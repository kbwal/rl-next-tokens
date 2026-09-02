#!/usr/bin/env python3
"""Measure the information-gain signal the GRPO reward actually exposes.

Positions are drawn fresh from openwebmath (not from the SFT trace file, which
the SFT checkpoint has memorised). For each position the observed continuation
is scored under several conditions:

  base no-think  : base model, prefix -> continuation. The honest counterfactual.
  pol no-think   : policy, prefix -> continuation, no think block.
  pol empty      : policy, prefix -> <think></think> -> continuation (seam only).
  pol sampled    : policy, prefix -> <think> rollout </think> -> continuation, G times.

What matters for GRPO is (a) real information gain against the base model,
(b) how much of it is pure formatting, and (c) the within-group spread, which is
the entire learning signal, versus the length/reward correlation that drives the
shorten-the-thought attractor.
"""

from __future__ import annotations

import argparse
import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-1.7B-Base"
CACHE = "/scratch/hub"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=BASE)
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--label", default="model")
    p.add_argument("--dataset-path", default="/scratch/datasets/openwebmath_600k")
    p.add_argument("--doc-offset", type=int, default=400_000)
    p.add_argument("--n-positions", type=int, default=64)
    p.add_argument("--g", type=int, default=16)
    p.add_argument("--continuation-length", type=int, default=16)
    p.add_argument("--min-prefix", type=int, default=128)
    p.add_argument("--max-prefix", type=int, default=768)
    p.add_argument("--max-think-tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--ref-device", default="cuda:2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    return p.parse_args()


@torch.no_grad()
def score(model, tokenizer, prefixes, conts, device, chunk=8):
    """Mean log-prob of each continuation given its prefix."""
    out = []
    for i in range(0, len(prefixes), chunk):
        ps, cs = prefixes[i : i + chunk], conts[i : i + chunk]
        full = [p + c for p, c in zip(ps, cs)]
        enc = tokenizer.pad({"input_ids": full}, return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        plen = torch.tensor([len(p) for p in ps], device=device)
        clen = len(cs[0])
        hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        idx = (plen - 1).unsqueeze(1) + torch.arange(clen, device=device).unsqueeze(0)
        h = torch.gather(hidden, 1, idx.unsqueeze(-1).expand(-1, -1, hidden.size(2)))
        logp = F.log_softmax(model.lm_head(h), dim=-1)
        labels = torch.gather(ids, 1, idx + 1)
        tok = logp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        out.extend(tok.mean(dim=-1).tolist())
    return out


@torch.no_grad()
def sample_thoughts(model, tokenizer, prompt_ids, g, max_new, temp, device, close_id):
    ids = torch.tensor([prompt_ids] * g, dtype=torch.long, device=device)
    gen = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temp,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[tokenizer.eos_token_id, close_id],
    )
    new = gen[:, len(prompt_ids) :]
    thoughts = []
    for row in new.tolist():
        body = []
        for t in row:
            if t == close_id:
                body.append(t)
                break
            if t == tokenizer.eos_token_id:
                break
            body.append(t)
        if not body or body[-1] != close_id:
            body.append(close_id)
        thoughts.append(body)
    return thoughts


def pearson(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def load_model(path, adapter, device):
    m = AutoModelForCausalLM.from_pretrained(
        path,
        cache_dir=CACHE if path == BASE else None,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    )
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
        m = m.merge_and_unload()
    m.eval()
    return m


def build_positions(args, tokenizer):
    data = load_from_disk(args.dataset_path)
    gen = torch.Generator().manual_seed(args.seed)
    positions = []
    i = args.doc_offset
    while len(positions) < args.n_positions and i < len(data):
        text = data[i]["text"]
        i += 1
        ids = [
            int(t)
            for t in tokenizer(text, add_special_tokens=False)["input_ids"][
                : args.max_prefix + args.continuation_length
            ]
        ]
        hi = min(len(ids) - args.continuation_length, args.max_prefix)
        if hi <= args.min_prefix + 1:
            continue
        sp = int(torch.randint(args.min_prefix + 1, hi, (1,), generator=gen).item())
        positions.append(
            (ids[:sp], ids[sp : sp + args.continuation_length])
        )
    return positions


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(BASE, cache_dir=CACHE)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    open_id = tokenizer.convert_tokens_to_ids("<think>")
    close_id = tokenizer.convert_tokens_to_ids("</think>")

    positions = build_positions(args, tokenizer)
    print(f"{len(positions)} held-out positions from {args.dataset_path}", flush=True)

    policy = load_model(args.model_path, args.adapter_path, args.device)
    policy.config.pad_token_id = tokenizer.pad_token_id
    ref = load_model(BASE, None, args.ref_device)
    ref.config.pad_token_id = tokenizer.pad_token_id

    per_position = []
    for prefix_ids, cont_ids in positions:
        r_base = score(ref, tokenizer, [prefix_ids], [cont_ids], args.ref_device)[0]
        r_nothink = score(policy, tokenizer, [prefix_ids], [cont_ids], args.device)[0]
        r_empty = score(
            policy,
            tokenizer,
            [prefix_ids + [open_id, close_id]],
            [cont_ids],
            args.device,
        )[0]

        prompt_ids = prefix_ids + [open_id]
        thoughts = sample_thoughts(
            policy,
            tokenizer,
            prompt_ids,
            args.g,
            args.max_think_tokens,
            args.temperature,
            args.device,
            close_id,
        )
        cand = [prompt_ids + t for t in thoughts]
        r_sampled = score(
            policy, tokenizer, cand, [cont_ids] * len(cand), args.device
        )
        lens = [len(t) for t in thoughts]

        entry = {
            "r_base_nothink": r_base,
            "r_policy_nothink": r_nothink,
            "r_policy_empty": r_empty,
            "r_sampled": r_sampled,
            "thought_lens": lens,
            "group_mean": float(np.mean(r_sampled)),
            "group_std": float(np.std(r_sampled)),
            "group_max": float(np.max(r_sampled)),
            "len_reward_corr": pearson(lens, r_sampled),
        }
        per_position.append(entry)
        print(
            f"[{len(per_position):3d}/{len(positions)}] "
            f"base={r_base:+.3f} pol_nothink={r_nothink:+.3f} empty={r_empty:+.3f} "
            f"grp_mean={entry['group_mean']:+.3f} grp_std={entry['group_std']:.3f} "
            f"grp_max={entry['group_max']:+.3f} len={np.mean(lens):.0f} "
            f"corr(len,r)={entry['len_reward_corr']:+.2f}",
            flush=True,
        )

    def avg(fn):
        return float(np.mean([fn(p) for p in per_position]))

    ig_vs_base = avg(lambda p: p["group_mean"] - p["r_base_nothink"])
    ig_best_vs_base = avg(lambda p: p["group_max"] - p["r_base_nothink"])
    ig_vs_self = avg(lambda p: p["group_mean"] - p["r_policy_nothink"])
    fmt_only = avg(lambda p: p["r_policy_empty"] - p["r_base_nothink"])
    sft_damage = avg(lambda p: p["r_policy_nothink"] - p["r_base_nothink"])
    thinking_over_empty = avg(lambda p: p["group_mean"] - p["r_policy_empty"])
    gstd = avg(lambda p: p["group_std"])
    across = float(np.std([p["group_mean"] for p in per_position]))
    corrs = [
        p["len_reward_corr"]
        for p in per_position
        if not math.isnan(p["len_reward_corr"])
    ]
    corr_mean = float(np.mean(corrs))
    frac_neg_corr = float(np.mean([1.0 if c < 0 else 0.0 for c in corrs]))
    frac_beat_base = avg(
        lambda p: 1.0 if p["group_mean"] > p["r_base_nothink"] else 0.0
    )
    frac_beat_empty = avg(
        lambda p: 1.0 if p["group_mean"] > p["r_policy_empty"] else 0.0
    )

    print("\n" + "=" * 74)
    print(
        f"{args.label}: {len(per_position)} held-out positions, G={args.g}, "
        f"cont_len={args.continuation_length}"
    )
    print("=" * 74)
    print(f"  SFT damage      policy no-think vs base no-think : {sft_damage:+.4f}")
    print(f"  format only     <think></think>  vs base no-think : {fmt_only:+.4f}")
    print(f"  total gain      sampled thought vs base no-think : {ig_vs_base:+.4f}")
    print(f"  best-of-{args.g} gain                              : {ig_best_vs_base:+.4f}")
    print(f"  REAL thinking   sampled thought vs empty think   : {thinking_over_empty:+.4f}")
    print(f"  (sampled vs policy's own no-think)               : {ig_vs_self:+.4f}")
    print("-" * 74)
    print(f"  within-group reward std (the entire RL signal)   : {gstd:.4f}")
    print(f"  across-position reward std                       : {across:.4f}")
    print(f"  mean within-group corr(thought len, reward)      : {corr_mean:+.4f}")
    print(f"  fraction of positions with negative len corr     : {frac_neg_corr:.3f}")
    print(f"  positions where thinking beats base no-think     : {frac_beat_base:.3f}")
    print(f"  positions where thinking beats empty think       : {frac_beat_empty:.3f}")
    if abs(thinking_over_empty) > 0:
        print("-" * 74)
        print(
            f"  |real thinking gain| / within-group std          : "
            f"{abs(thinking_over_empty) / gstd:.3f}"
        )
    print("=" * 74)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "label": args.label,
                    "args": vars(args),
                    "summary": {
                        "n": len(per_position),
                        "sft_damage": sft_damage,
                        "format_only_gain": fmt_only,
                        "total_gain_vs_base": ig_vs_base,
                        "best_of_g_gain_vs_base": ig_best_vs_base,
                        "real_thinking_gain_vs_empty": thinking_over_empty,
                        "gain_vs_policy_nothink": ig_vs_self,
                        "within_group_std": gstd,
                        "across_position_std": across,
                        "len_reward_corr": corr_mean,
                        "frac_negative_len_corr": frac_neg_corr,
                        "frac_beat_base": frac_beat_base,
                        "frac_beat_empty": frac_beat_empty,
                    },
                    "per_position": per_position,
                },
                f,
                indent=2,
            )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
