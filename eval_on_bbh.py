from __future__ import annotations
import argparse
import json
import os
from typing import Any
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_bbh_task(
    task_name: str, n_examples: int, data_dir: str = "./data/bbh"
) -> list[dict[str, Any]]:
    path = os.path.join(data_dir, f"{task_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"BBH task file not found at {path}")

    with open(path) as f:
        raw_data = json.load(f)

    all_examples = raw_data.get("examples", [])
    if n_examples is None or n_examples <= 0 or n_examples >= len(all_examples):
        raw_examples = all_examples
    else:
        raw_examples = all_examples[:n_examples]

    formatted = []
    for idx, ex in enumerate(raw_examples):
        prefix = ex["input"].rstrip()
        continuation = ex["target"].strip()

        formatted.append(
            {
                "id": f"{task_name}_{idx:03d}",
                "task": task_name,
                "prompt": prefix,
                "continuation": continuation,
                "expected_answer": ex["target"],
            }
        )

    return formatted


@torch.no_grad()
def _score_single_candidate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    contexts: list[str],
    continuations: list[str],
    device: str,
    batch_size: int = 16,
) -> list[float]:
    results: list[float] = []
    model.eval()

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    for i in range(0, len(contexts), batch_size):
        b_ctx = contexts[i : i + batch_size]
        b_cont = continuations[i : i + batch_size]

        b_ctx_ids = [tokenizer(c, add_special_tokens=False)["input_ids"] for c in b_ctx]
        b_cont_ids = [
            tokenizer(co, add_special_tokens=False)["input_ids"] for co in b_cont
        ]

        full_seqs = [ctx + cont for ctx, cont in zip(b_ctx_ids, b_cont_ids)]
        padded = tokenizer.pad(
            {"input_ids": full_seqs}, return_tensors="pt", padding=True
        )
        input_ids = padded["input_ids"].to(device)
        attention_mask = padded["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits, dim=-1)

        for j, (ctx_ids, cont_ids) in enumerate(zip(b_ctx_ids, b_cont_ids)):
            if len(cont_ids) == 0:
                results.append(-100.0)
                continue
            start_idx = len(ctx_ids) - 1
            end_idx = start_idx + len(cont_ids)
            cont_labels = torch.tensor(cont_ids, device=device)
            seq_log_probs = log_probs[j, start_idx:end_idx]
            token_log_probs = seq_log_probs.gather(
                1, cont_labels.unsqueeze(-1)
            ).squeeze(-1)
            results.append(float(token_log_probs.sum().item()))

    tokenizer.padding_side = orig_padding_side
    return results


@torch.no_grad()
def compute_continuation_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    contexts: list[str],
    continuations: list[str],
    device: str,
    batch_size: int = 16,
) -> tuple[list[float], list[float]]:
    cands_no_space = [c.strip() for c in continuations]
    cands_with_space = [" " + c.strip() for c in continuations]

    sum_lps_no_space = _score_single_candidate_batch(
        model, tokenizer, contexts, cands_no_space, device, batch_size=batch_size
    )
    sum_lps_with_space = _score_single_candidate_batch(
        model, tokenizer, contexts, cands_with_space, device, batch_size=batch_size
    )

    per_token_results = []
    seq_results = []
    for s0, s1, c0, c1 in zip(
        sum_lps_no_space, sum_lps_with_space, cands_no_space, cands_with_space
    ):
        ids0 = tokenizer(c0, add_special_tokens=False)["input_ids"]
        ids1 = tokenizer(c1, add_special_tokens=False)["input_ids"]
        if ids0 == ids1:
            total_logp = s0
        else:
            total_logp = float(np.logaddexp(s0, s1))
        target_len = max(1, len(ids0))
        per_token_results.append(total_logp / target_len)
        seq_results.append(total_logp)
    return per_token_results, seq_results


@torch.no_grad()
def sample_thoughts_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    device: str,
    batch_size: int = 16,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    force_open_think: bool = True,
) -> tuple[list[str], list[str], list[int], list[bool]]:
    model.eval()
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    eos_token_id = tokenizer.eos_token_id
    close_think_str = "</think>"
    close_think_ids = tokenizer(close_think_str, add_special_tokens=False)["input_ids"]

    if force_open_think:
        input_prompts = [p + "<think>" for p in prompts]
    else:
        input_prompts = prompts

    contexts: list[str] = []
    thoughts: list[str] = []
    lengths: list[int] = []
    closed: list[bool] = []

    for i in range(0, len(input_prompts), batch_size):
        b_prompts = input_prompts[i : i + batch_size]
        enc = tokenizer(b_prompts, return_tensors="pt", padding=True).to(device)

        out = model.generate(  # type: ignore
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=eos_token_id,
            eos_token_id=[eos_token_id] + close_think_ids,
        )

        for j in range(len(b_prompts)):
            gen_ids = out[j][enc.input_ids.shape[1] :].tolist()
            if eos_token_id in gen_ids:
                eos_pos = gen_ids.index(eos_token_id)
                gen_ids = gen_ids[:eos_pos]
            if len(close_think_ids) > 0 and close_think_ids[0] in gen_ids:
                close_pos = gen_ids.index(close_think_ids[0])
                gen_ids = gen_ids[: close_pos + 1]

            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            has_close = close_think_str in gen_text or (
                len(close_think_ids) > 0 and close_think_ids[0] in gen_ids
            )
            if not has_close:
                gen_text = gen_text + close_think_str  # type: ignore

            full_context = b_prompts[j] + gen_text  # type: ignore
            thought_tokens = len(gen_ids)

            contexts.append(full_context)
            thoughts.append(gen_text)  # type: ignore
            lengths.append(thought_tokens)
            closed.append(has_close)

    tokenizer.padding_side = orig_padding_side
    return contexts, thoughts, lengths, closed


def merge_bbh_results(
    task: str,
    results_dir: str = "results",
    model_names: list[str] | None = None,
    output_file: str | None = None,
) -> dict[str, Any] | None:
    if model_names is None:
        model_names = ["base", "sft", "grpo80", "grpo300"]

    merged_models: dict[str, Any] = {}
    n_examples = None

    for m in model_names:
        file_path = os.path.join(results_dir, f"{task}_{m}_results.json")
        if not os.path.exists(file_path):
            print(f"[Merge] Warning: {file_path} not found, skipping {m}.")
            continue
        with open(file_path) as f:
            data = json.load(f)
        merged_models[m] = data.get("details", data)
        if n_examples is None:
            n_examples = data.get(
                "n_examples", len(merged_models[m].get("logprobs", []))
            )

    if not merged_models:
        print(f"[Merge] No result files found in {results_dir} for task {task}.")
        return None

    base_mean = (
        merged_models["base"]["mean_logprob"] if "base" in merged_models else 0.0
    )
    base_seq_mean = (
        merged_models["base"].get("mean_seq_logprob", 0.0)
        if "base" in merged_models
        else 0.0
    )

    summary = {
        "task": task,
        "n_examples": n_examples,
        "mean_logprob": {m: merged_models[m]["mean_logprob"] for m in merged_models},
        "info_gain_vs_base": {
            m: merged_models[m]["mean_logprob"] - base_mean
            for m in ["sft", "grpo80", "grpo300"]
            if m in merged_models
        },
        "mean_seq_logprob": {
            m: merged_models[m].get(
                "mean_seq_logprob", merged_models[m]["mean_logprob"]
            )
            for m in merged_models
        },
        "seq_info_gain_vs_base": {
            m: merged_models[m].get("mean_seq_logprob", 0.0) - base_seq_mean
            for m in ["sft", "grpo80", "grpo300"]
            if m in merged_models
        },
        "mean_thought_length": {
            m: merged_models[m]["mean_thought_length"] for m in merged_models
        },
        "format_adherence": {
            m: merged_models[m]["format_adherence"] for m in merged_models
        },
    }

    merged_results = {
        "task": task,
        "n_examples": n_examples,
        "models": merged_models,
        "summary": summary,
    }

    if output_file is None:
        output_file = os.path.join(results_dir, f"{task}_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(merged_results, f, indent=2)

    for m in model_names:
        temp_file = os.path.join(results_dir, f"{task}_{m}_results.json")
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print("\n" + "=" * 80)
    print(f"MERGED BBH RESULTS: {task.upper()} ({n_examples} examples)")
    print("=" * 80)
    cols = [m for m in model_names if m in merged_models]
    header = f"{'Metric':<25} | " + " | ".join(f"{c.upper():<10}" for c in cols)
    print(header)
    print("-" * len(header))

    def fmt_val(m, metric):
        val = summary[metric].get(m)
        if val is None:
            return "N/A"
        if "info_gain" in metric:
            return f"{val:+10.4f}"
        if metric == "format_adherence":
            return f"{val*100:9.1f}%"
        if metric == "mean_thought_length":
            return f"{val:10.1f}"
        return f"{val:10.4f}"

    print(
        f"{'Per-Token Log-Prob':<25} | "
        + " | ".join(f"{fmt_val(c, 'mean_logprob'):<10}" for c in cols)
    )
    if "base" in merged_models:
        print(
            f"{'Per-Token Info Gain':<25} | "
            + " | ".join(
                f"{('0.0000' if c == 'base' else fmt_val(c, 'info_gain_vs_base')):<10}"
                for c in cols
            )
        )
    print(
        f"{'Total Seq Log-Prob':<25} | "
        + " | ".join(f"{fmt_val(c, 'mean_seq_logprob'):<10}" for c in cols)
    )
    if "base" in merged_models:
        print(
            f"{'Total Seq Info Gain':<25} | "
            + " | ".join(
                f"{('0.0000' if c == 'base' else fmt_val(c, 'seq_info_gain_vs_base')):<10}"
                for c in cols
            )
        )
    print(
        f"{'Mean Thought Length':<25} | "
        + " | ".join(f"{fmt_val(c, 'mean_thought_length'):<10}" for c in cols)
    )
    print(
        f"{'Format Adherence':<25} | "
        + " | ".join(f"{fmt_val(c, 'format_adherence'):<10}" for c in cols)
    )
    print("=" * 80)
    print(f"Merged results successfully saved to: {output_file}\n", flush=True)

    return merged_results


def benchmark_bbh(
    task: str = "multistep_arithmetic_two",
    n_examples: int = 50,
    data_dir: str = "./data/bbh",
    base_model_path: str = "Qwen/Qwen3-1.7B-Base",
    sft_adapter_path: str = "./sft-checkpoints/sft-good-splits-checkpoints/run-32-0.0003-1.0/batch_123",
    grpo80_path: str = "./grpo-checkpoints/important-checkpoints/checkpoint-80-91d428mf",
    grpo300_path: str = "./grpo-checkpoints/grpo-good-split-data-full-run-1-checkpoints/checkpoint-300",
    cache_dir: str = "/scratch/hub",
    batch_size: int = 16,
    seed: int = 10,
    output_json: str | None = None,
    force_open_think: bool = True,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    is_distributed = world_size > 1
    if is_distributed and not dist.is_initialized():
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", device_id=torch.device(device))

    models = [
        {"name": "base", "type": "base", "path": base_model_path, "adapter": None},
        {
            "name": "sft",
            "type": "peft",
            "path": base_model_path,
            "adapter": sft_adapter_path,
        },
        {"name": "grpo80", "type": "causal", "path": grpo80_path, "adapter": None},
        {"name": "grpo300", "type": "causal", "path": grpo300_path, "adapter": None},
    ]

    if rank >= len(models):
        raise ValueError(
            f"Rank {rank} exceeds available models list (len={len(models)})"
        )

    cfg = models[rank]

    if output_json is None:
        out_file = os.path.join("results", f"{task}_{cfg['name']}_results.json")
    elif os.path.isdir(output_json) or output_json.endswith("/"):
        out_file = os.path.join(output_json, f"{task}_{cfg['name']}_results.json")
    elif ".json" in output_json:
        out_file = output_json.replace(".json", f"_{cfg['name']}.json")
    else:
        out_file = f"{output_json}_{cfg['name']}.json"

    print(
        f"[Rank {rank}] Task: {task.upper()} | Model: {cfg['name'].upper()} on {device}"
    )
    print(
        f"[Rank {rank}] Device: {device} | Examples: {n_examples} | Batch Size: {batch_size}"
    )
    print(f"[Rank {rank}] Force Open Think: {force_open_think}")
    print(f"[Rank {rank}] Output File: {out_file}")

    dataset = load_bbh_task(task, n_examples=n_examples, data_dir=data_dir)
    prefixes = [item["prompt"] for item in dataset]
    continuations = [item["continuation"] for item in dataset]

    print(f"[Rank {rank}] Loaded {len(dataset)} examples. Sample 0:")
    print(f"[Rank {rank}] Prefix: {repr(prefixes[0])}")
    print(f"[Rank {rank}] Continuation: {repr(continuations[0])}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    if cfg["type"] == "base":
        print(
            f"[Rank {rank} | {cfg['name'].upper()}] Loading base model onto {device}...",
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            cache_dir=cache_dir,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        print(
            f"[Rank {rank} | {cfg['name'].upper()}] Computing continuation logprobs...",
            flush=True,
        )
        per_token_lps, seq_lps = compute_continuation_logprobs(
            model, tokenizer, prefixes, continuations, device, batch_size=batch_size
        )
        mean_pt_lp = float(np.mean(per_token_lps))
        mean_seq_lp = float(np.mean(seq_lps))
        model_res = {
            "logprobs": per_token_lps,
            "mean_logprob": mean_pt_lp,
            "std_logprob": float(np.std(per_token_lps)),
            "seq_logprobs": seq_lps,
            "mean_seq_logprob": mean_seq_lp,
            "std_seq_logprob": float(np.std(seq_lps)),
            "thought_lengths": [],
            "mean_thought_length": 0.0,
            "format_adherence": 1.0,
            "sample_thoughts": [],
        }
    else:
        if cfg["type"] == "peft":
            print(
                f"[Rank {rank} | {cfg['name'].upper()}] Loading SFT PEFT model onto {device}...",
                flush=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                cache_dir=cache_dir,
                device_map=device,
                torch_dtype=torch.bfloat16,
            )
            model = PeftModel.from_pretrained(
                base_model, cfg["adapter"], is_trainable=False
            ).merge_and_unload()  # type: ignore
        else:
            print(
                f"[Rank {rank} | {cfg['name'].upper()}] Loading checkpoint model onto {device}...",
                flush=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                cache_dir=cache_dir,
                device_map=device,
                torch_dtype=torch.bfloat16,
            )
        model.eval()

        print(f"[Rank {rank} | {cfg['name'].upper()}] Sampling thoughts...", flush=True)
        contexts, thoughts, lengths, closed = sample_thoughts_batch(
            model,
            tokenizer,
            prefixes,
            device,
            batch_size=batch_size,
            max_new_tokens=1024,
            temperature=0.7,
            force_open_think=force_open_think,
        )
        mean_len = float(np.mean(lengths))
        format_rate = float(np.mean(closed))
        print(
            f"[Rank {rank} | {cfg['name'].upper()}] Generated {len(thoughts)} thoughts (mean len: {mean_len:.1f}, format: {format_rate*100:.1f}%)",
            flush=True,
        )

        print(
            f"[Rank {rank} | {cfg['name'].upper()}] Computing continuation logprobs...",
            flush=True,
        )
        per_token_lps, seq_lps = compute_continuation_logprobs(
            model,
            tokenizer,
            contexts,
            continuations,
            device,
            batch_size=batch_size,
        )
        mean_pt_lp = float(np.mean(per_token_lps))
        mean_seq_lp = float(np.mean(seq_lps))
        model_res = {
            "logprobs": per_token_lps,
            "mean_logprob": mean_pt_lp,
            "std_logprob": float(np.std(per_token_lps)),
            "seq_logprobs": seq_lps,
            "mean_seq_logprob": mean_seq_lp,
            "std_seq_logprob": float(np.std(seq_lps)),
            "thought_lengths": lengths,
            "mean_thought_length": mean_len,
            "format_adherence": format_rate,
            "sample_thoughts": thoughts,
        }

    results = {
        "task": task,
        "model": cfg["name"],
        "rank": rank,
        "device": device,
        "n_examples": len(dataset),
        "summary": {
            "mean_logprob": model_res["mean_logprob"],
            "std_logprob": model_res["std_logprob"],
            "mean_seq_logprob": model_res["mean_seq_logprob"],
            "std_seq_logprob": model_res["std_seq_logprob"],
            "mean_thought_length": model_res["mean_thought_length"],
            "format_adherence": model_res["format_adherence"],
        },
        "details": model_res,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"[Rank {rank} | {cfg['name'].upper()}] Per-Token LP: {model_res['mean_logprob']:.4f} | "
        f"Seq LP: {model_res['mean_seq_logprob']:.4f} | "
        f"Len: {model_res['mean_thought_length']:.1f} | Format: {model_res['format_adherence']*100:.1f}% | "
        f"Saved to: {out_file}",
        flush=True,
    )

    if is_distributed:
        dist.barrier()
        if rank == 0:
            merge_bbh_results(task=task, results_dir="results")
        dist.destroy_process_group()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="multistep_arithmetic_two")
    parser.add_argument("--data-dir", type=str, default="./data/bbh")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--b", type=int, default=16)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path or directory",
    )
    parser.add_argument(
        "--no-force-think", action="store_true", help="Do not force open <think>"
    )
    args = parser.parse_args()

    benchmark_bbh(
        task=args.task,
        n_examples=args.n,
        data_dir=args.data_dir,
        batch_size=args.b,
        output_json=args.output,
        force_open_think=not args.no_force_think,
    )
