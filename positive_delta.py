import os
import gc
import json
import random
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/scratch/hub/qwen3_1.7b_sft_merged")
    parser.add_argument("--base-model-name", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--cache-dir", type=str, default="/scratch/hub")
    parser.add_argument("--dataset-path", type=str, default="/scratch/datasets/openwebmath_600k")
    parser.add_argument("--output-file", type=str, default="positive_delta_traces.jsonl")
    parser.add_argument("--total-docs", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("-g", "--g", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-prefix-len", type=int, default=128)
    parser.add_argument("--max-prefix-len", type=int, default=2048)
    parser.add_argument("--continuation-len", type=int, default=16)
    parser.add_argument("--max-think-tokens", type=int, default=200)
    parser.add_argument("--min-delta", type=float, default=0.05)
    parser.add_argument("--keep-top-k", type=int, default=1)
    parser.add_argument("--allow-empty-thinks", action="store_true")
    parser.add_argument("--base-score-device", type=str, default="cuda:1")
    parser.add_argument("--sft-score-device", type=str, default="cuda:2")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def score_continuation_batch(
    model, tokenizer, prefix_ids: list[list[int]], cont_ids: list[list[int]], device: str
) -> list[float]:
    prefix_ids = [[int(t) for t in p] for p in prefix_ids]
    cont_ids = [[int(t) for t in c] for c in cont_ids]
    full_seq_ids = [p + c for p, c in zip(prefix_ids, cont_ids)]
    full_enc = tokenizer.pad(
        {"input_ids": full_seq_ids}, return_tensors="pt", padding=True
    )

    full_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)

    prefix_lens = torch.tensor([len(p) for p in prefix_ids], device=device)
    # Continuation length is fixed by dataset construction.
    continuation_len = len(cont_ids[0])

    last_hidden = model.model(
        input_ids=full_ids, attention_mask=attention_mask
    ).last_hidden_state

    shift_logit_indexes = (prefix_lens - 1).unsqueeze(1) + torch.arange(
        continuation_len, device=device
    ).unsqueeze(0)
    continuation_hidden = torch.gather(
        last_hidden,
        1,
        shift_logit_indexes.unsqueeze(-1).expand(-1, -1, last_hidden.size(2)),
    )

    logits = model.lm_head(continuation_hidden)
    log_probs = F.log_softmax(logits, dim=-1)

    labels = torch.gather(full_ids, 1, shift_logit_indexes + 1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    sequence_log_probs = token_log_probs.mean(dim=-1)

    return sequence_log_probs.tolist()


def get_batch_pairs(
    raw_data,
    starting_idx: int,
    batch_size: int,
    tokenizer,
    min_prefix_len: int,
    max_prefix_len: int,
    continuation_len: int,
    generator: torch.Generator,
) -> tuple[list[str], list[str], list[list[int]], list[list[int]]]:
    prefixes = []
    continuations = []
    all_prefix_ids = []
    all_cont_ids = []

    end_idx = min(starting_idx + batch_size, len(raw_data))
    for i in range(starting_idx, end_idx):
        text = raw_data[i]["text"]
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        n = len(token_ids)
        max_split_point = min(n - continuation_len, max_prefix_len)

        if max_split_point <= min_prefix_len + 1:
            continue

        split_point = torch.randint(
            min_prefix_len + 1,
            max_split_point,
            (1,),
            generator=generator,
        ).item()

        prefix_ids = token_ids[:split_point]
        cont_ids = token_ids[split_point : split_point + continuation_len]

        prefixes.append(tokenizer.decode(prefix_ids))
        continuations.append(tokenizer.decode(cont_ids))
        all_prefix_ids.append([int(t) for t in prefix_ids])
        all_cont_ids.append([int(t) for t in cont_ids])

    return prefixes, continuations, all_prefix_ids, all_cont_ids


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, cache_dir=args.cache_dir)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=args.model_path,
        download_dir=args.cache_dir,
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.40,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=args.g,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_think_tokens,
        stop=["</think>"],
        include_stop_str_in_output=True,
    )

    base_scorer = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        cache_dir=args.cache_dir,
        device_map=args.base_score_device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base_scorer.config.pad_token_id = tokenizer.pad_token_id
    base_scorer.eval()

    sft_scorer = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        device_map=args.sft_score_device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    sft_scorer.config.pad_token_id = tokenizer.pad_token_id
    sft_scorer.eval()

    raw_data = load_from_disk(args.dataset_path).shuffle(seed=args.seed)
    total_docs = min(args.total_docs, len(raw_data))
    doc_generator = torch.Generator().manual_seed(args.seed)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(output_path, "w", encoding="utf-8")

    total_processed_prefixes = 0
    total_saved_traces = 0
    prefixes_with_positive = 0
    all_deltas = []
    saved_deltas = []
    start_time = time.time()

    for start_idx in range(0, total_docs, args.batch_size):
        (
            batch_prefixes,
            batch_continuations,
            batch_prefix_ids,
            batch_cont_ids,
        ) = get_batch_pairs(
            raw_data=raw_data,
            starting_idx=start_idx,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            min_prefix_len=args.min_prefix_len,
            max_prefix_len=args.max_prefix_len,
            continuation_len=args.continuation_len,
            generator=doc_generator,
        )
        if not batch_prefixes:
            continue

        logp_nothink_list = score_continuation_batch(
            base_scorer,
            tokenizer,
            batch_prefix_ids,
            batch_cont_ids,
            args.base_score_device,
        )

        token_prompts = [
            TokensPrompt(prompt_token_ids=ids) for ids in batch_prefix_ids
        ]
        vllm_outputs = llm.generate(token_prompts, sampling_params)

        all_candidate_prefix_ids = []
        all_candidate_cont_ids = []
        for p_ids, c_ids, out in zip(
            batch_prefix_ids, batch_cont_ids, vllm_outputs
        ):
            generated_prompt_ids = [int(t) for t in out.prompt_token_ids]
            for sample in out.outputs:
                think_ids = [int(t) for t in sample.token_ids]
                all_candidate_prefix_ids.append(generated_prompt_ids + think_ids)
                all_candidate_cont_ids.append(c_ids)

        logp_think_list = []
        score_chunk_size = 8
        for i in range(0, len(all_candidate_prefix_ids), score_chunk_size):
            chunk_p = all_candidate_prefix_ids[i : i + score_chunk_size]
            chunk_c = all_candidate_cont_ids[i : i + score_chunk_size]
            chunk_lps = score_continuation_batch(
                sft_scorer, tokenizer, chunk_p, chunk_c, args.sft_score_device
            )
            logp_think_list.extend(chunk_lps)

        think_idx = 0
        for prefix, prefix_ids, cont, cont_ids, out, lp_nothink in zip(
            batch_prefixes,
            batch_prefix_ids,
            batch_continuations,
            batch_cont_ids,
            vllm_outputs,
            logp_nothink_list,
        ):
            doc_candidates = []
            for sample in out.outputs:
                lp_think = logp_think_list[think_idx]
                think_idx += 1

                delta = lp_think - lp_nothink
                all_deltas.append(delta)

                thought = sample.text
                if "<think>" in thought and "</think>" in thought:
                    inner = thought.split("<think>")[1].split("</think>")[0].strip()
                    if (len(inner) > 0 or args.allow_empty_thinks) and delta >= args.min_delta:
                        think_ids = [int(t) for t in sample.token_ids]
                        doc_candidates.append(
                            (delta, thought, think_ids, lp_think, lp_nothink)
                        )

            doc_candidates.sort(key=lambda x: x[0], reverse=True)
            if args.keep_top_k > 0:
                doc_candidates = doc_candidates[: args.keep_top_k]

            if doc_candidates:
                prefixes_with_positive += 1
                for d, t, think_ids, lp_t, lp_nt in doc_candidates:
                    entry = {
                        "prefix": prefix,
                        "prefix_ids": [int(t) for t in prefix_ids],
                        "teacher_thinking_trace": t,
                        "teacher_thinking_trace_ids": think_ids,
                        "continuation": cont,
                        "continuation_ids": [int(t) for t in cont_ids],
                        "delta": round(d, 4),
                        "logp_think": round(lp_t, 4),
                        "logp_nothink": round(lp_nt, 4),
                    }
                    f_out.write(json.dumps(entry) + "\n")
                    total_saved_traces += 1
                    saved_deltas.append(d)

            total_processed_prefixes += 1

        f_out.flush()
        elapsed = time.time() - start_time
        hit_rate = (prefixes_with_positive / total_processed_prefixes) * 100 if total_processed_prefixes > 0 else 0.0
        mean_all_d = np.mean(all_deltas) if all_deltas else 0.0
        mean_pos_d = np.mean(saved_deltas) if saved_deltas else 0.0
        docs_per_sec = total_processed_prefixes / elapsed if elapsed > 0 else 0.0

        print(
            f"[Prefixes: {total_processed_prefixes}/{total_docs} | {docs_per_sec:.2f} docs/s] "
            f"Saved: {total_saved_traces} (Hit Rate: {hit_rate:.1f}%) | "
            f"Mean Delta (all): {mean_all_d:+.3f} | Mean Delta (saved): {mean_pos_d:+.3f}"
        )

    f_out.close()
    del llm, base_scorer, sft_scorer
    gc.collect()


if __name__ == "__main__":
    main()
    os._exit(0)
