import argparse
import json
import os
from pathlib import Path
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from load_data import create_grpo_dataset_good_splits


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate good splits dataset for GRPO training."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/scratch/datasets/openwebmath2M",
        help="Path to source corpus (disk path or HF dataset name)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/scratch/datasets/openwebmath_good_splits",
        help="Directory to save the pre-generated dataset",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
        help="Base model name used for scoring splits",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/scratch/hub",
        help="Model cache directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to run the scorer model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100_000,
        help="Total number of good splits to extract",
    )
    parser.add_argument(
        "--samples-per-doc",
        type=int,
        default=1,
        help="Max samples to pick per document",
    )
    parser.add_argument(
        "--min-prefix-len",
        type=int,
        default=128,
        help="Minimum prefix length in tokens",
    )
    parser.add_argument(
        "--max-prefix-len",
        type=int,
        default=2048,
        help="Maximum prefix length in tokens",
    )
    parser.add_argument(
        "--continuation-length",
        type=int,
        default=64,
        help="Continuation length in tokens",
    )
    parser.add_argument(
        "--min-logprob",
        type=float,
        default=-4.5,
        help="Minimum first-token logprob",
    )
    parser.add_argument(
        "--max-logprob",
        type=float,
        default=-1.0,
        help="Maximum first-token logprob",
    )
    parser.add_argument(
        "--max-rolling-logprob",
        type=float,
        default=-0.8,
        help="Maximum rolling logprob threshold",
    )
    parser.add_argument(
        "--score-window-len",
        type=int,
        default=16,
        help="Sliding window size for rolling logprob",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Save intermediate checkpoint every N samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--force-open-think",
        action="store_true",
        default=False,
        help="Append <think> tag to the prompt (default: False)",
    )
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = f"cuda:{local_rank}" if world_size > 1 else args.device
    num_samples = (
        (args.num_samples + world_size - 1) // world_size
        if world_size > 1
        else args.num_samples
    )
    seed = args.seed + rank

    if world_size > 1:
        corpus = (
            load_from_disk(args.dataset_path)
            .shard(num_shards=world_size, index=rank, contiguous=True)  # type: ignore
            .shuffle(seed=seed)
        )
        output_dir = Path(args.output_dir) / f"shard_{rank}"
    else:
        corpus = args.dataset_path
        output_dir = Path(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = output_dir / "samples.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    scorer_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    scorer_model.eval()
    scorer_model.requires_grad_(False)

    dataset_stream = create_grpo_dataset_good_splits(
        tokenizer=tokenizer,
        samples_per_doc=args.samples_per_doc,
        min_prefix_len=args.min_prefix_len,
        max_prefix_len=args.max_prefix_len,
        continuation_length=args.continuation_length,
        dataset_path=corpus,  # type: ignore
        seed=seed,
        force_open_think=args.force_open_think,
        scorer_model=scorer_model,
        scorer_device=device,
        min_logprob=args.min_logprob,
        max_logprob=args.max_logprob,
        score_window_len=args.score_window_len,
        max_rolling_logprob=args.max_rolling_logprob,
        filter_multidomain=True,
    )

    collected_samples = []

    jsonl_file = open(jsonl_path, "a", encoding="utf-8")
    desc = f"Good splits [Rank {rank}]" if world_size > 1 else "Good splits"
    pbar = tqdm(
        total=num_samples,
        initial=len(collected_samples),
        desc=desc,
        position=local_rank if world_size > 1 else 0,
        leave=True,
    )

    try:
        for item in dataset_stream:
            row = {
                "prompt": item["prompt"],
                "continuations": item["continuations"],
                "continuation_ids": item["continuation_ids"],
            }
            collected_samples.append(row)
            jsonl_file.write(json.dumps(row) + "\n")
            pbar.update(1)

            if len(collected_samples) % args.save_interval == 0:
                jsonl_file.flush()

            if len(collected_samples) >= num_samples:
                break
    finally:
        jsonl_file.flush()
        jsonl_file.close()
        pbar.close()

    final_ds = Dataset.from_list(collected_samples)
    final_ds.save_to_disk(output_dir)
    print(f"[Rank {rank}] Saved {len(final_ds)} samples to {output_dir}")

    if world_size > 1:
        (output_dir / ".done").touch()

        if rank == 0:
            import time

            print("Waiting for all ranks to complete before merging...")
            for r in range(world_size):
                done_marker = Path(args.output_dir) / f"shard_{r}" / ".done"
                while not done_marker.exists():
                    time.sleep(1)

            print("All ranks finished. Merging shards...")
            shard_datasets = [
                load_from_disk(str(Path(args.output_dir) / f"shard_{r}"))
                for r in range(world_size)
            ]
            merged_ds = concatenate_datasets(shard_datasets)  # type: ignore
            if len(merged_ds) > args.num_samples:
                merged_ds = merged_ds.select(range(args.num_samples))
            merged_ds.save_to_disk(args.output_dir)

            merged_jsonl = Path(args.output_dir) / "samples.jsonl"
            total_written = 0
            with open(merged_jsonl, "w", encoding="utf-8") as out_f:
                for r in range(world_size):
                    shard_jsonl = Path(args.output_dir) / f"shard_{r}" / "samples.jsonl"
                    if shard_jsonl.exists():
                        with open(shard_jsonl, "r", encoding="utf-8") as in_f:
                            for line in in_f:
                                if total_written < args.num_samples:
                                    out_f.write(line)
                                    total_written += 1
            print(
                f"Successfully saved merged dataset ({len(merged_ds)} samples) to {args.output_dir}"
            )


if __name__ == "__main__":
    main()
