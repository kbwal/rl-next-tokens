import os

os.environ["WANDB_PROJECT"] = "rl-ntp"

import argparse
from pathlib import Path
import random
import numpy as np
from typing import cast
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from load_data import create_grpo_dataset_good_splits


class ContinuationReward:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        completion_ids: list[list[int]],
        continuation_ids: list[list[int]],
        **kwargs,
    ) -> list[float | None]:
        rewards: list[float] = []
        was_training = self.model.training
        self.model.eval()
        # Match GRPOTrainer._tokenize_prompts exactly. The prompt text is decoded
        # dataset text, so stored pre-decode IDs are not necessarily the IDs used
        # to generate completion_ids.
        prompt_ids = self.tokenizer(prompts)["input_ids"]
        prompt_ids = [[int(t) for t in p] for p in prompt_ids]
        completion_ids = [[int(t) for t in c] for c in completion_ids]
        cont_ids = [[int(t) for t in c] for c in continuation_ids]

        # note: this only works if the lengths of the continuations are the same
        # otherwise it won't, and it just cuts off to the min
        length_of_continuation = min(len(c) for c in cont_ids)

        with torch.no_grad():
            for i in range(0, len(prompt_ids), self.batch_size):
                chunk_prompt_ids = prompt_ids[i : i + self.batch_size]
                chunk_completion_ids = completion_ids[i : i + self.batch_size]
                chunk_cont_ids = cont_ids[i : i + self.batch_size]

                chunk_full_seq_ids = [
                    p + c + cont
                    for p, c, cont in zip(
                        chunk_prompt_ids, chunk_completion_ids, chunk_cont_ids
                    )
                ]
                full_enc = self.tokenizer.pad(
                    {"input_ids": chunk_full_seq_ids},
                    return_tensors="pt",
                    padding=True,
                )
                full_ids = full_enc["input_ids"].to(self.model.device)
                attention_mask = full_enc["attention_mask"].to(self.model.device)
                prefix_len = torch.tensor(
                    [
                        len(p) + len(c)
                        for p, c in zip(chunk_prompt_ids, chunk_completion_ids)
                    ],
                    device=self.model.device,
                )

                last_hidden = self.model.model(
                    input_ids=full_ids, attention_mask=attention_mask
                ).last_hidden_state

                shift_logit_indexes = (prefix_len - 1).unsqueeze(1) + torch.arange(
                    length_of_continuation, device=self.model.device
                ).unsqueeze(0)

                continuation_hidden = torch.gather(
                    last_hidden,
                    1,
                    shift_logit_indexes.unsqueeze(-1).expand(
                        -1, -1, last_hidden.size(2)
                    ),
                )
                only_continuation_logits = self.model.lm_head(continuation_hidden)
                log_probs = F.log_softmax(only_continuation_logits, dim=-1)
                labels = torch.gather(full_ids, 1, shift_logit_indexes + 1)
                token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                sequence_log_probs = token_log_probs.mean(dim=-1)
                rewards.extend(sequence_log_probs.tolist())

        if was_training:
            self.model.train()

        return rewards  # type: ignore


class FormatPenaltyReward:
    def __init__(self, format_penalty: float):
        self.format_penalty = format_penalty

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for c in completions:
            has_extra_open = "<think>" in c
            close_count = c.count("</think>")
            if has_extra_open or close_count != 1:
                rewards.append(-self.format_penalty)
            elif not c.rstrip().endswith("</think>"):
                rewards.append(-self.format_penalty)
            else:
                rewards.append(0.0)
        return rewards


class LengthPenaltyReward:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, completion_ids: list[list[int]], **kwargs) -> list[float]:
        return [-self.alpha * float(len(c)) for c in completion_ids]


run_name = "grpo-good-split-data-full-run-1"
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--g", type=int, default=8)
parser.add_argument("--b", type=int, default=8)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=2048)
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--alpha", type=float, default=0.0005)
parser.add_argument("--format-penalty", type=float, default=3.0)
parser.add_argument("--min-prefix-len", type=int, default=128)
parser.add_argument("--max-prefix-len", type=int, default=2048)
parser.add_argument("--continuation-length", type=int, default=16)
parser.add_argument("--max-steps", type=int, default=2000)
parser.add_argument("--max-checkpoints", type=int, default=1)
parser.add_argument("--wandb", action="store_true")
args = parser.parse_args()
SEED = args.seed
G = args.g
B = args.b
T = args.temperature
MAX_NEW_TOKENS = args.max_new_tokens
LR = args.lr
ALPHA = args.alpha
FORMAT_PENALTY = args.format_penalty
MAX_CHECKPOINTS = args.max_checkpoints
CHECKPOINT_ROOT = Path(f"./grpo-checkpoints") / f"{run_name}-checkpoints"
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = f"cuda:{local_rank}"
torch.cuda.set_device(device)
device_map = {"": device}
model_name, cache_dir = "Qwen/Qwen3-1.7B-Base", "/scratch/hub"
adapter_path = (
    "./sft-checkpoints/sft-good-splits-checkpoints/run-32-0.0003-1.0/batch_123"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
dataloader_generator = torch.Generator().manual_seed(SEED)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    device_map=device_map,
    attn_implementation="sdpa",
)
base_model.config.use_cache = False
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()
model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
model = model.merge_and_unload()  # type: ignore
model = cast(PreTrainedModel, model)
for p in model.parameters():
    p.requires_grad = True

scorer_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    device_map=device_map,
    attn_implementation="sdpa",
)
scorer_model.eval()
scorer_model.requires_grad_(False)

closethink_id = tokenizer.convert_tokens_to_ids("</think>")
grpo_config = GRPOConfig(
    output_dir=str(CHECKPOINT_ROOT),
    run_name=run_name,
    num_generations=G,
    per_device_train_batch_size=G,
    gradient_accumulation_steps=B,
    steps_per_generation=B,
    max_completion_length=MAX_NEW_TOKENS,
    temperature=T,
    learning_rate=LR,
    optim="paged_adamw_8bit",
    num_iterations=1,
    generation_kwargs={"stop_token_ids": [tokenizer.eos_token_id, closethink_id]},
    seed=SEED,
    bf16=True,
    lr_scheduler_type="constant",
    max_steps=args.max_steps,
    warmup_steps=int(0.05 * args.max_steps),
    # warmup_ratio=0.05,
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=MAX_CHECKPOINTS,
    report_to="wandb" if args.wandb else "none",
    # num_train_epochs=100,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.25,
    vllm_max_model_length=args.max_prefix_len + args.max_new_tokens + 32,
    vllm_enable_sleep_mode=False,
    vllm_importance_sampling_correction=False,
    use_liger_kernel=True,
)

train_dataset = create_grpo_dataset_good_splits(
    tokenizer=tokenizer,
    samples_per_doc=1,
    min_prefix_len=args.min_prefix_len,
    max_prefix_len=args.max_prefix_len,
    continuation_length=args.continuation_length,
    dataset_path="/scratch/datasets/openwebmath2M",
    seed=SEED,
    force_open_think=True,
    scorer_model=scorer_model,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        ContinuationReward(model, tokenizer),
        FormatPenaltyReward(format_penalty=FORMAT_PENALTY),
        LengthPenaltyReward(alpha=ALPHA),
    ],  # type: ignore
    args=grpo_config,
    train_dataset=train_dataset,
)

trainer.train()
