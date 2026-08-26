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
from load_data import create_grpo_dataset, create_grpo_overfit_dataset


class ThinkReward:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        alpha: float,
        format_penalty: float,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.format_penalty = format_penalty

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        completion_ids: list[list[int]],
        continuations: list[str],
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
        full_seq_ids = [
            p + c + cont
            for p, c, cont in zip(prompt_ids, completion_ids, cont_ids)
        ]
        full_enc = self.tokenizer.pad(
            {"input_ids": full_seq_ids}, return_tensors="pt", padding=True
        )
        full_ids = full_enc["input_ids"].to(self.model.device)
        attention_mask = full_enc["attention_mask"].to(self.model.device)
        prefix_len = torch.tensor(
            [
                len(p) + len(c)
                for p, c in zip(prompt_ids, completion_ids)
            ],
            device=self.model.device,
        )
        thinking_len = torch.tensor(
            [len(c) for c in completion_ids],
            device=self.model.device,
            dtype=torch.float32,
        )

        format_penalties = []
        for c in completions:
            has_open = "<think>" in c
            has_close = "</think>" in c
            if not has_open or not has_close:
                format_penalties.append(self.format_penalty)
            else:
                open_idx = c.find("<think>")
                close_idx = c.rfind("</think>")
                if open_idx >= close_idx:
                    format_penalties.append(self.format_penalty)
                else:
                    format_penalties.append(0.0)

        format_penalties_tensor = torch.tensor(
            format_penalties, device=self.model.device, dtype=torch.float32
        )

        # note: this only works if the lengths of the continuations are the same
        # otherwise it won't, and it just cuts off to the min
        length_of_continuation = min(len(c) for c in cont_ids)

        with torch.no_grad():
            last_hidden = self.model.model(
                input_ids=full_ids, attention_mask=attention_mask
            ).last_hidden_state

            shift_logit_indexes = (prefix_len - 1).unsqueeze(1) + torch.arange(
                length_of_continuation, device=self.model.device
            ).unsqueeze(0)

            continuation_hidden = torch.gather(
                last_hidden,
                1,
                shift_logit_indexes.unsqueeze(-1).expand(-1, -1, last_hidden.size(2)),
            )
            only_continuation_logits = self.model.lm_head(continuation_hidden)
            log_probs = F.log_softmax(only_continuation_logits, dim=-1)
            labels = torch.gather(full_ids, 1, shift_logit_indexes + 1)
            token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
            sequence_log_probs = token_log_probs.mean(dim=-1)
            total_rewards = (
                sequence_log_probs - self.alpha * thinking_len - format_penalties_tensor
            )
            rewards.extend(total_rewards.tolist())

        if was_training:
            self.model.train()

        return rewards  # type: ignore


run_name = "grpo-overfitting"
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--g", type=int, default=8)
parser.add_argument("--b", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--alpha", type=float, default=0.0005)
parser.add_argument("--format-penalty", type=float, default=3.0)
parser.add_argument("--min-prefix-len", type=int, default=128)
parser.add_argument("--max-prefix-len", type=int, default=2048)
parser.add_argument("--continuation-length", type=int, default=16)
parser.add_argument("--max-steps", type=int, default=2000)
parser.add_argument("--max-checkpoints", type=int, default=3)
parser.add_argument("--wandb", action="store_true")
args = parser.parse_args()
SEED = args.seed
G = args.g
B = args.b
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
    "/home/kushalb/rl-ntp/sft-removed-length-bias-checkpoints/run-8-0.0001/batch_1096"
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

closethink_id = tokenizer.convert_tokens_to_ids("</think>")
grpo_config = GRPOConfig(
    output_dir=str(CHECKPOINT_ROOT),
    run_name=run_name,
    num_generations=G,
    per_device_train_batch_size=G,
    gradient_accumulation_steps=B,
    steps_per_generation=1,
    max_completion_length=200,
    temperature=1.0,
    learning_rate=LR,
    optim="paged_adamw_8bit",
    num_iterations=1,
    generation_kwargs={"eos_token_id": [tokenizer.eos_token_id, closethink_id]},
    seed=SEED,
    bf16=True,
    lr_scheduler_type="cosine",
    # max_steps=args.max_steps,
    # warmup_steps=int(0.05 * args.max_steps),
    warmup_ratio=0.05,
    logging_steps=5,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=MAX_CHECKPOINTS,
    report_to="wandb" if args.wandb else "none",
    num_train_epochs=100,
)

train_dataset = create_grpo_overfit_dataset(
    tokenizer=tokenizer,
    samples_per_doc=1,
    min_prefix_len=args.min_prefix_len,
    max_prefix_len=args.max_prefix_len,
    continuation_length=args.continuation_length,
    dataset_path="/scratch/datasets/openwebmath_600k",
    seed=SEED,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=ThinkReward(
        model, tokenizer, alpha=ALPHA, format_penalty=FORMAT_PENALTY
    ),
    args=grpo_config,
    train_dataset=train_dataset,
)

trainer.train()
