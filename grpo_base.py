import os

os.environ["WANDB_PROJECT"] = "rl-ntp"

import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from load_data import create_grpo_base_overfit_dataset_full

RLP_INSTRUCTION = (
    "You are a continuation-and-reasoning assistant. You receive the prefix of a "
    "context, problem, solution, or derivation. First, briefly think between "
    "<think> and </think> about what should come next. Then, after </think>, "
    "continue the text in the SAME style as the prefix (notation, LaTeX, tone), "
    "focusing on the next few steps rather than jumping to a final boxed answer. "
    "Do not restate the question or add meta commentary; simply continue the content."
)


class ThinkReward:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        alpha: float,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        completion_ids: list[list[int]],
        continuations: list[str],
        **kwargs,
    ) -> list[float | None]:
        rewards: list[float] = []
        was_training = self.model.training
        self.model.eval()
        full_prefix = [prompts[i] + completions[i] for i in range(len(prompts))]
        full_text = [
            prompts[i] + completions[i] + continuations[i] for i in range(len(prompts))
        ]
        prefix_ids = self.tokenizer(full_prefix, return_tensors="pt", padding=True)[
            "input_ids"
        ].to(self.model.device)
        full_enc = self.tokenizer(full_text, return_tensors="pt", padding=True)
        full_ids = full_enc["input_ids"].to(self.model.device)
        attention_mask = full_enc["attention_mask"].to(self.model.device)
        prefix_len = torch.where(prefix_ids != self.tokenizer.pad_token_id, 1, 0).sum(
            dim=1
        )
        full_len = torch.where(full_ids != self.tokenizer.pad_token_id, 1, 0).sum(dim=1)
        thinking_len = torch.tensor(
            [len(c) for c in completion_ids],
            device=self.model.device,
            dtype=torch.float32,
        )

        # note: this only works if the lengths of the continuations are the same
        # otherwise it won't, and it just cuts off to the min
        length_of_continuation = (full_len - prefix_len).min().item()

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
            total_rewards = sequence_log_probs - self.alpha * thinking_len
            rewards.extend(total_rewards.tolist())

        if was_training:
            self.model.train()

        return rewards  # type: ignore


run_name = "grpo-base-overfitting"
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--g", type=int, default=8)
parser.add_argument("--b", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--alpha", type=float, default=0.0)
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
MAX_CHECKPOINTS = args.max_checkpoints
CHECKPOINT_ROOT = Path(f"./grpo-checkpoints") / f"{run_name}-checkpoints"
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = f"cuda:{local_rank}"
torch.cuda.set_device(device)
device_map = {"": device}
model_name, cache_dir = "Qwen/Qwen3-1.7B-Base", "/scratch/hub"

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

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    device_map=device_map,
    attn_implementation="sdpa",
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

closethink_id = tokenizer.convert_tokens_to_ids("</think>")
grpo_config = GRPOConfig(
    output_dir=str(CHECKPOINT_ROOT),
    run_name=run_name,
    num_generations=G,
    per_device_train_batch_size=G,
    gradient_accumulation_steps=B,
    steps_per_generation=1,
    max_completion_length=2048,
    temperature=1.0,
    learning_rate=LR,
    optim="paged_adamw_8bit",
    num_iterations=1,
    generation_kwargs={"stop_token_ids": [tokenizer.eos_token_id, closethink_id]},
    seed=SEED,
    bf16=True,
    lr_scheduler_type="constant",
    # max_steps=args.max_steps,
    # warmup_steps=int(0.05 * args.max_steps),
    warmup_ratio=0.05,
    logging_steps=5,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=MAX_CHECKPOINTS,
    report_to="wandb" if args.wandb else "none",
    num_train_epochs=100,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.3,
    vllm_max_model_length=2560,
    vllm_enable_sleep_mode=True,
    vllm_importance_sampling_correction=False,
    use_liger_kernel=True,
)

train_dataset = create_grpo_base_overfit_dataset_full(
    tokenizer=tokenizer,
    samples_per_doc=1,
    min_prefix_len=args.min_prefix_len,
    max_prefix_len=args.max_prefix_len,
    continuation_length=args.continuation_length,
    seed=SEED,
    instruction=RLP_INSTRUCTION,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=ThinkReward(model, tokenizer, alpha=ALPHA),
    args=grpo_config,
    train_dataset=train_dataset,
)

trainer.train()
