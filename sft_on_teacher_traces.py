import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class TeacherDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["prefix"], item["teacher_thinking_trace"], item["continuation"]


def collate_teacher_batch(batch):
    prefixes, thoughts, continuations = zip(*batch)
    prefix_ids = tokenizer(list(prefixes), add_special_tokens=False)["input_ids"]
    thought_ids = tokenizer(list(thoughts), add_special_tokens=False)["input_ids"]
    continuation_ids = tokenizer(list(continuations), add_special_tokens=False)[
        "input_ids"
    ]

    input_rows = []
    attention_rows = []
    think_rows = []
    continuation_rows = []
    for prefix, thought, continuation in zip(prefix_ids, thought_ids, continuation_ids):
        input_rows.append(
            torch.tensor(prefix + thought + continuation, dtype=torch.long)
        )
        attention_rows.append(torch.ones(len(input_rows[-1]), dtype=torch.long))

        # true on thought tokens
        think_rows.append(
            torch.tensor(
                [False] * len(prefix)
                + [True] * len(thought)
                + [False] * len(continuation)
            )
        )
        # true on continuation tokens
        continuation_rows.append(
            torch.tensor(
                [False] * len(prefix)
                + [False] * len(thought)
                + [True] * len(continuation)
            )
        )

    return {
        "input_ids": pad_sequence(
            input_rows, batch_first=True, padding_value=tokenizer.pad_token_id
        ),
        "attention_mask": pad_sequence(
            attention_rows, batch_first=True, padding_value=0
        ),
        "think_token_mask": pad_sequence(
            think_rows, batch_first=True, padding_value=False
        ),
        "continuation_token_mask": pad_sequence(
            continuation_rows, batch_first=True, padding_value=False
        ),
    }


SEED = 0
R = 16
B = 8
LR = 3e-4
ALPHA = 0.25  # weight given to continuation wrt thinking
lora_config = LoraConfig(r=R, target_modules="all-linear", lora_alpha=2 * R)
device = "cuda:0"

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
dataloader_generator = torch.Generator().manual_seed(SEED)

model_name, cache_dir = "Qwen/Qwen3-1.7B-Base", "/scratch/hub"
model = AutoModelForCausalLM.from_pretrained(
    model_name, cache_dir=cache_dir, device_map=device, attn_implementation="sdpa"
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

lora_model = get_peft_model(model, lora_config)
optimizer = optim.AdamW(params=lora_model.parameters(), lr=LR, fused=True)
dataset = TeacherDataset("./teacher_traces.jsonl")
dataloader = DataLoader(
    dataset,
    batch_size=B,
    shuffle=True,
    collate_fn=collate_teacher_batch,
    generator=dataloader_generator,
)
lora_model.train()
lora_model.print_trainable_parameters()

for batch_idx, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    think_token_mask = batch["think_token_mask"].to(device)
    continuation_token_mask = batch["continuation_token_mask"].to(device)

    # 1: for targets
    think_targets = think_token_mask[:, 1:]
    continuation_targets = continuation_token_mask[:, 1:]

    last_hidden_state = lora_model.base_model.model.model(
        input_ids=input_ids, attention_mask=attention_mask
    ).last_hidden_state[:, :-1]
    no_prefix_hidden_state = last_hidden_state[think_targets | continuation_targets]
    logits = lora_model.base_model.model.lm_head(no_prefix_hidden_state)
    targets = input_ids[:, 1:][think_targets | continuation_targets]
    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets,
        reduction="none",
    )

    think_targets_cut = think_targets[think_targets | continuation_targets]
    continuation_targets_cut = continuation_targets[
        think_targets | continuation_targets
    ]

    loss_think = token_ce[think_targets_cut].mean()
    loss_cont = token_ce[continuation_targets_cut].mean()
    loss = loss_think + ALPHA * loss_cont

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(lora_model.parameters(), 1.0)
    optimizer.step()

    if batch_idx % 1 == 0:
        print(
            f"for batch {batch_idx}: loss={loss.item():.4f} thinking_loss={loss_think.item():.4f} continuation_loss={loss_cont.item():.4f}"
        )
