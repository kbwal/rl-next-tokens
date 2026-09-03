import copy
import os
import re
from typing import Any
from datasets import Dataset, IterableDataset, load_dataset, load_from_disk
import torch
import torch.nn.functional as F

FORCE_OPEN_THINK_TAG = "<think>"


def format_grpo_base_prompt(prefix: str, instruction: str) -> str:
    return instruction.strip() + "\n\n" + prefix


def _int_ids(ids) -> list[int]:
    return [int(t) for t in ids]


def _decode_exact(tokenizer, token_ids) -> str | None:
    token_ids = _int_ids(token_ids)
    text = tokenizer.decode(token_ids)
    reencoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text if _int_ids(reencoded) == token_ids else None


def _grpo_row(
    tokenizer,
    prefix_ids,
    continuation_ids,
    *,
    instruction: str | None = None,
    force_open_think: bool = True,
) -> dict | None:
    prefix_ids = _int_ids(prefix_ids)
    continuation_ids = _int_ids(continuation_ids)
    prefix = _decode_exact(tokenizer, prefix_ids)
    if prefix is None:
        return None
    if instruction is not None:
        prompt = format_grpo_base_prompt(prefix, instruction)
    else:
        prompt = prefix + FORCE_OPEN_THINK_TAG if force_open_think else prefix
    row = {
        "prompt": prompt,
        "continuations": tokenizer.decode(continuation_ids),
        "continuation_ids": continuation_ids,
    }
    if instruction is not None:
        row["raw_prefix"] = prefix
    return row


def _load_streaming_corpus(
    dataset_path: str,
    split: str,
    cache_dir: str,
    seed: int,
):
    return load_dataset(
        dataset_path,
        split=split,
        cache_dir=cache_dir,
        streaming=True,
    ).shuffle(seed=seed, buffer_size=1, max_buffer_input_shards=1)


def create_grpo_dataset(
    tokenizer,
    samples_per_doc: int,
    min_prefix_len: int,
    max_prefix_len: int,
    continuation_length: int,
    dataset_path: str,
    seed: int,
    force_open_think: bool = True,
) -> IterableDataset:
    raw_data: Any = load_from_disk(dataset_path).shuffle(seed=seed)
    generator = torch.Generator().manual_seed(seed)

    def gen():
        for item in raw_data:
            text = item["text"]
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            n = len(token_ids)
            max_split_point = min(n - continuation_length, max_prefix_len)

            if max_split_point <= min_prefix_len + 1:
                continue

            split_points = torch.randint(
                min_prefix_len + 1,
                max_split_point,
                (samples_per_doc,),
                generator=generator,
            ).tolist()

            for split_point in split_points:
                prefix_ids = token_ids[:split_point]
                continuation_ids = token_ids[
                    split_point : split_point + continuation_length
                ]
                row = _grpo_row(
                    tokenizer,
                    prefix_ids,
                    continuation_ids,
                    force_open_think=force_open_think,
                )
                if row is not None:
                    yield row

    return IterableDataset.from_generator(gen)


def score_document(
    scorer_model: Any,
    token_ids: list[int],
    scorer_device: Any = "cuda:0",
    score_window_len: int = 16,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if scorer_model is None:
        return None

    device = scorer_device
    full_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = scorer_model(input_ids=full_ids)
        logits = outputs.logits[:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        targets = full_ids[:, 1:]
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(
            -1
        )  # (1, T-1)

        if score_window_len > 1 and token_log_probs.shape[-1] >= score_window_len:
            k_len = score_window_len
            kernel = (
                torch.ones(1, 1, k_len, device=device, dtype=token_log_probs.dtype)
                / k_len
            )
            conv_out = (
                F.conv1d(token_log_probs.unsqueeze(1), kernel).squeeze(0).squeeze(0)
            )  # (T - k_len,)
        else:
            conv_out = token_log_probs.squeeze(0)

        return token_log_probs.squeeze(0), conv_out


def create_grpo_dataset_good_splits(
    tokenizer,
    samples_per_doc: int = 1,
    min_prefix_len: int = 128,
    max_prefix_len: int = 2048,
    continuation_length: int = 16,
    dataset_path: str = "open-web-math/open-web-math",
    seed: int = 0,
    force_open_think: bool = True,
    scorer_model: Any = None,
    scorer_device: str | None = None,
    min_logprob: float | None = -4.5,
    max_logprob: float | None = -1.0,
    score_window_len: int = 16,
    max_rolling_logprob: float | None = -0.8,
    filter_multidomain: bool = True,
    split: str = "train",
    num_proc: int = 8,
    cache_dir: str = "/scratch/datasets/openwebmath",
    shuffle_buffer_size: int | None = None,
) -> IterableDataset:
    if isinstance(dataset_path, (Dataset, IterableDataset)):
        raw_data: Any = dataset_path
    elif os.path.exists(dataset_path):
        raw_data = load_from_disk(dataset_path).shuffle(seed=seed)
    else:
        raw_data = load_dataset(
            dataset_path,
            split=split,
            num_proc=num_proc,
            cache_dir=cache_dir,
        ).shuffle(seed=seed)

    if scorer_device is not None:
        effective_device = scorer_device
    elif scorer_model is not None:
        try:
            effective_device = next(scorer_model.parameters()).device
        except Exception:
            effective_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        effective_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    generator = torch.Generator().manual_seed(seed)

    def gen():
        for item in raw_data:
            doc_text = item["text"]
            if not doc_text:
                continue

            ascii_and_math = sum(
                1
                for c in doc_text
                if ord(c) < 128
                or 0x0370 <= ord(c) <= 0x03FF
                or 0x2100 <= ord(c) <= 0x22FF
            )
            if (ascii_and_math / len(doc_text)) < 0.85:
                continue

            tokenized_doc = tokenizer(
                doc_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_prefix_len + continuation_length + 256,
            )["input_ids"]
            token_ids = _int_ids(tokenized_doc)
            n = len(token_ids)
            max_split_point = min(n - continuation_length, max_prefix_len)

            if max_split_point <= min_prefix_len + 1:
                continue

            candidate_splits = []
            if filter_multidomain:
                char_splits = find_multidomain_splits(doc_text)
                seen_splits = set()
                for char_idx in char_splits:
                    prefix_ids = tokenizer(
                        doc_text[:char_idx], add_special_tokens=False
                    )["input_ids"]
                    plen = len(prefix_ids)
                    if (
                        min_prefix_len + 1 <= plen <= max_split_point
                        and plen not in seen_splits
                    ):
                        seen_splits.add(plen)
                        candidate_splits.append(plen)
            else:
                num_rand = max(samples_per_doc * 10, 50)
                candidate_splits = torch.randint(
                    min_prefix_len + 1,
                    max_split_point,
                    (num_rand,),
                    generator=generator,
                ).tolist()

            if not candidate_splits:
                continue

            valid_splits = []
            if scorer_model is not None and (
                min_logprob is not None
                or max_logprob is not None
                or max_rolling_logprob is not None
            ):
                scoring_result = score_document(
                    scorer_model,
                    token_ids,
                    scorer_device=effective_device,
                    score_window_len=score_window_len,
                )
                if scoring_result is not None:
                    token_lps, conv_lps = scoring_result
                    for sp in candidate_splits:
                        target_idx = sp - 1
                        if target_idx < 0 or target_idx >= len(token_lps):
                            continue

                        first_tok_lp = token_lps[target_idx].item()
                        if (
                            min_logprob is not None
                            and first_tok_lp < min_logprob
                        ):
                            continue
                        if (
                            max_logprob is not None
                            and first_tok_lp > max_logprob
                        ):
                            continue

                        if max_rolling_logprob is not None and target_idx < len(
                            conv_lps
                        ):
                            rolling_mean = conv_lps[target_idx].item()
                            if rolling_mean > max_rolling_logprob:
                                continue

                        valid_splits.append(sp)
                else:
                    valid_splits = candidate_splits
            else:
                valid_splits = candidate_splits

            if not valid_splits:
                continue

            shuffled_indices = torch.randperm(
                len(valid_splits), generator=generator
            ).tolist()
            yielded_for_doc = 0
            for idx in shuffled_indices:
                split_point = valid_splits[idx]
                prefix_ids = token_ids[:split_point]
                continuation_ids = token_ids[
                    split_point : split_point + continuation_length
                ]
                row = _grpo_row(
                    tokenizer,
                    prefix_ids,
                    continuation_ids,
                    force_open_think=force_open_think,
                )
                if row is not None:
                    yield row
                    yielded_for_doc += 1
                    if yielded_for_doc >= samples_per_doc:
                        break

    dataset = IterableDataset.from_generator(gen)
    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(
            seed=seed,
            buffer_size=shuffle_buffer_size,
            max_buffer_input_shards=1,
        )
    return dataset


def create_grpo_overfit_dataset(
    tokenizer,
    samples_per_doc: int,
    min_prefix_len: int,
    max_prefix_len: int,
    continuation_length: int,
    dataset_path: str,
    seed: int,
    num_samples: int = 8,
    force_open_think: bool = True,
) -> Dataset:
    raw_data: Any = load_from_disk(dataset_path).shuffle(seed=seed)
    generator = torch.Generator().manual_seed(seed)

    samples = []
    for item in raw_data:
        text = item["text"]
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        n = len(token_ids)
        max_split_point = min(n - continuation_length, max_prefix_len)

        if max_split_point <= min_prefix_len + 1:
            continue

        split_points = torch.randint(
            min_prefix_len + 1,
            max_split_point,
            (samples_per_doc,),
            generator=generator,
        ).tolist()

        for split_point in split_points:
            prefix_ids = token_ids[:split_point]
            continuation_ids = token_ids[
                split_point : split_point + continuation_length
            ]
            row = _grpo_row(
                tokenizer,
                prefix_ids,
                continuation_ids,
                force_open_think=force_open_think,
            )
            if row is None:
                continue
            samples.append(row)
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    return Dataset.from_list(samples)


def create_grpo_dataset_full(
    tokenizer,
    samples_per_doc: int,
    min_prefix_len: int,
    max_prefix_len: int,
    continuation_length: int,
    dataset_path: str = "open-web-math/open-web-math",
    seed: int = 0,
    split: str = "train",
    cache_dir: str = "/scratch/datasets/openwebmath",
    force_open_think: bool = True,
    shuffle_buffer_size: int = 5_000,
) -> IterableDataset:
    tokenizer = copy.deepcopy(tokenizer)
    raw_data: Any = _load_streaming_corpus(dataset_path, split, cache_dir, seed)
    generator = torch.Generator().manual_seed(seed)

    def gen():
        for item in raw_data:
            text = item["text"]
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_prefix_len + continuation_length,
            )["input_ids"]
            n = len(token_ids)
            max_split_point = min(n - continuation_length, max_prefix_len)

            if max_split_point <= min_prefix_len + 1:
                continue

            split_points = torch.randint(
                min_prefix_len + 1,
                max_split_point,
                (samples_per_doc,),
                generator=generator,
            ).tolist()

            for split_point in split_points:
                prefix_ids = token_ids[:split_point]
                continuation_ids = token_ids[
                    split_point : split_point + continuation_length
                ]
                row = _grpo_row(
                    tokenizer,
                    prefix_ids,
                    continuation_ids,
                    force_open_think=force_open_think,
                )
                if row is not None:
                    yield row

    return IterableDataset.from_generator(gen).shuffle(
        seed=seed,
        buffer_size=shuffle_buffer_size,
        max_buffer_input_shards=1,
    )


def create_grpo_overfit_dataset_full(
    tokenizer,
    samples_per_doc: int,
    min_prefix_len: int,
    max_prefix_len: int,
    continuation_length: int,
    dataset_path: str = "open-web-math/open-web-math",
    seed: int = 0,
    num_samples: int = 8,
    split: str = "train",
    num_proc: int = 8,
    cache_dir: str = "/scratch/datasets/openwebmath",
    force_open_think: bool = True,
) -> Dataset:
    raw_data: Any = load_dataset(
        dataset_path,
        split=split,
        num_proc=num_proc,
        cache_dir=cache_dir,
    ).shuffle(seed=seed)
    generator = torch.Generator().manual_seed(seed)

    samples = []
    for item in raw_data:
        text = item["text"]
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        n = len(token_ids)
        max_split_point = min(n - continuation_length, max_prefix_len)

        if max_split_point <= min_prefix_len + 1:
            continue

        split_points = torch.randint(
            min_prefix_len + 1,
            max_split_point,
            (samples_per_doc,),
            generator=generator,
        ).tolist()

        for split_point in split_points:
            prefix_ids = token_ids[:split_point]
            continuation_ids = token_ids[
                split_point : split_point + continuation_length
            ]
            row = _grpo_row(
                tokenizer,
                prefix_ids,
                continuation_ids,
                force_open_think=force_open_think,
            )
            if row is None:
                continue
            samples.append(row)
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    return Dataset.from_list(samples)


def create_grpo_base_dataset(
    tokenizer,
    samples_per_doc: int,
    min_prefix_len: int,
    max_prefix_len: int,
    continuation_length: int,
    dataset_path: str,
    seed: int,
    instruction: str,
) -> IterableDataset:
    raw_data: Any = load_from_disk(dataset_path).shuffle(seed=seed)
    generator = torch.Generator().manual_seed(seed)

    def gen():
        for item in raw_data:
            text = item["text"]
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            n = len(token_ids)
            max_split_point = min(n - continuation_length, max_prefix_len)

            if max_split_point <= min_prefix_len + 1:
                continue

            split_points = torch.randint(
                min_prefix_len + 1,
                max_split_point,
                (samples_per_doc,),
                generator=generator,
            ).tolist()

            for split_point in split_points:
                prefix_ids = token_ids[:split_point]
                continuation_ids = token_ids[
                    split_point : split_point + continuation_length
                ]
                row = _grpo_row(
                    tokenizer,
                    prefix_ids,
                    continuation_ids,
                    instruction=instruction,
                )
                if row is not None:
                    yield row

    return IterableDataset.from_generator(gen)


def create_grpo_base_overfit_dataset(
    tokenizer,
    samples_per_doc: int,
    min_prefix_len: int,
    max_prefix_len: int,
    continuation_length: int,
    dataset_path: str,
    seed: int,
    instruction: str,
    num_samples: int = 8,
) -> Dataset:
    raw_data: Any = load_from_disk(dataset_path).shuffle(seed=seed)
    generator = torch.Generator().manual_seed(seed)

    samples = []
    for item in raw_data:
        text = item["text"]
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        n = len(token_ids)
        max_split_point = min(n - continuation_length, max_prefix_len)

        if max_split_point <= min_prefix_len + 1:
            continue

        split_points = torch.randint(
            min_prefix_len + 1,
            max_split_point,
            (samples_per_doc,),
            generator=generator,
        ).tolist()

        for split_point in split_points:
            prefix_ids = token_ids[:split_point]
            continuation_ids = token_ids[
                split_point : split_point + continuation_length
            ]
            row = _grpo_row(
                tokenizer,
                prefix_ids,
                continuation_ids,
                instruction=instruction,
            )
            if row is None:
                continue
            samples.append(row)
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    return Dataset.from_list(samples)


def create_grpo_base_dataset_full(
    tokenizer,
    samples_per_doc: int,
    min_prefix_len: int,
    max_prefix_len: int,
    continuation_length: int,
    dataset_path: str = "open-web-math/open-web-math",
    seed: int = 0,
    instruction: str = "",
    split: str = "train",
    cache_dir: str = "/scratch/datasets/openwebmath",
    shuffle_buffer_size: int = 5_000,
) -> IterableDataset:
    tokenizer = copy.deepcopy(tokenizer)
    raw_data: Any = _load_streaming_corpus(dataset_path, split, cache_dir, seed)
    generator = torch.Generator().manual_seed(seed)

    def gen():
        for item in raw_data:
            text = item["text"]
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_prefix_len + continuation_length,
            )["input_ids"]
            n = len(token_ids)
            max_split_point = min(n - continuation_length, max_prefix_len)

            if max_split_point <= min_prefix_len + 1:
                continue

            split_points = torch.randint(
                min_prefix_len + 1,
                max_split_point,
                (samples_per_doc,),
                generator=generator,
            ).tolist()

            for split_point in split_points:
                prefix_ids = token_ids[:split_point]
                continuation_ids = token_ids[
                    split_point : split_point + continuation_length
                ]
                row = _grpo_row(
                    tokenizer,
                    prefix_ids,
                    continuation_ids,
                    instruction=instruction,
                )
                if row is not None:
                    yield row

    return IterableDataset.from_generator(gen).shuffle(
        seed=seed,
        buffer_size=shuffle_buffer_size,
        max_buffer_input_shards=1,
    )


def create_grpo_base_overfit_dataset_full(
    tokenizer,
    samples_per_doc: int,
    min_prefix_len: int,
    max_prefix_len: int,
    continuation_length: int,
    dataset_path: str = "open-web-math/open-web-math",
    seed: int = 0,
    instruction: str = "",
    num_samples: int = 8,
    split: str = "train",
    num_proc: int = 8,
    cache_dir: str = "/scratch/datasets/openwebmath",
) -> Dataset:
    raw_data: Any = load_dataset(
        dataset_path,
        split=split,
        num_proc=num_proc,
        cache_dir=cache_dir,
    ).shuffle(seed=seed)
    generator = torch.Generator().manual_seed(seed)

    samples = []
    for item in raw_data:
        text = item["text"]
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        n = len(token_ids)
        max_split_point = min(n - continuation_length, max_prefix_len)

        if max_split_point <= min_prefix_len + 1:
            continue

        split_points = torch.randint(
            min_prefix_len + 1,
            max_split_point,
            (samples_per_doc,),
            generator=generator,
        ).tolist()

        for split_point in split_points:
            prefix_ids = token_ids[:split_point]
            continuation_ids = token_ids[
                split_point : split_point + continuation_length
            ]
            row = _grpo_row(
                tokenizer,
                prefix_ids,
                continuation_ids,
                instruction=instruction,
            )
            if row is None:
                continue
            samples.append(row)
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    return Dataset.from_list(samples)


class TrainingDataset:
    def __init__(
        self,
        tokenizer,
        seed: int = 0,
        dataset_path: str = "open-web-math/open-web-math",
        split: str = "train",
        num_proc: int = 8,
        cache_dir: str = "/scratch/datasets/openwebmath",
    ):
        self.data = load_dataset(
            dataset_path,
            split=split,
            num_proc=num_proc,
            cache_dir=cache_dir,
        ).shuffle(seed=seed)
        self.tokenizer = tokenizer
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.data)

    def get_document_batch(self, starting_doc_index: int, num_docs: int) -> list[str]:
        return self.data[starting_doc_index : starting_doc_index + num_docs]["text"]

    def get_prefix_continuation_pairs(
        self,
        starting_doc_index: int,
        num_docs: int,
        num_samples_per_doc: int,
        min_prefix_len: int,
        continuation_length: int,
        max_prefix_len: int,
    ) -> tuple[list[list[int]], list[list[int]]]:
        docs = self.get_document_batch(
            starting_doc_index=starting_doc_index, num_docs=num_docs
        )
        tokenized_docs = self.tokenizer(docs, add_special_tokens=False)["input_ids"]
        all_prefixes = []
        all_continuations = []

        for tokenized_doc in tokenized_docs:
            token_ids = _int_ids(tokenized_doc)
            n = len(token_ids)
            max_split_point = min(n - continuation_length, max_prefix_len)

            if max_split_point <= min_prefix_len + 1:
                continue

            # this indicates the points before which is the prefix
            # and split_point:split_point + continuation_length is the answer
            split_points = torch.randint(
                min_prefix_len + 1,
                max_split_point,
                (num_samples_per_doc,),
                generator=self.generator,
            ).tolist()
            for split_point in split_points:
                prefix_ids = token_ids[:split_point]
                if _decode_exact(self.tokenizer, prefix_ids) is None:
                    continue
                all_prefixes.append(prefix_ids)
                all_continuations.append(
                    token_ids[split_point : split_point + continuation_length]
                )
        return all_prefixes, all_continuations


def is_url_or_metadata(text: str, char_idx: int) -> bool:
    surrounding = text[max(0, char_idx - 35) : min(len(text), char_idx + 15)]

    # url
    url_substrings = (
        "http://",
        "https://",
        "www.",
        "?share=",
        "?Num=",
        "?v=",
        "?id=",
        "?p=",
        ".org/",
        ".com/",
    )
    if any(p in surrounding for p in url_substrings):
        return True
    if re.search(r"(https?://\S*|www\.\S*)", surrounding):
        return True

    # bibtex
    if re.search(
        r"(?i)\b(abstractnote|abstract|title|author|journal|volume|pages|doi|isbn|eprint|biburl)\s*=",
        surrounding,
    ):
        return True

    if re.search(r"(?i)\b(mode|gid|uid|pid|chmod|chown)\s*=", surrounding):
        return True

    return False


def find_multidomain_splits(text: str) -> list[int]:
    """
    Finds split character offsets across LaTeX Math, Code/Expressions, and Plaintext Logic,
    filtered to exclude URLs, BibTeX citations, and config metadata.
    """
    splits = []

    # math
    math_pattern = re.compile(
        r"(\$\$(.+?)\$\$|\$(.+?)\$|\\\[(.+?)\\\]|\\begin\{([a-z*]+)\}(.+?)\\end\{\5\})",
        re.DOTALL,
    )
    latex_rel_pattern = re.compile(
        r"(=|\\approx|\\le|\\ge|\\equiv|\\implies|\\to|\\sim|\\propto)"
    )

    for match in math_pattern.finditer(text):
        block_start = match.start()
        block_text = match.group(0)
        for rel_match in latex_rel_pattern.finditer(block_text):
            op_end = rel_match.end()
            while op_end < len(block_text) and block_text[op_end] == " ":
                op_end += 1
            remaining = block_text[op_end:].strip("$\n\\] ")
            if len(remaining) > 3:
                char_idx = block_start + op_end
                if not is_url_or_metadata(text, char_idx):
                    splits.append(char_idx)

    # code
    code_pattern = re.compile(
        r"(return\s+|assert\s+|yield\s+|->\s+|=>\s+|(?<=\w)\s*(?<!=)(=|\+=|-=|\*=|/=|==|!=|<=|>=)(?!=)\s*)"
    )
    for match in code_pattern.finditer(text):
        char_idx = match.end()
        if not is_url_or_metadata(text, char_idx):
            if not any(s_idx - 5 <= char_idx <= s_idx + 5 for s_idx in splits):
                splits.append(char_idx)

    # other
    logic_pattern = re.compile(
        r"(?:^|[.\n;]\s*|\bSince\b.+?,\s*)(Therefore,\s*|Thus,\s*|Hence,\s*|We get:\s*|We have:\s*|It follows that\s+|it follows that\s+|solving for\s+[a-zA-Z0-9_]+\s*,\s*we get\s*)"
    )
    for match in logic_pattern.finditer(text):
        char_idx = match.end()
        if not is_url_or_metadata(text, char_idx):
            if not any(s_idx - 5 <= char_idx <= s_idx + 5 for s_idx in splits):
                splits.append(char_idx)

    return splits


class ReasoningSplitsDataset:
    def __init__(
        self,
        tokenizer,
        scorer_model: Any = None,
        scorer_device: str = "cuda:0",
        seed: int = 0,
        dataset_path: str = "open-web-math/open-web-math",
        split: str = "train",
        num_proc: int = 8,
        cache_dir: str = "/scratch/datasets/openwebmath",
        min_logprob: float | None = -4.5,
        max_logprob: float | None = -1.0,
        score_window_len: int = 16,
        max_rolling_logprob: float | None = -0.8,
        filter_multidomain: bool = True,
    ):
        if os.path.exists(dataset_path):
            self.data = load_from_disk(dataset_path).shuffle(seed=seed)
        else:
            self.data = load_dataset(
                dataset_path,
                split=split,
                num_proc=num_proc,
                cache_dir=cache_dir,
            ).shuffle(seed=seed)
        self.tokenizer = tokenizer
        self.scorer_model = scorer_model
        self.scorer_device = scorer_device
        self.generator = torch.Generator().manual_seed(seed)
        self.min_logprob = min_logprob
        self.max_logprob = max_logprob
        self.score_window_len = score_window_len
        self.max_rolling_logprob = max_rolling_logprob
        self.filter_multidomain = filter_multidomain

    def __len__(self):
        return len(self.data)

    def get_document_batch(self, starting_doc_index: int, num_docs: int) -> list[str]:
        return self.data[starting_doc_index : starting_doc_index + num_docs]["text"]

    def score_document(
        self, token_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.scorer_model is None:
            return None

        device = self.scorer_device
        full_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = self.scorer_model(input_ids=full_ids)
            logits = outputs.logits[:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            targets = full_ids[:, 1:]
            token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(
                -1
            )  # (1, T-1)

            if self.score_window_len > 1:
                k_len = self.score_window_len
                kernel = (
                    torch.ones(1, 1, k_len, device=device, dtype=token_log_probs.dtype)
                    / k_len
                )
                conv_out = (
                    F.conv1d(token_log_probs.unsqueeze(1), kernel).squeeze(0).squeeze(0)
                )  # (T - k_len,)
            else:
                conv_out = token_log_probs.squeeze(0)

            return token_log_probs.squeeze(0), conv_out

    def get_prefix_continuation_pairs(
        self,
        starting_doc_index: int,
        num_docs: int,
        num_samples_per_doc: int,
        min_prefix_len: int,
        continuation_length: int,
        max_prefix_len: int,
    ) -> tuple[list[list[int]], list[list[int]]]:
        docs = self.get_document_batch(
            starting_doc_index=starting_doc_index, num_docs=num_docs
        )
        all_prefixes = []
        all_continuations = []

        for doc_text in docs:
            if not doc_text:
                continue
            ascii_and_math = sum(
                1
                for c in doc_text
                if ord(c) < 128
                or 0x0370 <= ord(c) <= 0x03FF
                or 0x2100 <= ord(c) <= 0x22FF
            )
            if (ascii_and_math / len(doc_text)) < 0.85:
                continue

            tokenized_doc = self.tokenizer(
                doc_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_prefix_len + continuation_length + 256,
            )["input_ids"]
            token_ids = _int_ids(tokenized_doc)
            n = len(token_ids)
            max_split_point = min(n - continuation_length, max_prefix_len)

            if max_split_point <= min_prefix_len + 1:
                continue

            candidate_splits = []
            if self.filter_multidomain:
                char_splits = find_multidomain_splits(doc_text)
                seen_splits = set()
                for char_idx in char_splits:
                    prefix_ids = self.tokenizer(
                        doc_text[:char_idx], add_special_tokens=False
                    )["input_ids"]
                    plen = len(prefix_ids)
                    if (
                        min_prefix_len + 1 <= plen <= max_split_point
                        and plen not in seen_splits
                    ):
                        seen_splits.add(plen)
                        candidate_splits.append(plen)
            else:
                num_rand = max(num_samples_per_doc * 10, 50)
                candidate_splits = torch.randint(
                    min_prefix_len + 1,
                    max_split_point,
                    (num_rand,),
                    generator=self.generator,
                ).tolist()

            if not candidate_splits:
                continue

            valid_splits = []
            if self.scorer_model is not None and (
                self.min_logprob is not None
                or self.max_logprob is not None
                or self.max_rolling_logprob is not None
            ):
                scoring_result = self.score_document(token_ids)
                if scoring_result is not None:
                    token_lps, conv_lps = scoring_result
                    for sp in candidate_splits:
                        target_idx = sp - 1
                        if target_idx < 0 or target_idx >= len(token_lps):
                            continue

                        first_tok_lp = token_lps[target_idx].item()
                        if (
                            self.min_logprob is not None
                            and first_tok_lp < self.min_logprob
                        ):
                            continue
                        if (
                            self.max_logprob is not None
                            and first_tok_lp > self.max_logprob
                        ):
                            continue

                        if self.max_rolling_logprob is not None and target_idx < len(
                            conv_lps
                        ):
                            rolling_mean = conv_lps[target_idx].item()
                            if rolling_mean > self.max_rolling_logprob:
                                continue

                        valid_splits.append(sp)
                else:
                    valid_splits = candidate_splits
            else:
                valid_splits = candidate_splits

            if not valid_splits:
                continue

            shuffled_indices = torch.randperm(
                len(valid_splits), generator=self.generator
            ).tolist()
            chosen_splits = [
                valid_splits[idx] for idx in shuffled_indices[:num_samples_per_doc]
            ]

            for split_point in chosen_splits:
                prefix_ids = token_ids[:split_point]
                if _decode_exact(self.tokenizer, prefix_ids) is None:
                    continue
                all_prefixes.append(prefix_ids)
                all_continuations.append(
                    token_ids[split_point : split_point + continuation_length]
                )

        return all_prefixes, all_continuations
