from typing import Any
from datasets import Dataset, IterableDataset, load_dataset, load_from_disk
import torch

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
    raw_data: Any = load_from_disk(dataset_path)
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
    raw_data: Any = load_from_disk(dataset_path)
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
) -> IterableDataset:
    raw_data: Any = load_dataset(
        dataset_path,
        split=split,
        cache_dir=cache_dir,
        streaming=True,
    )
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
    )
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
    raw_data: Any = load_from_disk(dataset_path)
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
    raw_data: Any = load_from_disk(dataset_path)
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
) -> IterableDataset:
    raw_data: Any = load_dataset(
        dataset_path,
        split=split,
        cache_dir=cache_dir,
        streaming=True,
    )
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
    )
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
        )
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
