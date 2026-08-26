from typing import Any, cast
from datasets import Dataset, IterableDataset, load_dataset, load_from_disk
import torch
from torch.nn.utils.rnn import pad_sequence

FORCE_OPEN_THINK_TAG = "<think>"


def format_grpo_base_prompt(prefix: str, instruction: str) -> str:
    return instruction.strip() + "\n\n" + prefix


def create_grpo_dataset(
    tokenizer,
    samples_per_doc: int,
    min_prefix_len: int,
    max_prefix_len: int,
    continuation_length: int,
    dataset_path: str,
    seed: int,
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
                yield {
                    "prompt": tokenizer.decode(prefix_ids),
                    "continuations": tokenizer.decode(continuation_ids),
                }

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
            samples.append(
                {
                    "prompt": tokenizer.decode(prefix_ids),
                    "continuations": tokenizer.decode(continuation_ids),
                }
            )
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
                prefix = tokenizer.decode(prefix_ids)
                yield {
                    "prompt": format_grpo_base_prompt(prefix, instruction),
                    "continuations": tokenizer.decode(continuation_ids),
                    "raw_prefix": prefix,
                }

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
            prefix = tokenizer.decode(prefix_ids)
            samples.append(
                {
                    "prompt": format_grpo_base_prompt(prefix, instruction),
                    "continuations": tokenizer.decode(continuation_ids),
                    "raw_prefix": prefix,
                }
            )
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
    num_proc: int = 8,
    cache_dir: str = "/scratch/datasets/openwebmath",
) -> IterableDataset:
    raw_data: Any = load_dataset(
        dataset_path,
        split=split,
        num_proc=num_proc,
        cache_dir=cache_dir,
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
                prefix = tokenizer.decode(prefix_ids)
                yield {
                    "prompt": format_grpo_base_prompt(prefix, instruction),
                    "continuations": tokenizer.decode(continuation_ids),
                    "raw_prefix": prefix,
                }

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
            prefix = tokenizer.decode(prefix_ids)
            samples.append(
                {
                    "prompt": format_grpo_base_prompt(prefix, instruction),
                    "continuations": tokenizer.decode(continuation_ids),
                    "raw_prefix": prefix,
                }
            )
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        docs = self.get_document_batch(
            starting_doc_index=starting_doc_index, num_docs=num_docs
        )
        tokenized_docs = self.tokenizer(docs)["input_ids"]
        all_prefixes = []
        all_continuations = []

        for tokenized_doc in tokenized_docs:
            token_ids = torch.tensor(tokenized_doc, dtype=torch.long)
            n = token_ids.shape[0]
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
            )
            for split_point in split_points:
                all_prefixes.append(token_ids[:split_point])
                all_continuations.append(
                    token_ids[split_point : split_point + continuation_length]
                )
        prefixes = pad_sequence(
            all_prefixes,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        continuations = torch.stack(all_continuations)
        return prefixes, continuations
