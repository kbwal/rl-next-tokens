from typing import Any, cast
from datasets import IterableDataset, load_dataset, load_from_disk
import torch
from torch.nn.utils.rnn import pad_sequence


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


class TrainingDataset:
    def __init__(self, tokenizer, seed=0):
        self.data = load_dataset(
            "open-web-math/open-web-math",
            split="train",
            num_proc=8,
            cache_dir="/scratch/datasets/openwebmath",
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
