# this file is mostly for me looking through the data
from datasets import load_dataset

data = load_dataset(
    "open-web-math/open-web-math",
    split="train",
    num_proc=8,
    cache_dir="/scratch/datasets/openwebmath",
)

print(data[0])
