from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from load_data import TrainingDataset
import gc
import json
import os
import random


def main():
    teacher_model_path = "/scratch/hub/gemma4_31b/models--Intel--gemma-4-31B-it-int4-AutoRound/snapshots/a428c96a57976947b0f12735f0cf5fcae69019ad"
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)

    max_model_len = 8192
    max_tokens = 2048
    prepended_think_tag = "<think>"
    system_prompt = """
You are a text predictor whose entire goal is to produce the correct continuation of a given document.
Before doing so, you are allowed to produce thinking to help you inside <think>thinking text here to help you sound out loud</think> (though this is not mandatory, if you feel it won't help your predictions then just produce <think></think>prediction here, i.e. close the tag instantly).
After your thinking is done, you will predict the correct continuation of the document, and your predictions should be as accurate as possible, as they will be scored. Any text produced as your internal reasoning will not be assessed, so use it as your scratchpad if you feel it will help (you may find it helpful to do a sort of step-by-step and logical reasoning).
Predict at least a few sentences of the document continuation (all documents given to you will have at least a few sentences to predict afterwards). Remember that any text you produce after the closing think tag will be assessed, and if you forget to close it, your prediction wont count; i.e. remember to close it!
"""
    mode_prompts = {
        "default": "",
        "few_words": "\nYou only have a few words (roughly a medium length sentence) to think, past which you will be penalized, so keep your reasoning short and include only the main key ideas.",
        "couple_sentences": "\nYou only have a couple of sentences (roughly 3-4 sentences) to think, past which you will be penalized, so keep your reasoning concise.",
    }
    system_prompt_len = len(tokenizer.encode(system_prompt))
    max_prefix_len = (
        max_model_len - max_tokens - system_prompt_len - 256
    )  # jic, accounts for mode_prompts
    filename = "./teacher_traces.jsonl"
    total_num_docs = 5_000
    B = 100
    num_samples_per_doc = 2
    min_prefix_len = 128
    continuation_length = 64

    teacher = LLM(
        teacher_model_path,
        tensor_parallel_size=4,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.6,
        top_p=0.9,
        max_tokens=max_tokens,
        stop=["</think>"],
        include_stop_str_in_output=True,
    )

    training_dataset = TrainingDataset(tokenizer)
    mode_rng = random.Random(0)

    with open(filename, "a", encoding="utf-8") as f:
        for starting_doc_index in range(0, total_num_docs, B):
            batch_num_docs = min(B, total_num_docs - starting_doc_index)
            tokenized_prefixes, tokenized_continuations = (
                training_dataset.get_prefix_continuation_pairs(
                    starting_doc_index=starting_doc_index,
                    num_docs=batch_num_docs,
                    num_samples_per_doc=num_samples_per_doc,
                    min_prefix_len=min_prefix_len,
                    continuation_length=continuation_length,
                    max_prefix_len=max_prefix_len,
                )
            )
            unformatted_prompts = tokenizer.batch_decode(
                tokenized_prefixes, skip_special_tokens=True
            )
            continuations = tokenizer.batch_decode(tokenized_continuations)

            thinking_modes = mode_rng.choices(
                ["nothink", "default", "few_words", "couple_sentences"],
                weights=[0.025, 0.225, 0.375, 0.375],
                k=len(unformatted_prompts),
            )
            generated_prompt_indices = [
                prompt_idx
                for prompt_idx, mode in enumerate(thinking_modes)
                if mode != "nothink"
            ]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": system_prompt
                            + mode_prompts[thinking_modes[prompt_idx]],
                        },
                        {"role": "user", "content": unformatted_prompts[prompt_idx]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                + prepended_think_tag
                for prompt_idx in generated_prompt_indices
            ]
            outputs = teacher.generate(prompts, sampling_params) if prompts else []
            outputs_by_prompt_idx = dict(zip(generated_prompt_indices, outputs))

            for prompt_idx, mode in enumerate(thinking_modes):
                if mode == "nothink":
                    trajectory_texts = ["<think></think>"]
                else:
                    trajectory_texts = [
                        prepended_think_tag + request_output.text
                        for request_output in outputs_by_prompt_idx[prompt_idx].outputs
                    ]

                for trajectory_text in trajectory_texts:
                    if (
                        trajectory_text.count("<think>") != 1
                        or trajectory_text.count("</think>") != 1
                        or not trajectory_text.endswith("</think>")
                    ):
                        continue

                    f.write(
                        json.dumps(
                            {
                                "prefix": unformatted_prompts[prompt_idx],
                                "teacher_thinking_trace": trajectory_text,
                                "continuation": continuations[prompt_idx],
                                "thinking_mode": mode,
                            }
                        )
                        + "\n"
                    )

            f.flush()
            print(
                f"Finished documents {starting_doc_index} through "
                f"{starting_doc_index + batch_num_docs - 1}",
                flush=True,
            )

    del teacher
    gc.collect()


if __name__ == "__main__":
    main()
    os._exit(0)
