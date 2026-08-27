from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from load_data import TrainingDataset
import gc
import json
import os


def main():
    teacher_model_path = "/scratch/hub/gemma4_31b/models--Intel--gemma-4-31B-it-int4-AutoRound/snapshots/a428c96a57976947b0f12735f0cf5fcae69019ad"
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    student_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B-Base", cache_dir="/scratch/hub"
    )

    max_model_len = 8192
    max_tokens = 2048
    prepended_think_tag = "<think>"
    system_prompt = """
Your goal is to produce thinking traces that would help someone go from the prefix to the continuation WITHOUT having seen the continuation and only having seen the prefix.
You will be given document prefixes and the true continuation, and your job is to produce these thinking traces, in between <think> and </think>.
Note: your job is to do this for someone who has NOT seen the actual continuation, ergo you should not mention that you know / have seen the true continuation.
Your reasoning should serve as a strong bridge, such that someone who has seen the prefix and your thinking should have a good chance at predicting what comes next without having seen it.
Do NOT just mention that you know the true continuation, instead you should be using it as a hint to guide you into better reasoning, not as a final answer.
Feel free to think as long as you'd like, but be specific and to the point. Your thinking should include drafts of a few potential continuations. No rambling about meta-commentary or useless things. It is okay if your thinking is highly information dense.
"""
    system_prompt_len = len(tokenizer.encode(system_prompt))
    max_prefix_len = max_model_len - max_tokens - system_prompt_len - 256  # jic buffer
    filename = f"./teacher_traces_new.jsonl"
    total_num_docs = 5000
    B = 100
    num_samples_per_doc = 1
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

    # Slice documents in the student's vocabulary so the saved IDs can be used
    # directly during SFT without a lossy decode/re-tokenize round trip.
    training_dataset = TrainingDataset(student_tokenizer)

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
            unformatted_prompts = student_tokenizer.batch_decode(tokenized_prefixes)
            continuations = student_tokenizer.batch_decode(tokenized_continuations)

            generated_prompt_indices = []
            prompts = []
            for prompt_idx, unformatted_prompt in enumerate(unformatted_prompts):
                prompt = (
                    tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": f"Prefix:\n{unformatted_prompt}\n\nContinuation:\n{continuations[prompt_idx]}",
                            },
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    + prepended_think_tag
                )
                prompt_length = len(tokenizer(prompt)["input_ids"])
                if prompt_length + max_tokens > max_model_len:
                    continue
                generated_prompt_indices.append(prompt_idx)
                prompts.append(prompt)
            outputs = teacher.generate(prompts, sampling_params) if prompts else []

            for prompt_idx, output in zip(generated_prompt_indices, outputs):
                trajectory_texts = [
                    prepended_think_tag + request_output.text
                    for request_output in output.outputs
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
                                "prefix_ids": [
                                    int(t) for t in tokenized_prefixes[prompt_idx]
                                ],
                                "teacher_thinking_trace": trajectory_text,
                                "continuation": continuations[prompt_idx],
                                "continuation_ids": [
                                    int(t) for t in tokenized_continuations[prompt_idx]
                                ],
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
