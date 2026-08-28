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
    max_tokens = 4096
    system_prompt = """You are a reasoning trace generator for language model training.

You will be given:
- <prefix>: The context the model has seen so far.
- <target_continuation>: The ground-truth continuation that follows the prefix (shown to you ONLY as reference for the target direction).

### Your Goal
Generate a high-quality, dense <think>...</think> reasoning trace that prepares a reader to predict what comes next.
Because open-ended text has many possible paths, use <target_continuation> to know which topic/direction to explore, but your reasoning MUST be strictly causal and grounded in the prefix.

### Rules
1. No telepathy / future citations:
   - You MUST NOT mention, quote, or copy any unseen proper nouns, specific author names, forum usernames, random dates, exact coordinates, or future URLs from <target_continuation> unless they were already stated in <prefix> or you can cleanly derive them.
   - If the continuation introduces a new entity or arbitrary number, reason about the *general category*, *structural transition*, or *relevant formulas/principles*, never the arbitrary specific token. Only mention the specific token if you have a clear way to lead to it.
   - E.g. it's okay to mention a specific math number as the answer to a formula if you perform the computation in your thinking, but it isn't okay to predict a citation that you have no way of seeing coming from the prefix.
2. Use <target_continuation> as a guide, not an answer sheet:
   - Look at the continuation -> ask: "What clues, premises, or structural setups in the prefix foreshadowed this direction?" -> build your internal thought around those prefix clues.
3. No meta-prompt artifacts:
   - Never say "the continuation", "true continuation", "given continuation", "in the prompt", "the target text", or "the author continues".
   - Basically, you're going in from the POV of someone who has never seen <target_continuation>
   - Speak in the first person present tense as an internal monologue looking ONLY at the prefix (e.g. "The prefix establishes...", "I need to calculate...", "The next logical point to address is...").

### Examples

Example 1: Next Paper / Document in a List
- <prefix>: "...quantum cohomology of toric Fano manifolds. Moreover, this holds for any"
- <target_continuation>: "Fano manifold. Working paper Adler D., Gritsenko V. 2019..."
- Cheating: "The next entry will be a working paper by Adler D. and Gritsenko V. from 2019."
- Causal: "The sentence is cut off after 'any', which naturally completes to 'Fano manifold' by dropping the 'toric' restriction. Following this entry, the document pattern indicates another working paper listing will follow, likely continuing with algebraic geometry or related polynomial rings."

Example 2: Forum Post User Reply
- <prefix>: "Why does my code throw this index error? 5. May 2, 2008"
- <target_continuation>: "### rock.freak667\\n\\nYou forgot to..."
- Cheating: "The next post is by user rock.freak667."
- Causal: "The prefix ends at a new post header. A responder will likely reply to address the index out-of-bounds error, probably checking array length or the loop bounds."

### Output Format:
You can speak normally before <think> to plan your strategy, but once you emit <think>, your thinking trace will be recorded.

[SCRATCHPAD]
1. Target Direction: (What general direction/topic does the continuation take?)
2. Prefix Clues: (What dormant clues in the prefix support this direction?)
3. Forbidden Tokens: (What specific unseen names, numbers, or exact quotes from the continuation must NOT appear in the thought?)

<think>
[Your grounded, clue-driven reasoning trace here]
</think>"""
    system_prompt_len = len(tokenizer.encode(system_prompt))
    max_prefix_len = max_model_len - max_tokens - system_prompt_len - 256  # jic buffer
    filename = f"./teacher_traces/scratchpad_prompt_traces.jsonl"
    total_num_docs = 2500
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
                user_content = (
                    f"<prefix>\n{unformatted_prompt}\n</prefix>\n\n"
                    f"<target_continuation>\n{continuations[prompt_idx]}\n</target_continuation>"
                )
                prompt = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_length = len(tokenizer(prompt)["input_ids"])
                if prompt_length + max_tokens > max_model_len:
                    continue
                generated_prompt_indices.append(prompt_idx)
                prompts.append(prompt)
            outputs = teacher.generate(prompts, sampling_params) if prompts else []

            for prompt_idx, output in zip(generated_prompt_indices, outputs):
                for request_output in output.outputs:
                    raw_text = request_output.text
                    if "<think>" not in raw_text or "</think>" not in raw_text:
                        continue

                    inner_thought = (
                        raw_text.split("<think>")[1].split("</think>")[0].strip()
                    )
                    if not inner_thought:
                        continue
                    trajectory_text = f"<think>\n{inner_thought}\n</think>"

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
