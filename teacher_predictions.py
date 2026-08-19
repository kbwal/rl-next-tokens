from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from load_data import TrainingDataset
import json
import os

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
system_prompt_len = len(tokenizer.encode(system_prompt))
max_prefix_len = max_model_len - max_tokens - system_prompt_len - 32  # jic
filename = "./teacher_traces.jsonl"

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
tokenized_prefixes, tokenized_continuations = (
    training_dataset.get_prefix_continuation_pairs(0, 4, 2, 256, 64, max_prefix_len)
)
unformatted_prompts = tokenizer.batch_decode(
    tokenized_prefixes, skip_special_tokens=True
)
continuations = tokenizer.batch_decode(tokenized_continuations)

prompts = [
    tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    + prepended_think_tag
    for prompt in unformatted_prompts
]
outputs = teacher.generate(prompts, sampling_params)
with open(filename, "a", encoding="utf-8") as f:
    for prompt_idx, output in enumerate(outputs):
        for request_output in output.outputs:
            f.write(
                json.dumps(
                    {
                        "prefix": unformatted_prompts[prompt_idx],
                        "teacher_thinking_trace": prepended_think_tag
                        + request_output.text,
                        "continuation": continuations[prompt_idx],
                    }
                )
                + "\n"
            )
            trajectory_text = prepended_think_tag + request_output.text
            token_ids = request_output.token_ids

del teacher
import gc
import os

gc.collect()
os._exit(0)
