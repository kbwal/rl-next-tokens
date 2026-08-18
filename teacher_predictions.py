from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

teacher_model_path = "/scratch/hub/gemma4_31b/models--Intel--gemma-4-31B-it-int4-AutoRound/snapshots/a428c96a57976947b0f12735f0cf5fcae69019ad"
tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)

teacher = LLM(
    teacher_model_path,
    tensor_parallel_size=4,
    max_model_len=2048,
    gpu_memory_utilization=0.8,
    dtype="bfloat16",
    trust_remote_code=True,
)
sampling_params = SamplingParams(
    n=1,
    temperature=0.6,
    top_p=0.9,
    max_tokens=1024,
    stop=["</think>"],
    include_stop_str_in_output=True,
)

raw_prompts = [
    "Holmes then saw the boy's eye blink. 3 fast, 3 slow, 3 fast. What does this mean? He heard a voice: it was "
]
prompts = [
    tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": """
          You are a text predictor whose entire goal is to produce the correct continuation of a given document.
          Before doing so, you are allowed to produce thinking to help you inside <think>thinking text here to help you sound out loud</think> (though this is not mandatory, if you feel it won't help your predictions then just produce <think></think>prediction here, i.e. close the tag instantly).
          After your thinking is done, you will predict the correct continuation of the document, and your predictions should be as accurate as possible, as they will be scored. Any text produced as your internal reasoning will not be assessed, so use it as your scratchpad if you feel it will help (you may find it helpful to do a sort of step-by-step and logical reasoning).
          Predict at least a few sentences of the document continuation (all documents given to you will have at least a few sentences to predict afterwards). Remember that any text you produce after the closing think tag will be assessed, and if you forget to close it, your prediction wont count; i.e. remember to close it!
          """,
            },
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in raw_prompts
]
outputs = teacher.generate(prompts, sampling_params)
for output in outputs:
    for i, request_output in enumerate(output.outputs):
        trajectory_text = request_output.text
        token_ids = request_output.token_ids
        print({trajectory_text.strip()})

del teacher
import gc
import os

gc.collect()
os._exit(0)
