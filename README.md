# RL-NTP: Reinforcement Learning for Next-Token Prediction

A project exploring whether language models can learn to "think" (`<think>...</think>`) before predicting next-token continuations during pretraining, without task-specific verifiers or ground-truth answer checkers.

Final runs were on [FineMath-4plus](https://huggingface.co/datasets/HuggingFaceTB/finemath). I originally tried [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), even with custom split points in documents, and it didn't work too well tbh.

---

## 1. The Direct Attempt: NVIDIA RLP Replication (`grpo_base.py`)

I first tried directly replicating NVIDIA's **RLP** (*Reinforcement as a Pretraining Objective*) on `Qwen/Qwen3-1.7B-Base`:
- Wrapped raw text prefixes in a prompt asking the model to think in `<think>...</think>` before continuing the text in the same style.
- Rewarded completions based on the continuation logprobs under the base model

**Result:** Pretty unsuccessful. In a full run of ~1k steps (32k unique prompts, 8 rollouts per prompt), rewards stayed pretty flat (if you squinted, it's possible it went from -16 to -15) with little policy improvement. The base model also doesn't spontaneously use the `<think>` tags, and the model doesn't discover how to do useful reasoning on its own (it basically stays in the base model regime, which is why the logprobs are -15).

I'm not sure how Nvidia solved this; my guesses are mostly scale right now (I think I'm at the point of scale where I should be ~2ooms below Nvidia).
They were able to afford much larger batch sizes than me, and this probably helps their convergence. I also ran this with continuation length of 768, and Nvidia mentioned that it only helped continuation logprobs over base b/w 512 and 1024 thinking tokens, so I might be roughly in the collapse regime anyways.

---

## 2. The Two-Stage Approach: SFT Warm-Start + GRPO (`grpo.py`)

Since cold-starting RL on a base model didn't pan out, I switched to a two-stage approach: teach the model *how* to format thoughts and structure scratchpads first via SFT, then let GRPO optimize the reasoning.

### Step 1: Data & Teacher Traces
- **[`precompute_good_splits.py`](precompute_good_splits.py)**: Random text splits usually land on trivial tokens (e.g. punctuation, stopwords). I score splits with the base model to find prefix/continuation boundaries where thinking actually has room to help (first-token logprob between -4.5 and -1.0). Note: I didn't do this until more recently, so some of my first commits still did the Nvidia way of choosing random splits (I found this led to too many tokens where thinking didn't help).
- **[`teacher_predictions.py`](teacher_predictions.py)**: Prompts a Gemma4-31B teacher to generate thinking traces conditioned on the prefix and continuation, and praying for no leakage.

### Step 2: SFT Warm-Start ([`sft_on_teacher_traces.py`](sft_on_teacher_traces.py))
- Trained a LoRA on `Qwen3-1.7B-Base` over the teacher traces (swept rank, lr, and loss weighting in [`sft_sweep.yml`](sft_sweep.yml), I basically always chose r=32, lr=3e-4, and alpha=1.0 for the next stage though).
- Loss combines thinking cross-entropy and continuation cross-entropy (`loss = loss_think + alpha * loss_continuation`).
- The model pretty reliably learned how to open/close `<think>` tags and write basic scratchpad reasoning.

### Step 3: On-Policy GRPO ([`grpo.py`](grpo.py))
- Merged the SFT adapter into the base model and initialized GRPO from it.
- Forced `<think>` at the end of the prompt and rewarded the model with:
  - **Continuation logprobs**: logprob of the next K ground-truth tokens (note: this is another difference from Nvidia! They used strictly K=1 for everything, whereas I've done runs with differing values of K between 1 and 64).
  - **Format penalty**: penalizes unclosed or repeated `<think>` tags (I'm not strictly sure if this is necessary! It seems to not matter too much, as I make `</think>` an EOS, so it should be auto-punished for not formatting right anyways)
  - **Length penalty**: this technically exists, so I should mention it, but in practice I'm setting this to 0 in all of my runs.
- **Result:** Better than direct RLP, but subject to pretty big failure modes. The biggest one I saw on OpenWebMath is length collapse. I'm suspecting this is due to the quality of the data, and nothink being a strong strange attractor basin that's hard to fall out of if a decent chunk of prompts aren't improved by thinking (i.e. the data quality isn't as good as I'd hope; the reason for switching to FineMath-4plus!)
- **Update:** FineMath-4plus didn't fix the length collapse either. BBH benchmarks (`eval_on_bbh.py`):
  - `multistep_arithmetic_two`: Base `-6.93` seq logprob -> SFT `+2.71` info gain (len 249) -> GRPO-300 `+1.61` info gain (len 197)
  - `boolean_expressions`: Base `-1.55` logprob -> SFT `+0.80` info gain (len 190) -> GRPO-300 `+0.66` info gain (len 138)
  - `dyck_languages`: Base `-2.62` seq logprob -> SFT `-1.18` info gain (len 257) -> GRPO-300 `-0.97` info gain (len 124)

---

## File Overview

- [`grpo_base.py`](grpo_base.py) - Direct base model GRPO (RLP replication attempt).
- [`precompute_good_splits.py`](precompute_good_splits.py) - Scores and filters challenging prefix/continuation splits.
- [`teacher_predictions.py`](teacher_predictions.py) - Generates causal teacher traces using Gemma-4-31B.
- [`sft_on_teacher_traces.py`](sft_on_teacher_traces.py) - LoRA SFT on teacher traces (`sft_sweep.yml` for sweeps).
- [`grpo.py`](grpo.py) - On-policy GRPO starting from merged SFT checkpoint.
- [`eval_on_bbh.py`](eval_on_bbh.py) - Multi-GPU evaluation script for BigBench-Hard tasks.
- [`load_data.py`](load_data.py) - Dataset streaming and split utilities.
