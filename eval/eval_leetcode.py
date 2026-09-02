#!/usr/bin/env python3
"""Evaluate base / SFT / GRPO checkpoints on a LeetCode medium/hard bank.

Protocol (matches GRPO training more closely than one-shot generation):
  1. Think-tagged models generate until ``</think>``, then code is generated
     as a second stage from that prefix.
  2. Scores are reported twice:
       - strict: valid think format, and only the post-think code is executed
       - oracle: AST/markdown salvage from the full sample, ignoring format
  3. Generation uses a fixed seed. The original LC3 / LC42 pair is pinned, and
     the remaining tasks are sampled from the medium/hard bank.

Stopping gate: abandon the current training recipe if the best GRPO checkpoint
fails to beat both SFT and base on macro strict pass@1.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import io
import json
import math
import os
import random
import re
import signal
import sys
import textwrap
import time
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from leetcode_problems import (
    PINNED_IDS,
    PROBLEM_BANK,
    problem_public_dict,
    results_equal,
    sample_problems,
    validate_problem_bank,
)

SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]*\|>")
MARKDOWN_RE = re.compile(r"```(?:python)?\s*\n?(.*?)(?:```|$)", re.DOTALL)
DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)


def has_method_def(text: str, method_name: str) -> bool:
    return (
        re.search(
            rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(method_name)}\s*\(",
            text,
        )
        is not None
    )


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def decode_keep_think(tokenizer, token_ids) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return SPECIAL_TOKEN_RE.sub("", text)


def first_stop_index(token_ids: list[int], stop_ids: set[int]) -> int | None:
    for i, tok in enumerate(token_ids):
        if int(tok) in stop_ids:
            return i
    return None


def think_format_status(thought: str, use_think: bool) -> dict:
    if not use_think:
        return {
            "closed_think": True,
            "extra_open_think": False,
            "close_count": 0,
            "format_ok": True,
        }
    extra_open = "<think>" in thought
    close_count = thought.count("</think>")
    closed = close_count == 1 and thought.rstrip().endswith("</think>")
    return {
        "closed_think": closed,
        "extra_open_think": extra_open,
        "close_count": close_count,
        "format_ok": closed and not extra_open,
    }


def try_parse(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def trim_until_parses(code: str) -> str | None:
    lines = code.splitlines()
    for end in range(len(lines), 0, -1):
        chunk = "\n".join(lines[:end])
        if try_parse(chunk) is not None:
            return chunk
    return None


def module_has_method(code: str, method_name: str) -> bool:
    tree = try_parse(code)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return True
    return False


def wrap_as_function(body: str, function_header: str) -> str:
    stripped = body.lstrip("\n")
    if not stripped.strip():
        return function_header + "\n    pass\n"
    first = stripped.splitlines()[0]
    if first.startswith("    ") or first.startswith("\t"):
        return function_header + "\n" + stripped
    return function_header + "\n" + textwrap.indent(stripped, "    ")


def candidate_snippets(text: str, method_name: str, function_header: str) -> list[str]:
    text = SPECIAL_TOKEN_RE.sub("", text)
    snippets = [text]
    for match in MARKDOWN_RE.finditer(text):
        block = match.group(1).strip()
        if block:
            snippets.append(block)

    no_think = re.sub(r"<think>.*?</think>", "\n", text, flags=re.DOTALL)
    if no_think != text:
        snippets.append(no_think)

    if "</think>" in text:
        snippets.append(text.split("</think>", 1)[1])

    for match in DEF_RE.finditer(text):
        snippets.append(text[match.start() :])

    out: list[str] = []
    seen: set[str] = set()
    for snippet in snippets:
        snippet = snippet.strip("\n")
        if not snippet or snippet in seen:
            continue
        seen.add(snippet)
        out.append(snippet)
        if not has_method_def(snippet, method_name) and "class Solution" not in snippet:
            wrapped = wrap_as_function(snippet, function_header)
            if wrapped not in seen:
                seen.add(wrapped)
                out.append(wrapped)
    return out


def isolate_executable(text: str, method_name: str, function_header: str) -> str | None:
    for snippet in candidate_snippets(text, method_name, function_header):
        trimmed = trim_until_parses(snippet)
        if trimmed is not None and module_has_method(trimmed, method_name):
            return trimmed
    return None


def extract_strict(
    code_region: str,
    method_name: str,
    function_header: str,
    format_ok: bool,
) -> tuple[str | None, str]:
    if not format_ok:
        return None, "strict: invalid think format"
    region = code_region
    if "<think>" in region:
        region = region.split("<think>", 1)[0]
    if "</think>" in region:
        region = region.split("</think>", 1)[0]
    region = SPECIAL_TOKEN_RE.sub("", region).strip("\n")
    if not region.strip():
        return None, "strict: empty code after </think>"
    code = isolate_executable(region, method_name, function_header)
    if code is None:
        return None, "strict: could not isolate a parseable function"
    return code, "ok"


def extract_oracle(
    full_text: str,
    code_region: str,
    method_name: str,
    function_header: str,
) -> tuple[str | None, str]:
    for source, label in (
        (code_region, "oracle: post-think"),
        (full_text, "oracle: full sample"),
    ):
        code = isolate_executable(source, method_name, function_header)
        if code is not None:
            return code, label
    return None, "oracle: no parseable function"


def classify_failure(detail: str) -> str:
    if detail.startswith("strict:"):
        return "format"
    if (
        "SyntaxError" in detail
        or "IndentationError" in detail
        or "could not isolate" in detail
    ):
        return "syntax"
    if "timed out" in detail or "Timeout" in detail:
        return "timeout"
    if detail.startswith("Runtime Error"):
        return "runtime"
    if detail.startswith("Failed on test"):
        return "wrong_answer"
    if "not found" in detail:
        return "interface"
    return "other"


def evaluate_code(
    code_str: str | None,
    test_cases: list[dict],
    method_name: str,
    compare: str,
    timeout_seconds: int = 2,
) -> tuple[bool, str, int, int]:
    total = len(test_cases)
    if code_str is None:
        return False, "No executable code", 0, total

    namespace: dict = {}
    old_handler = None
    has_alarm = hasattr(signal, "SIGALRM")
    passed = 0

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            if has_alarm:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)

            exec("from typing import List, Optional, Dict, Set, Tuple, Any", namespace)
            exec(code_str, namespace)

            func = None
            if method_name in namespace and callable(namespace[method_name]):
                func = namespace[method_name]
            elif "Solution" in namespace:
                try:
                    sol = namespace["Solution"]()
                except Exception as e:
                    return False, f"Error instantiating Solution class: {e}", 0, total
                if hasattr(sol, method_name):
                    func = getattr(sol, method_name)

            if func is None:
                return (
                    False,
                    f"Error: '{method_name}' function or 'Solution' class not found in generated code",
                    0,
                    total,
                )

            for i, case in enumerate(test_cases):
                args = copy.deepcopy(case["args"])
                res = func(*args)
                if not results_equal(res, case["expected"], compare):
                    return (
                        False,
                        f"Failed on test {i} (args={case['args']!r}): expected {case['expected']!r}, got {res!r}",
                        passed,
                        total,
                    )
                passed += 1
            return True, "All tests passed!", passed, total
        except TimeoutException as e:
            return False, f"Runtime Error: {e}", passed, total
        except Exception as e:
            return False, f"Runtime Error: {type(e).__name__}: {e}", passed, total
        finally:
            if has_alarm and old_handler is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unpadded_prompt_ids(input_ids, attention_mask) -> list[list[int]]:
    out = []
    for ids, mask in zip(input_ids, attention_mask):
        out.append([int(t) for t, m in zip(ids.tolist(), mask.tolist()) if m])
    return out


def generate_from_ids(
    model,
    tokenizer,
    prompt_ids: list[list[int]],
    max_new_tokens: int,
    eos_token_ids: list[int],
    seed: int,
    temperature: float,
    top_p: float,
) -> tuple[list[list[int]], list[str]]:
    set_seed(seed)
    device = model_device(model)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    max_len = max(len(x) for x in prompt_ids)
    padded = []
    attn = []
    for ids in prompt_ids:
        pad = max_len - len(ids)
        padded.append([pad_id] * pad + ids)
        attn.append([0] * pad + [1] * len(ids))
    enc = {
        "input_ids": torch.tensor(padded, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn, dtype=torch.long, device=device),
    }
    prompt_len = max_len
    stop_ids = {int(x) for x in eos_token_ids}
    eos_arg: int | list[int] = (
        eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids
    )

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
            eos_token_id=eos_arg,
        )

    gen = outputs[:, prompt_len:]
    kept: list[list[int]] = []
    texts: list[str] = []
    for i in range(gen.shape[0]):
        row = [int(t) for t in gen[i].tolist()]
        stop_at = first_stop_index(row, stop_ids)
        if stop_at is not None:
            row = row[: stop_at + 1]
        else:
            while row and row[-1] == pad_id:
                row.pop()
        kept.append(row)
        texts.append(decode_keep_think(tokenizer, row))
    return kept, texts


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    eos_token_ids: list[int],
    seed: int,
    temperature: float,
    top_p: float,
) -> tuple[list[list[int]], list[str], list[list[int]]]:
    set_seed(seed)
    device = model_device(model)
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = unpadded_prompt_ids(enc["input_ids"], enc["attention_mask"])
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]
    stop_ids = {int(x) for x in eos_token_ids}
    eos_arg: int | list[int] = (
        eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids
    )

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_arg,
        )

    gen = outputs[:, prompt_len:]
    kept: list[list[int]] = []
    texts: list[str] = []
    for i in range(gen.shape[0]):
        row = [int(t) for t in gen[i].tolist()]
        stop_at = first_stop_index(row, stop_ids)
        if stop_at is not None:
            row = row[: stop_at + 1]
        else:
            pad_id = tokenizer.pad_token_id
            if pad_id is not None:
                while row and row[-1] == pad_id:
                    row.pop()
        kept.append(row)
        texts.append(decode_keep_think(tokenizer, row))
    return kept, texts, prompt_ids


def generate_samples(
    model,
    tokenizer,
    prompt: str,
    n: int,
    use_think: bool,
    max_think_tokens: int,
    max_code_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> list[dict]:
    prompts = [prompt] * n
    eos_id = tokenizer.eos_token_id
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")

    if think_close_id is None or think_close_id == tokenizer.unk_token_id:
        raise RuntimeError(
            "Tokenizer does not contain a </think> token; two-stage eval cannot run"
        )

    if not use_think:
        ids, texts, _ = generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_code_tokens,
            eos_token_ids=[eos_id],
            seed=seed,
            temperature=temperature,
            top_p=top_p,
        )
        return [
            {
                "thought": "",
                "code_region": texts[i],
                "full_generated": texts[i],
                "thought_token_count": 0,
                "code_token_count": len(ids[i]),
                "closed_think": True,
            }
            for i in range(n)
        ]

    thought_ids, thought_texts, prompt_ids = generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_think_tokens,
        eos_token_ids=[eos_id, think_close_id],
        seed=seed,
        temperature=temperature,
        top_p=top_p,
    )

    stage2_ids = []
    stage2_index = []
    for i, (t_ids, t_text) in enumerate(zip(thought_ids, thought_texts)):
        closed = think_close_id in t_ids or t_text.rstrip().endswith("</think>")
        if not closed:
            continue
        cont = list(prompt_ids[i]) + list(t_ids)
        if t_ids and t_ids[-1] != think_close_id:
            cont.append(think_close_id)
        stage2_ids.append(cont)
        stage2_index.append(i)

    code_by_i = {i: "" for i in range(n)}
    code_tok_by_i = {i: 0 for i in range(n)}
    if stage2_ids:
        code_ids, code_texts = generate_from_ids(
            model,
            tokenizer,
            stage2_ids,
            max_new_tokens=max_code_tokens,
            eos_token_ids=[eos_id],
            seed=seed + 1,
            temperature=temperature,
            top_p=top_p,
        )
        for local_i, orig_i in enumerate(stage2_index):
            code_by_i[orig_i] = code_texts[local_i]
            code_tok_by_i[orig_i] = len(code_ids[local_i])

    samples = []
    for i in range(n):
        samples.append(
            {
                "thought": thought_texts[i],
                "code_region": code_by_i[i],
                "full_generated": thought_texts[i] + code_by_i[i],
                "thought_token_count": len(thought_ids[i]),
                "code_token_count": code_tok_by_i[i],
                "closed_think": think_close_id in thought_ids[i]
                or thought_texts[i].rstrip().endswith("</think>"),
            }
        )
    return samples


def pass_metrics(n: int, c: int) -> dict:
    ks = [1, 2, 4, 8, 10]
    out = {
        "num_samples": n,
        "passed_samples": c,
    }
    for k in ks:
        if k <= n:
            out[f"pass@{k}"] = round(calculate_pass_at_k(n, c, k), 4)
    return out


def mean_metric(per_problem: list[dict], key: str) -> float:
    vals = [p[key] for p in per_problem if key in p]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def default_models() -> list[dict]:
    return [
        {
            "name": "Base Model (Qwen3-1.7B-Base)",
            "type": "base",
            "model_path": "Qwen/Qwen3-1.7B-Base",
            "adapter_path": None,
            "use_think_prompt": False,
        },
        {
            "name": "SFT Model (Teacher Traces)",
            "type": "sft",
            "model_path": "Qwen/Qwen3-1.7B-Base",
            "adapter_path": "../sft-checkpoints/sft-new-scratchpad-data-checkpoints/run-32-0.0003-1.0/batch_276",
            "use_think_prompt": True,
        },
        {
            "name": "GRPO Run 1 (jo1azgcz)",
            "type": "grpo",
            "model_path": "../grpo-checkpoints/grpo-scratchpad-data-full-run-1-checkpoints/checkpoint-1000",
            "adapter_path": None,
            "use_think_prompt": True,
        },
        {
            "name": "GRPO Run 2 (rabh2bn9)",
            "type": "grpo",
            "model_path": "../grpo-checkpoints/grpo-scratchpad-data-full-run-2-checkpoints/checkpoint-1000",
            "adapter_path": None,
            "use_think_prompt": True,
        },
        {
            "name": "GRPO Run 3 (wnkvzp1h)",
            "type": "grpo",
            "model_path": "../grpo-checkpoints/grpo-scratchpad-data-full-run-3-checkpoints/checkpoint-100",
            "adapter_path": None,
            "use_think_prompt": True,
        },
        {
            "name": "GRPO Run 4 (64q9w4wa)",
            "type": "grpo",
            "model_path": "../grpo-checkpoints/grpo-scratchpad-data-full-run-4-checkpoints/checkpoint-200",
            "adapter_path": None,
            "use_think_prompt": True,
        },
        {
            "name": "GRPO Run 5 (iplwtqu4)",
            "type": "grpo",
            "model_path": "../grpo-checkpoints/grpo-scratchpad-data-full-run-5-checkpoints/checkpoint-100",
            "adapter_path": None,
            "use_think_prompt": True,
        },
    ]


def stopping_gate(results: dict) -> dict:
    def macro_strict(entry: dict) -> float:
        return entry.get("macro", {}).get("strict_pass@1", 0.0)

    base = [v for v in results.values() if v.get("model_type") == "base"]
    sft = [v for v in results.values() if v.get("model_type") == "sft"]
    grpo = [(name, v) for name, v in results.items() if v.get("model_type") == "grpo"]
    base_score = macro_strict(base[0]) if base else None
    sft_score = macro_strict(sft[0]) if sft else None
    best_grpo = None
    if grpo:
        name, entry = max(grpo, key=lambda kv: macro_strict(kv[1]))
        best_grpo = {"name": name, "strict_pass@1": macro_strict(entry)}

    beats_sft = (
        best_grpo is not None
        and sft_score is not None
        and best_grpo["strict_pass@1"] > sft_score
    )
    beats_base = (
        best_grpo is not None
        and base_score is not None
        and best_grpo["strict_pass@1"] > base_score
    )
    return {
        "metric": "macro strict pass@1",
        "base": base_score,
        "sft": sft_score,
        "best_grpo": best_grpo,
        "beats_sft": beats_sft,
        "beats_base": beats_base,
        "abandon_recipe": not (beats_sft and beats_base),
        "rule": (
            "Abandon the current training recipe if the best GRPO checkpoint "
            "fails to beat both SFT and base on strict pass@1 across this suite."
        ),
    }


def parse_args() -> argparse.Namespace:
    known_ids = ", ".join(p["id"] for p in PROBLEM_BANK)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2, help="Generation seed")
    parser.add_argument(
        "--n", type=int, default=16, help="Samples per problem (pass@k n)"
    )
    parser.add_argument(
        "--n-problems", type=int, default=10, help="How many bank problems to sample"
    )
    parser.add_argument(
        "--problem-seed", type=int, default=2, help="Seed for sampling problems"
    )
    parser.add_argument(
        "--problem-ids",
        type=str,
        default="",
        help=f"Comma-separated ids to run instead of sampling. Known: {known_ids}",
    )
    parser.add_argument(
        "--no-pin",
        action="store_true",
        help="Do not force-include LC3 and LC42 in the sampled set",
    )
    parser.add_argument(
        "--all-bank", action="store_true", help="Evaluate every problem in the bank"
    )
    parser.add_argument("--output", type=str, default="leetcode_eval_results.json")
    parser.add_argument("--max-think-tokens", type=int, default=1024)
    parser.add_argument("--max-code-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--cache-dir", type=str, default="/scratch/hub")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional case-insensitive substrings to filter model names",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the problem bank / sample, then exit without loading models",
    )
    parser.add_argument(
        "--truncate-raw",
        type=int,
        default=8000,
        help="Max characters of raw generation stored per sample",
    )
    return parser.parse_args()


def select_models(args: argparse.Namespace) -> list[dict]:
    models = default_models()
    if args.models:
        filters = [f.lower() for f in args.models]
        models = [
            m
            for m in models
            if any(f in m["name"].lower() or f in m["type"].lower() for f in filters)
        ]
        if not models:
            raise SystemExit(f"No models matched filters {args.models}")
    return models


def select_problems(args: argparse.Namespace) -> list[dict]:
    ids = [x.strip() for x in args.problem_ids.split(",") if x.strip()]
    if args.all_bank:
        return list(PROBLEM_BANK)
    return sample_problems(
        n=args.n_problems,
        seed=args.problem_seed,
        ids=ids or None,
        pin_original=not args.no_pin,
    )


def evaluate_problem_on_model(
    model,
    tokenizer,
    problem: dict,
    entry: dict,
    args: argparse.Namespace,
    model_index: int,
) -> dict:
    use_think = entry["use_think_prompt"]
    prompt = problem["prompt_think"] if use_think else problem["prompt_base"]
    method_name = problem["method_name"]
    header = problem["function_header"]
    seed = args.seed + 1009 * problem["number"] + 10007 * model_index

    t0 = time.time()
    samples = generate_samples(
        model,
        tokenizer,
        prompt,
        n=args.n,
        use_think=use_think,
        max_think_tokens=args.max_think_tokens,
        max_code_tokens=args.max_code_tokens,
        seed=seed,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    gen_s = time.time() - t0

    generations = []
    n_strict = 0
    n_oracle = 0
    for idx, sample in enumerate(samples):
        fmt = think_format_status(sample["thought"], use_think)
        strict_code, strict_why = extract_strict(
            sample["code_region"], method_name, header, fmt["format_ok"]
        )
        oracle_code, oracle_why = extract_oracle(
            sample["full_generated"], sample["code_region"], method_name, header
        )
        strict_ok, strict_detail, strict_p, strict_t = evaluate_code(
            strict_code,
            problem["test_cases"],
            method_name,
            problem["compare"],
            timeout_seconds=args.timeout,
        )
        if strict_code is None:
            strict_detail = strict_why
        oracle_ok, oracle_detail, oracle_p, oracle_t = evaluate_code(
            oracle_code,
            problem["test_cases"],
            method_name,
            problem["compare"],
            timeout_seconds=args.timeout,
        )
        if oracle_code is None:
            oracle_detail = oracle_why
        if strict_ok:
            n_strict += 1
        if oracle_ok:
            n_oracle += 1

        raw = sample["full_generated"]
        if args.truncate_raw and len(raw) > args.truncate_raw:
            raw = raw[: args.truncate_raw] + "\n...[truncated]..."

        generations.append(
            {
                "sample_index": idx + 1,
                "seed": seed,
                "format": fmt,
                "thought_trace": sample["thought"],
                "thought_token_count": sample["thought_token_count"],
                "code_token_count": sample["code_token_count"],
                "strict_code": strict_code,
                "strict_passed": strict_ok,
                "strict_detail": strict_detail,
                "strict_passed_tests": f"{strict_p}/{strict_t}",
                "strict_failure": (
                    None if strict_ok else classify_failure(strict_detail)
                ),
                "oracle_code": oracle_code,
                "oracle_passed": oracle_ok,
                "oracle_detail": oracle_detail,
                "oracle_passed_tests": f"{oracle_p}/{oracle_t}",
                "oracle_extract": oracle_why,
                "raw_generated_text": raw,
            }
        )
        print(
            f"  {problem['id']} sample {idx + 1}: "
            f"strict={'PASS' if strict_ok else 'FAIL'} ({strict_detail[:80]}) | "
            f"oracle={'PASS' if oracle_ok else 'FAIL'} ({oracle_detail[:80]})"
        )
        sys.stdout.flush()

    return {
        "problem_id": problem["id"],
        "title": problem["title"],
        "difficulty": problem["difficulty"],
        "generation_seconds": round(gen_s, 2),
        "strict": pass_metrics(args.n, n_strict),
        "oracle": pass_metrics(args.n, n_oracle),
        "generations": generations,
    }


def print_suite(problems: list[dict]) -> None:
    print("Problems:")
    for p in problems:
        print(
            f"  {p['id']:5} {p['difficulty']:6} {p['short_title']}  "
            f"({len(p['test_cases'])} tests)  {p['url']}"
        )
    print()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Using device: {device} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})"
    )

    validate_problem_bank()
    problems = select_problems(args)
    models_to_eval = [] if args.validate_only else select_models(args)
    print_suite(problems)
    print(
        f"Protocol: two-stage think->code, seed={args.seed}, n={args.n}, "
        f"max_think={args.max_think_tokens}, max_code={args.max_code_tokens}"
    )
    print(
        f"Pinned original pair: {PINNED_IDS if not args.no_pin and not args.problem_ids else 'none'} | "
        f"bank size={len(PROBLEM_BANK)}"
    )
    if args.validate_only:
        print("Validate-only: problem bank OK.")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B-Base", cache_dir=args.cache_dir
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    results: dict[str, dict] = {}
    payload = {
        "protocol": {
            "two_stage": True,
            "strict_and_oracle": True,
            "seed": args.seed,
            "num_samples": args.n,
            "max_think_tokens": args.max_think_tokens,
            "max_code_tokens": args.max_code_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "problem_sample_seed": args.problem_seed,
            "pinned_problem_ids": list(PINNED_IDS) if not args.no_pin else [],
        },
        "problems": [problem_public_dict(p) for p in problems],
        "results": results,
        "stopping_gate": None,
    }

    for model_index, entry in enumerate(models_to_eval):
        model_name = entry["name"]
        print(f"\n=======================================================")
        print(f"Evaluating: {model_name}")
        print(f"=======================================================")
        sys.stdout.flush()

        t0 = time.time()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        cache_dir = args.cache_dir if entry["type"] in ("base", "sft") else None
        base_m = AutoModelForCausalLM.from_pretrained(
            entry["model_path"],
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        if entry["adapter_path"] is not None:
            m = PeftModel.from_pretrained(base_m, entry["adapter_path"])
            m = m.merge_and_unload()
        else:
            m = base_m
        m.eval()
        print(f"Model loaded in {time.time() - t0:.2f}s")
        sys.stdout.flush()

        per_problem = []
        for problem in problems:
            print(f"\n-- {problem['title']} ({problem['difficulty']}) --")
            sys.stdout.flush()
            per_problem.append(
                evaluate_problem_on_model(
                    m, tokenizer, problem, entry, args, model_index
                )
            )

        strict_rows = [p["strict"] for p in per_problem]
        oracle_rows = [p["oracle"] for p in per_problem]
        results[model_name] = {
            "model_type": entry["type"],
            "per_problem": {p["problem_id"]: p for p in per_problem},
            "macro": {
                "strict_pass@1": mean_metric(strict_rows, "pass@1"),
                "strict_pass@2": mean_metric(strict_rows, "pass@2"),
                "strict_pass@4": mean_metric(strict_rows, "pass@4"),
                "oracle_pass@1": mean_metric(oracle_rows, "pass@1"),
                "oracle_pass@2": mean_metric(oracle_rows, "pass@2"),
                "oracle_pass@4": mean_metric(oracle_rows, "pass@4"),
            },
        }
        if args.n >= 8:
            results[model_name]["macro"]["strict_pass@8"] = mean_metric(
                strict_rows, "pass@8"
            )
            results[model_name]["macro"]["oracle_pass@8"] = mean_metric(
                oracle_rows, "pass@8"
            )

        payload["stopping_gate"] = stopping_gate(results)
        atomic_write_json(args.output, payload)
        print(
            f"--> {model_name}: strict pass@1={results[model_name]['macro']['strict_pass@1']:.3f} "
            f"oracle pass@1={results[model_name]['macro']['oracle_pass@1']:.3f}"
        )
        print(f"--> Updated {args.output}")
        sys.stdout.flush()

        del m
        del base_m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    gate = stopping_gate(results)
    payload["stopping_gate"] = gate
    atomic_write_json(args.output, payload)

    print("\n=======================================================")
    print("Stopping gate (macro strict pass@1)")
    print(f"  Base:      {gate['base']}")
    print(f"  SFT:       {gate['sft']}")
    if gate["best_grpo"]:
        print(
            f"  Best GRPO: {gate['best_grpo']['strict_pass@1']} "
            f"({gate['best_grpo']['name']})"
        )
    print(f"  Beats SFT:  {gate['beats_sft']}")
    print(f"  Beats base: {gate['beats_base']}")
    print(f"  Abandon recipe: {gate['abandon_recipe']}")
    print(f"All evaluations complete! Final results in {args.output}")
    print("=======================================================")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
