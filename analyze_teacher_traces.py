import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


MODE_ORDER = ["nothink", "default", "few_words", "couple_sentences"]
HISTOGRAM_BINS = [
    (0, 0),
    (1, 49),
    (50, 99),
    (100, 199),
    (200, 399),
    (400, 799),
    (800, 1599),
    (1600, math.inf),
]


def percentile(values: list[int], proportion: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[int]) -> dict[str, float]:
    return {
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": median(values),
        "mean": mean(values),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def bin_label(lower: int, upper: int | float) -> str:
    if lower == upper:
        return str(lower)
    if math.isinf(upper):
        return f"{lower}+"
    return f"{lower}-{upper}"


def analyze(path: Path) -> dict:
    mode_lengths = defaultdict(list)
    pair_counts = Counter()
    mode_counts = Counter()
    malformed_json_lines = []
    invalid_tag_lines = []

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_json_lines.append(line_number)
                continue

            mode = record.get("thinking_mode", "missing")
            trace = record.get("teacher_thinking_trace", "")
            mode_counts[mode] += 1
            pair_counts[(record.get("prefix"), record.get("continuation"))] += 1

            valid_tags = (
                trace.count("<think>") == 1
                and trace.count("</think>") == 1
                and trace.startswith("<think>")
                and trace.endswith("</think>")
            )
            if not valid_tags:
                invalid_tag_lines.append(line_number)
                continue

            thinking_text = trace[len("<think>") : -len("</think>")].strip()
            mode_lengths[mode].append(len(thinking_text))

    all_lengths = [
        length for lengths in mode_lengths.values() for length in lengths
    ]
    histogram = {}
    for lower, upper in HISTOGRAM_BINS:
        histogram[bin_label(lower, upper)] = sum(
            lower <= length <= upper for length in all_lengths
        )

    ordered_modes = MODE_ORDER + sorted(set(mode_counts) - set(MODE_ORDER))
    mode_summaries = {
        mode: summarize(mode_lengths[mode])
        for mode in ordered_modes
        if mode_lengths[mode]
    }

    return {
        "records": sum(mode_counts.values()),
        "malformed_json_lines": malformed_json_lines,
        "invalid_tag_lines": invalid_tag_lines,
        "mode_counts": {mode: mode_counts[mode] for mode in ordered_modes},
        "empty_traces": all_lengths.count(0),
        "empty_traces_by_mode": {
            mode: mode_lengths[mode].count(0) for mode in ordered_modes
        },
        "duplicate_prefix_continuation_examples": sum(
            count - 1 for count in pair_counts.values() if count > 1
        ),
        "overall_character_summary": summarize(all_lengths) if all_lengths else {},
        "mode_character_summaries": mode_summaries,
        "character_histogram": histogram,
    }


def print_report(result: dict) -> None:
    total = result["records"]
    print(f"Records: {total}")
    print(f"Malformed JSON lines: {len(result['malformed_json_lines'])}")
    print(f"Invalid tag records: {len(result['invalid_tag_lines'])}")
    print(f"Empty thinking traces: {result['empty_traces']}")
    print(
        "Empty traces by mode: "
        + ", ".join(
            f"{mode}={count}"
            for mode, count in result["empty_traces_by_mode"].items()
        )
    )
    print(
        "Repeated prefix/continuation examples: "
        f"{result['duplicate_prefix_continuation_examples']}"
    )

    print("\nMode distribution:")
    for mode, count in result["mode_counts"].items():
        percentage = 100 * count / total if total else 0
        print(f"  {mode:20} {count:5} ({percentage:5.1f}%)")

    print("\nThinking characters after stripping tags and surrounding whitespace:")
    print("  mode                    min    p25 median   mean    p75    p90    max")
    for mode, summary in result["mode_character_summaries"].items():
        print(
            f"  {mode:20} "
            f"{summary['min']:6.0f} {summary['p25']:6.0f} "
            f"{summary['median']:6.0f} {summary['mean']:6.1f} "
            f"{summary['p75']:6.0f} {summary['p90']:6.0f} "
            f"{summary['max']:6.0f}"
        )

    print("\nOverall character histogram:")
    for label, count in result["character_histogram"].items():
        percentage = 100 * count / total if total else 0
        print(f"  {label:10} {count:5} ({percentage:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze teacher trace JSONL output.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("teacher_traces.jsonl"),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead.")
    args = parser.parse_args()

    result = analyze(args.path)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
