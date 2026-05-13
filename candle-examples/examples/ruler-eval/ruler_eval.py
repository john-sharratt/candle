#!/usr/bin/env python3
"""
RULER benchmark orchestrator for PalQuant vs. fixed-format KV comparison.

Workflow:
  1. Generate RULER task JSONL for each (task, context_length) pair
     using the NVIDIA/RULER data-generation scripts.
  2. For each (compression, context_length) cell, invoke the Rust
     ruler-eval binary to produce a predictions JSONL.
  3. Score predictions using the RULER scoring script.
  4. Print a summary table.

Prerequisites:
  - RULER repo cloned (set RULER_DIR or pass --ruler-dir).
  - ruler-eval binary built:
      cargo build --release --features cuda --example ruler-eval
  - Python packages: numpy, datasets (pip install numpy datasets)

Usage:
  python ruler_eval.py \\
    --model qwen3-30b-a3b \\
    --compressions c5 c8 q4-0 none \\
    --lengths 4096 8192 16384 32768 \\
    --tasks niah_single_1 niah_multikey_2 vt cwe \\
    --ruler-dir C:/tools/RULER \\
    --work-dir ./ruler_work \\
    --binary ./target/release/examples/ruler-eval
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ── RULER task definitions ────────────────────────────────────────────────────

# Map from short task name → RULER generator script / config identifier.
# These correspond to the RULER repo's synthetic_data/ task configs.
TASK_CONFIGS = {
    "niah_single_1":   {"type": "niah", "sub": "single_1"},
    "niah_single_2":   {"type": "niah", "sub": "single_2"},
    "niah_single_3":   {"type": "niah", "sub": "single_3"},
    "niah_multikey_1": {"type": "niah", "sub": "multikey_1"},
    "niah_multikey_2": {"type": "niah", "sub": "multikey_2"},
    "niah_multikey_3": {"type": "niah", "sub": "multikey_3"},
    "niah_multivalue": {"type": "niah", "sub": "multivalue"},
    "niah_multiquery": {"type": "niah", "sub": "multiquery"},
    "vt":              {"type": "vt",   "sub": None},
    "cwe":             {"type": "cwe",  "sub": None},
    "fwe":             {"type": "fwe",  "sub": None},
    "qa_1":            {"type": "qa",   "sub": "1"},
    "qa_2":            {"type": "qa",   "sub": "2"},
}

BASE_SAMPLES = 100  # default at 4096 tokens; scaled down for longer contexts
MIN_SAMPLES = 10    # floor to keep evaluation meaningful


# ── Data generation ───────────────────────────────────────────────────────────

def samples_for_length(length: int, base: int, overrides: dict[int, int] | None = None) -> int:
    """Return the number of samples to generate/evaluate for a given context length.

    Scales inversely with length to keep total compute roughly constant:
        samples = max(MIN_SAMPLES, base * 4096 // length)

    ``overrides`` (length → count) lets the caller pin specific lengths.
    """
    if overrides and length in overrides:
        return overrides[length]
    return max(MIN_SAMPLES, base * 4096 // length)


def generate_ruler_jsonl(
    ruler_dir: Path,
    task: str,
    context_length: int,
    tokenizer_path: str,
    output_path: Path,
    num_samples: int = BASE_SAMPLES,
) -> None:
    """Call the RULER data-generation script to produce a task JSONL file."""
    if output_path.exists():
        print(f"  [skip] {output_path.name} already exists")
        return

    gen_script = ruler_dir / "scripts" / "data" / "prepare.sh"
    if not gen_script.exists():
        # Try the Python entrypoint used in some RULER versions.
        gen_script = ruler_dir / "scripts" / "data" / "prepare.py"

    cfg = TASK_CONFIGS[task]
    task_type = cfg["type"]
    task_sub = cfg["sub"] or ""

    env = os.environ.copy()
    env["TOKENIZER_PATH"] = tokenizer_path
    env["MAX_SEQ_LENGTH"] = str(context_length)
    env["NUM_SAMPLES"] = str(num_samples)

    if gen_script.suffix == ".sh":
        cmd = [
            "bash", str(gen_script),
            "--task", task_type,
            "--subtask", task_sub,
            "--output", str(output_path),
        ]
    else:
        cmd = [
            sys.executable, str(gen_script),
            "--task", task_type,
            "--subtask", task_sub,
            "--tokenizer", tokenizer_path,
            "--max-seq-len", str(context_length),
            "--num-samples", str(num_samples),
            "--output", str(output_path),
        ]

    print(f"  Generating: {output_path.name}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR generating {output_path.name}:")
        print(result.stderr[-2000:])
        raise RuntimeError(f"Data generation failed for {task} @ {context_length}")


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    binary: Path,
    model: str,
    compression: str,
    input_jsonl: Path,
    output_jsonl: Path,
    max_gen_tokens: int,
    model_file: str | None,
    tokenizer_file: str | None,
    limit: int | None,
) -> None:
    """Invoke the ruler-eval Rust binary for one (compression, task, length) cell."""
    if output_jsonl.exists():
        print(f"  [skip] {output_jsonl.name} already exists")
        return

    cmd = [
        str(binary),
        "--model", model,
        "--compression", compression,
        "--input-jsonl", str(input_jsonl),
        "--output-jsonl", str(output_jsonl),
        "--max-gen-tokens", str(max_gen_tokens),
    ]
    if model_file:
        cmd += ["--model-file", model_file]
    if tokenizer_file:
        cmd += ["--tokenizer", tokenizer_file]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    print(f"  Running inference: {output_jsonl.name}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"ruler-eval failed for {output_jsonl.name}")


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_predictions(
    ruler_dir: Path,
    task: str,
    context_length: int,
    input_jsonl: Path,
    pred_jsonl: Path,
) -> float:
    """Score predictions using RULER's exact-match scorer. Returns accuracy 0–100."""
    # Load expected outputs from input JSONL.
    expected: dict[int, list[str]] = {}
    with open(input_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            expected[obj["index"]] = [s.lower().strip() for s in obj["outputs"]]

    # Load predictions.
    predictions: dict[int, str] = {}
    with open(pred_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            predictions[obj["index"]] = obj["pred"].lower().strip()

    if not predictions:
        return 0.0

    # Exact-match: prediction must contain at least one expected string.
    correct = 0
    total = 0
    for idx, exp_list in expected.items():
        if idx not in predictions:
            total += 1
            continue
        pred = predictions[idx]
        hit = any(e in pred for e in exp_list)
        correct += int(hit)
        total += 1

    return 100.0 * correct / total if total > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RULER benchmark orchestrator")
    parser.add_argument("--model", default="qwen3-30b-a3b",
                        help="Model identifier (matches ruler-eval --model values)")
    parser.add_argument("--compressions", nargs="+",
                        default=["none", "c5", "c8", "q4-0"],
                        help="Compression modes to evaluate")
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=[4096, 8192, 16384, 32768],
                        help="Context lengths (tokens)")
    parser.add_argument("--tasks", nargs="+",
                        default=["niah_single_1", "niah_multikey_2", "vt", "cwe"],
                        help="RULER task names")
    parser.add_argument("--ruler-dir", default=os.environ.get("RULER_DIR", ""),
                        help="Path to cloned NVIDIA/RULER repository")
    parser.add_argument("--work-dir", default="./ruler_work",
                        help="Working directory for generated data and predictions")
    parser.add_argument("--binary",
                        default="./target/release/examples/ruler-eval",
                        help="Path to the ruler-eval binary")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer.json (downloads from HF if omitted)")
    parser.add_argument("--model-file", default=None,
                        help="Path to GGUF model file (downloads from HF if omitted)")
    parser.add_argument("--max-gen-tokens", type=int, default=50)
    parser.add_argument("--base-samples", type=int, default=BASE_SAMPLES,
                        help="Sample count at 4096-token context; longer lengths are scaled down "
                             "proportionally (base * 4096 / length). Default: %(default)s.")
    parser.add_argument("--samples", nargs="+", type=int, default=None,
                        metavar="N",
                        help="Explicit sample count per length (must match --lengths order). "
                             "Overrides --base-samples scaling. E.g.: --samples 100 50 25 12")
    parser.add_argument("--skip-data-gen", action="store_true",
                        help="Skip RULER data generation (assumes JSONL already exists)")
    args = parser.parse_args()

    # ── Resolve per-length sample counts ─────────────────────────────────────
    lengths = args.lengths  # alias for convenience
    if args.samples is not None:
        if len(args.samples) == 1:
            # single value → apply to all lengths
            sample_counts = {l: args.samples[0] for l in lengths}
        elif len(args.samples) == len(lengths):
            sample_counts = dict(zip(lengths, args.samples))
        else:
            sys.exit(
                f"--samples must have 1 value or exactly {len(lengths)} values "
                f"(one per --lengths entry), got {len(args.samples)}"
            )
    else:
        sample_counts = {l: samples_for_length(l, args.base_samples) for l in lengths}

    print("Sample counts by context length:")
    for l in lengths:
        print(f"  {l // 1024}K tokens → {sample_counts[l]} samples")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir = work_dir / "data"
    pred_dir = work_dir / "preds"
    data_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)

    ruler_dir = Path(args.ruler_dir) if args.ruler_dir else None
    binary = Path(args.binary)
    if not binary.exists():
        sys.exit(
            f"ruler-eval binary not found at {binary}. "
            "Build with: cargo build --release --features cuda --example ruler-eval"
        )

    # ── Phase 1: Data generation ──────────────────────────────────────────────
    if not args.skip_data_gen:
        if ruler_dir is None or not ruler_dir.exists():
            sys.exit(
                "RULER_DIR not set or doesn't exist. "
                "Clone https://github.com/NVIDIA/RULER and pass --ruler-dir, "
                "or use --skip-data-gen if JSONL files already exist."
            )
        tokenizer_for_gen = args.tokenizer or ""
        print("\n=== Phase 1: Generating RULER task data ===")
        for task in args.tasks:
            for length in lengths:
                out = data_dir / f"{task}_{length}.jsonl"
                generate_ruler_jsonl(
                    ruler_dir=ruler_dir,
                    task=task,
                    context_length=length,
                    tokenizer_path=tokenizer_for_gen,
                    output_path=out,
                    num_samples=sample_counts[length],
                )

    # ── Phase 2: Inference ────────────────────────────────────────────────────
    print("\n=== Phase 2: Running inference ===")
    for comp in args.compressions:
        for task in args.tasks:
            for length in lengths:
                input_path = data_dir / f"{task}_{length}.jsonl"
                if not input_path.exists():
                    print(f"  [missing] {input_path.name} — skipping")
                    continue
                comp_label = comp.replace("_", "-")
                pred_path = pred_dir / f"{comp_label}_{task}_{length}.jsonl"
                run_inference(
                    binary=binary,
                    model=args.model,
                    compression=comp,
                    input_jsonl=input_path,
                    output_jsonl=pred_path,
                    max_gen_tokens=args.max_gen_tokens,
                    model_file=args.model_file,
                    tokenizer_file=args.tokenizer,
                    limit=sample_counts[length],
                )

    # ── Phase 3: Scoring ──────────────────────────────────────────────────────
    print("\n=== Phase 3: Scoring ===")
    # scores[comp][length] = {task: score}
    scores: dict[str, dict[int, dict[str, float]]] = {}
    for comp in args.compressions:
        comp_label = comp.replace("_", "-")
        scores[comp] = {}
        for length in lengths:
            scores[comp][length] = {}
            for task in args.tasks:
                input_path = data_dir / f"{task}_{length}.jsonl"
                pred_path = pred_dir / f"{comp_label}_{task}_{length}.jsonl"
                if not pred_path.exists() or not input_path.exists():
                    scores[comp][length][task] = float("nan")
                    continue
                acc = score_predictions(ruler_dir or Path("."), task, length, input_path, pred_path)
                scores[comp][length][task] = acc

    # ── Phase 4: Print summary table ─────────────────────────────────────────
    print("\n=== RULER Accuracy (%) ===")
    col_w = 10
    task_w = 22

    # Header: one column per (task, length) — show sample count in length label
    header_cells = []
    for length in lengths:
        n = sample_counts[length]
        for task in args.tasks:
            header_cells.append(f"{task[:8]}@{length//1024}K(n={n})")
    print(f"{'Compression':<{task_w}}" + "".join(f"{c:>{col_w+4}}" for c in header_cells))
    print("-" * (task_w + (col_w + 4) * len(header_cells)))

    cw = col_w + 4
    for comp in args.compressions:
        row = f"{comp:<{task_w}}"
        for length in lengths:
            for task in args.tasks:
                s = scores.get(comp, {}).get(length, {}).get(task, float("nan"))
                if s != s:  # nan
                    row += f"{'—':>{cw}}"
                else:
                    row += f"{s:>{cw}.1f}"
        print(row)

    # Per-length averages
    print()
    avg_headers = [f"avg@{l//1024}K(n={sample_counts[l]})" for l in lengths]
    print(f"{'Compression':<{task_w}}" + "".join(f"{h:>{cw}}" for h in avg_headers))
    print("-" * (task_w + cw * len(lengths)))
    for comp in args.compressions:
        row = f"{comp:<{task_w}}"
        for length in lengths:
            cell_scores = [
                scores.get(comp, {}).get(length, {}).get(t, float("nan"))
                for t in args.tasks
            ]
            valid = [s for s in cell_scores if s == s]
            avg = sum(valid) / len(valid) if valid else float("nan")
            row += f"{avg:>{cw}.1f}" if avg == avg else f"{'—':>{cw}}"
        print(row)

    # Save JSON results
    results_path = work_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    main()
