//! decode_ab — A/B test harness for the paged-decode attention kernels.
//!
//! Drives two decode-attention kernels — the V2 persistent-slot-buffer kernel
//! (reference) and the new fused-attn-v1 INT8 kernel (candidate) — over the
//! same synthetic fixtures and reports numerical parity and throughput.
//!
//! Test data is fully synthetic and deterministic: a KV arena is populated by
//! prefilling hash-seeded tokens at a chosen storage format, then a single
//! decode step is run through each kernel on identical, freshly-rebuilt slot
//! metadata. No model download or capture/replay needed.
//!
//! Examples:
//!   cargo run --release --features cuda --example decode_ab -- compare
//!   cargo run --release --features cuda --example decode_ab -- compare \
//!       --scenarios gqa3_ctx512_b8 --formats q4_0,q8_0,f16 --out report.md
//!   cargo run --release --features cuda --example decode_ab -- bench --iters 200

mod fixture;
mod formats;
mod metrics;
mod report;
mod scenarios;


use anyhow::{bail, Context, Result};
use candle::Device;
use candle::quantized::pinned_staging::PinnedStager;
use candle_transformers::models::prefill_utils::DecodeBackend;
use clap::{Parser, Subcommand};

use fixture::Fixture;
use formats::{
    all_formats, default_formats, deep_formats, quant_formats, select_formats, ArenaFmt,
};
use metrics::Metrics;
use report::{
    render_bench, render_compare, render_golden, BenchRow, CompareOutcome, CompareRow,
    GoldenOutcome, GoldenRow,
};
use scenarios::{
    default_scenarios, perf_scenarios, select_scenarios, single_decode_scenarios,
    suite_deep_scenarios, suite_scenarios, Scenario,
};

#[derive(Parser)]
#[command(
    name = "decode_ab",
    about = "A/B test harness for paged-decode attention kernels (V2 vs fused-attn-v1)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,

    /// Comma-separated scenario names to run (default: all).
    #[arg(long, global = true)]
    scenarios: Option<String>,

    /// Comma-separated arena-format labels to run (e.g. f16,q4_0,q8_0).
    #[arg(long, global = true)]
    formats: Option<String>,

    /// Sweep every kernel-supported arena format (overrides the default set).
    #[arg(long, global = true)]
    all_formats: bool,

    /// Write the markdown report to this file in addition to stdout.
    #[arg(long, global = true)]
    out: Option<String>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compare kernel outputs for numerical parity across the matrix.
    Compare {
        // Defaults are calibrated to INT8-vs-FP precision, not bit-equality:
        // the fused Track-A kernel is V2 with INT8 MMA for QK^T and INT8 for PV,
        // so a format-independent residual on the order of a few e-3 MAE
        // (cosine ~0.994 at the most extreme short-context / hd64 case) is the
        // designed approximation, already validated end-to-end. The harness
        // flags divergence *beyond* this — as it did when it caught the
        // fixture-reuse bug (cosine 0.85). Tighten with the flags for
        // regression hunts.
        /// Max mean-abs error allowed for a pass.
        #[arg(long, default_value_t = 1.0e-2)]
        mae_tol: f32,
        /// Max absolute error allowed for a pass.
        #[arg(long, default_value_t = 1.0e-1)]
        max_abs_tol: f32,
        /// Min cosine similarity required for a pass.
        #[arg(long, default_value_t = 0.99)]
        cosine_tol: f32,
        /// Also compare BOTH kernels against an FP32 ground-truth reference
        /// (forces identity RoPE) and gate on it. Catches a bug that affects
        /// both kernels equally — which the A/B alone cannot.
        #[arg(long)]
        golden: bool,
        /// Golden pass gate: min cosine of each kernel vs FP32 truth. Cosine is
        /// precision-robust — even the most aggressive compression (L7 / Q2/Q1)
        /// keeps cosine ≥ ~0.96, while a structural K-read/palette bug craters it
        /// to ≲ 0.6 (e.g. the hd64 R16 case). 0.93 sits in that gap: it passes
        /// all legitimate quant loss and fails only correctness regressions.
        #[arg(long, default_value_t = 0.93)]
        golden_cosine_tol: f32,
    },
    /// Benchmark per-call kernel time for each backend across the matrix.
    Bench {
        /// Timed iterations per (scenario, format, backend).
        #[arg(long, default_value_t = 100)]
        iters: usize,
        /// Warmup iterations (untimed) before measuring.
        #[arg(long, default_value_t = 20)]
        warmup: usize,
    },
    /// Diagnostic: localize a divergence. For each scenario, build a FLOAT and
    /// a QUANT arena from identical synthetic K/V, run both kernels on each, and
    /// cross-compare to separate the INT8-compute delta from the quantized-read
    /// delta and attribute it to V2 vs fused.
    Xcheck {
        /// Quant format to use for the quantized arena (label, e.g. q4_0).
        #[arg(long, default_value = "q4_0")]
        quant: String,
    },
    /// Comprehensive regression suite: run BOTH gates — A/B parity (fused vs V2)
    /// and ground truth (vs FP32) — across the full quant × shape matrix in one
    /// pass. Defaults to `suite_scenarios` × `quant_formats` (every codec at
    /// every context depth, single and multi batch); `--scenarios` / `--formats`
    /// / `--all-formats` override either axis.
    Suite {
        /// Max mean-abs error allowed for an A/B pass.
        #[arg(long, default_value_t = 1.0e-2)]
        mae_tol: f32,
        /// Max absolute error allowed for an A/B pass.
        #[arg(long, default_value_t = 1.0e-1)]
        max_abs_tol: f32,
        /// Min cosine similarity required for an A/B pass.
        #[arg(long, default_value_t = 0.99)]
        cosine_tol: f32,
        /// Golden pass gate: min cosine of each kernel vs FP32 truth.
        #[arg(long, default_value_t = 0.93)]
        golden_cosine_tol: f32,
    },
}

fn resolve_matrix(cli: &Cli, is_suite: bool) -> Result<(Vec<Scenario>, Vec<ArenaFmt>)> {
    let scenarios = match &cli.scenarios {
        Some(f) => select_scenarios(f).map_err(|e| anyhow::anyhow!(e))?,
        None if is_suite => suite_scenarios(),
        None => default_scenarios(),
    };
    let formats = match (&cli.formats, cli.all_formats) {
        (Some(f), _) => select_formats(f).map_err(|e| anyhow::anyhow!(e))?,
        (None, true) => all_formats(),
        (None, false) if is_suite => quant_formats(),
        (None, false) => default_formats(),
    };
    if scenarios.is_empty() {
        bail!("no scenarios selected");
    }
    if formats.is_empty() {
        bail!("no formats selected");
    }
    Ok((scenarios, formats))
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let device = Device::cuda_if_available(0).context("opening CUDA device")?;
    if !device.is_cuda() {
        bail!("decode_ab requires a CUDA device (build with --features cuda and a GPU present)");
    }
    let stager = PinnedStager::new_from_device(&device);

    let is_suite = matches!(cli.cmd, Cmd::Suite { .. });
    let (scenarios, fmts) = resolve_matrix(&cli, is_suite)?;

    let markdown = match &cli.cmd {
        Cmd::Compare {
            mae_tol,
            max_abs_tol,
            cosine_tol,
            golden,
            golden_cosine_tol,
        } => {
            if *golden {
                std::env::set_var("DECODE_AB_IDENTITY_ROPE", "1");
                run_golden(&scenarios, &fmts, *golden_cosine_tol, &device, &stager)?
            } else {
                run_compare(
                    &scenarios,
                    &fmts,
                    (*mae_tol, *max_abs_tol, *cosine_tol),
                    &device,
                    &stager,
                )?
            }
        }
        Cmd::Bench { iters, warmup } => {
            // Default the bench to the batch-8 perf set (fills the MMA M dim)
            // plus the batch-1 deep-context single-decode set (the grid-starved
            // regime split-KV targets); an explicit --scenarios still overrides.
            let bench_scen = if cli.scenarios.is_none() {
                let mut s = perf_scenarios();
                s.extend(single_decode_scenarios());
                s
            } else {
                scenarios.clone()
            };
            run_bench(&bench_scen, &fmts, *iters, *warmup, &device, &stager)?
        }
        Cmd::Xcheck { quant } => run_xcheck(&scenarios, quant, &device, &stager)?,
        Cmd::Suite {
            mae_tol,
            max_abs_tol,
            cosine_tol,
            golden_cosine_tol,
        } => {
            let tol = (*mae_tol, *max_abs_tol, *cosine_tol);
            // Default suite = two sweeps:
            //   • CODEC: every quant format at the cheap shallow/mid shapes
            //     (resolve_matrix already set scenarios/fmts to
            //     suite_scenarios × quant_formats) — validates each codec.
            //   • DEPTH/SCALE: only the production native-INT8 formats at the
            //     expensive deep / large-batch shapes — exercises the deep-scan /
            //     split-KV path (codec-agnostic), without rebuilding every
            //     aggressive codec at depth.
            // An explicit --scenarios/--formats/--all-formats collapses to one
            // flat group for ad-hoc runs.
            let overridden = cli.scenarios.is_some() || cli.formats.is_some() || cli.all_formats;
            let groups: Vec<(Vec<Scenario>, Vec<ArenaFmt>, &str)> = if overridden {
                vec![(scenarios.clone(), fmts.clone(), "override")]
            } else {
                vec![
                    (
                        scenarios.clone(),
                        fmts.clone(),
                        "codec sweep — all quants × shallow/mid shapes",
                    ),
                    (
                        suite_deep_scenarios(),
                        deep_formats(),
                        "depth/scale — Q8_0/Q4_0/f16 × deep & large-batch shapes",
                    ),
                ]
            };
            // A/B parity first (real RoPE); DECODE_AB_IDENTITY_ROPE must be set
            // only AFTER the parity fixtures are built, since it forces identity
            // RoPE for the FP32 truth.
            let mut ab = String::from("# Decode suite — A/B parity (fused vs V2)\n");
            for (scn, fmt, label) in &groups {
                ab.push_str(&format!("\n## {label}\n\n"));
                ab.push_str(&run_compare(scn, fmt, tol, &device, &stager)?);
            }
            std::env::set_var("DECODE_AB_IDENTITY_ROPE", "1");
            let mut golden = String::from("# Decode suite — ground truth (vs FP32)\n");
            for (scn, fmt, label) in &groups {
                golden.push_str(&format!("\n## {label}\n\n"));
                golden.push_str(&run_golden(scn, fmt, *golden_cosine_tol, &device, &stager)?);
            }
            format!("{ab}\n{golden}")
        }
    };

    println!("{markdown}");
    if let Some(path) = &cli.out {
        std::fs::write(path, &markdown).with_context(|| format!("writing report to {path}"))?;
        eprintln!("report written to {path}");
    }
    Ok(())
}

/// Ground-truth gate: build both kernels' output for each cell, compare to the
/// FP32 golden, and pass/FAIL on structural correctness (cosine). Catches
/// both-kernels-wrong bugs (e.g. a palette/K-read regression) the A/B can't see.
fn run_golden(
    scenarios: &[Scenario],
    fmts: &[ArenaFmt],
    cosine_tol: f32,
    device: &Device,
    stager: &PinnedStager,
) -> Result<String> {
    let mut rows = Vec::new();
    let mut any_fail = false;
    'outer: for sc in scenarios {
        for &fmt in fmts {
            eprint!("golden  {:<24} {:<14} ... ", sc.name, fmt.label());
            let outcome = match golden_cell(sc, fmt, device, stager) {
                Ok((v2, fused, ab_mae)) => {
                    // Per-format floor: the most aggressive formats (1-bit Q1_S)
                    // have a legitimately lower structural-correctness ceiling.
                    let floor = fmt.golden_cosine_floor(cosine_tol);
                    let passed = v2.cosine >= floor && fused.cosine >= floor;
                    any_fail |= !passed;
                    eprintln!(
                        "{} legacy_cos={:.5} int8_cos={:.5} (legacy_mae={:.2e} int8_mae={:.2e})",
                        if passed { "pass" } else { "FAIL" },
                        v2.cosine,
                        fused.cosine,
                        v2.mae,
                        fused.mae,
                    );
                    GoldenOutcome::Ran {
                        v2,
                        fused,
                        ab_mae,
                        passed,
                    }
                }
                Err(e) => {
                    let msg = short_err(&e);
                    eprintln!("skip ({msg})");
                    if is_context_fatal(&msg) {
                        rows.push(GoldenRow {
                            scenario: sc.name.to_string(),
                            format: fmt.label(),
                            outcome: GoldenOutcome::Skipped(msg),
                        });
                        eprintln!("FATAL: CUDA context poisoned — stopping early.");
                        break 'outer;
                    }
                    GoldenOutcome::Skipped(msg)
                }
            };
            rows.push(GoldenRow {
                scenario: sc.name.to_string(),
                format: fmt.label(),
                outcome,
            });
        }
    }
    if any_fail {
        eprintln!("note: one or more GOLDEN cells FAILED — a kernel diverged from FP32 truth.");
    }
    Ok(render_golden(&rows, cosine_tol))
}

fn run_compare(
    scenarios: &[Scenario],
    fmts: &[ArenaFmt],
    tol: (f32, f32, f32),
    device: &Device,
    stager: &PinnedStager,
) -> Result<String> {
    let mut rows = Vec::new();
    let mut any_fail = false;
    'outer: for sc in scenarios {
        for &fmt in fmts {
            eprint!("compare {:<24} {:<14} ... ", sc.name, fmt.label());
            let outcome = compare_cell(sc, fmt, tol, device, stager);
            let mut fatal = false;
            match &outcome {
                CompareOutcome::Ran { metrics, passed } => {
                    eprintln!(
                        "{} mae={:.2e} cos={:.5}",
                        if *passed { "pass" } else { "FAIL" },
                        metrics.mae,
                        metrics.cosine
                    );
                    any_fail |= !*passed;
                }
                CompareOutcome::Skipped(why) => {
                    eprintln!("skip ({why})");
                    fatal = is_context_fatal(why);
                }
            }
            rows.push(CompareRow {
                scenario: sc.name.to_string(),
                format: fmt.label().to_string(),
                outcome,
            });
            if fatal {
                eprintln!(
                    "FATAL: CUDA context poisoned by {}/{} — stopping run early and \
                     emitting partial results. Re-run remaining cells in a fresh process.",
                    sc.name,
                    fmt.label()
                );
                break 'outer;
            }
        }
    }
    let md = render_compare(&rows, tol);
    if any_fail {
        eprintln!("note: one or more parity cells FAILED (see table).");
    }
    Ok(md)
}

fn compare_cell(
    sc: &Scenario,
    fmt: ArenaFmt,
    tol: (f32, f32, f32),
    device: &Device,
    stager: &PinnedStager,
) -> CompareOutcome {
    if !sc.head_dim_supported() {
        return CompareOutcome::Skipped(format!("head_dim {} unsupported", sc.head_dim));
    }
    // Each backend runs on its OWN freshly-built fixture. A decode commits the
    // write token (advancing the slot by one), so reusing one fixture across
    // backends would let the first kernel's commit change the context/chunk
    // layout the second kernel sees — spuriously inflating the divergence,
    // most visibly at a chunk boundary (e.g. ctx=31 → 32). Fresh, deterministic
    // fixtures give both kernels bit-identical pristine input.
    let out_v2 = match build_and_decode(sc, fmt, DecodeBackend::Legacy, device, stager) {
        Ok(o) => o,
        Err(e) => return CompareOutcome::Skipped(format!("v2: {}", short_err(&e))),
    };
    let out_fused = match build_and_decode(sc, fmt, DecodeBackend::Int8, device, stager) {
        Ok(o) => o,
        Err(e) => return CompareOutcome::Skipped(format!("fused: {}", short_err(&e))),
    };
    match Metrics::compute(&out_v2, &out_fused, sc.n_q_head, sc.head_dim) {
        Ok(metrics) => {
            let passed = metrics.passes(tol.0, tol.1, tol.2);
            CompareOutcome::Ran { metrics, passed }
        }
        Err(e) => CompareOutcome::Skipped(format!("metrics: {}", short_err(&e))),
    }
}

fn run_bench(
    scenarios: &[Scenario],
    fmts: &[ArenaFmt],
    iters: usize,
    warmup: usize,
    device: &Device,
    stager: &PinnedStager,
) -> Result<String> {
    let mut rows = Vec::new();
    'outer: for sc in scenarios {
        for &fmt in fmts {
            if !sc.head_dim_supported() {
                continue;
            }
            eprint!("bench   {:<24} {:<8} ... ", sc.name, fmt.label());
            let run = || -> candle::Result<(std::time::Duration, std::time::Duration)> {
                // Twin fixtures (identical seed), built once. We INTERLEAVE the
                // V2 and fused timed decodes per iteration so both see the same
                // GPU clock/thermal state back-to-back. Timing V2 fully then
                // fused fully would let a clock shift between the two phases bias
                // the ratio (the dominant noise source). The commit-drift from
                // the decode write token then affects both equally per iter.
                let fix_v2 = Fixture::build(sc, fmt, device, stager)?;
                let fix_fused = Fixture::build(sc, fmt, device, stager)?;
                for _ in 0..warmup {
                    let _ = fix_v2.decode(DecodeBackend::Legacy, device, stager)?;
                    let _ = fix_fused.decode(DecodeBackend::Int8, device, stager)?;
                }
                let mut v2s: Vec<std::time::Duration> = Vec::with_capacity(iters.max(1));
                let mut fs: Vec<std::time::Duration> = Vec::with_capacity(iters.max(1));
                for _ in 0..iters.max(1) {
                    let (_, dt_v2) = fix_v2.decode(DecodeBackend::Legacy, device, stager)?;
                    let (_, dt_f) = fix_fused.decode(DecodeBackend::Int8, device, stager)?;
                    v2s.push(dt_v2);
                    fs.push(dt_f);
                }
                v2s.sort();
                fs.sort();
                Ok((v2s[v2s.len() / 2], fs[fs.len() / 2]))
            };
            let (v2, fused) = match run() {
                Ok(d) => d,
                Err(e) => {
                    let msg = short_err(&e);
                    eprintln!("skip ({msg})");
                    if is_context_fatal(&msg) {
                        eprintln!(
                            "FATAL: CUDA context poisoned by {}/{} — stopping bench early.",
                            sc.name,
                            fmt.label()
                        );
                        break 'outer;
                    }
                    continue;
                }
            };
            let v2_us = v2.as_secs_f64() * 1e6;
            let fused_us = fused.as_secs_f64() * 1e6;
            let tps = |us: f64| sc.num_slots as f64 * 1e6 / us;
            eprintln!(
                "legacy={v2_us:.1}µs ({:.0} tok/s)  int8={fused_us:.1}µs ({:.0} tok/s)  {:.2}×",
                tps(v2_us),
                tps(fused_us),
                v2_us / fused_us
            );
            rows.push(BenchRow {
                scenario: sc.name.to_string(),
                format: fmt.label(),
                num_slots: sc.num_slots,
                v2_us,
                fused_us,
            });
        }
    }
    Ok(render_bench(&rows))
}

/// Build a fresh fixture and run a single decode on `backend`. The fresh build
/// guarantees pristine, deterministic input (a decode commits the write token,
/// so fixtures must not be reused across kernels — see `compare_cell`).
fn build_and_decode(
    sc: &Scenario,
    fmt: ArenaFmt,
    backend: DecodeBackend,
    device: &Device,
    stager: &PinnedStager,
) -> candle::Result<candle::Tensor> {
    let fix = Fixture::build(sc, fmt, device, stager)?;
    Ok(fix.decode(backend, device, stager)?.0)
}

/// Ground-truth check for one cell. Returns (v2-vs-golden, fused-vs-golden,
/// v2-vs-fused) MAE. If one kernel's vs-golden MAE is much larger than the
/// other's, that kernel has the bug; if BOTH are large but their A/B is small,
/// both share a wrong K-read (the case the A/B alone can't see).
fn golden_cell(
    sc: &Scenario,
    fmt: ArenaFmt,
    device: &Device,
    stager: &PinnedStager,
) -> candle::Result<(Metrics, Metrics, f32)> {
    let out_v2 = build_and_decode(sc, fmt, DecodeBackend::Legacy, device, stager)?;
    let out_fused = build_and_decode(sc, fmt, DecodeBackend::Int8, device, stager)?;
    let gold = fixture::golden_decode(sc, device)?;
    let v2 = Metrics::compute(&gold, &out_v2, sc.n_q_head, sc.head_dim)?;
    let fused = Metrics::compute(&gold, &out_fused, sc.n_q_head, sc.head_dim)?;
    let ab = Metrics::compute(&out_v2, &out_fused, sc.n_q_head, sc.head_dim)?.mae;
    Ok((v2, fused, ab))
}

/// Diagnostic cross-check: separate the INT8-compute delta from the
/// quantized-read delta to attribute a divergence to V2 vs fused.
fn run_xcheck(
    scenarios: &[Scenario],
    quant_label: &str,
    device: &Device,
    stager: &PinnedStager,
) -> Result<String> {
    let qfmt = *select_formats(quant_label)
        .map_err(|e| anyhow::anyhow!(e))?
        .first()
        .ok_or_else(|| anyhow::anyhow!("no quant format resolved from {quant_label}"))?;
    let ffmt = ArenaFmt::Float(candle::DType::F16);

    let mut s = String::new();
    s.push_str(&format!(
        "# decode A/B — xcheck (float=f16, quant={quant_label})\n\n\
         Columns (MAE, all over identical synthetic K/V):\n\
         - `int8Δ(float)` = fused vs V2 on the **float** arena → pure INT8-compute delta.\n\
         - `v2 qsens` = V2 float vs V2 quant → how much quantization perturbs **V2**.\n\
         - `fused qsens` = fused float vs fused quant → how much quantization perturbs **fused**.\n\
         - `total(quant)` = fused vs V2 on the **quant** arena.\n\n\
         If `fused qsens` ≫ `v2 qsens`, the fused **quantized-read** path is the suspect.\n\n",
    ));
    s.push_str(
        "| scenario | int8Δ(float) | v2 qsens | fused qsens | total(quant) | verdict |\n",
    );
    s.push_str("|---|---|---|---|---|---|\n");

    for sc in scenarios {
        if !sc.head_dim_supported() {
            continue;
        }
        eprint!("xcheck  {:<24} ... ", sc.name);
        let row = (|| -> candle::Result<(f32, f32, f32, f32)> {
            // Fresh fixture per decode — a decode commits the write token, so
            // reuse would cross-contaminate (see `compare_cell`).
            let v2_f = build_and_decode(sc, ffmt, DecodeBackend::Legacy, device, stager)?;
            let fu_f = build_and_decode(sc, ffmt, DecodeBackend::Int8, device, stager)?;
            let v2_q = build_and_decode(sc, qfmt, DecodeBackend::Legacy, device, stager)?;
            let fu_q = build_and_decode(sc, qfmt, DecodeBackend::Int8, device, stager)?;
            let m = |a, b| -> candle::Result<f32> {
                Ok(Metrics::compute(a, b, sc.n_q_head, sc.head_dim)?.mae)
            };
            Ok((
                m(&v2_f, &fu_f)?,
                m(&v2_f, &v2_q)?,
                m(&fu_f, &fu_q)?,
                m(&v2_q, &fu_q)?,
            ))
        })();
        match row {
            Ok((int8d, v2q, fuq, tot)) => {
                // Flag the fused quant-read path only when its quant
                // sensitivity is both materially large AND well above V2's —
                // avoids false positives when quant is near-lossless (both ~0).
                let verdict = if fuq > 1e-3 && fuq > 3.0 * v2q.max(1e-6) {
                    "fused quant-read"
                } else if int8d >= tot * 0.5 {
                    "INT8 compute (expected)"
                } else {
                    "mixed"
                };
                eprintln!(
                    "int8Δ={int8d:.2e} v2q={v2q:.2e} fuq={fuq:.2e} tot={tot:.2e} → {verdict}"
                );
                s.push_str(&format!(
                    "| {} | {int8d:.3e} | {v2q:.3e} | {fuq:.3e} | {tot:.3e} | {verdict} |\n",
                    sc.name
                ));
            }
            Err(e) => {
                eprintln!("skip ({})", short_err(&e));
                s.push_str(&format!("| {} | — | — | — | — | skip |\n", sc.name));
            }
        }
    }
    Ok(s)
}

/// Whether an error string indicates a CUDA context-poisoning failure (an
/// illegal memory access leaves the context dead; every subsequent CUDA call —
/// including unrelated allocations — will fail, so the run must stop).
fn is_context_fatal(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("illegal") || m.contains("drivererror") || m.contains("cuda_error")
}

/// First line of an error, trimmed for table cells.
fn short_err(e: &candle::Error) -> String {
    let s = e.to_string();
    let first = s.lines().next().unwrap_or("");
    if first.len() > 80 {
        format!("{}…", &first[..79])
    } else {
        first.to_string()
    }
}
