//! decode_ab — correctness + throughput harness for the paged-decode INT8 kernel.
//!
//! Drives the INT8 split-KV / warp-stripe / batched-M decode kernel over
//! synthetic fixtures and reports correctness against an FP32 ground truth plus
//! throughput. (Historically this was an A/B harness vs the legacy V2 kernel;
//! that kernel has been removed, so the FP32 golden is now the reference.)
//!
//! Test data is fully synthetic and deterministic: a KV arena is populated by
//! prefilling hash-seeded tokens at a chosen storage format, then a single
//! decode step is run on freshly-rebuilt slot metadata. No model download or
//! capture/replay needed.
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
use candle::quantized::pinned_staging::PinnedStager;
use candle::Device;
use clap::{Parser, Subcommand};

use fixture::Fixture;
use formats::{
    all_formats, deep_formats, default_formats, quant_formats, select_formats, ArenaFmt,
};
use metrics::Metrics;
use report::{render_bench, render_golden, BenchRow, GoldenOutcome, GoldenRow};
use scenarios::{
    default_scenarios, perf_scenarios, select_scenarios, single_decode_scenarios,
    suite_deep_scenarios, suite_scenarios, Scenario,
};

#[derive(Parser)]
#[command(
    name = "decode_ab",
    about = "Correctness + throughput harness for the INT8 paged-decode kernel"
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
    /// Check the INT8 kernel's output against an FP32 ground truth (identity
    /// RoPE) and gate on structural correctness (cosine).
    Compare {
        /// Pass gate: min cosine of the int8 output vs FP32 truth. Cosine is
        /// precision-robust — even the most aggressive compression (L7 / Q2/Q1)
        /// keeps cosine ≥ ~0.96, while a structural K-read/palette bug craters it
        /// to ≲ 0.6. 0.93 sits in that gap: it passes all legitimate quant loss
        /// and fails only correctness regressions.
        #[arg(long, default_value_t = 0.93)]
        golden_cosine_tol: f32,
    },
    /// Benchmark per-call INT8 kernel time across the matrix.
    Bench {
        /// Timed iterations per (scenario, format).
        #[arg(long, default_value_t = 100)]
        iters: usize,
        /// Warmup iterations (untimed) before measuring.
        #[arg(long, default_value_t = 20)]
        warmup: usize,
    },
    /// Comprehensive ground-truth regression suite: run the golden gate across
    /// the full quant × shape matrix in one pass. Defaults to a codec sweep
    /// (every codec at shallow/mid shapes) plus a depth/scale sweep (production
    /// native-INT8 formats at deep & large-batch shapes); `--scenarios` /
    /// `--formats` / `--all-formats` override either axis.
    Suite {
        /// Pass gate: min cosine of the int8 output vs FP32 truth.
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
        Cmd::Compare { golden_cosine_tol } => {
            std::env::set_var("DECODE_AB_IDENTITY_ROPE", "1");
            run_golden(&scenarios, &fmts, *golden_cosine_tol, &device, &stager)?
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
        Cmd::Suite { golden_cosine_tol } => {
            // Default suite = two sweeps:
            //   • CODEC: every quant format at the cheap shallow/mid shapes
            //     (resolve_matrix already set scenarios/fmts to
            //     suite_scenarios × quant_formats) — validates each codec.
            //   • DEPTH/SCALE: only the production native-INT8 formats at the
            //     expensive deep / large-batch shapes — exercises the deep-scan /
            //     split-KV path (codec-agnostic).
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
            std::env::set_var("DECODE_AB_IDENTITY_ROPE", "1");
            let mut golden = String::from("# Decode suite — ground truth (vs FP32)\n");
            for (scn, fmt, label) in &groups {
                golden.push_str(&format!("\n## {label}\n\n"));
                golden.push_str(&run_golden(scn, fmt, *golden_cosine_tol, &device, &stager)?);
            }
            golden
        }
    };

    println!("{markdown}");
    if let Some(path) = &cli.out {
        std::fs::write(path, &markdown).with_context(|| format!("writing report to {path}"))?;
        eprintln!("report written to {path}");
    }
    Ok(())
}

/// Ground-truth gate: build the int8 kernel's output for each cell, compare to
/// the FP32 golden, and pass/FAIL on structural correctness (cosine).
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
                Ok(metrics) => {
                    // Per-format floor: the most aggressive formats (1-bit Q1_S)
                    // have a legitimately lower structural-correctness ceiling.
                    let floor = fmt.golden_cosine_floor(cosine_tol);
                    let passed = metrics.cosine >= floor;
                    any_fail |= !passed;
                    eprintln!(
                        "{} int8_cos={:.5} int8_mae={:.2e}",
                        if passed { "pass" } else { "FAIL" },
                        metrics.cosine,
                        metrics.mae,
                    );
                    GoldenOutcome::Ran { metrics, passed }
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
        eprintln!("note: one or more GOLDEN cells FAILED — the kernel diverged from FP32 truth.");
    }
    Ok(render_golden(&rows, cosine_tol))
}

/// Ground-truth check for one cell: int8 kernel output vs the FP32 golden.
fn golden_cell(
    sc: &Scenario,
    fmt: ArenaFmt,
    device: &Device,
    stager: &PinnedStager,
) -> candle::Result<Metrics> {
    if !sc.head_dim_supported() {
        candle::bail!("head_dim {} unsupported", sc.head_dim);
    }
    let out = build_and_decode(sc, fmt, device, stager)?;
    let gold = fixture::golden_decode(sc, device)?;
    Metrics::compute(&gold, &out, sc.n_q_head, sc.head_dim)
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
            let run = || -> candle::Result<std::time::Duration> {
                let fix = Fixture::build(sc, fmt, device, stager)?;
                for _ in 0..warmup {
                    let _ = fix.decode(device, stager)?;
                }
                let mut ts: Vec<std::time::Duration> = Vec::with_capacity(iters.max(1));
                for _ in 0..iters.max(1) {
                    let (_, dt) = fix.decode(device, stager)?;
                    ts.push(dt);
                }
                ts.sort();
                Ok(ts[ts.len() / 2])
            };
            let int8 = match run() {
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
            let int8_us = int8.as_secs_f64() * 1e6;
            let tps = sc.num_slots as f64 * 1e6 / int8_us;
            eprintln!("int8={int8_us:.1}µs ({tps:.0} tok/s)");
            rows.push(BenchRow {
                scenario: sc.name.to_string(),
                format: fmt.label(),
                num_slots: sc.num_slots,
                int8_us,
            });
        }
    }
    Ok(render_bench(&rows))
}

/// Build a fresh fixture and run a single int8 decode. The fresh build
/// guarantees pristine, deterministic input (a decode commits the write token,
/// so fixtures must not be reused across calls).
fn build_and_decode(
    sc: &Scenario,
    fmt: ArenaFmt,
    device: &Device,
    stager: &PinnedStager,
) -> candle::Result<candle::Tensor> {
    let fix = Fixture::build(sc, fmt, device, stager)?;
    Ok(fix.decode(device, stager)?.0)
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
