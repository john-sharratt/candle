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
//!   cargo run --release --features cuda --example decode_ab -- profile \
//!       --scenarios gqa4_ctx2048_b16 --formats rq-uni-q0_v-L0,rq-uni-q0_x-L0

mod fixture;
mod formats;
mod metrics;
mod report;
mod scenarios;
mod timing;

use anyhow::{bail, Context, Result};
use candle::quantized::pinned_staging::PinnedStager;
use candle::{DType, Device};
use clap::{Parser, Subcommand};

use fixture::{Fixture, Rope};
use formats::{
    all_formats, deep_formats, default_formats, quant_formats, select_formats, ArenaFmt,
};
use metrics::Metrics;
use report::{
    render_bench, render_golden, render_profile, BenchRow, GoldenOutcome, GoldenRow, ProfileRow,
};
use scenarios::{
    default_scenarios, flash_next_deep_scenarios, flash_next_holed_scenarios, flash_next_scenarios,
    perf_scenarios, select_scenarios, single_decode_scenarios, suite_deep_scenarios,
    suite_scenarios, Scenario,
};
use timing::median;

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
    /// Check the INT8 decode kernel against FP32 attention (identity RoPE) over
    /// the context AS STORED — dequantized, scaled and placed exactly as the
    /// kernels read it — and gate on that; the cosine against the unquantized
    /// truth is reported beside it as the format's precision.
    Compare {
        /// Pass gate: min cosine of the int8 output vs FP32 attention over the
        /// stored context. With the storage format's loss taken out of the
        /// reference, what is left is the kernel's own INT8 arithmetic, the
        /// same few parts in 10⁴ at every format and compression level; a
        /// structural bug — a wrong block, rank, scale, side or codebook —
        /// falls far below it at any precision. Calibrated on the prefill
        /// sweep over every format (RTX 4090 Mobile, 2026-10-01): the lowest
        /// correct cell reads 0.99971 (uniform Q0_X), the host codec
        /// mismatches the gate was built to catch read 0.98 and below.
        #[arg(long, default_value_t = 0.999)]
        stored_cosine_tol: f32,
    },
    /// Check the INT8 prefill kernel the same way: FP32 causal attention over
    /// the stored context plus the fresh tokens, gated as `compare` gates the
    /// decode.
    ComparePrefill {
        /// Pass gate: min cosine vs FP32 attention over the stored context.
        #[arg(long, default_value_t = 0.999)]
        stored_cosine_tol: f32,
        /// Fresh tokens per slot in the checked prefill.
        #[arg(long, default_value_t = 64)]
        prefill_tokens: usize,
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
    /// Device time of every production stage that touches the stored arena:
    /// the seal (format selection + palette conversion), a decode step, and a
    /// batched prefill step over the stored context. Run it under `nsys
    /// profile -t cuda` for the per-kernel breakdown inside each stage.
    Profile {
        /// Timed decode steps per cell.
        #[arg(long, default_value_t = 50)]
        iters: usize,
        /// Timed prefill steps per cell.
        #[arg(long, default_value_t = 8)]
        prefill_iters: usize,
        /// New tokens per slot in each prefill step.
        #[arg(long, default_value_t = 64)]
        prefill_tokens: usize,
        /// Untimed warmup iterations of each timed stage.
        #[arg(long, default_value_t = 5)]
        warmup: usize,
    },
    /// Comprehensive ground-truth regression suite: run the golden gate across
    /// the full quant × shape matrix in one pass. Defaults to a codec sweep
    /// (every codec at shallow/mid shapes) plus a depth/scale sweep (production
    /// native-INT8 formats at deep & large-batch shapes); `--scenarios` /
    /// `--formats` / `--all-formats` override either axis.
    Suite {
        /// Pass gate: min cosine vs FP32 attention over the stored context.
        #[arg(long, default_value_t = 0.999)]
        stored_cosine_tol: f32,
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
        Cmd::Compare { stored_cosine_tol } => run_golden(
            &scenarios,
            &fmts,
            *stored_cosine_tol,
            Stage::Decode,
            &device,
            &stager,
        )?,
        Cmd::ComparePrefill {
            stored_cosine_tol,
            prefill_tokens,
        } => run_golden(
            &scenarios,
            &fmts,
            *stored_cosine_tol,
            Stage::Prefill(*prefill_tokens),
            &device,
            &stager,
        )?,
        Cmd::Bench { iters, warmup } => {
            // Default the bench to the batch-8 perf set (fills the MMA M dim),
            // the batch-1 deep-context single-decode set (the grid-starved
            // regime split-KV targets), and the wide-head Flash-Next shape at
            // shallow and deep context; an explicit --scenarios still overrides.
            let bench_scen = if cli.scenarios.is_none() {
                let mut s = perf_scenarios();
                s.extend(single_decode_scenarios());
                s.extend(flash_next_scenarios());
                s.extend(flash_next_deep_scenarios());
                s.extend(flash_next_holed_scenarios());
                s
            } else {
                scenarios.clone()
            };
            run_bench(&bench_scen, &fmts, *iters, *warmup, &device, &stager)?
        }
        Cmd::Profile {
            iters,
            prefill_iters,
            prefill_tokens,
            warmup,
        } => run_profile(
            &scenarios,
            &fmts,
            StageIters {
                decode: *iters,
                prefill: *prefill_iters,
                prefill_tokens: *prefill_tokens,
                warmup: *warmup,
            },
            &device,
            &stager,
        )?,
        Cmd::Suite { stored_cosine_tol } => {
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
            let mut golden = String::from("# Decode suite — vs FP32 over the stored context\n");
            for (scn, fmt, label) in &groups {
                golden.push_str(&format!("\n## {label}\n\n"));
                golden.push_str(&run_golden(
                    scn,
                    fmt,
                    *stored_cosine_tol,
                    Stage::Decode,
                    &device,
                    &stager,
                )?);
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

/// Which kernel a golden check runs: one decode step, or a prefill of that
/// many fresh tokens per slot over the stored context.
#[derive(Clone, Copy)]
enum Stage {
    Decode,
    Prefill(usize),
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Decode => "decode",
            Stage::Prefill(_) => "prefill",
        }
    }
}

/// The golden gate: run the int8 kernel for each cell, compare it to FP32
/// attention over the context as stored (the gate) and over the unquantized
/// truth (the format's precision, reported), and pass/FAIL on the first.
fn run_golden(
    scenarios: &[Scenario],
    fmts: &[ArenaFmt],
    stored_cosine_tol: f32,
    stage: Stage,
    device: &Device,
    stager: &PinnedStager,
) -> Result<String> {
    let mut rows = Vec::new();
    let mut any_fail = false;
    'outer: for sc in scenarios {
        for &fmt in fmts {
            eprint!("golden  {:<24} {:<14} ... ", sc.name, fmt.label());
            let outcome = match golden_cell(sc, fmt, stage, device, stager) {
                Ok((stored, truth)) => {
                    let passed = stored.cosine >= stored_cosine_tol;
                    any_fail |= !passed;
                    eprintln!(
                        "{} stored_cos={:.6} truth_cos={:.5} truth_mae={:.2e}",
                        if passed { "pass" } else { "FAIL" },
                        stored.cosine,
                        truth.cosine,
                        truth.mae,
                    );
                    GoldenOutcome::Ran {
                        stored,
                        truth,
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
        eprintln!(
            "note: one or more GOLDEN cells FAILED — the kernel diverged from attention over \
             the context it was given."
        );
    }
    Ok(render_golden(stage.name(), &rows, stored_cosine_tol))
}

/// One cell: the int8 kernel's output against FP32 attention over the stored
/// context, and against FP32 attention over the unquantized truth. The
/// fixture is built fresh with identity RoPE (the references are plain
/// attention), which guarantees pristine, deterministic input — a decode or
/// prefill commits its tokens, so fixtures are not reused across checks. The
/// stored context is read before the kernel runs; it is the prefilled
/// tokens, which neither kernel rewrites.
fn golden_cell(
    sc: &Scenario,
    fmt: ArenaFmt,
    stage: Stage,
    device: &Device,
    stager: &PinnedStager,
) -> candle::Result<(Metrics, Metrics)> {
    if !sc.head_dim_supported() {
        candle::bail!("head_dim {} unsupported", sc.head_dim);
    }
    if matches!(
        (stage, fmt),
        (Stage::Prefill(_), ArenaFmt::Float(DType::F8E4M3))
    ) {
        // FP8 is a post-seal storage format: a prefill's compute dtype is its
        // cache's, F16/BF16 only, and production never prefills over a cache
        // forced to F8E4M3 (`prefill_utils`). Its read path is the decode's.
        candle::bail!("no FP8 prefill: FP8 is a post-seal storage format");
    }
    let mut fix = Fixture::build(sc, fmt, Rope::Identity, device, stager)?;
    let stored: Vec<(Vec<f32>, Vec<f32>)> = (0..sc.num_slots)
        .map(|s| fix.stored_context(s))
        .collect::<candle::Result<_>>()?;
    let stored_ctx = |s: usize| Ok(stored[s].clone());
    let truth_ctx = |s: usize| fixture::synthetic_context(sc, s, device);
    let (out, gold_stored, gold_truth) = match stage {
        Stage::Decode => (
            fix.decode(device, stager)?.0,
            fixture::golden_decode(sc, device, stored_ctx)?,
            fixture::golden_decode(sc, device, truth_ctx)?,
        ),
        Stage::Prefill(n) => (
            fix.prefill_step(n, device, stager)?.0,
            fixture::golden_prefill(sc, n, device, stored_ctx)?,
            fixture::golden_prefill(sc, n, device, truth_ctx)?,
        ),
    };
    Ok((
        Metrics::compute(&gold_stored, &out, sc.n_q_head, sc.head_dim)?,
        Metrics::compute(&gold_truth, &out, sc.n_q_head, sc.head_dim)?,
    ))
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
                let mut fix = Fixture::build(sc, fmt, Rope::Real, device, stager)?;
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

/// How much of each stage `run_profile` times.
struct StageIters {
    decode: usize,
    prefill: usize,
    prefill_tokens: usize,
    warmup: usize,
}

/// The stage profile: per cell, build a real-RoPE fixture (the seal is timed
/// inside the build), then time decode steps, then prefill steps — each walks
/// the contexts forward, so the prefill reads the decoded tokens too.
fn run_profile(
    scenarios: &[Scenario],
    fmts: &[ArenaFmt],
    it: StageIters,
    device: &Device,
    stager: &PinnedStager,
) -> Result<String> {
    // A cold GPU clocks differently from one under load: the first cell of a
    // run otherwise reads ~20% slow. Run it once, untimed, before the matrix.
    if let (Some(sc), Some(&fmt)) = (
        scenarios.iter().find(|s| s.head_dim_supported()),
        fmts.first(),
    ) {
        eprintln!("profile warmup {} {}", sc.name, fmt.label());
        let mut fix = Fixture::build(sc, fmt, Rope::Real, device, stager)?;
        for _ in 0..it.decode.max(1) {
            fix.decode(device, stager)?;
        }
        for _ in 0..it.prefill.max(1) {
            fix.prefill_step(it.prefill_tokens, device, stager)?;
        }
    }
    let mut rows = Vec::new();
    'outer: for sc in scenarios {
        for &fmt in fmts {
            if !sc.head_dim_supported() {
                continue;
            }
            eprint!("profile {:<24} {:<14} ... ", sc.name, fmt.label());
            let run = || -> candle::Result<ProfileRow> {
                let mut fix = Fixture::build(sc, fmt, Rope::Real, device, stager)?;
                for _ in 0..it.warmup {
                    fix.decode(device, stager)?;
                }
                let decode = median(
                    (0..it.decode.max(1))
                        .map(|_| fix.decode(device, stager).map(|(_, dt)| dt))
                        .collect::<candle::Result<_>>()?,
                );
                for _ in 0..it.warmup {
                    fix.prefill_step(it.prefill_tokens, device, stager)?;
                }
                let prefill = median(
                    (0..it.prefill.max(1))
                        .map(|_| {
                            fix.prefill_step(it.prefill_tokens, device, stager)
                                .map(|(_, dt)| dt)
                        })
                        .collect::<candle::Result<_>>()?,
                );
                Ok(ProfileRow {
                    scenario: sc.name.to_string(),
                    format: fmt.label(),
                    seal_us: fix.seal.as_secs_f64() * 1e6,
                    decode_us: decode.as_secs_f64() * 1e6,
                    prefill_tokens: it.prefill_tokens,
                    prefill_us: prefill.as_secs_f64() * 1e6,
                })
            };
            match run() {
                Ok(row) => {
                    eprintln!(
                        "seal={:.1}µs decode={:.1}µs prefill={:.1}µs",
                        row.seal_us, row.decode_us, row.prefill_us
                    );
                    rows.push(row);
                }
                Err(e) => {
                    let msg = short_err(&e);
                    eprintln!("skip ({msg})");
                    if is_context_fatal(&msg) {
                        eprintln!("FATAL: CUDA context poisoned — stopping profile early.");
                        break 'outer;
                    }
                }
            }
        }
    }
    Ok(render_profile(&rows))
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
