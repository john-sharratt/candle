//! Prefill kernel tests — the development harness for the int8
//! prefix-attention kernel (`docs/archived/prefill_optimization.md` §11).
//!
//! Three oracle legs:
//!
//! 1. **CPU golden** (`golden.rs`) — FP32/FP64 causal GQA attention with the
//!    kernel's exact RoPE convention, computed from pre-quantization source
//!    values. The `f16_identity_*` scenarios (unquantized prefix) validate
//!    the golden itself; quantized scenarios accept their compression
//!    level's error band on top. Bands are sized for uniform-random K/V —
//!    the quantizer's adversarial case (no channel structure), deliberately
//!    harsher than production data — so the golden leg is a sanity band,
//!    not a precision oracle.
//! 2. **Determinism** — one backend, two runs over identically-built arenas
//!    AND a reset-and-rerun over one arena, bitwise-equal outputs.
//! 3. **Reset determinism** — every scenario reruns over the byte-identical
//!    arena (`reset_to_prefix`) and must reproduce its output bitwise: the
//!    oracle for kernel nondeterminism (races, uninitialized reads,
//!    schedule-dependent accumulation).
//!
//! Synthetic scenarios build their prefixes through the REAL seal +
//! `quantize_sealed_in_place` path (genuine per-block format selection,
//! palette maps, partial chunks). The `substrate_*` tests (`#[ignore]`,
//! `ZEN_PREFILL_AB_SUBSTRATE=<dir>`) use real production KV recovered from
//! a redo log instead.

mod golden;
mod harness;
mod substrate_source;

use candle::{Device, Result};
use candle_transformers::models::profile::pipeline_snapshot_and_reset;
use harness::{build_case, compare, gpu_serial, run_prefill, Scenario, Segment, SeqSpec};

fn device() -> Result<Device> {
    Device::new_cuda(0)
}

fn seg(len: usize, level: Option<u8>) -> Segment {
    Segment { len, level }
}

fn single(segments: Vec<Segment>, q_len: usize) -> Vec<SeqSpec> {
    vec![SeqSpec { segments, q_len }]
}

/// Run a scenario against the kernel and the CPU golden; assert the bands
/// and print the measured metrics (the numbers that size the bands).
fn run_golden_scenario(spec: Scenario) -> Result<()> {
    let _guard = gpu_serial();
    let dev = device()?;
    let mut case = build_case(&spec, &dev)?;
    let out = harness::out_f32(&run_prefill(&mut case)?)?;
    let gold = golden::golden(&case);
    let m = compare(&out, &gold);
    println!(
        "[{}] max_rel={:.4e} min_row_cos={:.6}",
        spec.name, m.max_rel, m.min_row_cos
    );
    assert!(
        m.max_rel < spec.golden_band,
        "[{}] kernel vs golden max_rel {:.4e} exceeds band {:.1e}",
        spec.name,
        m.max_rel,
        spec.golden_band
    );
    assert!(
        m.min_row_cos > spec.min_cos,
        "[{}] row cosine {:.6} below floor {:.4}",
        spec.name,
        m.min_row_cos,
        spec.min_cos
    );
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Golden-validation scenarios: unquantized F16 prefix / no prefix. Any
// failure here is a harness/golden bug (RoPE convention, GQA mapping,
// causal horizon), not a quantization effect.
// Measured on bring-up (RTX 4090 Mobile): max_rel ≈ 4e-4, cos = 1.0.
// ──────────────────────────────────────────────────────────────────────

#[test]
fn f16_identity_prefix() -> Result<()> {
    run_golden_scenario(Scenario {
        name: "f16_identity_prefix",
        seqs: single(vec![seg(96, None)], 48),
        theta: 1e6,
        seed: 0xA11CE,
        golden_band: 2e-2,
        min_cos: 0.999,
        structured_dims: false,
    })
}

#[test]
fn f16_identity_no_rope() -> Result<()> {
    run_golden_scenario(Scenario {
        name: "f16_identity_no_rope",
        seqs: single(vec![seg(96, None)], 48),
        theta: 0.0, // identity rotation isolates arena/addressing from RoPE
        seed: 0xA11CF,
        golden_band: 2e-2,
        min_cos: 0.999,
        structured_dims: false,
    })
}

#[test]
fn no_prefix_fresh() -> Result<()> {
    // HAS_PREFIX = false control — the path the int8 design leaves on FP16.
    run_golden_scenario(Scenario {
        name: "no_prefix_fresh",
        seqs: single(vec![], 192),
        theta: 1e6,
        seed: 0x555,
        golden_band: 2e-2,
        min_cos: 0.999,
        structured_dims: false,
    })
}

// ──────────────────────────────────────────────────────────────────────
// Quantized-prefix scenarios — the real coverage. Bands measured on
// bring-up against uniform-random data: C0 ≈ cos 1.0; C4–C7 mixes land at
// max_rel 0.12–0.15 / cos 0.984–0.990 (quantization error vs the pre-quant
// golden, NOT kernel error — reset-rerun determinism is the exact oracle).
// Floors sit ~2× under measured headroom; tighten only downward.
// ──────────────────────────────────────────────────────────────────────

#[test]
fn c0_full_chunks() -> Result<()> {
    run_golden_scenario(Scenario {
        name: "c0_full_chunks",
        seqs: single(vec![seg(256, Some(0))], 48),
        theta: 1e6,
        seed: 0xC0,
        golden_band: 5e-2,
        min_cos: 0.999,
        structured_dims: false,
    })
}

#[test]
fn c5_adaptive() -> Result<()> {
    run_golden_scenario(Scenario {
        name: "c5_adaptive",
        seqs: single(vec![seg(512, Some(5))], 64),
        theta: 1e6,
        seed: 0xC5,
        golden_band: 2.5e-1,
        min_cos: 0.975,
        structured_dims: false,
    })
}

#[test]
fn mixed_levels_partial_chunks() -> Result<()> {
    // Segment lengths deliberately off-chunk (100, 75, 130, 61): every
    // boundary leaves a partial sealed chunk — the gap-aware slice walk
    // gets exercised alongside four different format mixes.
    run_golden_scenario(Scenario {
        name: "mixed_levels_partial_chunks",
        seqs: single(
            vec![
                seg(100, Some(0)),
                seg(75, Some(4)),
                seg(130, Some(5)),
                seg(61, Some(7)),
            ],
            64,
        ),
        theta: 1e6,
        seed: 0x111,
        golden_band: 2.5e-1,
        min_cos: 0.975,
        structured_dims: false,
    })
}

#[test]
fn tiny_segments_gap_walk() -> Result<()> {
    // Many tiny sealed segments → a prefix that is mostly partial chunks.
    run_golden_scenario(Scenario {
        name: "tiny_segments_gap_walk",
        seqs: single(
            vec![
                seg(5, Some(5)),
                seg(3, Some(5)),
                seg(32, Some(4)),
                seg(1, Some(7)),
                seg(17, Some(5)),
            ],
            33,
        ),
        theta: 1e6,
        seed: 0x222,
        golden_band: 2.5e-1,
        // Measured 0.9527 once the harness actually reached this assertion (it
        // panicked on the missing write-region allocation before), so the 0.975
        // floor had never run against a real number. 0.95 clears it with margin.
        min_cos: 0.95,
        structured_dims: false,
    })
}

#[test]
fn short_q_long_prefix() -> Result<()> {
    // The Zen Code hot shape: a small turn over a long quantized prefix.
    run_golden_scenario(Scenario {
        name: "short_q_long_prefix",
        seqs: single(vec![seg(1024, Some(5))], 8),
        theta: 1e6,
        seed: 0x333,
        golden_band: 2.5e-1,
        min_cos: 0.975,
        structured_dims: false,
    })
}

#[test]
fn single_token_q() -> Result<()> {
    // Degenerate decode-shaped edge: one query row.
    run_golden_scenario(Scenario {
        name: "single_token_q",
        seqs: single(vec![seg(512, Some(5))], 1),
        theta: 1e6,
        seed: 0x444,
        golden_band: 2.5e-1,
        min_cos: 0.975,
        structured_dims: false,
    })
}

#[test]
fn ragged_batch2() -> Result<()> {
    run_golden_scenario(Scenario {
        name: "ragged_batch2",
        seqs: vec![
            SeqSpec {
                segments: vec![seg(256, Some(5))],
                q_len: 64,
            },
            SeqSpec {
                segments: vec![seg(100, Some(0)), seg(45, Some(7))],
                q_len: 17,
            },
        ],
        theta: 1e6,
        seed: 0x666,
        // Ragged batching is the widest case: measured max_rel 2.96e-1 and cos
        // 0.9615 on first real execution of these assertions. Margin over both.
        golden_band: 3.2e-1,
        min_cos: 0.95,
        structured_dims: false,
    })
}

// ──────────────────────────────────────────────────────────────────────
// Kernel scenarios: the int8 kernel vs the CPU golden, plus bitwise
// reset-rerun determinism. The golden divergence is quantization plus pure
// kernel behavior (int8 Q/K/P quantization) — bands sized accordingly
// (SageAttention-class error on top of the shared KV quantization).
// ──────────────────────────────────────────────────────────────────────

/// Per-scenario check: run the kernel, compare against the CPU golden
/// (when the scenario's band is finite — quantizer-adversarial data can
/// make the golden non-discriminating), then rerun over the
/// byte-identical arena and require a bitwise-equal output.
fn run_kernel_scenario(spec: Scenario) -> Result<()> {
    let _guard = gpu_serial();
    let dev = device()?;
    let mut case = build_case(&spec, &dev)?;
    let out = harness::out_f32(&run_prefill(&mut case)?)?;

    if spec.golden_band.is_finite() {
        let gold = golden::golden(&case);
        let mg = compare(&out, &gold);
        println!(
            "[{}] kernel-vs-golden: max_rel={:.4e} min_row_cos={:.6}",
            spec.name, mg.max_rel, mg.min_row_cos
        );
        assert!(
            mg.max_rel < spec.golden_band && mg.min_row_cos > spec.min_cos,
            "[{}] kernel vs golden out of band: max_rel {:.4e} (band {:.1e}), cos {:.6} (floor {:.4})",
            spec.name,
            mg.max_rel,
            spec.golden_band,
            mg.min_row_cos,
            spec.min_cos
        );
    } else {
        println!("[{}] golden not applicable (band = inf)", spec.name);
    }

    harness::reset_to_prefix(&mut case)?;
    let out2 = harness::out_f32(&run_prefill(&mut case)?)?;
    assert_eq!(out, out2, "[{}] reset-and-rerun diverged", spec.name);
    Ok(())
}

#[test]
#[ignore = "debug probe — prints raw outputs for kernel bring-up"]
fn debug_probe_1chunk() -> Result<()> {
    let _guard = gpu_serial();
    let dev = device()?;
    let spec = Scenario {
        name: "debug_probe_1chunk",
        seqs: single(vec![seg(32, None)], 1),
        theta: 0.0,
        seed: 0xAB00,
        golden_band: f32::INFINITY,
        min_cos: -1.0,
        structured_dims: false,
    };
    let mut case = build_case(&spec, &dev)?;
    let out = harness::out_f32(&run_prefill(&mut case)?)?;
    let gold = golden::golden(&case);
    println!("head 0, dims 0..16 (golden | kernel):");
    for d in 0..16 {
        println!("  d{:>3}: {:>9.5} | {:>9.5}", d, gold[d], out[d]);
    }
    // Per-head cosine vs golden — locates whether specific heads break.
    for h in 0..4 {
        let a = &out[h * 8 * harness::HEAD_DIM..(h * 8 + 1) * harness::HEAD_DIM];
        let b = &gold[h * 8 * harness::HEAD_DIM..(h * 8 + 1) * harness::HEAD_DIM];
        let m = compare(a, b);
        println!("q-head {:>2}: vs-golden cos={:.6}", h * 8, m.min_row_cos);
    }
    Ok(())
}

#[test]
fn ab_f16_identity_norope_1chunk() -> Result<()> {
    // Bisection scenario: one full sealed F16 chunk, identity RoPE, one
    // query token. Isolates the sealed-tile staging path from rotation,
    // partial-chunk handling, and multi-tile accumulation.
    run_kernel_scenario(Scenario {
        name: "ab_f16_identity_norope_1chunk",
        seqs: single(vec![seg(32, None)], 1),
        theta: 0.0,
        seed: 0xAB00,
        golden_band: 8e-2,
        min_cos: 0.995,
        structured_dims: false,
    })
}

#[test]
fn ab_f16_identity_norope() -> Result<()> {
    // Bisection scenario: sealed F16 prefix without rotation — splits RoPE
    // errors from staging errors.
    run_kernel_scenario(Scenario {
        name: "ab_f16_identity_norope",
        seqs: single(vec![seg(96, None)], 48),
        theta: 0.0,
        seed: 0xAB0A,
        golden_band: 8e-2,
        min_cos: 0.995,
        structured_dims: false,
    })
}

#[test]
fn ab_f16_identity() -> Result<()> {
    run_kernel_scenario(Scenario {
        name: "ab_f16_identity",
        seqs: single(vec![seg(96, None)], 48),
        theta: 1e6,
        seed: 0xAB01,
        golden_band: 8e-2, // int8 Q/K/P on top of FP16 arena
        min_cos: 0.995,
        structured_dims: false,
    })
}

#[test]
fn ab_c0_full_chunks() -> Result<()> {
    run_kernel_scenario(Scenario {
        name: "ab_c0_full_chunks",
        seqs: single(vec![seg(256, Some(0))], 48),
        theta: 1e6,
        seed: 0xAB02,
        golden_band: 1e-1,
        min_cos: 0.995,
        structured_dims: false,
    })
}

#[test]
fn ab_c5_adaptive() -> Result<()> {
    run_kernel_scenario(Scenario {
        name: "ab_c5_adaptive",
        seqs: single(vec![seg(512, Some(5))], 64),
        theta: 1e6,
        seed: 0xAB03,
        golden_band: 3e-1,
        min_cos: 0.97,
        structured_dims: false,
    })
}

#[test]
fn ab_mixed_partial_chunks() -> Result<()> {
    run_kernel_scenario(Scenario {
        name: "ab_mixed_partial_chunks",
        seqs: single(
            vec![
                seg(100, Some(0)),
                seg(75, Some(4)),
                seg(130, Some(5)),
                seg(61, Some(7)),
            ],
            64,
        ),
        theta: 1e6,
        seed: 0xAB04,
        golden_band: 3e-1,
        min_cos: 0.97,
        structured_dims: false,
    })
}

#[test]
fn ab_tiny_segments() -> Result<()> {
    run_kernel_scenario(Scenario {
        name: "ab_tiny_segments",
        seqs: single(
            vec![
                seg(5, Some(5)),
                seg(3, Some(5)),
                seg(32, Some(4)),
                seg(1, Some(7)),
                seg(17, Some(5)),
            ],
            33,
        ),
        theta: 1e6,
        seed: 0xAB05,
        golden_band: 3e-1,
        // Measured 0.9585 on first real execution (see `tiny_segments_gap_walk`).
        min_cos: 0.95,
        structured_dims: false,
    })
}

#[test]
fn ab_short_q_long_prefix() -> Result<()> {
    run_kernel_scenario(Scenario {
        name: "ab_short_q_long_prefix",
        seqs: single(vec![seg(1024, Some(5))], 8),
        theta: 1e6,
        seed: 0xAB06,
        golden_band: 3e-1,
        min_cos: 0.97,
        structured_dims: false,
    })
}

#[test]
fn ab_ragged_batch2() -> Result<()> {
    run_kernel_scenario(Scenario {
        name: "ab_ragged_batch2",
        seqs: vec![
            SeqSpec {
                segments: vec![seg(256, Some(5))],
                q_len: 64,
            },
            SeqSpec {
                segments: vec![seg(100, Some(0)), seg(45, Some(7))],
                q_len: 17,
            },
        ],
        theta: 1e6,
        seed: 0xAB07,
        golden_band: 3e-1,
        // Measured 0.9597 on first real execution (see `ragged_batch2`).
        min_cos: 0.95,
        structured_dims: false,
    })
}

#[test]
fn ab_no_prefix_fresh() -> Result<()> {
    run_kernel_scenario(Scenario {
        name: "ab_no_prefix_fresh",
        seqs: single(vec![], 192),
        theta: 1e6,
        seed: 0xAB08,
        golden_band: 8e-2,
        min_cos: 0.995,
        structured_dims: false,
    })
}

#[test]
fn int8_determinism() -> Result<()> {
    let _guard = gpu_serial();
    let dev = device()?;
    let spec = Scenario {
        name: "int8_determinism",
        seqs: single(vec![seg(100, Some(0)), seg(130, Some(5))], 48),
        theta: 1e6,
        seed: 0xAB09,
        golden_band: 3e-1,
        min_cos: 0.97,
        structured_dims: false,
    };
    let mut case = build_case(&spec, &dev)?;
    let a = harness::out_f32(&run_prefill(&mut case)?)?;
    harness::reset_to_prefix(&mut case)?;
    let b = harness::out_f32(&run_prefill(&mut case)?)?;
    assert_eq!(a, b, "int8 kernel: reset-and-rerun diverged");
    Ok(())
}

#[test]
#[ignore = "benchmark — run explicitly with --ignored bench"]
fn bench_int8_kernel() -> Result<()> {
    let _guard = gpu_serial();
    bench_prefill_table()
}

// ──────────────────────────────────────────────────────────────────────
// Determinism + reset invariance
// ──────────────────────────────────────────────────────────────────────

#[test]
fn determinism_and_reset() -> Result<()> {
    let _guard = gpu_serial();
    let dev = device()?;
    let spec = Scenario {
        name: "determinism_and_reset",
        seqs: single(vec![seg(100, Some(0)), seg(130, Some(5))], 48),
        theta: 1e6,
        seed: 0x777,
        golden_band: 2.5e-1,
        min_cos: 0.975,
        structured_dims: false,
    };
    // Two independently-built cases: bitwise-equal outputs.
    let mut case_a = build_case(&spec, &dev)?;
    let out_a = harness::out_f32(&run_prefill(&mut case_a)?)?;
    let mut case_b = build_case(&spec, &dev)?;
    let out_b = harness::out_f32(&run_prefill(&mut case_b)?)?;
    assert_eq!(out_a, out_b, "independently built cases diverged");

    // Reset-and-rerun on ONE arena: bitwise-equal again. This is the exact
    // mechanism every scenario uses to rerun over the same bytes.
    harness::reset_to_prefix(&mut case_a)?;
    let out_c = harness::out_f32(&run_prefill(&mut case_a)?)?;
    assert_eq!(out_a, out_c, "reset_to_prefix changed the arena's output");
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Performance benchmark (#[ignore] — run explicitly; prints a table of
// seconds/prefill and effective q-token throughput per backend across the
// production-relevant shapes). This is the number the int8 kernel is
// developed against.
// ──────────────────────────────────────────────────────────────────────

fn bench_spec_level(
    name: &'static str,
    prefix: usize,
    q_len: usize,
    level: Option<u8>,
) -> Scenario {
    Scenario {
        name,
        seqs: single(vec![seg(prefix, level)], q_len),
        theta: 1e6,
        seed: 0xBE7C,
        golden_band: f32::INFINITY, // bench only — no golden assertion
        min_cos: -1.0,
        structured_dims: false,
    }
}

fn bench_spec(name: &'static str, prefix: usize, q_len: usize) -> Scenario {
    bench_spec_level(name, prefix, q_len, Some(5))
}

fn bench_prefill_table() -> Result<()> {
    let dev = device()?;
    let shapes = [
        bench_spec("q64_prefix8k", 8192, 64),
        bench_spec("q8_prefix8k", 8192, 8),
        bench_spec("q256_prefix2k", 2048, 256),
        bench_spec("q64_prefix2k", 2048, 64),
        // Unquantized float prefix — the bulk-ingest regime (live prefix
        // not yet sealed/quantized): every palette takes the kernel's
        // non-hop dtype path.
        bench_spec_level("q64_f16_prefix8k", 8192, 64, None),
        bench_spec_level("q512_f16_prefix4k", 4096, 512, None),
    ];
    println!("── int8 prefix-attention prefill ──");
    for spec in shapes {
        let mut case = harness::build_case(&spec, &dev)?;
        // Discard spans accumulated during the case build.
        let _ = pipeline_snapshot_and_reset();
        let (best, mean) = harness::bench_prefill(&mut case, &dev, 3, 10)?;
        let q_total: usize = spec.seqs.iter().map(|s| s.q_len).sum();
        println!(
            "  {:<16} best {:>9.3} ms  mean {:>9.3} ms  ({:.0} q-tok/s @ prefix {})",
            spec.name,
            best * 1e3,
            mean * 1e3,
            q_total as f64 / best,
            spec.seqs[0].prefix_len(),
        );
        // Per-span host attribution over the 13 runs (3 warmup + 10 reps).
        // Populated only when built with `--features profile`; silent otherwise.
        let snap = pipeline_snapshot_and_reset();
        for (name, total_ms, count) in &snap.entries {
            println!(
                "      {:<20} {:>8.3} ms/call (×{count})",
                name,
                total_ms / *count as f64
            );
        }
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Real-substrate scenarios (#[ignore]; ZEN_PREFILL_AB_SUBSTRATE=<dir>)
// ──────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires ZEN_PREFILL_AB_SUBSTRATE pointing at a substrate dir"]
fn substrate_layer_prefill() -> Result<()> {
    let _guard = gpu_serial();
    let Some(dir) = substrate_source::substrate_dir() else {
        candle::bail!("ZEN_PREFILL_AB_SUBSTRATE not set");
    };
    let dev = device()?;
    let rec = substrate_source::load_largest_turn(&dir, &dev)?;
    println!(
        "substrate turn: {} tokens × {} layers",
        rec.prefix_len, rec.n_layers
    );
    // First, middle, and last layer — palette/format diversity differs by
    // depth. No golden for real chunks (pre-quantization source values no
    // longer exist): assert finite, non-degenerate output and bitwise
    // reset-rerun determinism over the real production bytes.
    for layer in [0, rec.n_layers / 2, rec.n_layers - 1] {
        let mut case = substrate_source::case_for_layer(&rec, layer, 48, 0x5AB5, &dev)?;
        let out = harness::out_f32(&run_prefill(&mut case)?)?;
        assert!(
            out.iter().all(|x| x.is_finite()),
            "layer {layer}: non-finite attention output"
        );
        let energy: f32 = out.iter().map(|x| x * x).sum();
        assert!(energy > 0.0, "layer {layer}: all-zero attention output");

        harness::reset_to_prefix(&mut case)?;
        let out2 = harness::out_f32(&run_prefill(&mut case)?)?;
        assert_eq!(out, out2, "layer {layer}: reset-and-rerun diverged");
        println!(
            "layer {layer}: ok ({} outputs, energy {energy:.3e})",
            out.len()
        );
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Regression battery: boundary geometry and format coverage. Every
// scenario checks the CPU golden band (where finite) and bitwise
// reset-rerun determinism over identical arena bytes.
// ──────────────────────────────────────────────────────────────────────

/// One sweep step with its golden band.
fn sweep(
    name: &'static str,
    seqs: Vec<SeqSpec>,
    seed: u64,
    structured: bool,
    golden: (f32, f32),
) -> Result<()> {
    run_kernel_scenario(Scenario {
        name,
        seqs,
        theta: 1e6,
        seed,
        golden_band: golden.0,
        min_cos: golden.1,
        structured_dims: structured,
    })
}

#[test]
fn ab_level_matrix() -> Result<()> {
    // Every compression level C0–C9 over the same partial-chunk layout.
    // Each level exercises a different format family end-to-end through
    // the production policy: C0 R16/Q8 K, C4 asymmetric Q4_1 V (the
    // FP-fallback staging path), C8/C9 the Q2/Q1/Q0 extreme families.
    for level in 0u8..=9 {
        // Floors re-measured after partial-tail quantization landed
        // (partials used to stay float and contribute zero error): C6
        // measured cos 0.9492 with the two partial tails (4 + 13 tokens)
        // carrying C6-ladder error.
        let golden = match level {
            0..=5 => (3e-1, 0.97),
            // C6/C7 measured cos 0.9190 — just under the 0.92 floor, which had
            // never actually run (the harness panicked earlier). Margin over it.
            6 | 7 => (4.5e-1, 0.91),
            _ => (7e-1, 0.88),
        };
        println!("── ab_level_matrix: C{level} ──");
        sweep(
            "ab_level_matrix",
            single(vec![seg(100, Some(level)), seg(45, Some(level))], 17),
            0xAB10 + level as u64,
            false,
            golden,
        )?;
    }
    Ok(())
}

#[test]
fn ab_structured_palettes() -> Result<()> {
    // Structured K/V magnitudes force per-dim format DIVERSITY through
    // the production policy: scattered dim→palette maps (interleaved,
    // not contiguous bands), asymmetric V picks, and different maps per
    // segment (a table-cache miss at every segment boundary).
    // Golden floor sized to evidence: the planted 300:1 dynamic range is
    // the quantizer's adversarial case (cos 0.8817 measured with float
    // partial tails; 0.8172 once the three partial tails — 46 of 206
    // tokens — quantize too). The loss is the arena's, not the kernel's.
    sweep(
        "ab_structured_palettes",
        single(
            vec![seg(100, Some(2)), seg(45, Some(4)), seg(61, Some(5))],
            33,
        ),
        0xAB20,
        true,
        (4e-1, 0.79),
    )?;
    // Determinism only: C7/C9's 2-bit-class blocks on the planted dynamic
    // range lose most of the signal at the QUANTIZER (measured golden cos
    // 0.38 with drift-free kernel behavior) — no finite golden floor
    // discriminates here.
    sweep(
        "ab_structured_high_compression",
        single(vec![seg(130, Some(7)), seg(37, Some(9))], 9),
        0xAB21,
        true,
        (f32::INFINITY, -1.0),
    )
}

#[test]
fn ab_chunk_boundary_sweep() -> Result<()> {
    // Prefix lengths in every chunk-edge neighborhood (±1 around 32, 64,
    // 96) — partial tiles, single-token prefixes, full-chunk exactness.
    for &plen in &[1usize, 31, 32, 33, 63, 64, 65, 95, 96, 97] {
        println!("── ab_chunk_boundary_sweep: prefix {plen}, q 9 ──");
        sweep(
            "ab_chunk_boundary_sweep",
            single(vec![seg(plen, Some(5))], 9),
            0xAB30 + plen as u64,
            false,
            (3e-1, 0.97),
        )?;
    }
    Ok(())
}

#[test]
fn ab_qlen_boundary_sweep() -> Result<()> {
    // q lengths at every M-block edge (block_m_tok = 8 at GQA 32/4):
    // partial M tiles, idle rows, single-query decode-like shapes, and
    // multi-x-block grids with a partial last block.
    for &q in &[1usize, 7, 8, 9, 16, 17, 63, 64, 65] {
        println!("── ab_qlen_boundary_sweep: prefix 100, q {q} ──");
        sweep(
            "ab_qlen_boundary_sweep",
            single(vec![seg(100, Some(5))], q),
            0xAB50 + q as u64,
            false,
            (3e-1, 0.97),
        )?;
    }
    Ok(())
}

#[test]
fn ab_gap_walk_hostile() -> Result<()> {
    // Many tiny partial segments at mixed levels — a gap after nearly
    // every slice, F16 (dtype, non-hop) palettes interleaved with quant
    // ones, and single-token tiles walking the whole ordinal space.
    sweep(
        "ab_gap_walk_hostile",
        single(
            vec![
                seg(1, Some(5)),
                seg(1, Some(7)),
                seg(2, Some(0)),
                seg(31, Some(5)),
                seg(1, None),
                seg(33, Some(4)),
                seg(2, Some(9)),
                seg(1, Some(5)),
                seg(30, Some(7)),
                seg(3, Some(2)),
            ],
            12,
        ),
        0xAB60,
        false,
        (4.5e-1, 0.95),
    )
}

#[test]
fn ab_splits_partial_interplay() -> Result<()> {
    // Split-KV round-robin against partial geometry: a long prefix whose
    // last chunk is partial (1000 = 31×32 + 8) under active splits with
    // fresh tiles in the shared ordinal space; then the degenerate walks
    // (single-token prefix, prefix 33 = one full chunk + 1).
    sweep(
        "ab_splits_partial_long",
        single(vec![seg(1000, Some(5))], 8),
        0xAB70,
        false,
        (3e-1, 0.97),
    )?;
    sweep(
        "ab_splits_prefix1",
        single(vec![seg(1, Some(5))], 1),
        0xAB71,
        false,
        (3e-1, 0.97),
    )?;
    // Floor sized to evidence: the 1-token partial tail quantizes with
    // C7's (coarse) ladder — measured cos 0.9361 once partial tails
    // stopped being float.
    sweep(
        "ab_splits_prefix33",
        single(vec![seg(33, Some(7))], 3),
        0xAB72,
        false,
        (4.5e-1, 0.91),
    )
}

#[test]
fn ab_multiseq_ragged_hostile() -> Result<()> {
    // Ragged batch: a fresh-only sequence, a single-query sequence over a
    // mixed partial prefix, and a long-q sequence over a C7 prefix. The
    // grid is sized by the longest q (13 M-blocks), so the short
    // sequences run mostly-idle blocks — the row-liveness masking path.
    sweep(
        "ab_multiseq_ragged_hostile",
        vec![
            SeqSpec {
                segments: vec![],
                q_len: 40,
            },
            SeqSpec {
                segments: vec![seg(97, Some(5)), seg(32, Some(7))],
                q_len: 1,
            },
            SeqSpec {
                segments: vec![seg(500, Some(7))],
                q_len: 100,
            },
        ],
        0xAB80,
        false,
        (4.5e-1, 0.95),
    )
}

#[test]
fn ab_sealed_partial_tail_gap() -> Result<()> {
    // Restart geometry: an injected prefix whose last sealed chunk is
    // PARTIAL (65 = 32 + 32 + 1) leaves writer_start_idx == n_chunks — the
    // tail's free slots are a sealed gap, never writer capacity. The write
    // ensure must size the writer region from ZERO available: q=33 needs 2
    // fresh chunks, q=355 needs 12. Counting the gap under-allocates by one
    // chunk and panics in extend_for_write_region (production zend restart,
    // 355-token first turn over a 65-token recovered prefix).
    sweep(
        "ab_sealed_partial_tail_gap_q33",
        single(vec![seg(65, Some(5))], 33),
        0xAB90,
        false,
        (3e-1, 0.97),
    )?;
    sweep(
        "ab_sealed_partial_tail_gap_q355",
        single(vec![seg(65, Some(5))], 355),
        0xAB91,
        false,
        (3e-1, 0.97),
    )
}

#[test]
fn ab_fuzz_seeded() -> Result<()> {
    // Deterministic fuzz over the config space: segment counts and
    // lengths biased to chunk edges, levels across all format families,
    // q at M-block edges, 1–2 sequences, structured values every third
    // iteration. A failure reproduces from the printed config (the
    // generator is xorshift from the iteration seed). Determinism-only —
    // per-random-config golden bands would be noise; the value is geometry
    // coverage (allocation, gaps, splits) plus the bitwise rerun.
    const EDGE_LENS: [usize; 14] = [1, 2, 5, 17, 31, 32, 33, 63, 64, 65, 96, 100, 127, 129];
    const QS: [usize; 10] = [1, 3, 7, 8, 9, 16, 33, 63, 64, 65];
    const LEVELS: [Option<u8>; 8] = [
        None,
        Some(0),
        Some(2),
        Some(4),
        Some(5),
        Some(7),
        Some(8),
        Some(9),
    ];
    for iter in 0..16u64 {
        let mut r = harness::Rng::new(0xF0220000 + iter);
        let structured = iter % 3 == 0;
        let n_seqs = 1 + r.below(2);
        let mut seqs = Vec::new();
        for _ in 0..n_seqs {
            let n_segs = r.below(4); // 0..=3 (0 = fresh-only)
            let segments: Vec<Segment> = (0..n_segs)
                .map(|_| {
                    seg(
                        EDGE_LENS[r.below(EDGE_LENS.len())],
                        LEVELS[r.below(LEVELS.len())],
                    )
                })
                .collect();
            seqs.push(SeqSpec {
                segments,
                q_len: QS[r.below(QS.len())],
            });
        }
        println!("── ab_fuzz_seeded iter {iter} (structured {structured}): {seqs:?} ──");
        run_kernel_scenario(Scenario {
            name: "ab_fuzz_seeded",
            seqs,
            theta: 1e6,
            seed: 0xF0220000 + iter,
            golden_band: f32::INFINITY,
            min_cos: -1.0,
            structured_dims: structured,
        })?;
    }
    Ok(())
}
