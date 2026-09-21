//! Correctness gates + tuning harness for `qsa_page_place`.
//!
//! The kernel lays a QSA index page out in the scorer's channel-blocked
//! `[head_dim/4, rows, 4]` layout. Pages are stored un-rotated and the scorer
//! rotates each key as it loads it, so placing is a pure transpose — it replaces
//! a `reshape → transpose → contiguous` chain with one batched launch.
//!
//! # What is gated, and why in this file
//!
//! The kernel is pure bandwidth — `rows · head_dim` floats in, the same out —
//! so it is tuned, and a tuned kernel needs its correctness pinned against a
//! reference that does not move while the tuning does. Both live here so one
//! command answers both questions:
//!
//! - **A placement is the transpose it replaces**, bit for bit.
//! - **A batch places every page into its own buffer**, the narrow jobs' spare
//!   row tiles writing nothing.
//! - **Every tile is the same kernel.** Each `tile_r` arm must agree with the
//!   default bit for bit; a tuning change cannot become a numerics change.
//!
//! # Running it
//!
//! ```text
//! cargo test --release --features cuda -p candle-transformers \
//!     --test qsa_page_place_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `bench_qsa_page_place_tile_sweep` says which shape is slow;
//! `bench_qsa_page_place_every_shape` launches each one exactly once so `ncu`
//! can say why, and `bench_qsa_page_place_plan_cost` reports the half of a
//! placement that is not the kernel. The sweep prints the `ncu` line.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use candle::{Device, Result, Tensor};
use candle_transformers::models::qwen4exp::place::{
    place_pages, PlacePage, Placement, PLACE_TILE_R,
};

/// The released indexer geometry (`qsa_index_bench`): 128-wide keys, one block
/// key per 4 tokens.
const HEAD_DIM: usize = 128;

/// Full-attention layers in the released checkpoint — a placement rebuilds
/// every page on every one of them, so this is the natural job multiplier.
const ATTN_LAYERS: usize = 12;

/// The tiles the kernel is compiled for. Every one must be numerically
/// identical; only their speed differs.
const TILES: [usize; 4] = [8, 16, 32, 64];

/// One GPU at a time: the deep cases allocate tens of MB per page per layer.
fn gpu_serial() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Deterministic values in a stable range — a hash, not an RNG, so a rerun on
/// another machine compares against the same numbers.
fn seeded(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            ((x >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn page_keys(rows: usize, seed: u64, dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(seeded(rows * HEAD_DIM, seed), (rows, HEAD_DIM), dev)
}

/// The layout the kernel replaces: `[rows, d] → [d/4, rows, 4]`.
fn reference_blocked(keys: &Tensor) -> Result<Tensor> {
    let (rows, dim) = keys.dims2()?;
    keys.reshape((rows, dim / 4, 4))?
        .transpose(0, 1)?
        .contiguous()
}

fn to_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_vec1::<f32>()
}

/// Bit-for-bit: a transpose moves bytes and computes nothing.
fn assert_bits(got: &Tensor, want: &Tensor, what: &str) -> Result<()> {
    let g = to_vec(got)?;
    let w = to_vec(want)?;
    assert_eq!(g.len(), w.len(), "{what}: length");
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: element {i} — kernel {a} against reference {b}",
        );
    }
    Ok(())
}

// ── Correctness ──────────────────────────────────────────────────────────────

/// A placement is the channel-blocked transpose of the page, bit for bit, at
/// every width — including ones that are not a multiple of any tile.
#[test]
fn a_placement_is_the_transpose_it_replaces() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    for &rows in &[1usize, 7, 64, 501] {
        let keys = page_keys(rows, 0xA11CE ^ rows as u64, &dev)?;
        let got = place_pages(&[PlacePage { keys: &keys }], PLACE_TILE_R)?;
        assert_bits(&got[0], &reference_blocked(&keys)?, &format!("rows {rows}"))?;
    }
    Ok(())
}

/// A placement is batched over every page, and the pages are not all the same
/// width. The narrow jobs' spare row-tiles must retire without writing
/// anything, and every job must land in its own buffer.
#[test]
fn a_batch_of_unequal_pages_each_get_their_own_placement() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let widths = [1usize, 33, 512, 7, 129];
    let keys: Vec<Tensor> = widths
        .iter()
        .enumerate()
        .map(|(i, &r)| page_keys(r, 0x5EED + i as u64, &dev))
        .collect::<Result<_>>()?;
    let pages: Vec<PlacePage<'_>> = keys.iter().map(|k| PlacePage { keys: k }).collect();

    let got = place_pages(&pages, PLACE_TILE_R)?;
    assert_eq!(got.len(), widths.len());
    for (i, (k, out)) in keys.iter().zip(&got).enumerate() {
        assert_bits(out, &reference_blocked(k)?, &format!("job {i}"))?;
    }
    Ok(())
}

/// **Tuning may not become a numerics change.** Every compiled tile is a work
/// partition of the same transpose, so they must agree with the default bit
/// for bit, and the sweep below is then free to pick any of them.
#[test]
fn every_row_tile_computes_the_same_placement() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let rows = 777;
    let keys = page_keys(rows, 0x0D1A_1CE5, &dev)?;
    let want = place_pages(&[PlacePage { keys: &keys }], PLACE_TILE_R)?;
    for &tile in &TILES {
        let got = place_pages(&[PlacePage { keys: &keys }], tile)?;
        assert_bits(&got[0], &want[0], &format!("tile_r {tile}"))?;
    }
    // And the default is one of the compiled arms, not a fifth path.
    assert!(
        TILES.contains(&PLACE_TILE_R),
        "PLACE_TILE_R {PLACE_TILE_R} is not one of the compiled tiles {TILES:?}",
    );
    Ok(())
}

// ── Tuning ───────────────────────────────────────────────────────────────────

/// One case's shape: how many pages of how many rows, across every attention
/// layer.
struct Case {
    label: &'static str,
    /// Pages the placement installs, per layer.
    pages: usize,
    /// Rows in each — `tokens / ratio`, so a 2,048-token turn is 512.
    rows: usize,
}

const CASES: [Case; 6] = [
    Case {
        label: "one short turn",
        pages: 1,
        rows: 128,
    },
    Case {
        label: "one long turn",
        pages: 1,
        rows: 2048,
    },
    Case {
        label: "a projection (8 turns)",
        pages: 8,
        rows: 512,
    },
    Case {
        label: "a projection (32 turns)",
        pages: 32,
        rows: 512,
    },
    Case {
        label: "many tiny pages",
        pages: 128,
        rows: 24,
    },
    Case {
        label: "one deep section",
        pages: 1,
        rows: 32_768,
    },
];

fn case_keys(case: &Case, dev: &Device) -> Result<Vec<Tensor>> {
    (0..case.pages * ATTN_LAYERS)
        .map(|i| page_keys(case.rows, 0x1234 + i as u64, dev))
        .collect()
}

/// Time `iters` launches of one planned placement, returning ms per launch.
///
/// The plan is hoisted deliberately: planning allocates one buffer per page per
/// attention layer and the kernel runs in tens of microseconds, so timing
/// plan-plus-launch measures the allocator. [`bench_qsa_page_place_plan_cost`]
/// reports that half.
fn time_launch(plan: &Placement, tile: usize, dev: &Device) -> Result<f64> {
    for _ in 0..3 {
        plan.run(tile)?;
    }
    dev.synchronize()?;
    let iters = 50;
    let t = Instant::now();
    for _ in 0..iters {
        plan.run(tile)?;
    }
    dev.synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

/// **The sweep.** Every case at every tile, reported as achieved bandwidth —
/// the kernel reads `rows · head_dim` floats and writes the same, so GB/s
/// against the card's peak is the only figure that says whether it is done.
#[test]
#[ignore = "tuning harness; run with --ignored --nocapture"]
fn bench_qsa_page_place_tile_sweep() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;

    println!("\n  qsa_page_place — row-tile sweep (ms per launch, GB/s achieved)");
    println!("  head_dim {HEAD_DIM}, {ATTN_LAYERS} layers");
    print!("  {:>24} {:>7} {:>7} {:>9}", "case", "pages", "rows", "MB");
    for t in TILES {
        print!(" {:>15}", format!("tile {t}"));
    }
    println!();

    for case in &CASES {
        let keys = case_keys(case, &dev)?;
        let pages: Vec<PlacePage<'_>> = keys.iter().map(|k| PlacePage { keys: k }).collect();
        let n_jobs = pages.len();
        // Read plus write, which is all this kernel does.
        let bytes = 2.0 * (n_jobs * case.rows * HEAD_DIM * 4) as f64;
        print!(
            "  {:>24} {:>7} {:>7} {:>9.1}",
            case.label,
            case.pages,
            case.rows,
            bytes / 2.0 / 1e6,
        );
        let plan = Placement::plan(&pages)?;
        let mut best = (f64::MAX, 0usize);
        let mut ms_of = Vec::with_capacity(TILES.len());
        for &tile in &TILES {
            let ms = time_launch(&plan, tile, &dev)?;
            if ms < best.0 {
                best = (ms, tile);
            }
            ms_of.push(ms);
        }
        for (&tile, &ms) in TILES.iter().zip(&ms_of) {
            let gbs = bytes / (ms * 1e-3) / 1e9;
            let mark = if tile == best.1 { '*' } else { ' ' };
            print!(" {:>13}{}", format!("{ms:.3}/{gbs:.0}"), mark);
        }
        println!();
    }

    println!(
        "\n  profile one shape:\n    ncu --set full --kernel-name regex:place_kernel \\\n\
         \x20     --launch-skip 8 --launch-count 1 \\\n\
         \x20     <this binary> --ignored --exact \\\n\
         \x20     bench_qsa_page_place_every_shape\n"
    );
    Ok(())
}

/// **The other half of a placement, and on the small shapes the larger half.**
///
/// Planning allocates one buffer per page per attention layer and resolves two
/// device pointers for each. The sweep above hoists that so it can see the
/// kernel; this reports it, because a projection pays both.
#[test]
#[ignore = "tuning harness; run with --ignored --nocapture"]
fn bench_qsa_page_place_plan_cost() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;

    println!("\n  qsa_page_place — plan against launch (ms)");
    println!(
        "  {:>24} {:>7} {:>9} {:>9} {:>9} {:>7}",
        "case", "jobs", "plan", "launch", "total", "plan %"
    );
    for case in &CASES {
        let keys = case_keys(case, &dev)?;
        let pages: Vec<PlacePage<'_>> = keys.iter().map(|k| PlacePage { keys: k }).collect();
        let n_jobs = pages.len();

        // Planning allocates, so it is timed on its own with the previous
        // plan already dropped — otherwise the pool grows across iterations
        // and the later ones measure a different allocator state.
        let iters = 20;
        for _ in 0..3 {
            let _ = Placement::plan(&pages)?;
        }
        dev.synchronize()?;
        let t = Instant::now();
        for _ in 0..iters {
            let _ = Placement::plan(&pages)?;
        }
        dev.synchronize()?;
        let plan_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let plan = Placement::plan(&pages)?;
        let launch_ms = time_launch(&plan, PLACE_TILE_R, &dev)?;
        let total = plan_ms + launch_ms;
        println!(
            "  {:>24} {n_jobs:>7} {plan_ms:>9.3} {launch_ms:>9.3} {total:>9.3} {:>6.0}%",
            case.label,
            100.0 * plan_ms / total,
        );
    }
    Ok(())
}

/// **Every case, launched exactly once, in order** — the profiling target.
///
/// ```text
/// ncu --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
/// sm__warps_active.avg.pct_of_peak_sustained_active,\
/// launch__registers_per_thread,launch__waves_per_multiprocessor,\
/// l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
///     --kernel-name regex:place_kernel --launch-count 6 \
///     <this binary> --ignored --exact bench_qsa_page_place_every_shape
/// ```
#[test]
#[ignore = "profiling target; run under ncu"]
fn bench_qsa_page_place_every_shape() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    for case in &CASES {
        let keys = case_keys(case, &dev)?;
        let pages: Vec<PlacePage<'_>> = keys.iter().map(|k| PlacePage { keys: k }).collect();
        let n_jobs = pages.len();
        let plan = Placement::plan(&pages)?;
        plan.run(PLACE_TILE_R)?;
        dev.synchronize()?;
        println!(
            "  launched: {:>24}  grid ({}, {n_jobs})",
            case.label,
            case.rows.div_ceil(PLACE_TILE_R),
        );
    }
    Ok(())
}
