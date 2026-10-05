//! Microbenchmark for the QSA **top-k selection kernel** alone
//! (`SelectionTable::fill_rows` → `run_qsa_topk_entries`).
//!
//! `qsa_index_bench` times the whole `select_layer` — key append, query
//! projection, scoring GEMMs and the top-k together — which is the right figure
//! for the layer and the wrong one for this kernel: at depth the scoring GEMM
//! dominates and a change to the selection disappears inside it. Here the
//! scores are synthetic and uploaded once, so the only work per iteration is the
//! selection launch and its packed-metadata upload — exactly what the layer pays
//! for the top-k.
//!
//! Geometry is the released checkpoint's: ratio 4, a 2048-position budget. The
//! axes are the ones that move the kernel — rows (decode ~16, a prefill chunk of
//! 1,024 or 8,192), candidate blocks per row (16K–262K blocks, i.e. 64K–1M
//! tokens), and the strata (`docs/qsa_stratified_selection.md`): the
//! checkpoint's one ranking, and the deployed 128K-position windows with an
//! 8K-position recent span ranked as a candidate or forced.
//!
//! ```text
//! cargo test --release --features cuda -p candle-transformers \
//!     --test qsa_topk_bench -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use std::time::Instant;

use candle::{Device, Result, Tensor};
use candle_transformers::models::qwen4exp::indexer::SelectionTable;
use candle_transformers::models::qwen4exp::qsa_select::{Recent, Strata};

const RATIO: usize = 4;
const TOP_K: usize = 2048;
const WARMUP: usize = 3;
/// The deployed system prompt, in blocks: ~8K positions.
const PROMPT_BLOCKS: u32 = 2000;

/// The strata the bench sweeps, by name.
fn strata() -> [(&'static str, Strata); 3] {
    let windowed = |recent| Strata {
        window_blocks: 32_768,
        recent_blocks: 2048,
        recent,
    };
    [
        ("whole", Strata::WHOLE),
        ("win+cand", windowed(Recent::Candidate)),
        ("win+forced", windowed(Recent::Forced)),
    ]
}

/// Deterministic, non-negative scores (a sum of ReLUs never goes below zero).
fn scores(rows: usize, blocks: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..rows * blocks)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 8.0
        })
        .collect()
}

/// Mean milliseconds per selection launch over `iters`, synchronized.
fn time_fill(rows: usize, blocks: usize, strata: &Strata, iters: usize) -> Result<f64> {
    let device = Device::new_cuda(0)?;
    let host = scores(rows, blocks, 0xA5A5 ^ (rows * 31 + blocks) as u64);
    let scores = Tensor::from_vec(host, (rows, blocks), &device)?;
    // Every row sees the whole candidate axis, so each one streams `blocks`
    // keys — the deepest-row case a wave sizes its buffers for.
    let cand: Vec<u32> = vec![blocks as u32; rows];
    let qpos: Vec<usize> = vec![blocks * RATIO; rows];
    let tail: Vec<u32> = vec![1; rows];
    let prompt: Vec<u32> = vec![PROMPT_BLOCKS; rows];
    let mut table = SelectionTable::new(rows, RATIO, TOP_K, blocks, strata, &device, None)?;
    for _ in 0..WARMUP {
        table.fill_rows(&scores, &cand, &qpos, &tail, &prompt, RATIO, TOP_K, 0)?;
    }
    device.synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        table.fill_rows(&scores, &cand, &qpos, &tail, &prompt, RATIO, TOP_K, 0)?;
    }
    device.synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

/// One launch per shape and strata and nothing else — the target for `ncu`,
/// which then captures exactly the decode and the prefill selection at the
/// depth the daemon runs (294,912 tokens):
///
/// ```text
/// ncu --kernel-name regex:qsa_topk --set full <test exe> \
///     profile_qsa_topk_at_depth --ignored --exact --nocapture
/// ```
#[test]
#[ignore = "profiling target; run under ncu"]
fn profile_qsa_topk_at_depth() -> Result<()> {
    for (_, s) in strata() {
        for rows in [16usize, 1024] {
            time_fill(rows, 73_728, &s, 1)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "microbenchmark; run with --ignored --nocapture"]
fn bench_qsa_topk_selection() -> Result<()> {
    println!("qsa top-k selection (ratio {RATIO}, budget {TOP_K}, prompt {PROMPT_BLOCKS} blocks)");
    println!(
        "{:>11} {:>6} {:>8} {:>10} {:>12}",
        "strata", "rows", "blocks", "tokens", "ms/launch"
    );
    for (name, s) in strata() {
        for &(rows, iters) in &[(16usize, 200usize), (1024, 20), (8192, 4)] {
            for &blocks in &[16_384usize, 32_768, 73_728, 262_144] {
                // An 8,192-row prefill over 1M tokens is 8 GB of scores; the
                // decode and 1,024-row cases cover that depth.
                if rows * blocks * 4 > 4 << 30 {
                    continue;
                }
                let ms = time_fill(rows, blocks, &s, iters)?;
                println!(
                    "{name:>11} {rows:>6} {blocks:>8} {:>10} {ms:>12.3}",
                    blocks * RATIO
                );
            }
        }
    }
    Ok(())
}
