//! Microbenchmark + oracle for the QSA **selection path** (`select_layer`).
//!
//! This is the span the full-model profile calls `q4e:qsa_select`: per
//! full-attention layer, per wave, it appends every sequence's index keys and —
//! once some row is past the budget — scores the cache and builds the wave's
//! selection table.
//!
//! # The two regimes, which are different code
//!
//! The path forks on `selection_engages`, and a change can move one side
//! without touching the other:
//!
//! - **Short context** (≤ `top_k + ratio − 1` = 2051 attended positions): the
//!   selection would be the identity, so no table, no query projection and no
//!   scoring runs. What remains is the mandatory work — the keys are cached at
//!   every depth because a later wave scores the blocks the waves below it
//!   built. This is the regime the production gate ladder runs in
//!   (713–1466 tokens a context), so it is the one a depth sweep never sees.
//! - **Long context**: adds the query projection, a GEMM per sequence against
//!   that sequence's cache, the head fold and the top-k.
//!
//! Both are benchmarked, and both sweep the axis that actually moves them:
//! short context sweeps **wave width** (sessions), long context sweeps
//! **depth**. Width is the interesting axis for the short regime because the
//! per-sequence loop is what costs there, not the arithmetic.
//!
//! Synthetic throughout — deterministic hash-seeded weights and hidden states,
//! no checkpoint, no model load. Run:
//!
//! ```text
//! cargo test --release --features cuda -p candle-transformers \
//!     --test qsa_index_bench -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use candle::{Device, Result, Tensor};
use candle_transformers::models::delta_net::mix::SeqSpan;
use candle_transformers::models::qwen4exp::config::IndexerConfig;
use candle_transformers::models::qwen4exp::indexer::{select_layer, IndexCache};
use candle_transformers::models::qwen4exp::qsa::IndexerWeights;
use candle_transformers::models::rope_schedule::{plain_inv_freq, FactoredRope};

/// The released geometry (`the_published_geometry_parses`): 4 indexer heads of
/// 128, a 2048-position budget, hidden 2560, ratio 4 — so the selection is the
/// identity at or below 2051 attended positions.
const N_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const TOP_K: usize = 2048;
const HIDDEN: usize = 2560;
const RATIO: usize = 4;
const ROPE_DIM: usize = 64;
const ROPE_THETA: f32 = 10_000.0;
const EPS: f64 = 1e-6;

/// One GPU at a time: every case here allocates device buffers sized by depth.
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

fn tensor(dims: (usize, usize), seed: u64, dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(seeded(dims.0 * dims.1, seed), dims, dev)
}

fn idx_cfg() -> IndexerConfig {
    IndexerConfig {
        n_heads: N_HEADS,
        head_dim: HEAD_DIM,
        top_k: TOP_K,
    }
}

fn weights(dev: &Device) -> Result<IndexerWeights> {
    Ok(IndexerWeights {
        q_proj: tensor((N_HEADS * HEAD_DIM, HIDDEN), 0x51, dev)?,
        k_proj: tensor((HEAD_DIM, HIDDEN), 0x52, dev)?,
        q_norm: Tensor::from_vec(seeded(HEAD_DIM, 0x53), HEAD_DIM, dev)?,
        k_norm: Tensor::from_vec(seeded(HEAD_DIM, 0x54), HEAD_DIM, dev)?,
    })
}

/// A wave of `n_seq` sequences, each already `depth` tokens deep, about to
/// contribute `rows_per_seq` new rows.
struct Wave {
    caches: HashMap<usize, Vec<IndexCache>>,
    h: Tensor,
    spans: Vec<SeqSpan>,
    offsets: Vec<usize>,
    total_rows: usize,
}

impl Wave {
    /// Build the caches by driving the real `select_layer` over the history in
    /// wave-sized steps — the cache's internal state (how the open block is
    /// split across appends, where the block boundaries fall) is part of what
    /// is being measured, so it has to arrive the way production builds it.
    fn build(
        n_seq: usize,
        depth: usize,
        rows_per_seq: usize,
        w: &IndexerWeights,
        rope: &FactoredRope,
        dev: &Device,
    ) -> Result<Self> {
        let mut caches: HashMap<usize, Vec<IndexCache>> = HashMap::new();
        for s in 0..n_seq {
            caches.insert(s, vec![IndexCache::new(HEAD_DIM, dev)?]);
        }
        let counter = AtomicU64::new(0);

        // Prefill each sequence's history in one span per sequence, which is
        // what a bulk prefill wave does.
        if depth > 0 {
            let total = n_seq * depth;
            let h = tensor((total, HIDDEN), 0x60, dev)?;
            let spans: Vec<SeqSpan> = (0..n_seq)
                .map(|s| SeqSpan {
                    seq: s,
                    start: s * depth,
                    len: depth,
                })
                .collect();
            let offsets = vec![0usize; n_seq];
            select_layer(
                0,
                RATIO,
                w,
                rope,
                &h,
                &spans,
                &offsets,
                &mut caches,
                None,
                total,
                &idx_cfg(),
                EPS,
                dev,
                &counter,
            )?;
        }

        let total_rows = n_seq * rows_per_seq;
        let spans: Vec<SeqSpan> = (0..n_seq)
            .map(|s| SeqSpan {
                seq: s,
                start: s * rows_per_seq,
                len: rows_per_seq,
            })
            .collect();
        Ok(Self {
            caches,
            h: tensor((total_rows, HIDDEN), 0x61, dev)?,
            spans,
            offsets: vec![depth; n_seq],
            total_rows,
        })
    }

    /// One layer's selection — exactly the call the wave engine makes.
    fn run(&mut self, w: &IndexerWeights, rope: &FactoredRope, dev: &Device) -> Result<bool> {
        let counter = AtomicU64::new(0);
        let sel = select_layer(
            0,
            RATIO,
            w,
            rope,
            &self.h,
            &self.spans,
            &self.offsets,
            &mut self.caches,
            None,
            self.total_rows,
            &idx_cfg(),
            EPS,
            dev,
            &counter,
        )?;
        Ok(sel.is_some())
    }
}

/// The indexer's factored table — it covers every position below its reach, so
/// no depth sizes it.
fn rope_for(dev: &Device) -> Result<FactoredRope> {
    FactoredRope::new(&plain_inv_freq(ROPE_DIM, ROPE_THETA), dev)
}

/// Time `iters` steps, each one wave's worth of selection for one layer.
///
/// The caches grow as it runs — which is what production does — so the reported
/// figure is the mean over the run rather than a single step.
fn time_steps(
    wave: &mut Wave,
    w: &IndexerWeights,
    rope: &FactoredRope,
    dev: &Device,
    warm: usize,
    iters: usize,
) -> Result<(f64, bool)> {
    let mut engaged = false;
    for _ in 0..warm {
        engaged |= wave.run(w, rope, dev)?;
        wave.advance();
    }
    dev.synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        engaged |= wave.run(w, rope, dev)?;
        wave.advance();
    }
    dev.synchronize()?;
    Ok((t.elapsed().as_secs_f64() * 1e3 / iters as f64, engaged))
}

impl Wave {
    /// Move every sequence forward by the rows this wave just consumed.
    fn advance(&mut self) {
        let rows = self.spans.first().map(|s| s.len).unwrap_or(0);
        for off in self.offsets.iter_mut() {
            *off += rows;
        }
    }
}

/// **Short context — the regime the production gate ladder runs in.**
///
/// Every sequence sits below the budget, so no selection is computed; the cost
/// is the mandatory key caching, and the axis that moves it is how many
/// sequences share the wave. A verify wave contributes ~5 rows a sequence.
#[test]
#[ignore = "microbenchmark; run with --ignored --nocapture"]
fn bench_qsa_select_cost_vs_width() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let w = weights(&dev)?;
    let rope = rope_for(&dev)?;

    println!("\n  QSA selection — SHORT context (identity regime), cost vs wave width");
    println!(
        "  {:>8} {:>7} {:>10} {:>12} {:>10}",
        "sessions", "depth", "rows/wave", "ms/layer", "µs/session"
    );
    for &n_seq in &[1usize, 4, 8, 16] {
        let depth = 1024;
        let mut wave = Wave::build(n_seq, depth, 5, &w, &rope, &dev)?;
        let (ms, engaged) = time_steps(&mut wave, &w, &rope, &dev, 3, 20)?;
        assert!(
            !engaged,
            "depth {depth} was expected to stay below the {} budget",
            TOP_K + RATIO - 1
        );
        println!(
            "  {n_seq:>8} {depth:>7} {:>10} {ms:>12.3} {:>10.1}",
            n_seq * 5,
            ms * 1000.0 / n_seq as f64
        );
    }
    Ok(())
}

/// **Long context — the engaged regime.**
///
/// One sequence, swept past the budget so the table, the query projection, the
/// per-sequence GEMM, the head fold and the top-k all run.
#[test]
#[ignore = "microbenchmark; run with --ignored --nocapture"]
fn bench_qsa_select_cost_vs_depth() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let w = weights(&dev)?;

    println!("\n  QSA selection — LONG context (engaged regime), cost vs depth");
    println!(
        "  {:>8} {:>9} {:>12} {:>10}",
        "depth", "engaged", "ms/layer", "vs 4K"
    );
    let mut base = 0f64;
    for (i, &depth) in [4096usize, 8192, 32768, 131_072].iter().enumerate() {
        let rope = rope_for(&dev)?;
        let mut wave = Wave::build(1, depth, 5, &w, &rope, &dev)?;
        let (ms, engaged) = time_steps(&mut wave, &w, &rope, &dev, 2, 10)?;
        if i == 0 {
            base = ms;
        }
        println!(
            "  {depth:>8} {:>9} {ms:>12.3} {:>9.2}×",
            if engaged { "yes" } else { "no" },
            ms / base
        );
    }
    Ok(())
}

/// The width sweep at engaged depth — both costs at once, which is the shape a
/// deep multi-session wave actually runs.
#[test]
#[ignore = "microbenchmark; run with --ignored --nocapture"]
fn bench_qsa_select_deep_and_wide() -> Result<()> {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0)?;
    let w = weights(&dev)?;
    let depth = 8192;
    let rope = rope_for(&dev)?;

    println!("\n  QSA selection — engaged, cost vs wave width at depth {depth}");
    println!(
        "  {:>8} {:>12} {:>10}",
        "sessions", "ms/layer", "µs/session"
    );
    for &n_seq in &[1usize, 4, 8, 16] {
        let mut wave = Wave::build(n_seq, depth, 5, &w, &rope, &dev)?;
        let (ms, engaged) = time_steps(&mut wave, &w, &rope, &dev, 2, 10)?;
        assert!(engaged, "depth {depth} must engage the selection");
        println!(
            "  {n_seq:>8} {ms:>12.3} {:>10.1}",
            ms * 1000.0 / n_seq as f64
        );
    }
    Ok(())
}
