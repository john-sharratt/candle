//! Calibration at load: the split's measured costs on this machine, as it is
//! (`docs/cpu_expert_compute.md` §5.3.1, §5.10).
//!
//! Run by the service thread on its own pool before it serves a layer — once
//! the first forward wakes it, not while the model loads: the loader's threads
//! hold the cores and DRAM the measurement is of, and a calibration taken under
//! them inflated c(n) up to 3.4× and out of order. The table is held monotone
//! ([`monotone`]) against what contention remains.
//!
//! | Cost | Measured as |
//! |---|---|
//! | `cpu_n_ns` | eight synthetic experts of the row's real geometry and KO width through the production layer schedule, at each count bucket, best of five; per expert. Every timed layer reads experts from DRAM, as a real one does: they rotate through a set spanning [`COLD_BYTES`] |
//! | `pool_fixed_ns` | a layer of one DRAM-cold expert at one token, less that expert's share of the throughput: the per-layer cost a small layer pays whatever its experts |
//! | `cpu_read_alone_ns` | one expert's bytes over the pool's measured DRAM read rate |
//! | `copy_ns`, `row_ns` | one expert's bytes, and one row out and its result back, over the link's measured rate |
//!
//! Two start as estimates the online loop (§5.3.2) refines from the device's
//! own timestamps: the GPU's per-expert GEMM time, `gpu_n_ns`, from an int8
//! rate typical of the cards here; and the read while the link streams,
//! `cpu_read_shared_ns`, from the laptop's measured ratio (52 → 49 GB/s on a
//! four-core pool, measurement 2).

use std::hint::black_box;
use std::time::{Duration, Instant};

use candle::quantized::ko_quant::quantize_ko;
use candle::quantized::GgmlDType;

use super::kernel::act::ActTiles;
use super::kernel::expert::Expert;
use super::kernel::ko::Ko;
use super::kernel::simd::Isa;
use super::layer::{run_layer, ExpertWork, LayerWork};
use super::params::{Measured, BUCKETS};
use super::pool::Pool;
use super::service::RowShape;

/// Experts per timed layer: enough that the pool's tasks spread as a real
/// layer's do, so the cost is a throughput, not one expert's latency.
const EXPERTS: usize = 8;
/// Distinct synthetic experts quantized. Their values do not affect the time,
/// and quantizing is the slow part, so the timed set is copies of these.
const DISTINCT: usize = 8;
/// Bytes the timed experts span. A real layer's experts come from DRAM, so a
/// timed layer must too: each takes experts no timed layer has read for at
/// least half this, past the last-level cache of every host here (24 MiB on
/// the laptop, 32 MiB per CCD on a Threadripper). Timed over one set of eight
/// experts — 16 MB, inside the laptop's L3 — the costs came out at half what
/// the pool then took in decode.
const COLD_BYTES: usize = 256 << 20;
/// Timed runs per bucket; the best is kept.
const REPS: usize = 5;
/// The shared-read estimate's ratio to the read alone (measurement 2).
const SHARED_READ_RATIO: f32 = 52.0 / 49.0;
/// The GPU's int8 MAC rate per expert GEMM the estimate assumes, MAC/ns.
const GPU_MAC_PER_NS: f32 = 2_000.0;
/// The GPU GEMM's fixed cost per expert, ns.
const GPU_FIXED_NS: f32 = 1_500.0;
/// The fixed handoff the measurement found (2.2 µs round trip) plus the pull.
const HANDOFF_NS: f32 = 5_000.0;

/// The costs for `shape` on `pool`, given the pool's DRAM read rate and the
/// link's, in bytes per second.
pub fn calibrate(
    pool: &Pool,
    isa: Isa,
    shape: &RowShape,
    dram_bps: f64,
    link_bps: f64,
) -> Measured {
    let expert_bytes: usize = shape.bytes.iter().sum();
    let distinct: Vec<Slot> = (0..DISTINCT as u64).map(|s| synthetic(shape, s)).collect();
    let cold = ColdSet::new(&distinct, expert_bytes);
    let mut at = 0;
    let max_n = *BUCKETS.last().unwrap();
    let mut seed = 0x0ca1_1b8a_u64;
    let x: Vec<f32> = (0..max_n * shape.hidden).map(|_| lcg(&mut seed)).collect();
    let act = ActTiles::quantize(&x, max_n, shape.hidden);
    // As the service runs a layer: each expert's `n` tokens, a result row per
    // assignment.
    let layer = |slots: &[usize], n: usize| LayerWork {
        act: &act,
        experts: slots
            .iter()
            .enumerate()
            .map(|(j, &s)| {
                let (bytes, off) = &cold.slots[s];
                ExpertWork {
                    expert: Expert::from_slot(bytes, *off, shape.hidden, shape.ff, shape.ko),
                    tokens: (0..n).collect(),
                    weights: vec![1.0; n],
                    outputs: (j * n..(j + 1) * n).collect(),
                }
            })
            .collect(),
        rows: slots.len() * n,
    };
    let mut cpu_n_ns = [0f32; BUCKETS.len()];
    for (i, &n) in BUCKETS.iter().enumerate() {
        let mut best = Duration::MAX;
        for _ in 0..REPS {
            let work = layer(&cold.next(&mut at, EXPERTS), n);
            let t = Instant::now();
            black_box(run_layer(pool, isa, &work));
            best = best.min(t.elapsed());
        }
        cpu_n_ns[i] = best.as_nanos() as f32 / EXPERTS as f32;
    }
    monotone(&mut cpu_n_ns);
    // The pool's fixed cost: a layer of one expert at one token, against that
    // expert's share of the throughput above.
    let mut alone = Duration::MAX;
    for _ in 0..REPS {
        let one = layer(&cold.next(&mut at, 1), 1);
        let t = Instant::now();
        black_box(run_layer(pool, isa, &one));
        alone = alone.min(t.elapsed());
    }
    let pool_fixed_ns = (alone.as_nanos() as f32 - cpu_n_ns[0]).max(0.0);
    let macs_per_token = 3.0 * shape.hidden as f32 * shape.ff as f32;
    let gpu_n_ns = BUCKETS.map(|n| GPU_FIXED_NS + n as f32 * macs_per_token / GPU_MAC_PER_NS);
    let read_alone = (expert_bytes as f64 / dram_bps * 1e9) as f32;
    let row_bytes = shape.hidden + shape.hidden / 128 * 16 + shape.hidden * 4;
    Measured {
        copy_ns: (expert_bytes as f64 / link_bps * 1e9) as f32,
        cpu_read_alone_ns: read_alone,
        cpu_read_shared_ns: read_alone * SHARED_READ_RATIO,
        cpu_n_ns,
        gpu_n_ns,
        handoff_ns: HANDOFF_NS,
        pool_fixed_ns,
        row_ns: (row_bytes as f64 / link_bps * 1e9) as f32,
    }
}

/// One synthetic expert's slot image, and its gate, up and down offsets in it.
type Slot = (Vec<u8>, (usize, usize, usize));

/// The timed experts: copies of the distinct synthetic ones, [`COLD_BYTES`] of
/// them, handed out in rotation so a slot comes round again only after every
/// other has been read.
struct ColdSet {
    slots: Vec<Slot>,
}

impl ColdSet {
    fn new(distinct: &[Slot], expert_bytes: usize) -> Self {
        let n = (COLD_BYTES / expert_bytes.max(1)).max(2 * EXPERTS);
        Self {
            slots: (0..n)
                .map(|i| distinct[i % distinct.len()].clone())
                .collect(),
        }
    }

    /// The next `k` slots in rotation from `at`, which moves past them.
    fn next(&self, at: &mut usize, k: usize) -> Vec<usize> {
        rotate(at, self.slots.len(), k)
    }
}

/// `k` indices from `*at` round a ring of `len`, advancing `*at` past them.
fn rotate(at: &mut usize, len: usize, k: usize) -> Vec<usize> {
    let out = (0..k).map(|i| (*at + i) % len).collect();
    *at = (*at + k) % len;
    out
}

/// Holds a cost table to its physical shape: an expert never costs less for
/// more tokens, so each bucket is at most every bucket above it. A bucket a
/// preemption or a neighbour's burst inflated takes the cheaper count's
/// measurement above it, and the cut it implies stays one contiguous range.
pub fn monotone(t: &mut [f32]) {
    for i in (0..t.len().saturating_sub(1)).rev() {
        t[i] = t[i].min(t[i + 1]);
    }
}

/// A random expert of `shape`, quantized to its KO width, as a slot image.
fn synthetic(shape: &RowShape, seed: u64) -> Slot {
    let dt = match shape.ko {
        Ko::Q3 => GgmlDType::Q3_KO,
        Ko::Q4 => GgmlDType::Q4_KO,
    };
    let mut s = seed.wrapping_add(1);
    let mut w = |n: usize| -> Vec<f32> { (0..n).map(|_| lcg(&mut s) * 0.05).collect() };
    let (h, f) = (shape.hidden, shape.ff);
    let (g, u, d) = (
        quantize_ko(&w(f * h), f, h, dt),
        quantize_ko(&w(f * h), f, h, dt),
        quantize_ko(&w(h * f), h, f, dt),
    );
    let offsets = (0, g.len(), g.len() + u.len());
    ([g, u, d].concat(), offsets)
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The run the gate measured under the load's contention, raw: buckets
    /// 2–12 inflated past bucket 16. Each takes the cheapest measurement at or
    /// above it; an already monotone table is untouched.
    #[test]
    fn a_table_is_held_monotone_from_above() {
        let mut t = [
            149.3, 272.2, 360.9, 347.9, 359.8, 404.3, 443.2, 150.0, 230.0, 281.6,
        ];
        monotone(&mut t);
        assert_eq!(
            t,
            [149.3, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 230.0, 281.6]
        );
        let mut m = [41.4, 43.7, 53.1, 66.2];
        monotone(&mut m);
        assert_eq!(m, [41.4, 43.7, 53.1, 66.2]);
        monotone(&mut []);
    }

    /// The rotation hands out every slot once before any comes round again,
    /// across calls of different widths.
    #[test]
    fn the_cold_set_rotates_through_every_slot() {
        let mut at = 0;
        assert_eq!(rotate(&mut at, 5, 3), [0, 1, 2]);
        assert_eq!(rotate(&mut at, 5, 1), [3]);
        assert_eq!(rotate(&mut at, 5, 3), [4, 0, 1]);
        assert_eq!(at, 2);
    }

    /// Flash-Next's expert, 1,996,800 bytes, spans 256 MiB in 134 copies: a
    /// slot is reused only after 133 others, 265 MB of reads, have passed it.
    #[test]
    fn the_cold_set_spans_past_the_last_level_cache() {
        let distinct = vec![(vec![0u8; 4], (0, 1, 2)); DISTINCT];
        assert_eq!(ColdSet::new(&distinct, 1_996_800).slots.len(), 134);
        assert_eq!(ColdSet::new(&distinct, 64 << 20).slots.len(), 2 * EXPERTS);
    }
}
