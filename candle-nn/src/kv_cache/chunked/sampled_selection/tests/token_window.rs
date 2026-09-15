//! The fused palette-4 selection reads every slot outside a chunk's token
//! window as zero. The inputs are staged in the production layout — K in R16
//! (one dim's 32 tokens per block, so the lane is the token) and V in F32
//! token-major `[t][pd]` bands (so a 32-lane load covers half a token row at
//! head_dim 256) — with windows that start past offset 0.

#![cfg(feature = "cuda")]

use super::{candidate_formats, f16, PagedSelectionGpuInputs, SampleFormat, CHUNK_SIZE};
use crate::kv_cache::arena_table::N_PALETTE;
use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
use crate::kv_cache::chunked::sampled_selection::{
    DEFAULT_REPORT_ARENA_CHUNKS, PRODUCTION_K_QREL_HIGH_THRESHOLDS,
    PRODUCTION_K_QREL_LOW_THRESHOLDS, PRODUCTION_V_QREL_HIGH_THRESHOLDS,
    PRODUCTION_V_QREL_LOW_THRESHOLDS,
};

const N_KV_HEAD: usize = 2;
const HEAD_DIM: usize = 256;
const SUB_HEAD_DIM: usize = HEAD_DIM / N_PALETTE;

/// Everything `select_palette4_formats_fused` returns.
type Selection = (
    Vec<[SampleFormat; 4]>,
    Vec<[SampleFormat; 4]>,
    Vec<[f32; 4]>,
    Vec<[f32; 4]>,
    Vec<u8>,
    Vec<u8>,
    Vec<f32>,
    Vec<f32>,
);

/// Index of head `h`, band `p`, token `t`, band dim `d` in the
/// `[H][P][T][D']` chunk layout the staging reads.
fn at(h: usize, p: usize, t: usize, d: usize) -> usize {
    (h * HEAD_DIM + p * SUB_HEAD_DIM) * CHUNK_SIZE + t * SUB_HEAD_DIM + d
}

/// One chunk with tokens `[lo, hi)` live and every other token `dead`.
fn chunk(lo: usize, hi: usize, dead: f32, phase: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; N_KV_HEAD * HEAD_DIM * CHUNK_SIZE];
    for h in 0..N_KV_HEAD {
        for p in 0..N_PALETTE {
            for t in 0..CHUNK_SIZE {
                for d in 0..SUB_HEAD_DIM {
                    let x = (h * HEAD_DIM + p * SUB_HEAD_DIM + d) as f32 * 0.013
                        + t as f32 * 0.07
                        + phase;
                    out[at(h, p, t, d)] = if (lo..hi).contains(&t) {
                        x.sin() * 0.9 + 0.05
                    } else {
                        dead
                    };
                }
            }
        }
    }
    out
}

/// `max |x|` of head `h` over the live tokens, after the staging's own
/// rounding (`as_f16` for the R16 K band).
fn live_amax(c: &[f32], lo: usize, hi: usize, h: usize, as_f16: bool) -> f32 {
    let mut amax = 0.0f32;
    for p in 0..N_PALETTE {
        for t in lo..hi {
            for d in 0..SUB_HEAD_DIM {
                let x = c[at(h, p, t, d)];
                let x = if as_f16 { f16::from_f32(x).to_f32() } else { x };
                amax = amax.max(x.abs());
            }
        }
    }
    amax
}

fn select(k: &[f32], v: &[f32], lo: usize, hi: usize, dev: &candle::CudaDevice) -> Selection {
    let q = vec![0.0f32; k.len()];
    let (_backing, inputs) = PagedSelectionGpuInputs::from_f32_chunks_with_q(
        &[k],
        &[v],
        &[q.as_slice()],
        N_KV_HEAD * HEAD_DIM,
        N_KV_HEAD,
        DEFAULT_REPORT_ARENA_CHUNKS,
        None,
        dev,
    )
    .expect("stage paged selection inputs");
    let candidates = candidate_formats();
    let window = ((lo as i32) << 8) | (hi - lo) as i32;
    inputs
        .select_palette4_formats_fused(
            &candidates,
            &candidates,
            PRODUCTION_K_QREL_HIGH_THRESHOLDS[4],
            PRODUCTION_K_QREL_LOW_THRESHOLDS[4],
            PRODUCTION_V_QREL_HIGH_THRESHOLDS[4],
            PRODUCTION_V_QREL_LOW_THRESHOLDS[4],
            Some(&[window][..]),
            None,
        )
        .expect("fused palette4 selection")
}

/// A partial chunk's dead slots hold whatever the recycled ground held, and
/// the selection takes head amaxes, block amaxes and candidate scales over
/// every slot it loads. It reads the slots outside the window as zero, so a
/// chunk whose dead slots hold `+inf` selects exactly what the zero-padded
/// chunk selects, and each head's amax is the maximum over its live tokens.
#[test]
fn fused_selection_reads_slots_outside_the_window_as_zero() {
    let _gpu = gpu_serial();
    let Ok(candle::Device::Cuda(dev)) = candle::Device::cuda_if_available(0) else {
        return;
    };
    for (lo, hi) in [(3usize, 10usize), (0, 7), (20, 32), (9, 10)] {
        let (k0, v0) = (chunk(lo, hi, 0.0, 0.0), chunk(lo, hi, 0.0, 1.3));
        let (k_inf, v_inf) = (
            chunk(lo, hi, f32::INFINITY, 0.0),
            chunk(lo, hi, f32::INFINITY, 1.3),
        );
        let zero_padded = select(&k0, &v0, lo, hi, &dev);
        let stale = select(&k_inf, &v_inf, lo, hi, &dev);
        assert_eq!(
            stale, zero_padded,
            "window [{lo}, {hi}): the selection changed with the dead slots' contents"
        );
        let (k_amax, v_amax) = (&zero_padded.6, &zero_padded.7);
        for h in 0..N_KV_HEAD {
            assert_eq!(
                k_amax[h],
                live_amax(&k0, lo, hi, h, true),
                "window [{lo}, {hi}): head {h} K amax is not its live tokens' maximum"
            );
            assert_eq!(
                v_amax[h],
                live_amax(&v0, lo, hi, h, false),
                "window [{lo}, {hi}): head {h} V amax is not its live tokens' maximum"
            );
        }
    }
}
