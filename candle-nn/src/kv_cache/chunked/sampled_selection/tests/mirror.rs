//! Shared harness for the tests that hold the GPU palette-4 selection to its CPU
//! mirror (`cpu_selection`): a deterministic data generator over one chunk, and
//! one call that runs both selections and returns their slot formats.

#![cfg(feature = "cuda")]

use half::f16;

use super::{token_major_bands, PagedSelectionGpuInputs, SampleFormat, CHUNK_SIZE};
use crate::kv_cache::chunked::cpu_selection::{select_palette4, SelectionInput};
use crate::kv_cache::chunked::sampled_selection::DEFAULT_REPORT_ARENA_CHUNKS;
use crate::kv_cache::{KvFormat, QuantFormat};

pub(super) const N_KV_HEAD: usize = 2;
/// The CPU mirror's width.
pub(super) const HEAD_DIM: usize = 128;

/// A deterministic value in [-1, 1) for `(seed, idx)`.
pub(super) fn unit(seed: u64, idx: usize) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(idx as u64);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x & 0x7FFF_FFFF) as f32 / 1_073_741_823.5 - 1.0
}

/// `[H][D][T]` data, rounded to f16 as the float arenas hold it. Each dim's
/// magnitude is its bit-reversed index, so no two blocks share an amax and the
/// sort has no ties; `shape(d, mag, t, i)` gives the value at token `t`.
pub(super) fn data(shape: impl Fn(usize, f32, usize, usize) -> f32) -> Vec<f32> {
    let mut out = vec![0.0f32; N_KV_HEAD * HEAD_DIM * CHUNK_SIZE];
    for h in 0..N_KV_HEAD {
        for d in 0..HEAD_DIM {
            let rev = (d as u32).reverse_bits() >> (32 - HEAD_DIM.trailing_zeros());
            let mag = 0.25 + rev as f32 * 0.02 + h as f32 * 0.01;
            for t in 0..CHUNK_SIZE {
                let i = ((h * HEAD_DIM + d) * CHUNK_SIZE) + t;
                out[i] = f16::from_f32(shape(d, mag, t, i)).to_f32();
            }
        }
    }
    out
}

/// Slot formats the GPU selects for K and V, and the mirror's, per head.
#[allow(clippy::type_complexity)]
pub(super) fn gpu_and_mirror(
    k: &[f32],
    v: &[f32],
    cands: &[QuantFormat],
    threshold: f32,
) -> (
    Vec<[SampleFormat; 4]>,
    Vec<[SampleFormat; 4]>,
    Vec<[SampleFormat; 4]>,
    Vec<[SampleFormat; 4]>,
) {
    let Ok(candle::Device::Cuda(dev)) = candle::Device::cuda_if_available(0) else {
        panic!("a CUDA device is required");
    };
    let sample: Vec<SampleFormat> = cands
        .iter()
        .map(|q| SampleFormat::from_kv_format(KvFormat::Quantized(*q)).expect("sample format"))
        .collect();
    let k_bands = token_major_bands(k, N_KV_HEAD, HEAD_DIM);
    let v_bands = token_major_bands(v, N_KV_HEAD, HEAD_DIM);
    let (_backing, inputs) = PagedSelectionGpuInputs::from_f32_chunks(
        &[k_bands.as_slice()],
        &[v_bands.as_slice()],
        N_KV_HEAD * HEAD_DIM,
        N_KV_HEAD,
        DEFAULT_REPORT_ARENA_CHUNKS,
        None,
        &dev,
    )
    .expect("stage paged selection inputs");
    let (k_rows, v_rows, ..) = inputs
        .select_palette4_formats_fused(
            &sample, &sample, threshold, threshold, threshold, threshold, None, None,
        )
        .expect("GPU fused palette4 selection");

    let q = vec![0u16; k.len()];
    let mirror = select_palette4(SelectionInput {
        k_data: k,
        v_data: v,
        q_data: &q,
        k_candidates: cands,
        v_candidates: cands,
        k_threshold_hi: threshold,
        k_threshold_lo: threshold,
        v_threshold_hi: threshold,
        v_threshold_lo: threshold,
        n_chunks: 1,
        n_kv_head: N_KV_HEAD,
        head_dim: HEAD_DIM,
    });
    let to_sample = |row: &[QuantFormat; 4]| -> [SampleFormat; 4] {
        std::array::from_fn(|s| {
            SampleFormat::from_kv_format(KvFormat::Quantized(row[s])).expect("sample format")
        })
    };
    let k_mirror = mirror
        .heads
        .iter()
        .map(|h| to_sample(&h.k_pal_format))
        .collect();
    let v_mirror = mirror
        .heads
        .iter()
        .map(|h| to_sample(&h.v_pal_format))
        .collect();
    (k_rows, v_rows, k_mirror, v_mirror)
}
