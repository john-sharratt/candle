//! The GPU palette-4 selection reads a float band in the arenas' token-major
//! layout, so its block `d` is dim `d`'s 32 tokens — the block the convert
//! encodes and the palette map assigns. On data whose dims have well
//! separated magnitudes its palette map is exactly the CPU mirror's
//! (`cpu_selection`, which takes dim-major blocks).

#![cfg(feature = "cuda")]

use super::{token_major_bands, PagedSelectionGpuInputs, SampleFormat, CHUNK_SIZE};
use crate::kv_cache::chunked::cpu_selection::{select_palette4, SelectionInput};
use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
use crate::kv_cache::chunked::sampled_selection::DEFAULT_REPORT_ARENA_CHUNKS;
use crate::kv_cache::QuantFormat;

const N_KV_HEAD: usize = 2;
/// The CPU mirror's width.
const HEAD_DIM: usize = 128;

/// `[H][D][T]` data in which each dim of each head has its own magnitude —
/// its bit-reversed index, so neighbouring dims sit far apart — and its
/// tokens alternate sign under a small ripple that peaks at 1.0. A dim's
/// block amax is exactly its magnitude, distinct from every other dim's by
/// at least 0.01, while every token row spans the same set of magnitudes.
fn separated_dims(offset: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; N_KV_HEAD * HEAD_DIM * CHUNK_SIZE];
    for h in 0..N_KV_HEAD {
        for d in 0..HEAD_DIM {
            let rev = (d as u32).reverse_bits() >> (32 - HEAD_DIM.trailing_zeros());
            let mag = 0.25 + rev as f32 * 0.02 + h as f32 * 0.01 + offset;
            for t in 0..CHUNK_SIZE {
                let sign = if t % 2 == 0 { 1.0 } else { -1.0 };
                let ripple = 1.0 - 0.002 * ((t * 7 + d) % 5) as f32;
                out[(h * HEAD_DIM + d) * CHUNK_SIZE + t] = sign * mag * ripple;
            }
        }
    }
    out
}

/// With one candidate format and thresholds every block passes, a palette map
/// is the ranking of the blocks' amaxes. A selection that read a token-major
/// float band as dim-major would rank token rows — near-ties here — and its
/// map would not be the mirror's.
#[test]
fn gpu_palette_map_matches_the_cpu_mirror_on_float_bands() {
    let _gpu = gpu_serial();
    let Ok(candle::Device::Cuda(dev)) = candle::Device::cuda_if_available(0) else {
        return;
    };
    let k = separated_dims(0.0);
    let v = separated_dims(0.5);
    let k_bands = token_major_bands(&k, N_KV_HEAD, HEAD_DIM);
    let v_bands = token_major_bands(&v, N_KV_HEAD, HEAD_DIM);
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
    let lenient = 1.0f32;
    let (_, _, _, _, k_gpu, v_gpu, _, _) = inputs
        .select_palette4_formats_fused(
            &[SampleFormat::Q8_0],
            &[SampleFormat::Q8_0],
            lenient,
            lenient,
            lenient,
            lenient,
            None,
            None,
        )
        .expect("GPU fused palette4 selection");

    let q = vec![0u16; k.len()];
    let mirror = select_palette4(SelectionInput {
        k_data: &k,
        v_data: &v,
        q_data: &q,
        k_candidates: &[QuantFormat::Q8_0],
        v_candidates: &[QuantFormat::Q8_0],
        k_threshold_hi: lenient,
        k_threshold_lo: lenient,
        v_threshold_hi: lenient,
        v_threshold_lo: lenient,
        n_chunks: 1,
        n_kv_head: N_KV_HEAD,
        head_dim: HEAD_DIM,
    });
    for (h, head) in mirror.heads.iter().enumerate() {
        let span = h * HEAD_DIM..(h + 1) * HEAD_DIM;
        assert_eq!(
            &k_gpu[span.clone()],
            &head.k_assignments[..],
            "head {h}: the GPU K palette map is not the CPU mirror's"
        );
        assert_eq!(
            &v_gpu[span],
            &head.v_assignments[..],
            "head {h}: the GPU V palette map is not the CPU mirror's"
        );
    }
}
