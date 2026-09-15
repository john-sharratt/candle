//! Test harness for `sampled_selection`: re-exports the CPU/GPU sampling API
//! under `pub(super)`, loads the golden Qwen3 KV dump
//! (`src/kv_cache/chunked/tests/data/qwen3-kv-data.bin`) when present, and
//! provides synthetic-batch/R16-packing generators used across the child test
//! modules (`benchmark`, `calibration`, `float_layout`, `gpu_vs_cpu`,
//! `helpers`, `model`, `projection`, `test_data`, `token_window`) —
//! CPU-vs-GPU parity, threshold calibration sweeps, format-selection
//! correctness against real and synthetic K/V, the selection's reading of the
//! float arenas' layout, and its bound on a partial chunk's dead slots.

// Test code: block alignment is written as the `% n == 0` the format contract
// is stated in.
#![allow(clippy::manual_is_multiple_of)]

#[allow(unused_imports)]
pub(super) use super::{
    model_compression_from_surface, sample_error_surface, sample_error_surface_cpu,
    select_smallest_passing, KvSampler, KvSamplerInputs, PagedSelectionCpuInputs, SampleFormat,
    SampleSide, SampledSelectionBenchmarkResult,
};
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub(super) use super::{
    sample_quantization_sweep_kv_paged, sample_quantization_sweep_paged, KvSamplerGpu,
    PagedSelectionGpuInputs,
};
pub(super) use crate::kv_cache::chunked::tests::dump_reader::load_dump;

#[cfg(feature = "cuda")]
pub(super) use super::sample_error_surface_gpu_paged;
pub(super) use half::f16;

use crate::kv_cache::arena_table::N_PALETTE;

pub(super) const DUMP_REL_PATH: &str = "src/kv_cache/chunked/tests/data/qwen3-kv-data.bin";
pub(super) const R16_DUMP_REL_PATH: &str = "src/kv_cache/chunked/tests/data/kv_cache_r16_dump.bin";
pub(super) const CHUNK_SIZE: usize = 32;

pub(super) fn dump_path() -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(DUMP_REL_PATH);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

pub(super) fn r16_dump_path() -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(R16_DUMP_REL_PATH);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

pub(super) fn make_synthetic_batch(n_batch: usize, n_head: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_batch * n_head * CHUNK_SIZE * head_dim];
    for b in 0..n_batch {
        for h in 0..n_head {
            for t in 0..CHUNK_SIZE {
                for d in 0..head_dim {
                    let idx = (((b * n_head + h) * head_dim + d) * CHUNK_SIZE) + t;
                    let base = (h as f32 * 0.35) + (d as f32 * 0.07) + (t as f32 * 0.015);
                    out[idx] = if b % 2 == 0 {
                        base.sin() * 0.8 + base.cos() * 0.2
                    } else {
                        let sign = if (t + d) % 2 == 0 { 1.0 } else { -1.0 };
                        sign * (0.25 + base.abs() * 1.4)
                    };
                }
            }
        }
    }
    out
}

/// `make_synthetic_batch`'s `[B][H][D][T]` chunks rearranged into the float
/// arenas' `[B][H][P][T][D']` layout: palette bands in order, each token-major.
#[cfg(feature = "cuda")]
pub(super) fn token_major_bands(dim_major: &[f32], n_head: usize, head_dim: usize) -> Vec<f32> {
    let sub = head_dim / N_PALETTE;
    let mut out = vec![0.0f32; dim_major.len()];
    for (chunk_in, chunk_out) in dim_major
        .chunks_exact(n_head * head_dim * CHUNK_SIZE)
        .zip(out.chunks_exact_mut(n_head * head_dim * CHUNK_SIZE))
    {
        for h in 0..n_head {
            for d in 0..head_dim {
                let (p, dp) = (d / sub, d % sub);
                for t in 0..CHUNK_SIZE {
                    chunk_out[(h * head_dim + p * sub) * CHUNK_SIZE + t * sub + dp] =
                        chunk_in[(h * head_dim + d) * CHUNK_SIZE + t];
                }
            }
        }
    }
    out
}

/// A `[H][P][T][D']` chunk (a dump chunk, a float arena band set) rearranged
/// into dim-major `[H][D][T]` — one dim's 32 tokens per block, the order
/// `pack_r16_blocks` packs. The inverse of [`token_major_bands`].
pub(super) fn dim_major_blocks(token_major: &[f32], n_head: usize, head_dim: usize) -> Vec<f32> {
    let sub = head_dim / N_PALETTE;
    let mut out = vec![0.0f32; token_major.len()];
    for (chunk_in, chunk_out) in token_major
        .chunks_exact(n_head * head_dim * CHUNK_SIZE)
        .zip(out.chunks_exact_mut(n_head * head_dim * CHUNK_SIZE))
    {
        for h in 0..n_head {
            for d in 0..head_dim {
                let (p, dp) = (d / sub, d % sub);
                for t in 0..CHUNK_SIZE {
                    chunk_out[(h * head_dim + d) * CHUNK_SIZE + t] =
                        chunk_in[(h * head_dim + p * sub) * CHUNK_SIZE + t * sub + dp];
                }
            }
        }
    }
    out
}

pub(super) fn candidate_formats() -> Vec<SampleFormat> {
    let (k_candidates, v_candidates) = crate::kv_cache::chunked::production_adaptive_candidates(5);
    let mut candidates = k_candidates
        .into_iter()
        .chain(v_candidates)
        .filter_map(SampleFormat::from_kv_format)
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        a.bits_per_elem()
            .partial_cmp(&b.bits_per_elem())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.table_index().cmp(&b.table_index()))
    });
    candidates.dedup();
    candidates
}

pub(super) fn pack_r16_blocks(k_data: &[f32], q_data: &[f32]) -> Vec<u8> {
    assert_eq!(k_data.len(), q_data.len());
    assert!(k_data.len() % CHUNK_SIZE == 0);
    let n_blocks = k_data.len() / CHUNK_SIZE;
    let mut buf = vec![0u8; n_blocks * 128];
    for b in 0..n_blocks {
        let block_start = b * 128;
        for i in 0..CHUNK_SIZE {
            let k_f16 = f16::from_f32(k_data[b * CHUNK_SIZE + i]);
            let q_f16 = f16::from_f32(q_data[b * CHUNK_SIZE + i]);
            buf[block_start + i * 2..block_start + i * 2 + 2].copy_from_slice(&k_f16.to_le_bytes());
            buf[block_start + 64 + i * 2..block_start + 64 + i * 2 + 2]
                .copy_from_slice(&q_f16.to_le_bytes());
        }
    }
    buf
}

pub(super) fn pack_f16(data: &[f32]) -> Vec<u8> {
    let mut buf = vec![0u8; data.len() * 2];
    for (i, &v) in data.iter().enumerate() {
        let h = f16::from_f32(v);
        buf[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
    }
    buf
}

mod benchmark;
mod calibration;
mod float_layout;
mod gpu_vs_cpu;
pub(super) mod helpers;
mod model;
mod projection;
pub(super) mod test_data;
mod token_window;
