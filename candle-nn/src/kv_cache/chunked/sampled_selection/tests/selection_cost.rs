//! The kernel bench for the fused palette-4 selection: production candidate
//! lists at C3, C4, C5 and C10, each over a 384-head and a 4,096-head grid.
//! Kernel times come from running it under Nsight Compute, which times every
//! launch at locked clocks:
//!
//! ```text
//! ncu --clock-control base --metrics gpu__time_duration.sum,smsp__inst_executed.sum \
//!     --kernel-name regex:'select_kv_format_palette4_paged|approximate_q_relevance_quantiles' \
//!     <candle_nn test binary> \
//!     kv_cache::chunked::sampled_selection::tests::selection_cost::selection_kernel_bench \
//!     --exact --ignored --nocapture --test-threads=1
//! ```
//!
//! Each selection call is one launch of each kernel, a workload is one warm-up
//! call then `BENCH_RUNS` timed ones, and the workloads run in the order the
//! fingerprint lines print. On a laptop part the 4,096-head times drift with
//! heat between runs even at locked clocks; compare instruction counts, or two
//! builds run back to back.
//!
//! The workload is what makes the candidate ladder work: dims of three kinds
//! (nearly flat, moderately varying, strongly varying), spread over several
//! orders of magnitude, with a sink spike in every sixteenth chunk, and a Q
//! stream so the K side's per-block q-relevance thresholds are in play.

#![cfg(feature = "cuda")]

use candle::CudaDevice;

use super::mirror::unit;
use super::{token_major_bands, PagedSelectionGpuInputs, SampleFormat, CHUNK_SIZE};
use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
use crate::kv_cache::chunked::sampled_selection::params::{
    production_adaptive_candidates, PRODUCTION_K_QREL_HIGH_THRESHOLDS,
    PRODUCTION_K_QREL_LOW_THRESHOLDS, PRODUCTION_V_QREL_HIGH_THRESHOLDS,
    PRODUCTION_V_QREL_LOW_THRESHOLDS,
};
use crate::kv_cache::chunked::sampled_selection::DEFAULT_REPORT_ARENA_CHUNKS;

const N_KV_HEAD: usize = 8;
const HEAD_DIM: usize = 128;
const BENCH_RUNS: usize = 5;

/// One chunk's `[H][D][T]` data. Dim `d` of head `h` has a magnitude on a
/// log scale over `[0.02, 4]` and one of three spreads across its tokens.
fn chunk_data(chunk: usize, seed: u64) -> Vec<f32> {
    let mut out = vec![0.0f32; N_KV_HEAD * HEAD_DIM * CHUNK_SIZE];
    for h in 0..N_KV_HEAD {
        for d in 0..HEAD_DIM {
            let key = (chunk * N_KV_HEAD + h) * HEAD_DIM + d;
            let mag = 0.02 * (200.0f32).powf((unit(seed ^ 0x11, key) + 1.0) * 0.5);
            let spread = match d % 3 {
                0 => 0.002,
                1 => 0.15,
                _ => 0.8,
            };
            for t in 0..CHUNK_SIZE {
                let i = key * CHUNK_SIZE + t;
                out[(h * HEAD_DIM + d) * CHUNK_SIZE + t] = mag * (1.0 + spread * unit(seed, i));
            }
        }
        if chunk % 16 == 0 {
            out[(h * HEAD_DIM) * CHUNK_SIZE] = 300.0;
        }
    }
    out
}

/// `n_chunks` chunks of the workload, staged on the device. The backing keeps
/// the arenas alive for as long as the inputs are used.
fn staged_inputs(dev: &CudaDevice, n_chunks: usize) -> (impl Sized, PagedSelectionGpuInputs) {
    let bands = |seed: u64| -> Vec<Vec<f32>> {
        (0..n_chunks)
            .map(|c| token_major_bands(&chunk_data(c, seed), N_KV_HEAD, HEAD_DIM))
            .collect()
    };
    let (k_chunks, v_chunks, q_chunks) = (bands(0xA1), bands(0xB2), bands(0xC3));
    let k_refs: Vec<&[f32]> = k_chunks.iter().map(Vec::as_slice).collect();
    let v_refs: Vec<&[f32]> = v_chunks.iter().map(Vec::as_slice).collect();
    let q_refs: Vec<&[f32]> = q_chunks.iter().map(Vec::as_slice).collect();
    PagedSelectionGpuInputs::from_f32_chunks_with_q(
        &k_refs,
        &v_refs,
        &q_refs,
        N_KV_HEAD * HEAD_DIM,
        N_KV_HEAD,
        DEFAULT_REPORT_ARENA_CHUNKS,
        None,
        dev,
    )
    .expect("stage paged selection inputs")
}

/// FNV-1a over every output of one selection: palette tags and scales, block
/// assignments and head amaxes. Two kernel builds that agree here made the same
/// selection bit for bit.
fn selection_fingerprint(
    inputs: &PagedSelectionGpuInputs,
    k: &[SampleFormat],
    v: &[SampleFormat],
    level: usize,
) -> u64 {
    let (k_tags, v_tags, k_scales, v_scales, k_map, v_map, k_amax, v_amax) = inputs
        .select_palette4_formats_fused(
            k,
            v,
            PRODUCTION_K_QREL_HIGH_THRESHOLDS[level],
            PRODUCTION_K_QREL_LOW_THRESHOLDS[level],
            PRODUCTION_V_QREL_HIGH_THRESHOLDS[level],
            PRODUCTION_V_QREL_LOW_THRESHOLDS[level],
            None,
            None,
        )
        .expect("GPU fused palette4 selection");
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    };
    for row in k_tags.iter().chain(v_tags.iter()) {
        for fmt in row {
            eat(format!("{fmt:?}").as_bytes());
        }
    }
    for row in k_scales.iter().chain(v_scales.iter()) {
        for s in row {
            eat(&s.to_bits().to_le_bytes());
        }
    }
    eat(&k_map);
    eat(&v_map);
    for a in k_amax.iter().chain(v_amax.iter()) {
        eat(&a.to_bits().to_le_bytes());
    }
    h
}

/// The bench. The fingerprint line names the selection each workload made: a
/// change to the kernel that is meant to be exact must leave every fingerprint
/// unchanged. 384 heads leaves most of the card idle, so it measures one head's
/// serial path; 4,096 heads fills it and measures throughput.
#[test]
#[ignore = "kernel bench: run under ncu on an idle card (see the module docs)"]
fn selection_kernel_bench() {
    let _gpu = gpu_serial();
    let Ok(candle::Device::Cuda(dev)) = candle::Device::cuda_if_available(0) else {
        return;
    };
    for n_chunks in [48usize, 512] {
        let (_backing, inputs) = staged_inputs(&dev, n_chunks);
        for level in [3usize, 4, 5, 10] {
            let (k_kv, v_kv) = production_adaptive_candidates(level as u8);
            let k: Vec<SampleFormat> = k_kv
                .iter()
                .filter_map(|f| SampleFormat::from_kv_format(*f))
                .collect();
            let v: Vec<SampleFormat> = v_kv
                .iter()
                .filter_map(|f| SampleFormat::from_kv_format(*f))
                .collect();
            let fp = selection_fingerprint(&inputs, &k, &v, level);
            for _ in 0..BENCH_RUNS {
                assert_eq!(
                    selection_fingerprint(&inputs, &k, &v, level),
                    fp,
                    "selection is not deterministic"
                );
            }
            eprintln!(
                "bench heads={} C{level} fingerprint {fp:016x}",
                n_chunks * N_KV_HEAD
            );
        }
    }
}
