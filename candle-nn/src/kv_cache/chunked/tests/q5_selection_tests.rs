//! Q5_0 / Q5_1 as selection candidates.
//!
//! The fused palette4 selection kernel only evaluates a candidate its
//! compile-time dispatch (`with_select_fmt`) knows. A format missing there is
//! skipped without error, and the search climbs to the next candidate — so a
//! ladder listing `[Q5_x, Q8_0]` silently stores everything at Q8_0. These
//! tests give the kernel data every 5-bit block passes by a wide margin and
//! require the 5-bit format to win on both sides.

#![cfg(feature = "cuda")]

use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::quantized::cuda::{ggml_to_select_qtype, select_kv_format_paged_per_head};
use candle::quantized::GgmlDType;
use candle::Device;

use crate::kv_cache::arena_table::N_PALETTE;
use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;

const HEAD_DIM: usize = 128;
const CHUNK_TOKENS: usize = 32;
const N_CHUNKS: usize = 3;
const ARENA_CHUNKS: i64 = 8192;
/// K and V for each of the four palette bands (`head * 8 + palette * 2 + is_v`).
const GIDS_PER_HEAD: usize = 8;

/// A deterministic, roughly Gaussian chunk: the sum of four LCG uniforms,
/// centred. Every 32-element block spans a similar range, so no block sits
/// near a 5-bit format's threshold.
fn chunk_values(seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let mut uniform = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    (0..CHUNK_TOKENS * HEAD_DIM)
        .map(|_| uniform() + uniform() + uniform() + uniform() - 2.0)
        .collect()
}

/// Run the per-head selection over `N_CHUNKS` F32 chunks with `candidates` on
/// both sides, returning the (K, V) head tags.
fn head_tags(candidates: &[GgmlDType]) -> (Vec<i32>, Vec<i32>) {
    let dev = match Device::cuda_if_available(0).expect("cuda_if_available") {
        Device::Cuda(d) => d,
        _ => panic!("q5 selection tests need a CUDA device"),
    };
    let stream = dev.cuda_stream();

    let chunks: Vec<_> = (0..N_CHUNKS)
        .map(|i| {
            let k = dev
                .memcpy_stod(&chunk_values(2 * i as u32 + 1))
                .expect("upload K");
            let v = dev
                .memcpy_stod(&chunk_values(2 * i as u32 + 2))
                .expect("upload V");
            (k, v)
        })
        .collect();

    // One `Palette4PerHeadEntry` row per chunk (n_kv_head = 1): four identical
    // band sub-entries, F32 source (metadata 0), unit outer scales.
    let chunk_byte_stride = (CHUNK_TOKENS * HEAD_DIM * 4) as i64;
    let outer_one = 1.0_f32.to_bits() as i64;
    let table: Vec<i64> = chunks
        .iter()
        .flat_map(|(k, v)| {
            let (k_ptr, _) = k.device_ptr(&stream);
            let (v_ptr, _) = v.device_ptr(&stream);
            [
                k_ptr as i64,
                v_ptr as i64,
                0,
                0,
                chunk_byte_stride,
                chunk_byte_stride,
                0,
                outer_one,
                outer_one,
            ]
            .repeat(N_PALETTE)
        })
        .collect();
    let table_gpu = dev.memcpy_stod(&table).expect("upload table");
    let gids: Vec<i64> = (0..N_CHUNKS)
        .flat_map(|i| [i as i64 * ARENA_CHUNKS; GIDS_PER_HEAD])
        .collect();

    // Both 5-bit formats sit near 0.03 on the K metric (mean top-4 |err| over
    // head amax) and near 3e-4 on the V metric (MSE over head amax²).
    let thr = 0.05;
    let (k_tags, v_tags) = select_kv_format_paged_per_head(
        &table_gpu,
        &gids,
        candidates,
        candidates,
        thr,
        thr,
        thr,
        thr,
        HEAD_DIM,
        1,
        ARENA_CHUNKS as usize,
        &dev,
    )
    .expect("select_kv_format_paged_per_head");
    (
        dev.memcpy_dtov(&k_tags).expect("download K tags"),
        dev.memcpy_dtov(&v_tags).expect("download V tags"),
    )
}

fn assert_selected(fmt: GgmlDType) {
    let _gpu = gpu_serial();
    let code = ggml_to_select_qtype(fmt).expect("select code");
    let (k, v) = head_tags(&[fmt, GgmlDType::Q8_0]);
    assert_eq!(k, vec![code; N_CHUNKS], "K head tags for {fmt:?}");
    assert_eq!(v, vec![code; N_CHUNKS], "V head tags for {fmt:?}");
}

#[test]
fn q5_0_is_selected_over_q8_0() {
    assert_selected(GgmlDType::Q5_0);
}

#[test]
fn q5_1_is_selected_over_q8_0() {
    assert_selected(GgmlDType::Q5_1);
}
