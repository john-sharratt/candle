//! The GPU palette-4 selection offers Q0 only the blocks that are flat to within
//! one INT8 step of the head scale, and divides its error metrics by that scale
//! capped at 8x the head's p95 |x|. On data that exercises both, its slot
//! formats are the CPU mirror's (`cpu_selection`).

#![cfg(feature = "cuda")]

use super::mirror::{data, gpu_and_mirror, unit, N_KV_HEAD};
use super::SampleFormat;
use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
use crate::kv_cache::{KvFormat, QuantFormat};

/// A third of the dims are flat (constant to within f16 noise), the rest swing
/// by up to ±50% of their magnitude. At a lenient threshold the error metric
/// alone lets Q0 take any of them; only the flat ones may have it.
#[test]
fn gpu_offers_q0_only_flat_blocks_as_the_mirror_does() {
    let _gpu = gpu_serial();
    let shape = |seed: u64| {
        move |d: usize, mag: f32, _t: usize, i: usize| {
            if d % 3 == 0 {
                mag
            } else {
                mag * (1.0 + 0.5 * unit(seed, i))
            }
        }
    };
    let k = data(shape(0xA1));
    let v = data(shape(0xB2));
    let cands = [QuantFormat::Q0, QuantFormat::Q4_0, QuantFormat::Q8_0];
    let (k_gpu, v_gpu, k_mirror, v_mirror) = gpu_and_mirror(&k, &v, &cands, 0.3);
    assert_eq!(k_gpu, k_mirror, "K slot formats differ from the mirror");
    assert_eq!(v_gpu, v_mirror, "V slot formats differ from the mirror");
    // The data has enough flat blocks for a Q0 slot and enough varied ones that
    // the remaining slots must not be Q0.
    let q0 = SampleFormat::from_kv_format(KvFormat::Quantized(QuantFormat::Q0)).unwrap();
    for rows in [&k_gpu, &v_gpu] {
        for row in rows {
            assert_eq!(row.iter().filter(|f| **f == q0).count(), 1, "{row:?}");
        }
    }
}

/// One element at 1000 against a bulk of σ = 1. The selection divides its
/// metrics by the capped head scale, so the slots that have a choice take Q8_0
/// where the raw amax would have let Q4_0 pass.
#[test]
fn gpu_caps_the_head_scale_as_the_mirror_does() {
    let _gpu = gpu_serial();
    let shape = |seed: u64| {
        move |d: usize, _mag: f32, t: usize, i: usize| {
            if d == 0 && t == 0 {
                1000.0
            } else {
                unit(seed, i) + unit(seed ^ 0x77, i)
            }
        }
    };
    let k = data(shape(0xC3));
    let v = data(shape(0xD4));
    let cands = [QuantFormat::Q4_0, QuantFormat::Q8_0];
    let (k_gpu, v_gpu, k_mirror, v_mirror) = gpu_and_mirror(&k, &v, &cands, 0.002);
    // The sink block is left to the last slot, where the two formats tie on its
    // error; the slots before it are the ones with a choice.
    for h in 0..N_KV_HEAD {
        assert_eq!(k_gpu[h][..3], k_mirror[h][..3], "K head {h}");
        assert_eq!(v_gpu[h][..3], v_mirror[h][..3], "V head {h}");
    }
    let q8 = SampleFormat::from_kv_format(KvFormat::Quantized(QuantFormat::Q8_0)).unwrap();
    for h in 0..N_KV_HEAD {
        assert_eq!(k_gpu[h][..3], [q8; 3], "K head {h}");
        assert_eq!(v_gpu[h][..3], [q8; 3], "V head {h}");
    }
}
