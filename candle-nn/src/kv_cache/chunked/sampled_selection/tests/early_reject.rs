//! The GPU search stops measuring a scale once it has failed more blocks than
//! the slot can spare. That must never change a selection: a dropped scale could
//! not have filled the slot, and when no candidate fills it the search runs
//! again without dropping to rank the fallback. The CPU mirror
//! (`cpu_selection`) measures every block at every scale, so on data that drives
//! both cases the two must pick the same slot formats.

#![cfg(feature = "cuda")]

use super::mirror::{data, gpu_and_mirror, unit};
use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
use crate::kv_cache::QuantFormat;

/// Six-scale and single-scale formats in ascending size: the cheap ones fail
/// most blocks and are dropped part-way, the larger ones decide. The GPU host
/// path sorts its candidates by size and the mirror walks them in the order
/// given, so the list is already in that order (Q2_S, 9 bytes, before Q2_A, 10).
const CANDS: [QuantFormat; 6] = [
    QuantFormat::Q1_S,
    QuantFormat::Q2_S,
    QuantFormat::Q2_A,
    QuantFormat::Q3_0,
    QuantFormat::Q4_0,
    QuantFormat::Q8_0,
];

fn varied(seed: u64) -> Vec<f32> {
    data(move |d: usize, mag: f32, t: usize, i: usize| {
        let swing = [0.05, 0.4, 1.0][d % 3];
        mag * (1.0 + swing * unit(seed, i)) * if t % 7 == 0 { -1.0 } else { 1.0 }
    })
}

/// At a zero threshold no format passes any block, so every scale is dropped
/// on the first search and every slot takes the fallback, which the second,
/// full search ranks.
#[test]
fn gpu_ranks_the_fallback_as_the_mirror_does_when_nothing_fills_a_slot() {
    let _gpu = gpu_serial();
    let (k, v) = (varied(0x11), varied(0x22));
    let (k_gpu, v_gpu, k_mirror, v_mirror) = gpu_and_mirror(&k, &v, &CANDS, 0.0);
    assert_eq!(k_gpu, k_mirror, "K slot formats differ from the mirror");
    assert_eq!(v_gpu, v_mirror, "V slot formats differ from the mirror");
}

/// At a moderate threshold the cheap formats fail and are dropped part-way
/// while a larger one fills each slot.
#[test]
fn gpu_picks_as_the_mirror_does_when_cheap_formats_are_dropped() {
    let _gpu = gpu_serial();
    let (k, v) = (varied(0x33), varied(0x44));
    for threshold in [0.002, 0.01, 0.05] {
        let (k_gpu, v_gpu, k_mirror, v_mirror) = gpu_and_mirror(&k, &v, &CANDS, threshold);
        assert_eq!(
            k_gpu, k_mirror,
            "K slot formats differ from the mirror at {threshold}"
        );
        assert_eq!(
            v_gpu, v_mirror,
            "V slot formats differ from the mirror at {threshold}"
        );
    }
}
