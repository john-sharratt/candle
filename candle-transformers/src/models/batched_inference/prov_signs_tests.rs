//! A scope's packed provenance signs: a block read from its R16 Q takes the
//! sign-pack's warps, a block the gap converted takes the bits it captured, and
//! both land in the one warp-major order the fold reads.

use std::sync::Arc;

use candle_nn::kv_cache::{ProvSource, CHUNK_SIZE};

use super::lay_out_prov_signs;

/// One block's warps at 1 head × 1 palette: `CHUNK_SIZE` words from `base`.
fn block(base: u64) -> Vec<u64> {
    (0..CHUNK_SIZE as u64).map(|t| base + t).collect()
}

fn q() -> ProvSource {
    ProvSource::Q(vec![0])
}

fn signed(base: u64) -> ProvSource {
    ProvSource::Signed(Arc::new(block(base)))
}

#[test]
fn read_and_captured_blocks_interleave_in_layer_then_block_order() {
    // Two layers of three blocks; the middle block of each was converted.
    let layers = vec![vec![q(), signed(500), q()], vec![q(), signed(700), q()]];
    // The kernel's output: every Q block's warps, layer by layer.
    let read: Vec<u64> = [100, 200, 300, 400].into_iter().flat_map(block).collect();
    let got = lay_out_prov_signs(&layers, &read, CHUNK_SIZE).unwrap();
    let want: Vec<u64> = [100, 500, 200, 300, 700, 400]
        .into_iter()
        .flat_map(block)
        .collect();
    assert_eq!(got, want);
}

#[test]
fn a_scope_wholly_converted_needs_no_kernel() {
    let layers = vec![vec![signed(1), signed(2)]];
    let got = lay_out_prov_signs(&layers, &[], CHUNK_SIZE).unwrap();
    assert_eq!(got, [block(1), block(2)].concat());
}

#[test]
fn a_kernel_output_not_the_q_blocks_warps_is_refused() {
    let layers = vec![vec![q(), q()]];
    // One block short.
    assert!(lay_out_prov_signs(&layers, &block(0), CHUNK_SIZE).is_none());
    // One block over.
    let three: Vec<u64> = [0, 1, 2].into_iter().flat_map(block).collect();
    assert!(lay_out_prov_signs(&layers, &three, CHUNK_SIZE).is_none());
    // Not a whole block.
    assert!(lay_out_prov_signs(&layers, &[0; CHUNK_SIZE * 2 + 1], CHUNK_SIZE).is_none());
}

#[test]
fn captured_bits_of_another_geometry_are_refused() {
    // Bits for 2 warps a block where this geometry has 1.
    let wide = ProvSource::Signed(Arc::new(vec![0; 2 * CHUNK_SIZE]));
    assert!(lay_out_prov_signs(&[vec![wide]], &[], CHUNK_SIZE).is_none());
    // Q pointers for 2 warps a block where this geometry has 1.
    let q2 = ProvSource::Q(vec![0, 0]);
    assert!(lay_out_prov_signs(&[vec![q2]], &[0; 2 * CHUNK_SIZE], CHUNK_SIZE).is_none());
    assert!(lay_out_prov_signs(&[vec![q()]], &[], 0).is_none());
}
