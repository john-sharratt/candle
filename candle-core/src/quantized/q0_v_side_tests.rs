//! Q0_V's raw-byte decode reads with the codebook of the side it is told.
//! K and V bands carry separate tables, so a K band read with the trait's
//! V-table decode comes back as some other block's values.

use super::k_quants::{
    decode_blocks_q0_v, decode_q0_v_bytes, encode_block_q0_v, encode_q0_v_bytes, BlockQ0V, GgmlType,
};

fn k_block() -> (Vec<u8>, BlockQ0V) {
    let xs: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.4).cos() * 0.6).collect();
    let block = encode_block_q0_v::<true>(&xs);
    (vec![block.lo, block.hi], block)
}

#[test]
fn raw_bytes_decode_with_the_named_side() {
    let (bytes, block) = k_block();
    let mut k = vec![0f32; 32];
    decode_blocks_q0_v::<true>(std::slice::from_ref(&block), &mut k);
    let mut v = vec![0f32; 32];
    BlockQ0V::to_float(std::slice::from_ref(&block), &mut v);

    assert_eq!(decode_q0_v_bytes::<true>(&bytes), k);
    assert_eq!(decode_q0_v_bytes::<false>(&bytes), v);
    assert_ne!(k, v, "the K and V codebooks decode one block differently");
}

#[test]
fn raw_bytes_encode_with_the_named_side() {
    let xs: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.4).cos() * 0.6).collect();
    let k: Vec<u8> = xs
        .chunks_exact(32)
        .flat_map(|b| {
            let blk = encode_block_q0_v::<true>(b);
            [blk.lo, blk.hi]
        })
        .collect();
    assert_eq!(encode_q0_v_bytes::<true>(&xs), k);
}
