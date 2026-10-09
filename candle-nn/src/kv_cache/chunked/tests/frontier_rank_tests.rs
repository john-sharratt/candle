//! Who holds the frontier — the region rank the compaction gate reads before it
//! chooses a pass.
//!
//! The gate sends a KV pass only when a KV arena stands at the frontier, and a
//! span-tenant pass otherwise. An arena that has been emptied but not yet swept
//! still holds its region, so it is the KV side's to lower; read as somebody
//! else's, it sent the gate to a span-tenant pass that could not move it.
#![cfg(feature = "cuda")]

use candle::{DType, Device, Tensor};

use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
use crate::kv_cache::ChunkedKvBacking;

const HEADS: usize = 16;
const HEAD_DIM: usize = 128;

/// **Freed is not swept.** A sequence's arena emptied by `free_sequence` holds
/// its region until a sweep returns it, and the KV top rank goes on naming it
/// until then — never a rank below it.
#[test]
fn an_emptied_arena_holds_the_kv_top_until_it_is_swept() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let b = ChunkedKvBacking::new(1, HEADS, HEAD_DIM, DType::F16, &dev, 4096).unwrap();
    // Empty arenas an earlier test left behind are not this test's ground.
    b.release_empty_arenas().unwrap();

    let seq = b.alloc_sequence().unwrap();
    let kv = Tensor::ones((1, HEADS, 256, HEAD_DIM), DType::F16, &dev).unwrap();
    b.write_contiguous(seq, 0, &kv, &kv).unwrap();
    b.set_len(seq, 256);
    let top = b.kv_top_rank().expect("the written arena holds a region");

    b.free_sequence(seq).unwrap();
    assert!(
        b.kv_top_rank() >= Some(top),
        "the emptied arena still holds region {top}, so the KV side still holds the top \
         (read {:?})",
        b.kv_top_rank()
    );
    b.release_empty_arenas().unwrap();
}
