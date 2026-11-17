//! A detached free hands the slot-state serialisation's pins back with the
//! chunks, so the caller drops both together, across threads, instead of the
//! pins dropping one layer at a time under the layer's state lock.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use candle::{DType, Device, Tensor};

use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
use crate::kv_cache::{ChunkedKvBacking, KvCache};

const N_KV_HEAD: usize = 2;
const HEAD_DIM: usize = 32;

/// **The freed table carries the serialisation's pins**: one per chunk the
/// decode slot buffer described, the very vector the slot held, and nothing is
/// left pinned once the table drops.
#[test]
fn a_detached_free_hands_back_the_pins_with_the_chunks() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let backing = ChunkedKvBacking::new(1, N_KV_HEAD, HEAD_DIM, DType::F16, &dev, 256).unwrap();
    let seq = backing.alloc_sequence().unwrap();
    let mut cache = KvCache::new(2, 256);
    cache.set_chunked_backing(&backing, seq, None).unwrap();
    let n = 40;
    KvCache::ensure_chunked_capacity_batch(&mut [&mut cache], &[0], n).unwrap();
    let kv = Tensor::ones((1, N_KV_HEAD, n, HEAD_DIM), DType::F16, &dev).unwrap();
    cache.chunked_write_kv(0, &kv, &kv).unwrap();
    cache.commit_written_tokens(0, n).unwrap();

    // A decode step serialises the slot, pinning every chunk it describes.
    backing.ensure_for_offset(seq, n, 1).unwrap();
    let info = backing.resolve_arena_info().unwrap();
    let generation = backing.begin_stager_generation_required();
    backing
        .sync_decode_gpu_chunks(&[(seq, n)], &info, &generation)
        .unwrap();
    let held = {
        let state = backing.state.read().unwrap();
        state.sequences[seq].as_ref().unwrap().gpu_chunk_pins()
    };
    assert_eq!(held.len(), 2, "two chunks: 32 tokens, then the writer");
    drop(cache);

    let table = backing.free_sequence_detached(seq).unwrap();
    assert_eq!(table.len(), 2);
    assert!(
        Arc::ptr_eq(&table.pins, &held),
        "the table holds the slot's own pins, not a fresh vector"
    );
    assert_eq!(
        Arc::strong_count(&held),
        2,
        "the slot let go of them: only the table and this test hold the vector"
    );
    drop(table);
    assert_eq!(
        Arc::strong_count(&held),
        1,
        "and the table's drop releases them"
    );
}
