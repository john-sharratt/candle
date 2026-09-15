//! A write outside the decode kernel leaves the cached decode slot buffer
//! current.
//!
//! The decode kernel keeps its slot buffer's writer length right on the device,
//! one commit per token, and the next decode reuses the buffer as it stands. A
//! prefill or a speculative verify block writes through the prefill kernel and
//! commits on the host only. Before `KvCache::commit_written_tokens` those
//! commits left the buffer's lengths short, the next decode wrote its token at
//! the stale slot, and the slot the host counted was never written — the NaN in
//! the MTP draft head's KV layer.
//!
//! Each case reads the lengths back from the device, because the host state was
//! always right: the whole defect was the device disagreeing with it.
#![cfg(feature = "cuda")]

use candle::cuda_backend::cudarc::driver::CudaSlice;
use candle::{DType, Device, Tensor};

use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
use crate::kv_cache::{ChunkedKvBacking, KvCache};

const N_KV_HEAD: usize = 2;
const HEAD_DIM: usize = 32;
const SLICE_BYTES: usize = 16;

/// A CUDA backing with one sequence and a `KvCache` over it.
fn setup(dev: &Device) -> (ChunkedKvBacking, KvCache, usize) {
    let backing = ChunkedKvBacking::new(1, N_KV_HEAD, HEAD_DIM, DType::F16, dev, 256).unwrap();
    let seq = backing.alloc_sequence().unwrap();
    let mut cache = KvCache::new(2, 256);
    cache.set_chunked_backing(&backing, seq, None).unwrap();
    (backing, cache, seq)
}

/// Write `n` tokens at `offset` the way a prefill does — outside the decode
/// kernel — and commit them.
fn write_outside_decode(cache: &mut KvCache, dev: &Device, offset: usize, n: usize) {
    KvCache::ensure_chunked_capacity_batch(&mut [&mut *cache], &[offset], n).unwrap();
    let kv = Tensor::ones((1, N_KV_HEAD, n, HEAD_DIM), DType::F16, dev).unwrap();
    cache.chunked_write_kv(offset, &kv, &kv).unwrap();
    cache.commit_written_tokens(offset, n).unwrap();
}

/// What a decode step at `offset` would read: sync the slot buffer the way the
/// decode metadata build does (ensure the write chunk, then reuse or rebuild),
/// and return every slice's `len` as the device holds it.
fn device_lens(dev: &Device, backing: &ChunkedKvBacking, seq: usize, offset: usize) -> Vec<u16> {
    device_slices(dev, backing, seq, offset)
        .into_iter()
        .map(|(len, _)| len)
        .collect()
}

/// [`device_lens`] with each slice's `rope` beside its `len` — the cumulative
/// position of the chunk's first token, which is how the kernel knows where a
/// slice sits in the sequence.
fn device_slices(
    dev: &Device,
    backing: &ChunkedKvBacking,
    seq: usize,
    offset: usize,
) -> Vec<(u16, u32)> {
    backing.ensure_for_offset(seq, offset, 1).unwrap();
    let info = backing.resolve_arena_info().unwrap();
    let (ptrs, _, _) = backing
        .sync_decode_gpu_chunks(&[(seq, offset)], &info)
        .unwrap();
    let (ptr, n_slices, _) = ptrs[0];
    let Device::Cuda(cuda) = dev else {
        unreachable!("a CUDA test");
    };
    let len = n_slices as usize * SLICE_BYTES;
    let stream = cuda.cuda_stream();
    // SAFETY: `ptr` is the live slot buffer the sync just returned, holding
    // `n_slices` 16-byte slice headers.
    let view: CudaSlice<u8> = unsafe { stream.upgrade_device_ptr::<u8>(ptr, len) };
    let host = cuda.memcpy_dtov(&view).unwrap();
    // A borrow of the buffer, not an owner: dropping it would free the slot.
    std::mem::forget(view);
    host.chunks_exact(SLICE_BYTES)
        .map(|s| {
            (
                u16::from_le_bytes([s[2], s[3]]),
                u32::from_le_bytes([s[4], s[5], s[6], s[7]]),
            )
        })
        .collect()
}

/// A decode that fills the writer and moves into a chunk claimed before the
/// buffer was serialised: nothing is pushed, so the buffer is reused, and the
/// new write slice must count every token of the chunk it left.
///
/// The kernel's committed total is where the write slice ends, `rope + len`.
/// The device keeps only the writer's `len` current; the next writer's `rope`
/// is whatever it was serialised as. A stale one — 30, where the host holds
/// 32 — puts the next write two slots past where the host commits it, and
/// those two positions read unwritten.
#[test]
fn a_writer_that_moves_into_a_claimed_chunk_is_reserialised() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let (backing, mut cache, seq) = setup(&dev);

    // Room for a 70-token prompt, claimed before the first pass: three chunks.
    KvCache::ensure_chunked_capacity_batch(&mut [&mut cache], &[0], 70).unwrap();
    write_outside_decode(&mut cache, &dev, 0, 30);
    assert_eq!(
        device_slices(&dev, &backing, seq, 30),
        vec![(30, 0), (0, 30), (0, 30)],
        "the first sync builds the buffer from the host state"
    );

    // Two decode steps, committed on the host as the driver does after each.
    cache.set_current_seq_len(31).unwrap();
    cache.set_current_seq_len(32).unwrap();
    assert_eq!(
        device_slices(&dev, &backing, seq, 32)[..2],
        [(32, 0), (0, 32)],
        "the writer moved to chunk 1: its rope must count chunk 0's 32 tokens"
    );
}

/// A commit that crosses into a claimed chunk with another still trailing:
/// both chunks it filled read their true lengths, and the new write slice
/// ends at the committed total.
#[test]
fn a_commit_that_crosses_into_a_claimed_chunk_ends_the_write_slice_at_the_total() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let (backing, mut cache, seq) = setup(&dev);

    KvCache::ensure_chunked_capacity_batch(&mut [&mut cache], &[0], 70).unwrap();
    write_outside_decode(&mut cache, &dev, 0, 10);
    assert_eq!(device_slices(&dev, &backing, seq, 10)[0], (10, 0));

    write_outside_decode(&mut cache, &dev, 10, 30);
    assert_eq!(
        device_slices(&dev, &backing, seq, 40)[..2],
        [(32, 0), (8, 32)],
        "chunk 0 filled on the way and chunk 1 holds the rest: 32 + 8 = 40"
    );

    // Decode steps fill chunk 1 and move the writer into claimed chunk 2 on
    // the reused buffer.
    for len in 41..=64 {
        cache.set_current_seq_len(len).unwrap();
    }
    assert_eq!(
        device_slices(&dev, &backing, seq, 64)[..3],
        [(32, 0), (32, 32), (0, 64)],
        "the writer moved to chunk 2: chunk 1 reads full and chunk 2 starts at 64"
    );
}

/// A block that fits the writer chunk: the reused buffer's writer length must
/// count it. Before the fix the device still said 8.
#[test]
fn a_write_that_fits_the_writer_chunk_is_counted_on_the_device() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let (backing, mut cache, seq) = setup(&dev);

    write_outside_decode(&mut cache, &dev, 0, 8);
    // A decode step builds the live buffer from the host state.
    assert_eq!(device_lens(&dev, &backing, seq, 8)[0], 8);

    // A verify block: 4 tokens through the prefill kernel, committed on the host.
    write_outside_decode(&mut cache, &dev, 8, 4);
    assert_eq!(
        device_lens(&dev, &backing, seq, 12)[0],
        12,
        "the next decode reuses the buffer, so its writer length must already count \
         the 4 tokens committed outside the decode kernel"
    );
}

/// A block that spills into the next chunk, over a buffer re-validated at the
/// old lengths after that chunk was pushed. Both the filled predecessor and the
/// new writer must read their true lengths; before the fix they read
/// `[30, 0]`.
#[test]
fn a_write_that_spills_into_the_next_chunk_is_counted_in_both() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let (backing, mut cache, seq) = setup(&dev);

    write_outside_decode(&mut cache, &dev, 0, 30);
    assert_eq!(device_lens(&dev, &backing, seq, 30)[0], 30);

    // Room for the spill is claimed, and a metadata build re-validates the
    // buffer at the pre-commit lengths before the block is written.
    KvCache::ensure_chunked_capacity_batch(&mut [&mut cache], &[30], 4).unwrap();
    assert_eq!(device_lens(&dev, &backing, seq, 30), vec![30, 0]);

    let kv = Tensor::ones((1, N_KV_HEAD, 4, HEAD_DIM), DType::F16, &dev).unwrap();
    cache.chunked_write_kv(30, &kv, &kv).unwrap();
    cache.commit_written_tokens(30, 4).unwrap();
    assert_eq!(
        device_lens(&dev, &backing, seq, 34),
        vec![32, 2],
        "a spill changes the full predecessor's length too, which a writer-slice \
         patch alone would leave stale"
    );
}

/// How many rebuilds a decode sync at `offset` performs: 1 when the live
/// buffer had been dropped, 0 when it was reused as it stood.
fn rebuilds_on_sync(backing: &ChunkedKvBacking, seq: usize, offset: usize) -> u64 {
    backing.ensure_for_offset(seq, offset, 1).unwrap();
    let info = backing.resolve_arena_info().unwrap();
    let (_, _, stats) = backing
        .sync_decode_gpu_chunks(&[(seq, offset)], &info)
        .unwrap();
    stats.rebuilds
}

/// An arena that moves leaves the live buffer's inline records — the writer
/// chunk's among them — naming the ground it left, and the decode path reuses
/// that buffer across forwards: a decode or draft write through it lands in
/// the old ground, and the chunk's new home reads that position unwritten.
/// Compaction therefore drops the buffer of every sequence holding a chunk in
/// the moved arena, and leaves the others alone.
#[test]
fn a_moved_arena_drops_the_decode_buffers_that_name_it() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let (backing, mut cache, seq) = setup(&dev);

    write_outside_decode(&mut cache, &dev, 0, 8);
    assert_eq!(
        rebuilds_on_sync(&backing, seq, 8),
        1,
        "the first sync builds the buffer"
    );
    let arena = backing.state.read().unwrap().sequences[seq]
        .as_ref()
        .unwrap()
        .chunks_slice()[0]
        .gids
        .as_slice()[0]
        .arena_idx();

    assert_eq!(backing.drop_decode_buffers_in(&[arena + 1000]).unwrap(), 0);
    assert_eq!(
        rebuilds_on_sync(&backing, seq, 8),
        0,
        "an arena the sequence does not use leaves its buffer in place"
    );

    assert_eq!(backing.drop_decode_buffers_in(&[arena]).unwrap(), 1);
    assert_eq!(
        rebuilds_on_sync(&backing, seq, 8),
        1,
        "the sequence's own arena moved, so its buffer is rebuilt against the new base"
    );
}

/// The latent wave commits at the backing rather than through a `KvCache`:
/// `set_len`, then `refresh_decode_writer_slice`, with no block length in hand.
/// The refresh must therefore cover a spill on its own — re-serialising the
/// whole writer region, the filled predecessor as well as the new writer — and
/// not patch the new writer while the full predecessor still reads 30.
#[test]
fn a_backing_refresh_after_a_spill_counts_both_chunks() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let (backing, mut cache, seq) = setup(&dev);

    write_outside_decode(&mut cache, &dev, 0, 30);
    backing.ensure_for_offset(seq, 30, 4).unwrap();
    assert_eq!(device_lens(&dev, &backing, seq, 30), vec![30, 0]);

    let kv = Tensor::ones((1, N_KV_HEAD, 4, HEAD_DIM), DType::F16, &dev).unwrap();
    cache.chunked_write_kv(30, &kv, &kv).unwrap();
    backing.set_len(seq, 34);
    backing.refresh_decode_writer_slice(&[(seq, 0)]).unwrap();
    assert_eq!(
        device_lens(&dev, &backing, seq, 34),
        vec![32, 2],
        "the writer moved to the next chunk, so the buffer's other slices are stale too"
    );
}
