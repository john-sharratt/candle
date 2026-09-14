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

use crate::kv_cache::chunked::backing::buffered_seq_indices;
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
        .map(|s| u16::from_le_bytes([s[2], s[3]]))
        .collect()
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

/// A sequence that has never decoded holds no slot buffer, so a commit has
/// nothing to bring up to date and the refresh must not name it — that is what
/// lets a fresh prefill skip the arena resolve entirely. Once a decode sync
/// builds the buffer, the same sequence is named.
#[test]
fn only_a_sequence_holding_a_decode_buffer_is_refreshed() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let (backing, mut cache, seq) = setup(&dev);

    write_outside_decode(&mut cache, &dev, 0, 8);
    assert_eq!(
        buffered_seq_indices(&backing.state.read().unwrap(), &[(seq, 0)]),
        Vec::<usize>::new(),
        "a prefilled sequence that has not decoded has no buffer to patch"
    );

    device_lens(&dev, &backing, seq, 8);
    assert_eq!(
        buffered_seq_indices(&backing.state.read().unwrap(), &[(seq, 0)]),
        vec![seq]
    );
}

/// The batched prefill commits a layer's whole batch at once. Every sequence in
/// it must come out counted on the device, exactly as a commit per sequence
/// would leave it.
#[test]
fn a_batch_commit_counts_every_sequence_on_the_device() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let backing = ChunkedKvBacking::new(2, N_KV_HEAD, HEAD_DIM, DType::F16, &dev, 256).unwrap();
    let seqs = [
        backing.alloc_sequence().unwrap(),
        backing.alloc_sequence().unwrap(),
    ];
    let mut caches = seqs.map(|seq| {
        let mut cache = KvCache::new(2, 256);
        cache.set_chunked_backing(&backing, seq, None).unwrap();
        cache
    });
    for (cache, &seq) in caches.iter_mut().zip(seqs.iter()) {
        write_outside_decode(cache, &dev, 0, 8);
        assert_eq!(device_lens(&dev, &backing, seq, 8)[0], 8);
    }

    // A verify block on both sequences, committed together.
    let [c0, c1] = &mut caches;
    let mut batch = [c0, c1];
    KvCache::ensure_chunked_capacity_batch(&mut batch, &[8, 8], 4).unwrap();
    let kv = Tensor::ones((1, N_KV_HEAD, 4, HEAD_DIM), DType::F16, &dev).unwrap();
    for cache in batch.iter_mut() {
        cache.chunked_write_kv(8, &kv, &kv).unwrap();
    }
    KvCache::commit_written_tokens_batch(&mut batch, &[8, 8], &[4, 4]).unwrap();

    for &seq in &seqs {
        assert_eq!(
            device_lens(&dev, &backing, seq, 12)[0],
            12,
            "sequence {seq}: the reused buffer must count the batch-committed block"
        );
    }
}
