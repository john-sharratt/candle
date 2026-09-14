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

use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use candle::cuda_backend::cudarc::driver::result::memcpy_dtod_async;
use candle::cuda_backend::cudarc::driver::result::stream::launch_host_function;
use candle::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
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

/// Write `n` tokens of KV at `offset` without committing them. The tensor is
/// returned so it outlives the caller's next steps: freeing it inside a region
/// the caller times could synchronise the device, and that wait would be read as
/// the code under test's.
fn write_kv_ahead(cache: &mut KvCache, dev: &Device, offset: usize, n: usize) -> Tensor {
    KvCache::ensure_chunked_capacity_batch(&mut [&mut *cache], &[offset], n).unwrap();
    let kv = Tensor::ones((1, N_KV_HEAD, n, HEAD_DIM), DType::F16, dev).unwrap();
    cache.chunked_write_kv(offset, &kv, &kv).unwrap();
    kv
}

/// What a decode step at `offset` would read: sync the slot buffer the way the
/// decode metadata build does (ensure the write chunk, then reuse or rebuild),
/// and return every slice's `len` as the device holds it.
fn device_lens(dev: &Device, backing: &ChunkedKvBacking, seq: usize, offset: usize) -> Vec<u16> {
    let (ptr, n_slices) = live_slot(backing, seq, offset);
    read_lens(dev, ptr, n_slices)
}

/// The live slot buffer a decode step at `offset` would read — its device
/// pointer and slice count — synced the way the decode metadata build does.
fn live_slot(backing: &ChunkedKvBacking, seq: usize, offset: usize) -> (u64, usize) {
    backing.ensure_for_offset(seq, offset, 1).unwrap();
    let info = backing.resolve_arena_info().unwrap();
    let (ptrs, _, _) = backing
        .sync_decode_gpu_chunks(&[(seq, offset)], &info)
        .unwrap();
    let (ptr, n_slices, _) = ptrs[0];
    (ptr, n_slices as usize)
}

/// Every slice's `len` as the device holds it at `ptr`.
fn read_lens(dev: &Device, ptr: u64, n_slices: usize) -> Vec<u16> {
    let Device::Cuda(cuda) = dev else {
        unreachable!("a CUDA test");
    };
    let len = n_slices * SLICE_BYTES;
    let stream = cuda.cuda_stream();
    // SAFETY: `ptr` is a live slot buffer holding `n_slices` 16-byte slice
    // headers.
    let view: CudaSlice<u8> = unsafe { stream.upgrade_device_ptr::<u8>(ptr, len) };
    let host = cuda.memcpy_dtov(&view).unwrap();
    // A borrow of the buffer, not an owner: dropping it would free the slot.
    std::mem::forget(view);
    slice_lens(&host)
}

/// The `len` field (bytes 2..4) of each 16-byte slice header in `bytes`.
fn slice_lens(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(SLICE_BYTES)
        .map(|s| u16::from_le_bytes([s[2], s[3]]))
        .collect()
}

/// A gate across a stream: a host function the stream runs when it reaches it,
/// which blocks until the test opens the gate — so everything enqueued after it
/// waits, deterministically and with no GPU work. It opens by itself after
/// [`StreamGate::TIMEOUT`], so a host that waits on work queued behind a closed
/// gate is held that long and then fails the test, rather than hanging it.
struct StreamGate {
    open: Arc<(Mutex<bool>, Condvar)>,
}

impl StreamGate {
    const TIMEOUT: Duration = Duration::from_secs(2);

    fn close_on(stream: &CudaStream) -> Self {
        let open = Arc::new((Mutex::new(false), Condvar::new()));
        let arg = Arc::into_raw(Arc::clone(&open)) as *mut c_void;
        // SAFETY: `wait_at_gate` takes back, exactly once, the reference leaked
        // here, when the stream runs it.
        unsafe { launch_host_function(stream.cu_stream(), wait_at_gate, arg) }.unwrap();
        Self { open }
    }

    fn open(&self) {
        let (opened, wake) = &*self.open;
        *opened.lock().unwrap() = true;
        wake.notify_all();
    }

    /// Open the gate from another thread after `delay` — for a test whose own
    /// thread is about to wait on work behind it.
    fn open_after(&self, delay: Duration) -> JoinHandle<()> {
        let gate = Self {
            open: Arc::clone(&self.open),
        };
        thread::spawn(move || {
            thread::sleep(delay);
            gate.open();
        })
    }
}

impl Drop for StreamGate {
    fn drop(&mut self) {
        self.open();
    }
}

/// The host function behind a [`StreamGate`]. It must not call CUDA, and does
/// not: it only waits.
unsafe extern "C" fn wait_at_gate(arg: *mut c_void) {
    // SAFETY: `arg` is the `Arc` leaked by `StreamGate::close_on`, taken back
    // once.
    let open = unsafe { Arc::from_raw(arg as *const (Mutex<bool>, Condvar)) };
    let (opened, wake) = &*open;
    let guard = opened
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _ = wake.wait_timeout_while(guard, StreamGate::TIMEOUT, |opened| !*opened);
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

/// The harness first: work enqueued behind a closed [`StreamGate`] does not
/// complete until the gate opens. Every test below that reads "the host did not
/// wait" off an event behind the gate depends on it.
#[test]
fn a_closed_gate_holds_the_work_behind_it() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let Device::Cuda(cuda) = &dev else {
        unreachable!("a CUDA test");
    };
    let stream = cuda.cuda_stream();
    dev.synchronize().unwrap();

    let gate = StreamGate::close_on(&stream);
    let behind = stream.context().new_event(None).unwrap();
    behind.record(&stream).unwrap();
    thread::sleep(Duration::from_millis(20));
    let held = !behind.is_complete();
    gate.open();
    dev.synchronize().unwrap();
    assert!(
        held,
        "work enqueued behind a closed gate completed before it opened"
    );
    assert!(behind.is_complete());
}

/// **A queued upload keeps its bytes, and the next commit does not wait for
/// it.**
///
/// A prefill commits each layer's tokens while that layer's slot-state upload
/// is still queued behind the attention kernel — and that kernel must read the
/// pre-commit lengths. So a commit may neither rewrite the bytes the queued
/// upload will carry nor hold the host until it has run: the first hands the
/// kernel lengths from its future, the second stops the host running ahead of
/// the GPU at every layer of every prefill.
///
/// A gate is closed across the stream; a first commit's upload queues behind
/// it; a device copy of the slot queues behind the upload — the reader that must
/// see the first commit's lengths — and a second commit follows at once.
#[test]
fn a_queued_upload_keeps_its_bytes_and_the_next_commit_does_not_wait() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let Device::Cuda(cuda) = &dev else {
        unreachable!("a CUDA test");
    };
    let stream = cuda.cuda_stream();
    let (backing, mut cache, seq) = setup(&dev);

    write_outside_decode(&mut cache, &dev, 0, 8);
    let (ptr, n_slices) = live_slot(&backing, seq, 8);
    assert_eq!(read_lens(&dev, ptr, n_slices), vec![8]);
    let bytes = n_slices * SLICE_BYTES;
    // SAFETY: fully overwritten by the copy below before anything reads it.
    let probe: CudaSlice<u8> = unsafe { stream.alloc::<u8>(bytes) }.unwrap();
    // The KV both commits cover is written first: only the commits run behind
    // the gate, so an allocation or a free the write makes cannot stand in for
    // the commit under test.
    let kv = write_kv_ahead(&mut cache, &dev, 8, 8);
    dev.synchronize().unwrap();

    let gate = StreamGate::close_on(&stream);
    // The first commit: its upload queues behind the gate.
    let t0 = Instant::now();
    cache.commit_written_tokens(8, 4).unwrap();
    let first_commit = t0.elapsed();
    {
        let (probe_ptr, _record) = probe.device_ptr(&stream);
        // SAFETY: both ranges are live device allocations of `bytes` bytes,
        // and the copy is ordered on the stream every slot-state upload uses.
        unsafe { memcpy_dtod_async(probe_ptr, ptr, bytes, stream.cu_stream()) }.unwrap();
    }
    // The second commit, at once.
    let t0 = Instant::now();
    cache.commit_written_tokens(12, 4).unwrap();
    let second_commit = t0.elapsed();

    gate.open();
    dev.synchronize().unwrap();
    drop(kv);
    assert_eq!(
        slice_lens(&cuda.memcpy_dtov(&probe).unwrap()),
        vec![12],
        "the upload queued by the first commit carried the second commit's bytes"
    );
    assert_eq!(read_lens(&dev, ptr, n_slices), vec![16]);
    // Work behind the closed gate cannot run until the gate opens or times
    // out, so a commit that waits for it takes the gate's whole timeout; one
    // that does not returns in microseconds.
    assert!(
        first_commit < StreamGate::TIMEOUT / 4 && second_commit < StreamGate::TIMEOUT / 4,
        "a commit held the host (first {first_commit:?}, second {second_commit:?}) — it \
         waited for work the closed gate holds for up to {:?}",
        StreamGate::TIMEOUT
    );
}

/// Two uploads queued behind unfinished work, and a third commit behind them:
/// the third finds both staging buffers still read by queued copies and waits
/// for the older one — and every reader still sees exactly the bytes its upload
/// was issued with.
#[test]
fn every_queued_upload_keeps_its_bytes_when_both_staging_buffers_are_in_flight() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let Device::Cuda(cuda) = &dev else {
        unreachable!("a CUDA test");
    };
    let stream = cuda.cuda_stream();
    let (backing, mut cache, seq) = setup(&dev);

    write_outside_decode(&mut cache, &dev, 0, 8);
    let (ptr, n_slices) = live_slot(&backing, seq, 8);
    let bytes = n_slices * SLICE_BYTES;
    // SAFETY: each is fully overwritten by its copy below before anything reads
    // it.
    let probes: Vec<CudaSlice<u8>> = (0..2)
        .map(|_| unsafe { stream.alloc::<u8>(bytes) }.unwrap())
        .collect();
    // Only the commits run behind the gate (see the test above).
    let kv = write_kv_ahead(&mut cache, &dev, 8, 12);
    dev.synchronize().unwrap();

    let gate = StreamGate::close_on(&stream);
    for (probe, offset) in probes.iter().zip([8, 12]) {
        cache.commit_written_tokens(offset, 4).unwrap();
        let (probe_ptr, _record) = probe.device_ptr(&stream);
        // SAFETY: both ranges are live device allocations of `bytes` bytes,
        // and the copy is ordered on the stream every slot-state upload uses.
        unsafe { memcpy_dtod_async(probe_ptr, ptr, bytes, stream.cu_stream()) }.unwrap();
    }
    // The third commit waits for the first upload, which is behind the gate:
    // open it from another thread once that wait has begun.
    let opener = gate.open_after(Duration::from_millis(200));
    cache.commit_written_tokens(16, 4).unwrap();
    opener.join().unwrap();

    dev.synchronize().unwrap();
    drop(kv);
    let seen: Vec<Vec<u16>> = probes
        .iter()
        .map(|p| slice_lens(&cuda.memcpy_dtov(p).unwrap()))
        .collect();
    assert_eq!(seen, vec![vec![12], vec![16]]);
    assert_eq!(read_lens(&dev, ptr, n_slices), vec![20]);
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
