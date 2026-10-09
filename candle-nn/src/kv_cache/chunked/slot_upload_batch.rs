//! One launch for every slot-state upload a decode metadata build makes on one
//! layer.
//!
//! Bringing a sequence's decode slot up to date — an extend across a 32-token
//! boundary, a resync, a rebuild — ends in an upload of the bytes it
//! re-serialised. Issued one by one, each costs a pinned-staging pack, one
//! `cuMemcpyHtoDAsync` per contiguous range (an extend has two: its headers and
//! its records) and an event record, about 10 µs of host time a slot. At 60
//! sequences × 24 layers every sequence crosses a boundary each 32 tokens, and
//! those driver calls were the GPU's idle gap before every such step.
//!
//! While a [`SlotUploadBatch`] is open on a thread, [`defer`] collects the
//! ranges instead, and [`SlotUploadBatch::flush`] packs every one into the
//! stager generation's device-mapped arena and copies them all with ONE
//! `rows_scatter` launch on the compute stream (hot-path invariant 2b: a
//! descriptor table, not a copy per destination). A thread recording a wave
//! defers nothing: its uploads are recorded into the wave's ring (see [`defer`]).
//!
//! # Why deferring is safe
//!
//! The batch is opened and flushed under the one backing state write lock that
//! owns every slot written into it, so no other thread can release, reclaim or
//! write one of those slots in between. A slot released on this thread while
//! its copies are still pending drops them ([`forget`]): the slot is no longer
//! this sequence's, and its next tenant's bytes must not be overwritten. Once
//! flushed, the scatter is ordered on the compute stream like every other copy
//! into these slots (`slot_state_arena`, "Why releasing needs no fence").
//!
//! A slot serialised twice in one batch is captured once: the earlier ranges
//! are merged into the later capture, read from the host copy that is
//! authoritative for every serialised byte, so no two runs of the scatter ever
//! write the same bytes.

use std::cell::RefCell;
use std::ops::Range;

use candle::cuda_backend::kernels::simple::rows_scatter::{
    run_rows_scatter, ROWS_SCATTER_INLINE_MAX, ROWS_SCATTER_WORDS,
};
use candle::quantized::pinned_staging::{Generation, GpuBuf};
use candle::{CudaDevice, Result};

/// Alignment of each run's bytes in the packed buffer, so a run whose length
/// and destination are 16-byte aligned takes the scatter's `uint4` path.
const PACK_ALIGN: usize = 16;

/// One deferred copy: `len` bytes from the packed buffer at `src` to `dst`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    /// The slot this run writes into — its base address — so a later capture or
    /// a release of the same slot finds every run it supersedes.
    slot: u64,
    dst: u64,
    src: usize,
    len: usize,
}

/// The copies collected on this thread since the batch opened.
#[derive(Default)]
struct Pending {
    bytes: Vec<u8>,
    runs: Vec<Run>,
    device: Option<CudaDevice>,
}

impl Pending {
    /// Capture `ranges` (offsets into `host`, which is the slot's whole host
    /// copy) for the slot at `slot`, merged with any ranges still pending for
    /// it. Their bytes are taken from `host` now.
    fn capture(&mut self, slot: u64, ranges: &[Range<usize>], host: &[u8]) {
        let mut merged: Vec<Range<usize>> = ranges.to_vec();
        self.runs.retain(|r| {
            if r.slot == slot {
                let off = (r.dst - slot) as usize;
                merged.push(off..off + r.len);
                false
            } else {
                true
            }
        });
        for range in coalesce(merged) {
            let src = self.bytes.len().next_multiple_of(PACK_ALIGN);
            self.bytes.resize(src, 0);
            self.bytes.extend_from_slice(&host[range.clone()]);
            self.runs.push(Run {
                slot,
                dst: slot + range.start as u64,
                src,
                len: range.len(),
            });
        }
    }

    /// Drop every pending run into the slot at `slot`.
    fn forget(&mut self, slot: u64) {
        self.runs.retain(|r| r.slot != slot);
    }
}

/// Sort `ranges` and merge every pair that overlaps or touches.
fn coalesce(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.retain(|r| !r.is_empty());
    ranges.sort_by_key(|r| r.start);
    let mut out: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for r in ranges {
        match out.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => out.push(r),
        }
    }
    out
}

thread_local! {
    static PENDING: RefCell<Option<Pending>> = const { RefCell::new(None) };
}

/// Defer the upload of `ranges` of a slot's host copy `host` into the device
/// slot at `slot`, when a batch is open on this thread. `false` — nothing
/// taken — when none is, for the caller to upload eagerly.
///
/// Also `false` on a thread recording a wave: there the caller records each
/// range into the wave's ring, a copy-engine node of the segment, where the
/// scatter would be a kernel reading the staged bytes over PCIe on the GPU's
/// critical path — measured as Flash-Next's verify losing 4–7% of decode,
/// once per attention layer per step.
pub(crate) fn defer(device: &CudaDevice, slot: u64, ranges: &[Range<usize>], host: &[u8]) -> bool {
    PENDING.with(|p| match p.borrow_mut().as_mut() {
        Some(_) if device.recording_segment().is_some() => false,
        Some(pending) => {
            pending.device.get_or_insert_with(|| device.clone());
            pending.capture(slot, ranges, host);
            true
        }
        None => false,
    })
}

/// Drop the copies pending on this thread into the slot at `slot`, which is
/// being released.
pub(crate) fn forget(slot: u64) {
    PENDING.with(|p| {
        if let Some(pending) = p.borrow_mut().as_mut() {
            pending.forget(slot);
        }
    });
}

/// An open batch on this thread. Flushed by [`Self::flush`], or on drop when a
/// caller leaves early on an error — the slots its copies describe are already
/// serialised on the host, so their bytes must reach the device either way.
pub(crate) struct SlotUploadBatch<'g> {
    generation: &'g Generation,
    open: bool,
}

impl<'g> SlotUploadBatch<'g> {
    /// Open a batch on this thread, staging through `generation`. Batches do not
    /// nest: one already open here is an error.
    pub(crate) fn open(generation: &'g Generation) -> Result<Self> {
        PENDING.with(|p| {
            let mut p = p.borrow_mut();
            if p.is_some() {
                candle::bail!("slot upload batch: one is already open on this thread");
            }
            *p = Some(Pending::default());
            Ok(())
        })?;
        Ok(Self {
            generation,
            open: true,
        })
    }

    /// Copy every deferred range in one launch and close the batch.
    pub(crate) fn flush(mut self) -> Result<()> {
        self.open = false;
        flush_pending(self.generation)
    }
}

impl Drop for SlotUploadBatch<'_> {
    fn drop(&mut self) {
        if self.open {
            if let Err(e) = flush_pending(self.generation) {
                log::error!("slot upload batch: flushing on an early exit failed: {e}");
            }
        }
    }
}

/// Take this thread's pending copies and launch them.
fn flush_pending(generation: &Generation) -> Result<()> {
    let Some(pending) = PENDING.with(|p| p.borrow_mut().take()) else {
        return Ok(());
    };
    let (Some(device), false) = (pending.device.as_ref(), pending.runs.is_empty()) else {
        return Ok(());
    };
    let bytes = stage(generation, &pending.bytes)?;
    let desc = descriptors(&pending.runs, bytes.dev_ptr())?;
    // Few runs ride in the kernel's parameters and the device table is never
    // read; more are read from the generation's arena in place.
    let table = if pending.runs.len() > ROWS_SCATTER_INLINE_MAX {
        Some(stage(generation, as_bytes(&desc))?)
    } else {
        None
    };
    let max_words = pending.runs.iter().map(|r| r.len / 4).max().unwrap_or(0);
    let stream = device.cuda_stream();
    // SAFETY: every descriptor names a source inside the staged bytes, which
    // the generation keeps alive past the launch, and a destination inside a
    // slot this thread's backing owns (see the module docs); `desc` and the
    // staged table hold the same words.
    let status = unsafe {
        run_rows_scatter(
            table
                .as_ref()
                .map_or(std::ptr::null(), |t| t.dev_ptr() as *const i64),
            desc.as_ptr(),
            pending.runs.len() as i32,
            max_words as i32,
            1,
            stream.cu_stream() as *mut std::ffi::c_void,
        )
    };
    if status != 0 {
        candle::bail!("slot upload batch: the scatter launch failed with CUDA error {status}");
    }
    Ok(())
}

/// The scatter's descriptor table for `runs`, their sources addressed from
/// `base`: one single-row run each, its width in 32-bit words.
fn descriptors(runs: &[Run], base: u64) -> Result<Vec<i64>> {
    let mut desc = Vec::with_capacity(runs.len() * ROWS_SCATTER_WORDS);
    for r in runs {
        if r.len % 4 != 0 || r.dst % 4 != 0 {
            candle::bail!(
                "slot upload batch: a {}-byte range at {:#x} is not 32-bit aligned — the \
                 scatter copies words",
                r.len,
                r.dst
            );
        }
        let words = (r.len / 4) as i64;
        desc.extend_from_slice(&[
            (base + r.src as u64) as i64,
            words,
            r.dst as i64,
            words,
            1,
            words,
        ]);
    }
    Ok(desc)
}

/// The bytes of `words`, in memory order.
fn as_bytes(words: &[i64]) -> &[u8] {
    // SAFETY: an `i64` slice is plain initialised memory of `8 × len` bytes.
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) }
}

/// `bytes` in a device-visible staging buffer of `generation`.
fn stage(generation: &Generation, bytes: &[u8]) -> Result<GpuBuf> {
    let mut pinned = generation.alloc(bytes.len())?;
    pinned.as_mut_slice().copy_from_slice(bytes);
    let gpu = generation.submit(pinned)?;
    if gpu.dev_ptr() == 0 {
        candle::bail!("slot upload batch: the staging generation has no device");
    }
    Ok(gpu)
}

// A one-element `&[a..b]` here is a slice holding one range — `capture` and
// `defer` take a slice of ranges — not a mistaken range-of-elements array.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;
    use candle::backend::BackendDevice;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::pinned_staging::PinnedStager;

    fn host(len: usize) -> Vec<u8> {
        (0..len).map(|i| i as u8).collect()
    }

    #[test]
    fn touching_and_overlapping_ranges_coalesce() {
        assert_eq!(
            coalesce(vec![32..48, 0..16, 16..20, 40..64, 100..100, 80..96]),
            vec![0..20, 32..64, 80..96]
        );
    }

    /// Each range lands at its offset in the slot, its bytes packed at a
    /// 16-byte boundary in the order captured.
    #[test]
    fn a_capture_packs_each_range_aligned() {
        let h = host(256);
        let mut p = Pending::default();
        p.capture(0x1000, &[0..20, 64..72], &h);
        assert_eq!(
            p.runs,
            vec![
                Run {
                    slot: 0x1000,
                    dst: 0x1000,
                    src: 0,
                    len: 20
                },
                Run {
                    slot: 0x1000,
                    dst: 0x1040,
                    src: 32,
                    len: 8
                },
            ]
        );
        assert_eq!(&p.bytes[0..20], &h[0..20]);
        assert_eq!(&p.bytes[20..32], &[0u8; 12]);
        assert_eq!(&p.bytes[32..40], &h[64..72]);
    }

    /// A second capture of one slot supersedes its earlier runs: the union of
    /// both range sets, read from the host copy as it stands at the second.
    #[test]
    fn a_second_capture_of_a_slot_merges_and_rereads() {
        let mut h = host(256);
        let mut p = Pending::default();
        p.capture(0x1000, &[0..16, 128..144], &h);
        p.capture(0x9000, &[0..8], &h);
        h[130] = 0xEE;
        h[20] = 0xDD;
        p.capture(0x1000, &[16..32], &h);
        let runs: Vec<(u64, usize)> = p
            .runs
            .iter()
            .filter(|r| r.slot == 0x1000)
            .map(|r| (r.dst, r.len))
            .collect();
        assert_eq!(runs, vec![(0x1000, 32), (0x1080, 16)]);
        let first = p.runs.iter().find(|r| r.dst == 0x1000).unwrap();
        assert_eq!(p.bytes[first.src + 20], 0xDD);
        let rec = p.runs.iter().find(|r| r.dst == 0x1080).unwrap();
        assert_eq!(p.bytes[rec.src + 2], 0xEE);
        assert_eq!(p.runs.iter().filter(|r| r.slot == 0x9000).count(), 1);
    }

    #[test]
    fn a_released_slot_drops_only_its_own_runs() {
        let h = host(64);
        let mut p = Pending::default();
        p.capture(0x1000, &[0..16], &h);
        p.capture(0x2000, &[0..16, 32..48], &h);
        p.forget(0x2000);
        assert_eq!(p.runs.len(), 1);
        assert_eq!(p.runs[0].slot, 0x1000);
    }

    /// One single-row run per range, its width in words, sources from `base`.
    #[test]
    fn descriptors_are_single_row_word_runs() {
        let runs = [
            Run {
                slot: 0x1000,
                dst: 0x1000,
                src: 0,
                len: 32,
            },
            Run {
                slot: 0x1000,
                dst: 0x1100,
                src: 32,
                len: 12,
            },
        ];
        assert_eq!(
            descriptors(&runs, 0x5000).unwrap(),
            vec![0x5000, 8, 0x1000, 8, 1, 8, 0x5020, 3, 0x1100, 3, 1, 3]
        );
    }

    #[test]
    fn an_unaligned_range_is_refused() {
        let runs = [Run {
            slot: 0x1000,
            dst: 0x1000,
            src: 0,
            len: 6,
        }];
        assert!(descriptors(&runs, 0).is_err());
    }

    /// A batch is open on its thread from `open` to `flush`, and only one at a
    /// time; with none open, `forget` touches nothing. An empty batch flushes
    /// without staging anything — the host-only stager here has no device.
    #[test]
    fn one_batch_is_open_from_open_to_flush() {
        let stager = PinnedStager::noop();
        let generation = stager.begin_generation();
        forget(0x1000);
        PENDING.with(|p| assert!(p.borrow().is_none()));
        let batch = SlotUploadBatch::open(&generation).unwrap();
        PENDING.with(|p| assert!(p.borrow().is_some()));
        assert!(
            SlotUploadBatch::open(&generation).is_err(),
            "batches do not nest"
        );
        batch.flush().unwrap();
        PENDING.with(|p| assert!(p.borrow().is_none()));
    }

    /// The bytes of a flushed batch land at their destinations: 40 bytes into the
    /// start of one buffer and 8 bytes into the middle of another, the rest of
    /// both left as they were. On a thread recording a wave nothing is deferred —
    /// the caller records its ranges into the wave's ring — and a batch opened
    /// there flushes nothing, with the capture intact.
    #[test]
    fn a_flushed_batch_lands_every_range() {
        let _gpu = crate::kv_cache::chunked::gpu_test_lock::gpu_serial();
        let dev = CudaDevice::new(0).unwrap();
        let stager = PinnedStager::new(&dev);
        for recording in [false, true] {
            let a = dev.memcpy_stod(&[0xAAu8; 64]).unwrap();
            let b = dev.memcpy_stod(&[0xBBu8; 64]).unwrap();
            let stream = dev.cuda_stream();
            let pa = a.device_ptr(&stream).0;
            let pb = b.device_ptr(&stream).0;
            let h = host(64);
            let capture = recording.then(|| {
                let c = dev.begin_wave_capture().unwrap();
                dev.record_launches().unwrap();
                c
            });
            let generation = stager.begin_generation();
            let batch = SlotUploadBatch::open(&generation).unwrap();
            assert_eq!(dev.recording_segment().is_some(), recording);
            assert_eq!(defer(&dev, pa, &[0..40], &h), !recording);
            assert_eq!(defer(&dev, pb, &[16..24], &h), !recording);
            batch.flush().unwrap();
            drop(generation);
            if let Some(c) = capture {
                c.finish().unwrap();
            }
            dev.synchronize().unwrap();
            let got_a = dev.memcpy_dtov(&a).unwrap();
            let got_b = dev.memcpy_dtov(&b).unwrap();
            let mut want_a = vec![0xAAu8; 64];
            let mut want_b = vec![0xBBu8; 64];
            if !recording {
                want_a[..40].copy_from_slice(&h[..40]);
                want_b[16..24].copy_from_slice(&h[16..24]);
            }
            assert_eq!(got_a, want_a, "recording {recording}");
            assert_eq!(got_b, want_b, "recording {recording}");
        }
    }

    /// A batch dropped without a flush still closes, so the next can open.
    #[test]
    fn a_dropped_batch_closes() {
        let stager = PinnedStager::noop();
        let generation = stager.begin_generation();
        drop(SlotUploadBatch::open(&generation).unwrap());
        PENDING.with(|p| assert!(p.borrow().is_none()));
        SlotUploadBatch::open(&generation).unwrap().flush().unwrap();
    }
}
