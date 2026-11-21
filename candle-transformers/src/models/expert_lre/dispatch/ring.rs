//! The mapped ring of routing summaries that the device writes and the host threads
//! read.
//!
//! Bucketize writes each slot through its device address and stores the sequence
//! word last, so a reader that sees the word sees every count in the slot. A waiter
//! spins briefly and then parks on the word's cache line. It gives up with an error
//! after a long deadline, which fires only when the device is dead. Zero is never a
//! sequence word, so a fresh slot reads as unwritten.

use super::*;

/// The routing summaries, in mapped pinned memory: bucketize writes a slot
/// through its device address and the host threads read it through its host
/// address once its sequence word is in.
pub(crate) struct SummaryRing {
    pub(super) dev_base: u64,
    pub(super) host: *mut u32,
    pub(super) n_experts: usize,
}

// SAFETY: a slot is written by one bucketize and read by the host threads only
// after its sequence word; the forward thread's ring hold keeps it from being
// rewritten while it is read.
unsafe impl Send for SummaryRing {}

unsafe impl Sync for SummaryRing {}

impl SummaryRing {
    pub(crate) fn new(n_experts: usize) -> Result<Self> {
        let len = SUMMARY_RING * (n_experts + 1);
        let mut raw: *mut c_void = null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, len * 4, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("expert cache: mapped routing-summary ring allocation failed: {r:?}");
        }
        let mut dev_base: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev_base, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe {
                sys::cuMemFreeHost(raw);
            }
            candle::bail!("expert cache: routing-summary ring has no device address: {r:?}");
        }
        // SAFETY: `len` u32s just allocated. Zero is never a sequence word, so
        // no slot reads as written before its first bucketize.
        unsafe { write_bytes(raw as *mut u32, 0, len) };
        Ok(Self {
            dev_base,
            host: raw as *mut u32,
            n_experts,
        })
    }

    pub(super) fn stride(&self) -> usize {
        self.n_experts + 1
    }

    pub(super) fn dev_slot(&self, slot: usize) -> u64 {
        self.dev_base + (slot * self.stride() * 4) as u64
    }

    pub(super) fn host_slot_ptr(&self, slot: usize) -> *mut u32 {
        assert!(
            slot < SUMMARY_RING,
            "summary ring slot {slot} of {SUMMARY_RING}"
        );
        // SAFETY: in bounds by the assertion.
        unsafe { self.host.add(slot * self.stride()) }
    }

    /// `slot`'s sequence word, which bucketize stores last: the line a waiter
    /// for the slot `umwait`s on. Live as long as the ring.
    pub(crate) fn word_ptr(&self, slot: usize) -> *const u32 {
        // SAFETY: the word follows the slot's `n_experts` counts, inside it.
        unsafe { self.host_slot_ptr(slot).add(self.n_experts) }
    }

    /// Whether bucketize has written `slot` for sequence word `word`.
    ///
    /// The word is the last thing bucketize stores, behind a system fence, so
    /// once it reads `word` every count is visible — and bucketize running
    /// means every kernel enqueued before it on the compute stream has
    /// completed. That is the whole signal: no event and no copy.
    pub(crate) fn ready(&self, slot: usize, word: u32) -> bool {
        // SAFETY: the word is inside the mapped ring; the device writes it.
        let ready = unsafe { read_volatile(self.word_ptr(slot)) } == word;
        if ready {
            fence(Ordering::Acquire);
        }
        ready
    }

    /// Wait for [`Self::ready`] — hot spins, then `umwait` on the sequence
    /// word's line, which the device's store wakes (`cpu_pool::backoff`) —
    /// calling `idle` every [`WAIT_IDLE_EVERY`] polls while hot and after
    /// every low-power wait: the waiter's chance to do work that must not wait
    /// for this word. A word that never arrives means the device faulted or the
    /// forward thread's chain died; that is reported rather than waited out.
    pub(crate) fn wait(
        &self,
        slot: usize,
        word: u32,
        mut idle: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        let start = Instant::now();
        let p = self.word_ptr(slot);
        let mut backoff = Backoff::new(INTER_LAYER_SPINS);
        let mut polls = 0u32;
        while !self.ready(slot, word) {
            polls += 1;
            if backoff.cooled() || polls.is_multiple_of(WAIT_IDLE_EVERY) {
                idle()?;
                if start.elapsed() > SUMMARY_DEADLINE {
                    candle::bail!(
                        "expert pipeline: routing summary for slot {slot} never arrived — the \
                         device stopped before this layer's bucketize"
                    );
                }
            }
            // SAFETY: the mapped ring, live while `self` is; the device writes
            // the word.
            unsafe {
                let seen = read_volatile(p);
                if seen != word {
                    backoff.wait_u32(p, seen, LOW_POWER_CYCLES);
                }
            }
        }
        Ok(())
    }

    /// The summary in `slot`: `count | pinned << 29 | cold << 30 | decode << 31`
    /// per expert.
    ///
    /// # Safety
    ///
    /// [`Self::ready`] must have returned true for this invocation, and the
    /// caller must not have released the slot to the forward thread yet.
    pub(crate) unsafe fn read(&self, slot: usize) -> &[u32] {
        slice::from_raw_parts(self.host_slot_ptr(slot), self.n_experts)
    }
}

/// Polls of a routing-summary word between two calls of the waiter's idle work:
/// often enough that a finished copy is landed within microseconds, rarely
/// enough that the poll stays a load of one mapped word.
pub(super) const WAIT_IDLE_EVERY: u32 = 64;

/// How long the pipeline thread waits for a routing summary before it decides
/// the device is not going to produce one. Far above any layer's compute — a
/// worker gives a cold expert up after [`SPIN_LIMIT_NS`] — so it fires only on
/// a dead device.
pub(super) const SUMMARY_DEADLINE: Duration = Duration::from_secs(30);

/// The sequence word bucketize stores for invocation `seq`: never zero, which
/// is what a fresh slot holds.
pub(crate) fn summary_word(seq: u64) -> u32 {
    (seq as u32).wrapping_add(1).max(1)
}

impl Drop for SummaryRing {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc` in `new`; the threads that read
        // it are joined before the cache drops.
        unsafe {
            sys::cuMemFreeHost(self.host as *mut c_void);
        }
    }
}

// Every invocation the ring hold can wait on has been launched.
const _: () = assert!(2 * MAX_UNFLUSHED_INVOCATIONS <= SUMMARY_RING);
