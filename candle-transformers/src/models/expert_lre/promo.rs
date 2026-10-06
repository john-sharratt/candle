//! The promotion ring — free VRAM slots the GPU fills with the experts it
//! misses.
//!
//! A routed expert that is not in VRAM is copied over the link by its layer's
//! GEMM workers anyway (`dispatch`), slice by slice, into VRAM scratch. If the
//! same slices also land in a free weight-zone slot, the expert is in VRAM once
//! the layer is done — a promotion that costs a VRAM write instead of a second
//! crossing of the link. This ring is how the slots get there:
//!
//! - the pipeline thread takes free slots (under the reclaim rule — nothing can
//!   be reading them) and pushes their addresses at `tail`;
//! - `moe_bucketize` gives each remote expert the next slot from `head`, and
//!   logs `summary_word << 32 | row << 16 | expert` at its index;
//! - the pipeline thread collects each logged slot, and once the invocation
//!   that took it has completed (its ticket is below the observed one) points
//!   the expert's entry at it.
//!
//! In mapped pinned memory: `u64 slots[cap] | u64 log[cap] | u32 head | u32
//! tail | u32 marks[rows][n_experts] | u32 reserve`. `tail` and `reserve` are
//! written only by the host, `head` only by the device. `reserve` is the stock
//! kept for decode-scored experts: a prompt-only expert takes a slot only while
//! more than it is stocked (`moe_bucketize.cu`, PROMOTION). An expert's mark is set by the device when it
//! gives the expert a slot and cleared by the host when the promotion lands (or
//! is dropped): until then a later invocation that still finds the expert
//! remote — a prefill visiting the row again before the host has caught up —
//! does not spend a second slot on it.

use candle::quantized::cuda::PromoRing;
use candle::Result;
use cudarc::driver::sys;
use std::sync::atomic::{fence, Ordering};

pub(crate) struct PromotionRing {
    host: *mut u8,
    dev: u64,
    cap: usize,
    n_experts: usize,
    /// Byte offset of the `reserve` word, after the marks.
    reserve_at: usize,
}

// SAFETY: `tail` and the slots are written only by the pipeline thread (the
// one owner of the host side), `head` and the log only by the device; every
// access is volatile and fenced.
unsafe impl Send for PromotionRing {}
unsafe impl Sync for PromotionRing {}

/// One logged promotion: which invocation's word, which row and expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Logged {
    pub(crate) word: u32,
    pub(crate) row: usize,
    pub(crate) expert: usize,
}

impl Logged {
    fn decode(v: u64) -> Self {
        Self {
            word: (v >> 32) as u32,
            row: ((v >> 16) & 0xffff) as usize,
            expert: (v & 0xffff) as usize,
        }
    }
}

/// The ticket whose summary word is `word`, nearest `near`. Words are the
/// tickets' low 32 bits, and a logged slot is never more than a ring's worth of
/// invocations from the one being served.
pub(crate) fn ticket_from_word(word: u32, near: u64) -> u64 {
    let base = near & !0xffff_ffff;
    let candidates = [base.wrapping_sub(1 << 32), base, base + (1 << 32)];
    candidates
        .into_iter()
        .map(|b| b | word as u64)
        .min_by_key(|&t| t.abs_diff(near))
        .expect("three candidates")
}

impl PromotionRing {
    pub(crate) fn new(cap: usize, rows: usize, n_experts: usize) -> Result<Self> {
        let reserve_at = cap * 16 + 8 + rows * n_experts * 4;
        let bytes = reserve_at + 4;
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("expert promotion ring: mapped allocation failed: {r:?}");
        }
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe {
                sys::cuMemFreeHost(raw);
            }
            candle::bail!("expert promotion ring has no device address: {r:?}");
        }
        // SAFETY: `bytes` just allocated, not yet visible to the device.
        unsafe { std::ptr::write_bytes(raw as *mut u8, 0, bytes) };
        let ring = Self {
            host: raw as *mut u8,
            dev,
            cap,
            n_experts,
            reserve_at,
        };
        // No prompt-only expert takes a slot until the host says how much is
        // decode's.
        ring.set_reserve(u32::MAX);
        Ok(ring)
    }

    /// Keep `n` stocked slots for decode-scored experts: a prompt-only expert
    /// takes one only while more than `n` are stocked.
    pub(crate) fn set_reserve(&self, n: u32) {
        // SAFETY: the reserve word this struct owns, after the marks.
        unsafe { std::ptr::write_volatile(self.host.add(self.reserve_at) as *mut u32, n) };
        fence(Ordering::SeqCst);
    }

    /// The expert's promotion has landed or been dropped: it may be given a
    /// slot again.
    pub(crate) fn clear_mark(&self, row: usize, expert: usize) {
        // SAFETY: inside the marks array, `rows × n_experts` u32s after the
        // counters; the device writes a mark only before handing out a slot.
        unsafe {
            let marks = self.host.add(self.cap * 16 + 8) as *mut u32;
            std::ptr::write_volatile(marks.add(row * self.n_experts + expert), 0);
        }
    }

    /// The device addresses `moe_bucketize` takes.
    pub(crate) fn ring(&self) -> PromoRing {
        PromoRing {
            slots: self.dev,
            log: self.dev + (self.cap * 8) as u64,
            head: self.dev + (self.cap * 16) as u64,
            tail: self.dev + (self.cap * 16 + 4) as u64,
            cap: self.cap as u32,
            marks: self.dev + (self.cap * 16 + 8) as u64,
            reserve: self.dev + self.reserve_at as u64,
        }
    }

    pub(crate) fn cap(&self) -> usize {
        self.cap
    }

    fn word(&self, at: usize) -> *mut u32 {
        // SAFETY: `at` is one of the two counter offsets inside the allocation.
        unsafe { self.host.add(at) as *mut u32 }
    }

    /// Slots the device has taken, ever (wrapping).
    pub(crate) fn head(&self) -> u32 {
        // SAFETY: the counter word this struct owns.
        let h = unsafe { std::ptr::read_volatile(self.word(self.cap * 16)) };
        // The log entries the device wrote before it.
        fence(Ordering::Acquire);
        h
    }

    /// Slots the host has published, ever (wrapping).
    pub(crate) fn tail(&self) -> u32 {
        // SAFETY: as `head`.
        unsafe { std::ptr::read_volatile(self.word(self.cap * 16 + 4)) }
    }

    /// Publish `slot_base` at `tail` and advance it. The caller must have room
    /// (`tail - taken < cap`, where `taken` is what it has collected).
    pub(crate) fn push(&self, slot_base: u64) {
        let tail = self.tail();
        let i = tail as usize % self.cap;
        // SAFETY: `i < cap`, inside the slots array.
        unsafe {
            std::ptr::write_volatile((self.host as *mut u64).add(i), slot_base);
        }
        fence(Ordering::SeqCst);
        // SAFETY: the counter word this struct owns.
        unsafe { std::ptr::write_volatile(self.word(self.cap * 16 + 4), tail.wrapping_add(1)) };
        fence(Ordering::SeqCst);
    }

    /// The log entry at ring index `i` (mod `cap`).
    pub(crate) fn log(&self, i: u32) -> Logged {
        let i = i as usize % self.cap;
        // SAFETY: inside the log array.
        let v = unsafe { std::ptr::read_volatile((self.host.add(self.cap * 8) as *const u64).add(i)) };
        Logged::decode(v)
    }

    /// Withdraw every published, untaken slot: `tail = head`. Only with no
    /// bucketize in flight and none able to begin.
    pub(crate) fn withdraw(&self) {
        self.withdraw_to(self.head());
    }

    /// Withdraw the published, untaken slots past `tail`: set `tail` back to
    /// it (`head <= tail <= self.tail()`). Only with no bucketize in flight and
    /// none able to begin — a bucketize reads `tail` once and takes up to it.
    pub(crate) fn withdraw_to(&self, tail: u32) {
        // SAFETY: the counter word this struct owns.
        unsafe { std::ptr::write_volatile(self.word(self.cap * 16 + 4), tail) };
        fence(Ordering::SeqCst);
    }
}

impl Drop for PromotionRing {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc` in `new`; the device and the
        // pipeline thread are done with it before the cache drops.
        unsafe {
            sys::cuMemFreeHost(self.host as *mut std::ffi::c_void);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A log entry carries the word, the row and the expert.
    #[test]
    fn a_log_entry_decodes() {
        let v = (0xdead_beefu64 << 32) | (39 << 16) | 255;
        assert_eq!(
            Logged::decode(v),
            Logged {
                word: 0xdead_beef,
                row: 39,
                expert: 255
            }
        );
    }

    /// Published slots land at `tail` in order and wrap at `cap`; withdrawing
    /// sets `tail` back so the slots past it are no longer offered, and the
    /// next push reuses the first withdrawn index.
    #[test]
    fn pushes_publish_in_order_and_a_withdraw_sets_the_tail_back() {
        let Ok(_device) = candle::Device::new_cuda(0) else {
            return;
        };
        let ring = PromotionRing::new(4, 2, 8).unwrap();
        assert_eq!((ring.head(), ring.tail()), (0, 0));
        for base in [0x1000u64, 0x2000, 0x3000] {
            ring.push(base);
        }
        assert_eq!(ring.tail(), 3);
        // SAFETY: the slots array is the allocation's first `cap` u64s.
        let slot = |i: usize| unsafe { std::ptr::read_volatile((ring.host as *const u64).add(i)) };
        assert_eq!([slot(0), slot(1), slot(2)], [0x1000, 0x2000, 0x3000]);

        ring.withdraw_to(1);
        assert_eq!(ring.tail(), 1);
        ring.push(0x4000);
        assert_eq!((ring.tail(), slot(1)), (2, 0x4000));

        ring.withdraw();
        assert_eq!(ring.tail(), 0);
        for base in [0x5000u64, 0x6000, 0x7000, 0x8000, 0x9000] {
            ring.push(base);
        }
        assert_eq!((ring.tail(), slot(0)), (5, 0x9000));
    }

    /// A cleared mark is zero in the device's marks array, at `row × E + e`.
    #[test]
    fn clearing_a_mark_zeroes_its_word() {
        let Ok(_device) = candle::Device::new_cuda(0) else {
            return;
        };
        let ring = PromotionRing::new(4, 2, 8).unwrap();
        // SAFETY: the marks array follows the slots, the log and the counters.
        let marks = unsafe { ring.host.add(4 * 16 + 8) as *mut u32 };
        // SAFETY: row 1, expert 3 is inside the 2 × 8 marks.
        unsafe { std::ptr::write_volatile(marks.add(11), 7) };
        ring.clear_mark(1, 3);
        // SAFETY: as above.
        assert_eq!(unsafe { std::ptr::read_volatile(marks.add(11)) }, 0);
    }

    /// The ticket nearest the one being served, across a 32-bit wrap either
    /// way.
    #[test]
    fn a_word_maps_to_the_nearest_ticket() {
        assert_eq!(ticket_from_word(7, 9), 7);
        assert_eq!(ticket_from_word(12, 9), 12);
        let near = (5u64 << 32) | 3;
        assert_eq!(ticket_from_word(0xffff_fffe, near), (4u64 << 32) | 0xffff_fffe);
        let near = (5u64 << 32) | 0xffff_fff0;
        assert_eq!(ticket_from_word(4, near), (6u64 << 32) | 4);
    }
}
