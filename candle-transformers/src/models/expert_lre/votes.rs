//! The router look-ahead votes, one ring slot per invocation.
//!
//! A decode launch's stacked router projection also applies the next rows'
//! routers to its own input (`Qwen35MoeBlock::forward_parts`), and
//! `moe_predict_votes` turns each of those into how many of the launch's tokens
//! put an expert in their top `k`: the experts the next rows are likely to
//! route. The counts go to this ring's slot for the invocation — the summary
//! ring's slot, so the same hold protects both (`Dispatch::hold_for_ring`) —
//! in mapped pinned memory, written before the invocation's bucketize stores
//! its summary word. The pipeline thread copies them out beside the summary
//! and predicts from them (`PipelineState::predict_rows`).
//!
//! Layout: `u32 [SUMMARY_RING][HOPS][n_experts]`.

use super::dispatch::SUMMARY_RING;
use super::transition::HOPS;
use candle::cuda_backend::cudarc::driver::sys;
use candle::Result;

pub(crate) struct VoteRing {
    host: *mut u32,
    dev: u64,
    n_experts: usize,
}

// SAFETY: written by the device only, read by the pipeline thread only after
// the slot's summary word, and rewritten only once that thread has served it.
unsafe impl Send for VoteRing {}
unsafe impl Sync for VoteRing {}

impl VoteRing {
    pub(crate) fn new(n_experts: usize) -> Result<Self> {
        let bytes = SUMMARY_RING * HOPS * n_experts * 4;
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("expert dispatch: mapped vote ring allocation failed: {r:?}");
        }
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe { sys::cuMemFreeHost(raw) };
            candle::bail!("expert dispatch: vote ring has no device address: {r:?}");
        }
        // SAFETY: `bytes` just allocated, not yet visible to the device.
        unsafe { std::ptr::write_bytes(raw as *mut u8, 0, bytes) };
        Ok(Self {
            host: raw as *mut u32,
            dev,
            n_experts,
        })
    }

    fn slot_words(&self) -> usize {
        HOPS * self.n_experts
    }

    /// Slot `slot`'s device address — the votes kernel's output.
    pub(crate) fn dev_slot(&self, slot: usize) -> u64 {
        assert!(
            slot < SUMMARY_RING,
            "vote ring slot {slot} of {SUMMARY_RING}"
        );
        self.dev + (slot * self.slot_words() * 4) as u64
    }

    /// Slot `slot`'s first `hops` hops of votes, `[hops][n_experts]`, copied.
    ///
    /// # Safety
    ///
    /// The slot's invocation's summary word must have been seen, and the slot
    /// not yet released to the forward thread.
    pub(crate) unsafe fn read(&self, slot: usize, hops: usize) -> Vec<u32> {
        assert!(slot < SUMMARY_RING && hops <= HOPS);
        let n = hops * self.n_experts;
        let src = self.host.add(slot * self.slot_words());
        (0..n)
            .map(|i| std::ptr::read_volatile(src.add(i)))
            .collect()
    }
}

impl Drop for VoteRing {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc` in `new`; the device and the
        // pipeline thread are done with it before the cache drops.
        unsafe {
            sys::cuMemFreeHost(self.host as *mut std::ffi::c_void);
        }
    }
}

/// Hop `hop`'s experts by votes, most first, ties by expert, at most `cap` —
/// `votes` is one slot's `[hops][n_experts]` copy.
pub(crate) fn ranked(votes: &[u32], n_experts: usize, hop: usize, cap: usize) -> Vec<usize> {
    let row = &votes[(hop - 1) * n_experts..hop * n_experts];
    let mut v: Vec<(u32, usize)> = row
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(e, &c)| (c, e))
        .collect();
    v.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    v.truncate(cap);
    v.into_iter().map(|(_, e)| e).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hop's experts, most votes first, ties by expert, zero votes left out,
    /// capped.
    #[test]
    fn a_hops_experts_rank_by_votes() {
        // Two hops of four experts.
        let votes = [0, 3, 1, 3, /* hop 2 */ 2, 0, 0, 5];
        assert_eq!(ranked(&votes, 4, 1, 8), vec![1, 3, 2]);
        assert_eq!(ranked(&votes, 4, 1, 2), vec![1, 3]);
        assert_eq!(ranked(&votes, 4, 2, 8), vec![3, 0]);
    }
}
