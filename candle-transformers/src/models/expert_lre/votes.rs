//! The router look-ahead votes, one ring slot per invocation.
//!
//! A decode launch's stacked router projection also applies the next rows'
//! routers to its own input (`Qwen35MoeBlock::forward_parts`), and
//! `moe_predict_votes` turns each of those into one word per expert: how many of
//! the launch's tokens put it in their top `k`, in the high half, above how many
//! reached it only in ranks `k+1 … k + LOOK_AHEAD_MARGIN` — the experts the next
//! rows are likely to route, then the ones just past the router's cut — and
//! beside it the router probability those picks carried (its *mass*), which
//! tells a firm pick from one that barely made the cut. Both go to this ring's
//! slot for the invocation — the summary
//! ring's slot, so the same hold protects both (`Dispatch::hold_for_ring`) —
//! in mapped pinned memory, written before the invocation's bucketize stores
//! its summary word. The pipeline thread copies them out beside the summary
//! and predicts from them (`PipelineState::predict_rows`).
//!
//! Layout: `[SUMMARY_RING]` slots of `u32 words[HOPS][n_experts] | f32
//! mass[HOPS][n_experts]`.

use super::dispatch::SUMMARY_RING;
use super::transition::HOPS;
use candle::cuda_backend::cudarc::driver::sys;
use candle::Result;
use std::ptr;

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
        let bytes = SUMMARY_RING * 2 * HOPS * n_experts * 4;
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

    /// 4-byte entries in one hop plane: `HOPS × n_experts`.
    fn plane(&self) -> usize {
        HOPS * self.n_experts
    }

    /// Slot `slot`'s device addresses — the votes kernel's outputs: its words,
    /// then its masses.
    pub(crate) fn dev_slot(&self, slot: usize) -> (u64, u64) {
        assert!(
            slot < SUMMARY_RING,
            "vote ring slot {slot} of {SUMMARY_RING}"
        );
        let words = self.dev + (slot * 2 * self.plane() * 4) as u64;
        (words, words + (self.plane() * 4) as u64)
    }

    /// Slot `slot`'s first `hops` hops, copied.
    ///
    /// # Safety
    ///
    /// The slot's invocation's summary word must have been seen, and the slot
    /// not yet released to the forward thread.
    pub(crate) unsafe fn read(&self, slot: usize, hops: usize) -> Votes {
        assert!(slot < SUMMARY_RING && hops <= HOPS);
        let n = hops * self.n_experts;
        let words = self.host.add(slot * 2 * self.plane());
        let mass = words.add(self.plane()) as *const f32;
        Votes {
            words: (0..n).map(|i| ptr::read_volatile(words.add(i))).collect(),
            mass: (0..n).map(|i| ptr::read_volatile(mass.add(i))).collect(),
        }
    }
}

/// One invocation's votes for its rows ahead, `[hops][n_experts]` each: the
/// vote words (`routed << 16 | margin`) and the router probability their picks
/// carried.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Votes {
    pub(crate) words: Vec<u32>,
    pub(crate) mass: Vec<f32>,
}

impl Votes {
    /// Hop `hop`'s words and masses, `[n_experts]` each.
    pub(crate) fn hop(&self, hop: usize, n_experts: usize) -> (&[u32], &[f32]) {
        let r = (hop - 1) * n_experts..hop * n_experts;
        (&self.words[r.clone()], &self.mass[r])
    }

    /// Hops voted for.
    pub(crate) fn hops(&self, n_experts: usize) -> usize {
        self.words.len() / n_experts.max(1)
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

/// Hop `hop`'s experts by vote word, highest first, ties by expert, at most `cap`
/// — `votes` is one slot's `[hops][n_experts]` copy. The word's routed count is
/// its high half, so every expert some token routes ranks above every expert
/// only reached in the margin, and a cap cuts the margin first.
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

    /// A slot's copy splits into hops of `n_experts`: its words and masses
    /// side by side, and how many hops it holds.
    #[test]
    fn a_slots_votes_split_into_hops() {
        let v = Votes {
            words: vec![1, 2, 3, 4, 5, 6],
            mass: vec![0.5, 0.25, 0.125, 1.0, 2.0, 4.0],
        };
        assert_eq!(v.hops(3), 2);
        assert_eq!(v.hop(1, 3), (&[1u32, 2, 3][..], &[0.5f32, 0.25, 0.125][..]));
        assert_eq!(v.hop(2, 3), (&[4u32, 5, 6][..], &[1.0f32, 2.0, 4.0][..]));
        assert_eq!(Votes::default().hops(3), 0);
    }

    /// Routed picks outrank margin picks however many margin votes pile up, and
    /// a cap cuts the margin first: e2 reached by five tokens' margins ranks
    /// below e3 routed by one.
    #[test]
    fn a_routed_pick_outranks_any_count_of_margin_picks() {
        let votes = [(2 << 16) | 1, 0, 5, 1 << 16];
        assert_eq!(ranked(&votes, 4, 1, 8), vec![0, 3, 2]);
        assert_eq!(ranked(&votes, 4, 1, 2), vec![0, 3]);
    }
}
