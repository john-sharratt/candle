//! Checks the deferred reset, the generation lifetime, clone sharing, the failure
//! of oversized staging, and the bounds of the buffer recycler.
//!
//! A generation still live must block the reset, and a device-free stager must
//! reset at once rather than wait on a fence nothing will fire. A clone of an
//! owned buffer shares its bytes without a second allocation. The recycler hands
//! a buffer back only to a request of exactly its size, refuses past its cap, and
//! keeps only write-combined buffers, so a pinned page is never reused as the
//! wrong kind of memory.

use super::*;

#[cfg(test)]
pub(super) mod deferred_reset_tests {
    use super::*;

    /// Dropping the last generation must NOT leave the arena permanently dirty.
    ///
    /// The reset moved from the drop (which synchronised the stream inline) to
    /// the next `begin_generation`, gated on a recorded event. The hazard in
    /// that move is a reset that never happens — the arena would grow overflow
    /// slabs forever. On the device-free stager there is no stream to fence
    /// against, so the drop must reset immediately rather than park a fence
    /// nothing will ever fire.
    #[test]
    fn deviceless_drop_resets_immediately_rather_than_parking_a_fence() {
        let stager = PinnedStager::noop();
        {
            let _g = stager.begin_generation();
            let mut inner = stager.inner.lock().unwrap();
            // Stand in for submitted bump allocations.
            inner.arena_dirty = true;
        }
        let inner = stager.inner.lock().unwrap();
        assert_eq!(inner.live_generations, 0, "guard released");
        assert!(
            !inner.arena_dirty,
            "deviceless drop must reset, not defer behind a fence that cannot fire",
        );
        assert!(
            inner.pending_reset.is_none(),
            "no fence should be parked when there is no stream",
        );
    }

    /// A generation still live must block the reset — the whole invariant the
    /// guard exists for. Nesting is what makes the deferred path safe: the
    /// inner drop sees a non-zero count and records nothing.
    #[test]
    fn nested_generation_keeps_arena_live() {
        let stager = PinnedStager::noop();
        let outer = stager.begin_generation();
        {
            let _inner_gen = stager.begin_generation();
            let mut inner = stager.inner.lock().unwrap();
            inner.arena_dirty = true;
        }
        {
            let inner = stager.inner.lock().unwrap();
            assert_eq!(inner.live_generations, 1, "outer guard still held");
            assert!(
                inner.arena_dirty,
                "arena reset while a generation was still live — submitted bump \
                 pointers would dangle",
            );
        }
        drop(outer);
        let inner = stager.inner.lock().unwrap();
        assert!(!inner.arena_dirty, "reset once the last guard released");
    }

    /// `begin_generation` bumps the epoch, so a consumer holding a device
    /// pointer from an earlier generation can detect that the arena underneath
    /// it was reset. Deferring the reset must not stall the epoch.
    #[test]
    fn epoch_advances_per_generation() {
        let stager = PinnedStager::noop();
        let a = stager.begin_generation().epoch();
        let b = stager.begin_generation().epoch();
        assert!(b > a, "epoch must advance so stale pointers are detectable");
    }
}

#[cfg(all(test, feature = "cuda"))]
pub(super) mod gpu_buf_clone_tests {
    use super::*;
    use crate::backend::BackendDevice;

    /// A clone of an owned buffer is another handle on the same device bytes:
    /// same address, no second allocation, and the bytes outlive the handle
    /// they were made with.
    #[test]
    fn a_clone_shares_an_owned_buffers_bytes() {
        let crate::Device::Cuda(dev) = crate::Device::new_cuda(0).unwrap() else {
            unreachable!("asked for a CUDA device")
        };
        let bytes: Vec<u8> = (0u8..64).collect();
        let original = GpuBuf::from_raw_owned(dev.memcpy_stod(&bytes).unwrap(), &dev);
        let clone = original.clone();
        assert_eq!(clone.dev_ptr(), original.dev_ptr());
        assert_eq!(clone.len(), 64);
        let shared = clone._owned.as_ref().expect("an owned buffer");
        assert_eq!(Arc::strong_count(shared), 2, "the clone allocated");
        drop(original);
        dev.synchronize().unwrap();
        let mut back = vec![0u8; 64];
        let stream = dev.cuda_stream();
        // SAFETY: `clone` holds the 64 bytes at `dev_ptr` alive, and the wait
        // below keeps `back` borrowed until the copy has landed.
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(
                &mut back,
                clone.dev_ptr(),
                stream.cu_stream(),
            )
        }
        .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(back, bytes);
    }
}

#[cfg(test)]
pub(super) mod fallible_alloc_tests {
    use super::*;

    /// The staging fallback must return `Err` — never abort the process — when
    /// neither pinned host memory nor the plain host heap can supply the size.
    /// This is the whole point of `try_reserve_exact` over `vec![0u8; n]`, whose
    /// `handle_alloc_error` aborted a full overnight load.
    #[test]
    fn errs_on_impossible_size_instead_of_aborting() {
        let r = PinnedBuf::alloc_default_or_host_fallible(usize::MAX);
        assert!(r.is_err(), "impossible alloc must return Err, not abort");
    }

    /// A modest size still succeeds (pinned when a CUDA context is present, else
    /// the fallible host heap).
    #[test]
    fn small_alloc_succeeds() {
        let b = PinnedBuf::alloc_default_or_host_fallible(4096).expect("4 KiB staging");
        assert!(b.len() >= 4096);
    }

    /// A stand-in buffer of `len` bytes. `Host`, so the recycler's bookkeeping is
    /// testable with no CUDA context — `cuMemHostAlloc` is exactly what these
    /// tests exist to avoid calling.
    fn stub(len: usize) -> PinnedBuf {
        PinnedBuf::Host {
            data: vec![0u8; len],
        }
    }

    fn empty_state() -> RecyclerState {
        RecyclerState {
            by_len: std::collections::HashMap::new(),
            bytes: 0,
        }
    }

    /// **A returned buffer comes back to the next taker of the same size**, which
    /// is the whole point: the second take must not reach the allocator.
    #[test]
    fn a_returned_buffer_is_handed_to_the_next_taker() {
        let mut pool = empty_state();
        assert!(pool.take(4096).is_none(), "nothing held yet");
        assert!(pool.give(stub(4096)).is_none(), "retained");
        assert_eq!(pool.bytes, 4096);
        let got = pool.take(4096).expect("the buffer comes back");
        assert_eq!(got.len(), 4096);
        assert_eq!(pool.bytes, 0, "taking it back un-counts the bytes");
    }

    /// **Sizes do not substitute for each other.** A 4 KiB buffer handed to a
    /// caller that asked for 8 KiB would be short, and the copy out of it would
    /// read past its end.
    #[test]
    fn a_different_size_is_not_a_hit() {
        let mut pool = empty_state();
        pool.give(stub(4096));
        assert!(pool.take(8192).is_none(), "8 KiB is not served by 4 KiB");
        assert!(pool.take(4096).is_some(), "its own size still hits");
    }

    /// **The pool is capped, and refuses by handing the buffer back.** Pinned
    /// pages are host-wide, so an unbounded pool would trade this latency fix for
    /// a footprint problem; a refusal returns the buffer so the caller's drop
    /// frees it.
    #[test]
    fn the_cap_refuses_and_returns_the_buffer() {
        let mut pool = empty_state();
        let big = RECYCLER_MAX_BYTES;
        assert!(pool.give(stub(big)).is_none(), "the first one fits exactly");
        assert_eq!(pool.bytes, big);
        let refused = pool.give(stub(4096)).expect("over the cap, handed back");
        assert_eq!(refused.len(), 4096);
        assert_eq!(pool.bytes, big, "a refusal does not count toward the pool");
    }

    /// **A drain hands back every size and leaves the pool empty**, so nothing it
    /// held is still counted against the host's pinned pages.
    #[test]
    fn a_drain_empties_every_size() {
        let mut pool = empty_state();
        pool.give(stub(4096));
        pool.give(stub(4096));
        pool.give(stub(8192));
        let mut lens: Vec<usize> = pool.drain().iter().map(|b| b.len()).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![4096, 4096, 8192]);
        assert_eq!(pool.bytes, 0);
        assert!(pool.take(4096).is_none(), "nothing left to hand out");
        assert!(pool.take(8192).is_none(), "nothing left to hand out");
    }

    /// A zero-length buffer is not worth a slot: `alloc_owned(0)` is already a
    /// no-CUDA `Bump` and every `Staging` starts with one.
    #[test]
    fn a_zero_length_buffer_is_not_pooled() {
        let mut pool = empty_state();
        assert!(pool.give(stub(0)).is_some(), "handed back, not retained");
        assert_eq!(pool.bytes, 0);
    }

    /// Only `Owned` is pooled. A `Bump` points into a stager arena that will reset
    /// under it, and a `Host` vector is not pinned, so neither may be recycled as
    /// though it were a pinned buffer this pool owns.
    #[test]
    fn the_public_return_path_keeps_only_owned_buffers() {
        let (_, before) = recycled_stats();
        give_recycled_wc(stub(4096));
        let (_, after) = recycled_stats();
        assert_eq!(after, before, "a Host buffer is not taken into the pool");
    }
}
