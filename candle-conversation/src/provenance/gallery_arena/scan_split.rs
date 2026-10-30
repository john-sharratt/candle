//! Where the arena's scans on this thread spent their time, summed until taken.
//!
//! A belief scan launches the arena several times — the collection, each turn
//! group, each again for the question window — and the per-launch lines are
//! `trace`. The scan's own summary line takes this split, so one line a scan
//! says whether its arena time went to the index, the kernel, the readback or the
//! host tally.

use std::cell::Cell;

/// The arena's work since the split was last taken.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaScanSplit {
    /// `scan_weighted` calls.
    pub scans: u64,
    /// Kernel launches (a scan whose probe rows were all cached launches none).
    pub launches: u64,
    /// Segments over all scans.
    pub segments: u64,
    /// Probe tokens scored fresh by a kernel.
    pub fresh_tokens: u64,
    /// Finding or building the paged index.
    pub index_us: u64,
    /// Of `index_us`: fingerprinting the segment set, over every window.
    pub fingerprint_us: u64,
    /// Windows fingerprinted, over all scans.
    pub windows: u64,
    /// Turns the index pinned resident, over all scans.
    pub pinned: u64,
    /// Probe upload.
    pub upload_us: u64,
    /// Enqueue to the kernel's completion on the stream — the scan shares the
    /// compute stream, so work queued ahead of it is counted here too.
    pub kernel_us: u64,
    /// The outputs back to the host.
    pub readback_us: u64,
    /// Row cache plan, insert and assembly.
    pub rows_us: u64,
    /// The host's needle tally.
    pub tally_us: u64,
}

thread_local! {
    static ARENA_SCAN_SPLIT: Cell<ArenaScanSplit> = const { Cell::new(ArenaScanSplit {
        scans: 0,
        launches: 0,
        segments: 0,
        fresh_tokens: 0,
        index_us: 0,
        fingerprint_us: 0,
        windows: 0,
        pinned: 0,
        upload_us: 0,
        kernel_us: 0,
        readback_us: 0,
        rows_us: 0,
        tally_us: 0,
    }) };
}

/// Where one kernel launch's time went, in microseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LaunchTiming {
    /// The probe upload.
    pub upload_us: u64,
    /// The kernel, to its completion.
    pub kernel_us: u64,
    /// The outputs back to the host.
    pub readback_us: u64,
}

/// Fold `f`'s changes into this thread's split.
pub(crate) fn add(f: impl FnOnce(&mut ArenaScanSplit)) {
    ARENA_SCAN_SPLIT.with(|c| {
        let mut s = c.get();
        f(&mut s);
        c.set(s);
    });
}

/// This thread's split since the last take, which it resets.
pub fn take() -> ArenaScanSplit {
    ARENA_SCAN_SPLIT.take()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Additions accumulate, and a take hands them back and starts over.
    #[test]
    fn additions_accumulate_until_taken() {
        take();
        add(|s| {
            s.scans += 1;
            s.kernel_us += 5;
        });
        add(|s| {
            s.scans += 1;
            s.kernel_us += 7;
        });
        let got = take();
        assert_eq!((got.scans, got.kernel_us), (2, 12));
        assert_eq!(take(), ArenaScanSplit::default());
    }
}
