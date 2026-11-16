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

/// Fold `other` — a split taken on another thread, for a scan of this
/// thread's — into this thread's.
pub(crate) fn merge(other: ArenaScanSplit) {
    add(|s| {
        s.scans += other.scans;
        s.launches += other.launches;
        s.segments += other.segments;
        s.fresh_tokens += other.fresh_tokens;
        s.index_us += other.index_us;
        s.fingerprint_us += other.fingerprint_us;
        s.windows += other.windows;
        s.pinned += other.pinned;
        s.upload_us += other.upload_us;
        s.kernel_us += other.kernel_us;
        s.readback_us += other.readback_us;
        s.rows_us += other.rows_us;
        s.tally_us += other.tally_us;
    });
}

/// Run `f` with this thread's split set aside, and hand back what `f` added:
/// a launch run here for a scan on another thread is that scan's to count.
pub(crate) fn apart<R>(f: impl FnOnce() -> R) -> (R, ArenaScanSplit) {
    let own = take();
    let out = f();
    let added = take();
    merge(own);
    (out, added)
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

    /// **Work done apart is handed back, and the thread's own is kept**; a
    /// merge adds every field.
    #[test]
    fn work_done_apart_is_handed_back_and_the_own_kept() {
        take();
        add(|s| s.scans = 3);
        let ((), added) = apart(|| {
            add(|s| {
                s.scans += 1;
                s.index_us += 9;
            })
        });
        assert_eq!((added.scans, added.index_us), (1, 9));
        let own = take();
        assert_eq!(
            (own.scans, own.index_us),
            (3, 0),
            "the thread's own work, untouched"
        );

        let every = ArenaScanSplit {
            scans: 1,
            launches: 2,
            segments: 3,
            fresh_tokens: 4,
            index_us: 5,
            fingerprint_us: 6,
            windows: 7,
            pinned: 8,
            upload_us: 9,
            kernel_us: 10,
            readback_us: 11,
            rows_us: 12,
            tally_us: 13,
        };
        merge(every);
        merge(every);
        let doubled = take();
        assert_eq!(
            doubled,
            ArenaScanSplit {
                scans: 2,
                launches: 4,
                segments: 6,
                fresh_tokens: 8,
                index_us: 10,
                fingerprint_us: 12,
                windows: 14,
                pinned: 16,
                upload_us: 18,
                kernel_us: 20,
                readback_us: 22,
                rows_us: 24,
                tally_us: 26,
            }
        );
    }
}
