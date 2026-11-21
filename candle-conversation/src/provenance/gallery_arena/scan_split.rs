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
    /// Scans that built their index rather than reused a cached one.
    pub index_built: u64,
    /// Of `index_us`: the builds.
    pub index_build_us: u64,
    /// Builds because no index was cached for the segment set.
    pub index_absent: u64,
    /// Builds because a turn the cached index held was evicted, moved or
    /// re-uploaded.
    pub index_stale: u64,
    /// Builds because the cached index scanned fewer fold groups.
    pub index_narrower: u64,
    /// Windows fingerprinted, over all scans.
    pub windows: u64,
    /// Turns the index pinned resident, over all scans.
    pub pinned: u64,
    /// Probe upload.
    pub upload_us: u64,
    /// Enqueue to the kernel's completion on the stream — the scan shares the
    /// compute stream, so work queued ahead of it is counted here too.
    pub kernel_us: u64,
    /// Of `kernel_us`: the kernel's own device time, between events recorded
    /// on the stream either side of it — what `kernel_us` holds beyond this
    /// is work queued ahead of the launch and the host's enqueue.
    pub kernel_gpu_us: u64,
    /// The outputs back to the host.
    pub readback_us: u64,
    /// Row cache plan, insert and assembly.
    pub rows_us: u64,
    /// The host's needle tally.
    pub tally_us: u64,
    /// Launches sent to the arena's scan thread from another
    /// ([`super::HeldScan`]).
    pub held: u64,
    /// Of the held launches: copying the segments into owned form on the
    /// sending thread.
    pub hold_us: u64,
    /// Of the held launches: waiting for the scan thread to take them up.
    pub queue_us: u64,
    /// Of the held launches: laying the owned segments back out on the scan
    /// thread before the launch.
    pub view_us: u64,
    /// Of `fresh_tokens`: those scored over an index this split's scans built
    /// — a built index starts with no rows, so every probe token is fresh.
    pub built_fresh_tokens: u64,
    /// The launch with the most device time: its fresh tokens, the corpus
    /// tokens its index scans, and that device time — what one launch of a
    /// slow scan was made of.
    pub widest_fresh: u64,
    pub widest_tokens: u64,
    pub widest_gpu_us: u64,
    /// Turns a reused index's check looked up — the turns changes named, or
    /// every turn when it was walked whole — and the checks that walked whole.
    pub checked_turns: u64,
    pub checks_whole: u64,
}

impl ArenaScanSplit {
    /// Keep `(fresh, tokens, gpu_us)` as the widest launch when it took more
    /// device time than the widest so far.
    pub(crate) fn note_launch(&mut self, fresh: u64, tokens: u64, gpu_us: u64) {
        if gpu_us >= self.widest_gpu_us && (gpu_us > 0 || self.widest_fresh == 0) {
            self.widest_fresh = fresh;
            self.widest_tokens = tokens;
            self.widest_gpu_us = gpu_us;
        }
    }
}

thread_local! {
    static ARENA_SCAN_SPLIT: Cell<ArenaScanSplit> = const { Cell::new(ArenaScanSplit {
        scans: 0,
        launches: 0,
        segments: 0,
        fresh_tokens: 0,
        index_us: 0,
        fingerprint_us: 0,
        index_built: 0,
        index_build_us: 0,
        index_absent: 0,
        index_stale: 0,
        index_narrower: 0,
        windows: 0,
        pinned: 0,
        upload_us: 0,
        kernel_us: 0,
        kernel_gpu_us: 0,
        readback_us: 0,
        rows_us: 0,
        tally_us: 0,
        held: 0,
        hold_us: 0,
        queue_us: 0,
        view_us: 0,
        built_fresh_tokens: 0,
        widest_fresh: 0,
        widest_tokens: 0,
        widest_gpu_us: 0,
        checked_turns: 0,
        checks_whole: 0,
    }) };
}

/// Where one kernel launch's time went, in microseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LaunchTiming {
    /// The probe upload.
    pub upload_us: u64,
    /// The kernel, to its completion.
    pub kernel_us: u64,
    /// Of `kernel_us`: the kernel's own device time.
    pub kernel_gpu_us: u64,
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
        s.index_built += other.index_built;
        s.index_build_us += other.index_build_us;
        s.index_absent += other.index_absent;
        s.index_stale += other.index_stale;
        s.index_narrower += other.index_narrower;
        s.windows += other.windows;
        s.pinned += other.pinned;
        s.upload_us += other.upload_us;
        s.kernel_us += other.kernel_us;
        s.kernel_gpu_us += other.kernel_gpu_us;
        s.readback_us += other.readback_us;
        s.rows_us += other.rows_us;
        s.tally_us += other.tally_us;
        s.held += other.held;
        s.hold_us += other.hold_us;
        s.queue_us += other.queue_us;
        s.view_us += other.view_us;
        s.built_fresh_tokens += other.built_fresh_tokens;
        s.checked_turns += other.checked_turns;
        s.checks_whole += other.checks_whole;
        if other.widest_gpu_us > 0 || other.widest_fresh > 0 {
            s.note_launch(other.widest_fresh, other.widest_tokens, other.widest_gpu_us);
        }
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
            index_built: 7,
            index_build_us: 8,
            index_absent: 9,
            index_stale: 10,
            index_narrower: 11,
            windows: 12,
            pinned: 13,
            upload_us: 14,
            kernel_us: 15,
            kernel_gpu_us: 16,
            readback_us: 17,
            rows_us: 18,
            tally_us: 19,
            held: 20,
            hold_us: 21,
            queue_us: 22,
            view_us: 23,
            built_fresh_tokens: 24,
            widest_fresh: 25,
            widest_tokens: 26,
            widest_gpu_us: 27,
            checked_turns: 28,
            checks_whole: 29,
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
                index_built: 14,
                index_build_us: 16,
                index_absent: 18,
                index_stale: 20,
                index_narrower: 22,
                windows: 24,
                pinned: 26,
                upload_us: 28,
                kernel_us: 30,
                kernel_gpu_us: 32,
                readback_us: 34,
                rows_us: 36,
                tally_us: 38,
                held: 40,
                hold_us: 42,
                queue_us: 44,
                view_us: 46,
                built_fresh_tokens: 48,
                widest_fresh: 25,
                widest_tokens: 26,
                widest_gpu_us: 27,
                checked_turns: 56,
                checks_whole: 58,
            }
        );
    }

    /// **The widest launch is the one with the most device time**, a later
    /// launch replacing it only by taking longer.
    #[test]
    fn the_widest_launch_is_the_longest() {
        let mut s = ArenaScanSplit::default();
        s.note_launch(64, 38_000_000, 7_000);
        s.note_launch(2_400, 70_000, 3_000);
        assert_eq!(
            (s.widest_fresh, s.widest_tokens, s.widest_gpu_us),
            (64, 38_000_000, 7_000)
        );
        s.note_launch(300, 38_000_000, 30_000);
        assert_eq!(
            (s.widest_fresh, s.widest_tokens, s.widest_gpu_us),
            (300, 38_000_000, 30_000)
        );
    }
}
