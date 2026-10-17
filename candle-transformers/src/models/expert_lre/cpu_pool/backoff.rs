//! How the expert cache's host threads wait, the way a mutex does: hot spins
//! first where the answer may already be on its way, then a low-power wait,
//! and only once inference has gone quiet for [`COOL_DOWN`] a sleep the
//! scheduler wakes them from (`docs/cpu_expert_compute.md` §5.6).
//!
//! The low-power wait is `umwait` on the line the answer is written to, where
//! the CPU has WAITPKG: the store wakes it at once, whether a host thread's or
//! the device's over PCIe (measured, `handoff`), and until then the core sits
//! in C0.1 rather than spinning beside its hyperthread sibling. A wait with no
//! line to watch uses `tpause` for a bounded time. Either is bounded, so a
//! waiter re-checks everything else it waits on — a channel, a give-up
//! condition — every few microseconds.

use std::hint::spin_loop;
#[cfg(target_arch = "x86_64")]
use std::ptr::read_volatile;
use std::time::Duration;

#[cfg(target_arch = "x86_64")]
use super::handoff::{has_waitpkg, tpause, umonitor, umwait};

/// Spins a wait takes before it goes to low power, by what it waits for. The
/// count is the lever: a spin is worth taking only while the answer may land
/// inside it, and a core spinning is a core the package pays for.
///
/// Inside a layer — the caller waiting for its participants' last tasks, a
/// participant waiting on another's, a once-cell's build — the answer is
/// microseconds away: a few hundred spins.
pub(crate) const INTRA_LAYER_SPINS: u32 = 256;
/// Between layers — a worker's next job, the doorbell, a summary word, a cold
/// expert's publish — the answer is most of a millisecond away, and `umwait`
/// wakes on its store at once: none.
pub(crate) const INTER_LAYER_SPINS: u32 = 0;

/// How long the threads stay warm — waiting in low power rather than asleep —
/// after inference was last seen: past any gap inside a step or between a
/// session's steps, short enough that an idle daemon's cores are back in deep
/// sleep within a blink.
pub(crate) const COOL_DOWN: Duration = Duration::from_millis(100);

/// One low-power wait's bound in TSC cycles: ~10 µs.
pub(crate) const LOW_POWER_CYCLES: u64 = 30_000;

/// One waiter's escalation: `hot` spins, then low power. Reset whenever the
/// wait is answered.
#[derive(Debug)]
pub(crate) struct Backoff {
    polls: u32,
    hot: u32,
}

impl Backoff {
    /// A wait that spins `hot` times before it goes to low power.
    pub(crate) fn new(hot: u32) -> Self {
        Self { polls: 0, hot }
    }

    /// The wait was answered: the next one starts hot again.
    pub(crate) fn reset(&mut self) {
        self.polls = 0;
    }

    /// Whether the wait is past its hot spins.
    pub(crate) fn cooled(&self) -> bool {
        self.polls >= self.hot
    }

    /// One hot spin, while there are any left; false once they are spent.
    fn spin(&mut self) -> bool {
        if self.cooled() {
            return false;
        }
        self.polls += 1;
        spin_loop();
        true
    }

    /// One wait for the `u64` at `word` to stop holding `seen`: a hot spin, or
    /// past them `umwait` on its line for at most `cycles`.
    ///
    /// # Safety
    /// `word` is a live, aligned `u64` for the duration of the call.
    pub(crate) unsafe fn wait_u64(&mut self, word: *const u64, seen: u64, cycles: u64) {
        if self.spin() {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if has_waitpkg() {
            // Re-checked after arming, so a store between the caller's read
            // and the arm is not slept through.
            umonitor(word as *const u8);
            if read_volatile(word) == seen {
                umwait(cycles);
            }
            return;
        }
        let _ = (word, seen, cycles);
        spin_loop();
    }

    /// The same for a `u32`.
    ///
    /// # Safety
    /// `word` is a live, aligned `u32` for the duration of the call.
    pub(crate) unsafe fn wait_u32(&mut self, word: *const u32, seen: u32, cycles: u64) {
        if self.spin() {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if has_waitpkg() {
            umonitor(word as *const u8);
            if read_volatile(word) == seen {
                umwait(cycles);
            }
            return;
        }
        let _ = (word, seen, cycles);
        spin_loop();
    }

    /// One wait with no line to watch: a hot spin, or past them `tpause` for
    /// `cycles`.
    pub(crate) fn pause(&mut self, cycles: u64) {
        if self.spin() {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if has_waitpkg() {
            // SAFETY: WAITPKG is present.
            unsafe { tpause(cycles) };
            return;
        }
        let _ = cycles;
        spin_loop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    /// Hot for exactly its `hot` waits, then cooled until reset; a wait with
    /// none is cooled from the start.
    #[test]
    fn hot_spins_then_cools_until_reset() {
        let mut b = Backoff::new(INTRA_LAYER_SPINS);
        for _ in 0..INTRA_LAYER_SPINS {
            assert!(!b.cooled());
            b.pause(1_000);
        }
        assert!(b.cooled());
        b.pause(1_000);
        assert!(b.cooled());
        b.reset();
        assert!(!b.cooled());
        assert!(Backoff::new(INTER_LAYER_SPINS).cooled());
    }

    /// A store from another thread ends a low-power wait well before its
    /// bound: the waiter sees it within the ~10 ms the test allows, though
    /// each wait may last up to its bound.
    #[test]
    fn a_store_answers_a_cooled_wait() {
        let word = Arc::new(AtomicU64::new(7));
        let w = word.clone();
        let writer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(2));
            w.store(8, Ordering::Release);
        });
        let mut b = Backoff::new(INTRA_LAYER_SPINS);
        let t0 = Instant::now();
        while word.load(Ordering::Acquire) == 7 {
            assert!(t0.elapsed() < Duration::from_secs(1), "never answered");
            // SAFETY: a live atomic.
            unsafe { b.wait_u64(word.as_ptr(), 7, LOW_POWER_CYCLES) };
        }
        writer.join().unwrap();
        assert!(b.cooled(), "the wait outlasted the hot spins");
    }
}
