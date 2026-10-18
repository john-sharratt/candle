//! Global GPU-context poison flag.
//!
//! A *sticky* GPU fault — a CUDA illegal address, launch failure, misaligned
//! access, device assert, ECC/NVLink uncorrectable — leaves the device context
//! permanently unusable: every later call returns the same error, forever. There
//! is no API to clear it on the same context, and recreating the context
//! in-process is unreliable (especially on WDDM). So instead of letting an
//! endless cascade of identical downstream errors spew until the process is
//! killed by hand, the backend FLAGS the context poisoned on the first such
//! fault. The daemon watches this flag and exits cleanly for a supervisor
//! restart — one root fault, then a fast reboot. The substrate redo log is
//! crash-safe, so the abrupt exit loses nothing durable.
//!
//! The flag lives here in candle-core (not the `cuda`-gated backend) so a host
//! daemon can poll it without a `cuda` cfg; the CUDA backend is the only writer.
//!
//! `OUT_OF_MEMORY` is NOT sticky on its own — a single occurrence is often a
//! request that was too large for the moment, and admission control is
//! designed to shed it and recover. But a device that reports nothing else for
//! a sustained stretch is operationally indistinguishable from a truly-dead
//! context: measured in production, a KV-migration OOM cascaded into every
//! later device call failing identically for 40+ minutes with zero requests
//! served, because nothing was watching for "OOM, and it never got better."
//! [`note_oom`] tracks an unbroken run of OOM-only errors and promotes it to a
//! full poison once it has lasted [`OOM_STICKY_AFTER`] with no successful
//! device operation in between ([`note_device_ok`] closes the run).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

static POISONED: AtomicBool = AtomicBool::new(false);
static ROOT_FAULT: OnceLock<String> = OnceLock::new();

/// When the open out-of-memory streak began: nanoseconds since [`epoch`], plus
/// one, so that `0` is "no streak". An atomic, not a lock: every device error
/// and every capacity report reads it.
static OOM_STREAK_START: AtomicU64 = AtomicU64::new(0);

/// How long an unbroken run of out-of-memory errors — no successful device
/// operation observed in between — is treated the same as a sticky fault.
pub const OOM_STICKY_AFTER: Duration = Duration::from_secs(60);

/// How long an unbroken run of out-of-memory errors makes the device one that
/// cannot take a turn. A busy device completes operations constantly and any
/// one of them closes the run, so a run this long means nothing is working —
/// long before [`OOM_STICKY_AFTER`] relaunches the process.
pub const FAILING_AFTER: Duration = Duration::from_secs(2);

/// The instant streak times are measured from.
fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn now_mark() -> u64 {
    (epoch().elapsed().as_nanos() as u64).saturating_add(1)
}

/// Flag the GPU context poisoned. `root` — the fault plus the recent
/// kernel-launch breadcrumb — is evaluated and captured ONCE, on the first
/// poison. Returns `true` iff this call was the transition, so the caller
/// records the root exactly once and later poisons stay quiet.
pub fn poison_gpu(root: impl FnOnce() -> String) -> bool {
    let first = !POISONED.swap(true, Ordering::SeqCst);
    if first {
        let _ = ROOT_FAULT.set(root());
    }
    first
}

/// Record an out-of-memory error. Opens an OOM streak if none is active, and
/// poisons the context once the open streak has run for [`OOM_STICKY_AFTER`]
/// — `root` is evaluated only in that case, on the same first-poison path as
/// [`poison_gpu`].
pub fn note_oom(root: impl FnOnce() -> String) {
    if oom_streak_exceeds(OOM_STICKY_AFTER) {
        poison_gpu(root);
    }
}

/// Opens the OOM streak if none is active, and reports whether it has now run
/// for at least `threshold`. Split from [`note_oom`] so the timing/reset logic
/// is testable against a small threshold instead of forcing a real
/// 60-second wait, and without ever touching the one-way [`POISONED`] latch
/// that a real test binary can only afford to flip once.
fn oom_streak_exceeds(threshold: Duration) -> bool {
    // Open the streak unless one is open; either way read the one that is.
    let _ = OOM_STREAK_START.compare_exchange(0, now_mark(), Ordering::AcqRel, Ordering::Acquire);
    oom_streak_age().is_some_and(|age| age >= threshold)
}

/// Record that a device operation completed without error, closing any open
/// out-of-memory streak. A later, unrelated OOM then starts its own fresh
/// countdown instead of inheriting one left over from a resolved episode.
#[inline]
pub fn note_device_ok() {
    if OOM_STREAK_START.load(Ordering::Relaxed) != 0 {
        OOM_STREAK_START.store(0, Ordering::Release);
    }
}

/// How long the open out-of-memory streak has run; `None` with none open.
pub fn oom_streak_age() -> Option<Duration> {
    let started = OOM_STREAK_START.load(Ordering::Acquire);
    (started != 0).then(|| Duration::from_nanos(now_mark().saturating_sub(started)))
}

/// Whether the device can take a turn: not when its context is poisoned, and
/// not while every operation has failed out of memory for [`FAILING_AFTER`].
/// A member reports itself faulted while this holds, so nothing is routed to a
/// card that would answer every turn with an error.
pub fn device_failing() -> bool {
    failing(is_gpu_poisoned(), oom_streak_age())
}

fn failing(poisoned: bool, streak: Option<Duration>) -> bool {
    poisoned || streak.is_some_and(|age| age >= FAILING_AFTER)
}

/// Whether a sticky GPU fault has poisoned the context.
#[inline]
pub fn is_gpu_poisoned() -> bool {
    POISONED.load(Ordering::Relaxed)
}

/// The captured root fault (error + recent launches) if the context is poisoned.
pub fn root_fault() -> Option<String> {
    ROOT_FAULT.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_poison_captures_root_and_reports_transition() {
        // Fresh statics per test binary; this is the only test touching them.
        assert!(!is_gpu_poisoned());
        assert!(
            poison_gpu(|| "root-A".to_string()),
            "first poison is the transition"
        );
        assert!(is_gpu_poisoned());
        assert_eq!(root_fault().as_deref(), Some("root-A"));
        // A second poison is not the transition and does not overwrite the root.
        assert!(!poison_gpu(|| "root-B".to_string()));
        assert_eq!(root_fault().as_deref(), Some("root-A"));
    }

    /// A poisoned context fails; so does a streak of out-of-memory errors
    /// once it has run [`FAILING_AFTER`]; a shorter one, or none, does not.
    #[test]
    fn a_device_fails_poisoned_or_out_of_memory_for_two_seconds() {
        assert!(failing(true, None));
        assert!(failing(false, Some(FAILING_AFTER)));
        assert!(!failing(false, Some(Duration::from_millis(1_999))));
        assert!(!failing(false, None));
    }

    /// Exercises the OOM-streak timing and reset in isolation, via
    /// `oom_streak_exceeds` directly — never through `note_oom`/`poison_gpu` —
    /// so it cannot flip the one-way `POISONED` latch the other test in this
    /// file owns. Disjoint statics (`OOM_STREAK_START` here, `POISONED` /
    /// `ROOT_FAULT` there), so the two tests are safe to run concurrently.
    #[test]
    fn oom_streak_tracks_elapsed_time_and_resets_on_recovery() {
        let tiny = Duration::from_millis(20);
        // A fresh streak has not run for `tiny` yet.
        assert!(!oom_streak_exceeds(tiny));
        std::thread::sleep(tiny * 2);
        // Same streak, now old enough.
        assert!(oom_streak_exceeds(tiny));
        assert!(oom_streak_age().is_some_and(|a| a >= tiny));
        // A successful device operation closes the streak...
        note_device_ok();
        assert_eq!(oom_streak_age(), None);
        // ...so a fresh check starts its own countdown from zero again.
        assert!(!oom_streak_exceeds(tiny));
        note_device_ok();
    }
}
