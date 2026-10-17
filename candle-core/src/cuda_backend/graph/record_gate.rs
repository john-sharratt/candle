//! Device-wide synchronisation against wave recording.
//!
//! While any stream of a context is capturing, the driver refuses a
//! context-wide synchronise — `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` — and the
//! refusal **invalidates the capture** on the other thread, whatever its capture
//! mode. A wave records between forwards as well as inside them (the draft walk,
//! the speculative rewind), so a thread that synchronises the whole device — the
//! KV region pool quiescing a dirty region before it re-tenants it — can land in
//! the middle of a recording. Measured in zend: the persistence thread's arena
//! creation recycled a region while the scheduler recorded a draft walk, the
//! walk failed with `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`, the next wave's
//! bucketize never ran, and the expert pipeline aborted for good.
//!
//! This gate keeps the two apart. Every recording segment counts itself in for
//! as long as it records; [`try_without_recording`] runs its body only when none
//! is recording, and holds new recordings off until the body returns. It never
//! waits for a recording to end: the caller holds locks a recording thread may
//! want, so it takes the refusal and comes back, as it already does for an open
//! forward. [`without_recording`] is the waiting form, for a caller that holds
//! nothing a recording thread wants — teardown, a module load.
//!
//! # Enforced by type, not by convention
//!
//! The gate hands its body a [`Quiet`]: proof that nothing records while it is
//! held. It cannot be built anywhere else and does not leave the thread, and
//! every operation the driver refuses during a capture is a method on it
//! ([`super::device_wide`]). The raw driver calls are banned everywhere else by
//! `clippy.toml`'s `disallowed-methods`, so a synchronise that skips the gate
//! fails the build rather than a wave. The rule was a comment before, kept at
//! three sites and missed at the others — a module load on another thread broke
//! captures in the test suite, and a context synchronise on the persistence
//! thread broke one in zend.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::{Condvar, Mutex, MutexGuard};

/// Proof, for as long as it is held, that no segment records anywhere and none
/// can begin: what an operation the driver refuses during a capture requires.
/// Made only by [`try_without_recording`] and [`without_recording`], and bound
/// to the thread that holds the gate.
pub struct Quiet {
    _thread_bound: PhantomData<*const ()>,
}

#[derive(Default)]
struct GateState {
    /// Recording segments in progress, on every thread and device.
    recording: usize,
    /// Device-wide operations running; recordings wait for all of them. They
    /// conflict only with a recording, never with each other, so any number
    /// run at once — two threads recycling regions must not turn each other
    /// away as though one of them were recording.
    quiet: usize,
}

static STATE: Mutex<GateState> = Mutex::new(GateState {
    recording: 0,
    quiet: 0,
});
static CHANGED: Condvar = Condvar::new();

thread_local! {
    /// This thread is recording a segment.
    static RECORDING_HERE: Cell<bool> = const { Cell::new(false) };
}

fn lock() -> MutexGuard<'static, GateState> {
    STATE.lock().unwrap_or_else(|p| p.into_inner())
}

/// A segment is about to start recording on this thread. Waits while a
/// device-wide operation runs — it is bounded, and it holds nothing a
/// recording thread holds.
pub(crate) fn recording_begins() {
    let mut st = lock();
    while st.quiet > 0 {
        st = CHANGED.wait(st).unwrap_or_else(|p| p.into_inner());
    }
    st.recording += 1;
    RECORDING_HERE.with(|r| r.set(true));
}

/// The segment this thread was recording has ended, captured or abandoned.
pub(crate) fn recording_ends() {
    let mut st = lock();
    st.recording = st.recording.saturating_sub(1);
    RECORDING_HERE.with(|r| r.set(false));
    drop(st);
    CHANGED.notify_all();
}

/// Run `f` — an operation the driver refuses while any stream captures, such as
/// a context-wide synchronise — only if no segment is recording anywhere, with
/// new recordings held off until it returns. `None`, without running `f`, when
/// one is recording: the caller retries once it has ended.
///
/// A thread that is itself recording always gets `None`: the operation would end
/// its own capture.
pub fn try_without_recording<R>(f: impl FnOnce(&Quiet) -> R) -> Option<R> {
    if RECORDING_HERE.with(|r| r.get()) {
        return None;
    }
    {
        let mut st = lock();
        if st.recording > 0 {
            return None;
        }
        st.quiet += 1;
    }
    Some(run_quiet(f))
}

/// Run `f` once no segment is recording anywhere, waiting for the ones that
/// are, with new recordings held off until it returns.
///
/// For a caller that holds nothing a recording thread may want — a span's
/// teardown, a module load made with no lock held. A caller holding such a lock
/// takes [`try_without_recording`]'s refusal instead, or the two wait on each
/// other. Refused, with an error, on a thread that is itself recording: waiting
/// would wait on itself, and running would end its own capture.
pub fn without_recording<R>(f: impl FnOnce(&Quiet) -> R) -> crate::Result<R> {
    if RECORDING_HERE.with(|r| r.get()) {
        crate::bail!(
            "an operation the driver refuses during a graph capture was asked for on the \
             thread that is recording one"
        );
    }
    {
        let mut st = lock();
        while st.recording > 0 {
            st = CHANGED.wait(st).unwrap_or_else(|p| p.into_inner());
        }
        st.quiet += 1;
    }
    Ok(run_quiet(f))
}

/// `f` with recordings held off, released however `f` leaves.
fn run_quiet<R>(f: impl FnOnce(&Quiet) -> R) -> R {
    struct Release;
    impl Drop for Release {
        fn drop(&mut self) {
            let mut st = lock();
            st.quiet = st.quiet.saturating_sub(1);
            drop(st);
            CHANGED.notify_all();
        }
    }
    let _release = Release;
    f(&Quiet {
        _thread_bound: PhantomData,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    /// The gate is process-wide, so its tests take turns.
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn a_recording_turns_the_operation_away_and_its_end_lets_it_run() {
        let _one = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        assert_eq!(try_without_recording(|_| 7), Some(7), "nothing recording");
        let started = Arc::new(Barrier::new(2));
        let ended = Arc::new(Barrier::new(2));
        let recorder = {
            let (started, ended) = (started.clone(), ended.clone());
            thread::spawn(move || {
                recording_begins();
                started.wait();
                ended.wait();
                recording_ends();
            })
        };
        started.wait();
        assert_eq!(
            try_without_recording(|_| 7),
            None,
            "another thread is recording"
        );
        ended.wait();
        recorder.join().unwrap();
        assert_eq!(try_without_recording(|_| 7), Some(7), "the recording ended");
    }

    #[test]
    fn the_recording_thread_is_always_turned_away() {
        let _one = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        recording_begins();
        assert_eq!(try_without_recording(|_| 1), None);
        assert!(without_recording(|_| 1).is_err(), "it would wait on itself");
        recording_ends();
        assert_eq!(try_without_recording(|_| 1), Some(1));
        assert_eq!(without_recording(|_| 1).unwrap(), 1);
    }

    /// **The waiting form runs once the recording ends, not before.**
    #[test]
    fn the_waiting_form_runs_after_the_recording_ends() {
        let _one = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let started = Arc::new(Barrier::new(2));
        let order = Arc::new(Mutex::new(Vec::new()));
        let recorder = {
            let (started, order) = (started.clone(), order.clone());
            thread::spawn(move || {
                recording_begins();
                started.wait();
                thread::sleep(std::time::Duration::from_millis(50));
                order.lock().unwrap().push("recording ended");
                recording_ends();
            })
        };
        started.wait();
        without_recording(|_| order.lock().unwrap().push("operation")).unwrap();
        recorder.join().unwrap();
        assert_eq!(*order.lock().unwrap(), vec!["recording ended", "operation"]);
    }

    /// A body that panics still releases the gate: recordings are not held off
    /// for good by an operation that failed.
    #[test]
    fn a_panicking_body_releases_the_gate() {
        let _one = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let caught = std::panic::catch_unwind(|| try_without_recording(|_| panic!("boom")));
        assert!(caught.is_err());
        assert_eq!(try_without_recording(|_| 3), Some(3));
    }

    /// **Two device-wide operations do not turn each other away.** Only a
    /// recording conflicts with one; two threads recycling regions at once
    /// were refused as though one of them were recording.
    #[test]
    fn two_operations_run_at_once() {
        let _one = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let inside = Arc::new(Barrier::new(2));
        let other = {
            let inside = inside.clone();
            thread::spawn(move || {
                try_without_recording(|_| {
                    inside.wait();
                    inside.wait();
                })
            })
        };
        let ran = try_without_recording(|_| {
            inside.wait();
            let nested = try_without_recording(|_| 5);
            inside.wait();
            nested
        });
        assert_eq!(ran, Some(Some(5)), "both ran while the other was inside");
        assert_eq!(other.join().unwrap(), Some(()));
    }

    #[test]
    fn a_recording_waits_for_the_operation_in_progress() {
        let _one = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let inside = Arc::new(Barrier::new(2));
        let order = Arc::new(Mutex::new(Vec::new()));
        let recorder = {
            let (inside, order) = (inside.clone(), order.clone());
            thread::spawn(move || {
                inside.wait();
                recording_begins();
                order.lock().unwrap().push("recording");
                recording_ends();
            })
        };
        let ran = try_without_recording(|_| {
            inside.wait();
            // The recorder is now blocked behind this body.
            thread::sleep(std::time::Duration::from_millis(50));
            order.lock().unwrap().push("operation");
        });
        assert_eq!(ran, Some(()));
        recorder.join().unwrap();
        assert_eq!(*order.lock().unwrap(), vec!["operation", "recording"]);
    }
}
