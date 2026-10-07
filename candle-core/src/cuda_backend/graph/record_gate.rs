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
//! forward.

use std::cell::Cell;
use std::sync::{Condvar, Mutex, MutexGuard};

#[derive(Default)]
struct GateState {
    /// Recording segments in progress, on every thread and device.
    recording: usize,
    /// A device-wide operation is running; recordings wait for it.
    exclusive: bool,
}

static STATE: Mutex<GateState> = Mutex::new(GateState {
    recording: 0,
    exclusive: false,
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
    while st.exclusive {
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
pub fn try_without_recording<R>(f: impl FnOnce() -> R) -> Option<R> {
    if RECORDING_HERE.with(|r| r.get()) {
        return None;
    }
    {
        let mut st = lock();
        if st.recording > 0 || st.exclusive {
            return None;
        }
        st.exclusive = true;
    }
    let out = f();
    lock().exclusive = false;
    CHANGED.notify_all();
    Some(out)
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
        assert_eq!(try_without_recording(|| 7), Some(7), "nothing recording");
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
            try_without_recording(|| 7),
            None,
            "another thread is recording"
        );
        ended.wait();
        recorder.join().unwrap();
        assert_eq!(try_without_recording(|| 7), Some(7), "the recording ended");
    }

    #[test]
    fn the_recording_thread_is_always_turned_away() {
        let _one = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        recording_begins();
        assert_eq!(try_without_recording(|| 1), None);
        recording_ends();
        assert_eq!(try_without_recording(|| 1), Some(1));
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
        let ran = try_without_recording(|| {
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
