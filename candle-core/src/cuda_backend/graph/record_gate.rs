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

//!
//! # Stream leases
//!
//! A device has one stream, and a capture on it takes in every launch issued on
//! it — from any thread. So a thread that is not recording takes a lease
//! ([`lease_begins`]) before it issues anything on a device's stream, and holds
//! it while it does: the lease waits while a segment records on a device stream,
//! and a segment about to record there checks for leases first
//! ([`try_stream_recording_begins`]). Finding one, the segment runs as issued
//! instead of waiting — the recording thread never waits on a lease, so a lease
//! holder blocked on a lock the recording thread holds cannot stall it.

use std::cell::Cell;
use std::marker::PhantomData;
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

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
    /// Of `recording`, the segments recording on a device's stream — the ones
    /// a lease waits for.
    on_stream: usize,
    /// Stream leases held, on every thread.
    leases: usize,
    /// A test holds the gate to itself ([`exclusive`]): leases from threads it
    /// has not admitted wait.
    #[cfg(test)]
    held_off: bool,
}

static STATE: Mutex<GateState> = Mutex::new(GateState {
    recording: 0,
    quiet: 0,
    on_stream: 0,
    leases: 0,
    #[cfg(test)]
    held_off: false,
});
static CHANGED: Condvar = Condvar::new();

/// How long a lease waits for a recording segment before it is refused as a
/// stall: a segment records for under a millisecond, so a wait this long is a
/// lease holder holding something the recording thread needs.
const LEASE_STALL: Duration = Duration::from_secs(30);

thread_local! {
    /// This thread is recording a segment.
    static RECORDING_HERE: Cell<bool> = const { Cell::new(false) };
    /// This thread is recording a segment on a device's stream: its own
    /// launches are what the segment captures, so its leases are not counted.
    static ON_STREAM_HERE: Cell<bool> = const { Cell::new(false) };
    /// Stream leases this thread holds — not in its way when it begins to
    /// record.
    static LEASES_HERE: Cell<usize> = const { Cell::new(0) };
    /// This thread belongs to the test holding the gate to itself.
    #[cfg(test)]
    static ADMITTED: Cell<bool> = const { Cell::new(false) };
}

fn lock() -> MutexGuard<'static, GateState> {
    STATE.lock().unwrap_or_else(|p| p.into_inner())
}

/// Whether a lease from this thread waits: while a segment records on a
/// device's stream, and — in tests — while another test holds the gate. A
/// thread already holding a lease is never held off: the test is waiting for
/// that lease to come back.
fn lease_waits(st: &GateState) -> bool {
    #[cfg(test)]
    if st.held_off && !ADMITTED.with(|a| a.get()) && LEASES_HERE.with(|l| l.get()) == 0 {
        return true;
    }
    st.on_stream > 0
}

/// The gate held by one test, with every stream lease from a thread outside it
/// held off until this drops.
///
/// The process has one device per ordinal and so one stream, shared by every
/// test the harness runs at once: another test's lease turns a wave test's
/// segment away, and another test's launches land in the middle of a gate
/// test's counts. The gate's tests and the wave tests take this instead, one at
/// a time; a thread a test spawns joins it with [`admit_this_thread`].
#[cfg(test)]
pub(crate) struct Exclusive {
    _turn: MutexGuard<'static, ()>,
}

#[cfg(test)]
pub(crate) fn exclusive() -> Exclusive {
    static TURN: Mutex<()> = Mutex::new(());
    let turn = TURN.lock().unwrap_or_else(|p| p.into_inner());
    ADMITTED.with(|a| a.set(true));
    let mut st = lock();
    st.held_off = true;
    // Leases already granted outside the test run to their end first.
    while st.leases > LEASES_HERE.with(|l| l.get()) {
        st = CHANGED.wait(st).unwrap_or_else(|p| p.into_inner());
    }
    Exclusive { _turn: turn }
}

#[cfg(test)]
impl Drop for Exclusive {
    fn drop(&mut self) {
        ADMITTED.with(|a| a.set(false));
        lock().held_off = false;
        CHANGED.notify_all();
    }
}

/// This thread belongs to the test holding the gate ([`exclusive`]).
#[cfg(test)]
pub(crate) fn admit_this_thread() {
    ADMITTED.with(|a| a.set(true));
}

/// A segment is about to start recording on this thread, on a stream of its
/// own. Waits while a device-wide operation runs — it is bounded, and it holds
/// nothing a recording thread holds.
#[cfg(test)]
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
    if ON_STREAM_HERE.with(|r| r.replace(false)) {
        st.on_stream = st.on_stream.saturating_sub(1);
    }
    RECORDING_HERE.with(|r| r.set(false));
    drop(st);
    CHANGED.notify_all();
}

/// A segment recording on a device's stream, counted in the gate until this
/// drops — which its thread does once the segment's graph is on the stream, so
/// a lease granted after it lands behind that launch.
#[must_use = "the segment stays counted until this drops"]
pub(crate) struct StreamRecording {
    _thread_bound: PhantomData<*const ()>,
}

impl StreamRecording {
    /// The recording this thread began with [`try_stream_recording_begins`],
    /// now ending.
    pub(crate) fn ending() -> Self {
        Self {
            _thread_bound: PhantomData,
        }
    }
}

impl Drop for StreamRecording {
    fn drop(&mut self) {
        recording_ends();
    }
}

/// A segment would start recording on a device's stream, on this thread:
/// `true` once it is counted in, `false` — counting nothing — while another
/// thread holds a stream lease, whose launches the capture would take in. The
/// caller then runs the segment as issued. Waits, like [`recording_begins`],
/// only while a device-wide operation runs.
pub(crate) fn try_stream_recording_begins() -> bool {
    let mut st = lock();
    while st.quiet > 0 {
        st = CHANGED.wait(st).unwrap_or_else(|p| p.into_inner());
    }
    let mine = LEASES_HERE.with(|l| l.get());
    if st.leases > mine {
        return false;
    }
    st.recording += 1;
    st.on_stream += 1;
    RECORDING_HERE.with(|r| r.set(true));
    ON_STREAM_HERE.with(|r| r.set(true));
    true
}

/// Take a stream lease for this thread: `true` when it was counted and must be
/// returned with [`lease_ends`], `false` when this thread is recording on a
/// device's stream and its launches belong to the segment.
///
/// Waits while a segment records on a device's stream on another thread. A wait
/// past [`LEASE_STALL`] is refused with a panic naming the site: a recording
/// segment ends within a millisecond, so this thread is holding something the
/// recording thread is waiting for, and waiting longer would hang both.
#[track_caller]
pub(crate) fn lease_begins() -> bool {
    if ON_STREAM_HERE.with(|r| r.get()) {
        return false;
    }
    let site = Location::caller();
    let started = Instant::now();
    let mut st = lock();
    let waited = lease_waits(&st);
    while lease_waits(&st) {
        let (next, _) = CHANGED
            .wait_timeout(st, Duration::from_millis(100))
            .unwrap_or_else(|p| p.into_inner());
        st = next;
        if lease_waits(&st) && started.elapsed() > LEASE_STALL {
            drop(st);
            panic!(
                "stream lease at {site} waited {LEASE_STALL:?} for a segment recording on \
                 another thread: this thread holds something that thread is waiting for"
            );
        }
    }
    st.leases += 1;
    LEASES_HERE.with(|l| l.set(l.get() + 1));
    drop(st);
    if waited {
        note_lease_wait(site, started.elapsed());
    }
    true
}

/// Leases that found a segment recording and waited for it, in all: how many,
/// and for how long. A thread issuing device work off the forward thread — the
/// expert pipeline, persistence — pays this wait on its own path, which no
/// segment count shows.
static LEASE_WAITS: AtomicU64 = AtomicU64::new(0);
static LEASE_WAIT_NS: AtomicU64 = AtomicU64::new(0);
/// The same, per call site, kept only for leases that waited.
static LEASE_WAIT_SITES: Mutex<Vec<(&'static Location<'static>, u64, u64)>> =
    Mutex::new(Vec::new());

fn note_lease_wait(site: &'static Location<'static>, waited: Duration) {
    let ns = waited.as_nanos() as u64;
    LEASE_WAITS.fetch_add(1, Ordering::Relaxed);
    LEASE_WAIT_NS.fetch_add(ns, Ordering::Relaxed);
    let mut sites = LEASE_WAIT_SITES.lock().unwrap_or_else(|p| p.into_inner());
    match sites.iter_mut().find(|(s, _, _)| *s == site) {
        Some((_, n, total)) => {
            *n += 1;
            *total += ns;
        }
        None => sites.push((site, 1, ns)),
    }
}

/// Leases that waited for a recording segment since start, and their wait in
/// all, in µs.
pub fn lease_wait_total() -> (u64, u64) {
    (
        LEASE_WAITS.load(Ordering::Relaxed),
        LEASE_WAIT_NS.load(Ordering::Relaxed) / 1_000,
    )
}

/// Per call site, since start: `(site, leases that waited, µs waited)`, the
/// longest total first.
pub fn lease_wait_sites() -> Vec<(String, u64, u64)> {
    let sites = LEASE_WAIT_SITES.lock().unwrap_or_else(|p| p.into_inner());
    let mut out: Vec<_> = sites
        .iter()
        .map(|(s, n, ns)| (s.to_string(), *n, ns / 1_000))
        .collect();
    out.sort_by(|a, b| b.2.cmp(&a.2));
    out
}

/// Return a lease [`lease_begins`] counted.
pub(crate) fn lease_ends() {
    let mut st = lock();
    st.leases = st.leases.saturating_sub(1);
    LEASES_HERE.with(|l| l.set(l.get().saturating_sub(1)));
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

    #[test]
    fn a_recording_turns_the_operation_away_and_its_end_lets_it_run() {
        let _one = exclusive();
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
        let _one = exclusive();
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
        let _one = exclusive();
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
        let _one = exclusive();
        let caught = std::panic::catch_unwind(|| try_without_recording(|_| panic!("boom")));
        assert!(caught.is_err());
        assert_eq!(try_without_recording(|_| 3), Some(3));
    }

    /// **Two device-wide operations do not turn each other away.** Only a
    /// recording conflicts with one; two threads recycling regions at once
    /// were refused as though one of them were recording.
    #[test]
    fn two_operations_run_at_once() {
        let _one = exclusive();
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

    /// **Another thread's lease turns a stream recording away, and only until
    /// it is returned.** The segment runs as issued instead of waiting.
    #[test]
    fn a_lease_elsewhere_turns_a_stream_recording_away() {
        let _one = exclusive();
        let taken = Arc::new(Barrier::new(2));
        let returned = Arc::new(Barrier::new(2));
        let holder = {
            let (taken, returned) = (taken.clone(), returned.clone());
            thread::spawn(move || {
                admit_this_thread();
                assert!(lease_begins(), "nothing records: the lease is counted");
                taken.wait();
                returned.wait();
                lease_ends();
            })
        };
        taken.wait();
        assert!(
            !try_stream_recording_begins(),
            "another thread holds a lease"
        );
        returned.wait();
        holder.join().unwrap();
        assert!(try_stream_recording_begins(), "the lease came back");
        recording_ends();
    }

    /// **A lease waits while a segment records on a stream, and is granted
    /// once it ends.**
    #[test]
    fn a_lease_waits_for_a_stream_recording() {
        let _one = exclusive();
        let started = Arc::new(Barrier::new(2));
        let order = Arc::new(Mutex::new(Vec::new()));
        let recorder = {
            let (started, order) = (started.clone(), order.clone());
            thread::spawn(move || {
                assert!(try_stream_recording_begins());
                started.wait();
                thread::sleep(std::time::Duration::from_millis(50));
                order.lock().unwrap().push("recording ended");
                recording_ends();
            })
        };
        started.wait();
        assert!(lease_begins());
        order.lock().unwrap().push("leased");
        lease_ends();
        recorder.join().unwrap();
        assert_eq!(*order.lock().unwrap(), vec!["recording ended", "leased"]);
    }

    /// **A thread's own leases are not in its way.** A launcher may hold the
    /// stream across the flush that begins this thread's next segment, and the
    /// recording thread's own leases are not counted at all: its launches are
    /// what the segment takes in.
    #[test]
    fn a_threads_own_lease_does_not_turn_its_recording_away() {
        let _one = exclusive();
        assert!(lease_begins());
        assert!(
            try_stream_recording_begins(),
            "only this thread holds a lease"
        );
        assert!(
            !lease_begins(),
            "the recording thread's lease is not counted"
        );
        recording_ends();
        lease_ends();
        assert!(try_stream_recording_begins());
        recording_ends();
    }

    /// **A recording on a stream of its own does not hold leases up.** Only a
    /// capture on a device's stream takes in other threads' launches.
    #[test]
    fn a_capture_on_its_own_stream_does_not_hold_leases_up() {
        let _one = exclusive();
        let started = Arc::new(Barrier::new(2));
        let done = Arc::new(Barrier::new(2));
        let recorder = {
            let (started, done) = (started.clone(), done.clone());
            thread::spawn(move || {
                recording_begins();
                started.wait();
                done.wait();
                recording_ends();
            })
        };
        started.wait();
        assert!(lease_begins(), "granted at once beside a private capture");
        lease_ends();
        done.wait();
        recorder.join().unwrap();
    }

    #[test]
    fn a_recording_waits_for_the_operation_in_progress() {
        let _one = exclusive();
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
