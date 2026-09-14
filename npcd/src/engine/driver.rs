//! The world's own clock.
//!
//! A world is shared by every mind in it, so no mind can drive it: sixteen
//! Makers ticking one building would each advance it, and a journey would cover
//! sixteen legs in the time it should cover one. The world gets its own clock,
//! running on an interval, and the minds read what it leaves behind.
//!
//! # It runs a closure, not the engine
//!
//! What a moment *is* — advance the journeys, tell the minds — belongs to
//! [`crate::engine::environment`]. This owns only when it happens, which makes
//! it a timer with a pause button and lets the thing under test be a plain
//! function called synchronously.
//!
//! # Not [`crate::clock`]
//!
//! That one answers *what time it is* in a world — an anchor and a scale,
//! narrative time computed from real time, nothing counting. This one answers
//! *when the world moves*. A world can have either without the other: a paused
//! metronome does not stop the narrative clock, and slowing narrative time does
//! not change how often journeys advance.
//!
//! # Pausable, because a world that cannot be held still cannot be debugged
//!
//! Every question an operator asks of a live world — why is that Maker in the
//! lift, what did it just read, what is holding that character — is asked of a
//! world that is moving underneath the answer. Pausing is not a test affordance
//! bolted on; it is the difference between an instrument and a log.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::runtime::Handle;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;

/// How often a world moves when nothing says otherwise.
///
/// Slow enough that a Maker crossing the building takes a few seconds rather
/// than being somewhere else before an operator has read where it was, and fast
/// enough that a journey is not a coffee break. The number is the felt speed of
/// the world and is meant to be set per world once there is more than one.
pub const EVERY: Duration = Duration::from_millis(500);

/// A clock, running something on an interval until it is stopped.
///
/// An async interval task rather than an OS thread: one hosted world costs a
/// parked timer on the daemon's runtime, not a thread, and a daemon hosting
/// many worlds carries one scheduler family instead of two. The moment itself
/// is synchronous work — the world's own locks, held for well under a beat —
/// which a worker thread absorbs the way it absorbs any brief host work.
pub struct Metronome {
    stop: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    moments: Arc<AtomicU64>,
    stumbles: Arc<AtomicU64>,
    /// Held by the task for exactly the span of one `moment` call, so
    /// [`Metronome::stop`] can wait out a moment in flight — the world's lock
    /// must not be torn down under it — from sync and async callers alike.
    in_moment: Arc<Mutex<()>>,
    task: Option<JoinHandle<()>>,
}

impl Metronome {
    /// Start a clock. It runs `moment` every `every` until dropped or stopped.
    ///
    /// `on` names the runtime the clock beats on — hosting happens from the
    /// startup loader thread as well as from request handlers, and the handle
    /// works from both.
    ///
    /// The first run happens after one interval rather than immediately, so a
    /// caller has the gap to bind bodies to minds before the world moves under
    /// them.
    /// # A moment that panics must not stop the world
    ///
    /// A moment reaches the scheduler, the bodies and the map, and a panic in
    /// any of them would otherwise unwind the task and end the loop. Nothing
    /// restarts it: `stop` was never set, so the world is not stopped, it is
    /// *dead* — and the daemon around it keeps answering, so the failure looks
    /// like a cast that has gone quiet rather than like a crash. The one signal
    /// is [`Metronome::moments`] ceasing to rise, which somebody has to be
    /// watching to notice.
    ///
    /// So the moment is caught. A panicking moment is counted as a stumble,
    /// logged with its payload, and the clock goes on to the next one — the
    /// world skips a beat instead of stopping forever.
    ///
    /// **Catching does not make it fine.** A poisoned `Mutex` inside the moment
    /// stays poisoned, and the next tick to touch it panics too, so a stumble
    /// count that climbs every interval is a broken world telling you so at
    /// 2 Hz. That is the point: [`Metronome::stumbles`] is a number a health
    /// check can read, where a dead clock was silence.
    pub fn start<F>(on: &Handle, every: Duration, mut moment: F) -> Metronome
    where
        F: FnMut() + Send + 'static,
    {
        let stop = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let moments = Arc::new(AtomicU64::new(0));
        let stumbles = Arc::new(AtomicU64::new(0));
        let in_moment = Arc::new(Mutex::new(()));

        let task = {
            let (stop, paused, moments, stumbles, in_moment) = (
                stop.clone(),
                paused.clone(),
                moments.clone(),
                stumbles.clone(),
                in_moment.clone(),
            );
            on.spawn(async move {
                let mut beat = tokio::time::interval(every);
                // A beat the clock could not serve on time is skipped, not
                // crammed in: the world moves at its felt speed, and a stall
                // followed by a flurry of catch-up moments would be a world
                // lurching rather than one that missed a step.
                beat.set_missed_tick_behavior(MissedTickBehavior::Skip);
                // `interval`'s first tick completes immediately — consume it,
                // so the first moment lands after one interval as promised.
                beat.tick().await;
                loop {
                    beat.tick().await;
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    if paused.load(Ordering::Relaxed) {
                        continue;
                    }
                    let guard = in_moment.lock().unwrap();
                    // Re-checked under the guard: a `stop` that arrived while
                    // this task was between the flag and the lock must not buy
                    // the world one more moment after `stop` returned.
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    // `AssertUnwindSafe` because the closure is `FnMut` and may
                    // hold state across moments. That state can be observed
                    // half-updated after a panic — which is exactly why the
                    // stumble is counted and logged rather than swallowed.
                    let landed =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(&mut moment));
                    drop(guard);
                    match landed {
                        Ok(()) => {
                            moments.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(payload) => {
                            let n = stumbles.fetch_add(1, Ordering::Relaxed) + 1;
                            tracing::error!(
                                stumbles = n,
                                "a world moment panicked: {}; the clock continues",
                                panic_text(&*payload)
                            );
                        }
                    }
                }
            })
        };

        Metronome {
            stop,
            paused,
            moments,
            stumbles,
            in_moment,
            task: Some(task),
        }
    }

    /// Hold the world still. Moments stop happening; the clock keeps running,
    /// so resuming does not have to rebuild anything.
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Relaxed);
    }

    pub fn resume(&self) {
        self.paused.store(false, Ordering::Relaxed);
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    /// How many moments have passed. What a health check reads: a world whose
    /// count has stopped rising is a world whose clock has died.
    pub fn moments(&self) -> u64 {
        self.moments.load(Ordering::Relaxed)
    }

    /// How many moments panicked. Zero in a healthy world, and any other number
    /// is a fault that has already happened — read it beside [`Self::moments`],
    /// because a clock that is beating entirely on stumbles is advancing its
    /// count of nothing.
    pub fn stumbles(&self) -> u64 {
        self.stumbles.load(Ordering::Relaxed)
    }

    /// Stop, and wait for the moment in flight to finish.
    ///
    /// Waiting matters: a moment holds the world's lock, and returning before
    /// it is released would let a caller tear the world down underneath it.
    pub fn stop(mut self) {
        self.halt();
    }

    fn halt(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // The moment guard is the wait: a moment in flight holds it, and the
        // task re-checks `stop` under it before starting another — so once
        // this lock is acquired, no moment is running and none will start.
        // A blocking lock rather than a task join, because it is bounded by
        // one moment and works identically from sync and async callers.
        drop(self.in_moment.lock().unwrap());
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

impl Drop for Metronome {
    fn drop(&mut self) {
        // A clock that outlived its owner would keep moving a world nobody is
        // watching, which is a leak that looks like a working daemon.
        self.halt();
    }
}

impl std::fmt::Debug for Metronome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metronome")
            .field("paused", &self.is_paused())
            .field("moments", &self.moments())
            .field("stumbles", &self.stumbles())
            .finish()
    }
}

/// The message out of a panic payload, for the log line.
///
/// `panic!` carries a `&str` for a literal and a `String` once it formats, and
/// neither downcast covers a payload of some other type — so an unknown one is
/// named rather than dropped, because "a moment panicked" with nothing after it
/// is the report that sent somebody looking in the wrong place.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "a panic payload of an unknown type".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::thread;
    use std::time::Instant;

    /// Wait for something to become true, or give up. Sleeping a fixed span and
    /// asserting afterwards makes a test that fails on a loaded machine and
    /// passes on a quiet one, which is worse than no test.
    ///
    /// A thread sleep on purpose: these tests run on a multi-thread runtime, so
    /// the clock's task keeps beating on a worker while the test thread waits —
    /// which is exactly the arrangement the live daemon has.
    fn until(what: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if what() {
                return true;
            }
            thread::sleep(Duration::from_millis(2));
        }
        false
    }

    fn counter() -> (Arc<AtomicUsize>, impl FnMut() + Send + 'static) {
        let n = Arc::new(AtomicUsize::new(0));
        let mine = n.clone();
        (n, move || {
            mine.fetch_add(1, Ordering::Relaxed);
        })
    }

    /// **A panicking moment must not be the end of the world.**
    ///
    /// Without the catch this thread unwinds on the first panic and the loop is
    /// gone: `moments` freezes, `stop` is still false, and every HTTP route
    /// keeps answering — so the world reads as quiet rather than as broken. The
    /// panic here alternates so the assertion is that beats continue *through*
    /// failures rather than merely that one failure was survived.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_moment_that_panics_costs_a_beat_and_not_the_clock() {
        let beats = Arc::new(AtomicUsize::new(0));
        let mine = beats.clone();
        let clock = Metronome::start(&Handle::current(), Duration::from_millis(1), move || {
            let n = mine.fetch_add(1, Ordering::Relaxed);
            if n.is_multiple_of(2) {
                panic!("the world came apart on beat {n}");
            }
        });

        assert!(
            until(|| clock.stumbles() >= 3),
            "a panicking moment killed the clock: {} stumbles, {} moments",
            clock.stumbles(),
            clock.moments()
        );
        // The odd beats returned normally, so the counter has to be rising too —
        // a clock that only ever stumbles is not running, it is failing at 1 kHz.
        assert!(
            until(|| clock.moments() >= 3),
            "no moment completed between the panics"
        );
        clock.stop();
    }

    /// The panic's own message reaches the log line, so an operator reading
    /// "a world moment panicked" is told *which* fault they are chasing.
    #[test]
    fn a_panic_payload_is_rendered_for_the_log() {
        let literal = std::panic::catch_unwind(|| panic!("a &'static str payload"))
            .expect_err("the closure panics");
        assert_eq!(panic_text(&*literal), "a &'static str payload");

        let n = 7;
        let formatted = std::panic::catch_unwind(|| panic!("a String payload: {n}"))
            .expect_err("the closure panics");
        assert_eq!(panic_text(&*formatted), "a String payload: 7");

        let odd = std::panic::catch_unwind(|| std::panic::panic_any(9_u8))
            .expect_err("the closure panics");
        assert_eq!(panic_text(&*odd), "a panic payload of an unknown type");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_clock_runs_its_moment_until_it_is_stopped() {
        let (n, moment) = counter();
        let clock = Metronome::start(&Handle::current(), Duration::from_millis(1), moment);
        assert!(
            until(|| n.load(Ordering::Relaxed) >= 5),
            "the clock never ran"
        );

        clock.stop();
        let after = n.load(Ordering::Relaxed);
        thread::sleep(Duration::from_millis(20));
        assert_eq!(n.load(Ordering::Relaxed), after, "it kept running");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_paused_clock_holds_the_world_still_and_resumes_where_it_was() {
        let (n, moment) = counter();
        let clock = Metronome::start(&Handle::current(), Duration::from_millis(1), moment);
        assert!(until(|| n.load(Ordering::Relaxed) >= 2));

        clock.pause();
        assert!(clock.is_paused());
        // Let anything already in flight land, then take the mark.
        thread::sleep(Duration::from_millis(20));
        let held = n.load(Ordering::Relaxed);
        thread::sleep(Duration::from_millis(20));
        assert_eq!(n.load(Ordering::Relaxed), held, "a paused world moved");

        clock.resume();
        assert!(!clock.is_paused());
        assert!(
            until(|| n.load(Ordering::Relaxed) > held),
            "it did not resume"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_count_of_moments_is_what_a_health_check_reads() {
        let (_n, moment) = counter();
        let clock = Metronome::start(&Handle::current(), Duration::from_millis(1), moment);
        assert!(until(|| clock.moments() >= 3));

        clock.pause();
        thread::sleep(Duration::from_millis(20));
        let held = clock.moments();
        thread::sleep(Duration::from_millis(20));
        assert_eq!(clock.moments(), held, "a paused clock counted moments");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn nothing_happens_before_the_first_interval() {
        // The gap a caller uses to bind bodies to minds before the world moves
        // under them.
        let (n, moment) = counter();
        let _clock = Metronome::start(&Handle::current(), Duration::from_millis(200), moment);
        assert_eq!(n.load(Ordering::Relaxed), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_clock_stops_moving_the_world() {
        // A clock that outlived its owner is a leak that looks like a working
        // daemon: the world keeps moving and nobody is reading it.
        let (n, moment) = counter();
        {
            let _clock = Metronome::start(&Handle::current(), Duration::from_millis(1), moment);
            assert!(until(|| n.load(Ordering::Relaxed) >= 3));
        }
        let after = n.load(Ordering::Relaxed);
        thread::sleep(Duration::from_millis(20));
        assert_eq!(n.load(Ordering::Relaxed), after, "it outlived its owner");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stopping_waits_for_the_moment_in_flight() {
        // A moment holds the world's lock. Returning before it is released
        // would let a caller tear the world down underneath it.
        let running = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let clock = {
            let (running, finished) = (running.clone(), finished.clone());
            Metronome::start(&Handle::current(), Duration::from_millis(1), move || {
                running.store(true, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(50));
                finished.store(true, Ordering::Relaxed);
            })
        };
        assert!(until(|| running.load(Ordering::Relaxed)));

        clock.stop();
        assert!(
            finished.load(Ordering::Relaxed),
            "stop returned mid-moment, with the lock still held"
        );
    }
}
