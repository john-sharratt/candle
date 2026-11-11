//! An idle scheduler closes its wave windows on time (`idle_wait`), so an idle
//! card answers every slot of `/v1/telemetry` rather than dropping out of it.

use super::*;
use flume::RecvTimeoutError;
use std::time::{Duration, Instant};

/// [`make_test_scheduler`] with its wave window opened now, so a test's waits
/// are measured from the window's start rather than from however long the
/// session took to build.
fn idle_scheduler() -> (Scheduler, Sender<SchedulerRequest>) {
    let (mut scheduler, tx) = make_test_scheduler();
    scheduler.wave_stats.window_start = Instant::now();
    (scheduler, tx)
}

/// **A wait longer than the window closes the window inside it.** Waiting
/// 2.6 s with nothing arriving closes the window that was due at 2 s, so the
/// window standing at the end is the 0.6 s after it, and holds only that
/// idle time — the 2 s before it went out with the closed one.
#[test]
fn an_idle_wait_closes_the_window_when_it_falls_due() {
    let (mut scheduler, _tx) = idle_scheduler();
    let started = Instant::now();
    let got = scheduler.recv_idle(Duration::from_millis(2_600));
    assert!(
        matches!(got, Err(RecvTimeoutError::Timeout)),
        "no request came"
    );
    assert!(
        started.elapsed() >= Duration::from_millis(2_600),
        "the whole wait ran"
    );
    assert!(
        scheduler.wave_stats.until_due() > Duration::ZERO,
        "a window closed during the wait and a fresh one stands"
    );
    assert!(
        scheduler.wave_stats.idle_ms < 2_000,
        "the standing window holds only the idle after the close: {} ms",
        scheduler.wave_stats.idle_ms
    );
}

/// **A wait shorter than the window leaves it open**, holding the idle time
/// the wait spent.
#[test]
fn an_idle_wait_inside_the_window_leaves_it_open() {
    let (mut scheduler, _tx) = idle_scheduler();
    let got = scheduler.recv_idle(Duration::from_millis(300));
    assert!(matches!(got, Err(RecvTimeoutError::Timeout)));
    assert!(
        scheduler.wave_stats.idle_ms >= 300,
        "{} ms",
        scheduler.wave_stats.idle_ms
    );
}

/// **A closed channel ends the wait at once**, whatever is left of it.
#[test]
fn a_closed_channel_ends_the_idle_wait() {
    let (mut scheduler, tx) = idle_scheduler();
    drop(tx);
    let started = Instant::now();
    let got = scheduler.recv_idle(Duration::from_secs(300));
    assert!(matches!(got, Err(RecvTimeoutError::Disconnected)));
    assert!(started.elapsed() < Duration::from_secs(1));
}
