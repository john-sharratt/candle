//! The wait's cadence, under paused time with fake looks.

use std::cell::RefCell;
use std::future::ready;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::watch;

use super::*;

fn standing(ended: bool) -> Standing {
    Standing {
        ended,
        line: "job: running".into(),
        detail: String::new(),
        total: 0,
        size: 0,
        marks: Vec::new(),
    }
}

/// Milliseconds from `start`.
fn ms(start: Instant) -> u128 {
    Instant::now().duration_since(start).as_millis()
}

/// **A task that ends on a bump answers at once**: nothing waits for the
/// two-second cadence.
#[tokio::test(start_paused = true)]
async fn a_task_that_ends_on_a_bump_answers_at_once() {
    let start = Instant::now();
    let done = Arc::new(AtomicBool::new(false));
    let (tx, rx) = watch::channel(0u64);
    let ender = Arc::clone(&done);
    tokio::spawn(async move {
        ender.store(true, Ordering::SeqCst);
        tx.send(1).unwrap();
    });
    let got = wait_on(
        1,
        |_| true,
        start + Duration::from_secs(60),
        |_| ready(Ok(standing(done.load(Ordering::SeqCst)))),
        Some(rx),
        |_, _| {},
    )
    .await;
    assert_eq!(ms(start), 0);
    assert!(got[0].as_ref().unwrap().ended);
}

/// **Wakes recheck tasks only**: three bumps within a second make one look
/// at the build beside the task, the one at the start.
#[tokio::test(start_paused = true)]
async fn bumps_do_not_hurry_the_other_jobs() {
    let start = Instant::now();
    let remote_looks = RefCell::new(0);
    let (tx, rx) = watch::channel(0u64);
    tokio::spawn(async move {
        for n in 1..=3 {
            sleep(Duration::from_millis(200)).await;
            tx.send(n).unwrap();
        }
        sleep(Duration::from_secs(10)).await;
    });
    wait_on(
        2,
        |i| i == 0,
        start + Duration::from_millis(1_500),
        |i| {
            if i == 1 {
                *remote_looks.borrow_mut() += 1;
            }
            ready(Ok(standing(false)))
        },
        Some(rx),
        |_, _| {},
    )
    .await;
    assert_eq!(*remote_looks.borrow(), 1);
}

/// **The other jobs keep their own cadence**: over five seconds, at 0, 2
/// and 4 s.
#[tokio::test(start_paused = true)]
async fn the_other_jobs_are_looked_at_every_two_seconds() {
    let start = Instant::now();
    let at = RefCell::new(Vec::new());
    wait_on(
        1,
        |_| false,
        start + Duration::from_secs(5),
        |_| {
            at.borrow_mut().push(ms(start));
            ready(Ok(standing(false)))
        },
        None,
        |_, _| {},
    )
    .await;
    assert_eq!(*at.borrow(), [0, 2_000, 4_000]);
}

/// **A wake whose sender is gone does not spin**: the task is looked at once
/// every two seconds, and once more at the deadline.
#[tokio::test(start_paused = true)]
async fn a_dropped_sender_makes_one_look_per_two_seconds() {
    let start = Instant::now();
    let at = RefCell::new(Vec::new());
    let (tx, rx) = watch::channel(0u64);
    drop(tx);
    wait_on(
        1,
        |_| true,
        start + Duration::from_secs(5),
        |_| {
            at.borrow_mut().push(ms(start));
            ready(Ok(standing(false)))
        },
        Some(rx),
        |_, _| {},
    )
    .await;
    assert_eq!(*at.borrow(), [0, 2_000, 4_000, 5_000]);
}
