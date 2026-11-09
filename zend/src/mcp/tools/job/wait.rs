//! Waiting on jobs: a task's change wakes the wait at once, through the MCP
//! state's change signal; builds, deploys and maintenance jobs are looked at
//! on their own cadence ([`LOOK_EVERY`]), together.
//!
//! The loop is a function of its looks and its wake, so it is tested with
//! fakes under paused time.

use std::future::Future;
use std::time::Duration;

use futures::future::join_all;
use tokio::sync::watch::Receiver;
use tokio::time::{sleep, timeout, Instant};

use super::{Standing, LOOK_EVERY};

/// Wait `left` at most, or until `wake` says something changed. A wake whose
/// sender is gone is dropped, and the wait sleeps the whole of `left`, so it
/// cannot spin.
pub async fn pause(wake: &mut Option<Receiver<u64>>, left: Duration) {
    let Some(rx) = wake.as_mut() else {
        sleep(left).await;
        return;
    };
    match timeout(left, rx.changed()).await {
        Ok(Ok(())) | Err(_) => {}
        Ok(Err(_)) => {
            *wake = None;
            sleep(left).await;
        }
    }
}

/// Look at `count` jobs until any has ended or `until` passes, and answer
/// where each stands. Job `i` is a task when `is_task(i)`; `look(i)` says
/// where it stands. Tasks are looked at again on every wake; the others on
/// their own cadence, from the first look. `said` hears each standing after
/// every look that does not end the wait.
pub async fn wait_on<F, Fut>(
    count: usize,
    is_task: impl Fn(usize) -> bool,
    until: Instant,
    look: F,
    mut wake: Option<Receiver<u64>>,
    mut said: impl FnMut(usize, &Standing),
) -> Vec<Result<Standing, String>>
where
    F: Fn(usize) -> Fut,
    Fut: Future<Output = Result<Standing, String>>,
{
    let tasks: Vec<usize> = (0..count).filter(|i| is_task(*i)).collect();
    let remote: Vec<usize> = (0..count).filter(|i| !is_task(*i)).collect();
    if tasks.is_empty() {
        wake = None;
    }
    let mut now: Vec<Option<Result<Standing, String>>> = (0..count).map(|_| None).collect();
    let mut next_remote = Instant::now();
    loop {
        for (i, s) in tasks
            .iter()
            .zip(join_all(tasks.iter().map(|&i| look(i))).await)
        {
            now[*i] = Some(s);
        }
        let at = Instant::now();
        if !remote.is_empty() && at >= next_remote {
            for (i, s) in remote
                .iter()
                .zip(join_all(remote.iter().map(|&i| look(i))).await)
            {
                now[*i] = Some(s);
            }
            next_remote = at + LOOK_EVERY;
        }
        let ended = now
            .iter()
            .flatten()
            .any(|s| s.as_ref().map_or(true, |s| s.ended));
        if ended || at >= until {
            break;
        }
        for (i, s) in now.iter().enumerate() {
            if let Some(Ok(s)) = s {
                said(i, s);
            }
        }
        let to = if remote.is_empty() {
            at + LOOK_EVERY
        } else {
            next_remote
        };
        pause(&mut wake, to.min(until).saturating_duration_since(at)).await;
    }
    now.into_iter()
        .map(|s| s.unwrap_or_else(|| Err("it was never looked at".to_string())))
        .collect()
}

#[cfg(test)]
mod tests;
