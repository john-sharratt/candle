//! Waiting on a sandbox job from a tool call.
//!
//! A tool call is synchronous and a job is a future that spawns tasks, so the
//! call drives the job to its end — on a thread of its own with a runtime of
//! its own, never on the caller's. Blocking on the caller's runtime works only
//! from a thread that may block and panics from an async task; a thread of its
//! own never panics that way, whatever calls it.
//!
//! The calling thread still waits, for as long as the job runs — up to its
//! timeout. The daemon's tool rounds run on Tokio's blocking pool, where a
//! wait is what the pool is for; called from an async task instead, the wait
//! holds that task's worker thread, which is a cost, not a fault.

use std::future::Future;
use std::panic::resume_unwind;
use std::thread;

use tokio::runtime::Builder;

/// Drive `future` to its end and return what it gives.
pub(super) fn on<F>(future: F) -> F::Output
where
    F: Future + Send,
    F::Output: Send,
{
    thread::scope(|scope| {
        scope
            .spawn(|| {
                Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("a current-thread runtime starts")
                    .block_on(future)
            })
            .join()
            .unwrap_or_else(|panic| resume_unwind(panic))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A job is driven from an async task as from a plain thread** — where
    /// blocking on the caller's own runtime would panic.
    #[tokio::test]
    async fn a_job_is_driven_from_inside_an_async_task() {
        let spawned = on(async { tokio::spawn(async { 41 + 1 }).await.unwrap() });
        assert_eq!(spawned, 42);
    }

    #[test]
    fn a_job_is_driven_from_a_plain_thread() {
        assert_eq!(on(async { "done" }), "done");
    }
}
