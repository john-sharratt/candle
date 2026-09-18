//! The thread pool the score-normalization warm-ups run on.
//!
//! # Why the warm-ups do not share rayon's global pool
//!
//! A warm-up replays stored turns as probes against every belief group and
//! collection to learn their hit levels — up to 512 dialogue turns, or every
//! ingested file, each scored on the CPU with `par_iter`. That is bulk work,
//! seconds to minutes of it, and it runs while the daemon is serving: after
//! `ready`, on the first dialogue query, and again after every file-watcher
//! reconcile.
//!
//! The live turn-boundary belief scan needs rayon too. Its kernel runs on the
//! GPU, but `needle_tally_segments` tallies the kernel's output on the host with
//! `into_par_iter` — and a `par_iter` issued from outside a pool queues behind
//! whatever that pool is already busy with, and blocks until it drains. With a
//! warm-up on the global pool, a three-window scan whose kernel finished in
//! microseconds waited 8 to 18 seconds for a worker to tally it: every tool
//! round of a conversation paid that gap, with the GPU idle and nothing logged,
//! because nothing was running on the thread that was waiting.
//!
//! Work run through [`run`] executes on this pool, and every `par_iter` inside
//! it stays on this pool, so the global pool's workers stay free for live
//! turns. The pool is sized to a quarter of the machine so a warm-up still
//! finishes promptly while leaving most cores to the work a user is waiting on.

use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Threads for the warm pool: a quarter of the machine, never fewer than two.
fn thread_count() -> usize {
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    (cores / 4).max(2)
}

fn pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(thread_count())
            .thread_name(|i| format!("norm-warm-{i}"))
            .build()
            .expect("the normalization warm pool builds")
    })
}

/// Run a normalization warm-up on the warm pool, blocking the caller until it
/// returns. Every `par_iter` inside `f` runs on the warm pool — see the module
/// docs for why that matters.
pub fn run<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    pool().install(f)
}

#[cfg(test)]
mod tests {
    use std::sync::{mpsc, Mutex};

    use rayon::prelude::*;

    use super::*;

    /// **Parallel work started inside a warm-up stays on the warm pool.** Every
    /// worker a `par_iter` inside [`run`] lands on belongs to a pool of the
    /// warm pool's size — so none of it can occupy the global pool's workers,
    /// which is the whole guarantee. Deterministic: it asserts where the work
    /// ran, not how long anything took.
    #[test]
    fn parallel_work_inside_a_warm_up_stays_on_the_warm_pool() {
        let sizes: Vec<usize> = run(|| {
            (0..256)
                .into_par_iter()
                .map(|_| {
                    assert!(
                        rayon::current_thread_index().is_some(),
                        "par_iter work ran outside any pool"
                    );
                    rayon::current_num_threads()
                })
                .collect()
        });
        assert!(sizes.iter().all(|&n| n == thread_count()), "{sizes:?}");
    }

    /// **A warm-up that occupies every warm worker does not hold up parallel
    /// work issued outside it.** The warm workers are parked on a channel until
    /// the outside `par_iter` has finished; if that work queued behind the
    /// warm-up, it would never finish and the channel would never close. None
    /// of it runs on a warm worker either.
    #[test]
    fn a_busy_warm_up_does_not_hold_up_parallel_work_outside_it() {
        let (release, parked) = mpsc::channel::<()>();
        let parked = Mutex::new(parked);
        let (started_tx, started) = mpsc::channel::<()>();
        let started_tx = Mutex::new(started_tx);
        std::thread::scope(|s| {
            let warm = s.spawn(|| {
                run(|| {
                    (0..thread_count()).into_par_iter().for_each(|_| {
                        started_tx.lock().unwrap().send(()).unwrap();
                        // Holding the lock across `recv` keeps the other warm
                        // workers blocked on the lock: every one stays busy.
                        let _ = parked.lock().unwrap().recv();
                    })
                })
            });
            started.recv().unwrap();

            let names: Vec<String> = (0..256)
                .into_par_iter()
                .map(|_| std::thread::current().name().unwrap_or("").to_owned())
                .collect();
            assert!(
                names.iter().all(|n| !n.starts_with("norm-warm-")),
                "outside work ran on a warm worker: {names:?}"
            );

            drop(release);
            warm.join().unwrap();
        });
    }

    #[test]
    fn the_warm_pool_leaves_most_of_the_machine_free() {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        assert!(thread_count() >= 2);
        assert!(
            thread_count() <= cores.max(8) / 4 + 1,
            "{} warm threads on {cores} cores",
            thread_count()
        );
    }
}
