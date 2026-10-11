//! The stager's read queue: demand reads before speculative ones.
//!
//! A demand read is a cold expert a GEMM worker is spinning on right now; a
//! speculative read is a predicted expert of a layer the wave has not reached.
//! Both go to the same reader threads, so a single FIFO would put a demand read
//! behind every speculative read queued before it — and the speculative reads
//! are only worth issuing if they never delay a demand. Readers therefore take
//! the oldest demand job while there is one, and a speculative job only when no
//! demand job waits. A speculative read the wave catches up with before it has
//! started ([`ReadQueue::promote`]) moves to the back of the demand queue, since
//! a worker is now waiting on it.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

/// Which queue a job goes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Priority {
    Demand,
    Speculative,
}

struct Queues<T> {
    demand: VecDeque<T>,
    spec: VecDeque<T>,
    closed: bool,
}

/// A two-level job queue shared by the reader threads.
pub(crate) struct ReadQueue<T> {
    queues: Mutex<Queues<T>>,
    ready: Condvar,
}

impl<T> ReadQueue<T> {
    pub(crate) fn new() -> Self {
        Self {
            queues: Mutex::new(Queues {
                demand: VecDeque::new(),
                spec: VecDeque::new(),
                closed: false,
            }),
            ready: Condvar::new(),
        }
    }

    /// Queue `job`. False once the queue is closed.
    pub(crate) fn push(&self, job: T, priority: Priority) -> bool {
        let Ok(mut q) = self.queues.lock() else {
            return false;
        };
        if q.closed {
            return false;
        }
        match priority {
            Priority::Demand => q.demand.push_back(job),
            Priority::Speculative => q.spec.push_back(job),
        }
        drop(q);
        self.ready.notify_one();
        true
    }

    /// Queue `jobs` together, in order, and only then wake the readers — so a
    /// reader that wakes finds the whole batch at the front and can take a run
    /// of it ([`Self::pop_run`]), where pushing one at a time hands each job to
    /// whichever reader was idle the moment it arrived. False once the queue is
    /// closed.
    pub(crate) fn push_all(&self, jobs: Vec<T>, priority: Priority) -> bool {
        let Ok(mut q) = self.queues.lock() else {
            return false;
        };
        if q.closed {
            return false;
        }
        let lane = match priority {
            Priority::Demand => &mut q.demand,
            Priority::Speculative => &mut q.spec,
        };
        lane.extend(jobs);
        drop(q);
        self.ready.notify_all();
        true
    }

    /// Move the first queued speculative job `is` matches to the demand queue.
    /// False when none is queued — it has started, or was never speculative.
    pub(crate) fn promote(&self, is: impl Fn(&T) -> bool) -> bool {
        let Ok(mut q) = self.queues.lock() else {
            return false;
        };
        let Some(i) = q.spec.iter().position(is) else {
            return false;
        };
        let job = q.spec.remove(i).expect("the position was just found");
        q.demand.push_back(job);
        true
    }

    /// The next job — demand first — blocking until there is one, then — from
    /// the queue that job came from — every job
    /// waiting at its front that `follows` the last one taken, up to `max` in
    /// all. A reader serves the run in one request: the pack's records are laid
    /// out in order, so jobs for adjacent records become one read. Only the
    /// front is taken, never a job further back, so no job is served ahead of
    /// one queued before it, and a run never takes a speculative job while a
    /// demand one waits. `None` once the queue is closed and drained.
    pub(crate) fn pop_run(&self, max: usize, follows: impl Fn(&T, &T) -> bool) -> Option<Vec<T>> {
        let mut q = self.queues.lock().ok()?;
        loop {
            let from_demand = !q.demand.is_empty();
            let lane = if from_demand {
                &mut q.demand
            } else {
                &mut q.spec
            };
            if let Some(job) = lane.pop_front() {
                let mut run = vec![job];
                while run.len() < max {
                    let last = run.last().expect("the run holds its first job");
                    match lane.front() {
                        Some(next) if follows(last, next) => {
                            run.push(lane.pop_front().expect("the front was just read"));
                        }
                        _ => break,
                    }
                }
                return Some(run);
            }
            if q.closed {
                return None;
            }
            q = self.ready.wait(q).ok()?;
        }
    }

    /// No more jobs: readers drain what is queued, then [`Self::pop_run`]
    /// returns `None`.
    pub(crate) fn close(&self) {
        if let Ok(mut q) = self.queues.lock() {
            q.closed = true;
        }
        self.ready.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// One job at a time, as a reader whose runs never extend.
    fn pop<T>(q: &ReadQueue<T>) -> Option<T> {
        q.pop_run(1, |_, _| false)?.pop()
    }

    #[test]
    fn a_demand_job_is_taken_before_every_queued_speculative_one() {
        let q = ReadQueue::new();
        assert!(q.push(1, Priority::Speculative));
        assert!(q.push(2, Priority::Speculative));
        assert!(q.push(10, Priority::Demand));
        assert!(q.push(11, Priority::Demand));
        let order: Vec<i32> = (0..4).map(|_| pop(&q).unwrap()).collect();
        assert_eq!(order, vec![10, 11, 1, 2]);
    }

    #[test]
    fn a_promoted_job_joins_the_back_of_the_demand_queue() {
        let q = ReadQueue::new();
        q.push(1, Priority::Speculative);
        q.push(2, Priority::Speculative);
        q.push(3, Priority::Speculative);
        q.push(10, Priority::Demand);
        assert!(q.promote(|&j| j == 2));
        assert!(!q.promote(|&j| j == 10), "a demand job is not speculative");
        assert!(!q.promote(|&j| j == 7), "nothing queued matches");
        let order: Vec<i32> = (0..4).map(|_| pop(&q).unwrap()).collect();
        assert_eq!(order, vec![10, 2, 1, 3]);
    }

    /// A run is the front job and the consecutive ones queued behind it in its
    /// own lane — capped, stopped at the first gap, and never borrowing from
    /// the speculative lane while demand is queued.
    #[test]
    fn a_run_takes_the_consecutive_front_of_one_lane() {
        let q = ReadQueue::new();
        for j in [10, 11, 12, 14, 15] {
            q.push(j, Priority::Demand);
        }
        q.push(16, Priority::Speculative);
        q.push(17, Priority::Speculative);
        let next = |a: &i32, b: &i32| *b == *a + 1;
        assert_eq!(
            q.pop_run(8, next),
            Some(vec![10, 11, 12]),
            "stops at the gap"
        );
        assert_eq!(
            q.pop_run(8, next),
            Some(vec![14, 15]),
            "not into the spec lane"
        );
        assert_eq!(q.pop_run(1, next), Some(vec![16]), "capped");
        assert_eq!(q.pop_run(8, next), Some(vec![17]));
        q.close();
        assert_eq!(q.pop_run(8, next), None);
    }

    /// A batch lands in its lane in order, behind what was already queued, and
    /// a reader takes it as one run.
    #[test]
    fn a_batch_is_queued_in_order_and_taken_as_a_run() {
        let q = ReadQueue::new();
        q.push(1, Priority::Demand);
        assert!(q.push_all(vec![5, 6, 7], Priority::Demand));
        let next = |a: &i32, b: &i32| *b == *a + 1;
        assert_eq!(q.pop_run(8, next), Some(vec![1]));
        assert_eq!(q.pop_run(8, next), Some(vec![5, 6, 7]));
        q.close();
        assert!(!q.push_all(vec![9], Priority::Demand));
    }

    #[test]
    fn close_drains_then_ends_and_refuses_new_jobs() {
        let q = ReadQueue::new();
        q.push(5, Priority::Speculative);
        q.close();
        assert!(!q.push(6, Priority::Demand));
        assert_eq!(pop(&q), Some(5));
        assert_eq!(pop(&q), None);
    }

    #[test]
    fn a_blocked_reader_wakes_for_a_job_and_for_close() {
        let q = Arc::new(ReadQueue::new());
        let reader = {
            let q = q.clone();
            std::thread::spawn(move || {
                let first = pop(&q);
                let second = pop(&q);
                (first, second)
            })
        };
        q.push(42, Priority::Demand);
        q.close();
        assert_eq!(reader.join().unwrap(), (Some(42), None));
    }
}
