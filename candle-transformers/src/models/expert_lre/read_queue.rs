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

    /// The next job — demand first — blocking until there is one. `None` once
    /// the queue is closed and drained.
    pub(crate) fn pop(&self) -> Option<T> {
        let mut q = self.queues.lock().ok()?;
        loop {
            if let Some(job) = q.demand.pop_front().or_else(|| q.spec.pop_front()) {
                return Some(job);
            }
            if q.closed {
                return None;
            }
            q = self.ready.wait(q).ok()?;
        }
    }

    /// No more jobs: readers drain what is queued, then [`Self::pop`] returns
    /// `None`.
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

    #[test]
    fn a_demand_job_is_taken_before_every_queued_speculative_one() {
        let q = ReadQueue::new();
        assert!(q.push(1, Priority::Speculative));
        assert!(q.push(2, Priority::Speculative));
        assert!(q.push(10, Priority::Demand));
        assert!(q.push(11, Priority::Demand));
        let order: Vec<i32> = (0..4).map(|_| q.pop().unwrap()).collect();
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
        let order: Vec<i32> = (0..4).map(|_| q.pop().unwrap()).collect();
        assert_eq!(order, vec![10, 2, 1, 3]);
    }

    #[test]
    fn close_drains_then_ends_and_refuses_new_jobs() {
        let q = ReadQueue::new();
        q.push(5, Priority::Speculative);
        q.close();
        assert!(!q.push(6, Priority::Demand));
        assert_eq!(q.pop(), Some(5));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn a_blocked_reader_wakes_for_a_job_and_for_close() {
        let q = Arc::new(ReadQueue::new());
        let reader = {
            let q = q.clone();
            std::thread::spawn(move || {
                let first = q.pop();
                let second = q.pop();
                (first, second)
            })
        };
        q.push(42, Priority::Demand);
        q.close();
        assert_eq!(reader.join().unwrap(), (Some(42), None));
    }
}
