//! The jobs a server keeps: the most recent [`KEPT_JOBS`], oldest let go
//! first, and never one still running.
//!
//! A job let go takes its log with it — the caller deletes the files
//! [`Jobs::insert`] returns — and asking after it is [`JobNotFound`] from then
//! on, exactly as for an id that never ran.
//!
//! [`JobNotFound`]: super::JobNotFound

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

use tokio::sync::watch;

use super::job_id::JobId;
use super::log::Written;
use super::status::{JobInfo, JobStatus};

/// How many jobs a server keeps.
pub const KEPT_JOBS: usize = 1000;

/// One kept job.
struct Kept {
    status: JobStatus,
    written: watch::Receiver<Written>,
    log: PathBuf,
}

/// The kept jobs, in the order they started.
pub(super) struct Jobs {
    capacity: usize,
    order: VecDeque<JobId>,
    kept: HashMap<JobId, Kept>,
}

impl Jobs {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: VecDeque::new(),
            kept: HashMap::new(),
        }
    }

    /// Keep `id`, queued, and let go of the oldest finished jobs past the
    /// capacity. Returns the logs of those let go.
    pub(super) fn insert(
        &mut self,
        id: JobId,
        log: PathBuf,
        written: watch::Receiver<Written>,
    ) -> Vec<PathBuf> {
        self.order.push_back(id.clone());
        self.kept.insert(
            id,
            Kept {
                status: JobStatus::Queued,
                written,
                log,
            },
        );
        let mut gone = Vec::new();
        let mut at = 0;
        while self.kept.len() > self.capacity && at < self.order.len() {
            let finished = self
                .kept
                .get(&self.order[at])
                .is_some_and(|k| k.status.is_finished());
            if !finished {
                at += 1;
                continue;
            }
            let id = self.order.remove(at).expect("in range");
            if let Some(kept) = self.kept.remove(&id) {
                gone.push(kept.log);
            }
        }
        gone
    }

    /// Where `id` stands and what its log holds.
    pub(super) fn info(&self, id: &JobId) -> Option<JobInfo> {
        let kept = self.kept.get(id)?;
        let written = *kept.written.borrow();
        Some(JobInfo {
            status: kept.status.clone(),
            lines: written.lines,
            bytes: written.bytes,
            log: kept.log.clone(),
        })
    }

    /// `id` has the checkout.
    pub(super) fn running(&mut self, id: &JobId) {
        self.advance(id, JobStatus::Running);
    }

    /// `id` has ended as `status`. A job already ended keeps how it ended:
    /// one cancelled is not reported as having run on.
    pub(super) fn ended(&mut self, id: &JobId, status: JobStatus) {
        self.advance(id, status);
    }

    fn advance(&mut self, id: &JobId, status: JobStatus) {
        if let Some(kept) = self.kept.get_mut(id) {
            if !kept.status.is_finished() {
                kept.status = status;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keep(jobs: &mut Jobs, n: u64) -> (JobId, Vec<PathBuf>) {
        let id = JobId::parse(&format!(
            "AAAAAAAAAA{}",
            ["A", "E", "I", "M", "Q"][n as usize]
        ))
        .unwrap();
        let (_, written) = watch::channel(Written::default());
        let gone = jobs.insert(id.clone(), PathBuf::from(format!("{n}.log")), written);
        (id, gone)
    }

    /// **Past the capacity the oldest finished job goes, log and all** — a
    /// job still running is passed over, however old.
    #[test]
    fn the_oldest_finished_job_goes_first() {
        let mut jobs = Jobs::new(2);
        let (a, _) = keep(&mut jobs, 0);
        let (b, _) = keep(&mut jobs, 1);
        jobs.running(&a);
        jobs.ended(&b, JobStatus::Exited { code: Some(0) });
        let (c, gone) = keep(&mut jobs, 2);
        assert_eq!(gone, [PathBuf::from("1.log")], "b went; a is still running");
        assert!(jobs.info(&b).is_none());
        assert_eq!(jobs.info(&a).unwrap().status, JobStatus::Running);
        assert_eq!(jobs.info(&c).unwrap().status, JobStatus::Queued);

        jobs.ended(&a, JobStatus::Exited { code: Some(1) });
        let (_, gone) = keep(&mut jobs, 3);
        assert_eq!(gone, [PathBuf::from("0.log")]);
        assert!(jobs.info(&a).is_none());
    }

    /// **How a job ended is kept**: a cancelled job is not later reported as
    /// having exited, and an ended job never goes back to running.
    #[test]
    fn an_ended_job_keeps_how_it_ended() {
        let mut jobs = Jobs::new(10);
        let (a, _) = keep(&mut jobs, 0);
        jobs.running(&a);
        jobs.ended(&a, JobStatus::Cancelled);
        jobs.ended(&a, JobStatus::Exited { code: Some(0) });
        jobs.running(&a);
        assert_eq!(jobs.info(&a).unwrap().status, JobStatus::Cancelled);
    }

    /// A query reads the log's count as it stands.
    #[test]
    fn a_query_reads_the_logs_count() {
        let mut jobs = Jobs::new(10);
        let id = JobId::parse("AAAAAAAAAAA").unwrap();
        let (tx, written) = watch::channel(Written::default());
        jobs.insert(id.clone(), PathBuf::from("x.log"), written);
        tx.send_modify(|w| {
            w.bytes = 12;
            w.lines = 3;
        });
        let info = jobs.info(&id).unwrap();
        assert_eq!((info.lines, info.bytes), (3, 12));
        assert_eq!(info.log, PathBuf::from("x.log"));
    }
}
