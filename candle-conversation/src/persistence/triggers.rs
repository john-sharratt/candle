//! Every attached substrate's persistence trigger, read as one.
//!
//! An engine runs one persistence thread per attached substrate
//! (`docs/shared_model_host.md` §3.2), and the scheduler — which serves them
//! all — wakes them after a seal, waits on their hot→warm drain under VRAM
//! pressure, and throttles admission off their backlog gauges. VRAM is one pool
//! whichever substrate's turns fill it, so a wake goes to every thread, a flush
//! waits for every thread, and each gauge is the sum over threads.

use std::time::{Duration, Instant};

use super::thread::PersistenceTrigger;

/// The id of the attachment an engine is built with; every later one takes the
/// next.
pub const FIRST_ATTACHMENT: u64 = 0;

/// The persistence triggers of every substrate attached to one engine, keyed by
/// the attachment's id.
#[derive(Clone)]
pub struct PersistenceTriggers {
    members: Vec<(u64, PersistenceTrigger)>,
}

impl PersistenceTriggers {
    /// The triggers of an engine whose first attachment is `id`.
    pub fn new(id: u64, trigger: PersistenceTrigger) -> Self {
        Self {
            members: vec![(id, trigger)],
        }
    }

    /// Add attachment `id`'s trigger.
    pub fn attach(&mut self, id: u64, trigger: PersistenceTrigger) {
        self.members.retain(|(m, _)| *m != id);
        self.members.push((id, trigger));
    }

    /// Remove attachment `id`'s trigger. Returns whether it was attached.
    pub fn detach(&mut self, id: u64) -> bool {
        let before = self.members.len();
        self.members.retain(|(m, _)| *m != id);
        self.members.len() != before
    }

    /// How many substrates are attached.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Wake every persistence thread.
    pub fn fire(&self) {
        for (_, t) in &self.members {
            t.fire();
        }
    }

    /// Block until every persistence thread has completed a pass that drains
    /// its pending hot→warm migration, within `timeout` overall. `true` only
    /// when every one acknowledged.
    pub fn flush_blocking(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut all = true;
        for (_, t) in &self.members {
            let left = deadline.saturating_duration_since(Instant::now());
            all &= t.flush_blocking(left);
        }
        all
    }

    /// The hot→warm drain backlog across every substrate, in bytes.
    pub fn pending_warm_bytes(&self) -> u64 {
        self.members
            .iter()
            .map(|(_, t)| t.pending_warm_bytes())
            .sum()
    }

    /// Pinned bytes every drain skipped, summed.
    pub fn pinned_undrainable_bytes(&self) -> u64 {
        self.members
            .iter()
            .map(|(_, t)| t.pinned_undrainable_bytes())
            .sum()
    }

    /// Residences holding a warm copy, summed.
    pub fn warm_resident_count(&self) -> u64 {
        self.members
            .iter()
            .map(|(_, t)| t.warm_resident_count())
            .sum()
    }

    /// Bytes held by warm copies, summed.
    pub fn warm_resident_bytes(&self) -> u64 {
        self.members
            .iter()
            .map(|(_, t)| t.warm_resident_bytes())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attach_and_detach_track_members_by_id() {
        let mut t = PersistenceTriggers::new(0, PersistenceTrigger::noop());
        t.attach(1, PersistenceTrigger::noop());
        assert_eq!(t.len(), 2);
        // Re-attaching an id replaces it rather than doubling it.
        t.attach(1, PersistenceTrigger::noop());
        assert_eq!(t.len(), 2);
        assert!(t.detach(0));
        assert!(!t.detach(0), "a detached id is gone");
        assert_eq!(t.len(), 1);
    }

    /// A flush against threads that are not running acknowledges nothing, and
    /// says so rather than waiting out its whole timeout per member.
    #[test]
    fn a_flush_no_thread_acknowledges_reports_false() {
        let mut t = PersistenceTriggers::new(0, PersistenceTrigger::noop());
        t.attach(1, PersistenceTrigger::noop());
        let started = Instant::now();
        assert!(!t.flush_blocking(Duration::from_millis(50)));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn idle_gauges_sum_to_zero() {
        let mut t = PersistenceTriggers::new(0, PersistenceTrigger::noop());
        t.attach(1, PersistenceTrigger::noop());
        assert_eq!(t.pending_warm_bytes(), 0);
        assert_eq!(t.pinned_undrainable_bytes(), 0);
        assert_eq!(t.warm_resident_count(), 0);
        assert_eq!(t.warm_resident_bytes(), 0);
    }
}
