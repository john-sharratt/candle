//! What `keep` does on one look at a service. No disk, no process, no bytes:
//! the caller holds the last-ready path and, when a swap is in progress, the
//! plan.

use std::time::Duration;

/// One look at a watched process.
#[derive(Debug, Clone, Copy)]
pub struct Look {
    /// A process with this service's executable name is running.
    pub alive: bool,
    /// Its ready check answered.
    pub ready: bool,
    /// A last-ready launch is on disk.
    pub has_last: bool,
    /// An upgrade plan is on disk.
    pub pending: bool,
    /// How long this look has been waiting on a process that is not ready,
    /// or on a start that has not shown a process yet.
    pub waited: Duration,
    /// How long a replacement may take to become ready.
    pub limit: Duration,
    /// A start was issued a moment ago, so a missing process is that start
    /// and not another one.
    pub just_started: bool,
}

/// What the caller does with the look.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Leave it. A replacement is still inside its time, or nothing is known.
    Hold,
    /// The running process is ready. Record its executable as last-ready.
    Remember,
    /// The process is gone. Start the last-ready executable.
    Start,
    /// A replacement did not become ready. Restore the previous executable
    /// and start that.
    Recover,
}

/// The action for `look`. A pending replacement is never started as the last
/// ready path: that path is the new bytes until `Recover` puts the old ones
/// back. A process that is simply gone is started from the last-ready path.
pub fn action(look: Look) -> Action {
    if look.pending {
        if look.ready {
            return Action::Remember;
        }
        if look.waited >= look.limit {
            return Action::Recover;
        }
        return Action::Hold;
    }
    if look.ready {
        return Action::Remember;
    }
    if !look.alive && look.has_last && !look.just_started {
        return Action::Start;
    }
    Action::Hold
}

#[cfg(test)]
mod tests {
    use super::*;

    fn look() -> Look {
        Look {
            alive: false,
            ready: false,
            has_last: true,
            pending: false,
            waited: Duration::ZERO,
            limit: Duration::from_secs(60),
            just_started: false,
        }
    }

    #[test]
    fn a_dead_process_is_started_from_the_last_ready_path() {
        assert_eq!(action(look()), Action::Start);
    }

    #[test]
    fn a_start_just_issued_is_not_issued_again() {
        let mut l = look();
        l.just_started = true;
        assert_eq!(action(l), Action::Hold);
    }

    #[test]
    fn a_ready_process_is_recorded() {
        let mut l = look();
        l.alive = true;
        l.ready = true;
        assert_eq!(action(l), Action::Remember);
    }

    #[test]
    fn a_pending_replacement_is_left_alone_until_its_time_is_up() {
        let mut l = look();
        l.alive = true;
        l.pending = true;
        l.waited = Duration::from_secs(59);
        assert_eq!(action(l), Action::Hold);
        l.waited = Duration::from_secs(60);
        assert_eq!(action(l), Action::Recover);
    }

    #[test]
    fn a_pending_replacement_that_is_ready_is_the_new_last() {
        let mut l = look();
        l.alive = true;
        l.ready = true;
        l.pending = true;
        assert_eq!(action(l), Action::Remember);
    }

    #[test]
    fn nothing_known_and_nothing_running_holds() {
        let mut l = look();
        l.has_last = false;
        assert_eq!(action(l), Action::Hold);
    }
}
