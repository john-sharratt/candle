//! The builds an install waits for (`docs/build_agent.md` §3, §4): a process
//! that runs a build agent hands the upgrader its [`BuildHold`], and an install
//! holds the agent's queue before it takes its upgrade leases, so the leases'
//! time is never spent waiting for a build and no build is cut short by the
//! restart.

use std::fmt;
use std::sync::Arc;

/// What holds a process's builds for a restart: the build agent in it.
pub trait BuildHold: Send + Sync {
    /// Stop the queue and deal with the running build; the hold, which says
    /// when it has drained. Answers at once.
    fn hold(&self) -> Arc<dyn HoldTicket>;
}

/// One hold of a [`BuildHold`], let go when dropped unless the process stops
/// first.
pub trait HoldTicket: Send + Sync {
    /// Whether the process may now be stopped: the running build has handed
    /// off, or was stopped and queued again, and the queue is parked.
    fn drained(&self) -> bool;
    /// What it is still waiting for, for the rollout's step:
    /// `battlecities deploy, about 1 min left`.
    fn waiting_for(&self) -> String;
}

impl fmt::Debug for dyn HoldTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HoldTicket")
            .field("drained", &self.drained())
            .finish()
    }
}

/// The step an install shows while its hold drains.
pub fn hold_step(ticket: &dyn HoldTicket) -> String {
    format!("waiting for builds: {}", ticket.waiting_for())
}

/// Whether an install may go on to take its leases: `Ok` once the process's
/// builds are held and drained, or when it runs none; otherwise the step to
/// show. The hold is asked for once and kept in `held` across ticks.
pub fn builds_drained(
    held: &mut Option<Arc<dyn HoldTicket>>,
    builds: Option<&Arc<dyn BuildHold>>,
) -> Result<(), String> {
    let Some(builds) = builds else {
        return Ok(());
    };
    let ticket = held.get_or_insert_with(|| builds.hold());
    if ticket.drained() {
        Ok(())
    } else {
        Err(hold_step(ticket.as_ref()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;

    struct Ticket(Arc<AtomicBool>);

    impl HoldTicket for Ticket {
        fn drained(&self) -> bool {
            self.0.load(Ordering::SeqCst)
        }
        fn waiting_for(&self) -> String {
            "battlecities deploy, about 1 min left".into()
        }
    }

    struct Agent {
        asked: AtomicUsize,
        drained: Arc<AtomicBool>,
    }

    impl BuildHold for Agent {
        fn hold(&self) -> Arc<dyn HoldTicket> {
            self.asked.fetch_add(1, Ordering::SeqCst);
            Arc::new(Ticket(Arc::clone(&self.drained)))
        }
    }

    #[test]
    fn the_step_names_what_the_hold_waits_for() {
        assert_eq!(
            hold_step(&Ticket(Arc::new(AtomicBool::new(false)))),
            "waiting for builds: battlecities deploy, about 1 min left"
        );
    }

    /// **An install asks for the hold once, waits on it across ticks, and goes
    /// on once it has drained**; a process with no builds goes on at once.
    #[test]
    fn an_install_holds_the_builds_once_and_goes_on_when_drained() {
        let agent = Arc::new(Agent {
            asked: AtomicUsize::new(0),
            drained: Arc::new(AtomicBool::new(false)),
        });
        let builds: Arc<dyn BuildHold> = agent.clone();
        let mut held = None;
        assert_eq!(
            builds_drained(&mut held, Some(&builds)),
            Err("waiting for builds: battlecities deploy, about 1 min left".into())
        );
        assert!(builds_drained(&mut held, Some(&builds)).is_err());
        agent.drained.store(true, Ordering::SeqCst);
        assert_eq!(builds_drained(&mut held, Some(&builds)), Ok(()));
        assert_eq!(agent.asked.load(Ordering::SeqCst), 1);
        assert!(held.is_some(), "the hold stands until the process stops");
        let mut none = None;
        assert_eq!(builds_drained(&mut none, None), Ok(()));
        assert!(none.is_none());
    }
}
