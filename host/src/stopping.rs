//! How the process comes to stop (`docs/replicated_substrate.md` §12.4): a
//! Ctrl-C, or a `shutdown` or `restart` command. Each asks once — the first
//! wins — and the stop then runs the same sequence, its step shown on every
//! tenant's loading page as it goes.

use std::sync::Mutex;

use tokio::sync::watch;

/// What the process does once it has stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    /// Exit: a Ctrl-C, or `shutdown`.
    Exit,
    /// Start the same binary again with the same arguments: `restart`.
    Relaunch,
}

/// The step a stop is on, for the loading page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopStep {
    pub label: &'static str,
    pub detail: String,
}

/// See the module documentation.
pub struct Stopping {
    asked: watch::Sender<Option<Ending>>,
    step: Mutex<Option<StopStep>>,
}

impl Default for Stopping {
    fn default() -> Self {
        Self {
            asked: watch::channel(None).0,
            step: Mutex::new(None),
        }
    }
}

impl Stopping {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the process to stop and then do `ending`; a later ask changes
    /// nothing. Whether this one was the first.
    pub fn ask(&self, ending: Ending) -> bool {
        self.asked.send_if_modified(|asked| {
            if asked.is_some() {
                return false;
            }
            *asked = Some(ending);
            true
        })
    }

    /// What was asked, once something was.
    pub fn ending(&self) -> Option<Ending> {
        *self.asked.borrow()
    }

    /// Resolves once a stop is asked, with what was asked.
    pub async fn asked(&self) -> Ending {
        let mut rx = self.asked.subscribe();
        loop {
            if let Some(e) = *rx.borrow_and_update() {
                return e;
            }
            if rx.changed().await.is_err() {
                // The sender lives as long as `self`, which this borrows.
                unreachable!("the stop's sender outlives its receivers");
            }
        }
    }

    pub fn set_step(&self, label: &'static str, detail: impl Into<String>) {
        *self.step.lock().unwrap_or_else(|e| e.into_inner()) = Some(StopStep {
            label,
            detail: detail.into(),
        });
    }

    /// The step the stop is on; `None` until it starts.
    pub fn step(&self) -> Option<StopStep> {
        self.step.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first ask wins; a waiter started before it hears it.
    #[tokio::test]
    async fn the_first_ask_wins() {
        let s = std::sync::Arc::new(Stopping::new());
        assert_eq!(s.ending(), None);
        let waiter = {
            let s = s.clone();
            tokio::spawn(async move { s.asked().await })
        };
        assert!(s.ask(Ending::Relaunch));
        assert!(!s.ask(Ending::Exit));
        assert_eq!(waiter.await.unwrap(), Ending::Relaunch);
        assert_eq!(s.ending(), Some(Ending::Relaunch));
        // A waiter started after it hears it at once.
        assert_eq!(s.asked().await, Ending::Relaunch);
    }

    #[test]
    fn the_step_is_the_last_one_set() {
        let s = Stopping::new();
        assert_eq!(s.step(), None);
        s.set_step("Draining", "handing back 3 claims");
        s.set_step("Sending", "12 events left");
        assert_eq!(
            s.step(),
            Some(StopStep {
                label: "Sending",
                detail: "12 events left".into()
            })
        );
    }
}
