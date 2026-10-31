//! One app's reads of its store for the facts (§3.8, §3.11) — its census and
//! the decisions under `certify/`, `rollout/` and `pause/` — run as one task the
//! tick waits for with a bound.
//!
//! **Why a task that outlives the tick.** Each read is a blocking call into the
//! store, and a store can be held for minutes by work on its own thread — on
//! zend, every census waits on the persistence thread. Dropping a timed-out
//! `spawn_blocking` does not stop it, so a tick that started a fresh read every
//! second would stack one more blocked thread a second for as long as the store
//! is held. Instead a read that does not answer in time is kept, and the next
//! tick looks at it again without waiting and without asking the store anything
//! new. Only one read per app is ever in flight.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio::time::timeout;

use crate::replicate::Replicator;
use crate::upgrade::decisions::{pause_prefix, rollout_prefix};

/// What one app's store answered.
#[derive(Debug, Default)]
pub(super) struct AppFacts {
    pub census: Vec<u8>,
    pub certify: Vec<(String, Vec<u8>)>,
    pub rollouts: Vec<(String, Vec<u8>)>,
    pub pauses: Vec<(String, Vec<u8>)>,
}

/// The decisions under which every build's verdicts are recorded for `app`.
pub(super) fn verdicts_prefix(app: &str) -> String {
    format!("certify/{app}/")
}

/// One app's facts read, in flight.
#[derive(Debug)]
pub(super) struct FactsRead {
    app: String,
    /// The store call it is waiting on now.
    waiting_on: Arc<Mutex<String>>,
    task: JoinHandle<AppFacts>,
}

impl FactsRead {
    /// Start reading `app`'s facts from `r`'s store.
    pub(super) fn start(app: &str, r: Arc<Replicator>) -> FactsRead {
        let waiting_on = Arc::new(Mutex::new(String::new()));
        let on = waiting_on.clone();
        let name = app.to_string();
        let task = tokio::spawn(async move {
            let set = |what: String| *on.lock().unwrap_or_else(|e| e.into_inner()) = what;
            set("census".into());
            let census = r.census().await;
            let cp = verdicts_prefix(&name);
            set(format!("decisions under {cp}"));
            let certify = r.decisions(&cp).await;
            let rp = rollout_prefix(&name);
            set(format!("decisions under {rp}"));
            let rollouts = r.decisions(&rp).await;
            let pp = pause_prefix(&name);
            set(format!("decisions under {pp}"));
            let pauses = r.decisions(&pp).await;
            AppFacts {
                census,
                certify,
                rollouts,
                pauses,
            }
        });
        FactsRead {
            app: app.to_string(),
            waiting_on,
            task,
        }
    }

    /// The app it reads for.
    pub(super) fn app(&self) -> &str {
        &self.app
    }

    /// Its answer, waiting up to `budget` — or itself back, still waiting. A
    /// read that panicked answers nothing, as a store call that panics does.
    pub(super) async fn wait(mut self, budget: Duration) -> Result<AppFacts, FactsRead> {
        match timeout(budget, &mut self.task).await {
            Ok(Ok(facts)) => Ok(facts),
            Ok(Err(e)) => {
                tracing::error!(app = self.app, "reading the upgrade facts failed: {e}");
                Ok(AppFacts::default())
            }
            Err(_) => Err(self),
        }
    }

    /// The upgrade step while this read has not answered within `wait`.
    pub(super) fn blocked_step(&self, wait: Duration) -> String {
        let on = self
            .waiting_on
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        blocked_step(&self.app, &on, wait)
    }
}

/// `blocked: <app>'s store did not answer <what> in <wait>`.
pub(super) fn blocked_step(app: &str, what: &str, wait: Duration) -> String {
    format!(
        "blocked: {app}'s store did not answer {what} in {}",
        span(wait)
    )
}

/// `10s`, or `250ms` for a wait that is not whole seconds.
fn span(d: Duration) -> String {
    if d.subsec_millis() == 0 {
        format!("{}s", d.as_secs())
    } else {
        format!("{}ms", d.as_millis())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blocked_step_names_the_app_the_call_and_the_wait() {
        assert_eq!(
            blocked_step("zend", "census", Duration::from_secs(10)),
            "blocked: zend's store did not answer census in 10s"
        );
        assert_eq!(
            blocked_step(
                "npcd",
                "decisions under pause/npcd/",
                Duration::from_millis(250)
            ),
            "blocked: npcd's store did not answer decisions under pause/npcd/ in 250ms"
        );
    }
}
