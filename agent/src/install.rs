//! An agent's install (`docs/replicated_substrate.md` §3.9): asked by the
//! process's upgrades, it stops the agent the way Ctrl-C does — its sockets
//! close, so no gateway sends it another job — waits, at most
//! [`JOBS_WAIT`], for the jobs it is running to end, and hands off to the
//! watchdog, which starts the new build once this process is gone.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use cluster::upgrade::runtime::{InstallJob, Installer};
use tokio::sync::Notify;
use tokio::time::{sleep, Instant};

use crate::jobs::Jobs;

/// The longest an install waits for the jobs in flight.
pub const JOBS_WAIT: Duration = Duration::from_secs(60);

/// The agent's side of an install.
#[derive(Default)]
pub struct AgentInstaller {
    job: Mutex<Option<InstallJob>>,
    asked: Notify,
}

impl AgentInstaller {
    pub fn new() -> Arc<AgentInstaller> {
        Arc::new(AgentInstaller::default())
    }

    /// Resolves once an install is asked for, with it.
    pub async fn asked(&self) -> InstallJob {
        loop {
            let notified = self.asked.notified();
            if let Some(job) = self.job.lock().unwrap_or_else(|e| e.into_inner()).clone() {
                return job;
            }
            notified.await;
        }
    }
}

impl Installer for AgentInstaller {
    fn install(&self, job: InstallJob) -> Result<(), String> {
        let mut held = self.job.lock().unwrap_or_else(|e| e.into_inner());
        if held.is_some() {
            return Err("an install is already under way".into());
        }
        *held = Some(job);
        self.asked.notify_waiters();
        Ok(())
    }
}

/// Wait, at most [`JOBS_WAIT`], for `jobs` to have nothing running; how many
/// were still running when it stopped waiting.
pub async fn jobs_ended(jobs: &Jobs) -> u32 {
    let deadline = Instant::now() + JOBS_WAIT;
    loop {
        let (_, running) = jobs.slots();
        if running == 0 || Instant::now() >= deadline {
            return running;
        }
        sleep(Duration::from_millis(200)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use cluster::frame::BuildId;
    use cluster::upgrade::swap::Source;

    fn job() -> InstallJob {
        InstallJob {
            build: BuildId {
                count: 12,
                hash: "h12".into(),
                local: String::new(),
            },
            source: Source::Kept("12-h12".into()),
            leases: vec![],
            protect: vec![],
        }
    }

    /// An install is handed to whoever waits for one; a second is refused.
    #[tokio::test]
    async fn an_install_reaches_the_agent_once() {
        let i = AgentInstaller::new();
        let waiter = {
            let i = i.clone();
            tokio::spawn(async move { i.asked().await })
        };
        tokio::task::yield_now().await;
        assert_eq!(i.install(job()), Ok(()));
        assert_eq!(waiter.await.unwrap(), job());
        assert_eq!(
            i.install(job()).unwrap_err(),
            "an install is already under way"
        );
        assert_eq!(i.asked().await, job(), "asked again, the same install");
    }
}
