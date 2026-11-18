//! An agent's install (`docs/replicated_substrate.md` §3.9): asked by the
//! process's upgrades once they have held the agent's builds and the hold has
//! drained (`docs/build_agent.md` §4), it stops the agent the way Ctrl-C does
//! — its sockets close, so no gateway sends it another job — and asks `keep`
//! to start the staged build. Nothing is running by then: a deploy's build
//! has handed off, any other run was set aside, and the queue waits on keep.

use std::sync::{Arc, Mutex};

use cluster::upgrade::runtime::{InstallJob, Installer};
use tokio::sync::Notify;

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
            connector: None,
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
