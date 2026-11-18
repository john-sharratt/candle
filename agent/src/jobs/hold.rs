//! The hold a planned restart asks for first (`docs/build_agent.md` §3).
//!
//! A hold stops the queue at once. The run holding the slot is dealt with by
//! what it is: a deploy's build is waited for until it has staged and asked
//! keep for its trial, at most the agent's `build_hold`; any other run is
//! stopped and put back at the head of the queue under its id, to run again
//! from its start. A process that is not well is not waited for. Then the queue
//! is parked on keep, and the hold has **drained**: the process may be stopped.
//!
//! Holds join: the queue moves again only once every hold has ended. A hold
//! ends when its [`Hold`] is dropped; if the process did not stop meanwhile,
//! keep lets the park go and the queue goes on.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cluster::upgrade::runtime::builds::{BuildHold, HoldTicket};
use tokio::runtime::Handle;
use tokio::sync::Notify;
use tokio::time::timeout;
use zend_vfs::{BranchName, JobStatus};

use super::queue::Holder;
use super::{Jobs, Kind, Prep};

/// How often a hold looks again at a deploy it waits for, for the process's
/// health and the time left, when the queue says nothing new.
const LOOK_EVERY: Duration = Duration::from_secs(1);

/// What a hold has come to, shared with the task draining it.
struct Drain {
    drained: AtomicBool,
    /// The hold was let go: its task stops.
    gone: AtomicBool,
    waiting_for: Mutex<String>,
    done: Notify,
}

/// One hold on an agent's queue: see the module documentation.
pub struct Hold {
    jobs: Arc<Jobs>,
    drain: Arc<Drain>,
}

impl Hold {
    /// Whether the process may now be stopped.
    pub fn drained(&self) -> bool {
        self.drain.drained.load(Ordering::SeqCst)
    }

    /// Until the hold has drained.
    pub async fn until_drained(&self) {
        loop {
            let notified = self.drain.done.notified();
            if self.drained() {
                return;
            }
            notified.await;
        }
    }

    /// What it is waiting for: `battlecities deploy, about 1 min left`.
    pub fn waiting_for(&self) -> String {
        self.drain
            .waiting_for
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        self.drain.gone.store(true, Ordering::SeqCst);
        if self.jobs.queue.release() {
            // The process did not stop: the park on keep is let go.
            if let Ok(rt) = Handle::try_current() {
                let jobs = Arc::clone(&self.jobs);
                rt.spawn(async move { jobs.discard_park().await });
            }
        }
    }
}

impl HoldTicket for Hold {
    fn drained(&self) -> bool {
        Hold::drained(self)
    }
    fn waiting_for(&self) -> String {
        Hold::waiting_for(self)
    }
}

/// The agent's [`BuildHold`], for the cluster's upgrader.
pub struct Builds(pub Arc<Jobs>);

impl BuildHold for Builds {
    fn hold(&self) -> Arc<dyn HoldTicket> {
        self.0.hold()
    }
}

impl Jobs {
    /// Hold the queue for a restart (see the module documentation). Answers
    /// at once; the hold drains in a task of its own.
    pub fn hold(self: &Arc<Self>) -> Arc<Hold> {
        self.queue.hold();
        let drain = Arc::new(Drain {
            drained: AtomicBool::new(false),
            gone: AtomicBool::new(false),
            waiting_for: Mutex::new("the run holding the build slot".into()),
            done: Notify::new(),
        });
        let jobs = Arc::clone(self);
        let draining = Arc::clone(&drain);
        tokio::spawn(async move { jobs.drain(&draining).await });
        Arc::new(Hold {
            jobs: Arc::clone(self),
            drain,
        })
    }

    async fn drain(self: Arc<Self>, drain: &Drain) {
        let mut changed = self.queue.watch();
        loop {
            if drain.gone.load(Ordering::SeqCst) {
                return;
            }
            let Some(holder) = self.queue.running() else {
                break;
            };
            let waiting = if !holder.deploy {
                if self.set_aside(&holder.id) {
                    continue;
                }
                "a run that is no deploy, staging what it built".to_string()
            } else {
                if !self.healthy() {
                    tracing::warn!(
                        job = holder.id,
                        "hold: the process is not well, so its deploy build is not waited for"
                    );
                    break;
                }
                if holder.since.elapsed() >= self.build_hold {
                    tracing::warn!(
                        job = holder.id,
                        "hold: the deploy build ran past the build hold and goes with the process"
                    );
                    break;
                }
                self.waiting_text(&holder)
            };
            *drain.waiting_for.lock().unwrap_or_else(|e| e.into_inner()) = waiting;
            let _ = timeout(LOOK_EVERY, changed.changed()).await;
        }
        self.park().await;
        drain.drained.store(true, Ordering::SeqCst);
        drain.done.notify_waiters();
    }

    /// Stop run `id`, which holds the slot and is no deploy, and put it back at
    /// the head of the queue under its id, to run again from its start; `false`
    /// when its command has already ended and it is only staging, which is
    /// waited out.
    fn set_aside(self: &Arc<Self>, id: &str) -> bool {
        let mut t = self.table();
        let Some(e) = t.0.get_mut(id) else {
            return false;
        };
        let repo = e.repo.clone();
        let Kind::Run {
            task,
            prep,
            request,
            overlay,
            gate,
            ..
        } = &mut e.kind
        else {
            return false;
        };
        if let Prep::Started(j) = prep {
            let finished = self
                .checkout(&repo)
                .ok()
                .and_then(|c| c.server.query_job(j).ok())
                .is_some_and(|info| info.status != JobStatus::Running);
            if finished {
                return false;
            }
        }
        let Ok(branch) = BranchName::parse(&request.branch) else {
            return false;
        };
        task.abort();
        *prep = Prep::Queued;
        e.line.slot_ms = None;
        self.queue.leave(id);
        self.queue.join(id, true);
        let (run, snapshot, gate) = (request.clone(), overlay.clone(), *gate);
        let respawned = self.spawn_carry(id, run, snapshot, branch, gate);
        if let Some(Kind::Run { task, .. }) = t.0.get_mut(id).map(|e| &mut e.kind) {
            *task = respawned;
        }
        tracing::warn!(
            job = id,
            "hold: a run that is no deploy was stopped for a restart, and runs again from its \
             start once the queue moves"
        );
        true
    }

    /// What a hold waiting on `holder`, a deploy's build, says it waits for:
    /// its service, and the time left by the same command's last success here.
    fn waiting_text(&self, holder: &Holder) -> String {
        let t = self.table();
        let Some(e) = t.0.get(&holder.id) else {
            return "the run holding the build slot".into();
        };
        let service = match &e.kind {
            Kind::Run { request, .. } => request.deploy.clone(),
            _ => None,
        }
        .unwrap_or_default();
        let took = t
            .1
            .iter()
            .rev()
            .filter_map(|other| t.0.get(other))
            .filter(|o| o.line.command == e.line.command && *o.ended.borrow())
            .find_map(|o| {
                let succeeded = match &o.kind {
                    Kind::Run { outcome, .. } => {
                        outcome.as_ref().is_some_and(|r| r.exit_code == Some(0))
                    }
                    Kind::Recalled { view } => view.status == JobStatus::Exited { code: Some(0) },
                    _ => false,
                };
                let from = o.line.slot_ms?;
                let to = o.line.ended_ms?;
                (succeeded && to > from).then(|| Duration::from_millis(to - from))
            });
        left_text(&service, took, holder.since.elapsed())
    }
}

/// `<service> deploy, about N min left`, from what the same build `took` last
/// time and how long this one has run.
pub fn left_text(service: &str, took: Option<Duration>, ran: Duration) -> String {
    match took {
        Some(took) if took > ran => {
            let mins = (took - ran).as_secs().div_ceil(60).max(1);
            format!("{service} deploy, about {mins} min left")
        }
        Some(_) => format!("{service} deploy, past the time it took last"),
        None => format!("{service} deploy, time left unknown"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_time_left_is_read_from_the_last_success() {
        let min = |m: u64| Duration::from_secs(m * 60);
        assert_eq!(
            left_text("battlecities", Some(min(2)), Duration::from_secs(70)),
            "battlecities deploy, about 1 min left"
        );
        assert_eq!(
            left_text("motherd", Some(min(12)), min(2)),
            "motherd deploy, about 10 min left"
        );
        assert_eq!(
            left_text("motherd", Some(min(2)), min(3)),
            "motherd deploy, past the time it took last"
        );
        assert_eq!(
            left_text("yenbot", None, min(1)),
            "yenbot deploy, time left unknown"
        );
    }
}
