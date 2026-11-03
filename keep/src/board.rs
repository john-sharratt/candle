//! What keep says of its services' trials (`docs/claude_zend_gateway.md`
//! §8.4): the trial running on each, and how the last one ended. A trial
//! answers only once it is proven or rolled back — up to ten minutes — so one
//! started from the cluster runs on its own thread and is read back here.
//!
//! A trial started by a deploy carries the deploy's id
//! (`docs/jobs.md` §1), so whoever reads the board — any member,
//! any session — can tell which deploy a trial belongs to and how it ended. A
//! trial asked to stop is marked cancelled here; the supervisor watching it
//! reads the mark and rolls it back.

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// keep's one board: every trial, whichever way it was asked for — the
/// loopback switch or the cluster — is kept here.
pub static BOARD: LazyLock<Board> = LazyLock::new(Board::default);

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A trial under way.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Trial {
    pub artifact: String,
    /// The deploy it belongs to; `None` for a trial asked for by hand.
    pub deploy: Option<String>,
    pub started_ms: u64,
    /// Asked to stop: it is being rolled back.
    pub cancelled: bool,
}

/// How a trial ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Ended {
    pub artifact: String,
    pub deploy: Option<String>,
    /// Proven and promoted, or rolled back to the previous stable file.
    pub proven: bool,
    /// Rolled back because it was cancelled.
    pub cancelled: bool,
    pub detail: String,
    pub started_ms: u64,
    pub ended_ms: u64,
}

/// One service's trials.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Trials {
    pub running: Option<Trial>,
    pub last: Option<Ended>,
}

/// Every service's trials, by name.
#[derive(Debug, Default)]
pub struct Board {
    services: Mutex<BTreeMap<String, Trials>>,
}

impl Board {
    fn services(&self) -> MutexGuard<'_, BTreeMap<String, Trials>> {
        self.services.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A trial of `artifact` for `deploy` began on `service`; refused while
    /// one runs.
    pub fn begin(&self, service: &str, artifact: &str, deploy: Option<&str>) -> Result<(), String> {
        let mut s = self.services();
        let t = s.entry(service.to_string()).or_default();
        if let Some(running) = &t.running {
            return Err(format!(
                "a trial of {} is already running on {service}",
                running.artifact
            ));
        }
        t.running = Some(Trial {
            artifact: artifact.to_string(),
            deploy: deploy.map(str::to_string),
            started_ms: now_ms(),
            cancelled: false,
        });
        Ok(())
    }

    /// Ask the trial on `service` to stop; the trial as it now stands, or why
    /// there is none to stop.
    pub fn cancel(&self, service: &str) -> Result<Trial, String> {
        let mut s = self.services();
        let running = s
            .get_mut(service)
            .and_then(|t| t.running.as_mut())
            .ok_or_else(|| format!("no trial is running on {service}"))?;
        running.cancelled = true;
        Ok(running.clone())
    }

    /// Whether the trial on `service` has been asked to stop.
    pub fn cancelled(&self, service: &str) -> bool {
        self.services()
            .get(service)
            .and_then(|t| t.running.as_ref())
            .is_some_and(|r| r.cancelled)
    }

    /// The trial on `service` ended.
    pub fn end(&self, service: &str, proven: bool, detail: &str) {
        let mut s = self.services();
        let t = s.entry(service.to_string()).or_default();
        let running = t.running.take();
        t.last = Some(Ended {
            artifact: running
                .as_ref()
                .map(|r| r.artifact.clone())
                .unwrap_or_default(),
            deploy: running.as_ref().and_then(|r| r.deploy.clone()),
            proven,
            cancelled: running.as_ref().is_some_and(|r| r.cancelled),
            detail: detail.trim().to_string(),
            started_ms: running.as_ref().map_or(0, |r| r.started_ms),
            ended_ms: now_ms(),
        });
    }

    pub fn trials(&self, service: &str) -> Trials {
        self.services().get(service).cloned().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trial_runs_alone_and_its_end_is_kept() {
        let b = Board::default();
        b.begin("motherd", "stage/motherd.exe", Some("76c3a0f4d39bd9db:Xe"))
            .unwrap();
        assert!(b
            .begin("motherd", "other.exe", None)
            .unwrap_err()
            .contains("already running"));
        assert!(
            b.begin("web", "stage/web.exe", None).is_ok(),
            "another service is apart"
        );
        b.end("motherd", false, "rolled back\nno serving line\n");
        let t = b.trials("motherd");
        assert_eq!(t.running, None);
        let last = t.last.unwrap();
        assert_eq!(
            (
                last.artifact.as_str(),
                last.deploy.as_deref(),
                last.proven,
                last.cancelled,
                last.detail.as_str()
            ),
            (
                "stage/motherd.exe",
                Some("76c3a0f4d39bd9db:Xe"),
                false,
                false,
                "rolled back\nno serving line"
            )
        );
        assert!(last.started_ms > 0 && last.ended_ms >= last.started_ms);
        assert!(b.begin("motherd", "again.exe", None).is_ok());
        assert_eq!(b.trials("unknown"), Trials::default());
    }

    /// **A cancel marks the running trial, and its end says it was
    /// cancelled**; with nothing running there is nothing to cancel.
    #[test]
    fn a_cancelled_trial_ends_marked() {
        let b = Board::default();
        assert!(b.cancel("web").unwrap_err().contains("no trial"));
        b.begin("web", "stage/web.exe", Some("d")).unwrap();
        assert!(!b.cancelled("web"));
        assert!(b.cancel("web").unwrap().cancelled);
        assert!(b.cancelled("web"));
        b.end("web", false, "rolled back\nthe trial was cancelled");
        assert!(!b.cancelled("web"), "nothing runs to be cancelled");
        assert!(b.trials("web").last.unwrap().cancelled);
    }
}
