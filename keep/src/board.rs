//! What keep says of its services' trials (`docs/claude_zend_gateway.md`
//! §8.4): the trial running on each, and how the last one ended. A trial
//! answers only once it is proven or rolled back — up to ten minutes — so one
//! started from the cluster runs on its own thread and is read back here.

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};
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
    pub started_ms: u64,
}

/// How a trial ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Ended {
    pub artifact: String,
    /// Proven and promoted, or rolled back to the previous stable file.
    pub proven: bool,
    pub detail: String,
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
    /// A trial of `artifact` began on `service`; refused while one runs.
    pub fn begin(&self, service: &str, artifact: &str) -> Result<(), String> {
        let mut s = self.services.lock().unwrap_or_else(|e| e.into_inner());
        let t = s.entry(service.to_string()).or_default();
        if let Some(running) = &t.running {
            return Err(format!(
                "a trial of {} is already running on {service}",
                running.artifact
            ));
        }
        t.running = Some(Trial {
            artifact: artifact.to_string(),
            started_ms: now_ms(),
        });
        Ok(())
    }

    /// The trial on `service` ended.
    pub fn end(&self, service: &str, proven: bool, detail: &str) {
        let mut s = self.services.lock().unwrap_or_else(|e| e.into_inner());
        let t = s.entry(service.to_string()).or_default();
        let artifact = t.running.take().map(|r| r.artifact).unwrap_or_default();
        t.last = Some(Ended {
            artifact,
            proven,
            detail: detail.trim().to_string(),
            ended_ms: now_ms(),
        });
    }

    pub fn trials(&self, service: &str) -> Trials {
        self.services
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(service)
            .cloned()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trial_runs_alone_and_its_end_is_kept() {
        let b = Board::default();
        b.begin("motherd", "stage/motherd.exe").unwrap();
        assert!(b
            .begin("motherd", "other.exe")
            .unwrap_err()
            .contains("already running"));
        assert!(
            b.begin("web", "stage/web.exe").is_ok(),
            "another service is apart"
        );
        b.end("motherd", false, "rolled back\nno serving line\n");
        let t = b.trials("motherd");
        assert_eq!(t.running, None);
        let last = t.last.unwrap();
        assert_eq!(
            (last.artifact.as_str(), last.proven, last.detail.as_str()),
            ("stage/motherd.exe", false, "rolled back\nno serving line")
        );
        assert!(b.begin("motherd", "again.exe").is_ok());
        assert_eq!(b.trials("unknown"), Trials::default());
    }
}
