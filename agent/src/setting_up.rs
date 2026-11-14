//! Which checkouts are still running their setup ([`crate::setup`]), and the
//! setup run holding each: a member's run on such a checkout waits until its
//! setup has ended, and is seen waiting behind that run.

use std::collections::HashMap;
use std::sync::Mutex;

use tokio::sync::watch;

/// Where a checkout's setup stands: `Some` — the setup runs it waits behind,
/// none while the file is being read — until it has ended, `None`.
type Standing = Option<Vec<String>>;

#[derive(Default)]
pub struct SettingUp {
    by_repo: Mutex<HashMap<String, watch::Sender<Standing>>>,
}

impl SettingUp {
    fn sender(&self, repo: &str) -> Option<watch::Sender<Standing>> {
        self.by_repo
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(repo)
            .cloned()
    }

    /// `repo`'s setup has begun: its runs wait from now on.
    pub fn begin(&self, repo: &str) {
        let (tx, _) = watch::channel(Some(Vec::new()));
        self.by_repo
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(repo.to_string(), tx);
    }

    /// `repo`'s setup is running the run `id`.
    pub fn running(&self, repo: &str, id: &str) {
        if let Some(tx) = self.sender(repo) {
            tx.send_replace(Some(vec![id.to_string()]));
        }
    }

    /// `repo`'s setup has ended: its runs go ahead.
    pub fn ended(&self, repo: &str) {
        if let Some(tx) = self.sender(repo) {
            tx.send_replace(None);
        }
    }

    /// The setup runs `repo`'s runs wait behind, or `None` when they wait for
    /// nothing.
    pub fn behind(&self, repo: &str) -> Option<Vec<String>> {
        self.sender(repo).and_then(|tx| tx.borrow().clone())
    }

    /// Until `repo`'s setup has ended; at once for a checkout with none.
    pub async fn until_ended(&self, repo: &str) {
        let Some(tx) = self.sender(repo) else {
            return;
        };
        let mut rx = tx.subscribe();
        let _ = rx.wait_for(Option::is_none).await;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::time::timeout;

    use super::*;

    #[tokio::test]
    async fn a_checkout_waits_while_its_setup_runs_and_one_without_setup_never_does() {
        let s = SettingUp::default();
        assert_eq!(s.behind("r"), None);
        s.until_ended("r").await;

        s.begin("r");
        assert_eq!(s.behind("r"), Some(Vec::new()));
        s.running("r", "job-1");
        assert_eq!(s.behind("r"), Some(vec!["job-1".to_string()]));
        assert!(
            timeout(Duration::from_millis(50), s.until_ended("r"))
                .await
                .is_err(),
            "a run waits while setup runs"
        );

        s.ended("r");
        assert_eq!(s.behind("r"), None);
        timeout(Duration::from_secs(1), s.until_ended("r"))
            .await
            .expect("an ended setup holds nothing");
    }
}
