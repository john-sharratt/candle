//! Whether this member is ready for its apps — the state `CAPACITY` carries to
//! the gateways, and what makes an app clustered.
//!
//! **A member is ready for any app only once every tenant in it has loaded.**
//! The tenants share one engine, and a tenant's load ends with forwards of its
//! own — zend primes, reading every repository through the model. A member
//! ready for npcd the moment npcd loaded was lent its share of the cast, whose
//! turns took the engine from zend: ten conversations a wave left zend's
//! priming without a single prefill token for forty minutes. Not ready, the
//! member is lent no characters and sent no requests; the cast runs on the
//! members that are, and nothing waits on this one.

use std::sync::Arc;
use std::time::Duration;

use tokio::time::interval;

use super::AppLink;
use crate::tenant::Reports;

/// Every second, tell each app's gateways whether this member is ready for
/// it: every tenant loaded, or not.
pub fn watch_readiness(tenants: Vec<(Arc<AppLink>, Arc<dyn Reports>)>) {
    tokio::spawn(async move {
        let mut tick = interval(Duration::from_secs(1));
        loop {
            tick.tick().await;
            let mut states = Vec::with_capacity(tenants.len());
            for (_, reports) in &tenants {
                states.push(reports.status().await.state);
            }
            let ready = all_loaded(&states);
            for (link, _) in &tenants {
                link.set_ready(ready);
            }
        }
    });
}

/// Whether every tenant, by its status `states`, has loaded.
fn all_loaded(states: &[&str]) -> bool {
    !states.is_empty() && states.iter().all(|s| *s == "ready")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **One tenant still loading holds the member back for every app**, and
    /// a member with no tenants is ready for none.
    #[test]
    fn a_member_is_ready_once_every_tenant_has_loaded() {
        assert!(all_loaded(&["ready", "ready"]));
        assert!(!all_loaded(&["ready", "loading"]));
        assert!(!all_loaded(&["loading", "ready"]));
        assert!(all_loaded(&["ready"]));
        assert!(!all_loaded(&[]));
    }
}
