//! What this member tells its gateways each second about its tenants: whether
//! each app's tenant is ready — `CAPACITY.state`, and what makes the app
//! clustered — and whether any tenant in the process is still loading —
//! `CAPACITY.process_loading`.
//!
//! **A member with a tenant still loading is lent no characters.** The tenants
//! share one engine, and a tenant's load ends with forwards of its own — zend
//! primes, reading every repository through the model. A member lent its share
//! of the cast the moment npcd loaded ran those turns over zend's priming: ten
//! conversations a wave left it without a single prefill token for forty
//! minutes. So the plan gives such a member no characters
//! (`docs/cooperative_admission.md` §4.4), and the cast runs on the members
//! that have loaded. Nothing else waits: each app is ready as its own tenant
//! is, so the member serves, holds its quorum place and takes the leases its
//! own load needs.

use std::sync::Arc;
use std::time::Duration;

use tokio::time::interval;

use super::AppLink;
use crate::tenant::Reports;

/// Every second, tell each app's gateways whether its tenant is ready, and
/// whether any tenant in the process is still loading.
pub fn watch_readiness(tenants: Vec<(Arc<AppLink>, Arc<dyn Reports>)>) {
    tokio::spawn(async move {
        let mut tick = interval(Duration::from_secs(1));
        loop {
            tick.tick().await;
            let mut states = Vec::with_capacity(tenants.len());
            for (_, reports) in &tenants {
                states.push(reports.status().await.state);
            }
            let loading = still_loading(&states);
            for ((link, _), state) in tenants.iter().zip(&states) {
                link.set_ready(*state == "ready");
                link.set_process_loading(loading);
            }
        }
    });
}

/// Whether any tenant, by its status `states`, is still loading.
fn still_loading(states: &[&str]) -> bool {
    states.iter().any(|s| *s != "ready")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **One tenant still loading marks the whole process**; every tenant
    /// ready, none.
    #[test]
    fn the_process_is_loading_while_any_tenant_is() {
        assert!(!still_loading(&["ready", "ready"]));
        assert!(still_loading(&["ready", "loading"]));
        assert!(still_loading(&["loading", "ready"]));
        assert!(!still_loading(&["ready"]));
    }
}
