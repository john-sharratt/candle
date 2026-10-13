//! What an upgrade asks of a member's replication (§3.8–§3.11): to propose a
//! decision — a verdict, a forced rollout, a pause — and to read the decisions
//! the group holds; and the `upgrade/<app>` leases, which a drain never hands
//! back, since the member holding one is draining exactly because it holds it.

use std::sync::Arc;

use super::replicator::Replicator;

/// The keys of the leases a member upgrades under (§3.9, §3.11).
pub const UPGRADE_PREFIX: &[u8] = b"upgrade/";

/// Whether `key` is an upgrade lease.
pub fn is_upgrade_lease(key: &[u8]) -> bool {
    key.starts_with(UPGRADE_PREFIX)
}

impl Replicator {
    /// Propose `value` for the one-time decision `key`. Refused — `false` —
    /// in fault mode, which proposes nothing (§7.7). Whatever is decided is
    /// written as a replicated record, which [`Self::decisions`] then reads.
    pub fn propose(self: &Arc<Self>, key: &[u8], value: Vec<u8>) -> bool {
        if self.faulted() {
            return false;
        }
        let now = self.now();
        let quorum = self.quorum();
        let tiebreak = self.tiebreak_allowed();
        let outs = {
            let mut p = self.paxos.lock().unwrap_or_else(|e| e.into_inner());
            p.set_quorum(quorum);
            p.allow_tiebreak(tiebreak);
            p.decide(key, value, now)
        };
        match outs {
            Ok(outs) => {
                self.dispatch(outs);
                true
            }
            Err(e) => {
                tracing::error!(node = self.node, "the vote journal failed: {e}");
                false
            }
        }
    }

    /// Every decision recorded under `prefix`, as `(key, value)`.
    pub async fn decisions(&self, prefix: &str) -> Vec<(String, Vec<u8>)> {
        let store = self.store.clone();
        let prefix = prefix.to_string();
        tokio::task::spawn_blocking(move || store.decisions(&prefix))
            .await
            .unwrap_or_default()
    }

    /// Every record type its substrate holds, by tag (§3.9).
    pub async fn census(&self) -> Vec<u8> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.census())
            .await
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_upgrade_keys_are_upgrade_leases() {
        assert!(is_upgrade_lease(b"upgrade/zend"));
        assert!(is_upgrade_lease(b"upgrade/zend/1"));
        assert!(!is_upgrade_lease(b"world/w1"));
    }
}
