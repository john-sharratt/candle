//! The claims a gateway has been told (§5.4): which member serves each sticky
//! key, and leases announced for routing (§9.6). Soft state: a gateway restart
//! forgets it, and members re-announce on reconnect. Of two claims on one key
//! the higher epoch keeps it, whatever order they arrive in.

use std::collections::HashMap;

use crate::frame::{ClaimInfo, Epoch};

#[derive(Debug, Default)]
pub struct Claims {
    held: HashMap<Vec<u8>, Epoch>,
}

impl Claims {
    pub fn new() -> Self {
        Self::default()
    }

    /// `CLAIM { key, epoch }`; returns whether it took the key.
    pub fn claim(&mut self, key: &[u8], epoch: Epoch) -> bool {
        match self.held.get(key) {
            Some(held) if *held >= epoch => false,
            _ => {
                self.held.insert(key.to_vec(), epoch);
                true
            }
        }
    }

    /// `UNCLAIM { key, epoch }`: let go only if that claim still holds it.
    pub fn unclaim(&mut self, key: &[u8], epoch: Epoch) -> bool {
        if self.held.get(key) == Some(&epoch) {
            self.held.remove(key);
            return true;
        }
        false
    }

    /// The node holding `key`.
    pub fn holder(&self, key: &[u8]) -> Option<u64> {
        self.held.get(key).map(|e| e.node)
    }

    /// The whole table, as `ADMIT` sends it.
    pub fn table(&self) -> Vec<ClaimInfo> {
        let mut t: Vec<ClaimInfo> = self
            .held
            .iter()
            .map(|(key, epoch)| ClaimInfo {
                key: key.clone(),
                epoch: *epoch,
            })
            .collect();
        t.sort_by(|a, b| a.key.cmp(&b.key));
        t
    }

    pub fn len(&self) -> usize {
        self.held.len()
    }

    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(ms: u64, node: u64) -> Epoch {
        Epoch { ms, node }
    }

    /// Of two first claims the higher epoch keeps the key, in either order.
    #[test]
    fn the_higher_epoch_keeps_the_key_in_any_order() {
        for order in [[e(10, 1), e(10, 2)], [e(10, 2), e(10, 1)]] {
            let mut c = Claims::new();
            for epoch in order {
                c.claim(b"conv", epoch);
            }
            assert_eq!(c.holder(b"conv"), Some(2));
        }
    }

    #[test]
    fn an_unclaim_lets_go_only_its_own_claim() {
        let mut c = Claims::new();
        c.claim(b"k", e(1, 1));
        c.claim(b"k", e(2, 2));
        assert!(!c.unclaim(b"k", e(1, 1)), "a superseded claim's release");
        assert_eq!(c.holder(b"k"), Some(2));
        assert!(c.unclaim(b"k", e(2, 2)));
        assert_eq!(c.holder(b"k"), None);
    }
}
