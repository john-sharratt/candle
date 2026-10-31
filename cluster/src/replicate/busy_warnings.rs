//! Which refused upgrade-lease rounds a member has logged (§3.9, §5.9).
//!
//! A member that wants `upgrade/<app>` asks every 5 s, and every round that
//! comes back `LeaseBusy` names the holder an acceptor still counts. One warn
//! per change of holder says who the member is waiting on — the clue a stale
//! lease at one acceptor leaves — without a line every retry. World and other
//! leases are refused as a matter of course and stay quiet.

use std::collections::HashMap;

use super::upgrade::is_upgrade_lease;

/// The holder last warned of, per upgrade lease.
#[derive(Debug, Default)]
pub struct BusyWarnings {
    warned: HashMap<Vec<u8>, u64>,
}

impl BusyWarnings {
    /// A round for `key` was refused because `holder` holds it: whether that
    /// is worth a warn — an upgrade lease, and a holder not already warned of.
    pub fn busy(&mut self, key: &[u8], holder: u64) -> bool {
        if !is_upgrade_lease(key) {
            return false;
        }
        self.warned.insert(key.to_vec(), holder) != Some(holder)
    }

    /// This member holds `key` now: a later refusal is news again, whoever
    /// it names.
    pub fn held(&mut self, key: &[u8]) {
        self.warned.remove(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A member retrying every 5 s warns once per holder, not once per retry.
    #[test]
    fn a_refusal_warns_once_per_holder() {
        let mut w = BusyWarnings::default();
        assert!(w.busy(b"upgrade/zend", 7));
        assert!(!w.busy(b"upgrade/zend", 7), "the same holder, retried");
        assert!(!w.busy(b"upgrade/zend", 7));
        assert!(w.busy(b"upgrade/zend", 9), "another holder");
        assert!(w.busy(b"upgrade/zend", 7), "and back");
        assert!(w.busy(b"upgrade/npcd", 7), "each key on its own");
    }

    #[test]
    fn holding_the_lease_makes_the_next_refusal_news() {
        let mut w = BusyWarnings::default();
        assert!(w.busy(b"upgrade/zend", 7));
        w.held(b"upgrade/zend");
        assert!(w.busy(b"upgrade/zend", 7));
    }

    #[test]
    fn other_leases_never_warn() {
        let mut w = BusyWarnings::default();
        assert!(!w.busy(b"world/w1", 7));
        assert!(!w.busy(b"git/candle", 7));
    }
}
