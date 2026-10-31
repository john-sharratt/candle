//! The leases an acceptor counts as armed, and how a status page shows one
//! (§5.9, §12.1): the lease that answers "why is this key busy?".

use std::time::Duration;

use serde::Serialize;

/// A lease an acceptor still counts: its key as the acceptor holds it, its
/// holder, and how long its term has left on the acceptor's clock.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArmedLease {
    pub key: Vec<u8>,
    pub holder: u64,
    pub left: Duration,
}

/// One armed lease on a status page: the key shown lossily as UTF-8, the
/// holder as a 16-hex node, and the whole seconds left, rounded up so a lease
/// still armed never reads as zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LeaseRow {
    pub key: String,
    pub holder: String,
    pub remaining_s: u64,
}

impl LeaseRow {
    pub fn of(lease: &ArmedLease) -> LeaseRow {
        LeaseRow {
            key: String::from_utf8_lossy(&lease.key).into_owned(),
            holder: format!("{:016x}", lease.holder),
            remaining_s: remaining_s(lease.left),
        }
    }
}

/// `left` in whole seconds, rounded up.
pub fn remaining_s(left: Duration) -> u64 {
    (left.as_millis() as u64).div_ceil(1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_names_the_key_its_holder_and_the_seconds_left() {
        let row = LeaseRow::of(&ArmedLease {
            key: b"upgrade/zend".to_vec(),
            holder: 0xab,
            left: Duration::from_millis(899_001),
        });
        assert_eq!(
            serde_json::to_value(&row).unwrap(),
            serde_json::json!({
                "key": "upgrade/zend",
                "holder": "00000000000000ab",
                "remaining_s": 900,
            })
        );
        assert_eq!(remaining_s(Duration::from_millis(1)), 1);
        assert_eq!(remaining_s(Duration::from_secs(30)), 30);
    }
}
