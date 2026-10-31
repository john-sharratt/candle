//! What `/v1/status`'s replication block shows of the member's acceptor.

use std::sync::{Arc, Weak};
use std::time::Duration;

use serde_json::json;

use super::memory::MemoryStore;
use super::replicator::{Replication, Replicator};
use crate::consensus::votes::{MemoryVotes, Vote, VoteStore};
use crate::consensus::{Value, WallMillis};
use crate::frame::{AcceptedValue, Ballot};

/// **The replication block lists every lease the acceptor counts as armed**,
/// with its holder and the seconds it has left — what answers "why is this
/// member refused `upgrade/zend`?" from the member's own status.
#[test]
fn the_replication_block_lists_the_armed_leases() {
    let mut votes = MemoryVotes::new();
    let ballot = Ballot { round: 1, node: 4 };
    // Node 4 took `upgrade/zend` for 20 minutes; 899.5 s of it are left now.
    votes
        .write(
            b"upgrade/zend",
            &Vote {
                promised: ballot,
                accepted: Some(AcceptedValue {
                    ballot,
                    value: Value::Lease {
                        holder: 4,
                        ttl: Duration::from_secs(20 * 60),
                    }
                    .encode(),
                }),
                lease_ends: WallMillis::now() + Duration::from_millis(899_500),
            },
        )
        .unwrap();
    // A decision is not a lease, and is not listed.
    votes
        .write(
            b"gen/1",
            &Vote {
                promised: ballot,
                accepted: Some(AcceptedValue {
                    ballot,
                    value: Value::Decision(b"x".to_vec()).encode(),
                }),
                lease_ends: WallMillis(0),
            },
        )
        .unwrap();
    let store = Arc::new(MemoryStore::new(1));
    let replication = Replication::open(store, Box::new(votes)).unwrap();
    let r = Replicator::new(replication, 1, Weak::new());
    let view = serde_json::to_value(r.view()).unwrap();
    assert_eq!(
        view["leases"],
        json!([{"key": "upgrade/zend", "holder": "0000000000000004", "remaining_s": 900}])
    );
}
