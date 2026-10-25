//! What operating a member reads of its substrate (`docs/replicated_substrate.md`
//! §7.7, §12), on CPU substrates: the failed-write latch and the test write,
//! the tombstones by how near each is to expiring, the deferred records, the
//! live bytes and the audit.

use tempfile::TempDir;

use super::operation::TombstoneCounts;
use super::{Conversation, GroupId, LayerId};
use crate::persistence::SubstratePersistence;
use crate::substrate::Substrate;

fn open(dir: &TempDir) -> Conversation {
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    Conversation::from_parts(substrate, sp)
}

/// A fresh substrate has no failed write, a test write succeeds, and it holds
/// nothing deferred and no tombstone.
#[test]
fn a_fresh_substrate_reads_clean() {
    let dir = TempDir::new().unwrap();
    let conv = open(&dir);
    assert!(!conv.out_of_space());
    conv.test_write().unwrap();
    assert!(!conv.out_of_space());
    assert!(conv.deferred_records().is_empty());
    assert_eq!(conv.tombstone_counts(), TombstoneCounts::default());
    assert!(conv.liveness_audit().is_ok());
}

/// A tombstoned timeline is held until a complete compaction has run past it,
/// and then waits only for its age; the live bytes are what that compaction
/// copies.
#[test]
fn a_tombstone_is_swept_by_a_complete_compaction() {
    let dir = TempDir::new().unwrap();
    let conv = open(&dir);
    let timeline = conv.mint_timeline(LayerId::new(1), GroupId::new(1));
    conv.tombstone_timeline(timeline).unwrap();
    // The append is the writer's. Drain it before reading the log: this shuts
    // the writer down, and the compaction below writes the log itself.
    conv.flush_writer();
    conv.commit_persistence_if_pending().unwrap();
    assert_eq!(
        conv.tombstone_counts(),
        TombstoneCounts { held: 1, swept: 0 }
    );
    assert!(conv.live_bytes() > 0, "the tombstone itself is live");
    conv.compact_substrate(None).unwrap();
    assert_eq!(
        conv.tombstone_counts(),
        TombstoneCounts { held: 1, swept: 1 }
    );
}

/// A turn tombstone reaches the log when the writer drains — a conversation's
/// drop does that — and a reopen restores the turn as dropped. The caller does
/// not wait for the append.
#[test]
fn a_turn_tombstone_survives_the_writer_draining() {
    let dir = TempDir::new().unwrap();
    let timeline = {
        let conv = open(&dir);
        let timeline = conv.mint_timeline(LayerId::new(1), GroupId::new(1));
        conv.tombstone_turn(timeline, 3).unwrap();
        assert!(conv.is_turn_tombstoned(timeline, 3));
        timeline
    };
    let conv = open(&dir);
    assert!(
        conv.is_turn_tombstoned(timeline, 3),
        "the drained writer left the turn tombstone in the log"
    );
    assert!(!conv.read().is_tombstoned(timeline));
}
