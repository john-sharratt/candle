//! **The applied set, the deferred set and the location index through the
//! store** (`docs/replicated_substrate.md` §6.2, §6.5, §14.1).

use tempfile::TempDir;

use super::identity::RecordIdentity;
use super::maintenance::MaintenanceOp;
use super::record::RecordType;
use super::replica::DeferredRecord;
use super::segment::SegmentId;
use super::SubstratePersistence;
use crate::substrate::Substrate;

const B: u64 = 0xB0B;

fn theirs(oseq: u64) -> RecordIdentity {
    RecordIdentity {
        origin: B,
        oseq,
        gen: 0,
        epoch: 0,
        hlc: oseq << 16,
    }
}

fn open(dir: &TempDir) -> (Substrate, SubstratePersistence) {
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    (substrate, sp)
}

/// Append a debug-id record carrying `id`, whose payload names `tag` — each
/// for a timeline of its own, so no two supersede each other.
fn append(sp: &mut SubstratePersistence, tag: &str, id: RecordIdentity) {
    let timeline = 0x1000 + id.oseq + if id.origin == B { 0x100 } else { 0 };
    let payload = format!("{{\"timeline_id\":{timeline},\"debug_id\":\"{tag}\"}}");
    sp.append_carried(RecordType::DebugId, 0, 0, 0, 0, 0, payload.as_bytes(), id)
        .unwrap();
}

fn debug_id_of(sp: &mut SubstratePersistence, origin: u64, oseq: u64) -> Option<String> {
    let record = sp.read_by_identity(origin, oseq).unwrap()?;
    let text = String::from_utf8(record.payload).unwrap();
    Some(
        text.split("\"debug_id\":\"")
            .nth(1)
            .unwrap()
            .trim_end_matches("\"}")
            .to_string(),
    )
}

/// **A record is found by its identity wherever maintenance moves it** —
/// staged, committed, relocated off a dropped segment, rewritten by a
/// compaction and reopened.
#[test]
fn a_record_is_found_by_identity_wherever_it_moves() {
    let dir = TempDir::new().unwrap();
    let mine = {
        let (_, mut sp) = open(&dir);
        let mine = sp.mint();
        append(&mut sp, "mine", mine);
        append(&mut sp, "theirs", theirs(3));
        assert_eq!(
            debug_id_of(&mut sp, mine.origin, mine.oseq).as_deref(),
            Some("mine")
        );
        assert_eq!(debug_id_of(&mut sp, B, 3).as_deref(), Some("theirs"));
        assert_eq!(debug_id_of(&mut sp, B, 4), None, "never held");
        sp.commit().unwrap();
        mine
    };
    // Reopened, so the substrate holds what the records say and every rewrite
    // carries them.
    let (mut substrate, mut sp) = open(&dir);
    sp.seal_active().unwrap();
    assert_eq!(sp.location_of(B, 3).unwrap().segment, SegmentId(1));

    sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(SegmentId(1)))
        .unwrap();
    let moved = sp.location_of(B, 3).expect("carried forward");
    assert_ne!(moved.segment, SegmentId(1), "out of the dropped segment");
    assert_eq!(debug_id_of(&mut sp, B, 3).as_deref(), Some("theirs"));

    sp.compact(&mut substrate, None).unwrap();
    assert_eq!(debug_id_of(&mut sp, B, 3).as_deref(), Some("theirs"));
    assert_eq!(
        debug_id_of(&mut sp, mine.origin, mine.oseq).as_deref(),
        Some("mine")
    );
    drop(sp);
    let (_, mut sp) = open(&dir);
    assert_eq!(debug_id_of(&mut sp, B, 3).as_deref(), Some("theirs"));
    assert_eq!(
        debug_id_of(&mut sp, mine.origin, mine.oseq).as_deref(),
        Some("mine")
    );
}

/// **What is applied of another origin is recorded exactly, gaps and all**,
/// and survives a reopen and a compaction — including the `oseq`s known only
/// as gone, which no record carries any more. This node's own records are
/// covered by its own `oseq`s.
#[test]
fn the_applied_set_keeps_its_gaps_through_reopen_and_compaction() {
    let dir = TempDir::new().unwrap();
    let (substrate, mut sp) = open(&dir);
    append(&mut sp, "1", theirs(1));
    append(&mut sp, "2", theirs(2));
    append(&mut sp, "5", theirs(5));
    let own = sp.mint();
    append(&mut sp, "own", own);
    assert_eq!(sp.applied_through(B), 2);
    assert!(sp.is_applied(B, 5) && !sp.is_applied(B, 3) && !sp.is_applied(B, 4));
    assert_eq!(sp.applied_through(own.origin), own.oseq);
    assert!(sp.is_applied(own.origin, own.oseq));
    assert!(!sp.is_applied(own.origin, own.oseq + 1));

    // A catch-up learned 7..=9 are gone everywhere.
    sp.mark_gone(B, 7, 9);
    sp.commit().unwrap();
    drop(sp);

    let check = |sp: &SubstratePersistence, context: &str| {
        assert_eq!(sp.applied_through(B), 2, "{context}");
        for oseq in [1, 2, 5, 7, 8, 9] {
            assert!(sp.is_applied(B, oseq), "{context}: {oseq}");
        }
        for oseq in [3, 4, 6, 10] {
            assert!(!sp.is_applied(B, oseq), "{context}: {oseq}");
        }
    };
    let (_, sp) = open(&dir);
    check(&sp, "reopened");
    drop(sp);
    let (mut substrate2, mut sp) = open(&dir);
    sp.compact(&mut substrate2, None).unwrap();
    check(&sp, "compacted");
    drop(sp);
    let (_, sp) = open(&dir);
    check(&sp, "reopened after compaction");
    drop(substrate);
}

/// **A deferred record waits for a build that knows its type**: it stays
/// deferred across a reopen while its type is unknown, and is released —
/// wanted again — by the first open of a build that decodes it.
#[test]
fn a_deferred_record_waits_for_a_build_that_knows_its_type() {
    let dir = TempDir::new().unwrap();
    {
        let (_, mut sp) = open(&dir);
        sp.defer(DeferredRecord {
            origin: B,
            oseq: 7,
            record_type: "a_type_from_the_future".into(),
        });
        sp.defer(DeferredRecord {
            origin: B,
            oseq: 8,
            record_type: "label".into(),
        });
        assert!(sp.is_deferred(B, 7) && sp.is_deferred(B, 8));
        assert!(!sp.is_applied(B, 7), "a deferred record is not applied");
        sp.commit().unwrap();
    }
    let (mut substrate, mut sp) = open(&dir);
    assert!(sp.is_deferred(B, 7), "still unknown to this build");
    assert!(
        !sp.is_deferred(B, 8),
        "a label is a type this build decodes"
    );
    sp.commit().unwrap();
    sp.compact(&mut substrate, None).unwrap();
    drop(sp);
    let (_, sp) = open(&dir);
    assert!(sp.is_deferred(B, 7));
    assert!(!sp.is_deferred(B, 8));
}

/// A node that never applies another origin's record and defers nothing
/// writes no `Replica` record at all.
#[test]
fn a_standalone_node_writes_no_replica() {
    let dir = TempDir::new().unwrap();
    let (_, mut sp) = open(&dir);
    let own = sp.mint();
    append(&mut sp, "own", own);
    sp.commit().unwrap();
    assert!(sp.replica().is_empty());
    drop(sp);
    let (_, sp) = open(&dir);
    assert_eq!(sp.open_census.get(RecordType::Replica), 0);
    assert_eq!(sp.open_census.get(RecordType::DebugId), 1);
}
