//! **A member away too long drops what it can no longer trust, and takes back
//! what the cluster still holds** (`docs/replicated_substrate.md` §7.8).

use tempfile::TempDir;

use super::identity::{now_ms, RecordIdentity, COUNTER_BITS};
use super::joined;
use super::record::RecordType;
use super::remote::{Refusal, RemoteOutcome};
use super::{SubstratePersistence, SUBSTRATE_DIR};
use crate::substrate::Substrate;

const B: u64 = 0xB0B;
const DAY: u64 = 24 * 60 * 60 * 1000;

fn open(dir: &TempDir) -> (Substrate, SubstratePersistence) {
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    (substrate, sp)
}

/// Append a debug-id record carrying `id` — each for a timeline of its own,
/// so no two supersede each other.
fn append(sp: &mut SubstratePersistence, tag: &str, id: RecordIdentity) {
    let timeline = 0x1000 + id.oseq + if id.origin == B { 0x100 } else { 0 };
    let payload = format!("{{\"timeline_id\":{timeline},\"debug_id\":\"{tag}\"}}");
    sp.append_carried(RecordType::DebugId, 0, 0, 0, 0, 0, payload.as_bytes(), id)
        .unwrap();
}

/// Another origin's record `oseq`, written at `at_ms`.
fn theirs(oseq: u64, at_ms: u64) -> RecordIdentity {
    RecordIdentity {
        origin: B,
        oseq,
        gen: 0,
        epoch: 0,
        hlc: at_ms << COUNTER_BITS,
    }
}

fn joined_at(dir: &TempDir) -> Option<u64> {
    joined::read(&dir.path().join(SUBSTRATE_DIR)).unwrap()
}

fn set_joined(dir: &TempDir, at_ms: u64) {
    joined::write(&dir.path().join(SUBSTRATE_DIR), at_ms).unwrap();
}

fn held(sp: &SubstratePersistence, origin: u64, oseq: u64) -> bool {
    sp.location_of(origin, oseq).is_some()
}

/// A store never judged starts its clock at its first open, and drops
/// nothing however old its records are.
#[test]
fn a_store_never_judged_is_marked_and_keeps_everything() {
    let dir = TempDir::new().unwrap();
    let before = now_ms();
    {
        let (_, mut sp) = open(&dir);
        append(&mut sp, "old", theirs(1, before - 400 * DAY));
        sp.commit().unwrap();
    }
    assert!(joined_at(&dir).is_some_and(|at| at >= before));
    let (_, sp) = open(&dir);
    assert!(held(&sp, B, 1));
    assert!(sp.is_applied(B, 1));
}

/// Twenty days away: every record another origin wrote is gone from the
/// store — compacted away, not just skipped — and the applied set has
/// forgotten the origin, gone `oseq`s included, so a catch-up fetches it all
/// again. Its own records stay. The clock starts again.
#[test]
fn twenty_days_away_drops_what_other_origins_wrote() {
    let dir = TempDir::new().unwrap();
    let now = now_ms();
    let own = {
        let (_, mut sp) = open(&dir);
        let own = sp.mint();
        append(&mut sp, "own", own);
        append(&mut sp, "theirs", theirs(1, now - 25 * DAY));
        sp.mark_gone(B, 2, 3);
        sp.commit().unwrap();
        own
    };
    set_joined(&dir, now - 21 * DAY);

    let (_, sp) = open(&dir);
    assert!(!held(&sp, B, 1));
    for oseq in 1..=3 {
        assert!(!sp.is_applied(B, oseq), "{oseq}");
    }
    assert!(held(&sp, own.origin, own.oseq));
    assert!(sp.is_applied(own.origin, own.oseq));
    assert!(sp
        .applied_set()
        .origin(own.origin)
        .unwrap()
        .gaps()
        .is_empty());
    assert!(joined_at(&dir).is_some_and(|at| at >= now));
    drop(sp);

    let (_, sp) = open(&dir);
    assert_eq!(
        sp.open_census.get(RecordType::DebugId),
        1,
        "the other origin's record is gone from disk"
    );
    assert!(!sp.is_applied(B, 1));
}

/// Thirty days away: its own records from before it was last joined go too,
/// leaving gaps in its written set that survive a reopen — until another
/// member's copy fills them, which writes the record back and closes the gap
/// for good. What it wrote after it was last joined is kept, and the `oseq`
/// counter never runs backwards.
#[test]
fn thirty_days_away_drops_own_records_from_before_the_join_until_refetched() {
    let dir = TempDir::new().unwrap();
    let now = now_ms();
    let joined = now - 31 * DAY;
    let (node, old, new, copy) = {
        let (_, mut sp) = open(&dir);
        let mut old = sp.mint();
        old.hlc = (joined - DAY) << COUNTER_BITS;
        append(&mut sp, "old", old);
        let new = sp.mint();
        append(&mut sp, "new", new);
        sp.commit().unwrap();
        let copy = sp.read_event(old.origin, old.oseq).unwrap().unwrap();
        (sp.node(), old, new, copy)
    };
    set_joined(&dir, joined);

    {
        let (_, sp) = open(&dir);
        assert!(!held(&sp, node, old.oseq));
        assert!(!sp.is_applied(node, old.oseq));
        assert!(held(&sp, node, new.oseq));
        assert!(sp.is_applied(node, new.oseq));
        let mine = sp.applied_set().origin(node).unwrap().clone();
        assert_eq!(mine.gaps(), vec![(old.oseq, old.oseq)]);
        assert_eq!(mine.top(), new.oseq);
        assert_eq!(sp.own_written().top(), new.oseq);
        assert!(sp.mint().oseq > new.oseq);
    }

    // Reopened before any catch-up: the gap stands.
    let (mut substrate, mut sp) = open(&dir);
    assert!(!sp.is_applied(node, old.oseq));
    let batch = sp.apply_remote(&mut substrate, &[&copy], 1).unwrap();
    assert!(
        matches!(batch.outcomes[..], [RemoteOutcome::Written(_)]),
        "{:?}",
        batch.outcomes
    );
    assert!(sp.is_applied(node, old.oseq));
    assert!(held(&sp, node, old.oseq));
    assert!(sp.own_written().gaps().is_empty());
    sp.commit().unwrap();
    drop(sp);

    let (mut substrate, mut sp) = open(&dir);
    assert!(sp.is_applied(node, old.oseq));
    assert!(held(&sp, node, old.oseq));
    assert!(sp.own_written().gaps().is_empty());
    // And a compaction carries the closed set.
    sp.compact(&mut substrate, None).unwrap();
    drop(sp);
    let (_, sp) = open(&dir);
    assert!(sp.own_written().gaps().is_empty());
    assert_eq!(sp.open_census.get(RecordType::DebugId), 2);
}

/// An own `oseq` a purge dropped that no member holds any more is closed
/// when a catch-up says it is gone, and stays closed.
#[test]
fn an_own_gap_known_gone_is_closed() {
    let dir = TempDir::new().unwrap();
    let now = now_ms();
    let joined = now - 40 * DAY;
    let (node, old) = {
        let (_, mut sp) = open(&dir);
        let mut old = sp.mint();
        old.hlc = (joined - DAY) << COUNTER_BITS;
        append(&mut sp, "old", old);
        let new = sp.mint();
        append(&mut sp, "new", new);
        sp.commit().unwrap();
        (sp.node(), old)
    };
    set_joined(&dir, joined);
    {
        let (_, mut sp) = open(&dir);
        assert!(!sp.is_applied(node, old.oseq));
        sp.mark_gone(node, old.oseq, old.oseq);
        assert!(sp.is_applied(node, old.oseq));
        sp.commit().unwrap();
    }
    let (_, sp) = open(&dir);
    assert!(sp.is_applied(node, old.oseq));
    assert!(sp.own_written().gaps().is_empty());
}

/// A record naming this node with an `oseq` it never minted is refused: no
/// purge left that gap.
#[test]
fn an_own_record_never_minted_is_refused() {
    let dir = TempDir::new().unwrap();
    let source = TempDir::new().unwrap();
    let (forged, node) = {
        let (_, mut sp) = open(&dir);
        let node = sp.node();
        let (_, mut other) = open(&source);
        let id = RecordIdentity {
            origin: node,
            oseq: 50,
            gen: 0,
            epoch: 0,
            hlc: now_ms() << COUNTER_BITS,
        };
        append(&mut other, "forged", id);
        other.commit().unwrap();
        sp.commit().unwrap();
        (other.read_event(node, 50).unwrap().unwrap(), node)
    };
    let (mut substrate, mut sp) = open(&dir);
    assert!(sp.minter().origination().next_oseq() <= 50);
    let batch = sp.apply_remote(&mut substrate, &[&forged], 1).unwrap();
    assert!(
        matches!(
            batch.outcomes[..],
            [RemoteOutcome::Refused(Refusal::OwnOrigin)]
        ),
        "{:?}",
        batch.outcomes
    );
    assert!(!held(&sp, node, 50));
}
