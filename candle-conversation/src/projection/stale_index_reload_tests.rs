//! **A runtime reload reads the log as an open would.** The in-RAM stream
//! index is insert-only, so a reload made while the process runs still sees
//! records the compactor has since dropped: a tombstoned timeline's, and a
//! distilled timeline's shed chunks and tokens. Replay at open finds those
//! records gone; a runtime reload must not read them as corrupt turns, nor
//! hand a distilled turn the cold references its chunks no longer back.

use tempfile::TempDir;

use super::{CorruptTurnPolicy, LayerId, TimelineId, TurnIndex};
use crate::persistence::identity::RecordIdentity;
use crate::persistence::manifest::{ChunkLoc, RecordLoc};
use crate::persistence::record::{DistillMode, RecordType};
use crate::persistence::resume::encode_token_ids;
use crate::persistence::segment::SegmentId;
use crate::persistence::streams::{StreamDecl, StreamId, TurnDecl};
use crate::persistence::SubstratePersistence;
use crate::projection::Conversation;
use crate::substrate::Substrate;

/// A segment no store ever wrote: what a location compaction has since
/// unlinked resolves to.
const UNLINKED: SegmentId = SegmentId(9_999);

fn at(origin: u64, hlc: u64) -> RecordIdentity {
    RecordIdentity {
        origin,
        oseq: hlc,
        gen: 0,
        epoch: 0,
        hlc: hlc << 16,
    }
}

fn decl(timeline_id: u64, turn_index: u32, block_end: u64) -> StreamDecl {
    StreamDecl::Turn(TurnDecl {
        timeline_id,
        turn_index,
        turn_id_day: 0,
        turn_id_seq: turn_index + 1,
        role: 1,
        block_start: 0,
        block_end,
        layer_id: 1,
        group_id: 1,
        anchored_prefix: Vec::new(),
        view: Vec::new(),
        segments: Vec::new(),
        tags: Vec::new(),
    })
}

/// Write turn `turn_index` of `timeline` as this store's own, its decl and
/// its tokens; returns its stream.
fn write_turn(
    sp: &mut SubstratePersistence,
    timeline: u64,
    turn_index: u32,
    block_end: u64,
    hlc: u64,
) -> StreamId {
    let node = sp.node();
    let d = decl(timeline, turn_index, block_end);
    let sid = d.stream_id();
    sp.append_carried(
        RecordType::StreamDecl,
        0,
        sid.0,
        0,
        0,
        0,
        &d.encode(),
        at(node, hlc),
    )
    .unwrap();
    sp.append_carried(
        RecordType::Tokens,
        0,
        sid.0,
        0,
        0,
        0,
        &encode_token_ids(&[7, 8]),
        at(node, hlc + 1),
    )
    .unwrap();
    sid
}

/// Open `dir` as a process start does, with the dialogue policy — a corrupt
/// turn tombstoned on its own, which a test can see.
fn open(dir: &TempDir) -> (Conversation, u64) {
    let mut substrate = Substrate::new();
    let p = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    let node = p.node();
    let conv = Conversation::from_parts(substrate, p);
    conv.set_layer_corrupt_turn_policy(LayerId::new(1), CorruptTurnPolicy::DropTurn);
    conv.reconstruct_from_log(1, None).unwrap();
    (conv, node)
}

/// Point `sid`'s tokens at a segment compaction unlinked, as the insert-only
/// index still does after the compactor drops the record.
fn unlink_tokens(conv: &Conversation, sid: StreamId, node: u64) {
    let loc = RecordLoc {
        segment: UNLINKED,
        offset: 0,
        payload_len: 8,
        record_size: 4096,
    };
    conv.write().apply_tokens_loc(sid, loc, at(node, 1_000));
}

/// **A deleted conversation is not recovered by a runtime reload**: its
/// records are gone from disk, and recovering it reported it corrupt and
/// wrote it a repair tombstone.
#[test]
fn a_runtime_reload_leaves_a_tombstoned_timeline_alone() {
    const TL: u64 = 0x5000_0001;
    let dir = TempDir::new().unwrap();
    let sid = {
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        write_turn(&mut sp, TL, 0, 0, 10);
        let sid = write_turn(&mut sp, TL, 1, 0, 20);
        sp.commit().unwrap();
        sid
    };
    let (conv, node) = open(&dir);
    let tl = TimelineId::from_raw(TL).unwrap();
    assert_eq!(conv.read().turn_count(tl), 2);

    conv.write().tombstone_timeline(tl);
    unlink_tokens(&conv, sid, node);
    conv.reconstruct_from_log(1, None).unwrap();
    assert!(
        !conv.read().is_turn_tombstoned(tl, 1),
        "a deleted conversation's turn was read as corrupt"
    );
}

/// **A runtime reload leaves every timeline the turns it holds.** The replay
/// appended each declared turn after the ones already there, so a two-turn
/// timeline came back with four, the copies at indices no chunk is keyed by.
/// On `.5` (2026-10-20) a gateway restart's rejoin reloaded the live substrate,
/// and every heavy-read projection then selected turns 2–5 of seven shared
/// timelines that hold two or three — `cold-load found no chunks`, withdrawn
/// from every prompt for 70 minutes.
#[test]
fn a_runtime_reload_leaves_every_timeline_its_turns() {
    const TL: u64 = 0x5000_0003;
    let dir = TempDir::new().unwrap();
    {
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        write_turn(&mut sp, TL, 0, 0, 10);
        write_turn(&mut sp, TL, 1, 0, 20);
        sp.commit().unwrap();
    }
    let (conv, _) = open(&dir);
    let tl = TimelineId::from_raw(TL).unwrap();
    let residences = |conv: &Conversation| {
        let read = conv.read();
        (0..read.turn_count(tl))
            .map(|i| read.turn_residence(tl, TurnIndex(i)))
            .collect::<Vec<_>>()
    };
    let before = residences(&conv);
    assert_eq!(before.len(), 2);

    conv.reconstruct_from_log(1, None).unwrap();
    conv.reconstruct_from_log(1, None).unwrap();
    assert_eq!(conv.read().turn_count(tl), 2, "the reload appended copies");
    assert_eq!(
        residences(&conv),
        before,
        "a turn the log still puts at its index keeps the residence its K/V is in"
    );
    for i in 0..2 {
        assert_eq!(conv.read().token_ids_of(tl, TurnIndex(i)), vec![7, 8]);
    }
}

/// **A distilled turn's shed records are absent, not corrupt**, and it comes
/// back with no cold tier: selection must never be offered K/V that
/// compaction shed.
#[test]
fn a_runtime_reload_restores_a_distilled_turn_without_its_shed_records() {
    const TL: u64 = 0x5000_0002;
    let dir = TempDir::new().unwrap();
    let sid = {
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        let sid = write_turn(&mut sp, TL, 0, 1, 10);
        sp.commit().unwrap();
        sid
    };
    let (conv, node) = open(&dir);
    let tl = TimelineId::from_raw(TL).unwrap();
    // The turn's one chunk, as the index names it once it has sealed.
    let chunk = ChunkLoc {
        segment: UNLINKED,
        offset: 0,
        payload_len: 100,
        record_size: 4096,
        token_count: 2,
        format: 4,
    };
    conv.write().apply_chunk_loc(sid, 0, chunk, at(node, 900));
    conv.write()
        .distill_timeline(tl, DistillMode::ProvenanceOnly);
    unlink_tokens(&conv, sid, node);

    conv.reconstruct_from_log(1, None).unwrap();
    let read = conv.read();
    assert!(
        !read.is_turn_tombstoned(tl, 0),
        "a distilled turn's shed tokens were read as corruption"
    );
    for i in 0..read.turn_count(tl) {
        let tier = read.turn_tier_state(tl, TurnIndex(i)).expect("restored");
        assert!(
            !tier.cold,
            "turn {i} came back with shed K/V as its cold tier"
        );
    }
}
