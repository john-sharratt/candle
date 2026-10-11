//! **Arrival order does not matter** (`docs/replicated_substrate.md` §6.9, §14.1).
//!
//! Each event set below is applied to a fresh substrate in every order —
//! exhaustively up to seven events, by seeded shuffles beyond — and the
//! replicated state ([`Substrate::replicated_state`]) must come out identical
//! every time. The sets cover competing writes to one key from two and three
//! origins, a label assembled field by field, children before their parent,
//! commits as a maximum, a retraction overtaking the snapshot it retracts,
//! records held for a timeline not registered yet, and two writers' turn *N*
//! placed in their timeline by stamp.
//!
//! The same state must also survive the log: written in an arbitrary order,
//! reopened, compacted and reopened again.

use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use tempfile::TempDir;

use super::Substrate;
use crate::persistence::identity::RecordIdentity;
use crate::persistence::manifest::{encode_conv_state_payload, encode_label_payload, ConvState};
use crate::persistence::record::{
    ChunkPayload, CustomObjectPayload, DebugIdPayload, Record, RecordHeader, RecordType,
    TreeMetadataPayload, TurnCouplingPayload,
};
use crate::persistence::segment::SegmentId;
use crate::persistence::streams::{StreamDecl, StreamId, TurnDecl};
use crate::persistence::walker::WalkEntry;
use crate::persistence::SubstratePersistence;

const A: u64 = 0xA;
const B: u64 = 0xB;
const C: u64 = 0xC;

fn at(origin: u64, hlc: u64) -> RecordIdentity {
    RecordIdentity {
        origin,
        oseq: hlc,
        gen: 0,
        epoch: 0,
        hlc,
    }
}

/// One record as it arrives: its header (identity included) and payload.
#[derive(Clone)]
struct Event {
    header: RecordHeader,
    payload: Vec<u8>,
}

fn event(
    rt: RecordType,
    stream_id: u64,
    chunk_index: u64,
    payload: Vec<u8>,
    id: RecordIdentity,
) -> Event {
    Event {
        header: RecordHeader {
            record_type: rt,
            payload_len: payload.len() as u64,
            stream_id,
            chunk_index,
            ..RecordHeader::NONE
        }
        .with_identity(id),
        payload,
    }
}

fn turn_decl(timeline_id: u64) -> StreamDecl {
    turn_decl_at(timeline_id, 0)
}

fn turn_decl_at(timeline_id: u64, turn_index: u32) -> StreamDecl {
    StreamDecl::Turn(TurnDecl {
        timeline_id,
        turn_index,
        turn_id_day: 0,
        turn_id_seq: turn_index + 1,
        role: 2,
        block_start: 0,
        block_end: 1,
        layer_id: 1,
        group_id: 1,
        anchored_prefix: Vec::new(),
        view: Vec::new(),
        segments: Vec::new(),
        tags: Vec::new(),
    })
}

fn decl(timeline_id: u64, id: RecordIdentity) -> Event {
    decl_at(timeline_id, 0, id)
}

fn decl_at(timeline_id: u64, turn_index: u32, id: RecordIdentity) -> Event {
    let d = turn_decl_at(timeline_id, turn_index);
    event(RecordType::StreamDecl, d.stream_id().0, 0, d.encode(), id)
}

fn label(tl: u64, conv_id: &str, title: &str, kv: &[(&str, &str)], id: RecordIdentity) -> Event {
    let custom: BTreeMap<String, String> = kv
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    event(
        RecordType::Label,
        0,
        0,
        encode_label_payload(tl, conv_id, title, &custom),
        id,
    )
}

fn state(tl: u64, archived: bool, active: u64, id: RecordIdentity) -> Event {
    let s = ConvState {
        archived,
        active,
        ..ConvState::default()
    };
    event(
        RecordType::ConvState,
        tl,
        0,
        encode_conv_state_payload(tl, &s),
        id,
    )
}

fn chunk(sid: StreamId, idx: u64, seed: u8, id: RecordIdentity) -> Event {
    let payload = ChunkPayload {
        offset: 0,
        k_formats: vec![4],
        v_formats: vec![5],
        k_pal: vec![seed],
        v_pal: vec![seed],
        k_scale: vec![1.0],
        v_scale: vec![1.0],
        kv_bytes: vec![seed; 64 + seed as usize],
    };
    let mut e = event(RecordType::Chunk, sid.0, idx, payload.encode(), id);
    e.header.token_count = 32;
    e.header.format = 4;
    e.header.crc = candle::fletcher::fletcher32(&payload.kv_bytes);
    e
}

fn blob(rt: RecordType, sid: u64, bytes: &[u8], id: RecordIdentity) -> Event {
    event(rt, sid, 0, bytes.to_vec(), id)
}

fn commit(sid: StreamId, through: u64, id: RecordIdentity) -> Event {
    event(RecordType::Commit, sid.0, through, Vec::new(), id)
}

fn debug_id(tl: u64, d: &str, id: RecordIdentity) -> Event {
    let p = DebugIdPayload {
        timeline_id: tl,
        debug_id: d.to_string(),
    };
    event(RecordType::DebugId, 0, 0, p.encode(), id)
}

fn tree(tl: u64, turn: u32, height: u8, id: RecordIdentity) -> Event {
    let p = TreeMetadataPayload {
        timeline_id: tl,
        turn_index: turn,
        kind: 1,
        tree_height: height,
        children: vec![turn],
    };
    event(RecordType::TreeMetadata, 0, 0, p.encode(), id)
}

fn coupling(tl: u64, from: u32, id: RecordIdentity) -> Event {
    let p = TurnCouplingPayload {
        timeline_id: tl,
        from_turn: from,
    };
    event(RecordType::TurnCoupling, 0, 0, p.encode(), id)
}

fn object(key: &str, n: u64, id: RecordIdentity) -> Event {
    let obj = CustomObjectPayload::new(key, serde_json::json!({ "n": n }));
    event(
        RecordType::CustomObject,
        obj.stream_id(),
        0,
        obj.encode(),
        id,
    )
}

const T1: u64 = 0x1000_0001;
const T2: u64 = 0x1000_0002;
const T3: u64 = 0x1000_0003;
const T4: u64 = 0x1000_0004;

/// A label assembled field by field from three origins, its state written
/// twice, all arriving around the declaration that registers the timeline.
fn labels_and_state() -> Vec<Event> {
    vec![
        label(T1, "c1", "", &[], at(A, 10)),
        label(T1, "", "title", &[], at(B, 12)),
        label(T1, "", "", &[("k", "v1")], at(A, 11)),
        label(T1, "", "", &[("k", "v2"), ("j", "w")], at(C, 13)),
        decl(T1, at(A, 9)),
        state(T1, true, 0, at(B, 14)),
        state(T1, false, 5, at(A, 15)),
    ]
}

/// A turn's keyed values: a chunk re-sealed, a commit as a maximum (the
/// greater mark written first), its tokens and its signature.
fn stream_values() -> Vec<Event> {
    let sid = turn_decl(T2).stream_id();
    vec![
        decl(T2, at(A, 1)),
        chunk(sid, 0, 1, at(A, 2)),
        chunk(sid, 0, 2, at(A, 3)),
        blob(RecordType::Tokens, sid.0, b"tokens-a", at(A, 4)),
        commit(sid, 7, at(A, 5)),
        commit(sid, 3, at(A, 6)),
        blob(RecordType::WideQSig, sid.0, b"sig-a", at(A, 7)),
    ]
}

const T5: u64 = 0x1000_0005;

/// Two writers' turn 1 of one timeline (§6.6): one stream id, two streams,
/// each with its own chunk and tokens, placed by their declarations' stamps —
/// and a later turn of the first writer placed above both.
fn two_writers_turn_n() -> Vec<Event> {
    let sid1 = turn_decl_at(T5, 1).stream_id();
    vec![
        decl_at(T5, 0, at(A, 1)),
        decl_at(T5, 1, at(A, 2)),
        decl_at(T5, 1, at(B, 3)),
        chunk(sid1, 0, 3, at(A, 4)),
        chunk(sid1, 0, 4, at(B, 5)),
        blob(RecordType::Tokens, sid1.0, b"tokens-b", at(B, 6)),
        decl_at(T5, 2, at(A, 7)),
    ]
}

/// Children of a timeline that is declared last, and a custom object written
/// by two origins.
fn children_first() -> Vec<Event> {
    vec![
        debug_id(T3, "d1", at(A, 3)),
        debug_id(T3, "d2", at(B, 4)),
        tree(T3, 0, 1, at(A, 5)),
        coupling(T3, 0, at(A, 6)),
        object("k", 1, at(A, 7)),
        object("k", 2, at(B, 8)),
        decl(T3, at(C, 2)),
    ]
}

/// A snapshot overtaken by its retraction, an older snapshot arriving after
/// both, competing branch checkpoints, and a turn's projection events and
/// index page.
fn recurrent_and_blobs() -> Vec<Event> {
    let snap = StreamId(0x5_0000_0001);
    let branch = StreamId(0x5_0000_0002);
    let sid = turn_decl(T4).stream_id();
    vec![
        blob(RecordType::Snapshot, snap.0, b"snapshot-10", at(A, 10)),
        blob(RecordType::Snapshot, snap.0, b"snapshot-11!", at(B, 11)),
        blob(RecordType::Snapshot, snap.0, b"", at(A, 12)),
        blob(RecordType::Snapshot, snap.0, b"snapshot-9", at(C, 9)),
        blob(
            RecordType::BranchCheckpoint,
            branch.0,
            b"branch-a",
            at(A, 3),
        ),
        blob(
            RecordType::BranchCheckpoint,
            branch.0,
            b"branch-bb",
            at(B, 4),
        ),
        blob(RecordType::ProjectionEvents, sid.0, b"[]", at(A, 5)),
    ]
}

fn walked(e: &Event, position: usize) -> WalkEntry {
    WalkEntry {
        segment: SegmentId(1),
        offset: 4096 * (position as u64 + 1),
        record: Record {
            header: e.header,
            payload: e.payload.clone(),
        },
        size: 4096,
    }
}

fn apply_in(events: &[Event], order: &[usize]) -> String {
    let mut sub = Substrate::new();
    for (position, &i) in order.iter().enumerate() {
        sub.apply_walker_entry(&walked(&events[i], position));
    }
    sub.replicated_state()
}

/// Every ordering of `0..n`.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn heap(k: usize, a: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if k == 1 {
            out.push(a.clone());
            return;
        }
        for i in 0..k {
            heap(k - 1, a, out);
            if k.is_multiple_of(2) {
                a.swap(i, k - 1);
            } else {
                a.swap(0, k - 1);
            }
        }
    }
    let mut a: Vec<usize> = (0..n).collect();
    let mut out = Vec::new();
    heap(n, &mut a, &mut out);
    out
}

/// Every ordering of `events` gives the state the first ordering gives.
fn assert_order_free(events: &[Event]) -> String {
    let orders = permutations(events.len());
    let reference = apply_in(events, &orders[0]);
    for order in &orders[1..] {
        let got = apply_in(events, order);
        assert_eq!(got, reference, "order {order:?} diverged");
    }
    reference
}

#[test]
fn a_label_assembled_from_three_origins_is_the_same_in_every_order() {
    let state = assert_order_free(&labels_and_state());
    // And it is the state the stamps say: each field's greatest record, the
    // greatest whole state, ranked by the least record that named it.
    assert!(state.contains("conv_id a/10@0.0.10 c1"), "{state}");
    assert!(state.contains("label b/12@0.0.12 title"), "{state}");
    assert!(state.contains("custom c/13@0.0.13 k=v2"), "{state}");
    assert!(state.contains("custom c/13@0.0.13 j=w"), "{state}");
    assert!(state.contains("order=10"), "{state}");
    assert!(
        state.contains("state a/15@0.0.15 archived=false active=5"),
        "{state}"
    );
}

#[test]
fn a_turns_values_are_the_same_in_every_order() {
    let state = assert_order_free(&stream_values());
    assert!(state.contains("chunk 0 a/3@0.0.3"), "{state}");
    assert!(state.contains("committed 7 a/5@0.0.5"), "{state}");
}

#[test]
fn two_writers_turn_n_is_the_same_in_every_order() {
    let state = assert_order_free(&two_writers_turn_n());
    let sid = |n: u32| turn_decl_at(T5, n).stream_id().0;
    let order = format!(
        "turns {T5:016x}\n  0 {:016x} writer a\n  1 {:016x} writer a\n  2 {:016x} writer b\n  3 {:016x} writer a\n",
        sid(0),
        sid(1),
        sid(1),
        sid(2)
    );
    assert!(state.contains(&order), "{state}");
    // Both writers' chunk 0 of the shared stream id are kept.
    assert!(state.contains("chunk 0 a/4@0.0.4"), "{state}");
    assert!(state.contains("chunk 0 b/5@0.0.5"), "{state}");
}

/// Every lookup by a turn's id reaches the turn at that position: the second
/// writer's turn 1 is reached as turn 2, with its own tokens, and the first
/// writer's turn 2 as turn 3.
#[test]
fn a_lookup_by_turn_index_reaches_the_turn_at_that_position() {
    let events = two_writers_turn_n();
    let mut sub = Substrate::new();
    for (position, e) in events.iter().enumerate().rev() {
        sub.apply_walker_entry(&walked(e, position));
    }
    let at_index = |n: u32| sub.stream_of(turn_decl_at(T5, n).stream_id()).unwrap();
    assert_eq!(at_index(1).ids.decl, at(A, 2));
    assert_eq!(at_index(2).ids.decl, at(B, 3));
    assert_eq!(at_index(2).ids.tokens, at(B, 6));
    assert!(at_index(1).tokens.is_none());
    assert_eq!(at_index(3).ids.decl, at(A, 7));
    let StreamDecl::Turn(d) = at_index(3).decl.as_ref().unwrap() else {
        unreachable!()
    };
    assert_eq!(d.turn_index, 2, "the decl is as its writer wrote it");
    assert!(sub.stream_of(turn_decl_at(T5, 4).stream_id()).is_none());
}

#[test]
fn children_arriving_before_their_timeline_are_the_same_in_every_order() {
    let state = assert_order_free(&children_first());
    assert!(state.contains("debug_id b/4@0.0.4 d2"), "{state}");
    assert!(state.contains("coupling 0 a/6@0.0.6"), "{state}");
    assert!(
        !state.contains("pending"),
        "everything held was applied: {state}"
    );
}

#[test]
fn a_retraction_and_the_snapshots_around_it_are_the_same_in_every_order() {
    let state = assert_order_free(&recurrent_and_blobs());
    assert!(state.contains("a/12@0.0.12 retracted"), "{state}");
    assert!(
        state.contains("branch") && state.contains("b/4@0.0.4 len=9"),
        "{state}"
    );
}

fn all_events() -> Vec<Event> {
    let mut all = Vec::new();
    all.extend(labels_and_state());
    all.extend(stream_values());
    all.extend(children_first());
    all.extend(recurrent_and_blobs());
    all.extend(two_writers_turn_n());
    all
}

/// Beyond seven events, seeded shuffles of the whole set.
#[test]
fn the_whole_set_is_the_same_under_seeded_shuffles() {
    let events = all_events();
    let reference = apply_in(&events, &(0..events.len()).collect::<Vec<_>>());
    let mut rng = StdRng::seed_from_u64(0x5EED_0001);
    let mut order: Vec<usize> = (0..events.len()).collect();
    for round in 0..5_000 {
        order.shuffle(&mut rng);
        assert_eq!(
            apply_in(&events, &order),
            reference,
            "shuffle {round}: {order:?}"
        );
    }
}

/// Write `events` into a fresh log in `order`, applying each as it lands —
/// the substrate and the log as a member that received them would hold them.
fn write_in(dir: &std::path::Path, events: &[Event], order: &[usize]) -> String {
    let mut sub = Substrate::new();
    let mut sp = SubstratePersistence::open_in_with_substrate(dir, &mut sub).unwrap();
    for &i in order {
        let e = &events[i];
        let h = &e.header;
        let placed = sp
            .append_carried(
                h.record_type,
                h.format,
                h.stream_id,
                h.chunk_index,
                h.token_count,
                h.crc,
                &e.payload,
                h.identity(),
            )
            .unwrap();
        sub.apply_walker_entry(&WalkEntry {
            segment: placed.segment,
            offset: placed.offset,
            record: Record {
                header: e.header,
                payload: e.payload.clone(),
            },
            size: placed.size,
        });
    }
    sp.commit().unwrap();
    sub.replicated_state()
}

/// Assert two renderings are equal, naming the lines each has that the other
/// does not.
fn assert_same(got: &str, want: &str, context: &str) {
    if got == want {
        return;
    }
    let got_lines: Vec<&str> = got.lines().collect();
    let want_lines: Vec<&str> = want.lines().collect();
    let extra: Vec<&&str> = got_lines
        .iter()
        .filter(|l| !want_lines.contains(l))
        .collect();
    let missing: Vec<&&str> = want_lines
        .iter()
        .filter(|l| !got_lines.contains(l))
        .collect();
    panic!("{context}: state diverged\n  unexpected: {extra:#?}\n  missing: {missing:#?}");
}

fn reopened(dir: &std::path::Path) -> (Substrate, SubstratePersistence) {
    let mut sub = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir, &mut sub).unwrap();
    (sub, sp)
}

/// **The state survives the log**: written in an arbitrary order, it reopens
/// to the same state, and so does a compaction of it and a reopen after that.
#[test]
fn the_state_survives_a_reopen_and_a_compaction_in_any_write_order() {
    let events = all_events();
    let reference = apply_in(&events, &(0..events.len()).collect::<Vec<_>>());
    let mut rng = StdRng::seed_from_u64(0x5EED_0002);
    let mut order: Vec<usize> = (0..events.len()).collect();
    for round in 0..8 {
        order.shuffle(&mut rng);
        let dir = TempDir::new().unwrap();
        let written = write_in(dir.path(), &events, &order);
        assert_same(&written, &reference, &format!("round {round}: as written"));

        let (mut sub, mut sp) = reopened(dir.path());
        assert_same(
            &sub.replicated_state(),
            &reference,
            &format!("round {round}: reopened"),
        );
        sp.compact(&mut sub, None).unwrap();
        assert_same(
            &sub.replicated_state(),
            &reference,
            &format!("round {round}: compacted"),
        );
        drop(sp);

        let (sub, _sp) = reopened(dir.path());
        assert_same(
            &sub.replicated_state(),
            &reference,
            &format!("round {round}: reopened after compaction"),
        );
    }
}
