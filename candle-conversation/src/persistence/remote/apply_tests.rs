//! **Applying another member's records** (`docs/replicated_substrate.md`
//! §6.5, §14.1 "apply"): each record is deferred, skipped or written by the
//! rules, a turn is adopted only once all of it is here, in any order of
//! arrival, and a turn ranked below one held asks for a reload.
//!
//! A sender store holds the records as their writers wrote them — each under
//! the identity its writer gave it — and ships them by identity, as a member
//! reads an event it owes.

use tempfile::TempDir;

use super::super::identity::{RecordIdentity, GEN_DETACHED};
use super::super::record::{
    encode_record, ChunkPayload, RecordHeader, RecordType, TombstonePayload,
};
use super::super::resume::encode_token_ids;
use super::super::streams::{StreamDecl, TurnDecl};
use super::super::SubstratePersistence;
use super::{Incoming, Refusal, RemoteBatch, RemoteOutcome};
use crate::projection::{TimelineId, TurnIndex};
use crate::substrate::Substrate;

const A: u64 = 0xA;
const C: u64 = 0xC;
const N_LAYERS: usize = 2;
const TL: u64 = 0x3000_0001;

struct Store {
    _dir: TempDir,
    substrate: Substrate,
    sp: SubstratePersistence,
}

fn store() -> Store {
    let dir = TempDir::new().unwrap();
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    Store {
        _dir: dir,
        substrate,
        sp,
    }
}

fn at(origin: u64, hlc: u64) -> RecordIdentity {
    RecordIdentity {
        origin,
        oseq: hlc,
        gen: 0,
        epoch: 0,
        hlc: hlc << 16,
    }
}

fn chunk_payload(seed: u8) -> ChunkPayload {
    ChunkPayload {
        offset: 0,
        k_formats: vec![4, 4],
        v_formats: vec![5, 5],
        k_pal: vec![seed; 2],
        v_pal: vec![seed; 2],
        k_scale: vec![1.0],
        v_scale: vec![0.5],
        kv_bytes: (0..256u32).map(|i| (i as u8).wrapping_add(seed)).collect(),
    }
}

impl Store {
    fn append(
        &mut self,
        rt: RecordType,
        stream: u64,
        chunk: u64,
        payload: &[u8],
        id: RecordIdentity,
    ) {
        self.sp
            .append_carried(rt, 0, stream, chunk, 0, 0, payload, id)
            .unwrap();
    }

    /// One turn of a two-layer, one-block grid as `origin` wrote it from
    /// `hlc` on — its declaration, its tokens and its two chunks — returned in
    /// that order.
    fn write_turn(
        &mut self,
        turn_index: u32,
        origin: u64,
        hlc: u64,
        tokens: &[u32],
    ) -> Vec<RecordIdentity> {
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: TL,
            turn_index,
            turn_id_day: 0,
            turn_id_seq: turn_index + 1,
            role: 1,
            block_start: 0,
            block_end: 1,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = decl.stream_id().0;
        let mut ids = vec![at(origin, hlc), at(origin, hlc + 1)];
        self.append(RecordType::StreamDecl, sid, 0, &decl.encode(), ids[0]);
        self.append(
            RecordType::Tokens,
            sid,
            0,
            &encode_token_ids(tokens),
            ids[1],
        );
        for layer in 0..N_LAYERS as u64 {
            let payload = chunk_payload(layer as u8 + tokens[0] as u8);
            let id = at(origin, hlc + 2 + layer);
            self.sp
                .append_carried(
                    RecordType::Chunk,
                    4,
                    sid,
                    layer,
                    tokens.len() as u64,
                    candle::fletcher::fletcher32(&payload.kv_bytes),
                    &payload.encode(),
                    id,
                )
                .unwrap();
            ids.push(id);
        }
        ids
    }

    fn tombstone(&mut self, origin: u64, hlc: u64) -> RecordIdentity {
        let payload = TombstonePayload {
            timeline_id: TL,
            turn_index: None,
            turn_writer: 0,
            reason: None,
        }
        .encode();
        let id = at(origin, hlc);
        self.append(RecordType::Tombstone, 0, 0, &payload, id);
        id
    }

    fn events(&mut self, ids: &[RecordIdentity]) -> Vec<Vec<u8>> {
        ids.iter()
            .map(|id| self.sp.read_event(id.origin, id.oseq).unwrap().unwrap())
            .collect()
    }

    fn apply(&mut self, events: &[Vec<u8>]) -> RemoteBatch {
        let refs: Vec<&[u8]> = events.iter().map(Vec::as_slice).collect();
        self.sp
            .apply_remote(&mut self.substrate, &refs, N_LAYERS)
            .unwrap()
    }

    fn tokens(&self) -> Vec<Vec<u32>> {
        let tl = TimelineId::from_raw(TL).unwrap();
        (0..self.substrate.turn_count(tl))
            .map(|i| self.substrate.token_ids_of(tl, TurnIndex(i)))
            .collect()
    }
}

fn written(batch: &RemoteBatch) -> usize {
    batch
        .outcomes
        .iter()
        .filter(|o| matches!(o, RemoteOutcome::Written(_)))
        .count()
}

fn tl() -> TimelineId {
    TimelineId::from_raw(TL).unwrap()
}

/// In every order of arrival of a turn's four records, one record per batch,
/// the turn is adopted exactly once — by the batch that brings its last
/// piece — and holds its writer's tokens.
#[test]
fn a_turn_is_adopted_once_all_of_it_arrives_in_any_order() {
    let mut sender = store();
    let ids = sender.write_turn(0, A, 10, &[7, 8]);
    let events = sender.events(&ids);
    let mut orders = Vec::new();
    permute(&mut vec![0, 1, 2, 3], 0, &mut orders);
    assert_eq!(orders.len(), 24);
    for order in orders {
        let mut receiver = store();
        for (step, &i) in order.iter().enumerate() {
            let batch = receiver.apply(&events[i..=i]);
            assert_eq!(written(&batch), 1, "order {order:?}");
            assert!(!batch.reload);
            let last = step == order.len() - 1;
            assert_eq!(
                batch.adopted,
                if last {
                    vec![(tl(), TurnIndex(0))]
                } else {
                    Vec::new()
                },
                "order {order:?} step {step}"
            );
        }
        assert_eq!(receiver.tokens(), vec![vec![7, 8]], "order {order:?}");
    }
}

fn permute(items: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
    if k == items.len() {
        out.push(items.clone());
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permute(items, k + 1, out);
        items.swap(k, i);
    }
}

/// A turn whose predecessor has not arrived is adopted above a gap, and the
/// predecessor takes its gap when it comes, moving nothing.
#[test]
fn an_early_turn_waits_above_a_gap_its_predecessor_fills() {
    let mut sender = store();
    let first = sender.write_turn(0, A, 10, &[1]);
    let second = sender.write_turn(1, A, 20, &[2]);
    let mut receiver = store();
    let batch = receiver.apply(&sender.events(&second));
    assert_eq!(batch.adopted, vec![(tl(), TurnIndex(1))]);
    assert_eq!(receiver.tokens(), vec![vec![], vec![2]]);
    assert!(receiver.substrate.is_gap_turn(tl(), TurnIndex(0)));

    let batch = receiver.apply(&sender.events(&first));
    assert!(!batch.reload);
    assert_eq!(batch.adopted, vec![(tl(), TurnIndex(0))]);
    assert_eq!(receiver.tokens(), vec![vec![1], vec![2]]);
}

/// A turn ranked below one this member holds moves it to another index: the
/// batch asks for a reload and adopts nothing.
#[test]
fn a_turn_ranked_below_a_held_turn_asks_for_a_reload() {
    let mut sender = store();
    let held = sender.write_turn(0, A, 100, &[1]);
    let below = sender.write_turn(0, C, 50, &[9]);
    let mut receiver = store();
    assert_eq!(
        receiver.apply(&sender.events(&held)).adopted,
        vec![(tl(), TurnIndex(0))]
    );
    let batch = receiver.apply(&sender.events(&below));
    assert!(batch.reload);
    assert!(batch.adopted.is_empty());
    assert_eq!(written(&batch), 4, "the records are kept for the reload");
}

/// A turn ranked above every turn held needs no reload.
#[test]
fn a_turn_ranked_above_needs_no_reload() {
    let mut sender = store();
    let held = sender.write_turn(0, A, 50, &[1]);
    let above = sender.write_turn(0, C, 100, &[9]);
    let mut receiver = store();
    receiver.apply(&sender.events(&held));
    let batch = receiver.apply(&sender.events(&above));
    assert!(!batch.reload);
    assert_eq!(batch.adopted, vec![(tl(), TurnIndex(1))]);
    assert_eq!(receiver.tokens(), vec![vec![1], vec![9]]);
}

/// A record already applied is a duplicate, written once.
#[test]
fn a_duplicate_is_not_written_again() {
    let mut sender = store();
    let ids = sender.write_turn(0, A, 10, &[1]);
    let events = sender.events(&ids);
    let mut receiver = store();
    assert_eq!(written(&receiver.apply(&events)), 4);
    let again = receiver.apply(&events);
    assert!(again
        .outcomes
        .iter()
        .all(|o| *o == RemoteOutcome::Duplicate));
    assert!(again.adopted.is_empty());
    assert_eq!(receiver.tokens(), vec![vec![1]]);
}

/// After its timeline's tombstone, a turn's declaration is written dead — it
/// names the timeline — and the rest of the turn is dominated: marked applied,
/// never written, never adopted.
#[test]
fn a_dead_timelines_records_are_dominated() {
    let mut sender = store();
    let ids = sender.write_turn(0, A, 10, &[1]);
    let tomb = sender.tombstone(A, 50);
    let mut receiver = store();
    receiver.apply(&sender.events(&[tomb]));
    let batch = receiver.apply(&sender.events(&ids));
    assert!(matches!(batch.outcomes[0], RemoteOutcome::Written(_)));
    assert!(batch.outcomes[1..]
        .iter()
        .all(|o| *o == RemoteOutcome::Dominated));
    assert!(batch.adopted.is_empty());
    for id in &ids {
        assert!(receiver.sp.is_applied(id.origin, id.oseq));
    }
    assert_eq!(receiver.substrate.turn_count(tl()), 0);
}

/// A chunk ranked below the one its key holds is superseded: marked applied,
/// not written.
#[test]
fn a_lesser_value_is_superseded() {
    let mut sender = store();
    let early = sender.write_turn(0, A, 10, &[1]);
    let payload = chunk_payload(3);
    let decl_event = &sender.events(&early[..1])[0];
    let stream = Incoming::parse(decl_event).unwrap().header.stream_id;
    let late = at(A, 90);
    sender
        .sp
        .append_carried(
            RecordType::Chunk,
            4,
            stream,
            0,
            1,
            candle::fletcher::fletcher32(&payload.kv_bytes),
            &payload.encode(),
            late,
        )
        .unwrap();
    let mut receiver = store();
    let batch = receiver.apply(&sender.events(&[late]));
    assert_eq!(written(&batch), 1);
    let batch = receiver.apply(&sender.events(&early[2..3]));
    assert_eq!(batch.outcomes, vec![RemoteOutcome::Superseded]);
    assert!(receiver.sp.is_applied(A, early[2].oseq));
}

/// A record whose payload changed in flight is refused and stays wanted.
#[test]
fn a_record_failing_its_digest_is_refused() {
    let mut sender = store();
    let ids = sender.write_turn(0, A, 10, &[1]);
    let mut event = sender.events(&ids[1..2]).remove(0);
    let line = event.iter().position(|&b| b == b'\n').unwrap();
    event[line + 1] ^= 0xFF;
    let mut receiver = store();
    let batch = receiver.apply(&[event]);
    assert_eq!(
        batch.outcomes,
        vec![RemoteOutcome::Refused(Refusal::Digest)]
    );
    assert!(!receiver.sp.is_applied(A, ids[1].oseq));
}

/// A type this build does not decode is deferred: neither written nor
/// applied.
#[test]
fn an_unknown_type_is_deferred() {
    let mut event =
        br#"{"type":"a_type_from_the_future","payload_len":1,"crc":0,"origin":10,"oseq":5,"hlc":65536}"#
            .to_vec();
    event.push(b'\n');
    event.push(7);
    let mut receiver = store();
    let batch = receiver.apply(&[event]);
    assert_eq!(batch.outcomes, vec![RemoteOutcome::Deferred]);
    assert!(receiver.sp.is_deferred(A, 5));
    assert!(!receiver.sp.is_applied(A, 5));
}

/// A record with no identity, and one of a type that never leaves its
/// writer, are refused.
#[test]
fn unidentified_and_local_records_are_refused() {
    let mut sender = store();
    let ids = sender.write_turn(0, A, 10, &[1]);
    let tokens = Incoming::parse(&sender.events(&ids[1..2])[0]).unwrap();
    let anonymous = encode_record(
        &RecordHeader {
            record_type: RecordType::Tokens,
            payload_len: tokens.payload.len() as u64,
            stream_id: tokens.header.stream_id,
            ..RecordHeader::NONE
        },
        &tokens.payload,
    );
    let replica = encode_record(
        &RecordHeader {
            record_type: RecordType::Replica,
            payload_len: 0,
            ..RecordHeader::NONE
        }
        .with_identity(at(A, 70)),
        &[],
    );
    let mut receiver = store();
    let batch = receiver.apply(&[anonymous, replica]);
    assert_eq!(
        batch.outcomes,
        vec![
            RemoteOutcome::Refused(Refusal::Unidentified),
            RemoteOutcome::Refused(Refusal::LocalOnly(RecordType::Replica)),
        ]
    );
}

/// A record still carrying the provisional generation of a member cut off is
/// one no member sends, and is refused.
#[test]
fn a_detached_record_is_refused() {
    let mut sender = store();
    let detached = RecordIdentity {
        gen: GEN_DETACHED,
        ..at(A, 10)
    };
    let payload = TombstonePayload {
        timeline_id: TL,
        turn_index: Some(0),
        turn_writer: A,
        reason: None,
    }
    .encode();
    sender.append(RecordType::Tombstone, 0, 0, &payload, detached);
    let mut receiver = store();
    let batch = receiver.apply(&sender.events(&[detached]));
    assert_eq!(
        batch.outcomes,
        vec![RemoteOutcome::Refused(Refusal::Detached)]
    );
    assert!(!receiver.sp.is_applied(A, 10));
}

/// Applying another member's record moves this member's clock past it: the
/// next record minted here ranks above it.
#[test]
fn the_clock_passes_every_record_applied() {
    let mut sender = store();
    let far = RecordIdentity {
        hlc: u64::MAX >> 2,
        ..at(A, 10)
    };
    let payload = TombstonePayload {
        timeline_id: TL,
        turn_index: Some(0),
        turn_writer: A,
        reason: None,
    }
    .encode();
    sender.append(RecordType::Tombstone, 0, 0, &payload, far);
    let mut receiver = store();
    receiver.apply(&sender.events(&[far]));
    assert!(receiver.sp.mint().hlc > far.hlc);
}

/// What a batch wrote is where the walk finds it: a reopen holds the same
/// streams, and the applied set covers every record.
#[test]
fn a_reopen_holds_what_the_batch_wrote() {
    let mut sender = store();
    let ids = sender.write_turn(0, A, 10, &[4, 5]);
    let events = sender.events(&ids);
    let dir = TempDir::new().unwrap();
    let before = {
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        let refs: Vec<&[u8]> = events.iter().map(Vec::as_slice).collect();
        sp.apply_remote(&mut substrate, &refs, N_LAYERS).unwrap();
        sp.commit().unwrap();
        substrate.replicated_state()
    };
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    assert_eq!(substrate.replicated_state(), before);
    for id in &ids {
        assert!(sp.is_applied(id.origin, id.oseq));
    }
}
