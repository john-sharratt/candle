//! **Another member's turn across a reload** (`docs/replicated_substrate.md`
//! §6.5): a turn whose records are still arriving reloads as a gap — pending,
//! never judged corrupt, never tombstoned — and is adopted into its gap once
//! the rest of it arrives. A repair this member makes of its own turns is
//! housekeeping, written with no identity, so no other member ever applies
//! it.

use tempfile::TempDir;

use super::{Conversation, TimelineId, TurnIndex};
use crate::persistence::identity::RecordIdentity;
use crate::persistence::record::{ChunkPayload, RecordType};
use crate::persistence::resume::encode_token_ids;
use crate::persistence::streams::{StreamDecl, TurnDecl};
use crate::persistence::SubstratePersistence;
use crate::substrate::Substrate;

const A: u64 = 0xA;
const N_LAYERS: usize = 2;
const TL: u64 = 0x4000_0001;

fn at(origin: u64, hlc: u64) -> RecordIdentity {
    RecordIdentity {
        origin,
        oseq: hlc,
        gen: 0,
        epoch: 0,
        hlc: hlc << 16,
    }
}

fn decl(turn_index: u32) -> StreamDecl {
    StreamDecl::Turn(TurnDecl {
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
    })
}

fn chunk(seed: u8) -> ChunkPayload {
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

fn open(dir: &TempDir) -> (Substrate, SubstratePersistence) {
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    (substrate, sp)
}

fn reload(dir: &TempDir) -> Conversation {
    let (substrate, sp) = open(dir);
    let conv = Conversation::from_parts(substrate, sp);
    conv.reconstruct_from_log(N_LAYERS, None).unwrap();
    conv
}

fn tokens(conv: &Conversation) -> Vec<Vec<u32>> {
    let tl = TimelineId::from_raw(TL).unwrap();
    let read = conv.read();
    (0..read.turn_count(tl))
        .map(|i| read.token_ids_of(tl, TurnIndex(i)))
        .collect()
}

/// Writer `A`'s turn `turn_index`, from `hlc` on: its declaration, tokens and
/// both chunks, as the bytes a member ships.
fn turn_events(
    sender: &mut SubstratePersistence,
    turn_index: u32,
    hlc: u64,
    token: u32,
) -> Vec<Vec<u8>> {
    let d = decl(turn_index);
    let sid = d.stream_id().0;
    let mut ids = vec![at(A, hlc), at(A, hlc + 1)];
    sender
        .append_carried(RecordType::StreamDecl, 0, sid, 0, 0, 0, &d.encode(), ids[0])
        .unwrap();
    sender
        .append_carried(
            RecordType::Tokens,
            0,
            sid,
            0,
            0,
            0,
            &encode_token_ids(&[token]),
            ids[1],
        )
        .unwrap();
    for layer in 0..N_LAYERS as u64 {
        let payload = chunk(token as u8 + layer as u8);
        let id = at(A, hlc + 2 + layer);
        sender
            .append_carried(
                RecordType::Chunk,
                4,
                sid,
                layer,
                1,
                candle::fletcher::fletcher32(&payload.kv_bytes),
                &payload.encode(),
                id,
            )
            .unwrap();
        ids.push(id);
    }
    ids.iter()
        .map(|id| sender.read_event(id.origin, id.oseq).unwrap().unwrap())
        .collect()
}

/// A turn one chunk short reloads as a gap under the complete turn above it,
/// its timeline untouched by any tombstone; the missing chunk then adopts it
/// into the gap, and a further reload holds both.
#[test]
fn a_pending_turn_reloads_as_a_gap_and_is_adopted_into_it() {
    let sender_dir = TempDir::new().unwrap();
    let (_, mut sender) = open(&sender_dir);
    let first = turn_events(&mut sender, 0, 10, 1);
    let second = turn_events(&mut sender, 1, 20, 2);

    let dir = TempDir::new().unwrap();
    {
        let (mut substrate, mut sp) = open(&dir);
        let arrived: Vec<&[u8]> = first[..3]
            .iter()
            .chain(second.iter())
            .map(Vec::as_slice)
            .collect();
        let batch = sp.apply_remote(&mut substrate, &arrived, N_LAYERS).unwrap();
        assert!(!batch.reload);
        sp.commit().unwrap();
    }

    let conv = reload(&dir);
    let tl = TimelineId::from_raw(TL).unwrap();
    assert_eq!(tokens(&conv), vec![vec![], vec![2]]);
    assert!(conv.read().is_gap_turn(tl, TurnIndex(0)));
    assert!(!conv.read().is_tombstoned(tl));
    assert!(!conv.read().is_turn_tombstoned(tl, 0));

    let batch = conv.apply_remote(&[first[3].as_slice()], N_LAYERS).unwrap();
    assert_eq!(batch.adopted, vec![(tl, TurnIndex(0))]);
    assert_eq!(tokens(&conv), vec![vec![1], vec![2]]);
    conv.commit_persistence().unwrap();
    drop(conv);

    assert_eq!(tokens(&reload(&dir)), vec![vec![1], vec![2]]);
}

/// This member's own turn, found incomplete at reload, is tombstoned by the
/// repair — and the tombstone takes no `oseq`, so it is never an event.
#[test]
fn a_repair_tombstone_carries_no_identity() {
    let dir = TempDir::new().unwrap();
    let issued = {
        let (_, mut sp) = open(&dir);
        let d = decl(0);
        let sid = sp.declare_stream(&d).unwrap().value;
        sp.write_chunk(sid, 0, 1, 4, None, &chunk(1)).unwrap();
        sp.commit().unwrap();
        sp.applied_through(sp.node())
    };

    let conv = reload(&dir);
    let tl = TimelineId::from_raw(TL).unwrap();
    assert!(conv.read().is_tombstoned(tl), "the repair tombstoned it");
    conv.commit_persistence().unwrap();
    drop(conv);

    let (substrate, sp) = open(&dir);
    assert!(substrate.is_tombstoned(tl));
    assert_eq!(sp.applied_through(sp.node()), issued);
}

/// Writer `A`'s commit of turn `turn_index` at `hlc`, as the bytes it ships.
fn commit(sender: &mut SubstratePersistence, turn_index: u32, hlc: u64) -> Vec<u8> {
    let sid = decl(turn_index).stream_id().0;
    let id = at(A, hlc);
    sender
        .append_carried(RecordType::Commit, 0, sid, 1, 0, 0, &[], id)
        .unwrap();
    sender.read_event(id.origin, id.oseq).unwrap().unwrap()
}

/// **A released conversation is held complete by its own records.** Its
/// releaser names its newest commit; a member holding that commit and every
/// turn up to the one it committed holds the conversation entire, whatever
/// else of the releaser's it lacks — and one turn short of a chunk, or asked
/// through a point that is not the conversation's commit, does not.
#[test]
fn a_released_conversation_is_held_complete_by_its_own_commit() {
    let sender_dir = TempDir::new().unwrap();
    let (_, mut sender) = open(&sender_dir);
    let first = turn_events(&mut sender, 0, 10, 1);
    let second = turn_events(&mut sender, 1, 20, 2);
    let committed = commit(&mut sender, 1, 30);
    let tl = TimelineId::from_raw(TL).unwrap();

    let whole = TempDir::new().unwrap();
    let conv = reload(&whole);
    let all: Vec<&[u8]> = first
        .iter()
        .chain(second.iter())
        .chain(std::iter::once(&committed))
        .map(Vec::as_slice)
        .collect();
    conv.apply_remote(&all, N_LAYERS).unwrap();
    assert!(
        conv.holds_complete_through(tl, A, 30),
        "both turns and the commit"
    );
    assert!(
        !conv.holds_complete_through(tl, A, 31),
        "not this conversation's commit"
    );
    assert_eq!(conv.last_commit_oseq(tl, A), Some(30));

    let short = TempDir::new().unwrap();
    let conv = reload(&short);
    let some: Vec<&[u8]> = first[..3]
        .iter()
        .chain(second.iter())
        .chain(std::iter::once(&committed))
        .map(Vec::as_slice)
        .collect();
    conv.apply_remote(&some, N_LAYERS).unwrap();
    assert!(
        !conv.holds_complete_through(tl, A, 30),
        "the first turn a chunk short is a gap below the commit"
    );
}

/// **A release waits for its turn's commit.** The turn seals before the
/// persistence writer commits it, and a release sent between the two named the
/// commit before — so every turn its writer declared with K/V must carry a
/// commit before the conversation is let go; another writer's turns are not
/// this one's to wait for.
#[test]
fn a_conversation_is_committed_once_every_turn_its_writer_declared_is() {
    let sender_dir = TempDir::new().unwrap();
    let (_, mut sender) = open(&sender_dir);
    let first = turn_events(&mut sender, 0, 10, 1);
    let second = turn_events(&mut sender, 1, 20, 2);
    let first_commit = commit(&mut sender, 0, 30);
    let second_commit = commit(&mut sender, 1, 31);
    let tl = TimelineId::from_raw(TL).unwrap();

    let dir = TempDir::new().unwrap();
    let conv = reload(&dir);
    let turns: Vec<&[u8]> = first
        .iter()
        .chain(second.iter())
        .map(Vec::as_slice)
        .collect();
    conv.apply_remote(&turns, N_LAYERS).unwrap();
    assert!(
        !conv.own_turns_committed(tl, A),
        "sealed, nothing committed"
    );
    conv.apply_remote(&[first_commit.as_slice()], N_LAYERS)
        .unwrap();
    assert!(
        !conv.own_turns_committed(tl, A),
        "the newest turn's commit is still owed"
    );
    conv.apply_remote(&[second_commit.as_slice()], N_LAYERS)
        .unwrap();
    assert!(conv.own_turns_committed(tl, A));
    assert!(
        conv.own_turns_committed(tl, 0xB),
        "another writer owes nothing here"
    );
}
