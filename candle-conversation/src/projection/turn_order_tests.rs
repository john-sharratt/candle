//! **A reload places every turn at the index its stamps give it**
//! (`docs/replicated_substrate.md` §6.6).
//!
//! Two writers' turn *N* share a stream id; the reload restores both, in the
//! order of their declarations' stamps, and each restored turn carries its own
//! writer's tokens. A turn whose declaration was lost leaves a gap the reload
//! fills with an empty placeholder, so the turns above it keep their indices.

use tempfile::TempDir;

use super::{Conversation, TimelineId, TurnIndex};
use crate::persistence::identity::RecordIdentity;
use crate::persistence::record::RecordType;
use crate::persistence::resume::encode_token_ids;
use crate::persistence::streams::{StreamDecl, TurnDecl};
use crate::persistence::SubstratePersistence;
use crate::substrate::Substrate;

const A: u64 = 0xA;
const B: u64 = 0xB;

fn at(origin: u64, hlc: u64) -> RecordIdentity {
    RecordIdentity {
        origin,
        oseq: hlc,
        gen: 0,
        epoch: 0,
        hlc,
    }
}

fn decl(timeline_id: u64, turn_index: u32) -> StreamDecl {
    StreamDecl::Turn(TurnDecl {
        timeline_id,
        turn_index,
        turn_id_day: 0,
        turn_id_seq: turn_index + 1,
        role: 1,
        block_start: 0,
        block_end: 0,
        layer_id: 1,
        group_id: 1,
        anchored_prefix: Vec::new(),
        view: Vec::new(),
        segments: Vec::new(),
        tags: Vec::new(),
    })
}

/// Append turn `turn_index` of `timeline` as `origin` wrote it at `hlc`: its
/// declaration, then its tokens.
fn write_turn(
    sp: &mut SubstratePersistence,
    timeline: u64,
    turn_index: u32,
    origin: u64,
    hlc: u64,
    tokens: &[u32],
) {
    let d = decl(timeline, turn_index);
    let sid = d.stream_id().0;
    sp.append_carried(
        RecordType::StreamDecl,
        0,
        sid,
        0,
        0,
        0,
        &d.encode(),
        at(origin, hlc),
    )
    .unwrap();
    sp.append_carried(
        RecordType::Tokens,
        0,
        sid,
        0,
        0,
        0,
        &encode_token_ids(tokens),
        at(origin, hlc + 1),
    )
    .unwrap();
}

fn reload(dir: &TempDir) -> Conversation {
    let mut substrate = Substrate::new();
    let p = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    let conv = Conversation::from_parts(substrate, p);
    conv.reconstruct_from_log(1, None).unwrap();
    conv
}

fn tokens_by_index(conv: &Conversation, timeline: u64) -> Vec<Vec<u32>> {
    let tl = TimelineId::from_raw(timeline).unwrap();
    let read = conv.read();
    (0..read.turn_count(tl))
        .map(|i| read.token_ids_of(tl, TurnIndex(i)))
        .collect()
}

/// Two writers' turn 1, written in the opposite order to their stamps: the
/// reload places the lesser stamp at 1 and the greater at 2, each with its own
/// tokens, and the first writer's turn 2 above both.
#[test]
fn two_writers_turn_n_reload_in_stamp_order() {
    const TL: u64 = 0x2000_0001;
    let dir = TempDir::new().unwrap();
    {
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        write_turn(&mut sp, TL, 2, A, 50, &[30, 31]);
        write_turn(&mut sp, TL, 1, B, 40, &[20, 21]);
        write_turn(&mut sp, TL, 1, A, 20, &[10, 11]);
        write_turn(&mut sp, TL, 0, A, 10, &[1, 2]);
        sp.commit().unwrap();
    }
    let conv = reload(&dir);
    assert_eq!(
        tokens_by_index(&conv, TL),
        vec![vec![1, 2], vec![10, 11], vec![20, 21], vec![30, 31]]
    );
}

/// A lost declaration leaves its index empty: the turn above it reloads at
/// the index it was written at, not one lower.
#[test]
fn a_lost_declaration_reloads_as_a_placeholder() {
    const TL: u64 = 0x2000_0002;
    let dir = TempDir::new().unwrap();
    {
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        write_turn(&mut sp, TL, 0, A, 10, &[1, 2]);
        write_turn(&mut sp, TL, 2, A, 30, &[5, 6]);
        sp.commit().unwrap();
    }
    let conv = reload(&dir);
    assert_eq!(
        tokens_by_index(&conv, TL),
        vec![vec![1, 2], Vec::new(), vec![5, 6]]
    );
}
