//! **Adopting a store written before identities** (`docs/replicated_substrate.md`
//! §6.2, §15 chunk 1 exit): its first writable open stamps every record as
//! this node's own, once, and it reopens holding the same turns at the same
//! indices, the same labels and file events, its tombstones still naming what
//! they named, and its sections' records still admitted.

use std::collections::BTreeMap;

use tempfile::TempDir;

use super::content_hash::{hash_bytes, ContentHash};
use super::identity::RecordIdentity;
use super::manifest::encode_label_payload;
use super::record::{ChunkPayload, RecordType, TombstonePayload};
use super::resume::encode_token_ids;
use super::streams::{ContentAddress, SectionDecl, StreamDecl, TurnDecl};
use super::SubstratePersistence;
use crate::projection::{Conversation, TimelineId, TurnIndex};
use crate::substrate::{StreamKey, Substrate};

const TL: u64 = 0x5000_0001;

fn turn(turn_index: u32) -> StreamDecl {
    StreamDecl::Turn(TurnDecl {
        timeline_id: TL,
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

fn section() -> StreamDecl {
    StreamDecl::PromptSection(SectionDecl {
        address: ContentAddress {
            prefix_hash: ContentHash { lo: 3, hi: 0 },
            section_hash: hash_bytes(b"prelude"),
        },
        debug_name: "prelude".into(),
    })
}

fn chunk() -> ChunkPayload {
    ChunkPayload {
        offset: 0,
        k_formats: vec![4],
        v_formats: vec![5],
        k_pal: vec![1],
        v_pal: vec![1],
        k_scale: vec![1.0],
        v_scale: vec![0.5],
        kv_bytes: (0..128u32).map(|i| i as u8).collect(),
    }
}

/// Append `payload` as a record written before identities.
fn legacy(sp: &mut SubstratePersistence, rt: RecordType, stream: u64, chunk: u64, payload: &[u8]) {
    sp.append_carried(rt, 0, stream, chunk, 0, 0, payload, RecordIdentity::NONE)
        .unwrap();
}

/// A store as builds before identities wrote it: three turns, declared out of
/// order; a section and its chunk; a label; a turn tombstone; and a file event
/// with the tombstone that killed an earlier one.
fn write_legacy_store(dir: &TempDir) {
    let (_, mut sp) = open(dir);
    for i in [2, 0, 1] {
        let d = turn(i);
        let sid = d.stream_id().0;
        legacy(&mut sp, RecordType::StreamDecl, sid, 0, &d.encode());
        legacy(
            &mut sp,
            RecordType::Tokens,
            sid,
            0,
            &encode_token_ids(&[10 + i]),
        );
    }
    let s = section();
    let ssid = s.stream_id().0;
    legacy(&mut sp, RecordType::StreamDecl, ssid, 0, &s.encode());
    sp.append_carried(
        RecordType::Chunk,
        4,
        ssid,
        0,
        32,
        candle::fletcher::fletcher32(&chunk().kv_bytes),
        &chunk().encode(),
        RecordIdentity::NONE,
    )
    .unwrap();
    let label = encode_label_payload(TL, "conv-a", "", &BTreeMap::new());
    legacy(&mut sp, RecordType::Label, 0, 0, &label);
    let tomb = TombstonePayload {
        timeline_id: TL,
        turn_index: Some(1),
        turn_writer: 0,
        reason: None,
    };
    legacy(&mut sp, RecordType::Tombstone, 0, 0, &tomb.encode());
    legacy(
        &mut sp,
        RecordType::VfsEvent,
        TL,
        0,
        br#"{"timeline_id":1342177281,"seq":0,"repo":"r","key":"a","body":{"n":0}}"#,
    );
    legacy(
        &mut sp,
        RecordType::VfsEvent,
        TL,
        1,
        br#"{"timeline_id":1342177281,"seq":1,"repo":"r","key":"a","body":{"n":1}}"#,
    );
    legacy(
        &mut sp,
        RecordType::VfsTombstone,
        TL,
        2,
        br#"{"timeline_id":1342177281,"seq":2,"repo":"r","key":"a","kills":[0]}"#,
    );
    sp.commit().unwrap();
}

fn open(dir: &TempDir) -> (Substrate, SubstratePersistence) {
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    (substrate, sp)
}

/// Every record's identity as a further open walks it.
fn walked(dir: &TempDir) -> Vec<(RecordType, RecordIdentity)> {
    let mut substrate = Substrate::new();
    let mut seen = Vec::new();
    SubstratePersistence::open_in_with_substrate_and_sink(dir.path(), &mut substrate, |e| {
        seen.push((e.record.header.record_type, e.record.header.identity()));
    })
    .unwrap();
    seen
}

#[test]
fn a_store_written_before_identities_is_adopted_once_and_keeps_its_state() {
    let dir = TempDir::new().unwrap();
    write_legacy_store(&dir);

    let (substrate, mut sp) = open(&dir);
    let node = sp.node();
    let tl = TimelineId::from_raw(TL).unwrap();

    // The turns keep the indices they were written at, though declared out
    // of order: their stamps rise with the index.
    let order: Vec<(u32, u32)> = substrate
        .turn_streams_of(TL)
        .map(|(pos, key, rt)| {
            assert_eq!(key.writer, node);
            let StreamDecl::Turn(d) = rt.decl.as_ref().unwrap() else {
                panic!("a turn");
            };
            (pos.0, d.turn_index)
        })
        .collect();
    assert_eq!(order, vec![(0, 0), (1, 1), (2, 2)]);

    // The section's chunk carries its declaration's new generation, so the
    // section holds it.
    let ssid = section().stream_id();
    let s = substrate
        .stream_at(StreamKey {
            stream_id: ssid,
            writer: node,
        })
        .unwrap();
    assert_eq!(s.chunks.len(), 1);
    assert_eq!(s.ids.chunks[&0].epoch, s.ids.decl.oseq);

    // The label, and the turn tombstone — which names this node as the
    // turn's writer now.
    assert_eq!(
        substrate.timeline_entry(tl).unwrap().conv_id.as_deref(),
        Some("conv-a")
    );
    assert!(substrate.is_turn_tombstoned(tl, 1));
    assert!(!substrate.is_turn_tombstoned(tl, 0));

    // The file events are this node's, and the kill still holds.
    let events = sp.vfs_events(TL).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!((events[0].origin, events[0].seq), (node, 1));

    drop(sp);

    // Every record an origin writes is now this node's.
    let before = walked(&dir);
    for (rt, id) in &before {
        match rt {
            RecordType::HeaderIndex => assert!(id.is_none()),
            _ => assert_eq!(id.origin, node, "{rt:?}"),
        }
    }
    // Adopted once: a further open stamps nothing anew.
    drop(open(&dir));
    assert_eq!(walked(&dir), before);
}

/// A store whose every record carries an identity is not rewritten at open.
#[test]
fn a_stamped_store_is_not_adopted() {
    let dir = TempDir::new().unwrap();
    {
        let (_, mut sp) = open(&dir);
        sp.declare_stream(&turn(0)).unwrap();
        sp.commit().unwrap();
    }
    let before = walked(&dir);
    let (_, sp) = open(&dir);
    drop(sp);
    assert_eq!(walked(&dir), before);
    assert!(before
        .iter()
        .all(|(rt, id)| *rt == RecordType::HeaderIndex || !id.is_none()));
}

/// The adopted turns reload with their tokens at their indices.
#[test]
fn adopted_turns_reload_at_their_indices() {
    let dir = TempDir::new().unwrap();
    write_legacy_store(&dir);
    drop(open(&dir));
    let (substrate, sp) = open(&dir);
    let conv = Conversation::from_parts(substrate, sp);
    conv.reconstruct_from_log(1, None).unwrap();
    let tl = TimelineId::from_raw(TL).unwrap();
    let read = conv.read();
    let tokens: Vec<Vec<u32>> = (0..read.turn_count(tl))
        .map(|i| read.token_ids_of(tl, TurnIndex(i)))
        .collect();
    assert_eq!(tokens, vec![vec![10], vec![], vec![12]]);
}
