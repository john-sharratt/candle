//! Record identity end to end through the persistence layer: minted on every
//! append, continued across a reopen by either recovery path, re-minted for a
//! copied directory, and kept by compaction.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use tempfile::TempDir;

use super::{NodeSource, RecordIdentity};
use crate::models::identity::ModelKey;
use crate::persistence::header_index::INDEX_FLUSH_ENTRIES;
use crate::persistence::record::{ChunkPayload, RecordType};
use crate::persistence::streams::{StreamDecl, TurnDecl};
use crate::persistence::{test_npc, Admit, SubstratePersistence};
use crate::substrate::Substrate;

fn turn(timeline_id: u64) -> StreamDecl {
    StreamDecl::Turn(TurnDecl {
        timeline_id,
        turn_index: 0,
        turn_id_day: 0,
        turn_id_seq: 1,
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

fn chunk(seed: u32) -> ChunkPayload {
    ChunkPayload {
        offset: seed as u16,
        k_formats: vec![4, 4, 4, 4],
        v_formats: vec![5, 5, 5, 5],
        k_pal: vec![seed as u8; 4],
        v_pal: vec![(seed + 1) as u8; 2],
        k_scale: vec![seed as f32],
        v_scale: vec![seed as f32 + 0.25],
        kv_bytes: (0..256u32).map(|i| ((i + seed * 13) % 256) as u8).collect(),
    }
}

fn model(file: &str) -> ModelKey {
    ModelKey {
        arch: "Qwen3".into(),
        repo: "test/repo".into(),
        file: file.into(),
        weights: [1; 32],
        tokenizer: [0; 32],
        rope: "file-stated min=none".into(),
    }
}

type Seen = (RecordType, u64, u64, RecordIdentity);

/// Every record on disk in append order: type, stream, chunk index, identity.
fn on_disk(dir: &Path) -> Vec<Seen> {
    let mut seen = Vec::new();
    let mut substrate = Substrate::new();
    SubstratePersistence::open_in_with_substrate_and_sink(
        dir,
        &mut substrate,
        Admit::Unbound,
        |entry| {
            let h = &entry.record.header;
            seen.push((h.record_type, h.stream_id, h.chunk_index, h.identity()));
        },
        &|_, _, _| {},
    )
    .unwrap();
    seen
}

/// The records that carry an identity: everything but the index.
fn stamped(seen: &[Seen]) -> Vec<RecordIdentity> {
    seen.iter()
        .filter(|r| r.0 != RecordType::HeaderIndex)
        .map(|r| r.3)
        .collect()
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// **A fresh substrate numbers its records 1, 2, 3… under its own node**, each
/// with a later clock reading than the one before, and a reopen continues the
/// sequence rather than restarting it.
#[test]
fn appends_carry_the_nodes_rising_sequence_and_a_reopen_continues_it() {
    let dir = TempDir::new().unwrap();
    let decl = turn(901);
    let sid = decl.stream_id();
    let node = {
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        assert_eq!(sp.node_source(), Some(NodeSource::Minted));
        sp.declare_stream(&decl).unwrap();
        for i in 0..3u64 {
            sp.write_chunk(sid, i, 32, 4, None, &chunk(i as u32))
                .unwrap();
        }
        sp.write_npc(&test_npc(5, "Perrin")).unwrap();
        sp.commit().unwrap();
        sp.node()
    };
    assert_ne!(node, 0);

    let ids = stamped(&on_disk(dir.path()));
    assert_eq!(ids.len(), 5, "the decl, three chunks and the character");
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(
            (id.origin, id.oseq, id.gen, id.epoch),
            (node, i as u64 + 1, 0, 0),
            "record {i}"
        );
        if i > 0 {
            assert!(id.hlc > ids[i - 1].hlc, "record {i}'s reading rises");
        }
    }
    let last = *ids.last().unwrap();

    {
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        assert_eq!(sp.node_source(), Some(NodeSource::Kept));
        assert_eq!(sp.node(), node);
        assert_eq!(sp.origination().next_oseq(), last.oseq + 1);
        assert!(sp.origination().last_hlc() >= last.hlc);
        sp.write_chunk(sid, 3, 32, 4, None, &chunk(3)).unwrap();
        sp.commit().unwrap();
    }
    let newest = on_disk(dir.path())
        .into_iter()
        .rev()
        .find(|r| r.0 == RecordType::Chunk && r.2 == 3)
        .unwrap()
        .3;
    assert_eq!((newest.origin, newest.oseq), (node, last.oseq + 1));
    assert!(newest.hlc > last.hlc);
}

/// **The index chain carries identity**, so an open that recovers through it
/// instead of walking the log continues the sequence just the same.
#[test]
fn a_reopen_through_the_index_chain_continues_the_sequence() {
    let dir = TempDir::new().unwrap();
    let decl = turn(902);
    let sid = decl.stream_id();
    let n = INDEX_FLUSH_ENTRIES as u64 + 5;
    let node = {
        let mut sp = SubstratePersistence::open_in(dir.path()).unwrap();
        sp.declare_stream(&decl).unwrap();
        for i in 0..n {
            sp.write_chunk(sid, i, 32, 4, None, &chunk(i as u32))
                .unwrap();
        }
        assert!(sp.last_index().is_some(), "the run flushed an index record");
        sp.commit().unwrap();
        sp.node()
    };
    let seen = on_disk(dir.path());
    let ids = stamped(&seen);
    assert_eq!(ids.len() as u64, n + 1);
    assert!(
        ids.iter()
            .enumerate()
            .all(|(i, id)| id.origin == node && id.oseq == i as u64 + 1),
        "every record the chain replays names its node and place"
    );
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    assert!(sp.last_index().is_some(), "the reopen took the chain");
    assert_eq!(sp.origination().next_oseq(), n + 2);
}

/// **A copied directory writes under a node of its own.** The copy keeps the
/// records it was copied with, under the node that wrote them; what it writes
/// itself starts a fresh sequence under a fresh node; and the original keeps
/// its node.
#[test]
fn a_copied_directory_writes_under_a_fresh_node() {
    let original = TempDir::new().unwrap();
    let decl = turn(903);
    let sid = decl.stream_id();
    let node_a = {
        let mut sp = SubstratePersistence::open_in(original.path()).unwrap();
        sp.declare_stream(&decl).unwrap();
        sp.write_chunk(sid, 0, 32, 4, None, &chunk(0)).unwrap();
        sp.commit().unwrap();
        sp.node()
    };
    let copy = TempDir::new().unwrap();
    copy_tree(original.path(), copy.path());

    let node_b = {
        let mut sp = SubstratePersistence::open_in(copy.path()).unwrap();
        assert_eq!(
            sp.node_source(),
            Some(NodeSource::Reminted { previous: node_a })
        );
        assert_ne!(sp.node(), node_a);
        assert_eq!(sp.origination().next_oseq(), 1);
        sp.write_chunk(sid, 1, 32, 4, None, &chunk(1)).unwrap();
        sp.commit().unwrap();
        sp.node()
    };
    let seen = on_disk(copy.path());
    let of_chunk = |idx: u64| {
        seen.iter()
            .find(|r| r.0 == RecordType::Chunk && r.2 == idx)
            .unwrap()
            .3
    };
    assert_eq!((of_chunk(0).origin, of_chunk(0).oseq), (node_a, 2));
    assert_eq!((of_chunk(1).origin, of_chunk(1).oseq), (node_b, 1));

    let sp = SubstratePersistence::open_in(original.path()).unwrap();
    assert_eq!(sp.node_source(), Some(NodeSource::Kept));
    assert_eq!(sp.node(), node_a);
}

/// **Compaction rewrites the log without originating anything**: every record
/// it writes carries the identity of the live record it stands for — the
/// latest chunk body, the stream declaration, the character, the model and the
/// template — and none is minted.
#[test]
fn compaction_keeps_every_live_records_identity() {
    let dir = TempDir::new().unwrap();
    let decl = turn(904);
    let sid = decl.stream_id();
    {
        let mut sp = SubstratePersistence::open_in(dir.path()).unwrap();
        sp.declare_stream(&decl).unwrap();
        sp.bind_model(&model("first-description")).unwrap();
        sp.bind_model(&model("live-description")).unwrap();
        sp.set_template(b"the-template").unwrap();
        sp.write_npc(&test_npc(6, "Wyneth")).unwrap();
        for seed in [97, 98, 99, 7] {
            sp.write_chunk(sid, 0, 32, 4, None, &chunk(seed)).unwrap();
        }
        sp.commit_stream(sid, 0).unwrap();
        sp.commit().unwrap();
    }
    let before = on_disk(dir.path());
    let mut live: HashMap<(RecordType, u64, u64), RecordIdentity> = HashMap::new();
    for r in &before {
        live.insert((r.0, r.1, r.2), r.3);
    }
    let next_before = {
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        let next = sp.origination().next_oseq();
        sp.compact(&mut substrate, None).unwrap();
        assert_eq!(
            sp.origination().next_oseq(),
            next,
            "compaction minted nothing"
        );
        next
    };

    let after = on_disk(dir.path());
    let mut carried = 0;
    for (rt, stream, idx, id) in &after {
        if *rt == RecordType::HeaderIndex {
            assert!(id.is_none());
            continue;
        }
        assert_eq!(
            Some(id),
            live.get(&(*rt, *stream, *idx)),
            "{rt:?} stream {stream} chunk {idx} carries the live record's identity"
        );
        assert!(id.oseq < next_before);
        carried += 1;
    }
    for rt in [
        RecordType::StreamDecl,
        RecordType::Chunk,
        RecordType::Npc,
        RecordType::ModelKey,
        RecordType::Template,
    ] {
        assert!(
            after.iter().any(|r| r.0 == rt),
            "the compacted log carries a {rt:?}"
        );
    }
    assert!(
        carried < stamped(&before).len(),
        "dead records were dropped"
    );
    let chunk0 = after
        .iter()
        .find(|r| r.0 == RecordType::Chunk && r.1 == sid.0)
        .unwrap()
        .3;
    assert_eq!(Some(&chunk0), live.get(&(RecordType::Chunk, sid.0, 0)));
}
