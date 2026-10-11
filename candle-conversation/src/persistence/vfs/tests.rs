//! A conversation's file events through a real store: written, reopened,
//! killed, compacted and maintained.

use std::path::{Path, PathBuf};

use serde_json::json;

use super::{VfsAppend, VfsEventId, VfsEventPayload, VfsKill, VfsWrite};
use crate::persistence::identity::RecordIdentity;
use crate::persistence::maintenance::MaintenanceOp;
use crate::persistence::record::{encode_record, RecordHeader, RecordType};
use crate::persistence::segment::SegmentId;
use crate::persistence::streams::{StreamDecl, TurnDecl};
use crate::persistence::SubstratePersistence;
use crate::substrate::Substrate;

fn tmp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("vfs_{tag}_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A turn of `timeline`, whose declaration registers the timeline on reopen.
fn turn(timeline_id: u64) -> StreamDecl {
    StreamDecl::Turn(TurnDecl {
        timeline_id,
        turn_index: 0,
        turn_id_day: 0,
        turn_id_seq: 1,
        role: 2,
        block_start: 0,
        block_end: 16,
        layer_id: 1,
        group_id: 1,
        anchored_prefix: Vec::new(),
        view: Vec::new(),
        segments: Vec::new(),
        tags: Vec::new(),
    })
}

fn open(dir: &Path) -> (SubstratePersistence, Substrate) {
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir, &mut substrate).unwrap();
    (sp, substrate)
}

fn append(key: &str, n: u64) -> VfsAppend {
    VfsAppend {
        repo: "candle".into(),
        key: key.into(),
        body: json!({"kind": "delta", "n": n}),
    }
}

fn events(key: &str, ns: &[u64]) -> VfsWrite {
    VfsWrite {
        tombstones: Vec::new(),
        events: ns.iter().map(|&n| append(key, n)).collect(),
    }
}

/// Kill the events `seqs` that `sp`'s own node wrote.
fn kill(sp: &SubstratePersistence, key: &str, seqs: &[u64]) -> VfsWrite {
    kill_ids(key, seqs.iter().map(|&seq| own(sp, seq)).collect())
}

fn kill_ids(key: &str, kills: Vec<VfsEventId>) -> VfsWrite {
    VfsWrite {
        tombstones: vec![VfsKill {
            repo: "candle".into(),
            key: key.into(),
            kills,
        }],
        events: Vec::new(),
    }
}

/// The event `seq` of `sp`'s own node.
fn own(sp: &SubstratePersistence, seq: u64) -> VfsEventId {
    VfsEventId {
        origin: sp.node(),
        seq,
    }
}

/// The sequence numbers of `ids`, which `write_vfs` returned.
fn seqs(ids: Vec<VfsEventId>) -> Vec<u64> {
    ids.into_iter().map(|id| id.seq).collect()
}

/// `(key, seq, n)` of every live event of `timeline`.
fn read(sp: &mut SubstratePersistence, timeline: u64) -> Vec<(String, u64, u64)> {
    sp.vfs_events(timeline)
        .unwrap()
        .into_iter()
        .map(|VfsEventPayload { key, seq, body, .. }| (key, seq, body["n"].as_u64().unwrap()))
        .collect()
}

/// **A write is staged, not committed** — the group commit makes it durable,
/// so a tool round never waits on an `fsync` — **and reads back at once**:
/// reading commits what is staged first.
#[test]
fn a_write_is_staged_and_reads_back_at_once() {
    let dir = tmp_dir("staged");
    {
        let (mut sp, _) = open(&dir);
        sp.write_vfs(5, &events("a.rs", &[1])).unwrap();
        assert!(sp.pending_bytes() > 0, "staged, not committed");
        assert_eq!(read(&mut sp, 5), vec![("a.rs".into(), 0, 1)]);
        assert_eq!(sp.pending_bytes(), 0, "the read committed it");
    }
    let (mut sp, _) = open(&dir);
    assert_eq!(read(&mut sp, 5), vec![("a.rs".into(), 0, 1)]);
    std::fs::remove_dir_all(&dir).ok();
}

/// **Events come back as written, in key and sequence order, after a
/// reopen** — and sequence numbers carry on past them.
#[test]
fn events_come_back_after_a_reopen() {
    let dir = tmp_dir("reopen");
    {
        let (mut sp, _) = open(&dir);
        assert_eq!(
            seqs(sp.write_vfs(5, &events("b.rs", &[10, 11])).unwrap()),
            vec![0, 1]
        );
        assert_eq!(
            seqs(sp.write_vfs(5, &events("a.rs", &[12])).unwrap()),
            vec![2]
        );
        sp.commit().unwrap();
    }
    let (mut sp, _) = open(&dir);
    assert_eq!(
        read(&mut sp, 5),
        vec![
            ("a.rs".into(), 2, 12),
            ("b.rs".into(), 0, 10),
            ("b.rs".into(), 1, 11)
        ]
    );
    assert_eq!(
        sp.write_vfs(5, &events("a.rs", &[13])).unwrap(),
        vec![own(&sp, 3)]
    );
    assert!(read(&mut sp, 6).is_empty(), "another conversation has none");
    std::fs::remove_dir_all(&dir).ok();
}

/// **A tombstone kills the events it names, and they stay dead across a
/// reopen**; sequence numbers carry on past the tombstone too.
#[test]
fn a_tombstone_kills_what_it_names_for_good() {
    let dir = tmp_dir("kill");
    {
        let (mut sp, _) = open(&dir);
        sp.write_vfs(5, &events("a.rs", &[1, 2])).unwrap();
        sp.write_vfs(5, &events("b.rs", &[3])).unwrap();
        let tomb = kill(&sp, "a.rs", &[0, 1]);
        sp.write_vfs(5, &tomb).unwrap();
        assert_eq!(read(&mut sp, 5), vec![("b.rs".into(), 2, 3)]);
    }
    let (mut sp, _) = open(&dir);
    assert_eq!(read(&mut sp, 5), vec![("b.rs".into(), 2, 3)]);
    assert_eq!(
        seqs(sp.write_vfs(5, &events("a.rs", &[4])).unwrap()),
        vec![4]
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// **A batch writes its events before its tombstones** — the events take
/// the lower sequence numbers — and both hold across a reopen.
#[test]
fn a_batch_writes_events_then_tombstones() {
    let dir = tmp_dir("batch");
    {
        let (mut sp, _) = open(&dir);
        sp.write_vfs(5, &events("a.rs", &[1])).unwrap();
        let batch = VfsWrite {
            tombstones: kill(&sp, "a.rs", &[0]).tombstones,
            events: vec![append("b.rs", 2), append("b.rs", 3)],
        };
        assert_eq!(seqs(sp.write_vfs(5, &batch).unwrap()), vec![1, 2]);
        let tl = sp.vfs_index().timeline(5).unwrap();
        assert_eq!(
            tl.tombstones().keys().copied().collect::<Vec<_>>(),
            vec![own(&sp, 3)]
        );
        sp.commit().unwrap();
    }
    let (mut sp, _) = open(&dir);
    assert_eq!(
        read(&mut sp, 5),
        vec![("b.rs".into(), 1, 2), ("b.rs".into(), 2, 3)]
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// **A tombstoned conversation's events go with it** — at once, and across a
/// reopen — with no tombstone of their own.
#[test]
fn a_tombstoned_conversation_takes_its_events() {
    let dir = tmp_dir("retire");
    {
        let (mut sp, _) = open(&dir);
        sp.declare_stream(&turn(5)).unwrap();
        sp.write_vfs(5, &events("a.rs", &[1])).unwrap();
        sp.write_vfs(6, &events("a.rs", &[2])).unwrap();
        sp.write_tombstone(5, None).unwrap();
        sp.commit().unwrap();
        assert!(read(&mut sp, 5).is_empty());
        assert!(sp.vfs_index().timeline(5).is_none());
    }
    let (mut sp, _) = open(&dir);
    assert!(read(&mut sp, 5).is_empty());
    assert_eq!(read(&mut sp, 6), vec![("a.rs".into(), 0, 2)]);
    std::fs::remove_dir_all(&dir).ok();
}

/// **Compaction carries a live conversation's events and nothing else** — not
/// a retired or unregistered conversation's, not a killed event, not a
/// tombstone — and what it carries reads back the same.
#[test]
fn compaction_carries_only_live_events() {
    let dir = tmp_dir("compact");
    {
        let (mut sp, _) = open(&dir);
        sp.declare_stream(&turn(5)).unwrap();
        sp.declare_stream(&turn(7)).unwrap();
        sp.write_vfs(5, &events("a.rs", &[1, 2])).unwrap();
        let tomb = kill(&sp, "a.rs", &[0]);
        sp.write_vfs(5, &tomb).unwrap();
        sp.write_vfs(6, &events("a.rs", &[3])).unwrap();
        sp.write_vfs(7, &events("a.rs", &[4])).unwrap();
        sp.write_tombstone(7, None).unwrap();
        sp.commit().unwrap();
    }
    {
        let (mut sp, mut substrate) = open(&dir);
        sp.compact(&mut substrate, None).unwrap();
        assert_eq!(read(&mut sp, 5), vec![("a.rs".into(), 1, 2)]);
        assert!(read(&mut sp, 6).is_empty(), "never registered: an orphan");
        assert!(read(&mut sp, 7).is_empty());
        let tl = sp.vfs_index().timeline(5).unwrap();
        assert!(tl.tombstones().is_empty(), "nothing left to kill");
    }
    let (mut sp, _) = open(&dir);
    assert_eq!(read(&mut sp, 5), vec![("a.rs".into(), 1, 2)]);
    std::fs::remove_dir_all(&dir).ok();
}

/// **Maintenance carries a tombstone off the segment it compacts**, so an
/// event it killed in an older segment it leaves alone stays dead — and the
/// live events it carries read back from their new home.
#[test]
fn maintenance_keeps_a_killed_event_dead() {
    let dir = tmp_dir("maintain");
    {
        let (mut sp, _) = open(&dir);
        sp.declare_stream(&turn(5)).unwrap();
        sp.write_vfs(5, &events("a.rs", &[1])).unwrap();
        sp.write_vfs(5, &events("b.rs", &[2])).unwrap();
        sp.seal_active().unwrap();
        let tomb = kill(&sp, "a.rs", &[0]);
        sp.write_vfs(5, &tomb).unwrap();
        sp.write_vfs(5, &events("c.rs", &[3])).unwrap();
        sp.seal_active().unwrap();
    }
    {
        let (mut sp, mut substrate) = open(&dir);
        let (tomb, event) = (own(&sp, 2), own(&sp, 3));
        let tl = sp.vfs_index().timeline(5).unwrap();
        assert_eq!(tl.tombstones()[&tomb].segment, SegmentId(2));
        sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(SegmentId(2)))
            .unwrap();
        let tl = sp.vfs_index().timeline(5).unwrap();
        assert_ne!(tl.tombstones()[&tomb].segment, SegmentId(2), "carried off");
        assert_ne!(tl.events()[&event].segment, SegmentId(2), "carried off");
        assert_eq!(
            read(&mut sp, 5),
            vec![("b.rs".into(), 1, 2), ("c.rs".into(), 3, 3)]
        );
    }
    let (mut sp, _) = open(&dir);
    assert_eq!(
        read(&mut sp, 5),
        vec![("b.rs".into(), 1, 2), ("c.rs".into(), 3, 3)],
        "the event killed in segment 1 stays dead"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// **A tombstone of another member's event outlives compaction**, so the
/// event still dies when it arrives after it, however late; and **two
/// writers' events of one key** — each numbered in its own sequence — come
/// back in the order of their stamps.
#[test]
fn a_kill_of_another_members_event_outlives_compaction() {
    const B: u64 = 0xB;
    let theirs = |seq: u64, n: u64| -> Vec<u8> {
        let payload = VfsEventPayload {
            timeline_id: 5,
            origin: B,
            seq,
            repo: "candle".into(),
            key: "a.rs".into(),
            body: json!({"kind": "delta", "n": n}),
        }
        .encode();
        let header = RecordHeader {
            record_type: RecordType::VfsEvent,
            payload_len: payload.len() as u64,
            stream_id: 5,
            chunk_index: seq,
            ..RecordHeader::NONE
        }
        .with_identity(RecordIdentity {
            origin: B,
            oseq: seq + 1,
            gen: 0,
            epoch: 0,
            hlc: (seq + 1) << 16,
        });
        encode_record(&header, &payload)
    };
    let dir = tmp_dir("remote_kill");
    {
        let (mut sp, _) = open(&dir);
        sp.declare_stream(&turn(5)).unwrap();
        sp.write_vfs(5, &kill_ids("a.rs", vec![VfsEventId { origin: B, seq: 0 }]))
            .unwrap();
        sp.commit().unwrap();
    }
    {
        let (mut sp, mut substrate) = open(&dir);
        sp.compact(&mut substrate, None).unwrap();
        let tl = sp.vfs_index().timeline(5).unwrap();
        assert_eq!(tl.tombstones().len(), 1, "kept: it kills another's event");
    }
    {
        let (mut sp, mut substrate) = open(&dir);
        let late = [theirs(0, 7), theirs(1, 8)];
        let refs: Vec<&[u8]> = late.iter().map(Vec::as_slice).collect();
        sp.apply_remote(&mut substrate, &refs, 1).unwrap();
        assert_eq!(read(&mut sp, 5), vec![("a.rs".into(), 1, 8)]);
        assert_eq!(
            seqs(sp.write_vfs(5, &events("a.rs", &[9])).unwrap()),
            vec![1],
            "this member's own sequence: its tombstone took 0"
        );
        assert_eq!(
            read(&mut sp, 5),
            vec![("a.rs".into(), 1, 8), ("a.rs".into(), 1, 9)]
        );
    }
    let (mut sp, _) = open(&dir);
    assert_eq!(
        read(&mut sp, 5),
        vec![("a.rs".into(), 1, 8), ("a.rs".into(), 1, 9)]
    );
    std::fs::remove_dir_all(&dir).ok();
}
