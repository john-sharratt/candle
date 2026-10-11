//! A substrate's replication primitives (`docs/replicated_substrate.md`
//! §6.3–§7.2), on CPU substrates: the tap, a rejoin's fixed records, the
//! write ticket, listing and reading for another member, and what a catch-up
//! learns is gone.

use std::sync::{Arc, Mutex};

use serde_json::json;
use tempfile::TempDir;

use super::{Conversation, GroupId, LayerId};
use crate::persistence::deleted::{DeletedEntry, DeletedTimeline};
use crate::persistence::identity::GEN_DETACHED;
use crate::persistence::record::decode_header;
use crate::persistence::SubstratePersistence;
use crate::substrate::Substrate;

fn open(dir: &TempDir) -> Conversation {
    let mut substrate = Substrate::new();
    let sp = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
    Conversation::from_parts(substrate, sp)
}

fn tapped(conv: &Conversation) -> Arc<Mutex<Vec<u64>>> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s = seen.clone();
    conv.set_replication_tap(Box::new(move |oseq| s.lock().unwrap().push(oseq)));
    seen
}

/// Each own record is tapped once, as it is written; one written cut off is
/// not, until a rejoin fixes it; a reopen taps nothing already held.
#[test]
fn own_records_are_tapped_once_and_cut_off_ones_at_the_rejoin() {
    let dir = TempDir::new().unwrap();
    let conv = open(&dir);
    let node = conv.node();
    let seen = tapped(&conv);
    conv.put_cluster_object("cluster/a", json!(1)).unwrap();
    let first = conv.last_oseq();
    assert_eq!(*seen.lock().unwrap(), vec![first]);

    conv.detach();
    conv.put_cluster_object("cluster/b", json!(2)).unwrap();
    let cut_off = conv.last_oseq();
    assert!(conv.holds_detached());
    assert_eq!(
        *seen.lock().unwrap(),
        vec![first],
        "nothing cut off is tapped"
    );
    assert_eq!(conv.applied_set().through(node), first);
    let raw = conv.read_event(node, cut_off).unwrap();
    let (h, _) = decode_header(&raw).unwrap();
    assert_eq!(h.identity().gen, GEN_DETACHED);

    conv.rejoin_at(4).unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![first, cut_off]);
    assert!(!conv.holds_detached());
    assert_eq!(conv.generation(), 4);
    assert_eq!(conv.applied_set().through(node), cut_off);
    let fixed = conv.read_event(node, cut_off).unwrap();
    assert_eq!(decode_header(&fixed).unwrap().0.identity().gen, 4);
    drop(conv);

    let again = open(&dir);
    let seen_again = tapped(&again);
    assert_eq!(again.applied_set().through(node), cut_off);
    again.put_cluster_object("cluster/c", json!(3)).unwrap();
    assert_eq!(*seen_again.lock().unwrap(), vec![cut_off + 1]);
}

/// A write through the writer thread is in the log, and tapped, once its
/// ticket is waited for.
#[test]
fn a_ticket_waits_for_the_writer() {
    let dir = TempDir::new().unwrap();
    let conv = open(&dir);
    let seen = tapped(&conv);
    let timeline = conv.mint_timeline(LayerId::new(1), GroupId::new(1));
    conv.set_conversation_label(timeline, "named").unwrap();
    let ticket = conv.write_ticket();
    assert!(ticket >= 1);
    conv.wait_written(ticket);
    let oseq = conv.last_oseq();
    assert!(seen.lock().unwrap().contains(&oseq));
    assert!(conv.read_event(conv.node(), oseq).is_some());
}

/// What one member lists and reads of its range applies on another, which
/// then lists the same records, digests and all — and reads them out exactly
/// as the first did, unpadded.
#[test]
fn a_range_listed_and_read_applies_elsewhere() {
    let (da, db) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let a = open(&da);
    let b = open(&db);
    for k in 0..3 {
        a.put_cluster_object(&format!("cluster/k{k}"), json!(k))
            .unwrap();
    }
    let node = a.node();
    let top = a.last_oseq();
    let held = a.list_held(node, 1, top).unwrap();
    assert_eq!(held.records.len() as u64, top);
    assert!(held.gone.is_empty());
    let bytes: Vec<Vec<u8>> = (1..=top).map(|q| a.read_event(node, q).unwrap()).collect();
    for (r, b) in held.records.iter().zip(&bytes) {
        assert_eq!(r.len as usize, b.len(), "listed at its unpadded length");
    }
    let refs: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
    let batch = b.apply_remote(&refs, 0).unwrap();
    assert!(!batch.reload);
    assert_eq!(b.applied_set().through(node), top);
    assert_eq!(b.list_held(node, 1, top).unwrap(), held);
    assert_eq!(
        (1..=top)
            .map(|q| b.read_event(node, q).unwrap())
            .collect::<Vec<_>>(),
        bytes
    );
    let objects = b.cluster_objects();
    assert_eq!(objects.len(), 3);
    assert!(objects
        .iter()
        .any(|(k, v, _)| k == "cluster/k2" && *v == json!(2)));
}

/// `oseq`s learned gone count as applied; one that was an expired tombstone
/// brings its deleted-set entry, which kills its timeline here and outlives a
/// restart.
#[test]
fn what_a_catch_up_learns_is_gone_is_kept() {
    let dir = TempDir::new().unwrap();
    let conv = open(&dir);
    let timeline = conv.mint_timeline(LayerId::new(1), GroupId::new(1));
    let entry = DeletedTimeline {
        timeline: timeline.raw(),
        origin: 0xA,
        oseq: 7,
    };
    conv.record_gone(
        0xA,
        &[
            (5, None),
            (6, None),
            (7, Some(DeletedEntry::Timeline(entry))),
        ],
    )
    .unwrap();
    assert!(conv.applied_set().contains(0xA, 6));
    assert_eq!(conv.read().deleted_entry(timeline), Some(entry));
    drop(conv);
    let again = open(&dir);
    assert!(again.applied_set().contains(0xA, 7));
    assert_eq!(again.read().deleted_entry(timeline), Some(entry));
}
