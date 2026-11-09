//! A substrate's replication primitives (`docs/replicated_substrate.md`
//! §6.3–§7.2), on CPU substrates: the tap, a rejoin's fixed records, the
//! write ticket, listing and reading for another member, and what a catch-up
//! learns is gone.

use std::sync::{Arc, Mutex};

use serde_json::json;
use tempfile::TempDir;

use super::{Conversation, GroupId, LayerId, TimelineId};
use crate::persistence::identity::GEN_DETACHED;
use crate::persistence::record::{decode_header, RecordType};
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

/// **A field written cut off does not spin the next edit of it.** The rejoin
/// fixes the record and mints under the decided generation, while the substrate
/// holds the value under the provisional one until its reload. Every identity
/// minted in between ranks below it, so an edit that only tried again never
/// returned, and said nothing: WhiteWorld's load sat in its first chain link
/// for 5.6 hours (2026-11-01).
#[test]
fn an_edit_of_a_field_written_cut_off_returns() {
    let dir = TempDir::new().unwrap();
    let conv = open(&dir);
    let timeline = conv.mint_timeline(LayerId::new(1), GroupId::new(1));
    conv.detach();
    conv.set_conversation_metadata(timeline, "forked_from", "7")
        .unwrap();
    conv.wait_written(conv.write_ticket());
    conv.rejoin_at(4).unwrap();

    let (tx, rx) = std::sync::mpsc::channel();
    let editing = conv.clone();
    std::thread::spawn(move || {
        let _ = tx.send(
            editing
                .update_conversation_metadata(timeline, "forked_from", |_, _| Some(String::new())),
        );
    });
    assert_eq!(
        rx.recv_timeout(std::time::Duration::from_secs(5))
            .expect("the edit is still spinning"),
        Some(String::new())
    );
    assert_eq!(
        conv.conversation_metadata_value(timeline, "forked_from"),
        Some(String::new())
    );
}

/// **One key is read in place**: its value for a present key, `None` for an
/// absent one and for an unregistered timeline, and a reader of several keys
/// sees exactly the keys set.
#[test]
fn one_metadata_key_is_read_without_the_bag() {
    let dir = TempDir::new().unwrap();
    let conv = open(&dir);
    let timeline = conv.mint_timeline(LayerId::new(1), GroupId::new(1));
    conv.set_conversation_metadata(timeline, "a", "1").unwrap();
    conv.set_conversation_metadata(timeline, "b", "2").unwrap();
    assert_eq!(
        conv.conversation_metadata_value(timeline, "a"),
        Some("1".to_string())
    );
    assert_eq!(conv.conversation_metadata_value(timeline, "c"), None);
    assert!(conv.has_conversation_metadata_key(timeline, "b"));
    assert!(!conv.has_conversation_metadata_key(timeline, "c"));
    let never = TimelineId::from_raw(timeline.raw() + 1_000).unwrap();
    assert_eq!(conv.conversation_metadata_value(never, "a"), None);
    assert!(!conv.has_conversation_metadata_key(never, "a"));
    assert_eq!(
        conv.conversation_metadata_with(timeline, |m| m
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>()),
        Some(vec![
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "2".to_string())
        ])
    );
    assert_eq!(conv.conversation_metadata_with(never, |m| m.len()), None);
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

/// A long range lists every record in `oseq` order, one each, and a record
/// appended a moment ago, with nothing committed, is listed with the rest: the
/// listing reads the index, not the file.
#[test]
fn a_long_range_lists_every_record_staged_or_written() {
    let dir = TempDir::new().unwrap();
    let conv = open(&dir);
    for k in 0..1031 {
        conv.put_cluster_object(&format!("cluster/k{k}"), json!(k))
            .unwrap();
    }
    let node = conv.node();
    let top = conv.persistence.call(|p| {
        p.append_record(RecordType::Tokens, 0, 9, 0, 0, 0, &[1, 2, 3])
            .unwrap();
        p.applied_through(node)
    });
    let held = conv.list_held(node, 1, top);
    assert_eq!(
        held.records.iter().map(|r| r.oseq).collect::<Vec<_>>(),
        (1..=top).collect::<Vec<_>>()
    );
    assert!(held.gone.is_empty());
    assert_eq!(held.records.last().unwrap().stream_id, 9);
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
    let held = a.list_held(node, 1, top);
    assert_eq!(held.records.len() as u64, top);
    assert!(held.gone.is_empty());
    let bytes: Vec<Vec<u8>> = (1..=top).map(|q| a.read_event(node, q).unwrap()).collect();
    for (r, b) in held.records.iter().zip(&bytes) {
        let crc_digits = decode_header(b).unwrap().0.crc.to_string().len();
        assert_eq!(
            r.len as usize,
            b.len() + 10 - crc_digits,
            "listed unpadded, its crc at its widest"
        );
    }
    let refs: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
    let batch = b.apply_remote(&refs, 0).unwrap();
    assert!(!batch.reload);
    assert_eq!(b.applied_set().through(node), top);
    assert_eq!(b.list_held(node, 1, top), held);
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

/// `oseq`s learned gone count as applied.
#[test]
fn what_a_catch_up_learns_is_gone_is_applied() {
    let dir = TempDir::new().unwrap();
    let conv = open(&dir);
    conv.record_gone(0xA, &[5, 6, 7]);
    assert!(conv.applied_set().contains(0xA, 5));
    assert!(conv.applied_set().contains(0xA, 7));
    assert!(!conv.applied_set().contains(0xA, 8));
}
