//! Tests for a stream's located values — its signature window and its
//! projection events: held until this process's append lands, then where the
//! log keeps them.
//!
//! The record already held, placed again — a settle or a relocation — moves
//! only its location: the window keeps its stamp, so a resident gallery uploads
//! nothing, and no gallery epoch moves. A newer record is a new value with a
//! new stamp. A dropped segment takes the locations into it.

use super::*;
use crate::persistence::segment::SegmentId;
use crate::provenance::{encode_wide_sigs, WideQSig};

fn decl() -> StreamDecl {
    StreamDecl::Turn(TurnDecl {
        timeline_id: 77,
        turn_index: 0,
        turn_id_day: 0,
        turn_id_seq: 1,
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

/// Record `oseq` of origin 1, its clock reading `oseq`.
fn id(oseq: u64) -> RecordIdentity {
    RecordIdentity {
        origin: 1,
        oseq,
        gen: 1,
        epoch: 0,
        hlc: oseq,
    }
}

fn loc(segment: u64, offset: u64, payload_len: u64) -> RecordLoc {
    RecordLoc {
        segment: SegmentId(segment),
        offset,
        payload_len,
        record_size: 4096,
    }
}

fn window() -> Vec<u8> {
    encode_wide_sigs(&[WideQSig {
        n_heads: 12,
        words: vec![5; 24],
    }])
}

fn declared() -> (Substrate, StreamId) {
    let d = decl();
    let sid = d.stream_id();
    let mut s = Substrate::new();
    s.apply_stream_decl(sid, d, id(1));
    (s, sid)
}

/// **A held window settles to its location under the stamp it had**, and the
/// bytes leave RAM.
#[test]
fn a_held_window_settles_to_its_location_under_its_stamp() {
    let (mut s, sid) = declared();
    let bytes = window();
    s.apply_wide_q_sigs(sid, bytes.clone(), id(2));
    let held = s.wide_sig_window(sid).expect("a held window");
    assert_eq!(s.held_payload_bytes().wide_q_sigs, bytes.len() as u64);
    let epoch = s.gallery_epoch();

    s.apply_wide_q_sigs_loc(sid, loc(3, 8192, bytes.len() as u64), id(2));
    let stream = s.stream_of(sid).unwrap();
    assert_eq!(
        stream.wide_q_sigs.as_ref().and_then(Located::stored),
        Some(loc(3, 8192, bytes.len() as u64))
    );
    assert_eq!(
        s.held_payload_bytes().wide_q_sigs,
        0,
        "settled: nothing held"
    );
    assert_eq!(
        s.gallery_epoch(),
        epoch,
        "a settle changes nothing a scan reads"
    );
    let settled = s
        .wide_sig_window(sid)
        .expect("sized from the header it kept");
    assert_eq!(settled.fingerprint(), held.fingerprint());
    assert_eq!(settled.len(), 1);
}

/// **A stored value moved by maintenance keeps its stamp.**
#[test]
fn a_relocated_window_keeps_its_stamp() {
    let (mut s, sid) = declared();
    let bytes = window();
    s.apply_wide_q_sigs(sid, bytes.clone(), id(2));
    s.apply_wide_q_sigs_loc(sid, loc(3, 8192, bytes.len() as u64), id(2));
    let stamp = s.stream_of(sid).unwrap().sig_stamp;
    let epoch = s.gallery_epoch();

    s.apply_stored_loc(
        sid,
        RecordType::WideQSig,
        loc(9, 0, bytes.len() as u64),
        id(2),
    );
    let stream = s.stream_of(sid).unwrap();
    assert_eq!(
        stream.wide_q_sigs.as_ref().and_then(Located::stored),
        Some(loc(9, 0, bytes.len() as u64))
    );
    assert_eq!(stream.sig_stamp, stamp);
    assert_eq!(s.gallery_epoch(), epoch);
    assert_eq!(
        s.stored_record_of(StreamKey::of(sid, &id(2)), RecordType::WideQSig),
        Some((loc(9, 0, bytes.len() as u64), id(2)))
    );
}

/// **A newer record is a new value**: a new stamp, its header read afresh.
#[test]
fn a_newer_window_is_stamped_anew() {
    let (mut s, sid) = declared();
    let bytes = window();
    s.apply_wide_q_sigs(sid, bytes.clone(), id(2));
    let first = s.wide_sig_window(sid).unwrap().fingerprint();
    s.apply_wide_q_sigs(sid, bytes, id(3));
    let second = s.wide_sig_window(sid).unwrap().fingerprint();
    assert_ne!(first, second, "a rewritten window must re-upload");
    assert_eq!(s.stream_of(sid).unwrap().ids.wide_q_sigs, id(3));
}

/// **An older record does not replace a newer one**, held or stored.
#[test]
fn an_older_record_never_replaces_a_newer_value() {
    let (mut s, sid) = declared();
    s.apply_projection_events(sid, b"[1]".to_vec(), id(5));
    s.apply_projection_events_loc(sid, loc(2, 0, 3), id(4));
    let stream = s.stream_of(sid).unwrap();
    assert_eq!(stream.ids.projection_events, id(5));
    assert_eq!(
        stream
            .projection_events
            .as_ref()
            .and_then(Located::held)
            .map(|b| b.to_vec()),
        Some(b"[1]".to_vec())
    );
}

/// **A dropped segment takes the stored locations into it**; values elsewhere
/// stay.
#[test]
fn a_dropped_segment_takes_the_locations_into_it() {
    let (mut s, sid) = declared();
    s.apply_wide_q_sigs_loc(sid, loc(4, 0, 32), id(2));
    s.apply_projection_events_loc(sid, loc(6, 0, 3), id(3));
    s.drop_locations_into(&HashSet::from([SegmentId(4)]));
    let stream = s.stream_of(sid).unwrap();
    assert!(stream.wide_q_sigs.is_none(), "the window's segment is gone");
    assert!(stream.ids.wide_q_sigs.is_none());
    assert_eq!(
        stream.projection_events.as_ref().and_then(Located::stored),
        Some(loc(6, 0, 3)),
        "the events' segment stays"
    );
}

/// **The replicated state renders a value by identity and length**, so a
/// member holding its bytes and one holding only where they are agree.
#[test]
fn held_and_stored_render_the_same_replicated_state() {
    let (mut held, sid) = declared();
    let bytes = window();
    held.apply_wide_q_sigs(sid, bytes.clone(), id(2));
    let (mut stored, _) = declared();
    stored.apply_wide_q_sigs_loc(sid, loc(1, 4096, bytes.len() as u64), id(2));
    assert_eq!(held.replicated_state(), stored.replicated_state());
}
