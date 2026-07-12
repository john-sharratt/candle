//! Regression: a turn recorded LIVE (in the current session, no reload) must
//! be visible to tag-scoped belief galleries immediately.
//!
//! `record_turn` persists the `TurnDecl` (carrying the gather-scope tags) to
//! the redo log AND mirrors it into the in-memory substrate. The gallery reads
//! tags from the in-memory decl — when the mirror is missing, every turn
//! sealed in the current session (notably a startup calibration corpus on a
//! fresh substrate) is invisible to the belief scan until the next daemon
//! restart's walker pass installs the decls from disk. Symptom: all-zero
//! belief scores and catalog-order tool selection despite a healthy on-disk
//! corpus.

use candle_conversation::persistence::content_hash::turn_stream_id;
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::projection::{GroupId, LayerId, TimelineId};
use candle_conversation::provenance::{encode_wide_sigs, WideQSig};
use candle_conversation::substrate::TurnPartWrite;
use candle_conversation::turn::Role;

mod common;
use common::open_conversation;

#[test]
fn live_recorded_turn_feeds_tag_scoped_gallery_without_reload() {
    let dir = tempfile::tempdir().unwrap();
    let conv = open_conversation(dir.path());

    let timeline = TimelineId::from_raw(42).expect("timeline id");
    conv.register_timeline(
        timeline,
        LayerId::from_raw(1).expect("layer id"),
        GroupId::from_raw(1).expect("group id"),
    );

    // Record a tool-tagged turn exactly as a calibration seal does — live, in
    // this session, with no reopen afterwards.
    let idx = conv
        .record_turn(
            timeline,
            Role::User,
            TurnPartWrite {
                token_count: 4,
                tags: vec!["tool".to_string(), "calculator".to_string()],
                ..Default::default()
            },
            |seqs| Ok(seqs.to_vec()),
        )
        .expect("record_turn");
    let stream_id = turn_stream_id(timeline.raw(), idx.0);

    // Attach a non-empty wide-Q signature, as the seal's provenance capture does.
    let band = vec![1.0f32; 64]; // one head, head_dim 64, all sign bits set
    let sig = WideQSig::from_band(&band, 64);
    conv.persist_wide_q_sigs(stream_id, &encode_wide_sigs(&[sig]))
        .expect("persist sigs");

    // The decl (and its tags) must be in the LIVE substrate — not only on disk.
    {
        let read = conv.read();
        let entry = read.stream_of(stream_id).expect("stream entry exists");
        let decl = entry.decl.as_ref().expect(
            "live-recorded turn must carry its decl in the in-memory substrate \
             (tag-scoped galleries read tags from it)",
        );
        match decl {
            StreamDecl::Turn(t) => {
                assert_eq!(t.tags, vec!["tool".to_string(), "calculator".to_string()]);
            }
            other => panic!("expected a Turn decl, got {other:?}"),
        }
    }

    // And the tag-scoped gallery admits it immediately.
    let (windows, slots) = conv.belief_gallery("tools", &["tool".to_string()], |name| {
        if name == "calculator" {
            Some(7)
        } else {
            None
        }
    });
    assert_eq!(
        windows.len(),
        1,
        "the live-sealed turn must enter the gallery without a restart"
    );
    assert_eq!(slots, vec![7], "slot resolves from the tool-name tag");
}
