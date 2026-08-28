//! Staged ingest provenance linkage — the storage chain a turn-level wide-Q
//! scan follows: a repo-scan turn's `ProjectionEvents` record must reference
//! the turn itself (and its predecessor) by `(timeline, index)`, its tags
//! must scope it into tagged galleries, and a summary node must inherit the
//! union of its children's tags. All of it must survive a reload from the
//! redo log.

use candle_conversation::persistence::content_hash::turn_stream_id;
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::projection::{
    decode_events, encode_events, staged_ingest_event, GroupId, LayerId, SelectedTurn, SystemItem,
    TimelineId, TurnIndex,
};
use candle_conversation::substrate::TurnPartWrite;
use candle_conversation::summary_tree::TurnKind;
use candle_conversation::turn::Role;

mod common;
use common::open_conversation;

const CODE_TAGS: [&str; 2] = ["code", "src/x.rs"];

fn code_tags() -> Vec<String> {
    CODE_TAGS.iter().map(|s| s.to_string()).collect()
}

fn record_tagged_turn(
    conv: &candle_conversation::projection::Conversation,
    timeline: TimelineId,
    tags: Vec<String>,
) -> u32 {
    conv.record_turn(
        timeline,
        Role::User,
        TurnPartWrite {
            token_count: 4,
            tags,
            ..Default::default()
        },
        |seqs| Ok(seqs.to_vec()),
    )
    .expect("record_turn")
    .0
}

fn sel_turn(timeline: TimelineId, index: u32) -> SelectedTurn {
    SelectedTurn {
        qualified: false,
        layer: "code_reading".to_string(),
        group: "scopes".to_string(),
        index,
        role: "user".to_string(),
        tokens: 4,
        kind: TurnKind::Normal,
        reason: None,
        timeline: Some(timeline.raw()),
        selected: true,
        score: 0.0,
    }
}

#[test]
fn staged_events_reference_their_turn_and_survive_reload() {
    let dir = tempfile::tempdir().unwrap();
    let timeline = TimelineId::from_raw(42).expect("timeline id");

    {
        let conv = open_conversation(dir.path());
        conv.register_timeline(
            timeline,
            LayerId::from_raw(1).expect("layer id"),
            GroupId::from_raw(1).expect("group id"),
        );

        let first = record_tagged_turn(&conv, timeline, code_tags());
        let second = record_tagged_turn(&conv, timeline, code_tags());
        assert_eq!((first, second), (0, 1));

        // The union of the two scope turns' tags — what a summary node over
        // them inherits.
        assert_eq!(
            conv.union_turn_tags(timeline, &[TurnIndex(0), TurnIndex(1)]),
            code_tags(),
            "duplicate tags dedup, order preserved"
        );
        assert!(
            conv.union_turn_tags(timeline, &[TurnIndex(9)]).is_empty(),
            "missing children contribute nothing"
        );

        // The two staged events for the second turn: user half at 0,
        // assistant half at its grid boundary — both referencing prev + self.
        let system = vec![SystemItem::Section {
            name: "frame".to_string(),
            tokens: 8,
        }];
        let turns = vec![sel_turn(timeline, 0), sel_turn(timeline, 1)];
        let events = [
            staged_ingest_event(0, 0.0, system.clone(), turns.clone()),
            staged_ingest_event(3, 0.0, system, turns),
        ];
        let stream_id = turn_stream_id(timeline.raw(), second);
        conv.persist_projection_events(stream_id, encode_events(&events));

        // Live read-back, no reload.
        {
            let read = conv.read();
            let blob = read
                .stream_of(stream_id)
                .and_then(|s| s.projection_events.clone())
                .expect("events mirrored into the live substrate");
            assert_eq!(decode_events(&blob), events.to_vec());
        }

        // Make the appended records durable before the simulated restart.
        conv.commit_persistence().expect("commit");
    }

    // Reopen from the redo log — the linkage must survive a restart.
    let conv = open_conversation(dir.path());
    let stream_id = turn_stream_id(timeline.raw(), 1);
    let read = conv.read();
    let entry = read.stream_of(stream_id).expect("stream recovered");
    let events = decode_events(entry.projection_events.as_ref().expect("events recovered"));
    assert_eq!(events.len(), 2, "user-half + assistant-half events");
    let self_ref = events[0]
        .selection
        .turns
        .last()
        .expect("self reference present");
    assert_eq!(
        (self_ref.timeline, self_ref.index),
        (Some(timeline.raw()), 1),
        "the event must name the turn its wide-Q sigs belong to"
    );
    assert_eq!(
        events[0].selection.turns[0].index, 0,
        "the predecessor scope is referenced too"
    );
    // The tags survive on the decl alongside the events.
    let decl_tags = match entry.decl.as_ref().expect("decl recovered") {
        StreamDecl::Turn(t) => t.tags.clone(),
        other => panic!("expected a Turn decl, got {other:?}"),
    };
    assert_eq!(decl_tags, code_tags());
}
