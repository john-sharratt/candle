//! Startup integrity repair — reload behavior for turns whose async records
//! were lost to a hard kill (`persistence::integrity`).
//!
//! Simulates the field failure directly: a turn's `TurnDecl` lands durably but
//! its `Tokens` record never does (the fire-and-forget writer died first). On
//! reload the verdict is `MissingTokens`; the owning layer's corrupt-turn
//! policy decides the repair — `DropConversation` (regenerable ingest content)
//! tombstones the timeline for re-ingest, `DropTurn` (dialogue, user history)
//! restores the turn and only warns.
//!
//! No CUDA required: the turns carry no sealed KV, and a workspace with no
//! chunk streams anywhere skips the KV-presence check (`compression_policy =
//! None` semantics), so the token check is exercised in isolation.

use candle_conversation::persistence::content_hash::turn_stream_id;
use candle_conversation::projection::CorruptTurnPolicy;
use candle_conversation::projection::{GroupId, LayerId, TimelineId};
use candle_conversation::substrate::TurnPartWrite;
use candle_conversation::turn::Role;

mod common;
use common::open_conversation;

const N_LAYERS: usize = 2;

/// Record one turn on `timeline`: decl always persists (as `record_turn`
/// does); the `Tokens` record persists only when `with_tokens` — omitting it
/// reproduces the async-writer loss.
fn seed_turn(
    conv: &candle_conversation::projection::Conversation,
    timeline: TimelineId,
    with_tokens: bool,
) {
    let idx = conv
        .record_turn(
            timeline,
            Role::User,
            TurnPartWrite {
                token_count: 4,
                // A declared block span = "this turn claims sealed content" —
                // the signal the integrity classifier keys on.
                block_end: 1,
                ..Default::default()
            },
            |seqs| Ok(seqs.to_vec()),
        )
        .expect("record_turn");
    if with_tokens {
        conv.persist_tokens_only(turn_stream_id(timeline.raw(), idx.0), &[1, 2, 3, 4])
            .expect("persist tokens");
    }
}

#[test]
fn lost_tokens_record_tombstones_regenerable_conversation_but_keeps_dialogue() {
    let dir = tempfile::tempdir().unwrap();
    let ingest_layer = LayerId::from_raw(1).unwrap();
    let dialogue_layer = LayerId::from_raw(2).unwrap();
    let group = GroupId::from_raw(1).unwrap();
    // One damaged conversation per policy, plus a healthy control.
    let tl_ingest = TimelineId::from_raw(100).unwrap();
    let tl_dialogue = TimelineId::from_raw(200).unwrap();
    let tl_healthy = TimelineId::from_raw(300).unwrap();

    {
        let conv = open_conversation(dir.path());
        conv.register_timeline(tl_ingest, ingest_layer, group);
        conv.register_timeline(tl_dialogue, dialogue_layer, group);
        conv.register_timeline(tl_healthy, ingest_layer, group);
        seed_turn(&conv, tl_ingest, false); // decl only — the lost-record state
        seed_turn(&conv, tl_dialogue, false); // same damage, dialogue policy
        seed_turn(&conv, tl_healthy, true); // complete
        conv.commit_persistence().expect("commit");
    }

    // Reopen exactly as the daemon does and reload.
    let conv = open_conversation(dir.path());
    conv.register_timeline(tl_ingest, ingest_layer, group);
    conv.register_timeline(tl_dialogue, dialogue_layer, group);
    conv.register_timeline(tl_healthy, ingest_layer, group);
    conv.set_layer_corrupt_turn_policy(ingest_layer, CorruptTurnPolicy::DropConversation);
    conv.set_layer_corrupt_turn_policy(dialogue_layer, CorruptTurnPolicy::DropTurn);
    conv.reconstruct_from_log(N_LAYERS, None).expect("reload");

    let read = conv.read();
    assert!(
        read.is_tombstoned(tl_ingest),
        "regenerable conversation with a lost Tokens record must be \
         tombstoned for re-ingest"
    );
    assert!(
        !read.is_tombstoned(tl_dialogue),
        "dialogue is user history — never auto-deleted for incompleteness"
    );
    assert!(
        !read.is_tombstoned(tl_healthy),
        "a complete conversation must reload untouched"
    );
    // The dialogue turn survived the reload (restored, warned about).
    assert_eq!(
        read.turn_count(tl_dialogue),
        1,
        "incomplete dialogue turn must still restore"
    );
    // The healthy turn's tokens round-tripped.
    assert_eq!(
        read.token_ids_of(tl_healthy, candle_conversation::projection::TurnIndex(0)),
        vec![1, 2, 3, 4],
        "complete turn's Tokens record must survive the reload"
    );
}

#[test]
fn repair_converges_after_reingest() {
    // The thrash guard: once the damaged conversation is re-recorded complete
    // (as the layer's refresh does after the tombstone), the next reload
    // passes it untouched — the repair fires once, not every startup.
    let dir = tempfile::tempdir().unwrap();
    let layer = LayerId::from_raw(1).unwrap();
    let group = GroupId::from_raw(1).unwrap();
    let tl_damaged = TimelineId::from_raw(100).unwrap();
    let tl_reingested = TimelineId::from_raw(101).unwrap();

    {
        let conv = open_conversation(dir.path());
        conv.register_timeline(tl_damaged, layer, group);
        seed_turn(&conv, tl_damaged, false);
        conv.commit_persistence().expect("commit");
    }
    // Reload 1: repair tombstones the damaged conversation; the refresh then
    // re-ingests the same content as a NEW timeline, complete this time.
    {
        let conv = open_conversation(dir.path());
        conv.register_timeline(tl_damaged, layer, group);
        conv.set_layer_corrupt_turn_policy(layer, CorruptTurnPolicy::DropConversation);
        conv.reconstruct_from_log(N_LAYERS, None).expect("reload 1");
        assert!(conv.read().is_tombstoned(tl_damaged));
        conv.register_timeline(tl_reingested, layer, group);
        seed_turn(&conv, tl_reingested, true);
        conv.commit_persistence().expect("commit");
    }
    // Reload 2: nothing left to repair.
    let conv = open_conversation(dir.path());
    conv.register_timeline(tl_reingested, layer, group);
    conv.set_layer_corrupt_turn_policy(layer, CorruptTurnPolicy::DropConversation);
    conv.reconstruct_from_log(N_LAYERS, None).expect("reload 2");
    let read = conv.read();
    assert!(
        !read.is_tombstoned(tl_reingested),
        "re-ingested conversation must reload untouched — repair converges"
    );
    assert_eq!(read.turn_count(tl_reingested), 1);
}
