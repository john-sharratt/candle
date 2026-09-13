//! A clean restart recovers every sealed turn, so the last turn's recurrent
//! memory installs on resume.
//!
//! On the hybrid lineage a sealed turn writes its recurrent snapshot at seal
//! time and the rest of its records as it moves through the tiers. A resume
//! installs the snapshot only when the recovered history reaches the turn it
//! was taken at (`snapshot_within_recovered_history`). A turn whose records do
//! not come back leaves the snapshot a turn ahead; the resume refuses it, and
//! the conversation continues with its recurrent layers at zero — fluent, and
//! forgetful. That refusal exists for a torn shutdown. After a clean one it
//! must never fire, and this is the test that says so.
//!
//! Runs on the 0.8B: the same hybrid stack as the production model, at a size
//! that loads twice in seconds.
//!
//! ```text
//! cargo test -p zend --features cuda --test restart_turn_recovery -- --test-threads=1
//! ```

mod common;

use candle::Device;
use candle_conversation::models::Model;
use common::{say, Workspace};

const MODEL: Model = Model::Qwen35_0_8B_Q8;

#[test]
fn a_clean_restart_recovers_every_sealed_turn() {
    let device = Device::new_cuda(0).expect("cuda");
    let ws = Workspace::for_model(MODEL);

    let (timeline, sealed_turns, snapshot_turn) = {
        let (engine, mut conv) = ws.open(&device);
        let timeline = conv.timeline_id();
        say(&mut conv, "Remember: the passphrase is 'harbour lantern'.");
        say(&mut conv, "Also remember: the project is called Meridian.");
        // After the scheduler has joined and the writer has flushed, every seal
        // has landed — so these two are what the disk is supposed to hold.
        engine.shutdown().expect("clean shutdown");
        let conversation = engine.conversation();
        let sealed_turns = conversation.read().turn_count(timeline);
        let snapshot = conversation
            .read_recurrent_snapshot(timeline)
            .expect("the snapshot is readable")
            .expect("a hybrid's sealed turn writes a recurrent snapshot");
        (timeline, sealed_turns, snapshot.turn_index)
    };
    assert!(
        snapshot_turn < sealed_turns,
        "before the restart the snapshot names turn {snapshot_turn} of a timeline \
         holding {sealed_turns} — the seal wrote memory for a turn it never recorded"
    );

    let (engine, _conv) = ws.open(&device);
    let recovered = engine.conversation().read().turn_count(timeline);
    assert_eq!(
        recovered, sealed_turns,
        "a clean restart recovered {recovered} of {sealed_turns} sealed turn(s); the \
         snapshot is for turn {snapshot_turn}, so the resume refuses it and the \
         conversation continues with no recurrent memory of its history"
    );
}

/// **A resumed conversation's new turns survive the next restart too.**
///
/// The daemon reconnects a client by resuming the conversation's timeline
/// (`fork_resuming`), so every turn after a conversation's first restart is
/// sealed on a resumed timeline — the shape almost every turn a daemon ever
/// seals has.
#[test]
fn a_resumed_conversation_recovers_every_sealed_turn() {
    let device = Device::new_cuda(0).expect("cuda");
    let ws = Workspace::for_model(MODEL);

    let timeline = {
        let (engine, mut conv) = ws.open(&device);
        say(&mut conv, "Remember: the passphrase is 'harbour lantern'.");
        engine.shutdown().expect("clean shutdown");
        conv.timeline_id()
    };

    let (timeline, sealed_turns, snapshot_turn) = {
        let (engine, base) = ws.open(&device);
        let mut resumed = base.fork_resuming(timeline).expect("resume the timeline");
        say(&mut resumed, "What is the passphrase?");
        engine.shutdown().expect("clean shutdown");
        let timeline = resumed.timeline_id();
        let conversation = engine.conversation();
        let sealed_turns = conversation.read().turn_count(timeline);
        let snapshot = conversation
            .read_recurrent_snapshot(timeline)
            .expect("the snapshot is readable")
            .expect("a hybrid's sealed turn writes a recurrent snapshot");
        (timeline, sealed_turns, snapshot.turn_index)
    };
    assert!(
        snapshot_turn < sealed_turns,
        "before the restart the resumed timeline's snapshot names turn \
         {snapshot_turn} of {sealed_turns} — the seal wrote memory for a turn it \
         never recorded"
    );

    let (engine, _conv) = ws.open(&device);
    let recovered = engine.conversation().read().turn_count(timeline);
    assert_eq!(
        recovered, sealed_turns,
        "a resumed conversation's restart recovered {recovered} of {sealed_turns} \
         sealed turn(s); the snapshot is for turn {snapshot_turn}, so the next resume \
         refuses it and the conversation loses its recurrent memory"
    );
}

/// **A shut-down engine something still holds does not cost the next engine on
/// its workspace any turns.**
///
/// An engine can outlive its `shutdown` while another thread holds a reference,
/// and the next engine opens the same workspace in the meantime. Nothing the
/// stale one does — while held, or when it is finally released — may reach the
/// log its successor writes.
#[test]
fn a_held_shut_down_engine_costs_its_successor_no_turns() {
    let device = Device::new_cuda(0).expect("cuda");
    let ws = Workspace::for_model(MODEL);

    let (stale_engine, mut stale_conv) = ws.open(&device);
    say(
        &mut stale_conv,
        "Remember: the passphrase is 'harbour lantern'.",
    );
    let timeline = stale_conv.timeline_id();
    stale_engine.shutdown().expect("clean shutdown");

    let (timeline, sealed_turns, snapshot_turn) = {
        let (engine, base) = ws.open(&device);
        let mut resumed = base.fork_resuming(timeline).expect("resume the timeline");
        say(&mut resumed, "What is the passphrase?");
        engine.shutdown().expect("clean shutdown");
        let timeline = resumed.timeline_id();
        let conversation = engine.conversation();
        let sealed_turns = conversation.read().turn_count(timeline);
        let snapshot = conversation
            .read_recurrent_snapshot(timeline)
            .expect("the snapshot is readable")
            .expect("a hybrid's sealed turn writes a recurrent snapshot");
        (timeline, sealed_turns, snapshot.turn_index)
    };
    drop(stale_conv);
    drop(stale_engine);

    let (engine, _conv) = ws.open(&device);
    let recovered = engine.conversation().read().turn_count(timeline);
    assert_eq!(
        recovered, sealed_turns,
        "with a shut-down engine still held on the workspace, its successor's restart \
         recovered {recovered} of {sealed_turns} sealed turn(s); the snapshot is for \
         turn {snapshot_turn}"
    );
}
