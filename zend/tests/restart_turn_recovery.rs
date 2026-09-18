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
use candle_conversation::projection::SectionLoads;
use candle_conversation::{SelectionState, TurnEvent};
use common::{say, sealed_memory_at, Workspace};

const MODEL: Model = Model::Qwen35_0_8B_Q8;

/// **A restart restores the prompt; it does not prefill it again.**
///
/// Every section the schema declares is sealed under a content address, so a
/// workspace reopened under the same schema already holds every section's K/V
/// in its redo log. Prefilling them again is a whole-prompt forward on every
/// boot — on the daemon, most of the time between "Loading model" and ready.
///
/// On this hybrid the sections persist on the six attention layers of
/// twenty-four, so a restore judged against transformer depth refuses every
/// one of them and the second boot prefills exactly what the first did.
#[test]
fn a_restart_restores_the_prompt_rather_than_prefilling_it() {
    let device = Device::new_cuda(0).expect("cuda");
    let ws = Workspace::for_model(MODEL);

    let first = {
        let (engine, _conv) = ws.open(&device);
        // The counters live on the engine's shared substrate handle, not on the
        // sequence: they count what this OPEN loaded, across every conversation.
        let loads = engine.conversation().section_loads();
        engine.shutdown().expect("clean shutdown");
        loads
    };
    assert!(
        first.prefilled > 0,
        "a fresh workspace prefills its prompt: {first:?}"
    );

    let (engine, _conv) = ws.open(&device);
    assert_eq!(
        engine.conversation().section_loads(),
        SectionLoads {
            restored: first.prefilled,
            prefilled: 0,
        },
        "the reopened workspace holds every section the first boot sealed ({first:?})"
    );
}

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

/// **A stuffed group's memory record belongs to its LAST case.**
///
/// A stuffed prefill lays several turns into one grid and seals them region by
/// region. The recurrence runs through the whole grid, so the state on the slot
/// after the forward is the state at the end of the last region — and that is
/// the only turn it may be recorded against. Exporting at every region wrote
/// that end-of-grid state under the FIRST region's turn index, then evicted it,
/// so every later region sealed with nothing: a resume installed a state ahead
/// of its K/V, and the log carried one warning per region.
#[test]
fn a_stuffed_group_records_its_memory_at_the_last_case() {
    let device = Device::new_cuda(0).expect("cuda");
    let ws = Workspace::for_model(MODEL);

    let (engine, mut conv) = ws.open(&device);
    let timeline = conv.timeline_id();
    let pad = engine
        .tokenizer()
        .token_to_id("<|im_end|>")
        .expect("the hybrid's turn terminator");
    let cases: Vec<(String, String, Vec<String>)> = (0..4)
        .map(|i| {
            (
                format!("Question {i}: what is {i} plus {i}?"),
                String::new(),
                Vec::new(),
            )
        })
        .collect();
    let (handle, regions) = conv
        .submit_prefilled_turn_group(&cases, SelectionState::default(), pad)
        .expect("submit the stuffed group");
    assert_eq!(
        regions.len(),
        4,
        "every case carries tokens, so each claims a region"
    );
    let mut response = None;
    for ev in handle.stream() {
        match ev {
            TurnEvent::Done(r) => {
                response = Some(r);
                break;
            }
            TurnEvent::Error(e) => panic!("stuffed group failed: {e}"),
            _ => {}
        }
    }
    let response = response.expect("the stream ended without Done");
    conv.finish_turn(handle, &response)
        .expect("finish the group");
    engine.shutdown().expect("clean shutdown");

    let sealed_turns = engine.conversation().read().turn_count(timeline);
    assert_eq!(sealed_turns, 4, "one sealed turn per case");
    // Polls until the record for exactly this turn is readable, and panics if
    // none has appeared within its window — before the fix the only record named
    // turn 0, so this waited out the window and failed.
    let record = sealed_memory_at(&engine, timeline, sealed_turns - 1);
    assert_eq!(record.turn_index, sealed_turns - 1);
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
