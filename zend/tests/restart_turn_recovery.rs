//! A clean restart recovers every sealed turn and every prompt section, so the
//! last turn's recurrent memory installs on resume and nothing is prefilled
//! twice.
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
//! that loads in seconds.
//!
//! **One story, three engines on one workspace.** Every phase below is an
//! invariant of a restart, and an engine boot is the cost of one — a model
//! load, the system prompt's sections, a log replay — so the phases are the
//! successive states of one workspace rather than four workspaces booted four
//! times. Each assertion names its phase, so a failure still says which
//! invariant broke. The turns are [`say_briefly`]: they seal like any turn, and
//! nothing here reads what they say.
//!
//! The test holds [`exclusive_engine_slot`] for its whole length. It opens
//! several engines, and the wave gate is one per device, not one per engine: a
//! forward on one engine refuses every other engine's arena creation, so tests
//! run side by side refused each other's conversations mid-setup. The slot runs
//! them one at a time whatever the thread count.
//!
//! ```text
//! cargo test -p zend --features cuda --test restart_turn_recovery
//! ```

mod common;

use candle::Device;
use candle_conversation::models::Model;
use candle_conversation::projection::SectionLoads;
use candle_conversation::{ConversationEngine, Sequence};
use common::{exclusive_engine_slot, say_briefly, Workspace};

const MODEL: Model = Model::Qwen35_0_8B_Q8;

/// What an engine's log holds of a timeline once it has shut down: how many
/// turns sealed, and the turn its recurrent snapshot was taken at.
struct Sealed {
    turns: u32,
    snapshot_turn: u32,
}

/// Shut `engine` down and read what its log holds of `conv`'s timeline.
fn shut_down_and_read(engine: &ConversationEngine, conv: &Sequence) -> Sealed {
    let timeline = conv.timeline_id();
    // After the scheduler has joined and the writer has flushed, every seal has
    // landed — so these are what the disk is supposed to hold.
    engine.shutdown().expect("clean shutdown");
    let conversation = engine.conversation();
    let turns = conversation.read().turn_count(timeline);
    let snapshot = conversation
        .read_recurrent_snapshot(timeline)
        .expect("the snapshot is readable")
        .expect("a hybrid's sealed turn writes a recurrent snapshot");
    Sealed {
        turns,
        snapshot_turn: snapshot.turn_index,
    }
}

#[test]
fn a_restart_recovers_every_sealed_turn_and_every_section() {
    let _slot = exclusive_engine_slot();
    let device = Device::new_cuda(0).expect("cuda");
    let ws = Workspace::for_model(MODEL);

    // ── Phase 1: a fresh workspace seals two turns and shuts down cleanly. ──
    //
    // **A fresh workspace restores nothing and prefills its system prompt.**
    let (stale_engine, mut stale_conv) = ws.open(&device);
    let first = stale_engine.conversation().section_loads();
    assert_eq!(
        first.restored, 0,
        "phase 1: a fresh workspace has nothing to restore"
    );
    assert!(
        first.prefilled > 0,
        "phase 1: the first open prefills the system prompt"
    );
    say_briefly(
        &mut stale_conv,
        "Remember: the passphrase is 'harbour lantern'.",
    );
    say_briefly(
        &mut stale_conv,
        "Also remember: the project is called Meridian.",
    );
    let timeline = stale_conv.timeline_id();
    let sealed = shut_down_and_read(&stale_engine, &stale_conv);
    assert!(
        sealed.snapshot_turn < sealed.turns,
        "phase 1: before the restart the snapshot names turn {} of a timeline holding {} — \
         the seal wrote memory for a turn it never recorded",
        sealed.snapshot_turn,
        sealed.turns
    );

    // ── Phase 2: a second engine opens on the workspace the first still holds. ──
    //
    // The first engine is shut down but not released: an engine can outlive its
    // `shutdown` while another thread holds a reference, and the next engine
    // opens the same workspace in the meantime. Nothing the stale one does —
    // while held, or when it is finally released — may reach the log its
    // successor writes.
    let (engine, base) = ws.open(&device);

    // **A clean restart recovers every sealed turn.** A turn whose records do
    // not come back leaves the snapshot ahead of the history, the resume
    // refuses it, and the conversation loses its recurrent memory.
    let recovered = engine.conversation().read().turn_count(timeline);
    assert_eq!(
        recovered, sealed.turns,
        "phase 2: a clean restart recovered {recovered} of {} sealed turn(s), with a shut-down \
         engine still held on the workspace; the snapshot is for turn {}, so the resume \
         refuses it and the conversation continues with no recurrent memory of its history",
        sealed.turns, sealed.snapshot_turn
    );

    // **A restart restores every prompt section from the log instead of
    // prefilling it again.** Sections are content-addressed in the redo log, so
    // an unchanged prompt has nothing left to compute on the next boot. On the
    // hybrid lineage every restore used to be refused: the triage checked the
    // persisted chunk grid against the model's transformer depth, while the grid
    // holds one chunk list per KV backing — the attention layers only — so every
    // boot prefilled the whole system prompt again, one section per forward.
    let reopened = engine.conversation().section_loads();
    assert_eq!(
        reopened,
        SectionLoads {
            restored: first.prefilled,
            prefilled: 0,
        },
        "phase 2: the restart prefilled {} of the {} section(s) the first open sealed",
        reopened.prefilled,
        first.prefilled
    );

    // **A resumed conversation's new turns survive the next restart too.** The
    // daemon reconnects a client by resuming the conversation's timeline
    // (`fork_resuming`), so every turn after a conversation's first restart is
    // sealed on a resumed timeline — the shape almost every turn a daemon ever
    // seals has.
    let mut resumed = base.fork_resuming(timeline).expect("resume the timeline");
    say_briefly(&mut resumed, "What is the passphrase?");
    let resumed_timeline = resumed.timeline_id();
    let resumed_sealed = shut_down_and_read(&engine, &resumed);
    assert!(
        resumed_sealed.snapshot_turn < resumed_sealed.turns,
        "phase 2: before the next restart the resumed timeline's snapshot names turn {} of {} — \
         the seal wrote memory for a turn it never recorded",
        resumed_sealed.snapshot_turn,
        resumed_sealed.turns
    );

    // ── Phase 3: the stale engine is released, and a third engine opens. ──
    drop(stale_conv);
    drop(stale_engine);
    let (engine, _conv) = ws.open(&device);
    let recovered = engine.conversation().read().turn_count(resumed_timeline);
    assert_eq!(
        recovered, resumed_sealed.turns,
        "phase 3: a resumed conversation's restart recovered {recovered} of {} sealed turn(s) \
         after the stale engine was released; the snapshot is for turn {}, so the next resume \
         refuses it and the conversation loses its recurrent memory",
        resumed_sealed.turns, resumed_sealed.snapshot_turn
    );
}
