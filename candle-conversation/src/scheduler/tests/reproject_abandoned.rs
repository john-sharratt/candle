//! A reprojection that fails while it holds the turn's state tells the turn.
//!
//! Between prepare and complete a reprojection holds the turn's
//! [`DecodeState`] out of `active_decodes`. A failure there once dropped it,
//! closing the turn's channel with neither `Done` nor `Error` — the caller could
//! not tell a fault from a vanished turn, evicted the conversation and kept
//! nothing of it. Measured live: `glue convention violated` at `kept prefix`,
//! then `turn ended without Done or Error`.

use super::*;
use crate::guest::Guests;

/// One decoding turn on a view of its own, its parent the slot it reads.
fn decoding_turn() -> (Scheduler, SequenceId, SequenceId, Receiver<TurnEvent>) {
    let (_tx, rx) = flume::bounded(16);
    let mut scheduler = Scheduler::new(
        rx,
        Box::new(DummyModel::new()),
        make_test_session(),
        make_dummy_tokenizer(),
        vec![0].into(),
        512,
        8,
        false,
        None,
        DecodeHealthConfig::default(),
        512,
        PersistenceTrigger::noop(),
        SummariserTrigger::noop(),
        projection_assembler::BoundaryMarkers::default(),
        Arc::new(Guests::new()),
    );
    let parent = SequenceId(scheduler.session.create_sequence().unwrap());
    let view = SequenceId(scheduler.session.create_sequence().unwrap());
    let (state, events) = boundary_state();
    scheduler.active_decodes.insert(view, state);
    (scheduler, parent, view, events)
}

/// A step of a reprojection that fails while it holds the turn's state, as
/// prepare's and complete's exits do.
fn failing_step(
    scheduler: &mut Scheduler,
    view: SequenceId,
    parent: SequenceId,
    view_live: bool,
) -> Result<(), ConversationError> {
    let state = scheduler
        .active_decodes
        .remove(&view)
        .expect("the turn is decoding");
    let failed: Result<(), ConversationError> = Err(ConversationError::Channel(
        "glue convention violated: slot 27 offset 158079 != logical_pos 158047".into(),
    ));
    or_abandon!(scheduler, view, parent, state, view_live, failed);
    Ok(())
}

/// What the turn's stream has carried: its errors, and whether it was told
/// the turn completed.
fn heard(events: &Receiver<TurnEvent>) -> (Vec<String>, bool) {
    let mut errors = Vec::new();
    let mut done = false;
    for e in events.try_iter() {
        match e {
            TurnEvent::Error(e) => errors.push(e.to_string()),
            TurnEvent::Done(_) => done = true,
            _ => {}
        }
    }
    (errors, done)
}

/// **The turn hears why, once, and is never told it completed**; the error
/// still reaches the reprojection's caller, and a view whose slot was still
/// held is freed.
#[test]
fn a_reprojection_failing_with_the_turns_state_tells_the_turn() {
    let (mut scheduler, parent, view, events) = decoding_turn();
    let returned = failing_step(&mut scheduler, view, parent, true);
    assert!(
        returned
            .unwrap_err()
            .to_string()
            .contains("glue convention violated"),
        "the caller still gets the error to log"
    );
    let (errors, done) = heard(&events);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].contains("the reprojection failed")
            && errors[0].contains("glue convention violated"),
        "{}",
        errors[0]
    );
    assert!(!done, "a turn that failed is never reported complete");
    assert!(!scheduler.active_decodes.contains_key(&view));
    assert_eq!(
        scheduler.session.sequence_offset(view.0),
        None,
        "the view's slot is freed"
    );
    assert!(
        scheduler.session.sequence_offset(parent.0).is_some(),
        "the parent is not the reprojection's to free"
    );
}

/// **After the swap freed the view, nothing is freed under its id** — it may
/// already be another view's — but the turn still hears why.
#[test]
fn a_reprojection_failing_after_the_swap_frees_nothing_and_still_tells_the_turn() {
    let (mut scheduler, parent, view, events) = decoding_turn();
    assert!(failing_step(&mut scheduler, view, parent, false).is_err());
    let (errors, done) = heard(&events);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(!done);
    assert!(
        scheduler.session.sequence_offset(view.0).is_some(),
        "an id the swap already let go is not freed again"
    );
}
