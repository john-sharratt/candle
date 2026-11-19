//! An append-only conversation keeps its lineage in its context.
//!
//! An append-only conversation (`disable_reprojection`) forked from a base
//! that never ran a turn carries the base's frame: its first turn lays the
//! system prompt onto its empty slot instead of projecting. A fork that
//! inherits a lineage — a calibration draft forked onto the reading it writes
//! questions about, a priming-chain link read behind the links before it — has
//! more to place than the system prompt, so its frame carries the lineage and
//! the lineage is in front of it. Without it, every calibration draft on BlackWorld
//! from 2026-11-15 on wrote a file's questions knowing only its name.
//!
//! Uses Qwen2-0.5B via HuggingFace and a CUDA GPU:
//! ```bash
//! cargo test -p candle-conversation --features hub --test append_only_lineage -- --nocapture
//! ```

use candle_conversation::{
    models::{Model, ModelBuilder},
    projection::{TurnIndex, TurnKey},
    ConversationEngine, SamplingConfig, SequenceConfig,
};

const TEST_MODEL: Model = Model::Qwen2_0_5B;

/// The codeword the lineage states and the fork is asked for.
const CODEWORD: &str = "4817";

fn test_builder() -> ModelBuilder {
    TEST_MODEL
        .builder()
        .sampling(SamplingConfig::argmax())
        .seed(42)
        .max_response_tokens(32)
        .max_concurrent(4)
}

fn engine() -> ConversationEngine {
    let device = candle::Device::cuda_if_available(0).expect("a CUDA device");
    test_builder().engine(&device).expect("the model loads")
}

/// The config of an append-only ingest: no projection after the first turn.
fn append_only() -> SequenceConfig {
    let mut config = test_builder().conversation_config();
    config.disable_reprojection = true;
    config
}

/// **A fork of a never-run append-only base onto a lineage places that
/// lineage**: its prompt holds the parent's turn, and it answers from it.
#[test]
fn an_append_only_fork_onto_a_lineage_holds_it() {
    let engine = engine();
    let prompt = test_builder().format_system_prompt();

    let mut parent = engine
        .new_conversation(&prompt, append_only())
        .expect("the parent");
    parent
        .insert_turn(
            format!("The vault code is {CODEWORD}. Remember it."),
            "Noted.",
        )
        .expect("the parent's turn seals");
    let lineage = parent.timeline_id();

    let base = engine
        .new_conversation(&prompt, append_only())
        .expect("a base that never runs a turn");
    let mut fork = base.fork().expect("the fork");
    engine
        .set_forked_from(fork.timeline_id(), lineage)
        .expect("the fork's lineage");

    let dump = fork
        .prompt_dump()
        .expect("a first turn that inherits a lineage is projected");
    let held: usize = dump
        .pieces
        .iter()
        .filter(|p| p.timeline == Some(lineage.raw()))
        .map(|p| p.token_count)
        .sum();
    assert!(held > 0, "the fork's prompt holds none of its lineage");

    let handle = fork
        .submit_turn("What is the vault code? Answer with the number only.")
        .expect("submit");
    let response = handle.wait().expect("the turn completes");
    assert!(
        response.text.contains(CODEWORD),
        "the fork did not answer from its lineage: {:?}",
        response.text
    );
}

/// **A fork with no lineage is framed by the system prompt alone**: its
/// frame lays down sections and no turn.
#[test]
fn an_append_only_fork_with_no_lineage_is_framed_by_its_sections() {
    let engine = engine();
    let prompt = test_builder().format_system_prompt();
    let base = engine
        .new_conversation(&prompt, append_only())
        .expect("a base that never runs a turn");
    let fork = base.fork().expect("the fork");
    let dump = fork.prompt_dump().expect("a first turn's frame");
    assert!(!dump.pieces.is_empty(), "the frame holds the system prompt");
    assert!(
        dump.pieces.iter().all(|p| p.timeline.is_none()),
        "a fork with no lineage frames no turn"
    );
}

/// **A frame window lays down exactly its turns of the lineage**, in place of
/// the inherited chain: a calibration draft over part of a long reading.
#[test]
fn a_frame_window_lays_down_its_turns_alone() {
    let engine = engine();
    let prompt = test_builder().format_system_prompt();
    let mut parent = engine
        .new_conversation(&prompt, append_only())
        .expect("the parent");
    for text in ["The first page.", "The second page.", "The third page."] {
        parent.insert_turn(text, "Read.").expect("a page seals");
    }
    let lineage = parent.timeline_id();

    let base = engine
        .new_conversation(&prompt, append_only())
        .expect("a base that never runs a turn");
    let mut fork = base.fork().expect("the fork");
    engine
        .set_forked_from(fork.timeline_id(), lineage)
        .expect("the fork's lineage");
    fork.frame_window(vec![
        TurnKey::new(lineage, TurnIndex(0)),
        TurnKey::new(lineage, TurnIndex(2)),
    ]);
    let dump = fork.prompt_dump().expect("a first turn's frame");
    let mut held: Vec<u32> = dump
        .pieces
        .iter()
        .filter(|p| p.timeline == Some(lineage.raw()))
        .filter_map(|p| p.index)
        .collect();
    held.dedup();
    assert_eq!(
        held,
        vec![0, 2],
        "the window's turns, in order, and no other"
    );
}
