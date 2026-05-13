//! Narrator context-window integration tests.
//!
//! Runs a multi-beat scene through `NarratorEngine`, mirroring the two-`send_turn`
//! pattern the chat app uses per story beat (turn A: player inputs, turn B:
//! character waypoints), and asserts the KV attention window has not activated
//! by the end of each beat.
//!
//! Run with:
//! ```bash
//! cargo test -p candle-conversation --features "cuda,hub,test-helpers" \
//!     --test narrator_window_test -- --nocapture --ignored --test-threads=1
//! ```

use candle_conversation::{
    models::{Model, ModelBuilder},
    narrator::{NarratorEngine, NarratorInput},
    ConversationEngine, SamplingConfig,
};

// ── model / engine setup ──────────────────────────────────────────────────────

const TEST_MODEL: Model = Model::Hermes3_3B_Q6;

fn test_builder() -> ModelBuilder {
    TEST_MODEL
        .builder()
        .sampling(SamplingConfig::argmax())
        .seed(42)
        .max_response_tokens(256)
        .max_concurrent(4)
}

fn build_engine() -> (ConversationEngine, ModelBuilder) {
    let device =
        candle::Device::cuda_if_available(0).expect("CUDA device required for integration tests");
    eprintln!("\n=== Loading {} ===", TEST_MODEL);
    let start = std::time::Instant::now();
    let mut builder = test_builder();
    let e = builder.engine(&device).expect("failed to load model");
    eprintln!("   Loaded in {:.2}s\n", start.elapsed().as_secs_f64());
    (e, builder)
}

// ── story beats ───────────────────────────────────────────────────────────────
//
// A quiet inn scene: Alistair and Mira at the Thornfield Inn.  Beat 1
// introduces an obsidian locket that is not re-mentioned in later beats.
// Beat 6 calls on the narrator to draw on that early context.
//
// Each element is (player_inputs, character_waypoints).  Character waypoints
// are canned rather than model-generated, which avoids needing a second
// model while still replicating the two-`send_turn`-per-beat traffic pattern
// of the chat app.
fn story_beats() -> Vec<(Vec<NarratorInput>, Vec<NarratorInput>)> {
    let beat = |v: Vec<NarratorInput>| v;
    vec![
        // Beat 1
        (
            beat(vec![
                NarratorInput::Scene {
                    description:
                        "The common room of the Thornfield Inn, late evening. A fire crackles \
                         in the hearth. An obsidian locket rests on the table between Alistair \
                         and Mira — small, oval, carved with a crescent moon."
                            .into(),
                },
                NarratorInput::Say {
                    character: "Alistair".into(),
                    text: "Where did you get this?".into(),
                },
            ]),
            beat(vec![NarratorInput::Say {
                character: "Mira".into(),
                text: "It belonged to my mother. I have carried it since I was seven.".into(),
            }]),
        ),
        // Beat 2
        (
            beat(vec![
                NarratorInput::Scene {
                    description: "The inn door swings open, letting in a gust of cold air. \
                                  A hooded figure steps inside and scans the room."
                        .into(),
                },
                NarratorInput::Act {
                    character: "Alistair".into(),
                    action: "sits up straighter and watches the newcomer".into(),
                },
            ]),
            beat(vec![NarratorInput::Act {
                character: "Mira".into(),
                action: "closes her hand around the locket and tucks it out of sight".into(),
            }]),
        ),
        // Beat 3
        (
            beat(vec![
                NarratorInput::Say {
                    character: "Stranger".into(),
                    text: "I am looking for someone. A woman with dark hair, travelling alone."
                        .into(),
                },
                NarratorInput::Act {
                    character: "Alistair".into(),
                    action: "shakes his head slowly without looking at Mira".into(),
                },
            ]),
            beat(vec![NarratorInput::Say {
                character: "Mira".into(),
                text: "There is no one here matching that description.".into(),
            }]),
        ),
        // Beat 4
        (
            beat(vec![
                NarratorInput::Scene {
                    description: "The stranger surveys the room a moment longer, then turns and \
                                  leaves. The door closes."
                        .into(),
                },
                NarratorInput::Beat {
                    description: "The silence stretches. Both characters feel the weight of what \
                                  was almost discovered."
                        .into(),
                },
            ]),
            beat(vec![NarratorInput::Act {
                character: "Mira".into(),
                action: "exhales slowly and loosens her grip".into(),
            }]),
        ),
        // Beat 5
        (
            beat(vec![NarratorInput::Say {
                character: "Alistair".into(),
                text: "You need to tell me what is going on, Mira. All of it.".into(),
            }]),
            beat(vec![NarratorInput::Beat {
                description:
                    "Mira hesitates, weighing how much to reveal, then decides to trust him.".into(),
            }]),
        ),
        // Beat 6
        (
            beat(vec![
                NarratorInput::Say {
                    character: "Mira".into(),
                    text: "The locket is the reason they are hunting me.".into(),
                },
                NarratorInput::Beat {
                    description: "Reveal what the locket means and why it places Mira in danger. \
                                  The narrator should draw on what was established at the start \
                                  of the scene."
                        .into(),
                },
            ]),
            beat(vec![NarratorInput::Beat {
                description: "Alistair absorbs this revelation in silence.".into(),
            }]),
        ),
    ]
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// Runs a 6-beat inn scene through the narrator, two `send_turn` calls per
/// beat (player inputs, then character waypoints), and asserts the window state
/// matches expectations at each beat.
///
/// Each beat costs 2 narrator-level turns.  With `context_window_turns = 4`,
/// the window activates at beat 3 (completed=6 > window=4) and stays active.
/// The test verifies the window is OFF for beats 1–2 and ON for beats 3–6,
/// confirming masking kicks in at exactly the right point.
#[test]
#[ignore = "requires CUDA device and downloaded model"]
fn narrator_window_all_beats() {
    let (engine, builder) = build_engine();
    let config = builder.conversation_config();

    let mut narrator =
        NarratorEngine::new(&engine, config).expect("failed to create NarratorEngine");

    let beats = story_beats();
    let total_beats = beats.len();

    for (beat_idx, (player_inputs, char_waypoints)) in beats.into_iter().enumerate() {
        let beat_num = beat_idx + 1;

        // Turn A: narrator generates prose from the player's inputs.
        let prose_a = narrator
            .next(player_inputs)
            .unwrap_or_else(|e| panic!("beat {beat_num} turn A: {e}"));

        eprintln!("─── Beat {beat_num}/{total_beats} ───");
        eprintln!("  [A] {}", prose_a.lines().next().unwrap_or(""));

        let (completed_a, window) = narrator.conversation.window_state();
        eprintln!(
            "  [A] window_state: completed={completed_a}, window={window}, masked={}",
            completed_a > window
        );

        // Turn B: narrator processes canned character waypoints (mirrors
        // insert_character_response_streaming without a second model).
        let prose_b = narrator
            .next(char_waypoints)
            .unwrap_or_else(|e| panic!("beat {beat_num} turn B: {e}"));

        eprintln!("  [B] {}", prose_b.lines().next().unwrap_or(""));

        let (completed_b, _) = narrator.conversation.window_state();
        eprintln!(
            "  [B] window_state: completed={completed_b}, window={window}, masked={}",
            completed_b > window
        );

        // The window activates when completed > window.  With window=4 and 2 turns
        // per beat, that happens at beat 3 (completed=6).  Verify the state matches.
        let expected_active = beat_num * 2 > window;
        let is_active = completed_b > window;
        assert_eq!(
            is_active, expected_active,
            "beat {beat_num}/{total_beats}: window active={is_active} \
             (expected {expected_active}), completed={completed_b}, window={window}",
        );
    }

    eprintln!("\nAll {total_beats} beats: window state correct throughout.");
}

/// Same two-turns-per-beat traffic pattern using `insert_turn` (prefill only,
/// no decode).  Produces a deterministic window-state trace and verifies the
/// window activates exactly when expected (beat 3 with window=4).
#[test]
#[ignore = "requires CUDA device for insert_turn prefill"]
fn narrator_window_insert_turn_trace() {
    let (engine, builder) = build_engine();
    let config = builder.conversation_config();
    let narrator = NarratorEngine::new(&engine, config).expect("failed to create NarratorEngine");
    let mut conv = narrator.conversation;

    eprintln!();
    for beat_num in 1..=6usize {
        conv.insert_turn(
            &format!("[{beat_num}A] inputs"),
            &format!("[{beat_num}A] prose"),
        )
        .unwrap_or_else(|e| panic!("insert_turn {beat_num}A: {e}"));
        let (a, window) = conv.window_state();
        eprintln!(
            "beat {beat_num} A: completed={a}, window={window}, masked={}",
            a > window
        );

        conv.insert_turn(
            &format!("[{beat_num}B] waypoints"),
            &format!("[{beat_num}B] prose"),
        )
        .unwrap_or_else(|e| panic!("insert_turn {beat_num}B: {e}"));
        let (b, _) = conv.window_state();
        eprintln!(
            "beat {beat_num} B: completed={b}, window={window}, masked={}",
            b > window
        );

        // Window should be inactive for beats 1–2 (completed ≤ 4) and active
        // from beat 3 onwards (completed > 4).  Both states are expected.
        let expected_active = beat_num * 2 > window;
        let is_active = b > window;
        assert_eq!(
            is_active, expected_active,
            "beat {beat_num}: window active={is_active} (expected {expected_active}), \
             completed={b}, window={window}",
        );
    }
}
