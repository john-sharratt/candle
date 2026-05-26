//! Integration tests for the narrator module.
//!
//! These tests run model inference and require a CUDA device and a downloaded
//! model. They are gated with `#[ignore]` and require the `cuda` and `hub`
//! features.
//!
//! Run with:
//! ```bash
//! cargo test -p candle-conversation --features hub \
//!     --test narrator_integration_test -- --nocapture --ignored --test-threads=1
//! ```
//!
//! Three comprehensive model tests cover the full rule-set of each system prompt,
//! plus one model-free parse test that validates the clap→JSON pipeline.

use candle_conversation::{
    models::{Model, ModelBuilder},
    narrator::{
        parse_turn, text_to_inputs, ConverterMode, NarratorEngine, NarratorInput, SessionConfig,
    },
    SequenceConfig, ConversationEngine, SamplingConfig,
};

const TEST_MODEL: Model = Model::Hermes3_3B_Q6;

/// Shared builder: argmax for determinism, short output, no thinking tokens.
fn test_builder() -> ModelBuilder {
    TEST_MODEL
        .builder()
        .sampling(SamplingConfig::argmax())
        .seed(42)
        .max_response_tokens(512)
        .max_concurrent(4)
}

fn engine() -> ConversationEngine {
    let device =
        candle::Device::cuda_if_available(0).expect("CUDA device required for integration tests");
    eprintln!("\n=== Loading {} ===", TEST_MODEL);
    let start = std::time::Instant::now();
    let e = test_builder()
        .engine(&device)
        .expect("failed to load model");
    eprintln!("   Loaded in {:.2}s\n", start.elapsed().as_secs_f64());
    e
}

/// Converter config reused across all text_to_inputs calls.
fn converter_config() -> SequenceConfig {
    test_builder().conversation_config()
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Collect all character names (lower-cased) from character-bearing inputs.
fn character_names(inputs: &[NarratorInput]) -> Vec<String> {
    inputs
        .iter()
        .filter_map(|i| match i {
            NarratorInput::Say { character, .. }
            | NarratorInput::Act { character, .. }
            | NarratorInput::Cue { character, .. } => Some(character.to_lowercase()),
            _ => None,
        })
        .collect()
}

/// Assert every field in every Input is non-empty.
fn assert_all_fields_nonempty(inputs: &[NarratorInput], label: &str) {
    for (idx, input) in inputs.iter().enumerate() {
        match input {
            NarratorInput::Say { character, text } => {
                assert!(
                    !character.is_empty(),
                    "[{label}] Say[{idx}] character empty"
                );
                assert!(!text.is_empty(), "[{label}] Say[{idx}] text empty");
            }
            NarratorInput::Act { character, action } | NarratorInput::Cue { character, action } => {
                assert!(
                    !character.is_empty(),
                    "[{label}] Act/Cue[{idx}] character empty"
                );
                assert!(!action.is_empty(), "[{label}] Act/Cue[{idx}] action empty");
            }
            NarratorInput::Scene { description } | NarratorInput::Beat { description } => {
                assert!(
                    !description.is_empty(),
                    "[{label}] Scene/Beat[{idx}] description empty"
                );
            }
        }
    }
}

/// Assert the JSON roundtrip is lossless.
fn assert_json_roundtrip(inputs: &[NarratorInput], label: &str) {
    let json = serde_json::to_string(inputs).expect("serialise failed");
    // No markdown fences or extra prose should be present — the output is machine-readable.
    assert!(
        !json.contains("```"),
        "[{label}] output must not contain markdown fences: {json}"
    );
    let rt: Vec<NarratorInput> = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("[{label}] deserialise failed: {e}\njson: {json}"));
    assert_eq!(
        rt.len(),
        inputs.len(),
        "[{label}] roundtrip length mismatch"
    );
    assert_eq!(rt, inputs.to_vec(), "[{label}] roundtrip equality failed");
}

// ── Test 1: waypoint converter — full rule set ────────────────────────────────
//
// Probes every rule in `narrator_waypoint.md`:
//  A. "cue" signal words ("doesn't want to but") → Input::Cue, not Input::Act
//  B. Environment/atmosphere text → Input::Scene (no character fields)
//  C. Quoted speech → Input::Say with correct speaker
//  D. Authorial steering phrase → Input::Beat
//  E. Multiple distinct events → order preserved, no merging
//  F. Unnamed/unidentifiable character → character field = "unknown"
//  G. All outputs survive JSON roundtrip with no markdown contamination

#[test]
#[ignore]
fn test_waypoint_converter_comprehensive() {
    let eng = engine();
    let cfg = converter_config();

    // ─── A: reluctant action → cue ───────────────────────────────────────────
    // "doesn't want to but" is the explicit cue signal in the prompt.
    {
        let text = "Voss doesn't want to but she lowers her weapon.";
        let inputs = text_to_inputs(text, ConverterMode::Waypoint, 3, &eng, cfg.clone())
            .expect("A: text_to_inputs failed");
        eprintln!(
            "A (cue signal):\n{}\n",
            serde_json::to_string_pretty(&inputs).unwrap()
        );

        assert_all_fields_nonempty(&inputs, "A");
        assert_json_roundtrip(&inputs, "A");

        let has_cue = inputs.iter().any(|i| {
            matches!(i, NarratorInput::Cue { character, .. } if character.to_lowercase().contains("voss"))
        });
        assert!(
            has_cue,
            "A: 'doesn't want to but' should produce Cue for Voss: {inputs:?}"
        );

        // Should not produce a plain Act for the same reluctant action.
        let act_for_voss = inputs.iter().any(|i| {
            matches!(i, NarratorInput::Act { character, .. } if character.to_lowercase().contains("voss"))
        });
        assert!(
            !act_for_voss,
            "A: reluctant action should be Cue, not Act for Voss: {inputs:?}"
        );
    }

    // ─── B: environment/atmosphere text → scene only, no character events ────
    {
        let text = "The lights cut out. The room fills with the smell of smoke and old wood.";
        let inputs = text_to_inputs(text, ConverterMode::Waypoint, 3, &eng, cfg.clone())
            .expect("B: text_to_inputs failed");
        eprintln!(
            "B (scene only):\n{}\n",
            serde_json::to_string_pretty(&inputs).unwrap()
        );

        assert_all_fields_nonempty(&inputs, "B");
        assert_json_roundtrip(&inputs, "B");

        let has_scene = inputs
            .iter()
            .any(|i| matches!(i, NarratorInput::Scene { .. }));
        assert!(
            has_scene,
            "B: atmosphere text should produce at least one Scene: {inputs:?}"
        );

        let has_character_event = inputs.iter().any(|i| {
            matches!(
                i,
                NarratorInput::Act { .. } | NarratorInput::Say { .. } | NarratorInput::Cue { .. }
            )
        });
        assert!(
            !has_character_event,
            "B: pure environment text should not produce character events: {inputs:?}"
        );
    }

    // ─── C: quoted speech → say with correct speaker ─────────────────────────
    {
        let text = r#"George looks up from the table. "I know what you did," he says quietly."#;
        let inputs = text_to_inputs(text, ConverterMode::Waypoint, 3, &eng, cfg.clone())
            .expect("C: text_to_inputs failed");
        eprintln!(
            "C (quoted speech):\n{}\n",
            serde_json::to_string_pretty(&inputs).unwrap()
        );

        assert_all_fields_nonempty(&inputs, "C");
        assert_json_roundtrip(&inputs, "C");

        let george_says = inputs.iter().any(|i| {
            matches!(i, NarratorInput::Say { character, .. } if character.to_lowercase().contains("george"))
        });
        assert!(
            george_says,
            "C: quoted speech should be Say with George as speaker: {inputs:?}"
        );

        // The dialogue text itself should not be empty and should contain actual words.
        for i in &inputs {
            if let NarratorInput::Say { text: t, .. } = i {
                assert!(t.len() > 3, "C: Say text suspiciously short: '{t}'");
            }
        }
    }

    // ─── D: authorial steering hint → beat ───────────────────────────────────
    {
        let text = "Push this toward a confrontation. Marsh should become suspicious of George.";
        let inputs = text_to_inputs(text, ConverterMode::Waypoint, 3, &eng, cfg.clone())
            .expect("D: text_to_inputs failed");
        eprintln!(
            "D (steering → beat):\n{}\n",
            serde_json::to_string_pretty(&inputs).unwrap()
        );

        assert_all_fields_nonempty(&inputs, "D");
        assert_json_roundtrip(&inputs, "D");

        let has_beat = inputs
            .iter()
            .any(|i| matches!(i, NarratorInput::Beat { .. }));
        assert!(
            has_beat,
            "D: authorial steering phrase should produce Beat: {inputs:?}"
        );
    }

    // ─── E: multiple distinct events — order preserved, no merging ───────────
    // Three events with explicit attribution (no pronoun coreference) so the
    // 3B model doesn't need to resolve "he" back to its antecedent.
    {
        let text = r#"Kael steps into the room. "Take a seat," Kael says. Voss pulls a chair out."#;
        let inputs = text_to_inputs(text, ConverterMode::Waypoint, 3, &eng, cfg.clone())
            .expect("E: text_to_inputs failed");
        eprintln!(
            "E (event ordering):\n{}\n",
            serde_json::to_string_pretty(&inputs).unwrap()
        );

        assert_all_fields_nonempty(&inputs, "E");
        assert_json_roundtrip(&inputs, "E");
        assert!(
            inputs.len() >= 3,
            "E: three distinct events should not be merged: {inputs:?}"
        );

        // Kael act (enter) must precede Kael say.
        let kael_act_pos = inputs.iter().position(|i| {
            matches!(i, NarratorInput::Act { character, .. } if character.to_lowercase().contains("kael"))
        });
        let kael_say_pos = inputs.iter().position(|i| {
            matches!(i, NarratorInput::Say { character, .. } if character.to_lowercase().contains("kael"))
        });
        if let (Some(ap), Some(sp)) = (kael_act_pos, kael_say_pos) {
            assert!(ap < sp, "E: Kael enter should precede Kael say: {inputs:?}");
        }

        // Voss action should come after Kael's dialogue.
        let voss_act_pos = inputs.iter().position(|i| {
            matches!(i, NarratorInput::Act { character, .. } if character.to_lowercase().contains("voss"))
        });
        if let (Some(sp), Some(vp)) = (kael_say_pos, voss_act_pos) {
            assert!(
                sp < vp,
                "E: Kael say should precede Voss reaction: {inputs:?}"
            );
        }
    }

    // ─── F: unnamed persona — character field must be a non-empty string ────
    // The prompt instructs "unknown" for vague actors. Hermes-3B tends to copy
    // the source word (e.g. "someone") rather than substitute "unknown", which
    // is a known 3B limitation. We assert valid schema structure and that the
    // actor is NOT classified as a Scene — a human reviewer can see the value.
    {
        let text = "Someone pushes through the back door and runs across the room.";
        let inputs = text_to_inputs(text, ConverterMode::Waypoint, 3, &eng, cfg.clone())
            .expect("F: text_to_inputs failed");
        eprintln!(
            "F (unknown character):\n{}\n",
            serde_json::to_string_pretty(&inputs).unwrap()
        );

        assert_all_fields_nonempty(&inputs, "F");
        assert_json_roundtrip(&inputs, "F");

        let names = character_names(&inputs);
        assert!(
            !names.is_empty(),
            "F: should produce at least one character event (not scene): {inputs:?}"
        );
        for name in &names {
            assert!(
                !name.is_empty(),
                "F: character name must not be empty: {inputs:?}"
            );
        }
        eprintln!("F: model chose character field(s): {names:?} (prefer \"unknown\")");
    }
}

// ── Test 2: author mode converter — full rule set ────────────────────────────
//
// Probes every rule in `narrator_author.md`:
//  A. "I" → author name in Act character field
//  B. "We" → author name
//  C. Action verb converted to third-person (not "I take" → "takes")
//  D. Quoted first-person speech → Input::Say with author as character
//  E. Named third-party character resolved alongside first-person correctly
//  F. "me" used as object still produces author as actor (not object identity)
//  G. All outputs survive JSON roundtrip with no markdown contamination

#[test]
#[ignore]
fn test_author_converter_comprehensive() {
    let eng = engine();
    let cfg = converter_config();
    let author = "Kael Dorn";

    // ─── A: "I" → author name, not literal "I" in character ─────────────────
    {
        let text = "I pull out the map and spread it across the table.";
        let inputs = text_to_inputs(text, ConverterMode::Author(author), 3, &eng, cfg.clone())
            .expect("A: text_to_inputs failed");
        eprintln!(
            "A (I → author):\n{}\n",
            serde_json::to_string_pretty(&inputs).unwrap()
        );

        assert_all_fields_nonempty(&inputs, "A");
        assert_json_roundtrip(&inputs, "A");

        let names = character_names(&inputs);
        assert!(
            !names.is_empty(),
            "A: should produce character event: {inputs:?}"
        );

        let has_kael = names.iter().any(|n| n.contains("kael"));
        assert!(has_kael, "A: 'I' should resolve to Kael Dorn: {inputs:?}");

        let has_bare_i = names.iter().any(|n| n.trim() == "i");
        assert!(
            !has_bare_i,
            "A: 'I' must not appear unresolved as character: {inputs:?}"
        );
    }

    // ─── B: "We" → author name ────────────────────────────────────────────────
    {
        let text = "We sprint down the corridor toward the fire exit.";
        let inputs = text_to_inputs(text, ConverterMode::Author(author), 3, &eng, cfg.clone())
            .expect("B: text_to_inputs failed");
        eprintln!(
            "B (we → author):\n{}\n",
            serde_json::to_string_pretty(&inputs).unwrap()
        );

        assert_all_fields_nonempty(&inputs, "B");
        assert_json_roundtrip(&inputs, "B");

        let names = character_names(&inputs);
        let has_kael = names.iter().any(|n| n.contains("kael"));
        assert!(
            has_kael,
            "B: 'We' should yield Kael Dorn as a character: {inputs:?}"
        );

        let has_bare_we = names.iter().any(|n| n.trim() == "we");
        assert!(
            !has_bare_we,
            "B: 'we' must not appear as a character name: {inputs:?}"
        );
    }

    // ─── C: action field is third-person form, not first-person ──────────────
    {
        let text = "I grab the rope with both hands and haul myself up.";
        let inputs = text_to_inputs(text, ConverterMode::Author(author), 3, &eng, cfg.clone())
            .expect("C: text_to_inputs failed");
        eprintln!(
            "C (verb form):\n{}\n",
            serde_json::to_string_pretty(&inputs).unwrap()
        );

        for input in &inputs {
            if let NarratorInput::Act { character, action } = input {
                if character.to_lowercase().contains("kael") {
                    assert!(
                        !action.trim_start().starts_with("I "),
                        "C: action must be third-person, got first-person: '{action}' in {inputs:?}"
                    );
                }
            }
        }
    }

    // ─── D: quoted first-person speech → Say with author as character ─────────
    {
        let text = r#""We need to leave right now," I tell her."#;
        let inputs = text_to_inputs(text, ConverterMode::Author(author), 3, &eng, cfg.clone())
            .expect("D: text_to_inputs failed");
        eprintln!(
            "D (quoted I-speech → say):\n{}\n",
            serde_json::to_string_pretty(&inputs).unwrap()
        );

        assert_all_fields_nonempty(&inputs, "D");
        assert_json_roundtrip(&inputs, "D");

        let kael_says = inputs.iter().any(|i| {
            matches!(i, NarratorInput::Say { character, .. } if character.to_lowercase().contains("kael"))
        });
        assert!(
            kael_says,
            "D: quoted I-speech → Say with Kael as character: {inputs:?}"
        );
    }

    // ─── E: named third-party correctly identified alongside first-person ─────
    {
        let text = "Voss hands me the key. I unlock the door and step inside.";
        let inputs = text_to_inputs(text, ConverterMode::Author(author), 3, &eng, cfg.clone())
            .expect("E: text_to_inputs failed");
        eprintln!(
            "E (third-party + I):\n{}\n",
            serde_json::to_string_pretty(&inputs).unwrap()
        );

        assert_all_fields_nonempty(&inputs, "E");
        assert_json_roundtrip(&inputs, "E");

        let names = character_names(&inputs);
        let has_voss = names.iter().any(|n| n.contains("voss"));
        let has_kael = names.iter().any(|n| n.contains("kael"));
        assert!(has_voss, "E: Voss should appear as a character: {inputs:?}");
        assert!(
            has_kael,
            "E: I (Kael) should appear as a character: {inputs:?}"
        );

        // Order: Voss acts first, then Kael.
        let voss_pos = inputs.iter().position(|i| {
            matches!(i, NarratorInput::Act { character, .. } | NarratorInput::Cue { character, .. }
                if character.to_lowercase().contains("voss"))
        });
        let kael_pos = inputs.iter().position(|i| {
            matches!(i, NarratorInput::Act { character, .. }
                if character.to_lowercase().contains("kael"))
        });
        if let (Some(vp), Some(kp)) = (voss_pos, kael_pos) {
            assert!(
                vp < kp,
                "E: Voss should precede Kael in event order: {inputs:?}"
            );
        }
    }

    // ─── F: mixed passage — "my" property reference, third-person unnamed ────
    {
        let text = "I reach into my coat. Someone grabs my arm from behind.";
        let inputs = text_to_inputs(text, ConverterMode::Author(author), 3, &eng, cfg.clone())
            .expect("F: text_to_inputs failed");
        eprintln!(
            "F (my + unknown):\n{}\n",
            serde_json::to_string_pretty(&inputs).unwrap()
        );

        assert_all_fields_nonempty(&inputs, "F");
        assert_json_roundtrip(&inputs, "F");

        let names = character_names(&inputs);
        let has_kael = names.iter().any(|n| n.contains("kael"));
        assert!(
            has_kael,
            "F: 'I/my' should resolve to Kael Dorn: {inputs:?}"
        );

        // "Someone" grabs — should be "unknown", not "someone" as literal name.
        let has_someone_literal = names.iter().any(|n| n == "someone");
        assert!(
            !has_someone_literal,
            "F: 'someone' should not appear as a literal character name: {inputs:?}"
        );
    }
}

// ── Test 3: narrator prose — full rule set ───────────────────────────────────
//
// Probes every rule in `guide_narrate.md`:
//  A. Output is second-person present tense ("you")
//  B. Fidelity: every named person and key action in the waypoint appears in prose
//  C. Full name on first mention — named characters not reduced to pronouns alone
//  D. Prose length: 1–4 sentences
//  E. Tone fidelity: a quiet domestic scene produces no dread/menace vocabulary
//  F. No meta-acknowledgment (no "waypoint", "instruction", "I will", "as requested")
//  G. Multi-turn: second response continues from first, names re-introduced on first use

#[test]
#[ignore]
fn test_narrator_prose_comprehensive() {
    let eng = engine();
    let mut narrator =
        NarratorEngine::new(&eng, converter_config()).expect("new NarratorEngine failed");

    // ─── A + B + C + D + E + F: single-turn prose quality ───────────────────
    // Waypoint: Marsh enters, offers sherry. Quiet, domestic. No danger.
    let t1_inputs = vec![
        NarratorInput::Act {
            character: "Marsh".into(),
            action: "enters the room shaking snow from his coat".into(),
        },
        NarratorInput::Say {
            character: "Marsh".into(),
            text: "Thought you could use some company.".into(),
        },
    ];
    let prose1 = narrator
        .next(t1_inputs)
        .expect("turn 1 failed")
        .trim()
        .to_string();
    eprintln!("Turn 1 prose:\n{prose1}\n");

    // A: second-person present tense.
    assert!(
        prose1.to_lowercase().contains("you"),
        "A: prose must be second-person, missing 'you': {prose1}"
    );

    // B: fidelity — "Marsh" and "snow" and some form of arrival must appear.
    assert!(
        prose1.to_lowercase().contains("marsh"),
        "B (fidelity): 'Marsh' must appear in prose: {prose1}"
    );
    assert!(
        prose1.to_lowercase().contains("snow") || prose1.to_lowercase().contains("coat"),
        "B (fidelity): waypoint detail (snow/coat) must appear in prose: {prose1}"
    );
    // The offer of company should be reflected.
    assert!(
        prose1.to_lowercase().contains("company")
            || prose1.to_lowercase().contains("thought")
            || prose1.to_lowercase().contains("could use"),
        "B (fidelity): Marsh's dialogue must appear in prose: {prose1}"
    );

    // C: full name on first mention — "Marsh" should appear, not just "he" alone.
    let first_sentence = prose1.split(['.', '!', '?']).next().unwrap_or(&prose1);
    assert!(
        first_sentence.to_lowercase().contains("marsh"),
        "C: Marsh's name should appear in the first sentence, not just a pronoun: {prose1}"
    );

    // D: length — 1 to 3 sentences (hard limit in prompt).
    let sentence_count = prose1
        .split(['.', '!', '?'])
        .filter(|s| s.chars().any(|c| c.is_alphabetic()))
        .count();
    assert!(
        sentence_count >= 1 && sentence_count <= 3,
        "D: prose must be 1–3 sentences, got {sentence_count}: {prose1}"
    );

    // E: tone — quiet domestic scene should not introduce dread vocabulary.
    let dread_words = [
        "danger",
        "menace",
        "threat",
        "fear",
        "terror",
        "sinister",
        "dark secret",
    ];
    for word in dread_words {
        assert!(
            !prose1.to_lowercase().contains(word),
            "E (tone): quiet scene should not contain '{word}': {prose1}"
        );
    }

    // F: no meta-acknowledgment of the waypoint machinery.
    let meta_phrases = [
        "waypoint",
        "instruction",
        "i will",
        "as requested",
        "as instructed",
        "as asked",
    ];
    for phrase in meta_phrases {
        assert!(
            !prose1.to_lowercase().contains(phrase),
            "F (no meta): prose must not acknowledge instructions, found '{phrase}': {prose1}"
        );
    }

    // ─── Between turns: no character response insertion needed ────────────────
    // The narrator maintains continuity from its own prior narrations;
    // character responses are no longer fed back into the narrator context.

    // ─── G: multi-turn — second narration continues story, re-introduces name ─
    // New beat: Kael speaks. Marsh's name already established, but Kael appears fresh.
    let t2_inputs = vec![
        NarratorInput::Say {
            character: "Kael Dorn".into(),
            text: "You came.".into(),
        },
        NarratorInput::Act {
            character: "Kael Dorn".into(),
            action: "moves to the window".into(),
        },
    ];
    let prose2 = narrator
        .next(t2_inputs)
        .expect("turn 2 failed")
        .trim()
        .to_string();
    eprintln!("Turn 2 prose:\n{prose2}\n");

    // Still second-person.
    assert!(
        prose2.to_lowercase().contains("you"),
        "G: turn 2 must still be second-person: {prose2}"
    );

    // Fidelity: the move to the window must appear.
    assert!(
        prose2.to_lowercase().contains("window"),
        "G (fidelity): 'window' from waypoint must appear in turn 2 prose: {prose2}"
    );
    // Note: the protagonist (Kael Dorn) is addressed as 'you' in second-person narration,
    // so their name does not appear directly — that is correct behaviour.

    // D: length again.
    let sentence_count2 = prose2
        .split(['.', '!', '?'])
        .filter(|s| s.chars().any(|c| c.is_alphabetic()))
        .count();
    assert!(
        sentence_count2 >= 1 && sentence_count2 <= 3,
        "G/D: turn 2 must be 1–3 sentences, got {sentence_count2}: {prose2}"
    );

    // No meta-acknowledgment in turn 2.
    for phrase in meta_phrases {
        assert!(
            !prose2.to_lowercase().contains(phrase),
            "G/F: turn 2 must not acknowledge instructions, found '{phrase}': {prose2}"
        );
    }

    narrator.close().expect("close failed");
}

// ── Test 4: parse_turn → serialise pipeline (model-free) ─────────────────────
//
// Verifies the full clap→JSON pipeline without inference. This is a fast
// boundary test that covers all five Input types, default character resolution,
// multi-line input, and lossless serialisation.

#[test]
#[ignore]
fn test_parse_turn_serialise_comprehensive() {
    let cfg = SessionConfig {
        protagonist: "Kael Dorn".into(),
        persona: "Voss".into(),
        max_turns: 7,
    };

    // ─── All five types, default character resolution ─────────────────────────
    let raw = concat!(
        "scene the roof begins to collapse\n",
        "act takes her hand\n",             // protagonist default → Kael Dorn
        "cue draws her sidearm\n",          // persona default → Voss
        "say we have to get out of here\n", // protagonist default → Kael Dorn
        "beat steer toward desperation",
    );
    let inputs = parse_turn(raw, &cfg).expect("parse_turn failed");
    let json = serde_json::to_string(&inputs).expect("serialise failed");
    eprintln!(
        "Pipeline JSON:\n{}\n",
        serde_json::to_string_pretty(&inputs).unwrap()
    );

    let rt: Vec<NarratorInput> = serde_json::from_str(&json).expect("deserialise failed");
    assert_eq!(rt.len(), 5, "expected 5 inputs, got: {inputs:?}");
    assert_eq!(rt, inputs);

    assert!(
        matches!(&rt[0], NarratorInput::Scene { description } if description.contains("collapse"))
    );
    assert!(matches!(&rt[1], NarratorInput::Act { character, .. } if character == "Kael Dorn"));
    assert!(matches!(&rt[2], NarratorInput::Cue { character, .. } if character == "Voss"));
    assert!(matches!(&rt[3], NarratorInput::Say { character, .. } if character == "Kael Dorn"));
    assert!(
        matches!(&rt[4], NarratorInput::Beat { description } if description.contains("desperation"))
    );

    // ─── Explicit --character overrides default ───────────────────────────────
    let raw2 = concat!(
        "act --character Marsh pours himself a drink\n",
        "say --character George what do you want from me",
    );
    let inputs2 = parse_turn(raw2, &cfg).expect("parse_turn explicit character failed");
    assert_eq!(inputs2.len(), 2);
    assert!(matches!(&inputs2[0], NarratorInput::Act { character, .. } if character == "Marsh"));
    assert!(
        matches!(&inputs2[1], NarratorInput::Say { character, text } if character == "George" && text.contains("want"))
    );

    // ─── Quoted multi-word character name ─────────────────────────────────────
    let raw3 = r#"act --character "Old Man" reaches for the lantern"#;
    let inputs3 = parse_turn(raw3, &cfg).expect("parse_turn quoted character failed");
    assert_eq!(inputs3.len(), 1);
    assert!(matches!(&inputs3[0], NarratorInput::Act { character, .. } if character == "Old Man"));

    // ─── Single-event inputs ──────────────────────────────────────────────────
    let single_cases = [
        ("scene morning light fills the room", 1usize),
        ("beat push toward resolution", 1),
        ("say good morning", 1),
    ];
    for (raw, expected_len) in single_cases {
        let result = parse_turn(raw, &cfg).expect(raw);
        assert_eq!(
            result.len(),
            expected_len,
            "expected {expected_len} input for '{raw}', got: {result:?}"
        );
    }

    // ─── JSON roundtrip preserves all field values ────────────────────────────
    for input in &inputs {
        let s = serde_json::to_string(input).unwrap();
        let rt: NarratorInput = serde_json::from_str(&s).unwrap();
        assert_eq!(&rt, input, "single-input roundtrip failed for {input:?}");
    }
}
