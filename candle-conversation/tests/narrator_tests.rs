//! Unit tests for the narrator module.
//!
//! All tests here are pure (no model, no GPU, no network). They cover:
//!
//! - [`Input`] serialisation and deserialisation for every variant
//! - [`SessionConfig`] and [`DEFAULT_MAX_TURNS`]
//! - [`parse_turn`] for every command type
//! - Default character resolution via [`SessionConfig`]
//! - Multi-line input producing multiple [`Input`] objects
//! - Error cases: empty input, unknown commands, missing required args
//! - [`correction_message`] format
//! - [`converter_system_prompt`] template substitution

use candle_conversation::narrator::{
    converter_system_prompt, correction_message, parse_turn, ConvertError, ConverterMode,
    NarratorInput, ParseError, SessionConfig, DEFAULT_MAX_TURNS,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn config() -> SessionConfig {
    SessionConfig {
        protagonist: "Kael Dorn".into(),
        persona: "Voss".into(),
        max_turns: 7,
    }
}

/// Config with empty protagonist/persona — used to test the edge case where
/// a caller neglects to set them (character resolves to an empty string).
fn empty_config() -> SessionConfig {
    SessionConfig {
        protagonist: String::new(),
        persona: String::new(),
        max_turns: 7,
    }
}

// ── Input serialisation ───────────────────────────────────────────────────────

#[test]
fn input_say_roundtrip() {
    let input = NarratorInput::Say {
        character: "Kael Dorn".into(),
        text: "we have to get out of here".into(),
    };
    let json = serde_json::to_string(&input).unwrap();
    assert!(json.contains("\"type\":\"say\""));
    assert!(json.contains("\"character\":\"Kael Dorn\""));
    assert!(json.contains("\"text\":\"we have to get out of here\""));
    let rt: NarratorInput = serde_json::from_str(&json).unwrap();
    assert_eq!(rt, input);
}

#[test]
fn input_act_roundtrip() {
    let input = NarratorInput::Act {
        character: "Voss".into(),
        action: "draws her sidearm".into(),
    };
    let json = serde_json::to_string(&input).unwrap();
    assert!(json.contains("\"type\":\"act\""));
    assert!(json.contains("\"character\":\"Voss\""));
    assert!(json.contains("\"action\":\"draws her sidearm\""));
    let rt: NarratorInput = serde_json::from_str(&json).unwrap();
    assert_eq!(rt, input);
}

#[test]
fn input_scene_roundtrip() {
    let input = NarratorInput::Scene {
        description: "The roof begins to collapse.".into(),
    };
    let json = serde_json::to_string(&input).unwrap();
    assert!(json.contains("\"type\":\"scene\""));
    assert!(json.contains("\"description\":\"The roof begins to collapse.\""));
    let rt: NarratorInput = serde_json::from_str(&json).unwrap();
    assert_eq!(rt, input);
}

#[test]
fn input_cue_roundtrip() {
    let input = NarratorInput::Cue {
        character: "Voss".into(),
        action: "steps in front of the exit".into(),
    };
    let json = serde_json::to_string(&input).unwrap();
    assert!(json.contains("\"type\":\"cue\""));
    assert!(json.contains("\"character\":\"Voss\""));
    let rt: NarratorInput = serde_json::from_str(&json).unwrap();
    assert_eq!(rt, input);
}

#[test]
fn input_beat_roundtrip() {
    let input = NarratorInput::Beat {
        description: "Voss is close to agreeing but needs one more push".into(),
    };
    let json = serde_json::to_string(&input).unwrap();
    assert!(json.contains("\"type\":\"beat\""));
    let rt: NarratorInput = serde_json::from_str(&json).unwrap();
    assert_eq!(rt, input);
}

#[test]
fn input_vec_roundtrip() {
    let inputs = vec![
        NarratorInput::Scene {
            description: "the roof begins to collapse".into(),
        },
        NarratorInput::Act {
            character: "Kael Dorn".into(),
            action: "takes her hand".into(),
        },
        NarratorInput::Cue {
            character: "Voss".into(),
            action: "draws her sidearm".into(),
        },
        NarratorInput::Say {
            character: "Kael Dorn".into(),
            text: "we have to get out of here".into(),
        },
        NarratorInput::Beat {
            description: "steer toward escape through the east corridor".into(),
        },
    ];
    let json = serde_json::to_string(&inputs).unwrap();
    let rt: Vec<NarratorInput> = serde_json::from_str(&json).unwrap();
    assert_eq!(rt, inputs);
}

#[test]
fn input_json_no_null_characters() {
    let input = NarratorInput::Say {
        character: "Alice".into(),
        text: "hello".into(),
    };
    let json = serde_json::to_string(&input).unwrap();
    assert!(
        !json.contains("null"),
        "Input JSON must not contain null: {json}"
    );
}

// ── SessionConfig ────────────────────────────────────────────────────────────

#[test]
fn session_config_default_max_turns() {
    assert_eq!(DEFAULT_MAX_TURNS, 7);
}

#[test]
fn session_config_fields_are_public() {
    let cfg = SessionConfig {
        protagonist: "Alice".into(),
        persona: "Bob".into(),
        max_turns: 10,
    };
    assert_eq!(cfg.protagonist, "Alice");
    assert_eq!(cfg.persona, "Bob");
    assert_eq!(cfg.max_turns, 10);
}

// ── parse_turn: each command type ────────────────────────────────────────────

#[test]
fn parse_turn_say_resolves_protagonist() {
    let inputs = parse_turn("say we have to get out of here", &config()).unwrap();
    assert_eq!(inputs.len(), 1);
    assert_eq!(
        inputs[0],
        NarratorInput::Say {
            character: "Kael Dorn".into(),
            text: "we have to get out of here".into(),
        }
    );
}

#[test]
fn parse_turn_say_explicit_character() {
    let inputs = parse_turn(
        "say --character Voss that is the third time tonight",
        &config(),
    )
    .unwrap();
    assert_eq!(inputs.len(), 1);
    assert_eq!(
        inputs[0],
        NarratorInput::Say {
            character: "Voss".into(),
            text: "that is the third time tonight".into(),
        }
    );
}

#[test]
fn parse_turn_say_short_flag() {
    let inputs = parse_turn("say -c Voss you sure about that", &config()).unwrap();
    assert_eq!(
        inputs[0],
        NarratorInput::Say {
            character: "Voss".into(),
            text: "you sure about that".into(),
        }
    );
}

#[test]
fn parse_turn_act_resolves_protagonist() {
    let inputs = parse_turn("act takes your hand", &config()).unwrap();
    assert_eq!(
        inputs[0],
        NarratorInput::Act {
            character: "Kael Dorn".into(),
            action: "takes your hand".into(),
        }
    );
}

#[test]
fn parse_turn_act_explicit_character() {
    let inputs = parse_turn(
        "act --character Voss moves towards the door with haste",
        &config(),
    )
    .unwrap();
    assert_eq!(
        inputs[0],
        NarratorInput::Act {
            character: "Voss".into(),
            action: "moves towards the door with haste".into(),
        }
    );
}

#[test]
fn parse_turn_scene_no_character_field() {
    let inputs = parse_turn("scene the roof begins to collapse", &config()).unwrap();
    assert_eq!(
        inputs[0],
        NarratorInput::Scene {
            description: "the roof begins to collapse".into(),
        }
    );
    // Scene must not carry a character field.
    let json = serde_json::to_string(&inputs[0]).unwrap();
    assert!(!json.contains("character"));
}

#[test]
fn parse_turn_cue_resolves_persona() {
    let inputs = parse_turn("cue draws her sidearm", &config()).unwrap();
    assert_eq!(
        inputs[0],
        NarratorInput::Cue {
            character: "Voss".into(),
            action: "draws her sidearm".into(),
        }
    );
}

#[test]
fn parse_turn_cue_explicit_character() {
    let inputs = parse_turn("cue --character Voss steps in front of the exit", &config()).unwrap();
    assert_eq!(
        inputs[0],
        NarratorInput::Cue {
            character: "Voss".into(),
            action: "steps in front of the exit".into(),
        }
    );
}

#[test]
fn parse_turn_beat() {
    let inputs = parse_turn(
        "beat steer toward escape through the east corridor",
        &config(),
    )
    .unwrap();
    assert_eq!(
        inputs[0],
        NarratorInput::Beat {
            description: "steer toward escape through the east corridor".into(),
        }
    );
}

// ── parse_turn: multi-line ────────────────────────────────────────────────────

#[test]
fn parse_turn_multi_line_full_turn() {
    let text = "scene the roof begins to collapse\n\
                act takes your hand\n\
                cue draws her sidearm\n\
                say we have to get out of here\n\
                beat steer toward escape through the east corridor";

    let inputs = parse_turn(text, &config()).unwrap();
    assert_eq!(inputs.len(), 5);
    assert_eq!(
        inputs[0],
        NarratorInput::Scene {
            description: "the roof begins to collapse".into(),
        }
    );
    assert_eq!(
        inputs[1],
        NarratorInput::Act {
            character: "Kael Dorn".into(),
            action: "takes your hand".into(),
        }
    );
    assert_eq!(
        inputs[2],
        NarratorInput::Cue {
            character: "Voss".into(),
            action: "draws her sidearm".into(),
        }
    );
    assert_eq!(
        inputs[3],
        NarratorInput::Say {
            character: "Kael Dorn".into(),
            text: "we have to get out of here".into(),
        }
    );
    assert_eq!(
        inputs[4],
        NarratorInput::Beat {
            description: "steer toward escape through the east corridor".into(),
        }
    );
}

#[test]
fn parse_turn_blank_lines_ignored() {
    let text = "\n  \nsay hello\n\n\nact waves\n  \n";
    let inputs = parse_turn(text, &config()).unwrap();
    assert_eq!(inputs.len(), 2);
}

#[test]
fn parse_turn_leading_trailing_whitespace_trimmed() {
    let inputs = parse_turn("  say   hello world  ", &config()).unwrap();
    assert_eq!(inputs.len(), 1);
    assert_eq!(
        inputs[0],
        NarratorInput::Say {
            character: "Kael Dorn".into(),
            text: "hello world".into(),
        }
    );
}

// ── parse_turn: character resolution with empty config ───────────────────────

#[test]
fn parse_turn_say_empty_protagonist_falls_back_to_empty_string() {
    let inputs = parse_turn("say hello", &empty_config()).unwrap();
    assert_eq!(
        inputs[0],
        NarratorInput::Say {
            character: "".into(),
            text: "hello".into(),
        }
    );
}

#[test]
fn parse_turn_cue_empty_persona_falls_back_to_empty_string() {
    let inputs = parse_turn("cue fires", &empty_config()).unwrap();
    assert_eq!(
        inputs[0],
        NarratorInput::Cue {
            character: "".into(),
            action: "fires".into(),
        }
    );
}

// ── parse_turn: error cases ───────────────────────────────────────────────────

#[test]
fn parse_turn_empty_input_returns_error() {
    let err = parse_turn("", &config()).unwrap_err();
    assert!(matches!(err, ParseError::EmptyInput));
}

#[test]
fn parse_turn_all_blank_lines_returns_error() {
    let err = parse_turn("   \n\n   \n", &config()).unwrap_err();
    assert!(matches!(err, ParseError::EmptyInput));
}

#[test]
fn parse_turn_unknown_command_returns_invalid_command() {
    let err = parse_turn("jump over the fence", &config()).unwrap_err();
    assert!(matches!(err, ParseError::InvalidCommand { .. }));
    if let ParseError::InvalidCommand { line, .. } = err {
        assert_eq!(line, "jump over the fence");
    }
}

#[test]
fn parse_turn_say_no_text_returns_invalid_command() {
    // `say` with no text produces EmptyContent, not InvalidCommand.
    let err = parse_turn("say", &config()).unwrap_err();
    assert!(matches!(err, ParseError::EmptyContent { .. }));
}

#[test]
fn parse_turn_act_no_action_returns_invalid_command() {
    let err = parse_turn("act", &config()).unwrap_err();
    assert!(matches!(err, ParseError::EmptyContent { .. }));
}

#[test]
fn parse_turn_scene_no_description_returns_invalid_command() {
    let err = parse_turn("scene", &config()).unwrap_err();
    assert!(matches!(err, ParseError::EmptyContent { .. }));
}

#[test]
fn parse_turn_cue_no_action_returns_invalid_command() {
    let err = parse_turn("cue", &config()).unwrap_err();
    assert!(matches!(err, ParseError::EmptyContent { .. }));
}

#[test]
fn parse_turn_beat_no_description_returns_invalid_command() {
    let err = parse_turn("beat", &config()).unwrap_err();
    assert!(matches!(err, ParseError::EmptyContent { .. }));
}

#[test]
fn parse_turn_first_line_bad_stops_on_error() {
    // The second line is valid but we stop at the first bad line.
    let err = parse_turn("badcmd foo\nsay hello", &config()).unwrap_err();
    assert!(matches!(err, ParseError::InvalidCommand { .. }));
}

// ── parse_turn: multi-word quoted character ───────────────────────────────────

#[test]
fn parse_turn_say_multi_word_character_quoted() {
    let inputs = parse_turn("say --character \"Old Man\" you need to leave", &config()).unwrap();
    assert_eq!(
        inputs[0],
        NarratorInput::Say {
            character: "Old Man".into(),
            text: "you need to leave".into(),
        }
    );
}

// ── ParseError and ConvertError display ──────────────────────────────────────

#[test]
fn parse_error_empty_input_display() {
    let err = ParseError::EmptyInput;
    let msg = err.to_string();
    assert!(msg.contains("no non-empty lines"));
}

#[test]
fn parse_error_invalid_command_display() {
    // Build a clap error by actually triggering a parse failure.
    let err = parse_turn("jump over the fence", &config()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("jump over the fence"));
}

#[test]
fn convert_error_max_attempts_display() {
    let err = ConvertError::MaxAttemptsExceeded {
        last_error: "unexpected token".into(),
        last_output: "[bad json".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("attempt limit"));
    assert!(msg.contains("unexpected token"));
}

#[test]
fn convert_error_inference_display() {
    let err = ConvertError::InferenceError("scheduler gone".into());
    let msg = err.to_string();
    assert!(msg.contains("scheduler gone"));
}

// ── correction_message format ─────────────────────────────────────────────────

#[test]
fn correction_message_contains_error_and_output() {
    let msg = correction_message("unexpected token at offset 3", "[{bad json");
    assert!(msg.contains("unexpected token at offset 3"));
    assert!(msg.contains("[{bad json"));
    assert!(msg.contains("Parse error:"));
    assert!(msg.contains("Your output was:"));
}

// ── converter_system_prompt ───────────────────────────────────────────────────

#[test]
fn converter_system_prompt_waypoint_mode_no_author_placeholder() {
    let prompt = converter_system_prompt(&ConverterMode::Waypoint);
    // Waypoint prompt must not contain the unresolved {author} marker.
    assert!(!prompt.contains("{author}"));
    // Should describe third-person conversion.
    assert!(prompt.contains("third-person"));
}

#[test]
fn converter_system_prompt_author_mode_substitutes_name() {
    let prompt = converter_system_prompt(&ConverterMode::Author("Kael Dorn"));
    // All {author} occurrences must be replaced.
    assert!(!prompt.contains("{author}"));
    // The actual author name must appear.
    assert!(prompt.contains("Kael Dorn"));
}

#[test]
fn converter_system_prompt_author_mode_multiple_substitutions() {
    // The template uses {author} more than once; all must be replaced.
    let prompt = converter_system_prompt(&ConverterMode::Author("Voss"));
    assert!(!prompt.contains("{author}"));
    let count = prompt.matches("Voss").count();
    // The template references the author name at least twice.
    assert!(
        count >= 2,
        "expected at least 2 occurrences of 'Voss', got {count}"
    );
}

#[test]
fn converter_system_prompt_waypoint_mentions_json_array() {
    let prompt = converter_system_prompt(&ConverterMode::Waypoint);
    assert!(prompt.to_lowercase().contains("json array"));
}

#[test]
fn converter_system_prompt_author_mentions_json_array() {
    let prompt = converter_system_prompt(&ConverterMode::Author("Alice"));
    assert!(prompt.to_lowercase().contains("json array"));
}
