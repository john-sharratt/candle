//! Roleplay Engine (RPE) — narrator and input parser.
//!
//! # Overview
//!
//! The narrator module provides two entry points:
//!
//! - [`parse_turn`] — pure, no-model: parses multi-line clap-style text into a
//!   [`Vec<Input>`]. Use this for live player input in the chat app.
//!
//! - [`text_to_inputs`] — model-backed: converts free-form prose into a
//!   [`Vec<Input>`] JSON array using a correction loop. Use this for
//!   third-party NPC prose (waypoint mode, `author: None`) or first-person
//!   player prose (author mode, `author: Some(name)`).
//!
//! # Input wire format
//!
//! [`Input`] is the crate's wire format. It is serialised as a JSON array and
//! appended as the user message in the narrator conversation each turn. The
//! model responds with narrative prose only.
//!
//! # Session defaults
//!
//! [`SessionConfig`] carries the protagonist name (default for `say`/`act`
//! commands) and persona name (default for `cue` commands). These are resolved
//! at parse time so `Vec<Input>` never contains null character fields.

mod converter;
mod engine;
mod error;
mod parse;
mod types;

pub use converter::{
    character_system_prompt, converter_system_prompt, correction_message, narrate_system_prompt,
    text_to_inputs, text_to_inputs_streaming, ConverterMode,
};
pub use engine::NarratorEngine;
pub use error::{ConvertError, ParseError};
pub use parse::parse_turn;
pub use types::{NarratorInput, SessionConfig, DEFAULT_MAX_TURNS};
