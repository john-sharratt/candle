//! Wire-format types for the narrator module.

use serde::{Deserialize, Serialize};

/// Wire-format input for one story event.
///
/// Serialised as a tagged JSON object with a `"type"` discriminant.
/// All `character` fields are concrete names — `null` is never valid.
///
/// # JSON schema
///
/// ```json
/// { "type": "say",   "character": "Kael Dorn", "text": "we have to get out" }
/// { "type": "act",   "character": "Kael Dorn", "action": "takes her hand" }
/// { "type": "scene", "description": "The roof begins to collapse." }
/// { "type": "cue",   "character": "Voss",      "action": "draws her sidearm" }
/// { "type": "beat",  "description": "steer toward the east corridor" }
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NarratorInput {
    /// A character speaks dialogue.
    Say { character: String, text: String },
    /// A character performs a physical action.
    Act { character: String, action: String },
    /// An environment or world state description not tied to any character.
    Scene { description: String },
    /// A forced character action — the character does not resist or deliberate.
    Cue { character: String, action: String },
    /// An optional single-turn narrative steering hint (not narrated directly).
    Beat { description: String },
}

/// Session-level defaults for [`parse_turn`](super::parse_turn) character resolution.
///
/// `protagonist` is used as the default `character` for `say` and `act`
/// commands when `--character` is omitted.
///
/// `persona` is used as the default `character` for `cue` commands when
/// `--character` is omitted.
///
/// `max_turns` controls the narrator conversation sliding window (caller
/// responsibility — not enforced by this crate).
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Default speaker/actor for `say` and `act` with no `--character` flag.
    pub protagonist: String,
    /// Default target for `cue` with no `--character` flag.
    pub persona: String,
    /// Maximum turns to retain in the narrator conversation window.
    /// Enforced by the caller. Use [`DEFAULT_MAX_TURNS`] if you have no
    /// preference.
    pub max_turns: usize,
}

/// Sensible default for [`SessionConfig::max_turns`].
pub const DEFAULT_MAX_TURNS: usize = 7;
