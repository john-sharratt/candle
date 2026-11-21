//! Reads a conversation's recovered turns and the errors recorded against
//! them, for the history view.
//!
//! History is decoded from the substrate with the summariser's ghost turns
//! hidden, one exchange per turn in turn order. A turn that failed keeps its
//! error under a key that carries the turn's substrate index, so each failure
//! stays beside its own turn after a reload. A key whose suffix is not a turn
//! index is not an error record and is ignored.

use super::*;

/// A dialogue conversation's recovered history, as `GET /v1/conversations/{id}`
/// shows it — the summariser's ghost summary turns hidden.
pub(super) fn dialogue_history(state: &InferenceState, conv_id: &str) -> Vec<RecoveredMessage> {
    let timeline = timeline_for(conv_id);
    let base = &state.base_conv;
    base.recovered_history(timeline, false)
}

/// The conversation metadata key a failed turn's error is kept under — one per
/// turn, by the turn's substrate index, so every failure stays beside its own
/// turn ([`ZendSession::conversation_turn_errors`]).
pub(super) fn turn_error_key(turn: u32) -> String {
    format!("{TURN_ERROR_PREFIX}{turn}")
}

/// The prefix of [`turn_error_key`].
pub(super) const TURN_ERROR_PREFIX: &str = "turn_error.";

/// Every failed turn's error in a conversation's metadata, by turn index. A key
/// whose suffix is not a turn index is not one of these.
pub(super) fn turn_errors(meta: &POrdMap<String, String>) -> BTreeMap<u32, String> {
    meta.iter()
        .filter_map(|(k, v)| {
            let turn = k.strip_prefix(TURN_ERROR_PREFIX)?.parse().ok()?;
            Some((turn, v.clone()))
        })
        .collect()
}

impl ZendSession {
    /// Decoded turn history for a single recovered conversation — backs
    /// `GET /v1/conversations/{id}`. Returns `None` when the model isn't
    /// loaded yet; an empty `Vec` when the conv_id has no recovered turns.
    pub fn conversation_history(&self, conv_id: &str) -> Option<Vec<RecoveredMessage>> {
        let state = self.inference.load_full()?;
        if let Some(timeline) = passthrough::timeline_of(conv_id) {
            // A passthrough conversation's turns are the client's exchanges.
            let engine = &state.engine;
            // One exchange per turn, in turn order.
            let bubble = |role, text, turn: usize| RecoveredMessage {
                role,
                text,
                turn: turn as u32,
                no_think: false,
                thinking: None,
                tool_tokens: Vec::new(),
            };
            return Some(
                passthrough::held_history(engine, timeline)
                    .into_iter()
                    .enumerate()
                    .flat_map(|(turn, e)| {
                        [
                            bubble(TurnRole::User, e.user.text(), turn),
                            bubble(TurnRole::Assistant, e.assistant, turn),
                        ]
                    })
                    .collect(),
            );
        }
        Some(dialogue_history(&state, conv_id))
    }

    /// The composer dials this conversation last ran under, for the GUI to set
    /// its dials from when the conversation is opened. `None` when the model
    /// isn't loaded, or when no turn has recorded dials — a conversation that
    /// has never run keeps the client's own defaults.
    /// Why each of the conversation's failed turns stopped, by the turn's
    /// substrate index — recorded as the turn sealed (`turn_error_key`).
    pub fn conversation_turn_errors(&self, conv_id: &str) -> BTreeMap<u32, String> {
        let Some(state) = self.inference.load_full() else {
            return BTreeMap::new();
        };
        let timeline = conversation_timeline(conv_id);
        state
            .engine
            .conversation_metadata_with(timeline, turn_errors)
            .unwrap_or_default()
    }
}
