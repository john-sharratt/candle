//! The conversation a title is decoded on: opened for that one title, and
//! retired with it.
//!
//! A title reads the titler's system prompt and the text it names, and nothing
//! else. Its conversation's group is **closed**, so no other title job's turns
//! are its candidates, and it is opened fresh, so it holds no title of its own
//! before this one: its projection places the frame and the message and scores
//! nothing. A titler conversation that lived across jobs held every title it
//! had ever written, and every title's projection scored them all on the
//! scheduler thread — 260 ms a title on BlackWorld (2026-11-19), which a new
//! conversation's first round waited behind.
//!
//! The frame — the system prompt's sealed section — is kept and shared by every
//! job (`ConversationEngine::plain_prompt_section`); the conversation is
//! transient and tombstoned however the job ends ([`Throwaway`]).

use std::sync::Arc;

use candle_conversation::projection::{Builder, DecodePriority, GroupId, LayerId, Reserved};
use candle_conversation::throwaway::Throwaway;
use candle_conversation::{ConversationEngine, Sequence, SequenceConfig};

use super::InferenceState;

/// The band a title decodes at: background to the dialogue it names, so it
/// never holds that dialogue's prompt back.
const TITLE_BAND: DecodePriority = DecodePriority::Normal;

/// What every title job opens its conversation from.
#[derive(Clone)]
pub(super) struct Titler {
    /// The system prompt, formatted in the dialect's markers.
    formatted: String,
    config: SequenceConfig,
}

/// One title's conversation, retired when it drops. The sequence is declared
/// first, so it — and its slot — drops ahead of the tombstone (`throwaway`).
pub(super) struct TitleConversation {
    pub sequence: Sequence,
    _retired: Throwaway<Arc<InferenceState>>,
}

impl Titler {
    pub(super) fn new(formatted: String, config: SequenceConfig) -> Self {
        Self { formatted, config }
    }

    /// Open the conversation one title is decoded on. Blocking: a scheduler
    /// round trip.
    pub(super) fn open(&self, state: &Arc<InferenceState>) -> anyhow::Result<TitleConversation> {
        let engine = &state.engine;
        let frame = engine.plain_prompt_section(&self.formatted)?;
        let builder =
            Builder::for_closed_reserved(&self.formatted, Reserved::Titler, frame, TITLE_BAND);
        let sequence = engine.new_conversation_with_projection(
            &self.formatted,
            builder,
            LayerId::reserved(Reserved::Titler),
            GroupId::reserved(Reserved::Titler),
            self.config.clone(),
        )?;
        let timeline = sequence.timeline_id();
        engine.mark_timeline_transient(timeline);
        Ok(TitleConversation {
            sequence,
            _retired: Throwaway::new(Arc::clone(state), timeline, "title"),
        })
    }
}

/// Tombstone every titler timeline the substrate holds: a title job's is gone
/// when the job is, so any left were written by a titler that kept its
/// conversation across jobs, holding the titles it wrote. Returns how many
/// went.
pub(super) fn retire_titler_timelines(engine: &ConversationEngine) -> usize {
    engine
        .group_conversations(GroupId::reserved(Reserved::Titler))
        .into_iter()
        .filter(|&tl| match engine.tombstone_timeline(tl) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(
                    timeline = tl.raw(),
                    "a titler timeline was not retired: {e}"
                );
                false
            }
        })
        .count()
}
