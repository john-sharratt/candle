//! Words, decoded for what a character means (`docs/npc_conversations.md`
//! §6.3, §11.4): free text on a fork of its live slot, under the response
//! templates of where the words go, with no acts and no reasoning.
//!
//! **Nothing is written back.** The fork reads the character's own system
//! prompt and history — it is the same person saying it — and is dropped once
//! the words come back, as a question put to it is.
//!
//! **No call and no reasoning can be decoded.** There is no grammar, so the
//! two are banned outright: `<think>` and the dialect's call marker are taken
//! out of the vocabulary for the turn, and the no-think selector is set.

use std::time::Instant;

use candle_conversation::projection::DecodePriority;
use candle_conversation::{AskScope, TurnOptions};

use super::Minds;
use crate::engine::identity::{self, Deliberation};
use crate::engine::prompt::Persona;
use crate::engine::reflect::plain_prose;
use crate::engine::talk::templates::Destination;
use crate::engine::tools::Within;

/// The tokens a voiced turn may not decode.
const BANNED: [&str; 2] = ["<think>", "<tool_call>"];

/// What to voice, and how.
pub struct Voicing<'a> {
    /// The user turn the words answer.
    pub prompt: &'a str,
    /// Where the words go: whose templates shape them.
    pub to: Destination,
    pub max_tokens: usize,
    pub band: DecodePriority,
}

impl Minds {
    /// Decode the words `v` asks `npc_id` for, on a fork of its live slot.
    ///
    /// Errors when the character has no live conversation, or the words came
    /// back empty.
    pub async fn voice(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        within: &Within,
        v: Voicing<'_>,
    ) -> anyhow::Result<String> {
        let started = Instant::now();
        let resident = self
            .resident_of(npc_id)
            .ok_or_else(|| anyhow::anyhow!("npc {npc_id} has no live conversation yet"))?;
        let (conversation, asker) = (&resident.conversation, &resident.asker);
        let mut selection = match conversation.try_lock() {
            Ok(mut live) => self.prepare(&mut live, npc_id, persona, within).await,
            Err(_) => asker.selection(),
        };
        identity::deliberate(&mut selection, Deliberation::None);
        if let Some(p) = self.projection.load().as_ref() {
            p.templates.offer(&mut selection, v.to);
        }
        let tok = self.engine.tokenizer();
        let banned: Vec<i32> = BANNED
            .iter()
            .filter_map(|t| tok.token_to_id(t))
            .map(|id| id as i32)
            .collect();
        let options = TurnOptions {
            sampling: Some(self.sampling_for(Deliberation::None)),
            max_tokens: Some(v.max_tokens),
            turn_banned_tokens: banned,
            selection,
            ..Default::default()
        };
        let response = asker
            .ask(v.prompt, v.band, AskScope::Whole, options)
            .await?;
        let words = plain_prose(&response.text);
        tracing::info!(
            npc_id,
            to = ?v.to,
            ms = started.elapsed().as_millis() as u64,
            tokens = response.stats.tokens_generated,
            "voiced"
        );
        if words.trim().is_empty() {
            anyhow::bail!("the words came back empty");
        }
        Ok(words)
    }
}
