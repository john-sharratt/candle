//! Sealed against fresh: one question decoded twice over exactly the same
//! tokens — once on the production path, where the knowledge base's files are
//! sealed members spliced into the prompt and the seed a sealed section, and
//! once with every one of those tokens prefilled afresh on a bare slot — and
//! the first decoded token where the two part
//! (`docs/provenance_system_collection.md` §8, `docs/results/
//! files_collection_rollout_2026-11-26.md`).
//!
//! The sealed side is a fork of the dialogue's base, its knowledge base pinned
//! (one file, or none — the seed alone), reprojection held to the opening and
//! tools off, so the prompt it decodes over is fixed and is the one
//! [`Sequence::prompt_dump_with`] lays out before the turn. The fresh side is
//! that prompt's ids and the turn's own user half — the ids the sealed turn
//! was prefilled with — run through every layer in order. Both decode argmax
//! under no steering, so the first divergence is the first token the two K/V
//! states disagree on, and nothing else.
//!
//! **The swap** ([`Recurrent::Fresh`]): the sealed side takes the recurrent
//! state a fresh prefill of its prompt leaves, installed on its slot before its
//! turn, under its own sealed K/V. Answering as fresh does, the sealed K/V is
//! read correctly and the recurrence is what differs; answering as sealed
//! does, the K/V differs.

use std::sync::Arc;

use anyhow::Context;
use candle_conversation::files::{FileKey, FilesPin};
use candle_conversation::projection::{
    DecodePriority, OptionalState, ReprojectionMode, SelectionState, TurnIndex,
    TOOLS_ENABLED_SELECTOR,
};
use candle_conversation::stencil::TriggerRegistry;
use candle_conversation::{SamplingConfig, TurnOptions};
use serde::{Deserialize, Serialize};

use super::ZendSession;
use crate::files::records;

/// The most tokens either side decodes.
pub const MAX_TOKENS: usize = 512;

/// What to decode.
#[derive(Debug, Clone, Deserialize)]
pub struct SealedFreshReq {
    /// The question, as the user's message.
    pub question: String,
    /// The knowledge base's one file, by its workspace path; absent places
    /// none, the seed alone.
    #[serde(default)]
    pub file: Option<String>,
    /// Tokens to decode on each side, at most [`MAX_TOKENS`].
    pub n: usize,
    /// The recurrent state the sealed side's turn starts from.
    #[serde(default)]
    pub recurrent: Recurrent,
}

/// Where the sealed side's recurrent state comes from: the swap that tells a
/// K/V cause from a recurrent one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Recurrent {
    /// As production: the prompt branch's checkpoint, which never ran over a
    /// placed file.
    #[default]
    Sealed,
    /// A fresh prefill's: the prompt's ids run through every layer from zero,
    /// installed on the sealed slot before its turn, under the sealed K/V.
    Fresh,
}

/// The two decodes and where they part.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SealedFresh {
    /// The recurrent state the sealed side started from.
    pub recurrent: Recurrent,
    /// Tokens of the prompt both decoded over: the projection's, then the
    /// turn's own user half.
    pub prompt_tokens: usize,
    /// The sealed turn's depth when its prefill finished, as the scheduler
    /// counted it: equal to `prompt_tokens` when the fresh side was handed
    /// exactly the tokens the sealed side ran on.
    pub sealed_prompt_tokens: usize,
    pub same_prompt: bool,
    /// The index of the first decoded token the two disagree on; `None`
    /// when every token both decoded agrees.
    pub first_divergence: Option<usize>,
    /// The divergent token on each side, decoded.
    pub sealed_token: Option<String>,
    pub fresh_token: Option<String>,
    pub agreed: String,
    pub sealed_text: String,
    pub fresh_text: String,
}

/// The first index where `a` and `b` differ, within the shorter; `None` when
/// they agree that far.
pub fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// The selection the sealed side runs under: tools off, the opening its only
/// projection, every other dial the dialogue's default.
fn selection() -> SelectionState {
    let mut s = SelectionState::new();
    s.set_optional(TOOLS_ENABLED_SELECTOR, OptionalState::Absent);
    s.set_reprojection_mode(ReprojectionMode::Opening);
    s
}

impl ZendSession {
    /// Decode `req` sealed and fresh over the same tokens ([`SealedFresh`]).
    pub fn sealed_fresh(&self, req: &SealedFreshReq) -> anyhow::Result<SealedFresh> {
        anyhow::ensure!(
            (1..=MAX_TOKENS).contains(&req.n),
            "n must be 1..={MAX_TOKENS}"
        );
        let state = self
            .inference
            .load_full()
            .context("the model is not loaded")?;
        let engine = &state.engine;
        let pin = match &req.file {
            None => FilesPin::Empty,
            Some(path) => {
                let record = records::of_path(&engine.conversation(), path)
                    .into_iter()
                    .next()
                    .with_context(|| format!("no record of {path}"))?;
                FilesPin::Only(FileKey::of(&record.content_key))
            }
        };
        let mut conv = state
            .base_conv
            .fork_at(DecodePriority::Low)
            .map_err(|e| anyhow::anyhow!("forking the dialogue's base: {e}"))?;
        let timeline = conv.timeline_id();
        engine.conversation().mark_timeline_transient(timeline);
        engine.set_timeline_summarize(timeline, false);
        engine.files_pin(timeline, pin);
        let out = (|| {
            let selection = selection();
            let dump = conv
                .prompt_dump_with(&selection)
                .context("the sealed side's prompt")?;
            if req.recurrent == Recurrent::Fresh {
                conv.install_fresh_recurrent_over(dump.token_ids.clone())
                    .map_err(|e| anyhow::anyhow!("the fresh recurrent state: {e}"))?;
            }
            let options = TurnOptions {
                max_tokens: Some(req.n),
                sampling: Some(SamplingConfig::argmax()),
                selection,
                triggers: Arc::new(TriggerRegistry::new()),
                ..Default::default()
            };
            let handle = conv
                .submit_turn_with_options(req.question.clone(), options)
                .map_err(|e| anyhow::anyhow!("submit: {e}"))?;
            let resp = handle
                .wait()
                .map_err(|e| anyhow::anyhow!("the sealed decode: {e}"))?;
            conv.finish_turn(handle, &resp)
                .map_err(|e| anyhow::anyhow!("finish: {e}"))?;
            let turn = engine
                .conversation()
                .read()
                .token_ids_of(timeline, TurnIndex(0));
            let user_half = resp.stats.turn_prefill_tokens.min(turn.len());
            let mut prompt = dump.token_ids.clone();
            prompt.extend_from_slice(&turn[..user_half]);
            let sealed: Vec<u32> = resp.token_ids.to_vec();
            let fresh = engine
                .infer_raw_token_ids(&prompt, req.n)
                .map_err(|e| anyhow::anyhow!("the fresh decode: {e}"))?;
            let decode = |ids: &[u32]| engine.tokenizer().decode(ids, false).unwrap_or_default();
            let at = first_divergence(&sealed, &fresh);
            let agreed = &sealed[..at.unwrap_or(sealed.len().min(fresh.len()))];
            Ok(SealedFresh {
                recurrent: req.recurrent,
                prompt_tokens: prompt.len(),
                sealed_prompt_tokens: resp.stats.prefill_token_count,
                same_prompt: prompt.len() == resp.stats.prefill_token_count,
                first_divergence: at,
                sealed_token: at.map(|i| decode(&sealed[i..=i])),
                fresh_token: at.map(|i| decode(&fresh[i..=i])),
                agreed: decode(agreed),
                sealed_text: decode(&sealed),
                fresh_text: decode(&fresh),
            })
        })();
        let _ = engine.tombstone_timeline(timeline);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The first divergence is the first index the two disagree at**, and
    /// none when one runs out first in agreement.
    #[test]
    fn the_first_divergence_is_the_first_disagreeing_index() {
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 4]), Some(2));
        assert_eq!(first_divergence(&[9], &[1]), Some(0));
        assert_eq!(first_divergence(&[1, 2], &[1, 2, 3]), None);
        assert_eq!(first_divergence(&[], &[1]), None);
    }

    /// **The sealed side keeps production's recurrence unless the swap is
    /// asked for.**
    #[test]
    fn the_recurrent_state_is_sealed_unless_asked_fresh() {
        let req: SealedFreshReq = serde_json::from_str(r#"{"question":"q","n":8}"#).unwrap();
        assert_eq!(req.recurrent, Recurrent::Sealed);
        let req: SealedFreshReq =
            serde_json::from_str(r#"{"question":"q","n":8,"recurrent":"fresh"}"#).unwrap();
        assert_eq!(req.recurrent, Recurrent::Fresh);
    }

    /// The sealed side runs with tools off and its opening its only
    /// projection.
    #[test]
    fn the_sealed_side_holds_its_prompt_to_the_opening() {
        let s = selection();
        assert_eq!(
            s.optional(TOOLS_ENABLED_SELECTOR),
            Some(OptionalState::Absent)
        );
        assert_eq!(s.reprojection_mode(), Some(ReprojectionMode::Opening));
    }
}
