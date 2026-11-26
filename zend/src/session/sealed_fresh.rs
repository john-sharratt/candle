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
//!
//! **The reverse swap** (`fresh_recurrent: sealed`): the fresh side prefills
//! the prompt's K/V afresh, then takes the recurrent state production's
//! recurrence has — the prompt run from zero without the placed file's member
//! sections — before its user half. Answering correctly, fresh K/V alone
//! suffices; answering as sealed does, the recurrence the file never reached
//! matters too. `fresh_recurrent: replayed` installs the state of the whole
//! prompt the same way — the control that must answer as the plain fresh side
//! does before either swap is read.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Context;
use candle_conversation::files::names::{chunk_name, heading_name};
use candle_conversation::files::{files_owner, FileRecord, FilesPin};
use candle_conversation::projection::Conversation;
use candle_conversation::projection::{
    DecodePriority, OptionalState, ReprojectionMode, SelectionState, TurnIndex,
    TOOLS_ENABLED_SELECTOR,
};
use candle_conversation::stencil::TriggerRegistry;
use candle_conversation::PromptDump;
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
    /// The recurrent state the fresh side's turn starts from:
    /// [`Recurrent::Fresh`], its own prefill's, by default;
    /// [`Recurrent::Sealed`] the one production's recurrence has — the prompt
    /// run from zero without the placed file's tokens — under the fresh K/V.
    #[serde(default = "fresh")]
    pub fresh_recurrent: Recurrent,
    /// Quantize the fresh side's prompt K/V in place at this compression level
    /// before its user half — `4` is a collection member's seal — so the fresh
    /// side attends over quantized K/V computed in its own context. Absent
    /// leaves it native.
    #[serde(default)]
    pub fresh_quantize: Option<u8>,
}

fn fresh() -> Recurrent {
    Recurrent::Fresh
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
    /// The whole prompt's ids run from zero and installed over the slot's
    /// own state: the swap's control. On the fresh side it must decode as
    /// [`Recurrent::Fresh`] does, or the install, not the state, is what the
    /// swapped arms measure; on the sealed side it is [`Recurrent::Fresh`].
    Replayed,
    /// The fresh side's prompt and user half as two raw turns on one slot,
    /// nothing installed between them: the control for the slot carrying its
    /// own K/V and state from one raw turn into the next. On the sealed side
    /// it is [`Recurrent::Fresh`].
    Split,
}

/// The two decodes and where they part.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SealedFresh {
    /// The recurrent state the sealed side started from.
    pub recurrent: Recurrent,
    /// The recurrent state the fresh side started from.
    pub fresh_recurrent: Recurrent,
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
    /// Tokens each side decoded, and why each stopped: `Stop` for the
    /// model's own end, `Length` for the budget.
    pub sealed_decoded: usize,
    pub fresh_decoded: usize,
    pub sealed_finish: String,
    pub fresh_finish: String,
}

/// The first index where `a` and `b` differ, within the shorter; `None` when
/// they agree that far.
pub fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// The raw ids of `record`'s member sections in this process.
fn member_ids(conversation: &Conversation, record: &FileRecord) -> HashSet<u32> {
    let mut names = vec![heading_name(&record.path, &record.blob)];
    names.extend(
        record
            .chunks
            .iter()
            .map(|c| chunk_name(&record.path, &record.blob, c.lines)),
    );
    names
        .iter()
        .filter_map(|n| conversation.owned_section_named(files_owner(), n))
        .map(|id| id.raw())
        .collect()
}

/// `dump`'s ids without the pieces that are `members`: the prompt as
/// production's recurrence has run over it.
fn without_members(dump: &PromptDump, members: &HashSet<u32>) -> Vec<u32> {
    dump.pieces
        .iter()
        .filter(|p| !p.section_id.is_some_and(|id| members.contains(&id)))
        .flat_map(|p| p.token_ids.iter().copied())
        .collect()
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
        let record = match &req.file {
            None => None,
            Some(path) => Some(
                records::of_path(&engine.conversation(), path)
                    .into_iter()
                    .next()
                    .with_context(|| format!("no record of {path}"))?,
            ),
        };
        let pin = record
            .as_ref()
            .map_or(FilesPin::Empty, |r| FilesPin::Only(r.key()));
        // The placed file's member sections in this process: what the prompt
        // holds that production's recurrence never ran over.
        let members: HashSet<u32> = record
            .as_ref()
            .map(|r| member_ids(&engine.conversation(), r))
            .unwrap_or_default();
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
            if req.recurrent != Recurrent::Sealed {
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
            // The fresh side in one pass, unless its state is swapped or its
            // K/V quantized between the prompt and the user half.
            let recurrent_over = match req.fresh_recurrent {
                Recurrent::Fresh | Recurrent::Split => None,
                Recurrent::Sealed => Some(without_members(&dump, &members)),
                Recurrent::Replayed => Some(dump.token_ids.clone()),
            };
            let one_pass = req.fresh_recurrent == Recurrent::Fresh && req.fresh_quantize.is_none();
            let fresh = if one_pass {
                engine.infer_raw_token_ids(&prompt, req.n)
            } else {
                engine.infer_raw_token_ids_swapped(
                    &dump.token_ids,
                    recurrent_over,
                    req.fresh_quantize,
                    &turn[..user_half],
                    req.n,
                )
            }
            .map_err(|e| anyhow::anyhow!("the fresh decode: {e}"))?;
            let fresh_finish = fresh.finish.map_or("none".into(), |f| format!("{f:?}"));
            let fresh = fresh.tokens;
            let decode = |ids: &[u32]| engine.tokenizer().decode(ids, false).unwrap_or_default();
            let at = first_divergence(&sealed, &fresh);
            let agreed = &sealed[..at.unwrap_or(sealed.len().min(fresh.len()))];
            Ok(SealedFresh {
                recurrent: req.recurrent,
                fresh_recurrent: req.fresh_recurrent,
                prompt_tokens: prompt.len(),
                sealed_prompt_tokens: resp.stats.prefill_token_count,
                same_prompt: prompt.len() == resp.stats.prefill_token_count,
                first_divergence: at,
                sealed_token: at.map(|i| decode(&sealed[i..=i])),
                fresh_token: at.map(|i| decode(&fresh[i..=i])),
                agreed: decode(agreed),
                sealed_text: decode(&sealed),
                fresh_text: decode(&fresh),
                sealed_decoded: sealed.len(),
                fresh_decoded: fresh.len(),
                sealed_finish: format!("{:?}", resp.stats.finish),
                fresh_finish,
            })
        })();
        let _ = engine.tombstone_timeline(timeline);
        out
    }
}

#[cfg(test)]
mod tests {
    use candle_conversation::{PieceKind, PromptPiece};

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
        assert_eq!(req.fresh_recurrent, Recurrent::Fresh);
        let req: SealedFreshReq =
            serde_json::from_str(r#"{"question":"q","n":8,"fresh_recurrent":"sealed"}"#).unwrap();
        assert_eq!(req.fresh_recurrent, Recurrent::Sealed);
        let req: SealedFreshReq =
            serde_json::from_str(r#"{"question":"q","n":8,"fresh_recurrent":"replayed"}"#).unwrap();
        assert_eq!(req.fresh_recurrent, Recurrent::Replayed);
        assert_eq!(req.fresh_quantize, None);
        let req: SealedFreshReq =
            serde_json::from_str(r#"{"question":"q","n":8,"fresh_quantize":4}"#).unwrap();
        assert_eq!(req.fresh_quantize, Some(4));
    }

    fn piece(section_id: Option<u32>, token_ids: Vec<u32>) -> PromptPiece {
        PromptPiece {
            kind: if section_id.is_some() {
                PieceKind::Section
            } else {
                PieceKind::Glue
            },
            name: String::new(),
            section_id,
            timeline: None,
            index: None,
            role: None,
            score: 0.0,
            qualified: false,
            token_count: token_ids.len(),
            token_ids,
            text: String::new(),
        }
    }

    /// **Production's recurrence runs over every piece but the placed
    /// file's members**, in order, glue and the seed included.
    #[test]
    fn the_sealed_recurrence_skips_only_the_members() {
        let pieces = vec![
            piece(Some(1), vec![10, 11]),
            piece(None, vec![20]),
            piece(Some(7), vec![30, 31, 32]),
            piece(Some(8), vec![40]),
            piece(Some(2), vec![50]),
        ];
        let dump = PromptDump {
            token_ids: pieces.iter().flat_map(|p| p.token_ids.clone()).collect(),
            token_count: 8,
            pieces,
            text: String::new(),
        };
        let members: HashSet<u32> = [7, 8].into_iter().collect();
        assert_eq!(without_members(&dump, &members), vec![10, 11, 20, 50]);
        assert_eq!(without_members(&dump, &HashSet::new()), dump.token_ids);
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
