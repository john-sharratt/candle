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
//! its tools the request's mode, so the prompt it decodes over is fixed and is
//! the one
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

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use anyhow::Context;
use candle_conversation::files::names::{chunk_name, heading_name};
use candle_conversation::files::{files_owner, FileRecord, FilesPin};
use candle_conversation::projection::Conversation;
use candle_conversation::projection::{
    DecodePriority, OptionalState, ReprojectionMode, SelectionState, TurnIndex, SEED_SELECTOR,
};
use candle_conversation::stencil::TriggerRegistry;
use candle_conversation::{
    format_name, ChunkLayout, PieceKind, PromptDump, SealAs, SlotLayout, SpanQuant, TransplantFrom,
};
use candle_conversation::{SamplingConfig, TurnOptions};
use serde::{Deserialize, Serialize};

use super::builders::turn_projection;
use super::sealed_fresh_layout::{compare, piece_chunks, LayoutComparison};
use super::sealed_fresh_reseal::{prefix_before, reseal_behind};
use super::ZendSession;
use crate::files::records;
use crate::journal::opening::SeedRead;
use crate::types::ToolMode;

/// The most tokens either side decodes.
pub const MAX_TOKENS: usize = 512;

/// The level a collection member's K/V is sealed at
/// (`Scheduler::section_compression_policy_member`).
const MEMBER_LEVEL: u8 = 4;

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
    /// Quantize the fresh side's prompt K/V in place before its user half:
    /// its sections at this level — `4` is a collection member's seal — and,
    /// laid out in spans, its turns and glue under the session's turn policy,
    /// so the fresh side attends over quantized K/V computed in its own
    /// context. Absent leaves it native.
    #[serde(default)]
    pub fresh_quantize: Option<u8>,
    /// How the fresh side's prompt lies in its slot.
    #[serde(default)]
    pub fresh_layout: Layout,
    /// Seal the file's members again behind the sections this prompt places
    /// ahead of them, and place those on the sealed side in place of the
    /// stored ones (`session/sealed_fresh_reseal.rs`).
    #[serde(default)]
    pub reseal: bool,
    /// Resealed, seal the file's members under a boundary section's policy —
    /// Q8 K and V, the seed's — instead of a member's.
    #[serde(default)]
    pub reseal_boundary: bool,
    /// Where the fresh side's prompt K/V comes from: its own prefill, or the
    /// sealed slot's chunks transplanted in their place.
    #[serde(default)]
    pub fresh_kv: Source,
    /// Where the fresh side's index comes from: its own prefill, or the sealed
    /// slot's index pages transplanted in their place.
    #[serde(default)]
    pub fresh_index: Source,
    /// The tools mode the sealed side projects under, as a chat turn's dial
    /// sets it — `none` places no tool block at all; absent keeps the
    /// dialogue base's own projection.
    #[serde(default)]
    pub tools: Option<ToolMode>,
    /// With a transplant, the prompt's last this many pieces are left out of
    /// it and prefilled live over the transplanted K/V and index — whether
    /// the prompt's own ending, computed in place, reads what sealed K/V
    /// before it holds.
    #[serde(default)]
    pub live_tail: usize,
    /// Other knowledge-base files placed after the file, in path order, until
    /// their tokens reach this — the file no longer the last thing the
    /// question reads.
    #[serde(default)]
    pub after_tokens: usize,
    /// The seed as files: the seed section left out (its dial absent) and
    /// each of its files placed instead as a member of the knowledge base,
    /// in the seed's order, ahead of the file — each sealed alone behind the
    /// prompt up to `files_open`, as the file is, none behind another.
    #[serde(default)]
    pub seed_as_files: bool,
}

/// The seed's files in the seed's order: each repository's root listing,
/// then the documents the seed read from it.
pub fn seed_paths(read: &SeedRead) -> Vec<String> {
    let mut out = Vec::new();
    for repo in &read.listings {
        out.push(format!("{repo}/"));
        let prefix = format!("{repo}/");
        out.extend(
            read.documents
                .iter()
                .filter(|d| d.starts_with(&prefix))
                .cloned(),
        );
    }
    out
}

/// The record this process can place for each of `paths`, in order; a path
/// with none is left out.
fn placeable_records(conversation: &Conversation, paths: &[String]) -> Vec<FileRecord> {
    paths
        .iter()
        .filter_map(|p| {
            records::of_path(conversation, p)
                .into_iter()
                .find(|r| records::placeable(conversation, r))
        })
        .collect()
}

/// The files placed after `needle` to reach `tokens`: every other path's
/// record this process can place, one version each, in path order, until
/// their members' tokens reach `tokens`; `exclude`'s paths are left out.
fn files_after(
    conversation: &Conversation,
    needle: &FileRecord,
    exclude: &[String],
    tokens: usize,
) -> Vec<FileRecord> {
    let mut by_path: BTreeMap<String, FileRecord> = BTreeMap::new();
    for r in records::all(conversation) {
        if r.path != needle.path
            && !exclude.contains(&r.path)
            && records::placeable(conversation, &r)
        {
            by_path.entry(r.path.clone()).or_insert(r);
        }
    }
    take_to(by_path.into_values(), tokens)
}

/// The leading `records` whose members' tokens reach `tokens`.
fn take_to(records: impl Iterator<Item = FileRecord>, tokens: usize) -> Vec<FileRecord> {
    let mut out = Vec::new();
    let mut held = 0usize;
    for r in records {
        if held >= tokens {
            break;
        }
        held += record_tokens(&r);
        out.push(r);
    }
    out
}

/// A record's members' tokens: its heading and every chunk.
fn record_tokens(r: &FileRecord) -> usize {
    r.heading.tokens as usize
        + r.chunks
            .iter()
            .map(|c| c.member.tokens as usize)
            .sum::<usize>()
}

/// Which side a part of the fresh slot comes from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// The fresh side's own.
    #[default]
    Fresh,
    /// The sealed slot's, transplanted.
    Sealed,
}

/// How the fresh side's prompt lies in its slot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// One prefill: the prompt's tokens packed into whole chunks.
    #[default]
    Contiguous,
    /// One prefill per piece of the prompt — every section, turn and run of
    /// glue in its own chunks, ending in its own partial chunk — as a
    /// projection lays sealed sections out.
    Spans,
    /// [`Layout::Spans`], mirroring the sealed side's policies: each piece
    /// quantized under the policy its sealed counterpart shows — a boundary
    /// section's Q8, where every band the sealed side holds for it is `Q8_KS`
    /// K and `Q8_0` V, else a member's level.
    Mirror,
}

/// A boundary section's formats: every K band `Q8_KS`, every V band `Q8_0`.
fn is_boundary(chunks: &[ChunkLayout]) -> bool {
    let all = |side: fn(&ChunkLayout) -> &Vec<Vec<u8>>, name: &str| {
        chunks
            .iter()
            .flat_map(|c| side(c).iter().flatten())
            .all(|&t| format_name(t) == name)
    };
    !chunks.is_empty() && all(|c| &c.k, "Q8_KS") && all(|c| &c.v, "Q8_0")
}

/// The fresh side's prompt as the pieces it is prefilled in, each with the
/// policy it is quantized under. [`Layout::Mirror`] reads each piece's policy
/// off `sealed`, the sealed side's slot; a section it cannot place there takes
/// the member level.
fn fresh_pieces(
    dump: &PromptDump,
    layout: Layout,
    level: Option<u8>,
    sealed: &SlotLayout,
) -> Vec<(Vec<u32>, SpanQuant)> {
    let section = level.map_or(SpanQuant::Native, SpanQuant::Level);
    match layout {
        Layout::Contiguous => vec![(dump.token_ids.clone(), section)],
        Layout::Mirror => {
            let member = SpanQuant::Level(level.unwrap_or(MEMBER_LEVEL));
            let pieces: Vec<_> = dump
                .pieces
                .iter()
                .filter(|p| !p.token_ids.is_empty())
                .collect();
            let tokens: Vec<usize> = pieces.iter().map(|p| p.token_ids.len()).collect();
            let cut = piece_chunks(&sealed.chunks, &tokens).ok();
            pieces
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let quant = match (&p.kind, &cut) {
                        (PieceKind::Section, Some(cut)) => {
                            let (a, b) = cut[i];
                            if is_boundary(&sealed.chunks[a..b]) {
                                SpanQuant::Boundary
                            } else {
                                member
                            }
                        }
                        (PieceKind::Section, None) => member,
                        _ => SpanQuant::Session,
                    };
                    (p.token_ids.clone(), quant)
                })
                .collect()
        }
        Layout::Spans => dump
            .pieces
            .iter()
            .map(|p| {
                let quant = match (&p.kind, level) {
                    (_, None) => SpanQuant::Native,
                    (PieceKind::Section, Some(_)) => section,
                    (_, Some(_)) => SpanQuant::Session,
                };
                (p.token_ids.clone(), quant)
            })
            .collect(),
    }
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
    /// The two slots' layouts compared piece by piece, when the fresh side
    /// ran in pieces it could be described in.
    pub layout: Option<LayoutComparison>,
    /// Resealed, whether the prompt with the resealed members placed holds
    /// exactly the tokens the stored members' prompt did.
    pub resealed_same_tokens: Option<bool>,
    /// The files placed after the file, in order, and their members' tokens.
    pub after_files: Vec<String>,
    pub after_tokens: usize,
    /// With the seed as files, the seed's files placed as members, in order,
    /// and the seed's files this process holds no record of.
    pub seed_files: Vec<String>,
    pub seed_missing: Vec<String>,
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

/// The selection the sealed side runs under: the opening its only projection,
/// every other dial the dialogue's default. Which tools it places is its
/// projection's ([`SealedFreshReq::tools`]).
fn selection(seed_as_files: bool) -> SelectionState {
    let mut s = SelectionState::new();
    s.set_reprojection_mode(ReprojectionMode::Opening);
    if seed_as_files {
        s.set_optional(SEED_SELECTOR, OptionalState::Absent);
    }
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
        // The files placed after the needle, to `after_tokens`.
        // The seed's files, placed as members ahead of the file when the seed
        // is asked for as files.
        let seed_paths = seed_paths(&state.branch_ingest.seed_read());
        let seed_files = if req.seed_as_files {
            placeable_records(&engine.conversation(), &seed_paths)
        } else {
            Vec::new()
        };
        let after = match &record {
            Some(r) => files_after(&engine.conversation(), r, &seed_paths, req.after_tokens),
            None => Vec::new(),
        };
        let pin_of = |needle: &FileRecord| {
            let mut keys: Vec<_> = seed_files.iter().map(FileRecord::key).collect();
            keys.push(needle.key());
            keys.extend(after.iter().map(FileRecord::key));
            FilesPin::Only(keys)
        };
        let pin = record.as_ref().map_or(FilesPin::Empty, pin_of);
        // The placed file's member sections in this process: what the prompt
        // holds that production's recurrence never ran over.
        let mut members: HashSet<u32> = record
            .as_ref()
            .map(|r| member_ids(&engine.conversation(), r))
            .unwrap_or_default();
        let mut probe: Option<FileRecord> = None;
        let mut conv = state
            .base_conv
            .fork_at(DecodePriority::Low)
            .map_err(|e| anyhow::anyhow!("forking the dialogue's base: {e}"))?;
        let timeline = conv.timeline_id();
        engine.conversation().mark_timeline_transient(timeline);
        engine.set_timeline_summarize(timeline, false);
        engine.files_pin(timeline, pin);
        if let Some(mode) = req.tools {
            conv.set_projection(turn_projection(&state, "sealed-fresh", None, mode));
        }
        let out = (|| {
            let selection = selection(req.seed_as_files);
            let mut dump = conv
                .prompt_dump_with(&selection)
                .context("the sealed side's prompt")?;
            // Resealed, the file's members are sealed again behind the
            // sections this prompt places ahead of them and placed in their
            // stead: the same tokens, read in the context they were computed in.
            let mut resealed_same_tokens = None;
            if let (true, Some(record)) = (req.reseal, record.as_ref()) {
                let conversation = engine.conversation();
                let heading = conversation
                    .read()
                    .owned_section_named(files_owner(), &heading_name(&record.path, &record.blob))
                    .context("the file's heading is not held here")?;
                let pieces: Vec<(Option<u32>, usize)> = dump
                    .pieces
                    .iter()
                    .map(|p| (p.section_id, p.token_ids.len()))
                    .collect();
                let prefix = prefix_before(&pieces, heading.raw())?;
                let decode = |ids: &[u32]| {
                    engine
                        .tokenizer()
                        .decode(ids, false)
                        .map_err(|e| anyhow::anyhow!("decoding a member: {e}"))
                };
                let kind = if req.reseal_boundary {
                    SealAs::Boundary
                } else {
                    SealAs::Member
                };
                let p = reseal_behind(
                    &conversation,
                    &state.base_conv,
                    decode,
                    record,
                    &prefix,
                    kind,
                )?;
                engine.files_pin(timeline, pin_of(&p));
                members = member_ids(&conversation, &p);
                probe = Some(p);
                let again = conv
                    .prompt_dump_with(&selection)
                    .context("the resealed prompt")?;
                resealed_same_tokens = Some(again.token_ids == dump.token_ids);
                dump = again;
            }
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
            let sealed_described = engine
                .describe_at_seal(conv.id())
                .map_err(|e| anyhow::anyhow!("describing the sealed slot: {e}"))?;
            let handle = conv
                .submit_turn_with_options(req.question.clone(), options)
                .map_err(|e| anyhow::anyhow!("submit: {e}"))?;
            let resp = handle
                .wait()
                .map_err(|e| anyhow::anyhow!("the sealed decode: {e}"))?;
            let sealed_layout = sealed_described
                .recv()
                .context("the sealed slot's description")?
                .map_err(|e| anyhow::anyhow!("describing the sealed slot: {e}"))?;
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
            let transplant = (req.fresh_kv == Source::Sealed || req.fresh_index == Source::Sealed)
                .then_some(TransplantFrom {
                    capture: sealed_layout.capture,
                    kv: req.fresh_kv == Source::Sealed,
                    index: req.fresh_index == Source::Sealed,
                    live_tail: req.live_tail,
                });
            if transplant.is_none() {
                engine.release_capture(sealed_layout.capture);
            }
            let one_pass = req.fresh_recurrent == Recurrent::Fresh
                && req.fresh_quantize.is_none()
                && req.fresh_layout == Layout::Contiguous
                && transplant.is_none();
            let fresh = if one_pass {
                engine.infer_raw_token_ids(&prompt, req.n)
            } else {
                engine.infer_raw_token_ids_swapped(
                    &fresh_pieces(&dump, req.fresh_layout, req.fresh_quantize, &sealed_layout),
                    recurrent_over,
                    transplant,
                    &turn[..user_half],
                    req.n,
                )
            }
            .map_err(|e| anyhow::anyhow!("the fresh decode: {e}"))?;
            let fresh_finish = fresh.finish.map_or("none".into(), |f| format!("{f:?}"));
            let layout = fresh
                .layout
                .as_ref()
                .map(|f| compare(&dump, &sealed_layout, f));
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
                layout,
                resealed_same_tokens,
                after_files: after.iter().map(|r| r.path.clone()).collect(),
                after_tokens: after.iter().map(record_tokens).sum(),
                seed_files: seed_files.iter().map(|r| r.path.clone()).collect(),
                seed_missing: seed_paths
                    .iter()
                    .filter(|p| req.seed_as_files && !seed_files.iter().any(|r| &r.path == *p))
                    .cloned()
                    .collect(),
            })
        })();
        let _ = engine.tombstone_timeline(timeline);
        if let Some(p) = &probe {
            let _ = records::retire(&engine.conversation(), p);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use candle_conversation::files::{ChunkRecord, FileUnit, LineSpan, MemberRef};
    use candle_conversation::persistence::content_hash::ContentHash;
    use candle_conversation::persistence::streams::ContentAddress;
    use candle_conversation::{PieceKind, PromptPiece};
    use candle_nn::kv_cache::ArenaFormatTag;

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

    /// **Laid out in spans, each piece is its own prefill under its own
    /// policy**: sections at the asked level, glue and turns under the
    /// session's; contiguous, the prompt is one piece; unquantized, native.
    #[test]
    fn the_fresh_prompt_is_cut_into_its_pieces_with_their_policies() {
        let pieces = vec![
            piece(Some(1), vec![10, 11]),
            piece(None, vec![20]),
            piece(Some(7), vec![30, 31]),
        ];
        let dump = PromptDump {
            token_ids: vec![10, 11, 20, 30, 31],
            token_count: 5,
            pieces,
            text: String::new(),
        };
        let none = layout_of(Vec::new());
        assert_eq!(
            fresh_pieces(&dump, Layout::Spans, Some(4), &none),
            vec![
                (vec![10, 11], SpanQuant::Level(4)),
                (vec![20], SpanQuant::Session),
                (vec![30, 31], SpanQuant::Level(4)),
            ]
        );
        assert_eq!(
            fresh_pieces(&dump, Layout::Contiguous, Some(4), &none),
            vec![(vec![10, 11, 20, 30, 31], SpanQuant::Level(4))]
        );
        assert!(fresh_pieces(&dump, Layout::Spans, None, &none)
            .iter()
            .all(|(_, q)| *q == SpanQuant::Native));
    }

    fn layout_of(chunks: Vec<ChunkLayout>) -> SlotLayout {
        SlotLayout {
            depth: 0,
            index_coverage: None,
            index_pages: Vec::new(),
            chunks,
            capture: 0,
        }
    }

    fn sealed_chunk(tokens: u16, k: ArenaFormatTag, v: ArenaFormatTag) -> ChunkLayout {
        ChunkLayout {
            offset: 0,
            tokens,
            k: vec![vec![k.as_u8(); 2]],
            v: vec![vec![v.as_u8(); 2]],
            layers_agree: true,
        }
    }

    /// **Mirrored, each section takes the policy its sealed counterpart
    /// shows**: Q8 where every band is `Q8_KS` K and `Q8_0` V, the member level
    /// where any band is adaptive, and glue the session's.
    #[test]
    fn a_mirrored_piece_takes_its_sealed_counterparts_policy() {
        let dump = PromptDump {
            token_ids: vec![10, 11, 20, 30, 31],
            token_count: 5,
            pieces: vec![
                piece(Some(1), vec![10, 11]),
                piece(None, vec![20]),
                piece(Some(7), vec![30, 31]),
            ],
            text: String::new(),
        };
        let sealed = layout_of(vec![
            sealed_chunk(2, ArenaFormatTag::Q8_KS, ArenaFormatTag::Q8_0),
            sealed_chunk(1, ArenaFormatTag::Q4_0, ArenaFormatTag::Q3_1),
            ChunkLayout {
                k: vec![vec![
                    ArenaFormatTag::Q8_KS.as_u8(),
                    ArenaFormatTag::Q5_1.as_u8(),
                ]],
                ..sealed_chunk(2, ArenaFormatTag::Q8_KS, ArenaFormatTag::Q8_0)
            },
        ]);
        assert_eq!(
            fresh_pieces(&dump, Layout::Mirror, None, &sealed),
            vec![
                (vec![10, 11], SpanQuant::Boundary),
                (vec![20], SpanQuant::Session),
                (vec![30, 31], SpanQuant::Level(MEMBER_LEVEL)),
            ]
        );
    }

    /// The sealed side's opening is its only projection.
    #[test]
    fn the_sealed_side_holds_its_prompt_to_the_opening() {
        assert_eq!(
            selection(false).reprojection_mode(),
            Some(ReprojectionMode::Opening)
        );
        assert_eq!(selection(false).optional(SEED_SELECTOR), None);
        assert_eq!(
            selection(true).optional(SEED_SELECTOR),
            Some(OptionalState::Absent)
        );
    }

    /// **The seed's files read in the seed's order**: each repository's
    /// listing, then that repository's documents.
    #[test]
    fn the_seeds_files_read_in_the_seeds_order() {
        let read = SeedRead {
            listings: vec!["candle".into(), "yenbot".into()],
            documents: vec![
                "candle/README.md".into(),
                "yenbot/README.md".into(),
                "candle/AGENTS.md".into(),
            ],
        };
        assert_eq!(
            seed_paths(&read),
            vec![
                "candle/",
                "candle/README.md",
                "candle/AGENTS.md",
                "yenbot/",
                "yenbot/README.md"
            ]
        );
    }

    fn record_of(path: &str, heading: u32, chunks: &[u32]) -> FileRecord {
        let member = |tokens| MemberRef {
            address: ContentAddress {
                prefix_hash: ContentHash::ZERO,
                section_hash: ContentHash::ZERO,
            },
            tokens,
        };
        FileRecord {
            content_key: format!("v1:{path}@b"),
            path: path.into(),
            blob: "b".into(),
            unit: FileUnit::File,
            lines: 1,
            lang: "rs".into(),
            branches: vec![],
            heading: member(heading),
            chunks: chunks
                .iter()
                .map(|&t| ChunkRecord {
                    lines: LineSpan { first: 1, last: 1 },
                    member: member(t),
                })
                .collect(),
        }
    }

    /// **Files are taken until their tokens reach the ask**, the one that
    /// crosses it included, and none past it.
    #[test]
    fn files_are_taken_until_their_tokens_reach_the_ask() {
        let records = [
            record_of("a", 10, &[90]),
            record_of("b", 10, &[40, 50]),
            record_of("c", 5, &[5]),
        ];
        assert_eq!(record_tokens(&records[1]), 100);
        let paths = |n: usize| -> Vec<String> {
            take_to(records.clone().into_iter(), n)
                .into_iter()
                .map(|r| r.path)
                .collect()
        };
        assert_eq!(paths(0), Vec::<String>::new());
        assert_eq!(paths(150), vec!["a", "b"]);
        assert_eq!(paths(10_000), vec!["a", "b", "c"]);
    }

    /// **Tools are the dialogue base's unless the request names a mode.**
    #[test]
    fn the_tools_mode_is_the_requests_when_named() {
        let req: SealedFreshReq = serde_json::from_str(r#"{"question":"q","n":8}"#).unwrap();
        assert_eq!(req.tools, None);
        let req: SealedFreshReq =
            serde_json::from_str(r#"{"question":"q","n":8,"tools":"none"}"#).unwrap();
        assert_eq!(req.tools, Some(ToolMode::None));
    }
}
