//! Restoring an archived conversation: its recorded turns decoded again onto a
//! conversation of their own.
//!
//! Archiving distils a conversation to text (`DistillMode::TextOnly`): its
//! token ids stay, and its K/V, signatures, index pages and projection points
//! are shed at the next compaction. The mark is one way, so the turns cannot be
//! filled in where they stand. Everything shed but the points is what a forward
//! computes from the ids, so decoding the turns again with their recorded ids
//! forced ([`replay_history`]) onto a timeline of their own rebuilds it — the
//! K/V, the recurrent snapshot, the pages the decode cuts and the signatures.
//!
//! The points cannot be rebuilt: each records what a turn's projection chose
//! over the substrate as it stood then. The restore keeps the ones its own
//! decode made, so the restored conversation carries a projection per turn, as
//! a conversation the chat ran does, and a diagnostic decode can run against
//! it.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{bail, Context};
use candle_conversation::projection::DecodePriority;
use candle_conversation::SelectionState;
use serde::Serialize;

use super::diag::HistoryDiffers;
use super::replay::{compare_turns, replay_history};
use super::{timeline_for, turn_projection, ZendSession};
use crate::types::ToolMode;

/// What to restore, and where to.
#[derive(Debug, Clone)]
pub struct RestoreSpec {
    /// The archived conversation.
    pub from: String,
    /// The conversation it is restored as. It must hold no turns.
    pub into: String,
    /// The turns' dials, as the chat builds them.
    pub selection: SelectionState,
    pub tools_mode: ToolMode,
    /// The identity the conversation runs as.
    pub identity: Option<String>,
}

/// A finished restore.
#[derive(Debug, Clone, Serialize)]
pub struct Restored {
    pub from: String,
    pub into: String,
    pub turns: usize,
    pub secs: f64,
    /// Turns whose restored ids or pages differ from the recording.
    pub differs: Vec<HistoryDiffers>,
}

/// The metadata a restore carries over: the dials the turns ran under and the
/// band the conversation decodes at. Everything else names the original — its
/// fork, its journal, its archiving — and stays with it.
fn carried(metadata: &BTreeMap<String, String>) -> Vec<(&str, &str)> {
    metadata
        .iter()
        .filter(|(k, _)| k.starts_with("dial_") || k.as_str() == DecodePriority::METADATA_KEY)
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect()
}

impl ZendSession {
    /// Run `spec`. Blocking, for as long as the turns take to decode — call it
    /// off the async runtime.
    pub fn restore_conversation(&self, spec: &RestoreSpec) -> anyhow::Result<Restored> {
        if spec.into == spec.from {
            bail!("a conversation is restored as another: the archived one stays distilled");
        }
        let state = self
            .inference
            .load_full()
            .context("the model is not loaded")?;
        let turns = self.recorded_conversation(&spec.from)?;
        if turns.is_empty() {
            bail!("{} has no turns to restore", spec.from);
        }
        let source = timeline_for(&spec.from);
        let recorded = state.base_conv.sealed_turns(source);
        let target = timeline_for(&spec.into);
        let held = state.base_conv.sealed_turns(target).len();
        if held > 0 {
            bail!("{} already holds {held} turns", spec.into);
        }

        let engine = &state.engine;
        engine
            .set_conversation_conv_id(target, &spec.into)
            .context("naming the restored conversation")?;
        if let Some(label) = engine.conversation_label_of(source) {
            engine
                .set_conversation_label(target, &label)
                .context("labelling the restored conversation")?;
        }
        if let Some(metadata) = engine.conversation_metadata(source) {
            for (key, value) in carried(&metadata) {
                engine
                    .set_conversation_metadata(target, key, value)
                    .with_context(|| format!("carrying {key} over"))?;
            }
        }

        let mut conv = state
            .base_conv
            .fork_resuming(target)
            .context("forking the restored conversation")?;
        conv.set_projection(turn_projection(
            &state,
            &spec.into,
            spec.identity.as_deref(),
            spec.tools_mode,
        ));
        let start = Instant::now();
        replay_history(
            &state,
            &mut conv,
            &turns,
            &recorded,
            &spec.selection,
            spec.tools_mode,
            &mut |_, _| {},
            // The points its decode streamed and the one at its seal, as the
            // chat persists a turn's.
            &mut |index, conv, mut points, response| {
                points.extend(conv.projection_event(&response.stats));
                conv.persist_projection_events(&points)
                    .with_context(|| format!("keeping turn {index}'s projections"))
            },
        )?;
        let secs = start.elapsed().as_secs_f64();
        let decode = |ids: &[u32]| state.tokenizer.decode(ids, false).unwrap_or_default();
        let differs = compare_turns(&recorded, &conv.sealed_turns(target), turns.len(), &decode)
            .iter()
            .filter(|t| !t.matches())
            .map(HistoryDiffers::of)
            .collect();
        Ok(Restored {
            from: spec.from.clone(),
            into: spec.into.clone(),
            turns: turns.len(),
            secs,
            differs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dials and the band go over; the fork, the journal and the archive
    /// flag stay with the original.
    #[test]
    fn a_restore_carries_the_dials_and_the_band_only() {
        let metadata: BTreeMap<String, String> = [
            ("dial_effort", "deep"),
            ("dial_think", "on"),
            (DecodePriority::METADATA_KEY, "urgent"),
            ("forked_from", "1880718856119606029"),
            ("journal.kept", "{}"),
            ("archived", "true"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let mut kept = carried(&metadata);
        kept.sort();
        let mut want = vec![
            ("dial_effort", "deep"),
            ("dial_think", "on"),
            (DecodePriority::METADATA_KEY, "urgent"),
        ];
        want.sort();
        assert_eq!(kept, want);
    }
}
