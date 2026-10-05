//! `--self-check`: every stored conversation asked whether it is intact.
//!
//! A boot step, run once the substrate is loaded and the sections are rebuilt,
//! and before anything ingests. Each conversation in the dialogue and the
//! `repo_map` / `code_reading` layers is read back through a fork of its
//! layer's base and asked the four [`questions::QUESTIONS`]; one that answers
//! no to any of them is corrupt, and is tombstoned. With `--dry-run` nothing
//! is tombstoned and every verdict is reported, so the questions can be judged
//! before they are trusted with a deletion.
//!
//! An ingest conversation tombstoned here is read again by the ingest that
//! follows — its content key is gone with it. A dialogue tombstoned here is
//! gone.

mod ask;
mod candidates;
mod questions;

use std::collections::HashSet;
use std::sync::Mutex;

use candle_conversation::projection::{LayerId, TimelineId};
use candle_conversation::{ConversationEngine, Sequence, SequenceConfig};
use futures::stream::{self, StreamExt};

use crate::config::SelfCheck;
use crate::loading::LoadProgress;
use ask::Asker;
use candidates::{select, Candidate, Entry};
use questions::Findings;

/// Conversations asked at once. Each holds four ephemeral forks while its
/// questions decode, so this bounds the scheduler's extra slots at four times
/// it; wide enough for the waves to batch, narrow enough to leave the card to
/// the projections.
const CONVERSATIONS_IN_FLIGHT: usize = 8;

/// What a run found.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Report {
    pub checked: usize,
    pub corrupt: usize,
    pub tombstoned: usize,
    /// Conversations whose questions could not be put — a fork or a decode
    /// failed. Never tombstoned: nothing was learned about them.
    pub unanswered: usize,
}

/// Ask every checkable conversation and act on the answers as `mode` says.
///
/// `fork` opens a reader on a stored timeline: a fork of `layer`'s base onto
/// that timeline, which projects its history and writes nothing until a turn
/// is sealed on it (none is). `layers` are the layers `fork` can serve.
pub fn run(
    engine: &Mutex<ConversationEngine>,
    config: &SequenceConfig,
    layers: &HashSet<LayerId>,
    fork: &dyn Fn(LayerId, TimelineId) -> anyhow::Result<Sequence>,
    mode: SelfCheck,
    progress: &LoadProgress,
) -> anyhow::Result<Report> {
    let (asker, candidates) = {
        let e = engine.lock().unwrap();
        let resolver = e.conversation();
        let entries: Vec<Entry> = e
            .live_conversations()
            .into_iter()
            .map(|timeline| Entry {
                timeline,
                layer: resolver.timeline_target(timeline).map(|(l, _)| l),
                turns: e.timeline_turn_count(timeline),
                conv_id: e.conversation_conv_id(timeline),
                metadata: e.conversation_metadata(timeline).unwrap_or_default(),
            })
            .collect();
        (Asker::new(&e, config)?, select(entries, layers))
    };
    let total = candidates.len() as u64;
    tracing::info!(
        conversations = total,
        dry_run = matches!(mode, SelfCheck::DryRun),
        "self-check: asking every stored conversation whether it is intact",
    );
    progress.set_step_progress(0, total);

    let mut report = Report {
        checked: candidates.len(),
        ..Report::default()
    };
    let mut done = 0u64;
    let verdicts = futures::executor::block_on(
        stream::iter(candidates)
            .map(|c| {
                let asker = &asker;
                async move {
                    let findings = match fork(c.layer, c.timeline) {
                        Ok(conversation) => asker.check(&conversation).await,
                        Err(e) => Err(e),
                    };
                    (c, findings)
                }
            })
            .buffer_unordered(CONVERSATIONS_IN_FLIGHT)
            .inspect(|_| {
                done += 1;
                progress.set_step_progress(done, total);
            })
            .collect::<Vec<_>>(),
    );

    for (c, findings) in verdicts {
        match findings {
            Ok(f) if f.is_corrupt() => {
                report.corrupt += 1;
                if act_on_corrupt(engine, &c, &f, mode) {
                    report.tombstoned += 1;
                }
            }
            Ok(_) => tracing::debug!(
                timeline = c.timeline.raw(),
                conversation = %c.label,
                "self-check: intact",
            ),
            Err(e) => {
                report.unanswered += 1;
                tracing::warn!(
                    timeline = c.timeline.raw(),
                    conversation = %c.label,
                    "self-check: the questions could not be put — left as it is: {e:#}",
                );
            }
        }
    }
    tracing::info!(
        checked = report.checked,
        corrupt = report.corrupt,
        tombstoned = report.tombstoned,
        unanswered = report.unanswered,
        "self-check complete",
    );
    Ok(report)
}

/// Report one corrupt conversation and, outside a dry run, tombstone it.
/// Whether it was tombstoned.
fn act_on_corrupt(
    engine: &Mutex<ConversationEngine>,
    c: &Candidate,
    findings: &Findings,
    mode: SelfCheck,
) -> bool {
    let failed = findings.failed();
    if !matches!(mode, SelfCheck::Tombstone) {
        tracing::warn!(
            timeline = c.timeline.raw(),
            conversation = %c.label,
            failed = %failed,
            "self-check: CORRUPT (dry run — left in place)",
        );
        return false;
    }
    match engine.lock().unwrap().tombstone_timeline(c.timeline) {
        Ok(()) => {
            tracing::warn!(
                timeline = c.timeline.raw(),
                conversation = %c.label,
                failed = %failed,
                "self-check: CORRUPT — tombstoned",
            );
            true
        }
        Err(e) => {
            tracing::error!(
                timeline = c.timeline.raw(),
                conversation = %c.label,
                failed = %failed,
                "self-check: CORRUPT, and tombstoning it failed: {e}",
            );
            false
        }
    }
}
