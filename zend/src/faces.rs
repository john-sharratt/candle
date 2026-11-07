//! Each ingest unit's face (`docs/zend_working_set.md` §4.9).
//!
//! A unit opens on the request that commissioned its reading — "Read the
//! entire contents of …", "Summarize …" — and a dialogue that held that
//! opening would hold a user turn the user never wrote. So a dialogue holds a
//! unit's opening by its face instead (`Substrate::face_of`): a one-turn
//! conversation on the reserved `Face` layer, framed by the prelude every
//! dialogue primes on, whose user half is the reading's anchor alone and whose
//! assistant half is the unit's own opening — the calls it made — prefilled.
//!
//! [`fill`] gives every committed unit without a live face one, at the start and
//! the end of each branch-ingest pass, so a unit read this pass and one read
//! before faces existed meet the same rule. A face is one short prefill; nothing is read
//! again.

use std::time::Instant;

use anyhow::Context;
use candle_conversation::projection::{DecodePriority, Reserved, TurnIndex, TurnKey};
use candle_conversation::substrate::Substrate;
use candle_conversation::{ConversationEngine, Sequence};

use crate::branch_ingest::plan::Committed;
use crate::ingest_lease::{self, Share};
use crate::refresh_ctx::RefreshContext;

/// What kind of reading a unit is, which says how its anchor names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reading {
    /// A `code_reading` unit: one file.
    File,
    /// A `repo_map` unit: one folder, or a repository's root.
    Folder,
}

/// A face's user half: the reading's anchor and nothing else — the string its
/// content is fenced under, `file=<repo>/<path>` or `folder=<repo>/<dir>`, with
/// no request, no count and nothing to carry out.
pub fn anchor(reading: Reading, subject: &str) -> String {
    match reading {
        Reading::File => format!("file={subject}"),
        Reading::Folder => format!("folder={subject}"),
    }
}

/// How many faceless units [`fill`] works through between progress lines.
const PROGRESS_EVERY: usize = 100;

/// What [`fill`] has done so far with the units that lacked a face.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Tally {
    sealed: usize,
    failed: usize,
    /// Leased to another member, which seals it.
    theirs: usize,
    /// Given a face while this member waited for the lease.
    arrived: usize,
}

/// Seal a face for every unit of `units` that has no live one, each under its
/// lease so two members do not both seal it. Returns how many were sealed.
///
/// A backfill over a whole corpus is thousands of prefills at background
/// priority, so it says how many units it found faceless, how far it has got
/// every [`PROGRESS_EVERY`] of them, and how it ended.
pub fn fill(ctx: &RefreshContext<'_>, units: &[(Reading, Committed)]) -> usize {
    let faceless: Vec<&(Reading, Committed)> = units
        .iter()
        .filter(|(_, unit)| !has_face(ctx.engine, unit))
        .collect();
    if faceless.is_empty() {
        return 0;
    }
    let started = Instant::now();
    let of = faceless.len();
    tracing::info!(
        target: "zend::faces",
        faceless = of,
        units = units.len(),
        "sealing faces for the units that lack one",
    );
    // Every face forks this one conversation, so the prelude framing them is
    // sealed once and shared — never once per face, which over a corpus of
    // thousands of units wrote the prelude's K/V that many times.
    let base = match opened_base(ctx) {
        Ok(base) => base,
        Err(e) => {
            tracing::warn!(
                target: "zend::faces",
                "opening the faces' base conversation failed; no face is sealed: {e:#}",
            );
            return 0;
        }
    };
    let mut tally = Tally::default();
    for (done, (reading, unit)) in faceless.into_iter().enumerate() {
        if crate::ingest_cancel::cancelled() {
            break;
        }
        match ingest_lease::share_face(&unit.key) {
            Share::Theirs => tally.theirs += 1,
            // A previous holder's face may have arrived while the lease was asked.
            Share::Mine(guard) if guard.recheck() && has_face(ctx.engine, unit) => {
                tally.arrived += 1
            }
            Share::Mine(_guard) => match seal(ctx, &base, *reading, unit) {
                Ok(()) => tally.sealed += 1,
                Err(e) => {
                    tally.failed += 1;
                    tracing::warn!(
                        target: "zend::faces",
                        subject = %unit.subject,
                        "sealing the unit's face failed; its own opening stands: {e:#}",
                    );
                }
            },
        }
        if (done + 1) % PROGRESS_EVERY == 0 {
            tracing::info!(
                target: "zend::faces",
                done = done + 1,
                of,
                sealed = tally.sealed,
                failed = tally.failed,
                theirs = tally.theirs,
                elapsed_s = started.elapsed().as_secs(),
                "faces: progress",
            );
        }
    }
    tracing::info!(
        target: "zend::faces",
        of,
        sealed = tally.sealed,
        failed = tally.failed,
        theirs = tally.theirs,
        arrived = tally.arrived,
        elapsed_s = started.elapsed().as_secs(),
        "sealed ingest units' faces",
    );
    // The base holds no turn of its own; only its forks are faces.
    if let Err(e) = ctx.engine.tombstone_timeline(base.timeline_id()) {
        tracing::warn!(target: "zend::faces", "retiring the faces' base failed: {e:#}");
    }
    tally.sealed
}

/// The conversation every face of one [`fill`] forks: the `Face` layer, framed
/// by the prelude every dialogue primes on, holding no turn.
fn opened_base(ctx: &RefreshContext<'_>) -> anyhow::Result<Sequence> {
    ctx.engine
        .new_reserved_conversation(
            ctx.formatted_prompt,
            Reserved::Face,
            DecodePriority::Low,
            ctx.config.clone(),
        )
        .context("faces' base conversation")
}

/// Whether `unit` has a live face.
fn has_face(engine: &ConversationEngine, unit: &Committed) -> bool {
    engine
        .conversation()
        .read()
        .face_of(TurnKey::new(unit.timeline, TurnIndex(0)))
        .is_some()
}

/// Seal `unit`'s face: the face names its unit before its turn is prefilled,
/// and the unit names the face only once the turn holds, so a face that
/// failed part-way is never one a dialogue is pointed at.
fn seal(
    ctx: &RefreshContext<'_>,
    base: &Sequence,
    reading: Reading,
    unit: &Committed,
) -> anyhow::Result<()> {
    let e = ctx.engine;
    let (_, opening) = e
        .turn_texts(unit.timeline)
        .into_iter()
        .next()
        .context("the unit holds no turn")?;
    let mut face = base.fork().context("face conversation")?;
    let timeline = face.timeline_id();
    e.set_conversation_metadata(
        timeline,
        Substrate::FACE_OF_KEY,
        &unit.timeline.raw().to_string(),
    )
    .context("naming the face's unit")?;
    face.insert_turn(anchor(reading, &unit.subject), &opening)
        .context("prefilling the face")?;
    e.set_conversation_metadata(
        unit.timeline,
        Substrate::FACE_KEY,
        &timeline.raw().to_string(),
    )
    .context("naming the unit's face")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A face's user half is the anchor its content is fenced under**, and
    /// nothing else.
    #[test]
    fn a_face_names_its_reading_by_its_anchor_alone() {
        assert_eq!(
            anchor(Reading::File, "candle/zend/src/held_values.rs"),
            "file=candle/zend/src/held_values.rs"
        );
        assert_eq!(
            anchor(Reading::Folder, "candle/zend/src"),
            "folder=candle/zend/src"
        );
        assert_eq!(anchor(Reading::Folder, "candle"), "folder=candle");
    }
}
