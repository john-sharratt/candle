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
//! [`fill`] gives every committed unit without a live face one, at the end of
//! each branch-ingest pass, so a unit read this pass and one read before faces
//! existed meet the same rule. A face is one short prefill; nothing is read
//! again.

use anyhow::Context;
use candle_conversation::projection::{DecodePriority, Reserved, TurnIndex, TurnKey};
use candle_conversation::substrate::Substrate;
use candle_conversation::ConversationEngine;

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

/// Seal a face for every unit of `units` that has no live one, each under its
/// lease so two members do not both seal it. Returns how many were sealed.
pub fn fill(ctx: &RefreshContext<'_>, units: &[(Reading, Committed)]) -> usize {
    let mut sealed = 0;
    for (reading, unit) in units {
        if crate::ingest_cancel::cancelled() {
            break;
        }
        if has_face(ctx.engine, unit) {
            continue;
        }
        let guard = match ingest_lease::share_face(&unit.key) {
            Share::Mine(guard) => guard,
            Share::Theirs => continue,
        };
        // A previous holder's face may have arrived while the lease was asked.
        if guard.recheck() && has_face(ctx.engine, unit) {
            continue;
        }
        match seal(ctx, *reading, unit) {
            Ok(()) => sealed += 1,
            Err(e) => tracing::warn!(
                target: "zend::faces",
                subject = %unit.subject,
                "sealing the unit's face failed; its own opening stands: {e:#}",
            ),
        }
    }
    if sealed > 0 {
        tracing::info!(target: "zend::faces", sealed, "sealed ingest units' faces");
    }
    sealed
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
fn seal(ctx: &RefreshContext<'_>, reading: Reading, unit: &Committed) -> anyhow::Result<()> {
    let e = ctx.engine;
    let (_, opening) = e
        .turn_texts(unit.timeline)
        .into_iter()
        .next()
        .context("the unit holds no turn")?;
    let mut face = e
        .new_reserved_conversation(
            ctx.formatted_prompt,
            Reserved::Face,
            DecodePriority::Low,
            ctx.config.clone(),
        )
        .context("face conversation")?;
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
        assert_eq!(anchor(Reading::Folder, "candle/zend/src"), "folder=candle/zend/src");
        assert_eq!(anchor(Reading::Folder, "candle"), "folder=candle");
    }
}
