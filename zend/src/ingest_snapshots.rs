//! Ingest conversations keep no recurrent snapshot.
//!
//! A `code_reading` file or a `repo_map` folder is read once and never
//! continued: a dialogue turn borrows its K/V through projection, and a changed
//! file is read again as a new conversation. The snapshot each of its seals
//! wrote — one GDN store, 113 MB on this model, fixed whatever the turn's
//! length — is never read back, and the substrate keeps one live tail per
//! conversation for good. Measured on BlackWorld, 2026-10-19: the substrate
//! grew from 332 GB to 526 GB in twelve hours of ingest and load, with only
//! 6.7% of it dead.
//!
//! **The priming chain's links are the exception.** A conversation forked
//! from the chain seeds its recurrent memory from the chain's end
//! (`seed_recurrent_from_lineage`), so a link keeps its tail. A link is known
//! by its path ([`is_link_dir`], [`is_link_file`]) — not by how it was read,
//! since a link that failed its turn in the chain is read by the pool like any
//! file, and is re-pointed as a link on a later boot.

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;

use crate::branch_ingest::prime::{is_link_dir, is_link_file};
use crate::code_read::PATH_KEY;
use crate::repo_scan::DIR_KEY;

/// Smallest snapshot taken to be a kept state rather than a retired tail,
/// which encodes in a few dozen bytes — npcd's figure for its own read-only
/// conversations.
const KEPT_SNAPSHOT_BYTES: u64 = 4096;

/// An ingest unit's conversation, as the sweep sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit<'a> {
    /// A `code_reading` file, by its path.
    File(&'a str),
    /// A `repo_map` folder, by its directory.
    Folder(&'a str),
    /// A unit of a layer built on another, by its subject — read once, never
    /// continued, and never a chain link.
    Layer(&'a str),
}

impl Unit<'_> {
    fn is_link(&self) -> bool {
        match self {
            Unit::File(path) => is_link_file(path),
            Unit::Folder(dir) => is_link_dir(dir),
            Unit::Layer(_) => false,
        }
    }
}

/// What to do with one unit's snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// A chain link: its tail is read by the conversations forked from it.
    Keep,
    /// Retire the kept tail, and write no more.
    Retire,
    /// Already retired or never kept: only remember to write none.
    Note,
}

/// The verdict for `unit`, whose snapshot is `bytes` long if it has one.
pub fn verdict(unit: Unit<'_>, bytes: Option<u64>) -> Verdict {
    if unit.is_link() {
        Verdict::Keep
    } else if bytes.is_some_and(|b| b >= KEPT_SNAPSHOT_BYTES) {
        Verdict::Retire
    } else {
        Verdict::Note
    }
}

/// `timeline` has finished reading `unit`: unless it is a chain link, its
/// tail is retired and no later seal writes another.
pub fn finished(engine: &ConversationEngine, timeline: TimelineId, unit: Unit<'_>) {
    if !unit.is_link() {
        engine.mark_timeline_unresumed(timeline);
    }
}

/// Retire the tail of every ingest conversation that is not a chain link,
/// and mark each so none is written again. For a store written before the
/// mark existed, and for marks a restart forgot. Answers how many tails were
/// retired; one already retired costs a lookup.
pub fn retire_finished(engine: &ConversationEngine) -> usize {
    let mut retired = 0;
    let files = engine.conversations_with_metadata_key(PATH_KEY);
    let folders = engine.conversations_with_metadata_key(DIR_KEY);
    let units = files
        .iter()
        .filter(|(tl, _)| kind_is(engine, *tl, "code_read"))
        .map(|(tl, p)| (*tl, Unit::File(p)))
        .chain(
            folders
                .iter()
                .filter(|(tl, _)| kind_is(engine, *tl, "repo_map"))
                .map(|(tl, d)| (*tl, Unit::Folder(d))),
        );
    for (timeline, unit) in units {
        match verdict(unit, engine.recurrent_snapshot_bytes(timeline)) {
            Verdict::Keep => {}
            Verdict::Retire => {
                engine.mark_timeline_unresumed(timeline);
                retired += 1;
            }
            Verdict::Note => engine.note_timeline_unresumed(timeline),
        }
    }
    retired
}

fn kind_is(engine: &ConversationEngine, timeline: TimelineId, kind: &str) -> bool {
    engine
        .conversation_metadata(timeline)
        .is_some_and(|m| m.get("kind").map(String::as_str) == Some(kind))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file or folder that is not a link has its kept tail retired; one
    /// already retired, or never kept, is only noted.
    #[test]
    fn an_ordinary_unit_keeps_no_snapshot() {
        let file = Unit::File("candle/candle-nn/src/lib.rs");
        assert_eq!(verdict(file, Some(113_246_208)), Verdict::Retire);
        assert_eq!(verdict(file, Some(40)), Verdict::Note);
        assert_eq!(verdict(file, None), Verdict::Note);
        let folder = Unit::Folder("candle/candle-nn/src/");
        assert_eq!(verdict(folder, Some(KEPT_SNAPSHOT_BYTES)), Verdict::Retire);
    }

    /// The chain's links keep theirs: what is forked from them seeds from it.
    #[test]
    fn a_chain_link_keeps_its_snapshot() {
        assert_eq!(
            verdict(Unit::File("candle/README.md"), Some(113_246_208)),
            Verdict::Keep
        );
        assert_eq!(
            verdict(Unit::File("candle/claude.md"), Some(113_246_208)),
            Verdict::Keep
        );
        assert_eq!(
            verdict(Unit::Folder("candle/"), Some(113_246_208)),
            Verdict::Keep
        );
    }
}
