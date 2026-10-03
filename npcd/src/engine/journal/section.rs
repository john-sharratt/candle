//! The journal as the sections a character's system prompt carries.
//!
//! `docs/journal.md` §6. A character's journal is a collection of sections of its
//! **own conversation**, shaped like its mission: one section per kept entry, the
//! newest [`IN_PROMPT`] of them, and one for the items left open. The mind submits
//! them as entries are written and removes the section of an entry that ages out
//! of the newest five, so they are sealed, persisted and tombstoned with the
//! conversation they belong to, and the journal is read under `journal_intro`
//! while a character that has written nothing reads `journal_none`.
//!
//! # What survives a restart
//!
//! A submitted section's text is not recoverable from the substrate, so the same
//! entries are also kept as a JSON record in the conversation's metadata
//! ([`META_KEPT`]), keyed by whose they are ([`META_OF`]). The record is what
//! [`restore`] rebuilds a character's [`JournalState`] from, and the same
//! sections resubmitted on the first turn restore the sealed streams rather than
//! prefilling them again.
//!
//! [`JournalState`]: crate::engine::journal::state::JournalState

use serde::{Deserialize, Serialize};

use crate::engine::identity::JOURNAL;
use crate::engine::journal::entry::{Entry, Item};
use crate::engine::journal::state::IN_PROMPT;

/// The section the open items are submitted under.
pub const OPEN_SECTION: &str = "journal/open";

/// Conversation metadata: whose journal a conversation holds.
pub const META_OF: &str = "journal.of";
/// Conversation metadata: the kept entries and open items, as JSON.
pub const META_KEPT: &str = "journal.kept";

/// The heading the open-items section opens with.
const OPEN_HEADING: &str = "What you have left open:";

/// The section entry `id` is submitted under.
pub fn entry_section(id: u64) -> String {
    format!("{JOURNAL}/{id}")
}

/// The open items as their section reads, or `None` when nothing is open.
pub fn render_open(items: &[Item]) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    let lines: Vec<String> = items
        .iter()
        .map(|i| format!("#{} {}", i.id, i.text.trim()))
        .collect();
    Some(format!("{OPEN_HEADING}\n{}", lines.join("\n")))
}

/// What a journal keeps across a restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Kept {
    pub entries: Vec<Entry>,
    pub open: Vec<Item>,
    pub next_item: u64,
}

impl Kept {
    /// The newest entry's id, zero when there is none.
    pub fn newest(&self) -> u64 {
        self.entries.iter().map(|e| e.id).max().unwrap_or(0)
    }
}

/// One section to be held: its name and what it reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub name: String,
    pub text: String,
}

/// The journal as a conversation should hold it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalPrompt {
    /// The sections to hold, oldest entry first, the open items last.
    pub sections: Vec<Section>,
    /// The section of the entry that has just left the newest [`IN_PROMPT`], which
    /// a conversation that did not see it leave may still hold.
    pub aged_out: Option<String>,
    record: String,
}

impl JournalPrompt {
    /// The sections for `kept`, whose entries are the newest ones.
    pub fn of(kept: &Kept) -> Self {
        let mut sections: Vec<Section> = kept
            .entries
            .iter()
            .map(|e| Section {
                name: entry_section(e.id),
                text: e.render(),
            })
            .collect();
        if let Some(text) = render_open(&kept.open) {
            sections.push(Section {
                name: OPEN_SECTION.to_string(),
                text,
            });
        }
        let aged_out = kept
            .newest()
            .checked_sub(IN_PROMPT as u64)
            .filter(|id| *id >= 1)
            .map(entry_section);
        Self {
            sections,
            aged_out,
            record: serde_json::to_string(kept).unwrap_or_default(),
        }
    }

    /// Whether the character has written anything down.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// The record the conversation's metadata holds.
    pub fn record(&self) -> &str {
        &self.record
    }
}

/// What a conversation's metadata record holds, if it holds one that parses.
pub fn restore(record: &str) -> Option<Kept> {
    serde_json::from_str(record).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::journal::entry::{Claim, Kind};

    fn item(id: u64, text: &str) -> Item {
        Item {
            id,
            text: text.into(),
        }
    }

    fn entry(id: u64) -> Entry {
        Entry {
            id,
            from_turn: 1,
            to_turn: 9,
            from_ms: 0,
            to_ms: 1,
            claims: vec![Claim {
                text: "The tower holds 40 ore.".into(),
                cite: vec![3],
                kind: Kind::Observed,
                perishable: true,
                typed: None,
                corrected: false,
            }],
            intend: vec![],
            opened: vec![],
            resolved: vec![],
        }
    }

    fn kept(ids: &[u64], open: Vec<Item>) -> Kept {
        Kept {
            entries: ids.iter().map(|id| entry(*id)).collect(),
            open,
            next_item: 9,
        }
    }

    #[test]
    fn a_journal_with_nothing_written_holds_no_sections() {
        let p = JournalPrompt::of(&kept(&[], vec![]));
        assert!(p.is_empty());
        assert_eq!(p.aged_out, None);
    }

    #[test]
    fn each_entry_is_a_section_named_for_it_and_open_items_come_last() {
        let p = JournalPrompt::of(&kept(&[3, 4], vec![item(2, " who holds it? ")]));
        let names: Vec<&str> = p.sections.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["journal/3", "journal/4", "journal/open"]);
        assert_eq!(p.sections[0].text, entry(3).render());
        assert_eq!(
            p.sections[2].text,
            "What you have left open:\n#2 who holds it?"
        );
    }

    #[test]
    fn nothing_open_is_no_section_rather_than_an_empty_one() {
        let p = JournalPrompt::of(&kept(&[1], vec![]));
        assert_eq!(p.sections.len(), 1);
        assert_eq!(render_open(&[]), None);
    }

    #[test]
    fn the_entry_that_just_left_the_newest_five_is_named_as_aged_out() {
        let young = JournalPrompt::of(&kept(&[1, 2, 3, 4, 5], vec![]));
        assert_eq!(young.aged_out, None);
        let old = JournalPrompt::of(&kept(&[2, 3, 4, 5, 6], vec![]));
        assert_eq!(old.aged_out.as_deref(), Some("journal/1"));
    }

    #[test]
    fn the_record_reads_back_as_what_was_kept() {
        let k = kept(&[6, 7], vec![item(4, "who holds it?")]);
        let p = JournalPrompt::of(&k);
        assert_eq!(restore(p.record()), Some(k));
    }

    #[test]
    fn a_record_that_is_not_one_is_not_restored() {
        assert_eq!(restore(""), None);
        assert_eq!(restore("{not json"), None);
    }

    #[test]
    fn the_newest_entry_is_the_highest_id() {
        assert_eq!(kept(&[4, 9, 6], vec![]).newest(), 9);
        assert_eq!(kept(&[], vec![]).newest(), 0);
    }
}
