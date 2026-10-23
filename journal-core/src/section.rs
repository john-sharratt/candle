//! The journal as the sections a system prompt carries, and the record a restart
//! rebuilds it from.
//!
//! A journal is a collection of sections of its writer's **own conversation**:
//! one section per kept entry, the newest [`IN_PROMPT`] of them, and one for the
//! items left open. They are submitted as entries are written and the section of
//! an entry that ages out of the newest few is removed, so they are sealed,
//! persisted and tombstoned with the conversation they belong to.
//!
//! # What survives a restart
//!
//! A submitted section's text is not recoverable from the substrate, so the same
//! entries are also kept as a JSON record in the conversation's metadata
//! ([`META_KEPT`]). The record is what [`restore`] reads, and the same sections
//! resubmitted on the first turn restore the sealed streams rather than
//! prefilling them again.

use serde::{Deserialize, Serialize};

use crate::entry::{Entry, Item};
use crate::journal::IN_PROMPT;
use crate::style::Style;

/// The collection a journal's sections are submitted to.
pub const JOURNAL: &str = "journal";

/// The section the open items are submitted under.
pub const OPEN_SECTION: &str = "journal/open";

/// Conversation metadata: the kept entries and open items, as JSON.
pub const META_KEPT: &str = "journal.kept";

/// The section entry `id` is submitted under.
pub fn entry_section(id: u64) -> String {
    format!("{JOURNAL}/{id}")
}

/// The open items as their section reads, or `None` when nothing is open.
pub fn render_open(items: &[Item], style: &Style) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    let lines: Vec<String> = items
        .iter()
        .map(|i| format!("#{} {}", i.id, i.text.trim()))
        .collect();
    Some(format!("{}\n{}", style.open_heading, lines.join("\n")))
}

/// What a journal keeps across a restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Kept {
    pub entries: Vec<Entry>,
    pub open: Vec<Item>,
    pub next_item: u64,
    /// The last turn settled — by a kept entry, or by a draft that found
    /// nothing to add. Not the entries' own coverage: a stretch skipped after
    /// the newest entry is settled too, and a restart must not draft it again.
    pub covered_to: u64,
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
    /// The section of the entry that has just left the newest [`IN_PROMPT`],
    /// which a conversation that did not see it leave may still hold.
    pub aged_out: Option<String>,
    record: String,
}

impl JournalPrompt {
    /// The sections for `kept`, whose entries are the newest ones, rendered in
    /// `style`.
    pub fn of(kept: &Kept, style: &Style) -> Self {
        let mut sections: Vec<Section> = kept
            .entries
            .iter()
            .map(|e| Section {
                name: entry_section(e.id),
                text: e.render(style),
            })
            .collect();
        if let Some(text) = render_open(&kept.open, style) {
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

    /// Whether anything has been written down.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// The record the conversation's metadata holds.
    pub fn record(&self) -> &str {
        &self.record
    }

    /// The section a conversation that did not see it leave may still hold: the
    /// entry that just aged out. A fresh reconcile removes it, because streams an
    /// earlier run sealed are invisible to it. Anything else stale is merely not
    /// selected — a journal is read by name.
    pub fn sweep(&self) -> Vec<&str> {
        self.aged_out.as_deref().into_iter().collect()
    }
}

/// What a conversation's metadata record holds, if it holds one that parses.
pub fn restore(record: &str) -> Option<Kept> {
    serde_json::from_str(record).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::testing::{entry, item};
    use crate::entry::{Claim, Kind};
    use crate::style::testing::PLAIN;

    fn entry_with_claim(id: u64) -> Entry {
        Entry {
            claims: vec![Claim {
                text: "The tower holds 40 ore.".into(),
                cite: vec![3],
                kind: Kind::Observed,
                perishable: true,
                typed: None,
                corrected: false,
            }],
            ..entry(id)
        }
    }

    fn kept(ids: &[u64], open: Vec<Item>) -> Kept {
        Kept {
            entries: ids.iter().map(|id| entry_with_claim(*id)).collect(),
            open,
            next_item: 9,
            covered_to: 0,
        }
    }

    #[test]
    fn a_journal_with_nothing_written_holds_no_sections() {
        let p = JournalPrompt::of(&kept(&[], vec![]), &PLAIN);
        assert!(p.is_empty());
        assert_eq!(p.aged_out, None);
    }

    #[test]
    fn each_entry_is_a_section_named_for_it_and_open_items_come_last() {
        let p = JournalPrompt::of(&kept(&[3, 4], vec![item(2, " who holds it? ")]), &PLAIN);
        let names: Vec<&str> = p.sections.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["journal/3", "journal/4", "journal/open"]);
        assert_eq!(p.sections[0].text, entry_with_claim(3).render(&PLAIN));
        assert_eq!(p.sections[2].text, "Open:\n#2 who holds it?");
    }

    #[test]
    fn nothing_open_is_no_section_rather_than_an_empty_one() {
        let p = JournalPrompt::of(&kept(&[1], vec![]), &PLAIN);
        assert_eq!(p.sections.len(), 1);
        assert_eq!(render_open(&[], &PLAIN), None);
    }

    #[test]
    fn the_entry_that_just_left_the_newest_five_is_named_as_aged_out() {
        let young = JournalPrompt::of(&kept(&[1, 2, 3, 4, 5], vec![]), &PLAIN);
        assert_eq!(young.aged_out, None);
        let old = JournalPrompt::of(&kept(&[2, 3, 4, 5, 6], vec![]), &PLAIN);
        assert_eq!(old.aged_out.as_deref(), Some("journal/1"));
        assert_eq!(old.sweep(), ["journal/1"]);
        assert!(young.sweep().is_empty());
    }

    #[test]
    fn the_record_reads_back_as_what_was_kept() {
        let k = kept(&[6, 7], vec![item(4, "who holds it?")]);
        let p = JournalPrompt::of(&k, &PLAIN);
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

    #[test]
    fn coverage_past_the_entries_reads_back() {
        let k = Kept {
            covered_to: 72,
            ..kept(&[3], vec![])
        };
        assert_eq!(restore(JournalPrompt::of(&k, &PLAIN).record()), Some(k));
    }
}
