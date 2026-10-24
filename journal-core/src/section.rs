//! The journal as the sections a system prompt carries, and the record a restart
//! rebuilds it from.
//!
//! A journal is a collection of sections of its writer's **own conversation**:
//! one section per entry it holds, and one for the items left open. They are
//! submitted as entries are written, and the sections of the entries a fold
//! takes are removed as the folded entry's is submitted, so they are sealed,
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

/// The journal's last section: the open items — or that nothing is open — and
/// the line that says the journal ends there.
pub fn render_open(items: &[Item], style: &Style) -> String {
    let open = if items.is_empty() {
        style.nothing_open.to_string()
    } else {
        let lines: Vec<String> = items
            .iter()
            .map(|i| format!("#{} {}", i.id, i.text.trim()))
            .collect();
        format!("{}\n{}", style.open_heading, lines.join("\n"))
    };
    format!("{open}\n\n{}", style.end)
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
    /// Entries folded or forgotten whose sections no reconcile has swept yet.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed: Vec<u64>,
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
    /// The sections to hold, oldest entry first, the open items last — that
    /// section present, even with nothing open, whenever there is a journal at
    /// all, because it is where the journal says it ends.
    pub sections: Vec<Section>,
    /// The sections of the entries a fold just took, which a conversation that
    /// did not see them go may still hold.
    pub removed: Vec<String>,
    record: String,
}

impl JournalPrompt {
    /// The sections for `kept`, whose entries are the newest ones, rendered in
    /// `style`, each dated against the one above it.
    pub fn of(kept: &Kept, style: &Style) -> Self {
        let mut sections: Vec<Section> = kept
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| Section {
                name: entry_section(e.id),
                text: e.render(i.checked_sub(1).map(|b| &kept.entries[b]), style),
            })
            .collect();
        if !(kept.entries.is_empty() && kept.open.is_empty()) {
            sections.push(Section {
                name: OPEN_SECTION.to_string(),
                text: render_open(&kept.open, style),
            });
        }
        Self {
            sections,
            removed: Vec::new(),
            record: serde_json::to_string(kept).unwrap_or_default(),
        }
    }

    /// This prompt, naming `removed` as the sections that just left it.
    pub fn without(self, removed: Vec<String>) -> Self {
        Self { removed, ..self }
    }

    /// Whether anything has been written down.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// The record the conversation's metadata holds.
    pub fn record(&self) -> &str {
        &self.record
    }

    /// The sections a conversation that did not see them go may still hold:
    /// the entries a fold just took. A fresh reconcile removes them, because
    /// streams an earlier run sealed are invisible to it. Anything else stale
    /// is merely not selected — a journal is read by name.
    pub fn sweep(&self) -> Vec<&str> {
        self.removed.iter().map(String::as_str).collect()
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
            removed: vec![],
        }
    }

    #[test]
    fn a_journal_with_nothing_written_holds_no_sections() {
        let p = JournalPrompt::of(&kept(&[], vec![]), &PLAIN);
        assert!(p.is_empty());
        assert!(p.removed.is_empty());
    }

    #[test]
    fn each_entry_is_a_section_named_for_it_and_open_items_come_last() {
        let p = JournalPrompt::of(&kept(&[3, 4], vec![item(2, " who holds it? ")]), &PLAIN);
        let names: Vec<&str> = p.sections.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["journal/3", "journal/4", "journal/open"]);
        assert_eq!(p.sections[0].text, entry_with_claim(3).render(None, &PLAIN));
        assert_eq!(
            p.sections[1].text,
            entry_with_claim(4).render(Some(&entry_with_claim(3)), &PLAIN)
        );
        assert_eq!(p.sections[2].text, "Open:\n#2 who holds it?\n\nEnd.");
    }

    /// The live failure: with no open section, a writer asked what was still
    /// open invented an answer.
    #[test]
    fn nothing_open_is_said_and_the_journal_says_where_it_ends() {
        let p = JournalPrompt::of(&kept(&[1], vec![]), &PLAIN);
        let names: Vec<&str> = p.sections.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["journal/1", "journal/open"]);
        assert_eq!(p.sections[1].text, "Open: nothing.\n\nEnd.");
    }

    #[test]
    fn the_sections_a_fold_took_are_swept() {
        let p = JournalPrompt::of(&kept(&[7, 8], vec![]), &PLAIN);
        assert!(
            p.sweep().is_empty(),
            "nothing is swept unless a fold took it"
        );
        let folded = p.without(vec![entry_section(1), entry_section(2)]);
        assert_eq!(folded.sweep(), ["journal/1", "journal/2"]);
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
