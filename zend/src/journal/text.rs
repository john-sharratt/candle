//! A dialogue's journal text: the turns behind each entry, whole, kept in the
//! substrate beside the conversation for `journal_read` and `journal_grep`.
//!
//! Each stretch a draft covers is one custom object under the conversation's
//! timeline ([`journal_core::transcript::key`]), written once and never
//! rewritten — so it is replicated and compacted like the rest of the
//! substrate, and ceased like any keyed object. The objects live as long as the
//! entries they are filed under: forgetting an entry ceases its own
//! ([`JournalText::cease_entry`]), and tombstoning or archiving the conversation
//! ceases every one of them ([`JournalText::cease_all`]) — an archived
//! conversation keeps its journal's entries, but nothing runs a tool on it
//! again, so the text behind them goes.

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::SharedSubstrate;
use candle_conversation::projection::TimelineId;
use journal_core::transcript::{
    entry_of, entry_prefix, owner_prefix, Book, Filed, Hits, Miss, Page, Part,
};
use serde_json::{json, Value};
use zend_tools::tools::journal::JournalBook;

use crate::journal::style::CODING;

/// The epoch a stretch is written under: none — the conversation's own writer
/// is the only one that files it.
const NO_LEASE: u64 = 0;

/// One conversation's journal text in the substrate.
pub struct JournalText {
    shared: SharedSubstrate,
    owner: String,
}

impl JournalText {
    pub fn of(shared: SharedSubstrate, timeline: TimelineId) -> Self {
        Self {
            shared,
            owner: timeline.raw().to_string(),
        }
    }

    /// Keep `filed`'s turns under its entry.
    pub fn file(&self, filed: &Filed) -> anyhow::Result<()> {
        if filed.parts.is_empty() {
            return Ok(());
        }
        let blob = json!({ "parts": filed.parts });
        self.shared
            .put_custom_object(CustomObjectPayload::new(filed.key(&self.owner), blob))
            .map_err(|e| anyhow::anyhow!("the journal's turns could not be kept: {e}"))
    }

    /// Every stretch kept, as `(entry, parts)` in key order.
    fn stretches(&self) -> Vec<(u64, Vec<Part>)> {
        self.shared
            .custom_objects_under(&owner_prefix(&self.owner))
            .into_iter()
            .filter_map(|(obj, _)| {
                let entry = entry_of(&self.owner, &obj.key)?;
                Some((entry, parts_of(&obj.blob)))
            })
            .collect()
    }

    /// Cease every stretch kept; how many there were.
    pub fn cease_all(&self) -> anyhow::Result<usize> {
        self.cease_under(&owner_prefix(&self.owner))
    }

    /// Cease the stretches filed under `entry`; how many there were.
    pub fn cease_entry(&self, entry: u64) -> anyhow::Result<usize> {
        self.cease_under(&entry_prefix(&self.owner, entry))
    }

    fn cease_under(&self, prefix: &str) -> anyhow::Result<usize> {
        let ceased: Vec<CustomObjectPayload> = self
            .shared
            .custom_objects_under(prefix)
            .into_iter()
            .map(|(obj, _)| CustomObjectPayload::ceased(obj.key))
            .collect();
        let n = ceased.len();
        self.shared
            .put_custom_objects_under(ceased, NO_LEASE)
            .map_err(|e| anyhow::anyhow!("the journal's turns could not be ceased: {e}"))?;
        Ok(n)
    }
}

/// The parts a stretch's blob holds; none when it does not read.
fn parts_of(blob: &Value) -> Vec<Part> {
    blob.get("parts")
        .and_then(|p| serde_json::from_value(p.clone()).ok())
        .unwrap_or_default()
}

impl JournalBook for JournalText {
    fn read(&self, entry: u64, from: usize, lines: usize) -> Result<Page, Miss> {
        Book::new(self.stretches(), &CODING).read(entry, from, lines)
    }

    fn grep(&self, pattern: &str, entry: Option<u64>, max: usize) -> Result<Hits, Miss> {
        Book::new(self.stretches(), &CODING).grep(pattern, entry, max)
    }
}

#[cfg(test)]
mod tests {
    use candle_conversation::persistence::Admit;

    use super::*;

    fn part(id: u64, text: &str) -> Part {
        Part {
            id,
            at_ms: 1_792_713_600_000 + id * 60_000,
            text: text.into(),
        }
    }

    fn opened(dir: &tempfile::TempDir) -> SharedSubstrate {
        SharedSubstrate::open_in(dir.path(), Admit::Unbound).unwrap()
    }

    fn timeline(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    #[test]
    fn a_filed_stretch_reads_back_under_its_entry_and_no_one_elses() {
        let dir = tempfile::tempdir().unwrap();
        let shared = opened(&dir);
        let mine = JournalText::of(shared.clone(), timeline(7));
        let theirs = JournalText::of(shared, timeline(70));
        mine.file(&Filed {
            entry: 1,
            parts: vec![
                part(1, "User: fix the scan"),
                part(2, "You: reading scan.rs"),
            ],
        })
        .unwrap();
        mine.file(&Filed {
            entry: 1,
            parts: vec![part(3, "Tool results: fn scan() {}")],
        })
        .unwrap();

        let page = mine.read(1, 1, 400).unwrap();
        assert_eq!(page.total, 3, "both stretches, in order: {}", page.text);
        assert_eq!(
            page.text,
            "#1 00:01 User: fix the scan\n#2 00:02 You: reading scan.rs\n#3 00:03 Tool results: \
             fn scan() {}"
        );
        assert_eq!(mine.grep("SCAN", None, 10).unwrap().total, 3);
        assert_eq!(theirs.read(1, 1, 10), Err(Miss::Nothing));
    }

    #[test]
    fn ceasing_a_conversations_text_leaves_nothing_to_read() {
        let dir = tempfile::tempdir().unwrap();
        let text = JournalText::of(opened(&dir), timeline(7));
        text.file(&Filed {
            entry: 2,
            parts: vec![part(5, "You: done")],
        })
        .unwrap();
        assert_eq!(text.cease_all().unwrap(), 1);
        assert_eq!(text.read(2, 1, 10), Err(Miss::Nothing));
        assert_eq!(text.cease_all().unwrap(), 0);
    }

    #[test]
    fn a_forgotten_entry_takes_only_its_own_text() {
        let dir = tempfile::tempdir().unwrap();
        let text = JournalText::of(opened(&dir), timeline(7));
        for entry in [1, 2] {
            text.file(&Filed {
                entry,
                parts: vec![part(entry, "You: step")],
            })
            .unwrap();
        }
        assert_eq!(text.cease_entry(1).unwrap(), 1);
        assert_eq!(text.read(1, 1, 10), Err(Miss::Entry(1)));
        assert_eq!(text.read(2, 1, 10).unwrap().total, 1);
    }

    #[test]
    fn a_stretch_with_no_turns_keeps_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let text = JournalText::of(opened(&dir), timeline(7));
        text.file(&Filed {
            entry: 1,
            parts: vec![],
        })
        .unwrap();
        assert_eq!(text.grep("x", None, 1), Err(Miss::Nothing));
    }
}
