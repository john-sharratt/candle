//! A dialogue's journal text: the turns behind each entry, whole, kept in the
//! substrate beside the conversation for `journal_read` and `journal_grep`
//! (through [`crate::journal::shelf`]).
//!
//! Each stretch a draft covers is one custom object under the conversation's
//! timeline ([`journal_core::transcript::key`]), written once and never
//! rewritten but by a refile (below) — so it is replicated and compacted like
//! the rest of the substrate, and ceased like any keyed object. The objects
//! live as long as the entries they are filed under: forgetting an entry
//! ceases its own ([`JournalText::forget`]), and tombstoning or archiving the conversation
//! ceases every one of them ([`JournalText::cease_all`]) — an archived
//! conversation keeps its journal's entries, but nothing runs a tool on it
//! again, so the text behind them goes.
//!
//! A fold ceases its entries' text, except in a conversation owned through
//! `/mcp`, whose entries are cited by the conversations below it: there the
//! fold refiles them under its own entry ([`JournalText::refile`]) and notes
//! where each went, so a folded entry's address still reads its turns.

use candle_conversation::persistence::record::CustomObjectPayload;
use candle_conversation::persistence::SharedSubstrate;
use candle_conversation::projection::TimelineId;
use journal_core::transcript::{
    entry_of, entry_prefix, owner_prefix, Book, Filed, Hits, Miss, Page, Part,
};
use serde_json::{json, Value};

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

    /// Every stretch kept under `prefix` — the conversation's, or one entry's —
    /// as `(entry, parts)` in key order.
    fn stretches(&self, prefix: &str) -> Vec<(u64, Vec<Part>)> {
        self.shared
            .custom_objects_under(prefix)
            .into_iter()
            .filter_map(|(obj, _)| {
                let entry = entry_of(&self.owner, &obj.key)?;
                Some((entry, parts_of(&obj.blob)))
            })
            .collect()
    }

    /// Cease every stretch kept, and every note of a fold; how many stretches
    /// there were.
    pub fn cease_all(&self) -> anyhow::Result<usize> {
        self.cease_under(&folded_prefix(&self.owner))?;
        self.cease_under(&owner_prefix(&self.owner))
    }

    /// The entry `entry` was folded into, when a fold refiled it.
    fn folded_into(&self, entry: u64) -> Option<u64> {
        self.shared
            .custom_object(&folded_key(&self.owner, entry))
            .and_then(|obj| obj.blob.get("into").and_then(Value::as_u64))
    }

    /// Where `entry`'s turns are filed: itself, or the fold it went into —
    /// and that fold's, when a later fold took it in turn.
    fn resolved(&self, entry: u64) -> u64 {
        let mut at = entry;
        for _ in 0..MAX_FOLDS {
            match self.folded_into(at) {
                Some(into) if into != at => at = into,
                _ => break,
            }
        }
        at
    }

    /// File the turns kept under each of `folded` under `into` — the entry a
    /// fold wrote them into — and cease them where they were, so a citation of
    /// a folded entry still has its evidence to read.
    ///
    /// Everything is written before anything is ceased, so a refile that stops
    /// partway leaves every turn readable, and one run again writes the same
    /// objects. The fold's digest is already filed under `into`, keyed by its
    /// last turn; a stretch that begins at that same turn is filed with the
    /// digest after its own turns, as one stretch, so neither replaces the
    /// other — the one key a refile rewrites.
    pub fn refile(&self, folded: &[u64], into: u64) -> anyhow::Result<()> {
        for &entry in folded {
            for (_, parts) in self.stretches(&entry_prefix(&self.owner, entry)) {
                let mut filed = Filed { entry: into, parts };
                if let Some(held) = self.shared.custom_object(&filed.key(&self.owner)) {
                    for part in parts_of(&held.blob) {
                        if !filed.parts.contains(&part) {
                            filed.parts.push(part);
                        }
                    }
                }
                self.file(&filed)?;
            }
        }
        for &entry in folded {
            self.shared
                .put_custom_object(CustomObjectPayload::new(
                    folded_key(&self.owner, entry),
                    json!({ "into": into }),
                ))
                .map_err(|e| {
                    anyhow::anyhow!("the fold of entry {entry} could not be noted: {e}")
                })?;
        }
        for &entry in folded {
            self.cease_entry(entry)?;
        }
        Ok(())
    }

    /// Every entry a fold took, wherever it went: what a citation may still
    /// name.
    pub fn folded(&self) -> Vec<u64> {
        let prefix = folded_prefix(&self.owner);
        self.shared
            .custom_objects_under(&prefix)
            .into_iter()
            .filter_map(|(obj, _)| obj.key.strip_prefix(&prefix)?.parse().ok())
            .collect()
    }

    /// Cease the stretches filed under `entry`; how many there were.
    pub fn cease_entry(&self, entry: u64) -> anyhow::Result<usize> {
        self.cease_under(&entry_prefix(&self.owner, entry))
    }

    /// Forget `entry`: cease its stretches, its own note of a fold, and every
    /// note of an entry folded into it, which would otherwise lead to nothing.
    pub fn forget(&self, entry: u64) -> anyhow::Result<usize> {
        let notes: Vec<CustomObjectPayload> = self
            .shared
            .custom_objects_under(&folded_prefix(&self.owner))
            .into_iter()
            .filter(|(obj, _)| {
                obj.key == folded_key(&self.owner, entry)
                    || obj.blob.get("into").and_then(Value::as_u64) == Some(entry)
            })
            .map(|(obj, _)| CustomObjectPayload::ceased(obj.key))
            .collect();
        self.shared
            .put_custom_objects_under(notes, NO_LEASE)
            .map_err(|e| anyhow::anyhow!("the notes of folds could not be ceased: {e}"))?;
        self.cease_entry(entry)
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

/// What every note of a fold — which entry went into which — starts with.
const FOLDED: &str = "journal/folded/";

/// The most folds one entry is followed through: a day folds into a week, a
/// week into a month; more is a loop.
const MAX_FOLDS: usize = 8;

/// Every note of a fold of `owner`'s starts with this.
fn folded_prefix(owner: &str) -> String {
    format!("{FOLDED}{owner}/")
}

/// The note that `owner`'s `entry` was folded.
fn folded_key(owner: &str, entry: u64) -> String {
    format!("{}{entry:010}", folded_prefix(owner))
}

/// The parts a stretch's blob holds; none when it does not read.
fn parts_of(blob: &Value) -> Vec<Part> {
    blob.get("parts")
        .and_then(|p| serde_json::from_value(p.clone()).ok())
        .unwrap_or_default()
}

impl JournalText {
    /// `lines` lines of `entry`'s turns from line `from`, counted from 1.
    /// Reads the entry's own stretches alone: an entry with none is a miss of
    /// that entry, whatever the others hold.
    ///
    /// An entry a fold took reads from the fold's entry, the page opening with
    /// where it went, so an address cited before the fold still reads its
    /// evidence.
    pub fn read(&self, entry: u64, from: usize, lines: usize) -> Result<Page, Miss> {
        let at = self.resolved(entry);
        let stretches = self.stretches(&entry_prefix(&self.owner, at));
        if stretches.is_empty() {
            return Err(Miss::Entry(entry));
        }
        let mut page = Book::new(stretches, &CODING).read(at, from, lines)?;
        if at != entry {
            page.text = format!(
                "Entry {entry} was folded into entry {at}; its turns are filed there.\n{}",
                page.text
            );
        }
        Ok(page)
    }

    /// The lines that hold `pattern`, ignoring case, in `entry`'s turns —
    /// wherever a fold filed them — or in every entry's; the first `max`.
    pub fn grep(&self, pattern: &str, entry: Option<u64>, max: usize) -> Result<Hits, Miss> {
        let entry = entry.map(|e| self.resolved(e));
        let prefix = match entry {
            Some(e) => entry_prefix(&self.owner, e),
            None => owner_prefix(&self.owner),
        };
        let stretches = self.stretches(&prefix);
        if let (true, Some(e)) = (stretches.is_empty(), entry) {
            return Err(Miss::Entry(e));
        }
        Book::new(stretches, &CODING).grep(pattern, entry, max)
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
        assert_eq!(theirs.read(1, 1, 10), Err(Miss::Entry(1)));
        assert_eq!(theirs.grep("scan", None, 10), Err(Miss::Nothing));
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
        assert_eq!(text.read(2, 1, 10), Err(Miss::Entry(2)));
        assert_eq!(text.grep("done", None, 10), Err(Miss::Nothing));
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

    /// **A fold of an `/mcp` conversation keeps its entries' evidence**: the
    /// folded entries' turns read back under the fold's entry, beside its
    /// digest, and nothing is left under the folded ones.
    #[test]
    fn refiled_turns_read_back_under_the_fold_beside_its_digest() {
        let dir = tempfile::tempdir().unwrap();
        let text = JournalText::of(opened(&dir), timeline(7));
        text.file(&Filed {
            entry: 1,
            parts: vec![part(1, "User: fix the scan"), part(2, "You: reading")],
        })
        .unwrap();
        text.file(&Filed {
            entry: 2,
            parts: vec![part(3, "You: fixed it")],
        })
        .unwrap();
        // The fold's digest, keyed by its last turn — the same turn the
        // second stretch begins at.
        text.file(&Filed {
            entry: 3,
            parts: vec![part(3, "Read scan.rs, fixed the race.")],
        })
        .unwrap();

        text.refile(&[1, 2], 3).unwrap();

        assert_eq!(
            text.read(3, 1, 400).unwrap().text,
            "#1 00:01 User: fix the scan\n#2 00:02 You: reading\n#3 00:03 You: fixed it\n\
             #3 00:03 Read scan.rs, fixed the race."
        );
        // **A folded entry's address resolves to the fold**, and says so.
        let page = text.read(1, 1, 1).unwrap();
        assert_eq!(page.entry, 3);
        assert_eq!(
            page.text,
            "Entry 1 was folded into entry 3; its turns are filed there.\n\
             #1 00:01 User: fix the scan"
        );
        assert_eq!(text.grep("fixed it", Some(2), 10).unwrap().total, 1);

        // A later fold takes the fold in turn: the address follows it.
        text.refile(&[3], 9).unwrap();
        assert_eq!(text.read(1, 1, 1).unwrap().entry, 9);
        let mut folded = text.folded();
        folded.sort();
        assert_eq!(folded, [1, 2, 3]);

        assert_eq!(text.cease_all().unwrap(), 2);
        assert_eq!(text.read(1, 1, 10), Err(Miss::Entry(1)));
        assert!(text.folded().is_empty());
    }

    /// **A refile run again writes the same objects**: the digest is not
    /// repeated where a stretch carries it.
    #[test]
    fn a_refile_run_again_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let text = JournalText::of(opened(&dir), timeline(7));
        text.file(&Filed {
            entry: 1,
            parts: vec![part(3, "You: fixed it")],
        })
        .unwrap();
        text.file(&Filed {
            entry: 2,
            parts: vec![part(3, "digest")],
        })
        .unwrap();
        // A stretch written under the fold, its entry not yet ceased — where a
        // refile that stopped partway leaves it.
        text.file(&Filed {
            entry: 2,
            parts: vec![part(3, "You: fixed it"), part(3, "digest")],
        })
        .unwrap();
        text.refile(&[1], 2).unwrap();
        assert_eq!(
            text.read(2, 1, 400).unwrap().text,
            "#3 00:03 You: fixed it\n#3 00:03 digest"
        );
    }

    /// **Forgetting the fold forgets where its entries went**, so no address
    /// leads to an entry with nothing behind it.
    #[test]
    fn forgetting_a_fold_ceases_the_notes_that_lead_to_it() {
        let dir = tempfile::tempdir().unwrap();
        let text = JournalText::of(opened(&dir), timeline(7));
        text.file(&Filed {
            entry: 1,
            parts: vec![part(1, "User: hello")],
        })
        .unwrap();
        text.refile(&[1], 5).unwrap();
        assert_eq!(text.folded(), [1]);
        assert_eq!(text.forget(5).unwrap(), 1);
        assert!(text.folded().is_empty());
        assert_eq!(text.read(1, 1, 10), Err(Miss::Entry(1)));
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
