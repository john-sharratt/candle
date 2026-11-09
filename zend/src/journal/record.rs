//! The journal's record in its conversation's metadata.
//!
//! A submitted section's text cannot be read back from the substrate, so the
//! entries and open items are also kept as JSON under
//! [`META_KEPT`](journal_core::section::META_KEPT) in the conversation's
//! metadata. It is replicated and tombstoned with the conversation, and it is
//! what a restart — or the member a conversation is handed to — rebuilds the
//! journal from.

use std::collections::BTreeMap;

use journal_core::section::{restore, JournalPrompt, Kept, META_KEPT};

/// The key the journal's record is kept under.
pub const KEY: &str = META_KEPT;

/// The journal a conversation's recorded [`KEY`] holds, if it reads.
pub fn kept_in(recorded: Option<&str>) -> Option<Kept> {
    restore(recorded?)
}

/// The metadata that records `prompt`'s journal.
pub fn metadata_of(prompt: &JournalPrompt) -> BTreeMap<String, String> {
    BTreeMap::from([(META_KEPT.to_string(), prompt.record().to_string())])
}

#[cfg(test)]
mod tests {
    use journal_core::entry::Item;

    use super::*;
    use crate::journal::style::CODING;

    #[test]
    fn a_recorded_journal_reads_back_from_the_metadata() {
        let kept = Kept {
            entries: vec![],
            open: vec![Item {
                id: 2,
                text: "rerun".into(),
            }],
            next_item: 3,
            covered_to: 12,
            removed: vec![],
        };
        let metadata = metadata_of(&JournalPrompt::of(&kept, &CODING));
        assert_eq!(kept_in(metadata.get(KEY).map(String::as_str)), Some(kept));
    }

    #[test]
    fn metadata_with_no_record_or_a_broken_one_holds_no_journal() {
        assert_eq!(kept_in(None), None);
        assert_eq!(kept_in(Some("{")), None);
    }
}
