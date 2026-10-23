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

/// The journal a conversation's metadata holds, if it holds one that reads.
pub fn kept_in(metadata: &BTreeMap<String, String>) -> Option<Kept> {
    restore(metadata.get(META_KEPT)?)
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
        };
        let metadata = metadata_of(&JournalPrompt::of(&kept, &CODING));
        assert_eq!(kept_in(&metadata), Some(kept));
    }

    #[test]
    fn metadata_with_no_record_or_a_broken_one_holds_no_journal() {
        assert_eq!(kept_in(&BTreeMap::new()), None);
        let broken = BTreeMap::from([(META_KEPT.to_string(), "{".to_string())]);
        assert_eq!(kept_in(&broken), None);
    }
}
