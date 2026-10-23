//! Every dialogue's journal, by conversation id.
//!
//! A journal is rebuilt from its conversation's metadata record the first time
//! anything asks for it, and held for the life of the process: a draft runs off
//! the conversation's lock and outlives the reply that started it, and the API
//! reads a journal while a reply holds that lock. What a draft needs of the
//! conversation itself — the asker its questions go through and the files its
//! claims are checked against — is registered when the conversation is opened,
//! so a draft can be started without the conversation's lock.
//!
//! The map's lock is never held while a record is read: the record is read
//! first, and kept only if no one registered the journal in the meantime.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_conversation::UnsealedAsker;
use journal_core::section::Kept;
use zend_vfs::RepoFiles;

use crate::journal::state::SessionJournal;

/// What a draft needs of an open conversation.
#[derive(Clone)]
pub struct Open {
    pub asker: UnsealedAsker,
    pub files: Arc<RepoFiles>,
}

struct Held {
    journal: Arc<Mutex<SessionJournal>>,
    open: Option<Open>,
}

#[derive(Default)]
pub struct Journals {
    held: Mutex<HashMap<String, Held>>,
}

impl Journals {
    /// `conv_id`'s journal, rebuilt from `restore` the first time it is asked
    /// for.
    pub fn of(
        &self,
        conv_id: &str,
        restore: impl FnOnce() -> Option<Kept>,
    ) -> Arc<Mutex<SessionJournal>> {
        if let Some(held) = self.get(conv_id) {
            return held;
        }
        let restored = restore().map_or_else(SessionJournal::new, SessionJournal::restore);
        let mut held = self.held.lock().unwrap();
        let entry = held.entry(conv_id.to_string()).or_insert_with(|| Held {
            journal: Arc::new(Mutex::new(restored)),
            open: None,
        });
        Arc::clone(&entry.journal)
    }

    /// `conv_id`'s journal, if anything has asked for it since the start.
    pub fn get(&self, conv_id: &str) -> Option<Arc<Mutex<SessionJournal>>> {
        self.held
            .lock()
            .unwrap()
            .get(conv_id)
            .map(|h| Arc::clone(&h.journal))
    }

    /// Note what a draft of `conv_id` is put through, now that it is open, and
    /// return its journal, rebuilt from `restore` if this is the first ask.
    pub fn opened(
        &self,
        conv_id: &str,
        open: Open,
        restore: impl FnOnce() -> Option<Kept>,
    ) -> Arc<Mutex<SessionJournal>> {
        let journal = self.of(conv_id, restore);
        if let Some(held) = self.held.lock().unwrap().get_mut(conv_id) {
            held.open = Some(open);
        }
        journal
    }

    /// What a draft of `conv_id` would be put through, while it is open.
    pub fn open(&self, conv_id: &str) -> Option<Open> {
        self.held.lock().unwrap().get(conv_id)?.open.clone()
    }

    /// `conv_id` was released from memory: its journal stays, its asker goes.
    pub fn closed(&self, conv_id: &str) {
        if let Some(held) = self.held.lock().unwrap().get_mut(conv_id) {
            held.open = None;
        }
    }

    /// `conv_id` was deleted, and its journal with it.
    pub fn forget(&self, conv_id: &str) {
        self.held.lock().unwrap().remove(conv_id);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use journal_core::entry::Entry;
    use journal_core::journal::Holds;

    use super::*;

    fn kept() -> Kept {
        Kept {
            entries: vec![Entry {
                id: 3,
                from_turn: 1,
                to_turn: 8,
                from_ms: 0,
                to_ms: 0,
                claims: vec![],
                intend: vec!["go on".into()],
                opened: vec![],
                resolved: vec![],
            }],
            open: vec![],
            next_item: 1,
        }
    }

    #[test]
    fn a_journal_is_restored_once_and_then_held() {
        let journals = Journals::default();
        let reads = Cell::new(0);
        let restore = || {
            reads.set(reads.get() + 1);
            Some(kept())
        };
        let first = journals.of("c-1", restore);
        assert_eq!(first.lock().unwrap().journal().written(), 3);
        let again = journals.of("c-1", || panic!("read again"));
        assert!(Arc::ptr_eq(&first, &again));
        assert_eq!(reads.get(), 1);
    }

    #[test]
    fn a_conversation_with_no_record_starts_an_empty_journal() {
        let journals = Journals::default();
        let j = journals.of("c-2", || None);
        assert_eq!(j.lock().unwrap().journal().written(), 0);
        assert!(journals.get("c-2").is_some());
        assert!(journals.get("c-3").is_none());
    }

    #[test]
    fn a_deleted_conversation_takes_its_journal_with_it() {
        let journals = Journals::default();
        journals.of("c-1", || Some(kept()));
        journals.forget("c-1");
        assert!(journals.get("c-1").is_none());
        let fresh = journals.of("c-1", || None);
        assert_eq!(fresh.lock().unwrap().journal().written(), 0);
    }

    #[test]
    fn a_conversation_never_opened_has_nothing_to_draft_through() {
        let journals = Journals::default();
        journals.of("c-1", || None);
        assert!(journals.open("c-1").is_none());
        journals.closed("c-1");
        assert!(journals.open("c-1").is_none());
        assert!(journals.open("nope").is_none());
    }
}
