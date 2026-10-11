//! Each sealed section's per-position model state, shared by every substrate
//! that holds the section.
//!
//! The index page a section's own prefill produced, kept so the next section
//! that borrows this one's chunks — or a projection that injects it — can be
//! handed the rows that go with them (`ManagedBatchedModel::push_positional_state`).
//! Empty for every model that carries no such state.
//!
//! **Keyed by the section's stream id, never its `SectionId`.** A `SectionId`
//! is its schema's own — allocated in order as the schema is built — so two
//! tenants sharing one engine each hold a section `7` that are different
//! sections. The stream id is the section's content address, so the same text
//! under the same prefix has one page whoever holds it, and two different
//! sections never share one. Each page records which substrates hold it, and
//! goes only when the last of them lets it go.
//!
//! Process-local, and that is a real limit rather than a choice: a section
//! recovered cold from the substrate on a later run brings its page from the
//! log (`RestoreSection`), and one that has none selects against a prefix it
//! never indexed and says so.

use std::collections::HashMap;
use std::sync::Arc;

use crate::index_pages::StoredPages;
use crate::persistence::streams::StreamId;
use crate::projection::{Conversation, SubstrateRef};

/// One section's page and the substrates holding it.
struct Held {
    page: Arc<StoredPages>,
    holders: Vec<SubstrateRef>,
}

/// Every sealed section's index page, by stream id.
#[derive(Default)]
pub(crate) struct SectionPages {
    pages: HashMap<StreamId, Held>,
}

impl SectionPages {
    /// `holder` holds the section at `stream`, whose page is `page`. A later
    /// page for the same stream replaces the earlier one: they index the same
    /// tokens under the same prefix.
    pub(crate) fn insert(
        &mut self,
        stream: StreamId,
        holder: &Conversation,
        page: Arc<StoredPages>,
    ) {
        let held = self.pages.entry(stream).or_insert_with(|| Held {
            page: Arc::clone(&page),
            holders: Vec::new(),
        });
        held.page = page;
        held.holders.retain(|h| !h.is_gone());
        if !held.holders.iter().any(|h| h.is(holder)) {
            held.holders.push(holder.substrate_ref());
        }
    }

    /// The page for the section at `stream`.
    pub(crate) fn get(&self, stream: StreamId) -> Option<&Arc<StoredPages>> {
        self.pages.get(&stream).map(|h| &h.page)
    }

    /// `holder` no longer holds the section at `stream`; its page goes when no
    /// substrate does.
    pub(crate) fn release(&mut self, stream: StreamId, holder: &Conversation) {
        let Some(held) = self.pages.get_mut(&stream) else {
            return;
        };
        held.holders.retain(|h| !h.is(holder) && !h.is_gone());
        if held.holders.is_empty() {
            self.pages.remove(&stream);
        }
    }

    /// `holder` has detached: it holds nothing any more.
    pub(crate) fn forget(&mut self, holder: &Conversation) {
        self.pages.retain(|_, held| {
            held.holders.retain(|h| !h.is(holder) && !h.is_gone());
            !held.holders.is_empty()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(byte: u8) -> Arc<StoredPages> {
        Arc::new(StoredPages::new(vec![byte; 4]))
    }

    fn stream(n: u64) -> StreamId {
        StreamId(n)
    }

    /// Two substrates holding one section share its page; it stays while
    /// either holds it, and goes with the last.
    #[test]
    fn a_page_shared_by_two_substrates_goes_with_the_last() {
        let (zend, npcd) = (Conversation::new(), Conversation::new());
        let mut pages = SectionPages::default();
        pages.insert(stream(1), &zend, page(1));
        pages.insert(stream(1), &npcd, page(1));
        assert_eq!(pages.pages.len(), 1);
        assert_eq!(pages.pages[&stream(1)].holders.len(), 2);

        pages.release(stream(1), &zend);
        assert!(pages.get(stream(1)).is_some(), "npcd still holds it");
        pages.release(stream(1), &npcd);
        assert!(pages.get(stream(1)).is_none());
    }

    /// A detach lets go of everything its substrate held, and nothing another
    /// substrate holds.
    #[test]
    fn a_detach_forgets_only_its_own() {
        let (zend, npcd) = (Conversation::new(), Conversation::new());
        let mut pages = SectionPages::default();
        pages.insert(stream(1), &zend, page(1));
        pages.insert(stream(2), &npcd, page(2));
        pages.insert(stream(3), &zend, page(3));
        pages.insert(stream(3), &npcd, page(3));
        pages.forget(&npcd);
        assert!(pages.get(stream(1)).is_some());
        assert!(pages.get(stream(2)).is_none());
        assert!(pages.get(stream(3)).is_some(), "zend still holds it");
    }

    /// A substrate whose every handle has gone holds nothing.
    #[test]
    fn a_vanished_holder_holds_nothing() {
        let mut pages = SectionPages::default();
        let gone = Conversation::new();
        pages.insert(stream(1), &gone, page(1));
        drop(gone);
        pages.release(stream(1), &Conversation::new());
        assert!(pages.get(stream(1)).is_none());
    }
}
