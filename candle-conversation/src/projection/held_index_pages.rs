//! The QSA index pages a live slot holds, found again without the log.
//!
//! A page left in the log is read on demand ([`IndexPage::Stored`]): recovery
//! no longer keeps every turn's page for the life of the process, which was
//! tens of GB of commit on a long-lived store. But a reprojection re-injects
//! the turns behind the first piece that changed — about eighty of them every
//! sixty-four tokens on a codebase tour — and each one read its page off the
//! log again: measured on BlackWorld (2026-10-21) at 7.1 s of a 14.9 s
//! rebuild total in a 56 s turn, 3,590 reads of pages up to a megabyte each.
//!
//! So the bytes a read produced are remembered here, weakly. A slot that
//! places a turn keeps that page's `Arc` with the piece it placed
//! (`PlacedPiece`), so a page stays reachable exactly while some live slot has
//! its turn placed, and goes when the last one lets go — the set held is the
//! working set of the slots, never the store.
//!
//! **Found by the record, not the stream.** An entry answers only while the
//! substrate still names the same page record ([`RecordIdentity`]) for its
//! stream. A relocated or re-emitted copy keeps its identity, so a page that
//! moved is still found; a new page for the stream is a new record, so the old
//! bytes are never handed out for it. A location would not do: whole-store
//! compaction writes everything again from the first segment, so one location
//! can come to hold a different page. A record with no identity (written before
//! identities existed) cannot be told from another, and is never held.
//!
//! Shared by every thread that projects, and lock-free: a concurrent map
//! whose sweep any inserter may run.
//!
//! [`IndexPage::Stored`]: crate::substrate::IndexPage::Stored

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use crate::index_pages::StoredPages;
use crate::persistence::identity::RecordIdentity;
use crate::persistence::streams::StreamId;

/// Entries a map holds before dead ones are first swept: a sweep runs when
/// the map reaches twice the size it was left at by the last one, so its cost
/// is spread over at least as many inserts as it visits.
const FIRST_SWEEP_AT: usize = 256;

#[derive(Debug)]
pub(crate) struct HeldIndexPages {
    pages: papaya::HashMap<StreamId, (RecordIdentity, Weak<StoredPages>)>,
    sweep_at: AtomicUsize,
}

impl Default for HeldIndexPages {
    fn default() -> Self {
        Self {
            pages: papaya::HashMap::new(),
            sweep_at: AtomicUsize::new(FIRST_SWEEP_AT),
        }
    }
}

impl HeldIndexPages {
    /// The page `stream_id`'s record `id` holds, if a slot still holds it.
    pub(crate) fn get(&self, stream_id: StreamId, id: RecordIdentity) -> Option<Arc<StoredPages>> {
        if id.is_none() {
            return None;
        }
        let pages = self.pages.pin();
        let (held_id, page) = pages.get(&stream_id)?;
        if *held_id != id {
            return None;
        }
        page.upgrade()
    }

    /// Remember `page` as what record `id` of `stream_id` holds. Nothing is
    /// kept alive by this: the entry answers only while a caller holds `page`.
    pub(crate) fn hold(&self, stream_id: StreamId, id: RecordIdentity, page: &Arc<StoredPages>) {
        if id.is_none() {
            return;
        }
        let pages = self.pages.pin();
        pages.insert(stream_id, (id, Arc::downgrade(page)));
        let at = self.sweep_at.load(Ordering::Acquire);
        // One inserter sweeps a given threshold: the one whose CAS moves it on.
        if pages.len() >= at
            && self
                .sweep_at
                .compare_exchange(at, usize::MAX, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            pages.retain(|_, (_, page)| page.strong_count() > 0);
            self.sweep_at
                .store((pages.len() * 2).max(FIRST_SWEEP_AT), Ordering::Release);
        }
    }

    /// Entries in the map, live or not yet swept.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.pages.pin().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(hlc: u64) -> RecordIdentity {
        RecordIdentity {
            origin: 7,
            oseq: hlc,
            gen: 1,
            epoch: 0,
            hlc,
        }
    }

    fn page(bytes: &[u8]) -> Arc<StoredPages> {
        Arc::new(StoredPages::new(bytes.to_vec()))
    }

    /// While a slot holds the page, the same record finds the same bytes —
    /// the very allocation, not a copy.
    #[test]
    fn a_held_page_is_found_by_its_record() {
        let held = HeldIndexPages::default();
        let p = page(b"rows");
        held.hold(StreamId(1), id(5), &p);
        let found = held.get(StreamId(1), id(5)).expect("held");
        assert!(Arc::ptr_eq(&found, &p));
        assert_eq!(found.bytes(), b"rows");
    }

    /// The map keeps nothing alive: once the last slot lets go, the page is
    /// read from the log again.
    #[test]
    fn a_page_no_slot_holds_is_not_found() {
        let held = HeldIndexPages::default();
        let p = page(b"rows");
        held.hold(StreamId(1), id(5), &p);
        drop(p);
        assert!(held.get(StreamId(1), id(5)).is_none());
    }

    /// A new page for the stream is a new record: the old bytes are never
    /// handed out for it, though a slot still holds them.
    #[test]
    fn a_new_record_for_the_stream_does_not_find_the_old_page() {
        let held = HeldIndexPages::default();
        let old = page(b"old rows");
        held.hold(StreamId(1), id(5), &old);
        assert!(held.get(StreamId(1), id(6)).is_none());
        assert!(held.get(StreamId(2), id(5)).is_none(), "another stream");
    }

    /// A record written before identities existed cannot be told from another
    /// such record, so it is never held.
    #[test]
    fn a_record_without_an_identity_is_never_held() {
        let held = HeldIndexPages::default();
        let p = page(b"rows");
        held.hold(StreamId(1), RecordIdentity::NONE, &p);
        assert_eq!(held.len(), 0);
        assert!(held.get(StreamId(1), RecordIdentity::NONE).is_none());
    }

    /// Dead entries are swept as the map grows, so it tracks the slots'
    /// working set rather than every page ever read.
    #[test]
    fn dead_entries_are_swept_as_the_map_grows() {
        let held = HeldIndexPages::default();
        let live = page(b"live");
        held.hold(StreamId(0), id(1), &live);
        for s in 1..(FIRST_SWEEP_AT as u64 * 4) {
            let p = page(b"gone");
            held.hold(StreamId(s), id(1), &p);
        }
        assert!(
            held.len() < FIRST_SWEEP_AT,
            "swept to {} entries",
            held.len()
        );
        assert!(
            held.get(StreamId(0), id(1)).is_some(),
            "the held page survives a sweep"
        );
    }
}
