//! The index pages the pool read most recently, held in host memory.
//!
//! A pooled conversation's pages are read once and held on the model while it
//! stays pooled ([`super::pool_refresh`]). One that leaves lets them go, and
//! the working set's pieces at the edge of its token budget leave and come
//! back from one reprojection to the next. On BlackWorld (2026-11-18) every
//! refresh admitted 6–25 pieces and dropped 3–37 of about 180, and each
//! re-admission read its pages off the log again: 2.8–127 MB in 8–155 ms per
//! refresh, the same few pieces each second.
//!
//! A page read is found again without the log while any holder keeps its
//! `Arc` ([`crate::projection`]'s held index pages). The ring keeps the pages
//! the pool used last, least recently used first out, up to a byte budget, so
//! a piece that comes back within that reach is read from memory. **Used, not
//! read:** a page the pool finds in memory moves to the newest end, so the
//! pieces it keeps coming back to stay, where a ring ordered by when a page was
//! read off the log let them go first.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::index_pages::StoredPages;

/// The host bytes the ring keeps at most.
pub(super) const POOL_READ_RING_BYTES: usize = 768 << 20;

/// See the module documentation.
#[derive(Debug)]
pub(super) struct PoolReadRing {
    pages: VecDeque<Arc<StoredPages>>,
    bytes: usize,
    cap: usize,
}

impl PoolReadRing {
    pub fn new(cap: usize) -> Self {
        Self {
            pages: VecDeque::new(),
            bytes: 0,
            cap,
        }
    }

    /// `page` was used: keep it as the newest, moving it there if the ring
    /// holds it, and let the least recently used go past the budget. A page
    /// larger than the whole budget is not kept.
    pub fn keep(&mut self, page: &Arc<StoredPages>) {
        let size = page.bytes().len();
        if size > self.cap {
            return;
        }
        if let Some(at) = self.pages.iter().position(|p| Arc::ptr_eq(p, page)) {
            if let Some(held) = self.pages.remove(at) {
                self.pages.push_back(held);
            }
            return;
        }
        self.pages.push_back(Arc::clone(page));
        self.bytes += size;
        while self.bytes > self.cap {
            let Some(old) = self.pages.pop_front() else {
                break;
            };
            self.bytes -= old.bytes().len();
        }
    }

    /// Bytes held now.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Pages held now.
    pub fn len(&self) -> usize {
        self.pages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(n: usize) -> Arc<StoredPages> {
        Arc::new(StoredPages::new(vec![0u8; n]))
    }

    /// **The ring keeps the newest pages within its budget**: past it the
    /// oldest go first, and a page that goes is no longer held by the ring.
    #[test]
    fn the_newest_pages_are_kept_within_the_budget() {
        let mut ring = PoolReadRing::new(100);
        let (a, b, c) = (page(40), page(40), page(40));
        ring.keep(&a);
        ring.keep(&b);
        assert_eq!((ring.len(), ring.bytes()), (2, 80));
        ring.keep(&c);
        assert_eq!((ring.len(), ring.bytes()), (2, 80), "the oldest went");
        assert_eq!(Arc::strong_count(&a), 1, "a is held only here");
        assert_eq!(Arc::strong_count(&b), 2);
        assert_eq!(Arc::strong_count(&c), 2);
    }

    /// **A page used again moves to the newest end**, so the next to go is
    /// the one least recently used, not the one read off the log first; using
    /// it again does not hold it twice.
    #[test]
    fn a_page_used_again_is_the_last_to_go() {
        let mut ring = PoolReadRing::new(100);
        let (a, b, c) = (page(40), page(40), page(40));
        ring.keep(&a);
        ring.keep(&b);
        ring.keep(&a);
        assert_eq!((ring.len(), ring.bytes()), (2, 80));
        assert_eq!(Arc::strong_count(&a), 2, "held once");
        ring.keep(&c);
        assert_eq!(Arc::strong_count(&b), 1, "b went: a was used after it");
        assert_eq!(Arc::strong_count(&a), 2);
        assert_eq!(Arc::strong_count(&c), 2);
    }

    /// A page larger than the whole budget is not kept, and keeps nothing out.
    #[test]
    fn a_page_past_the_budget_is_not_kept() {
        let mut ring = PoolReadRing::new(100);
        let small = page(30);
        ring.keep(&small);
        ring.keep(&page(101));
        assert_eq!((ring.len(), ring.bytes()), (1, 30));
    }
}
