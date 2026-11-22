//! The cache proper: the slab, the entries it holds, and the one lock both are
//! touched under (`docs/hit_driven_projection.md` §7.6).
//!
//! **A block is in exactly one place at a time.** It is free in the slab, or
//! the block of one entry in the map, or reserved by the one reader filling
//! it. A reader reserves under the lock, fills outside it — no other thread
//! can reach a reserved block — and publishes under the lock again. Two
//! readers of the same page may both fill a block; the second to publish finds
//! the first's entry, gives its own block back, and answers with the entry.
//!
//! **An entry is evicted only when nothing holds it.** The map's `Arc` is then
//! the only reference to the page, and the page's bytes are referenced by that
//! page alone. A new reference is handed out only under the lock, so once the
//! lock sees an entry unheld, nothing can take it up again before it is gone.
//! A placement that uploads a page's bytes holds the page until its copies
//! have finished, so no copy reads a block after it is handed back.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use candle::host_region::HostRegion;

use crate::index_pages::StoredPages;
use crate::persistence::identity::RecordIdentity;
use crate::persistence::streams::StreamId;

use super::cached::CachedBytes;
use super::classes::{class_bytes, class_of};
use super::slab::{Block, Slab};

/// Entries evicted together when a class has no free block: the scan that
/// finds the oldest is paid once for several misses.
const EVICT_BATCH: usize = 8;

type Key = (StreamId, RecordIdentity);

/// See the module documentation.
pub struct IndexPageCache {
    state: Mutex<State>,
    hits: AtomicU64,
    misses: AtomicU64,
    uncached: AtomicU64,
    evicted: AtomicU64,
}

struct State {
    slab: Slab,
    entries: HashMap<Key, Entry>,
    /// Bytes of the blocks entries hold.
    held_bytes: usize,
    /// Recency clock: an entry's `used` is the clock when last handed out.
    clock: u64,
}

struct Entry {
    pages: Arc<StoredPages>,
    block: Block,
    used: u64,
}

impl Entry {
    /// Whether anything but the map holds the page or its bytes.
    fn held(&self) -> bool {
        Arc::strong_count(&self.pages) > 1 || self.pages.cached_holders() > 1
    }
}

/// What the cache holds and has done since it was made.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub pinned_bytes: u64,
    pub pageable_bytes: u64,
    /// Bytes of the blocks entries hold.
    pub held_bytes: u64,
    pub entries: u64,
    /// Pages found in the cache.
    pub hits: u64,
    /// Pages read into the cache.
    pub misses: u64,
    /// Pages read and not cached: larger than the largest block, or no block
    /// could be had.
    pub uncached: u64,
    pub evicted: u64,
}

impl IndexPageCache {
    pub fn new(regions: Vec<HostRegion>) -> Self {
        Self {
            state: Mutex::new(State {
                slab: Slab::new(regions),
                entries: HashMap::new(),
                held_bytes: 0,
                clock: 0,
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            uncached: AtomicU64::new(0),
            evicted: AtomicU64::new(0),
        }
    }

    /// The page record `id` of `stream` holds, if the cache has it.
    pub fn get(&self, stream: StreamId, id: RecordIdentity) -> Option<Arc<StoredPages>> {
        let mut state = self.lock();
        let found = state.hand_out(&(stream, id));
        if found.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        found
    }

    /// Read record `id` of `stream` — `record_len` bytes, header and padding
    /// included — into a block of the cache: `read` fills the block's first
    /// `record_len` bytes and answers the payload's range of them, or `None`
    /// when it could not. `None` when no block could be had, the record has no
    /// identity, or `read` failed; the caller then reads the page elsewhere.
    pub fn read_into(
        &self,
        stream: StreamId,
        id: RecordIdentity,
        record_len: usize,
        read: impl FnOnce(&mut [u8]) -> Option<Range<usize>>,
    ) -> Option<Arc<StoredPages>> {
        let (block, base) = self.reserve(id, record_len)?;
        // SAFETY: the block is reserved for this call alone — in no entry and
        // no free list — and holds `class_bytes(block.class)` ≥ `record_len`
        // bytes of a live region.
        let dest = unsafe { std::slice::from_raw_parts_mut(base, record_len) };
        let range = read(dest).filter(|r| r.start <= r.end && r.end <= record_len);
        self.publish(stream, id, block, base, range)
    }

    /// `payload`, the page record `id` of `stream` holds, copied into a block
    /// of the cache. `None` as for [`Self::read_into`].
    pub fn insert(
        &self,
        stream: StreamId,
        id: RecordIdentity,
        payload: &[u8],
    ) -> Option<Arc<StoredPages>> {
        self.read_into(stream, id, payload.len(), |dest| {
            dest.copy_from_slice(payload);
            Some(0..payload.len())
        })
    }

    pub fn stats(&self) -> CacheStats {
        let state = self.lock();
        let (pinned, pageable) = state.slab.capacity();
        CacheStats {
            pinned_bytes: pinned as u64,
            pageable_bytes: pageable as u64,
            held_bytes: state.held_bytes as u64,
            entries: state.entries.len() as u64,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            uncached: self.uncached.load(Ordering::Relaxed),
            evicted: self.evicted.load(Ordering::Relaxed),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // A panic under the lock leaves the map and slab consistent at every
        // step, so the state is taken up again rather than poisoning reads.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A block for a record of `record_len` bytes, and its first byte.
    fn reserve(&self, id: RecordIdentity, record_len: usize) -> Option<(Block, *mut u8)> {
        let class = class_of(record_len).filter(|_| !id.is_none());
        let mut state = self.lock();
        let block = class.and_then(|c| state.take(c, &self.evicted));
        match block {
            Some(b) => Some((b, state.slab.base(b))),
            None => {
                self.uncached.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Install what a reader filled into `block`, or give the block back.
    fn publish(
        &self,
        stream: StreamId,
        id: RecordIdentity,
        block: Block,
        base: *mut u8,
        range: Option<Range<usize>>,
    ) -> Option<Arc<StoredPages>> {
        let mut state = self.lock();
        let Some(range) = range else {
            state.slab.give(block);
            return None;
        };
        let key = (stream, id);
        if let Some(found) = state.hand_out(&key) {
            // Another reader published the same page first.
            state.slab.give(block);
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(found);
        }
        // SAFETY: `range` lies inside the block's first `record_len` bytes,
        // which the reader filled and nothing writes again while the bytes
        // are held; the regions are kept alive by the bytes themselves.
        let bytes = unsafe {
            CachedBytes::new(
                Arc::clone(state.slab.regions()),
                base.add(range.start),
                range.len(),
            )
        };
        let pages = Arc::new(StoredPages::cached(Arc::new(bytes)));
        state.clock += 1;
        let used = state.clock;
        state.held_bytes += class_bytes(block.class);
        state.entries.insert(
            key,
            Entry {
                pages: Arc::clone(&pages),
                block,
                used,
            },
        );
        self.misses.fetch_add(1, Ordering::Relaxed);
        Some(pages)
    }
}

impl State {
    /// The entry's page, marked most recently used.
    fn hand_out(&mut self, key: &Key) -> Option<Arc<StoredPages>> {
        self.clock += 1;
        let clock = self.clock;
        let entry = self.entries.get_mut(key)?;
        entry.used = clock;
        Some(Arc::clone(&entry.pages))
    }

    /// A free block of `class`, evicting the least recently used entries of
    /// that class nothing holds when the slab has none.
    fn take(&mut self, class: usize, evicted: &AtomicU64) -> Option<Block> {
        if let Some(block) = self.slab.take(class) {
            return Some(block);
        }
        let mut idle: Vec<(u64, Key)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.block.class == class && !e.held())
            .map(|(k, e)| (e.used, *k))
            .collect();
        idle.sort_unstable_by_key(|&(used, _)| used);
        for (_, key) in idle.into_iter().take(EVICT_BATCH) {
            if let Some(entry) = self.entries.remove(&key) {
                self.held_bytes -= class_bytes(entry.block.class);
                self.slab.give(entry.block);
                evicted.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.slab.take(class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u64) -> RecordIdentity {
        RecordIdentity {
            origin: 7,
            oseq: n,
            gen: 1,
            epoch: 0,
            hlc: n,
        }
    }

    fn cache(bytes: usize) -> IndexPageCache {
        IndexPageCache::new(vec![HostRegion::pageable(bytes).unwrap()])
    }

    /// A page read in is found again by its record, with its payload's bytes
    /// exactly, and a different record of the same stream is not it.
    #[test]
    fn a_page_read_in_is_found_by_its_record() {
        let c = cache(64 * 1024);
        let page = c
            .read_into(StreamId(1), id(5), 8, |dest| {
                dest.copy_from_slice(b"HDRpage!");
                Some(3..8)
            })
            .unwrap();
        assert_eq!(page.bytes(), b"page!");
        let again = c.get(StreamId(1), id(5)).unwrap();
        assert!(Arc::ptr_eq(&page, &again));
        assert!(c.get(StreamId(1), id(6)).is_none());
        assert!(c.get(StreamId(2), id(5)).is_none());
        let s = c.stats();
        assert_eq!((s.hits, s.misses, s.entries), (1, 1, 1));
        assert_eq!(s.held_bytes, 4096);
    }

    /// A read that fails, or names a range outside the record, caches
    /// nothing and gives its block back.
    #[test]
    fn a_failed_read_caches_nothing() {
        let c = cache(4096);
        assert!(c.read_into(StreamId(1), id(1), 16, |_| None).is_none());
        assert!(c
            .read_into(StreamId(1), id(1), 16, |_| Some(4..17))
            .is_none());
        assert_eq!(c.stats().entries, 0);
        // The one block came back each time.
        assert!(c.insert(StreamId(1), id(1), b"bytes").is_some());
    }

    /// A record with no identity cannot be told from another, so it is never
    /// cached; neither is one larger than the largest block.
    #[test]
    fn records_the_cache_cannot_hold_are_not_cached() {
        let c = cache(4096);
        assert!(c.insert(StreamId(1), RecordIdentity::NONE, b"x").is_none());
        let big = vec![0u8; (8 << 20) + 1];
        assert!(c.insert(StreamId(1), id(1), &big).is_none());
        assert_eq!(c.stats().uncached, 2);
    }

    /// Full, the cache evicts the least recently used page nothing holds, and
    /// never one a holder still has.
    #[test]
    fn eviction_takes_the_oldest_unheld_page_and_never_a_held_one() {
        let c = cache(3 * 4096);
        let held = c.insert(StreamId(1), id(1), b"held").unwrap();
        drop(c.insert(StreamId(2), id(1), b"old").unwrap());
        drop(c.insert(StreamId(3), id(1), b"new").unwrap());
        // Stream 2's page is the older unheld one; using it makes 3's older.
        drop(c.get(StreamId(2), id(1)).unwrap());
        // The slab is full: a fourth page evicts every unheld page of its
        // class, oldest first, up to the batch — here 3 and 2 — and never 1.
        let fourth = c.insert(StreamId(4), id(1), b"fourth").unwrap();
        assert_eq!(fourth.bytes(), b"fourth");
        assert_eq!(held.bytes(), b"held", "a held page's bytes are untouched");
        assert!(c.get(StreamId(1), id(1)).is_some());
        assert!(c.get(StreamId(3), id(1)).is_none());
        assert_eq!(c.stats().evicted, 2);
    }

    /// A value copy of a page holds its bytes as the page itself does.
    #[test]
    fn a_copy_of_a_page_holds_it_against_eviction() {
        let c = cache(4096);
        let page = c.insert(StreamId(1), id(1), b"kept").unwrap();
        let copy: StoredPages = (*page).clone();
        drop(page);
        assert!(c.insert(StreamId(2), id(1), b"other").is_none(), "held");
        assert_eq!(copy.bytes(), b"kept");
        drop(copy);
        assert!(c.insert(StreamId(2), id(1), b"other").is_some());
    }

    /// When two readers fill the same page, the second gives its block back
    /// and answers with the first's entry.
    #[test]
    fn a_page_published_twice_is_held_once() {
        let c = cache(2 * 4096);
        let (first, base) = c.reserve(id(1), 4).unwrap();
        let (second, base2) = c.reserve(id(1), 4).unwrap();
        // SAFETY: each block is reserved for this test alone.
        unsafe {
            std::slice::from_raw_parts_mut(base, 4).copy_from_slice(b"aaaa");
            std::slice::from_raw_parts_mut(base2, 4).copy_from_slice(b"bbbb");
        }
        let a = c
            .publish(StreamId(1), id(1), first, base, Some(0..4))
            .unwrap();
        let b = c
            .publish(StreamId(1), id(1), second, base2, Some(0..4))
            .unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(b.bytes(), b"aaaa");
        let s = c.stats();
        assert_eq!((s.entries, s.held_bytes), (1, 4096));
        // The second block is free again: another page fits beside the first.
        assert!(c.insert(StreamId(2), id(1), b"more").is_some());
    }

    /// Readers on several threads at once each get their own page back,
    /// whole, and the cache ends holding every one of them.
    #[test]
    fn concurrent_readers_each_get_their_own_page() {
        let c = Arc::new(cache(64 * 4096));
        let handles: Vec<_> = (0..8u64)
            .map(|t| {
                let c = Arc::clone(&c);
                std::thread::spawn(move || {
                    for i in 0..8u64 {
                        let want = [(t * 8 + i) as u8; 100];
                        let page = c.insert(StreamId(t * 8 + i), id(1), &want).unwrap();
                        assert_eq!(page.bytes(), &want[..]);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(c.stats().entries, 64);
        for s in 0..64u64 {
            let page = c.get(StreamId(s), id(1)).unwrap();
            assert_eq!(page.bytes(), &[s as u8; 100][..]);
        }
    }
}
