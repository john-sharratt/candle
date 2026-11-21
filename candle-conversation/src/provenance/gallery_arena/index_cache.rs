//! Reuses a scan's index while its turns are still resident where it was built,
//! so a reprojection does not rebuild the index of a large corpus.
//!
//! An index is valid only while every turn it names still carries the run id it
//! was built against. That is checked per turn rather than by a device-wide
//! generation, so another conversation's uploads leave it alone. An index that
//! scans more fold groups than a reader weighs serves that reader exactly, since
//! groups are scanned independently. A stale index is removed at once, because a
//! moved turn keeps its new run id and can never be valid again. An index that
//! goes unused for a long time is dropped, and the least recently used one goes
//! when the cache passes its byte budget.

use super::*;

impl GalleryArena {
    /// Reuse the cached index if it matches `fingerprint`, scans every fold
    /// group each segment weighs (`weighed`, one set per segment), AND every
    /// turn it references is still resident on the run it was built against —
    /// in which case this **leases** it, holding its turns as one for the scan.
    /// Otherwise says why it missed — no index cached, one scanning fewer
    /// groups, or a turn it holds evicted, moved or re-uploaded — and the caller
    /// rebuilds.
    ///
    /// Takes no lock unless a resident turn changed since the index was last
    /// checked (`lease`).
    ///
    /// An index scanning more groups than a segment weighs serves it exactly:
    /// groups are scanned independently, and the rows of the groups it does
    /// not weigh are read as no vote ([`widen_rows`](token_votes::widen_rows)).
    /// So two readers of one segment set weighing different groups share one
    /// index — and its rows — over the union.
    ///
    /// A turn whose run id still matches has not been re-uploaded, evicted or
    /// moved since the build, so every address the index holds for it is still
    /// its page. That is checked per referenced turn — O(turns), not
    /// O(tokens) — so another conversation's uploads leave this index valid.
    pub(super) fn reuse_index(
        &self,
        fingerprint: u64,
        weighed: &[GroupSet],
    ) -> StdResult<IndexLease, IndexMiss> {
        let idx = self
            .indexes
            .load()
            .iter()
            .find(|(fp, idx)| *fp == fingerprint && !idx.hold.retire_seen())
            .map(|(_, idx)| idx.clone())
            .ok_or(IndexMiss::Absent)?;
        idx.used.store(self.since_born_ms(), Ordering::Relaxed);
        if !idx.serves(weighed) {
            return Err(IndexMiss::Narrower {
                covering: idx.covering(weighed),
            });
        }
        let lease = IndexLease::take(idx.clone()).ok_or(IndexMiss::Absent)?;
        if !idx.hold.current(&self.changes) {
            if idx.hold.stale() {
                self.check_index(fingerprint, &idx)?;
            } else {
                // The count moved for turns this index does not name
                // (`lease`): it is current without the walk.
                idx.hold.trusted(self.changes.load(Ordering::SeqCst));
            }
        }
        self.touch_index(&idx);
        Ok(lease)
    }

    /// Check that every turn `idx` references is still resident on the run it
    /// was built against, under the residency lock, leased so that nothing
    /// changes behind the check. A stale index leaves the cache: it can never
    /// be valid again — a moved turn keeps its new run id — so its device
    /// arrays go back rather than wait for something to overwrite the key.
    pub(super) fn check_index(
        &self,
        fingerprint: u64,
        idx: &Arc<scan::PagedIndex>,
    ) -> StdResult<(), IndexMiss> {
        let res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        let changes = self.changes.load(Ordering::SeqCst);
        let resident = |sid: &StreamId, run: u64| res.get(sid).is_some_and(|rt| rt.run_id == run);
        // The turns changes named since the last check, or every turn
        // (`lease`): one no change named is as it was checked.
        let marked = idx.hold.marked();
        let (looked, whole) = match &marked {
            Some(sids) => (sids.len(), false),
            None => (idx.pinned_sids.len(), true),
        };
        let current = match marked {
            Some(sids) => sids
                .iter()
                .all(|sid| idx.members.get(sid).is_none_or(|&run| resident(sid, run))),
            None => idx
                .pinned_sids
                .iter()
                .zip(&idx.run_ids)
                .all(|(sid, &run)| resident(sid, run)),
        };
        scan_split::add(|s| {
            s.checked_turns += looked as u64;
            s.checks_whole += u64::from(whole);
        });
        // Cleared under the lock every change takes, so no change's mark is
        // lost between the check and the clearing.
        if current {
            idx.hold.checked(changes);
        }
        drop(res);
        if !current {
            let mut cache = self.index_cache.lock().unwrap_or_else(|e| e.into_inner());
            if cache.get(&fingerprint).is_some_and(|c| Arc::ptr_eq(c, idx)) {
                cache.remove(&fingerprint);
            }
            self.publish_indexes(&cache);
            return Err(IndexMiss::Stale);
        }
        Ok(())
    }

    /// Record that a scan used `idx`'s turns, at most every
    /// [`TOUCH_EVERY_MS`], and never waiting for the residency lock: a busy
    /// lock leaves it to the next scan.
    pub(super) fn touch_index(&self, idx: &scan::PagedIndex) {
        let now_ms = self.born.elapsed().as_millis() as u64;
        let was = idx.touched_ms.load(Ordering::Relaxed);
        if was != 0 && now_ms.saturating_sub(was) < TOUCH_EVERY_MS {
            return;
        }
        let Ok(mut res) = self.residency.try_lock() else {
            return;
        };
        let lru = self.lru_clock.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();
        for sid in &idx.pinned_sids {
            if let Some(rt) = res.get_mut(sid) {
                rt.lru = lru;
                rt.touch(now);
            }
        }
        idx.touched_ms.store(now_ms.max(1), Ordering::Relaxed);
    }

    /// Publish `cache` for scans to read, keeping an index that left it until
    /// a later publish finds it marked and unleased (`lease`).
    pub(super) fn publish_indexes(&self, cache: &HashMap<u64, Arc<scan::PagedIndex>>) {
        let mut next: Vec<(u64, Arc<scan::PagedIndex>)> =
            cache.iter().map(|(fp, idx)| (*fp, idx.clone())).collect();
        for (fp, idx) in self.indexes.load().iter() {
            if cache.values().any(|c| Arc::ptr_eq(c, idx)) {
                continue;
            }
            let marked = idx.hold.retire();
            if !marked || idx.hold.leased() {
                next.push((*fp, idx.clone()));
            }
        }
        self.indexes.store(Arc::new(next));
    }

    /// Cache `idx` under `fingerprint` for reuse, dropping the indices no scan
    /// has used for [`INDEX_IDLE_MS`] and, past [`Self::index_cache_bytes`],
    /// the least recently used ([`index_cache_evictions`]). An index larger
    /// than the cap on its own is not cached.
    pub(super) fn store_index(&self, fingerprint: u64, idx: Arc<scan::PagedIndex>) {
        let now_ms = self.since_born_ms();
        let bytes = idx.device_bytes();
        let mut cache = self.index_cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.remove(&fingerprint);
        if bytes > self.index_cache_bytes {
            self.publish_indexes(&cache);
            return;
        }
        let entries: Vec<(u64, u64, u64)> = cache
            .iter()
            .map(|(&fp, e)| (fp, e.used.load(Ordering::Relaxed), e.device_bytes()))
            .collect();
        for fp in index_cache_evictions(
            &entries,
            bytes,
            now_ms,
            INDEX_IDLE_MS,
            self.index_cache_bytes,
        ) {
            cache.remove(&fp);
        }
        idx.used.store(now_ms, Ordering::Relaxed);
        cache.insert(fingerprint, idx);
        self.publish_indexes(&cache);
    }

    /// Milliseconds since the arena began — the clock an index's use is kept on.
    pub(super) fn since_born_ms(&self) -> u64 {
        self.born.elapsed().as_millis() as u64
    }
}
