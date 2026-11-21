//! Gives up resident gallery turns, least recently used or past their keep time,
//! so the arena stays within its budget.
//!
//! Dropping a turn drops its pages, which returns their slots to the arena. Pinned
//! turns never go, so an in-flight scan keeps its pages. The keep time is set from
//! the weight side's measured marginal value, so a turn that is not scanned often
//! enough to earn its ground is given up first. A guest's drain may give up every
//! unpinned turn, because nothing scans while it runs and each turn rebuilds from
//! the substrate on its next scan.

use super::*;

impl GalleryArena {
    /// Give up every unpinned turn, working set and all, and answer the
    /// reservation bytes that went back to the span. **Only for a guest's
    /// drain**: nothing scans while it runs, and every turn rebuilds from the
    /// substrate on its next scan — the 1.6–2.0 s an index rebuild costs, once,
    /// against a draw that otherwise cannot run.
    pub fn evict_all_for_guest(&self) -> u64 {
        let before = self.resident_bytes();
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        self.note_change();
        let gone: Vec<StreamId> = res
            .iter()
            .filter(|(sid, rt)| rt.pinned == 0 && !self.leased(sid))
            .map(|(sid, _)| *sid)
            .collect();
        for sid in &gone {
            res.remove(sid); // drops the run → frees pages
            self.residency_gen.fetch_add(1, Ordering::Relaxed);
        }
        drop(res);
        let after = self.resident_bytes();
        let freed = before.saturating_sub(after);
        tracing::info!(
            target: "candle_conversation::provenance",
            turns = gone.len(),
            resident_mib_before = before >> 20,
            resident_mib_after = after >> 20,
            "gallery emptied for a guest"
        );
        freed
    }

    /// Evict least-recently-used unpinned turns until at least `want` bytes are
    /// freed, regardless of recent use. Returns the bytes freed. The tests use it
    /// to empty the arena and prove a rebuild reproduces the scan; nothing in the
    /// engine sheds a working set this way.
    #[cfg(test)]
    pub fn evict_lru(&self, want: u64) -> u64 {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        self.evict_lru_locked(&mut res, want)
    }

    /// [`evict_lru`](Self::evict_lru) under a residency guard the caller holds.
    #[cfg(test)]
    pub(super) fn evict_lru_locked(
        &self,
        res: &mut HashMap<StreamId, ResidentTurn>,
        want: u64,
    ) -> u64 {
        // Order candidates by LRU ascending (oldest first), skipping pins.
        self.note_change();
        let mut cands: Vec<(u64, StreamId, u64)> = res
            .iter()
            .filter(|(sid, rt)| rt.pinned == 0 && !self.leased(sid))
            .map(|(sid, rt)| (rt.lru, *sid, self.run_bytes(rt)))
            .collect();
        cands.sort_by_key(|(lru, _, _)| *lru);
        let mut freed = 0u64;
        for (_, sid, bytes) in cands {
            if freed >= want {
                break;
            }
            res.remove(&sid); // drops the run → frees pages
            self.residency_gen.fetch_add(1, Ordering::Relaxed);
            freed += bytes;
        }
        freed
    }

    /// How often a turn must be scanned to keep its ground — `T*`, set by the
    /// scheduler from the weight side's measured marginal value
    /// ([`eviction::keep_within`]); `None` keeps everything.
    pub fn set_keep_within(&self, keep: Option<Duration>) {
        let ns = keep.map_or(u64::MAX, |d| d.as_nanos().min(u64::MAX as u128 - 1) as u64);
        self.keep_within_ns.store(ns, Ordering::Relaxed);
    }

    /// How often a turn must be scanned to keep its ground; `None` keeps
    /// everything ([`Self::set_keep_within`]).
    pub fn keep_within(&self) -> Option<Duration> {
        match self.keep_within_ns.load(Ordering::Relaxed) {
            u64::MAX => None,
            ns => Some(Duration::from_nanos(ns)),
        }
    }

    /// What a scan waits per byte of its working set when that set is not all
    /// resident — the uploads of the turns it finds evicted and the index over
    /// them, in microseconds, timed on the scans that rebuilt — and so what a
    /// turn's residency spares per byte at each scan. Zero until one has.
    pub fn upload_us_per_byte(&self) -> f64 {
        f64::from_bits(self.upload_us_per_byte.load(Ordering::Relaxed))
    }

    /// An upload of `bytes` took `elapsed`: it joins the per-byte average.
    pub(super) fn note_upload(&self, bytes: u64, elapsed: Duration) {
        if bytes == 0 {
            return;
        }
        let sample = elapsed.as_secs_f64() * 1e6 / bytes as f64;
        let was = self.upload_us_per_byte();
        let next = if was > 0.0 {
            was + UPLOAD_WEIGHT * (sample - was)
        } else {
            sample
        };
        self.upload_us_per_byte
            .store(next.to_bits(), Ordering::Relaxed);
    }

    /// Give up every turn not worth its ground ([`eviction::past_keep`]) — the
    /// governor's relief rung, a scan's index build, and a preload. Returns
    /// the bytes freed. Pinned turns never go, so an in-flight scan's pages
    /// stay.
    pub fn evict_past_keep(&self) -> u64 {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        self.evict_past_keep_locked(&mut res)
    }

    pub(super) fn evict_past_keep_locked(&self, res: &mut HashMap<StreamId, ResidentTurn>) -> u64 {
        let keep = self.keep_within();
        if keep.is_none() {
            return 0;
        }
        let now = Instant::now();
        let (sids, turns): (Vec<StreamId>, Vec<Resident>) = res
            .iter()
            .map(|(sid, rt)| {
                (
                    *sid,
                    Resident {
                        lru: rt.lru,
                        idle: now.saturating_duration_since(rt.used),
                        interval: rt.interval,
                        bytes: self.run_bytes(rt),
                        pinned: rt.pinned != 0,
                    },
                )
            })
            .unzip();
        let mut freed = 0u64;
        let gone = past_keep(&turns, keep);
        if !gone.is_empty() {
            self.note_change();
        }
        let gone: Vec<usize> = gone
            .into_iter()
            .filter(|&i| !self.leased(&sids[i]))
            .collect();
        for &i in &gone {
            res.remove(&sids[i]); // drops the run → frees pages
            self.residency_gen.fetch_add(1, Ordering::Relaxed);
            freed += turns[i].bytes;
        }
        if !gone.is_empty() {
            tracing::info!(
                target: "candle_conversation::provenance",
                turns = gone.len(),
                freed_mib = freed >> 20,
                keep_ms = keep.map_or(0, |k| k.as_millis() as u64),
                upload_ns_per_kib = (self.upload_us_per_byte() * 1024.0 * 1000.0) as u64,
                kept = res.len(),
                "gallery gave up the turns past its keep time"
            );
        }
        freed
    }
}
