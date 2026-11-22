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

/// The fewest page bytes an upload must hold to be timed: under it, a clock
/// tick is a large share of the work, and a fast clock read of a small
/// upload as nothing would set the cost at zero for good.
pub(super) const MIN_UPLOAD_SAMPLE: u64 = 1 << 20;

/// The per-byte upload cost after `was` (µs, zero for none yet) meets an
/// upload of `bytes` that took `host`: **the least seen**, of uploads of
/// [`MIN_UPLOAD_SAMPLE`] or more.
///
/// The least, because nothing makes an upload faster than its work and
/// everything else makes it slower: a residency lock waited on, a busy CPU, a
/// stream queued behind a wave. Each of those is the moment's load, not what
/// evicting a turn costs, and an average of them priced uploads at 25× their
/// work on WhiteWorld's busy card (8 → 200 µs/KiB in a minute, 2026-11-22),
/// lifting the keep time with it. It is timed on the upload alone — not on
/// the index rebuild around it, which divided by the whole scanned set read a
/// rebuild that uploaded nothing as a near-free upload.
pub(super) fn fold_upload(was: f64, bytes: u64, host: Duration) -> f64 {
    if bytes < MIN_UPLOAD_SAMPLE || host.is_zero() {
        return was;
    }
    let sample = host.as_secs_f64() * 1e6 / bytes as f64;
    if was > 0.0 {
        was.min(sample)
    } else {
        sample
    }
}

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

    /// The host's part of re-uploading a turn, per page byte, in microseconds:
    /// decoding its signatures, transposing them into pages and enqueueing the
    /// copies — the least any upload of [`MIN_UPLOAD_SAMPLE`] or more has
    /// taken ([`fold_upload`]). The copy over the link is the caller's to add
    /// at the link's rate. Zero until an upload is timed.
    pub fn upload_us_per_byte(&self) -> f64 {
        f64::from_bits(self.upload_us_per_byte.load(Ordering::Relaxed))
    }

    /// An upload of `bytes` page bytes took `host` of host work: it joins the
    /// least per-byte cost seen ([`fold_upload`]).
    pub(super) fn note_upload(&self, bytes: u64, host: Duration) {
        let was = self.upload_us_per_byte();
        let next = fold_upload(was, bytes, host);
        if next != was {
            self.upload_us_per_byte
                .store(next.to_bits(), Ordering::Relaxed);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    /// **The cost is the least seen**: a slower upload — the card busy, the
    /// lock waited on — never raises it, a faster one lowers it.
    #[test]
    fn the_upload_cost_keeps_the_fastest_upload() {
        let first = fold_upload(0.0, 4 * MIB, Duration::from_micros(4_194_304));
        assert_eq!(first, 1.0, "4 MiB in 4.19 s: 1 µs a byte");
        let slower = fold_upload(first, 4 * MIB, Duration::from_secs(100));
        assert_eq!(slower, 1.0, "a contended upload changes nothing");
        let faster = fold_upload(slower, 2 * MIB, Duration::from_micros(524_288));
        assert_eq!(faster, 0.25);
    }

    /// An upload under a MiB, or timed at nothing, is no sample: a clock that
    /// read a small upload as instant would price every upload at zero for
    /// good.
    #[test]
    fn a_small_or_untimed_upload_is_no_sample() {
        assert_eq!(fold_upload(0.0, MIB - 1, Duration::from_nanos(1)), 0.0);
        assert_eq!(fold_upload(0.5, 64 << 10, Duration::from_nanos(1)), 0.5);
        assert_eq!(fold_upload(0.5, 8 * MIB, Duration::ZERO), 0.5);
        assert_eq!(
            fold_upload(0.0, MIB, Duration::from_micros(1_048_576)),
            1.0,
            "a MiB exactly is a sample"
        );
    }
}
