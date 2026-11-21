//! Makes a turn resident on the card for a read, uploading its pages when they are
//! absent or do not hold what the read needs.
//!
//! A hit is a read whose pages already serve it. A miss frees the stale pages and
//! uploads the turn again, holding the union of what it held and what is now asked.
//! Addresses are resolved while the residency lock is still held, so an eviction
//! cannot free a slot the caller is about to use. The keep eviction runs before an
//! upload rather than before a hit, because walking every resident turn on each
//! ensure cost seconds per index build. A replaced run is counted as a change,
//! since a cached index over it is now stale.

use super::*;

impl GalleryArena {
    /// Give up the unpinned turns among `sids` — turns no scan can read any
    /// more (`docs/gallery_residency.md` §3). A pinned one stays: an in-flight
    /// scan reads its pages. Returns the turns and page bytes freed.
    pub fn drop_turns(&self, sids: &[StreamId]) -> (usize, u64) {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        let mut freed = (0usize, 0u64);
        self.note_change();
        for sid in sids {
            if res
                .get(sid)
                .is_some_and(|rt| rt.pinned == 0 && !self.leased(sid))
            {
                if let Some(rt) = res.remove(sid) {
                    freed.0 += 1;
                    freed.1 += self.run_bytes(&rt);
                }
            }
        }
        if freed.0 > 0 {
            self.residency_gen.fetch_add(1, Ordering::Relaxed);
        }
        drop(res);
        self.dedup.forget(sids);
        freed
    }

    /// Claim and upload a turn's pages, returning the run.
    ///
    /// One claim for the whole turn, so the arena window opens at most once however
    /// many new arenas it needs. A refusal means the span itself is full, and it
    /// **is** an error: routine growth is bounded well before this by the gallery's
    /// own keep rule (`evict_past_keep_locked`, run once before a scan's index build,
    /// a preload, and a single turn's miss), so
    /// reaching a refusal is the KV side having no region to spare for a gallery
    /// that is already inside its budget. The pages hold `groups` of the
    /// tokens `mask` keeps only.
    pub(super) fn alloc_and_upload(
        &self,
        sigs: TurnSigs<'_>,
        groups: GroupSet,
        mask: &KeepMask,
    ) -> Result<PageRun> {
        let n_pages = pages_for(mask.kept());
        if n_pages == 0 {
            return Ok(PageRun { slots: Vec::new() });
        }
        let slots = claim_arena_slots(
            &self.device,
            SlotTenant::Gallery,
            self.page_bytes_for(groups) as usize,
            n_pages,
        )?;
        let stride_words = slots[0].stride() / std::mem::size_of::<u64>();
        let host_pages =
            sigs.with_decoded(|sigs| transpose_to_pages(sigs, mask, self.gw, groups, stride_words));
        page_io::write_pages(&self.device, &slots, &host_pages)?;
        Ok(PageRun { slots })
    }

    /// Ensure a turn is resident under `fingerprint` (holding the residency
    /// guard), returning its pages' device **addresses** (page order) and the
    /// run id they belong to ([`ResidentTurn::run_id`]). A matching
    /// fingerprint is a hit; a mismatch (or absence) frees any stale pages and
    /// re-uploads. When `pin`, the turn's pin count is bumped **atomically with
    /// residency** so a concurrent [`evict_past_keep`](Self::evict_past_keep) can never free it
    /// between here and the launch that reads its pages. The addresses are
    /// resolved while the residency lock is still held, so the gids cannot be
    /// recycled out from under them.
    ///
    /// **Makes no room.** The keep eviction walks every resident turn, so it is
    /// the caller's to run once before what it admits: a single turn's
    /// [`ensure_resident`](Self::ensure_resident) before its miss, a scan's
    /// index build and a preload once before their every turn. Its cost per
    /// ensure is what this avoids: over a 27,217-turn `code_reading` index it
    /// came to 20.6 s a build (BlackWorld, 2026-10-26), stalling every
    /// reprojection.
    ///
    /// `holder` is who this read is for; the turn's holder widens by it
    /// ([`Holder::widen`]), on a hit and across a re-upload alike. `groups` are
    /// the fold groups the read weighs: a turn whose pages hold them all is a
    /// hit, one missing any is uploaded again holding the union of what it held
    /// and what this read weighs (`docs/gallery_residency.md` §4). `keep` is
    /// which of its tokens the read asks the card to hold: a turn held under a
    /// keep covering it is a hit, one held under a narrower keep is uploaded
    /// again under the union (`docs/gallery_dedup.md` §3).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ensure_locked(
        &self,
        res: &mut HashMap<StreamId, ResidentTurn>,
        sid: StreamId,
        sigs: TurnSigs<'_>,
        fingerprint: u64,
        pin: bool,
        holder: Holder,
        groups: GroupSet,
        keep: Keep,
    ) -> Result<Ensured> {
        debug_assert!(
            sigs.first_width().unwrap_or(self.wpt) == self.wpt,
            "gallery sig width {} != arena wpt {} — folded-geometry mismatch (wrong \
             head_dim?); every token would be dropped and the scan silently zeroed",
            sigs.first_width().unwrap_or(0),
            self.wpt
        );
        let lru = self.lru_clock.fetch_add(1, Ordering::Relaxed);
        // Addresses are resolved while `res` is still held: evicting or replacing
        // a run needs the residency lock, so its slots can't be freed in this
        // window.
        match res.get_mut(&sid) {
            Some(rt) if rt.serves(fingerprint, groups, keep) => Ok(rt.hit(lru, holder, pin)),
            _ => self
                .replace_locked(res, sid, sigs, fingerprint, pin, lru, holder, groups, keep)
                .map(|(addrs, run_id, groups, mask)| Ensured {
                    addrs,
                    run_id,
                    groups,
                    mask,
                    uploaded: true,
                }),
        }
    }

    /// The miss / stale-fingerprint / narrow-pages branch of
    /// [`ensure_locked`](Self::ensure_locked): free the old run and upload fresh
    /// pages, returning the new pages' addresses and the groups they hold. A
    /// re-uploaded turn keeps the wider of its old holder and `holder`, the
    /// union of the groups its old pages held and `groups`, and of its old keep
    /// and `keep` — the old one only while its content is unchanged: a re-seal
    /// is a new turn to its readers.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn replace_locked(
        &self,
        res: &mut HashMap<StreamId, ResidentTurn>,
        sid: StreamId,
        sigs: TurnSigs<'_>,
        fingerprint: u64,
        pin: bool,
        lru: u64,
        holder: Holder,
        groups: GroupSet,
        keep: Keep,
    ) -> Result<(Vec<u64>, u64, GroupSet, Arc<KeepMask>)> {
        let book = self.dedup.book();
        let held = Held::widened(
            res.get(&sid),
            fingerprint,
            holder,
            groups,
            keep,
            book.is_some(),
        );
        let Held {
            holder,
            groups,
            keep,
        } = held;
        self.drop_replaced(res, sid);
        let mask = self
            .dedup
            .mask(sid, fingerprint, sigs, keep, book.as_deref());
        let run = self.alloc_and_upload(sigs, groups, &mask)?;
        let addrs = run.addrs();
        let run_id = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        res.insert(
            sid,
            ResidentTurn {
                fingerprint,
                run,
                run_id,
                mask: Arc::clone(&mask),
                keep,
                lru,
                used: Instant::now(),
                interval: None,
                pinned: u32::from(pin),
                holder,
                groups,
            },
        );
        Ok((addrs, run_id, groups, mask))
    }

    /// Take `sid`'s resident entry out before its fresh upload, freeing its
    /// pages, and count the change.
    pub(super) fn drop_replaced(&self, res: &mut HashMap<StreamId, ResidentTurn>, sid: StreamId) {
        // A re-seal changes a turn an index may name: asked, so each such index
        // is marked stale, in every build.
        let leased = res.contains_key(&sid) && {
            self.note_change();
            self.leased(&sid)
        };
        if let Some(old) = res.remove(&sid) {
            // The scan thread always unpins before the next ensure on that thread,
            // so a replaced entry is never pinned. If this ever fires, an in-flight
            // scan's pages are about to be recycled — the arena would need a
            // multi-version turn to stay safe under concurrent re-seal + scan.
            debug_assert!(
                old.pinned == 0 && !leased,
                "re-seal of a turn pinned by an active scan — pages would be freed \
                 under an in-flight kernel"
            );
            drop(old); // frees the old run's pages before the fresh upload
        }
        // Counted NOW — the pages are freed even if the upload after fails. The
        // old run id left with the removed entry, so any cached index over it is
        // already invalid.
        self.residency_gen.fetch_add(1, Ordering::Relaxed);
    }

    /// Ensure a turn is resident under `fingerprint`, returning its pages' device
    /// addresses (page order). A matching fingerprint is a hit (no upload); a
    /// mismatch (or absence) frees any stale pages and re-uploads exactly this
    /// turn — the delta a seal produces. `sigs` must be the turn's full window;
    /// a stored window is decoded only on the miss.
    /// Does NOT pin — used off the scan hot path (and by tests).
    ///
    /// A miss first gives up the turns past their keep time: this is where
    /// gallery growth is bounded outside a scan, since `alloc_and_upload` adds
    /// a slab whenever the page pool is empty and nothing else ever shrinks the
    /// arena. A hit admits nothing, so it makes no room. Held for `holder`,
    /// holding at least `groups` of the tokens `keep` keeps, under a codebook
    /// already trained — a keep with none yet holds every token.
    pub fn ensure_resident<'a>(
        &self,
        sid: StreamId,
        sigs: impl Into<TurnSigs<'a>>,
        fingerprint: u64,
        holder: Holder,
        groups: GroupSet,
        keep: Keep,
    ) -> Result<Vec<u64>> {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        if res
            .get(&sid)
            .is_none_or(|rt| !rt.serves(fingerprint, groups, keep))
        {
            self.evict_past_keep_locked(&mut res);
        }
        self.ensure_locked(
            &mut res,
            sid,
            sigs.into(),
            fingerprint,
            false,
            holder,
            groups,
            keep,
        )
        .map(|ensured| ensured.addrs)
    }

    /// [`ensure_resident`](Self::ensure_resident) for many turns at once — a
    /// gallery preload, of shared turns, each holding the groups its readers
    /// weigh of the tokens they keep — giving up the turns past their keep
    /// time once, before the first, rather than before each miss: the pass
    /// walks every resident turn. A turn kept by code trains the codebook from
    /// the turns kept by code, when none is trained yet, and their masks are
    /// computed together before the first upload. One result per turn, in
    /// order.
    pub fn ensure_all_resident<'a>(
        &self,
        turns: impl IntoIterator<Item = (StreamId, TurnSigs<'a>, u64, GroupSet, Keep)>,
    ) -> Vec<Result<()>> {
        let turns: Vec<(StreamId, TurnSigs<'a>, u64, GroupSet, Keep)> = turns.into_iter().collect();
        self.prepare_keeps(
            turns
                .iter()
                .map(|&(sid, sigs, fp, _, keep)| (sid, sigs, fp, keep)),
        );
        self.evict_past_keep();
        turns
            .into_iter()
            .map(|(sid, sigs, fingerprint, groups, keep)| {
                let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
                self.ensure_locked(
                    &mut res,
                    sid,
                    sigs,
                    fingerprint,
                    false,
                    Holder::Shared,
                    groups,
                    keep,
                )
                .map(|_| ())
            })
            .collect()
    }

    /// Ready the overlay for `turns` about to be read, each under the keep its
    /// reader asks for: train the codebook from the ones kept by code if none
    /// is trained, and compute the masks they lack in parallel — so the
    /// uploads, one turn at a time under the residency lock, and the index
    /// build after them find theirs computed. A turn resident under a wider
    /// keep still needs its reader's mask: that is which of its tokens the
    /// reader scans.
    pub(in super::super) fn prepare_keeps<'a>(
        &self,
        turns: impl IntoIterator<Item = (StreamId, TurnSigs<'a>, u64, Keep)>,
    ) {
        let mut coded: Vec<(StreamId, TurnSigs<'a>, u64, Keep)> =
            turns.into_iter().filter(|t| t.3 != Keep::All).collect();
        if coded.is_empty() {
            return;
        }
        coded.sort_by_key(|t| (t.0 .0, t.3));
        coded.dedup_by_key(|t| (t.0, t.3));
        let mut corpus: Vec<(StreamId, TurnSigs<'a>)> = coded.iter().map(|t| (t.0, t.1)).collect();
        corpus.dedup_by_key(|t| t.0);
        let book = self
            .dedup
            .book_from(&corpus.into_iter().map(|t| t.1).collect::<Vec<_>>());
        self.dedup.prepare(&coded, &book);
    }

    /// The mask a reader keeping `keep` scans `sigs` — the turn `sid` under
    /// content `fingerprint` — by: every token without a codebook.
    pub(in super::super) fn reader_mask(
        &self,
        sid: StreamId,
        fingerprint: u64,
        sigs: TurnSigs<'_>,
        keep: Keep,
    ) -> Arc<KeepMask> {
        let book = self.dedup.book();
        self.dedup
            .mask(sid, fingerprint, sigs, keep, book.as_deref())
    }

    /// Release the pins [`scan_ensure_all`](Self::scan_ensure_all) took on
    /// `sids`, in one pass under one lock.
    pub(in super::super) fn unpin_all(&self, sids: &[StreamId]) {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        for sid in sids {
            if let Some(rt) = res.get_mut(sid) {
                rt.pinned = rt.pinned.saturating_sub(1);
            }
        }
    }

    /// A resident turn an index could name is about to change. Comes before
    /// the change and before [`Self::leased`] (`lease`).
    pub(super) fn note_change(&self) {
        self.changes.fetch_add(1, Ordering::SeqCst);
    }

    /// Whether a scan holds `sid` through a leased index. Asked after
    /// [`Self::note_change`], of the turn about to go, under the residency
    /// lock. Every index naming `sid` is marked stale first, before its lease
    /// is read (`lease`): a later scan of it checks its turns, and one that
    /// names none of the changed turns is trusted without the walk.
    pub(super) fn leased(&self, sid: &StreamId) -> bool {
        let mut leased = false;
        for (_, idx) in self.indexes.load().iter() {
            if idx.members.contains_key(sid) {
                idx.hold.mark_stale(*sid);
                leased |= idx.hold.leased();
            }
        }
        leased
    }
}
