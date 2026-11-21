//! Builds the paged index that a scan reads: the resident turns it needs, their
//! page addresses, and the entries each segment scans.
//!
//! Each turn is uploaded at most once per scan and pinned until the launch that
//! reads it. A turn gets one page address per launch group, and a group the turn
//! does not hold points at the arena's zero page. Each segment's index covers
//! every group it weighs, and every group an earlier reader of the same segment
//! weighed, so one index serves both. Windows are emitted in case order, which
//! the BMMA backend's dense ranking depends on. Pins are released on error, so a
//! failed build holds nothing.

use super::*;

impl GalleryArena {
    /// Ensure every referenced turn is resident and assemble the paged index.
    /// A turn is uploaded at most once per scan (deduped by `sid`), holding the
    /// groups every segment reading it weighs; its pages get one `page_ptr`
    /// entry per launch group, in first-encounter order, and every window into
    /// that turn resolves against the recorded base page. A launch group the
    /// turn does not hold — weighed only by another segment — points at the
    /// arena's zero page.
    ///
    /// Each segment is built for `seg_groups` (one set per segment): at least
    /// the groups it weighs, and the groups an earlier reader of the same
    /// segment set weighed, so the index serves both.
    pub(super) fn build_index(
        &self,
        segments: &[PagedSegment],
        seg_groups: &[GroupSet],
    ) -> Result<PagedIndex> {
        let wpt = self.wpt();
        let launch = seg_groups
            .iter()
            .fold(GroupSet::default(), |g, s| g.union(*s));
        let gw_bytes = (PAGE_TOKENS * self.gw() * std::mem::size_of::<u64>()) as u64;
        // What each turn is read for, over every segment that reads it: the
        // widest holder, the union of the groups, and the union of the keeps.
        let mut wants: HashMap<StreamId, (Holder, GroupSet, Keep)> = HashMap::new();
        // Every turn under every keep by code a segment reads it by.
        let mut reads: HashMap<(StreamId, Keep), (TurnSigs<'_>, u64)> = HashMap::new();
        for (seg, &groups) in segments.iter().zip(seg_groups) {
            for w in &seg.windows {
                wants
                    .entry(w.sid)
                    .and_modify(|(h, g, k)| {
                        *h = h.widen(seg.holder);
                        *g = g.union(groups);
                        *k = k.union(seg.keep);
                    })
                    .or_insert((seg.holder, groups, seg.keep));
                if seg.keep != Keep::All {
                    reads
                        .entry((w.sid, seg.keep))
                        .or_insert((w.turn, w.fingerprint));
                }
            }
        }
        // Train the codebook and compute the masks the uploads and the walk
        // below need, in parallel, before the first: they run a turn at a time.
        let t_keeps = Instant::now();
        self.prepare_keeps(
            reads
                .iter()
                .map(|(&(sid, keep), &(turn, fp))| (sid, turn, fp, keep)),
        );
        let keeps_time = t_keeps.elapsed();
        // Which of a turn's tokens a segment scans: its own keep's, each kept
        // token at its place among the tokens the turn's pages hold — those of
        // a keep covering it, so a reader scans what it asked for whoever
        // else holds the turn wider.
        let mut scanned: HashMap<(StreamId, Keep), Arc<KeepMask>> = HashMap::new();
        let mut zero: Option<u64> = None;
        let mut turn_base: HashMap<StreamId, (usize, Arc<KeepMask>)> = HashMap::new();
        let mut n_pages = 0usize;
        let mut page_ptr: Vec<u64> = Vec::new();
        let mut entries = IndexEntries::new();
        let mut seg_case = vec![0i32];
        let mut case_off = 0usize;
        let mut max_seg_cases = 1usize;

        // Give up the turns past their keep time before the build admits
        // anything — once per build, not per turn: the pass is O(resident
        // turns). What the build uploads is held by its own pins until the
        // launch, so nothing it admits is given up under it.
        self.evict_past_keep();
        // What the build cost, and why: the turns it had to upload against the
        // turns already resident.
        let t_build = Instant::now();
        // Every turn the walk below reads, admitted and pinned at once, in the
        // order the walk first meets them.
        let mut first_seen: HashMap<StreamId, usize> = HashMap::new();
        let mut wanted: Vec<Wanted<'_>> = Vec::new();
        for seg in segments {
            for w in seg.windows.iter().filter(|w| w.case < seg.n_cases) {
                if let Entry::Vacant(e) = first_seen.entry(w.sid) {
                    e.insert(wanted.len());
                    let (holder, groups, keep) = wants[&w.sid];
                    wanted.push(Wanted {
                        sid: w.sid,
                        sigs: w.turn,
                        fingerprint: w.fingerprint,
                        holder,
                        groups,
                        keep,
                    });
                }
            }
        }
        let mut admitted: Vec<Option<Ensured>> = self
            .scan_ensure_all(&wanted)?
            .into_iter()
            .map(Some)
            .collect();
        let ensure_time = t_build.elapsed();
        let uploaded = admitted
            .iter()
            .filter(|e| e.as_ref().is_some_and(|e| e.uploaded))
            .count();
        // Every turn admitted is pinned, and released on error so a failed
        // build never leaks pins.
        let all_pinned: Vec<StreamId> = wanted.iter().map(|w| w.sid).collect();
        let mut pinned_sids: Vec<StreamId> = Vec::with_capacity(wanted.len());
        let mut run_ids: Vec<u64> = Vec::with_capacity(wanted.len());
        for seg in segments {
            // Emit each segment's windows in CASE order, so gallery case ids are
            // non-decreasing over the scan order. The scan math is order-independent
            // (per-case max/sum), but the BMMA backend dense-ranks each chunk's
            // cases with a single monotone walk — this ordering is its invariant.
            // The resolver's exchange slots already arrive sorted (stable no-op).
            let mut order: Vec<usize> = (0..seg.windows.len()).collect();
            order.sort_by_key(|&i| seg.windows[i].case);
            for &wi in &order {
                let w = &seg.windows[wi];
                // Mirror `PackedGallery::from_windows`: drop an out-of-range case.
                if w.case >= seg.n_cases {
                    continue;
                }
                let (base, held) = match turn_base.get(&w.sid) {
                    Some((b, m)) => (*b, Arc::clone(m)),
                    None => {
                        // Pinned atomically with residency by the admission
                        // above, so the governor can't free its pages before
                        // the launch reads them.
                        let ensured = admitted[first_seen[&w.sid]]
                            .take()
                            .expect("a turn is first met once");
                        pinned_sids.push(w.sid);
                        run_ids.push(ensured.run_id);
                        if !ensured.groups.covers(launch) && zero.is_none() {
                            match self.zero_page_addr() {
                                Ok(z) => zero = Some(z),
                                Err(e) => {
                                    self.unpin_all(&all_pinned);
                                    return Err(e);
                                }
                            }
                        }
                        for &addr in &ensured.addrs {
                            for g in launch.iter() {
                                page_ptr.push(match ensured.groups.place(g) {
                                    Some(place) => addr + place as u64 * gw_bytes,
                                    None => zero.expect("claimed above for a narrower turn"),
                                });
                            }
                        }
                        let b = n_pages;
                        n_pages += ensured.addrs.len();
                        turn_base.insert(w.sid, (b, Arc::clone(&ensured.mask)));
                        (b, ensured.mask)
                    }
                };
                let mask = if seg.keep == Keep::All {
                    Arc::clone(&held)
                } else {
                    Arc::clone(scanned.entry((w.sid, seg.keep)).or_insert_with(|| {
                        self.reader_mask(w.sid, w.fingerprint, w.turn, seg.keep)
                    }))
                };
                for t in w.start..w.end.min(w.turn.len()) {
                    // Uniform width only (mirror the reference's admission guard),
                    // and only the tokens this segment keeps, at their place
                    // among the ones the pages hold.
                    if !w.turn.token_has_width(t, wpt) || !mask.keeps(t) {
                        continue;
                    }
                    debug_assert!(held.keeps(t), "a reader's kept token the pages lack");
                    let place = held.rank(t);
                    let page = base + place / PAGE_TOKENS;
                    entries.push(page as u32, place % PAGE_TOKENS, (case_off + w.case) as u32);
                }
            }
            case_off += seg.n_cases;
            entries.end_segment();
            seg_case.push(case_off as i32);
            max_seg_cases = max_seg_cases.max(seg.n_cases);
        }

        let n_tokens = entries.tokens();
        let n_slots = entries.slots_len();
        let seg_law: Vec<VoteLaw> = segments.iter().map(|s| s.law).collect();
        if let Some(bad) = seg_law.iter().find(|l| !l.is_valid()) {
            self.unpin_all(&all_pinned);
            return Err(candle::Error::Msg(format!(
                "paged scan: a segment's vote law {bad:?} is outside 1..={} votes",
                VoteLaw::MAX_TOP_K
            )));
        }
        let seg_top_k: Vec<i32> = seg_law.iter().map(|l| l.top_k as i32).collect();
        let max_k = seg_law.iter().map(|l| l.top_k).max().unwrap_or(1);
        let walked = t_build.elapsed();
        let t_upload = Instant::now();
        let device = if n_tokens == 0 {
            None
        } else {
            match self.upload_index(&page_ptr, &entries, &seg_case, &seg_top_k) {
                Ok(d) => Some(d),
                Err(e) => {
                    self.unpin_all(&all_pinned);
                    return Err(e);
                }
            }
        };
        tracing::debug!(
            target: "candle_conversation::provenance::gallery_arena",
            segments = segments.len(),
            turns = pinned_sids.len(),
            uploaded,
            tokens = n_tokens,
            entries = entries.page.len(),
            index_bytes = (page_ptr.len() * 8) as u64 + entries.bytes(),
            keeps_ms = keeps_time.as_millis() as u64,
            ensure_with_lock_wait_ms = ensure_time.as_millis() as u64,
            walk_ms = walked.saturating_sub(ensure_time).as_millis() as u64,
            index_upload_ms = t_upload.elapsed().as_millis() as u64,
            "built a scan index"
        );
        Ok(PagedIndex {
            device,
            groups: launch,
            seg_groups: seg_groups.to_vec(),
            n_tokens,
            n_slots,
            seg_case,
            seg_tok: entries.seg_tok.clone(),
            n_cases: case_off,
            n_segments: segments.len(),
            max_seg_cases,
            seg_law,
            max_k,
            members: pinned_sids
                .iter()
                .copied()
                .zip(run_ids.iter().copied())
                .collect(),
            pinned_sids,
            run_ids,
            hold: ScanHold::default(),
            touched_ms: AtomicU64::new(0),
            used: AtomicU64::new(0),
            rows: Mutex::new(ProbeRows::new(launch.len() * segments.len() * max_k)),
        })
    }

    /// Upload a built index's arrays once, onto the arena device's stream, for
    /// every launch that reuses the index to read in place.
    pub(super) fn upload_index(
        &self,
        page_ptr: &[u64],
        entries: &IndexEntries,
        seg_case: &[i32],
        seg_top_k: &[i32],
    ) -> Result<DeviceIndex> {
        let Device::Cuda(dev) = &self.device else {
            return Err(candle::Error::Msg("paged scan requires CUDA".into()));
        };
        // Held: the index is reused by every scan until the segments change.
        Ok(DeviceIndex {
            page_ptr: Held::from_host(dev, page_ptr)?,
            entry_page: Held::from_host(dev, &entries.page)?,
            entry_slots: Held::from_host(dev, &entries.slots)?,
            entry_case: Held::from_host(dev, &entries.case)?,
            seg_slot: Held::from_host(dev, &entries.seg_slot)?,
            seg_tok: Held::from_host(dev, &entries.seg_tok)?,
            seg_case: Held::from_host(dev, seg_case)?,
            seg_top_k: Held::from_host(dev, seg_top_k)?,
        })
    }
}
