//! Packs the gallery's pages toward the low end of the span, so a high frontier
//! does not deny the weight side the ground below it.
//!
//! A page copy is exact, because a page's words encode nothing about their own
//! address and no record needs rewriting. Pinned turns are never moved, since a
//! scan is dereferencing their pages. A turn that does move takes a new run id, so
//! a cached index cannot vouch for addresses it no longer holds. Planning happens
//! before the residency lock is taken, because a claim quiesces the device, and a
//! turn evicted in between simply loses its move. A pass with no moves changes
//! nothing and tells no index that anything happened.

use super::*;

/// What one gallery compaction pass did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GalleryCompaction {
    /// Moves planned, each with its destination claimed.
    pub planned: usize,
    /// Pages that moved. Short of `planned` when a source belonged to a pinned
    /// turn, or to one evicted between the plan and the apply.
    pub moved: usize,
    pub regions_before: usize,
    pub regions_after: usize,
}

impl GalleryCompaction {
    /// Regions the pass handed back to the span.
    pub fn regions_released(&self) -> usize {
        self.regions_before.saturating_sub(self.regions_after)
    }
}

impl GalleryArena {
    /// Pack the gallery's pages toward the low end of the span — the two-cursor
    /// pass the KV pools and the recurrent state run, applied to page slots.
    ///
    /// # Why this is needed even when the gallery reads nearly full
    ///
    /// A snapshot of a *growing* gallery is dense: pages are claimed in runs and
    /// nothing has been freed yet. The scattered state is what churn produces —
    /// runs are **variable length** (a short turn is one page, a long one
    /// hundreds) and they are freed in a different order than they were claimed
    /// (LRU eviction, and a re-seal freeing a turn's old run mid-corpus). That is
    /// the classic external-fragmentation generator, and its effect is not lost
    /// capacity but a **held frontier**: a handful of live pages in a high arena
    /// denies the weight side every region below them. Measuring a young corpus
    /// and concluding there is nothing to pack is measuring the wrong phase.
    ///
    /// # What makes the move safe
    ///
    /// - **A page copy is exact.** Its bytes are folded signature words and encode
    ///   nothing about their own address — unlike a `KvHead` record, whose bytes
    ///   *are* addresses — so there is no fill step and no minting.
    /// - **Pinned turns are never moved.** A scan pins every turn it reads and
    ///   unpins only after its launch has synchronised, so a pinned turn's pages
    ///   are being dereferenced right now. Their sources stay put and the
    ///   destinations claimed for them go back unused.
    /// - **A turn with a moved page gets a fresh run id.** [`scan::PagedIndex`]
    ///   caches raw page addresses and `reuse_index` revalidates them only by
    ///   each turn's unchanged run id. A page that moved without the new id
    ///   would hand the scan kernel a stale address — the one failure here that
    ///   surfaces as quietly wrong retrieval rather than as a fault.
    ///
    /// Planning happens **before** the residency lock is taken, because a claim
    /// opens the arena window and quiesces the device; holding `residency` across
    /// that would block every scan. A turn evicted in that gap simply loses its
    /// move.
    ///
    /// **Planned per page size.** A page is as wide as the groups it holds, so
    /// the gallery's slots come in one size per group count, each its own
    /// arenas; a move never crosses sizes. `max_moves` (zero for no bound) is
    /// the pass's budget across them all.
    pub fn compact(&self, max_moves: usize) -> Result<GalleryCompaction> {
        let regions_before = arena_regions(&self.device, SlotTenant::Gallery);
        let mut by_src: HashMap<u64, ArenaSlot> = HashMap::new();
        for k in 1..=self.n_groups {
            let left = if max_moves == 0 {
                0
            } else {
                match max_moves.saturating_sub(by_src.len()) {
                    0 => break,
                    n => n,
                }
            };
            let moves = plan_slot_moves(
                &self.device,
                SlotTenant::Gallery,
                self.page_bytes_for(GroupSet::all(k)) as usize,
                left,
            )?;
            by_src.extend(moves.into_iter().map(|m| (m.src, m.dst)));
        }
        let planned = by_src.len();
        // **A pass with no moves changes nothing**, and says so to no index: a
        // change noted, or a turn asked whether it is leased, marks the indexes
        // naming it stale, and the next scan of each walks its turns again.
        if planned == 0 {
            return Ok(GalleryCompaction {
                planned,
                moved: 0,
                regions_before,
                regions_after: regions_before,
            });
        }
        let mut moved = 0usize;
        let outcome = {
            let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
            // **A moved turn's run id is renewed failure included.** A copy that
            // errors part-way has already reassigned every slot before it, so
            // leaving that turn's id alone would let a cached `PagedIndex` vouch
            // for addresses it no longer occupies — the one failure here that
            // surfaces as quietly wrong retrieval rather than as a fault. The
            // result is carried past the renewal instead of propagated through it.
            let mut outcome = Ok(());
            self.note_change();
            for (sid, rt) in res.iter_mut() {
                // Only a turn the plan moves is asked whether it is leased —
                // which marks the indexes naming it stale — so a pass leaves
                // every index of turns it does not touch trusted.
                if by_src.is_empty() {
                    break;
                }
                let touched = rt.run.slots.iter().any(|s| by_src.contains_key(&s.ptr()));
                if !touched || rt.pinned != 0 || self.leased(sid) {
                    continue;
                }
                let mut moved_here = 0usize;
                let page_bytes = self.page_bytes_for(rt.groups) as usize;
                for slot in &mut rt.run.slots {
                    let Some(dst) = by_src.remove(&slot.ptr()) else {
                        continue;
                    };
                    if let Err(e) = slot.copy_into(&dst, page_bytes, &self.device) {
                        outcome = Err(e);
                        break;
                    }
                    *slot = dst;
                    moved_here += 1;
                }
                if moved_here > 0 {
                    // Under the residency lock, with the moves: a scan taking the
                    // lock after this sees both the new addresses and the new id.
                    rt.run_id = self.next_run_id.fetch_add(1, Ordering::Relaxed);
                    moved += moved_here;
                }
                if outcome.is_err() {
                    break;
                }
            }
            if moved > 0 {
                self.residency_gen.fetch_add(1, Ordering::Relaxed);
            }
            outcome
        };
        outcome?;
        // Destinations nothing claimed go back to their arenas.
        drop(by_src);
        Ok(GalleryCompaction {
            planned,
            moved,
            regions_before,
            regions_after: arena_regions(&self.device, SlotTenant::Gallery),
        })
    }
}
