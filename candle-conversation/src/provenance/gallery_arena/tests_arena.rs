//! Checks the gallery arena's residency and compaction against raw page words,
//! not against tolerances.
//!
//! A round trip reads each page back and compares its transpose exactly, since a
//! page is bytes and a copy that changed one is a corrupted signature. Pins are
//! shown to survive eviction, a preload makes room once rather than per turn, and
//! a compaction moves pages without changing what any surviving turn reads. The
//! tests run one gallery at a time on the card, because galleries on one device
//! share a slot pool, and a compaction over a shared pool would plan against
//! another test's holes.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::batch_ensure::Wanted;
    use super::*;
    use crate::projection::TimelineId;
    use crate::provenance::WideQSig;
    use candle::Device;

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: (0..24)
                .map(|w| fill.wrapping_add((w as u64) << 8))
                .collect(),
        }
    }

    fn sid(n: u64) -> StreamId {
        crate::persistence::content_hash::turn_stream_id(1, n as u32)
    }

    /// A turn made resident as most tests read one: held shared, every fold
    /// group, every token.
    trait Shared {
        fn shared(&self, sid: StreamId, sigs: &[WideQSig], fp: u64) -> Result<Vec<u64>>;
        /// The same, pinned as a scan pins it.
        fn pinned(&self, sid: StreamId, sigs: &[WideQSig], fp: u64) -> Result<Ensured>;
    }

    impl Shared for GalleryArena {
        fn shared(&self, sid: StreamId, sigs: &[WideQSig], fp: u64) -> Result<Vec<u64>> {
            self.ensure_resident(sid, sigs, fp, Holder::Shared, GroupSet::all(3), Keep::All)
        }

        fn pinned(&self, sid: StreamId, sigs: &[WideQSig], fp: u64) -> Result<Ensured> {
            let wanted = Wanted {
                sid,
                sigs: TurnSigs::from(sigs),
                fingerprint: fp,
                holder: Holder::Shared,
                groups: GroupSet::all(3),
                keep: Keep::All,
            };
            Ok(self.scan_ensure_all(&[wanted])?.remove(0))
        }
    }

    /// Held by every test that builds a gallery on the card.
    ///
    /// **The galleries on one device share one slot pool**, and a compaction
    /// plans over the whole pool: run beside another test's gallery, a packed
    /// gallery found that gallery's holes and planned moves, and a scattered
    /// one found its holes filled. One gallery at a time makes each test's
    /// pool its own.
    pub(in super::super) fn gallery_pool() -> std::sync::MutexGuard<'static, ()> {
        static POOL: Mutex<()> = Mutex::new(());
        POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Round-trip: upload a turn, read its pages back, verify the group-major
    /// transpose survived the H2D exactly (raw bytes, not a threshold).
    #[test]
    fn resident_pages_roundtrip_group_major() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU — skip
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs: Vec<WideQSig> = (0..40).map(|t| sig((t as u64) << 40)).collect(); // 2 pages
        let fp = 0xDEADBEEF;
        let addrs = arena.shared(sid(0), &sigs, fp).unwrap();
        assert_eq!(addrs.len(), 2, "40 tokens → 2 pages");
        assert_eq!(arena.resident_turns(), 1);

        // Read pages back and check the transpose token-by-token.
        let pw = page_words(8, GroupSet::all(3));
        let expect = transpose_to_pages(&sigs, &KeepMask::all(40), 8, GroupSet::all(3), pw);
        let res = arena.residency.lock().unwrap();
        let rt = res.get(&sid(0)).unwrap();
        assert_eq!(
            rt.run.addrs(),
            addrs,
            "the addresses handed out are the run's"
        );
        for (p, slot) in rt.run.slots.iter().enumerate() {
            let got = page_io::read_page(&device, slot, pw).unwrap();
            assert_eq!(
                got,
                expect[p * pw..(p + 1) * pw],
                "page {p} bytes differ after H2D"
            );
        }
    }

    /// A page is one slot of the gallery's own arenas, a 6 KiB stride: 2,730 to
    /// a region, and none of them in another tenant's arena.
    #[test]
    fn pages_are_gallery_tenant_slots() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs: Vec<WideQSig> = (0..64).map(|t| sig(t as u64)).collect(); // 2 pages
        arena.shared(sid(0), &sigs, 1).unwrap();
        let res = arena.residency.lock().unwrap();
        let run = &res.get(&sid(0)).unwrap().run;
        assert_eq!(run.slots.len(), 2);
        for slot in &run.slots {
            assert_eq!(slot.stride(), 6144);
            assert_eq!(slot.tenant(), SlotTenant::Gallery);
        }
        assert!(arena.resident_bytes() >= SpanRegion::bytes() as u64);
        assert!(arena.page_bytes() >= 2 * 6144);
    }

    /// A matching fingerprint is a hit: same addresses, no new pages allocated.
    #[test]
    fn fingerprint_hit_reuses_pages() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs: Vec<WideQSig> = (0..10).map(|t| sig(t as u64)).collect();
        let a1 = arena.shared(sid(0), &sigs, 7).unwrap();
        let live1 = arena.live_pages();
        let a2 = arena.shared(sid(0), &sigs, 7).unwrap();
        let live2 = arena.live_pages();
        assert_eq!(a1, a2, "hit must return identical page addresses");
        assert_eq!(live1, live2, "hit must not allocate new pages");
        assert_eq!(arena.resident_turns(), 1);
    }

    /// A changed fingerprint (a re-seal) frees the old pages and re-uploads.
    #[test]
    fn fingerprint_miss_reuploads_and_recycles() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs: Vec<WideQSig> = (0..10).map(|t| sig(t as u64)).collect();
        let _ = arena.shared(sid(0), &sigs, 1).unwrap();
        let live1 = arena.live_pages();
        // Re-seal to a bigger window under a new fingerprint.
        let sigs2: Vec<WideQSig> = (0..40).map(|t| sig((t + 100) as u64)).collect();
        let _ = arena.shared(sid(0), &sigs2, 2).unwrap();
        let live2 = arena.live_pages();
        // Old 1 page freed, 2 new pages live → net live == 2.
        assert_eq!(live1, 1);
        assert_eq!(live2, 2);
        assert_eq!(arena.resident_turns(), 1);
    }

    /// **A resident hit evicts nothing; an upload makes room first.** With a
    /// keep time every turn is past, ensuring a resident turn leaves the other
    /// in place, and only admitting a new one gives the idle turns up.
    #[test]
    fn only_an_upload_runs_the_keep_eviction() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs = |t: u64| -> Vec<WideQSig> { (0..32).map(|k| sig(t * 1000 + k)).collect() };
        arena.shared(sid(0), &sigs(0), 1).unwrap();
        arena.shared(sid(1), &sigs(1), 2).unwrap();
        arena.set_keep_within(Some(Duration::from_nanos(1)));
        std::thread::sleep(Duration::from_millis(2));
        arena.shared(sid(0), &sigs(0), 1).unwrap();
        assert_eq!(arena.resident_turns(), 2, "a hit admits nothing");
        std::thread::sleep(Duration::from_millis(2));
        arena.shared(sid(2), &sigs(2), 3).unwrap();
        let res = arena.residency.lock().unwrap();
        assert_eq!(res.len(), 1, "the upload gave both idle turns up");
        assert!(res.contains_key(&sid(2)));
    }

    /// **A preload makes room once, before its first turn**, then admits every
    /// turn it names, in order: the idle turns past the keep time go, and the
    /// batch's own turns — one re-uploaded, one new — are all resident after.
    #[test]
    fn a_preload_gives_up_idle_turns_once_then_admits_its_own() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs = |t: u64| -> Vec<WideQSig> { (0..32).map(|k| sig(t * 1000 + k)).collect() };
        arena.shared(sid(0), &sigs(0), 1).unwrap();
        arena.shared(sid(1), &sigs(1), 2).unwrap();
        arena.set_keep_within(Some(Duration::from_nanos(1)));
        std::thread::sleep(Duration::from_millis(2));
        let (s0, s2) = (sigs(0), sigs(2));
        let done = arena.ensure_all_resident([
            (sid(0), TurnSigs::from(&s0), 1, GroupSet::all(3), Keep::All),
            (sid(2), TurnSigs::from(&s2), 3, GroupSet::all(3), Keep::All),
        ]);
        assert_eq!(done.len(), 2);
        assert!(done.iter().all(|r| r.is_ok()));
        let res = arena.residency.lock().unwrap();
        let mut held: Vec<StreamId> = res.keys().copied().collect();
        held.sort_by_key(|s| s.0);
        let mut want = vec![sid(0), sid(2)];
        want.sort_by_key(|s| s.0);
        assert_eq!(held, want, "the idle turn 1 went; the batch's turns stay");
    }

    /// **A turn is held for whoever reads it, widening only**, and
    /// `drop_turns` gives up what it names but a pinned turn. One conversation's
    /// own scan holds turn 0 for it; a re-seal under a new fingerprint keeps
    /// that; a shared read of turn 1 after a conversation's makes it shared.
    #[test]
    fn holders_widen_and_drop_turns_skips_pins() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let tl = TimelineId::from_raw(7).unwrap();
        let own = Holder::Conversation(tl);
        let rows: Vec<WideQSig> = (0..32).map(sig).collect();
        let all = GroupSet::all(3);
        let every = Keep::All;
        let ensure = |s, fp, holder| {
            arena
                .ensure_resident(s, &rows, fp, holder, all, every)
                .unwrap()
        };
        ensure(sid(0), 1, own);
        ensure(sid(0), 2, own);
        ensure(sid(1), 3, own);
        ensure(sid(1), 3, Holder::Shared);
        let holder_of = |s: StreamId| {
            arena
                .resident_runs()
                .into_iter()
                .find(|r| r.sid == s)
                .map(|r| r.holder)
        };
        assert_eq!(holder_of(sid(0)), Some(own), "a re-seal keeps its holder");
        assert_eq!(holder_of(sid(1)), Some(Holder::Shared));

        arena.pinned(sid(1), &rows, 3).unwrap();
        let (turns, bytes) = arena.drop_turns(&[sid(0), sid(1), sid(9)]);
        assert_eq!((turns, bytes), (1, 6144), "only the unpinned turn 0 goes");
        assert_eq!(holder_of(sid(0)), None);
        arena.unpin_all(&[sid(1)]);
        assert_eq!(arena.drop_turns(&[sid(1)]), (1, 6144));
        assert_eq!(arena.resident_turns(), 0);
    }

    /// Eviction frees LRU turns and skips the pinned working set.
    #[test]
    fn evict_lru_skips_pins() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        for t in 0..4u64 {
            let sigs: Vec<WideQSig> = (0..32).map(|k| sig(t * 1000 + k)).collect(); // 1 page each
            arena.shared(sid(t), &sigs, t + 1).unwrap();
        }
        assert_eq!(arena.resident_turns(), 4);
        // Pin turns 2 and 3 via a scan-style ensure (hit → pins; fp = t+1).
        let dummy: Vec<WideQSig> = (0..32).map(sig).collect();
        arena.pinned(sid(2), &dummy, 3).unwrap();
        arena.pinned(sid(3), &dummy, 4).unwrap();
        // Evict everything possible — the pinned two must survive.
        let freed = arena.evict_lru(u64::MAX);
        assert!(freed > 0);
        assert_eq!(
            arena.resident_turns(),
            2,
            "only the two pinned turns survive"
        );
        {
            let res = arena.residency.lock().unwrap();
            assert!(res.contains_key(&sid(2)) && res.contains_key(&sid(3)));
        }
        // Unpin → they become evictable again.
        arena.unpin_all(&[sid(2), sid(3)]);
        let freed2 = arena.evict_lru(u64::MAX);
        assert!(freed2 > 0);
        assert_eq!(arena.resident_turns(), 0, "unpinned turns now evict");
    }

    /// **A compaction empties an arena and changes nothing a reader can see.**
    ///
    /// Scatter across two arenas — fill the first, spill a few turns into the
    /// second, then drop turns from the first — so the second's pages fit the
    /// holes below and the pass can give its region back. A pass only moves
    /// pages it can empty an arena with (`plan_slot_moves`); holes inside a
    /// single arena free nothing and are left alone. Then read every surviving
    /// page back, compared against the expected group-major transpose as **raw
    /// words**, not a tolerance: a page is bytes, and a copy that alters one is
    /// a corrupted signature, not an approximation.
    #[test]
    fn compaction_moves_pages_without_changing_them() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs = |t: u64, n: usize| -> Vec<WideQSig> {
            (0..n).map(|k| sig((t << 40) + k as u64)).collect()
        };
        // Turns of 64 pages, enough to fill an arena and spill two into the next.
        let per_arena = candle_nn::kv_cache::REGION_BYTES
            / candle_nn::kv_cache::slot_stride(arena.page_bytes_for(GroupSet::all(3)) as usize);
        let turn_tokens = 64 * PAGE_TOKENS;
        let n_turns = (per_arena / 64 + 3) as u64;
        for t in 0..n_turns {
            arena
                .ensure_resident(
                    sid(t),
                    &sigs(t, turn_tokens),
                    t + 1,
                    Holder::Shared,
                    GroupSet::all(3),
                    Keep::All,
                )
                .unwrap();
        }
        let regions_before = arena_regions(&device, SlotTenant::Gallery);
        assert!(regions_before >= 2, "the turns spill into a second arena");
        // Holes in the first arena, more than the second holds.
        let dropped = 0..4u64;
        for t in dropped.clone() {
            arena.drop_turns(&[sid(t)]);
        }
        let live_before = arena.live_pages();

        let report = arena.compact(0).unwrap();
        assert!(
            report.moved > 0 && report.moved == report.planned,
            "expected the second arena's pages to move, got {report:?}",
        );
        assert_eq!(arena.live_pages(), live_before, "a move frees no page");
        assert!(
            arena_regions(&device, SlotTenant::Gallery) < regions_before,
            "the emptied arena's region went back",
        );

        // Every surviving turn still reads back exactly its own transpose.
        let res = arena.residency.lock().unwrap();
        for t in 0..n_turns {
            if dropped.contains(&t) {
                assert!(!res.contains_key(&sid(t)), "dropped turn is gone");
                continue;
            }
            let pw = page_words(8, GroupSet::all(3));
            let turn = sigs(t, turn_tokens);
            let expect =
                transpose_to_pages(&turn, &KeepMask::all(turn.len()), 8, GroupSet::all(3), pw);
            let run = &res.get(&sid(t)).expect("survivor").run;
            assert_eq!(run.slots.len() * pw, expect.len());
            for (p, slot) in run.slots.iter().enumerate() {
                let got = page_io::read_page(&device, slot, pw).unwrap();
                assert_eq!(
                    got,
                    expect[p * pw..(p + 1) * pw],
                    "turn {t} page {p} after compaction"
                );
            }
        }
    }

    /// A pinned turn is never moved — a scan is dereferencing its pages — and the
    /// generation moves only when something actually did.
    #[test]
    fn compaction_skips_pins_and_bumps_the_generation_only_on_a_move() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let rows: Vec<WideQSig> = (0..64).map(sig).collect();
        for t in 0..4u64 {
            arena.shared(sid(t), &rows, t + 1).unwrap();
        }
        arena.drop_turns(&[sid(0), sid(2)]);
        // Pin one survivor; record where its pages sit.
        arena.pinned(sid(1), &rows, 2).unwrap();
        let pinned_addrs: Vec<u64> = {
            let res = arena.residency.lock().unwrap();
            res.get(&sid(1)).unwrap().run.addrs()
        };

        let gen_before = arena.residency_gen();
        let report = arena.compact(0).unwrap();
        let res = arena.residency.lock().unwrap();
        assert_eq!(
            res.get(&sid(1)).unwrap().run.addrs(),
            pinned_addrs,
            "a pinned turn's pages must not move under an in-flight scan",
        );
        drop(res);
        if report.moved > 0 {
            assert!(
                arena.residency_gen() > gen_before,
                "a cached index must be invalidated when a page moves",
            );
        } else {
            assert_eq!(arena.residency_gen(), gen_before, "no move, no bump");
        }
    }

    /// A gallery with nothing to pack plans nothing and bumps nothing — the pass
    /// has to be free to run every relief episode. **Nor does it tell any index a
    /// change happened**: the change count stands, so a cached index's next scan
    /// trusts it without walking its turns.
    #[test]
    fn compaction_of_a_packed_gallery_is_a_no_op() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let rows: Vec<WideQSig> = (0..32).map(sig).collect();
        for t in 0..3u64 {
            arena.shared(sid(t), &rows, t + 1).unwrap();
        }
        let gen_before = arena.residency_gen();
        let changes_before = arena.changes.load(Ordering::SeqCst);
        let report = arena.compact(0).unwrap();
        assert_eq!((report.planned, report.moved), (0, 0));
        assert_eq!(arena.residency_gen(), gen_before);
        assert_eq!(arena.changes.load(Ordering::SeqCst), changes_before);
    }
}
