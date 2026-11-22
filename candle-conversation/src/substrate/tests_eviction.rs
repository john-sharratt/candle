//! Tests the warm-RAM purge, the hot-VRAM eviction and demotion, and the install
//! paths that move residences between tiers under a budget.
//!
//! A warm copy is dropped only where a lower tier backs it, so the purge keeps a
//! turn's only copy, while a transient residence whose cold copy never comes can
//! be reclaimed. The purge is bounded per pass, so a zero budget sheds
//! incrementally rather than draining the tier at once. Eviction takes the least
//! recently promoted residences first and spares the keep set and every live
//! working set. An ingest demotion keeps its rolling window resident.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::projection::{GroupId, LayerId, SectionId, TimelineAllocator, TimelineId};

    use super::super::tests_working_set::tests::{make_timeline, unshared_timeline};

    use super::super::tests_timeline_meta::tests::{identity_migrate, minimal_sealed_layer};

    /// Two successive appends produce independent entries.
    #[test]
    fn multiple_appends_independent() {
        let (_, _, timeline, mut sub) = make_timeline();

        let migrate = |_: &[SealedSequence]| -> candle::Result<Vec<SealedSequence>> {
            Ok(vec![minimal_sealed_layer()])
        };

        let idx0 = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    sealed_gpu: Some(Arc::new(vec![])),
                    ..Default::default()
                },
                migrate,
            )
            .unwrap();
        let idx1 = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    sealed_gpu: Some(Arc::new(vec![])),
                    ..Default::default()
                },
                migrate,
            )
            .unwrap();

        assert!(sub.turn_sealed_of(timeline, idx0).is_some());
        assert!(sub.turn_sealed_of(timeline, idx1).is_some());
        assert_ne!(idx0, idx1);
    }

    /// Helper: install a warm residence with a known byte_size. Returns the
    /// residence index. Used by the purge tests below to drive the warm LRU
    /// without going through the (CUDA-only) migrate path.
    ///
    /// `cold_backed` mirrors the realistic post-persist state: once a turn's
    /// warm→cold write lands, its warm copy is purgeable (cold is the durable
    /// backup). When `false`, the residence is warm-only with NO lower-tier
    /// backup — the purge must refuse to drop it (its K/V would be lost).
    pub(in super::super) fn install_warm_only(
        sub: &mut Substrate,
        timeline: TimelineId,
        bytes: u64,
        cold_backed: bool,
    ) -> ResidenceIndex {
        let idx = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    sealed_gpu: Some(Arc::new(vec![])),
                    ..Default::default()
                },
                identity_migrate,
            )
            .unwrap();
        let residence = sub.turn_residence(timeline, idx).unwrap();
        // Drop hot, install a marker warm payload, set byte_size for
        // accounting. The actual SealedSequence content doesn't matter
        // for the purge — purge_warm_to_budget counts the residence's
        // byte size, not the warm vec's bytes.
        sub.residences.take_hot(residence);
        sub.residence(residence).set_byte_size(bytes);
        // A non-empty placeholder so the warm slot is `Some`.
        let placeholder = vec![SealedSequence {
            chunks: Vec::new(),
            token_count: 0,
            chunk_size: 32,
            location: candle_nn::kv_cache::ArenaLocation::Cpu,
        }];
        sub.install_warm(residence, placeholder);
        if cold_backed {
            // A non-empty cold marker (its chunks are irrelevant to the purge —
            // only `cold.is_some()` gates the drop).
            sub.install_cold(
                residence,
                vec![StoredSequence {
                    chunks: Vec::new(),
                    token_count: 0,
                }],
            );
        }
        residence
    }

    /// Ample headroom → purge is a no-op even with warm-resident slots.
    #[test]
    fn purge_with_ample_headroom_is_noop() {
        let (_, _, timeline, mut sub) = make_timeline();
        install_warm_only(&mut sub, timeline, 1_000_000, true);
        install_warm_only(&mut sub, timeline, 1_000_000, true);

        // Warm holds 2 MB against a 32 GiB budget — nothing to shed.
        let r = sub.purge_warm_to_budget(32 * 1024 * 1024 * 1024, 1_000_000);
        assert_eq!(r.count, 0);
        assert_eq!(r.bytes, 0);
    }

    /// Tight headroom → keep popping LRU until projected available
    /// covers the threshold.
    #[test]
    fn purge_drops_lru_until_threshold_met() {
        let (_, _, timeline, mut sub) = make_timeline();
        // Order matters: a is LRU (installed first → pushed to front,
        // then b → b is at front, a slides toward back). pop_back
        // returns a first.
        let a = install_warm_only(&mut sub, timeline, 500_000_000, true);
        let b = install_warm_only(&mut sub, timeline, 500_000_000, true);

        // Warm holds 1 GB; the budget is 500 MB and an incoming 500 MB recall
        // needs its room too → target 0 → both LRU victims go.
        let r = sub.purge_warm_to_budget(500_000_000, 500_000_000);
        assert_eq!(r.count, 2, "both warm residences should be popped");
        assert_eq!(r.bytes, 1_000_000_000);
        assert!(!sub.residence(a).has_warm(), "a (LRU) dropped first");
        assert!(!sub.residence(b).has_warm(), "b also dropped");
    }

    /// A warm residence with NO lower-tier backup (its cold write hasn't landed,
    /// or failed) is the turn's ONLY copy — the purge must refuse to drop it even
    /// under pressure, or the K/V is lost. The cold-backed victim is still freed.
    #[test]
    fn purge_keeps_warm_without_backup() {
        let (_, _, timeline, mut sub) = make_timeline();
        // a installed first → slides to the LRU tail → `pop_back` sees it first.
        // a has no cold backup; b does.
        let a = install_warm_only(&mut sub, timeline, 500_000_000, false);
        let b = install_warm_only(&mut sub, timeline, 500_000_000, true);

        // A zero budget would otherwise purge both.
        let r = sub.purge_warm_to_budget(0, 0);
        // Only the cold-backed residence (b) is dropped; the un-backed one (a)
        // is preserved and stays warm-resident.
        assert_eq!(r.count, 1, "only the cold-backed warm is purgeable");
        assert_eq!(r.bytes, 500_000_000);
        assert!(
            sub.residence(a).has_warm(),
            "un-backed warm must be kept (its K/V would otherwise be lost)"
        );
        assert!(!sub.residence(b).has_warm(), "cold-backed warm dropped");
    }

    /// Empty warm LRU → purge exits gracefully without panicking even
    /// when the threshold can't be met.
    #[test]
    fn purge_handles_empty_warm_set() {
        let sub = Substrate::new();
        // No warm residences exist; warm_now = 0 <= any target → no-op even at
        // a zero budget with a large incoming reservation.
        let r = sub.purge_warm_to_budget(0, 10 * 1024 * 1024 * 1024);
        assert_eq!(r.count, 0);
        assert_eq!(r.bytes, 0);
    }

    /// A ZERO budget must not drain the whole warm tier in one pass.
    ///
    /// The zero-budget machine still fills warm transiently mid-drain; shedding
    /// it all at once each pass would turn every later read into a cold
    /// re-fetch. The per-pass bound turns "over budget" into steady incremental
    /// shedding.
    #[test]
    fn a_zero_budget_sheds_incrementally_not_everything() {
        let (_, _, timeline, mut sub) = make_timeline();
        // 24 warm residences of 256 MiB each = 6 GiB of warm tier.
        const VICTIM: u64 = 256 * 1024 * 1024;
        for _ in 0..24 {
            install_warm_only(&mut sub, timeline, VICTIM, true);
        }
        assert_eq!(sub.warm_resident_count(), 24);

        let r = sub.purge_warm_to_budget(0, 0);

        // Exactly the per-pass bound is shed — 1 GiB, i.e. four victims — and the
        // rest of the warm tier survives for the next pass.
        assert_eq!(r.bytes, Substrate::WARM_PURGE_MAX_BYTES_PER_PASS);
        assert_eq!(r.count, 4);
        assert_eq!(sub.warm_resident_count(), 20);

        // A second pass sheds the next slice rather than everything at once.
        let r2 = sub.purge_warm_to_budget(0, 0);
        assert_eq!(r2.count, 4);
        assert_eq!(sub.warm_resident_count(), 16);
    }

    /// A budget the warm tier only slightly exceeds sheds exactly enough — the
    /// bound must not make the purge over-shed when the target is reachable.
    #[test]
    fn a_reachable_budget_stops_at_the_target() {
        let (_, _, timeline, mut sub) = make_timeline();
        const VICTIM: u64 = 256 * 1024 * 1024;
        for _ in 0..24 {
            install_warm_only(&mut sub, timeline, VICTIM, true);
        }
        // 6 GiB warm against a budget one victim short of it.
        let budget = 23 * VICTIM;
        let r = sub.purge_warm_to_budget(budget, 0);
        assert_eq!(r.count, 1);
        assert_eq!(r.bytes, VICTIM);
        assert_eq!(sub.warm_resident_count(), 23);
    }

    /// The persistence thread gates its RAM purge on this count, so it must
    /// track the warm population regardless of HOW warm was produced — the
    /// migration is not warm's only producer (cold recalls land warm too), and a
    /// gate that misses those routes lets warm grow unbounded while the purge
    /// stays silent.
    #[test]
    fn warm_resident_count_tracks_the_purgeable_population() {
        let (_, _, timeline, mut sub) = make_timeline();
        assert_eq!(sub.warm_resident_count(), 0);

        install_warm_only(&mut sub, timeline, 1024, true);
        install_warm_only(&mut sub, timeline, 2048, true);
        assert_eq!(sub.warm_resident_count(), 2);

        // Purging drops them out of the population, so a workspace that has been
        // purged clean stops paying for the OS query.
        let r = sub.purge_warm_to_budget(0, 0);
        assert_eq!(r.count, 2);
        assert_eq!(sub.warm_resident_count(), 0);
    }

    /// The purge fires exactly at the budget boundary: warm at or under the
    /// budget is untouched; one byte of incoming reservation over tips it.
    #[test]
    fn purge_fires_exactly_at_the_budget_boundary() {
        let (_, _, timeline, mut sub) = make_timeline();
        // One LRU warm victim of 2 GB.
        install_warm_only(&mut sub, timeline, 2_000_000_000, true);

        // Warm == budget → nothing shed.
        let r = sub.purge_warm_to_budget(2_000_000_000, 0);
        assert_eq!(r.count, 0, "warm exactly at budget stays");

        // Incoming reservation pushes the target below current warm → shed.
        let r = sub.purge_warm_to_budget(2_000_000_000, 1);
        assert_eq!(r.count, 1);
        assert_eq!(r.bytes, 2_000_000_000);
    }

    /// Helper: install a turn that is **both** hot (on `hot_lru`) and warm,
    /// with a known `byte_size`. A non-empty sealed payload is required so
    /// `append_complete` crosses `install_hot`'s `!is_empty()` gate (an empty
    /// vec is treated as a cold-marker → `hot = None`). Returns
    /// `(turn_index, residence)`.
    pub(in super::super) fn install_hot_and_warm(
        sub: &mut Substrate,
        timeline: TimelineId,
        bytes: u64,
    ) -> (TurnIndex, ResidenceIndex) {
        let idx = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    sealed_gpu: Some(Arc::new(vec![minimal_sealed_layer()])),
                    token_count: 1,
                    ..Default::default()
                },
                identity_migrate,
            )
            .unwrap();
        let residence = sub.turn_residence(timeline, idx).unwrap();
        sub.residence(residence).set_byte_size(bytes);
        sub.install_warm(residence, vec![minimal_sealed_layer()]);
        (idx, residence)
    }

    /// Budget-aware eviction frees the **least-recently-promoted** turns
    /// first (back of `hot_lru`) and stops as soon as the target is covered,
    /// leaving the newest turn hot and every evicted turn's **warm copy
    /// intact** (eviction is hot→warm, never hot→cold).
    #[test]
    fn evict_hot_to_free_oldest_first_keeps_warm() {
        let (_, _, timeline, mut sub) = make_timeline();
        // install_hot pushes to the FRONT, so install order a,b,c leaves
        // hot_lru = [c, b, a] (front newest). Eviction walks from the back:
        // a first, then b.
        let (_, a) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, b) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, c) = install_hot_and_warm(&mut sub, timeline, 100_000_000);

        // Target 150 MB → evict a (100 MB) then b (200 MB ≥ 150 MB) and stop.
        let report = sub.evict_hot_to_free(&[], &[], 150_000_000);
        assert_eq!(report.count, 2, "a + b cover the 150 MB target");
        assert_eq!(report.bytes, 200_000_000);
        // Evicted turns: hot dropped, warm KEPT (the fast-reload backup).
        assert!(sub.residence(a).hot().is_none(), "a hot evicted");
        assert!(sub.residence(a).has_warm(), "a warm KEPT (→warm not →cold)");
        assert!(sub.residence(b).hot().is_none(), "b hot evicted");
        assert!(sub.residence(b).has_warm(), "b warm KEPT");
        // Newest turn untouched — the working set stays resident.
        assert!(sub.residence(c).has_hot(), "c (newest) still hot");
    }

    /// The selection (keep set) is never evicted, even when it's the oldest
    /// turn — eviction skips it and frees a younger one instead.
    #[test]
    fn evict_hot_to_free_protects_keep_set() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (a_idx, a) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, b) = install_hot_and_warm(&mut sub, timeline, 100_000_000);

        // a is oldest (would be evicted first) but it's in the keep set, so
        // eviction must skip it and take b instead.
        let keep = [TurnKey {
            timeline,
            index: a_idx,
        }];
        let report = sub.evict_hot_to_free(&[], &keep, 100_000_000);
        assert_eq!(report.count, 1);
        assert!(sub.residence(a).has_hot(), "a protected by keep set");
        assert!(sub.residence(b).hot().is_none(), "b evicted instead");
    }

    /// A residence any slot's live working set attends is protected, not only
    /// the projection being elevated: one dialogue's elevation must not evict
    /// what another slot's wave reads.
    #[test]
    fn evict_hot_to_free_protects_every_live_working_set() {
        let (layer, group, file, mut sub) = make_timeline();
        let (a_idx, a) = install_hot_and_warm(&mut sub, file, 100_000_000);
        let other = unshared_timeline(1);
        sub.register_timeline(other, layer, group);
        let (_, b) = install_hot_and_warm(&mut sub, other, 100_000_000);
        sub.set_working_set_pins(&[TurnKey::new(file, a_idx)], &[]);

        let report = sub.evict_hot_to_free(&[], &[], 100_000_000);
        assert_eq!(report.count, 1);
        assert!(sub.residence(a).has_hot(), "the attended file stays hot");
        assert!(sub.residence(b).hot().is_none(), "the other one goes");
    }

    /// A guest's drain takes a live working set too, and still spares the
    /// keep list.
    #[test]
    fn evict_hot_for_guest_takes_live_working_sets_but_not_the_keep_list() {
        let (layer, group, file, mut sub) = make_timeline();
        let (a_idx, a) = install_hot_and_warm(&mut sub, file, 100_000_000);
        let other = unshared_timeline(1);
        sub.register_timeline(other, layer, group);
        let (b_idx, b) = install_hot_and_warm(&mut sub, other, 100_000_000);
        sub.set_working_set_pins(&[TurnKey::new(file, a_idx)], &[]);

        let report = sub.evict_hot_for_guest(&[], &[TurnKey::new(other, b_idx)], u64::MAX);
        assert_eq!(report.count, 1);
        assert!(sub.residence(a).hot().is_none(), "the attended file goes");
        assert!(sub.residence(a).has_warm(), "and keeps its warm copy");
        assert!(sub.residence(b).has_hot(), "the kept turn stays");
    }

    /// The census counts what the eviction would take and names what it
    /// protects, and evicts nothing itself.
    #[test]
    fn hot_census_reads_as_the_eviction_would() {
        let (layer, group, file, mut sub) = make_timeline();
        let (a_idx, a) = install_hot_and_warm(&mut sub, file, 100_000_000);
        let other = unshared_timeline(1);
        sub.register_timeline(other, layer, group);
        let (_, b) = install_hot_and_warm(&mut sub, other, 50_000_000);
        sub.set_working_set_pins(&[TurnKey::new(file, a_idx)], &[]);

        let census = sub.hot_census(&[], &[]);
        assert_eq!(census.pinned, 100_000_000);
        assert_eq!(census.evictable, 50_000_000);
        assert!(sub.residence(a).has_hot() && sub.residence(b).has_hot());
        assert_eq!(
            sub.evict_hot_to_free(&[], &[], u64::MAX).bytes,
            census.evictable
        );
    }

    /// Targeted demotion drops the hot copy of exactly the NAMED turns (keeping
    /// warm) and leaves every other hot turn resident — the inverse of the
    /// keep-set eviction. A turn with no warm copy is left hot (dropping it would
    /// lose its K/V), and re-demoting an already-warm-only turn is a no-op.
    #[test]
    fn demote_turns_to_warm_drops_named_keeps_warm_and_rest() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (a_idx, a) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (b_idx, b) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (c_idx, c) = install_hot_and_warm(&mut sub, timeline, 100_000_000);

        let demoted = sub.demote_turns_to_warm(&[
            TurnKey {
                timeline,
                index: a_idx,
            },
            TurnKey {
                timeline,
                index: b_idx,
            },
        ]);
        assert_eq!(demoted, 2, "a + b demoted");
        assert!(sub.residence(a).hot().is_none(), "a hot dropped");
        assert!(sub.residence(a).has_warm(), "a warm KEPT (cheap reload)");
        assert!(sub.residence(b).hot().is_none(), "b hot dropped");
        assert!(sub.residence(b).has_warm(), "b warm KEPT");
        assert!(sub.residence(c).has_hot(), "c (unnamed) still hot");

        // Idempotent: a is now warm-only, so re-demoting it frees nothing.
        assert_eq!(
            sub.demote_turns_to_warm(&[TurnKey {
                timeline,
                index: a_idx
            }]),
            0,
            "already warm-only → no-op"
        );

        // A hot turn with NO warm copy is left hot — dropping it would lose K/V.
        sub.residences.take_warm(c);
        assert_eq!(
            sub.demote_turns_to_warm(&[TurnKey {
                timeline,
                index: c_idx
            }]),
            0,
            "hot-without-warm is not demotable"
        );
        assert!(sub.residence(c).has_hot(), "c hot-without-warm untouched");
    }

    /// Gentle-early ingest demotion drops the hot copy of an ingest timeline's
    /// sealed, warm-backed turns EXCEPT the newest `keep_recent` (the rolling
    /// window), keeping warm; a hot-without-warm turn (the active writer) is left
    /// hot.
    #[test]
    fn demote_cold_ingest_keeps_window_and_warm() {
        let (_, _, timeline, mut sub) = make_timeline();
        // Five sealed, hot+warm turns, oldest→newest.
        let (_, t0) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t1) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t2) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t3) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t4) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        // The newest turn is the active writer: hot-only (no warm copy yet).
        sub.residences.take_warm(t4);

        let mut ingest = std::collections::HashSet::new();
        ingest.insert(timeline);

        // keep_recent = 2 → protect t3, t4 (the rolling window); no live working
        // set; unbounded target → demote the whole LRU tail t0, t1, t2 (all
        // warm-backed), oldest-first.
        let report = sub.demote_cold_ingest(&ingest, &[], &[], 2, u64::MAX);
        assert_eq!(report.count, 3, "t0,t1,t2 demoted; t3,t4 window kept");
        assert_eq!(report.bytes, 300_000_000);
        for r in [t0, t1, t2] {
            assert!(
                sub.residence(r).hot().is_none(),
                "older ingest turn demoted"
            );
            assert!(sub.residence(r).has_warm(), "warm KEPT (cheap reload)");
        }
        assert!(sub.residence(t3).has_hot(), "window turn stays hot");
        assert!(
            sub.residence(t4).has_hot(),
            "hot-without-warm writer stays hot"
        );

        // Idempotent: the demoted turns are now warm-only, so a second pass frees
        // nothing.
        assert_eq!(
            sub.demote_cold_ingest(&ingest, &[], &[], 2, u64::MAX).count,
            0,
            "already demoted"
        );
    }

    /// `target_bytes` bounds the LRU walk: with a target that covers only two
    /// turns, the demote stops after shedding the two oldest and leaves the rest
    /// hot — relief to the watermark, never the whole working set.
    #[test]
    fn demote_cold_ingest_stops_at_target() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_, t0) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t1) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t2) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t3) = install_hot_and_warm(&mut sub, timeline, 100_000_000);

        let mut ingest = std::collections::HashSet::new();
        ingest.insert(timeline);

        // keep_recent = 0 (nothing protected), target = 150 MB → the walk frees t0
        // (100 MB, still < target) then t1 (200 MB, ≥ target) and stops. t2, t3 stay
        // hot.
        let report = sub.demote_cold_ingest(&ingest, &[], &[], 0, 150_000_000);
        assert_eq!(report.count, 2, "only the two oldest shed to meet target");
        assert_eq!(report.bytes, 200_000_000);
        assert!(sub.residence(t0).hot().is_none(), "oldest demoted");
        assert!(sub.residence(t1).hot().is_none(), "second-oldest demoted");
        assert!(sub.residence(t2).has_hot(), "target met → t2 kept hot");
        assert!(sub.residence(t3).has_hot(), "target met → t3 kept hot");
    }

    /// An ingest timeline that was never registered must be skipped, not
    /// panic — and must not stop the registered ones being demoted.
    ///
    /// This case only became reachable when the walk was driven by the ingest
    /// set instead of by the timeline map: the old form iterated registered
    /// timelines and filtered, so an unregistered id could not be visited at
    /// all. Driving from the set means the lookup can miss, and the miss has to
    /// be tolerated.
    #[test]
    fn demote_cold_ingest_skips_unregistered_ingest_timeline() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_, a) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, b) = install_hot_and_warm(&mut sub, timeline, 100_000_000);

        let mut ingest = std::collections::HashSet::new();
        ingest.insert(timeline);
        // Never registered on this substrate.
        ingest.insert(TimelineId::from_raw(4242).unwrap());

        let report = sub.demote_cold_ingest(&ingest, &[], &[], 0, u64::MAX);
        assert_eq!(report.count, 2, "the registered ingest turns still demote");
        for r in [a, b] {
            assert!(sub.residence(r).hot().is_none(), "demoted");
        }
    }

    /// A timeline NOT in the ingest set is never touched — the demotion is scoped
    /// to append-only ingest timelines, so chat working sets are safe.
    #[test]
    fn demote_cold_ingest_ignores_non_ingest_timeline() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_, a) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, b) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, c) = install_hot_and_warm(&mut sub, timeline, 100_000_000);

        // Empty ingest set → this timeline is not an ingest timeline, so even an
        // unbounded target frees nothing.
        let report =
            sub.demote_cold_ingest(&std::collections::HashSet::new(), &[], &[], 0, u64::MAX);
        assert_eq!(report.count, 0);
        for r in [a, b, c] {
            assert!(sub.residence(r).has_hot(), "non-ingest timeline untouched");
        }
    }

    /// `install_cold` frees a section's hot VRAM only when its residence is
    /// flagged `evict_when_cold` (the prefix-transparent collection-member
    /// offload path); a plain/boundary section keeps its hot copy resident.
    #[test]
    fn install_cold_frees_section_hot_only_when_evict_flagged() {
        let mut sub = Substrate::new();
        let install = |sub: &mut Substrate, id: u32| {
            sub.set_section_full(
                SectionId::new(id),
                StreamId::default(),
                32,
                vec![minimal_sealed_layer()],
                Arc::new(vec![]),
            )
        };
        let cold = || {
            vec![StoredSequence {
                chunks: vec![StoredChunk {
                    log_offset: 0,
                    record_len: 1024,
                    token_count: 32,
                }],
                token_count: 32,
            }]
        };

        let member = install(&mut sub, 1);
        let boundary = install(&mut sub, 2);
        assert!(sub.residence(member).has_hot());
        assert!(sub.residence(boundary).has_hot());

        // Only the collection member is flagged for offload-on-persist.
        sub.mark_section_evict_when_cold(member);

        sub.install_cold(member, cold());
        sub.install_cold(boundary, cold());

        // Flagged member: cold copy lands AND hot VRAM is freed.
        assert!(
            sub.residence(member).hot().is_none(),
            "flagged member: VRAM offloaded once cold lands"
        );
        assert!(sub.residence(member).has_cold());
        // Boundary section: cold copy lands but it stays hot for the build.
        assert!(
            sub.residence(boundary).has_hot(),
            "boundary section stays hot"
        );
        assert!(sub.residence(boundary).has_cold());
    }

    /// A completed-ingest turn flagged `evict_when_cold` is fully evicted from
    /// BOTH resident tiers when its cold copy lands: `install_cold` drops hot
    /// AND warm and clears both LRUs, leaving it cold-only on NVMe. An unflagged
    /// turn keeps its resident copies.
    #[test]
    fn install_cold_fully_evicts_flagged_turn_from_both_tiers() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_, flagged) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, kept) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let cold = || {
            vec![StoredSequence {
                chunks: vec![StoredChunk {
                    log_offset: 0,
                    record_len: 1024,
                    token_count: 32,
                }],
                token_count: 32,
            }]
        };

        // One turn is a completed-ingest residence flagged for full eviction; the
        // other stays live. Both start hot + warm (on both LRUs).
        sub.residence(flagged).set_flag(EVICT_WHEN_COLD);
        assert!(
            sub.residences.hot_members().contains(&flagged)
                && sub.residences.warm_members().contains(&flagged)
        );

        sub.install_cold(flagged, cold());
        sub.install_cold(kept, cold());

        // Flagged: cold durable, and BOTH resident tiers freed + off both LRUs.
        assert!(sub.residence(flagged).has_cold(), "cold durable");
        assert!(sub.residence(flagged).hot().is_none(), "VRAM reclaimed");
        assert!(sub.residence(flagged).warm().is_none(), "RAM reclaimed");
        assert!(
            !sub.residences.hot_members().contains(&flagged),
            "off the hot set"
        );
        assert!(
            !sub.residences.warm_members().contains(&flagged),
            "off the warm set"
        );

        // Unflagged: cold lands but the turn stays fully resident.
        assert!(sub.residence(kept).has_cold());
        assert!(sub.residence(kept).has_hot(), "kept stays hot");
        assert!(sub.residence(kept).has_warm(), "kept stays warm");
    }

    /// **A recall stays hot while a pin names it, and never takes a warm
    /// copy.** A completed-ingest turn, offloaded to cold-only, is pulled back
    /// hot by `install_promoted`. It is never queued for a hot→warm copy — the
    /// log already holds its bytes. While the published pins name it (the
    /// scheduler holds a slot's previous working set for one projection) the
    /// pass leaves it hot, so a re-selection finds it there; once no pin
    /// names it, the offload takes it back to cold-only. A flagged turn no
    /// pin names is offloaded at once.
    #[test]
    fn a_recall_stays_hot_while_pinned_and_takes_no_warm_copy() {
        let (_, _, timeline, mut sub) = make_timeline();
        let cold = || {
            vec![StoredSequence {
                chunks: vec![StoredChunk {
                    log_offset: 0,
                    record_len: 1024,
                    token_count: 32,
                }],
                token_count: 32,
            }]
        };
        let (recalled_idx, recalled) = install_hot_and_warm(&mut sub, timeline, 10);
        let (_, untouched) = install_hot_and_warm(&mut sub, timeline, 10);
        for r in [recalled, untouched] {
            sub.residence(r).set_flag(EVICT_WHEN_COLD);
            sub.install_cold(r, cold());
            assert!(
                !sub.residence(r).has_hot() && !sub.residence(r).has_warm(),
                "offloaded to cold-only"
            );
        }

        // The projection recalls one turn: pinned, then lifted hot.
        let key = TurnKey::new(timeline, recalled_idx);
        sub.set_working_set_pins(&[key], &[]);
        let installed = sub.install_promoted(
            vec![ColdRecall {
                kind: PromotionItemKind::Turn(key),
                residence: recalled,
                hot: vec![minimal_sealed_layer()],
            }],
            Vec::new(),
        );
        assert_eq!(installed, 1);

        // Its hot copy is the stored form the log holds: no hot→warm copy.
        assert!(
            sub.snapshot_pending_warm()
                .iter()
                .all(|(idx, _, _)| *idx != recalled),
            "a recall is never queued for a warm copy"
        );
        assert_eq!(sub.pending_warm_bytes(), 0);

        // Held by a pin, a pass leaves it hot for the next selection.
        assert_eq!(sub.demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS), 0);
        assert!(sub.turn_tier_state(timeline, recalled_idx).unwrap().hot);

        // No pin names it any more: offloaded back to cold-only.
        sub.set_working_set_pins(&[], &[]);
        assert_eq!(sub.demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS), 1);
        assert!(!sub.residence(recalled).has_hot(), "back to cold-only");
        assert!(!sub.residence(recalled).has_warm(), "no warm copy made");
        assert!(sub.residence(recalled).has_cold());
        assert!(sub.residence(recalled).has_flag(EVICT_WHEN_COLD));

        // A flagged turn no projection ever named is offloaded on the spot.
        let (_, ingested) = install_hot_and_warm(&mut sub, timeline, 10);
        sub.residence(ingested).set_flag(EVICT_WHEN_COLD);
        assert_eq!(sub.demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS), 1);
        assert!(!sub.residence(ingested).has_hot());

        // The turn no projection recalled is still offload-only.
        assert!(!sub.residence(untouched).has_hot() && !sub.residence(untouched).has_warm());
    }

    /// `mark_timeline_evict_when_cold` flags every turn residence of exactly the
    /// named timeline (leaving other timelines' turns untouched) and reports the
    /// count; an unknown timeline is a no-op.
    #[test]
    fn mark_timeline_evict_when_cold_flags_every_turn_of_that_timeline() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let ingest = alloc.next();
        let live = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(ingest, layer, group);
        sub.register_timeline(live, layer, group);

        let (_, a) = install_hot_and_warm(&mut sub, ingest, 10);
        let (_, b) = install_hot_and_warm(&mut sub, ingest, 10);
        let (_, c) = install_hot_and_warm(&mut sub, live, 10);

        let n = sub.mark_timeline_evict_when_cold(ingest);
        assert_eq!(n, 2, "both ingest turns flagged");
        assert!(sub.residence(a).has_flag(EVICT_WHEN_COLD));
        assert!(sub.residence(b).has_flag(EVICT_WHEN_COLD));
        assert!(
            !sub.residence(c).has_flag(EVICT_WHEN_COLD),
            "the other timeline's turn is untouched"
        );
        // Already-warm ingest turns get their hot copy dropped immediately (VRAM
        // reclaimed now) while warm is kept; the other timeline stays fully hot.
        assert!(sub.residence(a).hot().is_none() && sub.residence(a).has_warm());
        assert!(sub.residence(b).hot().is_none() && sub.residence(b).has_warm());
        assert!(
            !sub.residences.hot_members().contains(&a)
                && !sub.residences.hot_members().contains(&b)
        );
        assert!(sub.residence(c).has_hot() && sub.residence(c).has_warm());

        // Unknown timeline → no-op.
        assert_eq!(sub.mark_timeline_evict_when_cold(alloc.next()), 0);
    }

    /// `timeline_hot_residences` counts exactly the named timeline's turns that
    /// still hold a hot copy, and reads zero once the eviction has dropped them.
    #[test]
    fn timeline_hot_residences_counts_that_timelines_hot_turns() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let evicted = alloc.next();
        let live = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(evicted, layer, group);
        sub.register_timeline(live, layer, group);
        install_hot_and_warm(&mut sub, evicted, 10);
        install_hot_and_warm(&mut sub, evicted, 10);
        install_hot_and_warm(&mut sub, live, 10);

        assert_eq!(sub.timeline_hot_residences(evicted), 2);
        assert_eq!(sub.timeline_hot_residences(live), 1);
        sub.mark_timeline_evict_when_cold(evicted);
        assert_eq!(sub.timeline_hot_residences(evicted), 0);
        assert_eq!(sub.timeline_hot_residences(live), 1);
        assert_eq!(sub.timeline_hot_residences(alloc.next()), 0);
    }

    /// A residence pinned into the current wave's working set is NOT evicted by
    /// `install_cold` even when flagged `evict_when_cold`: the drop is deferred
    /// (hot+warm kept, flag still set) so the persistence thread can't free a
    /// turn the scheduler is attending. An identically-flagged but unpinned turn
    /// evicts as normal. This is the multi-timeline-scan free-under-read fix.
    #[test]
    fn working_set_pin_defers_install_cold_eviction() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (pinned_idx, pinned) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, unpinned) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let cold = || {
            vec![StoredSequence {
                chunks: vec![StoredChunk {
                    log_offset: 0,
                    record_len: 1024,
                    token_count: 32,
                }],
                token_count: 32,
            }]
        };

        // Both are completed-ingest residences flagged for full eviction.
        sub.residence(pinned).set_flag(EVICT_WHEN_COLD);
        sub.residence(unpinned).set_flag(EVICT_WHEN_COLD);
        // Only the first is in the scheduler's live working set.
        sub.set_working_set_pins(&[TurnKey::new(timeline, pinned_idx)], &[]);

        sub.install_cold(pinned, cold());
        sub.install_cold(unpinned, cold());

        // Pinned: cold durable, but hot+warm KEPT (deferred), flag still set.
        assert!(sub.residence(pinned).has_cold(), "cold durable");
        assert!(sub.residence(pinned).has_hot(), "pinned keeps hot");
        assert!(sub.residence(pinned).has_warm(), "pinned keeps warm");
        assert!(
            sub.residences.hot_members().contains(&pinned)
                && sub.residences.warm_members().contains(&pinned)
        );
        assert!(
            sub.residence(pinned).has_flag(EVICT_WHEN_COLD),
            "flag stays set — eviction is deferred, not cancelled"
        );
        // Unpinned: evicted from both resident tiers as usual.
        assert!(
            sub.residence(unpinned).hot().is_none(),
            "unpinned VRAM reclaimed"
        );
        assert!(
            sub.residence(unpinned).warm().is_none(),
            "unpinned RAM reclaimed"
        );
    }

    /// `install_warm_and_evict_hot` keeps the hot copy (dual hot+warm residency)
    /// when the residence is pinned, and drops it when not — so a mid-migrate
    /// install can't yank a residence the current wave is using.
    #[test]
    fn working_set_pin_defers_install_warm_and_evict_hot() {
        let (_, _, timeline, mut sub) = make_timeline();
        // Two hot-only residences (append_complete installs hot; no warm yet).
        let pinned_idx = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    sealed_gpu: Some(Arc::new(vec![minimal_sealed_layer()])),
                    ..Default::default()
                },
                identity_migrate,
            )
            .unwrap();
        let pinned = sub.turn_residence(timeline, pinned_idx).unwrap();
        let unpinned_idx = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    sealed_gpu: Some(Arc::new(vec![minimal_sealed_layer()])),
                    ..Default::default()
                },
                identity_migrate,
            )
            .unwrap();
        let unpinned = sub.turn_residence(timeline, unpinned_idx).unwrap();

        sub.set_working_set_pins(&[TurnKey::new(timeline, pinned_idx)], &[]);

        // The migrate produced a warm copy for each; install it and evict hot.
        sub.install_warm_and_evict_hot(pinned, vec![minimal_sealed_layer()]);
        sub.install_warm_and_evict_hot(unpinned, vec![minimal_sealed_layer()]);

        // Pinned: hot KEPT (now dual hot+warm); unpinned: hot dropped, warm only.
        assert!(sub.residence(pinned).has_hot(), "pinned keeps hot");
        assert!(sub.residence(pinned).has_warm(), "pinned gained warm");
        assert!(sub.residences.hot_members().contains(&pinned));
        assert!(
            sub.residence(unpinned).hot().is_none(),
            "unpinned hot dropped"
        );
        assert!(
            sub.residence(unpinned).has_warm(),
            "unpinned warm installed"
        );
        assert!(!sub.residences.hot_members().contains(&unpinned));
    }
}
