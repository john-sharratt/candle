//! Tests the rules that decide which residences a copy, a distillation or an idle
//! demote may take, so none of them takes KV a live wave is reading.
//!
//! A copy lands only over the hot content it was made from, so a residence killed
//! or replaced while it copied takes no warm copy. Idle demotion, distillation and
//! the warm purge each skip a pinned residence and one still owed a quantize.
//! Distillation releases both resident tiers because the signature, not the K/V,
//! answers the belief scan. A transient residence is reclaimable from warm RAM,
//! while an ordinary one keeps its last durable copy.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::projection::{GroupId, LayerId, SectionId, TimelineAllocator, TimelineId};

    use super::super::tests_working_set::tests::make_timeline;

    use super::super::tests_timeline_meta::tests::{identity_migrate, minimal_sealed_layer};

    use super::super::tests_eviction::tests::install_hot_and_warm;

    /// **A copy's warm install leaves the hot copy as it was wherever hot
    /// stays.** Unflagged, both tiers stand; flagged and unpinned, hot goes and
    /// warm stays; flagged but in a live working set, hot stays for the
    /// reading wave, with the flag still set for the demote that sheds it
    /// later.
    #[test]
    fn a_copy_keeps_the_hot_copy_wherever_hot_stays() {
        let (_, _, timeline, mut sub) = make_timeline();
        let fresh = |sub: &mut Substrate| {
            let idx = sub
                .append_complete(
                    timeline,
                    TurnPartWrite {
                        sealed_gpu: Some(Arc::new(vec![minimal_sealed_layer()])),
                        ..Default::default()
                    },
                    identity_migrate,
                )
                .unwrap();
            (idx, sub.turn_residence(timeline, idx).unwrap())
        };
        let (_, plain) = fresh(&mut sub);
        let (_, flagged) = fresh(&mut sub);
        let (pinned_idx, flagged_pinned) = fresh(&mut sub);
        sub.residence(flagged).set_flag(EVICT_WHEN_COLD);
        sub.residence(flagged_pinned).set_flag(EVICT_WHEN_COLD);
        sub.set_working_set_pins(&[TurnKey::new(timeline, pinned_idx)], &[]);
        let mut read = Vec::new();
        for r in [plain, flagged, flagged_pinned] {
            let from = sub.residence(r).hot().expect("hot before the copy");
            assert!(sub.install_copied(r, &from, vec![minimal_sealed_layer()]));
            read.push(from);
        }
        let same = |r: ResidenceIndex, from: &Arc<HotKv>| {
            sub.residence(r)
                .hot()
                .is_some_and(|h| h.lineage() == from.lineage())
        };

        assert!(
            same(plain, &read[0]),
            "unflagged: the hot copy stays as read"
        );
        assert!(sub.residence(plain).has_warm());

        assert!(
            !sub.residence(flagged).has_hot(),
            "flagged, unpinned: hot goes"
        );
        assert!(sub.residence(flagged).has_warm());

        assert!(
            same(flagged_pinned, &read[2]),
            "flagged and pinned: the hot copy stays for the wave"
        );
        assert!(sub.residence(flagged_pinned).has_warm());
        assert!(
            sub.residence(flagged_pinned).has_flag(EVICT_WHEN_COLD),
            "the flag stands"
        );
    }

    /// **A copy lands only over the hot content it was made from.** A
    /// residence killed while it copied, or whose hot copy was taken and
    /// replaced — a gap's quantize among them — takes no warm copy: it
    /// describes state that has moved on.
    #[test]
    fn a_copy_over_a_residence_that_moved_on_installs_nothing() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_, killed) = install_hot_and_warm(&mut sub, timeline, 10);
        sub.residences.take_warm(killed);
        let from_killed = sub.residence(killed).hot().unwrap();
        sub.residences.kill(killed);
        assert!(!sub.install_copied(killed, &from_killed, vec![minimal_sealed_layer()]));
        assert!(!sub.residence(killed).has_hot() && !sub.residence(killed).has_warm());

        let (_, replaced) = install_hot_and_warm(&mut sub, timeline, 10);
        sub.residences.take_warm(replaced);
        let stale = sub.residence(replaced).hot().unwrap();
        sub.residences.take_hot(replaced);
        sub.residences
            .install_hot(replaced, vec![minimal_sealed_layer()]);
        assert!(!sub.install_copied(replaced, &stale, vec![minimal_sealed_layer()]));
        assert!(
            !sub.residence(replaced).has_warm(),
            "no warm copy of the bytes the residence no longer holds"
        );
    }

    /// **A pinned fresh seal is migrated; a pinned persisted residence is not.**
    /// The migrate is what swaps a seal's native hot copy for its quantized one,
    /// so deferring it while pinned left a turn its own conversation keeps
    /// attending native for the conversation's whole life. A residence with its
    /// cold write in flight is held back while the working set names it; one
    /// recalled from cold is never migrated, pinned or not — the log already
    /// holds its bytes. Every count that mirrors the work list agrees.
    #[test]
    fn a_pinned_fresh_seal_is_migrated_and_a_pinned_recall_is_not() {
        let (_, _, timeline, mut sub) = make_timeline();
        let fresh_hot = |sub: &mut Substrate| {
            let idx = sub
                .append_complete(
                    timeline,
                    TurnPartWrite {
                        sealed_gpu: Some(Arc::new(vec![minimal_sealed_layer()])),
                        ..Default::default()
                    },
                    identity_migrate,
                )
                .unwrap();
            (idx, sub.turn_residence(timeline, idx).unwrap())
        };
        let (fresh_idx, fresh) = fresh_hot(&mut sub);
        let (recalled_idx, recalled) = fresh_hot(&mut sub);
        let (writing_idx, writing) = fresh_hot(&mut sub);
        let (_, unpinned_recall) = fresh_hot(&mut sub);
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
        sub.residences.set_cold(recalled, Some(cold()));
        sub.residences.set_cold(unpinned_recall, Some(cold()));
        sub.residence(writing).set_flag(COLD_PENDING);
        for (r, bytes) in [
            (fresh, 1_000),
            (recalled, 200),
            (writing, 30),
            (unpinned_recall, 4),
        ] {
            sub.residence(r).set_byte_size(bytes);
        }
        sub.set_working_set_pins(
            &[
                TurnKey::new(timeline, fresh_idx),
                TurnKey::new(timeline, recalled_idx),
                TurnKey::new(timeline, writing_idx),
            ],
            &[],
        );

        let mut pending: Vec<ResidenceIndex> = sub
            .snapshot_pending_warm_for(None)
            .into_iter()
            .map(|(idx, _)| idx)
            .collect();
        pending.sort_unstable_by_key(|r| r.0);
        assert_eq!(pending, vec![fresh]);
        assert_eq!(sub.pending_warm_count(), 1);
        assert_eq!(sub.pending_warm_bytes(), 1_000);
        assert_eq!(sub.warm_backlog_split(), (1, 1_000, 1, 30));
    }

    /// `mark_timeline_evict_when_cold`'s immediate hot-drop skips a pinned
    /// residence (deferred, flag still set) while dropping an unpinned sibling.
    #[test]
    fn working_set_pin_defers_mark_timeline_immediate_drop() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (pinned_idx, pinned) = install_hot_and_warm(&mut sub, timeline, 10);
        let (_, unpinned) = install_hot_and_warm(&mut sub, timeline, 10);

        sub.set_working_set_pins(&[TurnKey::new(timeline, pinned_idx)], &[]);
        let n = sub.mark_timeline_evict_when_cold(timeline);
        assert_eq!(n, 2, "both turns flagged");

        // Both flagged; the immediate hot-drop fires only for the unpinned turn.
        assert!(sub.residence(pinned).has_flag(EVICT_WHEN_COLD));
        assert!(sub.residence(unpinned).has_flag(EVICT_WHEN_COLD));
        assert!(
            sub.residence(pinned).has_hot(),
            "pinned keeps hot (deferred)"
        );
        assert!(sub.residences.hot_members().contains(&pinned));
        assert!(
            sub.residence(unpinned).hot().is_none(),
            "unpinned hot dropped now"
        );
        assert!(!sub.residences.hot_members().contains(&unpinned));
    }

    /// **The idle demote cannot take KV the current wave is using.**
    ///
    /// This is the whole safety property. A residence in the working set is
    /// stamped with the current epoch as the pins are published, so it fails the
    /// staleness test arithmetically — the active set is excluded by
    /// construction rather than by a filter that could drift. The idle sibling
    /// beside it is shed on the same call, which is what proves the test is
    /// exercising the predicate and not simply flagging nothing.
    #[test]
    fn idle_demote_never_takes_the_current_working_set() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (active_idx, active) = install_hot_and_warm(&mut sub, timeline, 10);
        let (_idle_idx, idle) = install_hot_and_warm(&mut sub, timeline, 10);

        // Run many passes with `active` re-published each time — exactly a live
        // conversation. Each pass ticks the clock, so `idle` ages past the grace
        // window while `active` keeps being re-stamped at the current epoch.
        let mut shed = 0;
        for _ in 0..(IDLE_DEMOTE_GRACE_EPOCHS + 8) {
            sub.set_working_set_pins(&[TurnKey::new(timeline, active_idx)], &[]);
            shed += sub.demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS);
        }
        assert_eq!(shed, 1, "only the idle residence is shed");
        assert!(
            sub.residence(active).has_hot(),
            "a residence the current wave is attending must keep its hot copy"
        );
        assert!(
            sub.residence(idle).hot().is_none(),
            "the idle one's hot is dropped, since warm already exists"
        );
        assert!(!sub.residences.hot_members().contains(&idle));
        // **Shedding hot is not a verdict on the residence's kind.** The demote
        // must leave `evict_when_cold` exactly as it found it: the flag says a
        // residence's cold copy is disposable (a transient timeline), it is
        // latched — nothing in the tree ever clears it — and a dormant
        // conversation is not transient. Setting it here condemned ordinary KV
        // to be dropped rather than persisted, permanently, on the strength of
        // one quiet minute.
        for idx in [active, idle] {
            assert!(
                !sub.residence(idx).has_flag(EVICT_WHEN_COLD),
                "the idle demote must not latch evict_when_cold on a durable residence"
            );
        }
    }

    /// Within the grace window nothing is shed: an idle-for-a-moment residence
    /// is not the same as a dormant one, and shedding it would charge a
    /// warm→hot lift on a conversation's very next turn.
    #[test]
    fn idle_demote_respects_the_grace_window() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_i, _r) = install_hot_and_warm(&mut sub, timeline, 10);

        // Each pass ticks the clock. Inside the window nothing is shed.
        for pass in 1..=IDLE_DEMOTE_GRACE_EPOCHS {
            assert_eq!(
                sub.demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS),
                0,
                "pass {pass} is still inside the grace window"
            );
        }
        assert_eq!(
            sub.demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS),
            1,
            "the first pass beyond grace sheds it"
        );
    }

    /// A residence still owed a quantize keeps its hot copy. Its `hot` is the
    /// interim native form; letting the demote push that to disk would leave the
    /// cold copy diverging from the Q form the drain is about to install.
    #[test]
    fn idle_demote_skips_residences_pending_quantize() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_i, r) = install_hot_and_warm(&mut sub, timeline, 10);
        sub.residence(r).set_flag(PENDING_QUANTIZE);
        let mut shed = 0;
        for _ in 0..(IDLE_DEMOTE_GRACE_EPOCHS + 8) {
            shed += sub.demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS);
        }

        assert_eq!(
            shed, 0,
            "a pending-quantize residence is not the demote's to take"
        );
        assert!(sub.residence(r).has_hot());
    }

    /// **The demote is the retry for an offload that stood aside for a compaction.**
    /// `mark_timeline_evict_when_cold`'s immediate hot-drop defers while a pass holds
    /// chunk locations, leaving a flagged residence hot with a warm copy beside it.
    /// The next demote sheds it on the first pass, inside the grace window: the flag
    /// already says it is not wanted resident, so its age is not the question.
    #[test]
    fn a_deferred_offload_is_shed_by_the_next_demote_whatever_its_age() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_i, r) = install_hot_and_warm(&mut sub, timeline, 10);
        sub.residence(r).set_flag(EVICT_WHEN_COLD);

        assert_eq!(sub.demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS), 1);
        assert!(sub.residence(r).hot().is_none(), "the hot copy is shed");
        assert!(!sub.residences.hot_members().contains(&r));
        assert!(
            sub.residence(r).has_warm(),
            "warm stays until cold lands — it is the only durable-in-session copy"
        );
    }

    /// The same retry once the cold copy has landed: `install_cold` stood aside, so
    /// both resident tiers are still there, and the demote sheds both — what
    /// `install_cold` would have done.
    #[test]
    fn a_deferred_cold_offload_sheds_both_resident_tiers() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_i, r) = install_hot_and_warm(&mut sub, timeline, 10);
        sub.residence(r).set_flag(EVICT_WHEN_COLD);
        sub.residences.set_cold(
            r,
            Some(vec![StoredSequence {
                chunks: vec![StoredChunk {
                    log_offset: 0,
                    record_len: 1024,
                    token_count: 32,
                }],
                token_count: 32,
            }]),
        );

        assert_eq!(sub.demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS), 1);
        assert!(sub.residence(r).hot().is_none(), "VRAM reclaimed");
        assert!(sub.residence(r).warm().is_none(), "RAM reclaimed");
        assert!(
            !sub.residences.hot_members().contains(&r)
                && !sub.residences.warm_members().contains(&r)
        );
        assert!(sub.residence(r).has_cold(), "cold durable");
    }

    /// **The idle demote reaches SECTIONS, which the recency order of turns
    /// leaves out.**
    ///
    /// `elevate_to_hot` lifts sections and turns alike, and §8.1 of
    /// `docs/vram_governor_design.md` names "nothing demoted a section once it
    /// was elevated" as the gap this closes. A walk of the turns oldest-first
    /// reaches only turns — sections are deliberately not ranked against them —
    /// so it would miss the very case it was written for.
    #[test]
    fn idle_demote_reaches_sections_not_only_turns() {
        let mut sub = Substrate::new();
        let section = SectionId::new(42);
        let r = sub.set_section_full(
            section,
            StreamId::default(),
            10,
            vec![minimal_sealed_layer()],
            Arc::new(vec![1u32, 2, 3]),
        );
        // A warm copy must exist for the demote to have something to fall back
        // on — the same condition it applies to turns.
        sub.install_warm(r, vec![minimal_sealed_layer()]);
        assert!(sub.residence(r).has_hot());
        assert!(
            !sub.hot_turns_oldest_first().contains(&r),
            "sections are deliberately not in the turns' recency order — that is \
             what makes this test about the scan rather than about the order",
        );

        let mut shed = 0;
        for _ in 0..(IDLE_DEMOTE_GRACE_EPOCHS + 8) {
            shed += sub.demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS);
        }

        assert_eq!(shed, 1, "the idle section is shed");
        assert!(
            sub.residence(r).hot().is_none(),
            "an idle section must give its VRAM back like any other residence",
        );
        assert!(sub.residence(r).has_warm(), "warm survives the demote");
    }

    /// …but a section the current wave is attending is untouchable, exactly as
    /// for a turn.
    #[test]
    fn idle_demote_never_takes_a_pinned_section() {
        let mut sub = Substrate::new();
        let section = SectionId::new(7);
        let r = sub.set_section_full(
            section,
            StreamId::default(),
            10,
            vec![minimal_sealed_layer()],
            Arc::new(vec![1u32, 2, 3]),
        );
        sub.install_warm(r, vec![minimal_sealed_layer()]);

        let mut shed = 0;
        for _ in 0..(IDLE_DEMOTE_GRACE_EPOCHS + 8) {
            sub.set_working_set_pins(&[], &[section]);
            shed += sub.demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS);
        }

        assert_eq!(shed, 0);
        assert!(
            sub.residence(r).has_hot(),
            "a pinned section is part of the live projection and must stay hot",
        );
    }

    /// **A transient warm-only residence is reclaimable; an ordinary one is
    /// not.** The purge normally refuses to drop warm bytes that are a turn's
    /// last copy. `no_cold_persist` says the cold copy is never coming, so that
    /// refusal is permanent — measured as 8.8 GiB of warm host RAM that could
    /// not be shed during a full calibration.
    #[test]
    fn the_warm_purge_sheds_transient_warm_only_and_keeps_durable_warm_only() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_i, durable) = install_hot_and_warm(&mut sub, timeline, 4096);
        let (_j, transient) = install_hot_and_warm(&mut sub, timeline, 4096);
        // Both warm-only: no hot, no cold.
        for idx in [durable, transient] {
            sub.residences.take_hot(idx);
        }
        sub.residence(transient).set_flag(NO_COLD_PERSIST);

        // A budget of zero asks the purge to shed everything it safely can.
        let report = sub.purge_warm_to_budget(0, 0);

        assert!(
            sub.residence(durable).has_warm(),
            "a durable warm-only residence is its turn's last copy and must survive",
        );
        assert!(
            sub.residence(transient).warm().is_none(),
            "a transient warm-only residence has no cold copy coming, so holding \
             it is holding it forever",
        );
        assert_eq!(report.count, 1);
    }

    /// …but not one the current wave is attending. The copy rule was standing in
    /// for a pin check on this path; removing it for transients means the pin
    /// check has to be real.
    #[test]
    fn the_warm_purge_never_sheds_a_pinned_transient() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (idx, r) = install_hot_and_warm(&mut sub, timeline, 4096);
        sub.residences.take_hot(r);
        sub.residence(r).set_flag(NO_COLD_PERSIST);
        sub.set_working_set_pins(&[TurnKey::new(timeline, idx)], &[]);

        sub.purge_warm_to_budget(0, 0);

        assert!(
            sub.residence(r).has_warm(),
            "elevate stamps pins before it reads warm to build hot; the purge \
             must not pull those bytes out from under it",
        );
    }

    /// **Marking a timeline for distillation frees its K/V immediately.**
    ///
    /// The mark says the K/V is disposable — both [`DistillMode`]s drop KV
    /// chunks — so holding it until a compactor runs is holding garbage. On a
    /// full tool calibration that garbage reached 7.2 GiB of warm host RAM
    /// across 2,222 archived exemplars, and took the machine to 0.9 GiB free.
    #[test]
    fn distilling_a_timeline_releases_its_hot_and_warm_kv() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_i, a) = install_hot_and_warm(&mut sub, timeline, 10);
        let (_j, b) = install_hot_and_warm(&mut sub, timeline, 10);
        assert!(sub.residence(a).has_hot() && sub.residence(a).has_warm());

        sub.distill_timeline(timeline, DistillMode::ProvenanceOnly);

        for idx in [a, b] {
            assert!(
                sub.residence(idx).hot().is_none(),
                "a distilled turn must not keep hot K/V",
            );
            assert!(
                sub.residence(idx).warm().is_none(),
                "a distilled turn must not keep warm K/V — this is the 7.2 GiB",
            );
            assert!(!sub.residences.hot_members().contains(&idx));
            assert!(!sub.residences.warm_members().contains(&idx));
        }
        // The mark itself still stands: the compactor sheds the record and the
        // cold copy, and the signature keeps answering the belief scan.
        assert_eq!(
            sub.distill_mode(timeline),
            Some(DistillMode::ProvenanceOnly)
        );
    }

    /// `TextOnly` drops KV chunks too, so it releases as well — the two modes
    /// differ in what they KEEP beside the K/V, not in whether the K/V goes.
    #[test]
    fn text_only_distillation_also_releases_kv() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_i, r) = install_hot_and_warm(&mut sub, timeline, 10);
        sub.distill_timeline(timeline, DistillMode::TextOnly);
        assert!(sub.residence(r).hot().is_none());
        assert!(sub.residence(r).warm().is_none());
    }

    /// The release honours the working set, like every other drop path: a
    /// residence the current wave is attending is left alone even when its
    /// timeline is marked. Distillation is a statement about the corpus, not a
    /// licence to pull K/V out from under a live forward.
    #[test]
    fn distillation_never_takes_the_current_working_set() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (active_idx, active) = install_hot_and_warm(&mut sub, timeline, 10);
        let (_j, idle) = install_hot_and_warm(&mut sub, timeline, 10);
        sub.set_working_set_pins(&[TurnKey::new(timeline, active_idx)], &[]);

        sub.distill_timeline(timeline, DistillMode::ProvenanceOnly);

        assert!(
            sub.residence(active).has_hot(),
            "a pinned residence keeps its K/V through a distill mark",
        );
        assert!(sub.residence(idle).hot().is_none());
        assert!(sub.residence(idle).warm().is_none());
    }

    /// A residence still owed a quantize keeps its K/V: its `hot` is the interim
    /// native form the drain is about to replace, and the compactor reclaims
    /// whatever this pass skips exactly as it did before.
    #[test]
    fn distillation_skips_residences_pending_quantize() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_i, r) = install_hot_and_warm(&mut sub, timeline, 10);
        sub.residence(r).set_flag(PENDING_QUANTIZE);
        sub.distill_timeline(timeline, DistillMode::ProvenanceOnly);
        assert!(sub.residence(r).has_hot());
        assert!(sub.residence(r).has_warm());
    }

    /// **Hot is only dropped where warm already holds the bytes.** The demote
    /// sheds a copy; it must never be the thing that loses the only one. With no
    /// warm tier the drop would leave the section reachable solely through the
    /// cold log — and, for a residence whose cold copy has not landed yet, not
    /// reachable at all.
    #[test]
    fn idle_demote_never_takes_the_last_copy() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_i, r) = install_hot_and_warm(&mut sub, timeline, 10);
        sub.residences.take_warm(r);

        let mut shed = 0;
        for _ in 0..(IDLE_DEMOTE_GRACE_EPOCHS + 8) {
            shed += sub.demote_idle_hot(IDLE_DEMOTE_GRACE_EPOCHS);
        }

        assert_eq!(shed, 0, "a hot-only residence is not the demote's to take");
        assert!(sub.residence(r).has_hot());
    }

    /// A stamp that runs ahead of the scan's epoch reads as "just used", never
    /// as a huge age. The scan and the publish are on different threads, so an
    /// underflow here would evict the live working set.
    #[test]
    fn idle_for_never_underflows_into_a_huge_age() {
        assert_eq!(
            idle_for(10, 4),
            0,
            "a stamp ahead of the epoch is not stale"
        );
        assert_eq!(idle_for(4, 10), 6);
        assert_eq!(idle_for(0, 1), 1, "never attended is stale from birth");
    }

    /// `set_working_set_pins` replaces the keep-set wholesale — a residence that
    /// leaves the working set drops its pin and becomes evictable again.
    #[test]
    fn set_working_set_pins_replaces_wholesale() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (a_idx, a) = install_hot_and_warm(&mut sub, timeline, 10);
        let (b_idx, _b) = install_hot_and_warm(&mut sub, timeline, 10);
        sub.residence(a).set_flag(EVICT_WHEN_COLD);

        // Wave 1 pins A; wave 2 pins only B → A is no longer protected.
        sub.set_working_set_pins(&[TurnKey::new(timeline, a_idx)], &[]);
        sub.set_working_set_pins(&[TurnKey::new(timeline, b_idx)], &[]);

        let cold = vec![StoredSequence {
            chunks: vec![StoredChunk {
                log_offset: 0,
                record_len: 1024,
                token_count: 32,
            }],
            token_count: 32,
        }];
        sub.install_cold(a, cold);
        assert!(
            sub.residence(a).hot().is_none(),
            "A left the working set → eviction resumes"
        );
    }

    /// **The pinned hot KV is told by its owning timeline.** Two timelines'
    /// turns pinned and one left out: each pinned timeline is reported with
    /// its pinned turns' bytes, largest first, the unpinned one not at all,
    /// and the cut at `top` keeps the largest.
    #[test]
    fn pinned_hot_kv_is_reported_by_its_timeline() {
        let (layer, group, a, mut sub) = make_timeline();
        let alloc = TimelineAllocator::new();
        let b = alloc.next();
        sub.register_timeline(b, layer, group);
        let (a1, _) = install_hot_and_warm(&mut sub, a, 10);
        let (a2, _) = install_hot_and_warm(&mut sub, a, 20);
        install_hot_and_warm(&mut sub, a, 40);
        let (b1, _) = install_hot_and_warm(&mut sub, b, 100);
        sub.set_working_set_pins(
            &[
                TurnKey::new(a, a1),
                TurnKey::new(a, a2),
                TurnKey::new(b, b1),
            ],
            &[],
        );

        let pinned = sub.pinned_hot_by_timeline(8);
        assert_eq!(pinned.pins, 3);
        assert_eq!(pinned.pinned_bytes, 130);
        assert_eq!(pinned.turn_bytes, 130);
        assert_eq!(pinned.by_timeline, vec![(b, 1, 100), (a, 2, 30)]);
        assert_eq!(sub.pinned_hot_by_timeline(1).by_timeline, vec![(b, 1, 100)]);
    }

    /// A sealed turn spanning REAL blocks, hot + warm installed — the shape a
    /// live ingest unit has once its seal lands. Unlike `install_hot_and_warm`
    /// this sets `block_end`, which is what makes the turn one that *should*
    /// hold K/V: a zero-block turn holds none by construction and intactness
    /// rightly ignores it.
    pub(in super::super) fn install_blocked_turn(
        sub: &mut Substrate,
        timeline: TimelineId,
        blocks: u64,
    ) -> ResidenceIndex {
        let idx = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    sealed_gpu: Some(Arc::new(vec![minimal_sealed_layer()])),
                    block_end: blocks,
                    token_count: 32,
                    ..Default::default()
                },
                identity_migrate,
            )
            .unwrap();
        let residence = sub.turn_residence(timeline, idx).unwrap();
        sub.install_warm(residence, vec![minimal_sealed_layer()]);
        residence
    }

    /// Drop every tier's copy — the pin-wedge end state, and also what
    /// `release_distilled_kv` leaves behind: a turn with token_ids and a block
    /// range and no chunks anywhere.
    pub(in super::super) fn reclaim_all_tiers(sub: &mut Substrate, residence: ResidenceIndex) {
        sub.residences.take_hot(residence);
        sub.residences.take_warm(residence);
        sub.residences.set_cold(residence, None);
    }

    /// The resume cache is a claim about a stored ANSWER, not just about the
    /// input hash: a live conversation that lost a turn's K/V must stop matching
    /// so the unit re-ingests, or its hash keeps matching forever and the hole
    /// is never repaired.
    #[test]
    fn a_damaged_conversation_leaves_the_resume_cache() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let intact = alloc.next();
        let damaged = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(intact, layer, group);
        sub.register_timeline(damaged, layer, group);
        install_blocked_turn(&mut sub, intact, 3);
        let broken = install_blocked_turn(&mut sub, damaged, 3);
        let hash_meta = |v: &str| BTreeMap::from([("content_sha256".to_string(), v.to_string())]);
        sub.merge_custom(intact, &hash_meta("aaa"), RecordIdentity::NONE);
        sub.merge_custom(damaged, &hash_meta("bbb"), RecordIdentity::NONE);

        assert!(sub.timeline_kv_intact(intact));
        assert!(sub.timeline_kv_intact(damaged));
        assert_eq!(
            sub.metadata_values_for_key("content_sha256"),
            ["aaa".to_string(), "bbb".to_string()].into(),
        );

        reclaim_all_tiers(&mut sub, broken);
        assert!(
            !sub.timeline_kv_intact(damaged),
            "a chunkless live turn is damage"
        );
        assert_eq!(
            sub.metadata_values_for_key("content_sha256"),
            ["aaa".to_string()].into(),
            "the damaged unit's hash must stop matching so it re-ingests",
        );
    }

    /// **Deliberately reclaimed K/V is not damage.** A tool-calibration exemplar
    /// ends archived + distilled(`ProvenanceOnly`) with its chunks dropped on
    /// purpose, and must keep matching the resume cache — re-ingesting the
    /// calibration corpus because its K/V is legitimately gone would be a far
    /// worse bug than the one this predicate exists to catch.
    #[test]
    fn deliberately_reclaimed_kv_still_counts_as_intact() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        for (archived, distilled) in [(true, false), (false, true), (true, true)] {
            let tl = alloc.next();
            let mut sub = Substrate::new();
            sub.register_timeline(tl, layer, group);
            let r = install_blocked_turn(&mut sub, tl, 3);
            // Content reclaimed, exactly as `release_distilled_kv` leaves it.
            reclaim_all_tiers(&mut sub, r);
            assert!(
                !sub.timeline_kv_intact(tl),
                "guard precondition: this turn IS chunkless, so the archived / \
                 distilled exemption below is what has to carry it",
            );
            if archived {
                assert!(sub.set_archived(tl, true));
            }
            if distilled {
                sub.distill_timeline(tl, DistillMode::ProvenanceOnly);
            }
            assert!(
                sub.timeline_kv_intact(tl),
                "archived={archived} distilled={distilled}: reclaimed content is not damage",
            );
        }
    }

    /// Intactness is scoped to a timeline's OWN turns. A fork whose ancestor is
    /// damaged is itself fine: ancestors resolve live through
    /// `inherited_chain`, so the hole is repaired once, in the ancestor, for
    /// every descendant. Walking the chain here would condemn the whole corpus
    /// — every `code_read` file hangs off the priming chain's end.
    #[test]
    fn a_damaged_ancestor_does_not_dirty_its_descendants() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let ancestor = alloc.next();
        let child = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(ancestor, layer, group);
        sub.register_timeline(child, layer, group);
        let broken = install_blocked_turn(&mut sub, ancestor, 3);
        install_blocked_turn(&mut sub, child, 3);
        sub.merge_custom(
            child,
            &BTreeMap::from([(
                Substrate::FORKED_FROM_KEY.to_string(),
                ancestor.raw().to_string(),
            )]),
            RecordIdentity::NONE,
        );
        assert_eq!(
            sub.parent_timeline(child),
            Some(ancestor),
            "the fork pointer is what a descendant resolves through",
        );

        reclaim_all_tiers(&mut sub, broken);

        assert!(!sub.timeline_kv_intact(ancestor), "the ancestor is damaged");
        assert!(
            sub.timeline_kv_intact(child),
            "the child owns intact turns — an ancestor's hole is not its own",
        );
    }

    /// **A pinned warm turn is written cold like any other, and keeps its
    /// resident copies.** A pin governs eviction, never durability: a
    /// conversation pins its own turns for as long as it is active, and a
    /// write deferred by the pin left a killed member with those turns' tokens
    /// and no K/V — empty to the projection, never resumable (BlackWorld,
    /// 2026-11-21; earlier, every `repo_map` unit's first turn under a pin
    /// that outlived its slot).
    #[test]
    fn a_pinned_warm_turn_is_written_cold_and_keeps_its_copies() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (pinned_idx, pinned) = install_hot_and_warm(&mut sub, timeline, 10);
        let (_, other) = install_hot_and_warm(&mut sub, timeline, 10);
        sub.set_working_set_pins(&[TurnKey::new(timeline, pinned_idx)], &[]);

        let pending: Vec<ResidenceIndex> = sub
            .snapshot_pending_cold()
            .into_iter()
            .map(|(idx, _, _)| idx)
            .collect();
        assert!(
            pending.contains(&pinned),
            "a pinned turn is gathered warm→cold"
        );
        assert!(pending.contains(&other));
        assert_eq!(
            sub.pending_cold_count(),
            2,
            "the count follows the same filter"
        );

        // The cold copy lands under the pin, with the evict-when-cold flag
        // set: the hot and warm copies the live slot reads stay.
        sub.residence(pinned).set_flag(EVICT_WHEN_COLD);
        sub.install_cold(
            pinned,
            vec![StoredSequence {
                chunks: Vec::new(),
                token_count: 0,
            }],
        );
        let r = sub.residence(pinned);
        assert!(r.has_cold() && r.has_hot() && r.has_warm());
        assert_eq!(sub.pending_cold_count(), 1, "written once");
    }
}
