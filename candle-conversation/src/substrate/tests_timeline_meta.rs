//! Tests lineage, base repair, turn resolution and registry bookkeeping for
//! timelines.
//!
//! A parent named by content key resolves to the surviving live copy, so members
//! that minted the same link still agree on its parent. A target's group resolves
//! to the target's own conversation, which keeps a second conversation's turns
//! from being looked up under the first. Re-registering a timeline keeps its
//! turns, because replay registers each declaration in turn and a wipe would lose
//! history. A conversation's own token total excludes the shared sections, which
//! would otherwise be counted once per conversation.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::projection::{GroupId, LayerId, ProjectionTarget, TimelineAllocator, TimelineId};

    use super::super::tests_working_set::tests::{
        forked_line, make_timeline, set_parent, unshared_timeline,
    };

    /// **A parent named by content key resolves to whichever copy survives.**
    /// Two members read the same changed link at once and each holds a copy;
    /// the next link names it by key, so it resolves to the oldest live copy,
    /// then to the other once that one is retired — never to an archived one,
    /// and to nothing when no copy is left.
    #[test]
    fn a_parent_named_by_key_resolves_to_the_live_copy() {
        let alloc = TimelineAllocator::new();
        let (layer, group) = (LayerId::for_test(1), GroupId::for_test(1));
        let [first, second, child] = std::array::from_fn(|_| alloc.next());
        let mut sub = Substrate::new();
        for tl in [first, second, child] {
            sub.register_timeline(tl, layer, group);
            sub.append_with_blocks(tl, 10, 0, 1);
        }
        let key = "v5:journals/@listing";
        let kv = |k: &str, v: &str| BTreeMap::from([(k.to_string(), v.to_string())]);
        for copy in [first, second] {
            sub.merge_custom(copy, &kv(Substrate::CONTENT_KEY, key), RecordIdentity::NONE);
        }
        sub.merge_custom(
            child,
            &kv(Substrate::FORKED_FROM_KEY, &Substrate::unit_parent(key)),
            RecordIdentity::NONE,
        );
        assert_eq!(sub.parent_timeline(child), Some(first), "the oldest copy");
        sub.tombstoned_timelines.insert(first);
        assert_eq!(
            sub.parent_timeline(child),
            Some(second),
            "the surviving copy"
        );
        arc_mut(&mut sub.timelines, &second).unwrap().archived = true;
        assert_eq!(sub.parent_timeline(child), None, "no live copy is left");
    }

    /// **The repair finds every conversation parented on a base** — a
    /// timeline of the base's layer and group with no turn and no `conv_id`
    /// — and none parented on the seed, on a unit, or on a named
    /// conversation.
    #[test]
    fn the_children_of_every_base_are_found() {
        let (mut sub, chain) = forked_line(1, 10);
        let alloc = TimelineAllocator::new();
        let (layer, group) = (LayerId::for_test(1), GroupId::for_test(1));
        let [old_base, base, named, a, b, on_seed, on_unit, on_named] =
            std::array::from_fn(|_| alloc.next());
        for tl in [old_base, base, named] {
            sub.register_timeline(tl, layer, group);
        }
        sub.set_conv_id(named, "a-conversation", RecordIdentity::NONE);
        for (child, parent) in [
            (a, old_base),
            (b, base),
            (on_unit, chain[0]),
            (on_named, named),
        ] {
            sub.register_timeline(child, layer, group);
            sub.append_with_blocks(child, 10, 0, 1);
            set_parent(&mut sub, child, parent);
        }
        sub.register_timeline(on_seed, layer, group);
        sub.merge_custom(
            on_seed,
            &BTreeMap::from([(
                Substrate::FORKED_FROM_KEY.to_string(),
                Substrate::SEED_PARENT.to_string(),
            )]),
            RecordIdentity::NONE,
        );
        let mut found = sub.parented_on_a_base(base);
        found.sort();
        let mut want = vec![a, b];
        want.sort();
        assert_eq!(found, want);
    }

    #[test]
    fn a_parent_pointer_naming_an_unknown_timeline_is_ignored() {
        let (_layer, _group, tl, mut sub) = make_timeline();
        let mut kv = BTreeMap::new();
        kv.insert(Substrate::FORKED_FROM_KEY.to_string(), "999999".to_string());
        sub.merge_custom(tl, &kv, RecordIdentity::NONE);
        assert_eq!(sub.parent_timeline(tl), None);
        assert_eq!(sub.inherited_chain(tl), vec![tl]);
    }

    #[test]
    fn an_unparseable_parent_pointer_is_ignored() {
        let (_layer, _group, tl, mut sub) = make_timeline();
        let mut kv = BTreeMap::new();
        kv.insert(
            Substrate::FORKED_FROM_KEY.to_string(),
            "not-a-timeline".to_string(),
        );
        sub.merge_custom(tl, &kv, RecordIdentity::NONE);
        assert_eq!(sub.parent_timeline(tl), None);
    }

    /// Regression: two conversations (timelines) in the SAME group is the case
    /// that lost history on reproject.  `timelines_for_group(group).next()`
    /// returns the FIRST-registered timeline, so a turn in the SECOND
    /// conversation was looked up under the first and dropped (`turn_tier_state`
    /// `None` → the slot rebuilt with `turns=0`).  `resolve_turn_timeline` pins
    /// the target group to the conversation actually being projected, so the
    /// second conversation's turns survive.  Every turn-timeline call site routes
    /// through it, so the SubmitTurn / reproject / inject paths can't disagree.
    #[test]
    fn resolve_turn_timeline_pins_target_group_to_its_own_timeline() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let first = alloc.next();
        let second = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(first, layer, group);
        sub.register_timeline(second, layer, group);

        // A turn lives ONLY in the second conversation.
        let idx = sub.append_with_blocks(second, 10, 0, 1);
        let target = |tl| ProjectionTarget {
            layer,
            group,
            timeline: tl,
        };

        // Registration order makes `.next()` resolve to `first` — the old bug.
        assert_eq!(sub.timelines_for_group(group).next(), Some(first));

        // The fix: a target pinned to the second conversation resolves to it, not
        // the first-registered timeline.
        assert_eq!(
            sub.resolve_turn_timeline(Some(target(second)), group),
            Some(second)
        );
        assert_eq!(
            sub.resolve_turn_timeline(Some(target(first)), group),
            Some(first)
        );

        // And that's what keeps the turn: under the correctly-resolved timeline
        // the second conversation's turn is found (kept); under the timeline the
        // old `.next()` picked it does not exist (dropped — the history loss).
        let resolved = sub
            .resolve_turn_timeline(Some(target(second)), group)
            .unwrap();
        assert!(
            sub.turn_tier_state(resolved, idx).is_some(),
            "turn is kept under the correctly-resolved timeline"
        );
        assert!(
            sub.turn_tier_state(first, idx).is_none(),
            "the old `.next()` timeline drops the second conversation's turn"
        );

        // A non-target group has a single timeline, so it still falls back to
        // registration order (the target only pins the target's own group).
        let other_group = GroupId::for_test(2);
        let other_tl = alloc.next();
        sub.register_timeline(other_tl, layer, other_group);
        assert_eq!(
            sub.resolve_turn_timeline(Some(target(second)), other_group),
            Some(other_tl)
        );

        // No target (utility pass) falls back to registration order everywhere.
        assert_eq!(sub.resolve_turn_timeline(None, group), Some(first));
    }

    // ── Phase 1: TreeNodeMeta + debug_id substrate APIs ──────────────────

    #[test]
    fn appended_turn_gets_default_tree_meta_as_normal() {
        let (_, _, timeline, mut sub) = make_timeline();
        let idx = sub.append_with_blocks(timeline, 10, 0, 1);
        let meta = sub.tree_meta_of(timeline, idx).expect("meta present");
        assert_eq!(meta.kind, TurnKind::Normal);
        assert!(meta.children.is_empty());
        assert_eq!(meta.tree_height, 0);
    }

    /// A timeline's turn count is its own turns, and an unknown timeline holds
    /// none.
    #[test]
    fn a_turn_count_is_the_timelines_own() {
        let (_, _, timeline, mut sub) = make_timeline();
        assert_eq!(sub.turn_count_of(timeline), 0);
        sub.append_with_blocks(timeline, 10, 0, 1);
        sub.append_with_blocks(timeline, 10, 1, 2);
        assert_eq!(sub.turn_count_of(timeline), 2);
        let unknown = TimelineId::from_raw(timeline.raw() + 1000).unwrap();
        assert_eq!(sub.turn_count_of(unknown), 0);
    }

    #[test]
    fn appended_turn_pushed_onto_pending_summary_queue() {
        let (_, _, timeline, mut sub) = make_timeline();
        let idx_a = sub.append_with_blocks(timeline, 10, 0, 1);
        let idx_b = sub.append_with_blocks(timeline, 10, 1, 2);
        assert_eq!(sub.pending_summary_len(timeline), 2);
        assert_eq!(sub.pop_pending_summary(timeline), Some(idx_a));
        assert_eq!(sub.pop_pending_summary(timeline), Some(idx_b));
        assert_eq!(sub.pop_pending_summary(timeline), None);
    }

    #[test]
    fn set_tree_meta_overwrites_in_place() {
        let (_, _, timeline, mut sub) = make_timeline();
        let idx = sub.append_with_blocks(timeline, 10, 0, 1);
        let meta = TreeNodeMeta {
            kind: TurnKind::SummaryOfTurns,
            children: vec![TurnIndex(99)],
            tree_height: 1,
        };
        sub.set_tree_meta(timeline, idx, meta.clone(), RecordIdentity::NONE);
        assert_eq!(sub.tree_meta_of(timeline, idx), Some(&meta));
        let leaves = sub.summary_leaves_chrono(timeline);
        assert_eq!(leaves, vec![idx]);
    }

    /// **A rewrite carries the tree nodes a record set, and only those.** A
    /// freshly appended turn's default node is implied by the turn, so it is not
    /// re-emitted — written out, it would be a record with no identity, and the
    /// store would read as predating identities at its next open. A node the
    /// summariser set goes out with the identity it was set under.
    #[test]
    fn only_recorded_tree_nodes_are_re_emitted() {
        let (_, _, timeline, mut sub) = make_timeline();
        let plain = sub.append_with_blocks(timeline, 10, 0, 1);
        let leaf = sub.append_with_blocks(timeline, 10, 1, 2);
        assert!(
            sub.tree_meta_of(timeline, plain).is_some(),
            "every turn has a node"
        );
        assert!(sub.live_tree_metadata_payloads().is_empty());

        let id = RecordIdentity {
            origin: 7,
            oseq: 3,
            gen: 0,
            epoch: 0,
            hlc: 3 << 16,
        };
        let meta = TreeNodeMeta {
            kind: TurnKind::SummaryOfTurns,
            children: vec![plain],
            tree_height: 1,
        };
        sub.set_tree_meta(timeline, leaf, meta, id);
        let emitted = sub.live_tree_metadata_payloads();
        assert_eq!(emitted.len(), 1);
        let (payload, got) = &emitted[0];
        assert_eq!((payload.turn_index, payload.kind, got), (leaf.0, 1, &id));
        assert_eq!(payload.children, vec![plain.0]);
    }

    /// Three SoT leaves carry into one ternary SoS; the SoS is the sole peak.
    #[test]
    fn peaks_of_derives_forest() {
        let (_, _, timeline, mut sub) = make_timeline();
        let mut leaves = Vec::new();
        for n in 0..MERGE_FANOUT as u32 {
            let idx = sub.append_with_blocks(timeline, 10, n as u64, n as u64 + 1);
            sub.set_tree_meta(
                timeline,
                idx,
                TreeNodeMeta {
                    kind: TurnKind::SummaryOfTurns,
                    children: vec![idx],
                    tree_height: 1,
                },
                RecordIdentity::NONE,
            );
            leaves.push(idx);
        }
        // Before the SoS exists, all MERGE_FANOUT leaves are peaks; reconcile wants
        // to build the SoS over them.
        assert_eq!(sub.peaks_of(timeline).len(), MERGE_FANOUT);
        let next = sub.reconcile_next(timeline).expect("an SoS to build");
        assert_eq!(next, leaves);
        // Record the SoS; now it is the single peak and the forest is whole.
        let sos =
            sub.append_with_blocks(timeline, 10, MERGE_FANOUT as u64, MERGE_FANOUT as u64 + 1);
        sub.set_tree_meta(
            timeline,
            sos,
            TreeNodeMeta {
                kind: TurnKind::SummaryOfSummaries,
                children: leaves.clone(),
                tree_height: 2,
            },
            RecordIdentity::NONE,
        );
        assert_eq!(sub.peaks_of(timeline), vec![(sos, 2)]);
        assert_eq!(sub.reconcile_next(timeline), None);
    }

    #[test]
    fn debug_id_round_trip_and_lookup() {
        let (_, _, timeline, mut sub) = make_timeline();
        assert_eq!(sub.lookup_by_debug_id("foo"), None);
        sub.set_debug_id(timeline, "coherent-50", RecordIdentity::NONE);
        assert_eq!(sub.lookup_by_debug_id("coherent-50"), Some(timeline));
        assert_eq!(sub.debug_id_of(timeline), Some("coherent-50"));
        // Replacing supersedes — old key disappears.
        sub.set_debug_id(timeline, "two-topics-100", RecordIdentity::NONE);
        assert_eq!(sub.lookup_by_debug_id("coherent-50"), None);
        assert_eq!(sub.lookup_by_debug_id("two-topics-100"), Some(timeline));
    }

    /// `mark_for_reconcile` purges non-canonical (old binary) SoS meta and arms
    /// the reconcile hint so the ternary nodes get rebuilt.
    #[test]
    fn mark_for_reconcile_purges_noncanonical_sos() {
        let (_, _, timeline, mut sub) = make_timeline();
        let leaf_a = sub.append_with_blocks(timeline, 10, 0, 1);
        let leaf_b = sub.append_with_blocks(timeline, 10, 1, 2);
        for leaf in [leaf_a, leaf_b] {
            sub.set_tree_meta(
                timeline,
                leaf,
                TreeNodeMeta {
                    kind: TurnKind::SummaryOfTurns,
                    children: vec![leaf],
                    tree_height: 1,
                },
                RecordIdentity::NONE,
            );
        }
        // A legacy binary SoS (2 children) — not canonical under MERGE_FANOUT.
        let bin = sub.append_with_blocks(timeline, 10, 2, 3);
        sub.set_tree_meta(
            timeline,
            bin,
            TreeNodeMeta {
                kind: TurnKind::SummaryOfSummaries,
                children: vec![leaf_a, leaf_b],
                tree_height: 2,
            },
            RecordIdentity::NONE,
        );
        assert!(sub.tree_meta_of(timeline, bin).is_some());
        sub.mark_for_reconcile(timeline);
        // Binary SoS purged; reconcile armed; the two leaves survive.
        assert!(sub.tree_meta_of(timeline, bin).is_none());
        assert!(sub.needs_reconcile(timeline));
        assert_eq!(sub.peaks_of(timeline).len(), 2);
    }

    /// Identity migration: returns the input unchanged (simulates CPU == GPU in tests).
    pub(in super::super) fn identity_migrate(
        seqs: &[SealedSequence],
    ) -> candle::Result<Vec<SealedSequence>> {
        Ok(seqs.to_vec())
    }

    /// Structurally-minimal `SealedSequence` suitable for migration-
    /// closure tests that don't care about chunk contents. The
    /// substrate's residence model treats a `Vec<SealedSequence>`
    /// containing zero elements as cold-marker (`hot = None`), so
    /// tests that want `turn_sealed_of` / `section_sealed_of` to
    /// return `Some` must pass at least one element. The empty
    /// `chunks` field is fine — it just means "this layer's sequence
    /// has no chunks", which is structurally valid.
    pub(in super::super) fn minimal_sealed_layer() -> SealedSequence {
        SealedSequence {
            chunks: Vec::new(),
            token_count: 0,
            chunk_size: 32,
            location: candle_nn::kv_cache::ArenaLocation::Cpu,
        }
    }

    /// **The hot→warm groups are named without cloning, and each is cloned on
    /// its own.** The persistence thread snapshots one group under its migrate
    /// guard, so the per-policy snapshot must return exactly that policy's
    /// pending residences — and skip what the whole snapshot skips: a
    /// residence already warm, and a persisted one pinned into the wave's
    /// working set.
    #[test]
    fn pending_warm_is_named_by_policy_and_snapshotted_per_group() {
        let lossless = ConvCompression {
            lossless: true,
            level: None,
            disable_k_override: false,
            force_k: None,
            force_v: None,
        };
        let sub = Substrate::new();
        let plain = sub.alloc_residence(StreamId(1), None);
        let exact = sub.alloc_residence(StreamId(2), Some(lossless));
        let warm = sub.alloc_residence(StreamId(3), None);
        let pinned = sub.alloc_residence(StreamId(4), Some(lossless));
        for r in [plain, exact, warm, pinned] {
            sub.install_hot(r, vec![minimal_sealed_layer()]);
        }
        sub.install_warm(warm, vec![minimal_sealed_layer()]);
        sub.residence(pinned).set_flag(COLD_PENDING);
        sub.residences
            .publish_pins(SCHEDULER_PINS, [pinned.0].into_iter().collect());

        let mut policies = sub.pending_warm_policies();
        policies.sort_by_key(|p| p.is_some());
        assert_eq!(policies, [None, Some(lossless)]);

        let of = |p| -> Vec<usize> {
            sub.snapshot_pending_warm_for(p)
                .into_iter()
                .map(|(idx, _)| idx.0)
                .collect()
        };
        assert_eq!(of(None), [plain.0]);
        assert_eq!(of(Some(lossless)), [exact.0]);
    }

    /// A one-chunk layer whose bands are all `tag`, with `tokens` tokens.
    pub(in super::super) fn layer_tagged(tag: u8, tokens: u16) -> SealedSequence {
        use candle_nn::kv_cache::{ArenaLocation, ChunkGid, HeadGids, SealedChunk, N_PALETTE};
        SealedSequence {
            chunks: vec![SealedChunk {
                gids: HeadGids::uniform(ChunkGid::detached(0), 1),
                offset: 0,
                token_count: tokens,
                k_pal: Arc::new(Vec::new()),
                v_pal: Arc::new(Vec::new()),
                k_scale: Arc::new(Vec::new()),
                v_scale: Arc::new(Vec::new()),
                k_fmt: Arc::new(vec![tag; N_PALETTE]),
                v_fmt: Arc::new(vec![tag; N_PALETTE]),
                byte_size: 0,
                meta: None,
                q_sign: None,
            }],
            token_count: tokens as usize,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }
    }

    /// **The gap is named the oldest pending turn still holding native
    /// chunks** — never a quantized one, an empty one, a warm one, or a
    /// lossless conversation's, whose turns stay native by design.
    #[test]
    fn a_native_pending_turn_is_named_for_the_gap_and_nothing_else_is() {
        use candle::DType;
        use candle_nn::kv_cache::{KvFormat, QuantFormat};
        let f16 = KvFormat::Float(DType::F16).to_tag();
        let q8 = KvFormat::Quantized(QuantFormat::Q8_0).to_tag();
        let lossless = ConvCompression {
            lossless: true,
            level: None,
            disable_k_override: false,
            force_k: None,
            force_v: None,
        };
        let sub = Substrate::new();
        let quantized = sub.alloc_residence(StreamId(1), None);
        let empty = sub.alloc_residence(StreamId(2), None);
        let kept = sub.alloc_residence(StreamId(3), Some(lossless));
        let warm = sub.alloc_residence(StreamId(4), None);
        let native = sub.alloc_residence(StreamId(5), None);
        sub.install_hot(quantized, vec![layer_tagged(q8, 32)]);
        sub.install_hot(empty, vec![layer_tagged(f16, 0)]);
        sub.install_hot(kept, vec![layer_tagged(f16, 32)]);
        sub.install_hot(warm, vec![layer_tagged(f16, 32)]);
        sub.install_warm(warm, vec![minimal_sealed_layer()]);
        sub.install_hot(native, vec![layer_tagged(f16, 16)]);
        let (idx, hot, cc) = sub.native_pending_warm().expect("the native turn");
        assert_eq!((idx.0, cc), (native.0, None));

        // Swapped in over the copy it read, it is native no longer.
        assert!(sub.install_gap_quantized(idx, &hot, vec![layer_tagged(q8, 16)]));
        assert!(sub.native_pending_warm().is_none());
        // A second install over the same read finds the content changed.
        assert!(!sub.install_gap_quantized(idx, &hot, vec![layer_tagged(q8, 16)]));
    }

    /// **A conversation's own tokens exclude the shared sections, and the corpus counts
    /// them once.** `total_token_count` adds the sections to every conversation — right
    /// for one conversation's "materialized / N" — so a sum of it across a layer counted
    /// the tool catalog once per conversation: 37 folder units read as 1.5 M tokens.
    #[test]
    fn a_conversations_tokens_exclude_the_sections_the_corpus_counts_once() {
        let (layer, group, first, mut sub) = make_timeline();
        let second = unshared_timeline(1);
        let retired = unshared_timeline(2);
        sub.register_timeline(second, layer, group);
        sub.register_timeline(retired, layer, group);
        sub.append_with_blocks(first, 100, 0, 4);
        sub.append_with_blocks(second, 30, 0, 1);
        sub.append_with_blocks(retired, 7, 0, 1);
        sub.tombstone_timeline(retired);
        sub.section_token_total = 40;

        assert_eq!(sub.total_token_count(first), 140);
        assert_eq!(sub.timeline_token_total(first), 100);
        assert_eq!(sub.timeline_token_total(second), 30);
        assert_eq!(sub.corpus_token_total(), 100 + 30 + 40);
    }

    /// `register_timeline` is data-idempotent — calling it again on a
    /// timeline that already has turns must NOT wipe those turns. This
    /// is the regression test for a real bug where substrate replay
    /// (`reconstruct_from_log`) destroyed every turn but the last,
    /// because each per-decl `register_timeline` call clobbered the
    /// previous `TimelineEntry`.
    #[test]
    fn re_registering_a_timeline_preserves_its_turns() {
        let (layer, group, timeline, mut sub) = make_timeline();

        // Land two turns under this timeline.
        sub.append_with_blocks(timeline, 3, 0, 1);
        sub.append_with_blocks(timeline, 5, 1, 2);
        assert_eq!(sub.turn_count(timeline), 2);

        // Re-register the same timeline. Used to wipe the turns map.
        sub.register_timeline(timeline, layer, group);
        assert_eq!(
            sub.turn_count(timeline),
            2,
            "re-registering a timeline must preserve its existing turns",
        );

        // Inverse index must still have exactly one entry — not two —
        // confirming we didn't double-push on re-registration.
        let listed: Vec<_> = sub.timelines_for_group(group).collect();
        assert_eq!(listed, vec![timeline]);
    }
}
