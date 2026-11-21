//! Tests the working-set, retrieval-scope and inherited-chain rules that decide
//! which conversations a dialogue may draw on.
//!
//! A pinned item is a promise to the model, so a tombstoned, archived, empty or
//! incomplete conversation reads as a miss and the call goes to a real read. An
//! inherited chain runs oldest-first and is capped by dropping its oldest
//! ancestors, so the nearest survive. A cycle in the lineage truncates the walk
//! rather than hanging the projection. The seed alias resolves to the chain this
//! member holds now, so a rebuilt chain takes effect without rewriting any
//! conversation.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::projection::{GroupId, LayerId, TimelineAllocator, TimelineId};

    /// Blob writes for tests about what the substrate does with the bytes,
    /// not about which of two records wins: every write carries no identity,
    /// so the last one holds.
    impl Substrate {
        pub(in super::super) fn set_wide_q_sigs_blob(
            &mut self,
            stream_id: StreamId,
            payload: Vec<u8>,
        ) {
            self.apply_wide_q_sigs(stream_id, payload, RecordIdentity::NONE);
        }

        pub(in super::super) fn set_projection_events_blob(
            &mut self,
            stream_id: StreamId,
            payload: Vec<u8>,
        ) {
            self.apply_projection_events(stream_id, payload, RecordIdentity::NONE);
        }
    }

    /// A timeline id no allocator in a test can also mint. Allocators are
    /// microsecond clocks, so two fresh ones asked in the same microsecond
    /// return the same id — two "different" timelines then share their turns.
    pub(in super::super) fn unshared_timeline(n: u64) -> TimelineId {
        TimelineId::from_raw((1 << 62) | n).expect("non-zero timeline id")
    }

    /// A dialogue of `turns` turns, each of `coupled` calling a tool the next
    /// turn answers.
    pub(in super::super) fn dialogue(turns: u32, coupled: &[u32]) -> (Substrate, TimelineId) {
        let mut sub = Substrate::new();
        let timeline = unshared_timeline(31);
        sub.register_timeline(timeline, LayerId::for_test(1), GroupId::for_test(1));
        for _ in 0..turns {
            sub.append_with_blocks(timeline, 10, 0, 1);
        }
        for &from in coupled {
            sub.couple_turn(timeline, from, RecordIdentity::NONE);
        }
        (sub, timeline)
    }

    /// While a tool loop runs, the turn in flight continues it: the loop's
    /// rounds keep their reasoning, the request before it does not.
    #[test]
    fn a_running_tool_loop_keeps_its_rounds_reasoning() {
        // #0–#1 a finished request; #2 the user's message, calling; #3 calling.
        let (sub, timeline) = dialogue(4, &[2, 3]);
        assert_eq!(
            sub.reasoning_turns(timeline),
            vec![TurnIndex(2), TurnIndex(3)]
        );
    }

    /// A long loop keeps only its newest `REASONING_ROUNDS` rounds.
    #[test]
    fn a_long_tool_loop_keeps_its_newest_rounds() {
        let coupled: Vec<u32> = (0..12).collect();
        let (sub, timeline) = dialogue(12, &coupled);
        let kept: Vec<TurnIndex> = (4..12).map(TurnIndex).collect();
        assert_eq!(REASONING_ROUNDS, 8);
        assert_eq!(sub.reasoning_turns(timeline), kept);
    }

    /// A newest turn that called nothing finished its request: the next turn
    /// is a new user message, and no reasoning is carried into it.
    #[test]
    fn a_finished_request_keeps_no_reasoning() {
        let (sub, timeline) = dialogue(4, &[1, 2]);
        assert_eq!(sub.reasoning_turns(timeline), Vec::<TurnIndex>::new());
    }

    /// A lift multiplies a scored turn, gives an unscored one what its reads
    /// are worth over the base, and leaves a turn it does not name alone.
    #[test]
    fn a_hit_lift_amplifies_scores_and_credits_reads() {
        let tl = TimelineId::from_raw(7).unwrap();
        let mut scores = ProjectionScores::new();
        scores.set_turn(tl, TurnIndex(0), 200.0);
        let other = TimelineId::from_raw(8).unwrap();
        scores.lift_turns(
            [
                (TurnKey::new(tl, TurnIndex(0)), 2.5),
                (TurnKey::new(tl, TurnIndex(1)), 4.0),
                (TurnKey::new(tl, TurnIndex(2)), 1.0),
                (TurnKey::new(other, TurnIndex(0)), 4.0),
            ],
            1000.0,
            |k| k.timeline == tl,
        );
        assert_eq!(scores.turn(tl, TurnIndex(0)), 500.0);
        assert_eq!(scores.turn(tl, TurnIndex(1)), 3000.0, "1000 · gain · h");
        assert_eq!(scores.turn(tl, TurnIndex(2)), 0.0, "no reads, no credit");
        assert_eq!(scores.turn(other, TurnIndex(0)), 0.0, "not credited");
        assert_eq!(scores.turn_count(), 2);
        // Every lifted turn records its reads, credited or not.
        assert_eq!(scores.reads(tl, TurnIndex(0)), 1.5);
        assert_eq!(scores.reads(tl, TurnIndex(1)), 3.0);
        assert_eq!(scores.reads(tl, TurnIndex(2)), 0.0);
        assert_eq!(scores.reads(other, TurnIndex(0)), 3.0);
        assert_eq!(scores.reads(tl, TurnIndex(3)), 0.0, "never lifted");
    }

    /// **An append-only layer reads every group self-local, but those of the
    /// layers it compounds on.** A unit of an analysis layer retrieves the
    /// analyses below it of other files; its own layer's siblings, and every
    /// group it was not given, stay masked to it. A dialogue layer is never
    /// self-local.
    #[test]
    fn a_compounding_layer_reads_only_its_lower_groups_across_conversations() {
        let mut sub = Substrate::new();
        let (dialogue, ingest) = (LayerId::for_test(1), LayerId::for_test(2));
        let (own, lower, other) = (
            GroupId::for_test(3),
            GroupId::for_test(4),
            GroupId::for_test(5),
        );
        assert!(!sub.is_self_local_group(dialogue, lower));
        sub.mark_layer_append_only(ingest);
        assert!(
            sub.is_self_local_group(ingest, lower),
            "wholly self-local by default"
        );
        sub.set_ingest_sees(ingest, [lower].into_iter().collect());
        assert!(!sub.is_self_local_group(ingest, lower));
        assert!(sub.is_self_local_group(ingest, own));
        assert!(sub.is_self_local_group(ingest, other));
        assert!(!sub.is_self_local_group(dialogue, own));
    }

    pub(in super::super) fn make_timeline() -> (LayerId, GroupId, TimelineId, Substrate) {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(timeline, layer, group);
        // Production leaves every timeline with `summarize` off (the summariser
        // is disconnected — see `Engine::new`). These tests exercise the enqueue
        // MECHANISM, which is retained, so they opt in explicitly rather than
        // riding a default that no longer exists.
        sub.set_timeline_summarize(timeline, true);
        (layer, group, timeline, sub)
    }

    // ── fork lineage (`inherited_chain`) ────────────────────────────────────

    /// Build `n` timelines in one group, each forked from the one before, every
    /// turn costing `tokens_each`. Returns them oldest-first.
    pub(in super::super) fn forked_line(
        n: usize,
        tokens_each: usize,
    ) -> (Substrate, Vec<TimelineId>) {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let mut sub = Substrate::new();
        let mut tls = Vec::new();
        for _ in 0..n {
            let tl = alloc.next();
            sub.register_timeline(tl, layer, group);
            sub.append_with_blocks(tl, tokens_each, 0, 1);
            if let Some(parent) = tls.last().copied() {
                set_parent(&mut sub, tl, parent);
            }
            tls.push(tl);
        }
        (sub, tls)
    }

    pub(in super::super) fn set_parent(sub: &mut Substrate, child: TimelineId, parent: TimelineId) {
        let mut kv = BTreeMap::new();
        kv.insert(
            Substrate::FORKED_FROM_KEY.to_string(),
            parent.raw().to_string(),
        );
        sub.merge_custom(child, &kv, RecordIdentity::NONE);
    }

    // ── retrieval scope ─────────────────────────────────────────────────────

    /// Two targets and three ingest conversations in one group.
    /// Three timelines carrying `key`: live `a`, live `b`, tombstoned `a`.
    pub(in super::super) fn metadata_fixture() -> (Substrate, [TimelineId; 3]) {
        let alloc = TimelineAllocator::new();
        let mut sub = Substrate::new();
        let tls = [alloc.next(), alloc.next(), alloc.next()];
        for (tl, value) in tls.iter().zip(["a", "b", "a"]) {
            sub.register_timeline(*tl, LayerId::for_test(1), GroupId::for_test(1));
            arc_mut(&mut sub.timelines, tl)
                .unwrap()
                .custom
                .insert("key".into(), value.into());
        }
        sub.tombstoned_timelines.insert(tls[2]);
        (sub, tls)
    }

    /// **Many values are looked up in one pass**: each value maps to its live
    /// timelines, a tombstoned one is not among them, and a value nothing
    /// carries is absent.
    #[test]
    fn many_metadata_values_are_found_in_one_pass() {
        let (sub, [tl1, tl2, _]) = metadata_fixture();
        let asked = BTreeSet::from(["a".to_string(), "b".to_string(), "c".to_string()]);
        assert_eq!(
            sub.timelines_with_metadata_values("key", &asked),
            BTreeMap::from([("a".to_string(), vec![tl1]), ("b".to_string(), vec![tl2])])
        );
    }

    /// **One value's membership stops at its first live, intact match**: a
    /// value carried only by a tombstoned timeline is not there.
    #[test]
    fn one_metadata_value_is_probed_by_its_first_match() {
        let (mut sub, [tl1, _, _]) = metadata_fixture();
        assert!(sub.has_live_intact_metadata("key", "b"));
        assert!(!sub.has_live_intact_metadata("key", "c"));
        arc_mut(&mut sub.timelines, &tl1)
            .unwrap()
            .custom
            .insert("key".into(), "z".into());
        assert!(
            !sub.has_live_intact_metadata("key", "a"),
            "only the tombstoned timeline carries `a` now"
        );
        assert_eq!(
            sub.has_live_intact_metadata("key", "b"),
            sub.metadata_values_for_key("key").contains("b")
        );
    }

    pub(in super::super) fn scope_fixture() -> (Substrate, GroupId, [TimelineId; 2], [TimelineId; 3])
    {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let mut sub = Substrate::new();
        let targets = [alloc.next(), alloc.next()];
        let ingested = [alloc.next(), alloc.next(), alloc.next()];
        for tl in ingested {
            sub.register_timeline(tl, layer, group);
        }
        (sub, group, targets, ingested)
    }

    pub(in super::super) fn scoped(
        sub: &Substrate,
        group: GroupId,
        target: TimelineId,
    ) -> Vec<TimelineId> {
        sub.scoped_timelines_for_group(group, target).collect()
    }

    /// **A group that is not scoped offers every live conversation**, to
    /// every target — a scope given to it changes nothing.
    #[test]
    fn an_unscoped_group_offers_everything() {
        let (mut sub, group, [a, _], ingested) = scope_fixture();
        sub.set_retrieval_scope(a, group, Arc::new(HashSet::from([ingested[0]])));
        assert!(!sub.is_group_scoped(group));
        assert_eq!(scoped(&sub, group, a), ingested.to_vec());
    }

    /// **A scoped group offers each target exactly what its scope names**,
    /// nothing to a target with no scope, and never a tombstoned conversation
    /// even when named.
    #[test]
    fn a_scoped_group_offers_each_target_its_own_scope() {
        let (mut sub, group, [a, b], ingested) = scope_fixture();
        sub.mark_group_scoped(group);
        assert!(sub.is_group_scoped(group));
        assert!(
            scoped(&sub, group, a).is_empty(),
            "no scope, nothing offered"
        );

        sub.set_retrieval_scope(
            a,
            group,
            Arc::new(HashSet::from([ingested[0], ingested[2]])),
        );
        sub.set_retrieval_scope(b, group, Arc::new(HashSet::from([ingested[1]])));
        assert_eq!(scoped(&sub, group, a), vec![ingested[0], ingested[2]]);
        assert_eq!(scoped(&sub, group, b), vec![ingested[1]]);

        sub.tombstoned_timelines.insert(ingested[2]);
        assert_eq!(scoped(&sub, group, a), vec![ingested[0]]);

        sub.set_retrieval_scope(a, group, Arc::new(HashSet::from([ingested[1]])));
        assert_eq!(
            scoped(&sub, group, a),
            vec![ingested[1]],
            "replaced, not merged"
        );

        sub.clear_retrieval_scope(a);
        assert!(scoped(&sub, group, a).is_empty());
        assert_eq!(scoped(&sub, group, b), vec![ingested[1]], "b's is its own");
    }

    /// **A target that can take no more turns keeps no scope**: archiving it
    /// or tombstoning it lets every scope it was given go.
    #[test]
    fn an_archived_or_tombstoned_target_lets_its_scope_go() {
        let (mut sub, group, [a, b], ingested) = scope_fixture();
        sub.mark_group_scoped(group);
        let dialogue = GroupId::for_test(2);
        for target in [a, b] {
            sub.register_timeline(target, LayerId::for_test(2), dialogue);
            sub.set_retrieval_scope(target, group, Arc::new(HashSet::from([ingested[0]])));
        }
        assert!(sub.set_archived(a, true));
        assert!(scoped(&sub, group, a).is_empty());
        assert_eq!(scoped(&sub, group, b), vec![ingested[0]]);
        sub.tombstone_timeline(b);
        assert!(sub.retrieval_scopes.is_empty());
        assert!(
            sub.scope_digests.is_empty(),
            "the digests go with the scopes"
        );
    }

    /// **Targets whose scopes of a group name the same conversations share
    /// its digest, whatever order the set was built in**; another set is
    /// another digest, a scoped group's target with no scope reads as the
    /// empty one, and a group that is not scoped has none at all.
    #[test]
    fn a_scope_digest_names_the_set_a_target_is_offered() {
        let (mut sub, group, [a, b], ingested) = scope_fixture();
        assert_eq!(sub.scope_digest(group, a), None, "not scoped");
        sub.mark_group_scoped(group);
        assert_eq!(sub.scope_digest(group, a), Some(EMPTY_SCOPE_DIGEST));

        sub.set_retrieval_scope(
            a,
            group,
            Arc::new(HashSet::from([ingested[0], ingested[2]])),
        );
        sub.set_retrieval_scope(
            b,
            group,
            Arc::new(HashSet::from([ingested[2], ingested[0]])),
        );
        let shared = sub.scope_digest(group, a);
        assert_eq!(shared, sub.scope_digest(group, b));
        assert_ne!(shared, Some(EMPTY_SCOPE_DIGEST));

        sub.set_retrieval_scope(b, group, Arc::new(HashSet::from([ingested[1]])));
        assert_ne!(sub.scope_digest(group, b), shared);
        assert_eq!(sub.scope_digest(group, a), shared, "a's is its own");

        sub.set_retrieval_scope(a, group, Arc::new(HashSet::new()));
        assert_eq!(sub.scope_digest(group, a), Some(EMPTY_SCOPE_DIGEST));
        sub.clear_retrieval_scope(b);
        assert_eq!(sub.scope_digest(group, b), Some(EMPTY_SCOPE_DIGEST));
    }

    /// **A drawing group's candidates are its source's, as its own scope names
    /// them**, and a unit it draws is placed through it — while the source,
    /// scoped without it, no longer offers it.
    #[test]
    fn a_drawing_group_places_what_its_scope_draws_from_its_source() {
        let (mut sub, files, [a, b], ingested) = scope_fixture();
        let guides = GroupId::for_test(3);
        sub.set_group_draws(guides, files);
        sub.mark_group_scoped(files);
        sub.mark_group_scoped(guides);
        assert_eq!(
            sub.timelines_for_group(guides).count(),
            0,
            "it holds none of its own"
        );
        assert_eq!(
            sub.active_timelines_for_group(guides).collect::<Vec<_>>(),
            ingested.to_vec(),
            "it selects from its source's"
        );
        sub.set_retrieval_scope(a, files, Arc::new(HashSet::from([ingested[0]])));
        sub.set_retrieval_scope(a, guides, Arc::new(HashSet::from([ingested[1]])));
        assert_eq!(scoped(&sub, files, a), vec![ingested[0]]);
        assert_eq!(scoped(&sub, guides, a), vec![ingested[1]]);
        assert_eq!(sub.placing_group(a, ingested[0]), Some(files));
        assert_eq!(sub.placing_group(a, ingested[1]), Some(guides));
        assert_eq!(
            sub.placing_group(b, ingested[1]),
            Some(files),
            "a target whose scope draws nothing places through the owner"
        );
        assert!(scoped(&sub, guides, b).is_empty());
    }

    /// **A drawn unit draws on the drawing group's share** of the working set.
    #[test]
    fn a_drawn_unit_draws_on_the_drawing_groups_share() {
        let (mut sub, target, reads) = working_set_fixture(2, 100);
        let files = GroupId::for_test(1);
        let guides = GroupId::for_test(3);
        sub.set_working_set_share(guides, WorkingSetShare::Guides);
        sub.set_group_draws(guides, files);
        sub.mark_group_scoped(guides);
        sub.set_retrieval_scope(target, guides, Arc::new(HashSet::from([reads[1]])));
        let share = |tl| sub.working_set_candidate(target, tl).and_then(|c| c.share);
        assert_eq!(share(reads[0]), Some(WorkingSetShare::Readings));
        assert_eq!(share(reads[1]), Some(WorkingSetShare::Guides));
    }

    // ── working sets ────────────────────────────────────────────────────────

    /// A dialogue plus `n` ingest conversations of `tokens_each`, none of them
    /// related — a working set carries conversations that never forked from
    /// one another.
    pub(in super::super) fn working_set_fixture(
        n: usize,
        tokens_each: usize,
    ) -> (Substrate, TimelineId, Vec<TimelineId>) {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let mut sub = Substrate::new();
        sub.set_working_set_share(group, WorkingSetShare::Readings);
        let target = alloc.next();
        sub.register_timeline(target, layer, group);
        let mut reads = Vec::new();
        for _ in 0..n {
            let tl = alloc.next();
            sub.register_timeline(tl, layer, group);
            sub.append_with_blocks(tl, tokens_each, 0, 1);
            reads.push(tl);
        }
        (sub, target, reads)
    }

    /// Observe `fresh` into `target`'s working set at the defaults.
    pub(in super::super) fn ws_observe(
        sub: &mut Substrate,
        target: TimelineId,
        fresh: &HashMap<TimelineId, f32>,
    ) {
        sub.working_set_observe(
            target,
            fresh,
            &HashMap::new(),
            &HitsConfig::default(),
            0.2,
            100.0,
        );
    }

    /// **The pool is the card's size, and a piece is pooled whole**: two
    /// 300-token readings fit a pool of 600, and only one fits a pool of 500;
    /// no per-piece cap refuses either.
    #[test]
    fn the_pool_holds_whole_pieces_within_the_cards_size() {
        let members = |sub: &Substrate, target| -> Vec<TimelineId> {
            sub.working_set(target)
                .map(|ws| ws.members())
                .unwrap_or_default()
        };
        for (tokens, pooled) in [(600, 2), (500, 1)] {
            let (mut sub, target, reads) = working_set_fixture(2, 300);
            sub.set_pool_size(PoolSize {
                tokens,
                per_kind: 0,
            });
            ws_observe(
                &mut sub,
                target,
                &HashMap::from([(reads[0], 5000.0), (reads[1], 4000.0)]),
            );
            assert_eq!(members(&sub, target).len(), pooled, "a pool of {tokens}");
            assert_eq!(members(&sub, target)[0], reads[0], "the strongest first");
        }
    }

    /// A link of the dialogue's own inherited chain is projected as its opening
    /// already, so it is never a member, however strongly the scan scores it.
    /// Admitted, it emitted twice.
    #[test]
    fn a_link_of_the_dialogues_own_chain_is_never_a_member() {
        let (mut sub, target, reads) = working_set_fixture(2, 10);
        set_parent(&mut sub, target, reads[0]);
        assert!(sub.working_set_candidate(target, reads[0]).is_none());
        let fresh = HashMap::from([(reads[0], 5000.0), (reads[1], 5000.0)]);
        ws_observe(&mut sub, target, &fresh);
        let members = sub.working_set(target).unwrap().members();
        assert_eq!(members, vec![reads[1]]);
    }

    /// A conversation in no working-set group can never be a member.
    #[test]
    fn a_conversation_outside_every_working_set_group_is_gone() {
        let (mut sub, target, _) = working_set_fixture(0, 10);
        let elsewhere = unshared_timeline(7);
        sub.register_timeline(elsewhere, LayerId::for_test(1), GroupId::for_test(9));
        sub.append_with_blocks(elsewhere, 10, 0, 1);
        let candidate = sub.working_set_candidate(target, elsewhere);
        assert!(
            candidate.is_some_and(|c| c.share.is_none()),
            "no group's share"
        );
        ws_observe(&mut sub, target, &HashMap::from([(elsewhere, 5000.0)]));
        assert!(sub.working_set(target).unwrap().members().is_empty());
    }

    /// A retired conversation reads as a miss. Its turns are gone, so serving
    /// it would tell the model it has a file that nothing will deliver.
    #[test]
    fn a_tombstoned_archived_empty_or_own_conversation_is_gone() {
        let (mut sub, target, reads) = working_set_fixture(2, 10);
        sub.tombstone_timeline(reads[0]);
        sub.set_archived(reads[1], true);
        let empty = unshared_timeline(1);
        sub.register_timeline(empty, LayerId::for_test(1), GroupId::for_test(1));
        for tl in [reads[0], reads[1], empty, target] {
            assert!(sub.working_set_candidate(target, tl).is_none(), "{tl:?}");
        }
        ws_observe(
            &mut sub,
            target,
            &HashMap::from([(reads[0], 5000.0), (empty, 5000.0)]),
        );
        assert!(sub.working_set(target).unwrap().provenance().is_empty());
    }

    /// **A gap is a turn still arriving.** A timeline holds one from its first
    /// restored gap; one whose turns are all its own holds none, and an
    /// unknown timeline holds none.
    #[test]
    fn a_timeline_holds_a_gap_while_a_turn_is_still_arriving() {
        let (mut sub, _, _) = working_set_fixture(0, 10);
        let pending = unshared_timeline(5);
        sub.register_timeline(pending, LayerId::for_test(1), GroupId::for_test(1));
        sub.append_with_blocks(pending, 10, 0, 1);
        assert!(!sub.holds_gap(pending), "every turn its own");
        sub.restore_gap_turn(pending);
        assert!(sub.holds_gap(pending), "a turn still arriving");
        assert!(
            !sub.holds_gap(unshared_timeline(6)),
            "a timeline it does not hold"
        );
    }

    /// **A conversation that is not whole is gone.** A gap — a turn another
    /// member wrote and this one has not fetched — and a retired turn are both
    /// empty, and the projection skips an empty turn: admitted, the
    /// conversation would place less than it is charged.
    #[test]
    fn a_conversation_that_is_not_whole_is_gone() {
        let (mut sub, target, _) = working_set_fixture(0, 10);
        let partial = unshared_timeline(3);
        sub.register_timeline(partial, LayerId::for_test(1), GroupId::for_test(1));
        sub.restore_gap_turn(partial);
        sub.append_with_blocks(partial, 10, 0, 1);
        assert_eq!(sub.turn_count(partial), 2);
        assert!(sub.working_set_candidate(target, partial).is_none());
        ws_observe(&mut sub, target, &HashMap::from([(partial, 5000.0)]));
        assert!(sub.working_set(target).unwrap().members().is_empty());

        let retired = unshared_timeline(4);
        sub.register_timeline(retired, LayerId::for_test(1), GroupId::for_test(1));
        sub.append_with_blocks(retired, 10, 0, 1);
        sub.append_with_blocks(retired, 10, 1, 2);
        assert!(sub.holds_whole(retired));
        sub.tombstone_turn(retired, 1);
        assert!(!sub.holds_whole(retired));
        assert!(sub.working_set_candidate(target, retired).is_none());
    }

    /// **A calibration names what it calibrates, and is never a member.**
    /// **A candidate's type is read from the subject its metadata names**: a
    /// document is typed, a source file and a conversation with no subject are
    /// not, and with no types installed nothing is.
    #[test]
    fn a_candidate_is_typed_by_its_subject() {
        let (mut sub, target, reads) = working_set_fixture(3, 10);
        let kind = |sub: &Substrate, tl| sub.working_set_candidate(target, tl).and_then(|c| c.kind);
        for (tl, path) in [
            (reads[0], "candle/docs/x.md"),
            (reads[1], "candle/zend/src/a.rs"),
        ] {
            let kv = BTreeMap::from([("path".to_string(), path.to_string())]);
            sub.merge_custom(tl, &kv, RecordIdentity::NONE);
        }
        assert_eq!(kind(&sub, reads[0]), None, "no types installed yet");
        let types =
            TypeShares::new(&[("docs".to_string(), vec![r"\.md$".to_string()], 0.35)]).unwrap();
        sub.set_working_set_types(types, "path");
        let docs = kind(&sub, reads[0]).expect("a document");
        assert_eq!((docs.id, docs.per_mille), (0, 350));
        assert_eq!(kind(&sub, reads[1]), None, "source is unbounded");
        assert_eq!(kind(&sub, reads[2]), None, "no subject, no type");
    }

    /// The lookup keys each live calibration by its conversation — archived
    /// ones included, tombstoned ones not — and a calibration is never a
    /// candidate however whole it is.
    #[test]
    fn a_calibration_is_found_by_its_conversation_and_never_delivered() {
        let (mut sub, target, reads) = working_set_fixture(1, 10);
        let file = reads[0];
        let calibrate = |sub: &mut Substrate, raw: u64| {
            let tl = unshared_timeline(raw);
            sub.register_timeline(tl, LayerId::for_test(1), GroupId::for_test(1));
            sub.append_with_blocks(tl, 4, 0, 1);
            let kv = BTreeMap::from([(
                Substrate::CALIBRATES_KEY.to_string(),
                file.raw().to_string(),
            )]);
            sub.merge_custom(tl, &kv, RecordIdentity::NONE);
            tl
        };
        let (a, b, gone) = (
            calibrate(&mut sub, 11),
            calibrate(&mut sub, 12),
            calibrate(&mut sub, 13),
        );
        sub.set_archived(b, true);
        sub.tombstone_timeline(gone);

        assert_eq!(sub.calibrates(a), Some(file));
        assert_eq!(sub.calibrates(file), None);
        let mut expect = vec![a, b];
        expect.sort();
        assert_eq!(sub.calibrations(), HashMap::from([(file, expect)]));
        assert!(sub.working_set_candidate(target, a).is_none());
        assert!(sub.working_set_candidate(target, file).is_some());

        // Re-pointed at another unit, a calibration moves with it.
        let other = unshared_timeline(14);
        let kv = BTreeMap::from([(
            Substrate::CALIBRATES_KEY.to_string(),
            other.raw().to_string(),
        )]);
        sub.merge_custom(a, &kv, RecordIdentity::NONE);
        assert_eq!(sub.calibrations_of(file).collect::<Vec<_>>(), vec![b]);
        assert_eq!(sub.calibrations_of(other).collect::<Vec<_>>(), vec![a]);

        // Named before it registered, a calibration is found once it does.
        let early = unshared_timeline(15);
        sub.merge_custom(early, &kv, RecordIdentity::NONE);
        assert_eq!(sub.calibrations_of(other).collect::<Vec<_>>(), vec![a]);
        sub.register_timeline(early, LayerId::for_test(1), GroupId::for_test(1));
        let mut expect = vec![a, early];
        expect.sort();
        assert_eq!(sub.calibrations_of(other).collect::<Vec<_>>(), expect);
    }

    /// A tombstoned conversation leaves every dialogue's working set — a member
    /// whose turns are gone would place nothing — and a tombstoned dialogue's
    /// own set goes with it.
    #[test]
    fn a_tombstone_leaves_every_working_set() {
        let (mut sub, target, reads) = working_set_fixture(3, 10);
        ws_observe(
            &mut sub,
            target,
            &HashMap::from([(reads[0], 1000.0), (reads[1], 1000.0)]),
        );
        ws_observe(&mut sub, target, &HashMap::from([(reads[2], 500.0)]));
        assert_eq!(sub.working_set(target).unwrap().members().len(), 3);
        for r in &reads {
            sub.tombstone_timeline(*r);
        }
        let ws = sub.working_set(target).unwrap();
        assert!(ws.provenance().is_empty());
        sub.tombstone_timeline(target);
        assert!(sub.working_set(target).is_none());
    }

    /// **A new question opens the next boundary even on a set that does not
    /// exist yet**: a dialogue opens with its set cleared away, and its first
    /// question must still choose its slot whole, with no allowance.
    #[test]
    fn a_first_question_opens_the_first_boundary_of_an_absent_set() {
        let (mut sub, target, _) = working_set_fixture(0, 10);
        sub.working_set_clear(target);
        assert!(sub.working_set(target).is_none());
        sub.working_set_clear_withheld(target);
        Substrate::working_set_boundary(&mut sub, target, &[]);
        assert!(sub.working_set(target).unwrap().opening());
        Substrate::working_set_boundary(&mut sub, target, &[]);
        assert!(!sub.working_set(target).unwrap().opening());
    }

    #[test]
    fn clearing_drops_the_set_for_a_rebuild() {
        let (mut sub, target, reads) = working_set_fixture(2, 10);
        ws_observe(
            &mut sub,
            target,
            &HashMap::from([(reads[0], 1000.0), (reads[1], 1000.0)]),
        );
        sub.working_set_clear(target);
        assert!(sub.working_set(target).is_none());
    }

    #[test]
    fn a_timeline_that_never_forked_inherits_only_itself() {
        let (_layer, _group, tl, sub) = make_timeline();
        assert_eq!(sub.inherited_chain(tl), vec![tl]);
        assert_eq!(sub.parent_timeline(tl), None);
    }

    /// The whole point: the chain reads oldest-first, so concatenating each
    /// timeline's turns yields the documents in the order they were read.
    #[test]
    fn an_inherited_chain_runs_oldest_ancestor_first_and_ends_at_the_target() {
        let (sub, tls) = forked_line(4, 10);
        let target = *tls.last().unwrap();
        assert_eq!(sub.inherited_chain(target), tls);
    }

    #[test]
    fn a_tombstoned_ancestor_ends_the_chain_instead_of_failing() {
        let (mut sub, tls) = forked_line(3, 10);
        sub.tombstone_timeline(tls[0]);
        // tls[2] -> tls[1] still resolves; tls[1] -> tls[0] is now dangling.
        assert_eq!(sub.inherited_chain(tls[2]), vec![tls[1], tls[2]]);
    }

    /// An ancestor a layer's own reconciliation has archived drops out the same
    /// way — the child keeps working, it just inherits less.
    #[test]
    fn an_archived_ancestor_ends_the_chain() {
        let (mut sub, tls) = forked_line(3, 10);
        sub.set_archived(tls[0], true);
        assert_eq!(sub.inherited_chain(tls[2]), vec![tls[1], tls[2]]);
    }

    /// The cap drops the FRONT — the oldest, most general ancestors — and keeps
    /// the nearest ones, which are the most specific to the target.
    #[test]
    fn the_token_cap_drops_the_front_of_the_chain_and_keeps_the_nearest() {
        // Four ancestors at two fifths of the ceiling each: two fit inside it
        // (four fifths), a third would pass it (six fifths).
        let (sub, tls) = forked_line(5, Substrate::INHERITED_CHAIN_TOKEN_CAP * 2 / 5);
        let target = *tls.last().unwrap();
        let chain = sub.inherited_chain(target);
        assert_eq!(chain, vec![tls[2], tls[3], target]);
    }

    /// A conversation never loses its OWN turns to the inherited budget, however
    /// large they are — the cap governs what it inherits, not what it is.
    #[test]
    fn the_target_survives_a_cap_its_own_turns_already_exceed() {
        let (sub, tls) = forked_line(2, Substrate::INHERITED_CHAIN_TOKEN_CAP * 2);
        let target = *tls.last().unwrap();
        assert_eq!(sub.inherited_chain(target), vec![target]);
    }

    /// `forked_from` is durable, user-visible metadata, so the walk must not
    /// trust it to be acyclic — a cycle would otherwise hang every projection.
    #[test]
    fn a_cycle_in_the_lineage_truncates_rather_than_hanging() {
        let (mut sub, tls) = forked_line(3, 10);
        set_parent(&mut sub, tls[0], tls[2]);
        let chain = sub.inherited_chain(tls[2]);
        assert_eq!(chain.len(), 3, "each timeline appears exactly once");
        assert_eq!(*chain.last().unwrap(), tls[2]);
    }

    /// The lineage walk sees past the token cap and stops on a cycle already
    /// recorded.
    #[test]
    fn lineage_contains_walks_the_whole_lineage() {
        let (mut sub, tls) = forked_line(4, Substrate::INHERITED_CHAIN_TOKEN_CAP);
        assert!(sub.lineage_contains(tls[3], tls[0]), "past the token cap");
        assert!(
            sub.lineage_contains(tls[3], tls[3]),
            "a timeline is its own"
        );
        assert!(!sub.lineage_contains(tls[0], tls[3]), "a descendant is not");
        set_parent(&mut sub, tls[0], tls[2]);
        assert!(
            !sub.lineage_contains(tls[1], tls[3]),
            "a cycle ends the walk"
        );
        assert!(
            sub.lineage_contains(tls[0], tls[1]),
            "round a recorded cycle"
        );
    }

    /// **The seed alias resolves to whatever chain this member holds now**:
    /// nothing before a seed is named, the chain's end once one is, and the
    /// new end when the chain is rebuilt — with nothing rewritten on the
    /// conversation.
    #[test]
    fn the_seed_alias_follows_this_members_chain() {
        let (mut sub, chain) = forked_line(2, 10);
        let dialogue = TimelineAllocator::new().next();
        sub.register_timeline(dialogue, LayerId::for_test(1), GroupId::for_test(1));
        sub.append_with_blocks(dialogue, 10, 0, 1);
        let kv = BTreeMap::from([(
            Substrate::FORKED_FROM_KEY.to_string(),
            Substrate::SEED_PARENT.to_string(),
        )]);
        sub.merge_custom(dialogue, &kv, RecordIdentity::NONE);
        assert_eq!(sub.parent_timeline(dialogue), None, "no seed yet");
        sub.set_seed(Some(chain[1]));
        assert_eq!(
            sub.inherited_chain(dialogue),
            vec![chain[0], chain[1], dialogue]
        );
        sub.set_seed(Some(chain[0]));
        assert_eq!(
            sub.inherited_chain(dialogue),
            vec![chain[0], dialogue],
            "a rebuilt chain is followed"
        );
    }
}
