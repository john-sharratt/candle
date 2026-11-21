//! Tests that labels, conversation state and tombstones land the same whatever
//! order replay presents them in.
//!
//! A label can be replayed before its timeline's declaration, so records for an
//! unregistered timeline are held and merged when it registers. Each label field
//! is decided by the greatest stamp that set it, and a conversation ranks by the
//! least record that named it. A live turn tombstone empties the turn, so a
//! retired answer is neither selected nor handed on as history. The
//! distilled-tombstone lookup admits a conversation only when its signatures were
//! kept by design.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::projection::{GroupId, LayerId, TimelineAllocator, TimelineId};

    use crate::token_buffer::TokenBuffer;

    use super::super::tests_working_set::tests::make_timeline;

    use super::super::tests_timeline_meta::tests::minimal_sealed_layer;

    /// Regression: zend's redo log writes the `Label` record carrying
    /// `conv_id` immediately on conversation creation, before any
    /// TurnDecl exists.  On reopen, the walker would replay the
    /// Label first and the TurnDecl second.  If `apply_conv_meta`
    /// dropped the meta when the timeline wasn't registered yet,
    /// every restored conversation would vanish from the sidebar
    /// because `conv_id` never reached the TimelineEntry.
    #[test]
    fn conv_meta_before_timeline_registration_survives() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        let mut sub = Substrate::new();

        let meta = super::ConvMeta {
            conv_id: "1780659918260".to_string(),
            label: String::new(),
            custom: Default::default(),
        };
        sub.apply_conv_meta(timeline.raw(), &meta, RecordIdentity::NONE);
        sub.apply_conv_state(
            timeline.raw(),
            super::ConvState::default(),
            RecordIdentity::NONE,
        );
        assert!(
            sub.known_conversations().is_empty(),
            "pre-registration: meta stashed, not visible yet"
        );

        sub.register_timeline(timeline, layer, group);

        let convs = sub.known_conversations();
        assert_eq!(convs.len(), 1, "post-registration: drained meta visible");
        assert_eq!(convs[0].1, "1780659918260");
    }

    /// Partial Label payloads merge in the stash — an earlier
    /// conv_id-only Label and a later label-only Label must both
    /// land on the TimelineEntry when registration finally drains
    /// the pending state.
    #[test]
    fn pending_conv_meta_merges_partial_updates() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        let mut sub = Substrate::new();

        sub.apply_conv_meta(
            timeline.raw(),
            &super::ConvMeta {
                conv_id: "abc".into(),
                label: String::new(),
                custom: Default::default(),
            },
            RecordIdentity::NONE,
        );
        sub.apply_conv_meta(
            timeline.raw(),
            &super::ConvMeta {
                conv_id: String::new(),
                label: "tour".into(),
                custom: Default::default(),
            },
            RecordIdentity::NONE,
        );
        sub.register_timeline(timeline, layer, group);

        let convs = sub.known_conversations();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].1, "abc");
        assert_eq!(convs[0].2, "tour");
    }

    /// `order` for labels written before identities existed is counted in the
    /// sequence their first `conv_id` replays in, and is stable: re-setting a
    /// conv_id doesn't re-stamp, and `live_label_records` (the compactor's Label
    /// source) emits in that order so a reload re-derives the same sequence
    /// instead of scrambling the sidebar on every compaction.
    #[test]
    fn conv_order_stamps_creation_sequence_and_emits_stably() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let mut sub = Substrate::new();

        let (a, b, c) = (alloc.next(), alloc.next(), alloc.next());
        for (tl, id) in [(a, "a"), (b, "b"), (c, "c")] {
            sub.register_timeline(tl, layer, group);
            sub.set_conv_id(tl, id, RecordIdentity::NONE);
        }

        let order_of = |sub: &Substrate, want: &str| {
            sub.known_conversations()
                .into_iter()
                .find(|(_, cid, ..)| cid == want)
                .map(|(_, _, _, _, o)| o)
                .expect("conversation present")
        };
        assert_eq!(order_of(&sub, "a"), 1);
        assert_eq!(order_of(&sub, "b"), 2);
        assert_eq!(order_of(&sub, "c"), 3);

        // Re-setting the same conv_id must not bump the counter or re-stamp.
        sub.set_conv_id(a, "a", RecordIdentity::NONE);
        assert_eq!(order_of(&sub, "a"), 1);
        assert_eq!(order_of(&sub, "c"), 3, "counter untouched by a re-set");

        // The compactor's Label source emits in creation order, so reload is stable.
        let emitted: Vec<String> = sub
            .live_label_records()
            .into_iter()
            .map(|(_, meta, _)| meta.conv_id)
            .collect();
        assert_eq!(emitted, ["a", "b", "c"]);
    }

    /// **A stamped conversation ranks by the clock reading of the least record
    /// that named it**, so every member orders it the same whatever order the
    /// records arrived in, and the record that named it first survives a
    /// rewrite even when a later one holds the `conv_id` field.
    #[test]
    fn a_stamped_conversation_ranks_by_its_first_naming_record() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let at = |hlc: u64| RecordIdentity {
            origin: 5,
            oseq: hlc,
            gen: 0,
            epoch: 0,
            hlc,
        };
        let tl = alloc.next();
        for order in [[30u64, 10, 20], [10, 20, 30], [20, 30, 10]] {
            let mut sub = Substrate::new();
            sub.register_timeline(tl, layer, group);
            for hlc in order {
                sub.set_conv_id(tl, &format!("c{hlc}"), at(hlc));
            }
            let entry = sub.timeline_entry(tl).unwrap();
            assert_eq!(entry.order, 10, "order {order:?}");
            assert_eq!(entry.conv_id.as_deref(), Some("c30"), "order {order:?}");
            let records = sub.live_label_records();
            let carried: Vec<(String, u64)> = records
                .iter()
                .map(|(_, m, id)| (m.conv_id.clone(), id.hlc))
                .collect();
            assert_eq!(
                carried,
                vec![("c10".to_string(), 10), ("c30".to_string(), 30)],
                "the first namer and the field's holder are both carried"
            );
        }
    }

    /// **Each label field keeps the greatest record that set it**, so a rename
    /// and a tag written by two members at once both land, in any order, and a
    /// stale restatement does not roll either back.
    #[test]
    fn label_fields_merge_by_stamp_in_any_order() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let tl = alloc.next();
        let at = |origin: u64, hlc: u64| RecordIdentity {
            origin,
            oseq: hlc,
            gen: 0,
            epoch: 0,
            hlc,
        };
        let meta = |label: &str, k: Option<(&str, &str)>| super::ConvMeta {
            conv_id: String::new(),
            label: label.to_string(),
            custom: k
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        };
        let records = [
            (meta("old title", Some(("k", "v1"))), at(1, 10)),
            (meta("renamed", None), at(2, 20)),
            (meta("", Some(("k", "v2"))), at(3, 30)),
            (meta("stale", None), at(1, 15)),
        ];
        let mut first: Option<(Option<String>, POrdMap<String, String>)> = None;
        for perm in [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]] {
            let mut sub = Substrate::new();
            sub.register_timeline(tl, layer, group);
            for i in perm {
                let (m, id) = &records[i];
                sub.apply_conv_meta(tl.raw(), m, *id);
            }
            let entry = sub.timeline_entry(tl).unwrap();
            let got = (entry.label.clone(), entry.custom.clone());
            assert_eq!(got.0.as_deref(), Some("renamed"), "perm {perm:?}");
            assert_eq!(
                got.1.get("k").map(String::as_str),
                Some("v2"),
                "perm {perm:?}"
            );
            match &first {
                None => first = Some(got),
                Some(f) => assert_eq!(f, &got, "perm {perm:?}"),
            }
        }
    }

    // ── Custom metadata (content-addressed cache) ──────────────────────

    #[test]
    fn custom_metadata_merges_and_is_searchable() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let tl = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(tl, layer, group);

        let mut kv = std::collections::BTreeMap::new();
        kv.insert("kind".to_string(), "code_read".to_string());
        kv.insert("path".to_string(), "src/lib.rs".to_string());
        kv.insert("content_sha256".to_string(), "abc123".to_string());
        sub.merge_custom(tl, &kv, RecordIdentity::NONE);

        let got = sub
            .custom_of(tl)
            .expect("registered timeline has custom map");
        assert_eq!(got.get("kind").map(String::as_str), Some("code_read"));
        assert_eq!(
            got.get("content_sha256").map(String::as_str),
            Some("abc123")
        );

        // Exact (key, value) search — the resume-cache + invalidation lookup.
        assert_eq!(
            sub.timelines_with_metadata("content_sha256", "abc123"),
            vec![tl]
        );
        assert_eq!(sub.timelines_with_metadata("path", "src/lib.rs"), vec![tl]);
        assert!(sub
            .timelines_with_metadata("content_sha256", "nope")
            .is_empty());
        assert!(sub.timelines_with_metadata("absent_key", "x").is_empty());

        // live_label_records carries custom so compaction can re-emit it.
        let live = sub.live_label_records();
        let entry = live
            .iter()
            .find(|e| e.0 == tl.raw())
            .expect("timeline in live meta");
        assert_eq!(
            entry.1.custom.get("path").map(String::as_str),
            Some("src/lib.rs")
        );
    }

    // ── Conversation state (archived + branches) ────────────────────────

    /// **A branch is set per repository, and only a change counts.** The
    /// state carries it beside `archived`, and `live_conv_states` — what
    /// compaction and maintenance re-emit — carries every non-default state
    /// whole, so a branch survives a rewrite of the log.
    #[test]
    fn branches_are_set_per_repository_and_re_emitted_whole() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let (a, b) = (alloc.next(), alloc.next());
        let mut sub = Substrate::new();
        sub.register_timeline(a, layer, group);
        sub.register_timeline(b, layer, group);

        assert_eq!(sub.conv_state(a), Some(super::ConvState::default()));
        assert!(sub.set_branch(a, "candle", "main"));
        assert!(!sub.set_branch(a, "candle", "main"), "no change, no write");
        assert!(sub.set_branch(a, "mind", "master"));
        assert!(sub.set_branch(a, "candle", "zen/work"));
        assert!(sub.set_archived(b, true));
        assert!(
            !sub.set_branch(alloc.next(), "candle", "main"),
            "unregistered"
        );

        let a_state = super::ConvState {
            archived: false,
            branches: [("candle", "zen/work"), ("mind", "master")]
                .into_iter()
                .map(|(r, b)| (r.to_string(), b.to_string()))
                .collect(),
            active: 0,
        };
        assert_eq!(sub.conv_state(a), Some(a_state.clone()));
        let b_state = super::ConvState {
            archived: true,
            branches: Default::default(),
            active: 0,
        };
        assert_eq!(
            sub.live_conv_states(),
            vec![
                (a.raw(), a_state, RecordIdentity::NONE),
                (b.raw(), b_state, RecordIdentity::NONE)
            ]
        );

        // A replayed record replaces the state whole.
        sub.apply_conv_state(a.raw(), super::ConvState::default(), RecordIdentity::NONE);
        assert_eq!(sub.conv_state(a), Some(super::ConvState::default()));
        assert_eq!(sub.live_conv_states().len(), 1, "a is untouched again");
    }

    /// **The last use outranks every earlier one, and survives replay.** Each
    /// touch stamps the next rank; a replayed state restores its rank exactly,
    /// whatever order the records arrive in, and the next touch still lands
    /// above every rank replayed.
    #[test]
    fn a_touch_outranks_every_earlier_use_and_replay_restores_it() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let (a, b) = (alloc.next(), alloc.next());
        let mut sub = Substrate::new();
        sub.register_timeline(a, layer, group);
        sub.register_timeline(b, layer, group);

        assert!(sub.touch_conversation(a));
        assert!(sub.touch_conversation(b));
        assert!(sub.touch_conversation(a));
        assert!(!sub.touch_conversation(alloc.next()), "unregistered");
        assert_eq!(sub.conv_state(a).unwrap().active, 3);
        assert_eq!(sub.conv_state(b).unwrap().active, 2);

        // Replay into a fresh substrate, the later-used record first — the
        // order compaction's timeline-ordered re-emit can produce.
        let states = sub.live_conv_states();
        let mut replayed = Substrate::new();
        replayed.register_timeline(a, layer, group);
        replayed.register_timeline(b, layer, group);
        for (tl, state, id) in states.into_iter().rev() {
            replayed.apply_conv_state(tl, state, id);
        }
        assert_eq!(replayed.conv_state(a).unwrap().active, 3);
        assert_eq!(replayed.conv_state(b).unwrap().active, 2);
        assert!(replayed.touch_conversation(b));
        assert_eq!(replayed.conv_state(b).unwrap().active, 4);
    }

    #[test]
    fn custom_metadata_survives_pre_registration_and_merges() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let tl = alloc.next();
        let mut sub = Substrate::new();

        // Two partial Label payloads (different custom keys) arrive before
        // the timeline registers — both must survive and merge on drain.
        let mut m1 = std::collections::BTreeMap::new();
        m1.insert("path".to_string(), "src/a.rs".to_string());
        sub.apply_conv_meta(
            tl.raw(),
            &super::ConvMeta {
                conv_id: String::new(),
                label: String::new(),
                custom: m1,
            },
            RecordIdentity::NONE,
        );
        let mut m2 = std::collections::BTreeMap::new();
        m2.insert("content_sha256".to_string(), "h1".to_string());
        sub.apply_conv_meta(
            tl.raw(),
            &super::ConvMeta {
                conv_id: String::new(),
                label: String::new(),
                custom: m2,
            },
            RecordIdentity::NONE,
        );

        assert!(
            sub.custom_of(tl).is_none(),
            "pre-registration: not visible yet"
        );
        sub.register_timeline(tl, layer, group);

        let got = sub.custom_of(tl).expect("custom drained on registration");
        assert_eq!(got.get("path").map(String::as_str), Some("src/a.rs"));
        assert_eq!(got.get("content_sha256").map(String::as_str), Some("h1"));
        assert_eq!(
            sub.timelines_with_metadata("content_sha256", "h1"),
            vec![tl]
        );
    }

    /// The distill-inclusive lookup admits a tombstoned timeline ONLY when it is
    /// distilled, and an ordinary tombstone stays invisible.
    ///
    /// A calibration exemplar's designed end state is archived + distilled
    /// (`ProvenanceOnly`) + tombstoned — out of the live gather, still answering
    /// the belief scan by signature. The live-only lookup could not see it, so
    /// the calibration resume filter read every finished exemplar as "never ran"
    /// and regenerated it. Widening the filter to ALL tombstones would have been
    /// wrong in the other direction: a genuinely retired conversation must stay
    /// ignored by provenance and by these checks. The distill flag is the only
    /// thing that separates the two.
    #[test]
    fn distill_inclusive_lookup_admits_only_distilled_tombstones() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let live = alloc.next();
        let corpus = alloc.next();
        let retired = alloc.next();
        let mut sub = Substrate::new();
        for t in [live, corpus, retired] {
            sub.register_timeline(t, layer, group);
            let mut m = std::collections::BTreeMap::new();
            m.insert("calib".to_string(), "marker-1".to_string());
            sub.merge_custom(t, &m, RecordIdentity::NONE);
        }
        // `corpus` is the provenance corpus: distilled AND tombstoned.
        sub.distill_timeline(corpus, DistillMode::ProvenanceOnly);
        sub.tombstone_timeline(corpus);
        // `retired` is an ordinary tombstone — gone, and must stay gone.
        sub.tombstone_timeline(retired);

        let mut got = sub.timelines_with_metadata_including_distilled("calib", "marker-1");
        got.sort();
        let mut want = vec![live, corpus];
        want.sort();
        assert_eq!(got, want, "distilled tombstone admitted, ordinary one not");
        assert!(
            !got.contains(&retired),
            "an ordinary tombstone must remain invisible to calibration/provenance"
        );

        // The live-only lookup is unchanged — everything else still wants it.
        assert_eq!(sub.timelines_with_metadata("calib", "marker-1"), vec![live]);
    }

    #[test]
    fn timelines_with_metadata_matches_only_exact_pairs() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let a = alloc.next();
        let b = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(a, layer, group);
        sub.register_timeline(b, layer, group);
        let mut ma = std::collections::BTreeMap::new();
        ma.insert("path".to_string(), "x.rs".to_string());
        let mut mb = std::collections::BTreeMap::new();
        mb.insert("path".to_string(), "y.rs".to_string());
        sub.merge_custom(a, &ma, RecordIdentity::NONE);
        sub.merge_custom(b, &mb, RecordIdentity::NONE);

        assert_eq!(sub.timelines_with_metadata("path", "x.rs"), vec![a]);
        assert_eq!(sub.timelines_with_metadata("path", "y.rs"), vec![b]);
        assert!(sub.timelines_with_metadata("path", "z.rs").is_empty());
    }

    /// A group holding MANY conversations must surface every one of them.
    ///
    /// `code_reading` declares one conversation per file, so this is the shape
    /// the coding agent actually runs. The resolver used to collapse a group to
    /// its first-registered timeline, which made every other file structurally
    /// unreachable — not scored by provenance, not projectable — no matter how
    /// well it matched the query. Tombstoned conversations still drop out.
    #[test]
    fn group_turns_span_every_conversation_in_the_group() {
        use crate::projection::TurnKey;

        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let (file_a, file_b, retired) = (alloc.next(), alloc.next(), alloc.next());
        let mut sub = Substrate::new();
        for tl in [file_a, file_b, retired] {
            sub.register_timeline(tl, layer, group);
        }
        // Two turns per conversation — note the indices COLLIDE across
        // timelines, which is exactly why a bare `(group, index)` key is
        // ambiguous and the turn identity has to carry the timeline.
        for tl in [file_a, file_b, retired] {
            sub.append_with_blocks(tl, 10, 0, 0);
            sub.append_with_blocks(tl, 10, 0, 0);
        }
        sub.tombstone_timeline(retired);

        let turns = ContentResolver::group_turns(&sub, group);
        assert_eq!(
            turns,
            vec![
                TurnKey::new(file_a, TurnIndex(0)),
                TurnKey::new(file_a, TurnIndex(1)),
                TurnKey::new(file_b, TurnIndex(0)),
                TurnKey::new(file_b, TurnIndex(1)),
            ],
            "both live conversations enumerate in registration order; the \
             tombstoned one drops out",
        );
        // Each key resolves against its OWN conversation.
        for key in &turns {
            assert_eq!(ContentResolver::turn_token_count(&sub, *key), 10);
        }
    }

    // ── Tombstone ─────────────────────────────────────────────────────

    #[test]
    fn tombstone_hides_timeline_from_active_iterator() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let alive = alloc.next();
        let dead = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(alive, layer, group);
        sub.register_timeline(dead, layer, group);

        // Before tombstone: both surface.
        let pre: Vec<TimelineId> = sub.active_timelines_for_group(group).collect();
        assert!(pre.contains(&alive));
        assert!(pre.contains(&dead));

        sub.tombstone_timeline(dead);
        assert!(sub.is_tombstoned(dead));
        assert!(!sub.is_tombstoned(alive));

        // After tombstone: only the alive one.
        let post: Vec<TimelineId> = sub.active_timelines_for_group(group).collect();
        assert_eq!(post, vec![alive]);
    }

    #[test]
    fn tombstone_for_unregistered_timeline_survives_registration() {
        // Walker replay order: a `Tombstone` record can precede the
        // matching `StreamDecl::Turn` (the redo log is append-only
        // and a refresh may write the tombstone before the new
        // generation's turn decls).  The tombstone bit must persist
        // through the eventual registration so the just-registered
        // timeline is observably dead from the start.
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        let mut sub = Substrate::new();

        sub.apply_tombstone(&super::TombstonePayload {
            timeline_id: timeline.raw(),
            turn_index: None,
            turn_writer: 0,
            reason: None,
        });
        assert!(sub.is_tombstoned(timeline));

        sub.register_timeline(timeline, layer, group);
        assert!(sub.is_tombstoned(timeline));
        assert!(sub.active_timelines_for_group(group).all(|t| t != timeline));
    }

    #[test]
    fn tombstoned_timelines_set_includes_unregistered_entries() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let registered = alloc.next();
        let unregistered = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(registered, layer, group);
        sub.tombstone_timeline(registered);
        sub.apply_tombstone(&super::TombstonePayload {
            timeline_id: unregistered.raw(),
            turn_index: None,
            turn_writer: 0,
            reason: None,
        });

        let set = sub.tombstoned_timelines();
        assert!(set.contains(&registered));
        assert!(set.contains(&unregistered));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn turn_scoped_tombstone_kills_only_the_turn_not_the_timeline() {
        let alloc = TimelineAllocator::new();
        let tl = alloc.next();
        let mut sub = Substrate::new();
        // A turn-scoped tombstone (`turn_index = Some`) drops only that turn.
        sub.apply_tombstone(&super::TombstonePayload {
            timeline_id: tl.raw(),
            turn_index: Some(2),
            turn_writer: 0,
            reason: Some("corrupt reload (turn 2): chunk mismatch".to_string()),
        });
        assert!(sub.is_turn_tombstoned(tl, 2));
        assert!(
            !sub.is_turn_tombstoned(tl, 1),
            "a different turn of the same timeline is unaffected"
        );
        assert!(
            !sub.is_tombstoned(tl),
            "the timeline itself stays live — only the turn is dead"
        );
    }

    /// **A live turn tombstone empties the turn in place — the state a reload
    /// gives it.** Marking the set alone left the turn's content in the
    /// timeline, and the live projection reads the content, not the set: a
    /// retired tail stayed selectable until the next restart, and a replay that
    /// retired one run's answer handed it to the next run as history.
    #[test]
    fn a_tombstoned_turn_is_emptied_in_place_as_a_reload_leaves_it() {
        let alloc = TimelineAllocator::new();
        let tl = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(tl, LayerId::for_test(1), GroupId::for_test(1));
        let kept = sub.restore_turn(
            tl,
            TurnLayout::default(),
            TokenBuffer::from(vec![11u32, 12, 13]),
            3,
            None,
            0,
            1,
        );
        let retired = sub.restore_turn(
            tl,
            TurnLayout::default(),
            TokenBuffer::from(vec![21u32, 22]),
            2,
            None,
            1,
            2,
        );

        sub.tombstone_turn(tl, retired.0);

        assert!(sub.is_turn_tombstoned(tl, retired.0));
        assert_eq!(sub.turn_token_count_of(tl, retired), 0);
        assert_eq!(sub.token_ids_of(tl, retired), Vec::<u32>::new());
        assert_eq!(
            sub.turn_indices(tl).collect::<Vec<_>>(),
            vec![kept, retired],
            "the index stays occupied, so later turns keep their stream ids"
        );
        assert_eq!(
            sub.turn_token_count_of(tl, kept),
            3,
            "the kept turn is untouched"
        );
        assert_eq!(sub.token_ids_of(tl, kept), vec![11u32, 12, 13]);
        assert_eq!(
            sub.total_token_count(tl),
            3,
            "the timeline's total counts only what its turns still pin"
        );
    }

    #[test]
    fn corrupt_turn_policy_defaults_to_drop_conversation_and_can_be_set() {
        let mut sub = Substrate::new();
        let layer = LayerId::for_test(9);
        assert_eq!(
            sub.corrupt_turn_policy_for(layer),
            CorruptTurnPolicy::DropConversation,
            "an unset layer defaults to drop_conversation"
        );
        sub.set_layer_corrupt_turn_policy(layer, CorruptTurnPolicy::DropTurn);
        assert_eq!(
            sub.corrupt_turn_policy_for(layer),
            CorruptTurnPolicy::DropTurn
        );
    }

    /// Regression: `snapshot_promotion_state` used to misclassify
    /// every tier-less turn (a turn whose residence has no
    /// hot/warm/cold installed — the design-intended state for
    /// ghost summary turns the substrate-summariser appends via
    /// `append_with_blocks(0..0)`) as `missing`, which used the
    /// same bucket as "substrate has no record of this turn at all."
    /// `elevate_to_hot` then logged a WARN per item, flooding the
    /// production daemon's trace with apparent corruption warnings
    /// every time a parallel-ingest worker submitted a turn whose
    /// projection plan included any of those tree-meta-only nodes.
    ///
    /// The fix splits the two cases — `missing` keeps the "not in
    /// substrate" semantic; `tier_less` is the new bucket for
    /// "in substrate but nothing to elevate."  Both get skipped by
    /// the elevation orchestrator, but only `missing` triggers the
    /// WARN.
    #[test]
    fn snapshot_promotion_state_classifies_tier_less_turns_separately_from_missing() {
        let (_layer, _group, timeline, mut sub) = make_timeline();
        // 4 turns with `block_range = 0..*` but no sealed K/V.
        // `append_with_blocks` allocates a residence slot with
        // hot/warm/cold all None — exactly what the summariser does
        // for ghost summary turns.
        for i in 0..4 {
            sub.append_with_blocks(timeline, 10, i, i + 1);
        }

        let stored_indices: Vec<TurnIndex> = sub.turn_indices(timeline).collect();
        assert_eq!(
            stored_indices,
            vec![TurnIndex(0), TurnIndex(1), TurnIndex(2), TurnIndex(3)],
        );

        let turn_keys: Vec<TurnKey> = stored_indices
            .iter()
            .map(|idx| TurnKey::new(timeline, *idx))
            .collect();
        let plan = sub.snapshot_promotion_state(&[], &turn_keys);

        // Every appended turn is tracked but tier-less — they
        // belong in `tier_less`, NOT `missing`.
        assert!(
            plan.missing.is_empty(),
            "no turn should be reported missing — they're all in the substrate; \
             missing={:?}",
            plan.missing,
        );
        assert_eq!(
            plan.tier_less.len(),
            4,
            "every tier-less ghost turn lands in `tier_less`; tier_less={:?}",
            plan.tier_less,
        );
        assert_eq!(plan.already_hot.len(), 0);
        assert_eq!(plan.warm_to_hot.len(), 0);
        assert_eq!(plan.cold_to_hot.len(), 0);
    }

    /// **A plan shares a warm turn's sequences with its residence**: the very
    /// vector the residence holds, one refcount, not a per-chunk copy — a lift
    /// priced three times over a reprojection copies nothing.
    #[test]
    fn a_promotion_plan_shares_the_residences_warm_sequences() {
        let (_layer, _group, timeline, mut sub) = make_timeline();
        sub.append_with_blocks(timeline, 10, 0, 1);
        let residence = sub.turn(timeline, TurnIndex(0)).unwrap().content.residence;
        sub.install_warm(residence, vec![minimal_sealed_layer()]);

        let plan = sub.snapshot_promotion_state(&[], &[TurnKey::new(timeline, TurnIndex(0))]);
        assert_eq!(plan.warm_to_hot.len(), 1);
        let held = sub.residence(residence).warm().unwrap();
        assert!(Arc::ptr_eq(&plan.warm_to_hot[0].warm, &held));
        assert_eq!(plan.warm_to_hot[0].warm.len(), 1);
    }

    /// Counterpart to the regression test: a TurnKey naming an
    /// index the substrate genuinely has no record of must still
    /// land in `missing` (not `tier_less`), so the WARN in
    /// `elevate_to_hot` still fires for true corruption / stale
    /// projection plans.
    #[test]
    fn snapshot_promotion_state_marks_truly_unknown_turns_as_missing() {
        let (_layer, _group, timeline, mut sub) = make_timeline();
        sub.append_with_blocks(timeline, 10, 0, 1);

        let turn_keys = vec![
            TurnKey::new(timeline, TurnIndex(0)),  // exists, tier-less
            TurnKey::new(timeline, TurnIndex(99)), // does not exist
        ];
        let plan = sub.snapshot_promotion_state(&[], &turn_keys);

        assert_eq!(plan.tier_less.len(), 1);
        assert_eq!(plan.missing.len(), 1);
        match &plan.missing[0] {
            PromotionItemKind::Turn(k) => {
                assert_eq!(k.index, TurnIndex(99));
                assert_eq!(k.timeline, timeline);
            }
            other => panic!("expected Turn variant, got {other:?}"),
        }
    }
}
