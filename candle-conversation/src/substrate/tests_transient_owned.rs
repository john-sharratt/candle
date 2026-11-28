//! Tests which timelines and sections are ephemeral or owned, and what releasing
//! each one frees.
//!
//! A transient timeline keeps neither its K/V nor its recurrent snapshot on disk,
//! and a splice source keeps its hot copy until its fork is tombstoned. Only the
//! owner releases an owned section, which ends its stream, while a shared
//! section's stream is left for every reader. A released section submitted again
//! under its address lands on a new generation and is readable. A persisted
//! section is never removed from memory, since the next load would register it
//! again.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::projection::{GroupId, LayerId, SectionId, TimelineAllocator, TimelineId};

    use crate::recovered::recovered_from;

    use super::super::tests_working_set::tests::make_timeline;

    use super::super::tests_timeline_meta::tests::{identity_migrate, minimal_sealed_layer};

    /// **An ephemeral timeline is also skipped by the recurrent snapshot.**
    ///
    /// The two halves of a turn's durable state cost very differently, and only
    /// one of them scales with tokens. Recurrent state is fixed-size *per
    /// sequence*, so a 19-token calibration question writes the same ~65 MB as a
    /// full trajectory — measured, that is ~195 GB across a 2,998-exemplar
    /// corpus, dwarfing its KV and enough on its own to fill the disk. Flagging
    /// the KV without the snapshot removed a quarter of the writes and left the
    /// failure in place, so this pins the query the snapshot guard reads.
    #[test]
    fn an_ephemeral_timeline_is_marked_for_both_kv_and_snapshot() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let normal = alloc.next();
        let ephemeral = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(normal, layer, group);
        sub.register_timeline(ephemeral, layer, group);
        sub.mark_timeline_transient(ephemeral);

        assert!(
            sub.is_timeline_transient(ephemeral),
            "the snapshot guard must see the mark — without this the recurrent \
             state is still written and the KV flag alone fixes almost nothing"
        );
        assert!(
            !sub.is_timeline_transient(normal),
            "a durable conversation must keep its snapshot: it is what restart \
             and fork-resume read"
        );
    }

    /// A sealed turn of `timeline` holding `user` and `assistant` text.
    pub(in super::super) fn seal_text_turn(
        sub: &mut Substrate,
        timeline: TimelineId,
        user: &str,
        assistant: &str,
    ) {
        let migrate = |_input: &[SealedSequence]| -> candle::Result<Vec<SealedSequence>> {
            Ok(vec![minimal_sealed_layer(), minimal_sealed_layer()])
        };
        let write = TurnPartWrite {
            layout: TurnLayout::from_flat_grid(
                0,
                0,
                0,
                3,
                0,
                0,
                user.to_string(),
                Some(assistant.to_string()),
                false,
            ),
            token_count: 3,
            block_end: 1,
            sealed_gpu: Some(Arc::new(vec![])),
            ..Default::default()
        };
        sub.append_complete(timeline, write, migrate).unwrap();
    }

    /// **A history read from a turn reads only from it on**: of three turns,
    /// reading from turn 1 gives exactly turns 1's and 2's bubbles.
    #[test]
    fn a_history_from_a_turn_holds_only_the_turns_from_it() {
        let alloc = TimelineAllocator::new();
        let tl = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(tl, LayerId::for_test(1), GroupId::for_test(1));
        for n in 0..3 {
            seal_text_turn(
                &mut sub,
                tl,
                &format!("asked {n}"),
                &format!("answered {n}"),
            );
        }
        let got: Vec<(u32, String)> = recovered_from(&sub, tl, false, 1, |_| 0, &[])
            .into_iter()
            .map(|m| (m.turn, m.text))
            .collect();
        let want: Vec<(u32, String)> = [1, 2]
            .into_iter()
            .flat_map(|n| [(n, format!("asked {n}")), (n, format!("answered {n}"))])
            .collect();
        assert_eq!(got, want);
        assert_eq!(recovered_from(&sub, tl, false, 0, |_| 0, &[]).len(), 6);
    }

    /// A transient (scratch-fork) timeline's turns are flagged `no_cold_persist`
    /// and excluded from the warm→cold gather, so their KV is never written to
    /// cold — the fix for the orphaned code_read-fork chunks. A normal timeline's
    /// turn is gathered as usual, and tombstoning the transient fork frees its warm.
    #[test]
    fn transient_fork_turns_never_cold_persist() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let normal_tl = alloc.next();
        let fork_tl = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(normal_tl, layer, group);
        sub.register_timeline(fork_tl, layer, group);
        sub.mark_timeline_transient(fork_tl);

        let migrate = |_input: &[SealedSequence]| -> candle::Result<Vec<SealedSequence>> {
            Ok(vec![minimal_sealed_layer(), minimal_sealed_layer()])
        };
        let write = || TurnPartWrite {
            layout: TurnLayout::from_flat_grid(
                0,
                0,
                0,
                3,
                0,
                0,
                String::new(),
                Some("x".to_string()),
                false,
            ),
            token_count: 3,
            block_end: 1,
            sealed_gpu: Some(Arc::new(vec![])),
            ..Default::default()
        };

        let n_idx = sub.append_complete(normal_tl, write(), migrate).unwrap();
        let f_idx = sub.append_complete(fork_tl, write(), migrate).unwrap();
        let n_res = sub.turn_residence(normal_tl, n_idx).unwrap();
        let f_res = sub.turn_residence(fork_tl, f_idx).unwrap();

        // The fork turn's residence is flagged; the normal one is not.
        assert!(
            !sub.residence(n_res).has_flag(NO_COLD_PERSIST),
            "a normal turn cold-persists"
        );
        assert!(
            sub.residence(f_res).has_flag(NO_COLD_PERSIST),
            "a transient fork turn is flagged no_cold_persist"
        );

        // Stage both in the warm tier — the warm→cold gather's input.
        sub.install_warm(n_res, vec![minimal_sealed_layer(), minimal_sealed_layer()]);
        sub.install_warm(f_res, vec![minimal_sealed_layer(), minimal_sealed_layer()]);

        // The gather includes the normal residence, excludes the transient fork.
        let pending = sub.snapshot_pending_cold();
        assert!(
            pending.iter().any(|(r, _, _)| *r == n_res),
            "normal turn gathered for cold"
        );
        assert!(
            !pending.iter().any(|(r, _, _)| *r == f_res),
            "transient fork turn NEVER gathered for cold (no orphan)"
        );
        assert_eq!(
            sub.pending_cold_count(),
            1,
            "only the normal turn awaits a cold write"
        );

        // Tombstoning the fork frees its warm RAM — no cold copy will ever land, so
        // `evict_when_cold`'s warm-drop-on-cold-land never fires.
        sub.tombstone_timeline(fork_tl);
        assert!(
            !sub.residence(f_res).has_warm(),
            "transient fork's warm freed at tombstone"
        );
    }

    /// A code_read scope fork's turns are splice sources: `adopt_turn` reads the
    /// hot copy and nothing else, so no automatic hot-drop may take it before the
    /// fork is tombstoned — even with its warm copy landed and every eviction
    /// path asked for everything. An ordinary turn in the same state IS evicted
    /// (the control), and the fork's tombstone frees its hot copy.
    #[test]
    fn a_splice_source_keeps_its_hot_copy_until_tombstoned() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let normal_tl = alloc.next();
        let fork_tl = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(normal_tl, layer, group);
        sub.register_timeline(fork_tl, layer, group);
        sub.mark_timeline_transient(fork_tl);
        sub.mark_timeline_splice_source(fork_tl);

        let migrate = |_input: &[SealedSequence]| -> candle::Result<Vec<SealedSequence>> {
            Ok(vec![minimal_sealed_layer(), minimal_sealed_layer()])
        };
        let write = || TurnPartWrite {
            layout: TurnLayout::from_flat_grid(
                0,
                0,
                0,
                3,
                0,
                0,
                String::new(),
                Some("x".to_string()),
                false,
            ),
            token_count: 3,
            block_end: 1,
            sealed_gpu: Some(Arc::new(vec![])),
            ..Default::default()
        };
        let n_idx = sub.append_complete(normal_tl, write(), migrate).unwrap();
        let f_idx = sub.append_complete(fork_tl, write(), migrate).unwrap();
        let n_res = sub.turn_residence(normal_tl, n_idx).unwrap();
        let f_res = sub.turn_residence(fork_tl, f_idx).unwrap();
        assert!(sub.residence(f_res).has_flag(SPLICE_SOURCE));
        assert!(!sub.residence(n_res).has_flag(SPLICE_SOURCE));

        // Both warm-backed, so every hot-drop path's precondition holds.
        sub.install_warm(n_res, vec![minimal_sealed_layer(), minimal_sealed_layer()]);
        sub.install_warm(f_res, vec![minimal_sealed_layer(), minimal_sealed_layer()]);

        sub.evict_hot_to_free(&[], &[], u64::MAX);
        assert!(
            sub.turn_sealed_of(normal_tl, n_idx).is_none(),
            "control: relief evicts an ordinary warm-backed turn"
        );
        assert!(
            sub.turn_sealed_of(fork_tl, f_idx).is_some(),
            "relief must not take a splice source"
        );

        let fork_key = TurnKey {
            timeline: fork_tl,
            index: f_idx,
        };
        sub.evict_hot_except(&[], &[]);
        sub.demote_idle_hot(0);
        sub.demote_turns_to_warm(&[fork_key]);
        let ingest: HashSet<TimelineId> = [fork_tl].into_iter().collect();
        sub.demote_cold_ingest(&ingest, &[], &[], 0, u64::MAX);
        sub.mark_timeline_evict_when_cold(fork_tl);
        assert!(
            sub.turn_sealed_of(fork_tl, f_idx).is_some(),
            "no demote path takes a splice source"
        );

        sub.tombstone_timeline(fork_tl);
        assert!(
            sub.turn_sealed_of(fork_tl, f_idx).is_none(),
            "the fork's tombstone frees its hot copy"
        );
        assert!(!sub.residence(f_res).has_flag(SPLICE_SOURCE));
    }

    /// `append_complete` calls the migration closure and installs the
    /// result into the assistant residence's hot tier — `turn_sealed_of`
    /// returns it.
    #[test]
    fn append_complete_stores_migrated_result() {
        let (_, _, timeline, mut sub) = make_timeline();

        // Migration that returns a 2-layer minimal sealed sequence
        // (non-empty so it crosses install_hot's `!is_empty()` gate
        // and the residence lands hot).
        let migrate = |_input: &[SealedSequence]| -> candle::Result<Vec<SealedSequence>> {
            Ok(vec![minimal_sealed_layer(), minimal_sealed_layer()])
        };

        let idx = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    layout: TurnLayout::from_flat_grid(
                        0,
                        0,
                        0,
                        3,
                        0,
                        0,
                        String::new(),
                        Some("hello".to_string()),
                        false,
                    ),
                    token_count: 3,
                    block_end: 1,
                    sealed_gpu: Some(Arc::new(vec![])),
                    ..Default::default()
                },
                migrate,
            )
            .unwrap();

        let stored = sub.turn_sealed_of(timeline, idx).unwrap();
        assert_eq!(stored.len(), 2, "two layers installed");
    }

    /// `set_section_full` installs the sealed layers into the section
    /// residence's hot tier.
    #[test]
    fn set_section_full_stores_its_sealed_layers() {
        let mut sub = Substrate::new();
        let section = SectionId::new(42);

        sub.set_section_full(
            section,
            StreamId::default(),
            10,
            vec![minimal_sealed_layer(), minimal_sealed_layer()],
            Arc::new(vec![1u32, 2, 3]),
        );

        let stored = sub.section_sealed_of(section).expect("hot installed");
        assert_eq!(stored.len(), 2, "two layers installed");
        assert_eq!(
            sub.sections.get(&section).unwrap().tokens.as_slice(),
            &[1u32, 2, 3]
        );
    }

    /// Install `section` with one hot and one warm layer.
    pub(in super::super) fn install_section_hot_and_warm(
        sub: &mut Substrate,
        section: SectionId,
    ) -> ResidenceIndex {
        let r = sub.set_section_full(
            section,
            StreamId::default(),
            10,
            vec![minimal_sealed_layer()],
            Arc::new(vec![1u32]),
        );
        sub.install_warm(r, vec![minimal_sealed_layer()]);
        r
    }

    /// **A transient section never reaches the disk.** Marked before it
    /// installs, its residence is `no_cold_persist`, so its warm copy is never
    /// counted as awaiting a cold write.
    #[test]
    fn a_transient_section_is_installed_never_to_cold_persist() {
        let mut sub = Substrate::new();
        let (kept, scratch) = (SectionId::new(7), SectionId::new(8));
        sub.mark_section_transient(scratch);
        let kept_r = install_section_hot_and_warm(&mut sub, kept);
        let scratch_r = install_section_hot_and_warm(&mut sub, scratch);

        assert!(!sub.residence(kept_r).has_flag(NO_COLD_PERSIST));
        assert!(sub.residence(scratch_r).has_flag(NO_COLD_PERSIST));
        assert_eq!(
            sub.pending_cold_count(),
            1,
            "only the kept section awaits a cold write"
        );
        assert!(sub.is_section_transient(scratch) && !sub.is_section_transient(kept));
    }

    /// Marked after it installed, the residence it already has is flagged.
    #[test]
    fn marking_an_installed_section_flags_its_residence() {
        let mut sub = Substrate::new();
        let s = SectionId::new(9);
        let r = install_section_hot_and_warm(&mut sub, s);
        sub.mark_section_transient(s);
        assert!(sub.residence(r).has_flag(NO_COLD_PERSIST));
    }

    /// **Retiring a transient section removes it and frees its KV**; a
    /// persisted one is refused, because the next load would register it again.
    #[test]
    fn only_a_transient_section_is_retired() {
        let mut sub = Substrate::new();
        let (kept, scratch) = (SectionId::new(11), SectionId::new(12));
        sub.mark_section_transient(scratch);
        install_section_hot_and_warm(&mut sub, kept);
        let r = install_section_hot_and_warm(&mut sub, scratch);
        assert_eq!(sub.section_token_total, 20);

        assert!(
            !sub.retire_section(kept),
            "a persisted section is not retired"
        );
        assert!(sub.section_exists(kept));

        assert!(sub.retire_section(scratch));
        assert!(!sub.section_exists(scratch));
        assert!(!sub.residence(r).has_hot() && !sub.residence(r).has_warm());
        assert!(
            !sub.residences.warm_members().contains(&r),
            "its warm copy left the warm set with it"
        );
        assert!(!sub.is_section_transient(scratch));
        assert_eq!(sub.section_token_total, 10, "its tokens left the total");
        assert!(
            !sub.retire_section(scratch),
            "a second retire finds nothing"
        );
    }

    /// Install `section` under its own `stream` with one hot and one warm layer.
    pub(in super::super) fn install_owned(
        sub: &mut Substrate,
        owner: TimelineId,
        name: &str,
        stream: u64,
    ) -> (SectionId, ResidenceIndex) {
        let section = sub.allocate_owned_section().unwrap();
        sub.register_owned_section(owner, name, section);
        let r = sub.set_section_full(
            section,
            StreamId(stream),
            10,
            vec![minimal_sealed_layer()],
            Arc::new(vec![1u32]),
        );
        sub.install_warm(r, vec![minimal_sealed_layer()]);
        (section, r)
    }

    pub(in super::super) fn owner(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    /// **Releasing an owned section ends its stream and removes it** — entry,
    /// token total, hot and warm K/V — and hands back the stream so the caller
    /// can write the on-disk tombstone. Another owner's section is untouched.
    #[test]
    fn releasing_an_owned_section_ends_its_stream_and_frees_it() {
        let mut sub = Substrate::new();
        let (mine, theirs) = (owner(1), owner(2));
        let (a, ra) = install_owned(&mut sub, mine, "a", 101);
        let (b, rb) = install_owned(&mut sub, theirs, "b", 102);
        assert_eq!(sub.section_token_total, 20);

        assert_eq!(
            sub.release_owned_section(mine, a),
            Some((StreamId(101), None)),
            "an undeclared stream ends unstamped"
        );

        assert!(!sub.section_exists(a));
        assert!(sub.is_section_tombstoned(StreamId(101)));
        assert!(!sub.residence(ra).has_hot() && !sub.residence(ra).has_warm());
        assert!(
            !sub.residences.warm_members().contains(&ra)
                && !sub.residences.hot_members().contains(&ra)
        );
        assert_eq!(sub.section_token_total, 10);
        assert!(sub.owned_sections_of(mine).is_empty());

        assert!(sub.section_exists(b), "another owner's section survives");
        assert!(!sub.is_section_tombstoned(StreamId(102)));
        assert!(sub.residence(rb).has_hot() && sub.residence(rb).has_warm());
        assert_eq!(sub.owned_section_named(theirs, "b"), Some(b));
    }

    /// Declare `stream` a prompt section under `debug_name`.
    pub(in super::super) fn declare_section(sub: &mut Substrate, stream: u64, debug_name: &str) {
        use crate::persistence::streams::{ContentAddress, SectionDecl};
        sub.apply_stream_decl(
            StreamId(stream),
            StreamDecl::PromptSection(SectionDecl {
                address: ContentAddress::default(),
                debug_name: debug_name.to_string(),
            }),
            RecordIdentity::NONE,
        );
    }

    /// **Releasing a shared section ends only this conversation's hold**: its
    /// entry and claim go, its stream is not ended — another conversation's
    /// section on the same stream keeps its resident K/V — and nothing is handed
    /// back to tombstone on disk (`docs/shared_journal_sections.md` §6).
    #[test]
    fn releasing_a_shared_section_leaves_its_stream_to_every_reader() {
        let mut sub = Substrate::new();
        declare_section(&mut sub, 301, "shared/journal/1");
        let (a, ra) = install_owned(&mut sub, owner(1), "journal/1", 301);
        let (b, rb) = install_owned(&mut sub, owner(2), "journal/1", 301);
        assert!(sub.is_shared_section_stream(StreamId(301)));
        assert_eq!(sub.shared_section_streams(), vec![StreamId(301)]);
        assert_eq!(
            sub.held_section_streams(),
            HashSet::from([StreamId(301)]),
            "both entries read the one stream"
        );

        assert_eq!(sub.release_owned_section(owner(1), a), None);
        assert!(!sub.section_exists(a));
        assert!(!sub.residence(ra).has_warm());
        assert!(!sub.is_section_tombstoned(StreamId(301)));
        assert!(sub.section_exists(b), "the other reader's section survives");
        assert!(sub.residence(rb).has_hot() && sub.residence(rb).has_warm());
        assert!(sub.owned_sections_of(owner(1)).is_empty());

        assert_eq!(sub.release_owned_section(owner(2), b), None);
        assert!(sub.held_section_streams().is_empty());
        assert_eq!(
            sub.shared_section_streams(),
            vec![StreamId(301)],
            "the stream stays for the next conversation, and the sweep"
        );
    }

    /// **The sweep never chooses a knowledge-base member.** A member is a
    /// shared stream (`shared/files/…`) that nothing holds after a restart;
    /// offered to the sweep, it was retired as a superseded journal section.
    #[test]
    fn a_knowledge_base_member_is_no_sweep_candidate() {
        let mut sub = Substrate::new();
        declare_section(&mut sub, 301, "shared/journal/1");
        declare_section(&mut sub, 302, "shared/files/candle/zend/src/lib.rs@b#1-200");
        declare_section(&mut sub, 303, "conv/7/journal/1");
        assert!(sub.is_shared_section_stream(StreamId(302)));
        assert_eq!(sub.shared_section_streams(), vec![StreamId(301)]);
    }

    /// **A section's end state goes with its generation**: the snapshot tail
    /// under its end-state stream is dropped when the section stream ends, and a
    /// later generation keeps one again.
    #[test]
    fn a_sections_end_state_goes_with_its_generation() {
        use crate::persistence::segment::SegmentId;
        let mut sub = Substrate::new();
        declare_section(&mut sub, 401, "shared/journal/close");
        let end = section_end_state_stream_id(StreamId(401));
        let at = |offset| RecordLoc {
            segment: SegmentId(1),
            offset,
            payload_len: 64,
            record_size: 128,
        };
        assert!(sub.apply_snapshot_loc(end, at(0), RecordIdentity::NONE));
        assert_eq!(sub.recurrent_snapshot_loc(end), Some(at(0)));

        sub.tombstone_section(StreamId(401));
        assert_eq!(sub.recurrent_snapshot_loc(end), None);

        assert!(
            sub.apply_snapshot_loc(end, at(4096), RecordIdentity::NONE),
            "not retired for good"
        );
        assert_eq!(sub.recurrent_snapshot_loc(end), Some(at(4096)));
    }

    /// Only the owner can release: a section is never torn down by a
    /// conversation that merely knows its id.
    #[test]
    fn only_the_owner_releases_an_owned_section() {
        let mut sub = Substrate::new();
        let (a, _) = install_owned(&mut sub, owner(1), "a", 101);
        assert_eq!(sub.release_owned_section(owner(2), a), None);
        assert!(sub.section_exists(a));
        assert!(!sub.is_section_tombstoned(StreamId(101)));
        assert_eq!(
            sub.release_owned_section(owner(1), a),
            Some((StreamId(101), None))
        );
        assert_eq!(
            sub.release_owned_section(owner(1), a),
            None,
            "a second release finds nothing"
        );
    }

    /// A section that was submitted but never sealed has no stream to end; its
    /// claim is still dropped.
    #[test]
    fn releasing_an_unsealed_owned_section_drops_only_the_claim() {
        let mut sub = Substrate::new();
        let section = sub.allocate_owned_section().unwrap();
        sub.register_owned_section(owner(1), "pending", section);
        assert_eq!(sub.release_owned_section(owner(1), section), None);
        assert!(sub.owned_sections_of(owner(1)).is_empty());
    }

    /// Releasing everything an owner holds returns every stream, in submission
    /// order, and touches nothing else.
    #[test]
    fn releasing_all_of_an_owners_sections_returns_their_streams() {
        let mut sub = Substrate::new();
        let (mine, theirs) = (owner(1), owner(2));
        let (a, _) = install_owned(&mut sub, mine, "a", 101);
        let (b, _) = install_owned(&mut sub, mine, "b", 102);
        let (c, _) = install_owned(&mut sub, theirs, "c", 103);

        assert_eq!(
            sub.release_owned_sections(mine),
            vec![(StreamId(101), None), (StreamId(102), None)]
        );
        assert!(!sub.section_exists(a) && !sub.section_exists(b));
        assert!(sub.section_exists(c));
        assert_eq!(sub.section_token_total, 10);
        assert!(sub.release_owned_sections(mine).is_empty());
    }

    /// Removing a section and submitting the same name again lands on a new
    /// generation of the same stream: the tombstone lifts when the stream is
    /// declared again, so the second section is readable.
    #[test]
    fn a_released_owned_section_can_be_sealed_again_under_its_address() {
        let mut sub = Substrate::new();
        use crate::persistence::streams::{ContentAddress, SectionDecl};

        let (a, _) = install_owned(&mut sub, owner(1), "a", 101);
        sub.release_owned_section(owner(1), a);
        let (again, _) = install_owned(&mut sub, owner(1), "a", 101);
        assert_ne!(a, again, "an id is never reused");
        assert!(
            !sub.section_exists(again),
            "the stream stays ended until the new generation is declared"
        );
        sub.apply_stream_decl(
            StreamId(101),
            StreamDecl::PromptSection(SectionDecl {
                address: ContentAddress::default(),
                debug_name: "conv/1/a".to_string(),
            }),
            RecordIdentity::NONE,
        );
        assert!(sub.section_exists(again));
    }

    /// An owner's persisted streams are found from their declarations alone —
    /// no ownership registry — by name or all together, and never another
    /// owner's, including one whose id begins with the same digits.
    #[test]
    fn an_owners_persisted_streams_are_found_by_their_declared_name() {
        use crate::persistence::streams::{ContentAddress, SectionDecl};
        let mut sub = Substrate::new();
        for (stream, name) in [
            (201, "conv/1/mission"),
            (202, "conv/1/mission"),
            (203, "conv/1/note"),
            (204, "conv/12/mission"),
            (205, "tool_catalog"),
        ] {
            sub.apply_stream_decl(
                StreamId(stream),
                StreamDecl::PromptSection(SectionDecl {
                    address: ContentAddress::default(),
                    debug_name: name.to_string(),
                }),
                RecordIdentity::NONE,
            );
        }
        let one = owner(1);
        assert_eq!(
            sub.persisted_owned_streams(one, Some("mission")),
            vec![StreamId(201), StreamId(202)]
        );
        assert_eq!(
            sub.persisted_owned_streams(one, None),
            vec![StreamId(201), StreamId(202), StreamId(203)]
        );
        assert_eq!(
            sub.persisted_owned_streams(owner(12), None),
            vec![StreamId(204)]
        );

        sub.tombstone_section(StreamId(201));
        assert_eq!(
            sub.persisted_owned_streams(one, Some("mission")),
            vec![StreamId(202)],
            "a tombstoned stream is no longer the owner's"
        );
    }

    /// After `reset()`, the substrate is empty.
    #[test]
    fn reset_clears_substrate() {
        let (_, _, timeline, mut sub) = make_timeline();

        sub.append_complete(
            timeline,
            TurnPartWrite {
                sealed_gpu: Some(Arc::new(vec![])),
                ..Default::default()
            },
            identity_migrate,
        )
        .unwrap();

        sub.reset();
        assert_eq!(sub.turn_count(timeline), 0);
        assert!(sub.turn_sealed_of(timeline, TurnIndex(0)).is_none());
    }

    /// `total_token_count(timeline)` is this conversation's turn tokens plus the
    /// shared section tokens, served from the O(1) maintained counters.
    #[test]
    fn total_token_count_sums_turns_and_sections() {
        let (_, _, timeline, mut sub) = make_timeline();
        for tc in [30usize, 12] {
            sub.append_complete(
                timeline,
                TurnPartWrite {
                    token_count: tc,
                    sealed_gpu: Some(Arc::new(vec![])),
                    ..Default::default()
                },
                identity_migrate,
            )
            .unwrap();
        }
        // A 10-token ingested section adds to the total.
        sub.set_section_full(
            SectionId::new(7),
            StreamId::default(),
            10,
            vec![minimal_sealed_layer()],
            Arc::new(vec![1u32, 2, 3]),
        );
        assert_eq!(sub.total_token_count(timeline), 30 + 12 + 10);
    }

    /// The maintained counter handles section re-ingest (overwrite, not add)
    /// and turn extension without drifting.
    #[test]
    fn total_token_count_handles_overwrite_and_extend() {
        let (_, _, timeline, mut sub) = make_timeline();
        sub.append_complete(
            timeline,
            TurnPartWrite {
                token_count: 20,
                sealed_gpu: Some(Arc::new(vec![])),
                ..Default::default()
            },
            identity_migrate,
        )
        .unwrap();
        sub.set_section_full(
            SectionId::new(9),
            StreamId::default(),
            100,
            vec![minimal_sealed_layer()],
            Arc::new(vec![1u32]),
        );
        assert_eq!(sub.total_token_count(timeline), 120);
        // Re-ingest the SAME section smaller — replaces its 100 with 60.
        sub.set_section_full(
            SectionId::new(9),
            StreamId::default(),
            60,
            vec![minimal_sealed_layer()],
            Arc::new(vec![1u32]),
        );
        assert_eq!(sub.total_token_count(timeline), 80);
        // Extend the turn by 5 tokens.
        sub.extend_turn(timeline, TurnIndex(0), 5, 2);
        assert_eq!(sub.total_token_count(timeline), 85);
    }
}
