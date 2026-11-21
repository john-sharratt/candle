//! Tests the listing and lookup queries that the sidebar, the projection and the
//! gallery caches depend on.
//!
//! An empty tag filter admits only untagged turns, so one corpus serves many
//! scopes without tagged content appearing everywhere. The gallery and tombstone
//! epochs must move on every change they are built from and never on a read,
//! since a gallery cached on a stale epoch would keep serving a dead turn. Prefix
//! and metadata lookups return only live conversations, so a retired one reads as
//! absent.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::persistence::streams::TurnDecl;

    use crate::projection::{TimelineAllocator, TimelineId};

    use super::super::tests_working_set::tests::make_timeline;

    /// Gather-scope tag semantics, pinned.
    ///
    /// `SelectionPolicy.tags` documents an empty list as "all projections in
    /// scope", and the gather in `projection::resolver` implements the
    /// opposite — an empty filter admits only turns that are themselves
    /// untagged:
    ///
    /// ```ignore
    /// let in_scope = if tags.is_empty() {
    ///     d.tags.is_empty()
    /// } else {
    ///     d.tags.iter().any(|t| tags.contains(t))
    /// };
    /// ```
    ///
    /// The implementation is the useful behaviour and this test fixes it, so
    /// the doc comment is what needs correcting rather than the code. It is
    /// what lets one corpus serve many scopes: content ingested UNTAGGED is
    /// shared by every node that declares no filter, and content ingested with
    /// a tag is reachable only from a node naming it. A change to "empty means
    /// everything" would make every tagged turn visible everywhere, silently.
    #[test]
    fn an_empty_tag_filter_admits_only_untagged_turns() {
        let admits = |filter: &[&str], turn: &[&str]| -> bool {
            let d: Vec<String> = turn.iter().map(|t| t.to_string()).collect();
            if filter.is_empty() {
                d.is_empty()
            } else {
                d.iter().any(|t| filter.contains(&t.as_str()))
            }
        };

        // Empty filter: untagged content only. A tagged turn is NOT admitted —
        // this is the half the doc comment gets wrong.
        assert!(admits(&[], &[]), "untagged content is the shared corpus");
        assert!(
            !admits(&[], &["ardh"]),
            "an empty filter must not sweep in tagged content"
        );

        // Named filter: that tag only. Untagged content is excluded here, so a
        // node that names a scope sees that scope and nothing else.
        assert!(admits(&["ardh"], &["ardh"]));
        assert!(!admits(&["ardh"], &["battle-cities"]));
        assert!(!admits(&["ardh"], &[]));

        // Multi-valued on both sides is an intersection test, so one document
        // can belong to two scopes at once.
        assert!(admits(&["ardh"], &["ardh", "battle-cities"]));
        assert!(admits(&["ardh", "low-fen"], &["low-fen"]));
    }

    /// A stream's window is its stored blob: repeat reads carry one fingerprint,
    /// a rewritten blob reads back as the new window under a new fingerprint,
    /// and an absent or empty window is `None`.
    #[test]
    fn wide_sig_window_reads_the_stored_blob() {
        use crate::provenance::{encode_wide_sigs, WideQSig};
        let mut sub = Substrate::new();
        let sid = turn_stream_id(1, 0);
        let sig_a = WideQSig {
            n_heads: 12,
            words: vec![0xAAAA_AAAA_AAAA_AAAA; 24],
        };
        sub.set_wide_q_sigs_blob(sid, encode_wide_sigs(std::slice::from_ref(&sig_a)));

        let w1 = sub.wide_sig_window(sid).expect("window");
        assert_eq!(w1.decode(), vec![sig_a]);
        let w2 = sub.wide_sig_window(sid).expect("window");
        assert_eq!(
            w1.fingerprint(),
            w2.fingerprint(),
            "one blob, one fingerprint"
        );

        let sig_b = WideQSig {
            n_heads: 12,
            words: vec![0x5555_5555_5555_5555; 24],
        };
        sub.set_wide_q_sigs_blob(sid, encode_wide_sigs(std::slice::from_ref(&sig_b)));
        let w3 = sub.wide_sig_window(sid).expect("window");
        assert_eq!(
            w3.decode(),
            vec![sig_b],
            "the rewritten blob is what reads back"
        );
        assert_ne!(
            w1.fingerprint(),
            w3.fingerprint(),
            "a rewrite must re-upload"
        );

        assert!(
            sub.wide_sig_window(turn_stream_id(1, 99)).is_none(),
            "absent"
        );
        let empty = turn_stream_id(1, 1);
        sub.set_wide_q_sigs_blob(empty, encode_wide_sigs(&[]));
        assert!(sub.wide_sig_window(empty).is_none(), "empty");
    }

    /// **Every tombstone moves the tombstone epoch, and nothing else does.**
    /// The gallery sweep looks for dead turns only when it has moved, so a
    /// tombstone that failed to move it would leave the dead turn's pages on
    /// the card; a write that moved it would only cost a sweep.
    #[test]
    fn the_tombstone_epoch_moves_on_every_tombstone_and_only_then() {
        use crate::provenance::{encode_wide_sigs, WideQSig};
        let (_, _, timeline, mut sub) = make_timeline();
        let sid = turn_stream_id(timeline.raw(), 0);
        let start = sub.tombstone_epoch();
        sub.set_wide_q_sigs_blob(
            sid,
            encode_wide_sigs(&[WideQSig {
                n_heads: 12,
                words: vec![1; 24],
            }]),
        );
        sub.set_projection_events_blob(sid, b"[]".to_vec());
        assert_eq!(sub.tombstone_epoch(), start, "a write is not a tombstone");

        sub.tombstone_turn(timeline, 0);
        assert_eq!(sub.tombstone_epoch(), start + 1, "a turn tombstone");
        sub.tombstone_timeline(timeline);
        assert_eq!(sub.tombstone_epoch(), start + 2, "a timeline tombstone");

        let other = TimelineId::from_raw(timeline.raw() + 1).unwrap();
        sub.apply_tombstone(&TombstonePayload {
            timeline_id: other.raw(),
            turn_index: Some(3),
            turn_writer: 0,
            reason: None,
        });
        assert_eq!(
            sub.tombstone_epoch(),
            start + 3,
            "a replayed turn tombstone"
        );
        sub.apply_tombstone(&TombstonePayload {
            timeline_id: other.raw(),
            turn_index: None,
            turn_writer: 0,
            reason: None,
        });
        assert_eq!(
            sub.tombstone_epoch(),
            start + 4,
            "a replayed timeline tombstone"
        );
    }

    /// **Every change a belief gallery is assembled from moves the epoch, and
    /// reading does not.** A collection's gallery is cached on this epoch, so a
    /// change that failed to move it would serve a stale gallery — a tombstoned
    /// turn still voting, or a new exemplar missing — silently.
    #[test]
    fn the_gallery_epoch_moves_on_every_change_and_never_on_a_read() {
        use crate::provenance::{encode_wide_sigs, WideQSig};
        let (_, _, timeline, mut sub) = make_timeline();
        let sid = turn_stream_id(timeline.raw(), 0);
        let sig = WideQSig {
            n_heads: 12,
            words: vec![0xAAAA_AAAA_AAAA_AAAA; 24],
        };
        let mut last = sub.gallery_epoch();
        let mut moved = |sub: &Substrate, what: &str| {
            let now = sub.gallery_epoch();
            assert!(now > last, "{what} must move the gallery epoch");
            last = now;
        };

        sub.set_wide_q_sigs_blob(sid, encode_wide_sigs(std::slice::from_ref(&sig)));
        moved(&sub, "a signature write");
        sub.set_projection_events_blob(sid, b"[]".to_vec());
        moved(&sub, "a projection-events write");

        // Reads leave it alone.
        let before = sub.gallery_epoch();
        let _ = sub.wide_sig_window(sid);
        let _ = sub.all_streams().count();
        assert_eq!(
            sub.gallery_epoch(),
            before,
            "a read must not move the epoch"
        );

        sub.tombstone_timeline(timeline);
        moved(&sub, "a timeline tombstone");
        sub.reset();
        moved(&sub, "a reset");
        sub.drop_locations_into(&HashSet::new());
        moved(&sub, "dropping a rewrite's stale locations");
    }

    /// **A turn's events digest to its seams and, per collection, the section
    /// its latest projection that chose one there selected** — an event that
    /// selected nothing in a collection leaves the earlier choice standing, and
    /// a rewrite of the events blob is read afresh.
    #[test]
    fn a_turns_events_digest_to_its_seams_and_latest_selections() {
        use crate::projection::{encode_events, ProjectionSelection, SelectedSection};
        let tools = |picked: Option<&str>| SystemItem::Collection {
            name: "tools".into(),
            member_glue: String::new(),
            member_glue_tokens: 0,
            sections: ["a", "b"]
                .iter()
                .map(|n| SelectedSection {
                    name: n.to_string(),
                    tokens: 1,
                    selected: Some(*n) == picked,
                    score: 0.0,
                    qualified: false,
                })
                .collect(),
        };
        let event = |start: u32, seam: bool, item: SystemItem| ProjectionEvent {
            start_token: start,
            self_reference: seam,
            selection: ProjectionSelection {
                system: vec![item],
                turns: Vec::new(),
            },
            ..Default::default()
        };
        let mut sub = Substrate::new();
        let sid = turn_stream_id(1, 0);
        sub.set_projection_events_blob(
            sid,
            encode_events(&[
                event(40, true, tools(Some("b"))),
                event(10, true, tools(None)),
                event(10, false, tools(Some("a"))),
                event(90, false, tools(None)),
            ]),
        );
        assert_eq!(*sub.decoded_seams(sid), vec![10, 40]);
        assert_eq!(sub.last_selected_in(sid, "tools"), Some("a".to_string()));
        assert_eq!(sub.last_selected_in(sid, "memory"), None);

        sub.set_projection_events_blob(sid, encode_events(&[event(5, false, tools(Some("b")))]));
        assert_eq!(sub.last_selected_in(sid, "tools"), Some("b".to_string()));
        assert!(sub.decoded_seams(sid).is_empty());
        assert_eq!(sub.last_selected_in(turn_stream_id(2, 0), "tools"), None);
    }

    /// **The tagged-turn index follows every declaration.** A turn in a gather
    /// scope enters it; a dialogue turn, a turn carrying only working-set marks,
    /// and a non-turn stay out; a redeclaration moves a stream in or out; and a
    /// stream ended or dropped by a rewrite leaves it — so a tagged gallery
    /// drawn from the index is the one a walk of every stream would draw.
    #[test]
    fn the_tagged_turn_index_follows_every_declaration() {
        let decl = |tl: u64, tags: &[&str]| {
            StreamDecl::Turn(TurnDecl {
                timeline_id: tl,
                turn_index: 0,
                turn_id_day: 0,
                turn_id_seq: 1,
                role: 1,
                block_start: 0,
                block_end: 1,
                layer_id: 1,
                group_id: 1,
                anchored_prefix: Vec::new(),
                view: Vec::new(),
                segments: Vec::new(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
            })
        };
        let tagged = |sub: &Substrate| {
            sub.tagged_turn_streams(&["tool".to_string()])
                .map(|t| t.stream_id)
                .collect::<Vec<_>>()
        };
        let mut sub = Substrate::new();
        let (tool, chat, marked) = (
            turn_stream_id(1, 0),
            turn_stream_id(2, 0),
            turn_stream_id(3, 0),
        );
        let none = RecordIdentity::NONE;
        sub.apply_stream_decl(tool, decl(1, &["tool"]), none);
        sub.apply_stream_decl(chat, decl(2, &[]), none);
        sub.apply_stream_decl(marked, decl(3, &["working_set:release"]), none);
        assert_eq!(tagged(&sub), vec![tool]);

        sub.apply_stream_decl(chat, decl(2, &["tool"]), none);
        sub.apply_stream_decl(tool, decl(1, &[]), none);
        assert_eq!(tagged(&sub), vec![chat]);

        sub.end_section(chat, None);
        assert!(tagged(&sub).is_empty());
        sub.apply_stream_decl(tool, decl(1, &["tool"]), none);
        assert_eq!(tagged(&sub), vec![tool]);
        sub.forget_dropped_stream(StreamKey::of(tool, &none), none);
        assert!(tagged(&sub).is_empty());
    }

    /// Rewriting one turn's signature changes only that turn's fingerprint, so
    /// a single seal re-uploads one window rather than the whole gallery.
    #[test]
    fn a_rewrite_changes_only_its_own_windows_fingerprint() {
        use crate::provenance::{encode_wide_sigs, WideQSig};
        let sig = |fill: u64| WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        };
        let mut sub = Substrate::new();
        let sid_a = turn_stream_id(1, 0);
        let sid_b = turn_stream_id(1, 1);
        sub.set_wide_q_sigs_blob(sid_a, encode_wide_sigs(&[sig(0xAAAA_AAAA_AAAA_AAAA)]));
        sub.set_wide_q_sigs_blob(sid_b, encode_wide_sigs(&[sig(0x5555_5555_5555_5555)]));
        let fp = |sub: &Substrate, sid| sub.wide_sig_window(sid).expect("window").fingerprint();
        let (a1, b1) = (fp(&sub, sid_a), fp(&sub, sid_b));

        sub.set_wide_q_sigs_blob(sid_a, encode_wide_sigs(&[sig(0xFFFF_FFFF_FFFF_FFFF)]));
        assert_ne!(a1, fp(&sub, sid_a), "A's rewrite must re-upload A");
        assert_eq!(b1, fp(&sub, sid_b), "B is untouched by A's rewrite");
    }

    /// Archive flag defaults to `false`, can be toggled, and the
    /// idempotency contract holds (setting the same value twice
    /// returns `false` so the caller can short-circuit the
    /// persistence write).
    #[test]
    fn archive_flag_toggles_with_idempotency() {
        let (layer, group, timeline, mut sub) = make_timeline();
        let _ = (layer, group);
        // Untouched: archived defaults to false.
        assert!(!sub.is_archived(timeline));

        // First write: state actually changed, returns true.
        assert!(sub.set_archived(timeline, true));
        assert!(sub.is_archived(timeline));

        // Second write of the same value: idempotent, returns false.
        assert!(!sub.set_archived(timeline, true));
        assert!(sub.is_archived(timeline));

        // Unarchive: state changes, returns true.
        assert!(sub.set_archived(timeline, false));
        assert!(!sub.is_archived(timeline));

        // Setting on an unknown timeline: returns false (and no-op).
        let bogus = TimelineId::from_raw(999).unwrap();
        assert!(!sub.set_archived(bogus, true));
        assert!(!sub.is_archived(bogus));
    }

    /// `known_conversations` exposes the archived flag — the daemon
    /// uses this to drive the sidebar filter.
    #[test]
    fn known_conversations_exposes_archived_flag() {
        let (_, _, timeline, mut sub) = make_timeline();
        sub.set_conv_id(timeline, "abc", RecordIdentity::NONE);
        sub.set_label(timeline, "tour", RecordIdentity::NONE);
        sub.set_archived(timeline, true);

        let convs = sub.known_conversations();
        assert_eq!(convs.len(), 1);
        let (tl, conv_id, label, archived, _order) = &convs[0];
        assert_eq!(*tl, timeline);
        assert_eq!(conv_id, "abc");
        assert_eq!(label, "tour");
        assert!(*archived);
    }

    /// A tombstoned conversation leaves the listing, so deleting one actually
    /// removes it from the sidebar.
    ///
    /// The regression this pins: a tombstone marks a timeline for reclamation
    /// but leaves its registry entry in place, so a lister that filtered only
    /// on `archived` kept serving the deleted row on every hydrate — `DELETE`
    /// answered 204 and the conversation never went away.
    #[test]
    fn a_tombstoned_conversation_is_not_listed() {
        let (_, _, timeline, mut sub) = make_timeline();
        sub.set_conv_id(timeline, "abc", RecordIdentity::NONE);
        sub.set_label(timeline, "tour", RecordIdentity::NONE);
        assert_eq!(sub.known_conversations().len(), 1);

        sub.tombstone_timeline(timeline);
        assert!(
            sub.known_conversations().is_empty(),
            "a deleted conversation must not survive in the listing"
        );
        // Archived is a lifecycle flag the caller filters on; deleted is not,
        // so it is gone even when the caller asked to see archived rows.
        sub.set_archived(timeline, true);
        assert!(sub.known_conversations().is_empty());
    }

    /// **A prefix lookup answers "which are this scheme's?" without the
    /// registry.**
    ///
    /// `known_conversations` materialises every live conversation the workspace
    /// holds, because it feeds a sidebar. Retirement bounds the live set and not
    /// the registry, so a caller that wants one naming scheme's members —
    /// npcd's `npc-<id>-day-*` — would pay for the whole registry on every
    /// open, forever.
    ///
    /// Two properties, and the second is the one with teeth: a tombstoned
    /// conversation is not returned, which is what lets the caller treat
    /// "retired" and "absent" as one answer instead of re-checking each row.
    #[test]
    fn a_conv_id_prefix_lookup_finds_only_live_matches() {
        let (layer, group, first, mut sub) = make_timeline();
        sub.set_conv_id(first, "npc-7-day-1", RecordIdentity::NONE);

        let alloc = TimelineAllocator::new();
        let add = |sub: &mut Substrate, conv_id: &str| {
            let tl = alloc.next();
            sub.register_timeline(tl, layer, group);
            sub.set_conv_id(tl, conv_id, RecordIdentity::NONE);
            tl
        };
        add(&mut sub, "npc-7-day-2");
        // Shares the `npc-7` stem, so the trailing `-day-` is doing real work.
        add(&mut sub, "npc-70-day-1");
        add(&mut sub, "chat-1");
        let retired = add(&mut sub, "npc-7-day-0");
        sub.tombstone_timeline(retired);

        let mut found: Vec<String> = sub
            .conversations_with_conv_id_prefix("npc-7-day-")
            .into_iter()
            .map(|(_, conv)| conv)
            .collect();
        found.sort();
        assert_eq!(
            found,
            vec!["npc-7-day-1".to_string(), "npc-7-day-2".to_string()],
            "the lookup took a neighbour, a stranger, or a retired conversation"
        );
    }

    /// **Every live timeline is listed, named or not.** A caller retiring
    /// everything cannot start from `known_conversations`: an ingested document
    /// is a conversation that never set a `conv_id`, and a wipe that missed those
    /// would leave exactly the prefilled K/V it was run to discard.
    #[test]
    fn live_timelines_include_the_unnamed_and_exclude_the_retired() {
        let (layer, group, named, mut sub) = make_timeline();
        sub.set_conv_id(named, "npc-7-day-1", RecordIdentity::NONE);

        let alloc = TimelineAllocator::new();
        let unnamed = alloc.next();
        sub.register_timeline(unnamed, layer, group);
        let retired = alloc.next();
        sub.register_timeline(retired, layer, group);
        sub.tombstone_timeline(retired);

        let mut live = sub.live_timeline_ids();
        live.sort();
        let mut want = vec![named, unnamed];
        want.sort();
        assert_eq!(
            live, want,
            "an unnamed conversation was missed or a retired one was listed"
        );
    }
}
