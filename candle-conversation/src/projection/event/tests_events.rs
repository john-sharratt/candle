//! Tests of the projection event's aggregation, its classification from a
//! projection, and its record encoding.
//!
//! A point decodes from the same record as the full event with its panel fields
//! skipped, so loading a conversation's timeline does not build its panels. The
//! tests check the bucket order and sums, the glue interleaved between the members
//! of a tree collection, and that decoding tolerates garbage rather than breaking
//! a conversation's hydrate. A record written before the belief fields existed
//! still deserializes with their defaults.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::builder::Builder;
    use super::super::super::ids::{GroupId, Reserved, SectionId, TurnId, TurnIndex, TurnKey};
    use super::super::super::project::{
        GeneratedIdentity, ProjectionSegment, ResolvedSection, ResolvedTurn, SealedKind,
    };
    use super::super::super::scan_stats::{GroupScan, ScanStats};
    use super::{
        aggregate, decode_events, decode_points, encode_events, from_projection,
        staged_ingest_event, summary_node_event, BucketKind, GroupScanRecord, MaterializedPiece,
        ProjectionBucket, ProjectionPoint, SegmentTokens, SelectedTurn, SelectionScores,
        SystemItem,
    };
    use crate::substrate::ContentResolver;
    use crate::summary_tree::TurnKind;
    use std::collections::HashMap;
    use std::sync::Arc;

    // ── aggregate (pure) ─────────────────────────────────────────────────────

    fn seg(kind: BucketKind, label: &str, tokens: u32) -> SegmentTokens {
        SegmentTokens {
            kind,
            label: label.to_string(),
            tokens,
        }
    }

    #[test]
    fn aggregate_orders_system_then_sections_then_turns() {
        // Deliberately out of order on input.
        let segs = vec![
            seg(BucketKind::Turns, "conversation", 100),
            seg(BucketKind::Section, "code_read", 800),
            seg(BucketKind::System, "system", 320),
            seg(BucketKind::Section, "repo_map", 200),
        ];
        let ev = aggregate(&segs, 10_000, 0, 10.0);
        let order: Vec<(&str, BucketKind, u32)> = ev
            .buckets
            .iter()
            .map(|b| (b.label.as_str(), b.kind, b.tokens))
            .collect();
        assert_eq!(
            order,
            vec![
                ("system", BucketKind::System, 320),
                ("code_read", BucketKind::Section, 800),
                ("repo_map", BucketKind::Section, 200),
                ("conversation", BucketKind::Turns, 100),
            ]
        );
    }

    #[test]
    fn aggregate_sums_same_bucket_and_totals_materialized() {
        let segs = vec![
            seg(BucketKind::Turns, "conversation", 40),
            seg(BucketKind::Turns, "conversation", 60),
            seg(BucketKind::System, "system", 100),
        ];
        let ev = aggregate(&segs, 5_000, 10, 5.0);
        // conversation merged to 100, system 100.
        assert_eq!(ev.buckets.len(), 2);
        assert_eq!(ev.materialized_tokens, 200);
        assert_eq!(ev.substrate_tokens, 5_000);
        // The point sits at `start_token`; the interval it governs and its
        // throughput are derived by the consumer from the event sequence.
        assert_eq!(ev.start_token, 10);
        assert_eq!(ev.seconds, 5.0);
    }

    #[test]
    fn aggregate_empty_is_empty() {
        let ev = aggregate(&[], 0, 0, 1.0);
        assert!(ev.buckets.is_empty());
        assert_eq!(ev.materialized_tokens, 0);
    }

    #[test]
    fn encode_decode_events_round_trips() {
        let events = vec![
            aggregate(
                &[
                    seg(BucketKind::System, "system", 320),
                    seg(BucketKind::Section, "code_read", 800),
                ],
                42_000,
                0,
                3.0,
            ),
            aggregate(
                &[seg(BucketKind::Turns, "conversation", 540)],
                42_000,
                120,
                8.0,
            ),
        ];
        assert_eq!(super::decode_events(&super::encode_events(&events)), events);
    }

    #[test]
    fn decode_events_tolerates_garbage() {
        assert!(super::decode_events(b"not json").is_empty());
        assert!(super::decode_events(&[]).is_empty());
        assert!(decode_points(b"not json").is_empty());
    }

    /// **A point is the event without its panel fields**, read from the same
    /// record: every timeline field arrives, and a selection and spine in the
    /// payload are skipped rather than refused.
    #[test]
    fn a_record_decodes_to_its_light_points() {
        let mut full = aggregate(
            &[seg(BucketKind::Turns, "conversation", 540)],
            42_000,
            120,
            8.0,
        );
        full.self_reference = true;
        full.materialized = vec![MaterializedPiece::Glue {
            text: "<|im_start|>user\n".into(),
        }];
        full.selection.system = vec![SystemItem::Glue {
            name: "system_start".into(),
            content: "<|im_start|>system\n".into(),
            tokens: 3,
        }];
        assert_eq!(
            decode_points(&encode_events(&[full])),
            vec![ProjectionPoint {
                start_token: 120,
                seconds: 8.0,
                materialized_tokens: 540,
                substrate_tokens: 42_000,
                buckets: vec![ProjectionBucket {
                    label: "conversation".into(),
                    kind: BucketKind::Turns,
                    tokens: 540,
                }],
                self_reference: true,
            }]
        );
    }

    // ── from_projection (schema + resolver classification) ───────────────────

    const YAML: &str = r#"
system_prompt:
  items:
    - kind: section
      id: frame
      content: "You are Zen-Code."
    - kind: collection
      name: code_read
      summary:
        chunk: 4
        categorize:
          max_tokens: 256
          system_prompt: Propose categories.
          user_prompt: Propose categories.
        assign:
          max_tokens: 128
          system_prompt: Assign by number.
          user_prompt: Assign by number.
      selection: { kind: top_k, k: 3 }
      sections:
        - id: file_a
          content: "fn a() {}"
        - id: file_b
          content: "fn b() {}"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: conversation
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 8
"#;

    /// Returns explicit token counts for sections (by raw id) and turns.
    #[derive(Default)]
    struct TokResolver {
        section_tokens: HashMap<u32, usize>,
        turn_tokens: HashMap<(u32, u32), usize>,
        group_scans: HashMap<u32, GroupScan>,
    }

    impl ContentResolver for TokResolver {
        fn group_turns(&self, _group: GroupId) -> Vec<TurnKey> {
            Vec::new()
        }
        fn group_scan(&self, group: GroupId) -> Option<GroupScan> {
            self.group_scans.get(&group.raw()).copied()
        }
        /// Fixtures are keyed by `(group, index)`; the emitted turns carry a
        /// timeline whose raw id is the group id (see the mock in
        /// `projection::tests`), so the lookup re-derives it.
        fn turn_token_count(&self, turn: TurnKey) -> usize {
            *self
                .turn_tokens
                .get(&(turn.timeline.raw() as u32, turn.index.0))
                .unwrap_or(&0)
        }
        fn turn_score(&self, _turn: TurnKey) -> f32 {
            0.0
        }
        fn trained_window(&self) -> Option<usize> {
            None
        }
        fn section_token_count(&self, section: SectionId) -> usize {
            *self.section_tokens.get(&section.raw()).unwrap_or(&0)
        }
    }

    #[test]
    fn from_projection_buckets_system_collection_and_turns() {
        let b = Builder::from_yaml(YAML).unwrap();
        let schema = b.schema();
        let conv = b.id_for_group("conversation").unwrap();
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let frame = b.id_for_system_section("frame").unwrap();
        let file_a = b.id_for_system_section("file_a").unwrap();
        let file_b = b.id_for_system_section("file_b").unwrap();

        let mut res = TokResolver::default();
        res.section_tokens.insert(frame.raw(), 64); // bare → system
        res.section_tokens.insert(file_a.raw(), 300); // collection code_read
        res.section_tokens.insert(file_b.raw(), 500); // collection code_read
        res.turn_tokens.insert((conv.raw(), 0), 120);
        res.turn_tokens.insert((conv.raw(), 1), 80);

        let turn = |idx: u32| {
            ProjectionSegment::Sealed(SealedKind::Turn(
                ResolvedTurn {
                    id: TurnId {
                        layer_id: dialogue,
                        group_id: conv,
                        index: TurnIndex(idx),
                    },
                    // A turn resolves by (timeline, index); the fixture keys its
                    // token counts by the group id, so the conversation id matches.
                    timeline: crate::projection::TimelineId::from_raw(conv.raw() as u64),
                },
                crate::Role::Assistant,
            ))
        };
        let section = |id: SectionId| {
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(id)))
        };

        let segments = vec![
            section(frame),
            section(file_a),
            section(file_b),
            turn(0),
            turn(1),
            // structural framing — must be ignored
            ProjectionSegment::Generated {
                tokens: Arc::new(vec![1, 2, 3]),
                identity: GeneratedIdentity {
                    name: "system_open".to_string(),
                    position: 0,
                },
            },
            ProjectionSegment::NewUserMessage {
                tokens: Arc::new(vec![9, 9, 9, 9]),
            },
        ];

        let ev = from_projection(
            &segments,
            schema,
            &res,
            &SelectionScores::default(),
            99_999,
            0,
            12.0,
        );

        // system (64) leftmost, then code_read (300+500=800), then turns:
        // conversation (120+80=200) and current message (4).
        let got: Vec<(&str, BucketKind, u32)> = ev
            .buckets
            .iter()
            .map(|b| (b.label.as_str(), b.kind, b.tokens))
            .collect();
        assert_eq!(
            got,
            vec![
                ("system", BucketKind::System, 64),
                ("code_read", BucketKind::Section, 800),
                ("conversation", BucketKind::Turns, 200),
                ("current message", BucketKind::Turns, 4),
            ]
        );
        // materialized excludes the skipped Generated run (3 tokens).
        assert_eq!(ev.materialized_tokens, 64 + 800 + 200 + 4);
        assert_eq!(ev.substrate_tokens, 99_999);
        assert_eq!(ev.seconds, 12.0);
        assert!(ev.groups.is_empty(), "no group was scanned");
    }

    /// A scanned group's statistics ride the event under the group's name,
    /// whether or not anything of it was selected; an unscanned group is absent.
    #[test]
    fn from_projection_records_each_scanned_group_by_name() {
        let b = Builder::from_yaml(YAML).unwrap();
        let conv = b.id_for_group("conversation").unwrap();
        let stats = |top: f32| ScanStats {
            members: 12,
            top,
            p90: 40.0,
            median: 10.0,
        };
        let scan = GroupScan {
            tail: stats(90.0),
            question: Some(stats(300.0)),
        };
        let mut res = TokResolver::default();
        res.group_scans.insert(conv.raw(), scan);
        let ev = from_projection(
            &[],
            b.schema(),
            &res,
            &SelectionScores::default(),
            0,
            0,
            0.0,
        );
        assert_eq!(
            ev.groups,
            vec![GroupScanRecord {
                group: "conversation".to_string(),
                scan,
            }]
        );
        assert_eq!(
            serde_json::to_value(&ev.groups).unwrap(),
            serde_json::json!([{
                "group": "conversation",
                "tail": {"members": 12, "top": 90.0, "p90": 40.0, "median": 10.0},
                "question": {"members": 12, "top": 300.0, "p90": 40.0, "median": 10.0},
            }])
        );
    }

    const TREE_GLUE_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: collection
          name: tools
          selection: { kind: top_k, k: 3 }
          member_glue: "\n"
          summary:
            chunk: 4
            categorize:
              max_tokens: 256
              system_prompt: cat
              user_prompt: cat
            assign:
              max_tokens: 128
              system_prompt: assign
              user_prompt: assign
          sections:
            - id: tool_a
              content: "a"
            - id: tool_b
              content: "b"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: conversation
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 8
"#;

    #[test]
    fn from_projection_interleaves_member_glue_between_tree_collection_members() {
        use super::super::super::schema::SystemPromptItem;
        let mut b = Builder::from_yaml(TREE_GLUE_YAML).unwrap();
        // Tokenise so the glue carries a real token count (mock: byte → token).
        b.tokenize_templates(|s: &str| Ok::<_, ()>(s.bytes().map(u32::from).collect()))
            .unwrap();
        let schema = b.schema();

        // The tools tree-collection node + its (single-branch) member variants.
        let tc = schema
            .system_prompt
            .items
            .iter()
            .find_map(|it| match it {
                SystemPromptItem::SectionTree(t) => {
                    t.nodes.iter().find_map(|n| n.collection.as_ref())
                }
                _ => None,
            })
            .unwrap();
        let key = tc.default_branch;
        let a = tc.member_variant(0, key).unwrap().id;
        let bb = tc.member_variant(1, key).unwrap().id;

        let mut res = TokResolver::default();
        res.section_tokens.insert(a.raw(), 10);
        res.section_tokens.insert(bb.raw(), 20);

        let section = |id: SectionId| {
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(id)))
        };
        let segments = vec![section(a), section(bb)];
        let ev = from_projection(
            &segments,
            schema,
            &res,
            &SelectionScores::default(),
            0,
            0,
            1.0,
        );

        // The materialized system view is: member → REAL glue token → member,
        // not two members run together — so the GUI panel and its copy-all
        // reproduce the exact bytes (the glue is no longer baked into a seal).
        let sys = &ev.selection.system;
        assert_eq!(sys.len(), 3, "expected member, glue, member; got {sys:?}");
        match &sys[0] {
            SystemItem::Section { name, .. } => assert_eq!(name, "tool_a"),
            o => panic!("expected tool_a section, got {o:?}"),
        }
        match &sys[1] {
            SystemItem::Glue {
                content, tokens, ..
            } => {
                assert_eq!(content, "\n", "glue content must be the literal newline");
                assert_eq!(*tokens, 1, "glue must report its token count");
            }
            o => panic!("expected a glue row between members, got {o:?}"),
        }
        match &sys[2] {
            SystemItem::Section { name, .. } => assert_eq!(name, "tool_b"),
            o => panic!("expected tool_b section, got {o:?}"),
        }
    }

    #[test]
    fn from_projection_captures_selection_with_skipped_sections() {
        let b = Builder::from_yaml(YAML).unwrap();
        let schema = b.schema();
        let conv = b.id_for_group("conversation").unwrap();
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let frame = b.id_for_system_section("frame").unwrap();
        let file_a = b.id_for_system_section("file_a").unwrap();
        let file_b = b.id_for_system_section("file_b").unwrap();

        let mut res = TokResolver::default();
        res.section_tokens.insert(frame.raw(), 64);
        res.section_tokens.insert(file_a.raw(), 300);
        res.section_tokens.insert(file_b.raw(), 500);
        res.turn_tokens.insert((conv.raw(), 0), 120);

        // Select frame + file_a, but NOT file_b, plus one turn.
        let segments = vec![
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(frame))),
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(file_a))),
            ProjectionSegment::Sealed(SealedKind::Turn(
                ResolvedTurn {
                    id: TurnId {
                        layer_id: dialogue,
                        group_id: conv,
                        index: TurnIndex(0),
                    },
                    // Turns resolve by (timeline, index); the fixture keys its
                    // token counts by the group id, so the ids line up.
                    timeline: crate::projection::TimelineId::from_raw(conv.raw() as u64),
                },
                crate::Role::User,
            )),
        ];

        let sel = from_projection(
            &segments,
            schema,
            &res,
            &SelectionScores::default(),
            1000,
            0,
            1.0,
        )
        .selection;

        // System prompt in order: bare `frame` section, then the `code_read`
        // collection (no template items in this fixture, so no glue rows).
        assert_eq!(sel.system.len(), 2);
        match &sel.system[0] {
            SystemItem::Section { name, tokens } => {
                assert_eq!(name, "frame");
                assert_eq!(*tokens, 64);
            }
            other => panic!("expected frame section, got {other:?}"),
        }
        // The code_read collection shows BOTH members, file_b flagged skipped.
        match &sel.system[1] {
            SystemItem::Collection { name, sections, .. } => {
                assert_eq!(name, "code_read");
                assert_eq!(sections.len(), 2);
                let a = sections.iter().find(|s| s.name == "file_a").unwrap();
                let b = sections.iter().find(|s| s.name == "file_b").unwrap();
                assert!(a.selected && a.tokens == 300);
                assert!(!b.selected && b.tokens == 500);
            }
            other => panic!("expected code_read collection, got {other:?}"),
        }

        // The selected turn is captured.
        assert_eq!(sel.turns.len(), 1);
        assert_eq!(sel.turns[0].role, "user");
        assert_eq!(sel.turns[0].index, 0);
        assert_eq!(sel.turns[0].tokens, 120);
    }

    #[test]
    fn from_projection_surfaces_emitted_template_glue_in_order() {
        use candle_transformers::models::dialect::Dialect;
        const YAML_T: &str = r#"
system_prompt:
  items:
    - kind: template
      id: system_open
      dialect: system_start
    - kind: section
      id: frame
      content: "You are Zen-Code."
    - kind: template
      id: system_close
      dialect: system_end
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: conversation
        selection: { kind: always_visible }
"#;
        let dlct = Dialect::chat_ml();
        let b = Builder::from_yaml_with_vars_and_dialect(YAML_T, &[], Some(&dlct)).unwrap();
        let schema = b.schema();
        let frame = b.id_for_system_section("frame").unwrap();

        let mut res = TokResolver::default();
        res.section_tokens.insert(frame.raw(), 64);

        // The assembler emits both template items as Generated glue, with the
        // content section sealed between them — exactly the shape the view must
        // reproduce so the closing markers stay visible.
        let segments = vec![
            ProjectionSegment::Generated {
                tokens: Arc::new(vec![1, 2, 3]),
                identity: GeneratedIdentity {
                    name: "system_open".to_string(),
                    position: 0,
                },
            },
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(frame))),
            ProjectionSegment::Generated {
                tokens: Arc::new(vec![9, 9]),
                identity: GeneratedIdentity {
                    name: "system_close".to_string(),
                    position: 2,
                },
            },
        ];

        let sel = from_projection(
            &segments,
            schema,
            &res,
            &SelectionScores::default(),
            1000,
            0,
            1.0,
        )
        .selection;

        assert_eq!(sel.system.len(), 3, "glue + section + glue, in order");
        match &sel.system[0] {
            SystemItem::Glue {
                name,
                content,
                tokens,
            } => {
                assert_eq!(name, "system_open");
                assert_eq!(content, "<|im_start|>system\n");
                assert_eq!(*tokens, 3);
            }
            other => panic!("expected system_open glue, got {other:?}"),
        }
        match &sel.system[1] {
            SystemItem::Section { name, tokens } => {
                assert_eq!(name, "frame");
                assert_eq!(*tokens, 64);
            }
            other => panic!("expected frame section, got {other:?}"),
        }
        match &sel.system[2] {
            SystemItem::Glue {
                name,
                content,
                tokens,
            } => {
                assert_eq!(name, "system_close");
                assert_eq!(content, "<|im_end|>\n");
                assert_eq!(*tokens, 2);
            }
            other => panic!("expected system_close glue, got {other:?}"),
        }
    }

    #[test]
    fn from_projection_summary_shows_outside_tools_markers() {
        use candle_transformers::models::dialect::Dialect;
        const YAML: &str = r#"
system_prompt:
  items:
    - kind: template
      id: tools_open
      dialect: tool_block_open
      depends_on: tools
    - kind: collection
      name: tools
      selection: { kind: top_k, k: 1 }
      summary:
        chunk: 4
        categorize:
          max_tokens: 256
          system_prompt: x
          user_prompt: x
        assign:
          max_tokens: 128
          system_prompt: x
          user_prompt: x
      sections:
        - id: t1
          content: "tool"
    - kind: template
      id: tools_close
      dialect: tool_block_close
      depends_on: tools
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: c
          user_prompt: c
        assistant:
          system_prompt: c
          user_prompt: c
    groups:
      - id: conversation
        selection: { kind: always_visible }
"#;
        let dlct = Dialect::chat_ml();
        let mut b = Builder::from_yaml_with_vars_and_dialect(YAML, &[], Some(&dlct)).unwrap();
        let tools = b.id_for_system_collection("tools").unwrap();
        let summary = SectionId::reserved(Reserved::ToolSummary);
        b.set_collection_summary_section(tools, summary).unwrap();
        let t1 = b.id_for_system_section("t1").unwrap();
        let schema = b.schema();

        let mut res = TokResolver::default();
        res.section_tokens.insert(summary.raw(), 40);
        res.section_tokens.insert(t1.raw(), 8);

        // The projected order: summary OUTSIDE, then <tools>, member, </tools>.
        let segments = vec![
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(summary))),
            ProjectionSegment::Generated {
                tokens: Arc::new(vec![1]),
                identity: GeneratedIdentity {
                    name: "tools_open".to_string(),
                    position: 0,
                },
            },
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(t1))),
            ProjectionSegment::Generated {
                tokens: Arc::new(vec![2]),
                identity: GeneratedIdentity {
                    name: "tools_close".to_string(),
                    position: 2,
                },
            },
        ];
        let sel = from_projection(
            &segments,
            schema,
            &res,
            &SelectionScores::default(),
            1000,
            0,
            1.0,
        )
        .selection;

        // The catalog summary must appear BEFORE the `tools_open` glue — i.e.
        // outside the `<tools>` block — in the panel composition.
        let sum_pos = sel
            .system
            .iter()
            .position(
                |it| matches!(it, SystemItem::Section { name, .. } if name == "tools summary"),
            )
            .unwrap_or_else(|| panic!("summary not in composition: {:?}", sel.system));
        let open_pos = sel
            .system
            .iter()
            .position(|it| matches!(it, SystemItem::Glue { name, .. } if name == "tools_open"))
            .unwrap_or_else(|| panic!("tools_open glue not in composition: {:?}", sel.system));
        assert!(
            sum_pos < open_pos,
            "summary must show before <tools>: {:?}",
            sel.system
        );
    }

    // ── staged_ingest_event / summary_node_event (pure constructors) ─────────

    fn sel_turn(index: u32, tokens: u32, timeline: u64) -> SelectedTurn {
        SelectedTurn {
            qualified: false,
            layer: "code_reading".to_string(),
            group: "scopes".to_string(),
            index,
            role: "user".to_string(),
            tokens,
            kind: TurnKind::Normal,
            reason: None,
            timeline: Some(timeline),
            selected: true,
            score: 0.0,
        }
    }

    #[test]
    fn staged_ingest_event_carries_self_and_prev_references_and_buckets() {
        let system = vec![SystemItem::Section {
            name: "frame".to_string(),
            tokens: 40,
        }];
        let turns = vec![sel_turn(4, 300, 77), sel_turn(5, 250, 77)];
        let ev = staged_ingest_event(12, 1.5, system, turns);

        assert_eq!(ev.start_token, 12);
        assert_eq!(ev.seconds, 1.5);
        assert_eq!(ev.selection.turns.len(), 2, "prev + self");
        assert_eq!(ev.selection.turns[0].index, 4);
        assert_eq!(ev.selection.turns[1].index, 5);
        assert_eq!(
            ev.selection.turns[1].timeline,
            Some(77),
            "the (timeline, index) reference must be resolvable"
        );
        let by_kind: Vec<(BucketKind, u32)> =
            ev.buckets.iter().map(|b| (b.kind, b.tokens)).collect();
        assert_eq!(
            by_kind,
            vec![(BucketKind::System, 40), (BucketKind::Turns, 550)]
        );
        assert_eq!(ev.materialized_tokens, 590);
    }

    #[test]
    fn staged_ingest_event_roundtrips_through_encode_decode() {
        let ev = staged_ingest_event(
            0,
            0.0,
            vec![SystemItem::Section {
                name: "frame".to_string(),
                tokens: 8,
            }],
            vec![sel_turn(0, 100, 42)],
        );
        let events = vec![ev.clone(), staged_ingest_event(9, 0.0, Vec::new(), vec![])];
        let decoded = decode_events(&encode_events(&events));
        assert_eq!(decoded, events);
        assert_eq!(decoded[0].selection.turns[0].timeline, Some(42));
    }

    #[test]
    fn selected_turn_deserializes_pre_belief_records_with_defaults() {
        // A redo-log record written before the belief-carry fields existed has
        // neither `selected` nor `score`; serde defaults keep it loadable, so
        // an old substrate opens without a migration.
        let json = r#"{
            "layer": "dialogue",
            "group": "conversation",
            "index": 7,
            "role": "user",
            "tokens": 42,
            "timeline": 5
        }"#;
        let t: SelectedTurn = serde_json::from_str(json).unwrap();
        assert_eq!(t.index, 7);
        assert!(!t.selected, "missing `selected` defaults to false");
        assert_eq!(t.score, 0.0, "missing `score` defaults to 0.0");
        // New records round-trip the fields.
        let full = SelectedTurn {
            selected: true,
            score: 900.0,
            ..t
        };
        let s = serde_json::to_string(&full).unwrap();
        let back: SelectedTurn = serde_json::from_str(&s).unwrap();
        assert!(back.selected);
        assert_eq!(back.score, 900.0);
    }

    #[test]
    fn summary_node_event_lists_node_first_then_children() {
        let ev = summary_node_event(
            "code_reading",
            "scopes",
            77,
            (TurnIndex(9), TurnKind::SummaryOfTurns, 120),
            vec![
                (TurnIndex(3), TurnKind::Normal, 400),
                (TurnIndex(4), TurnKind::Normal, 500),
            ],
        );
        assert!(
            ev.selection.system.is_empty(),
            "compression frames run no projection"
        );
        assert_eq!(ev.selection.turns.len(), 3);
        assert_eq!(ev.selection.turns[0].index, 9, "the node itself is first");
        assert_eq!(ev.selection.turns[0].kind, TurnKind::SummaryOfTurns);
        assert_eq!(ev.selection.turns[0].timeline, Some(77));
        assert_eq!(ev.selection.turns[1].index, 3);
        assert_eq!(ev.selection.turns[2].index, 4);
        assert_eq!(ev.buckets.len(), 1);
        assert_eq!(ev.buckets[0].kind, BucketKind::Turns);
        assert_eq!(ev.buckets[0].tokens, 120 + 400 + 500);
        assert_eq!(ev.materialized_tokens, 1020);

        let decoded = decode_events(&encode_events(std::slice::from_ref(&ev)));
        assert_eq!(decoded, vec![ev]);
    }
}
