//! Tests the projection schema: its labels, the working set it carries, and the
//! turn sinks it derives from its structure.
//!
//! The embedded schema is checked for the working-set configuration the
//! dialogue and the ingest layers depend on. The turn-sink plan is checked to
//! come from structure rather than annotation, and a coding workspace's folder
//! named like a pipeline layer is checked never to be ingested as raw. The
//! persona is checked to name the served model, and the journal to stay pinned
//! within the position budget.

use super::*;

#[cfg(test)]
pub(super) mod projection_schema_tests {
    use super::build_projection_builder;
    use candle_conversation::projection::{SelectionRule, WorkingSetShare};
    use host::substrate_views::system_prompt_labels;
    use std::collections::HashMap;
    use std::path::Path;

    /// The dialogue layer's system-prompt labels name its framing sections, its
    /// tools collection (with a member count), and its section tree — the
    /// structure the viewer lists under "Target layer".
    #[test]
    fn dialogue_system_prompt_labels_cover_sections_collection_and_tree() {
        let builder = build_projection_builder(Path::new("demo-project"));
        let labels = system_prompt_labels(&builder.schema().system_prompt);

        assert!(
            labels.iter().any(|l| l == "history_stance"),
            "a bare framing section is named verbatim: {labels:?}"
        );
        assert!(
            labels.iter().any(|l| l == "tools · collection"),
            "the tools collection is labelled (members listed separately): {labels:?}"
        );
        assert!(
            labels.iter().any(|l| l.starts_with("section tree (")),
            "the adaptive-thinking section tree is labelled: {labels:?}"
        );
    }

    /// The persona names the model the host serves, filled in where the schema
    /// is loaded, beside the workspace it works in.
    #[test]
    fn the_persona_names_the_served_model() {
        use crate::model::MODEL;
        use crate::repo_scan::shared_prompt_body;
        use host::model::ModelView;
        let builder = build_projection_builder(Path::new("demo-project"));
        let body = shared_prompt_body(&builder);
        let model = ModelView::of_model(&MODEL);
        let expected = format!(
            "You are Zen, an AI coding assistant running on {}, working in the demo-project\n\
             workspace, which holds one or more repositories.",
            model.release()
        );
        assert!(body.contains(&expected), "{body}");
    }

    /// **The prompt tells the model where the files it has read are**: under
    /// Knowledge Base at the end of the prompt, the turns being its work with
    /// the user, and it answers from both (`docs/provenance_system_collection.md`
    /// §7.2).
    #[test]
    fn the_prompt_says_the_files_are_in_the_knowledge_base() {
        use crate::repo_scan::shared_prompt_body;
        let builder = build_projection_builder(Path::new("demo-project"));
        let body = shared_prompt_body(&builder);
        assert!(
            body.contains(
                "Earlier turns here are your own work with the user, even where they appear\n\
                 condensed or summarised, so build on what you already read, found and decided\n\
                 instead of repeating it. The files you have read are under Knowledge Base at the\n\
                 end of this prompt. What to do now is always the user's latest message.\n"
            ),
            "{body}"
        );
        assert!(body.contains(
            "Answer from what is actually in this conversation and the files in the knowledge\n\
             base. Your tools run for real"
        ));
        assert!(!body.contains("the reading done before this"));
    }

    /// **The placed sections take their reserved ranges**
    /// (`docs/placed_section_layout.md` §3.7b): the journal 8K (2K below a
    /// 128K native context) from the bottom, the tools 8K and the knowledge
    /// base 96K from the top.
    #[test]
    fn embedded_schema_reserves_the_placed_sections_ranges() {
        use candle_conversation::projection::{RangeFill, ReservedRange, SystemPromptItem};
        let builder = build_projection_builder(Path::new("demo-project"));
        let range = |name: &str| {
            builder
                .schema()
                .system_prompt
                .items
                .iter()
                .find_map(|item| match item {
                    SystemPromptItem::Collection(c) if c.name == name => Some(c.range),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("collection {name} present"))
        };
        assert_eq!(
            range("journal"),
            Some(ReservedRange {
                tokens: 8_192,
                small_context: 2_048,
                fill: RangeFill::Bottom,
            })
        );
        assert_eq!(
            range("tools"),
            Some(ReservedRange {
                tokens: 8_192,
                small_context: 8_192,
                fill: RangeFill::Top,
            })
        );
        assert_eq!(
            range("files"),
            Some(ReservedRange {
                tokens: 114_688,
                small_context: 114_688,
                fill: RangeFill::Top,
            })
        );
    }

    /// The embedded schema carries the working set (`docs/zend_working_set.md`)
    /// onto the right nodes: `repo_map/structure` and `code_reading/scopes`
    /// fill from the dialogue's working set, in tool rounds too, with the
    /// scan's settings kept (they build the momentum) and the belief-only
    /// tuning — locality, anchor, adaptive budgets — gone; the dialogue layer
    /// declares the budget; the tools collection stays additive (the measured
    /// per-axis split).
    #[test]
    fn embedded_schema_carries_the_working_set_config() {
        use crate::calibrate::ask::AskScope;
        use crate::calibrate::count::QuestionCount;
        use crate::calibrate::CalibrateSpec;
        use candle_conversation::projection::{ScanCases, ScanNorm, SystemPromptItem};
        use candle_conversation::provenance::{FusionMode, VoteLaw};
        let builder = build_projection_builder(Path::new("demo-project"));
        let schema = builder.schema();

        let layer = |name: &str| {
            schema
                .layers
                .iter()
                .find(|l| l.name == name)
                .unwrap_or_else(|| panic!("layer {name} present"))
        };
        for (name, share) in [
            ("repo_map", WorkingSetShare::Folders),
            ("code_reading", WorkingSetShare::Readings),
        ] {
            let l = layer(name);
            let g = &l.groups[0];
            assert_eq!(g.selection, SelectionRule::WorkingSet { share }, "{name}");
            assert!(l.in_tool_rounds, "{name} is in tool rounds");
            // Content axes are ADDITIVE — content_gated was reverted after live
            // testing showed it zeros NL→code retrieval (the domain gap).
            assert_eq!(g.policy.scan.fusion, FusionMode::Additive, "{name}");
            assert!(g.policy.scan.question_pin, "{name}");
            assert!(g.locality.is_none() && g.anchor.is_none(), "{name}");
            assert!(g.budget_adaptive.is_none(), "{name}");
            assert!(l.budget.adaptive.is_none(), "{name}");
        }

        // The guides and the journals are drawn out of the files, each as its
        // own kind, with their hub files left behind; a journal entry fades
        // over a 30-day half-life and a guide does not.
        let group = |name: &str| {
            layer("code_reading")
                .groups
                .iter()
                .find(|g| g.name == name)
                .unwrap_or_else(|| panic!("code_reading declares the {name} group"))
        };
        let guides = group("guidance");
        assert_eq!(
            guides.selection,
            SelectionRule::WorkingSet {
                share: WorkingSetShare::Guides
            }
        );
        let draws = guides.draws.as_ref().expect("guides draw");
        assert_eq!(draws.paths, ["candle/mcp/src/guides/"]);
        assert_eq!(draws.except, ["README.md"]);
        assert!(guides.recency.is_none(), "a guide does not fade");
        let journals = group("journals");
        assert_eq!(
            journals.selection,
            SelectionRule::WorkingSet {
                share: WorkingSetShare::Journals
            }
        );
        let draws = journals.draws.as_ref().expect("journals draw");
        assert_eq!(draws.paths, ["journals/"]);
        assert_eq!(
            draws.except,
            ["README.md", "group.yaml", "journal.yaml", "person.yaml"]
        );
        let recency = journals.recency.as_ref().expect("journals age");
        assert_eq!(
            (recency.half_life_days, recency.dated_by.as_str()),
            (30.0, "path")
        );

        // A whole file is one case of one competition, scored under the
        // scoring lab's formula: every fold group alike in both competitions,
        // the calibration weighed alike with the content, each token voting
        // for its best 8 files at power 0.5 (the calibration keeping one vote
        // a token), a file read against its own kind, the lineage not offered.
        let g = &layer("code_reading").groups[0];
        assert_eq!(g.policy.scan.cases, ScanCases::File);
        assert_eq!(g.policy.layer_weights, vec![1.0, 1.0, 1.0]);
        assert_eq!(g.policy.calibration_layer_weights, vec![1.0, 1.0, 1.0]);
        assert_eq!(g.policy.scan.calibration_weight, 1.0);
        assert_eq!(
            g.policy.scan.law,
            VoteLaw {
                top_k: 8,
                power: 0.5
            }
        );
        assert_eq!(
            g.policy.scan.calibration_law(),
            VoteLaw {
                top_k: 1,
                power: 0.5
            }
        );
        assert_eq!(g.policy.scan.normalize, ScanNorm::KindZ);
        assert!(g.policy.scan.exclude_lineage);
        assert_eq!(
            layer("repo_map").groups[0].policy.scan.cases,
            ScanCases::Exchange
        );
        // Each section of a file is calibrated with the questions it answers,
        // each opening the thinking it starts, asked by what was read: a
        // manifest what it declares, a test what it proves, a kernel what it
        // computes, code what it defines and who uses it, a document what it
        // states — each prompt on its own budget, per section, and the file's
        // purpose once per file (docs/placed_section_layout.md §3.5d).
        let calibrate = CalibrateSpec::of(layer("code_reading"))
            .unwrap()
            .expect("code_reading declares a calibration");
        assert_eq!(calibrate.think, 64);
        let rule = |per_tokens, min, max| QuestionCount {
            per_tokens,
            min,
            max,
        };
        let asked = |subject: &str| {
            let profile = calibrate.profile_for(subject);
            let asks = calibrate
                .asks_for(subject)
                .iter()
                .map(|a| (a.name.clone(), a.questions, a.per))
                .collect::<Vec<_>>();
            (profile.name.clone(), asks)
        };
        let named = |profile: &str, asks: &[(&str, QuestionCount, AskScope)]| {
            (
                profile.to_string(),
                asks.iter()
                    .map(|(n, q, p)| (n.to_string(), *q, *p))
                    .collect::<Vec<_>>(),
            )
        };
        let (section, file) = (AskScope::Section, AskScope::File);
        assert_eq!(
            asked("candle/zend/Cargo.toml"),
            named(
                "manifest",
                &[
                    ("dependencies", rule(240, 2, 8), section),
                    ("dependents", rule(800, 1, 4), section),
                    ("purpose", rule(4000, 1, 2), file),
                ]
            )
        );
        for test in [
            "candle/zend/tests/aggregate_js.rs",
            "candle/zend/src/mcp/latency_tests.rs",
            "candle/tools/test_bench.py",
            "candle/web/tests/perf.spec.ts",
        ] {
            assert_eq!(asked(test).0, "tests", "{test}");
        }
        assert_eq!(
            asked("candle/candle-kernels/src/paged-decode/paged_decode.cu").0,
            "kernels"
        );
        assert_eq!(
            asked("candle/zend/src/calibrate/run.rs"),
            named(
                "code",
                &[
                    ("definitions", rule(600, 2, 14), section),
                    ("uses", rule(800, 1, 10), section),
                    ("messages", rule(3000, 0, 4), section),
                    ("purpose", rule(3000, 2, 6), file),
                ]
            )
        );
        assert_eq!(asked("candle/zend/src/prompts/projection.yaml").0, "config");
        assert_eq!(asked("candle/docs/unbounded_agents.md").0, "docs");
        assert_eq!(asked("candle/tools/quality_harness.sh").0, "scripts");
        assert_eq!(asked("candle/keep/install.ps1").0, "scripts");
        assert_eq!(asked("candle/LICENSE-MIT").0, "default");
        assert!(calibrate.profiles.iter().all(|p| p
            .asks
            .iter()
            .all(|a| a.prompt.contains("<<path>>") && a.prompt.contains("<<n>>"))));
        let guidance = calibrate.guidance.expect("the guidance group draws");
        assert_eq!(guidance.ask.questions, rule(250, 8, 64));

        let ws = layer("dialogue")
            .working_set
            .as_ref()
            .expect("the dialogue declares a working set");
        assert_eq!(ws.beta, 0.2);
        assert_eq!(ws.min_momentum, 100.0);
        // The reserve fits Flash-Next's trained window, and the slot's room
        // is shared out by kind as the fill's caps, the shares making the
        // whole room (`docs/hit_driven_projection.md` §4, §7.2 step 4).
        assert_eq!(ws.reserve_tokens, 16_384);
        assert_eq!(ws.slot_window(262_144).window, 262_144);
        let caps = ws.fill_shares.caps(233_000);
        assert_eq!(
            (
                caps.readings,
                caps.folders,
                caps.guides,
                caps.journals,
                caps.dialogue
            ),
            (52_425, 4_660, 11_650, 24_465, 139_800)
        );
        // The persona and the journal are pinned, within 1,024 positions.
        assert!(!schema.system_prompt.pins.is_empty());
        assert_eq!(schema.system_prompt.pins.tokens, 1024);

        let tools = schema
            .system_prompt
            .items
            .iter()
            .find_map(|i| match i {
                SystemPromptItem::Collection(c) if c.name == "tools" => Some(c),
                _ => None,
            })
            .expect("tools collection");
        assert_eq!(
            tools.policy.scan.fusion,
            FusionMode::Additive,
            "tools MUST stay additive (results doc §25.8: gating collapses Top-1 97.3% → 32.9%)"
        );
        assert_eq!(
            tools.policy.scan.law,
            VoteLaw::SINGLE,
            "tools keep one vote a token: top-8 costs Top-1 80% -> 64% \
             (docs/results/provenance_scoring_lab_2026-11-16.md)"
        );
        // Strict normalized-band gate (reverted from the too-permissive min 112).
        assert_eq!(tools.policy.config.min_score, 800.0);
        // The count's ceiling, binding only without a range; the tools' range
        // is their size (docs/placed_section_layout.md §3.5).
        assert_eq!(tools.policy.config.budget_max, 5);
        assert!(tools.range.is_some());

        // Dialogue stays recency-driven — no scan concepts apply.
        let dialogue = layer("dialogue");
        assert!(!dialogue.groups[0].is_belief_driven());
    }

    /// The turn-sink load plan is DERIVED from the embedded schema's structure —
    /// no `ingest:` annotations: the built-in `repo_map` (folder scan) and
    /// `code_reading` (file read) over every repository — an empty scope — the
    /// live `dialogue` layer excluded.
    #[test]
    fn embedded_turn_sinks_are_derived_from_structure() {
        use crate::ingest::IngestMode;
        let builder = build_projection_builder(Path::new("demo-project"));
        // "demo-project" doesn't exist on disk, so no folder-backed raw sinks
        // resolve — only the two built-in pipelines.
        let layers = crate::ingest::ingest_layers(
            builder.schema(),
            Path::new("demo-project"),
            &HashMap::new(),
        );
        let got: Vec<(&str, IngestMode, &str)> = layers
            .iter()
            .map(|l| (l.name.as_str(), l.mode, l.folder.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("repo_map", IngestMode::Folders, ""),
                ("code_reading", IngestMode::Files, ""),
            ],
            "derived turn-sinks: {got:?}"
        );
        // Every layer keeps the id its conversations were persisted under.
        let ids: Vec<(&str, u32)> = builder
            .schema()
            .layers
            .iter()
            .map(|l| (l.name.as_str(), l.id.raw()))
            .collect();
        assert_eq!(
            ids,
            vec![("repo_map", 1), ("code_reading", 2), ("dialogue", 10),]
        );
        // The live dialogue layer is never a turn-sink.
        assert!(
            !layers.iter().any(|l| l.name == "dialogue"),
            "dialogue must be excluded as the live conversation layer",
        );
        // The coding-assistant schema has no folder-backed section collections:
        // its collections are the registry-backed `tools`, the journal and the
        // knowledge base's `files`, all excluded.
        assert!(
            crate::ingest::section_sinks(builder.schema()).is_empty(),
            "embedded schema should derive no section-collection sinks",
        );
    }

    /// **The system prompt ends on the knowledge base**: `files_open`, the
    /// `files` collection with no schema member, then the close — so the
    /// birth prefix and every branch checkpoint end `files_open`, the files'
    /// range, `system_close`, with no knowledge-base content in them: the
    /// seed is the collection's mandatory files, placed as the others are
    /// (docs/placed_section_layout.md §3.5b).
    #[test]
    fn the_system_prompt_ends_on_the_knowledge_base() {
        use candle_conversation::projection::SystemPromptItem;
        let builder = build_projection_builder(Path::new("demo-project"));
        let items = &builder.schema().system_prompt.items;
        let tail: Vec<String> = items[items.len() - 3..]
            .iter()
            .map(|item| match item {
                SystemPromptItem::Section(s) => format!("section {}", s.name),
                SystemPromptItem::Collection(c) => {
                    format!("collection {} ({} members)", c.name, c.sections.len())
                }
                SystemPromptItem::SectionTree(t) => {
                    let n = &t.nodes[0];
                    let opts: Vec<&str> = n.options.iter().map(|o| o.id.as_str()).collect();
                    format!("tree {} [{}]", n.name, opts.join("|"))
                }
            })
            .collect();
        assert_eq!(
            tail,
            vec![
                "section files_open",
                "collection files (0 members)",
                "section system_close",
            ]
        );
        let SystemPromptItem::Section(open) = &items[items.len() - 3] else {
            unreachable!()
        };
        assert_eq!(open.content, "\n# Knowledge Base\n\n");
        assert!(
            open.depends_on.is_none(),
            "files_open is in the birth prefix"
        );
    }

    /// `--ingest-dir <layer>=<path>` replaces a derived layer's content root, so
    /// a rebuild can be scoped to a subtree instead of sweeping the workspace.
    /// Layers without an override keep their derived root.
    #[test]
    fn ingest_dir_overrides_a_layers_content_root() {
        use crate::ingest::IngestMode;
        let builder = build_projection_builder(Path::new("demo-project"));
        let mut dirs = HashMap::new();
        dirs.insert("code_reading".to_string(), "candle/zend/src".to_string());

        let layers =
            crate::ingest::ingest_layers(builder.schema(), Path::new("demo-project"), &dirs);
        let code = layers
            .iter()
            .find(|l| l.name == "code_reading")
            .expect("code_reading derives");
        assert_eq!(code.folder, "candle/zend/src", "the scope is the override");
        assert_eq!(code.mode, IngestMode::Files, "mode is unchanged");
        assert!(
            code.display.contains("candle/zend/src"),
            "the scoped root shows in the loading phase label: {:?}",
            code.display
        );

        // An un-overridden layer keeps its derived scope: every repository.
        let repo = layers
            .iter()
            .find(|l| l.name == "repo_map")
            .expect("repo_map derives");
        assert_eq!(repo.folder, "");
        assert_eq!(repo.display, "Scanning repositories");
    }

    /// A coding-agent workspace with a folder matching a declared-but-pipeline-fed
    /// layer name (e.g. `code_reading/`) must NOT be ingested as a raw sink; only a
    /// mind (a workspace with its own `projection.yaml`) draws raw sinks from its
    /// folders. Guards the folder-name-collision fix.
    #[test]
    fn coding_workspace_folder_never_ingests_a_pipeline_layer() {
        use crate::ingest::IngestMode;
        let builder = build_projection_builder(Path::new("demo-project"));
        let ws = std::env::temp_dir().join(format!("zend_ingestws_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&ws);
        std::fs::create_dir_all(ws.join("code_reading")).unwrap();

        // Not a mind (no projection.yaml): the colliding `code_reading/` folder
        // is never a raw sink — the layer is read by its own pipeline.
        let coding = crate::ingest::ingest_layers(builder.schema(), &ws, &HashMap::new());
        assert!(
            !coding.iter().any(|l| l.mode == IngestMode::Raw),
            "no raw sinks in a coding-agent workspace"
        );
        assert!(coding
            .iter()
            .any(|l| l.name == "code_reading" && l.mode == IngestMode::Files));

        let _ = std::fs::remove_dir_all(&ws);
    }

    /// The shipped `projection.yaml` parses, and the repo map a dialogue sees is
    /// its working set's folders — no group-level threshold or default: a cold
    /// probe selects only what is pinned, because a folder is a provenance
    /// candidate only once its momentum clears `min_momentum`.
    #[test]
    fn projection_yaml_parses_and_repo_map_is_the_working_sets_folders() {
        let builder = build_projection_builder(Path::new("demo-project"));
        let structure = builder
            .id_for_group("structure")
            .expect("repo_map declares a 'structure' group");
        let group = builder.group(structure).expect("group schema present");
        assert_eq!(
            group.selection,
            SelectionRule::WorkingSet {
                share: WorkingSetShare::Folders
            },
        );
        assert_eq!(group.score_threshold, None);
    }
}
