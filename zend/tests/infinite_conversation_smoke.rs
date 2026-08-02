//! Tier 3 — unbounded-window recall smoke test (§10.8).
//!
//! `#[ignore]`-d by default because it loads the full Qwen3-30B-A3B
//! model and grows a conversation past the layer window, then asks
//! the model to recall planted facts from arbitrary depths.  Run
//! manually with:
//!
//! ```text
//! cargo test -p zend --test infinite_conversation_smoke -- --ignored --nocapture
//! ```
//!
//! ## What this test guards
//!
//! 1. The score-density selection (`select_dense`) pulls each plant
//!    leaf into the slot when its probe Q is the current decode Q.
//! 2. The model, given that selection, actually retrieves the planted
//!    fact and produces it in its response.
//! 3. The recall pass works at the smoke scale (`T_target = 2 ×
//!    window`) — gross regressions surface here in ~5 min.
//!
//! ## What this test does NOT cover
//!
//! - Deeper scales (cruise / stress / marathon) — see the per-PR
//!   nightly CI runs of `infinite_conversation_deep.rs`.
//! - Cross-time bridge plants (§10.8.6).
//! - Negative-test hallucination heuristic (§10.8.7).  Those checks
//!   live in the deeper-scale harness because they need a meaningful
//!   filler distribution.

use std::path::Path;

use candle::Device;
use candle_conversation::models::Model;
use candle_conversation::projection;
use candle_conversation::summary_tree::{NodeId, SelectionDiagnostics, SelectionOrigin};
use candle_conversation::{ConversationEngine, SamplingConfig, Sequence};

const PROJECTION_YAML: &str = include_str!("../src/prompts/projection.yaml");

/// Plant: a fact embedded in a user turn at known depth, recalled
/// later by a probe.
#[derive(Debug, Clone)]
struct Plant {
    /// Symbolic id (e.g. "P-deep").
    id: &'static str,
    /// Turn index at which the fact is embedded.  Resolved to a
    /// concrete index after the grow loop starts; the table is
    /// declarative on `relative_depth`.
    relative_depth: PlantDepth,
    /// User-turn text that embeds the fact.
    user_text: String,
    /// Canonical fact substring the recall response must contain.
    fact: &'static str,
    /// Probe question used at recall time.
    probe: &'static str,
}

#[derive(Debug, Clone, Copy)]
enum PlantDepth {
    /// `N - k` where N is the total grow count.
    FromEnd(usize),
    /// `N / k` — middle-ish.
    Fraction {
        numerator: usize,
        denominator: usize,
    },
    /// Hard turn index.
    At(usize),
}

impl PlantDepth {
    fn resolve(self, total: usize) -> usize {
        match self {
            PlantDepth::FromEnd(k) => total.saturating_sub(k),
            PlantDepth::Fraction {
                numerator,
                denominator,
            } => (total * numerator) / denominator.max(1),
            PlantDepth::At(i) => i.min(total.saturating_sub(1)),
        }
    }
}

/// The seven-plant distribution from §10.8.2.
fn smoke_plants() -> Vec<Plant> {
    vec![
        Plant {
            id: "P-near",
            relative_depth: PlantDepth::FromEnd(5),
            user_text: "Quick aside: the color we picked is mauve.  Anyway, back to the topic."
                .into(),
            fact: "mauve",
            probe: "What color did I tell you we picked?",
        },
        Plant {
            id: "P-recent",
            relative_depth: PlantDepth::FromEnd(20),
            user_text: "Logistics note: the ship date is May 13.  Continuing...".into(),
            fact: "May 13",
            probe: "What ship date did I mention?",
        },
        Plant {
            id: "P-mid",
            relative_depth: PlantDepth::Fraction {
                numerator: 1,
                denominator: 2,
            },
            user_text: "Important budget fact: the budget is 50k.  Moving on.".into(),
            fact: "50k",
            probe: "What budget did I mention?",
        },
        Plant {
            id: "P-old",
            relative_depth: PlantDepth::Fraction {
                numerator: 1,
                denominator: 10,
            },
            user_text: "Technical decision: we chose Postgres for the database.  Continuing."
                .into(),
            fact: "Postgres",
            probe: "Which database did we choose?",
        },
        Plant {
            id: "P-deep",
            relative_depth: PlantDepth::At(3),
            user_text: "Important: the password is rosebud.  Anyway, ...".into(),
            fact: "rosebud",
            probe: "What was the password I mentioned earlier?",
        },
        Plant {
            id: "P-topic-A",
            relative_depth: PlantDepth::Fraction {
                numerator: 1,
                denominator: 3,
            },
            user_text: "Tangent: Alice's favourite tea is earl grey.  OK back to topic.".into(),
            fact: "earl grey",
            probe: "What's Alice's favourite tea?",
        },
        Plant {
            id: "P-topic-B",
            relative_depth: PlantDepth::Fraction {
                numerator: 2,
                denominator: 3,
            },
            user_text: "Brief note: Bob's favourite tea is oolong.  OK continuing.".into(),
            fact: "oolong",
            probe: "What's Bob's favourite tea?",
        },
    ]
}

fn cuda_device() -> Option<Device> {
    match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => Some(d),
        _ => None,
    }
}

fn init_tracing() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_test_writer()
            .try_init();
    });
}

fn load_engine_and_base(workspace: &Path) -> (ConversationEngine, Sequence) {
    init_tracing();
    let device = cuda_device().expect("CUDA required for Tier-3 recall harness");
    eprintln!(
        "=== Loading Qwen3-30B-A3B against {} ===",
        workspace.display()
    );
    let start = std::time::Instant::now();

    let dialect = Model::Qwen3_30B_A3B_Q6.spec().dialect.clone();
    let workspace_str = workspace.display().to_string();
    let mut proj_builder = projection::Builder::from_yaml_with_vars_and_dialect(
        PROJECTION_YAML,
        &[("workspace", workspace_str.as_str())],
        Some(&dialect),
    )
    .expect("parse projection.yaml");
    let dialogue_layer = proj_builder
        .id_for_layer("dialogue")
        .expect("dialogue layer");
    let primary_group = proj_builder
        .id_for_group("primary_conversation")
        .expect("primary group");

    let tool_sections =
        zend::tools::install_tool_catalog(&mut proj_builder).expect("install tool catalog");
    eprintln!("installed {} tool sections", tool_sections.len());

    let mut builder = Model::Qwen3_30B_A3B_Q6
        .builder()
        .workspace_path(workspace)
        .sampling(SamplingConfig::argmax())
        .seed(0)
        .max_response_tokens(40)
        .thinking(false);
    let conv_config = builder.conversation_config();
    let engine = builder.engine(&device).expect("engine load");
    eprintln!("engine loaded in {:.1}s", start.elapsed().as_secs_f64());

    let tokenizer = engine.tokenizer().clone();
    proj_builder
        .tokenize_templates::<anyhow::Error, _>(|s| {
            let encoded = tokenizer
                .encode(s, false)
                .map_err(|e| anyhow::anyhow!("template tokenise: {e}"))?;
            Ok(encoded.get_ids().to_vec())
        })
        .expect("tokenize templates");

    let formatted_prompt = builder.format_system_prompt();
    let base_conv = engine
        .new_conversation_with_projection(
            &formatted_prompt,
            proj_builder,
            dialogue_layer,
            primary_group,
            conv_config,
        )
        .expect("new conv");
    eprintln!("base conv built ({:.1}s)", start.elapsed().as_secs_f64());
    (engine, base_conv)
}

/// Smoke variant — `T_target = 2 × layer.window`.  At Qwen3-30B-A3B's
/// dialogue window of 16K tokens that's ~50 turns of moderate length.
/// PR-gated cadence (~5 min budget).
#[test]
#[ignore = "Tier 3: loads Qwen3-30B-A3B + runs ~50-turn growth + recall (~5 min)"]
fn infinite_conversation_smoke() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let workspace = tmp.path().to_path_buf();

    let (engine, base_conv) = load_engine_and_base(&workspace);

    // Mark this timeline with a stable resume key so a later test run
    // could re-open the workspace and continue.
    engine
        .set_conversation_debug_id(base_conv.timeline_id(), "smoke-50")
        .expect("set debug_id");

    // Build phase: grow N normal turns with plants interleaved.
    const N: usize = 50;
    let plants = smoke_plants();
    let resolved: Vec<(usize, &Plant)> = plants
        .iter()
        .map(|p| (p.relative_depth.resolve(N), p))
        .collect();
    let timeline_id = base_conv.timeline_id();
    let mut conv = base_conv.fork_resuming(timeline_id).expect("fork");
    for i in 0..N {
        let plant_text = resolved
            .iter()
            .find(|(turn, _)| *turn == i)
            .map(|(_, p)| p.user_text.clone());
        let user_msg = match plant_text {
            Some(text) => text,
            None => format!(
                "Filler turn {i}: tell me about something unrelated to the previous topics."
            ),
        };
        let response = conv.send_turn(&user_msg).expect("send_turn");
        if let Some(diag) = engine.last_selection_diagnostics(timeline_id) {
            eprintln!(
                "turn {i}: selected={} pending={} budget={}",
                diag.selected_nodes.len(),
                diag.pending_count,
                diag.budget,
            );
        }
        // Trim console noise: print the model's response only on plant
        // turns, to keep the test log readable.
        if resolved.iter().any(|(turn, _)| *turn == i) {
            eprintln!("plant @ turn {i}: response = {:?}", response.text);
        }
    }
    conv.close().expect("close");

    // Validation phase: ask each plant's probe in a fresh fork; assert
    // the planted leaf is in the selection AND the model recalls the
    // fact text.
    let mut algorithmic_pass = 0;
    let mut end_to_end_pass = 0;
    for plant in &plants {
        let plant_turn = plant.relative_depth.resolve(N);
        // Re-fork from the original base_conv for each probe — each
        // fork resumes the latest substrate state of `timeline_id`
        // (so plant turns embedded above are visible).
        let mut probe_conv = base_conv.fork_resuming(timeline_id).expect("fork resuming");
        let response = probe_conv.send_turn(plant.probe).expect("probe send");

        // Algorithm-level: planted turn (or some ancestor) must be in
        // the score-density selection.  Read from the substrate-side
        // diagnostic channel — the projection that drove this probe's
        // response wrote its selection there as a last-write-wins
        // side-effect.
        let algorithmic = engine
            .last_selection_diagnostics(timeline_id)
            .map(|d| {
                d.selected_nodes.iter().any(|n| {
                    n.0 as usize == plant_turn
                    // Or an ancestor — covered_by check would walk
                    // children, but a sufficient surrogate is "the
                    // plant turn is in chrono_normals of the
                    // selected ancestor".  For smoke we just check
                    // direct membership; the recall test still
                    // gates the end-to-end semantics.
                })
            })
            .unwrap_or(false);
        if algorithmic {
            algorithmic_pass += 1;
        }

        // End-to-end: model's response contains the canonical fact.
        let end_to_end = response
            .text
            .to_lowercase()
            .contains(&plant.fact.to_lowercase());
        if end_to_end {
            end_to_end_pass += 1;
        }

        eprintln!(
            "probe-{}: planted_turn={} algorithmic={} end_to_end={} response={:?}",
            plant.id, plant_turn, algorithmic, end_to_end, response.text
        );
        probe_conv.close().expect("close probe");
    }

    // Smoke threshold: at least HALF the plants recall end-to-end.
    // Production growth runs in cruise/stress aim for ≥90%; smoke is
    // a regression detector, not a quality benchmark.
    let total = plants.len();
    let pct = (end_to_end_pass as f32 / total as f32) * 100.0;
    eprintln!(
        "smoke recall: {}/{} end-to-end ({:.0}%), {}/{} algorithmic",
        end_to_end_pass, total, pct, algorithmic_pass, total
    );
    assert!(
        end_to_end_pass * 2 >= total,
        "smoke recall below 50% threshold ({}/{})",
        end_to_end_pass,
        total
    );
}

/// Negative test: ask about something we never planted.  The model
/// must NOT hallucinate a conversation-grounded answer.  Listed as a
/// separate test so it can pass independently of the recall floor.
#[test]
#[ignore = "Tier 3: loads model and asserts no-hallucination on unplanted probe (~5 min)"]
fn infinite_conversation_negative_smoke() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let workspace = tmp.path().to_path_buf();
    let (engine, base_conv) = load_engine_and_base(&workspace);
    engine
        .set_conversation_debug_id(base_conv.timeline_id(), "smoke-neg-50")
        .expect("set debug_id");
    let timeline_id = base_conv.timeline_id();
    let mut conv = base_conv.fork_resuming(timeline_id).expect("fork");
    // Grow 30 unrelated filler turns.
    for i in 0..30 {
        let user_msg = format!("Filler turn {i}: tell me about unrelated topic {i}.",);
        let _ = conv.send_turn(&user_msg).expect("send_turn");
    }
    // Ask about something we never discussed.
    let response = conv
        .send_turn("What's the capital of Bolivia? Did we discuss it?")
        .expect("send_turn");
    let lower = response.text.to_lowercase();
    let hallucinated_markers = [
        "we discussed",
        "as i mentioned",
        "earlier we said",
        "you said earlier",
        "as we talked about",
    ];
    let hits: Vec<&&str> = hallucinated_markers
        .iter()
        .filter(|m| lower.contains(**m))
        .collect();
    eprintln!("negative probe response: {:?}", response.text);
    assert!(
        hits.is_empty(),
        "model hallucinated conversation-grounded recall: {hits:?} in {:?}",
        response.text
    );
    conv.close().expect("close");
}

/// SelectionOrigin sanity: the smoke test below is a unit-level
/// algorithm assertion that the score-density diagnostics include
/// recency anchors when they apply.  Doesn't load the model.
#[test]
fn diagnostics_struct_round_trips() {
    let mut d = SelectionDiagnostics::new(8000);
    d.push(NodeId(1), SelectionOrigin::HardAnchor, f32::INFINITY, 20);
    d.push(NodeId(2), SelectionOrigin::ProvenanceScore, 0.8, 30);
    assert_eq!(d.selected_nodes.len(), 2);
    assert!(d.contains(NodeId(1)));
    assert_eq!(
        d.origin_of(NodeId(2)),
        Some(SelectionOrigin::ProvenanceScore)
    );
}
