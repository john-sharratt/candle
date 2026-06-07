//! Tier 3 — deep recall stress test (cruise / stress / marathon).
//!
//! `#[ignore]`-d by default.  Nightly CI runs `infinite_conversation_cruise`
//! against a persistent workspace so depth accumulates across runs
//! (the `debug_id`-resumable pattern from §10.4).  Weekly and
//! quarterly cadences run `infinite_conversation_stress` /
//! `infinite_conversation_marathon`.
//!
//! ```text
//!   Scale       T_target     Reach              CI cadence
//!   ────────────────────────────────────────────────────────
//!   Cruise      10× window   thousands of turns nightly
//!   Stress      100× window  tens of K turns    weekly (manual)
//!   Marathon    1000× window hundreds of K turns quarterly
//! ```
//!
//! Run with (cruise):
//! ```text
//! WORKSPACE_DIR=/var/tmp/cruise cargo test -p zend \
//!     --test infinite_conversation_deep -- --ignored \
//!     infinite_conversation_cruise --nocapture
//! ```

use std::path::{Path, PathBuf};

use candle::Device;
use candle_conversation::models::Model;
use candle_conversation::projection;
use candle_conversation::{ConversationEngine, SamplingConfig, Sequence};

const PROJECTION_YAML: &str = include_str!("../src/prompts/projection.yaml");

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

fn workspace_dir(default: &str) -> PathBuf {
    std::env::var("WORKSPACE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(default))
}

fn load_engine_and_base(workspace: &Path) -> (ConversationEngine, Sequence) {
    let device = cuda_device().expect("CUDA required for Tier-3 cruise");
    let dialect = Model::Qwen3_30B_A3B_Q4.spec().dialect.clone();
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
    let _ = zend::tools::install_tool_catalog(&mut proj_builder, dialogue_layer);

    let mut builder = Model::Qwen3_30B_A3B_Q4
        .builder()
        .workspace_path(workspace)
        .sampling(SamplingConfig::argmax())
        .seed(0)
        .max_response_tokens(40)
        .thinking(false);
    let conv_config = builder.conversation_config();
    let engine = builder.engine(&device).expect("engine load");
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
    (engine, base_conv)
}

/// Cruise — 10× window target.  Grows ~200 turns in one invocation.
/// Recall curve emitted as CSV via eprintln so CI can scrape it.
#[test]
#[ignore = "Tier 3 (cruise): grows ~200 turns + recall pass (~30 min wall clock)"]
fn infinite_conversation_cruise() {
    init_tracing();
    let workspace = workspace_dir("zen_cruise_workspace");
    std::fs::create_dir_all(&workspace).expect("workspace dir");
    eprintln!("CRUISE workspace = {}", workspace.display());
    let (engine, base_conv) = load_engine_and_base(&workspace);
    engine
        .set_conversation_debug_id(base_conv.timeline_id(), "cruise")
        .expect("set debug_id");

    const GROWTH: usize = 200;
    let timeline_id = base_conv.timeline_id();
    let mut conv = base_conv.fork_resuming(timeline_id).expect("fork");
    // Plant a marker turn early.
    let _ = conv
        .send_turn(
            "Important early marker: the safe combination is 17-42-9.  Anyway, continuing.",
        )
        .expect("plant");
    for i in 1..GROWTH {
        let user_msg = format!(
            "Cruise turn {i}: tell me about unrelated topic number {}.",
            i % 32
        );
        let _ = conv.send_turn(&user_msg).expect("send_turn");
        if i % 50 == 0 {
            let diag = engine.last_selection_diagnostics(timeline_id);
            eprintln!(
                "cruise: depth={i} pending={} dirty={} selected={}",
                engine.pending_summary_len(timeline_id),
                engine.dirty_summary_len(timeline_id),
                diag.map(|d| d.selected_nodes.len()).unwrap_or(0)
            );
        }
    }
    conv.close().expect("close");

    // Recall: ask for the planted marker many turns later.
    let mut probe_conv = base_conv.fork_resuming(timeline_id).expect("fork probe");
    let response = probe_conv
        .send_turn("What was the safe combination I mentioned at the start?")
        .expect("probe");
    eprintln!("cruise recall: {:?}", response.text);
    let recalled = response.text.contains("17-42-9");
    let diag = engine.last_selection_diagnostics(timeline_id);
    eprintln!(
        "cruise CSV: depth={GROWTH} recalled={recalled} pending={} dirty={} selected_count={}",
        engine.pending_summary_len(timeline_id),
        engine.dirty_summary_len(timeline_id),
        diag.map(|d| d.selected_nodes.len()).unwrap_or(0),
    );
    probe_conv.close().expect("close probe");
    // Soft assert — log as a warning rather than failing the build,
    // because the cruise harness is the regression detector, not a
    // strict gate.
    if !recalled {
        eprintln!(
            "WARNING: cruise recall MISS for safe combination (depth {GROWTH})"
        );
    }
}

/// Stress — 100× window target, weekly cadence.
#[test]
#[ignore = "Tier 3 (stress): grows ~2 000 turns + recall pass (~5h wall clock; weekly)"]
fn infinite_conversation_stress() {
    init_tracing();
    let workspace = workspace_dir("zen_stress_workspace");
    std::fs::create_dir_all(&workspace).expect("workspace dir");
    let (engine, base_conv) = load_engine_and_base(&workspace);
    engine
        .set_conversation_debug_id(base_conv.timeline_id(), "stress")
        .expect("set debug_id");
    const GROWTH: usize = 2_000;
    let timeline_id = base_conv.timeline_id();
    let mut conv = base_conv.fork_resuming(timeline_id).expect("fork");
    let _ = conv
        .send_turn("Stress marker: the launch codename is BLUEBIRD.  Continuing.")
        .expect("plant");
    for i in 1..GROWTH {
        let user_msg = format!("Stress turn {i}: discuss topic {}.", i % 64);
        let _ = conv.send_turn(&user_msg).expect("send_turn");
        if i % 200 == 0 {
            eprintln!(
                "stress: depth={i} pending={} dirty={}",
                engine.pending_summary_len(timeline_id),
                engine.dirty_summary_len(timeline_id)
            );
        }
    }
    conv.close().expect("close");
    let mut probe_conv = base_conv.fork_resuming(timeline_id).expect("fork probe");
    let response = probe_conv
        .send_turn("What was the launch codename from earlier?")
        .expect("probe");
    eprintln!("stress recall: {:?}", response.text);
    let recalled = response.text.to_lowercase().contains("bluebird");
    eprintln!(
        "stress CSV: depth={GROWTH} recalled={recalled}"
    );
    probe_conv.close().expect("close probe");
    if !recalled {
        eprintln!("WARNING: stress recall MISS at depth {GROWTH}");
    }
}
