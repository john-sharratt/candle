//! Regression test for the inline section-quantize boundary fix in
//! `candle-conversation`'s scheduler.
//!
//! Lives in `zend` rather than `candle-conversation` because it
//! exercises the full daemon stack: the production projection.yaml,
//! the real `zend::tools::install_tool_catalog`, and the real
//! Qwen3-30B-A3B model end-to-end.  Putting it here removes the
//! dev-dependency cycle that would otherwise exist
//! (`candle-conversation` test → `zend` → `candle-conversation`).
//!
//! Loads Qwen3-30B-A3B against a fresh empty workspace, installs the
//! production tool catalog, and verifies that:
//!
//!  1. After `base_conv` build returns, every section's hot bytes in
//!     `substrate.section.hot` are in the quantized byte-size range
//!     (i.e. the scheduler's `PrimingProjection`-end drain ran and
//!     swapped each residence's native bytes for their quantized form),
//!     and
//!
//!  2. A subsequent user turn produces a coherent response — not the
//!     `"\" {\" \"; 00 //1 \"1\"…"` gibberish that earlier
//!     asynchronous quantize paths produced when `substrate.section.hot`
//!     was mutated while priming projection / prefill kernels were
//!     still reading it.
//!
//! See `candle-conversation/src/scheduler/mod.rs::quantize_pending_sections`
//! for the production code path this test guards.

use std::path::Path;

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

/// Load the engine + base conversation exactly the way the daemon does:
/// production projection.yaml, full tool catalog from `zend::tools`,
/// argmax sampling, no thinking blocks, capped at 40 response tokens.
fn load_engine_and_base(workspace: &Path) -> (ConversationEngine, Sequence) {
    init_tracing();
    let device = cuda_device().expect("CUDA required");
    eprintln!(
        "=== Loading Qwen3-30B-A3B against {} ===",
        workspace.display()
    );
    let start = std::time::Instant::now();

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

    let tool_sections =
        zend::tools::install_tool_catalog(&mut proj_builder).expect("install tool catalog");
    eprintln!(
        "installed {} tool sections into projection schema",
        tool_sections.len()
    );

    let mut builder = Model::Qwen3_30B_A3B_Q4
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
    eprintln!("building base conversation...");
    let base_conv = engine
        .new_conversation_with_projection(
            &formatted_prompt,
            proj_builder,
            dialogue_layer,
            primary_group,
            conv_config,
        )
        .expect("new conv");
    eprintln!(
        "base conv built ({:.1}s since model load), turn_count = {}",
        start.elapsed().as_secs_f64(),
        base_conv.turn_count()
    );

    (engine, base_conv)
}

/// End-to-end check: the scheduler's `SealAction::Section →
/// PrimingProjection drain` path leaves every section in its quantized
/// form by the time `base_conv` build returns, *and* a subsequent
/// user turn against those quantized sections produces a coherent
/// response.
///
/// Earlier asynchronous attempts (persistence-thread Phase 2.5,
/// inline-at-section-seal) corrupted the prompt's K/V across 48 layers
/// of attention and produced JSON-fragment gibberish; this test guards
/// against regressing back to that path.
#[test]
#[ignore = "slow: full fresh prefill of 93 tool sections (~5 min)"]
fn section_quantize_end_to_end() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let workspace = tmp.path().to_path_buf();
    eprintln!("fresh workspace = {}", workspace.display());

    let (engine, base_conv) = load_engine_and_base(&workspace);

    // Native F16 chunks for Qwen3-30B-A3B (4 KV heads × head_dim 128 ×
    // CHUNK_SIZE 32 × 2 bytes × {K, V}) land at ~98 KiB.  Boundary
    // sections under C0/Q8/Q8 land at ~33 KiB; collection-member
    // sections under C3/Q8_KS K + adaptive V land at ~17 KiB.  Anything
    // < 64 KiB is unambiguously quantized.
    const NATIVE_F16_THRESHOLD: u64 = 65_536;
    let view = engine.conversation();
    let view = view.read();
    let section_ids = view.all_section_ids();
    let mut native_count = 0usize;
    let mut quantized_count = 0usize;
    for sid in &section_ids {
        let Some(sealed) = view.section_sealed_of(*sid) else {
            continue;
        };
        let Some(l0) = sealed.first() else { continue };
        let Some(c0) = l0.chunks.first() else {
            continue;
        };
        if c0.byte_size >= NATIVE_F16_THRESHOLD {
            native_count += 1;
        } else {
            quantized_count += 1;
        }
    }
    eprintln!(
        "post-build sections: total={}, quantized={}, native={}",
        section_ids.len(),
        quantized_count,
        native_count
    );
    assert_eq!(
        native_count, 0,
        "every section must be quantized post-build, but {native_count} are still native",
    );
    assert!(
        quantized_count > 0,
        "expected at least one quantized section",
    );
    drop(view);

    // Coherence check.  The exact wording shifts with the policy
    // (small K-side noise can nudge the model's tool selection), so
    // we don't pin a substring.  We reject the failure mode the
    // broken paths exhibited: short outputs full of unbalanced quotes,
    // `//` markers, and `\\\\` runs.
    let timeline_id = base_conv.timeline_id();
    let mut conv = base_conv.fork_resuming(timeline_id).expect("fork");
    let response = conv
        .send_turn("What's the current date and time?")
        .expect("send_turn");
    let text = response.text.clone();
    conv.close().expect("close");
    eprintln!("response: {:?}", text);
    assert!(
        text.len() >= 20,
        "response too short to be coherent: {:?}",
        text
    );
    let gibberish_score =
        text.matches("//").count() + text.matches("\\\\").count() + text.matches("\"\"").count();
    assert!(
        gibberish_score < 3,
        "response shows the broken-quantize-path failure pattern (gibberish_score={}): {:?}",
        gibberish_score,
        text
    );
}
