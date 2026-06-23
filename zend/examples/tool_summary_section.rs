//! `tool_summary_section` — verify the tool-summary **storage**: that a generated
//! summary, sealed as a section re-prefilled with the tools' prefix, pins KV in
//! the substrate and survives a reload (restored from disk, content-addressed).
//!
//! This exercises exactly what the daemon's startup hook does for the section
//! seal (minus the model generation — a fake summary stands in), so it confirms
//! `insert_section_with_prefix` works when called after the base conversation is
//! constructed, and that the reserved `ToolSummary` section round-trips a restart.
//!
//! ```bash
//! cargo run -p zend --example tool_summary_section --features cuda --release
//! ```

use candle_conversation::models::Model;
use candle_conversation::projection::{Builder, Reserved, SectionId, SystemPromptItem};
use candle_conversation::{ConversationEngine, SequenceConfig};
use zend::tools::install_tool_catalog;

const ZEND_YAML: &str = include_str!("../src/prompts/projection.yaml");
const FAKE_SUMMARY: &str = "## Files\n  file_read, file_write\n## Network\n  dns_lookup, port_scan";

/// Build the projection + install the catalog. Returns the builder (ready to hand
/// to `new_conversation_with_projection`) plus the dialogue/group ids and the
/// non-template section ids before the `tools` collection (the seal prefix).
fn build_projection(
    engine: &ConversationEngine,
    cfg: &SequenceConfig,
) -> anyhow::Result<(
    Builder,
    candle_conversation::projection::LayerId,
    candle_conversation::projection::GroupId,
    String,
    Vec<SectionId>,
)> {
    let mut proj = Builder::from_yaml_with_vars_and_dialect(
        ZEND_YAML,
        &[("workspace", "ts-sec")],
        Some(&cfg.dialect),
    )
    .map_err(|e| anyhow::anyhow!("projection parse: {e}"))?;
    let tok = engine.tokenizer();
    proj.tokenize_templates::<String, _>(|s| {
        tok.encode(s, false)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| e.to_string())
    })
    .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let dialogue = proj
        .id_for_layer("dialogue")
        .ok_or_else(|| anyhow::anyhow!("no dialogue layer"))?;
    let group = proj
        .id_for_group("primary_conversation")
        .ok_or_else(|| anyhow::anyhow!("no primary_conversation group"))?;
    install_tool_catalog(&mut proj, dialogue)?;

    // Prelude text (sections before the tools collection) + the non-template
    // section ids that make up the seal prefix.
    let layer = proj
        .schema()
        .layers
        .iter()
        .find(|l| l.id == dialogue)
        .ok_or_else(|| anyhow::anyhow!("dialogue layer vanished"))?;
    let mut prelude_text = String::new();
    let mut prelude_ids = Vec::new();
    for item in &layer.system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => {
                prelude_text.push_str(&s.content);
                if !s.is_template {
                    prelude_ids.push(s.id);
                }
            }
            SystemPromptItem::SectionTree(t) => {
                for n in &t.nodes {
                    prelude_text.push_str(&n.options[n.chosen(&t.default_selection)].content);
                }
                prelude_ids.extend(t.default_present_ids.iter().copied());
            }
            SystemPromptItem::Collection(_) => break,
        }
    }
    let formatted = cfg.dialect.format_system_prompt(&prelude_text);
    Ok((proj, dialogue, group, formatted, prelude_ids))
}

fn load_engine(
    device: &candle::Device,
    ws: &std::path::Path,
) -> anyhow::Result<ConversationEngine> {
    Model::Qwen3_30B_A3B_Q4
        .builder()
        .seed(42)
        .max_concurrent(4)
        .workspace_path(ws.to_path_buf())
        .engine(device)
        .map_err(|e| anyhow::anyhow!("engine: {e}"))
}

fn main() -> anyhow::Result<()> {
    let device = match candle::Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("tool_summary_section needs a CUDA device. Aborting.");
            return Ok(());
        }
    };
    let ws = tempfile::tempdir()?;
    let sid = SectionId::reserved(Reserved::ToolSummary);

    // ── Load 1: seal the summary section, verify it pins KV.
    {
        let engine = load_engine(&device, ws.path())?;
        let cfg = Model::Qwen3_30B_A3B_Q4.builder().conversation_config();
        let (proj, dialogue, group, formatted, prelude) = build_projection(&engine, &cfg)?;
        println!("prefix sections before tools: {}", prelude.len());

        let mut base = engine
            .new_conversation_with_projection(&formatted, proj, dialogue, group, cfg)
            .map_err(|e| anyhow::anyhow!("base conv: {e}"))?;

        base.insert_section_with_prefix(sid, FAKE_SUMMARY, &prelude)
            .map_err(|e| anyhow::anyhow!("seal summary section: {e}"))?;

        let conv = engine.conversation();
        let sealed = conv.read().section_sealed_of(sid).is_some();
        let toks = conv.read().section_tokens_of(sid).len();
        let (b0, b1) = conv.read().section_block_range(sid);
        println!("load 1: sealed={sealed}  tokens={toks}  blocks=[{b0},{b1})");
        assert!(sealed, "summary section must be pinned in the substrate");
        assert!(toks > 0, "summary section must carry its tokens");
        engine.checkpoint_persistence()?;
        println!("load 1: sealed + persisted ✓");
    }

    // ── Load 2 (restart): re-seal with the same text → restored from disk.
    {
        let engine = load_engine(&device, ws.path())?;
        let cfg = Model::Qwen3_30B_A3B_Q4.builder().conversation_config();
        let (proj, dialogue, group, formatted, prelude) = build_projection(&engine, &cfg)?;
        let mut base = engine
            .new_conversation_with_projection(&formatted, proj, dialogue, group, cfg)
            .map_err(|e| anyhow::anyhow!("base conv: {e}"))?;
        // Same text + same prefix → same content address → restore (no re-prefill).
        base.insert_section_with_prefix(sid, FAKE_SUMMARY, &prelude)
            .map_err(|e| anyhow::anyhow!("restore summary section: {e}"))?;
        let conv = engine.conversation();
        let sealed = conv.read().section_sealed_of(sid).is_some();
        let toks = conv.read().section_tokens_of(sid).len();
        println!("load 2: sealed={sealed}  tokens={toks}");
        assert!(sealed, "summary section must restore across restart");
        assert!(toks > 0);
        println!("load 2: restored from disk ✓");
    }

    println!("\nALL CHECKS PASSED");
    Ok(())
}
