//! `tool_summary_cache` — end-to-end check of the tool-summary cache mechanism
//! through the **real engine** (persist → reload → cache hit), without paying the
//! minute-long categorize→assign generation (a fake summary stands in; the model
//! generation itself is covered by `compress_tools`).
//!
//! It loads an engine on a workspace, installs the real catalog, hashes it, and
//! persists a tool summary keyed by that hash; then it drops the engine, reloads
//! a fresh engine on the **same** workspace, and asserts the reloaded substrate
//! still reports the cached hash — i.e. a restart is a cache hit. Finally it
//! supersedes with a second hash and confirms last-writer-wins.
//!
//! ```bash
//! cargo run -p zend --example tool_summary_cache --features cuda --release
//! ```

use candle_conversation::models::Model;
use candle_conversation::persistence::record::{ToolSummaryEntry, ToolSummaryPayload};
use candle_conversation::projection::Builder;
use zend::tool_summary::catalog_hash;
use zend::tools::install_tool_catalog;

const ZEND_YAML: &str = include_str!("../src/prompts/projection.yaml");

/// Build a single-entry payload (comprehensive slot) for the cache demo.
fn comp_payload(catalog_hash: u128, summary: &str) -> ToolSummaryPayload {
    ToolSummaryPayload {
        comprehensive: Some(ToolSummaryEntry {
            catalog_hash,
            summary: summary.to_string(),
        }),
        restricted: None,
    }
}

/// Install the real catalog into a throwaway projection and return its hash.
/// The hash is over the tools' content (name + JSON), independent of the dialect
/// used to parse the surrounding template, so any valid dialect serves.
fn catalog_hash_for(engine: &candle_conversation::ConversationEngine) -> anyhow::Result<u128> {
    let dialect = Model::Qwen3_30B_A3B_Q4
        .builder()
        .conversation_config()
        .dialect;
    let mut proj = Builder::from_yaml_with_vars_and_dialect(
        ZEND_YAML,
        &[("workspace", "tool-cache")],
        Some(&dialect),
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
    let tools = install_tool_catalog(&mut proj, dialogue)?;
    Ok(catalog_hash(&tools))
}

fn load_engine(
    device: &candle::Device,
    workspace: &std::path::Path,
) -> anyhow::Result<candle_conversation::ConversationEngine> {
    Model::Qwen3_30B_A3B_Q4
        .builder()
        .seed(42)
        .max_concurrent(4)
        .workspace_path(workspace.to_path_buf())
        .engine(device)
        .map_err(|e| anyhow::anyhow!("engine: {e}"))
}

fn main() -> anyhow::Result<()> {
    let device = match candle::Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("tool_summary_cache needs a CUDA device. Aborting.");
            return Ok(());
        }
    };
    // One workspace, kept alive across both engine loads.
    let ws = tempfile::tempdir()?;

    // ── Load 1: empty cache → persist a summary keyed by the catalog hash.
    let hash1;
    {
        let engine = load_engine(&device, ws.path())?;
        hash1 = catalog_hash_for(&engine)?;
        let conv = engine.conversation();
        let before = conv.read().tool_summary_hash(false);
        println!("load 1: catalog hash = {hash1:032x}, cached = {before:?}");
        assert_eq!(before, None, "fresh workspace must have no cached summary");
        conv.write_tool_summary(comp_payload(hash1, "FAKE SUMMARY v1"))?;
        engine.checkpoint_persistence()?;
        println!("load 1: persisted summary for hash {hash1:032x}");
    }

    // ── Load 2 (restart): the reloaded substrate must report the cached hash,
    // and the recomputed catalog hash must match → cache hit, no regeneration.
    {
        let engine = load_engine(&device, ws.path())?;
        let hash2 = catalog_hash_for(&engine)?;
        let cached = engine.conversation().read().tool_summary_hash(false);
        let text = engine
            .conversation()
            .read()
            .tool_summary_text(false)
            .map(str::to_string);
        println!("load 2: recomputed hash = {hash2:032x}, cached = {cached:?}, text = {text:?}");
        assert_eq!(hash2, hash1, "catalog hash must be stable across restarts");
        assert_eq!(cached, Some(hash1), "restart must see the cached summary");
        assert_eq!(text.as_deref(), Some("FAKE SUMMARY v1"));
        println!("load 2: CACHE HIT — catalog unchanged, no regeneration needed ✓");

        // ── Supersede: a different hash overwrites; last-writer-wins.
        let conv = engine.conversation();
        conv.write_tool_summary(comp_payload(hash1 ^ 0xFFFF, "FAKE SUMMARY v2"))?;
        engine.checkpoint_persistence()?;
        assert_eq!(conv.read().tool_summary_hash(false), Some(hash1 ^ 0xFFFF));
        assert_eq!(
            conv.read().tool_summary_text(false),
            Some("FAKE SUMMARY v2")
        );
        println!("supersede: new hash + text won (last-writer-wins) ✓");
    }

    // ── Load 3: the superseded value is what survives the reload.
    {
        let engine = load_engine(&device, ws.path())?;
        let cached = engine.conversation().read().tool_summary_hash(false);
        assert_eq!(
            cached,
            Some(hash1 ^ 0xFFFF),
            "the superseding value survives reload"
        );
        println!("load 3: superseded value survived reload ✓");
    }

    println!("\nALL CHECKS PASSED");
    Ok(())
}
