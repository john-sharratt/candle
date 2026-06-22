//! `compress_tools` — run the tools-collection categorize→assign workflow over
//! the **real** zend tool catalog and print the result.
//!
//! This drives the production execution path directly: it installs the real
//! catalog via [`install_tool_catalog`], reads the `tools` collection's
//! [`GroupSummary`] out of `projection.yaml`, hashes the catalog with
//! [`catalog_hash`], and generates the grouped summary with
//! [`generate_tool_summary`] — the same functions the daemon calls at startup
//! (`zend::session`). The summary is printed, not persisted (the daemon persists
//! it, keyed by the catalog hash).
//!
//! ```bash
//! cargo run -p zend --example compress_tools --features cuda --release
//! ```

use std::time::Instant;

use candle_conversation::models::Model;
use candle_conversation::projection::Builder;
use candle_conversation::SamplingConfig;
use zend::tool_summary::{catalog_hash, generate_tool_summary};
use zend::tools::install_tool_catalog;

const ZEND_YAML: &str = include_str!("../src/prompts/projection.yaml");

fn main() -> anyhow::Result<()> {
    let device = match candle::Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("compress_tools needs a CUDA device (it loads a real model). Aborting.");
            return Ok(());
        }
    };

    let tmp = tempfile::tempdir()?;
    let mut builder = Model::Qwen3_30B_A3B_Q4
        .builder()
        .sampling(SamplingConfig::top_k_top_p(40, 0.9, 0.5).with_repeat_penalty(1.1))
        .seed(42)
        .max_response_tokens(1024)
        .max_concurrent(4)
        .workspace_path(tmp.path());
    let config = builder.conversation_config();

    eprintln!("=== Loading {} ===", Model::Qwen3_30B_A3B_Q4);
    let t0 = Instant::now();
    let engine = builder
        .engine(&device)
        .map_err(|e| anyhow::anyhow!("engine: {e}"))?;
    eprintln!("    loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // Build the projection and install the real tool catalog into the dialogue
    // layer's `tools` collection — exactly what the daemon does at startup.
    let mut proj = Builder::from_yaml_with_vars_and_dialect(
        ZEND_YAML,
        &[("workspace", "tools-demo")],
        Some(&config.dialect),
    )
    .map_err(|e| anyhow::anyhow!("projection parse: {e}"))?;
    {
        let tok = engine.tokenizer();
        proj.tokenize_templates::<String, _>(|s| {
            tok.encode(s, false)
                .map(|enc| enc.get_ids().to_vec())
                .map_err(|e| e.to_string())
        })
        .map_err(|e| anyhow::anyhow!("tokenize templates: {e}"))?;
    }
    let dialogue = proj
        .id_for_layer("dialogue")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'dialogue' layer"))?;
    let tools = install_tool_catalog(&mut proj, dialogue)?;
    eprintln!("Installed {} tools.", tools.len());

    // Read the categorize→assign workflow out of the schema (config, not code).
    let gs = proj
        .schema()
        .layers
        .iter()
        .find(|l| l.id == dialogue)
        .and_then(|l| l.system_prompt.collection_named("tools"))
        .map(|c| c.summary.clone())
        .ok_or_else(|| anyhow::anyhow!("dialogue layer has no 'tools' collection"))?;

    eprintln!("catalog hash = {:032x}", catalog_hash(&tools));
    eprintln!("Generating summary (categorize → assign)...");
    let summary = generate_tool_summary(&engine, &tools, &gs, &config)?;

    println!("\n================ TOOLS SUMMARY ================\n");
    println!("{summary}");
    println!("\n==============================================");
    Ok(())
}
