//! `gen_tool_cases` — capture per-tool decode-Q probes (and user-turn raw Q/V)
//! against the **full materialised tool view** in the substrate.
//!
//! Solves the chicken-and-egg problem: provenance can't select a tool that isn't
//! in context, and a tool that isn't selected can't be invoked. Here the `tools`
//! collection is forced to `AllVisible`, so **every** turn materialises all ~93
//! tool sections' KV. The user/assistant turns therefore react directly to the
//! same tool KV we model the provenance scan against.
//!
//! Per tool we capture a **train** and a **holdout** case (prompts authored in
//! `prompts.json`): submit the user question, let the model emit a tool call,
//! stop at end-of-turn (the call is captured, never executed), and record:
//!   - the **decode-Q** sign-signatures (the invocation probe) → `signatures.prov`;
//!   - the **user-turn raw K/V/Q** vectors (band layers)        → `raw_kvq.prov`.
//!
//! ```sh
//! cargo run -p zend --example gen_tool_cases --release --features cuda -- \
//!   --model-path  <gguf> --tokenizer-path <tok> \
//!   --output      candle-conversation/tests/tool_cases
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle::Device;
use candle_conversation::models::{Dialect, Model};
use candle_conversation::projection::{Builder, SystemPromptItem};
use candle_conversation::provenance::{
    band_layer_indices, build_token_blob, RawFileHeader, RawProvenanceFile, RawSigEntry,
};
use candle_conversation::{ProvenanceFile, SamplingConfig, SigEntry};
use clap::Parser;
use serde::{Deserialize, Serialize};
use zend::tools::install_tool_catalog;

const PROJECTION_SCHEMA_TEMPLATE: &str = include_str!("../src/prompts/projection.yaml");
/// Shared Rust/CUDA block size (`CHUNK_SIZE`); zend has no direct dep.
const CHUNK_SIZE: usize = 32;

#[derive(Parser)]
#[command(about = "Capture per-tool decode-Q probes + user-turn raw Q/V over the full tool view")]
struct Args {
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long)]
    model_path: Option<PathBuf>,
    #[arg(long)]
    tokenizer_path: Option<PathBuf>,

    /// Directory with prompts.json (per-tool train + holdout prompts) and the
    /// output fixtures (signatures.prov / raw_kvq.prov / MANIFEST.json).
    #[arg(
        long,
        default_value_os_t = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../candle-conversation/tests/tool_cases"
        ))
    )]
    output: PathBuf,

    /// Scratch workspace whose substrate seals the full tool catalog (NOT your
    /// real daemon workspace). The probes + corpus both come from here.
    #[arg(long, default_value_os_t = std::env::temp_dir().join("zend_tool_cases"))]
    workspace: PathBuf,

    /// Token cap per case (the model stops at end-of-turn after the tool call).
    #[arg(long, default_value = "96")]
    max_tokens: usize,

    /// Only run the first N tool cases (quick verification). 0 = all.
    #[arg(long, default_value = "0")]
    limit: usize,

    #[arg(long)]
    force: bool,
}

#[derive(Deserialize)]
struct PromptCase {
    tool: String,
    train: String,
    holdout: String,
}
#[derive(Deserialize)]
struct PromptsFile {
    cases: Vec<PromptCase>,
}

/// Output scenario — superset of what `tool_select_from_substrate` reads
/// (`split` distinguishes train vs holdout).
#[derive(Serialize)]
struct OutScenario {
    id: String,
    tool: Option<String>,
    case_type: String,
    split: String,
    user_prompt: String,
    generated_text: String,
    byte_offset: u64,
    token_count: u16,
    /// Index into RAW_MANIFEST.json for this case's user-turn raw K/V/Q.
    raw_byte_offset: Option<u64>,
    raw_token_count: Option<u16>,
}

#[derive(Serialize)]
struct OutManifest {
    version: u32,
    model: String,
    context: String,
    provenance_layer_indices: [usize; 6],
    scenarios: Vec<OutScenario>,
}

#[derive(Serialize)]
struct RawOutManifest {
    version: u32,
    model: String,
    n_kv_heads: u32,
    head_dim: u32,
    n_layers_per_band: u32,
    band_half_width: u32,
    band_centers: [u32; 3],
    n_total_layers: u32,
}

/// Build the production projection, but with the `tools` collection forced to
/// `AllVisible` so the entire catalog materialises every turn.
fn build_projection_builder(workspace: &std::path::Path) -> anyhow::Result<Builder> {
    let name = workspace
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("this project");
    let forced = PROJECTION_SCHEMA_TEMPLATE.replace(
        "name: tools\n          selection: { kind: top_k, k: 5 }",
        "name: tools\n          selection: { kind: always_visible }",
    );
    if forced == PROJECTION_SCHEMA_TEMPLATE {
        anyhow::bail!("could not force tools collection to AllVisible — YAML shape changed");
    }
    let dialect = Dialect::chat_ml();
    Builder::from_yaml_with_vars_and_dialect(&forced, &[("workspace", name)], Some(&dialect))
        .map_err(|e| anyhow::anyhow!("projection.yaml parse: {e}"))
}

/// Dialogue-layer section content up to the first collection (the ChatML-wrapped
/// system prompt; everything from `tools` on is expanded by the projection).
fn pre_collection_prelude(builder: &Builder) -> String {
    let Some(layer) = builder
        .schema()
        .layers
        .iter()
        .find(|l| l.name == "dialogue")
    else {
        return String::new();
    };
    let mut out = String::new();
    for item in &layer.system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => out.push_str(&s.content),
            SystemPromptItem::SectionTree(t) => {
                for n in &t.nodes {
                    out.push_str(&n.options[n.chosen(&t.default_selection)].content);
                }
            }
            SystemPromptItem::Collection(_) => break,
        }
    }
    out
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // ── Prompts.
    let prompts_path = args.output.join("prompts.json");
    let prompts: PromptsFile = serde_json::from_str(
        &std::fs::read_to_string(&prompts_path)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", prompts_path.display()))?,
    )?;
    println!(
        "loaded {} tool cases from {}",
        prompts.cases.len(),
        prompts_path.display()
    );

    // ── Output guard.
    let sig_path = args.output.join("signatures.prov");
    let raw_path = args.output.join("raw_kvq.prov");
    for p in [&sig_path, &raw_path] {
        if p.exists() {
            if !args.force {
                anyhow::bail!("{} exists — pass --force to overwrite", p.display());
            }
            std::fs::remove_file(p)?;
        }
    }
    std::fs::create_dir_all(&args.workspace)?;

    // ── Projection (tools forced AllVisible) + catalog.
    let mut proj = build_projection_builder(&args.workspace)?;
    let dialogue = proj
        .id_for_layer("dialogue")
        .ok_or_else(|| anyhow::anyhow!("schema missing 'dialogue' layer"))?;
    let group = proj
        .id_for_group("primary_conversation")
        .ok_or_else(|| anyhow::anyhow!("schema missing 'primary_conversation' group"))?;
    let tool_sections = install_tool_catalog(&mut proj, dialogue)?;
    println!(
        "installed {} tools (forced AllVisible)",
        tool_sections.len()
    );
    // Capture mandate: the model answers knowledge questions directly unless
    // forced. Require exactly one tool call per turn so every case yields an
    // invocation probe — the model still chooses WHICH tool.
    let before_text = format!(
        "{}\n\n# Tool-capture mode\n\
         For EVERY user message you MUST respond with exactly one tool call of the form \
         <tool_call>{{\"name\": \"<tool>\", \"arguments\": {{...}}}}</tool_call> and nothing else. \
         Always choose the single most appropriate tool for the request. Never answer from your \
         own knowledge, never refuse, never add any text before or after — output only the tool call.\n",
        pre_collection_prelude(&proj),
    );

    // ── Model.
    let device = Device::cuda_if_available(0)?;
    println!("device: {device:?}");
    let t_load = Instant::now();
    let mut mbuilder = Model::Qwen3_30B_A3B_Q4
        .builder()
        .system_prompt(&before_text)
        .workspace_path(args.workspace.clone())
        .thinking(false)
        .sampling(SamplingConfig::argmax())
        .max_response_tokens(args.max_tokens)
        .max_concurrent(1);
    if let Some(ref d) = args.model_dir {
        mbuilder = mbuilder.model_dir(d);
    }
    if let Some(ref p) = args.model_path {
        mbuilder = mbuilder.model_path(p);
    }
    if let Some(ref p) = args.tokenizer_path {
        mbuilder = mbuilder.tokenizer_path(p);
    }
    let engine = mbuilder.engine(&device)?;
    let conv_config = mbuilder.conversation_config();
    let formatted_prompt = mbuilder.format_system_prompt();
    println!("model loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    let tokenizer = engine.tokenizer();
    proj.tokenize_templates::<anyhow::Error, _>(|s| {
        Ok(tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("template tokenise: {e}"))?
            .get_ids()
            .to_vec())
    })?;

    // ── Raw-band geometry (user-turn K/V/Q capture across a 20% layer band).
    let core = engine.model_core_properties();
    let prov_layers = core.provenance_layer_indices.as_array();
    let n_kv_heads = core.n_kv_heads;
    let head_dim = core.head_dim;
    let n_total_layers = core.num_layers;
    let n_layers_per_band = (n_total_layers * 20 / 100).max(1);
    let band_half = n_layers_per_band / 2;
    let bands: [Vec<usize>; 3] = [
        band_layer_indices(prov_layers[1], band_half, n_total_layers, n_layers_per_band),
        band_layer_indices(prov_layers[3], band_half, n_total_layers, n_layers_per_band),
        band_layer_indices(prov_layers[5], band_half, n_total_layers, n_layers_per_band),
    ];
    let unique_layers: Vec<usize> = {
        let mut seen = std::collections::HashSet::new();
        bands
            .iter()
            .flatten()
            .copied()
            .filter(|l| seen.insert(*l))
            .collect()
    };
    let raw_header = RawFileHeader {
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        n_layers_per_band: n_layers_per_band as u32,
        band_half_width: band_half as u32,
        band_centers: [
            prov_layers[1] as u32,
            prov_layers[3] as u32,
            prov_layers[5] as u32,
        ],
        n_total_layers: n_total_layers as u32,
    };

    // ── Output files.
    let out_pf = ProvenanceFile::open(&sig_path)?;
    let out_raw = RawProvenanceFile::create(&raw_path, raw_header)?;
    let pf = engine.provenance_file();
    let chunk_size = CHUNK_SIZE;

    let mut scenarios: Vec<OutScenario> = Vec::new();
    let t_start = Instant::now();
    let n_cases = if args.limit > 0 {
        args.limit.min(prompts.cases.len())
    } else {
        prompts.cases.len()
    };
    let total = n_cases * 2;
    let mut done = 0usize;

    let cases: Vec<&PromptCase> = if args.limit > 0 {
        prompts.cases.iter().take(args.limit).collect()
    } else {
        prompts.cases.iter().collect()
    };
    for case in cases {
        for (split, prompt) in [("train", &case.train), ("holdout", &case.holdout)] {
            done += 1;
            let id = format!("{}_{}", case.tool, split);

            let mut conv = engine.new_conversation_with_projection(
                &formatted_prompt,
                proj.clone(),
                dialogue,
                group,
                conv_config.clone(),
            )?;
            let handle = conv.submit_turn(prompt)?;
            let resp = handle.wait()?;
            conv.finish_turn(handle, &resp)?;

            let Some(seal) = resp.seal.as_ref() else {
                eprintln!("  [{done}/{total}] {id} — no seal, skipped");
                drop(conv);
                continue;
            };

            // Partition the turn's sigs: prefill_in_turn = total − generated.
            let gen_tokens = tokenizer
                .encode(resp.text.as_str(), false)
                .map(|e| e.get_ids().len())
                .unwrap_or(0);
            let total_turn: usize = seal
                .new_sig_entries
                .iter()
                .map(|e| e.token_count as usize)
                .sum();
            let prefill_in_turn = total_turn.saturating_sub(gen_tokens);
            let n_prefill_blocks = prefill_in_turn.div_ceil(chunk_size);

            let mut tok = 0usize;
            let mut decode_entries: Vec<&SigEntry> = Vec::new();
            for e in &seal.new_sig_entries {
                if tok < prefill_in_turn {
                    tok += e.token_count as usize;
                } else {
                    decode_entries.push(e);
                }
            }

            // User-turn raw K/V/Q over the prefill block range (before KV freed).
            let raw_entry: Option<RawSigEntry> = capture_user_raw(
                &conv,
                &out_raw,
                &raw_header,
                &bands,
                &unique_layers,
                seal.block_from,
                n_prefill_blocks.max(1),
                prefill_in_turn,
                chunk_size,
            );

            drop(conv);

            // Decode-Q probe.
            let mut syn = Vec::new();
            let mut sem = Vec::new();
            let mut prag = Vec::new();
            for e in &decode_entries {
                let (a, b, c) = pf.read_entry(**e)?;
                syn.extend(a);
                sem.extend(b);
                prag.extend(c);
            }
            if syn.is_empty() {
                eprintln!("  [{done}/{total}] {id} — no decode sigs, skipped");
                continue;
            }
            let oe = out_pf.append(&syn, &sem, &prag)?;

            let invoked = resp.text.contains("<tool_call>") || resp.text.contains("\"name\"");
            println!(
                "  [{done}/{total}] {id:<28} {} {:>3}tok  [{}]",
                if invoked { "call" } else { "TEXT" },
                oe.token_count,
                resp.text.chars().take(44).collect::<String>(),
            );

            scenarios.push(OutScenario {
                id,
                tool: Some(case.tool.clone()),
                case_type: "positive".into(),
                split: split.into(),
                user_prompt: prompt.clone(),
                generated_text: resp.text,
                byte_offset: oe.byte_offset,
                token_count: oe.token_count,
                raw_byte_offset: raw_entry.map(|e| e.byte_offset),
                raw_token_count: raw_entry.map(|e| e.token_count),
            });
        }
    }

    // ── Manifests.
    let manifest = OutManifest {
        version: 1,
        model: "Qwen3-30B-A3B-Q4_K_M".into(),
        context: format!("full_catalog_{}_tools_allvisible", tool_sections.len()),
        provenance_layer_indices: prov_layers,
        scenarios,
    };
    std::fs::write(
        args.output.join("MANIFEST.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    let raw_manifest = RawOutManifest {
        version: 1,
        model: "Qwen3-30B-A3B-Q4_K_M".into(),
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        n_layers_per_band: n_layers_per_band as u32,
        band_half_width: band_half as u32,
        band_centers: [
            prov_layers[1] as u32,
            prov_layers[3] as u32,
            prov_layers[5] as u32,
        ],
        n_total_layers: n_total_layers as u32,
    };
    std::fs::write(
        args.output.join("RAW_MANIFEST.json"),
        serde_json::to_string_pretty(&raw_manifest)?,
    )?;

    let called = manifest
        .scenarios
        .iter()
        .filter(|s| {
            s.generated_text.contains("<tool_call>") || s.generated_text.contains("\"name\"")
        })
        .count();
    println!(
        "\ndone — {}/{} cases captured ({} emitted a tool call) in {:.1}s",
        manifest.scenarios.len(),
        total,
        called,
        t_start.elapsed().as_secs_f64(),
    );
    println!("  signatures.prov : {}", sig_path.display());
    println!("  raw_kvq.prov    : {}", raw_path.display());
    Ok(())
}

/// Extract the user-turn raw K/V/Q over its prefill block range and append it to
/// the raw store. Returns the appended entry, or `None` on any extraction issue.
#[allow(clippy::too_many_arguments)]
fn capture_user_raw(
    conv: &candle_conversation::Sequence,
    out_raw: &RawProvenanceFile,
    header: &RawFileHeader,
    bands: &[Vec<usize>; 3],
    unique_layers: &[usize],
    block_from: usize,
    n_prefill_blocks: usize,
    prefill_tokens: usize,
    chunk_size: usize,
) -> Option<RawSigEntry> {
    let range = Some((block_from, block_from + n_prefill_blocks));
    let layer_data = conv.extract_raw_kvq(unique_layers.to_vec(), range).ok()?;
    let layer_map: std::collections::HashMap<usize, &Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>> =
        layer_data.iter().map(|(li, b)| (*li, b)).collect();

    let mut bytes: Vec<u8> = Vec::new();
    for block_offset in 0..n_prefill_blocks {
        let tokens_this_block = if block_offset + 1 < n_prefill_blocks {
            chunk_size
        } else {
            let rem = prefill_tokens % chunk_size;
            if rem == 0 {
                chunk_size
            } else {
                rem
            }
        };
        let abs_block = block_from + block_offset;
        let band_layer_data: [Vec<(&[f32], &[f32])>; 3] = std::array::from_fn(|band| {
            bands[band]
                .iter()
                .map(|&layer| {
                    layer_map
                        .get(&layer)
                        .and_then(|blocks| blocks.iter().find(|(bi, ..)| *bi == abs_block))
                        .map(|(_, k, _v, q)| (k.as_slice(), q.as_slice()))
                        .unwrap_or((&[][..], &[][..]))
                })
                .collect()
        });
        bytes.extend_from_slice(&build_token_blob(
            header,
            tokens_this_block,
            &band_layer_data,
        ));
    }
    out_raw.append(&bytes).ok()
}
