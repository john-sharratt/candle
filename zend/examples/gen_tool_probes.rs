//! `gen_tool_probes` — capture decode-phase Q probes **in the real 93-tool
//! context**, so they align with the substrate's tool-section corpus.
//!
//! The existing `tool_provenance_real_data` probes were captured with a
//! *single* tool in the system prompt; the daemon seals the **full 93-tool
//! catalog**. A probe Q taken in a 1-tool context shares only the prompt
//! baseline with the 93-tool corpus, so it ranks tools at chance (see
//! `tool_select_from_substrate`). This regenerates the probes against the
//! exact production projection: same schema, same `install_tool_catalog`,
//! same `new_conversation_with_projection` sealing — so the captured decode Q
//! is conditioned on the identical sealed catalog the substrate holds.
//!
//! It reads the question prompts (user_prompt + expected tool) from an existing
//! real-data MANIFEST, re-runs each through the model in the 93-tool context,
//! and writes fresh `signatures.prov` + `MANIFEST.json` that
//! `tool_select_from_substrate` consumes unchanged.
//!
//! ```sh
//! cargo run -p zend --example gen_tool_probes --release --features cuda -- \
//!   --model-dir /path/to/Qwen3-30B-A3B-Q4_K_M \
//!   --prompts   candle-conversation/tests/tool_provenance_real_data \
//!   --output    candle-conversation/tests/tool_provenance_ctx93
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle::Device;
use candle_conversation::models::{Dialect, Model};
use candle_conversation::projection::{Builder, SystemPromptItem};
use candle_conversation::{ProvenanceFile, SamplingConfig, SigEntry};
use clap::Parser;
use serde::{Deserialize, Serialize};
use zend::tools::install_tool_catalog;

const PROJECTION_SCHEMA_TEMPLATE: &str = include_str!("../src/prompts/projection.yaml");

#[derive(Parser)]
#[command(about = "Capture decode-Q tool probes in the real 93-tool projection context")]
struct Args {
    /// Directory containing the model GGUF + tokenizer.json (else HF hub cache).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Explicit GGUF path (overrides --model-dir; pairs with --tokenizer-path).
    #[arg(long)]
    model_path: Option<PathBuf>,

    /// Explicit tokenizer.json path (pairs with --model-path).
    #[arg(long)]
    tokenizer_path: Option<PathBuf>,

    /// Directory holding the source MANIFEST.json (question prompts + expected tool).
    #[arg(
        long,
        default_value_os_t = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../candle-conversation/tests/tool_provenance_real_data"
        ))
    )]
    prompts: PathBuf,

    /// Output directory for the re-captured probes (created if absent).
    #[arg(
        long,
        default_value_os_t = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../candle-conversation/tests/tool_provenance_ctx93"
        ))
    )]
    output: PathBuf,

    /// Scratch workspace for the capture conversations (a throwaway substrate;
    /// the catalog seals here, NOT in your real daemon workspace).
    #[arg(
        long,
        default_value_os_t = std::env::temp_dir().join("zend_gen_tool_probes")
    )]
    workspace: PathBuf,

    /// Max tokens to generate per prompt (enough to emit the tool-call).
    #[arg(long, default_value = "96")]
    max_tokens: usize,

    /// Overwrite existing output without prompting.
    #[arg(long)]
    force: bool,
}

/// One source prompt (only the fields we read from the input manifest).
#[derive(Deserialize)]
struct InScenario {
    id: String,
    tool: Option<String>,
    #[serde(default)]
    case_type: String,
    #[serde(default)]
    user_prompt: String,
}

#[derive(Deserialize)]
struct InManifest {
    scenarios: Vec<InScenario>,
}

/// Output scenario — matches what `tool_select_from_substrate` reads.
#[derive(Serialize)]
struct OutScenario {
    id: String,
    tool: Option<String>,
    case_type: String,
    user_prompt: String,
    generated_text: String,
    byte_offset: u64,
    token_count: u16,
    block_count: usize,
}

#[derive(Serialize)]
struct OutManifest {
    version: u32,
    model: String,
    context: String,
    provenance_layer_indices: [usize; 6],
    scenarios: Vec<OutScenario>,
}

/// Replicates the daemon's `build_projection_builder` (session.rs).
fn build_projection_builder(workspace: &std::path::Path) -> anyhow::Result<Builder> {
    let name = workspace
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("this project");
    let dialect = Dialect::chat_ml();
    Builder::from_yaml_with_vars_and_dialect(
        PROJECTION_SCHEMA_TEMPLATE,
        &[("workspace", name)],
        Some(&dialect),
    )
    .map_err(|e| anyhow::anyhow!("projection.yaml parse: {e}"))
}

/// Replicates the daemon's `pre_collection_prelude`: the dialogue layer's
/// section content up to (not including) the first collection — the text the
/// engine ChatML-wraps as the system prompt; everything from the `tools`
/// collection on is expanded inside `new_conversation_with_projection`.
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
            SystemPromptItem::Collection(_) => break,
        }
    }
    out
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // ── Load source prompts.
    let in_path = args.prompts.join("MANIFEST.json");
    let in_manifest: InManifest = serde_json::from_str(
        &std::fs::read_to_string(&in_path)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", in_path.display()))?,
    )?;
    println!(
        "loaded {} prompts from {}",
        in_manifest.scenarios.len(),
        in_path.display()
    );

    // ── Guard + prepare output.
    std::fs::create_dir_all(&args.output)?;
    let sig_path = args.output.join("signatures.prov");
    // Parallel fixture dir for the user-prompt PREFILL-Q probe (the clean
    // intent query, same phase as the tool-definition corpus).
    let prefill_dir = {
        let mut d = args.output.clone();
        let name = d
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        d.set_file_name(format!("{name}_prefill"));
        d
    };
    std::fs::create_dir_all(&prefill_dir)?;
    let prefill_sig_path = prefill_dir.join("signatures.prov");
    for p in [&sig_path, &prefill_sig_path] {
        if p.exists() {
            if !args.force {
                anyhow::bail!("{} exists — pass --force to overwrite", p.display());
            }
            std::fs::remove_file(p)?;
        }
    }
    std::fs::create_dir_all(&args.workspace)?;

    // ── Build the production projection: same schema + catalog the daemon seals.
    let mut proj = build_projection_builder(&args.workspace)?;
    let dialogue = proj
        .id_for_layer("dialogue")
        .ok_or_else(|| anyhow::anyhow!("schema missing 'dialogue' layer"))?;
    let group = proj
        .id_for_group("primary_conversation")
        .ok_or_else(|| anyhow::anyhow!("schema missing 'primary_conversation' group"))?;
    let tool_sections = install_tool_catalog(&mut proj, dialogue)?;
    println!(
        "installed {} tools into the projection",
        tool_sections.len()
    );
    let before_text = pre_collection_prelude(&proj);

    // ── Load the model (the only GPU step).
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
    if let Some(ref dir) = args.model_dir {
        mbuilder = mbuilder.model_dir(dir);
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

    // Tokenise the schema's template items against the engine tokenizer.
    let tokenizer = engine.tokenizer();
    proj.tokenize_templates::<anyhow::Error, _>(|s| {
        Ok(tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("template tokenise: {e}"))?
            .get_ids()
            .to_vec())
    })?;

    let prov_layers = engine
        .model_core_properties()
        .provenance_layer_indices
        .as_array();

    // ── Capture loop: one fresh projection conversation per prompt, so each
    // decode Q is conditioned on the full sealed catalog with no prior history.
    let out_pf = ProvenanceFile::open(&sig_path)?;
    let out_prefill_pf = ProvenanceFile::open(&prefill_sig_path)?;
    let pf = engine.provenance_file();
    let mut out_scenarios: Vec<OutScenario> = Vec::new();
    let mut out_prefill_scenarios: Vec<OutScenario> = Vec::new();
    let t_start = Instant::now();

    for (i, s) in in_manifest.scenarios.iter().enumerate() {
        if s.user_prompt.is_empty() {
            continue;
        }
        let mut conv = engine.new_conversation_with_projection(
            &formatted_prompt,
            proj.clone(),
            dialogue,
            group,
            conv_config.clone(),
        )?;
        let handle = conv.submit_turn(&s.user_prompt)?;
        let resp = handle.wait()?;
        conv.finish_turn(handle, &resp)?;

        let Some(seal) = resp.seal.as_ref() else {
            eprintln!("  [{}] {} — no seal, skipped", i + 1, s.id);
            continue;
        };

        // The pre-sealed catalog is cached, so `resp.stats.prefill_token_count`
        // counts the full context (thousands) while `new_sig_entries` holds only
        // this turn's new tokens (user prefill + decode). Partition by the turn's
        // OWN prefill = total turn tokens − generated tokens, so the decode tail
        // (the model emitting the tool-call) is the probe.
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
        let mut tokens_so_far = 0usize;
        let mut prefill_entries: Vec<&SigEntry> = Vec::new();
        let mut decode_entries: Vec<&SigEntry> = Vec::new();
        for entry in &seal.new_sig_entries {
            if tokens_so_far < prefill_in_turn {
                prefill_entries.push(entry);
                tokens_so_far += entry.token_count as usize;
            } else {
                decode_entries.push(entry);
            }
        }

        // Prefill-Q probe (the user-prompt query) → parallel fixture dir.
        let mut psyn = Vec::new();
        let mut psem = Vec::new();
        let mut pprag = Vec::new();
        for entry in &prefill_entries {
            let (a, b, c) = pf.read_entry(**entry)?;
            psyn.extend(a);
            psem.extend(b);
            pprag.extend(c);
        }
        if !psyn.is_empty() {
            let pe = out_prefill_pf.append(&psyn, &psem, &pprag)?;
            out_prefill_scenarios.push(OutScenario {
                id: s.id.clone(),
                tool: s.tool.clone(),
                case_type: s.case_type.clone(),
                user_prompt: s.user_prompt.clone(),
                generated_text: resp.text.clone(),
                byte_offset: pe.byte_offset,
                token_count: pe.token_count,
                block_count: seal.block_count,
            });
        }
        if i < 3 {
            eprintln!(
                "    [diag {}] entries={} turn_tokens={total_turn} gen={gen_tokens} \
                 prefill_in_turn={prefill_in_turn} decode_entries={} stats.prefill={}",
                s.id,
                seal.new_sig_entries.len(),
                decode_entries.len(),
                resp.stats.prefill_token_count,
            );
        }

        let mut syn = Vec::new();
        let mut sem = Vec::new();
        let mut prag = Vec::new();
        for entry in &decode_entries {
            let (s_, se_, p_) = pf.read_entry(**entry)?;
            syn.extend(s_);
            sem.extend(se_);
            prag.extend(p_);
        }
        if syn.is_empty() {
            eprintln!("  [{}] {} — no decode sigs, skipped", i + 1, s.id);
            continue;
        }

        let out_entry = out_pf.append(&syn, &sem, &prag)?;
        let preview: String = resp.text.chars().take(48).collect();
        println!(
            "  [{:>3}/{}] {:<22} {:>3} tok  [{}]",
            i + 1,
            in_manifest.scenarios.len(),
            s.id,
            out_entry.token_count,
            preview,
        );

        out_scenarios.push(OutScenario {
            id: s.id.clone(),
            tool: s.tool.clone(),
            case_type: s.case_type.clone(),
            user_prompt: s.user_prompt.clone(),
            generated_text: resp.text,
            byte_offset: out_entry.byte_offset,
            token_count: out_entry.token_count,
            block_count: seal.block_count,
        });
    }

    // ── Write the output manifest.
    let manifest = OutManifest {
        version: 1,
        model: "Qwen3-30B-A3B-Q4_K_M".to_string(),
        context: format!("full_catalog_{}_tools", tool_sections.len()),
        provenance_layer_indices: prov_layers,
        scenarios: out_scenarios,
    };
    let manifest_path = args.output.join("MANIFEST.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

    let prefill_manifest = OutManifest {
        version: 1,
        model: "Qwen3-30B-A3B-Q4_K_M".to_string(),
        context: format!("full_catalog_{}_tools_prefillQ", tool_sections.len()),
        provenance_layer_indices: prov_layers,
        scenarios: out_prefill_scenarios,
    };
    let prefill_manifest_path = prefill_dir.join("MANIFEST.json");
    std::fs::write(
        &prefill_manifest_path,
        serde_json::to_string_pretty(&prefill_manifest)?,
    )?;
    println!(
        "  prefill-Q probes: {} → {}",
        prefill_manifest.scenarios.len(),
        prefill_dir.display()
    );

    let elapsed = t_start.elapsed().as_secs_f64();
    println!(
        "\ndone — {} probes captured in the {}-tool context in {:.1}s",
        manifest.scenarios.len(),
        tool_sections.len(),
        elapsed,
    );
    println!("  signatures.prov : {}", sig_path.display());
    println!("  MANIFEST.json   : {}", manifest_path.display());
    println!(
        "\nnext: cargo run -p zend --example tool_select_from_substrate --release -- \\\n  <daemon-workspace> {}",
        args.output.display()
    );
    Ok(())
}
