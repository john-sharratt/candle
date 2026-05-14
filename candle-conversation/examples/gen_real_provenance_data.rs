//! Generate real tool-provenance KV/Q fixtures using Qwen3-30B-A3B.
//!
//! Runs all 128 tool-scenario prompts through the model in parallel batches,
//! capturing real Q sign-bit signatures from the KV-cache seal at each turn.
//!
//! # How parallelism works
//!
//! `submit_turn` is non-blocking: it queues a `SubmitTurn` message on the
//! scheduler channel and returns a `TurnHandle` immediately.  Within one
//! batch we submit all turns before waiting on any, so the scheduler sees
//! `batch_size` pending sequences and batches their prefill + decode steps
//! together — this is the same mechanism that drives the 64-session
//! benchmark throughput.  We then drain handles in submission order; by
//! the time we call `handle.wait()` on later entries the scheduler has
//! typically already finished them.
//!
//! # Run
//!
//! ```sh
//! # Generate prompts manifest first (fast, no GPU):
//! cargo run -p candle-conversation --example gen_tool_provenance_data
//!
//! # Generate real signatures:
//! cargo run -p candle-conversation --example gen_real_provenance_data \
//!   --release --features "cuda,hub" -- \
//!   --model-dir /path/to/Qwen3-30B-A3B-Q4_K_M
//! ```
//!
//! Output: `tests/tool_provenance_real_data/{signatures.prov,MANIFEST.json}`

use std::path::PathBuf;
use std::time::Instant;

use candle::Device;
use candle_conversation::{
    models::{Dialect, Model},
    ProvenanceFile, SamplingConfig, SigEntry, TurnHandle,
};
use candle_conversation::Sequence;
use clap::Parser;
use serde::{Deserialize, Serialize};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Generate real tool-provenance fixtures via batched Qwen3-30B-A3B inference")]
struct Args {
    /// Directory containing the model GGUF and tokenizer.json.
    /// If omitted, the model is fetched from the HuggingFace hub cache.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Output directory for real signatures (created if absent).
    #[arg(
        long,
        default_value_os_t = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/tool_provenance_real_data"
        ))
    )]
    output: PathBuf,

    /// Directory containing synthetic MANIFEST.json (prompt source).
    #[arg(
        long,
        default_value_os_t = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/tool_provenance_data"
        ))
    )]
    prompts_dir: PathBuf,

    /// Number of scenarios to run concurrently inside the scheduler.
    /// Higher = more GPU utilisation; lower = less KV pressure.
    #[arg(long, default_value = "8")]
    batch_size: usize,

    /// Max tokens to generate per scenario.
    #[arg(long, default_value = "128")]
    max_tokens: usize,

    /// Overwrite existing output without prompting.
    #[arg(long)]
    force: bool,
}

// ── Manifest types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CaseType {
    Positive,
    Boundary,
    Negative,
    NoTool,
}

impl std::fmt::Display for CaseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaseType::Positive => write!(f, "pos"),
            CaseType::Boundary => write!(f, "bnd"),
            CaseType::Negative => write!(f, "neg"),
            CaseType::NoTool => write!(f, "no_tool"),
        }
    }
}

/// Source scenario from synthetic MANIFEST.json.
#[derive(Debug, Deserialize)]
struct InScenario {
    id: String,
    tool: Option<String>,
    case_type: CaseType,
    system_prompt: String,
    user_prompt: String,
}

#[derive(Debug, Deserialize)]
struct InManifest {
    scenarios: Vec<InScenario>,
}

/// Output scenario written to real MANIFEST.json.
#[derive(Debug, Serialize)]
struct OutScenario {
    id: String,
    tool: Option<String>,
    case_type: CaseType,
    user_prompt: String,
    generated_text: String,
    turn_id: u64,
    byte_offset: u64,
    token_count: u16,
    block_count: usize,
}

#[derive(Debug, Serialize)]
struct OutManifest {
    version: u32,
    model: String,
    provenance_layer_indices: [usize; 3],
    scenarios: Vec<OutScenario>,
}

// ── In-flight batch slot ──────────────────────────────────────────────────────

/// One active slot: holds the Sequence alive while the scheduler works.
struct Slot {
    global_idx: usize,
    id: String,
    tool: Option<String>,
    case_type: CaseType,
    user_prompt: String,
    /// Kept alive until `finish_turn` — dropping it frees the KV cache slot.
    conv: Sequence,
    handle: TurnHandle,
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Load prompts from synthetic manifest.
    let prompts_manifest = args.prompts_dir.join("MANIFEST.json");
    let text = std::fs::read_to_string(&prompts_manifest).map_err(|e| {
        anyhow::anyhow!(
            "failed to read {}: {} — run gen_tool_provenance_data first",
            prompts_manifest.display(),
            e
        )
    })?;
    let in_manifest: InManifest = serde_json::from_str(&text)?;
    let total = in_manifest.scenarios.len();
    println!("Loaded {total} scenarios from {}", prompts_manifest.display());

    // Guard output directory.
    let sig_path = args.output.join("signatures.prov");
    if sig_path.exists() {
        if !args.force {
            eprintln!(
                "Output already exists at {}.  Use --force to overwrite.",
                sig_path.display()
            );
            std::process::exit(1);
        }
        // Remove the stale file so ProvenanceFile::open starts at offset 0,
        // not at the end of the old file.
        std::fs::remove_file(&sig_path)?;
    }
    std::fs::create_dir_all(&args.output)?;

    // ── Load model ────────────────────────────────────────────────────────────

    println!("Loading model from {}...", args.model_dir.as_deref().map(|p| p.display().to_string()).unwrap_or_else(|| "HuggingFace hub".into()));
    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);
    let t_load = Instant::now();

    let mut builder = Model::Qwen3_30B_A3B_Q4
        .builder()
        .thinking(false)
        .sampling(SamplingConfig::argmax())
        .max_response_tokens(args.max_tokens)
        .max_concurrent(args.batch_size);

    if let Some(ref dir) = args.model_dir {
        builder = builder.model_dir(dir);
    }

    let engine = builder.engine(&device)?;
    let conv_config = builder.conversation_config();
    let dialect: Dialect = builder.spec().dialect.clone();

    println!("Model loaded in {:.1}s", t_load.elapsed().as_secs_f64());
    let prov_layers = engine.provenance_layer_indices();
    println!("Provenance layer indices: {prov_layers:?}");
    println!("Batch size: {}", args.batch_size);

    // ── Output ProvenanceFile (persistent) ───────────────────────────────────

    let out_pf = ProvenanceFile::open(&sig_path)?;
    let mut out_scenarios: Vec<OutScenario> = Vec::with_capacity(total);

    // ── Batched inference loop ────────────────────────────────────────────────
    //
    // Phase A — submit: create batch_size sequences, call submit_turn on each
    //           (non-blocking).  All SubmitTurn messages queue in the scheduler
    //           channel; the scheduler batches their prefills and decodes.
    //
    // Phase B — drain: wait for each handle in submission order, finish_turn,
    //           read real Q signatures, write to output file.  By the time we
    //           reach the last handle in the batch, the scheduler has typically
    //           finished it already.

    let t_start = Instant::now();
    let mut completed = 0usize;

    for (batch_start, chunk) in in_manifest.scenarios.chunks(args.batch_size).enumerate() {
        let batch_idx = batch_start * args.batch_size;
        println!(
            "\n── Batch {}/{} (scenarios {}-{}) ──",
            batch_start + 1,
            (total + args.batch_size - 1) / args.batch_size,
            batch_idx + 1,
            (batch_idx + chunk.len()).min(total)
        );

        // ── Phase A: submit all turns in this batch (non-blocking) ───────────

        let t_submit = Instant::now();
        let mut slots: Vec<Slot> = Vec::with_capacity(chunk.len());

        for (i, s) in chunk.iter().enumerate() {
            let global_idx = batch_idx + i;
            let formatted_sp = dialect.format_system_prompt(&s.system_prompt);
            let mut conv = engine.new_conversation(&formatted_sp, conv_config.clone())?;
            // submit_turn queues the request and returns immediately.
            let handle = conv.submit_turn(&s.user_prompt)?;
            slots.push(Slot {
                global_idx,
                id: s.id.clone(),
                tool: s.tool.clone(),
                case_type: s.case_type,
                user_prompt: s.user_prompt.clone(),
                conv,
                handle,
            });
        }
        println!(
            "  Submitted {} turns in {:.1}ms — scheduler is running them in parallel",
            slots.len(),
            t_submit.elapsed().as_secs_f64() * 1000.0
        );

        // ── Phase B: drain handles in submission order ────────────────────────

        let pf = engine.provenance_file();

        for slot in slots {
            let Slot { global_idx, id, tool, case_type, user_prompt, mut conv, handle } = slot;

            let resp = handle.wait()?;
            conv.finish_turn(handle, &resp)?;
            drop(conv); // release KV slot for the next batch

            let seal = match &resp.seal {
                Some(s) => s,
                None => {
                    eprintln!("  [{global_idx}] WARNING: no seal for {id} — skipping");
                    continue;
                }
            };

            // Read real Q sign-bits from engine's internal ProvenanceFile.
            let mut all_syn = Vec::new();
            let mut all_sem = Vec::new();
            let mut all_prag = Vec::new();
            for entry in &seal.new_sig_entries {
                let (syn, sem, prag) = pf.read_entry(*entry)?;
                all_syn.extend(syn);
                all_sem.extend(sem);
                all_prag.extend(prag);
            }

            if all_syn.is_empty() {
                eprintln!("  [{global_idx}] WARNING: no signatures for {id} — skipping");
                continue;
            }

            let out_entry: SigEntry = out_pf.append(&all_syn, &all_sem, &all_prag)?;
            let preview: String = resp.text.chars().take(55).collect();
            println!(
                "  [{:>3}] {} ({}) — {} tok, {} blk  [{}]",
                global_idx + 1,
                id,
                case_type,
                out_entry.token_count,
                seal.block_count,
                preview,
            );

            out_scenarios.push(OutScenario {
                id,
                tool,
                case_type,
                user_prompt,
                generated_text: resp.text,
                turn_id: global_idx as u64,
                byte_offset: out_entry.byte_offset,
                token_count: out_entry.token_count,
                block_count: seal.block_count,
            });
            completed += 1;
        }
    }

    // ── Write manifest ────────────────────────────────────────────────────────

    let out_manifest = OutManifest {
        version: 1,
        model: "Qwen3-30B-A3B-Q4_K_M".to_string(),
        provenance_layer_indices: prov_layers,
        scenarios: out_scenarios,
    };
    let manifest_out = args.output.join("MANIFEST.json");
    std::fs::write(&manifest_out, serde_json::to_string_pretty(&out_manifest)?)?;

    let elapsed = t_start.elapsed().as_secs_f64();
    println!(
        "\nDone — {completed}/{total} scenarios in {elapsed:.1}s ({:.1} s/scenario avg)",
        elapsed / completed.max(1) as f64,
    );
    println!("  signatures.prov : {} bytes", std::fs::metadata(&sig_path)?.len());
    println!("  MANIFEST.json   : {} bytes", std::fs::metadata(&manifest_out)?.len());

    Ok(())
}
