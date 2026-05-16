//! Generate real layer-provenance KV/Q fixtures using Qwen3-30B-A3B.
//!
//! Reads prompts from `tests/<type>_provenance_data/MANIFEST.json` (written by
//! `gen_layer_provenance_data`), runs batched inference, and writes real
//! sign-bit signatures + raw K/Q vectors to `tests/<type>_provenance_real_data/`.
//!
//! # Run
//!
//! ```sh
//! # Generate synthetic prompts first (fast, no GPU):
//! cargo run -p candle-conversation --example gen_layer_provenance_data -- --all --force
//!
//! # Generate real signatures for all content types:
//! cargo run -p candle-conversation --example gen_real_layer_provenance_data \
//!   --release --features "cuda,hub" -- \
//!   --all --force
//!
//! # Or one type at a time:
//! cargo run -p candle-conversation --example gen_real_layer_provenance_data \
//!   --release --features "cuda,hub" -- \
//!   --content-type bug-analysis --force
//! ```

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle::Device;
use candle_conversation::{
    models::{Dialect, Model},
    provenance::{
        band_layer_indices, build_token_blob, RawFileHeader, RawProvenanceFile, RawSigEntry,
    },
    ConversationEngine, ProvenanceFile, SamplingConfig, SequenceConfig, SigEntry, TurnHandle,
};
use candle_conversation::Sequence;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Generate real layer-provenance KV/Q fixtures via Qwen3-30B-A3B inference")]
struct Args {
    /// Content type to generate (mutually exclusive with --all).
    #[arg(long, value_enum)]
    content_type: Option<ContentType>,

    /// Generate all content types in sequence.
    #[arg(long, conflicts_with = "content_type")]
    all: bool,

    /// If omitted, the model is fetched from the HuggingFace hub cache.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Override output directory (only valid with a single --content-type).
    #[arg(long)]
    output: Option<PathBuf>,

    /// Number of scenarios to run concurrently inside the scheduler.
    #[arg(long, default_value = "8")]
    batch_size: usize,

    /// Max tokens to generate per scenario.
    #[arg(long, default_value = "128")]
    max_tokens: usize,

    /// Overwrite existing output without prompting.
    #[arg(long)]
    force: bool,

    /// Skip generating raw KVQ data (only sign-bit signatures).
    #[arg(long)]
    skip_raw: bool,
}

// ── Content type ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ContentType {
    CodeReading,
    StaticAnalysis,
    DependencyAnalysis,
    ArchitecturalAnalysis,
    CriticalAnalysis,
    BugAnalysis,
    DailyHistory,
    DreamLog,
}

const ALL_TYPES: &[ContentType] = &[
    ContentType::CodeReading,
    ContentType::StaticAnalysis,
    ContentType::DependencyAnalysis,
    ContentType::ArchitecturalAnalysis,
    ContentType::CriticalAnalysis,
    ContentType::BugAnalysis,
    ContentType::DailyHistory,
    ContentType::DreamLog,
];

impl ContentType {
    fn dir_name(self) -> &'static str {
        match self {
            Self::CodeReading => "code_reading",
            Self::StaticAnalysis => "static_analysis",
            Self::DependencyAnalysis => "dependency_analysis",
            Self::ArchitecturalAnalysis => "architectural_analysis",
            Self::CriticalAnalysis => "critical_analysis",
            Self::BugAnalysis => "bug_analysis",
            Self::DailyHistory => "daily_history",
            Self::DreamLog => "dream_log",
        }
    }
}

// ── Manifest types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CaseType {
    Positive,
    Boundary,
    Negative,
    NoMatch,
}

impl std::fmt::Display for CaseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaseType::Positive => write!(f, "pos"),
            CaseType::Boundary => write!(f, "bnd"),
            CaseType::Negative => write!(f, "neg"),
            CaseType::NoMatch => write!(f, "no_match"),
        }
    }
}

/// Source scenario from synthetic MANIFEST.json.
#[derive(Debug, Deserialize)]
struct InScenario {
    id: String,
    item: Option<String>,
    case_type: CaseType,
    system_prompt: String,
    user_prompt: String,
}

#[derive(Debug, Deserialize)]
struct InManifest {
    scenarios: Vec<InScenario>,
}

/// Output scenario written to real MANIFEST.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OutScenario {
    id: String,
    item: Option<String>,
    case_type: CaseType,
    user_prompt: String,
    generated_text: String,
    turn_id: u64,
    byte_offset: u64,
    token_count: u16,
    block_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct OutManifest {
    version: u32,
    content_type: String,
    model: String,
    provenance_layer_indices: [usize; 6],
    scenarios: Vec<OutScenario>,
}

/// Output scenario written to RAW_MANIFEST.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawOutScenario {
    id: String,
    item: Option<String>,
    case_type: CaseType,
    raw_byte_offset: u64,
    raw_token_count: u16,
}

#[derive(Debug, Serialize, Deserialize)]
struct RawOutManifest {
    version: u32,
    content_type: String,
    model: String,
    n_kv_heads: u32,
    head_dim: u32,
    n_layers_per_band: u32,
    band_half_width: u32,
    band_centers: [u32; 3],
    n_total_layers: u32,
    scenarios: Vec<RawOutScenario>,
}

// ── In-flight batch slot ──────────────────────────────────────────────────────

struct Slot {
    global_idx: usize,
    id: String,
    item: Option<String>,
    case_type: CaseType,
    user_prompt: String,
    conv: Sequence,
    handle: TurnHandle,
}

// ── Band computation ──────────────────────────────────────────────────────────

fn compute_bands(
    prov_layers: [usize; 6],
    n_total_layers: usize,
    n_layers_per_band: usize,
) -> [[Vec<usize>; 3]; 1] {
    let half = n_layers_per_band / 2;
    [[
        band_layer_indices(prov_layers[1], half, n_total_layers, n_layers_per_band),
        band_layer_indices(prov_layers[3], half, n_total_layers, n_layers_per_band),
        band_layer_indices(prov_layers[5], half, n_total_layers, n_layers_per_band),
    ]]
}

// ── Per-content-type generation ───────────────────────────────────────────────

struct ModelCtx<'a> {
    engine: &'a ConversationEngine,
    conv_config: SequenceConfig,
    dialect: Dialect,
    prov_layers: [usize; 6],
    n_kv_heads: usize,
    head_dim: usize,
    n_total_layers: usize,
    n_layers_per_band: usize,
    band_half_width: usize,
    bands: [Vec<usize>; 3],
    unique_layers: Vec<usize>,
    raw_header: RawFileHeader,
    batch_size: usize,
    max_tokens: usize,
    skip_raw: bool,
    /// Cross-type summary log written to tests/gen_progress.log
    summary_log: Arc<Mutex<std::fs::File>>,
}

// ── Write manifests (atomic, incremental) ─────────────────────────────────────

fn write_manifests(
    out_dir: &std::path::Path,
    type_name: &str,
    ctx: &ModelCtx<'_>,
    out_scenarios: &[OutScenario],
    raw_scenarios: &[RawOutScenario],
) -> anyhow::Result<()> {
    let out_manifest = OutManifest {
        version: 1,
        content_type: type_name.to_string(),
        model: "Qwen3-30B-A3B-Q4_K_M".to_string(),
        provenance_layer_indices: ctx.prov_layers,
        scenarios: out_scenarios.to_vec(),
    };
    let manifest_path = out_dir.join("MANIFEST.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&out_manifest)?)?;

    if !ctx.skip_raw && !raw_scenarios.is_empty() {
        let raw_manifest = RawOutManifest {
            version: 1,
            content_type: type_name.to_string(),
            model: "Qwen3-30B-A3B-Q4_K_M".to_string(),
            n_kv_heads: ctx.n_kv_heads as u32,
            head_dim: ctx.head_dim as u32,
            n_layers_per_band: ctx.n_layers_per_band as u32,
            band_half_width: ctx.band_half_width as u32,
            band_centers: [
                ctx.prov_layers[1] as u32,
                ctx.prov_layers[3] as u32,
                ctx.prov_layers[5] as u32,
            ],
            n_total_layers: ctx.n_total_layers as u32,
            scenarios: raw_scenarios.to_vec(),
        };
        let raw_manifest_path = out_dir.join("RAW_MANIFEST.json");
        std::fs::write(&raw_manifest_path, serde_json::to_string_pretty(&raw_manifest)?)?;
    }

    Ok(())
}

fn generate_one(
    ct: ContentType,
    output_override: Option<&PathBuf>,
    force: bool,
    ctx: &ModelCtx<'_>,
) -> anyhow::Result<()> {
    let type_name = ct.dir_name();
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let prompts_dir = crate_root.join("tests").join(format!("{}_provenance_data", type_name));
    let default_out = crate_root.join("tests").join(format!("{}_provenance_real_data", type_name));
    let out_dir = output_override.unwrap_or(&default_out);

    // Load synthetic prompts manifest.
    let prompts_path = prompts_dir.join("MANIFEST.json");
    let text = std::fs::read_to_string(&prompts_path).map_err(|e| {
        anyhow::anyhow!(
            "failed to read {}: {} — run gen_layer_provenance_data --content-type {} first",
            prompts_path.display(), e, type_name
        )
    })?;
    let in_manifest: InManifest = serde_json::from_str(&text)?;
    let total = in_manifest.scenarios.len();
    let msg = format!("\n══ {type_name}: {total} scenarios ══");
    println!("{msg}");
    {
        let mut log = ctx.summary_log.lock().unwrap();
        writeln!(log, "{msg}")?;
    }

    // Guard / resume detection.
    let sig_path = out_dir.join("signatures.prov");
    let raw_path = out_dir.join("raw_kvq.prov");
    let manifest_path = out_dir.join("MANIFEST.json");
    let raw_manifest_path = out_dir.join("RAW_MANIFEST.json");

    let (mut out_scenarios, mut raw_scenarios, resuming) = if force {
        // Delete all existing output and start fresh.
        for path in [&sig_path, &raw_path, &manifest_path, &raw_manifest_path] {
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }
        (Vec::<OutScenario>::new(), Vec::<RawOutScenario>::new(), false)
    } else if manifest_path.exists() {
        // Resume: deserialize existing manifests.
        let existing_text = std::fs::read_to_string(&manifest_path)?;
        let existing: OutManifest = serde_json::from_str(&existing_text)?;
        let existing_scenarios = existing.scenarios;
        let existing_raw = if raw_manifest_path.exists() {
            let raw_text = std::fs::read_to_string(&raw_manifest_path)?;
            let raw: RawOutManifest = serde_json::from_str(&raw_text)?;
            raw.scenarios
        } else {
            Vec::new()
        };
        let msg = format!("  Resuming — {}/{} already done", existing_scenarios.len(), total);
        println!("{msg}");
        {
            let mut log = ctx.summary_log.lock().unwrap();
            writeln!(log, "{msg}")?;
        }
        (existing_scenarios, existing_raw, true)
    } else {
        // Orphaned .prov files (crash before manifest was written) — clean up.
        for path in [&sig_path, &raw_path] {
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }
        (Vec::<OutScenario>::new(), Vec::<RawOutScenario>::new(), false)
    };

    std::fs::create_dir_all(out_dir)?;

    // Open .prov files (append-safe for signatures; create or open for raw).
    let out_pf = ProvenanceFile::open(&sig_path)?;
    let out_raw_pf = if ctx.skip_raw {
        None
    } else if resuming && raw_path.exists() {
        Some(RawProvenanceFile::open(&raw_path)?)
    } else {
        Some(RawProvenanceFile::create(&raw_path, ctx.raw_header)?)
    };

    // Build pending list — skip already-completed scenario IDs.
    let done_ids: std::collections::HashSet<String> =
        out_scenarios.iter().map(|s| s.id.clone()).collect();

    let pending: Vec<(usize, &InScenario)> = in_manifest.scenarios.iter()
        .enumerate()
        .filter(|(_, s)| !done_ids.contains(&s.id))
        .collect();

    if pending.is_empty() {
        println!("  Already complete — nothing to do.");
        return Ok(());
    }

    let t_start = Instant::now();
    let n_batches = (pending.len() + ctx.batch_size - 1) / ctx.batch_size;

    for (batch_num, chunk) in pending.chunks(ctx.batch_size).enumerate() {
        let first_global = chunk.first().map(|(i, _)| i + 1).unwrap_or(1);
        let last_global = chunk.last().map(|(i, _)| i + 1).unwrap_or(1);
        let batch_msg = format!(
            "  Batch {}/{} (scenarios {}-{})",
            batch_num + 1, n_batches, first_global, last_global,
        );
        println!("{batch_msg}");
        {
            let mut log = ctx.summary_log.lock().unwrap();
            writeln!(log, "{batch_msg}")?;
        }

        // Phase A: submit all turns (non-blocking).
        let mut slots: Vec<Slot> = Vec::with_capacity(chunk.len());
        for &(global_idx, s) in chunk.iter() {
            let formatted_sp = ctx.dialect.format_system_prompt(&s.system_prompt);
            let mut conv = ctx.engine.new_conversation(&formatted_sp, ctx.conv_config.clone())?;
            let handle = conv.submit_turn(&s.user_prompt)?;
            slots.push(Slot {
                global_idx,
                id: s.id.clone(),
                item: s.item.clone(),
                case_type: s.case_type,
                user_prompt: s.user_prompt.clone(),
                conv,
                handle,
            });
        }

        // Phase B: drain handles in submission order.
        let pf = ctx.engine.provenance_file();

        for slot in slots {
            let Slot { global_idx, id, item, case_type, user_prompt, mut conv, handle } = slot;

            let resp = handle.wait()?;
            conv.finish_turn(handle, &resp)?;

            // Extract raw KVQ before KV cache slot is freed.
            let raw_entry: Option<RawSigEntry> = if let Some(ref raw_pf) = out_raw_pf {
                let seal = resp.seal.as_ref();
                if let Some(seal) = seal {
                    let block_range = Some((seal.block_from, seal.block_to));
                    match conv.extract_raw_kvq(ctx.unique_layers.clone(), block_range) {
                        Ok(layer_data) => {
                            let layer_map: std::collections::HashMap<usize, &Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>> =
                                layer_data.iter().map(|(li, blocks)| (*li, blocks)).collect();

                            let n_blocks = seal.block_to.saturating_sub(seal.block_from);
                            let total_tokens = seal.turn_token_count;
                            let chunk_size = candle_nn::CHUNK_SIZE;
                            let mut all_token_bytes: Vec<u8> = Vec::new();

                            for block_offset in 0..n_blocks {
                                let tokens_this_block = if block_offset + 1 < n_blocks {
                                    chunk_size
                                } else {
                                    let rem = total_tokens % chunk_size;
                                    if rem == 0 { chunk_size } else { rem }
                                };

                                let band_layer_data: [Vec<(&[f32], &[f32])>; 3] = std::array::from_fn(|band| {
                                    ctx.bands[band]
                                        .iter()
                                        .map(|&layer_idx| {
                                            if let Some(blocks) = layer_map.get(&layer_idx) {
                                                let abs_block = seal.block_from + block_offset;
                                                if let Some((_, k, _v, q)) = blocks.iter().find(|(bi, ..)| *bi == abs_block) {
                                                    (k.as_slice(), q.as_slice())
                                                } else {
                                                    (&[][..], &[][..])
                                                }
                                            } else {
                                                (&[][..], &[][..])
                                            }
                                        })
                                        .collect()
                                });

                                let blob = build_token_blob(&ctx.raw_header, tokens_this_block, &band_layer_data);
                                all_token_bytes.extend_from_slice(&blob);
                            }

                            match raw_pf.append(&all_token_bytes) {
                                Ok(e) => Some(e),
                                Err(err) => {
                                    eprintln!("  [{global_idx}] WARNING: raw append failed: {err}");
                                    None
                                }
                            }
                        }
                        Err(err) => {
                            eprintln!("  [{global_idx}] WARNING: extract_raw_kvq failed: {err}");
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            drop(conv);

            let seal = match &resp.seal {
                Some(s) => s,
                None => {
                    eprintln!("  [{global_idx}] WARNING: no seal for {id} — skipping");
                    continue;
                }
            };

            // Read real Q sign-bits from engine's provenance file.
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
            let scenario_msg = format!(
                "    [{:>3}] {} ({}) — {} tok, {} blk  [{}]",
                global_idx + 1, id, case_type,
                out_entry.token_count, seal.block_count, preview,
            );
            println!("{scenario_msg}");
            {
                let mut log = ctx.summary_log.lock().unwrap();
                writeln!(log, "{scenario_msg}")?;
            }

            if let Some(re) = raw_entry {
                raw_scenarios.push(RawOutScenario {
                    id: id.clone(),
                    item: item.clone(),
                    case_type,
                    raw_byte_offset: re.byte_offset,
                    raw_token_count: re.token_count,
                });
            }

            out_scenarios.push(OutScenario {
                id,
                item,
                case_type,
                user_prompt,
                generated_text: resp.text,
                turn_id: global_idx as u64,
                byte_offset: out_entry.byte_offset,
                token_count: out_entry.token_count,
                block_count: seal.block_count,
            });

            // Flush manifest after every scenario so a kill loses at most this one.
            write_manifests(out_dir, type_name, ctx, &out_scenarios, &raw_scenarios)?;
        }
    }

    let completed = out_scenarios.len();
    let elapsed = t_start.elapsed().as_secs_f64();

    let done_msg = format!(
        "  Done — {completed}/{total} in {elapsed:.1}s ({:.1} s/scenario avg)",
        elapsed / completed.max(1) as f64
    );
    let sig_size = std::fs::metadata(&sig_path)?.len();
    let manifest_size = std::fs::metadata(&out_dir.join("MANIFEST.json"))?.len();

    println!("{done_msg}");
    println!("  signatures.prov  : {sig_size} bytes");
    println!("  MANIFEST.json    : {manifest_size} bytes");

    {
        let mut log = ctx.summary_log.lock().unwrap();
        writeln!(log, "{done_msg}")?;
        writeln!(log, "  signatures.prov  : {sig_size} bytes")?;
        writeln!(log, "  MANIFEST.json    : {manifest_size} bytes")?;
    }

    if out_raw_pf.is_some() && !raw_scenarios.is_empty() {
        let raw_size = std::fs::metadata(&raw_path)?.len();
        println!("  raw_kvq.prov     : {raw_size} bytes");
    }

    Ok(())
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if !args.all && args.content_type.is_none() {
        eprintln!("Either --content-type <type> or --all is required.");
        std::process::exit(1);
    }
    if args.all && args.output.is_some() {
        eprintln!("--output cannot be used with --all (each type writes to its own directory).");
        std::process::exit(1);
    }

    // Open cross-type progress log (append so restarts don't lose history).
    let log_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("gen_progress.log");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let summary_log = Arc::new(Mutex::new(log_file));
    {
        let mut log = summary_log.lock().unwrap();
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        writeln!(log, "\n=== run started (unix {secs}) ===")?;
    }
    println!("Progress log: {}", log_path.display());

    // Load model once; reuse for all content types.
    println!("Loading Qwen3-30B-A3B-Q4 from {}...",
        args.model_dir.as_deref().map(|p| p.display().to_string())
            .unwrap_or_else(|| "HuggingFace hub".into()));
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

    let model_core = engine.model_core_properties();
    let prov_layers = model_core.provenance_layer_indices.as_array();
    let n_kv_heads = model_core.n_kv_heads;
    let head_dim = model_core.head_dim;
    let n_total_layers = model_core.num_layers;
    let n_layers_per_band = (n_total_layers * 20 / 100).max(1);
    let band_half_width = n_layers_per_band / 2;

    println!("Provenance layers: {prov_layers:?}");
    println!("n_kv_heads={n_kv_heads}  head_dim={head_dim}  n_total_layers={n_total_layers}");
    println!("band: n_layers_per_band={n_layers_per_band}  half={band_half_width}");

    let bands = compute_bands(prov_layers, n_total_layers, n_layers_per_band)[0].clone();
    let all_layer_indices: Vec<usize> = bands.iter().flat_map(|b| b.iter().copied()).collect();
    let unique_layers: Vec<usize> = {
        let mut seen = std::collections::HashSet::new();
        all_layer_indices.into_iter().filter(|l| seen.insert(*l)).collect()
    };

    let raw_header = RawFileHeader {
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        n_layers_per_band: n_layers_per_band as u32,
        band_half_width: band_half_width as u32,
        band_centers: [
            prov_layers[1] as u32,
            prov_layers[3] as u32,
            prov_layers[5] as u32,
        ],
        n_total_layers: n_total_layers as u32,
    };

    let ctx = ModelCtx {
        engine: &engine,
        conv_config,
        dialect,
        prov_layers,
        n_kv_heads,
        head_dim,
        n_total_layers,
        n_layers_per_band,
        band_half_width,
        bands,
        unique_layers,
        raw_header,
        batch_size: args.batch_size,
        max_tokens: args.max_tokens,
        skip_raw: args.skip_raw,
        summary_log,
    };

    let types: &[ContentType] = if args.all {
        ALL_TYPES
    } else {
        std::slice::from_ref(args.content_type.as_ref().unwrap())
    };

    for &ct in types {
        generate_one(ct, args.output.as_ref(), args.force, &ctx)?;
    }

    println!("\nAll done.");
    Ok(())
}
