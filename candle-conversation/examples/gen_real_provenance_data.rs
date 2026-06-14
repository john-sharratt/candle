//! Generate real tool-provenance KV/Q fixtures using Qwen3-30B-A3B.
//!
//! Produces two output files:
//!
//! 1. `signatures.prov` + `MANIFEST.json` — existing sign-bit signatures
//!    (used by the projection harness and the live BDP scanner).
//!
//! 2. `raw_kvq.prov` + `RAW_MANIFEST.json` — raw f32 K and Q vectors across
//!    a 20%-wide layer band centred on the three provenance depth slices.
//!    Used by the harness to sweep signature strategies offline.
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
//!   --release --features hub -- \
//!   --model-dir /path/to/Qwen3-30B-A3B-Q4_K_M
//! ```
//!
//! Output: `tests/tool_provenance_real_data/{signatures.prov,MANIFEST.json}`
//!          `tests/tool_provenance_real_data/{raw_kvq.prov,RAW_MANIFEST.json}`

use std::path::PathBuf;
use std::time::Instant;

use candle::Device;
use candle_conversation::Sequence;
use candle_conversation::{
    models::{Dialect, Model},
    provenance::{
        band_layer_indices, build_token_blob, RawFileHeader, RawProvenanceFile, RawSigEntry,
    },
    ProvenanceFile, SamplingConfig, SigEntry, TurnHandle,
};
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

    /// Skip generating raw KVQ data (only sign-bit signatures).
    #[arg(long)]
    skip_raw: bool,

    /// Append new scenarios to an existing output corpus.
    /// Reads existing MANIFEST.json, skips already-processed scenario IDs,
    /// and appends only the new entries to signatures.prov + MANIFEST.json.
    #[arg(long)]
    append: bool,
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
#[derive(Debug, Serialize, Deserialize)]
struct OutScenario {
    id: String,
    tool: Option<String>,
    case_type: CaseType,
    user_prompt: String,
    generated_text: String,
    turn_id: u64,
    /// Decode-phase Q vectors (model generating the tool-call response).
    byte_offset: u64,
    token_count: u16,
    block_count: usize,
    /// Prefill-phase Q vectors (model reading the user prompt only).
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_byte_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_token_count: Option<u16>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OutManifest {
    version: u32,
    model: String,
    provenance_layer_indices: [usize; 6],
    scenarios: Vec<OutScenario>,
}

/// Output scenario written to RAW_MANIFEST.json.
#[derive(Debug, Serialize, Deserialize)]
struct RawOutScenario {
    id: String,
    tool: Option<String>,
    case_type: CaseType,
    raw_byte_offset: u64,
    raw_token_count: u16,
}

#[derive(Debug, Serialize, Deserialize)]
struct RawOutManifest {
    version: u32,
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

// ── Band computation ──────────────────────────────────────────────────────────

/// Build the three-band layer index lists from provenance centres and model size.
fn compute_bands(
    prov_layers: [usize; 6],
    n_total_layers: usize,
    n_layers_per_band: usize,
) -> [[Vec<usize>; 3]; 1] {
    let half = n_layers_per_band / 2;
    // prov_layers = [syn_l0, syn_l4, sem_l0, sem_l4, prag_l0, prag_l4]
    // Band centres are the l4 values (indices 1, 3, 5).
    [[
        band_layer_indices(prov_layers[1], half, n_total_layers, n_layers_per_band),
        band_layer_indices(prov_layers[3], half, n_total_layers, n_layers_per_band),
        band_layer_indices(prov_layers[5], half, n_total_layers, n_layers_per_band),
    ]]
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
    let all_count = in_manifest.scenarios.len();

    // Guard output directory.
    let sig_path = args.output.join("signatures.prov");
    let prefill_sig_path = args.output.join("prefill_signatures.prov");
    let raw_path = args.output.join("raw_kvq.prov");

    // Collect already-processed IDs when appending.
    let existing_ids: std::collections::HashSet<String> = if args.append {
        let manifest_out = args.output.join("MANIFEST.json");
        if manifest_out.exists() {
            let text = std::fs::read_to_string(&manifest_out)?;
            // OutManifest has a `scenarios` array; parse just enough.
            #[derive(serde::Deserialize)]
            struct MinScenario {
                id: String,
            }
            #[derive(serde::Deserialize)]
            struct MinManifest {
                scenarios: Vec<MinScenario>,
            }
            let m: MinManifest = serde_json::from_str(&text)?;
            m.scenarios.into_iter().map(|s| s.id).collect()
        } else {
            std::collections::HashSet::new()
        }
    } else {
        std::collections::HashSet::new()
    };

    // In append mode, only process scenarios not yet in the output manifest.
    let scenarios_to_run: Vec<&InScenario> = if args.append && !existing_ids.is_empty() {
        in_manifest
            .scenarios
            .iter()
            .filter(|s| !existing_ids.contains(&s.id))
            .collect()
    } else {
        in_manifest.scenarios.iter().collect()
    };
    let total = scenarios_to_run.len();
    if args.append {
        println!(
            "Loaded {all_count} scenarios from {}; {total} new (skipping {} already done)",
            prompts_manifest.display(),
            all_count - total
        );
    } else {
        println!(
            "Loaded {total} scenarios from {}",
            prompts_manifest.display()
        );
    }

    if !args.append {
        for path in [&sig_path, &prefill_sig_path, &raw_path] {
            if path.exists() {
                if !args.force {
                    eprintln!(
                        "Output already exists at {}.  Use --force to overwrite.",
                        path.display()
                    );
                    std::process::exit(1);
                }
                std::fs::remove_file(path)?;
            }
        }
    }
    std::fs::create_dir_all(&args.output)?;

    // ── Load model ────────────────────────────────────────────────────────────

    println!(
        "Loading model from {}...",
        args.model_dir
            .as_deref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "HuggingFace hub".into())
    );
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

    println!("Provenance layer indices: {prov_layers:?}");
    println!("Model: n_kv_heads={n_kv_heads}, head_dim={head_dim}, n_layers={n_total_layers}");
    println!("Raw band: n_layers_per_band={n_layers_per_band}, half={band_half_width}");
    println!("Batch size: {}", args.batch_size);

    // Compute band layer indices for raw extraction.
    let bands = &compute_bands(prov_layers, n_total_layers, n_layers_per_band)[0];
    let all_layer_indices: Vec<usize> = bands.iter().flat_map(|b| b.iter().copied()).collect();
    // Deduplicated but keep first-occurrence order for the layer→(band, layer_in_band) map.
    let mut layer_to_band_slot: std::collections::HashMap<usize, Vec<(usize, usize)>> =
        std::collections::HashMap::new();
    for (band, band_layers) in bands.iter().enumerate() {
        for (slot, &layer_idx) in band_layers.iter().enumerate() {
            layer_to_band_slot
                .entry(layer_idx)
                .or_default()
                .push((band, slot));
        }
    }
    let unique_layers: Vec<usize> = {
        let mut seen = std::collections::HashSet::new();
        all_layer_indices
            .iter()
            .copied()
            .filter(|l| seen.insert(*l))
            .collect()
    };

    // Build raw file header.
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

    // ── Output files ─────────────────────────────────────────────────────────

    let out_pf = ProvenanceFile::open(&sig_path)?;
    let out_prefill_pf = ProvenanceFile::open(&prefill_sig_path)?;
    let out_raw_pf = if args.skip_raw {
        None
    } else if args.append && raw_path.exists() {
        Some(RawProvenanceFile::open(&raw_path)?)
    } else {
        Some(RawProvenanceFile::create(&raw_path, raw_header)?)
    };

    let mut out_scenarios: Vec<OutScenario> = Vec::with_capacity(total);
    let mut raw_scenarios: Vec<RawOutScenario> = Vec::with_capacity(total);

    // ── Batched inference loop ────────────────────────────────────────────────

    let t_start = Instant::now();
    let mut completed = 0usize;

    for (batch_start, chunk) in scenarios_to_run.chunks(args.batch_size).enumerate() {
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
            "  Submitted {} turns in {:.1}ms",
            slots.len(),
            t_submit.elapsed().as_secs_f64() * 1000.0
        );

        // ── Phase B: drain handles in submission order ────────────────────────

        let pf = engine.provenance_file();

        for slot in slots {
            let Slot {
                global_idx,
                id,
                tool,
                case_type,
                user_prompt,
                mut conv,
                handle,
            } = slot;

            let resp = handle.wait()?;
            conv.finish_turn(handle, &resp)?;

            // ── Extract raw KVQ before the KV cache is freed ─────────────────
            let raw_entry: Option<RawSigEntry> = if let Some(ref raw_pf) = out_raw_pf {
                let seal = resp.seal.as_ref();
                if let Some(seal) = seal {
                    let block_range = Some((seal.block_from, seal.block_to));
                    match conv.extract_raw_kvq(unique_layers.clone(), block_range) {
                        Ok(layer_data) => {
                            // Build a (layer_idx → blocks) map.
                            let layer_map: std::collections::HashMap<
                                usize,
                                &Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>,
                            > = layer_data
                                .iter()
                                .map(|(li, blocks)| (*li, blocks))
                                .collect();

                            // Collect per-band-layer raw block data for build_token_blob.
                            let n_blocks = seal.block_to.saturating_sub(seal.block_from);
                            let total_tokens = seal.turn_token_count;
                            let chunk_size = candle_nn::CHUNK_SIZE;

                            let mut all_token_bytes: Vec<u8> = Vec::new();

                            for block_offset in 0..n_blocks {
                                let tokens_this_block = if block_offset + 1 < n_blocks {
                                    chunk_size
                                } else {
                                    let rem = total_tokens % chunk_size;
                                    if rem == 0 {
                                        chunk_size
                                    } else {
                                        rem
                                    }
                                };

                                // Build band_layer_data for this block: [band][layer_in_band] → (k, q)
                                let band_layer_data: [Vec<(&[f32], &[f32])>; 3] =
                                    std::array::from_fn(|band| {
                                        bands[band]
                                            .iter()
                                            .map(|&layer_idx| {
                                                if let Some(blocks) = layer_map.get(&layer_idx) {
                                                    // Find the block at block_offset within the seal range.
                                                    let abs_block = seal.block_from + block_offset;
                                                    if let Some((_, k, _v, q)) = blocks
                                                        .iter()
                                                        .find(|(bi, ..)| *bi == abs_block)
                                                    {
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

                                let blob = build_token_blob(
                                    &raw_header,
                                    tokens_this_block,
                                    &band_layer_data,
                                );
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

            drop(conv); // release KV slot for the next batch

            let seal = match &resp.seal {
                Some(s) => s,
                None => {
                    eprintln!("  [{global_idx}] WARNING: no seal for {id} — skipping");
                    continue;
                }
            };

            // Partition `seal.new_sig_entries` into prefill-phase and
            // decode-phase chunks using `prefill_token_count` reported
            // by the scheduler.  The two phases land in distinct sets
            // of chunks (the scheduler push_empty_writer_chunk's after
            // prefill ends), so accumulating `token_count` along the
            // entries gives us the partition without a second
            // forward pass — this replaces the old `prefill_sigs_for`
            // call which ran a separate prefill of just `user_prompt`
            // on a scratch slot (different prefix shape, separate
            // provenance write).  The new path captures prefill sigs
            // under the actual ChatML-formatted prefix the model
            // sees at runtime — same code path, fewer artefacts.
            let prefill_tokens_target = resp.stats.prefill_token_count;
            let mut prefill_sig_entries: Vec<&SigEntry> = Vec::new();
            let mut decode_sig_entries: Vec<&SigEntry> = Vec::new();
            {
                let mut tokens_so_far: usize = 0;
                for entry in &seal.new_sig_entries {
                    if tokens_so_far < prefill_tokens_target {
                        prefill_sig_entries.push(entry);
                        tokens_so_far += entry.token_count as usize;
                    } else {
                        decode_sig_entries.push(entry);
                    }
                }
            }

            // Write prefill sigs to their dedicated provenance file.
            let prefill_out_entry: Option<SigEntry> = if !prefill_sig_entries.is_empty() {
                let mut pre_syn = Vec::new();
                let mut pre_sem = Vec::new();
                let mut pre_prag = Vec::new();
                let mut ok = true;
                for entry in &prefill_sig_entries {
                    match pf.read_entry(**entry) {
                        Ok((s, se, p)) => {
                            pre_syn.extend(s);
                            pre_sem.extend(se);
                            pre_prag.extend(p);
                        }
                        Err(e) => {
                            eprintln!("  [{global_idx}] WARNING: prefill read_entry failed: {e}");
                            ok = false;
                            break;
                        }
                    }
                }
                if ok && !pre_syn.is_empty() {
                    match out_prefill_pf.append(&pre_syn, &pre_sem, &pre_prag) {
                        Ok(e) => Some(e),
                        Err(e) => {
                            eprintln!("  [{global_idx}] WARNING: prefill append failed: {e}");
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            // Read real Q sign-bits from engine's internal ProvenanceFile.
            // Decode-only sigs go to signatures.prov (prefill chunks
            // were already accounted for above).
            let mut all_syn = Vec::new();
            let mut all_sem = Vec::new();
            let mut all_prag = Vec::new();
            for entry in &decode_sig_entries {
                let (syn, sem, prag) = pf.read_entry(**entry)?;
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

            if let Some(re) = raw_entry {
                raw_scenarios.push(RawOutScenario {
                    id: id.clone(),
                    tool: tool.clone(),
                    case_type,
                    raw_byte_offset: re.byte_offset,
                    raw_token_count: re.token_count,
                });
            }

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
                prefill_byte_offset: prefill_out_entry.map(|e| e.byte_offset),
                prefill_token_count: prefill_out_entry.map(|e| e.token_count),
            });
            completed += 1;
        }
    }

    // ── Write manifests ───────────────────────────────────────────────────────

    // In append mode, merge new scenarios with existing ones.
    let final_scenarios = if args.append {
        let manifest_out_path = args.output.join("MANIFEST.json");
        if manifest_out_path.exists() {
            let existing_text = std::fs::read_to_string(&manifest_out_path)?;
            let mut existing: OutManifest = serde_json::from_str(&existing_text)?;
            existing.scenarios.extend(out_scenarios);
            existing.scenarios
        } else {
            out_scenarios
        }
    } else {
        out_scenarios
    };

    let out_manifest = OutManifest {
        version: 1,
        model: "Qwen3-30B-A3B-Q4_K_M".to_string(),
        provenance_layer_indices: prov_layers,
        scenarios: final_scenarios,
    };
    let manifest_out = args.output.join("MANIFEST.json");
    std::fs::write(&manifest_out, serde_json::to_string_pretty(&out_manifest)?)?;

    if out_raw_pf.is_some() && !raw_scenarios.is_empty() {
        // In append mode, merge with existing raw manifest.
        let final_raw_scenarios = if args.append {
            let raw_manifest_out_path = args.output.join("RAW_MANIFEST.json");
            if raw_manifest_out_path.exists() {
                let existing_text = std::fs::read_to_string(&raw_manifest_out_path)?;
                let mut existing: RawOutManifest = serde_json::from_str(&existing_text)?;
                existing.scenarios.extend(raw_scenarios);
                existing.scenarios
            } else {
                raw_scenarios
            }
        } else {
            raw_scenarios
        };
        let raw_manifest = RawOutManifest {
            version: 1,
            model: "Qwen3-30B-A3B-Q4_K_M".to_string(),
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
            scenarios: final_raw_scenarios,
        };
        let raw_manifest_out = args.output.join("RAW_MANIFEST.json");
        std::fs::write(
            &raw_manifest_out,
            serde_json::to_string_pretty(&raw_manifest)?,
        )?;
        println!(
            "  raw_kvq.prov    : {} bytes",
            std::fs::metadata(&raw_path)?.len()
        );
        println!(
            "  RAW_MANIFEST.json: {} bytes",
            std::fs::metadata(&raw_manifest_out)?.len()
        );
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    println!(
        "\nDone — {completed}/{total} scenarios in {elapsed:.1}s ({:.1} s/scenario avg)",
        elapsed / completed.max(1) as f64,
    );
    println!(
        "  signatures.prov : {} bytes",
        std::fs::metadata(&sig_path)?.len()
    );
    println!(
        "  MANIFEST.json   : {} bytes",
        std::fs::metadata(&manifest_out)?.len()
    );

    Ok(())
}
