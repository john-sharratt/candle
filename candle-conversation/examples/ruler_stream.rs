//! Continuous RULER benchmark using [`ConversationEngine`] for automatic batching.
//!
//! Spawns N worker threads (one per slot in the context-length config); all share
//! a single [`ConversationEngine`].  The scheduler's `batch_decode_step` groups
//! all concurrently active decode sequences into one GPU forward pass, so 4 K and
//! 32 K contexts make progress in parallel.
//!
//! # Run (benchmark)
//!
//! ```powershell
//! cargo run --example ruler_stream -p candle-conversation --release --features "cuda,hub" `
//!     -- --mode C8 --log ruler_stream_C8.jsonl
//! ```
//!
//! # Run (report only, no GPU)
//!
//! ```powershell
//! cargo run --example ruler_stream -p candle-conversation --release `
//!     -- --report --log ruler_stream_C8.jsonl
//! ```
//!
//! # Output format (one JSON line per inference)
//!
//! ```json
//! {"quant":"C8","ctx":4096,"task":"niah_single_1","sample":3,"status":"success","elapsed":1.23}
//! ```

use candle::Device;
use candle_conversation::{models::Model, ConversationEngine};
use candle_transformers::models::batch_test::ruler_gen::{
    generate_ruler_samples, parse_ruler_mode, print_ruler_report, score_ruler_sample, RulerTask,
};
use candle_transformers::models::batched_inference::BatchedConfig;
use clap::Parser;
use std::{
    io::Write,
    sync::{Arc, Mutex},
};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "ruler_stream",
    about = "Continuous RULER benchmark with automatic batching via ConversationEngine"
)]
struct Args {
    /// KV compression mode: F16, Q8_0, Q8_Q4, Q4_0, C5, C8, C9, C10
    #[arg(long, default_value = "F16")]
    mode: String,

    /// Output JSONL file (appended, not overwritten)
    #[arg(long, default_value = "ruler_stream.jsonl")]
    log: String,

    /// Print accuracy table from an existing log file and exit (no GPU needed)
    #[arg(long)]
    report: bool,

    /// CUDA device index
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// Maximum new tokens to generate per sample
    #[arg(long, default_value_t = 50)]
    max_gen: usize,

    /// KV cache token budget (must cover longest context + generation)
    #[arg(long, default_value_t = 32_768)]
    token_budget: usize,

    /// Verbosity: -v=debug, -vv=trace (default: info)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    verbose: u8,
}

// ── Fixed config ──────────────────────────────────────────────────────────────

/// (context length in tokens, number of concurrent worker threads)
///
/// F16 uses a reduced thread count: it has no quantized-arena reconciliation
/// path, so KV memory grows with high-water-mark across slot turnover and the
/// scheduler degrades sharply once VRAM is under pressure (observed F16: 12-seq
/// 512-token prefills going from ~3 s → ~15 s after a few free/recreate
/// cycles). All quantized modes use the wider config.
const CTX_CONFIG_DEFAULT: &[(usize, usize)] = &[
    (4_096, 4),
    (8_192, 1),
    (16_384, 1),
];

const CTX_CONFIG_C5: &[(usize, usize)] = &[
    (4_096, 6),
    (8_192, 1),
    (16_384, 1),
];

const CTX_CONFIG_C10: &[(usize, usize)] = &[
    (4_096, 6),
    (8_192, 1),
    (16_384, 1),
];

const CTX_CONFIG_HIGH: &[(usize, usize)] = &[
    (4_096, 12),
    (8_192, 3),
    (16_384, 1),
];

fn ctx_config_for(mode_label: &str) -> &'static [(usize, usize)] {
    let label = mode_label.to_ascii_uppercase();
    if label == "C5" {
        CTX_CONFIG_C5
    } else if label == "C10" {
        CTX_CONFIG_C10
    } else if label.starts_with('C') {
        // C8, C9 — adaptive compression modes get the wide config.
        CTX_CONFIG_HIGH
    } else {
        // F16, Q4_0, Q8_0, Q8_Q4 — narrow config (KV grows fast, no reconciliation).
        CTX_CONFIG_DEFAULT
    }
}

const TASKS: &[RulerTask] = &[
    RulerTask::NiahSingle1,
    RulerTask::NiahMultiKey2,
    RulerTask::Vt,
];

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // ── Tracing console subscriber ───────────────────────────────────────────
    let level = match args.verbose {
        0 => tracing::Level::INFO,
        1 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(true)
        .with_writer(std::io::stderr)
        .try_init()
        .ok();

    // ── Report-only mode (no GPU required) ───────────────────────────────────
    if args.report {
        return print_ruler_report(std::path::Path::new(&args.log))
            .map_err(|e| anyhow::anyhow!("{e}"));
    }

    // ── Benchmark mode ────────────────────────────────────────────────────────
    let (mode, mode_label) =
        parse_ruler_mode(&args.mode).map_err(|e| anyhow::anyhow!("{e}"))?;

    let batched_config = match mode {
        Some(ref m) => BatchedConfig {
            k_format: m.k_format(),
            v_format: m.v_format(),
            compression_level: m.compression_level(),
            ..BatchedConfig::default()
        },
        None => BatchedConfig::default(),
    };

    let ctx_config = ctx_config_for(&mode_label);

    let device = Device::new_cuda(args.device)?;
    let builder = Model::Qwen3_8B_Q6
        .builder()
        .max_concurrent(ctx_config.iter().map(|(_, n)| n).sum::<usize>() + 1)
        .max_seq_len(args.token_budget);
    let (model_path, tokenizer_path) = builder.resolve_paths_pub()?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let model = builder.load_model(&model_path, &device)?;
    let mut engine_config = builder.engine_config(&tokenizer);
    engine_config.batched_config = batched_config;

    let engine = Arc::new(ConversationEngine::new(model, tokenizer.clone(), engine_config)?);

    // Open the log file in append mode so repeated runs accumulate data.
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.log)?;
    let writer = Arc::new(Mutex::new(std::io::BufWriter::new(file)));

    let tier_summary = ctx_config
        .iter()
        .map(|(ctx, n)| format!("{}×{}K", n, ctx / 1024))
        .collect::<Vec<_>>()
        .join("  ");
    println!(
        "=== RULER stream: mode={mode_label}  log={} ===\n\
         Threads: {tier_summary} — Ctrl+C to stop\n",
        args.log,
    );

    let max_gen = args.max_gen;
    let mut handles = vec![];
    for &(ctx_len, n_threads) in ctx_config {
        for thread_id in 0..n_threads {
            let engine = Arc::clone(&engine);
            let tokenizer = tokenizer.clone();
            let writer = Arc::clone(&writer);
            let mode_label = mode_label.clone();

            handles.push(std::thread::spawn(move || {
                // xorshift64 seeded distinctly per thread
                let seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .subsec_nanos() as u64;
                let mut rng: u64 = seed
                    ^ ((ctx_len as u64) << 32)
                    ^ (thread_id as u64).wrapping_mul(0x9e3779b97f4a7c15);
                let mut xorshift = move || -> u64 {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    rng
                };

                // Each thread starts at a different sample index and strides by
                // n_threads so all threads together cover distinct samples.
                let mut sample_idx: u64 = thread_id as u64;

                loop {
                    let task = TASKS[xorshift() as usize % TASKS.len()];
                    let samples =
                        generate_ruler_samples(&tokenizer, task, ctx_len, 1, sample_idx);
                    let sample = &samples[0];

                    // RULER samples are already fully formatted ChatML strings;
                    // tokenize without adding special tokens (they're inline).
                    let tokens = match tokenizer.encode(sample.input.as_str(), false) {
                        Ok(enc) => enc.get_ids().to_vec(),
                        Err(e) => {
                            eprintln!("[WARN] tokenize error: {e}");
                            continue;
                        }
                    };

                    let t0 = std::time::Instant::now();
                    let pred = match engine.infer_raw_tokens(&tokens, max_gen) {
                        Ok(p) => p,
                        Err(e) => {
                            let err_str = e.to_string();
                            eprintln!(
                                "[WARN] {mode_label} {ctx_len} {} s={sample_idx}: {err_str}",
                                task.name()
                            );
                            let elapsed = t0.elapsed().as_secs_f64();
                            let line = format!(
                                "{{\"quant\":\"{mode_label}\",\"ctx\":{ctx_len},\
                                 \"task\":\"{}\",\"sample\":{sample_idx},\
                                 \"status\":\"error\",\"elapsed\":{elapsed:.3},\
                                 \"error\":{}}}\n",
                                task.name(),
                                serde_json::to_string(&err_str).unwrap_or_else(|_| "\"unknown\"".into()),
                            );
                            {
                                let mut w = writer.lock().unwrap();
                                let _ = w.write_all(line.as_bytes());
                                let _ = w.flush();
                            }
                            println!(
                                "[{mode_label} {}K {} s={sample_idx}]  !  {elapsed:.1}s (error)",
                                ctx_len / 1024,
                                task.name(),
                            );
                            continue;
                        }
                    };
                    let elapsed = t0.elapsed().as_secs_f64();
                    let correct = score_ruler_sample(task, &pred, &sample.outputs);

                    let line = if correct {
                        format!(
                            "{{\"quant\":\"{mode_label}\",\"ctx\":{ctx_len},\
                             \"task\":\"{}\",\"sample\":{sample_idx},\
                             \"status\":\"success\",\"elapsed\":{elapsed:.3}}}\n",
                            task.name()
                        )
                    } else {
                        let pred_json = serde_json::to_string(pred.trim()).unwrap_or_else(|_| "\"\"".into());
                        let expected_json = serde_json::to_string(&sample.outputs).unwrap_or_else(|_| "[]".into());
                        format!(
                            "{{\"quant\":\"{mode_label}\",\"ctx\":{ctx_len},\
                             \"task\":\"{}\",\"sample\":{sample_idx},\
                             \"status\":\"failed\",\"elapsed\":{elapsed:.3},\
                             \"pred\":{pred_json},\"expected\":{expected_json}}}\n",
                            task.name()
                        )
                    };
                    {
                        let mut w = writer.lock().unwrap();
                        let _ = w.write_all(line.as_bytes());
                        let _ = w.flush();
                    }

                    println!(
                        "[{mode_label} {}K {} s={sample_idx}]  {}  {elapsed:.1}s",
                        ctx_len / 1024,
                        task.name(),
                        if correct { "✓" } else { "✗" },
                    );

                    sample_idx += n_threads as u64;
                }
            }));
        }
    }

    for h in handles {
        let _ = h.join();
    }
    Ok(())
}
