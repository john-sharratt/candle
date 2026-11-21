//! Runs the benchmark in two phases: a sweep that finds the batch size each
//! compression mode can hold, then the full evaluation at those batch sizes.
//!
//! The sweep runs the needle task with no compression, doubling the batch until the
//! token budget or throughput limit is reached, and scales that batch by each mode's
//! estimated compression ratio with a safety margin. The scaling starts from the
//! budget ceiling rather than the throughput optimum, so a compressed mode is not
//! held back because the uncompressed one stopped improving early. The evaluation
//! then runs every mode, length and task, loads or generates each set of samples
//! once, and reports per-mode averages. Mode labels are parsed into the inference
//! mode they name, with the uncompressed baseline as the reserved label.

use super::*;

// ── Benchmark orchestrator ────────────────────────────────────────────────────

/// Configuration for [`run_ruler_benchmark`].
pub struct RulerBenchConfig<'a> {
    /// Model name used in printed output.
    pub model_name: &'a str,
    /// EOS token IDs (e.g. [`QWEN3_EOS_IDS`]).
    pub eos_ids: &'a [u32],
    /// Maximum total KV tokens across all concurrent sequences.
    /// Batch size is capped at `token_budget / context_len`.
    pub token_budget: usize,
    /// Maximum decode tokens per sequence.
    pub max_gen_tokens: usize,
    /// Compression modes to sweep: `(mode, label)`.
    /// `None` = F16 lossless baseline.
    pub modes: &'a [(Option<InferenceMode>, &'a str)],
    /// `(context_len, n_samples)` pairs.
    /// Keep `context_len * max_batch ≤ token_budget`.
    pub lengths_samples: &'a [(usize, usize)],
    /// Tasks to evaluate.
    pub tasks: &'a [RulerTask],
    /// Where to load evaluation samples from.
    /// Defaults to [`RulerDataSource::Generated`] when using `..Default::default()`.
    pub data_source: RulerDataSource<'a>,
}

impl Default for RulerBenchConfig<'static> {
    /// Paper-run defaults: 4 tasks × 4 lengths × 4 compression modes.
    /// Sample counts scale inversely with context length (base=20 @ 4K).
    fn default() -> Self {
        Self {
            model_name: "unknown",
            eos_ids: QWEN3_EOS_IDS,
            token_budget: 32_768,
            max_gen_tokens: 50,
            modes: &[
                (None, "F16 (lossless)"),
                (Some(InferenceMode::Q4_0), "Q4_0 (4.5 BPE)"),
                (Some(InferenceMode::C5), "C5  (PalQuant ~4.4 BPE)"),
                (Some(InferenceMode::Q3_0), "Q3_0 (3.5 BPE)"),
                (Some(InferenceMode::C8), "C8  (PalQuant ~3.3 BPE)"),
            ],
            lengths_samples: &[(4_096, 20), (8_192, 10), (16_384, 5)],
            tasks: &[
                RulerTask::NiahSingle1,
                RulerTask::NiahMultiKey2,
                RulerTask::Vt,
                RulerTask::Cwe,
            ],
            data_source: RulerDataSource::Generated,
        }
    }
}

/// Phase 1: parallelism sweep.
///
/// Returns the estimated KV-cache compression ratio (relative to F16) for a
/// given inference mode.  Used to scale the VRAM-safe batch ceiling when modes
/// other than F16 are evaluated in Phase 2.
pub(super) fn kv_compression_ratio(mode: Option<InferenceMode>) -> f64 {
    match mode {
        None => 1.0, // F16 baseline
        Some(m) => match m {
            InferenceMode::Q8_0 | InferenceMode::Q8_1 | InferenceMode::Q8_KS => 2.0,
            InferenceMode::Q8_Q4 => 2.67,
            InferenceMode::Q8_Q4KS | InferenceMode::Q8_Q8KS => 2.0,
            InferenceMode::Q4_0 | InferenceMode::Q4_1 | InferenceMode::Q4_KS => 4.0,
            InferenceMode::Q3_0 => 5.3,
            InferenceMode::Q2_0 => 8.0,
            InferenceMode::C5 => 3.7,
            InferenceMode::C8 => 4.8,
            InferenceMode::C9 => 5.4,
            InferenceMode::C10 => 7.4,
            InferenceMode::C0 => 1.1,
            InferenceMode::C1 => 1.3,
            InferenceMode::C2 => 1.7,
            InferenceMode::C3 => 2.2,
            InferenceMode::C4 => 2.9,
            InferenceMode::C6 => 4.1,
            InferenceMode::C7 => 4.5,
            _ => 1.0, // conservative fallback for unknown modes
        },
    }
}

/// Runs NIAH-S1 at each context length with `mode=None` (F16), doubling the
/// batch size from 1 until `batch * ctx_len > token_budget` or throughput
/// plateaus.  Returns the F16 throughput-optimal batch, then scales it by each
/// mode's compression ratio to produce per-mode batch ceilings for Phase 2.
pub fn sweep_parallelism<M: ManagedBatchedModel>(
    model: &M,
    tokenizer: &Tokenizer,
    cfg: &RulerBenchConfig<'_>,
) -> Result<Vec<(usize, Vec<usize>)>> {
    println!("\n=== Phase 1: Parallelism sweep (NIAH-S1, F16 KV) ===");
    println!(
        "{:<10} {:>8} {:>14} {:>14} {:>10}",
        "Ctx", "Batch", "Prefill t/s", "Decode t/s", "Selected"
    );

    // Safety factor applied when scaling the F16 batch ceiling by compression ratio.
    // Accounts for imprecision in ratio estimates and activation memory overhead.
    // 0.63 = 0.90 baseline × 0.70 (extra 30% headroom requested).
    const SAFETY: f64 = 0.63;

    let mut best_per_length: Vec<(usize, Vec<usize>)> = Vec::new(); // (ctx_len, [batch_per_mode])

    for &(ctx_len, _) in cfg.lengths_samples {
        // Generate a fixed set of samples for the sweep (seed=0).
        // We need enough samples to fill the largest batch we'll try.
        let max_batch = cfg.token_budget / ctx_len;
        let n_sweep = max_batch.max(1);
        let samples =
            generate_ruler_samples(tokenizer, RulerTask::NiahSingle1, ctx_len, n_sweep, 0);

        let mut best_batch = 1usize;
        let mut best_decode_tps = 0.0f64;

        let mut batch = 1usize;
        loop {
            if batch > samples.len() {
                break;
            }
            let batch_samples = &samples[..batch];

            let mut timings = (0.0f64, 0.0f64);
            let preds = run_ruler_eval(
                model,
                tokenizer,
                batch_samples,
                None,
                cfg.max_gen_tokens,
                cfg.eos_ids,
                model.device(),
                Some(&mut timings),
            )?;
            let _ = preds; // discard predictions for sweep

            let (prefill_secs, decode_secs) = timings;

            // Prompt tokens for this batch.
            let prompt_toks: usize = batch_samples
                .iter()
                .map(|s| {
                    tokenizer
                        .encode(s.input.as_str(), false)
                        .map(|e| e.len())
                        .unwrap_or(0)
                })
                .sum();
            let decode_toks = batch * cfg.max_gen_tokens;

            let prefill_tps = if prefill_secs > 0.0 {
                prompt_toks as f64 / prefill_secs
            } else {
                0.0
            };
            let decode_tps = if decode_secs > 0.0 {
                decode_toks as f64 / decode_secs
            } else {
                0.0
            };

            let selected = if decode_tps >= best_decode_tps * 0.97 {
                // Still improving (or within 3% noise) — record as best.
                best_decode_tps = decode_tps;
                best_batch = batch;
                "  <--"
            } else {
                "  plateau"
            };

            println!(
                "{:<10} {:>8} {:>14.0} {:>14.0}{}",
                format!("{}K", ctx_len / 1024),
                batch,
                prefill_tps,
                decode_tps,
                selected,
            );

            // Next batch size: double, but cap at token_budget / ctx_len.
            let next = (batch * 2).min(cfg.token_budget / ctx_len);
            if next == batch || decode_tps < best_decode_tps * 0.90 {
                break; // plateau or cap reached
            }
            batch = next;
        }

        println!("  → Best F16 batch for {}K: {}", ctx_len / 1024, best_batch);

        // Scale to per-mode batch using compression ratio × safety factor.
        // token_budget / ctx_len is the F16 hard ceiling; best_batch may be lower.
        let f16_ceiling = cfg.token_budget / ctx_len;
        println!(
            "  → Projected batches for {}K (safety={:.0}%):",
            ctx_len / 1024,
            SAFETY * 100.0
        );
        let mode_batches: Vec<usize> = cfg
            .modes
            .iter()
            .map(|&(mode, label)| {
                let ratio = kv_compression_ratio(mode);
                // Scale from the F16 ceiling (not just best_batch) so compressed
                // modes are not penalised if F16 plateaued before hitting the wall.
                let b = ((f16_ceiling as f64 * ratio * SAFETY).floor() as usize).max(1);
                println!("      {:<24}  ratio={:.1}×  →  batch {}", label, ratio, b);
                b
            })
            .collect();

        best_per_length.push((ctx_len, mode_batches));
    }

    Ok(best_per_length)
}

/// Phase 2: full RULER benchmark.
///
/// Uses the best batch sizes from the sweep.  Evaluates all `cfg.modes ×
/// cfg.lengths_samples × cfg.tasks` and prints an accuracy table.
pub fn run_ruler_benchmark<M: ManagedBatchedModel>(
    model: &M,
    tokenizer: &Tokenizer,
    cfg: &RulerBenchConfig<'_>,
) -> Result<()> {
    let best_batches = sweep_parallelism(model, tokenizer, cfg)?;

    let n_modes = cfg.modes.len();
    let n_tasks = cfg.tasks.len();
    let n_lengths = cfg.lengths_samples.len();

    // scores[mode][length][task]
    let mut scores = vec![vec![vec![f64::NAN; n_tasks]; n_lengths]; n_modes];

    println!(
        "\n=== Phase 2: Full RULER benchmark — {} ===",
        cfg.model_name
    );

    for (li, &(ctx_len, n_samples)) in cfg.lengths_samples.iter().enumerate() {
        let mode_batches: &[usize] = best_batches
            .iter()
            .find(|(c, _)| *c == ctx_len)
            .map(|(_, mb)| mb.as_slice())
            .unwrap_or(&[]);

        println!(
            "\n─── Context {}K  ({} samples/task) ───",
            ctx_len / 1024,
            n_samples
        );

        for (ti, &task) in cfg.tasks.iter().enumerate() {
            // Load or generate data once; reuse across compression modes.
            let all_samples = match &cfg.data_source {
                RulerDataSource::Generated => {
                    print!("  Generating {} … ", task);
                    let seed = (li as u64) * 1000 + (ti as u64);
                    generate_ruler_samples(tokenizer, task, ctx_len, n_samples, seed)
                }
                RulerDataSource::Jsonl(dir) => {
                    let filename = ruler_jsonl_filename(task, ctx_len);
                    let path = dir.join(&filename);
                    print!("  Loading {} … ", path.display());
                    let mut v = load_ruler_samples_jsonl(&path)?;
                    v.truncate(n_samples);
                    v
                }
            };
            let prompt_tok_count = all_samples
                .first()
                .map(|s| {
                    tokenizer
                        .encode(s.input.as_str(), false)
                        .map(|e| e.len())
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            println!(
                "{} samples, prompt ≈ {} tok",
                all_samples.len(),
                prompt_tok_count
            );

            for (mi, &(mode, mode_label)) in cfg.modes.iter().enumerate() {
                let mode_batch = mode_batches.get(mi).copied().unwrap_or(1).max(1);
                print!("    [{mode_label}] (b={mode_batch}) ");
                #[allow(unused_imports)]
                use std::io::Write as _;
                std::io::stdout().flush().ok();

                // Run in mini-batches of `mode_batch`.
                let mut correct = 0usize;
                let t0 = std::time::Instant::now();
                let mut i = 0;
                while i < all_samples.len() {
                    let end = (i + mode_batch).min(all_samples.len());
                    let batch = &all_samples[i..end];
                    let preds = run_ruler_eval(
                        model,
                        tokenizer,
                        batch,
                        mode,
                        cfg.max_gen_tokens,
                        cfg.eos_ids,
                        model.device(),
                        None,
                    )?;
                    for (pred, sample) in preds.iter().zip(batch.iter()) {
                        if score_ruler_sample(task, pred, &sample.outputs) {
                            correct += 1;
                        }
                    }
                    i = end;
                }
                let elapsed = t0.elapsed().as_secs_f64();
                let acc = 100.0 * correct as f64 / all_samples.len() as f64;
                scores[mi][li][ti] = acc;
                println!(
                    "{correct}/{} = {acc:.1}%  ({elapsed:.1}s)",
                    all_samples.len()
                );
            }
        }
    }

    // ── Summary table ─────────────────────────────────────────────────────────
    println!("\n=== RULER Accuracy (%) — {} ===", cfg.model_name);
    let col_w = 14usize;
    let lbl_w = 26usize;

    let header_cells: Vec<String> = cfg
        .lengths_samples
        .iter()
        .flat_map(|&(ctx_len, n)| {
            cfg.tasks
                .iter()
                .map(move |&t| format!("{}@{}K(n={})", &t.name()[..4], ctx_len / 1024, n))
        })
        .collect();

    print!("{:<lbl_w$}", "Compression");
    for h in &header_cells {
        print!("{h:>col_w$}");
    }
    println!();
    println!("{}", "-".repeat(lbl_w + col_w * header_cells.len()));

    for (mi, &(_, mode_label)) in cfg.modes.iter().enumerate() {
        print!("{:<lbl_w$}", mode_label);
        for li in 0..n_lengths {
            for ti in 0..n_tasks {
                let s = scores[mi][li][ti];
                if s.is_nan() {
                    print!("{:>col_w$}", "—");
                } else {
                    print!("{s:>col_w$.1}");
                }
            }
        }
        println!();
    }

    // Per-mode averages.
    println!("{}", "-".repeat(lbl_w + col_w * header_cells.len()));
    for (mi, &(_, mode_label)) in cfg.modes.iter().enumerate() {
        let vals: Vec<f64> = {
            let mut v = Vec::new();
            for li in 0..n_lengths {
                for ti in 0..n_tasks {
                    v.push(scores[mi][li][ti]);
                }
            }
            v
        };
        let vals: Vec<f64> = vals.into_iter().filter(|s| !s.is_nan()).collect();
        let avg = if vals.is_empty() {
            f64::NAN
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        };
        print!("{:<lbl_w$}", mode_label);
        for _ in 0..header_cells.len().saturating_sub(1) {
            print!("{:>col_w$}", "");
        }
        if avg.is_nan() {
            println!("{:>col_w$}  ← avg", "—");
        } else {
            println!("{avg:>col_w$.1}  ← avg");
        }
    }

    Ok(())
}

// ── Streaming benchmark (Mode 1) ──────────────────────────────────────────────

/// Parse a KV-compression mode label (e.g. "F16", "C8", "Q8_0") into an
/// `Option<InferenceMode>` plus a canonical label string.
///
/// Returns `Ok((None, "F16"))` for F16 (lossless).
/// Returns `Err` for unrecognised labels.
pub fn parse_ruler_mode(s: &str) -> Result<(Option<InferenceMode>, String)> {
    let upper = s.to_uppercase();
    let mode = match upper.as_str() {
        "F16" | "NONE" => None,
        "Q8_0" => Some(InferenceMode::Q8_0),
        "Q8_Q4" | "Q8/Q4" => Some(InferenceMode::Q8_Q4),
        "Q4_0" => Some(InferenceMode::Q4_0),
        "Q3_0" => Some(InferenceMode::Q3_0),
        "C5" => Some(InferenceMode::C5),
        "C8" => Some(InferenceMode::C8),
        "C9" => Some(InferenceMode::C9),
        "C10" => Some(InferenceMode::C10),
        _ => return Err(candle::Error::Msg(format!("Unknown RULER mode: '{s}'"))),
    };
    Ok((mode, upper))
}
