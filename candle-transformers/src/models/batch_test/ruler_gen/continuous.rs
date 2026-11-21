//! Runs the RULER benchmark as an open-ended loop that appends one result line per
//! sample to a log, for a long-running accuracy record.
//!
//! Each (task, context length) pair is a slot, and the slots are cycled so every
//! task gets fair coverage: among the slots that are ready, one is picked at random.
//! A slot that fails, for example by running out of memory, backs off exponentially
//! up to thirty seconds and retries, while the other slots keep running, and a
//! success resets its backoff. A line is written and flushed for every sample, so
//! the log survives the process being killed. The report reads that log back and
//! tabulates scored results by quantisation mode, context length and task, counting
//! errored samples separately from scored ones.

use super::*;

/// Continuous append-mode RULER benchmark (Mode 1).
///
/// Runs until the process is killed.  After every batch one JSONL line per
/// sample is appended to `log_path` and the file is flushed.
///
/// Each output line:
/// ```json
/// {"quant":"C8","ctx":4096,"task":"niah_single_1","sample":7,"correct":true,"elapsed":1.23}
/// ```
///
/// Slots are (task × ctx_len) pairs cycled in a flat round-robin — each slot
/// advances independently.  If a slot fails (e.g. OOM) it backs off
/// exponentially (1 → 2 → 4 → … → 30 s) before retrying, while other slots
/// continue uninterrupted.
pub fn run_ruler_continuous<M: ManagedBatchedModel>(
    model: &M,
    tokenizer: &Tokenizer,
    mode: Option<InferenceMode>,
    mode_label: &str,
    log_path: &std::path::Path,
    tasks: &[RulerTask],
    ctx_config: &[(usize, usize)], // (ctx_len, n_concurrent)
    max_gen_tokens: usize,
    eos_ids: &[u32],
) -> Result<()> {
    use std::io::Write as _;

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| candle::Error::Msg(format!("open {:?}: {e}", log_path)))?;
    let mut writer = std::io::BufWriter::new(file);

    println!(
        "\n=== RULER stream: mode={mode_label}  log={} ===",
        log_path.display()
    );
    println!(
        "Ctx batches: {}",
        ctx_config
            .iter()
            .map(|(c, n)| format!("{}K×{n}", c / 1024))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Tasks: {}",
        tasks
            .iter()
            .map(|t| t.name())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("Press Ctrl+C to stop — results flushed after each batch.\n");

    // Per-slot state
    struct Slot {
        task: RulerTask,
        ctx_len: usize,
        n_concurrent: usize,
        batch_idx: u64, // monotonic; used as seed and for sample numbering
        backoff_secs: f64,
        next_run_at: std::time::Instant, // schedule: run the most-overdue slot first
    }

    let now = std::time::Instant::now();
    let mut slots: Vec<Slot> = tasks
        .iter()
        .flat_map(|&task| {
            ctx_config.iter().map(move |&(ctx_len, n_concurrent)| Slot {
                task,
                ctx_len,
                n_concurrent,
                batch_idx: 0,
                backoff_secs: 0.0,
                next_run_at: now,
            })
        })
        .collect();

    let device = model.device();

    // Simple xorshift64 RNG — no external dep needed.
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    let mut rng: u64 = if seed == 0 { 0xdeadbeefcafe1234 } else { seed };
    let mut xorshift = move || -> u64 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    loop {
        // Among all slots that are "ready" (have the minimum next_run_at),
        // pick one at random so every task type gets fair coverage.
        let min_t = slots.iter().map(|s| s.next_run_at).min().unwrap();
        let candidates: Vec<usize> = slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.next_run_at == min_t)
            .map(|(i, _)| i)
            .collect();
        let slot_idx = candidates[xorshift() as usize % candidates.len()];
        let slot = &mut slots[slot_idx];

        // Exponential backoff: sleep if the slot failed last time.
        if slot.backoff_secs > 0.0 {
            let sleep_ms = (slot.backoff_secs * 1000.0) as u64;
            std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
        }

        let seed = slot.batch_idx;
        let sample_base = slot.batch_idx * slot.n_concurrent as u64;
        let samples =
            generate_ruler_samples(tokenizer, slot.task, slot.ctx_len, slot.n_concurrent, seed);

        let t0 = std::time::Instant::now();
        let preds = match run_ruler_eval(
            model,
            tokenizer,
            &samples,
            mode,
            max_gen_tokens,
            eos_ids,
            device,
            None,
        ) {
            Ok(p) => {
                // Reset backoff on success.
                slot.backoff_secs = 0.0;
                p
            }
            Err(e) => {
                // Exponential backoff: 1 → 2 → 4 → … capped at 30 s.
                slot.backoff_secs = if slot.backoff_secs < 1.0 {
                    1.0
                } else {
                    (slot.backoff_secs * 2.0).min(30.0)
                };
                eprintln!(
                    "[WARN] batch failed ({mode_label} {}K {}) — backoff {:.0}s: {e}",
                    slot.ctx_len / 1024,
                    slot.task.name(),
                    slot.backoff_secs,
                );
                // On error: push next_run_at forward by backoff so we don't spin.
                slot.next_run_at = std::time::Instant::now()
                    + std::time::Duration::from_secs_f64(slot.backoff_secs);
                continue;
            }
        };
        let elapsed = t0.elapsed().as_secs_f64();
        let per_sample = elapsed / slot.n_concurrent.max(1) as f64;

        // Schedule next run for this slot at "now" so it is eligible
        // immediately. Slower slots will accumulate larger next_run_at
        // values naturally, letting faster slots run more often.
        slot.next_run_at = std::time::Instant::now();

        let mut batch_correct = 0usize;
        for (i, (pred, sample)) in preds.iter().zip(samples.iter()).enumerate() {
            let correct = score_ruler_sample(slot.task, pred, &sample.outputs);
            if correct {
                batch_correct += 1;
            }
            let line = format!(
                    "{{\"quant\":\"{mode_label}\",\"ctx\":{},\"task\":\"{}\",\"sample\":{},\"correct\":{correct},\"elapsed\":{per_sample:.3}}}\n",
                    slot.ctx_len,
                    slot.task.name(),
                    sample_base + i as u64,
                );
            writer
                .write_all(line.as_bytes())
                .map_err(|e| candle::Error::Msg(e.to_string()))?;
        }
        writer
            .flush()
            .map_err(|e| candle::Error::Msg(e.to_string()))?;

        println!(
            "[{mode_label} {}K {}  batch={}]  {batch_correct}/{} correct  {elapsed:.1}s",
            slot.ctx_len / 1024,
            slot.task.name(),
            slot.batch_idx,
            slot.n_concurrent,
        );

        slot.batch_idx += 1;
    }
}

// ── Report (Mode 2) ────────────────────────────────────────────────────────────

/// Read a JSONL log produced by [`run_ruler_continuous`] and print an accuracy
/// table grouped by (quant × ctx_len × task).
pub fn print_ruler_report(log_path: &std::path::Path) -> Result<()> {
    use std::collections::{BTreeMap, BTreeSet};

    let text = std::fs::read_to_string(log_path)
        .map_err(|e| candle::Error::Msg(format!("read {:?}: {e}", log_path)))?;

    // (quant, ctx_len, task) → (correct, scored_total, errored)
    // scored_total = success + failed (excludes errored)
    let mut data: BTreeMap<(String, u64, String), (usize, usize, usize)> = BTreeMap::new();
    let mut skipped = 0usize;
    let mut total_errors = 0usize;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Prefer new "status" field; fall back to legacy "correct" bool.
        let status = extract_json_str(line, "status");
        let legacy_correct = extract_json_bool(line, "correct");
        let resolved: Option<&'static str> = match (status.as_deref(), legacy_correct) {
            (Some("success"), _) => Some("success"),
            (Some("failed"), _) => Some("failed"),
            (Some("error"), _) => Some("error"),
            (None, Some(true)) => Some("success"),
            (None, Some(false)) => Some("failed"),
            _ => None,
        };
        match (
            extract_json_str(line, "quant"),
            extract_json_u64(line, "ctx"),
            extract_json_str(line, "task"),
            resolved,
        ) {
            (Some(q), Some(c), Some(t), Some(st)) => {
                let e = data.entry((q, c, t)).or_insert((0, 0, 0));
                match st {
                    "success" => {
                        e.0 += 1;
                        e.1 += 1;
                    }
                    "failed" => {
                        e.1 += 1;
                    }
                    "error" => {
                        e.2 += 1;
                        total_errors += 1;
                    }
                    _ => {}
                }
            }
            _ => skipped += 1,
        }
    }

    if data.is_empty() {
        println!("No data found in {:?}", log_path);
        return Ok(());
    }
    if skipped > 0 {
        println!("(skipped {skipped} malformed lines)");
    }
    if total_errors > 0 {
        println!("(excluded {total_errors} errored runs from accuracy totals)");
    }

    let quants: Vec<String> = data
        .keys()
        .map(|(q, _, _)| q.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let ctxs: Vec<u64> = data
        .keys()
        .map(|(_, c, _)| *c)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let task_names: Vec<String> = data
        .keys()
        .map(|(_, _, t)| t.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let col_w = 12usize;
    let lbl_w = 10usize;
    let n_cols = ctxs.len() * task_names.len();

    println!("\n=== RULER Report: {} ===", log_path.display());
    print!("{:<lbl_w$}", "Mode");
    for ctx in &ctxs {
        for task in &task_names {
            let abbr = &task[..task.len().min(4)];
            print!("{:>col_w$}", format!("{}@{}K", abbr, ctx / 1024));
        }
    }
    println!("  {:>8}  {:>6}", "Avg%", "Err");
    println!("{}", "─".repeat(lbl_w + col_w * n_cols + 18));

    for quant in &quants {
        print!("{:<lbl_w$}", quant);
        let (mut tot_c, mut tot_n, mut tot_err) = (0usize, 0usize, 0usize);
        for ctx in &ctxs {
            for task in &task_names {
                match data.get(&(quant.clone(), *ctx, task.clone())) {
                    Some((c, n, err)) => {
                        tot_c += c;
                        tot_n += n;
                        tot_err += err;
                        let cell = if *n > 0 {
                            format!("{:.0}%({})", 100.0 * *c as f64 / *n as f64, n)
                        } else {
                            // No scored runs — only errors (or nothing); show err count.
                            format!("E{}", err)
                        };
                        print!("{:>col_w$}", cell);
                    }
                    None => print!("{:>col_w$}", "—"),
                }
            }
        }
        let avg = if tot_n > 0 {
            format!("{:.1}", 100.0 * tot_c as f64 / tot_n as f64)
        } else {
            "—".to_string()
        };
        println!("  {:>8}  {:>6}", avg, tot_err);
    }

    Ok(())
}
