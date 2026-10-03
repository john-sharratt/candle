use super::params::SELECT_BLOCK;
use super::profile::sampled_profile_record_duration;
use super::{
    CompressionSummary, ErrorSurface, SampleFormat, SampleSide, SampledSelectionBenchmarkResult,
};
use candle::Result;
use rayon::prelude::*;
use std::time::Instant;

pub fn cpu_parallel_kernel_map<T, R, F>(items: &[T], kernel: F) -> Vec<R>
where
    T: Copy + Sync + Send,
    R: Send,
    F: Fn(T) -> R + Send + Sync,
{
    items.par_iter().map(|item| kernel(*item)).collect()
}

pub fn cpu_parallel_kernel_range<R, F>(item_count: usize, kernel: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Send + Sync,
{
    (0..item_count).into_par_iter().map(kernel).collect()
}

pub fn select_smallest_passing(
    surface: &ErrorSurface,
    threshold: f32,
    candidates: &[SampleFormat],
    mut benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
) -> Vec<usize> {
    let start = benchmark_result.as_ref().map(|_| Instant::now());
    let mut out =
        vec![surface.n_quant.saturating_sub(1); surface.n_batch * surface.n_head * surface.n_dim];
    for b in 0..surface.n_batch {
        for h in 0..surface.n_head {
            for d in 0..surface.n_dim {
                let out_idx = ((b * surface.n_head) + h) * surface.n_dim + d;
                let mut best_idx: Option<usize> = None;
                let mut best_bpe = f32::INFINITY;
                let mut best_err = f32::INFINITY;
                let mut least_error_idx = surface.n_quant.saturating_sub(1);
                let mut least_error = f32::INFINITY;
                let mut least_error_bpe = f32::INFINITY;
                for qidx in 0..surface.n_quant {
                    let err = surface.get(b, d, qidx, h);
                    let bpe = candidates
                        .get(qidx)
                        .map_or(qidx as f32, |f| f.bits_per_elem());
                    if err < least_error || (err == least_error && bpe < least_error_bpe) {
                        least_error = err;
                        least_error_idx = qidx;
                        least_error_bpe = bpe;
                    }
                    if err <= threshold && (bpe < best_bpe || (bpe == best_bpe && err < best_err)) {
                        best_idx = Some(qidx);
                        best_bpe = bpe;
                        best_err = err;
                    }
                }
                out[out_idx] = best_idx.unwrap_or(least_error_idx);
            }
        }
    }
    if let Some(start) = start {
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            match surface.side {
                SampleSide::Key => "quantization.key.select.total",
                SampleSide::Value => "quantization.value.select.total",
            },
            start.elapsed(),
            1,
        );
    }
    out
}

fn palette4_head_bits_from_needed(
    needed: &[usize],
    candidates: &[SampleFormat],
    chunk_size: usize,
    palette_overhead_bits: f64,
) -> f64 {
    const PALETTE_SLOTS: usize = 4;

    if needed.is_empty() || candidates.is_empty() {
        return palette_overhead_bits;
    }

    let mut active = needed.to_vec();
    let mut claimed = vec![false; needed.len()];
    let mut assigned = vec![candidates.len().saturating_sub(1); needed.len()];
    let mut palette = [candidates.len().saturating_sub(1); PALETTE_SLOTS];
    let mut claimed_total = 0usize;
    let target_per_palette = needed.len() / PALETTE_SLOTS;

    for slot in 0..PALETTE_SLOTS {
        let remaining = needed.len().saturating_sub(claimed_total);
        if remaining == 0 {
            palette[slot] = if slot > 0 {
                palette[slot - 1]
            } else {
                candidates.len().saturating_sub(1)
            };
            continue;
        }

        let slot_target = if slot + 1 == PALETTE_SLOTS {
            remaining
        } else {
            target_per_palette
        };
        if slot_target == 0 {
            palette[slot] = if slot > 0 {
                palette[slot - 1]
            } else {
                candidates.len().saturating_sub(1)
            };
            continue;
        }

        let slot_idx = loop {
            let mut counts = vec![0usize; candidates.len()];
            let mut min_idx = usize::MAX;
            let mut max_idx = 0usize;

            for (i, &idx) in active.iter().enumerate() {
                if claimed[i] {
                    continue;
                }
                counts[idx] += 1;
                min_idx = min_idx.min(idx);
                max_idx = max_idx.max(idx);
            }

            if min_idx == usize::MAX {
                break if slot > 0 {
                    palette[slot - 1]
                } else {
                    candidates.len().saturating_sub(1)
                };
            }
            if slot + 1 == PALETTE_SLOTS {
                break max_idx;
            }
            if let Some(found_idx) = (min_idx..counts.len()).find(|&idx| counts[idx] >= slot_target)
            {
                break found_idx;
            }

            let promote_idx = (min_idx + 1).min(counts.len().saturating_sub(1));
            if promote_idx == min_idx {
                break max_idx;
            }
            for (i, idx) in active.iter_mut().enumerate() {
                if !claimed[i] && *idx == min_idx {
                    *idx = promote_idx;
                }
            }
        };

        palette[slot] = slot_idx;

        if slot + 1 == PALETTE_SLOTS {
            for (i, was_claimed) in claimed.iter_mut().enumerate() {
                if !*was_claimed {
                    *was_claimed = true;
                    active[i] = slot_idx;
                    assigned[i] = slot_idx;
                    claimed_total += 1;
                }
            }
            continue;
        }

        let eligible: Vec<usize> = active
            .iter()
            .enumerate()
            .filter_map(|(i, &idx)| (!claimed[i] && idx == slot_idx).then_some(i))
            .collect();

        let mut taken = 0usize;
        if eligible.len() <= slot_target {
            for &i in &eligible {
                claimed[i] = true;
                assigned[i] = slot_idx;
                taken += 1;
            }
        } else {
            for pick in 0..slot_target {
                let pos = (pick * eligible.len()).div_ceil(slot_target);
                let i = eligible[pos];
                if !claimed[i] {
                    claimed[i] = true;
                    assigned[i] = slot_idx;
                    taken += 1;
                }
            }
            for &i in &eligible {
                if taken >= slot_target {
                    break;
                }
                if !claimed[i] {
                    claimed[i] = true;
                    assigned[i] = slot_idx;
                    taken += 1;
                }
            }
        }

        let promote_idx = (slot_idx + 1).min(candidates.len().saturating_sub(1));
        if promote_idx != slot_idx {
            for (i, idx) in active.iter_mut().enumerate() {
                if !claimed[i] && *idx == slot_idx {
                    *idx = promote_idx;
                }
            }
        }
        claimed_total += taken;
    }

    let mut head_bits = palette_overhead_bits;
    for &chosen in &assigned {
        head_bits += candidates[chosen].bits_per_elem() as f64 * chunk_size as f64;
    }
    head_bits
}

pub fn model_compression_from_surface(
    surface: &ErrorSurface,
    winners: &[usize],
    candidates: &[SampleFormat],
    mut benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
) -> Result<CompressionSummary> {
    let start = benchmark_result.as_ref().map(|_| Instant::now());
    let expected = surface.n_batch * surface.n_head * surface.n_dim;
    if winners.len() != expected {
        candle::bail!(
            "winners length mismatch: got {}, expected {}",
            winners.len(),
            expected
        );
    }
    if candidates.len() != surface.n_quant {
        candle::bail!(
            "candidate count mismatch: got {}, expected {}",
            candidates.len(),
            surface.n_quant
        );
    }

    let mut ideal_bits = 0.0f64;
    let mut head_bits = 0.0f64;
    let mut pal4_bits = 0.0f64;
    let elems_per_head = (surface.n_dim * surface.chunk_size) as f64;
    let palette_overhead_bits = (surface.n_dim * 2 + 4 * 8) as f64;

    for b in 0..surface.n_batch {
        for h in 0..surface.n_head {
            let mut head_worst = 0usize;
            let mut freq = vec![0usize; surface.n_quant];
            for d in 0..surface.n_dim {
                let idx = ((b * surface.n_head) + h) * surface.n_dim + d;
                let win = winners[idx].min(surface.n_quant - 1);
                freq[win] += 1;
                if win > head_worst {
                    head_worst = win;
                }
                ideal_bits += candidates[win].bits_per_elem() as f64 * surface.chunk_size as f64;
            }
            head_bits += candidates[head_worst].bits_per_elem() as f64 * elems_per_head;

            let needed: Vec<usize> = (0..surface.n_dim)
                .map(|d| {
                    let idx = ((b * surface.n_head) + h) * surface.n_dim + d;
                    winners[idx].min(surface.n_quant - 1)
                })
                .collect();
            pal4_bits += palette4_head_bits_from_needed(
                &needed,
                candidates,
                surface.chunk_size,
                palette_overhead_bits,
            );
        }
    }

    let total_elems =
        (surface.n_batch * surface.n_head * surface.n_dim * surface.chunk_size) as f64;
    let ideal_bpe = if total_elems > 0.0 {
        ideal_bits / total_elems
    } else {
        16.0
    };
    let head_bpe = if total_elems > 0.0 {
        head_bits / total_elems
    } else {
        16.0
    };
    let palette4_bpe = if total_elems > 0.0 {
        pal4_bits / total_elems
    } else {
        16.0
    };

    if let Some(start) = start {
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            match surface.side {
                SampleSide::Key => "quantization.key.summary.total",
                SampleSide::Value => "quantization.value.summary.total",
            },
            start.elapsed(),
            1,
        );
    }
    Ok(CompressionSummary {
        ideal_bpe,
        head_bpe,
        palette4_bpe,
        ideal_cr: 16.0 / ideal_bpe.max(1e-9),
        head_cr: 16.0 / head_bpe.max(1e-9),
        palette4_cr: 16.0 / palette4_bpe.max(1e-9),
    })
}

/// Fused sweep: select winners and compute compression summaries for all thresholds in one
/// pass over the error surface instead of N separate `select_smallest_passing` +
/// `model_compression_from_surface` call pairs.
///
/// Returns one `(winners, CompressionSummary)` per threshold, in input order.
/// Summarise pre-computed GPU winner indices into `CompressionSummary` values.
///
/// `winners` has layout `[n_thresholds × n_cells]` where
/// `n_cells = n_batch × n_head × n_dim` and each entry is a `u8` candidate index.
/// This is the output of the GPU `select_winners_kv_paged` kernel.
///
/// All thresholds are processed in **parallel** via rayon.  Each threshold works on
/// a disjoint slice of `winners` and independent accumulators, so there is no
/// contention.  `benchmark_result` is ignored (pass `None`; timings should be
/// recorded by the caller around the whole call).
pub fn batch_summarize_from_winners(
    winners: &[u8],
    n_thresholds: usize,
    n_batch: usize,
    n_head: usize,
    n_dim: usize,
    chunk_size: usize,
    candidates: &[SampleFormat],
    side: SampleSide,
) -> Result<Vec<(Vec<u8>, CompressionSummary)>> {
    let n_q = candidates.len();
    let n_bh = n_batch * n_head;
    let n_cells = n_bh * n_dim;
    let total_elems = (n_bh * n_dim * chunk_size) as f64;
    let elems_per_head = (n_dim * chunk_size) as f64;
    let palette_overhead_bits = (n_dim * 2 + 4 * 8) as f64;
    let _ = side; // kept for caller symmetry / future profiling

    if n_thresholds == 0 || n_cells == 0 {
        return Ok(Vec::new());
    }
    debug_assert_eq!(
        winners.len(),
        n_thresholds * n_cells,
        "batch_summarize_from_winners: winners length mismatch"
    );

    let results: Vec<(Vec<u8>, CompressionSummary)> = (0..n_thresholds)
        .into_par_iter()
        .map(|t| {
            let slice = &winners[t * n_cells..(t + 1) * n_cells];

            // Accumulate freq, head_worst, ideal_bits from the pre-selected winners.
            // u32 is sufficient for freq (max value = n_dim ≤ a few thousand).
            // u8 is sufficient for head_worst (indexes into candidates, max ~8).
            let mut freq = vec![0u32; n_bh * n_q];
            let mut head_worst = vec![0u8; n_bh];
            let mut ideal_bits = 0.0f64;

            for bh in 0..n_bh {
                for d in 0..n_dim {
                    let cell = bh * n_dim + d;
                    let chosen = (slice[cell] as usize).min(n_q.saturating_sub(1));
                    freq[bh * n_q + chosen] += 1;
                    if chosen > head_worst[bh] as usize {
                        head_worst[bh] = chosen as u8;
                    }
                    ideal_bits += candidates[chosen].bits_per_elem() as f64 * chunk_size as f64;
                }
            }

            // Phase 2: per-head palette construction (identical to batch_select_and_summarize).
            let mut head_bits = 0.0f64;
            let mut pal4_bits = 0.0f64;

            for bh in 0..n_bh {
                head_bits +=
                    candidates[head_worst[bh] as usize].bits_per_elem() as f64 * elems_per_head;

                let needed: Vec<usize> = (0..n_dim)
                    .map(|d| {
                        let cell = bh * n_dim + d;
                        (slice[cell] as usize).min(n_q.saturating_sub(1))
                    })
                    .collect();
                pal4_bits += palette4_head_bits_from_needed(
                    &needed,
                    candidates,
                    chunk_size,
                    palette_overhead_bits,
                );
            }

            let ideal_bpe = if total_elems > 0.0 {
                ideal_bits / total_elems
            } else {
                16.0
            };
            let head_bpe = if total_elems > 0.0 {
                head_bits / total_elems
            } else {
                16.0
            };
            let palette4_bpe = if total_elems > 0.0 {
                pal4_bits / total_elems
            } else {
                16.0
            };

            // Return the raw u8 slice directly — no widening needed.
            (
                slice.to_vec(),
                CompressionSummary {
                    ideal_bpe,
                    head_bpe,
                    palette4_bpe,
                    ideal_cr: 16.0 / ideal_bpe.max(1e-9),
                    head_cr: 16.0 / head_bpe.max(1e-9),
                    palette4_cr: 16.0 / palette4_bpe.max(1e-9),
                },
            )
        })
        .collect();

    Ok(results)
}

pub fn batch_select_and_summarize(
    surface: &ErrorSurface,
    thresholds: &[f32],
    candidates: &[SampleFormat],
    mut benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
) -> Result<Vec<(Vec<u8>, CompressionSummary)>> {
    let n_t = thresholds.len();
    let n_bh = surface.n_batch * surface.n_head;
    let n_bhd = n_bh * surface.n_dim;
    let n_q = surface.n_quant;

    // ── Phase 1: single surface scan ──────────────────────────────────────
    // For each (b, h, d), load the n_quant costs once into a local buffer, then
    // compute the winner for every threshold cheaply from that buffer.
    // u8 for all_winners/head_worst (indices into candidates, ≤ 255).
    // u32 for freq (count per head per quant level, ≤ n_dim ≤ a few thousand).
    let select_start = benchmark_result.as_ref().map(|_| Instant::now());

    let mut all_winners: Vec<Vec<u8>> = vec![vec![n_q.saturating_sub(1) as u8; n_bhd]; n_t];
    let mut head_worst: Vec<Vec<u8>> = vec![vec![0u8; n_bh]; n_t];
    // freq[t][bh * n_q + q] — linearised to avoid Vec<Vec<Vec>>
    let mut freq: Vec<u32> = vec![0u32; n_t * n_bh * n_q];
    let mut ideal_bits: Vec<f64> = vec![0.0f64; n_t];

    let mut costs = vec![0.0f32; n_q]; // reused scratch buffer
    for b in 0..surface.n_batch {
        for h in 0..surface.n_head {
            let bh = b * surface.n_head + h;
            for d in 0..surface.n_dim {
                // Load all quant costs for this (b, h, d) once.
                let base = ((b * surface.n_dim + d) * n_q) * surface.n_head + h;
                for (q, cost) in costs.iter_mut().enumerate().take(n_q) {
                    *cost = surface.data[base + q * surface.n_head];
                }

                let out_idx = bh * surface.n_dim + d;
                for t in 0..n_t {
                    let threshold = thresholds[t];
                    let mut chosen = n_q.saturating_sub(1);
                    let mut best_bpe = f32::INFINITY;
                    let mut best_err = f32::INFINITY;
                    let mut least_error_idx = chosen;
                    let mut least_error = f32::INFINITY;
                    let mut least_error_bpe = f32::INFINITY;
                    for (q, &cost) in costs.iter().enumerate() {
                        let bpe = candidates[q].bits_per_elem();
                        if cost < least_error || (cost == least_error && bpe < least_error_bpe) {
                            least_error = cost;
                            least_error_idx = q;
                            least_error_bpe = bpe;
                        }
                        if cost <= threshold
                            && (bpe < best_bpe || (bpe == best_bpe && cost < best_err))
                        {
                            chosen = q;
                            best_bpe = bpe;
                            best_err = cost;
                        }
                    }
                    if !best_err.is_finite() {
                        chosen = least_error_idx;
                    }
                    all_winners[t][out_idx] = chosen as u8;
                    freq[t * n_bh * n_q + bh * n_q + chosen] += 1;
                    if chosen > head_worst[t][bh] as usize {
                        head_worst[t][bh] = chosen as u8;
                    }
                    ideal_bits[t] +=
                        candidates[chosen].bits_per_elem() as f64 * surface.chunk_size as f64;
                }
            }
        }
    }

    if let Some(start) = select_start {
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            match surface.side {
                SampleSide::Key => "quantization.key.select.total",
                SampleSide::Value => "quantization.value.select.total",
            },
            start.elapsed(),
            1,
        );
    }

    // ── Phase 2: per-threshold palette summarisation ─────────────────────
    let summary_start = benchmark_result.as_ref().map(|_| Instant::now());

    let elems_per_head = (surface.n_dim * surface.chunk_size) as f64;
    let palette_overhead_bits = (surface.n_dim * 2 + 4 * 8) as f64;
    let total_elems =
        (surface.n_batch * surface.n_head * surface.n_dim * surface.chunk_size) as f64;

    let mut results = Vec::with_capacity(n_t);
    for t in 0..n_t {
        let mut head_bits = 0.0f64;
        let mut pal4_bits = 0.0f64;

        for b in 0..surface.n_batch {
            for h in 0..surface.n_head {
                let bh = b * surface.n_head + h;
                head_bits +=
                    candidates[head_worst[t][bh] as usize].bits_per_elem() as f64 * elems_per_head;

                let needed: Vec<usize> = (0..surface.n_dim)
                    .map(|d| {
                        let idx = bh * surface.n_dim + d;
                        all_winners[t][idx] as usize
                    })
                    .collect();
                pal4_bits += palette4_head_bits_from_needed(
                    &needed,
                    candidates,
                    surface.chunk_size,
                    palette_overhead_bits,
                );
            }
        }

        let ideal_bpe = if total_elems > 0.0 {
            ideal_bits[t] / total_elems
        } else {
            16.0
        };
        let head_bpe = if total_elems > 0.0 {
            head_bits / total_elems
        } else {
            16.0
        };
        let palette4_bpe = if total_elems > 0.0 {
            pal4_bits / total_elems
        } else {
            16.0
        };
        results.push((
            std::mem::take(&mut all_winners[t]),
            CompressionSummary {
                ideal_bpe,
                head_bpe,
                palette4_bpe,
                ideal_cr: 16.0 / ideal_bpe.max(1e-9),
                head_cr: 16.0 / head_bpe.max(1e-9),
                palette4_cr: 16.0 / palette4_bpe.max(1e-9),
            },
        ));
    }

    if let Some(start) = summary_start {
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            match surface.side {
                SampleSide::Key => "quantization.key.summary.total",
                SampleSide::Value => "quantization.value.summary.total",
            },
            start.elapsed(),
            1,
        );
    }

    Ok(results)
}

/// CPU mirror of the palette4 slot-grouping logic in `select_kv_format_palette4_paged`.
///
/// Takes per-block format selections (flat layout: `n_heads × blocks_per_head`)
/// and applies the same quota-based 4-slot palette reduction: walk unclaimed
/// blocks from lowest BPE upward, pick the first tier that fills the slot quota,
/// then claim those blocks to that slot. Returns `(bpe_per_element,
/// effective_format_per_block)` where every block carries the format of the
/// palette slot it was assigned to — identical to what `effective_block_tags`
/// holds after the GPU kernel runs.
pub fn cpu_palette4_reduce(
    fmts: &[SampleFormat],
    blocks_per_head: usize,
) -> (f64, Vec<SampleFormat>) {
    let elems_per_block = SELECT_BLOCK as f64;
    let palette_overhead_bits = (blocks_per_head * 2 + 4 * 8) as f64;
    let max_quant_ti = SampleFormat::Q8KS.table_index();
    let num_heads = fmts.len().checked_div(blocks_per_head).unwrap_or(0);

    let mut total_bits = 0.0f64;
    let mut total_elems = 0.0f64;
    let mut effective = Vec::with_capacity(fmts.len());

    for head in 0..num_heads {
        let base = head * blocks_per_head;
        let head_fmts = &fmts[base..base + blocks_per_head];
        let mut active_ti: Vec<usize> = head_fmts.iter().map(|f| f.table_index()).collect();
        // Promotion targets must be ti's that are present in this head's input
        // (i.e. formats the upstream kernel actually emits, which is the
        // candidate ladder).  Walking ti+1 one slot at a time would deposit
        // blocks at intermediate slots like Q5_0/Q5_1/Q8_1 that aren't in
        // either side's ladder, which then disappear from the per-side report.
        let valid_ti_sorted: Vec<usize> = {
            let mut s = active_ti.clone();
            s.sort_unstable();
            s.dedup();
            s
        };
        let next_valid_ti = |t: usize| -> usize {
            valid_ti_sorted
                .iter()
                .copied()
                .find(|&v| v > t)
                .unwrap_or(t)
        };
        let mut claimed = vec![false; blocks_per_head];
        let mut assigned_ti = vec![max_quant_ti; blocks_per_head];
        let target_per_palette = blocks_per_head / 4;
        let mut claimed_total = 0usize;
        let mut worst_ti = max_quant_ti;

        for slot in 0..4usize {
            let remaining = blocks_per_head.saturating_sub(claimed_total);
            if remaining == 0 {
                continue;
            }
            let slot_target = if slot == 3 {
                remaining
            } else {
                target_per_palette
            };

            let slot_ti = loop {
                let mut counts = [0usize; 23];
                let mut min_ti = usize::MAX;
                let mut max_ti = 0usize;
                for (i, &ti) in active_ti.iter().enumerate() {
                    if claimed[i] {
                        continue;
                    }
                    counts[ti] += 1;
                    min_ti = min_ti.min(ti);
                    max_ti = max_ti.max(ti);
                }
                if min_ti == usize::MAX {
                    break worst_ti;
                }
                if slot == 3 {
                    break max_ti;
                }
                if let Some(found_ti) = (min_ti..counts.len()).find(|&ti| counts[ti] >= slot_target)
                {
                    break found_ti;
                }
                let promote_ti = next_valid_ti(min_ti);
                if promote_ti == min_ti || promote_ti > max_quant_ti {
                    break max_ti;
                }
                for (i, ti) in active_ti.iter_mut().enumerate() {
                    if !claimed[i] && *ti == min_ti {
                        *ti = promote_ti;
                    }
                }
            };

            worst_ti = worst_ti.max(slot_ti);

            if slot == 3 {
                for (i, was_claimed) in claimed.iter_mut().enumerate() {
                    if !*was_claimed {
                        *was_claimed = true;
                        assigned_ti[i] = slot_ti;
                        claimed_total += 1;
                    }
                }
                continue;
            }

            let eligible: Vec<usize> = active_ti
                .iter()
                .enumerate()
                .filter_map(|(i, &ti)| (!claimed[i] && ti == slot_ti).then_some(i))
                .collect();
            let mut taken = 0usize;
            if eligible.len() <= slot_target {
                for &i in &eligible {
                    claimed[i] = true;
                    assigned_ti[i] = slot_ti;
                    taken += 1;
                }
            } else {
                for pick in 0..slot_target {
                    let pos = (pick * eligible.len()).div_ceil(slot_target);
                    let i = eligible[pos];
                    if !claimed[i] {
                        claimed[i] = true;
                        assigned_ti[i] = slot_ti;
                        taken += 1;
                    }
                }
                for &i in &eligible {
                    if taken >= slot_target {
                        break;
                    }
                    if !claimed[i] {
                        claimed[i] = true;
                        assigned_ti[i] = slot_ti;
                        taken += 1;
                    }
                }
            }
            let promote_ti = next_valid_ti(slot_ti);
            if promote_ti != slot_ti && promote_ti <= max_quant_ti {
                for (i, ti) in active_ti.iter_mut().enumerate() {
                    if !claimed[i] && *ti == slot_ti {
                        *ti = promote_ti;
                    }
                }
            }
            claimed_total += taken;
        }

        let mut head_bits = palette_overhead_bits;
        for &ti in &assigned_ti {
            let fmt = SampleFormat::from_table_index(ti);
            head_bits += fmt.bits_per_elem() as f64 * elems_per_block;
            total_elems += elems_per_block;
            effective.push(fmt);
        }
        total_bits += head_bits;
    }

    let bpe = if total_elems > 0.0 {
        total_bits / total_elems
    } else {
        16.0
    };
    (bpe, effective)
}

pub(super) fn compute_head_amax(values: &[f32]) -> f32 {
    values.iter().copied().map(f32::abs).fold(0.0f32, f32::max)
}

pub(super) fn safe_head_scale(amax: f32) -> f32 {
    if amax < 1.0e-8 {
        1.0
    } else {
        amax
    }
}

fn max_abs_error(orig: &[f32; SELECT_BLOCK], recon: &[f32; SELECT_BLOCK]) -> f32 {
    orig.iter()
        .zip(recon.iter())
        .map(|(x, xr)| (x - xr).abs())
        .fold(0.0f32, f32::max)
}

pub(super) fn normalised_error(
    orig: &[f32; SELECT_BLOCK],
    recon: &[f32; SELECT_BLOCK],
    head_scale: f32,
) -> f32 {
    max_abs_error(orig, recon) / head_scale
}

pub(super) fn round_trip_bf16(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let bits = x.to_bits();
        let bias = 0x7FFFu32 + ((bits >> 16) & 1);
        out[i] = f32::from_bits(bits.wrapping_add(bias) & 0xFFFF_0000);
    }
    out
}
