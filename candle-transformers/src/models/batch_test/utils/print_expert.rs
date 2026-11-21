//! Prints the expert-cache tables that explain how a streamed model's weights
//! were resident and what it missed.
//!
//! Each phase gets its own pipeline table. Its rows split misses by where they
//! were found, show whether the promotion ring kept pace with the GPU, and
//! compare the predictors with what an eviction policy could reach. The same
//! routing is replayed at the zone's capacity under plain LRU and under Belady's
//! MIN, so the gap to the best possible eviction belongs to the policy in use.
//! Blend cells, victim tags and an embedding tier's row cache are printed beside
//! the pipeline, so a miss can be read against what caused it.

use super::*;

impl TestParams {
    /// The prediction blend's learned cells (`expert_lre::blend`), as the last
    /// config left them — the learner runs across the whole gate — one line per
    /// hop: each judged cell's measured probability and its candidate count,
    /// most probable first.
    pub(super) fn print_blend_cells(results: &[TestResults]) {
        let Some(s) = results.iter().rev().find_map(|r| r.expert_stats.as_ref()) else {
            return;
        };
        let cells = &s.cells;
        if cells.preds.iter().all(|h| h.iter().all(|&n| n == 0)) {
            return;
        }
        println!("\n=== Prediction cells (v votes 0-3, m Markov 0-3, r recent) ===");
        for hop in 1..=cells.preds.len() {
            let mut judged: Vec<(f64, usize, u32)> = (0..CELLS)
                .filter(|&c| cells.preds[hop - 1][c] > 0)
                .map(|c| (cells.probability(hop, c), c, cells.preds[hop - 1][c]))
                .collect();
            judged.sort_by(|a, b| b.0.total_cmp(&a.0));
            let line: Vec<String> = judged
                .iter()
                .map(|&(p, c, n)| format!("{} {:.0}%/{n}", cell_label(c), 100.0 * p))
                .collect();
            println!("  hop {hop}: {}", line.join("  "));
        }
    }

    /// Decode's evicted victims by what was known of them when evicted
    /// (`expert_lre::regret`), one line per config: each tag's share routed
    /// again at its row's next visit and its count, most frequent first.
    pub(super) fn print_victim_tags(results: &[TestResults]) {
        let any = results.iter().any(|r| {
            r.expert_stats
                .as_ref()
                .is_some_and(|s| s.victim_tag_evicted.iter().any(|&n| n > 0))
        });
        if !any {
            return;
        }
        println!(
            "\n=== Decode victims by tag (p prediction band, hit on last visit, place, reload): regret%/n ==="
        );
        for (i, r) in results.iter().enumerate() {
            let Some(s) = r.expert_stats.as_ref() else {
                continue;
            };
            let mut tags: Vec<(usize, usize)> = (0..TAGS)
                .filter(|&t| s.victim_tag_evicted[t] > 0)
                .map(|t| (t, s.victim_tag_evicted[t]))
                .collect();
            tags.sort_by_key(|&(_, n)| Reverse(n));
            let line: Vec<String> = tags
                .iter()
                .map(|&(t, n)| {
                    format!(
                        "{} {:.0}%/{n}",
                        tag_label(t),
                        PipelineStats::percent(s.victim_tag_regretted[t], n)
                    )
                })
                .collect();
            println!(
                "  #{} {:?}×{}: {}",
                i + 1,
                r.config.mode,
                r.config.num_contexts,
                line.join("  ")
            );
        }
    }

    /// One phase's expert pipeline table: rows = metrics, columns = configs.
    pub(super) fn print_expert_phase_table(
        phase: &str,
        results: &[TestResults],
        pick: impl Fn(&TestResults) -> Option<&PipelineStats>,
    ) {
        // Build column headers from config descriptions.
        let headers: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("#{} {:?}×{}", i + 1, r.config.mode, r.config.num_contexts))
            .collect();

        // Determine column width (minimum 12 to fit numbers).
        let col_w = headers.iter().map(|h| h.len()).max().unwrap_or(10).max(12);

        const GIB: f64 = (1u64 << 30) as f64;
        // Metric rows: (label, extractor returning formatted string).
        let metrics: Vec<(&str, Box<dyn Fn(&PipelineStats) -> String>)> = vec![
            (
                "Expert hits",
                Box::new(|s: &PipelineStats| format!("{}", s.expert_hits)),
            ),
            (
                "Expert misses",
                Box::new(|s: &PipelineStats| format!("{}", s.expert_misses)),
            ),
            (
                "Hit rate",
                Box::new(|s: &PipelineStats| format!("{:.1}%", s.hit_rate())),
            ),
            // The same routing replayed at the zone's capacity: under plain
            // LRU, and under what no eviction policy could beat (Belady's
            // MIN) — the gap to MIN is the policy's.
            (
                "  LRU replay",
                Box::new(|s: &PipelineStats| format!("{:.1}%", s.hit_lru)),
            ),
            (
                "  ceiling (MIN)",
                Box::new(|s: &PipelineStats| format!("{:.1}%", s.hit_ceiling)),
            ),
            // Misses split by where bucketize found them: pinned (a warm or pad
            // slot — computed by the GEMM workers at once) or cold (waited on
            // until the stager staged them).
            (
                "  pinned misses",
                Box::new(|s: &PipelineStats| format!("{}", s.worker_pinned)),
            ),
            (
                "  cold misses",
                Box::new(|s: &PipelineStats| format!("{}", s.worker_cold)),
            ),
            (
                "Worker promotions",
                Box::new(|s: &PipelineStats| format!("{}", s.worker_promotions)),
            ),
            // Whether the promotion ring kept up with the GPU: the slots bucketize
            // took from it (of them the lazy victims it evicted, and the ones it
            // passed over for its own launch), the misses that took none —
            // computed from scratch and claimed on their next launch, the decode
            // rows' counted apart — and how many invocations the device had
            // begun past the one the pipeline thread was serving, per routed
            // layer.
            (
                "Ring slots taken",
                Box::new(|s: &PipelineStats| format!("{}", s.ring_taken)),
            ),
            (
                "  victims claimed",
                Box::new(|s: &PipelineStats| format!("{}", s.victims_claimed)),
            ),
            (
                "  victims skipped",
                Box::new(|s: &PipelineStats| format!("{}", s.victims_skipped)),
            ),
            (
                "Misses unslotted",
                Box::new(|s: &PipelineStats| format!("{}", s.ring_unslotted)),
            ),
            (
                "  decode unslotted",
                Box::new(|s: &PipelineStats| format!("{}", s.decode_unslotted)),
            ),
            (
                "Pipeline lag (inv.)",
                Box::new(|s: &PipelineStats| {
                    if s.routed_messages == 0 {
                        "-".to_string()
                    } else {
                        format!("{:.2}", s.pipeline_lag as f64 / s.routed_messages as f64)
                    }
                }),
            ),
            // The two gauges that turn the miss counts into a tier decomposition:
            // how much of the model is resident in VRAM, and how much of the KV
            // side's ground the weight zone is still willing to concede.
            (
                "Hot VRAM (resident)",
                Box::new(|s: &PipelineStats| format!("{} MiB", s.resident_vram_bytes >> 20)),
            ),
            (
                "Zone cedeable",
                Box::new(|s: &PipelineStats| format!("{} MiB", s.zone_cedeable_bytes >> 20)),
            ),
            (
                "Warm tier slots",
                Box::new(|s: &PipelineStats| {
                    if s.total_experts == 0 {
                        format!("{}", s.warm_slots)
                    } else {
                        format!(
                            "{} ({:.0}%)",
                            s.warm_slots,
                            100.0 * s.warm_slots as f64 / s.total_experts as f64
                        )
                    }
                }),
            ),
            (
                "  of which pageable",
                Box::new(|s: &PipelineStats| format!("{}", s.warm_paged_slots)),
            ),
            (
                "Pad slots",
                Box::new(|s: &PipelineStats| format!("{}", s.pad_slots)),
            ),
            (
                "Staged (cold)",
                Box::new(|s: &PipelineStats| format!("{}", s.staged_cold)),
            ),
            (
                "  from pageable",
                Box::new(|s: &PipelineStats| format!("{}", s.staged_paged)),
            ),
            (
                "Stage requests",
                Box::new(|s: &PipelineStats| format!("{}", s.stage_requests)),
            ),
            (
                "Staged ahead",
                Box::new(|s: &PipelineStats| format!("{}", s.staged_speculative)),
            ),
            (
                "  then routed",
                Box::new(|s: &PipelineStats| format!("{}", s.staged_ahead_routed)),
            ),
            // The same split by who asked: the pipeline's predictions, then
            // the stager's own lookahead — reads, and of them routed.
            (
                "  predicted: routed",
                Box::new(|s: &PipelineStats| {
                    format!("{} of {}", s.staged_predicted_routed, s.staged_predicted)
                }),
            ),
            (
                "  lookahead: routed",
                Box::new(|s: &PipelineStats| {
                    format!(
                        "{} of {}",
                        s.staged_ahead_routed - s.staged_predicted_routed,
                        s.staged_speculative - s.staged_predicted
                    )
                }),
            ),
            (
                "  listed on landing",
                Box::new(|s: &PipelineStats| format!("{}", s.staged_listed)),
            ),
            (
                "Pad evictions",
                Box::new(|s: &PipelineStats| format!("{}", s.pad_evictions)),
            ),
            (
                "Evictions",
                Box::new(|s: &PipelineStats| format!("{}", s.evictions)),
            ),
            // The ring's claims judged at their row's next routing, by claim
            // kind: the share of victims routed again (regret) beside the
            // share of promoted experts routed again (paid) — a claim gains
            // only while the second exceeds the first.
            (
                "  regret (demand)",
                Box::new(|s: &PipelineStats| regret(s.claim_regretted[0], s.claim_evicted[0])),
            ),
            (
                "  paid (demand)",
                Box::new(|s: &PipelineStats| regret(s.claim_paid[0], s.claim_promoted[0])),
            ),
            (
                "  regret (ahead)",
                Box::new(|s: &PipelineStats| regret(s.claim_regretted[1], s.claim_evicted[1])),
            ),
            (
                "  evict cost µs",
                Box::new(|s: &PipelineStats| {
                    let c = s.eviction_cost_us;
                    if c.iter().all(|&x| x == 0.0) {
                        "-".to_string()
                    } else {
                        format!("{:.0}/{:.0}/{:.0}/{:.0}", c[0], c[1], c[2], c[3])
                    }
                }),
            ),
            (
                "  paid (ahead)",
                Box::new(|s: &PipelineStats| regret(s.claim_paid[1], s.claim_promoted[1])),
            ),
            // Read-ahead: the slots the GPU claimed for experts predicted for
            // later rows, of them those copied from the pad, how many the rows
            // then routed, the window each layer gave it and how far ahead it
            // looked.
            (
                "Read-ahead claims",
                Box::new(|s: &PipelineStats| format!("{}", s.ahead_claims)),
            ),
            (
                "  from the pad",
                Box::new(|s: &PipelineStats| format!("{}", s.ahead_pad_claims)),
            ),
            (
                "  judged",
                Box::new(|s: &PipelineStats| format!("{}", s.predicted_total)),
            ),
            (
                "  precision",
                Box::new(|s: &PipelineStats| format!("{:.1}%", s.prediction_precision())),
            ),
            // The router look-ahead: of the experts each hop's votes predicted,
            // the share the target row routed.
            (
                "Look-ahead precision",
                Box::new(|s: &PipelineStats| {
                    if s.lookahead_judged.iter().all(|&j| j == 0) {
                        "-".to_string()
                    } else {
                        (1..=s.lookahead_judged.len())
                            .map(|h| format!("{:.0}", s.lookahead_precision(h)))
                            .collect::<Vec<_>>()
                            .join("/")
                    }
                }),
            ),
            // Of the experts the judged rows routed, the share the votes had
            // predicted — what a wider prediction buys, beside what it wastes.
            (
                "  recall",
                Box::new(|s: &PipelineStats| {
                    if s.lookahead_routed.iter().all(|&r| r == 0) {
                        "-".to_string()
                    } else {
                        (1..=s.lookahead_routed.len())
                            .map(|h| format!("{:.0}", s.lookahead_recall(h)))
                            .collect::<Vec<_>>()
                            .join("/")
                    }
                }),
            ),
            // The list each row ahead was actually given (`expert_lre::blend`),
            // by hop — only experts a copy could serve, those not already held:
            // its precision, its recall of the routed experts not held, and the
            // recall of every such candidate any predictor proposed — the
            // ceiling a ranking can reach.
            (
                "List precision",
                Box::new(|s: &PipelineStats| per_hop(&s.selected_hits, &s.selected_judged)),
            ),
            (
                "  recall",
                Box::new(|s: &PipelineStats| per_hop(&s.selected_hits, &s.copy_routed)),
            ),
            (
                "  union recall",
                Box::new(|s: &PipelineStats| per_hop(&s.union_hits, &s.copy_routed)),
            ),
            (
                "Markov precision",
                Box::new(|s: &PipelineStats| per_hop(&s.markov_hits, &s.markov_judged)),
            ),
            (
                "Recency precision",
                Box::new(|s: &PipelineStats| per_hop(&s.recent_hits, &s.recent_judged)),
            ),
            (
                "  window (images)",
                Box::new(|s: &PipelineStats| format!("{}", s.ahead_window)),
            ),
            (
                "  depth (rows)",
                Box::new(|s: &PipelineStats| format!("{}", s.prefetch_depth)),
            ),
            (
                "  pad pins held",
                Box::new(|s: &PipelineStats| format!("{}", s.ahead_pinned)),
            ),
            // Bytes each mover put over its link: read-ahead into VRAM, staging
            // off the drive (or out of pageable RAM) into the pad.
            (
                "Read-ahead GiB",
                Box::new(|s: &PipelineStats| format!("{:.2}", s.ahead_bytes as f64 / GIB)),
            ),
            (
                "Staged GiB",
                Box::new(|s: &PipelineStats| format!("{:.2}", s.staged_bytes as f64 / GIB)),
            ),
            (
                "Stage read GB/s",
                Box::new(|s: &PipelineStats| {
                    if s.stage_read_ns == 0 {
                        "-".to_string()
                    } else {
                        format!("{:.2}", s.staged_bytes as f64 / s.stage_read_ns as f64)
                    }
                }),
            ),
            // The slowest single read per source, and the reads over the slow
            // threshold: a cold wait that traps was waiting on one of these.
            (
                "  pack reads",
                Box::new(|s: &PipelineStats| read_latency_cell(&s.pack_reads)),
            ),
            // Records per request: how far the readers' runs coalesce them.
            (
                "    requests",
                Box::new(|s: &PipelineStats| match s.pack_requests {
                    0 => "-".to_string(),
                    n => format!(
                        "{n} · {:.2} records each",
                        s.pack_reads.reads as f64 / n as f64
                    ),
                }),
            ),
            // Why staged reads waited: no pad slot the queue may take, or a
            // victim still in use by an invocation that had begun.
            (
                "  spec stalls",
                Box::new(|s: &PipelineStats| {
                    format!(
                        "{} no slot · {} in use",
                        s.spec_stalls_no_slot, s.spec_stalls_in_use
                    )
                }),
            ),
            (
                "  pageable reads",
                Box::new(|s: &PipelineStats| read_latency_cell(&s.paged_reads)),
            ),
            (
                "Routed layers",
                Box::new(|s: &PipelineStats| format!("{}", s.routed_messages)),
            ),
        ];

        let label_w = metrics
            .iter()
            .map(|(l, _)| l.len())
            .max()
            .unwrap_or(16)
            .max(16);

        // ── Header ──
        println!("\n=== Expert Pipeline Stats: {phase} ===");
        print!("┌{:─<lw$}", "", lw = label_w + 2);
        for _ in &headers {
            print!("┬{:─<cw$}", "", cw = col_w + 2);
        }
        println!("┐");

        print!("│ {:label_w$} ", "Metric");
        for h in &headers {
            print!("│ {:>col_w$} ", h);
        }
        println!("│");

        print!("├{:─<lw$}", "", lw = label_w + 2);
        for _ in &headers {
            print!("┼{:─<cw$}", "", cw = col_w + 2);
        }
        println!("┤");

        // ── Data rows ──
        let default_stats = PipelineStats::default();
        for (label, extractor) in &metrics {
            print!("│ {:label_w$} ", label);
            for r in results {
                let stats = pick(r).unwrap_or(&default_stats);
                let val = extractor(stats);
                print!("│ {:>col_w$} ", val);
            }
            println!("│");
        }

        // ── Footer ──
        print!("└{:─<lw$}", "", lw = label_w + 2);
        for _ in &headers {
            print!("┴{:─<cw$}", "", cw = col_w + 2);
        }
        println!("┘");
    }

    /// Print the disk-resident embedding tier's row-cache hit rate, if the
    /// model serves one.
    ///
    /// The counters are cumulative over the whole run, so the last config's
    /// snapshot is the total. This is the number §0.1's cost model turns on —
    /// the tier is IOPS-shaped, and the miss count IS the IOPS — and it went
    /// unread for the whole bring-up because nothing printed it.
    pub(super) fn print_row_cache_report(&self, results: &[TestResults]) {
        let Some((hits, misses, evictions)) = results.iter().rev().find_map(|r| r.row_cache_stats)
        else {
            return;
        };
        let total = hits + misses;
        if total == 0 {
            return;
        }
        println!("\n=== Embedding Row Cache (disk-resident tier) ===");
        println!(
            "  lookups {total} | hits {hits} ({:.1}%) | misses {misses} | evictions {evictions}",
            100.0 * hits as f64 / total as f64
        );
        // A miss is one row read from the tier; §0.1 sizes the tier by that
        // rate, so it is reported as reads rather than left to be inferred.
        println!("  tier reads (row granularity): {misses}");
    }

    /// Print what host RAM the engine page-locked, and what bounded it.
    ///
    /// The warm tier's size is the single largest lever on cold-load rate, and
    /// it is decided once at load by whichever of three ceilings is lowest. The
    /// expert table above reports the *outcome* (`Warm tier slots`) with no way
    /// to tell a tier that took everything it could from one that was refused —
    /// which is the difference between "buy more RAM" and "raise a cap".
    #[cfg(feature = "cuda")]
    pub(super) fn print_pinned_ram_report() {
        use crate::models::expert_lre::{
            last_warm_tier_sizing, CEILING_AVAILABLE, CEILING_HOST_BUDGET, CEILING_PINNABLE,
        };
        use candle::vram::{host_pinned_breakdown, host_pinned_bytes};

        let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("\n=== Host Pinned RAM ===");

        let held = host_pinned_bytes();
        let Some(s) = last_warm_tier_sizing() else {
            println!("  (no warm tier was sized in this process)");
            println!("  pinned held: {:.2} GiB", gib(held));
            return;
        };

        println!(
            "  Machine: {:.2} GiB total | {:.2} GiB free at launch (the baseline) | \
             {:.2} GiB free mid-load | {:.2} GiB already pinned when sized",
            gib(s.total_ram),
            gib(s.launch_ram),
            gib(s.available_ram),
            gib(s.already_pinned),
        );
        println!("  Ceilings on the warm tier (lowest wins):");
        for (name, v) in [
            (CEILING_HOST_BUDGET, s.expert_pinned_budget),
            (CEILING_AVAILABLE, s.available_less_headroom),
            (CEILING_PINNABLE, s.pinnable_cap),
        ] {
            println!(
                "    {:<38} {:>8.2} GiB{}",
                name,
                gib(v),
                if s.bound_by == name {
                    "   ← binds"
                } else {
                    ""
                }
            );
        }
        println!("  Bound by: {}", s.bound_by);
        println!(
            "  Warm tier: {} of {} evictable experts ({:.0}%), {:.2} GiB at {:.2} MiB/slot",
            s.slots,
            s.wanted_slots,
            100.0 * s.slots as f64 / s.wanted_slots.max(1) as f64,
            gib(s.taken_bytes),
            s.stride as f64 / (1024.0 * 1024.0),
        );
        let short = s.wanted_slots.saturating_sub(s.slots);
        println!(
            "  Shortfall: {short} experts with no warm copy, {:.2} GiB — every miss on \
             one of these reads the pack",
            gib(short as u64 * s.stride as u64),
        );

        match candle::vram::available_low_water() {
            Some(low) => println!(
                "  Free RAM low-water across the run: {:.2} GiB  (headroom asked for {:.2} GiB)",
                gib(low),
                gib(crate::models::expert_lre::WARM_TIER_HEADROOM),
            ),
            None => println!("  Free RAM low-water: not sampled"),
        }
        // **Is the wave transient tier being used at all?** A zero peak reads
        // identically to "never seeded": if no tensor is ever carved from the
        // arena, no operand carries a ticket, and every op on the wave path
        // allocates from the driver however correctly each one propagates.
        match candle_nn::kv_cache::wave_domain_stats(0) {
            Some(d) => {
                let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
                println!(
                    "  Wave arenas (used/peak/cap MiB): fwd {:.1}/{:.1}/{:.1} | attn {:.1}/{:.1}/{:.1} \
                     | ffn {:.1}/{:.1}/{:.1}",
                    mib(d[0].0), mib(d[0].1), mib(d[0].2),
                    mib(d[1].0), mib(d[1].1), mib(d[1].2),
                    mib(d[2].0), mib(d[2].1), mib(d[2].2),
                );
            }
            None => println!("  Wave arenas: no domain on device 0"),
        }
        // Which DeltaNet dispatch path each wave took, and what rejected it.
        // Counts, not timings — see `docs/qwen36_performance_plan.md` T1.
        let dt = crate::models::delta_net::mix::dispatch_tally();
        if !dt.is_empty() {
            println!("\n=== DeltaNet dispatch (path taken) ===");
            for (path, outcome, n) in dt {
                let tag = if outcome == "FUSED" {
                    "fused"
                } else {
                    "FELL BACK on"
                };
                println!("  {path:<12} {tag:>13} {outcome:<14} {n:>8}");
            }
        }
        let g = crate::models::expert_lre::grow_tally();
        println!(
            "  Boundary growth: asked {} | no-spare {} | spare offered {} regions | \
             target-unchanged {} | target-backwards {} | floor-refused {} | at-limit {} | \
             SLOTS GAINED {}",
            g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7],
        );
        print_span("end of run", None);

        let s = candle_nn::kv_cache::spare_tally();
        println!(
            "  Spare calc: observing {} | pressure {} | occupancy-bound {} | fragmented {} \
             | granted {} regions || KV purchases: conceded {} / refused {}",
            s[0], s[1], s[2], s[6], s[3], s[4], s[5],
        );
        // The other half of the same question. `Spare calc` says what the pool
        // offered; this says what the layer zone did with it, and a collapse
        // that shows spare being granted while the zone stays on its floor is
        // only attributable with both.
        let g = crate::models::layer_stream::growth_tally();
        println!(
            "  Zone growth: called {} | already whole {} | no spare {} | bought no layer {} \
             | under step floor {} | wave live {} | APPLIED {}",
            g[0], g[1], g[2], g[3], g[4], g[6], g[5],
        );
        println!("  Pinned now, by consumer:");
        for (use_, bytes) in host_pinned_breakdown() {
            if bytes > 0 {
                println!("    {:<38} {:>8.2} GiB", use_.label(), gib(bytes));
            }
        }
        println!("    {:<38} {:>8.2} GiB", "TOTAL", gib(held));
    }

    #[cfg(not(feature = "cuda"))]
    pub(super) fn print_pinned_ram_report() {}
}
