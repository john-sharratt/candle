//! Prints the table that puts a run's configurations side by side.
//!
//! Each row is measured under one int8 mode, against the first row as the
//! baseline. The `ex-gap` rate takes the gap quantize's host time out of the
//! clock, so it is comparable with a run that sealed outside the clocks. The
//! harness's rows are the ceiling, and a caller's engine rows are printed below
//! them in the same table against the same baseline, since the delivered rate
//! is the point of the comparison. Host RAM and the spare-ground attribution are
//! printed here rather than after the expert table. That table is never reached
//! by a dense model, and it is where a streamed model's residency is explained.

use super::*;

impl TestParams {
    /// Print comparison table of test results
    pub(super) fn print_comparison_table(&self, results: &[TestResults]) {
        if results.is_empty() {
            return;
        }

        println!("\n\n=== Performance Comparison ===");
        // `ex-gap` is the rate with the gap quantize's host time inside the
        // clock taken out (`gap_tally`): the forwards' own rate, comparable
        // with a run that sealed outside the clocks. The gap's kernels run on
        // their own stream beside the forwards; what they cost the forwards is
        // in both columns.
        println!("┌──────────┬──────┬─────────┬──────────┬───────┬────────────┬──────────────┬─────────────┬───────────────┬─────────────┬───────────────┬───────────┬──────────┬────────────┬──────────┬───────┐");
        println!("│ KvMode   │ int8 │ Batched │ Contexts │ Valid │ t/s (bulk) │ t/s (single) │ ex-gap bulk │ ex-gap single │ perf (bulk) │ perf (single) │ %Quantized│ Compress │ Peak Tokens│ Frontier │ Eff%  │");
        println!("├──────────┼──────┼─────────┼──────────┼───────┼────────────┼──────────────┼─────────────┼───────────────┼─────────────┼───────────────┼───────────┼──────────┼────────────┼──────────┼───────┤");

        // Baseline is the first config
        let baseline = &results[0];
        let baseline_prompt_tps = baseline.prompt_tokens_per_sec;
        let baseline_generate_tps = baseline.generate_tokens_per_sec;

        for (i, result) in results.iter().enumerate() {
            let mode_str = format!("{:?}", result.config.mode);
            // int8 weight mode for this run (one mode per load, same for every row).
            let int8_str = match self.int8mode {
                Int8Mode::Off => "off",
                Int8Mode::Performance => "perf",
                Int8Mode::Precision => "prec",
            };
            let batched_str = if result.config.use_batched {
                "  yes  "
            } else {
                "  no   "
            };
            let contexts = result.config.num_contexts;
            let valid_str = match result.effective_test_mode {
                TestMode::NameGreeting | TestMode::Skip => "  -  ",
                _ if result.all_valid => "  ✓  ",
                _ => "  ✗  ",
            };
            let prompt_tps = result.prompt_tokens_per_sec;
            let generate_tps = result.generate_tokens_per_sec;

            let prompt_perf = if i == 0 {
                "baseline".to_string()
            } else {
                format!(
                    "{:+.1}%",
                    ((prompt_tps / baseline_prompt_tps - 1.0) * 100.0)
                )
            };

            let generate_perf = if i == 0 {
                "baseline".to_string()
            } else {
                format!(
                    "{:+.1}%",
                    ((generate_tps / baseline_generate_tps - 1.0) * 100.0)
                )
            };

            let quant_str = match result.quantized_token_percent {
                Some(pct) => format!("{:>8.1}%", pct),
                None => "     -   ".to_string(),
            };

            let compress_str = match result.compression_ratio {
                Some(ratio) => format!("{:>6.2}x", ratio),
                None => "    -  ".to_string(),
            };

            println!(
                "│ {:>8} │ {:>4} │ {} │ {:>8} │ {} │ {:>10.1} │ {:>12.1} │ {:>11.1} │ {:>13.1} │ {:>11} │ {:>13} │ {} │ {:>8} │ {:>10} │ {:>8} │ {:>5} │",
                mode_str,
                int8_str,
                batched_str,
                contexts,
                valid_str,
                prompt_tps,
                generate_tps,
                result.prompt_ex_gap_tps,
                result.generate_ex_gap_tps,
                prompt_perf,
                generate_perf,
                quant_str,
                compress_str,
                result.peak_tokens,
                // A harness row cannot fragment — every sequence is freed and every
                // empty arena released between configs, with a gate asserting nothing
                // is live before the next one starts. So these are not missing numbers,
                // they are numbers the row's own construction rules out.
                "-",
                "-",
            );
        }

        // **The caller's rows, below the harness's, in the same table.** Measured one
        // crate up through the real engine, so they carry admission, per-turn
        // projection, the persistence thread and whatever fragmentation the pool has
        // accumulated — everything the rows above exclude by design. Same baseline for
        // the `perf` columns, because comparing the delivered rate against the ceiling
        // is the whole reason they share a table.
        for row in &self.extra_rows {
            let valid_str = match row.valid {
                None => "  -  ".to_string(),
                Some((pass, total)) if pass == total => "  ✓  ".to_string(),
                Some(_) => "  ✗  ".to_string(),
            };
            let prompt_perf = format!(
                "{:+.1}%",
                (row.prompt_tokens_per_sec / baseline_prompt_tps - 1.0) * 100.0
            );
            let generate_perf = format!(
                "{:+.1}%",
                (row.generate_tokens_per_sec / baseline_generate_tps - 1.0) * 100.0
            );
            let compress_str = match row.compression_ratio {
                Some(ratio) => format!("{:>6.2}x", ratio),
                None => "    -  ".to_string(),
            };
            let frontier_str = match row.frontier_regions {
                Some(f) => f.to_string(),
                None => "-".to_string(),
            };
            let eff_str = match row.efficiency_pct {
                Some(e) => format!("{e}%"),
                None => "-".to_string(),
            };
            // `%Quantized` has no meaning for an engine row: the workload chose its own
            // formats per turn rather than running one mode, so there is no single figure
            // to report and a number here would invite a comparison that is not available.
            println!(
                "│ {:>8} │  eng │  engine │ {:>8} │ {} │ {:>10.1} │ {:>12.1} │ {:>11} │ {:>13} │ {:>11} │ {:>13} │      -    │ {:>8} │ {:>10} │ {:>8} │ {:>5} │",
                row.label,
                row.contexts,
                valid_str,
                row.prompt_tokens_per_sec,
                row.generate_tokens_per_sec,
                // The engine runs its own gaps through the scheduler, which
                // reports them on its `kv_gap` line rather than per row.
                "-",
                "-",
                prompt_perf,
                generate_perf,
                compress_str,
                row.peak_tokens,
                frontier_str,
                eff_str,
            );
        }

        println!("└──────────┴──────┴─────────┴──────────┴───────┴────────────┴──────────────┴─────────────┴───────────────┴─────────────┴───────────────┴───────────┴──────────┴────────────┴──────────┴───────┘");
        let side_rows: Vec<_> = results
            .iter()
            .filter(|r| r.compression_ratio.is_some())
            .map(|r| {
                (
                    format!("{:?}", r.config.mode),
                    r.compression_ratio,
                    &r.k_side,
                    &r.v_side,
                )
            })
            .collect();
        print_side_table(&side_rows);
        if !self.extra_rows.is_empty() {
            println!(
                "  Rows marked `engine` run through the real ConversationEngine — admission, \
                 per-turn projection,\n  the persistence thread, KV compaction — so they show \
                 what a daemon delivers. The rows above\n  them drive `forward_wave` from a \
                 clean slate and show how fast the forward itself can go.\n  Frontier is the \
                 arena watermark in 16 MiB regions, Eff% the share of it holding KV."
            );
        }
        // Where this process's host RAM stood as each config's decode ended —
        // what the warm expert tier is sized against, beside what it holds.
        let host: Vec<(String, Option<ProcessRam>)> = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                (
                    format!("#{} {:?}×{}", i + 1, r.config.mode, r.config.num_contexts),
                    r.host_after_decode.clone(),
                )
            })
            .collect();
        print_host_ram(&host);
        // **Here, not after the expert table.** The pinned-RAM report carries the
        // boundary's `Spare calc:` attribution — which of the four gates refused
        // the weight side ground — and it used to hang off
        // `print_expert_stats_table`, which returns immediately when no config
        // has expert stats. That is every dense model, so the one report that
        // explains a streamed model's residency was unreachable for exactly the
        // models whose whole decode rate is residency. A 10× decode collapse read
        // as unexplained in this table while the number naming its cause was
        // being computed and dropped.
        Self::print_pinned_ram_report();
    }

    /// Print a transposed expert pipeline stats table.
    ///
    /// Rows = metrics, Columns = test configs (#1, #2, …).
    /// Only printed if at least one config has expert stats.
    pub(super) fn print_expert_stats_table(&self, results: &[TestResults]) {
        // Skip entirely if no config has expert stats.
        if !results.iter().any(|r| r.expert_stats.is_some()) {
            return;
        }
        // One table per phase: the counters are reset at the prefill → decode
        // boundary, and the gauges in the prefill table are what decode inherits.
        Self::print_expert_phase_table("prefill", results, |r| r.prefill_expert_stats.as_ref());
        Self::print_expert_phase_table("decode", results, |r| r.expert_stats.as_ref());
        Self::print_blend_cells(results);
        Self::print_victim_tags(results);
        self.print_row_cache_report(results);
    }
}
