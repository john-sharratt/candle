//! Runs one configuration through the batched session, from prefill through
//! timed decode, with its memory and tier checks around the run.
//!
//! Each repeat rewinds the K/V and forks each sequence's recurrent state aside
//! and back, so every repeat is a true re-prefill from identical state, and the
//! logits must come back bit-identical. A prefill that cannot be placed as one
//! wave is sliced into passes priced by the model's own placement. Each
//! sequence's offset advances to a target, so a sliced prefill is not counted
//! twice. Every sequence is freed, with its recurrent state released, before the
//! next configuration begins, so no leftover region makes it slower. The clocks
//! stop before reporting, since reporting walks the address space and would
//! otherwise be charged to decode.

use super::*;

impl TestParams {
    /// Run a configuration in batched mode using BatchedInferenceSession
    pub(super) fn run_batched_config<M>(
        &self,
        config: &TestConfig,
        model: &M,
    ) -> Result<TestResults>
    where
        M: ManagedBatchedModel + ?Sized,
    {
        // Create the batch session from the model using the inference mode's KV format.
        let batch_config = BatchedConfig {
            k_format: config.mode.k_format(),
            v_format: config.mode.v_format(),
            compression_level: config.mode.compression_level(),
            override_k_quant: self.override_k_quant,
            ..Default::default()
        };
        // One loaded model serves every config in the sweep and the configs
        // differ in KV dtype, so the norm weights are re-materialised per config.
        // `create_batched_session` does that itself — see `maybe_change_dtype`.
        let mut session = model.create_batched_session(batch_config)?;

        // RAII diagnostic guard — MUST be declared AFTER `session`.
        // Rust drops locals in reverse declaration order, so this guard drops
        // BEFORE `session` on any early return via `?`.  That means the backings
        // are still alive in the global registry when we capture arena state.
        // On the happy path we set fire=false before the Ok().
        struct ArenaErrGuard {
            fire: bool,
        }
        impl Drop for ArenaErrGuard {
            fn drop(&mut self) {
                if self.fire {
                    // Register correct arena bytes while session is still alive
                    let arena_bytes = candle_nn::kv_cache::global_arena_gpu_bytes();
                    candle::gpu_memory::register("arenas (all backings)", arena_bytes);
                    // Register per-format breakdown for the gpu_memory report
                    let arena_report = candle_nn::kv_cache::global_arena_memory_report();
                    for (backing_idx, label, count, bytes) in &arena_report {
                        candle::gpu_memory::register(
                            &format!("  arena[{}] {} (×{})", backing_idx, label, count),
                            *bytes,
                        );
                    }
                    // Print the full per-arena breakdown table
                    candle_nn::kv_cache::global_print_arena_table();
                }
            }
        }
        let mut arena_guard = ArenaErrGuard { fire: true };

        // Allocate all sequences first
        let mut sequence_indices = Vec::with_capacity(config.num_contexts);
        for _ in 0..config.num_contexts {
            let seq_idx = session.create_sequence()?;
            // Opted in while the slot is still empty, which is the only point
            // it can be: the adapter changes the projections that write K/V, so
            // a slot that has already prefilled cannot switch without attending
            // its own keys with queries from a different model.
            if let Some(name) = &self.lora {
                session.set_sequence_adapter(seq_idx, Some(std::sync::Arc::from(name.as_str())))?;
            }
            sequence_indices.push(seq_idx);
        }

        // Collect system prompts with their sequence indices and lengths
        let system_data: Vec<(usize, Vec<u32>, Tensor)> = (0..config.num_contexts)
            .map(|n| {
                let tokens = self.system_prompt_tokens(n);
                let tensor = self.tokens_to_tensor(&tokens)?;
                Ok((sequence_indices[n], tokens, tensor))
            })
            .collect::<Result<Vec<_>>>()?;

        // Group by token length for batched processing
        let mut by_length: std::collections::BTreeMap<usize, Vec<(usize, Tensor)>> =
            std::collections::BTreeMap::new();
        for (seq_idx, tokens, tensor) in system_data {
            by_length
                .entry(tokens.len())
                .or_default()
                .push((seq_idx, tensor));
        }

        // Process each length group in batched forward
        let mut logits_map: std::collections::HashMap<usize, Tensor> =
            std::collections::HashMap::new();
        for (token_len, group) in by_length {
            let seq_idxs: Vec<usize> = group.iter().map(|(idx, _)| *idx).collect();
            // A session with no system turn has nothing to prefill here. Its run
            // takes a placeholder the prompt phase overwrites with the logits it
            // actually samples from, as every run's are.
            if token_len == 0 {
                for &seq_idx in &seq_idxs {
                    logits_map.insert(seq_idx, Tensor::zeros(1, DType::F32, &Device::Cpu)?);
                }
                continue;
            }
            let tensors: Vec<Tensor> = group.into_iter().map(|(_, t)| t).collect();

            let nl = model.num_layers();
            let logits_vec = model
                .forward_wave(
                    &mut session,
                    &[],
                    &[],
                    &seq_idxs,
                    &tensors,
                    &[],
                    &[],
                    0,
                    nl,
                    None,
                )?
                .logits_owned()?;

            // Store logits and advance sequences
            for (&seq_idx, logits) in seq_idxs.iter().zip(logits_vec.into_iter()) {
                session.advance_sequence(seq_idx, token_len)?;
                logits_map.insert(seq_idx, logits);
            }
        }

        let effective_mode = config.test_mode.unwrap_or(self.test_mode);

        // Build runs in original order
        let mut runs = Vec::with_capacity(config.num_contexts);
        for (n, &seq_idx) in sequence_indices.iter().enumerate() {
            let logits = logits_map
                .remove(&seq_idx)
                .ok_or_else(|| candle::Error::Msg(format!("missing logits for seq {}", seq_idx)))?;
            runs.push(self.create_test_run(n, logits, effective_mode));
        }

        // Prompt phase — per-session user prompts, padded to uniform length
        // for single-batch processing.  NameGreeting mode embeds {INSERT_NAME}
        // in the user turn, producing slightly different token counts per
        // session.  We right-pad shorter sequences so all sessions can be
        // processed in one forward_batched call.  For StoryRewrite mode the
        // user prompt is identical across sessions so no padding occurs.
        //
        // IMPORTANT: padding is applied to the *user content* tokens only,
        // BEFORE appending the turn-end / assistant-start suffix.  This keeps
        // pad tokens inside the user turn where they are harmless trailing
        // whitespace, rather than after <|im_start|>assistant where they
        // would cause the model to emit EOS immediately.
        let suffix_tokens = self.user_suffix_tokens();
        let content_tokens_per_session: Vec<Vec<u32>> = (0..config.num_contexts)
            .map(|n| {
                let name = &self.names[n % self.names.len()];
                self.user_content_tokens(&self.prompt_user.replace("{INSERT_NAME}", name))
            })
            .collect();

        // Ragged-batch validation: feed `forward_batched` genuinely UNEVEN
        // prefill lengths (no padding to a common length). For StoryRewrite the
        // user prompt is identical across sessions, so when the batch has >1
        // sequence we make the LAST context LONGER by more than one 32-token
        // CHUNK_SIZE of trailing whitespace — harmless padding inside the user
        // turn that the model ignores, so NO session's output changes (every
        // session still validates), yet the per-sequence chunk COUNT now differs
        // across the batch. That difference is what exercises per-seq KV
        // capacity + the decode writer slice: the bug (over-allocating every
        // slot to max(q_lens)) would leave extra tail chunks on the *shorter*
        // full-story sessions and desync their decode — so a regression shows up
        // as those validated sessions diverging/asserting. A 1-token diff stays
        // within one chunk and misses the class entirely.
        let pad_token = self
            .tokenizer
            .encode(" ", false)
            .ok()
            .and_then(|enc| enc.get_ids().first().copied())
            .unwrap_or(0);
        let user_tokens_per_session: Vec<Vec<u32>> = content_tokens_per_session
            .into_iter()
            .enumerate()
            .map(|(n, mut tokens)| {
                if config.num_contexts > 1 && n + 1 == config.num_contexts {
                    // > CHUNK_SIZE (32) so the chunk count actually differs.
                    tokens.extend(std::iter::repeat(pad_token).take(40));
                }
                tokens.extend_from_slice(&suffix_tokens);
                tokens
            })
            .collect();
        let user_lens: Vec<usize> = user_tokens_per_session.iter().map(|t| t.len()).collect();

        let user_tensors: Vec<Tensor> = user_tokens_per_session
            .iter()
            .map(|tokens| self.tokens_to_tensor(tokens))
            .collect::<Result<Vec<_>>>()?;

        // Throughput repeats re-prefill the SAME user tokens on the SAME
        // sequences. The wave-entry offset reconciler advances any offset that
        // is behind its physical backing, so a naive re-run APPENDS the prompt
        // again instead of overwriting it — after N repeats the model would see
        // the story N times and the system prompt would be drowned out of the
        // attention window. Truncate each sequence back to its pre-prompt
        // length between repeats so every repeat is a true re-prefill from
        // identical state; the final repeat leaves exactly one prompt in KV.
        //
        // **"Identical state" has to mean all of it.** The truncation restores
        // the K/V, which is the whole of a sequence's memory only for a model
        // that keeps nothing else. On a model carrying recurrent state, repeat
        // N would enter having absorbed the previous N−1 prompts — the story
        // read N times after all, just invisibly, and attributed to the model
        // rather than to this loop. So each sequence's state is forked aside at
        // the base offset and forked back between repeats, which is exactly the
        // snapshot the K/V gets for free by being append-only. Both calls are
        // no-ops on a model with no such state.
        let repeat_base_offsets: Vec<usize> = sequence_indices
            .iter()
            .map(|&i| session.sequence_offset(i).unwrap_or(0))
            .collect();
        // Only when the loop will actually repeat: the shadows cost a slot each.
        let repeat_shadows: Vec<usize> = if config.num_repeats.max(1) > 1 {
            let mut shadows = Vec::with_capacity(sequence_indices.len());
            for &seq_idx in &sequence_indices {
                let shadow = session.create_sequence()?;
                model.fork_recurrent(seq_idx, shadow)?;
                shadows.push(shadow);
            }
            shadows
        } else {
            Vec::new()
        };
        // Start from an idle device: this config's session setup (norm weights
        // re-materialised for its dtype) and its system-prompt prefill are
        // still queued here, and would otherwise finish inside the prompt
        // timer and be reported as the prompt's prefill.
        self.device.synchronize()?;
        // The gap quantize, run as the scheduler runs it: after the prefill
        // here, and around the settle after every decode step. Its work lands
        // inside the clocks, which is where a daemon pays it.
        #[cfg(feature = "cuda")]
        let mut gaps = GapTally::default();
        let prompt_start = std::time::Instant::now();
        let t_prompt_total = profile_now();
        let mut repeat_base_logits: Option<Vec<Tensor>> = None;
        for repeat in 0..config.num_repeats.max(1) {
            if repeat > 0 {
                for (i, (&seq_idx, &base)) in sequence_indices
                    .iter()
                    .zip(repeat_base_offsets.iter())
                    .enumerate()
                {
                    session.truncate_sequence_to_tokens(seq_idx, base)?;
                    if let Some(&shadow) = repeat_shadows.get(i) {
                        model.fork_recurrent(shadow, seq_idx)?;
                    }
                }
            }
            let nl = model.num_layers();
            // **Prefill is sliced only when the wave as composed cannot be
            // placed** ([`prefill_pass_fits`]).
            //
            // A wave's transient tier grows with its rows and its sequences, so
            // a prompt submitted whole asks for a tier proportional to all of it:
            // at 127K tokens that was a measured 9.4 GB against a 3.2 GB span, and
            // eight 520-token prompts on Qwen3.8-Flash-Next at Q3 on a 16 GB card
            // are a 4,160-row wave whose tier the partition cannot place. Such a
            // prefill goes in passes, each as wide as the placement will take.
            //
            // **Priced by the model's own placement arithmetic, not by the width
            // cap.** `wave_tier_bytes` is the function the placement itself
            // calls, so a wave this approves is one the placement takes. The cap
            // is a cruder bound in both directions: it prices a one-sequence
            // wave, which under-prices a wide fleet, and the trait's default cap
            // is conservative enough that obeying it cut the Qwen3.8-27B's ×20
            // prefill into six passes its tier never needed, at a fraction of the
            // throughput and with the chunk boundaries moved under a calibrated
            // ladder. A wave that fits runs as the single call it always did. The
            // cap is printed beside the decision for comparison.
            let cap = model.prefill_width_cap(session.activation_dtype()).max(1);
            let lens: Vec<usize> = user_tensors
                .iter()
                .map(|t| t.dims().last().copied().unwrap_or(0))
                .collect();
            let total: usize = lens.iter().sum();
            let cost = prefill_pass_cost(model, &session, &sequence_indices, &lens);
            let whole = cost.as_ref().is_none_or(|c| c.tier + c.kv <= c.ground);
            if repeat == 0 {
                // The composed wave's price beside the ground it can reach, so a
                // sliced prefill says by how much it missed.
                let priced = cost.as_ref().map_or(String::new(), |c| {
                    format!(
                        " (tier {} + KV {} of {} MiB ground)",
                        c.tier >> 20,
                        c.kv >> 20,
                        c.ground >> 20
                    )
                });
                println!(
                    "  - prefill: {total} rows across {} sequences, cap {cap} rows per wave{priced}{}",
                    lens.len(),
                    if whole { "" } else { " — sliced" }
                );
            }
            let logits_vec = if whole {
                model
                    .forward_wave(
                        &mut session,
                        &[],
                        &[],
                        &sequence_indices,
                        &user_tensors,
                        &[],
                        &[],
                        0,
                        nl,
                        None,
                    )?
                    .logits_owned()?
            } else {
                // Each pass gives every sequence still prefilling an equal share
                // of the cap. Sequences whose prompt has been fully submitted
                // drop out rather than being padded: the wave takes a ragged
                // batch, and padding would put tokens in their KV that the
                // prompt never contained. A sequence's logits are kept from the
                // pass that submitted its last token — a shorter prompt finishes
                // a pass before a longer one, and only that pass predicts its
                // first generated token.
                let mut done = vec![0usize; lens.len()];
                let mut finals: Vec<Option<Tensor>> = vec![None; lens.len()];
                loop {
                    let active: Vec<usize> =
                        (0..lens.len()).filter(|&i| done[i] < lens[i]).collect();
                    if active.is_empty() {
                        break;
                    }
                    let idx: Vec<usize> = active.iter().map(|&i| sequence_indices[i]).collect();
                    let takes_at = |step: usize| -> Vec<usize> {
                        active
                            .iter()
                            .map(|&i| step.min(lens[i] - done[i]))
                            .collect()
                    };
                    // Starting from every remaining row, narrowed until the
                    // pass as composed (this many sequences at these depths)
                    // fits — the widest pass the placement will take — and
                    // carries no more rows than one forward can
                    // (`forward_row_ceiling`), whatever the ground.
                    let ceiling = model.forward_row_ceiling().unwrap_or(usize::MAX);
                    let mut step = active.iter().map(|&i| lens[i] - done[i]).max().unwrap_or(1);
                    while step > 1
                        && (takes_at(step).iter().sum::<usize>() > ceiling
                            || !prefill_pass_fits(model, &session, &idx, &takes_at(step)))
                    {
                        step = (step * 3 / 4).max(1);
                    }
                    let takes = takes_at(step);
                    if repeat == 0 {
                        // Each pass's price beside the ground it was approved
                        // against, and how much of that ground the weight side
                        // was counted on to concede — what a placement that
                        // refuses the pass is read beside.
                        let depth = idx
                            .iter()
                            .map(|&s| session.sequence_offset(s).unwrap_or(0))
                            .max()
                            .unwrap_or(0);
                        if let Some(c) = prefill_pass_cost(model, &session, &idx, &takes) {
                            println!(
                                "    pass: {} rows from depth {depth}: tier {} + KV {} of {} MiB \
                                 ground ({} MiB of it the weight side's to concede)",
                                takes.iter().sum::<usize>(),
                                c.tier >> 20,
                                c.kv >> 20,
                                c.ground >> 20,
                                model.reclaimable_kv_bytes() >> 20,
                            );
                        }
                    }
                    let mut parts: Vec<Tensor> = Vec::with_capacity(active.len());
                    for (&i, &take) in active.iter().zip(&takes) {
                        let t = &user_tensors[i];
                        parts.push(t.narrow(t.rank() - 1, done[i], take)?);
                        done[i] += take;
                    }
                    let logits = model
                        .forward_wave(&mut session, &[], &[], &idx, &parts, &[], &[], 0, nl, None)?
                        .logits_owned()?;
                    // Each pass advances the sequences it wrote, as the scheduler
                    // advances a chunk's members after their wave: the next
                    // pass is priced from the depth this one left, and its
                    // selection carve follows that depth. Left to the next
                    // forward's entry reconciler, the logical offset trailed the
                    // backing between passes, and a 128K prompt's second pass was
                    // priced from depth 0 — a 42,992 MiB tier for a pass that
                    // placed 57,280.
                    for (&s, &take) in idx.iter().zip(&takes) {
                        session.advance_sequence(s, take)?;
                    }
                    for (k, &i) in active.iter().enumerate() {
                        if done[i] == lens[i] {
                            finals[i] = logits.get(k).cloned();
                        }
                    }
                }
                finals
                    .into_iter()
                    .collect::<Option<Vec<Tensor>>>()
                    .ok_or_else(|| {
                        candle::Error::Msg(
                            "sliced prefill: a sequence finished without logits".into(),
                        )
                    })?
            };

            // Idempotence gate: with the truncate above, every repeat runs the
            // same tokens from the same state through the same kernels, so the
            // logits must be bit-identical. Any drift means repeat state leaked
            // (offset/backing divergence) and the throughput numbers are
            // measuring a different workload than reported.
            //
            // Every session's row at once: one per-row max|delta| on the device
            // and one read-back, rather than a synchronising read per session
            // inside the timed window.
            match &repeat_base_logits {
                None => repeat_base_logits = Some(logits_vec.clone()),
                Some(base) => {
                    let a = Tensor::cat(&base.iter().collect::<Vec<_>>(), 0)?;
                    let b = Tensor::cat(&logits_vec.iter().collect::<Vec<_>>(), 0)?;
                    let deltas = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
                        .abs()?
                        .flatten_from(1)?
                        .max(1)?
                        .to_vec1::<f32>()?;
                    for (i, &d) in deltas.iter().enumerate() {
                        if d != 0.0 {
                            candle::bail!(
                                "prompt repeat {} session {} is not idempotent: \
                                 logits max|delta| = {d:e} vs repeat 0",
                                repeat,
                                i
                            );
                        }
                    }
                }
            }

            for (logits, run) in logits_vec.into_iter().zip(runs.iter_mut()) {
                run.logits = logits;
            }
        }
        // The shadows exist only to hold the base state across the repeats.
        for &shadow in &repeat_shadows {
            session.free_sequence(shadow)?;
            model.release_sequence(shadow)?;
        }
        // Advance each sequence to the offset its own (ragged) prompt puts it
        // at — **to a target, not by a length.**
        //
        // A single prefill wave leaves the logical offset untouched, so the whole
        // prompt is still to be counted. A *sliced* prefill has counted it
        // already, pass by pass. Adding `ulen` on top of that counts the prompt
        // twice, and the second count is silent — it surfaces later, in the
        // decode step, as a sequence standing at roughly twice its real depth:
        // measured as "cannot truncate sequence 0 to 62934 tokens — it stands
        // at 32172" on the recurrent lineage, and as "computed write len 30720
        // is invalid for chunk_size 32" on the qwen4exp engine.
        //
        // Advancing to `base + ulen` is correct for both shapes and needs to
        // know nothing about which of them ran.
        for ((&seq_idx, &ulen), &base) in sequence_indices
            .iter()
            .zip(user_lens.iter())
            .zip(repeat_base_offsets.iter())
        {
            let target = base + ulen;
            let now = session.sequence_offset(seq_idx).unwrap_or(0);
            match target.checked_sub(now) {
                Some(0) | None => {}
                Some(gap) => session.advance_sequence(seq_idx, gap)?,
            }
            let after = session.sequence_offset(seq_idx).unwrap_or(0);
            if after != target {
                candle::bail!(
                    "prefill left sequence {seq_idx} at {after} tokens, not the {target} its \
                     {ulen}-token prompt puts it at (from {base}) — the prompt has been counted \
                     more than once, and every depth measured from here would be wrong"
                );
            }
        }
        // The gap after the prefill: its complete blocks quantized on the
        // device before the clock stops, put in place by the first decode
        // step's gap — as the scheduler's next wave would.
        #[cfg(feature = "cuda")]
        gaps.quantize(&mut session, GapKind::Prefill)?;
        self.device.synchronize()?;
        pipeline_record("bench:bulk_total", t_prompt_total);
        let prompt_duration = prompt_start.elapsed();
        // After the clock stops: the walk visits every page of the address space
        // — ~17M on Flash-Next's 54 GB mapping — and inside the window it would
        // be charged to prefill.
        print_host_ram_line(&format!(
            "{:?}×{} after prefill",
            config.mode, config.num_contexts
        ));
        let prompt_tokens = user_lens.iter().sum::<usize>() * config.num_repeats.max(1);
        let prompt_tokens_per_sec = (prompt_tokens as f64) / prompt_duration.as_secs_f64();

        // Snapshot bulk profile at prompt→generate boundary
        let bulk_profile = model.snapshot_profiles();
        // Capture + reset the pipeline spans here so the prefill (bulk) phase is
        // attributed separately from decode — the final `pipeline_snapshot_and_reset`
        // below then contains only the generate (decode) phase. Without this split
        // the MoE / mHC spans (which run in both phases) are un-attributable.
        // Harvest the GPU spans before snapshotting: their elapsed times are read
        // from CUDA events, so an unharvested pair contributes nothing and the
        // phase silently under-reports. The prompt clock synchronised just
        // above, so the blocking drain costs nothing extra.
        gpu_drain_blocking();
        let pipeline_bulk_profile = pipeline_snapshot_and_reset();
        // The expert counters split at the same boundary, for the same reason:
        // `expert_stats` settles the pipeline thread, so every routed prefill
        // layer is counted here and none in the decode figures that follow.
        let prefill_expert_stats = expert_stats_with_ceiling(model);
        model.reset_expert_stats();

        // Generate phase. Budget and early-stop live inside the driver's per
        // sequence emit sinks now, so there is nothing to track out here.
        self.device.synchronize()?;
        #[cfg(feature = "cuda")]
        gaps.set_window(Window::Decode);
        let generate_start = std::time::Instant::now();
        let t_decode_total = profile_now();
        // The steady-state decode loop is the hot loop the transient tier
        // exists for, so it is the window worth measuring: every device
        // allocation inside it is one the wave path should have taken from a
        // bump range instead. The warmup step above is deliberately outside —
        // its first-touch allocations are unavoidable and would drown the
        // per-step traffic that matters.
        //
        // Arming is scoped to this block so an early `?` cannot leave the
        // detector on for the sealing and reporting that follow.
        let detector = forbidden_alloc::armed();
        // What the wave chains did across the decode (`docs/decode_graphs.md`):
        // how finely the forwards were cut, and how often a segment could be
        // folded into its executable rather than instantiated afresh.
        #[cfg(feature = "cuda")]
        let graphs_before = match &self.device {
            Device::Cuda(dev) => Some(dev.capture_stats()),
            _ => None,
        };
        #[cfg(feature = "cuda")]
        let leases_before = crate::models::batch_test::graph_report::LeaseWaits::now();
        #[cfg(feature = "cuda")]
        let cpu_before = crate::models::batch_test::graph_report::CpuServed::now();
        // ONE generate path for every gate and every model. A model with a
        // drafter speculates; a model without one proposes nothing and the
        // driver degrades to a token-per-step decode, which is what the
        // one-token loop this replaced used to do. Because speculation accepts
        // only the model's own greedy continuation, both produce identical text,
        // so every gate's expected-output check is now also a losslessness check
        // — across the whole C0–C10 ladder, not just the speculative gates.
        let steps_run = self.speculative_decode_phase(
            &mut session,
            &sequence_indices,
            &mut runs,
            model,
            self.speculative_max_draft
                .resolve(model, sequence_indices.len()),
            self.tokens_for(config.mode),
            #[cfg(feature = "cuda")]
            &mut gaps,
        )?;
        self.device.synchronize()?;
        drop(detector);
        #[cfg(feature = "cuda")]
        if let (Some(before), Device::Cuda(dev)) = (graphs_before, &self.device) {
            print_graph_line(&before, &dev.capture_stats());
        }
        #[cfg(feature = "cuda")]
        leases_before.print_since();
        #[cfg(feature = "cuda")]
        cpu_before.print_since();
        // **The clock stops here, before the reporting below** — the same rule the
        // prefill clock keeps. `ProcessRam::capture` walks every page of the address
        // space (~17M on Flash-Next's 54 GB mapping), and inside the window it was
        // charged to decode: ~0.9 s a row, which read as a 43–54% decode regression
        // at one context while the speculative steps themselves ran at the recorded
        // rate.
        pipeline_record("bench:decode_total", t_decode_total);
        let generate_duration = generate_start.elapsed();
        // Every session of the row is still alive here, so KV stands at its
        // high-water for the row and the weight zone at what decode ran against.
        print_span(
            &format!("{:?} x{} decode end", config.mode, sequence_indices.len()),
            model.expert_stats().as_ref(),
        );
        let host_after_decode = ProcessRam::capture();
        let forbidden = forbidden_alloc::take_report();
        if !forbidden.is_clean() {
            eprintln!("[{:?}] {}", config.mode, forbidden);
        }
        // The other half of the picture: the detector says what did NOT come
        // from an arena, this says how much did. A phase whose peak is zero has
        // a chain that never started, which reads identically in the detector to
        // a chain that started and was never converted.
        #[cfg(feature = "cuda")]
        if let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() {
            if let Some([attn, ffn, fwd]) = candle_nn::kv_cache::wave_domain_stats(gpu_id) {
                // `peak` is a process-lifetime high-water mark, so once one
                // config saturates a span every later config reports the same
                // number. Read it as "the worst this process ever saw", not as a
                // per-config figure.
                //
                // **Capacity is instantaneous and is 0 here by design.** The wave
                // tier is placed inside the span per forward and torn down when
                // the last generation closes (`BumpArena::detached`), so between
                // configs there is no span to report. This used to print
                // "N B of 0 B", which reads as an arena with no budget — i.e. as
                // an allocation escaping the reservation — and was misdiagnosed
                // exactly that way. The peak is the number that means something.
                let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
                eprintln!(
                    "[{:?}] wave arena peaks (process-wide; tier is placed per forward and \
                     absent between them): attention {:.1} MiB, ffn {:.1} MiB, forward {:.1} MiB",
                    config.mode,
                    mib(attn.1),
                    mib(ffn.1),
                    mib(fwd.1),
                );
            }
        }
        // A dense model whose layers are slot tenants: which tier every layer
        // read came from. Cumulative for the process, like the arena peaks
        // above — the shape that matters is the ratio, not the absolute.
        // `cold` is the expensive column: a pack read is a synchronous NVMe
        // round trip on the forward thread, where a warm hit is one H2D.
        if let Some([hits, warm, cold, joins, evictions, homed, streaming]) =
            model.layer_stream_stats()
        {
            // `homed`/`streaming` are the zone's state **now**, not at its carve:
            // the boundary grows into spare KV ground over the first forwards, so
            // a run measured at full residency and a run that never streamed both
            // show near-zero transfers and only this tells them apart.
            eprintln!(
                "[{:?}] layer stream: hits {hits} warm {warm} cold {cold} \
                 joins {joins} evictions {evictions} | zone now {homed} homed, \
                 {streaming} streaming{}",
                config.mode,
                if streaming == 0 { " (whole)" } else { "" }
            );
        }
        let generate_tokens = steps_run * config.num_contexts;
        let generate_tokens_per_sec = if generate_tokens == 0 {
            0.0
        } else {
            (generate_tokens as f64) / generate_duration.as_secs_f64()
        };

        // Gaps until the cycle is quiet, before measuring, so %Quantized and the
        // compression ratio cover every complete block the decode wrote; the
        // writer chunk each sequence still appends to stays native, as it does
        // between a daemon's waves. Outside the clocks.
        // The rates with the gap's time taken out, read once the settle has
        // caught up the last gap the decode launched.
        #[cfg(feature = "cuda")]
        let (prompt_ex_gap_tps, generate_ex_gap_tps) = {
            gaps.settle(&mut session)?;
            eprintln!(
                "{}",
                gaps.line(&format!("{:?} x{}", config.mode, sequence_indices.len()))
            );
            (
                rate_ex_gap(
                    prompt_tokens,
                    prompt_duration,
                    gaps.window_cost(Window::Prefill).host(),
                ),
                rate_ex_gap(
                    generate_tokens,
                    generate_duration,
                    gaps.window_cost(Window::Decode).host(),
                ),
            )
        };
        #[cfg(not(feature = "cuda"))]
        let (prompt_ex_gap_tps, generate_ex_gap_tps) =
            (prompt_tokens_per_sec, generate_tokens_per_sec);

        // Calculate quantized token percentage for quantized modes (reads the
        // post-quantize arena formats set above).
        let quantized_token_percent = if config.mode.is_quantized() {
            session.estimate_quantized_percentage_by_sequences(&sequence_indices)
        } else {
            None
        };

        let compression_ratio = if config.mode.is_quantized() {
            session.compression_ratio_by_sequences(&sequence_indices)
        } else {
            None
        };
        let (k_side, v_side) = if config.mode.is_quantized() {
            (
                SideCompression::measure(&session, &sequence_indices, false),
                SideCompression::measure(&session, &sequence_indices, true),
            )
        } else {
            Default::default()
        };

        #[cfg(feature = "verbose")]
        let compression_distribution = if config.mode.is_quantized() {
            Some(session.compression_dist_by_sequences(&sequence_indices))
        } else {
            None
        };

        // Calculate peak tokens: sum of all tokens across all sessions
        let peak_tokens: usize = (0..config.num_contexts)
            .map(|n| {
                self.system_prompt_tokens(n).len() + user_lens[n] + self.tokens_for(config.mode)
            })
            .sum();

        // Verbose palette4 diagnostic: print per-chunk arena format distribution for the
        // last sequence at layer 0 before we free anything.
        #[cfg(feature = "verbose")]
        if let Some(&last_seq) = sequence_indices.last() {
            session.print_palette4_stats(last_seq);
        }

        // Print the distribution
        #[cfg(feature = "verbose")]
        if let Some(compression_distribution) = compression_distribution {
            session.print_compression_distribution(&compression_distribution);
        }

        // Palette sanity check: an all-zero palette map means the backing was allocated but the
        // identity bytes were never written, which silently produces wrong sub-band assignments.
        // Panic here so the bug is caught at test time rather than producing subtle accuracy drift.
        // Note: empty k_pal / v_pal is valid and means "use the shared identity palette".
        for &seq_idx in &sequence_indices {
            if let Some(backing) = session.backings().first() {
                if let Some(chunks) = backing.live_chunks_as_sealed(seq_idx) {
                    for (ci, chunk) in chunks.iter().enumerate() {
                        if !chunk.k_pal.is_empty() && chunk.k_pal.iter().all(|&b| b == 0) {
                            panic!(
                                "UNINITIALIZED PALETTE: seq={seq_idx} blk={ci} k_pal is non-empty \
                                 but all-zeros (expected identity bytes, got uninitialised buffer)"
                            );
                        }
                        if !chunk.v_pal.is_empty() && chunk.v_pal.iter().all(|&b| b == 0) {
                            panic!(
                                "UNINITIALIZED PALETTE: seq={seq_idx} blk={ci} v_pal is non-empty \
                                 but all-zeros (expected identity bytes, got uninitialised buffer)"
                            );
                        }
                    }
                }
            }
        }

        // Explicitly free sequences and compact to release GPU memory before next config
        //
        // **Both halves of a sequence, because they have different owners.** The
        // session owns the KV slots; the *model* owns the recurrent state, keyed
        // by sequence id, and `release_sequence` is the hook that ties the two
        // lifetimes together (its own doc says so). Freeing only the KV half
        // leaves the recurrent store — 48 DeltaNet layers of `s` and conv tails
        // per sequence on the 27B — holding reservation regions for the life of
        // the process. It is not KV, so the arena sweep cannot see it and every
        // KV-side diagnostic reports the pool as healthy while the weight zone
        // is squeezed: measured, 21 regions after one context, 84 after four,
        // 404 after twenty, never falling.
        for &seq_idx in &sequence_indices {
            session.free_sequence(seq_idx)?;
            model.release_sequence(seq_idx)?;
        }
        let t_cleanup_sweep = profile_now();
        // Return the freed sequences' regions to the pool before the next
        // config claims them.
        let _ = session.release_empty_arenas();
        pipeline_record("bench:cleanup_sweep", t_cleanup_sweep);
        drop(session);
        self.device.synchronize()?;

        // ── The KV must be gone before the next config starts ──
        //
        // **A gate, because the symptom is a throughput figure and not an
        // error.** Every sequence has been freed and the session dropped, so
        // nothing above holds a chunk; anything still live is held by a
        // reference this cleanup does not reach, and it is held for the rest of
        // the run. The weight zone is carved from what the KV side leaves, so a
        // config that keeps its regions makes *every later config* slower —
        // measured on the 27B, `live` went 21 → 84 → 404 and never fell, a
        // single-context config holding 84 regions where the same config run
        // first holds 21, and the zone stuck at half its size from C8 onward.
        //
        // Two assertions, because they fail for different reasons. Regions still
        // live means a chunk is still referenced. An arena that a second sweep
        // can still free means the first sweep did not run to completion — the
        // ground was free and simply not collected.
        let which = format!("{:?}×{}", config.mode, config.num_contexts);
        let residual = candle_nn::kv_cache::reclaim_empty_arenas();
        assert_eq!(
            residual, 0,
            "config {which}: a second sweep freed {residual} more arenas — the \
             cleanup above left empty arenas holding regions"
        );
        if let Some(rs) = candle_nn::kv_cache::region_stats(0) {
            // The held ground's arenas stand across configs by design: they are
            // the headroom provisioned for the next forward's held buffers, and
            // go back to the span once a window of forwards leaves them idle.
            let held = candle_nn::kv_cache::held_ground_stats(0)
                .map_or(0, |h| h.arena_bytes / candle_nn::kv_cache::REGION_BYTES);
            let live = rs.live.saturating_sub(held);
            assert_eq!(
                live,
                0,
                "config {which}: {live} KV regions ({} MiB) are still live after every \
                 sequence was freed and the session dropped ({held} held-arena regions \
                 not counted). Something outlives the cleanup and holds them for the \
                 rest of the run, so the weight zone can never grow back and every \
                 later config is slower.",
                live * (candle_nn::kv_cache::REGION_BYTES / (1024 * 1024)),
            );
        }

        // Snapshot single (generate) profile
        let single_profile = model.snapshot_profiles();

        // Success path: disarm the error guard so it doesn't fire on normal drop.
        arena_guard.fire = false;
        Ok(TestResults {
            config: config.clone(),
            sessions: runs,
            prompt_tokens_per_sec,
            generate_tokens_per_sec,
            prompt_ex_gap_tps,
            generate_ex_gap_tps,
            all_valid: true,
            quantized_token_percent,
            compression_ratio,
            k_side,
            v_side,
            peak_tokens,
            expert_stats: None, // Filled by run() after collection
            prefill_expert_stats,
            host_after_decode,
            row_cache_stats: None, // Likewise
            bulk_profile,
            single_profile,
            pipeline_bulk_profile,
            pipeline_profile: {
                gpu_drain_blocking();
                pipeline_snapshot_and_reset()
            },
            effective_test_mode: effective_mode,
        })
    }
}
