//! Loads a model and runs its configurations, with a watchdog that ends a run
//! that hangs.
//!
//! Loading is separated from running, so a caller with several sweeps loads a
//! large checkpoint once and runs each sweep against it. The watchdog ends the
//! process, so it is cancelled when the run returns however it returns, and it
//! wakes periodically so an early finish releases its thread at once. Each
//! configuration reports the KV regions and free memory it began with, so a
//! configuration that fails to hand its memory back shows up as a later one
//! that is slower. The card's wave-by-wave figures go to a CSV under `target/`,
//! and engine tracing follows the `RUST_LOG` filter, defaulting to INFO.

use super::*;

impl TestParams {
    /// Load a model and run `configs` against it.
    ///
    /// A caller that runs several *sweeps* over one checkpoint — the draft
    /// budgets of `speculative_gate`, say — should load once and call
    /// [`Self::run_loaded`] per sweep instead. Loading is not incidental cost
    /// on these models: a 124 GB artifact takes minutes to open, so a
    /// five-budget sweep spends most of its wall clock re-reading weights that
    /// never changed.
    // Sized, unlike the rest of the chain: this one *constructs* the model from a
    // closure, and a closure cannot return an unsized value. A caller that already holds
    // a boxed trait object uses `run_loaded` or `run_loaded_collect` instead.
    pub fn run<M>(self, configs: Vec<TestConfig>, load_model: impl Fn() -> Result<M>) -> Result<()>
    where
        M: ManagedBatchedModel,
    {
        let model = account_model_load(&self.device, load_model)?;
        self.run_loaded(configs, &model)
    }

    /// Run `configs` against a model the caller already holds.
    ///
    /// The memory bookkeeping around the load belongs to whoever performed it,
    /// so this reports the run's own figures and nothing about weights it did
    /// not bring in.
    pub fn run_loaded<M>(mut self, configs: Vec<TestConfig>, model: &M) -> Result<()>
    where
        M: ManagedBatchedModel + ?Sized,
    {
        let mut results = self.run_configs(configs, model)?;
        self.validate_and_print_results(&mut results)
    }

    /// Measure every config, and return the rows. See [`Self::run_loaded_collect`] for
    /// why measuring is separable from reporting.
    pub(super) fn run_configs<M>(
        &mut self,
        configs: Vec<TestConfig>,
        model: &M,
    ) -> Result<Vec<TestResults>>
    where
        M: ManagedBatchedModel + ?Sized,
    {
        // Under `verbose`, surface the engine's `tracing` events (expert-cache
        // sizing, warm-tier decisions, elastic-boundary moves) in the gate's
        // output — the sizing inputs are runtime facts of the machine and the
        // moment, and a gate that hides them cannot be used to diagnose a
        // mis-sized tier. `try_init` so a second `run` in one process is fine.
        //
        // **The level comes from `RUST_LOG`, defaulting to INFO.** A hard
        // `with_max_level(INFO)` stood here and silently made this comment false
        // for the most important item in its own list: every elastic-boundary
        // move — both `concede_kv_ground` and `reclaim_spare_ground` — is a
        // `debug!`, so the one class of event a gate needs in order to explain a
        // residency collapse was the one class it could not print, and no
        // `RUST_LOG` could turn it on. Targets are kept on for the same reason:
        // filtering by target is the point of an env filter.
        #[cfg(feature = "verbose")]
        {
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .compact()
                .try_init();
        }
        println!("✓ TestParams created successfully");
        println!("  - Dialect: {:?}", self.dialect.dialect_type);
        println!("  - Generate tokens: {}", self.generate_token_count);
        println!(
            "  - Prompt system length: {} chars",
            self.prompt_system.len()
        );
        println!("  - Prompt user length: {} chars", self.prompt_user.len());
        println!("  - Names available: {}", self.names.len());

        // Hard-terminate the process if this run hangs.
        //
        // **The watchdog must be cancellable.** It kills the PROCESS, not the
        // test, and `#[test]` binaries run many tests in one process — so an
        // uncancelled timer set by a fast test goes off later and shoots
        // whichever test happens to be running then. That is not hypothetical:
        // a whole-suite run of the batched-forward tests died at test 8 of 12
        // with `exceeded 180 seconds`, which was the budget of test 4 (a 0.5B
        // model that had long since passed). The report blames the wrong test
        // and the remaining tests never run at all.
        //
        // So the timer checks whether its own run finished before firing.
        let finished = Arc::new(AtomicBool::new(false));
        let timeout_secs = self.timeout_secs;
        {
            let finished = Arc::clone(&finished);
            std::thread::spawn(move || {
                // Wake periodically rather than sleeping the whole budget, so a
                // run that ends early releases the thread promptly instead of
                // leaving it armed over its successors.
                let deadline =
                    std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
                while std::time::Instant::now() < deadline {
                    if finished.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                if finished.load(Ordering::Relaxed) {
                    return;
                }
                eprintln!(
                    "\n❌ TIMEOUT: Test exceeded {} seconds - forcibly terminating process",
                    timeout_secs
                );
                std::process::exit(1);
            });
        }
        // Disarms the watchdog however this function leaves — including on the
        // `?` early-returns below, which a bare set at the end would miss.
        struct Disarm(Arc<AtomicBool>);
        impl Drop for Disarm {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let _disarm = Disarm(Arc::clone(&finished));

        let mut results = Vec::new();

        // The card wave by wave, to a CSV under `target/` as the run goes — the
        // performance page's bands, for the whole sweep. Closed when this
        // function returns, however it returns.
        #[cfg(feature = "cuda")]
        let wave_log = VramWaveLog::start(&self.device, model.expert_stats_shared());

        // The load's VRAM accounting is NOT done here: the weights are already
        // resident by the time this runs, so a baseline taken now measures
        // nothing. [`account_model_load`] wraps the load itself — `run` calls it
        // around `load_model`, and a caller that loads once for several sweeps
        // wraps its own load in it.

        for (n, config) in configs.into_iter().enumerate() {
            println!("\n=== Running tests for config: {:?} ===", config);
            #[cfg(feature = "cuda")]
            if let Some(log) = &wave_log {
                log.set_config(&format!("{:?} x{}", config.mode, config.num_contexts));
            }

            // Per-config user-prompt override (used by routing-trace capture).
            if let Some(p) = self.per_config_prompts.get(n) {
                if !p.is_empty() {
                    self.prompt_user = p.clone();
                    println!("  - Prompt override ({} chars)", self.prompt_user.len());
                }
            }

            // Tag any captured routing records with this config index.
            crate::models::routing_capture::begin_config(n);

            // Prune cached embedding variants before each config
            model.prune()?;

            // Reset expert pipeline telemetry before each config
            model.reset_expert_stats();

            // Reset profile accumulators so each config gets clean data.
            // (snapshot_profiles() does snapshot+reset atomically, so just
            // discard the stale data from startup / previous config tail.)
            let _ = model.snapshot_profiles();

            // GPU memory tracking: snapshot before config and register arena state
            let arena_bytes = candle_nn::kv_cache::global_arena_gpu_bytes();
            candle::gpu_memory::register("arenas (all backings)", arena_bytes);
            // **What the KV side is holding as this config starts.** A gate runs
            // its configs against ONE loaded model, so anything a config fails to
            // hand back is still held when the next one begins — and the symptom
            // is a throughput figure, not an error. Printing the region ledger
            // per config is what turns "the third config is slow" into "the third
            // config started with N fewer free regions than the first".
            // Free host RAM at every config boundary, so the run's trough is
            // recorded rather than inferred from the one reading at load.
            candle::vram::sample_available_low_water();
            if let Some(rs) = candle_nn::kv_cache::region_stats(0) {
                let (swept, sweeps) = candle_nn::kv_cache::empty_sweep_stats();
                // `blocked` is split, because its two halves call for opposite
                // responses and reading the total alone cannot tell them apart:
                // ground under a standing tier comes back at the end of the
                // wave, ground above the weight floor comes back only if the
                // weight side concedes. `ceiling` says which bound is binding.
                // Driver-side free VRAM beside the pool's own view. The two
                // answer different questions and the difference is the whole
                // out-of-reservation story: the pool can report ground to spare
                // while the *device* has none, because everything allocated
                // outside the span — and everything an async free has not yet
                // returned — is invisible to the pool.
                let dev_free_mib = self
                    .device
                    .mem_get_info()
                    .map(|(f, _)| f >> 20)
                    .unwrap_or(0);
                println!(
                    "  [regions] total {} | live {} | free {} | blocked {} \
                     | ceiling {} | arenas {} MiB | weight {} MiB \
                     | device free {dev_free_mib} MiB \
                     | mid-wave claims {} refused {} \
                     | swept {swept} arenas / {sweeps} sweeps",
                    rs.total,
                    rs.live,
                    rs.free,
                    rs.blocked,
                    rs.transient_ceiling,
                    arena_bytes >> 20,
                    rs.weight_bytes >> 20,
                    // Zero is the precondition for anchoring the tier at the
                    // arena frontier: a claim that arrives after admit meets a
                    // ceiling admit did not plan against, and is refused with
                    // nowhere to go.
                    rs.fresh_claims_during_wave,
                    rs.refusals_during_wave,
                );
            }
            let _ = candle::gpu_memory::snapshot(
                &format!("before_{:?}×{}", config.mode, config.num_contexts),
                &self.device,
            );

            // The harness is batched-only — the sequential/`forward_with_context` path is retired.
            let result = match self.run_batched_config(&config, model) {
                Ok(r) => r,
                Err(e) => {
                    // Arena bytes and table were already captured by the ArenaErrGuard
                    // inside run_batched_config (while the session was still alive).
                    // Just take the OOM snapshot — it will use those pre-registered values.
                    let _ = candle::gpu_memory::snapshot(
                        &format!("OOM_{:?}×{}", config.mode, config.num_contexts),
                        &self.device,
                    );
                    let _ = candle::gpu_memory::print_report(&self.device);
                    return Err(e);
                }
            };

            // Collect expert pipeline stats for this config
            let mut result = result;
            result.expert_stats = expert_stats_with_ceiling(model);
            result.row_cache_stats = model.row_cache_stats();
            if let Some(dir) = &self.routing_trace_dir {
                let path = dir.join(format!(
                    "{:02}_{:?}x{}.bin",
                    n + 1,
                    config.mode,
                    config.num_contexts
                ));
                model.write_expert_routing_trace(&path)?;
            }

            self.print_row_profiles(&result, self.config_labels.get(n).map(String::as_str));
            results.push(result);

            // Release GPU logits tensors from completed config — they're not needed
            // for validation (which only uses generated token IDs and expected text).
            // This prevents accumulating GPU memory across configs.
            if let Some(last) = results.last_mut() {
                for run in &mut last.sessions {
                    run.logits = Tensor::zeros(1, DType::F32, &Device::Cpu)?;
                }
            }

            // Prune again after config to release any accumulated dtype variants
            model.prune()?;
            // Sync device between configs to ensure memory is released
            self.device.synchronize()?;

            // GPU memory tracking: snapshot after config cleanup
            let _ = candle::gpu_memory::snapshot(
                &format!("after_{:?}×{}", config.mode, config.num_contexts),
                &self.device,
            );
        }

        Ok(results)
    }

    /// Run `configs` and hand back their results **without printing or validating**.
    ///
    /// The half of [`Self::run_loaded`] that measures. It exists so a caller one crate
    /// up can put its own rows in the same table: the engine-driven rows have to be
    /// produced *after* this model has been moved into a
    /// `ConversationEngine`, and the table has to be printed after both — which is
    /// impossible while running and printing are one call.
    ///
    /// [`Self::run_loaded`] is this followed by [`Self::validate_and_print`], so every
    /// existing caller measures, validates and prints exactly as it did before.
    pub fn run_loaded_collect<M>(
        &mut self,
        configs: Vec<TestConfig>,
        model: &M,
    ) -> Result<Vec<TestResults>>
    where
        M: ManagedBatchedModel + ?Sized,
    {
        self.run_configs(configs, model)
    }

    /// Helper to create a TestRun with expected output
    pub(super) fn create_test_run(
        &self,
        index: usize,
        logits: Tensor,
        effective_mode: TestMode,
    ) -> TestRun {
        let name = &self.names[index % self.names.len()];
        let name_capitalized = format!(
            "{}{}",
            name.chars().next().unwrap().to_uppercase(),
            name.chars().skip(1).collect::<String>().to_lowercase()
        );
        let expected_output = match effective_mode {
            TestMode::StoryRewrite => self
                .prompt_user
                .replace("Marcus", &name_capitalized)
                .replace("marcus", &name.to_lowercase())
                .replace("MARCUS", &name.to_uppercase()),
            TestMode::NameGreeting | TestMode::CoherenceCheck | TestMode::Skip => {
                // For these modes, expected is just the name.
                name_capitalized.clone()
            }
        };

        TestRun {
            output: Default::default(),
            logits,
            expected: expected_output,
            prompt_phase: PhaseResults {
                avg_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                runs_used: 0,
            },
            generate_avg_per_token: 0.0,
            generate_total_ms: 0.0,
            total_time_ms: 0.0,
        }
    }

    /// A prompt piece as the model is handed it: on the host, as the scheduler
    /// hands its prompts over. A wave's ids reach the device in one upload onto
    /// the forward's span (`wave_token_ids::device_token_ids`), never as one
    /// device allocation per sequence ahead of it.
    pub(super) fn tokens_to_tensor(&self, tokens: &[u32]) -> Result<Tensor> {
        Tensor::new(tokens, &Device::Cpu)?.unsqueeze(0)
    }
}
