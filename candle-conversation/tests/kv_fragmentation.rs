//! Per-model KV-fragmentation gates, and the combined comparison table.
//!
//! Two shapes of test:
//!
//! - [`qwen3_30b_a3b_q4`] — the probe alone. Fast to read, fails on any of its three
//!   gates, and is what to run when iterating on compaction.
//! - [`qwen3_30b_a3b_q4_combined`] — the batched forward ladder **and** the engine-driven
//!   rows in one table. The ladder rows drive `forward_wave` from a clean slate and show
//!   how fast the forward itself can go; the engine rows run through the real
//!   `ConversationEngine` — admission, per-turn projection, the persistence thread, KV
//!   compaction — and show what a daemon delivers. One model load serves both.
//!
//! The 30B carries neither a drafter nor recurrent state, so its rows say nothing about
//! the verify path or a recurrent store. [`qwen36_35b_a3b_q4_combined`] is the same
//! comparison on a checkpoint with both — MoE, its NextN head speculating, and a 3:1
//! DeltaNet hybrid — and [`qwen38_flash_next_combined`] the flagship's.
//!
//! Adding a model is a `ModelProfile` row in `fragmentation_probe::profile` and a case
//! here naming it, with the model's own forward ladder.
//!
//! **Every case is `#[ignore]`d**, for the reasons the forward gates are: each loads a
//! multi-gigabyte checkpoint, sizes itself from the whole card, and runs for minutes.
//! They are gates, run deliberately, one `cargo` process at a time with nothing else
//! holding the GPU — never as part of the default suite.
//!
//! ```text
//! cargo test -p candle-conversation --features hub --test kv_fragmentation \
//!   qwen3_30b_a3b_q4_combined -- --exact --ignored --nocapture --test-threads=1
//! ```
//!
//! # What a failure means
//!
//! The three gates are not interchangeable. A **story** failure is a correctness defect:
//! a compaction relocated a chunk and left a holder naming the vacated slot, which does
//! not fault and instead answers from another sequence's KV. A **VRAM efficiency** or
//! **weight uptake** failure is the compaction pass not keeping up, or the weight side
//! not taking what it released — throughput, not correctness. Read which one failed
//! before reaching for a threshold.

use candle::Device;
use candle_conversation::fragmentation_probe::{
    names, profile, run, run_baseline, run_on_model, ModelProfile, Probe, ProbeOutcome,
};
use candle_conversation::models::Model;
use candle_conversation::ManagedBatchedModel;
use candle_transformers::models::batch_test::utils::{TestConfig, TestMode, TestParams};
use candle_transformers::models::batched_inference::InferenceMode;
use candle_transformers::models::quant_ladder;
use candle_transformers::models::{
    quantized_qwen36_moe, quantized_qwen38_moe, quantized_qwen3_moe,
};

/// Resolve a profile by name, or say what could have been named instead.
fn resolved(name: &str) -> ModelProfile {
    profile(name).unwrap_or_else(|| {
        panic!(
            "no model profile named {name:?}. Known: {}",
            names().join(", ")
        )
    })
}

/// The Flash-Next row, running the preset this card's rung loads.
///
/// Every machine holds only its own rung's engine artifact, so the probe takes the
/// same `quant_ladder::expert_format` choice the forward gate does — the 16 GB
/// laptop runs `Q2_KO` experts, the 72 GB card `Q4_KO`, from one row of thresholds.
fn flash_next_row() -> ModelProfile {
    let mut row = resolved("qwen38-flash-next");
    let device = Device::new_cuda(0).expect("CUDA device");
    let gib = quant_ladder::device_vram_gib(&device).expect("the card's VRAM");
    let experts = quant_ladder::expert_format(gib);
    row.model = Model::qwen38_flash_next_for(experts).unwrap_or_else(|| {
        panic!("a {gib} GiB card's rung ({experts:?}) has no Flash-Next preset")
    });
    row
}

/// Install the log subscriber every probe in this file needs.
///
/// **A probe that fails its story gate must be able to say why.** The engine already
/// carries the instrument for this fault: a degenerate token run calls
/// `describe_recurrent_state`, which reports per-layer finiteness of the persistent
/// recurrent state at `error` on `candle_conversation::eos`. Without a subscriber that
/// report is discarded, and the failure reads as "8 characters did not match" — the one
/// question the test exists to answer, thrown away at the last step.
///
/// INFO by default, because the span table cannot answer how WIDE a wave was. The spans
/// say what each phase cost per call; the scheduler's own per-wave line says how many
/// sequences that call carried. A decode phase costing 88 ms is a different diagnosis at
/// 20 sequences than at 1, and only the log distinguishes them.
///
/// Through `RUST_LOG` rather than a fixed level, so the next question down is reachable
/// without editing this file. The admission pass reports its verdict (`admitted`,
/// `queued`, `stopped_on_weights`, `stopped_on_rate`) at debug on
/// `candle_conversation::scheduler::throttle`, and a width of one is either that gate
/// refusing or the deadlock-freedom rule forcing a single head; those are opposite fixes
/// and the log line is what separates them.
///
/// `try_init` because the tests share a process: the second caller finds a subscriber
/// already installed, which is success, not failure.
fn logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init()
        .ok();
}

/// Print the outcome's own summary line — the figures the table cannot carry.
fn summarise(name: &str, outcome: &ProbeOutcome) {
    let b = &outcome.baseline;
    println!(
        "\n{name}: clean C{}×{} baseline {}/{} at {:.1} t/s prefill, {:.1} t/s decode",
        b.level,
        b.width,
        b.story_pass,
        b.width,
        b.timing.prefill_tps(),
        b.timing.decode_tps(),
    );
    println!(
        "{name}: story {}/{}, worst sustained efficiency {}%, worst single sample {}%, {}",
        outcome.story_pass,
        outcome.story_total,
        outcome.worst_sustained_efficiency,
        outcome.worst_single_efficiency,
        outcome.weight_uptake_label(),
    );
}

/// Qwen3-30B-A3B Q4_K_M — the probe's three gates on their own.
#[test]
#[ignore = "loads the 30B-A3B and runs for several minutes; needs the card to itself"]
fn qwen3_30b_a3b_q4() {
    logging();
    let probe = Probe::new(resolved("qwen3-30b-a3b-q4"));
    let outcome = run(&probe).expect("the probe ran");
    summarise("qwen3-30b-a3b-q4", &outcome);
    outcome.assert_passed();
}

/// Qwen3.8-Flash-Next — **the recurrent-state case**, and the reason this model has a row.
///
/// The only arch in the table carrying per-sequence state outside the paged K/V: the
/// DeltaNet recurrence, the PLE convolution tail and the QSA index cache. A compaction
/// relocates K/V chunks and a successful pass asks the weight side to take the ground the
/// frontier gave up, which moves `weight_floor` — and the recurrent store lives in the same
/// span. Nothing in the K/V sweep touches that state, so nothing in the 30B row can catch
/// it being trodden on.
///
/// On the live daemon it was: two compaction passes, then 15 of 36 layers of persistent
/// recurrent state non-finite, an all-NaN logits row, and a turn truncated on a forced EOS.
/// This test is that sequence, reproducible.
#[test]
#[ignore = "loads Qwen3.8-Flash-Next at this card's rung and runs the engine probe; \
            needs the card to itself"]
fn qwen38_flash_next() {
    logging();
    let probe = Probe::new(flash_next_row());
    let outcome = run(&probe).expect("the probe ran");
    summarise("qwen38-flash-next", &outcome);
    outcome.assert_passed();
}

/// **Flash-Next's clean `C5 ×8` baseline alone** — the iteration loop for optimising the
/// engine against the gate's `C5 ×8` row, as [`qwen36_35b_a3b_q4_baseline`] is for the
/// 3.6. The gate row to read it against comes from [`qwen38_flash_next_combined`].
#[test]
#[ignore = "loads Qwen3.8-Flash-Next at this card's rung and runs the engine's clean C5×8 \
            baseline; needs the card to itself"]
fn qwen38_flash_next_baseline() {
    logging();
    let probe = Probe::new(flash_next_row());
    let (row, fails) = run_baseline(&probe).expect("the baseline ran");
    println!(
        "\nqwen38-flash-next: clean C{}×{} baseline {}/{} at {:.1} t/s prefill, {:.1} t/s decode",
        row.level,
        row.width,
        row.story_pass,
        row.width,
        row.timing.prefill_tps(),
        row.timing.decode_tps(),
    );
    print_pipeline_profile("Flash-Next clean C5×8 baseline — measured batch");
    let fails = fails.join("\n  ");
    assert!(fails.is_empty(), "the clean baseline failed:\n  {fails}");
}

/// **The flagship's ceiling and its delivered rows, from one model load.**
///
/// The question this answers is how much of the forward path's throughput the
/// conversation engine actually delivers. Answering it by reading the engine
/// probe against the standalone gate in `candle-transformers` does not work: that
/// gate builds the model at 262,144 context, which is a different RoPE
/// configuration and therefore different numerics and a different cost. The
/// ladder has to be measured on the model the engine runs, which is what
/// [`ladder_and_engine`] does — same load, same context, same tokenizer, both
/// sets of rows. The ladder is Flash-Next's own gate ladder, whose `C5 ×8` row is
/// the one the engine's clean baseline reproduces.
#[test]
#[ignore = "loads Qwen3.8-Flash-Next once and runs both the forward ladder and the \
            engine probe — tens of minutes; needs the card to itself"]
fn qwen38_flash_next_combined() {
    ladder_and_engine(
        flash_next_row(),
        quantized_qwen38_moe::batched_forward_configs,
        true,
    );
}

/// Qwen3.6-35B-A3B — the probe's three gates on their own, **with speculation on.**
///
/// The 30B row cannot exercise the verify path: it has no drafter, so every phase-B
/// step is a one-token row. This checkpoint drafts with its NextN head at budget 2 for
/// the whole batch, and every accepted block rewinds a DeltaNet store as well as K/V —
/// so a compaction that moved something a rewind still names fails the story here.
#[test]
#[ignore = "loads Qwen3.6-35B-A3B (22 GB) and runs the engine probe with speculative \
            decode; needs the card to itself"]
fn qwen36_35b_a3b_q4() {
    logging();
    let probe = Probe::new(resolved("qwen36-35b-a3b-q4"));
    let outcome = run(&probe).expect("the probe ran");
    summarise("qwen36-35b-a3b-q4", &outcome);
    outcome.assert_passed();
}

/// **The engine's clean `C5 ×8` baseline alone** — the iteration loop for optimising the
/// engine against the gate's `C5 ×8` row.
///
/// No ladder and no churn: one load, one unmeasured warm-up batch, one measured batch.
/// The gate row to read it against comes from [`qwen36_35b_a3b_q4_combined`], which
/// prints both from one load and is the run to quote.
#[test]
#[ignore = "loads Qwen3.6-35B-A3B and runs the engine's clean C5×8 baseline; ~1 minute, \
            needs the card to itself"]
fn qwen36_35b_a3b_q4_baseline() {
    logging();
    let probe = Probe::new(resolved("qwen36-35b-a3b-q4"));
    let (row, fails) = run_baseline(&probe).expect("the baseline ran");
    println!(
        "\nqwen36-35b-a3b-q4: clean C{}×{} baseline {}/{} at {:.1} t/s prefill, {:.1} t/s decode",
        row.level,
        row.width,
        row.story_pass,
        row.width,
        row.timing.prefill_tps(),
        row.timing.decode_tps(),
    );
    // Empty unless built with `--features hub,profile`: the measured batch's spans,
    // for finding where the engine spends what the gate does not.
    print_pipeline_profile("Qwen3.6 clean C5×8 baseline — measured batch");
    let fails = fails.join("\n  ");
    assert!(fails.is_empty(), "the clean baseline failed:\n  {fails}");
}

/// **The forward ceiling and the engine's delivered rows on a speculating MoE hybrid,
/// from one model load.**
///
/// The ladder is the 3.6's own gate ladder, not the 30B's: its rows are the ones
/// `quantized_qwen36_moe`'s forward gate validates, and its C10 ×8 and ×16 rows decode
/// speculatively — so the ceiling above the engine rows is measured on the same path
/// the engine takes.
#[test]
#[ignore = "loads Qwen3.6-35B-A3B once and runs both its forward ladder and the engine \
            probe — several minutes; needs the card to itself"]
fn qwen36_35b_a3b_q4_combined() {
    ladder_and_engine(
        resolved("qwen36-35b-a3b-q4"),
        quantized_qwen36_moe::batched_forward_configs,
        true,
    );
}

/// The flagship's engine phase alone, with the span breakdown.
///
/// Read against [`qwen38_flash_next_combined`]'s ladder rows: the ladder is
/// what the forward costs, this is where the rest of the wall clock goes. The
/// host and device tables are kept apart by [`print_pipeline_profile`] because
/// device time overlaps the host and a shared denominator understates every host
/// span.
///
/// **A profiled build is for attribution, not for throughput.**
/// `docs/performance.md` §2.4: the spans cost 5–24% of decode and 1–3% of
/// prefill, so the t/s this prints is lower than the engine's real rate. Quote
/// the uninstrumented run for rate and this one for where the time went.
#[test]
#[ignore = "profile run: the flagship's engine probe alone, for the span breakdown. \
            Needs --features hub,profile and the card to itself"]
fn qwen38_flash_next_profile_engine() {
    logging();
    let probe = Probe::new(flash_next_row());
    let outcome = run(&probe).expect("the probe ran");
    summarise("qwen38-flash-next (profile)", &outcome);
    print_pipeline_profile("Flash-Next engine — full wave loop");
}

/// The forward ladder **alone**, at the daemon's declared context.
///
/// Same rows and same method as the standalone gate in `candle-transformers`, but with
/// the model wrapped for the context the profile asks for rather than the gate's
/// 262,144 — which is a different RoPE configuration, and therefore different numerics.
/// That is the point of having it: the ladder the engine rows are compared against has to
/// be measured on the model the engine actually runs.
///
/// Separate from [`qwen3_30b_a3b_q4_combined`] so a threshold can be iterated without
/// paying for the engine phase every time. Nothing about the ladder depends on the engine
/// rows, and the engine phase is the long half.
#[test]
#[ignore = "loads the 30B-A3B and runs the full forward ladder; minutes, needs the card"]
fn qwen3_30b_a3b_q4_ladder() {
    ladder_and_engine(
        resolved("qwen3-30b-a3b-q4"),
        |_| quantized_qwen3_moe::batched_forward_configs(),
        false,
    );
}

/// The **combined table**: the forward ladder's ceiling rows and the engine's delivered
/// rows, one model load, one comparison.
#[test]
#[ignore = "loads the 30B-A3B once and runs both the forward ladder and the engine \
            probe — tens of minutes; needs the card to itself"]
fn qwen3_30b_a3b_q4_combined() {
    ladder_and_engine(
        resolved("qwen3-30b-a3b-q4"),
        |_| quantized_qwen3_moe::batched_forward_configs(),
        true,
    );
}

/// **Profile A — one ladder row, alone.** `Q8_0 × 20`: the widest validated row, and the
/// shape the engine row is compared against.
///
/// Alone on purpose. The pipeline profile accumulates process-wide, so a run of the whole
/// ladder blends eighteen configurations' spans into one breakdown and answers nothing
/// about any of them. One row, one breakdown.
///
/// Run with `--features hub,profile`; without it the spans compile to nothing and the
/// table below is empty, which is the intended zero-cost default rather than a failure.
#[test]
#[ignore = "profile run: Q8_0x20 alone, for the span breakdown. Needs --features \
            hub,profile and the card to itself"]
fn qwen3_30b_a3b_q4_profile_ladder_q8x20() {
    let probe = Probe::new(resolved("qwen3-30b-a3b-q4"));
    let device = Device::new_cuda(probe.device).expect("CUDA device");
    let (params, mut results, _model) = ladder_rows(
        &probe,
        &device,
        vec![TestConfig {
            mode: InferenceMode::Q8_0,
            use_batched: true,
            num_contexts: 20,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        }],
    );
    params
        .validate_and_print(&mut results)
        .expect("the row validated");
}

/// **Profile B — the engine, alone.** The same width through the real
/// `ConversationEngine`, with no ladder row before it to mix into the accumulator.
///
/// Read against profile A: both are 20 concurrent sequences on one checkpoint, so the
/// difference between the two breakdowns is where the engine's time goes.
#[test]
#[ignore = "profile run: the engine probe alone, for the span breakdown. Needs \
            --features hub,profile and the card to itself"]
fn qwen3_30b_a3b_q4_profile_engine() {
    logging();
    let probe = Probe::new(resolved("qwen3-30b-a3b-q4"));
    let outcome = run(&probe).expect("the probe ran");
    summarise("qwen3-30b-a3b-q4 (profile)", &outcome);
    print_pipeline_profile("Engine — full wave loop");
}

/// Print the process-wide pipeline accumulator, costliest span first.
///
/// The gate's own profile tables are per-config and keyed off `TestResults`, which an
/// engine run does not produce — so the same accumulator is read directly here. Sorted by
/// cost rather than execution order because the question this answers is "where did the
/// time go", and the answer is the first few rows.
fn print_pipeline_profile(title: &str) {
    let snap = candle_transformers::models::profile::pipeline_snapshot_and_reset();
    if snap.entries.is_empty() {
        println!(
            "\n=== {title} ===\n  (no spans recorded — build with `--features profile`, \
             or the hooks compile to nothing)"
        );
        return;
    }
    // **Device time and host time are not the same quantity and must not share a
    // denominator.** `gpu_span` records CUDA stream duration between two events; the
    // scheduler's spans record wall clock on the loop thread. Summing them gives a total
    // that means nothing — device work overlaps the host, so it can exceed the run — and a
    // share against it silently understates every host span. That is not a cosmetic point:
    // read as one table, `wv:sweep` looked like the largest cost in the engine when it is
    // device time that overlaps everything else.
    //
    // Membership is by name because that is where the distinction is decided: a span is
    // GPU exactly when its call site used `gpu_span`.
    // **Membership is the exact set of names `gpu_span` records, not a prefix
    // guess.** A prefix rule cannot decide this, because host and device spans
    // share namespaces in both directions: `decode:alloc` / `decode:meta`
    // (`batched_layer.rs`), `moe:route` / `moe:sort` / `moe:submit`
    // (`latent_moe/engine.rs`) and `decode:prep` / `prefill:prep`
    // (`latent_moe/wave.rs`) are host `span()` calls that a `decode:`/`moe:`/
    // `prefill:` prefix would file as device, while every `hc_mix:*` is a
    // `gpu_span` that no prefix in the old list caught and that therefore read as
    // host — 11 s of device time counted against the loop thread.
    //
    // A name absent here is treated as host, which is the safe direction: host is
    // the residual the reader is hunting, so a device span added later inflates it
    // visibly rather than hiding somewhere. Regenerate with:
    //   grep -rhoE 'gpu_span(_if|_phase)?\("[a-z0-9_:]+"' --include=*.rs \
    //     candle-transformers/src candle-conversation/src | grep -oE '"[a-z0-9_:]+"' | sort -u
    const GPU_SPANS: [&str; 48] = [
        "decode:kernel",
        "decode:out_proj",
        "decode:qkv_proj",
        "dn:ffn",
        "dn:mix",
        "dn:out_proj",
        "dn:proj",
        "fwd:dntab",
        "fwd:embed",
        "fwd:head",
        "fwd:layer",
        "fwd:meta",
        "glue:hdr_meta",
        "glue:kernel",
        "hc_mix:gate_mean",
        "hc_mix:inject",
        "hc_mix:lowrank",
        "hc_mix:norm",
        "moe:bucketize",
        "moe:down",
        "moe:gate_up",
        "moe:gather",
        "moe:scatter",
        "moe:silu",
        "prefill:kernel",
        "prefill:out_proj",
        "prefill:pack",
        "prefill:qkv_proj",
        "probe:bounded",
        "probe:cpu",
        "probe:cpu_scope",
        "probe:gpu_matmul",
        "probe:recycle",
        "q4e:attn_decode",
        "q4e:attn_prefill",
        "q4e:attn_verify",
        "q4e:gr_combine",
        "q4e:gr_combine_ffn",
        "q4e:gr_pre",
        "q4e:gr_pre_ffn",
        "q4e:moe_acts",
        "q4e:moe_routed:bulk",
        "q4e:moe_routed:step",
        "q4e:mtp_head",
        "q4e:ple",
        "q4e:qsa_select",
        "verify:fwd",
        "wv:sweep",
    ];
    let is_gpu = |name: &str| GPU_SPANS.contains(&name);

    let mut host: Vec<_> = snap
        .entries
        .iter()
        .filter(|(n, _, _)| !is_gpu(n))
        .cloned()
        .collect();
    let mut gpu: Vec<_> = snap
        .entries
        .iter()
        .filter(|(n, _, _)| is_gpu(n))
        .cloned()
        .collect();
    host.sort_by(|a, b| b.1.total_cmp(&a.1));
    gpu.sort_by(|a, b| b.1.total_cmp(&a.1));

    let table = |label: &str, rows: &[(String, f64, u64)]| {
        if rows.is_empty() {
            return;
        }
        let total: f64 = rows.iter().map(|(_, ms, _)| *ms).sum();
        println!("\n  {label}");
        println!(
            "  {:<34} {:>12} {:>8} {:>10} {:>10}",
            "span", "total ms", "share", "calls", "ms/call"
        );
        for (name, ms, calls) in rows {
            println!(
                "  {:<34} {:>12.1} {:>7.1}% {:>10} {:>10.2}",
                name,
                ms,
                ms / total.max(1e-9) * 100.0,
                calls,
                ms / (*calls).max(1) as f64,
            );
        }
        println!("  {:<34} {:>12.1}", "— sum —", total);
    };
    println!("\n=== {title} ===");
    table("HOST (wall clock on the loop thread)", &host);
    table("DEVICE (CUDA stream time; overlaps the host)", &gpu);
    println!(
        "\n  Spans nest, so shares do not sum to 100%: `loop:housekeeping` contains the\n  \
         `loop:*` parts below it, `loop:compact_kv` contains every `compact:*`, and\n  \
         `wv:sweep` contains the per-layer device spans. Read a parent against its own\n  \
         table's total, and children against their parent."
    );
}

/// Load the model and measure `configs`, returning the params, the rows, and the model.
///
/// The model comes back so a caller can go on to move it into an engine; dropping it here
/// would make the two halves of the combined table need two loads.
fn ladder_rows(
    probe: &Probe,
    device: &Device,
    configs: Vec<TestConfig>,
) -> (
    TestParams,
    Vec<candle_transformers::models::batch_test::utils::TestResults>,
    Box<dyn ManagedBatchedModel + Send>,
) {
    let builder = probe.builder();
    let (model_path, tokenizer_path) = builder.resolve_paths_pub().expect("resolved paths");
    let tokenizer_json = std::fs::read_to_string(&tokenizer_path).expect("tokenizer json");
    println!("Loading {model_path:?} …");
    let loaded = builder
        .load_model(&model_path, device, None)
        .expect("model loaded");
    let model = loaded.model;

    // **The int8 column has to name the mode the model was actually loaded in.** It is a
    // label, not a lever — nothing in the harness reads it — so it is the one the load
    // reported, never worked out again here.
    let loaded_mode = loaded.int8_mode;
    println!("int8 mode the builder loaded at = {loaded_mode:?}");
    let mut params = TestParams::new(10, &tokenizer_json, builder.spec().dialect.clone())
        .expect("TestParams")
        .with_suppress_thinking(true)
        .with_print_outputs(false)
        .with_int8mode(loaded_mode)
        .with_timeout_secs(1200);
    let results = params
        .run_loaded_collect(configs, &*model)
        .expect("the forward ladder ran");
    (params, results, model)
}

/// Run the ladder, and optionally the engine probe, printing one table.
///
/// `ladder` is the model's own forward-gate ladder, given the device because some
/// ladders widen on a big card. The dialect is the preset's — the one the engine
/// formats every turn in — so both halves of the table prompt the model identically.
///
/// Order matters and is not incidental. The ladder runs first, on a borrowed model,
/// because it needs a clean pool — it frees every sequence and releases every empty arena
/// between configs, and it asserts nothing is live before each one. Only then is the model
/// moved into a `ConversationEngine`, which is where fragmentation becomes possible at
/// all. Reversed, the ladder's own gate would fail on KV the engine left behind.
fn ladder_and_engine(
    model_profile: ModelProfile,
    ladder: fn(&Device) -> Vec<TestConfig>,
    with_engine: bool,
) {
    logging();
    let name = model_profile.name;
    let batch = model_profile.batch;
    let speculates = model_profile.speculates;
    let probe = Probe::new(model_profile);
    let device = Device::new_cuda(probe.device).expect("CUDA device");

    // The daemon's own loader, for both row sets. The ceiling is then measured on the
    // model a daemon actually runs, rather than on a differently-configured twin — which
    // is what makes the two rows comparable at all.
    let builder = probe.builder();
    let (model_path, tokenizer_path) = builder.resolve_paths_pub().expect("resolved paths");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).expect("tokenizer");
    let tokenizer_json = std::fs::read_to_string(&tokenizer_path).expect("tokenizer json");
    println!("Loading {model_path:?} …");
    let loaded = builder
        .load_model(&model_path, &device, None)
        .expect("model loaded");
    let model = loaded.model;

    // **Refuse a model that does not speculate the way its profile says.** Speculation
    // is lossless, so a checkpoint whose head failed to load passes every gate here
    // and only the speed is gone — the engine rows would measure plain decode under a
    // speculating model's name. A zero budget with the head present is the profile's
    // batch sitting outside the draft ladder, which is the same wrong answer.
    let budget = model.draft_budget(batch);
    println!(
        "draft budget: width 1 = {}, phase-B width {batch} = {budget}",
        model.draft_budget(1)
    );
    assert_eq!(
        budget > 0,
        speculates,
        "{name}: the profile says speculates = {speculates}, the loaded model drafts \
         {budget} at its phase-B width {batch}"
    );

    // ── Ceiling: the forward ladder, clean slate, `forward_wave` ──────────────
    //
    // **The int8 column has to name the mode the model was actually loaded in.** It is a
    // label, not a lever — nothing in the harness reads it — so it is the one the load
    // reported, never worked out again here.
    let loaded_mode = loaded.int8_mode;
    println!("int8 mode the builder loaded at = {loaded_mode:?}");
    let mut params = TestParams::new(10, &tokenizer_json, builder.spec().dialect.clone())
        .expect("TestParams")
        .with_suppress_thinking(true)
        .with_print_outputs(false)
        .with_int8mode(loaded_mode)
        .with_timeout_secs(1200);
    let mut results = params
        .run_loaded_collect(ladder(&device), &*model)
        .expect("the forward ladder ran");

    // ── The baseline's own gate row, at the baseline's token count ───────────
    //
    // The ladder generates 10 tokens, which is what its rows are validated and
    // calibrated at, and a decode window that short is ~4 speculative steps: one
    // compaction pass landing in it moved the engine's decode by 20%. So the engine
    // baseline decodes `probe.baseline_decode` tokens, and the gate row it is set
    // beside runs here at that same count — same mode, width and prompts.
    let baseline_mode = [
        InferenceMode::C0,
        InferenceMode::C1,
        InferenceMode::C2,
        InferenceMode::C3,
        InferenceMode::C4,
        InferenceMode::C5,
        InferenceMode::C6,
        InferenceMode::C7,
        InferenceMode::C8,
        InferenceMode::C9,
        InferenceMode::C10,
    ]
    .into_iter()
    .find(|m| m.compression_level() == Some(probe.compression_level))
    .expect("the probe's compression level names a C-ladder mode");
    let mut long_params = TestParams::new(
        probe.baseline_decode,
        &tokenizer_json,
        builder.spec().dialect.clone(),
    )
    .expect("TestParams")
    .with_suppress_thinking(true)
    .with_print_outputs(false)
    .with_int8mode(loaded_mode)
    .with_timeout_secs(1200);
    let mut long_results = long_params
        .run_loaded_collect(
            vec![TestConfig {
                mode: baseline_mode,
                use_batched: true,
                num_contexts: probe.baseline_width,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            }],
            &*model,
        )
        .expect("the baseline's gate row ran");

    // ── Delivery: the same model, moved into the engine ──────────────────────
    //
    // Skipped when the caller only wants the ladder. The model is dropped either way; the
    // borrow above has ended, so nothing keeps it alive past this point.
    let outcome = if with_engine {
        let outcome = run_on_model(&probe, &device, tokenizer, model).expect("the probe ran");
        summarise(name, &outcome);
        Some(outcome)
    } else {
        drop(model);
        println!("\n(engine rows skipped — ladder only)");
        None
    };

    // Two tables. The ladder with phase B on the fragmented pool beneath it; then
    // the clean baseline — the gate's C5 ×8 row through the engine — beside that
    // gate row at the baseline's own token count.
    let phase_b = outcome
        .iter()
        .map(|o| o.as_table_row(format!("eng×{}", o.story_total)))
        .collect();
    params
        .with_extra_rows(phase_b)
        .validate_and_print(&mut results)
        .expect("every ladder row validated");
    let baseline = outcome.iter().map(|o| o.baseline.as_table_row()).collect();
    println!(
        "\n=== Clean baseline: gate C{} ×{} against the engine, {} tokens per session ===",
        probe.compression_level, probe.baseline_width, probe.baseline_decode
    );
    long_params
        .with_extra_rows(baseline)
        .validate_and_print(&mut long_results)
        .expect("the baseline's gate row validated");
    if let Some(outcome) = outcome {
        outcome.assert_passed();
    }
}
