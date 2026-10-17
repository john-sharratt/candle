//! **Where the CPU expert class's time goes, on the real model.** Alternating
//! all-GPU and CPU-split runs of the prescribed Flash-Next engine, prefill and
//! decode measured apart, with the pool's per-layer phases
//! (`cpu_pool::stats`) printed for each CPU run.
//!
//! The online loop is held still (an effectively infinite time constant), so
//! the split keeps handing the pool what calibration says it does faster and
//! the pool's real costs are what get measured — the loop would otherwise
//! price a slow pool out within a run and hide it.
//!
//! ```text
//! cargo test --release -p candle-transformers --features cuda --test cpu_expert_speed -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

use candle::quantized::Int8Mode;
use candle::{Device, IndexOp, Result, Tensor};
use candle_transformers::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
use candle_transformers::models::expert_lre::cpu_pool::params::{shared, Promote};
use candle_transformers::models::expert_lre::cpu_pool::stats::{stats, ClassStats};
use candle_transformers::models::expert_lre::{last_warm_tier_sizing, PipelineStats};
use candle_transformers::models::profile::{
    gpu_drain_blocking, pipeline_snapshot_and_reset, set_gpu_span_period, ProfileSnapshot,
};
use candle_transformers::models::quantized_qwen38_moe::{
    prepared_engine_pack, TOKENIZER_REPO, TOKENIZER_REV,
};
use candle_transformers::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
use hf_hub::{api::sync::Api, Repo, RepoType};

const PROMPT_TOKENS: usize = 1024;
const STEPS: usize = 128;
const RUNS: usize = 2;

fn tokenizer_path(repo: &str, rev: &str) -> Result<PathBuf> {
    let repo = Repo::with_revision(repo.to_string(), RepoType::Model, rev.to_string());
    Api::new()
        .map_err(|e| candle::Error::Msg(format!("hf api: {e}")))?
        .repo(repo)
        .get("tokenizer.json")
        .map_err(|e| candle::Error::Msg(format!("fetch tokenizer.json: {e}")))
}

fn engine(device: &Device) -> Result<Qwen4ExpGpu> {
    let tokenizer_json = |repo: &str, rev: &str| -> Result<String> {
        let p = tokenizer_path(repo, rev)?;
        std::fs::read_to_string(&p).map_err(|e| candle::Error::Msg(format!("read {p:?}: {e}")))
    };
    let pack = prepared_engine_pack(device, Some(Int8Mode::auto(device)), &tokenizer_json)?;
    Qwen4ExpGpu::load(&pack, device)
}

fn prompt_ids() -> Result<Vec<u32>> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer_path(TOKENIZER_REPO, TOKENIZER_REV)?)
        .map_err(|e| candle::Error::Msg(format!("load tokenizer: {e}")))?;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has a parent")
        .join("docs")
        .join("cpu_expert_compute.md");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| candle::Error::Msg(format!("read {}: {e}", path.display())))?;
    let mut ids = tok
        .encode(text, false)
        .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
        .get_ids()
        .to_vec();
    ids.truncate(PROMPT_TOKENS);
    Ok(ids)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("non-empty logits")
}

fn set_cap(cap: u32) {
    shared().lock().unwrap().knobs.max_cpu_assignments = cap;
}

/// Prefill `ids` at `prefill_cap`, then `STEPS` greedy decode steps at
/// `decode_cap`; the prefill and decode wall times, and the expert cache's
/// tallies over the decode.
fn run(
    model: &Qwen4ExpBatched,
    ids: &[u32],
    prefill_cap: u32,
    decode_cap: u32,
) -> Result<(Duration, Duration, PipelineStats)> {
    let experts = &model.engine().experts;
    let n_layers = ManagedBatchedModel::num_layers(model);
    let mut s = model.create_batched_session(BatchedConfig::default())?;
    let q = s.create_sequence()?;
    set_cap(prefill_cap);
    let t0 = Instant::now();
    let t = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &Device::Cpu)?;
    let st = model.forward_wave(
        &mut s,
        &[],
        &[],
        &[q],
        std::slice::from_ref(&t),
        &[],
        &[],
        0,
        n_layers,
        None,
    )?;
    s.advance_sequence(q, ids.len())?;
    s.mark_decode_slot_states(&[q])?;
    let row = |l: Tensor| -> Result<Vec<f32>> { l.i(0)?.to_dtype(candle::DType::F32)?.to_vec1() };
    let mut lg = row(st.logits_owned()?.remove(0))?;
    drop(st);
    let prefill = t0.elapsed();
    experts.reset_expert_stats();
    // The decode's spans alone (a `profile` build; nothing otherwise).
    gpu_drain_blocking();
    let _ = pipeline_snapshot_and_reset();
    set_cap(decode_cap);
    let t1 = Instant::now();
    for _ in 0..STEPS {
        let x = Tensor::from_vec(vec![argmax(&lg)], (1, 1), &Device::Cpu)?;
        let st = model.forward_wave(
            &mut s,
            &[q],
            std::slice::from_ref(&x),
            &[],
            &[],
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        s.advance_sequence(q, 1)?;
        lg = row(st.logits_owned()?.remove(0))?;
    }
    let decode = t1.elapsed();
    gpu_drain_blocking();
    spans(&pipeline_snapshot_and_reset());
    Ok((prefill, decode, experts.expert_stats()))
}

/// A `profile` build's decode spans: the heaviest, ms and calls per step.
fn spans(snap: &ProfileSnapshot) {
    let mut e = snap.entries.clone();
    if e.is_empty() {
        return;
    }
    e.sort_by(|a, b| b.1.total_cmp(&a.1));
    let line: Vec<String> = e
        .iter()
        .take(18)
        .map(|(name, ms, n)| {
            format!(
                "{name} {:.2} ms ×{:.0}",
                ms / STEPS as f64,
                *n as f64 / STEPS as f64
            )
        })
        .collect();
    println!("      spans per step: {}", line.join("; "));
}

fn cache(s: &PipelineStats) {
    let per_step = |n: usize| n as f64 / STEPS as f64;
    println!(
        "    expert cache per decode step: hits {:.1}, misses {:.1} (pinned {:.1}, cold {:.1}), \
         promotions {:.1}, read-ahead claims {:.1} ({:.1} from the pad, {:.0}% routed), \
         evictions {:.1}, staged cold {:.1} ({:.1} paged, {:.1} ahead); link µs {:.0}, \
         window {}, warm {} of {}",
        per_step(s.expert_hits),
        per_step(s.expert_misses),
        per_step(s.worker_pinned),
        per_step(s.worker_cold),
        per_step(s.worker_promotions),
        per_step(s.ahead_claims),
        per_step(s.ahead_pad_claims),
        100.0 * s.predicted_hits as f64 / s.predicted_total.max(1) as f64,
        per_step(s.evictions),
        per_step(s.staged_cold),
        per_step(s.staged_paged),
        per_step(s.staged_speculative),
        s.ahead_link_us,
        s.ahead_window,
        s.warm_slots,
        s.total_experts,
    );
    println!(
        "      stager per step: requests {:.1}, ahead routed {:.1}, pad evictions {:.1}, \
         stalls no-slot {:.1} in-use {:.1}; pack reads {:.1} at {:.2} ms mean ({:.2} max, {} slow); \
         pad {} slots, {} held by read-ahead",
        per_step(s.stage_requests),
        per_step(s.staged_ahead_routed),
        per_step(s.pad_evictions),
        per_step(s.spec_stalls_no_slot),
        per_step(s.spec_stalls_in_use),
        per_step(s.pack_reads.reads),
        s.pack_reads.total_ns as f64 / s.pack_reads.reads.max(1) as f64 / 1e6,
        s.pack_reads.max_ns as f64 / 1e6,
        s.pack_reads.slow,
        s.pad_slots,
        s.ahead_pinned,
    );
    // Residents by depth: the layers in eighths, shallow to deep.
    let eighth = s.resident_rows.len().div_ceil(8).max(1);
    let by_depth: Vec<usize> = s
        .resident_rows
        .chunks(eighth)
        .map(|c| c.iter().sum())
        .collect();
    println!(
        "      resident pad: {} slots; per step {:.2} promoted, {:.2} demoted; by depth {:?}",
        s.resident_capacity,
        per_step(s.resident_promotions),
        per_step(s.resident_demotions),
        by_depth,
    );
    println!(
        "      ring per step: taken {:.1} ({:.1} carried behind the pool), misses unslotted {:.1} \
         ({:.1} decode), victims skipped {:.1}",
        per_step(s.ring_taken),
        per_step(s.carry_claims),
        per_step(s.ring_unslotted),
        per_step(s.decode_unslotted),
        per_step(s.victims_skipped),
    );
    println!(
        "      cold waits per step: {:.1}, {:.0} µs each, {:.1} ms summed",
        per_step(s.cold_waits),
        s.cold_wait_ns as f64 / s.cold_waits.max(1) as f64 / 1e3,
        s.cold_wait_ns as f64 / STEPS as f64 / 1e6,
    );
    println!(
        "      pipeline: lag {:.2} invocations; predicted stagings {:.1} ({:.1} routed), \
         lookahead stagings {:.1}",
        s.pipeline_lag as f64 / s.routed_messages.max(1) as f64,
        per_step(s.staged_predicted),
        per_step(s.staged_predicted_routed),
        per_step(s.staged_speculative.saturating_sub(s.staged_predicted)),
    );
    // The predictors by hop: precision (routed of proposed) and recall (of the
    // routed experts a copy could serve, the share proposed).
    let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
    let hops: Vec<String> = (0..s.lookahead_judged.len())
        .map(|h| {
            format!(
                "+{}: votes {:.0}%/{:.0}% of {:.1}, markov {:.0}%/{:.0}%, selected {:.0}%/{:.0}%",
                h + 1,
                pct(s.lookahead_hits[h], s.lookahead_judged[h]),
                pct(s.lookahead_hits[h], s.lookahead_routed[h]),
                per_step(s.lookahead_judged[h]),
                pct(s.markov_hits[h], s.markov_judged[h]),
                pct(s.markov_hits[h], s.copy_routed[h]),
                pct(s.selected_hits[h], s.selected_judged[h]),
                pct(s.selected_hits[h], s.copy_routed[h]),
            )
        })
        .collect();
    println!("      predictors (precision/recall): {}", hops.join("; "));
    // Victims by backing — the tag index's low bit is "a warm copy backs it".
    let by_backing = |v: &[usize], warm: bool| -> usize {
        v.iter()
            .enumerate()
            .filter(|(i, _)| (i % 2 == 1) == warm)
            .map(|(_, n)| n)
            .sum()
    };
    println!(
        "      victims per step: warm-backed {:.1} ({:.1} routed again), pack-only {:.1} \
         ({:.1} routed again); class costs µs {:.0?}",
        per_step(by_backing(&s.victim_tag_evicted, true)),
        per_step(by_backing(&s.victim_tag_regretted, true)),
        per_step(by_backing(&s.victim_tag_evicted, false)),
        per_step(by_backing(&s.victim_tag_regretted, false)),
        s.eviction_cost_us,
    );
}

fn phases(name: &str, s: &ClassStats) {
    let [parse, compute, cold_wait, publish, wait] = s.per_layer_us();
    let n = s.layers.max(1) as f64;
    println!(
        "    {name}: {} layers, {:.1} experts ({:.1} cold) and {:.1} rows a layer; per layer µs: \
         parse {parse:.1}, compute {compute:.1}, cold wait {cold_wait:.1}, publish {publish:.1}; \
         the device waited {wait:.1}",
        s.layers,
        s.experts as f64 / n,
        s.cold_experts as f64 / n,
        s.rows as f64 / n,
    );
    let [participants, join, work, in_job] = s.per_run();
    println!(
        "      pool runs {:.2} a layer, {participants:.1} participants; last joined at \
         {join:.1} µs; per participant µs: work {work:.1} of {in_job:.1} in the job",
        s.runs as f64 / n,
    );
}

#[test]
#[ignore = "loads the full model; minutes"]
fn where_the_cpu_class_spends_its_time() -> Result<()> {
    let cap = {
        let params = shared();
        let mut p = params.lock().unwrap();
        p.enabled = true;
        // Held still: the split keeps what calibration measured.
        p.knobs.online_tau_ms = 1e12;
        p.knobs.max_cpu_assignments
    };
    // A `profile` build times one forward in four (`set_gpu_span_period`):
    // every forward's spans would slow decode enough to measure another
    // program.
    set_gpu_span_period(4);
    let device = Device::new_cuda(0)?;
    let model = Qwen4ExpBatched::new(engine(&device)?)?;
    let ids = prompt_ids()?;
    let t0 = Instant::now();
    run(&model, &ids, 0, 0)?;
    while shared().lock().unwrap().measured.is_none() {
        assert!(
            t0.elapsed() < Duration::from_secs(120),
            "the pool never calibrated"
        );
        sleep(Duration::from_millis(100));
    }
    {
        let params = shared();
        let p = params.lock().unwrap();
        println!(
            "calibrated: cut {} tokens; {:?}",
            p.split().max_n,
            p.measured
        );
    }
    println!("warm tier: {:?}", last_warm_tier_sizing());
    println!(
        "host RAM now: {:?} available of {:?}",
        candle::vram::available_physical_ram(),
        candle::vram::total_physical_ram()
    );
    {
        let s = model.engine().experts.expert_stats();
        println!(
            "warm slots {} ({} pageable), pad {}, of {} experts",
            s.warm_slots, s.warm_paged_slots, s.pad_slots, s.total_experts
        );
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        println!(
            "zone: {:.2} GiB ({} slots; may move {:.2}–{:.2} GiB), {:.2} GiB resident, {:.2} GiB \
             cedeable",
            gib(s.zone_bytes),
            s.zone_bytes / s.expert_slot_bytes.max(1),
            gib(s.zone_min_bytes),
            gib(s.zone_max_bytes),
            gib(s.resident_vram_bytes),
            gib(s.zone_cedeable_bytes),
        );
    }
    // All-GPU, then the split promoting none, the cold, and every one of the
    // experts its pool serves (§5.11), alternating within each run.
    let arms: [(&str, u32, Promote); 4] = [
        ("all-GPU", 0, Promote::None),
        ("split, promote none", cap, Promote::None),
        ("split, promote cold", cap, Promote::Cold),
        ("split, promote all", cap, Promote::All),
    ];
    // The laptop's CPU throttles as a run goes on, so each run starts at the
    // next arm: no arm always runs hottest.
    for r in 0..RUNS {
        for a in 0..arms.len() {
            let (name, arm_cap, promote) = arms[(r + a) % arms.len()];
            shared().lock().unwrap().knobs.promote = promote;
            let before = stats();
            let (p, d, s) = run(&model, &ids, arm_cap, arm_cap)?;
            let after = stats();
            println!(
                "run {r} {name}: prefill {:.0} t/s, decode {:.2} t/s",
                ids.len() as f64 / p.as_secs_f64(),
                STEPS as f64 / d.as_secs_f64()
            );
            cache(&s);
            if arm_cap > 0 {
                phases("decode-sized", &after[0].since(&before[0]));
                phases("prompt-sized", &after[1].since(&before[1]));
            }
        }
    }
    Ok(())
}
