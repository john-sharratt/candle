//! **The CPU expert class on the real model** (`docs/cpu_expert_compute.md`
//! §6–§8): accuracy against all-GPU, and decode speed beside it, on one load
//! of the prescribed Flash-Next engine.
//!
//! The split's cap is a live knob, so one engine serves every arm: a cap of 0
//! is all-GPU — the reference, exactly as the engine runs without the class —
//! and the default cap lets the device hand the pool what its costs say the
//! pool does faster. The GPU path is bit-identical whichever experts happen to
//! be resident, so the reference does not move as the cache warms between
//! arms; a second all-GPU arm proves it, exactly.
//!
//! **Accuracy.** Each CPU arm — decode only, prefill only, both — is
//! teacher-forced on the reference's greedy tokens, so every step compares the
//! same position's logits. The gate is on the arm with both
//! ([`KL_GATE`], [`TOP1_GATE`]). The pool must have served layers, or the
//! comparison proves nothing, and every work slot must have named its
//! doorbell when read.
//!
//! **Speed.** Decode runs alternate reference and CPU arms, so the laptop's
//! throttle swings land on both; each arm's runs are reported, not averaged
//! away.
//!
//! ```text
//! cargo test --release -p candle-transformers --features cuda --test cpu_expert_gate -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

use candle::quantized::Int8Mode;
use candle::{Device, IndexOp, Result, Tensor};
use candle_transformers::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel,
};
use candle_transformers::models::expert_lre::cpu_pool::params::shared;
use candle_transformers::models::expert_lre::cpu_pool::service::{
    served_experts, served_ns, served_rows, served_total, stale_slots,
};
use candle_transformers::models::quantized_qwen38_moe::{
    prepared_engine_pack, TOKENIZER_REPO, TOKENIZER_REV,
};
use candle_transformers::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
use hf_hub::{api::sync::Api, Repo, RepoType};

/// Prompt tokens: a real document, long enough that prefill routes experts
/// at prompt-width counts as well as decode's.
const PROMPT_TOKENS: usize = 2048;
/// Decode steps per run.
const STEPS: usize = 256;
/// Decode runs per arm, alternating.
const RUNS: usize = 2;
/// How long the pool's calibration may take after the load returns.
const CALIBRATION_WAIT: Duration = Duration::from_secs(120);
/// The end-to-end gate (`docs/cpu_expert_compute.md` §6): mean KL against
/// all-GPU and top-1 agreement over the prompt's last position and every
/// decode step, with the split on in both phases. Set from what the CPU path
/// measures with the GPU's kernels left as they are — its experts are the
/// GPU's bit for bit but for the device's approximate SiLU division, which
/// rarely flips an int8 code; once one has, the run's logits move. Measured
/// with the split on in both phases: mean KL 4.1e-2, top-1 92.2%.
const KL_GATE: f64 = 0.1;
const TOP1_GATE: f64 = 0.9;

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

/// The prompt: this design's own document, cut to [`PROMPT_TOKENS`].
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
    assert!(
        ids.len() >= PROMPT_TOKENS,
        "the document is {} tokens",
        ids.len()
    );
    ids.truncate(PROMPT_TOKENS);
    Ok(ids)
}

fn row(logits: Tensor) -> Result<Vec<f32>> {
    logits.i(0)?.to_dtype(candle::DType::F32)?.to_vec1::<f32>()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("non-empty logits")
}

fn softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = v.iter().map(|x| ((x - m) as f64).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

/// KL(p_a ‖ p_b) in nats over the full softmax.
fn kl(a: &[f32], b: &[f32]) -> f64 {
    let (pa, pb) = (softmax(a), softmax(b));
    pa.iter()
        .zip(&pb)
        .filter(|(x, _)| **x > 1e-12)
        .map(|(x, y)| x * (x / y.max(1e-30)).ln())
        .sum()
}

/// The split's cap through each phase of a run.
#[derive(Clone, Copy)]
struct Caps {
    prefill: u32,
    decode: u32,
}

impl Caps {
    fn all(cap: u32) -> Self {
        Self::phases(cap, cap)
    }

    fn phases(prefill: u32, decode: u32) -> Self {
        Self { prefill, decode }
    }
}

/// One run: prefill `ids`, then `STEPS` decode steps — greedy, or forced onto
/// `forced` — with the split's cap set per phase. Each step's logits go to
/// `on_step`; returns the tokens fed and the prefill and decode wall times.
fn run(
    model: &Qwen4ExpBatched,
    ids: &[u32],
    forced: Option<&[u32]>,
    caps: Caps,
    mut on_step: impl FnMut(usize, &[f32]),
) -> Result<(Vec<u32>, Duration, Duration)> {
    let n_layers = ManagedBatchedModel::num_layers(model);
    let mut s: BatchedInferenceSession = model.create_batched_session(BatchedConfig::default())?;
    let q = s.create_sequence()?;
    set_cap(caps.prefill);
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
    let mut lg = row(st.logits_owned()?.remove(0))?;
    // The step holds the forward's wave generation: released before the
    // next forward opens one.
    drop(st);
    let prefill = t0.elapsed();
    on_step(0, &lg);
    set_cap(caps.decode);
    let mut fed = Vec::with_capacity(STEPS);
    let t1 = Instant::now();
    for step in 0..STEPS {
        let nx = forced.map_or_else(|| argmax(&lg), |f| f[step]);
        fed.push(nx);
        let x = Tensor::from_vec(vec![nx], (1, 1), &Device::Cpu)?;
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
        on_step(step + 1, &lg);
    }
    Ok((fed, prefill, t1.elapsed()))
}

fn set_cap(cap: u32) {
    shared().lock().unwrap().knobs.max_cpu_assignments = cap;
}

#[test]
#[ignore = "loads the full model and decodes ~1.5K tokens; minutes"]
fn cpu_experts_match_all_gpu_and_report_decode_speed() -> Result<()> {
    // The class starts with the engine, so it is on before the load.
    let cap = {
        let params = shared();
        let mut p = params.lock().unwrap();
        p.enabled = true;
        p.knobs.max_cpu_assignments
    };
    assert!(cap > 0);
    let device = Device::new_cuda(0)?;
    let model = Qwen4ExpBatched::new(engine(&device)?)?;
    let ids = prompt_ids()?;

    // A warm-up, all-GPU: first-touch costs land on neither measured arm, and
    // its first forward releases the pool's calibration.
    let t0 = Instant::now();
    run(&model, &ids, None, Caps::all(0), |_, _| {})?;
    while shared().lock().unwrap().measured.is_none() {
        assert!(
            t0.elapsed() < CALIBRATION_WAIT,
            "the pool never calibrated: the class did not start (see the load's log)"
        );
        sleep(Duration::from_millis(100));
    }
    println!(
        "calibrated {:.1} s after the first forward: {:?}",
        t0.elapsed().as_secs_f64(),
        shared().lock().unwrap().measured
    );

    // The reference: all-GPU, greedy, every step's logits kept.
    let mut reference: Vec<Vec<f32>> = Vec::with_capacity(STEPS + 1);
    let (tokens, ref_prefill, ref_decode) = run(&model, &ids, None, Caps::all(0), |_, lg| {
        reference.push(lg.to_vec())
    })?;

    let n = (STEPS + 1) as f64;
    // A run forced onto the reference's tokens, against the reference.
    struct Versus {
        /// KL at the prompt's last position: the prefill alone.
        kl_prefill: f64,
        kl_mean: f64,
        kl_max: f64,
        top1: f64,
        served: u64,
        /// Assignments and experts the pool computed.
        rows: u64,
        experts: u64,
        prefill: Duration,
        decode: Duration,
    }
    // Every arm starts from the calibrated costs: the online loop's correction
    // from one arm must not decide what the next places.
    let calibrated = shared().lock().unwrap().measured;
    let (layers_before, rows_start, ns_before) = (served_total(), served_rows(), served_ns());
    let versus = |caps: Caps| -> Result<Versus> {
        shared().lock().unwrap().measured = calibrated;
        let (mut kl_sum, mut kl_max, mut agree, mut kl_prefill) = (0f64, 0f64, 0usize, 0f64);
        let before = served_total();
        let (rows_before, experts_before) = (served_rows(), served_experts());
        let (_, prefill, decode) = run(&model, &ids, Some(&tokens), caps, |step, lg| {
            let d = kl(&reference[step], lg);
            if step == 0 {
                kl_prefill = d;
            }
            kl_sum += d;
            kl_max = kl_max.max(d);
            agree += usize::from(argmax(&reference[step]) == argmax(lg));
        })?;
        Ok(Versus {
            kl_prefill,
            kl_mean: kl_sum / n,
            kl_max,
            top1: agree as f64 / n,
            served: served_total() - before,
            rows: served_rows() - rows_before,
            experts: served_experts() - experts_before,
            prefill,
            decode,
        })
    };
    let report = |name: &str, v: &Versus| {
        println!(
            "{name:>26}: KL prefill {:.2e}, mean {:.2e}, max {:.2e} nats; top-1 {:.2}%; \
             the pool: {} layers, {} experts, {} rows",
            v.kl_prefill,
            v.kl_mean,
            v.kl_max,
            100.0 * v.top1,
            v.served,
            v.experts,
            v.rows
        );
    };

    // The control: all-GPU again, after the cache has moved under the first.
    // The GPU path does not depend on residency, so it must be exact; anything
    // the CPU arms show is then the CPU path's.
    let control = versus(Caps::all(0))?;
    report("control, all-GPU", &control);
    assert_eq!(control.kl_max, 0.0, "the all-GPU path moved with residency");
    // Each phase on its own, then both.
    let decode_only = versus(Caps::phases(0, cap))?;
    report("CPU in decode only", &decode_only);
    let prefill_only = versus(Caps::phases(cap, 0))?;
    report("CPU in prefill only", &prefill_only);
    let both = versus(Caps::all(cap))?;
    report("CPU in both", &both);
    {
        let params = shared();
        let p = params.lock().unwrap();
        println!(
            "after the CPU arms: cut {} tokens; measured {:?}",
            p.split().max_n,
            p.measured
        );
    }
    let ns = served_ns() - ns_before;
    let layers = served_total() - layers_before;
    println!(
        "the pool across the CPU arms: {layers} layers, {:.1} µs a layer, {:.1} rows a layer; \
         {} work slots read before they named their doorbell",
        ns as f64 / layers.max(1) as f64 / 1e3,
        (served_rows() - rows_start) as f64 / layers.max(1) as f64,
        stale_slots()
    );
    let Versus {
        kl_mean,
        top1,
        served,
        prefill: cpu_prefill,
        decode: cpu_decode,
        ..
    } = both;
    println!(
        "first runs: prefill {PROMPT_TOKENS} tokens GPU {:.0} ms, CPU {:.0} ms; decode {STEPS} \
         tokens GPU {:.1} t/s, CPU {:.1} t/s",
        ref_prefill.as_secs_f64() * 1e3,
        cpu_prefill.as_secs_f64() * 1e3,
        STEPS as f64 / ref_decode.as_secs_f64(),
        STEPS as f64 / cpu_decode.as_secs_f64(),
    );

    // Speed: alternate arms, greedy, each run reported.
    for r in 0..RUNS {
        for (name, c) in [("all-GPU", 0), ("CPU split", cap)] {
            let (_, prefill, decode) = run(&model, &ids, None, Caps::all(c), |_, _| {})?;
            println!(
                "run {r} {name:>9}: prefill {:.0} t/s, decode {:.1} t/s",
                PROMPT_TOKENS as f64 / prefill.as_secs_f64(),
                STEPS as f64 / decode.as_secs_f64()
            );
        }
    }
    assert!(served > 0, "the pool served no layer: nothing was compared");
    assert_eq!(
        stale_slots(),
        0,
        "a work slot was read before it named its doorbell"
    );
    assert!(
        kl_mean <= KL_GATE,
        "mean KL {kl_mean:.2e} over the {KL_GATE:.0e} gate"
    );
    assert!(
        top1 >= TOP1_GATE,
        "top-1 agreement {:.2}% under the {:.0}% gate",
        100.0 * top1,
        100.0 * TOP1_GATE
    );
    Ok(())
}
