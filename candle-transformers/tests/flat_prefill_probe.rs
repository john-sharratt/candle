//! **Does the model call a tool when the SAME prompt is prefilled flat?**
//!
//! A temporary diagnostic, not a gate. The daemon's projected path refuses a
//! tool call on roughly half of cold turns, while provenance selects the right
//! tool with a healthy score and the catalog renders into the system prompt
//! correctly (verified against `/v1/conversations/{id}`: `calculator`
//! `selected:true qualified:true`, ~1.5K tokens of catalog, identical bucket
//! composition on passing and failing runs).
//!
//! That leaves two candidates, and this separates them:
//!
//! - **The prompt text.** If the model refuses the same tokens prefilled as one
//!   flat sequence, the composition is what it is and the projection machinery
//!   is exonerated.
//! - **The injection machinery.** A projected turn is assembled from borrowed
//!   K/V, injected QSA index pages, and regenerated glue seams between sections.
//!   A flat prefill has none of those: one contiguous forward, its own index
//!   built from its own hidden states, no seams. If the flat run calls the tool
//!   where the projected run refuses, the fault is in that machinery.
//!
//! Decoding is `argmax`, so a run is a function of the prompt alone — no seed,
//! no sampling variance to argue about.
//!
//! Run with the prompt files produced by `scratchpad/dump_prompt.ps1`:
//!
//! ```text
//! cargo test -p candle-transformers --features cuda --test flat_prefill_probe -- --nocapture --ignored
//! ```
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use candle::{Device, IndexOp, Result, Tensor};
use candle_transformers::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel,
};
use candle_transformers::models::quantized_qwen38_moe::{TOKENIZER_REPO, TOKENIZER_REV};
use candle_transformers::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// The turn terminators. Decoding past either is the model continuing something
/// it already finished, so what it emits there is not a claim about anything.
const IM_END: u32 = 248_046;
const ENDOFTEXT: u32 = 248_044;

/// The checkpoint's own tokenizer, from the same repo + pinned revision the
/// engine's tests use — the GGUF directory carries no `tokenizer.json`.
fn tokenizer() -> Result<tokenizers::Tokenizer> {
    let repo = Repo::with_revision(
        TOKENIZER_REPO.to_string(),
        RepoType::Model,
        TOKENIZER_REV.to_string(),
    );
    let p = Api::new()
        .map_err(|e| candle::Error::Msg(format!("hf api: {e}")))?
        .repo(repo)
        .get("tokenizer.json")
        .map_err(|e| candle::Error::Msg(format!("fetch tokenizer.json: {e}")))?;
    tokenizers::Tokenizer::from_file(&p)
        .map_err(|e| candle::Error::Msg(format!("load tokenizer: {e}")))
}

/// The daemon's own assembled prompt for "hi - how are you?", ending exactly at
/// `<|im_start|>assistant\n` — nothing appended, so each probe below decides for
/// itself what (if anything) opens the assistant turn.
///
/// Read from the workspace root, beside the other `flat_*.txt` fixtures. These
/// call sites used to name an absolute path inside a session-scoped scratch
/// directory, which is deleted when that session ends: every probe here would
/// have started failing on a missing file, and the module docs of
/// `stencil/think.rs` cite this file as the authority for the measurement that
/// set the block opener.
fn daemon_greeting_prompt() -> Result<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has a parent")
        .join("flat_rep1_fixed.txt");
    std::fs::read_to_string(&path)
        .map_err(|e| candle::Error::Msg(format!("read {}: {e}", path.display())))
}

/// How the next token is chosen.
///
/// The daemon SAMPLES — `temp=0.700 top_k=0 top_p=0.900` with a fresh
/// nanosecond seed per turn (`session.rs`: `sampling.seed = SystemTime::now()
/// …as_nanos()`). Comparing a greedy flat run against that measures the
/// sampler as much as the K/V path, so the probe offers both: `Argmax` to ask
/// whether the engine is deterministic at all, and `TopP` to put the flat run
/// on the daemon's own terms.
#[derive(Clone, Copy)]
enum Pick {
    Argmax,
    TopP { temp: f32, top_p: f32, seed: u64 },
}

/// Temperature + nucleus sampling over the raw logits, matching the daemon's
/// dial. `top_k = 0` there means the k-gate is off, so only `top_p` truncates.
fn sample_top_p(logits: &[f32], temp: f32, top_p: f32, rng: &mut StdRng) -> u32 {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    // Softmax over the temperature-scaled logits, max-shifted for stability.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|l| ((l - max) / temp).exp()).collect();
    let sum: f32 = p.iter().sum();
    for v in p.iter_mut() {
        *v /= sum;
    }
    idx.sort_unstable_by(|a, b| p[*b as usize].total_cmp(&p[*a as usize]));

    // Nucleus: keep the shortest prefix whose mass reaches `top_p`.
    let mut cum = 0.0f32;
    let mut cut = idx.len();
    for (i, &t) in idx.iter().enumerate() {
        cum += p[t as usize];
        if cum >= top_p {
            cut = i + 1;
            break;
        }
    }
    let keep = &idx[..cut];
    let mass: f32 = keep.iter().map(|&t| p[t as usize]).sum();
    let mut r = rng.random::<f32>() * mass;
    for &t in keep {
        r -= p[t as usize];
        if r <= 0.0 {
            return t;
        }
    }
    keep[keep.len() - 1]
}

/// Prefill `text` split at explicit boundaries and return the FINAL position's
/// logits — the vector the first sampled token is drawn from.
///
/// No decode, so this isolates the prefill: if two split patterns over the same
/// token ids produce different logits here, the K/V they built differs, and
/// everything downstream is a consequence rather than a cause.
fn prefill_logits(model: &Qwen4ExpBatched, ids: &[u32], splits: &[usize]) -> Result<Vec<f32>> {
    let n_layers = ManagedBatchedModel::num_layers(model);
    let mut session = model.create_batched_session(BatchedConfig::default())?;
    let seq = session.create_sequence()?;
    let mut bounds: Vec<usize> = splits.to_vec();
    bounds.push(ids.len());
    let mut step = None;
    let mut at = 0usize;
    for end in bounds {
        if end <= at {
            continue;
        }
        let part = &ids[at..end];
        let t = Tensor::from_vec(part.to_vec(), (1, part.len()), &Device::Cpu)?;
        step = Some(model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?);
        session.advance_sequence(seq, part.len())?;
        at = end;
    }
    step.expect("non-empty prompt").logits_owned()?[0]
        .i(0)?
        .to_dtype(candle::DType::F32)?
        .to_vec1::<f32>()
}

/// Compare two logit vectors in terms that separate *rounding* from
/// *corruption*.
///
/// A max-|delta| over a 248K vocab is dominated by outliers on tokens nobody
/// would sample, so it cannot tell a different attention tiling's float
/// non-associativity (expected, harmless) from a genuinely different K/V. What
/// distinguishes them is what happens to the part of the distribution the
/// sampler actually sees: the top few tokens' ordering, and the probability
/// mass moved.
struct Cmp {
    max_abs: f32,
    top1_same: bool,
    top5_same: bool,
    /// KL(p_a || p_b) in nats over the full softmax — total distributional
    /// movement, not a single coordinate.
    kl: f32,
    /// Probability the reference assigns its own top token, and what the other
    /// run assigns the SAME token. A large drop is the sampler seeing a
    /// materially different choice.
    p_top_a: f32,
    p_top_b: f32,
}

fn softmax(v: &[f32]) -> Vec<f32> {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut e: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    for x in e.iter_mut() {
        *x /= s;
    }
    e
}

fn top_n(v: &[f32], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|a, b| v[*b].total_cmp(&v[*a]));
    idx.truncate(n);
    idx
}

fn compare(a: &[f32], b: &[f32]) -> Cmp {
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let (pa, pb) = (softmax(a), softmax(b));
    let ta = top_n(a, 5);
    let tb = top_n(b, 5);
    let kl: f32 = pa
        .iter()
        .zip(&pb)
        .filter(|(x, _)| **x > 1e-12)
        .map(|(x, y)| x * (x / y.max(1e-30)).ln())
        .sum();
    Cmp {
        max_abs,
        top1_same: ta[0] == tb[0],
        top5_same: ta == tb,
        kl,
        p_top_a: pa[ta[0]],
        p_top_b: pb[ta[0]],
    }
}

/// Prefill `text` and decode `steps` tokens.
///
/// `chunks` splits the prompt into that many successive prefill forwards on the
/// SAME sequence instead of one contiguous forward. The projected path never
/// prefills a slot in one shot — it injects sealed sections and fires gap-fill
/// runs between them — so this asks whether merely *breaking the prefill into
/// pieces* changes the answer, with no substrate, no borrowed K/V and no
/// injected index pages in play. One chunk reproduces the flat baseline.
fn flat_run(
    model: &Qwen4ExpBatched,
    tok: &tokenizers::Tokenizer,
    text: &str,
    steps: usize,
    pick: Pick,
    chunks: usize,
    cfg: BatchedConfig,
) -> Result<String> {
    let ids: Vec<u32> = tok
        .encode(text, false)
        .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
        .get_ids()
        .to_vec();
    println!("  prompt: {} tokens", ids.len());

    let n_layers = ManagedBatchedModel::num_layers(model);
    let mut session = model.create_batched_session(cfg)?;
    let seq = session.create_sequence()?;

    // Split into `chunks` successive prefill forwards. Only the LAST one's
    // logits matter — the earlier forwards exist to build K/V, exactly as the
    // projected path's gap-fill runs do.
    let n = ids.len();
    let per = n.div_ceil(chunks.max(1));
    let mut step = None;
    let mut at = 0usize;
    while at < n {
        let end = (at + per).min(n);
        let part = &ids[at..end];
        let t = Tensor::from_vec(part.to_vec(), (1, part.len()), &Device::Cpu)?;
        step = Some(model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?);
        session.advance_sequence(seq, part.len())?;
        at = end;
    }
    let step = step.expect("a non-empty prompt produces at least one forward");

    let logits = step.logits_owned()?;
    let m = logits[0].abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    assert!(m.is_finite(), "non-finite logits from the flat prefill");

    // **No RNG exists on the argmax path.** `Option<StdRng>` rather than a
    // seeded-but-unused generator: under `Pick::Argmax` this is `None`, nothing
    // is ever constructed, and `choose` cannot draw from it — so "deterministic"
    // is a property of the code, not a claim about how the value happens to be
    // used. `Tensor::argmax` returns the FIRST maximal index, so even an exact
    // logit tie resolves the same way every run.
    let mut rng: Option<StdRng> = match pick {
        Pick::TopP { seed, .. } => Some(StdRng::seed_from_u64(seed)),
        Pick::Argmax => None,
    };
    let choose = |row: &Tensor, rng: &mut Option<StdRng>| -> Result<u32> {
        match pick {
            Pick::Argmax => row.argmax(0)?.to_scalar::<u32>(),
            Pick::TopP { temp, top_p, .. } => {
                let v = row.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
                let rng = rng.as_mut().expect("TopP implies an RNG");
                Ok(sample_top_p(&v, temp, top_p, rng))
            }
        }
    };

    let mut next = choose(&logits[0].i(0)?, &mut rng)?;
    let mut gen = vec![next];
    for _ in 0..steps {
        if next == IM_END || next == ENDOFTEXT {
            break;
        }
        let t = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
        let step = model.forward_wave(
            &mut session,
            &[seq],
            std::slice::from_ref(&t),
            &[],
            &[],
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, 1)?;
        next = choose(&step.logits_owned()?[0].i(0)?, &mut rng)?;
        gen.push(next);
    }
    tok.decode(&gen, false)
        .map_err(|e| candle::Error::Msg(format!("decode: {e}")))
}

/// Temperature sweep on ONE fixed prompt: the daemon's exact greeting prompt
/// plus the force-emitted `<think>\nOkay,` seed, decoded flat. Ten samples per
/// temperature at top_p 0.9 (the daemon's nucleus). Prints each sample's answer
/// (the text after `</think>`) so the sweet spot — where the forced opener stops
/// derailing into confabulated arithmetic and produces a real greeting — can be
/// read off. `flat_rep1_fixed.txt` is the clean dump of the daemon's assembled
/// prompt for "hi - how are you?" (system prompt + user turn + assistant opener).
/// **CONTROL: no forced opener at all.**
///
/// Prefill stops at `<|im_start|>assistant\n` — exactly where the daemon's
/// prompt ends — and the model generates everything after it, including its own
/// `<think>` if it wants one. Same prompt, same temperature (0.75), same 20
/// seeds as the forced-opener sweep, so this is a PAIRED comparison against
/// that run's 0.75 row (10/20 usable, 6 genuinely correct, 7 empty).
///
/// If quality jumps here, the forced `<think>\nOkay,` is the cause of the
/// confabulated questions, the empty answers, the persona break and the
/// "They said 'Okay'" misattribution. If it does not, the opener is exonerated
/// and the fault is elsewhere in the prompt.
#[test]
#[ignore = "needs the merged engine GGUF and a CUDA device"]
fn no_forced_opener_control() -> Result<()> {
    let merged =
        PathBuf::from(r"D:\models\qwen38-flash-next\Qwen3.8-Flash-Next-Q4KOEXP-merged.gguf");
    if !merged.exists() {
        candle::bail!("merged engine GGUF absent — run prepare_engine_gguf first");
    }
    let device = Device::new_cuda(0)?;
    let mut gpu = Qwen4ExpGpu::load(&merged, &device, candle::quantized::Int8Mode::auto(&device))?;
    gpu.mtp = None;
    gpu.cfg.num_mtp_layers = 0;
    let model = Qwen4ExpBatched::new(gpu)?;
    let tok = tokenizer()?;

    // Ends at `<|im_start|>assistant\n`. NOTHING is appended.
    let text = daemon_greeting_prompt()?;

    const N: usize = 20;
    const TEMP: f32 = 0.75;
    println!("\n=== CONTROL: no forced opener, temp {TEMP}, top_p 0.9, {N} samples ===");
    println!("=== prompt ends at <|im_start|>assistant + LF; model writes the rest ===");
    let (mut opened_think, mut empty_ans) = (0usize, 0usize);
    for r in 0..N {
        let out = flat_run(
            &model,
            &tok,
            &text,
            400,
            Pick::TopP { temp: TEMP, top_p: 0.9, seed: 0xC0FFEE + r as u64 },
            1,
            BatchedConfig::default(),
        )?;
        let has_think = out.contains("<think>");
        if has_think {
            opened_think += 1;
        }
        let think = out.split("</think>").next().unwrap_or("");
        let answer = out.split_once("</think>").map(|(_, a)| a.trim()).unwrap_or(out.trim());
        if answer.len() < 15 {
            empty_ans += 1;
        }
        let th: String = think.replace('\n', " ").chars().take(110).collect();
        let an: String = answer.replace('\n', " ").chars().take(150).collect();
        println!("\n[ctl #{r}] own_think={has_think}");
        println!("   THINK : {th}");
        println!("   ANSWER: {an}");
    }
    println!("\n*** CONTROL temp {TEMP}: opened own <think> {opened_think}/{N} | empty answers {empty_ans}/{N} ***");
    Ok(())
}

/// **The shipped prefill: `<think>` + LF, and nothing else.**
///
/// The stencil opens the thinking block for the model — so the block is always
/// entered and the dial's budget applies — but it stops at the newline instead
/// of also writing the first words of the thought. That distinction is the whole
/// experiment: `<think>\nOkay,` puts *content* in the block, which the model then
/// reads as something in the conversation to interpret (it confabulates a
/// question, quotes its own opener back as the user's words, drops the persona);
/// `<think>\n` puts only *structure* there, leaving the first thought to the
/// model.
///
/// Same prompt, same temperature, same 20 seeds as both baselines, so the three
/// arms are directly comparable:
///
/// | arm | usable | empty |
/// |---|---|---|
/// | `<think>\nOkay,` (old stencil) | 10/20 | 7/20 |
/// | nothing (`no_forced_opener_control`) | 20/20 | 0/20 |
/// | `<think>\n` (this test) | ? | ? |
///
/// The question this answers: does opening the block cost anything on its own,
/// or was the `Okay,` the entire defect? A result near the control means the
/// stencil can keep its structural control over the block for free.
#[test]
#[ignore = "needs the merged engine GGUF and a CUDA device"]
fn think_open_only_prefill() -> Result<()> {
    let merged =
        PathBuf::from(r"D:\models\qwen38-flash-next\Qwen3.8-Flash-Next-Q4KOEXP-merged.gguf");
    if !merged.exists() {
        candle::bail!("merged engine GGUF absent — run prepare_engine_gguf first");
    }
    let device = Device::new_cuda(0)?;
    let mut gpu = Qwen4ExpGpu::load(&merged, &device, candle::quantized::Int8Mode::auto(&device))?;
    gpu.mtp = None;
    gpu.cfg.num_mtp_layers = 0;
    let model = Qwen4ExpBatched::new(gpu)?;
    let tok = tokenizer()?;

    let prompt = daemon_greeting_prompt()?;
    // The new stencil opener: the tag and its newline, nothing more.
    let text = format!("{prompt}<think>\n");

    const N: usize = 20;
    // 0.70 is what the checkpoint's own GGUF declares; 0.75 came out of the
    // temperature sweep run against the broken opener. Both arms run here so the
    // daemon can ship the checkpoint's value unless 0.75 actually earns the
    // divergence.
    const TEMPS: [f32; 2] = [0.70, 0.75];
    for temp in TEMPS {
        println!("\n=== OPEN-ONLY: prefill `<think>` + LF, temp {temp}, top_p 0.9, {N} samples ===");
        let mut empty_ans = 0usize;
        for r in 0..N {
            let out = flat_run(
                &model,
                &tok,
                &text,
                400,
                Pick::TopP { temp, top_p: 0.9, seed: 0xC0FFEE + r as u64 },
                1,
                BatchedConfig::default(),
            )?;
            // The prompt carried `<think>\n`, so everything up to `</think>` is
            // the model's own thought — no opener to strip.
            let think = out.split("</think>").next().unwrap_or("");
            let answer = out.split_once("</think>").map(|(_, a)| a.trim()).unwrap_or(out.trim());
            if answer.len() < 15 {
                empty_ans += 1;
            }
            println!(
                "\n[open t{temp} #{r}] think_chars={} answer_chars={}",
                think.trim().len(),
                answer.len()
            );
            println!("   THINK : {}", think.trim());
            println!("   ANSWER: {answer}");
        }
        println!("\n*** OPEN-ONLY temp {temp}: empty answers {empty_ans}/{N} ***");
    }
    Ok(())
}

/// **Two named, seed-locked failures of the forced `<think>\nOkay,` opener.**
///
/// Found by hand-reading a 0.5→0.8 sweep of the daemon's own greeting prompt
/// ("hi - how are you?"). Both bifurcate at exactly temp 0.625 and then hold
/// identical across higher temperatures, so each is a single deterministic
/// sample rather than a rate:
///
/// - **persona break** (sample seed 0xC0FFEE+1): the answer becomes "I'm Qwen,
///   a large multimodal model developed by…", though the system prompt opens
///   "You are Zen, an AI coding assistant". Stable at 0.625 / 0.65 / 0.675;
///   below 0.625 the same seed answers "I don't have access to personal
///   information" instead (still wrong, but no persona break).
/// - **seed misattribution** (sample seed 0xC0FFEE+17): the think block reads
///   "The user just sent a message. Let me see… They said \"Okay\"" — the model
///   attributes the FORCED OPENER to the user, i.e. it reads its own injected
///   scratch token as the user's turn. Stable at 0.625 / 0.65; below 0.625 the
///   same seed thinks "the user is asking me to help them with something".
///
/// Each case runs twice: WITH the forced opener (reproducing the failure) and
/// WITHOUT it (the control). If a failure disappears in the control arm, the
/// forced opener causes it — which is the whole question. Full text is printed,
/// untruncated, so the prose can actually be read.
#[test]
#[ignore = "needs the merged engine GGUF and a CUDA device"]
fn forced_okay_named_failures() -> Result<()> {
    let merged =
        PathBuf::from(r"D:\models\qwen38-flash-next\Qwen3.8-Flash-Next-Q4KOEXP-merged.gguf");
    if !merged.exists() {
        candle::bail!("merged engine GGUF absent — run prepare_engine_gguf first");
    }
    let device = Device::new_cuda(0)?;
    let mut gpu = Qwen4ExpGpu::load(&merged, &device, candle::quantized::Int8Mode::auto(&device))?;
    gpu.mtp = None;
    gpu.cfg.num_mtp_layers = 0;
    let model = Qwen4ExpBatched::new(gpu)?;
    let tok = tokenizer()?;

    let prompt = daemon_greeting_prompt()?;

    // (label, sample index — the sweep's seed is 0xC0FFEE + index, temp, marker)
    let cases: [(&str, u64, f32, &str); 2] = [
        ("persona-break (expect: NOT 'Qwen')", 1, 0.625, "Qwen"),
        ("seed-misattribution (expect: NOT 'They said')", 17, 0.625, "They said"),
    ];

    for (label, idx, temp, marker) in cases {
        for with_seed in [true, false] {
            let text = if with_seed {
                format!("{prompt}<think>\nOkay,")
            } else {
                prompt.clone()
            };
            let out = flat_run(
                &model,
                &tok,
                &text,
                400,
                Pick::TopP { temp, top_p: 0.9, seed: 0xC0FFEE + idx },
                1,
                BatchedConfig::default(),
            )?;
            let think = out.split("</think>").next().unwrap_or(&out);
            let answer = out.split_once("</think>").map(|(_, a)| a.trim()).unwrap_or("");
            let hit = out.contains(marker);
            let arm = if with_seed { "WITH forced 'Okay,'" } else { "CONTROL (no forced opener)" };
            println!("\n───────────────────────────────────────────────────────────");
            println!("CASE {label}  seed=0xC0FFEE+{idx}  temp={temp}  [{arm}]");
            println!("  marker '{marker}' present: {hit}   <-- true = failure reproduced");
            println!("  THINK  : {}", if with_seed { format!("Okay,{think}") } else { think.to_string() });
            println!("  ANSWER : {answer}");
        }
    }
    Ok(())
}

/// Classify a think block the model produced AFTER the forced `Okay,`.
/// `gen_think` is the decoded text from the first generated token up to
/// `</think>` (the prompt already carried `<think>\nOkay,`).
fn classify_think(gen_think: &str) -> &'static str {
    let t = gen_think.trim();
    let cjk = t.chars().filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c)).count();
    if cjk > 3 {
        "CJK"
    } else if t.len() < 3 {
        "EMPTY(closed on Okay,)"
    } else if t.starts_with(|c: char| c.is_ascii_digit())
        || t.starts_with("10^")
        || t.starts_with("10 ")
        || t.starts_with("10-")
        || t.starts_with("^")
    {
        "MATH-CONFAB"
    } else {
        let low = t.to_lowercase();
        if low.contains("how are you")
            || low.contains("greeting")
            || low.contains("hello")
            || low.contains("assist")
            || low.contains("respond to")
            || low.contains("say")
        {
            "ON-TOPIC"
        } else {
            "OTHER"
        }
    }
}

/// **Does the flat test reproduce the bare-greeting collapse (Problem B)?**
///
/// Run #4 of the daemon's ten samples: "hi - how are you?" with the forced
/// `<think>\nOkay,` seed derailed into "10^6 = 1,000,000" and answered nothing.
/// No tool, no selection — just prompt + seed + sampler. This reproduces it in
/// the flat harness at the daemon's exact profile (temp 0.7, top_p 0.9) and
/// prints the THINK block, not just the answer. Two arms:
///
/// - **contiguous**: prompt + `<think>\nOkay,` prefilled as ONE forward.
/// - **split**: prompt prefilled, then `<think>\nOkay,` prefilled as its OWN
///   forward (a chunk boundary at the seam), the way the daemon force-emits it.
///
/// If contiguous derails like the daemon, Problem B is pure prompt+seed and the
/// flat test is the iteration loop. If contiguous stays clean and only the split
/// derails, the forced-emit boundary is the cause.
#[test]
#[ignore = "needs the merged engine GGUF and a CUDA device"]
fn reproduce_run4_greeting_collapse() -> Result<()> {
    let merged =
        PathBuf::from(r"D:\models\qwen38-flash-next\Qwen3.8-Flash-Next-Q4KOEXP-merged.gguf");
    if !merged.exists() {
        candle::bail!("merged engine GGUF absent — run prepare_engine_gguf first");
    }
    let device = Device::new_cuda(0)?;
    let mut gpu = Qwen4ExpGpu::load(&merged, &device, candle::quantized::Int8Mode::auto(&device))?;
    gpu.mtp = None;
    gpu.cfg.num_mtp_layers = 0;
    let model = Qwen4ExpBatched::new(gpu)?;
    let tok = tokenizer()?;

    let prompt = daemon_greeting_prompt()?;
    const SEED_TEXT: &str = "<think>\nOkay,";

    let p_ids: Vec<u32> = tok.encode(prompt.as_str(), false)
        .map_err(|e| candle::Error::Msg(format!("encode prompt: {e}")))?.get_ids().to_vec();
    let s_ids: Vec<u32> = tok.encode(SEED_TEXT, false)
        .map_err(|e| candle::Error::Msg(format!("encode seed: {e}")))?.get_ids().to_vec();
    let n_layers = ManagedBatchedModel::num_layers(&model);

    // One sample: prefill `segments` in order (each its own forward — the last
    // segment's boundary is where decode begins), then decode at temp 0.7.
    let run = |segments: &[&[u32]], seed: u64| -> Result<String> {
        let mut session = model.create_batched_session(BatchedConfig::default())?;
        let sq = session.create_sequence()?;
        let mut step = None;
        for seg in segments {
            if seg.is_empty() { continue; }
            let t = Tensor::from_vec(seg.to_vec(), (1, seg.len()), &Device::Cpu)?;
            step = Some(model.forward_wave(&mut session, &[], &[], &[sq],
                std::slice::from_ref(&t), &[], &[], 0, n_layers, None)?);
            session.advance_sequence(sq, seg.len())?;
        }
        let step = step.expect("non-empty prompt");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut next = sample_top_p(&step.logits_owned()?[0].i(0)?.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?, 0.7, 0.9, &mut rng);
        let mut gen = vec![next];
        for _ in 0..400 {
            if next == IM_END || next == ENDOFTEXT { break; }
            let t = Tensor::from_vec(vec![next], (1, 1), &Device::Cpu)?;
            let st = model.forward_wave(&mut session, &[sq], std::slice::from_ref(&t),
                &[], &[], &[], &[], 0, n_layers, None)?;
            session.advance_sequence(sq, 1)?;
            next = sample_top_p(&st.logits_owned()?[0].i(0)?.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?, 0.7, 0.9, &mut rng);
            gen.push(next);
        }
        tok.decode(&gen, false).map_err(|e| candle::Error::Msg(format!("decode: {e}")))
    };

    let contig: Vec<u32> = p_ids.iter().chain(s_ids.iter()).copied().collect();
    for (label, segs) in [
        ("CONTIGUOUS (prompt+seed one forward)", vec![contig.as_slice()]),
        ("SPLIT (prompt | seed, seed its own forward)", vec![p_ids.as_slice(), s_ids.as_slice()]),
    ] {
        println!("\n=== {label} — 10 samples, temp 0.7 top_p 0.9 ===");
        let (mut math, mut empty, mut cjk, mut topic, mut other, mut noans) = (0, 0, 0, 0, 0, 0);
        for r in 0..10u64 {
            let out = run(&segs, 0xD00D + r)?;
            let gen_think = out.split("</think>").next().unwrap_or(&out);
            let answer = out.split_once("</think>").map(|(_, a)| a.trim()).unwrap_or("");
            let v = classify_think(gen_think);
            match v { "MATH-CONFAB" => math += 1, "EMPTY(closed on Okay,)" => empty += 1,
                      "CJK" => cjk += 1, "ON-TOPIC" => topic += 1, _ => other += 1 }
            let ans_ok = answer.len() >= 15 && answer.chars().filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c)).count() == 0;
            if !ans_ok { noans += 1; }
            let th: String = gen_think.replace('\n', " ").chars().take(70).collect();
            let an: String = answer.replace('\n', " ").chars().take(50).collect();
            println!("  #{r} think={v:<20} ans_ok={ans_ok} :: Okay,{th} || {an}");
        }
        println!("  *** {label}: think math={math} empty={empty} cjk={cjk} on-topic={topic} other={other} | no-answer={noans}/10 ***");
    }
    Ok(())
}

#[test]
#[ignore = "needs the merged engine GGUF and a CUDA device"]
fn forced_okay_temperature_sweep() -> Result<()> {
    let merged =
        PathBuf::from(r"D:\models\qwen38-flash-next\Qwen3.8-Flash-Next-Q4KOEXP-merged.gguf");
    if !merged.exists() {
        candle::bail!("merged engine GGUF absent — run prepare_engine_gguf first");
    }
    let device = Device::new_cuda(0)?;
    let mut gpu = Qwen4ExpGpu::load(&merged, &device, candle::quantized::Int8Mode::auto(&device))?;
    gpu.mtp = None;
    gpu.cfg.num_mtp_layers = 0;
    let model = Qwen4ExpBatched::new(gpu)?;
    let tok = tokenizer()?;

    let prompt = daemon_greeting_prompt()?;
    // The daemon force-emits this at the start of every assistant turn.
    let text = format!("{prompt}<think>\nOkay,");

    const N: usize = 20;
    // Fine sweep 0.5..=0.8 in 0.025 steps.
    let temps = [
        0.5f32, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8,
    ];
    println!("\n=== forced-Okay temperature sweep (top_p 0.9, {N} samples each) ===");
    for &temp in &temps {
        let mut proper = 0usize;
        let (mut math, mut empty, mut cjk_t, mut topic, mut other) = (0, 0, 0, 0, 0);
        for r in 0..N {
            let pick = if temp == 0.0 {
                Pick::Argmax
            } else {
                Pick::TopP { temp, top_p: 0.9, seed: 0xC0FFEE + r as u64 }
            };
            let out = flat_run(&model, &tok, &text, 400, pick, 1, BatchedConfig::default())?;
            // The think block the model produced after the forced `Okay,` (the
            // prompt already carried `<think>\nOkay,`), and the answer after it.
            let gen_think = out.split("</think>").next().unwrap_or(&out);
            let answer = out.split_once("</think>").map(|(_, a)| a).unwrap_or("").trim();
            let think_verdict = classify_think(gen_think);
            match think_verdict {
                "MATH-CONFAB" => math += 1,
                "EMPTY(closed on Okay,)" => empty += 1,
                "CJK" => cjk_t += 1,
                "ON-TOPIC" => topic += 1,
                _ => other += 1,
            }
            let cjk = answer.chars().filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c)).count();
            let ok = answer.len() >= 15 && cjk == 0;
            if ok {
                proper += 1;
            }
            let th: String = gen_think.replace('\n', " ").chars().take(60).collect();
            let an: String = answer.replace('\n', " ").chars().take(45).collect();
            println!(
                "  [temp {temp:.2} #{r}] ans={} think={think_verdict:<20} :: Okay,{th} || {an}",
                if ok { "OK " } else { "BAD" }
            );
        }
        println!(
            "  *** temp {temp:.2}: answers {proper}/{N} proper | think: math={math} empty={empty} cjk={cjk_t} on-topic={topic} other={other} ***\n"
        );
    }
    Ok(())
}

#[test]
#[ignore = "needs the merged engine GGUF and a CUDA device"]
fn flat_prefill_of_a_refused_prompt() -> Result<()> {
    let merged =
        PathBuf::from(r"D:\models\qwen38-flash-next\Qwen3.8-Flash-Next-Q4KOEXP-merged.gguf");
    if !merged.exists() {
        candle::bail!("merged engine GGUF absent — run prepare_engine_gguf first");
    }
    let device = Device::new_cuda(0)?;
    let mut gpu = Qwen4ExpGpu::load(&merged, &device, candle::quantized::Int8Mode::auto(&device))?;

    // **Is the MTP draft head responsible?** It is a real KV layer (the last),
    // written on every wave where `layer_end == num_layers` — which is every
    // forward this probe makes, prefill included. So it participates even
    // though this probe never runs a speculative verify. Dropping it is the
    // clean A/B: `DROP_MTP=1`-style toggles are forbidden here, so this is a
    // recompile-to-switch constant, flipped by hand for the comparison.
    const DROP_MTP: bool = true;
    println!(
        "  num_mtp_layers={}  mtp_present={}  DROP_MTP={DROP_MTP}",
        gpu.cfg.num_mtp_layers,
        gpu.mtp.is_some()
    );
    if DROP_MTP {
        gpu.mtp = None;
        gpu.cfg.num_mtp_layers = 0;
    }

    let model = Qwen4ExpBatched::new(gpu)?;
    let tok = tokenizer()?;

    // Prompts dumped from conversations whose PROJECTED run refused the call.
    // Cargo runs an integration test with CWD = the CRATE root, not the
    // workspace root, so a bare relative name silently misses and the test
    // "passes" having checked nothing.
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has a parent");
    for name in ["flat_pinned.txt", "flat_live_time.txt", "flat_live_time_t1.txt"] {
        let path = workspace.join(name);
        if !path.exists() {
            println!("SKIP {name}: absent (run scratchpad/dump_prompt.ps1)");
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| candle::Error::Msg(format!("read {}: {e}", path.display())))?;
        println!("\n=== {name} ===");

        // **The measurement that matters.** The daemon's own prompt, byte for
        // byte (extracted from its `context-dump`, not reconstructed — an
        // earlier PowerShell reconstruction round-tripped the body through
        // CP1252 and turned every em-dash into 3 mojibake chars, so the two
        // arms were never the same text). Sampled on the daemon's dial so the
        // rate is comparable to its 2/6.
        {
            const N: usize = 12;
            let mut called = 0usize;
            for r in 0..N {
                let out = flat_run(
                    &model,
                    &tok,
                    &text,
                    220,
                    Pick::TopP {
                        temp: 0.7,
                        top_p: 0.9,
                        seed: 0xA11CE + r as u64,
                    },
                    1,
                    BatchedConfig::default(),
                )?;
                if out.contains("<tool_call>") {
                    called += 1;
                } else {
                    let head: String = out.replace('\n', " ").chars().take(100).collect();
                    println!("    [flat sampled {r} NO TOOL] {head}");
                }
            }
            println!("  *** FLAT (daemon's exact prompt, temp 0.7/top_p 0.9): {called}/{N} ***");
        }

        // **Is piecewise prefill enough to explain the daemon's rate?**
        // The projected path never prefills contiguously: it injects sealed
        // sections and fires gap-fill runs between them. That differs from the
        // flat arm in TWO ways — the prefill is piecewise, and the section K/V
        // is borrowed rather than computed here. This isolates the first: same
        // tokens, computed here, but split the way the projection splits them.
        // If a chunked flat run drops to the daemon's ~2/6, the fault is the
        // piecewise prefill; if it stays near 12/12, it is the borrowed K/V.
        for ch in [2usize, 8, 16] {
            const N: usize = 12;
            let mut called = 0usize;
            for r in 0..N {
                let out = flat_run(
                    &model,
                    &tok,
                    &text,
                    220,
                    Pick::TopP {
                        temp: 0.7,
                        top_p: 0.9,
                        seed: 0xB0B + r as u64,
                    },
                    ch,
                    BatchedConfig::default(),
                )?;
                if out.contains("<tool_call>") {
                    called += 1;
                }
            }
            println!("  *** FLAT chunks={ch} sampled: {called}/{N} ***");
        }

        // **Borrowed K/V — the projection's actual mechanism.**
        //
        // Everything else is now ruled out (quantization, chunking, MTP, QSA
        // selection, recurrent reset, prompt text), so this reproduces what the
        // projected path really does, with no conversation layer: seal the
        // system prompt from one session, Arc-inject it into a FRESH session,
        // then prefill only the user turn on top and decode. If the rate
        // collapses toward the daemon's 2/6, borrowed K/V is the fault.
        {
            // Split where the daemon does: system prompt | user turn + opener.
            let marker = "<|im_start|>user\n";
            let cut = text.rfind(marker).expect("prompt carries a user turn");
            let sys_ids: Vec<u32> = tok
                .encode(&text[..cut], false)
                .map_err(|e| candle::Error::Msg(format!("encode sys: {e}")))?
                .get_ids()
                .to_vec();
            let tail_ids: Vec<u32> = tok
                .encode(&text[cut..], false)
                .map_err(|e| candle::Error::Msg(format!("encode tail: {e}")))?
                .get_ids()
                .to_vec();
            println!(
                "\n  borrowed-KV split: system={} tokens, user tail={} tokens",
                sys_ids.len(),
                tail_ids.len()
            );

            let n_layers = ManagedBatchedModel::num_layers(&model);
            // Seal the system prompt once, exactly as section ingest does.
            //
            // The sealing session must OUTLIVE every injection: a
            // `SealedSequence` holds `Arc<ChunkGid>` refs to chunks, but the
            // ARENA those chunks live in belongs to the session. Dropping it
            // frees the arena under the still-referenced chunks and the next
            // slot-header build fails with "the block table lost its backing".
            // Production keeps them alive through the section residences; here
            // the owner is this binding.
            // ONE session throughout. Arenas are session-scoped, so a snapshot
            // taken in session A cannot be injected into session B — the chunks'
            // Arc refs survive but their arena does not, and the slot-header
            // build fails with "the block table lost its backing". Production
            // has the same constraint: sections live in the conversation's
            // residences and `elevate_to_hot` scatters them into THIS session's
            // backings. So seal from one slot and inject into sibling slots of
            // the same session, which is what the scheduler does.
            let mut s = model.create_batched_session(BatchedConfig::default())?;
            let seal_seq = s.create_sequence()?;
            let sealed = {
                let q = seal_seq;
                let t = Tensor::from_vec(sys_ids.clone(), (1, sys_ids.len()), &Device::Cpu)?;
                model.forward_wave(
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
                s.advance_sequence(q, sys_ids.len())?;
                s.snapshot_sequence_per_layer(q)?
            };

            // `inject_sealed_at_tail` moves ATTENTION K/V only. This stack also
            // carries three per-sequence recurrences (GDN store, PLE state, QSA
            // index) which were built in the sealing slot and belong to it. A
            // slot that receives the K/V but not those runs its recurrent layers
            // — the majority of a hybrid — as if the system prompt were absent.
            // `FORK_RECURRENT` toggles carrying them, so the two runs differ in
            // exactly that.
            const FORK_RECURRENT: bool = true;
            const N: usize = 12;
            let mut called = 0usize;
            for r in 0..N {
                let q = s.create_sequence()?;
                // Arc-clone the sealed system prompt onto a fresh slot, then
                // prefill ONLY the user turn against it — the projection's shape.
                s.inject_sealed_at_tail(q, &sealed)?;
                s.advance_sequence(q, sys_ids.len())?;
                if FORK_RECURRENT {
                    model.fork_recurrent(seal_seq, q)?;
                }
                let t = Tensor::from_vec(tail_ids.clone(), (1, tail_ids.len()), &Device::Cpu)?;
                let step = model.forward_wave(
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
                s.advance_sequence(q, tail_ids.len())?;

                let mut rng = StdRng::seed_from_u64(0xBEEF + r as u64);
                let mut lg: Vec<f32> = step.logits_owned()?[0]
                    .i(0)?
                    .to_dtype(candle::DType::F32)?
                    .to_vec1::<f32>()?;
                let mut gen = Vec::new();
                for _ in 0..220 {
                    let nx = sample_top_p(&lg, 0.7, 0.9, &mut rng);
                    gen.push(nx);
                    if nx == IM_END || nx == ENDOFTEXT {
                        break;
                    }
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
                    lg = st.logits_owned()?[0]
                        .i(0)?
                        .to_dtype(candle::DType::F32)?
                        .to_vec1::<f32>()?;
                }
                let out = tok
                    .decode(&gen, false)
                    .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
                if out.contains("<tool_call>") {
                    called += 1;
                } else if r < 3 {
                    let head: String = out.replace('\n', " ").chars().take(110).collect();
                    println!("    [borrowed {r} NO TOOL] {head}");
                }
            }
            println!("  *** BORROWED-KV sampled: {called}/{N} ***");

            // Rates at n=12 cannot separate 4/12 from 7/12. The logits can:
            // compare the SAME position — right after the user turn, before any
            // sampling — between a contiguous prefill and an injected one. The
            // K/V values are the identical Arc'd chunks, so any difference here
            // is in how they are READ (position mapping, block layout), not in
            // what they contain.
            let all_ids: Vec<u32> = tok
                .encode(text.as_str(), false)
                .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
                .get_ids()
                .to_vec();
            let contiguous_lg = prefill_logits(&model, &all_ids, &[])?;
            for fork in [false, true] {
                let q = s.create_sequence()?;
                s.inject_sealed_at_tail(q, &sealed)?;
                s.advance_sequence(q, sys_ids.len())?;
                if fork {
                    model.fork_recurrent(seal_seq, q)?;
                }
                let t = Tensor::from_vec(tail_ids.clone(), (1, tail_ids.len()), &Device::Cpu)?;
                let step = model.forward_wave(
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
                s.advance_sequence(q, tail_ids.len())?;
                let lg: Vec<f32> = step.logits_owned()?[0]
                    .i(0)?
                    .to_dtype(candle::DType::F32)?
                    .to_vec1::<f32>()?;
                let c = compare(&contiguous_lg, &lg);
                println!(
                    "  borrowed(fork={fork:<5}) vs contiguous: max|d|={:<9.4} top1={:<5} top5={:<5} KL={:.6}",
                    c.max_abs, c.top1_same, c.top5_same, c.kl
                );
            }
        }

        // **The quantization sweep.**
        // The daemon stores a collection member (each tool definition) at
        // `section_compression_policy_member` = C4, fully adaptive, while this
        // probe has been running BF16 uncompressed. If reading the `datetime`
        // definition out of C4 K/V is what loses the call, the flat rate should
        // collapse toward the daemon's 2/6 when the same compression is applied
        // — with everything else held identical.
        for level in [0u8, 4, 5] {
            const N: usize = 12;
            let cfg = BatchedConfig {
                compression_level: Some(level),
                ..BatchedConfig::default()
            };
            let mut called = 0usize;
            for r in 0..N {
                let out = flat_run(
                    &model,
                    &tok,
                    &text,
                    220,
                    Pick::TopP {
                        temp: 0.7,
                        top_p: 0.9,
                        seed: 0xC4C4 + r as u64,
                    },
                    1,
                    cfg.clone(),
                )?;
                if out.contains("<tool_call>") {
                    called += 1;
                }
            }
            println!("  *** FLAT compression C{level} sampled: {called}/{N} ***");
        }

        // **The chunk ladder.** One contiguous prefill is the known-good
        // baseline (8/8 sampled, 3/3 argmax). Each rung splits the SAME tokens
        // into more successive prefill forwards on the same sequence. Nothing
        // else changes — no substrate, no borrowed K/V, no injected index pages
        // — so a rung that flips the answer indicts the prefill seam itself.
        //
        // Argmax throughout: each rung is then one deterministic run, and a
        // difference between rungs cannot be sampling.
        let mut baseline: Option<String> = None;
        for chunks in [1usize, 2, 4, 8, 16, 32] {
            let out = flat_run(
                &model,
                &tok,
                &text,
                220,
                Pick::Argmax,
                chunks,
                BatchedConfig::default(),
            )?;
            let called = out.contains("<tool_call>");
            let same = baseline.as_ref().map(|b| *b == out);
            println!(
                "  chunks={chunks:<3} tool_call={called:<5} same_as_1chunk={}",
                match same {
                    None => "(baseline)".to_string(),
                    Some(s) => s.to_string(),
                }
            );
            if !called || same == Some(false) {
                let head: String = out.replace('\n', " ").chars().take(200).collect();
                println!("      -> {head}");
            }
            if baseline.is_none() {
                baseline = Some(out);
            }
        }
        println!(
            "  ----- 1-chunk argmax output -----\n{}\n  ------------------",
            baseline.unwrap()
        );

        // ── Where does the K/V actually diverge? ─────────────────────────────
        // Same ids, one contiguous prefill vs a single split at position `p`.
        // Pure prefill, no decode: any difference here is the seam.
        let ids: Vec<u32> = tok
            .encode(text.as_str(), false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let n = ids.len();
        let contiguous = prefill_logits(&model, &ids, &[])?;

        // A LOCKSTEP DECODE WALKER USED TO LIVE HERE. It is gone, deliberately.
        //
        // It reported a dramatic "every split diverges at decode step 2 to
        // `</think>`" result that two rounds of analysis were built on. The
        // result was the instrument's, not the model's: the walker held TWO
        // sessions open at once and each called `create_sequence()`, so both got
        // sequence id 0 — and the carried state (`recurrent`, `ple`, `index`) is
        // a `HashMap<usize, _>` keyed by sequence id ON THE MODEL, not per
        // session. The two arms clobbered each other's GDN/PLE/index state on
        // every alternating forward. Running the same splits ONE session at a
        // time gives 12/12 identical first-3 tokens and no divergence at all.
        //
        // Anything that compares two decode trajectories must give the two arms
        // distinct sequence ids, or run them one after the other.

        // Is QSA selection even engaging? At 1266 tokens the prompt is below the
        // 2051-cell identity threshold, so `qsa_rows_selected` should stay flat:
        // every query inside the budget reads densely and no selection is built.
        // If it moves, selection is engaging where it should be inert — and if
        // it does not, the index BUILD (page opens/closes at the break tokens)
        // is still a candidate even though selection is not.
        // **Bug or chaos?** Steps 0 and 1 are forced (p = 1.0000), so step 2 is
        // the first choice with any freedom. If splitting merely perturbs the
        // K/V numerically, the step-2 winner should SCATTER across split
        // positions. If every split lands on the same token, something is
        // biasing that specific choice and it is a defect.
        {
            let n_layers = ManagedBatchedModel::num_layers(&model);
            let three = |sp: &[usize]| -> Result<Vec<u32>> {
                let mut s = model.create_batched_session(BatchedConfig::default())?;
                let q = s.create_sequence()?;
                let mut bounds: Vec<usize> = sp.to_vec();
                bounds.push(ids.len());
                let mut st = None;
                let mut at = 0usize;
                for end in bounds {
                    if end <= at {
                        continue;
                    }
                    let part = &ids[at..end];
                    let t = Tensor::from_vec(part.to_vec(), (1, part.len()), &Device::Cpu)?;
                    st = Some(model.forward_wave(
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
                    )?);
                    s.advance_sequence(q, part.len())?;
                    at = end;
                }
                let mut lg: Vec<f32> = st.expect("non-empty").logits_owned()?[0]
                    .i(0)?
                    .to_dtype(candle::DType::F32)?
                    .to_vec1::<f32>()?;
                let mut got = Vec::new();
                for _ in 0..3 {
                    let t = top_n(&lg, 1)[0] as u32;
                    got.push(t);
                    let x = Tensor::from_vec(vec![t], (1, 1), &Device::Cpu)?;
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
                    lg = st.logits_owned()?[0]
                        .i(0)?
                        .to_dtype(candle::DType::F32)?
                        .to_vec1::<f32>()?;
                }
                Ok(got)
            };
            let base = three(&[])?;
            println!("\n  contiguous first 3 tokens: {base:?}");
            let mut same = 0usize;
            let mut think_close = 0usize;
            let probes = [
                2usize, 37, 100, 211, 333, 450, 512, 700, 800, 911, 1000, 1100,
            ];
            for p in probes {
                if p >= n {
                    continue;
                }
                let g = three(&[p])?;
                if g == base {
                    same += 1;
                } else if g.get(2) == Some(&248069) {
                    think_close += 1;
                }
                println!(
                    "    split@{p:<5} first3={g:?}{}",
                    if g == base { "  (matches)" } else { "" }
                );
            }
            println!(
                "  => {same} matched contiguous, {think_close} ended step2 on </think> (248069)"
            );
        }

        let rows_before = model.qsa_rows_selected();
        println!(
            "  qsa_rows_selected: {} -> {} (delta {})",
            rows_before,
            model.qsa_rows_selected(),
            model.qsa_rows_selected() - rows_before
        );

        // ── Is the split run seeing only its LAST chunk? ─────────────────────
        // This stack carries three recurrences per sequence (GDN store, PLE
        // state, QSA index). If a second prefill forward resets them instead of
        // carrying, the split run's recurrent path sees only the tokens in its
        // final chunk — while attention still sees everything, because that
        // lives in the paged cache. The discriminator: compare split@p against
        // a contiguous prefill of ids[p..] ALONE. Matching that, rather than the
        // full contiguous run, is the signature of a reset.
        {
            let p = 633usize;
            let full = prefill_logits(&model, &ids, &[])?;
            let split = prefill_logits(&model, &ids, &[p])?;
            let tail_only = prefill_logits(&model, &ids[p..], &[])?;
            let vs_full = compare(&full, &split);
            let vs_tail = compare(&tail_only, &split);
            println!(
                "\n  split@{p} vs FULL      : max|d|={:.5} top1={} top5={}",
                vs_full.max_abs, vs_full.top1_same, vs_full.top5_same
            );
            println!(
                "  split@{p} vs TAIL-ONLY : max|d|={:.5} top1={} top5={}",
                vs_tail.max_abs, vs_tail.top1_same, vs_tail.top5_same
            );
            println!("  (closer to TAIL-ONLY would mean the recurrences were reset)");
        }

        let report = |label: String, c: Cmp| {
            println!(
                "  {label:<26} max|d|={:<9.5} top1={:<5} top5={:<5} KL={:<10.6} p(top)={:.4}->{:.4}",
                c.max_abs, c.top1_same, c.top5_same, c.kl, c.p_top_a, c.p_top_b
            );
        };

        let again = prefill_logits(&model, &ids, &[])?;
        report("contiguous vs itself".into(), compare(&contiguous, &again));

        for p in [1usize, 8, 64, 256, 633, n - 64, n - 8, n - 1] {
            if p == 0 || p >= n {
                continue;
            }
            let split = prefill_logits(&model, &ids, &[p])?;
            report(format!("split@{p}"), compare(&contiguous, &split));
        }
    }
    Ok(())
}

/// **What does the padding hole actually cost, in tool calls?**
///
/// Every other measurement here is a logit delta or a text diff. This one is the
/// behaviour the daemon is judged on, at the daemon's own split, with the two
/// arms sharing seeds so the sampler is not part of the comparison:
///
/// - **seam** — the system prompt and the user turn prefilled as two forwards on
///   one slot. This is the right control: it has the daemon's piecewise prefill
///   but computes its own K/V.
/// - **injected** — the same split with the system prompt borrowed, which is
///   what a projected turn does. The system prompt is 1225 tokens, not a
///   multiple of 32, so this arm carries the padding hole.
///
/// The earlier probe compared the injected arm against a CONTIGUOUS prefill,
/// which charges the injection for the seam's cost as well as its own. If the
/// two arms here come out level, the hole is not what loses the call and the
/// alignment work would be wasted; if the injected arm sits well below, the hole
/// is worth removing.
#[test]
#[ignore = "needs the merged engine GGUF and a CUDA device"]
fn what_the_injection_hole_costs_at_the_daemons_own_split() -> Result<()> {
    let merged =
        PathBuf::from(r"D:\models\qwen38-flash-next\Qwen3.8-Flash-Next-Q4KOEXP-merged.gguf");
    if !merged.exists() {
        candle::bail!("merged engine GGUF absent — run prepare_engine_gguf first");
    }
    let device = Device::new_cuda(0)?;
    let mut gpu = Qwen4ExpGpu::load(&merged, &device, candle::quantized::Int8Mode::auto(&device))?;
    gpu.mtp = None;
    gpu.cfg.num_mtp_layers = 0;
    let model = Qwen4ExpBatched::new(gpu)?;
    let tok = tokenizer()?;

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has a parent");
    let text = std::fs::read_to_string(workspace.join("flat_pinned.txt"))
        .map_err(|e| candle::Error::Msg(format!("read flat_pinned.txt: {e}")))?;

    // The daemon's split: everything before the user turn is the borrowed part.
    let marker = "<|im_start|>user\n";
    let cut_at = text.rfind(marker).expect("prompt carries a user turn");
    let sys: Vec<u32> = tok
        .encode(&text[..cut_at], false)
        .map_err(|e| candle::Error::Msg(format!("encode sys: {e}")))?
        .get_ids()
        .to_vec();
    let tail: Vec<u32> = tok
        .encode(&text[cut_at..], false)
        .map_err(|e| candle::Error::Msg(format!("encode tail: {e}")))?
        .get_ids()
        .to_vec();
    const CHUNK: usize = 32;
    println!(
        "system {} tokens ({} past a chunk boundary), user tail {} tokens",
        sys.len(),
        sys.len() % CHUNK,
        tail.len()
    );

    const N: usize = 10;
    let n_layers = ManagedBatchedModel::num_layers(&model);
    let mut s = model.create_batched_session(BatchedConfig::default())?;
    let donor = s.create_sequence()?;
    prefill_into(&model, &mut s, donor, &sys)?;
    let sealed = s.snapshot_sequence_per_layer(donor)?;

    // **Are the two arms even different?** With the same seed, identical logits
    // decode to identical text, so any gap in the rates below means the arms
    // part company before the sampler — and this says by how much, and whether
    // the layouts agree, without waiting for twenty generations to disagree.
    {
        let a = s.create_sequence()?;
        prefill_into(&model, &mut s, a, &sys)?;
        let la = prefill_into(&model, &mut s, a, &tail)?;
        let b = s.create_sequence()?;
        s.inject_sealed_at_tail(b, &sealed)?;
        s.advance_sequence(b, sys.len())?;
        model.fork_recurrent(donor, b)?;
        // What the scheduler does between an inject and the prefill that
        // follows it (`push_empty_if_sealed`). Without it this arm takes a path
        // no projected turn takes, and a probe that skips it cannot see what the
        // production path does with the shared tail.
        s.push_empty_writer_chunk(b)?;
        let lb = prefill_into(&model, &mut s, b, &tail)?;
        let total = sys.len() + tail.len();
        let (lay_a, lay_b) = (
            s.provenance_chunk_layout(a, total),
            s.provenance_chunk_layout(b, total),
        );
        let tail_of = |v: &[(u16, u16, usize)]| -> String {
            v.iter()
                .skip(v.len().saturating_sub(3))
                .map(|(o, l, cum)| format!("(o{o},l{l},@{cum})"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!("  seam     blocks={} tail={}", lay_a.len(), tail_of(&lay_a));
        println!("  injected blocks={} tail={}", lay_b.len(), tail_of(&lay_b));
        let d = compare(&la, &lb);
        println!(
            "  injected vs seam at the prefill boundary: max|d|={:.6} KL={:.6} top1={}",
            d.max_abs, d.kl, d.top1_same
        );
    }

    // Same seeds down both arms, and the arms INTERLEAVED rather than run as
    // two blocks. Running all of one arm first is not a controlled comparison:
    // by the time the second arm starts, the session is holding ten more live
    // slots of history than the first arm ever saw, so a rate gap could be the
    // session's state rather than the arms'. Alternating puts each pair in the
    // same conditions, and comparing the generated TEXT per seed is a far
    // sharper instrument than comparing two rates out of ten.
    // **Three arms, and the third is the one that makes the other two mean
    // anything.** `seam` and `seam2` are the SAME construction run twice: same
    // tokens, same seed, both computing their own K/V in place. They must agree,
    // or a disagreement between `seam` and `injected` says nothing about the
    // injection. Every rate comparison in this file before this control existed
    // was assuming an answer to that question rather than measuring it.
    let mut called = [0usize; 3];
    let mut same_text = 0usize;
    let mut same_control = 0usize;
    for r in 0..N {
        let mut outs: Vec<String> = Vec::with_capacity(3);
        let mut first: Vec<Vec<f32>> = Vec::with_capacity(3);
        for borrowed in [false, false, true] {
            let q = s.create_sequence()?;
            if borrowed {
                s.inject_sealed_at_tail(q, &sealed)?;
                s.advance_sequence(q, sys.len())?;
                model.fork_recurrent(donor, q)?;
                s.push_empty_writer_chunk(q)?;
            } else {
                prefill_into(&model, &mut s, q, &sys)?;
            }
            let mut lg = prefill_into(&model, &mut s, q, &tail)?;
            first.push(lg.clone());

            let mut rng = StdRng::seed_from_u64(0x5EED + r as u64);
            let mut gen = Vec::new();
            for _ in 0..220 {
                let nx = sample_top_p(&lg, 0.7, 0.9, &mut rng);
                gen.push(nx);
                if nx == IM_END || nx == ENDOFTEXT {
                    break;
                }
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
                lg = st.logits_owned()?[0]
                    .i(0)?
                    .to_dtype(candle::DType::F32)?
                    .to_vec1::<f32>()?;
            }
            let out = tok
                .decode(&gen, false)
                .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
            let slot = outs.len();
            called[slot] += usize::from(out.contains("<tool_call>"));
            outs.push(out);
        }
        let ctl_agrees = outs[0] == outs[1];
        let inj_agrees = outs[0] == outs[2];
        same_control += usize::from(ctl_agrees);
        same_text += usize::from(inj_agrees);
        // The prefill logits are the arms' last shared quantity. Identical
        // logits and a differing generation means the decode parted them;
        // differing logits means the injection did, in THIS rep — which is not
        // the same claim as the one-off check above, because a copy that is
        // exact the first time can stop being exact once something has written
        // through it.
        println!(
            "  seed {r:<2} tools seam={:<5} seam2={:<5} inj={:<5} | seam==seam2 {ctl_agrees:<5} \
             seam==inj {inj_agrees:<5} | prefill max|d| seam2={:.6} inj={:.6}",
            outs[0].contains("<tool_call>"),
            outs[1].contains("<tool_call>"),
            outs[2].contains("<tool_call>"),
            compare(&first[0], &first[1]).max_abs,
            compare(&first[0], &first[2]).max_abs
        );
    }
    println!(
        "\ntool calls: seam {}/{N}  seam2 {}/{N}  injected {}/{N}",
        called[0], called[1], called[2]
    );
    println!(
        "identical generations: seam==seam2 {same_control}/{N}, seam==injected {same_text}/{N}"
    );
    Ok(())
}

/// **Which layer does the crossing first change?**
///
/// Two slots holding bit-identical K/V decode identically until one fills its
/// writer chunk. This runs the very forward where that happens as a layer-by-
/// layer sweep on both arms, carrying the residual stream between slices, and
/// reports the first layer whose residual differs.
///
/// The answer partitions the search. A difference present at the first attention
/// layer is the paged read or the K/V write. A difference that appears only at a
/// later layer, with the attention layers before it identical, is downstream —
/// an expert-routing flip amplifying a tiny input change, which would say the
/// crossing is numerically benign and the model merely sits on a knife edge.
#[test]
#[ignore = "needs the merged engine GGUF and a CUDA device"]
fn which_layer_a_chunk_boundary_crossing_first_changes() -> Result<()> {
    let merged =
        PathBuf::from(r"D:\models\qwen38-flash-next\Qwen3.8-Flash-Next-Q4KOEXP-merged.gguf");
    if !merged.exists() {
        candle::bail!("merged engine GGUF absent — run prepare_engine_gguf first");
    }
    let device = Device::new_cuda(0)?;
    let mut gpu = Qwen4ExpGpu::load(&merged, &device, candle::quantized::Int8Mode::auto(&device))?;
    gpu.mtp = None;
    gpu.cfg.num_mtp_layers = 0;
    let model = Qwen4ExpBatched::new(gpu)?;
    let tok = tokenizer()?;

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has a parent");
    let text = std::fs::read_to_string(workspace.join("flat_pinned.txt"))
        .map_err(|e| candle::Error::Msg(format!("read flat_pinned.txt: {e}")))?;
    let ids: Vec<u32> = tok
        .encode(text.as_str(), false)
        .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
        .get_ids()
        .to_vec();
    let n = ids.len();
    let n_layers = ManagedBatchedModel::num_layers(&model);

    // The daemon's own split, where the tail is short enough that the copied
    // trailing chunk is still the slot's decode writer. The two arms agree on
    // the layout and on the prefill logits to the bit and part company on the
    // FIRST decode forward, so no steps are walked before the sweep.
    let c = n - 14;
    let pre_steps = 0usize;
    println!(
        "prompt {n}, cut {c}: tail {} tokens, {pre_steps} steps before the sweep",
        n - c
    );

    let mut s = model.create_batched_session(BatchedConfig::default())?;
    let donor = s.create_sequence()?;
    prefill_into(&model, &mut s, donor, &ids[..c])?;
    let sealed = s.snapshot_sequence_per_layer(donor)?;

    let ctl = s.create_sequence()?;
    prefill_into(&model, &mut s, ctl, &ids[..c])?;
    let mut lg = prefill_into(&model, &mut s, ctl, &ids[c..])?;

    let inj = s.create_sequence()?;
    s.inject_sealed_at_tail(inj, &sealed)?;
    s.advance_sequence(inj, c)?;
    model.fork_recurrent(donor, inj)?;
    s.push_empty_writer_chunk(inj)?;
    prefill_into(&model, &mut s, inj, &ids[c..])?;

    // Walk both arms forward on a shared token stream to the step BEFORE the
    // crossing. They are bit-identical here, which the last comparison in the
    // sweep below re-establishes rather than assumes.
    let mut tokens = Vec::new();
    for _ in 0..pre_steps {
        let nx = top_n(&lg, 1)[0] as u32;
        tokens.push(nx);
        let x = Tensor::from_vec(vec![nx], (1, 1), &Device::Cpu)?;
        let mut last = None;
        for slot in [ctl, inj] {
            let st = model.forward_wave(
                &mut s,
                &[slot],
                std::slice::from_ref(&x),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            s.advance_sequence(slot, 1)?;
            last = Some(
                st.logits_owned()?[0]
                    .i(0)?
                    .to_dtype(candle::DType::F32)?
                    .to_vec1::<f32>()?,
            );
        }
        lg = last.expect("two slots stepped");
    }

    // The crossing forward, one layer at a time. The residual stream is carried
    // between slices and copied off the forward span each time — the span is
    // reclaimed when its guard drops, so a borrowed tensor would dangle into the
    // next slice's own span.
    let nx = top_n(&lg, 1)[0] as u32;
    let x = Tensor::from_vec(vec![nx], (1, 1), &Device::Cpu)?;
    println!("\ncrossing forward (token {nx}), layer by layer:");
    let mut residuals: Vec<Vec<Vec<f32>>> = Vec::new();
    for slot in [ctl, inj] {
        let mut per_layer = Vec::with_capacity(n_layers);
        let mut carry: Option<Tensor> = None;
        for l in 0..n_layers {
            let st = model.forward_wave(
                &mut s,
                &[slot],
                std::slice::from_ref(&x),
                &[],
                &[],
                &[],
                &[],
                l,
                l + 1,
                carry.take(),
            )?;
            match &st.residual {
                Some(t) => {
                    let owned = t.copy()?;
                    per_layer.push(
                        owned
                            .flatten_all()?
                            .to_dtype(candle::DType::F32)?
                            .to_vec1::<f32>()?,
                    );
                    carry = Some(owned);
                }
                None => {
                    // The final slice runs the head instead of emitting a
                    // residual; record the logits so the sweep ends on the
                    // quantity the sampler actually sees.
                    per_layer.push(
                        st.logits_owned()?[0]
                            .i(0)?
                            .to_dtype(candle::DType::F32)?
                            .to_vec1::<f32>()?,
                    );
                }
            }
        }
        s.advance_sequence(slot, 1)?;
        residuals.push(per_layer);
    }

    let mut first_bad = None;
    for (l, (a, b)) in residuals[0].iter().zip(residuals[1].iter()).enumerate() {
        let d = a
            .iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        if d > 0.0 && first_bad.is_none() {
            first_bad = Some(l);
        }
        if d > 0.0 || l + 1 == n_layers || first_bad.is_none() {
            println!("  layer {l:<3} max|d| = {d:.6}");
        }
    }
    match first_bad {
        Some(l) => println!("\nFIRST LAYER THAT DIFFERS: {l}"),
        None => println!("\nno layer differs — the crossing did not change this forward"),
    }
    Ok(())
}

/// **Does crossing a 32-token chunk boundary change the answer?**
///
/// [`borrowed_kv_against_the_same_seam_computed_in_place`] shows that two slots
/// holding bit-identical K/V at identical positions decode identically until one
/// of them fills its writer chunk and pushes a new one — and that the step they
/// part company on is exactly one past whichever crosses first. That says the
/// crossing perturbs the result, but not which side of it is right, because both
/// arms are decodes and neither is a reference.
///
/// This supplies the reference. One sequence decodes normally while, at every
/// step, a FRESH sequence prefills the same tokens contiguously — no crossing,
/// no writer chunk, the whole context computed in one forward. The prompt length
/// is chosen so the crossing lands early in the window, with a second length
/// whose crossing lands outside it as the control.
///
/// Prefill and decode are different kernels, so the comparison has a noise
/// floor; the floor is what the pre-crossing steps measure. A crossing that is
/// merely a different summation order stays in it. A step change out of it says
/// the decode reads something different after the push, and the sign of the
/// change says whether the crossing broke a good state or repaired a bad one.
#[test]
#[ignore = "needs the merged engine GGUF and a CUDA device"]
fn a_chunk_boundary_crossing_during_decode_against_a_fresh_prefill() -> Result<()> {
    let merged =
        PathBuf::from(r"D:\models\qwen38-flash-next\Qwen3.8-Flash-Next-Q4KOEXP-merged.gguf");
    if !merged.exists() {
        candle::bail!("merged engine GGUF absent — run prepare_engine_gguf first");
    }
    let device = Device::new_cuda(0)?;
    let mut gpu = Qwen4ExpGpu::load(&merged, &device, candle::quantized::Int8Mode::auto(&device))?;
    gpu.mtp = None;
    gpu.cfg.num_mtp_layers = 0;
    let model = Qwen4ExpBatched::new(gpu)?;
    let tok = tokenizer()?;

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has a parent");
    let path = workspace.join("flat_pinned.txt");
    if !path.exists() {
        candle::bail!("flat_pinned.txt absent — extract the daemon's prompt first");
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| candle::Error::Msg(format!("read {}: {e}", path.display())))?;
    let all: Vec<u32> = tok
        .encode(text.as_str(), false)
        .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
        .get_ids()
        .to_vec();

    const CHUNK: usize = 32;
    const STEPS: usize = 12;
    let n_layers = ManagedBatchedModel::num_layers(&model);

    // Two prompt lengths: one whose writer chunk is nearly full (crosses inside
    // the window) and one that has just started a chunk (crosses well outside
    // it). Both are prefixes of the same prompt, so nothing but the length —
    // and therefore the phase — differs.
    let base = all.len() / CHUNK * CHUNK;
    for p_len in [base - 2, base - 20] {
        let ids = &all[..p_len];
        let crossing = CHUNK - p_len % CHUNK;
        println!(
            "\n── prompt {p_len} tokens: writer holds {}, crosses at decode step {crossing} \
             ({}) ──",
            p_len % CHUNK,
            if crossing < STEPS {
                "INSIDE the window"
            } else {
                "outside the window"
            }
        );

        // One session throughout, distinct sequence ids: the carried
        // recurrences are keyed by sequence id on the MODEL, so two sessions
        // would both start at id 0 and overwrite each other's state.
        let mut s = model.create_batched_session(BatchedConfig::default())?;
        let live = s.create_sequence()?;
        let mut lg = prefill_into(&model, &mut s, live, ids)?;

        let mut gen: Vec<u32> = Vec::new();
        for step in 0..STEPS {
            // The reference: this exact context, computed in one forward.
            let mut full = ids.to_vec();
            full.extend_from_slice(&gen);
            let refq = s.create_sequence()?;
            let reference = prefill_into(&model, &mut s, refq, &full)?;
            let d = compare(&reference, &lg);
            println!(
                "  step {step:<3}{} decode vs fresh prefill: max|d|={:<9.4} KL={:<10.6} top1={}",
                if step == crossing {
                    " *CROSS*"
                } else {
                    "        "
                },
                d.max_abs,
                d.kl,
                d.top1_same
            );

            let nx = top_n(&lg, 1)[0] as u32;
            gen.push(nx);
            if nx == IM_END || nx == ENDOFTEXT {
                break;
            }
            let x = Tensor::from_vec(vec![nx], (1, 1), &Device::Cpu)?;
            let st = model.forward_wave(
                &mut s,
                &[live],
                std::slice::from_ref(&x),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            s.advance_sequence(live, 1)?;
            lg = st.logits_owned()?[0]
                .i(0)?
                .to_dtype(candle::DType::F32)?
                .to_vec1::<f32>()?;
        }
    }
    Ok(())
}

/// Prefill `part` onto `(session, q)` and return the last position's logits.
fn prefill_into(
    model: &Qwen4ExpBatched,
    s: &mut BatchedInferenceSession,
    q: usize,
    part: &[u32],
) -> Result<Vec<f32>> {
    let n_layers = ManagedBatchedModel::num_layers(model);
    let t = Tensor::from_vec(part.to_vec(), (1, part.len()), &Device::Cpu)?;
    let st = model.forward_wave(
        s,
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
    s.advance_sequence(q, part.len())?;
    // What the scheduler does after every prefill (`prefill.rs`): the forward
    // wrote K/V without the decode kernel's self-increment, so the cached
    // decode slot buffer's lengths are re-serialised before the next decode.
    // Without it the decode reads the pre-prefill lengths — a header
    // production never decodes with.
    s.refresh_decode_slot_state(q)?;
    st.logits_owned()?[0]
        .i(0)?
        .to_dtype(candle::DType::F32)?
        .to_vec1::<f32>()
}

/// Greedy-decode `(session, q)` forward from an already-computed logit row.
fn argmax_continue(
    model: &Qwen4ExpBatched,
    tok: &tokenizers::Tokenizer,
    s: &mut BatchedInferenceSession,
    q: usize,
    first: &[f32],
    steps: usize,
) -> Result<String> {
    let n_layers = ManagedBatchedModel::num_layers(model);
    let mut lg = first.to_vec();
    let mut gen = Vec::new();
    for _ in 0..steps {
        let nx = top_n(&lg, 1)[0] as u32;
        gen.push(nx);
        if nx == IM_END || nx == ENDOFTEXT {
            break;
        }
        let x = Tensor::from_vec(vec![nx], (1, 1), &Device::Cpu)?;
        let st = model.forward_wave(
            s,
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
        lg = st.logits_owned()?[0]
            .i(0)?
            .to_dtype(candle::DType::F32)?
            .to_vec1::<f32>()?;
    }
    tok.decode(&gen, false)
        .map_err(|e| candle::Error::Msg(format!("decode: {e}")))
}

/// **Borrowed K/V measured against the RIGHT control.**
///
/// The first probe compared an injected prefix against a *contiguous* prefill,
/// which conflates two changes: the prefill is now piecewise AND the prefix is
/// borrowed. The chunked ladder shows piecewise prefill alone already costs
/// something, so that comparison cannot attribute the rest.
///
/// The control here is the same seam computed in place: slot `S` prefills
/// `ids[..c]` then `ids[c..]` as two forwards on its own slot, and slot `Y`
/// injects a sealed `ids[..c]` and prefills `ids[c..]` on top. Both have the
/// break at exactly `c`; the ONLY difference is where the prefix's K/V came
/// from. Any divergence between them is the injection's.
///
/// And the cut is swept across a 32-token chunk boundary. A sealed prefix whose
/// length is not a multiple of `CHUNK_SIZE` ends in a PARTIAL chunk, and
/// `inject_sealed_at_tail` pushes the writer past it (`set_writer_start_idx`),
/// so the tail's tokens land in a fresh chunk while the in-place control packs
/// them into the partial one. The physical layouts differ; the logical windows
/// are supposed to agree. Aligned cuts remove that difference entirely, so:
///
/// - divergence at BOTH aligned and unaligned cuts ⇒ the fault is in the
///   borrowed content or the recurrences, not the layout;
/// - divergence only at unaligned cuts ⇒ it is the partial-chunk seam, and the
///   position/window derivation over a padded chunk is where to look.
#[test]
#[ignore = "needs the merged engine GGUF and a CUDA device"]
fn borrowed_kv_against_the_same_seam_computed_in_place() -> Result<()> {
    let merged =
        PathBuf::from(r"D:\models\qwen38-flash-next\Qwen3.8-Flash-Next-Q4KOEXP-merged.gguf");
    if !merged.exists() {
        candle::bail!("merged engine GGUF absent — run prepare_engine_gguf first");
    }
    let device = Device::new_cuda(0)?;
    let mut gpu = Qwen4ExpGpu::load(&merged, &device, candle::quantized::Int8Mode::auto(&device))?;
    // Same reason as the first probe: the MTP head is a real KV layer written on
    // every wave, so it participates even without a speculative verify. Held out
    // so a difference here is the trunk's.
    gpu.mtp = None;
    gpu.cfg.num_mtp_layers = 0;
    let model = Qwen4ExpBatched::new(gpu)?;
    let tok = tokenizer()?;

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has a parent");
    let path = workspace.join("flat_pinned.txt");
    if !path.exists() {
        candle::bail!("flat_pinned.txt absent — extract the daemon's prompt first");
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| candle::Error::Msg(format!("read {}: {e}", path.display())))?;
    let ids: Vec<u32> = tok
        .encode(text.as_str(), false)
        .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
        .get_ids()
        .to_vec();
    let n = ids.len();

    // The chunk size the arenas are built on. Not imported: this test asserts
    // nothing against it, it only needs cut positions on either side of a
    // boundary, and a wrong guess shows up as two cuts that behave alike.
    const CHUNK: usize = 32;
    let deep = n * 3 / 4 / CHUNK * CHUNK;
    let shallow = n * 2 / 5 / CHUNK * CHUNK;
    // The cuts are chosen to make the two arms cross a 32-token chunk boundary
    // at DIFFERENT decode steps, because that is the hypothesis under test.
    //
    // The control's layout never depends on the cut — it always ends with
    // `n % CHUNK` tokens in its writer, so it crosses after `CHUNK - n % CHUNK`
    // decode steps whatever `c` is. The injected arm's writer holds
    // `(n - c) % CHUNK` tokens, so `c` moves ITS crossing freely. A cut with a
    // nearly-full injected writer makes the injected arm cross FIRST. If the
    // divergence step tracks whichever arm crosses first, the crossing is the
    // mechanism and the injection is only what desynchronises the two layouts.
    // **A sweep of the TAIL length, because that is what separates the two
    // outcomes.** With 734 tokens prefilled on top of the borrowed prefix the
    // two arms are identical for the whole generation; with 14 they part at the
    // first decode step, even though the layouts and the prefill logits agree
    // exactly in both cases. Something the fork installs is being rebuilt by a
    // long tail prefill and not by a short one, and the length at which the
    // divergence disappears is that state's refresh window — which names it.
    //
    // The window lands between a 14-token tail (diverges) and a 32-token one
    // (identical), where two rules coincide: "the tail is at least a chunk" and
    // "the tail prefill crosses a chunk boundary, so the slot's final writer is
    // a freshly allocated chunk rather than the COPIED one". These cuts break
    // the tie — 24, 26 and 31 all cross a boundary with fewer than 32 tokens.
    // If those come out identical the rule is the crossing, which indicts the
    // copied chunk in its role as the decode writer; if they diverge the rule is
    // the length, which indicts a carried state the fork did not reproduce.
    let _ = (shallow, deep);
    let cuts = [n - 2, n - 14, n - 24, n - 26, n - 31, n - 32];
    let free_of = |used: usize| CHUNK - used % CHUNK;
    println!(
        "prompt {n} tokens (CHUNK={CHUNK}); control crosses at decode step {}",
        free_of(n)
    );
    for c in cuts {
        println!(
            "  cut {c:<5} aligned={:<5} tail={:<4} writer-free-before-tail={:<3} \
             tail_crosses={:<5} writer holds {:<2} after",
            c % CHUNK == 0,
            n - c,
            free_of(c),
            n - c > free_of(c),
            (n - c) % CHUNK,
        );
    }

    // Anchor: one contiguous prefill of the whole prompt, its own session.
    let contiguous = {
        let mut s = model.create_batched_session(BatchedConfig::default())?;
        let q = s.create_sequence()?;
        let lg = prefill_into(&model, &mut s, q, &ids)?;
        let out = argmax_continue(&model, &tok, &mut s, q, &lg, 220)?;
        println!(
            "contiguous: tool_call={}  head={}",
            out.contains("<tool_call>"),
            out.replace('\n', " ").chars().take(90).collect::<String>()
        );
        lg
    };

    for c in cuts {
        let aligned = c % CHUNK == 0;
        println!("\n── cut {c} ({} of {n}) aligned={aligned} ──", c);
        let mut s = model.create_batched_session(BatchedConfig::default())?;

        // The donor. Sealed and then never written again: its partial tail chunk
        // is Arc-shared with the injected copy, so a further write into it would
        // change what the "borrowed" arm reads.
        let x = s.create_sequence()?;
        prefill_into(&model, &mut s, x, &ids[..c])?;
        let sealed = s.snapshot_sequence_per_layer(x)?;

        // Control: the identical seam, computed in place.
        let ctl = s.create_sequence()?;
        prefill_into(&model, &mut s, ctl, &ids[..c])?;
        let l_ctl = prefill_into(&model, &mut s, ctl, &ids[c..])?;

        // Treatment: the same seam over a borrowed prefix.
        let inj = s.create_sequence()?;
        s.inject_sealed_at_tail(inj, &sealed)?;
        s.advance_sequence(inj, c)?;
        model.fork_recurrent(x, inj)?;
        s.push_empty_writer_chunk(inj)?;
        let l_inj = prefill_into(&model, &mut s, inj, &ids[c..])?;

        // The physical layouts, side by side. `provenance_chunk_layout` is the
        // window attention actually reads — `(offset, len, cum_before)` per
        // chunk — so a padded chunk shows up as a `len` short of 32 with a
        // following chunk that still starts at the right cumulative position.
        let lay_ctl = s.provenance_chunk_layout(ctl, n);
        let lay_inj = s.provenance_chunk_layout(inj, n);
        println!(
            "  blocks: control={} injected={}   real tokens: control={} injected={}",
            lay_ctl.len(),
            lay_inj.len(),
            lay_ctl.iter().map(|e| e.1 as usize).sum::<usize>(),
            lay_inj.iter().map(|e| e.1 as usize).sum::<usize>()
        );
        let tail_of = |v: &[(u16, u16, usize)]| -> String {
            v.iter()
                .skip(v.len().saturating_sub(4))
                .map(|(o, l, cum)| format!("(o{o},l{l},@{cum})"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!("  control  tail: {}", tail_of(&lay_ctl));
        println!("  injected tail: {}", tail_of(&lay_inj));
        let rope_ctl = s.backings()[0].chunk_rope_positions(ctl);
        let rope_inj = s.backings()[0].chunk_rope_positions(inj);
        let last = |v: &[i32], k: usize| -> Vec<i32> { v.iter().take(k).copied().collect() };
        println!(
            "  rope[{}..]: control={:?} injected={:?}",
            lay_ctl.len().saturating_sub(3),
            last(&rope_ctl[lay_ctl.len().saturating_sub(3)..], 3),
            last(&rope_inj[lay_inj.len().saturating_sub(3)..], 3)
        );

        let vs_ctl = compare(&l_ctl, &l_inj);
        let ctl_vs_contig = compare(&contiguous, &l_ctl);
        println!(
            "  seam vs contiguous : max|d|={:<8.4} top1={:<5} KL={:.6}",
            ctl_vs_contig.max_abs, ctl_vs_contig.top1_same, ctl_vs_contig.kl
        );
        println!(
            "  injected vs seam   : max|d|={:<8.4} top1={:<5} KL={:.6}   <<< the injection's share",
            vs_ctl.max_abs, vs_ctl.top1_same, vs_ctl.kl
        );

        // **Lockstep decode, one token stream.** The prefill logits above are
        // bit-identical, so a divergence in the generated text has to be
        // introduced by the decode. Feeding BOTH slots the same token isolates
        // that: the two arms can no longer drift apart because they picked
        // different words, only because they read their K/V differently. The
        // first step with a non-zero delta names the forward that did it.
        //
        // Both sequences live in ONE session with DISTINCT ids, because the
        // carried recurrences (`recurrent`/`ple`/`index`) are keyed by sequence
        // id on the MODEL — two sessions would both start at id 0 and clobber
        // each other, which is exactly how an earlier version of this probe
        // produced a confident and entirely false answer.
        let n_layers = ManagedBatchedModel::num_layers(&model);
        let mut a = l_ctl.clone();
        let mut b = l_inj.clone();
        let mut first_bad: Option<(usize, f32)> = None;
        let mut gen = Vec::new();
        for step in 0..220 {
            let d = compare(&a, &b);
            if first_bad.is_none() && d.max_abs > 0.0 {
                first_bad = Some((step, d.max_abs));
                println!(
                    "  FIRST DECODE DIVERGENCE at step {step}: max|d|={:.6} KL={:.6} top1_same={}",
                    d.max_abs, d.kl, d.top1_same
                );
            }
            let nx = top_n(&a, 1)[0] as u32;
            gen.push(nx);
            if nx == IM_END || nx == ENDOFTEXT {
                break;
            }
            let x = Tensor::from_vec(vec![nx], (1, 1), &Device::Cpu)?;
            for (slot, out) in [(ctl, &mut a), (inj, &mut b)] {
                let st = model.forward_wave(
                    &mut s,
                    &[slot],
                    std::slice::from_ref(&x),
                    &[],
                    &[],
                    &[],
                    &[],
                    0,
                    n_layers,
                    None,
                )?;
                s.advance_sequence(slot, 1)?;
                *out = st.logits_owned()?[0]
                    .i(0)?
                    .to_dtype(candle::DType::F32)?
                    .to_vec1::<f32>()?;
            }
        }
        match first_bad {
            None => println!("  lockstep decode: identical for all {} steps", gen.len()),
            Some((step, m)) => {
                println!("  lockstep decode: diverged at step {step} (max|d|={m:.6})")
            }
        }
        let shared = tok
            .decode(&gen, false)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        println!(
            "  shared-token text: tool_call={}  head={}",
            shared.contains("<tool_call>"),
            shared
                .replace('\n', " ")
                .chars()
                .take(90)
                .collect::<String>()
        );
    }
    Ok(())
}
