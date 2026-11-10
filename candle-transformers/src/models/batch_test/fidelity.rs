//! Output fidelity against depth: how far a compressed KV cache moves the
//! model's next-token distribution, as the context it reads grows.
//!
//! The Asymptotic Numerical Stability theorem (`docs/unbounded_agents.md`
//! §11.2) bounds the stored error a step is *exposed* to, and the paper says
//! plainly that this is not the error in the step's output: softmax attention
//! averages its values, so a full-attention model's output divergence need not
//! grow with depth at all. This gate measures the output side directly.
//!
//! At each depth one needle prompt (`yarn_gates::needle_prompt`: four codes of
//! random characters spread through the long-context filler, asked back at the
//! end) is prefilled under every mode, and the reference answer is read back
//! **teacher-forced** — every mode is fed the same answer tokens, so every step
//! compares two distributions over the same prefix. Per depth and mode:
//!
//! - **KL** — mean and worst per-step `KL(reference ‖ mode)`, nats;
//! - **Top-1 agreement** — the share of steps whose argmax matches the
//!   reference's;
//! - **NLL** — the answer's mean negative log-likelihood under the mode, and
//!   whether greedy decode would have reproduced it.
//!
//! The reference is the model's own float KV mode. A model whose attention
//! reads the whole prefix exposes each step to every stored token; one that
//! selects a bounded set does not. Whether the measured divergence grows with
//! depth in the first case and stays flat in the second is the question.

use candle::{DType, Result, Tensor, D};
use tokenizers::Tokenizer;

use super::yarn_gates::{decode_wave, needle_answer, needle_prompt, prefill};
use crate::models::batched_inference::{BatchedConfig, InferenceMode, ManagedBatchedModel};

/// One teacher-forced read: the log-probabilities the model gave at each
/// answer step, over the whole vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    /// `steps[i][v]` — log-probability of token `v` at answer step `i`.
    pub steps: Vec<Vec<f32>>,
}

/// How far one mode's trace sits from the reference's.
#[derive(Debug, Clone, PartialEq)]
pub struct Divergence {
    /// Mean per-step `KL(reference ‖ mode)`, nats.
    pub mean_kl: f64,
    /// Worst per-step `KL(reference ‖ mode)`, nats.
    pub max_kl: f64,
    /// Share of steps whose argmax equals the reference's.
    pub top1: f64,
    /// The answer's mean NLL under this mode, nats per token.
    pub nll: f64,
    /// Whether every step's argmax is the answer token.
    pub recovered: bool,
}

/// `KL(p ‖ q)` for two log-probability rows over one vocabulary, in nats.
/// Terms where `p` carries no mass contribute nothing; the sum is accumulated
/// in f64 so a 150K-entry row does not lose the small terms.
pub fn kl(p: &[f32], q: &[f32]) -> f64 {
    p.iter()
        .zip(q)
        .map(|(&lp, &lq)| {
            let pp = (lp as f64).exp();
            if pp == 0.0 {
                0.0
            } else {
                pp * (lp as f64 - lq as f64)
            }
        })
        .sum()
}

/// The index of a row's largest entry; the first on a tie.
pub fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

/// `other` against `reference`, both teacher-forced on `answer`.
pub fn divergence(reference: &Trace, other: &Trace, answer: &[u32]) -> Divergence {
    let n = answer
        .len()
        .min(reference.steps.len())
        .min(other.steps.len());
    let mut sum_kl = 0f64;
    let mut max_kl = 0f64;
    let mut agree = 0usize;
    let mut nll = 0f64;
    let mut recovered = true;
    for i in 0..n {
        let (r, o) = (&reference.steps[i], &other.steps[i]);
        let k = kl(r, o);
        sum_kl += k;
        max_kl = max_kl.max(k);
        let top = argmax(o);
        if top == argmax(r) {
            agree += 1;
        }
        if top != answer[i] as usize {
            recovered = false;
        }
        nll -= o[answer[i] as usize] as f64;
    }
    let steps = n.max(1) as f64;
    Divergence {
        mean_kl: sum_kl / steps,
        max_kl,
        top1: agree as f64 / steps,
        nll: nll / steps,
        recovered: recovered && n == answer.len(),
    }
}

/// The batched config a mode runs under — the same mapping the width harness
/// uses (`utils::TestParams`).
pub fn config_for(mode: InferenceMode) -> BatchedConfig {
    BatchedConfig {
        k_format: mode.k_format(),
        v_format: mode.v_format(),
        compression_level: mode.compression_level(),
        ..Default::default()
    }
}

/// Prefill `prompt` on a fresh session under `mode`, seal it as the substrate
/// seals a turn — the prompt's K/V quantized in place at the mode's level, the
/// answer decoding into a fresh chunk — and read `answer` back teacher-forced,
/// keeping every step's log-probabilities. Sealing is where compression is
/// paid (`BatchedInferenceSession::quantize_and_seal_sequences`); without it
/// every mode would decode against the prompt's float K/V. It is a no-op for
/// a float mode.
pub fn forced_trace<M: ManagedBatchedModel>(
    model: &M,
    mode: InferenceMode,
    prompt: &[u32],
    answer: &[u32],
) -> Result<Trace> {
    let mut session = model.create_batched_session(config_for(mode))?;
    let seq = session.create_sequence()?;
    let mut logits = prefill(model, &mut session, seq, prompt)?;
    #[cfg(feature = "cuda")]
    session.quantize_and_seal_sequences(&[seq], true)?;
    let mut steps = Vec::with_capacity(answer.len());
    for (i, &want) in answer.iter().enumerate() {
        steps.push(log_probs(&logits)?);
        if i + 1 < answer.len() {
            logits = decode_wave(model, &mut session, &[seq], &[want])?.remove(0);
        }
    }
    model.release_sequence(seq)?;
    Ok(Trace { steps })
}

/// One logits row as host log-probabilities. This is the measurement's one
/// readback per step, outside any hot path.
fn log_probs(logits: &Tensor) -> Result<Vec<f32>> {
    candle_nn::ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)?.to_vec1::<f32>()
}

/// One row of the ladder: a depth, a mode, and how far it sits from the
/// reference at that depth.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// Requested depth, tokens.
    pub depth: usize,
    /// The prompt as tokenised, tokens.
    pub prompt: usize,
    /// The mode measured.
    pub mode: InferenceMode,
    /// Its divergence from the reference at this depth.
    pub div: Divergence,
}

/// The ladder: at every depth, the reference mode's trace and every other
/// mode's divergence from it. The reference's own row compares a second
/// reference read against the first, so the table carries the model's
/// run-to-run spread at every depth — a divergence no larger than that is not
/// one the compression made.
pub fn ladder<M: ManagedBatchedModel>(
    model: &M,
    tok: &Tokenizer,
    reference: InferenceMode,
    modes: &[InferenceMode],
    depths: &[usize],
) -> Result<Vec<Row>> {
    let answer = needle_answer(tok)?;
    let mut rows = Vec::new();
    for &depth in depths {
        let prompt = needle_prompt(tok, depth)?;
        let base = forced_trace(model, reference, &prompt, &answer)?;
        let said: Vec<u32> = base.steps.iter().map(|s| argmax(s) as u32).collect();
        println!(
            "  depth {depth:>7}: the reference's greedy choice at each answer step reads {:?}",
            tok.decode(&said, false).unwrap_or_default()
        );
        let again = forced_trace(model, reference, &prompt, &answer)?;
        let mut push = |mode, div: Divergence| {
            println!(
                "  depth {depth:>7} ({:>7} tok)  {mode:<5?}  KL mean {:.3e} max {:.3e}  \
                 top-1 {:>5.1}%  NLL {:.3e}  recovered {}",
                prompt.len(),
                div.mean_kl,
                div.max_kl,
                div.top1 * 100.0,
                div.nll,
                div.recovered
            );
            rows.push(Row {
                depth,
                prompt: prompt.len(),
                mode,
                div,
            });
        };
        push(reference, divergence(&base, &again, &answer));
        for &mode in modes {
            let t = forced_trace(model, mode, &prompt, &answer)?;
            push(mode, divergence(&base, &t, &answer));
        }
    }
    Ok(rows)
}

/// Print the ladder as the markdown table the results file records.
pub fn print_table(label: &str, rows: &[Row]) {
    println!("\n### {label}\n");
    println!("| Depth | Prompt tok | Mode | KL mean (nats) | KL max | Top-1 | NLL | Recovered |");
    println!("|---:|---:|---|---:|---:|---:|---:|---|");
    for r in rows {
        println!(
            "| {} | {} | {:?} | {:.3e} | {:.3e} | {:.1}% | {:.3e} | {} |",
            r.depth,
            r.prompt,
            r.mode,
            r.div.mean_kl,
            r.div.max_kl,
            r.div.top1 * 100.0,
            r.div.nll,
            r.div.recovered
        );
    }
}

/// Every divergence in the ladder is a finite number, and the ladder holds a
/// reference row to read the others against.
///
/// Whether the reference *recovers* the needles is reported, not required:
/// teacher forcing compares two distributions over the same prefix whatever
/// the model would have said, and a checkpoint that answers in its own words
/// (a bullet, a bold code) still measures the cache's divergence exactly.
pub fn assert_sound(label: &str, rows: &[Row], reference: InferenceMode) {
    for r in rows {
        let d = &r.div;
        assert!(
            d.mean_kl.is_finite() && d.max_kl.is_finite() && d.nll.is_finite(),
            "{label}: depth {} {:?} produced a non-finite divergence {d:?}",
            r.depth,
            r.mode
        );
    }
    assert!(
        rows.iter().any(|r| r.mode == reference),
        "{label}: the ladder has no reference row"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ln` of each entry: a log-probability row from probabilities.
    fn logs(p: &[f64]) -> Vec<f32> {
        p.iter().map(|x| x.ln() as f32).collect()
    }

    /// Identical rows are zero apart.
    #[test]
    fn a_row_is_no_distance_from_itself() {
        let p = logs(&[0.5, 0.25, 0.25]);
        assert_eq!(kl(&p, &p), 0.0);
    }

    /// KL of `[1/2, 1/2]` from `[1/4, 3/4]` is
    /// `½·ln(2) + ½·ln(2/3)` = 0.143841… nats.
    #[test]
    fn kl_matches_the_closed_form() {
        let got = kl(&logs(&[0.5, 0.5]), &logs(&[0.25, 0.75]));
        let want = 0.5 * (2.0f64).ln() + 0.5 * (2.0f64 / 3.0).ln();
        assert!((got - want).abs() < 1e-6, "{got} against {want}");
    }

    /// A token the reference gives no mass contributes nothing, even where the
    /// other row gives it all of its own.
    #[test]
    fn a_massless_reference_entry_contributes_nothing() {
        let p = vec![0.0f32, f32::NEG_INFINITY];
        let q = logs(&[0.5, 0.5]);
        assert!((kl(&p, &q) - (2.0f64).ln()).abs() < 1e-6);
    }

    /// The first of two equal maxima is the argmax.
    #[test]
    fn argmax_takes_the_first_of_a_tie() {
        assert_eq!(argmax(&[0.1, 0.7, 0.7, 0.2]), 1);
        assert_eq!(argmax(&[-3.0]), 0);
    }

    /// Two steps: one agreeing, one not. The divergence counts each exactly.
    #[test]
    fn divergence_counts_every_step() {
        let reference = Trace {
            steps: vec![logs(&[0.5, 0.5 - 1e-9, 1e-9]), logs(&[0.1, 0.8, 0.1])],
        };
        let other = Trace {
            steps: vec![logs(&[0.5, 0.5 - 1e-9, 1e-9]), logs(&[0.6, 0.3, 0.1])],
        };
        let d = divergence(&reference, &other, &[0, 1]);
        assert_eq!(d.top1, 0.5, "step 0 agrees, step 1 does not");
        assert!(!d.recovered, "step 1's argmax is 0, not the answer's 1");
        let want_kl1 = 0.1 * (0.1f64 / 0.6).ln() + 0.8 * (0.8f64 / 0.3).ln();
        assert!((d.max_kl - want_kl1).abs() < 1e-6, "{}", d.max_kl);
        assert!((d.mean_kl - want_kl1 / 2.0).abs() < 1e-6, "{}", d.mean_kl);
        let want_nll = -((0.5f64).ln() + (0.3f64).ln()) / 2.0;
        assert!((d.nll - want_nll).abs() < 1e-6, "{}", d.nll);
    }

    /// A trace shorter than the answer cannot have recovered it.
    #[test]
    fn a_short_trace_has_not_recovered_the_answer() {
        let t = Trace {
            steps: vec![logs(&[0.9, 0.1])],
        };
        assert!(!divergence(&t, &t, &[0, 0]).recovered);
        assert!(divergence(&t, &t, &[0]).recovered);
    }

    /// Each mode runs under the formats and level the width harness gives it.
    #[test]
    fn a_mode_runs_under_its_own_formats() {
        let c = config_for(InferenceMode::C10);
        assert_eq!(c.compression_level, Some(10));
        assert_eq!(c.k_format, InferenceMode::C10.k_format());
        assert_eq!(config_for(InferenceMode::BF16).compression_level, None);
    }
}
