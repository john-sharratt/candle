//! The progressive-YaRN gates (`docs/progressive_yarn.md` §10, gates 3–5):
//! what a model's rungs buy past its trained window, measured on the model.
//!
//! Every gate plants four needles — vault codes of random characters, each
//! stated once, at a tenth, a third, three fifths and five sixths of the way
//! through a long run of the long-context filler
//! (`long_context::padding_prose`) — and asks for all four back, in order. It
//! reads the answer **teacher-forced**: each answer token is fed as the model's
//! input, and at each step the harness records whether greedy decode would have
//! chosen it and the negative log-likelihood the model gave it. One pass
//! therefore answers both questions:
//!
//! - **Recovered** — greedy decode reproduces the whole answer, because every
//!   step's argmax is the reference token (so greedy would have followed
//!   exactly this path).
//! - **How well** — the answer's mean NLL, which compares two schedules even
//!   where both recover it or both fail.
//!
//! Four codes, not one phrase, because one phrase does not separate schedules:
//! a model reads a single distinctive line back near-certainly well past its
//! window under either one (measured on Qwen3-30B-A3B: an NLL of order 1e-7
//! at twice the window under both). Codes of random characters leave nothing
//! to guess from, and four of them at different distances make the answer
//! depend on the position of every one.
//!
//! The control for gate 3 is the same model on **rung-1 extrapolation**: its
//! trained RoPE at every length, the schedule a stack without progressive
//! rungs runs. Below the trained window the two schedules are the same rung, so
//! the gate also asserts they agree there.

use candle::{DType, IndexOp, Result, Tensor, D};
use tokenizers::Tokenizer;

use super::long_context::padding_prose;
use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel, WaveResult,
};

/// The needles: four vaults, each with its code and the fraction of the filler
/// it is stated at. Codes of random characters, so nothing in the filler or
/// the model's priors predicts them.
const NEEDLES: [(&str, &str, f64); 4] = [
    ("Kirkwall", "7QX4-MZ9K", 0.10),
    ("Tampere", "R2VD-8HJ3", 0.35),
    ("Broome", "N6TC-4WQP", 0.60),
    ("Galway", "B9LF-2XRE", 0.85),
];

/// The first needle's code: nothing in the filler resembles it, so an answer
/// containing it read the needle back.
pub const NEEDLE_KEY: &str = "7QX4-MZ9K";

/// Tokens a prefill wave takes at once.
const PREFILL_CHUNK: usize = 8_192;

/// A ChatML user turn around `body`, then an assistant turn with thinking
/// suppressed, so the answer is the first thing decoded.
pub fn chatml_turn(body: &str) -> String {
    format!("<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// The question that reads the needles back.
fn needle_question() -> &'static str {
    "List the four vault codes stated in the notes above, in the order they \
     appear, one per line as `Vault: CODE`, and nothing else."
}

/// The answer the question asks for.
fn needle_answer_text() -> String {
    NEEDLES
        .iter()
        .map(|(vault, code, _)| format!("{vault}: {code}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A paragraph boundary near `at` of `padding`'s length, as a byte offset.
fn paragraph_at(padding: &str, at: f64) -> usize {
    let mut cut = (((padding.len() as f64) * at) as usize).min(padding.len());
    while !padding.is_char_boundary(cut) {
        cut -= 1;
    }
    padding[..cut].rfind("\n\n").map(|i| i + 2).unwrap_or(0)
}

/// `padding` with every needle's sentence stated at its paragraph boundary.
fn with_needles(padding: &str) -> String {
    let mut out = String::with_capacity(padding.len() + 256);
    let mut from = 0;
    for (vault, code, at) in NEEDLES {
        let cut = paragraph_at(padding, at).max(from);
        out.push_str(&padding[from..cut]);
        out.push_str(&format!("The code for the {vault} vault is {code}.\n\n"));
        from = cut;
    }
    out.push_str(&padding[from..]);
    out
}

/// One needle prompt of about `depth` tokens, the question at the end — as
/// token ids.
pub fn needle_prompt(tok: &Tokenizer, depth: usize) -> Result<Vec<u32>> {
    let padding = padding_prose(tok, depth);
    let body = format!("{}\n\n{}", with_needles(&padding), needle_question());
    encode(tok, &chatml_turn(&body))
}

/// The reference answer's token ids.
pub fn needle_answer(tok: &Tokenizer) -> Result<Vec<u32>> {
    encode(tok, &needle_answer_text())
}

fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok
        .encode(text, false)
        .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
        .get_ids()
        .to_vec())
}

/// The last logits row of one sequence's step, `[vocab]` in f32.
fn last_row(step: &WaveResult, i: usize) -> Result<Tensor> {
    let logits = step.logits_owned()?;
    let row = logits
        .get(i)
        .ok_or_else(|| candle::Error::Msg(format!("wave returned no logits for row {i}")))?;
    let r = row.i(row.dim(0)? - 1)?;
    r.to_dtype(DType::F32)
}

/// Prefill `ids` into `seq` in [`PREFILL_CHUNK`]-token waves; the last
/// position's logits.
pub fn prefill<M: ManagedBatchedModel>(
    model: &M,
    session: &mut BatchedInferenceSession,
    seq: usize,
    ids: &[u32],
) -> Result<Tensor> {
    let n_layers = model.num_layers();
    let mut last = None;
    for chunk in ids.chunks(PREFILL_CHUNK) {
        let t = Tensor::from_vec(chunk.to_vec(), (1, chunk.len()), session.device())?;
        let step = model.forward_wave(
            session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&t),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, chunk.len())?;
        last = Some(last_row(&step, 0)?);
    }
    last.ok_or_else(|| candle::Error::Msg("prefill of an empty prompt".into()))
}

/// One decode step for each of `seqs`, all in one wave; each sequence's next
/// logits, in `seqs` order.
pub fn decode_wave<M: ManagedBatchedModel>(
    model: &M,
    session: &mut BatchedInferenceSession,
    seqs: &[usize],
    tokens: &[u32],
) -> Result<Vec<Tensor>> {
    let device = session.device().clone();
    let inputs: Vec<Tensor> = tokens
        .iter()
        .map(|&t| Tensor::from_vec(vec![t], (1, 1), &device))
        .collect::<Result<_>>()?;
    let step = model.forward_wave(
        session,
        seqs,
        &inputs,
        &[],
        &[],
        &[],
        &[],
        0,
        model.num_layers(),
        None,
    )?;
    for &s in seqs {
        session.advance_sequence(s, 1)?;
    }
    (0..seqs.len()).map(|i| last_row(&step, i)).collect()
}

/// A teacher-forced read of the reference answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Read {
    /// Whether greedy decode reproduces the whole answer.
    pub recovered: bool,
    /// The answer's mean negative log-likelihood, nats per token.
    pub nll: f64,
    /// Greedy's choice at each step, decoded — what the model would have said.
    pub greedy: Vec<u32>,
}

/// Score `answer` teacher-forced, starting from the logits the prompt left.
pub fn read_answer<M: ManagedBatchedModel>(
    model: &M,
    session: &mut BatchedInferenceSession,
    seq: usize,
    first: Tensor,
    answer: &[u32],
) -> Result<Read> {
    let mut logits = first;
    let mut nll = 0f64;
    let mut greedy = Vec::with_capacity(answer.len());
    for (i, &want) in answer.iter().enumerate() {
        let lp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        nll -= lp.i(want as usize)?.to_scalar::<f32>()? as f64;
        greedy.push(logits.argmax(D::Minus1)?.to_scalar::<u32>()?);
        if i + 1 < answer.len() {
            logits = decode_wave(model, session, &[seq], &[want])?.remove(0);
        }
    }
    Ok(Read {
        recovered: greedy == answer,
        nll: nll / answer.len().max(1) as f64,
        greedy,
    })
}

/// Gate 3's measurement: the needles at `depth`, read back on a fresh session.
/// Returns the read and the rung the prompt's reach put the sequence on.
pub fn needle_read<M: ManagedBatchedModel>(
    model: &M,
    config: BatchedConfig,
    tok: &Tokenizer,
    depth: usize,
) -> Result<(Read, u32, usize)> {
    let ids = needle_prompt(tok, depth)?;
    let answer = needle_answer(tok)?;
    let mut session = model.create_batched_session(config)?;
    let seq = session.create_sequence()?;
    let rung = session.rope_rung_for(ids.len() + answer.len())?;
    let first = prefill(model, &mut session, seq, &ids)?;
    let read = read_answer(model, &mut session, seq, first, &answer)?;
    model.release_sequence(seq)?;
    Ok((read, rung, ids.len()))
}

/// Gate 4: `n` greedy tokens after each prompt, the sequences decoding
/// **together** in one wave each step. Each sequence's tokens, in order.
pub fn greedy_together<M: ManagedBatchedModel>(
    model: &M,
    config: BatchedConfig,
    prompts: &[Vec<u32>],
    n: usize,
) -> Result<(Vec<Vec<u32>>, Vec<u32>)> {
    let mut session = model.create_batched_session(config)?;
    let mut seqs = Vec::with_capacity(prompts.len());
    let mut next = Vec::with_capacity(prompts.len());
    let mut rungs = Vec::with_capacity(prompts.len());
    for p in prompts {
        let s = session.create_sequence()?;
        rungs.push(session.rope_rung_for(p.len() + n)?);
        let l = prefill(model, &mut session, s, p)?;
        next.push(l.argmax(D::Minus1)?.to_scalar::<u32>()?);
        seqs.push(s);
    }
    let mut out: Vec<Vec<u32>> = next.iter().map(|&t| vec![t]).collect();
    for _ in 1..n {
        let logits = decode_wave(model, &mut session, &seqs, &next)?;
        for (i, l) in logits.iter().enumerate() {
            next[i] = l.argmax(D::Minus1)?.to_scalar::<u32>()?;
            out[i].push(next[i]);
        }
    }
    for s in seqs {
        model.release_sequence(s)?;
    }
    Ok((out, rungs))
}

/// Gate 5: a conversation that crosses a ceiling mid-way. The needles in
/// `first_depth` tokens of filler, answered with a short acknowledgement
/// (rung `before`); then a tool result of `second_depth` more tokens and the
/// question (rung `after`). Returns `(before, after, read)`.
pub fn crossing_read<M: ManagedBatchedModel>(
    model: &M,
    config: BatchedConfig,
    tok: &Tokenizer,
    first_depth: usize,
    second_depth: usize,
) -> Result<(u32, u32, Read)> {
    let padding = padding_prose(tok, first_depth);
    let turn1 = chatml_turn(&format!(
        "{}\n\nAcknowledge these notes with the single word OK.",
        with_needles(&padding)
    ));
    let ack = encode(tok, "OK")?;
    let tool = padding_prose(tok, second_depth);
    let turn2 = format!(
        "<|im_end|>\n{}",
        chatml_turn(&format!(
            "<tool_response>\n{tool}\n</tool_response>\n\n{}",
            needle_question()
        ))
    );
    let ids1 = encode(tok, &turn1)?;
    let ids2 = encode(tok, &turn2)?;
    let answer = needle_answer(tok)?;

    let mut session = model.create_batched_session(config)?;
    let seq = session.create_sequence()?;
    prefill(model, &mut session, seq, &ids1)?;
    // The acknowledgement decodes on the first rung.
    let before_rung = session.rope_rung_for(ids1.len() + ack.len())?;
    for &t in &ack {
        decode_wave(model, &mut session, &[seq], &[t])?;
    }
    let first = prefill(model, &mut session, seq, &ids2)?;
    let reach = ids1.len() + ack.len() + ids2.len() + answer.len();
    let after_rung = session.rope_rung_for(reach)?;
    let read = read_answer(model, &mut session, seq, first, &answer)?;
    model.release_sequence(seq)?;
    Ok((before_rung, after_rung, read))
}

/// `tokens` up to the turn's end — the first `<|im_end|>` or `<|endoftext|>`,
/// resolved by name. Past either the model is continuing a turn it already
/// finished, and what it emits there is not an answer: which of two near-tied
/// continuations wins is a batch-composition coin-flip, not a property of the
/// schedule.
pub fn answer_of(tok: &Tokenizer, tokens: &[u32]) -> Result<Vec<u32>> {
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .map(|s| {
            tok.token_to_id(s)
                .ok_or_else(|| candle::Error::Msg(format!("tokenizer has no {s}")))
        })
        .collect::<Result<_>>()?;
    Ok(tokens
        .iter()
        .copied()
        .take_while(|t| !stop.contains(t))
        .collect())
}

/// Print one gate-3 row.
pub fn report(label: &str, depth: usize, prompt: usize, rung: u32, read: &Read, tok: &Tokenizer) {
    let said = tok.decode(&read.greedy, false).unwrap_or_default();
    println!(
        "  {label:<26} depth {depth:>7} ({prompt:>7} tok, rung {rung}): recovered {:<5} \
         NLL {:.4e}  greedy {said:?}",
        read.recovered, read.nll
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A needle lands on a paragraph boundary near its fraction, and a
    /// fraction short of the first boundary lands at the start.
    #[test]
    fn a_needle_lands_on_a_paragraph_boundary() {
        let text = "aaaa\n\nbbbb\n\ncccc\n\ndddd\n\n";
        assert_eq!(paragraph_at(text, 0.5), "aaaa\n\nbbbb\n\n".len());
        assert_eq!(paragraph_at(text, 0.0), 0);
        assert_eq!(paragraph_at(text, 0.1), 0);
    }

    /// Every needle is stated once, in order, and the filler keeps every byte
    /// around them.
    #[test]
    fn the_needles_are_stated_in_order_around_the_whole_filler() {
        let text: String = (0..20).map(|i| format!("paragraph {i}\n\n")).collect();
        let out = with_needles(&text);
        let at: Vec<usize> = NEEDLES
            .iter()
            .map(|(_, code, _)| out.find(code).expect("every needle is stated"))
            .collect();
        assert!(at.windows(2).all(|w| w[0] < w[1]), "in order: {at:?}");
        let mut stripped = out.clone();
        for (vault, code, _) in NEEDLES {
            stripped = stripped.replace(
                &format!("The code for the {vault} vault is {code}.\n\n"),
                "",
            );
        }
        assert_eq!(stripped, text);
        assert_eq!(
            needle_answer_text(),
            "Kirkwall: 7QX4-MZ9K\nTampere: R2VD-8HJ3\nBroome: N6TC-4WQP\nGalway: B9LF-2XRE"
        );
    }

    /// The turn frames the body exactly as the Qwen chat template does with
    /// thinking suppressed.
    #[test]
    fn the_turn_is_chatml_with_empty_thinking() {
        assert_eq!(
            chatml_turn("hi"),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
