//! Checks whether greedy decode and prefill are bitwise repeatable from
//! identical state.
//!
//! Token streams can pass a gate while the engine returns different text on
//! different runs, and logits can differ in a single ULP where the argmax does
//! not, so these probes compare the streams and the logits directly. Each replay
//! runs on a fresh sequence rather than a rewound one, because a rewind restores
//! the K/V and not a model's recurrent state. The first replay is discarded,
//! since the first decode starts from a different state than the rest. A
//! forward's result is dropped before the next forward starts, because a result
//! still in scope holds the forward span open. The counters are printed, so a
//! clean run is read against whether it evicted at all.

use super::*;

/// **Is greedy decode reproducible, run to run?**
///
/// Greedy argmax over a fixed prompt from a fresh session is a pure function
/// of the weights: `passes` independent decodes must produce identical token
/// streams. Anything else is a bug — a race, a read of memory another agent is
/// writing, or an unwritten element — not acceptable numerical noise.
///
/// **No existing gate can see this.** StoryRewrite checks each session
/// reproduces its own name-substituted story; `wave_paris` checks the answer is
/// "Paris". Both pass happily while the engine returns different text every
/// run. Reproducibility needs its own assertion, and this is it.
///
/// Shared across models on purpose: DeepSeek-V4-Flash fails this, and the
/// question that matters next is whether the fault is in *shared*
/// infrastructure or in DeepSeek-specific code. Running the identical check
/// against a model that shares the `expert_lre` cache (Qwen3-MoE) and one that
/// has no expert cache at all (Llama) answers it.
///
/// Returns the per-pass token streams so a caller can report more than the
/// verdict; logs every pairwise first-difference under `label`.
///
/// `device` is the MODEL's device, and it matters: these models' wave reads
/// token ids on the device, unlike DeepSeek's, which takes them on the host
/// (its embedding lookup is a host `index_select`). Passing CPU ids here fails
/// with `RmsNorm::forward_dynamic(int8) requires a CUDA tensor`.
pub fn decode_reproducibility<M: ManagedBatchedModel>(
    model: &M,
    device: &Device,
    prompt_ids: &[u32],
    decode_tokens: usize,
    passes: usize,
    label: &str,
) -> Result<(bool, Vec<Vec<u32>>)> {
    let n_layers = model.num_layers();
    let mut streams: Vec<Vec<u32>> = Vec::with_capacity(passes);

    // ONE session, truncated back to empty between passes — the harness's own
    // idiom for repeating a prefill ("so every repeat is a true re-prefill
    // from identical state"). A fresh session per pass reads more naturally
    // but does not work: the second pass fails with "the Forward span already
    // has a live generation", i.e. the KV arena's forward guard does not come
    // back when a session is dropped. That is worth fixing on its own; it is
    // not this test's subject.
    let mut session = model.create_batched_session(BatchedConfig::default())?;
    let seq = session.create_sequence()?;

    for _ in 0..passes {
        session.truncate_sequence_to_tokens(seq, 0)?;

        // **Each `WaveResult` is scoped so it DROPS before the next forward.**
        // It holds the KV arena's forward-span guard (`WaveResult::_forward`,
        // "held, never read"), so a step still in scope keeps that span open
        // and the next `forward_wave` fails with "the Forward span already has
        // a live generation". Binding a step with `let` across a loop
        // iteration is enough to trigger it.
        let prompt = Tensor::new(prompt_ids, device)?.unsqueeze(0)?;
        let mut next = {
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                std::slice::from_ref(&prompt),
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, prompt_ids.len())?;
            greedy_token(&step.logits_owned()?[0])?
        };
        let mut gen = vec![next];
        for _ in 1..decode_tokens {
            let tok = Tensor::new(&[next][..], device)?.unsqueeze(0)?;
            next = {
                let step = model.forward_wave(
                    &mut session,
                    &[seq],
                    std::slice::from_ref(&tok),
                    &[],
                    &[],
                    &[],
                    &[],
                    0,
                    n_layers,
                    None,
                )?;
                session.advance_sequence(seq, 1)?;
                greedy_token(&step.logits_owned()?[0])?
            };
            gen.push(next);
        }
        streams.push(gen);

        // **Did this model's expert cache actually EVICT?** A model whose
        // experts all fit in VRAM exercises dispatch but never the load/evict
        // path, so a green verdict from it exonerates far less than it appears
        // to. Print the counters so the cross-model comparison is read against
        // what each run really exercised rather than against an assumption.
        if let Some(s) = model.expert_stats() {
            eprintln!(
                "[repro:{label}] experts hits={} misses={} evictions={} read-ahead={} cold={}",
                s.expert_hits, s.expert_misses, s.evictions, s.ahead_claims, s.worker_cold
            );
        }
    }

    let mut agree = true;
    for i in 0..streams.len() {
        for j in (i + 1)..streams.len() {
            let d = streams[i].iter().zip(&streams[j]).position(|(a, b)| a != b);
            agree &= d.is_none();
            eprintln!(
                "[repro:{label}] pass{}v{} first difference = {:?}",
                i + 1,
                j + 1,
                d
            );
        }
    }
    eprintln!(
        "[repro:{label}] {}",
        if agree {
            "REPRODUCIBLE"
        } else {
            "NOT reproducible"
        }
    );
    Ok((agree, streams))
}

/// **Replay one decode step from identical KV state and compare bitwise.**
///
/// Sharper and far cheaper than comparing token streams: prefill once, then
/// repeatedly {decode one token, record, roll the KV back}, so every sample
/// sees the same inputs and the model is loaded once. Bitwise, so a single
/// differing ULP is caught where argmax would swallow it.
///
/// **Full depth only, deliberately.** An earlier version of this also swept
/// `layer_end` to name the layer that introduces a fault. That sweep is invalid
/// and was removed: a partial-depth `forward_wave` returns a *paused wave's*
/// residual, meant to be resumed on a later forward, and replaying that path
/// does not restore identical state. Run against Qwen3-MoE — whose full-depth
/// replays here are bit-identical — it reported every layer as dirty. Any
/// per-layer instrument must clear that same control before its output means
/// anything.
///
/// **Shared across models on purpose, as the harness control.** The probe's own
/// reset has to restore byte-identical state, or it reports its own drift as
/// model non-determinism. Running the identical code against a model known to
/// be reproducible is what rules that out.
///
/// **Each replay runs on a FRESH SEQUENCE.** It used to rewind one sequence with
/// `truncate_sequence_to_tokens`, which restores the K/V and nothing else —
/// correct exactly as long as the K/V is the whole of a sequence's memory. On a
/// model carrying recurrent state it is not: the probe's decode wave commits,
/// the state advances by a token, and a K/V rewind cannot reach it, so every
/// replay would enter from a state one token further along and the divergence
/// reported would be the harness's own. Allocating a sequence per replay
/// restores identical state on *every* model rather than only on models that
/// have no state to restore, at the cost of one prefill per replay.
///
/// Returns the number of replays that diverged from their predecessor.
pub fn decode_replay_probe<M: ManagedBatchedModel>(
    model: &M,
    device: &Device,
    prompt_ids: &[u32],
    repeats: usize,
    label: &str,
) -> Result<usize> {
    let n_layers = model.num_layers();
    let prompt_len = prompt_ids.len();
    let mut session = model.create_batched_session(BatchedConfig::default())?;
    let prompt = Tensor::new(prompt_ids, device)?.unsqueeze(0)?;

    // Prefill a sequence and return it alongside the prompt's own next-token
    // argmax. Each `WaveResult` is scoped so it drops: it holds the arena's
    // forward-span guard, and a step still in scope makes the next
    // `forward_wave` fail.
    let prefill = |session: &mut BatchedInferenceSession| -> Result<(usize, u32)> {
        let seq = session.create_sequence()?;
        let step = model.forward_wave(
            session,
            &[],
            &[],
            &[seq],
            std::slice::from_ref(&prompt),
            &[],
            &[],
            0,
            n_layers,
            None,
        )?;
        session.advance_sequence(seq, prompt_len)?;
        let tok = greedy_token(&step.logits_owned()?[0])?;
        Ok((seq, tok))
    };

    let (warm_seq, first) = prefill(&mut session)?;
    session.free_sequence(warm_seq)?;
    model.release_sequence(warm_seq)?;

    // One replay: a fresh sequence, prefilled from scratch, decoding the same
    // single token — so every sample starts from a state that is identical by
    // construction rather than by a rewind that only covers the K/V.
    let mut probe = || -> Result<Vec<u32>> {
        let (seq, _) = prefill(&mut session)?;
        let tok = Tensor::new(&[first][..], device)?.unsqueeze(0)?;
        let t = {
            let step = model.forward_wave(
                &mut session,
                &[seq],
                std::slice::from_ref(&tok),
                &[],
                &[],
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            let l = step.logits_owned()?;
            l.into_iter()
                .next()
                .ok_or_else(|| candle::Error::msg("decode wave produced no logits"))?
        };
        let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        session.free_sequence(seq)?;
        model.release_sequence(seq)?;
        Ok(v.iter().map(|x| x.to_bits()).collect())
    };

    fn diff(a: &[u32], b: &[u32]) -> (usize, f32) {
        let mut n = 0usize;
        let mut worst = 0f32;
        for (&x, &y) in a.iter().zip(b) {
            if x != y {
                n += 1;
                worst = worst.max((f32::from_bits(x) - f32::from_bits(y)).abs());
            }
        }
        (n, worst)
    }

    // **Warm-up, then discard.** The very first decode runs from the
    // post-prefill state while every later one runs from a post-truncate state,
    // and those are not the same thing — the first decode can seal or quantize
    // the prompt's live chunks. Comparing run 0 against run 1 therefore measures
    // that transition, not reproducibility. One throwaway replay puts every
    // recorded sample on the same side of it.
    let _ = probe()?;

    let mut dirty_repeats = 0usize;
    let mut prev: Option<Vec<u32>> = None;
    for r in 0..repeats {
        let s = probe()?;
        if let Some(p) = &prev {
            let (n, worst) = diff(p, &s);
            if n > 0 {
                dirty_repeats += 1;
                eprintln!(
                    "[replay:{label}] run{} vs run{r}: {n}/{} differ, max|Δ|={worst:.3e}",
                    r - 1,
                    s.len()
                );
            }
        }
        prev = Some(s);
    }
    eprintln!(
        "[replay:{label}] dirty repeats = {dirty_repeats}/{}",
        repeats - 1
    );

    // A run with NO evictions never exercises the load/evict path, so a clean
    // verdict from it says nothing about that path. Print the counters so the
    // comparison is read against what the run actually exercised.
    if let Some(s) = model.expert_stats() {
        eprintln!(
            "[replay:{label}] experts hits={} misses={} evictions={}",
            s.expert_hits, s.expert_misses, s.evictions
        );
    }
    Ok(dirty_repeats)
}

/// **Is PREFILL bitwise repeatable?**
///
/// The companion to [`decode_replay_probe`], and needed for the same reason:
/// end-to-end token streams cannot tell "prefill wrote different KV" from
/// "decode computed differently", so a fix to one gets judged by a test the
/// other also fails. This re-prefills the same prompt from a truncated
/// sequence and compares the prefill's own logits bitwise — nothing downstream
/// of prefill runs at all.
///
/// Returns the number of prefills that diverged from their predecessor.
pub fn prefill_replay_probe<M: ManagedBatchedModel>(
    model: &M,
    device: &Device,
    prompt_ids: &[u32],
    repeats: usize,
    label: &str,
) -> Result<usize> {
    use candle::IndexOp;

    let n_layers = model.num_layers();
    let mut session = model.create_batched_session(BatchedConfig::default())?;
    let seq = session.create_sequence()?;

    let once = |session: &mut BatchedInferenceSession| -> Result<Vec<u32>> {
        session.truncate_sequence_to_tokens(seq, 0)?;
        let prompt = Tensor::new(prompt_ids, device)?.unsqueeze(0)?;
        let t = {
            let step = model.forward_wave(
                session,
                &[],
                &[],
                &[seq],
                std::slice::from_ref(&prompt),
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            step.logits_owned()?
                .into_iter()
                .next()
                .ok_or_else(|| candle::Error::msg("prefill produced no logits"))?
        };
        let v = t
            .i(0)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        Ok(v.iter().map(|x| x.to_bits()).collect())
    };

    // Warm-up discarded: the first prefill of a session runs against a cold
    // expert cache, which is a different situation from every later one.
    let _ = once(&mut session)?;

    let mut dirty = 0usize;
    let mut prev: Option<Vec<u32>> = None;
    for r in 0..repeats {
        let s = once(&mut session)?;
        if let Some(p) = &prev {
            let mut n = 0usize;
            let mut worst = 0f32;
            for (&x, &y) in p.iter().zip(&s) {
                if x != y {
                    n += 1;
                    worst = worst.max((f32::from_bits(x) - f32::from_bits(y)).abs());
                }
            }
            if n > 0 {
                dirty += 1;
                eprintln!(
                    "[prefill:{label}] run{} vs run{r}: {n}/{} differ, max|Δ|={worst:.3e}",
                    r - 1,
                    s.len()
                );
            }
        }
        prev = Some(s);
    }
    eprintln!("[prefill:{label}] dirty = {dirty}/{}", repeats - 1);
    Ok(dirty)
}
