//! Issuing the one wave a speculative verify step runs on, with whatever else
//! the scheduler is co-batching onto it.
//!
//! # Why a verify block cannot simply be "a prefill"
//!
//! The engine's decode wave is not a decode wave. It is the continuous-fair
//! wave (`docs/continuous_fair_waves.md`): one forward folding every class of
//! work through the shared grouped GEMM, so one expert load per layer serves
//! them all. Its members split into two sweep classes, and `forward_wave`'s
//! three group slots — `[decode | prefill | glue]` — cut across that split
//! rather than along it:
//!
//! * **Full-sweep** members traverse every layer this wave. Decode rows (one
//!   token per sequence) and glue scatter rows are full-sweep.
//! * **Creep** members — the wave group of dialogue prefills and section-ingest
//!   chunks — share the GEMM only inside a layer window, holding their
//!   inter-layer residual across waves so the full-sweep members overtake them.
//!
//! A verify block is multi-token, so it cannot ride the decode slot: that slot
//! is one row per sequence, priced and admitted as one row per sequence. But it
//! is emphatically **full-sweep** — its logits are consumed by the accept walk
//! in the same step that produced them, so a block held mid-sweep with the creep
//! would stall the very decode it exists to accelerate.
//!
//! So verify rows occupy the prefill slot *alongside* the creep, and the layer
//! sweep is cut into up to three segments so that each class crosses the layers
//! it is supposed to:
//!
//! ```text
//!   seg 1  [0, cursor)          decode | verify |        | glue
//!   seg 2  [cursor, win_end)    decode | verify | creep  | glue
//!   seg 3  [win_end, layer_end) decode | verify |        | glue
//! ```
//!
//! The creep's residual is spliced in before segment 2 and lifted back out
//! after it, exactly as the plain decode path already does for a wave with no
//! verify rows on it — this module generalises that splice rather than
//! introducing a second one.
//!
//! With no creep group the window is empty, all three segments collapse to a
//! single `[0, layer_end)` forward, and the whole thing is the standalone verify
//! a test or the batch harness runs.

use std::ops::Range;

use candle::{Result, Tensor};

use super::batched_inference::{BatchedInferenceSession, ManagedBatchedModel};

/// The work riding the verify wave besides the verify blocks themselves.
///
/// Built by whoever owns the wave policy — the scheduler — and passed down
/// through `verify_blocks` so a model's verify path issues the *fair* wave
/// rather than a private one. [`WaveCoBatch::standalone`] is the degenerate
/// case: nothing else on the wave.
pub struct WaveCoBatch<'a> {
    /// Partial-sweep multi-token rows: dialogue prefills and section-ingest
    /// chunks. These advance only inside [`Self::creep_window`].
    pub creep_seqs: &'a [usize],
    pub creep_inputs: &'a [Tensor],
    /// Full-sweep scatter rows. Carried through every segment.
    pub glue_seqs: &'a [usize],
    pub glue_inputs: &'a [Tensor],
    /// The creep's inter-layer residual, held from the wave that last advanced
    /// it. `None` when the creep starts at layer 0 and embeds fresh.
    pub creep_residual: Option<Tensor>,
    /// The layer window the creep shares the GEMM in. Empty when there is no
    /// creep, which collapses the sweep to one segment.
    pub creep_window: Range<usize>,
}

impl WaveCoBatch<'_> {
    /// A verify wave carrying nothing else — one segment over every layer.
    pub fn standalone() -> Self {
        Self {
            creep_seqs: &[],
            creep_inputs: &[],
            glue_seqs: &[],
            glue_inputs: &[],
            creep_residual: None,
            creep_window: 0..0,
        }
    }

    /// Whether a creep group is actually present. An empty group is not the
    /// same as a zero-width window — a caller can hand over a window it did not
    /// fill — so both are checked wherever the segmentation branches.
    fn has_creep(&self) -> bool {
        !self.creep_seqs.is_empty() && !self.creep_window.is_empty()
    }
}

/// What one verify wave produced.
pub struct VerifyWaveOutput {
    /// Per-row logits in caller order, truncated to `[decode | verify]` — the
    /// only rows the accept walk reads. Creep rows promote through the
    /// scheduler's own path and glue rows carry no logits at all.
    pub logits: Vec<Tensor>,
    /// The creep's residual after this wave, to hold for the next one. `None`
    /// when no creep rode along.
    pub creep_residual: Option<Tensor>,
}

/// The rows a verify step feeds its wave, and which group slot each goes in.
///
/// This is the whole of what a model has to say about its verify wave, and it
/// exists so the model does not have to *own* that wave. The scheduler's
/// forward is the continuous-fair wave, with the creep group and the glue on
/// it; a model method that issued its own `forward_wave` would have to be
/// handed all of that, and would then be the custodian of scheduler state it
/// has no business holding. Instead the model plans the rows, the caller runs
/// whatever wave it runs, and the model reads the logits back.
///
/// The two slots are not interchangeable. The decode slot is one row per
/// sequence — priced, admitted and attended that way — so a multi-token block
/// cannot go in it. The prefill slot takes multi-token members. Models split
/// this differently and both are legal: the `qwen35` lineage puts plain rows in
/// decode and each block in prefill as one multi-token member, while
/// DeepSeek-V4 puts *every* row — plain tokens and block positions alike — in
/// the decode slot as its own one-token row.
pub struct VerifyPlan {
    /// One-token rows for the wave's decode slot.
    pub decode_seqs: Vec<usize>,
    pub decode_inputs: Vec<Tensor>,
    /// Multi-token rows for the wave's prefill slot.
    pub verify_seqs: Vec<usize>,
    pub verify_inputs: Vec<Tensor>,
    /// Scored rows the wave must hand back — checked against what it did, so a
    /// head that scored one row per member instead of one per token fails here
    /// rather than silently reading another sequence's row.
    pub rows: usize,
}

/// Token-row counts per group, which are what the residual splice narrows on.
///
/// Decode contributes one row per sequence whatever its input says; every other
/// group contributes one row per token. Getting this wrong does not fail — it
/// slices another group's rows — so it is derived once, here, and every narrow
/// below is expressed in terms of it.
struct Rows {
    decode: usize,
    verify: usize,
    creep: usize,
    glue: usize,
}

impl Rows {
    fn tokens(inputs: &[Tensor]) -> usize {
        inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(0))
            .sum()
    }

    fn new(n_decode: usize, verify: &[Tensor], co: &WaveCoBatch<'_>) -> Self {
        Self {
            decode: n_decode,
            verify: Self::tokens(verify),
            creep: Self::tokens(co.creep_inputs),
            glue: Self::tokens(co.glue_inputs),
        }
    }

    /// Rows ahead of the creep in caller order — `[decode | verify]`.
    fn before_creep(&self) -> usize {
        self.decode + self.verify
    }
}

/// Concatenate the present pieces of a residual in caller order.
///
/// Tokens are dim 1. A single piece is passed through rather than cat'd with
/// itself, and no pieces at all means "embed fresh", which is what a wave
/// starting at layer 0 wants.
fn splice(parts: &[Option<&Tensor>]) -> Result<Option<Tensor>> {
    let present: Vec<&Tensor> = parts.iter().copied().flatten().collect();
    match present.len() {
        0 => Ok(None),
        1 => Ok(Some(present[0].clone())),
        _ => Ok(Some(Tensor::cat(&present, 1)?)),
    }
}

/// Split a segment's `[decode | verify | glue]` residual into the part ahead of
/// where the creep belongs and the part behind it.
fn around_creep(
    residual: &Option<Tensor>,
    rows: &Rows,
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let Some(r) = residual else {
        return Ok((None, None));
    };
    let head = rows.before_creep();
    let ahead = if head > 0 {
        Some(r.narrow(1, 0, head)?)
    } else {
        None
    };
    let behind = if rows.glue > 0 {
        Some(r.narrow(1, head, rows.glue)?)
    } else {
        None
    };
    Ok((ahead, behind))
}

/// Issue the verify wave: up to three segments so the full-sweep members cross
/// every layer while the creep advances only inside its window.
///
/// `decode_seqs`/`decode_inputs` are the plain one-token rows (sequences whose
/// drafter proposed nothing) and `verify_seqs`/`verify_inputs` the multi-token
/// blocks. Both are full-sweep. Returns the `[decode | verify]` logits prefix
/// and the creep's residual to hold.
///
/// The model is borrowed shared, so a caller holding `&mut` on its own wave
/// state — cursor, residual, group membership — can still call this with the
/// model it also owns.
pub fn issue_verify_wave<M>(
    model: &M,
    session: &mut BatchedInferenceSession,
    decode_seqs: &[usize],
    decode_inputs: &[Tensor],
    verify_seqs: &[usize],
    verify_inputs: &[Tensor],
    co: &WaveCoBatch<'_>,
    layer_end: usize,
) -> Result<VerifyWaveOutput>
where
    M: ManagedBatchedModel + ?Sized,
{
    if decode_seqs.len() != decode_inputs.len() || verify_seqs.len() != verify_inputs.len() {
        candle::bail!(
            "issue_verify_wave: {} decode seqs against {} inputs, {} verify seqs against {}",
            decode_seqs.len(),
            decode_inputs.len(),
            verify_seqs.len(),
            verify_inputs.len()
        );
    }
    let rows = Rows::new(decode_seqs.len(), verify_inputs, co);
    let want = rows.before_creep();

    // No creep to work around: one forward over every layer, which is what a
    // standalone verify and every non-creep wave run.
    if !co.has_creep() {
        let step = model.forward_wave(
            session,
            decode_seqs,
            decode_inputs,
            verify_seqs,
            verify_inputs,
            co.glue_seqs,
            co.glue_inputs,
            0,
            layer_end,
            None,
        )?;
        let mut logits = step.logits_owned()?;
        logits.truncate(want);
        return Ok(VerifyWaveOutput {
            logits,
            creep_residual: co.creep_residual.clone(),
        });
    }

    let cursor = co.creep_window.start.min(layer_end);
    let win_end = co.creep_window.end.min(layer_end);
    // The creep group as the prefill slot sees it in segment 2: verify rows
    // first so `[decode | verify]` stays a contiguous prefix in every segment,
    // which is what makes the logits truncation above and the narrows below
    // read the same rows regardless of which segment reached the head.
    let mid_seqs: Vec<usize> = verify_seqs.iter().chain(co.creep_seqs).copied().collect();
    let mid_inputs: Vec<Tensor> = verify_inputs
        .iter()
        .chain(co.creep_inputs)
        .cloned()
        .collect();

    // Segment 1 — full-sweep only, over [0, cursor).
    let seg1: Option<Tensor> = if cursor > 0 {
        model
            .forward_wave(
                session,
                decode_seqs,
                decode_inputs,
                verify_seqs,
                verify_inputs,
                co.glue_seqs,
                co.glue_inputs,
                0,
                cursor,
                None,
            )?
            .into_residual()
    } else {
        None
    };
    let (seg1_ahead, seg1_behind) = around_creep(&seg1, &rows)?;

    // Segment 2 — everything, over [cursor, win_end). Caller order
    // `[decode | verify | creep | glue]`.
    let seg2_in = splice(&[
        seg1_ahead.as_ref(),
        co.creep_residual.as_ref(),
        seg1_behind.as_ref(),
    ])?;
    let seg2 = model.forward_wave(
        session,
        decode_seqs,
        decode_inputs,
        &mid_seqs,
        &mid_inputs,
        co.glue_seqs,
        co.glue_inputs,
        cursor,
        win_end,
        seg2_in,
    )?;

    // The head runs at the end of the sweep, so whichever segment finishes at
    // `layer_end` is the one carrying logits.
    if win_end >= layer_end {
        let mut logits = seg2.logits_owned()?;
        logits.truncate(want);
        // The creep did not finish its own sweep, so its residual is held even
        // though this wave reached the head for everyone else.
        let creep_residual = seg2
            .into_residual()
            .map(|r| r.narrow(1, rows.before_creep(), rows.creep))
            .transpose()?;
        return Ok(VerifyWaveOutput {
            logits,
            creep_residual,
        });
    }

    // Lift the creep's slice out to hold, and carry `[decode | verify | glue]`
    // on into segment 3.
    let seg2_res = seg2.into_residual();
    let (creep_residual, seg3_in) = match &seg2_res {
        Some(r) => {
            let held = r.narrow(1, rows.before_creep(), rows.creep)?;
            let ahead = if want > 0 {
                Some(r.narrow(1, 0, want)?)
            } else {
                None
            };
            let behind = if rows.glue > 0 {
                Some(r.narrow(1, rows.before_creep() + rows.creep, rows.glue)?)
            } else {
                None
            };
            (Some(held), splice(&[ahead.as_ref(), behind.as_ref()])?)
        }
        None => (None, None),
    };

    // Segment 3 — full-sweep only, over [win_end, layer_end).
    let seg3 = model.forward_wave(
        session,
        decode_seqs,
        decode_inputs,
        verify_seqs,
        verify_inputs,
        co.glue_seqs,
        co.glue_inputs,
        win_end,
        layer_end,
        seg3_in,
    )?;
    let mut logits = seg3.logits_owned()?;
    logits.truncate(want);
    Ok(VerifyWaveOutput {
        logits,
        creep_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn row(tokens: usize) -> Tensor {
        Tensor::zeros((1, tokens), candle::DType::U32, &Device::Cpu).unwrap()
    }

    /// Decode contributes one row per SEQUENCE and everything else one row per
    /// TOKEN. The splice narrows on these counts, so a decode row that was
    /// priced by its token length would slice into the verify block's rows.
    #[test]
    fn decode_counts_sequences_and_the_rest_count_tokens() {
        let verify = [row(3), row(2)];
        let creep = [row(40)];
        let glue = [row(7), row(1)];
        let co = WaveCoBatch {
            creep_seqs: &[9],
            creep_inputs: &creep,
            glue_seqs: &[10, 11],
            glue_inputs: &glue,
            creep_residual: None,
            creep_window: 4..8,
        };
        let rows = Rows::new(5, &verify, &co);
        assert_eq!(rows.decode, 5);
        assert_eq!(rows.verify, 5);
        assert_eq!(rows.creep, 40);
        assert_eq!(rows.glue, 8);
        // `[decode | verify]` is the prefix every segment shares.
        assert_eq!(rows.before_creep(), 10);
    }

    /// A standalone verify has no creep however the window is set, and a creep
    /// group with an empty window is equally absent — a caller can hand over a
    /// window it did not fill, and segmenting on it would run a zero-layer
    /// forward.
    #[test]
    fn creep_needs_both_members_and_a_window() {
        assert!(!WaveCoBatch::standalone().has_creep());
        let creep = [row(4)];
        let with_members_no_window = WaveCoBatch {
            creep_seqs: &[1],
            creep_inputs: &creep,
            creep_window: 3..3,
            ..WaveCoBatch::standalone()
        };
        assert!(!with_members_no_window.has_creep());
        let real = WaveCoBatch {
            creep_seqs: &[1],
            creep_inputs: &creep,
            creep_window: 3..9,
            ..WaveCoBatch::standalone()
        };
        assert!(real.has_creep());
    }

    /// The splice drops absent pieces, passes a lone piece through untouched,
    /// and concatenates on the TOKEN axis (dim 1) — the residual is
    /// `[batch, tokens, hidden]`, so cat'ing on dim 0 would silently build a
    /// batch instead of a sequence.
    #[test]
    fn splice_concatenates_on_the_token_axis() -> Result<()> {
        let a = Tensor::zeros((1, 2, 4), candle::DType::F32, &Device::Cpu)?;
        let b = Tensor::zeros((1, 5, 4), candle::DType::F32, &Device::Cpu)?;
        assert!(splice(&[None, None])?.is_none());
        assert_eq!(splice(&[Some(&a), None])?.unwrap().dims(), &[1, 2, 4]);
        assert_eq!(splice(&[Some(&a), Some(&b)])?.unwrap().dims(), &[1, 7, 4]);
        Ok(())
    }

    /// The creep's slot in caller order sits between `[decode | verify]` and
    /// the glue, so segment 1's residual splits into exactly those two pieces
    /// with the creep's own residual inserted between them.
    #[test]
    fn segment_one_residual_splits_around_the_creep_slot() -> Result<()> {
        let verify = [row(3)];
        let glue = [row(2)];
        let co = WaveCoBatch {
            creep_seqs: &[9],
            creep_inputs: &[row(40)],
            glue_seqs: &[10],
            glue_inputs: &glue,
            creep_residual: None,
            creep_window: 4..8,
        };
        let rows = Rows::new(2, &verify, &co);
        // [decode 2 | verify 3 | glue 2] = 7 token rows.
        let seg1 = Some(Tensor::zeros((1, 7, 4), candle::DType::F32, &Device::Cpu)?);
        let (ahead, behind) = around_creep(&seg1, &rows)?;
        assert_eq!(ahead.unwrap().dims(), &[1, 5, 4]);
        assert_eq!(behind.unwrap().dims(), &[1, 2, 4]);
        Ok(())
    }

    /// With no glue there is nothing behind the creep, and the split must say
    /// so rather than narrowing a zero-width tail.
    #[test]
    fn no_glue_leaves_nothing_behind_the_creep() -> Result<()> {
        let verify = [row(3)];
        let co = WaveCoBatch {
            creep_seqs: &[9],
            creep_inputs: &[row(40)],
            creep_window: 4..8,
            ..WaveCoBatch::standalone()
        };
        let rows = Rows::new(2, &verify, &co);
        let seg1 = Some(Tensor::zeros((1, 5, 4), candle::DType::F32, &Device::Cpu)?);
        let (ahead, behind) = around_creep(&seg1, &rows)?;
        assert_eq!(ahead.unwrap().dims(), &[1, 5, 4]);
        assert!(behind.is_none());
        Ok(())
    }
}
