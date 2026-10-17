//! Issuing the one wave a standalone speculative verify step runs on.
//!
//! # Why a verify block cannot simply be "a prefill"
//!
//! A verify block is multi-token, so it cannot ride the decode slot: that slot
//! is one row per sequence, priced and admitted as one row per sequence. But it
//! is emphatically **full-sweep** — its logits are consumed by the accept walk
//! in the same step that produced them, so it crosses every layer of the wave
//! it rides. So verify rows occupy the prefill slot as full-sweep members.
//!
//! A standalone verify — a test, the batch harness — is one forward over every
//! layer carrying nothing but its own rows, which is what [`issue_verify_wave`]
//! runs. The scheduler's verify rides its co-batched step instead, beside the
//! creep (`docs/wave_step.md` §2), through the same plan ([`VerifyPlan`]) and
//! the same read-back.

use candle::{Device, Result, Tensor};

use super::batched_inference::{BatchedInferenceSession, ManagedBatchedModel, WaveResult};

/// A verify plan's token rows, built **once per group** on the host: the plain
/// cohort's tokens as one `[n, 1]` tensor with each sequence's input a
/// `[1, 1]` view of its row, and the blocks as one `[1, Σ len]` tensor with
/// each block a `[1, len]` view of its run. Returned as `(plain inputs, block
/// inputs)`, in the order given.
///
/// **On the host, and the forward uploads them.** Every model lays a wave's
/// ids on its forward span in one copy (`wave_token_ids`), host ids beside any
/// already on the device, so building them on the device here would be a pool
/// allocation ahead of the forward for ids the forward copies anyway.
pub fn upload_plan_rows(plain: &[u32], blocks: &[Vec<u32>]) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    let device = &Device::Cpu;
    let plain_inputs = if plain.is_empty() {
        Vec::new()
    } else {
        let all = Tensor::from_vec(plain.to_vec(), (plain.len(), 1), device)?;
        (0..plain.len())
            .map(|i| all.narrow(0, i, 1))
            .collect::<Result<_>>()?
    };
    let block_inputs = if blocks.is_empty() {
        Vec::new()
    } else {
        let flat: Vec<u32> = blocks.iter().flatten().copied().collect();
        let total = flat.len();
        let all = Tensor::from_vec(flat, (1, total), device)?;
        let mut start = 0;
        let mut out = Vec::with_capacity(blocks.len());
        for b in blocks {
            out.push(all.narrow(1, start, b.len())?);
            start += b.len();
        }
        out
    };
    Ok((plain_inputs, block_inputs))
}

/// What one verify wave produced.
pub struct VerifyWaveOutput {
    /// Per-row logits in caller order, `[decode | verify]` — the rows the
    /// accept walk reads.
    ///
    /// **Views on the head's span, not copies.** They are valid while
    /// [`Self::head`] is held; a caller reads them — the accept walk — and
    /// drops both before the next forward opens, so a decode step allocates and
    /// copies nothing to hand its scored rows to the walk.
    pub logits: Vec<Tensor>,
    /// The forward that reached the head — the guard keeping [`Self::logits`]
    /// valid. Dropped once they have been read.
    pub head: WaveResult,
}

/// The rows a verify step feeds its wave, and which group slot each goes in.
///
/// This is the whole of what a model has to say about its verify wave, and it
/// exists so the model does not have to *own* that wave. The scheduler's
/// forward is the co-batched step, with the creep and the glue on it; a model
/// method that issued its own forward would have to be handed all of that, and
/// would then be the custodian of scheduler state it has no business holding.
/// Instead the model plans the rows, the caller runs whatever wave it runs, and
/// the model reads the logits back.
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

/// Issue a standalone verify wave: one forward over `[0, layer_end)` carrying
/// the plain one-token rows (`decode_seqs`/`decode_inputs`, sequences whose
/// drafter proposed nothing) and the multi-token blocks (`verify_seqs`/
/// `verify_inputs`). Returns the `[decode | verify]` logits with the forward
/// that holds them.
///
/// The model is borrowed shared, so a caller holding `&mut` on its own state
/// can still call this with the model it also owns.
pub fn issue_verify_wave<M>(
    model: &M,
    session: &mut BatchedInferenceSession,
    decode_seqs: &[usize],
    decode_inputs: &[Tensor],
    verify_seqs: &[usize],
    verify_inputs: &[Tensor],
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
    let head = model.forward_wave(
        session,
        decode_seqs,
        decode_inputs,
        verify_seqs,
        verify_inputs,
        &[],
        &[],
        0,
        layer_end,
        None,
    )?;
    // The scored rows stay on the head's span, read through the result that
    // holds it.
    let logits = head.logits_on_span();
    Ok(VerifyWaveOutput { logits, head })
}
