//! Assembles the per-token metadata a gap-fill forward needs to scatter
//! reprojection glue tokens into their reserved gap chunks.
//!
//! Each glue token's position comes from its chunk's rope base, the convention
//! decode already reads, so no separate position map is carried. The staged
//! descriptor records only where each glue token scatters and how far forward
//! it bridges. When no slot carries glue there is nothing to build, and the
//! caller falls back to plain prefill. The route a gap-fill took is logged once
//! per reprojection.

use super::*;

/// Assemble a [`GlueMeta`] from the wave's per-slot glue descriptors for a
/// gap-fill forward. Each `pending[i]`'s three vectors have length
/// `input_lens[i]` (the slot's glue-token count); they are concatenated in the
/// forward's flat `q` order. There is no `col_actual_pos`: the kernel derives
/// every column's position from its chunk `rope_base` (`slice_rope`), the same
/// convention decode reads. Returns `None` (falls back to plain prefill) only
/// when no slot actually carries glue.
#[cfg(feature = "cuda")]
pub(crate) fn build_glue_meta(
    pending: Vec<PendingGlue>,
    input_lens: &[usize],
    device: &Device,
    ticket: Option<WaveTicket>,
) -> Result<Option<GlueMeta>> {
    if pending.len() != input_lens.len() {
        candle::bail!(
            "build_glue_meta: {} glue descriptors vs {} input_lens",
            pending.len(),
            input_lens.len()
        );
    }
    let total: usize = input_lens.iter().sum();
    let mut write_slice: Vec<u32> = Vec::with_capacity(total);
    let mut write_in_blk: Vec<u32> = Vec::with_capacity(total);
    let mut fwd_ahead: Vec<u32> = Vec::with_capacity(total);
    for (i, p) in pending.iter().enumerate() {
        if p.write_slice.len() != input_lens[i]
            || p.write_in_blk.len() != input_lens[i]
            || p.fwd_ahead.len() != input_lens[i]
        {
            candle::bail!(
                "build_glue_meta: slot {i} glue len ({}/{}/{}) != input_len {}",
                p.write_slice.len(),
                p.write_in_blk.len(),
                p.fwd_ahead.len(),
                input_lens[i]
            );
        }
        write_slice.extend_from_slice(&p.write_slice);
        write_in_blk.extend_from_slice(&p.write_in_blk);
        fwd_ahead.extend_from_slice(&p.fwd_ahead);
    }
    if write_slice.is_empty() {
        return Ok(None);
    }
    let n = write_slice.len();
    // Confirms the gap-fill forward took the paged-glue route (HD128) — one line
    // per reproject, under the scheduler's reproject log target.
    tracing::trace!(
        target: "candle_conversation::scheduler::reproject",
        slots = pending.len(),
        total_glue = total,
        "paged-glue route active"
    );
    // On the forward span `ticket` names: three per-row tables every layer's
    // glue scatter reads (`WaveBuffer::GlueWriteTables`).
    Ok(Some(GlueMeta {
        glue_write_slice: wave_from_vec_ticketed(write_slice, n, device, ticket)?,
        glue_write_in_blk: wave_from_vec_ticketed(write_in_blk, n, device, ticket)?,
        fwd_ahead: wave_from_vec_ticketed(fwd_ahead, n, device, ticket)?,
    }))
}

/// Per-slot reprojection-glue descriptor staged on the session for one gap-fill
/// `forward_batched`. The glue tokens are reserved IN PLACE as gap chunks during
/// assembly, so there is no `col_actual_pos`: every column's sequence position
/// comes from its chunk's `rope_base` (`slice_rope`), the same convention decode
/// reads. This carries only what the kernel can't derive from the slot itself —
/// where each glue token scatters, and how far it bridges forward.
#[derive(Clone, Debug)]
pub struct PendingGlue {
    /// Per glue token: the gap chunk's block index its K/V scatters into.
    pub write_slice: Vec<u32>,
    /// Per glue token: the in-block offset within its gap chunk.
    pub write_in_blk: Vec<u32>,
    /// Per glue token: forward bridge window in tokens (`0` == backward-only).
    pub fwd_ahead: Vec<u32>,
}
