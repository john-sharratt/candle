//! A sequence's position frontier, kept apart from its token count
//! (`docs/placed_section_layout.md` §4.1).
//!
//! The offset is the tokens a sequence holds; its position frontier is where
//! its next token is read. They differ by the holes its block table leaves —
//! the empty part of a reserved range, the gap between two ranges — which every
//! layer's backing carries alike. Kernels read key and query positions off the
//! slices' RoPE bases, so the frontier is what a rung is chosen by and what an
//! index row is placed at; the offset is what a write length and a key count
//! are measured in.

use super::*;

impl BatchedInferenceSession {
    /// The position `idx`'s next token is read at: its offset plus the holes
    /// its block table leaves. `None` for an unallocated sequence.
    pub fn sequence_position(&self, idx: usize) -> Option<usize> {
        let offset = self.sequence_offset(idx)?;
        let holes = self
            .backings
            .first()
            .map_or(Ok(0), |b| b.lead_total(idx))
            .ok()?;
        Some(offset + holes)
    }

    /// Leave `n` positions empty at `idx`'s tail on every layer: what the
    /// sequence gains next is read `n` positions further on, and its offset
    /// does not move.
    pub fn push_sequence_position_gap(&self, idx: usize, n: usize) -> Result<()> {
        if self.sequence_offset(idx).is_none() {
            candle::bail!("push_sequence_position_gap: sequence {idx} not allocated");
        }
        for backing in &self.backings {
            backing.push_position_gap(idx, n)?;
        }
        Ok(())
    }

    /// Set `idx`'s holes on every layer from its layout, the sequence holding
    /// exactly `blocks` blocks: each `(block, lead)` is the hole before that
    /// block, `pending` the hole after the last
    /// (the backing's `set_position_leads`). Its offset does not move.
    pub fn set_sequence_position_leads(
        &self,
        idx: usize,
        blocks: usize,
        leads: &[(usize, u32)],
        pending: u32,
    ) -> Result<()> {
        if self.sequence_offset(idx).is_none() {
            candle::bail!("set_sequence_position_leads: sequence {idx} not allocated");
        }
        for backing in &self.backings {
            backing.set_position_leads(idx, blocks, leads, pending)?;
        }
        Ok(())
    }
}
