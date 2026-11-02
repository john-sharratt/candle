//! The scheduler's side of the capped diagnostic decode
//! (`candle_transformers::models::decode_diag`): hold the card to the armed
//! timeline, and record what each of its steps read and produced. Reached only
//! behind `decode_diag::armed()`.

use candle::{DType, Tensor};
use candle_transformers::models::decode_diag;

use super::stencil_wave::{StencilRow, StencilWave};
use super::*;

impl Scheduler {
    /// Whether `id` decodes on `timeline`.
    fn on_timeline(&self, id: SequenceId, timeline: u64) -> bool {
        self.active_decodes
            .get(&id)
            .and_then(|s| s.reprojection.as_ref())
            .is_some_and(|p| p.target.timeline.raw() == timeline)
    }

    /// Hold this wave to the armed timeline, alone, so nothing another
    /// conversation does can reach the armed decode, and tell the capture
    /// which step it is and what the slot holds. A held row waits, as an
    /// adapter's does.
    #[cold]
    pub(super) fn diag_step(&mut self, seq_ids: &mut Vec<SequenceId>, stencil: &mut StencilWave) {
        let Some(arm) = decode_diag::armed_with() else {
            return;
        };
        seq_ids.retain(|&id| self.on_timeline(id, arm.timeline));
        let (carried, held): (Vec<StencilRow>, Vec<StencilRow>) = std::mem::take(&mut stencil.rows)
            .into_iter()
            .partition(|r| self.on_timeline(r.seq, arm.timeline));
        stencil.rows = carried;
        self.stencil_rows.extend(held);

        let Some(&lead) = seq_ids.first().or(stencil.rows.first().map(|r| &r.seq)) else {
            return;
        };
        let step = self
            .active_decodes
            .get(&lead)
            .map_or(0, |s| s.generated_tokens.len());
        let rows = seq_ids.len() + stencil.rows.len();
        let recording = step >= arm.from;
        let seqs = seq_ids
            .iter()
            .map(|id| id.0)
            .filter(|_| recording)
            .collect::<Vec<_>>();
        decode_diag::set_step(seqs, step as u64, rows);
        let parent = self.turn_views.get(&lead).map_or(lead, |v| v.parent_id);
        if let Some(slot) = self
            .slot_projection_state
            .get(&parent)
            .filter(|_| step < arm.live_at)
        {
            decode_diag::set_pieces(slot.placed_spans());
        }
    }

    /// Record the rows behind `next_tokens`, as sampled.
    #[cold]
    pub(super) fn diag_commit(
        &self,
        seq_ids: &[SequenceId],
        next_tokens: &[u32],
        logits_vec: &[Tensor],
    ) {
        for (i, id) in seq_ids.iter().enumerate() {
            if !decode_diag::watching(id.0) {
                continue;
            }
            let row = logits_vec[i]
                .to_dtype(DType::F32)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>());
            match row {
                Ok(row) => decode_diag::record_step(id.0, &row, next_tokens[i]),
                Err(e) => tracing::warn!(seq = id.0, "diag: reading the logits failed ({e})"),
            }
        }
    }

    /// Under the fresh control, forget what the slot holds so the rebuild
    /// splices nothing.
    #[cold]
    pub(super) fn diag_forget_placement(&mut self, view: SequenceId) {
        let parent = self.turn_views.get(&view).map_or(view, |v| v.parent_id);
        if let Some(slot) = self.slot_projection_state.get_mut(&parent) {
            slot.placed_pieces.clear();
        }
    }
}
