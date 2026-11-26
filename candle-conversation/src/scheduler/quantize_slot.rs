//! Quantizes a live slot's K/V in place ([`SchedulerRequest::QuantizeSlot`]).
//!
//! Every layer's blocks are recorded as one sealed sequence, quantized under
//! the adaptive policy at the asked level, and put back at the slot's head in
//! place of the native chunks — the same record → quantize → inject a section
//! seal and a projection make, without the substrate. The slot keeps its
//! depth, its recurrent state and its index; only the blocks' format changes,
//! so its next turn attends over the K/V production would hand it.

use candle::Device;
use candle_nn::kv_cache::{quantize_sealed_in_place, CompressionPolicy};

use super::*;

impl Scheduler {
    /// Quantize every block `sequence_id` holds at `level`; the blocks it holds
    /// afterwards.
    pub(super) fn quantize_slot(
        &mut self,
        sequence_id: SequenceId,
        level: u8,
    ) -> Result<usize, ConversationError> {
        let device = self.session.device().clone();
        let copy_stream = match &device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => {
                return Err(ConversationError::Channel(
                    "quantize_slot requires a CUDA device".into(),
                ))
            }
        };
        let policy = CompressionPolicy::new(level);
        let backings = self.session.backings();
        let mut quantized = Vec::with_capacity(backings.len());
        for backing in backings.iter() {
            let native = backing
                .record_turn(sequence_id.0)
                .map_err(ConversationError::Model)?;
            let mut out = quantize_sealed_in_place(
                backing,
                &[&native],
                &policy,
                &device,
                &copy_stream,
                &mut self.quantize_pinned_scratch,
            )
            .map_err(ConversationError::Model)?;
            quantized.push(out.remove(0));
        }
        copy_stream
            .synchronize()
            .map_err(|e| ConversationError::Channel(format!("quantize_slot sync: {e}")))?;
        let mut blocks = 0;
        for (backing, sealed) in backings.iter().zip(&quantized) {
            backing
                .truncate_sequence_to_blocks(sequence_id.0, 0)
                .map_err(ConversationError::Model)?;
            let (_, end) = backing
                .inject_sealed_at_tail(sequence_id.0, sealed)
                .map_err(ConversationError::Model)?;
            blocks = end;
        }
        Ok(blocks)
    }
}
