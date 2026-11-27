//! Quantizes a live slot's K/V in place, span by span
//! ([`SchedulerRequest::QuantizeSpans`]).
//!
//! The slot's chunks are cut into the spans it was asked about — each a run of
//! tokens the slot was prefilled with as one piece, ending in its own partial
//! chunk — and every layer's chunks of a span are recorded as one sealed
//! sequence, quantized under the span's policy, and put back in order in place
//! of the native chunks. It is the record → quantize → inject a section seal
//! and a projection make, partial final blocks included, without the
//! substrate. The slot keeps its depth, its recurrent state and its index; only
//! the blocks' format changes, so its next turn attends over K/V in the layout
//! and the formats production hands it.

use candle::Device;
use candle_nn::kv_cache::{quantize_sealed_in_place, CompressionPolicy, SealedSequence};

use super::*;

/// The policy one span of a slot is quantized under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanQuant {
    /// Left as the forward wrote it.
    Native,
    /// The adaptive policy at a level — `4` is a collection member's seal.
    Level(u8),
    /// A boundary section's seal: Q8 K and V.
    Boundary,
    /// The session's turn policy, as the gap quantizes a turn's blocks.
    Session,
}

/// Where each span's chunks sit: `(first_block, end_block)` for every span,
/// in order, from the chunks' token counts. A chunk holding tokens of two
/// spans is refused — the slot was not laid out one piece per span — and an
/// empty chunk (a writer opened and never written) belongs to the span before
/// it.
pub(super) fn span_blocks(
    chunk_tokens: &[usize],
    span_tokens: &[usize],
) -> Result<Vec<(usize, usize)>, ConversationError> {
    let mut out = Vec::with_capacity(span_tokens.len());
    let mut block = 0;
    for (i, &want) in span_tokens.iter().enumerate() {
        let first = block;
        let mut held = 0;
        while held < want {
            let Some(&n) = chunk_tokens.get(block) else {
                return Err(ConversationError::Channel(format!(
                    "quantize_spans: span {i} wants {want} tokens, the slot ran out at {held}"
                )));
            };
            held += n;
            block += 1;
        }
        if held != want {
            return Err(ConversationError::Channel(format!(
                "quantize_spans: span {i} ends inside block {} ({held} tokens against {want}) \
                 — the slot was not laid out one piece per span",
                block - 1
            )));
        }
        while chunk_tokens.get(block) == Some(&0) {
            block += 1;
        }
        out.push((first, block));
    }
    if block != chunk_tokens.len() {
        return Err(ConversationError::Channel(format!(
            "quantize_spans: the spans cover {block} of the slot's {} blocks",
            chunk_tokens.len()
        )));
    }
    Ok(out)
}

impl Scheduler {
    /// Quantize the slot `sequence_id`, laid out one piece per span, span by
    /// span under each span's policy; the blocks it holds afterwards.
    pub(super) fn quantize_spans(
        &mut self,
        sequence_id: SequenceId,
        spans: &[(usize, SpanQuant)],
    ) -> Result<usize, ConversationError> {
        let device = self.session.device().clone();
        let copy_stream = match &device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => {
                return Err(ConversationError::Channel(
                    "quantize_spans requires a CUDA device".into(),
                ))
            }
        };
        let session_policy = self.session.seal_policy();
        let policy_of = |q: SpanQuant| -> Option<CompressionPolicy> {
            match q {
                SpanQuant::Native => None,
                SpanQuant::Level(level) => Some(CompressionPolicy::new(level)),
                SpanQuant::Boundary => Some(SectionSeal::Boundary.policy()),
                SpanQuant::Session => session_policy,
            }
        };
        let backings = self.session.backings();
        let layout = backings
            .first()
            .ok_or_else(|| ConversationError::Channel("quantize_spans: no layers".into()))?
            .record_turn(sequence_id.0)
            .map_err(ConversationError::Model)?;
        let chunk_tokens: Vec<usize> = layout
            .chunks
            .iter()
            .map(|c| c.token_count as usize)
            .collect();
        drop(layout);
        let span_tokens: Vec<usize> = spans.iter().map(|&(n, _)| n).collect();
        let ranges = span_blocks(&chunk_tokens, &span_tokens)?;

        let mut rebuilt: Vec<Vec<SealedSequence>> = Vec::with_capacity(backings.len());
        for backing in backings.iter() {
            let mut layer = Vec::with_capacity(spans.len());
            for (&(start, end), &(_, quant)) in ranges.iter().zip(spans) {
                let native = backing
                    .record_turn_blocks(sequence_id.0, start, end)
                    .map_err(ConversationError::Model)?;
                let sealed = match policy_of(quant) {
                    None => native,
                    Some(policy) => quantize_sealed_in_place(
                        backing,
                        &[&native],
                        &policy,
                        &device,
                        &copy_stream,
                        &mut self.quantize_pinned_scratch,
                    )
                    .map_err(ConversationError::Model)?
                    .remove(0),
                };
                layer.push(sealed);
            }
            rebuilt.push(layer);
        }
        copy_stream
            .synchronize()
            .map_err(|e| ConversationError::Channel(format!("quantize_spans sync: {e}")))?;
        let mut blocks = 0;
        for (backing, layer) in backings.iter().zip(&rebuilt) {
            backing
                .truncate_sequence_to_blocks(sequence_id.0, 0)
                .map_err(ConversationError::Model)?;
            for sealed in layer {
                let (_, end) = backing
                    .inject_sealed_at_tail(sequence_id.0, sealed)
                    .map_err(ConversationError::Model)?;
                blocks = end;
            }
        }
        Ok(blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Each span takes the blocks its tokens filled**, a partial tail
    /// included, and an empty writer left behind it goes with it.
    #[test]
    fn each_span_takes_the_blocks_its_tokens_filled() {
        // 40 tokens = one full block and a partial of 8; then 5 in one
        // partial, an empty writer after it; then 32 exactly.
        let chunks = [32, 8, 5, 0, 32];
        assert_eq!(
            span_blocks(&chunks, &[40, 5, 32]).unwrap(),
            vec![(0, 2), (2, 4), (4, 5)]
        );
    }

    /// **A chunk holding two spans' tokens is refused**: the slot was laid out
    /// contiguously, not one piece per span.
    #[test]
    fn a_chunk_shared_by_two_spans_is_refused() {
        let err = span_blocks(&[32, 13], &[40, 5]).unwrap_err().to_string();
        assert!(err.contains("ends inside block 1"), "{err}");
    }

    /// **Spans that do not cover the slot are refused**, short or long.
    #[test]
    fn spans_must_cover_the_slot_exactly() {
        assert!(span_blocks(&[32, 8], &[32]).is_err());
        assert!(span_blocks(&[32], &[32, 4]).is_err());
    }
}
