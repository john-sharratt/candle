//! Placing a span of a turn's prefill from another turn's K/V
//! (`docs/inline_read_injection.md` §3).
//!
//! The prefill has run every layer up to the span's start. The span goes onto
//! the slot's tail as the source turn's K/V windowed to it, with the source's
//! index pages sliced to it, and the prefill carries on past it: its tokens are
//! never forwarded, so the recurrent state passes over them unchanged.

use candle_transformers::models::piece_key::PieceKey;

use super::{ConversationError, Scheduler, SequenceId};
use crate::conversation::window_sealed_tokens;
use crate::index_pages;
use crate::placed_span::PlacedSpan;

impl Scheduler {
    /// Place `span` on `seq`, whose prefill has just reached its start.
    ///
    /// Everything that can refuse is read before the slot is touched: the
    /// source's hot K/V, its index pages and their slices. Only then does the
    /// index tail close — the head's rows become a page, so the placed pages
    /// land on an empty tail — and the K/V and pages go in.
    pub(super) fn place_span(
        &mut self,
        seq: SequenceId,
        span: &PlacedSpan,
    ) -> Result<(), ConversationError> {
        let refused = |why: String| {
            ConversationError::Other(format!(
                "placing turn {}/{} tokens {:?} at {:?}: {why}",
                span.source.raw(),
                span.turn.0,
                span.source_tokens,
                span.tokens
            ))
        };
        let width = span.source_tokens.len();
        if span.tokens.len() != width {
            return Err(refused(format!(
                "the span is {} tokens and its source {width}",
                span.tokens.len()
            )));
        }
        let parent = self.turn_views.get(&seq).map_or(seq, |v| v.parent_id);
        let Some(conversation) = self.slot_conversations.get(&parent).cloned() else {
            return Err(refused(format!("no conversation holds slot {parent}")));
        };
        let Some(sealed) = conversation.read().turn_sealed_of(span.source, span.turn) else {
            return Err(refused("the source turn holds no hot K/V".to_string()));
        };
        let kv = window_sealed_tokens(&sealed, span.source_tokens.start, span.source_tokens.end);
        if let Some(short) = kv.iter().find(|layer| layer.token_count != width) {
            return Err(refused(format!(
                "the source's K/V holds {} of the {width} tokens",
                short.token_count
            )));
        }

        let mut pages: Vec<Vec<u8>> = Vec::new();
        if self.model.carries_positional_state() {
            let Some(stored) = conversation.turn_index_page(span.source, span.turn)? else {
                return Err(refused("the source turn has no index pages".to_string()));
            };
            let framed = index_pages::decode(stored.bytes()).map_err(|e| refused(e.to_string()))?;
            let mut start = 0usize;
            for (page_width, bytes) in framed {
                let end = start + page_width;
                let from = span.source_tokens.start.max(start);
                let to = span.source_tokens.end.min(end);
                if from < to {
                    let Some(slice) = self
                        .model
                        .slice_positional_page(bytes, from - start..to - start)
                        .map_err(|e| refused(e.to_string()))?
                    else {
                        return Err(refused(
                            "the model carries pages it cannot slice".to_string(),
                        ));
                    };
                    pages.push(slice);
                }
                start = end;
            }
            if start < span.source_tokens.end {
                return Err(refused(format!(
                    "the source's pages cover {start} tokens, short of the span"
                )));
            }
        }

        self.model
            .close_positional_page(seq.0)
            .map_err(ConversationError::Model)?;
        let injected = self
            .session
            .inject_sealed_at_tail(seq.0, &kv)
            .map_err(ConversationError::Model)?;
        if injected.tokens != width || injected.was_cut() {
            return Err(refused(format!(
                "the slot took {} of the {width} tokens",
                injected.tokens
            )));
        }
        for page in &pages {
            self.model
                .push_positional_state(seq.0, page, &PieceKey::of(page))
                .map_err(ConversationError::Model)?;
        }
        tracing::info!(
            target: "candle_conversation::scheduler::placed",
            seq = seq.0,
            source = span.source.raw(),
            turn = span.turn.0,
            at = span.tokens.start,
            tokens = width,
            pages = pages.len(),
            "placed a span from a sealed turn"
        );
        Ok(())
    }
}
