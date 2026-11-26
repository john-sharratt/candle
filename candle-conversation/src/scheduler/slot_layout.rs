//! Describes what a slot holds at its turn's end, before its seal: every K/V
//! chunk on every layer — its window and the format of each `(head, palette)`
//! band — and every page of the model's per-position index, with the tokens
//! it covers ([`SchedulerRequest::DescribeAtSeal`]).
//!
//! Taken where the turn's view has been finalized onto its parent and nothing
//! has been sealed, re-quantized or truncated, so two slots described here are
//! compared exactly as their last forward read them.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use candle_nn::kv_cache::ArenaFormatTag;
use serde::Serialize;

use super::*;

/// One chunk of a slot: its window, and its bands' formats on every layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChunkLayout {
    /// The window's start within its physical chunk.
    pub offset: u16,
    /// Valid tokens in the window.
    pub tokens: u16,
    /// Per layer, the K bands' format tags in `(head, palette)` order
    /// ([`format_name`] reads one); empty for a chunk that names none.
    pub k: Vec<Vec<u8>>,
    /// The same for V.
    pub v: Vec<Vec<u8>>,
    /// Whether every layer holds this chunk with the same window.
    pub layers_agree: bool,
}

/// One page of the model's per-position index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndexPage {
    pub tokens: usize,
    pub bytes: usize,
    /// A hash of the page's bytes, to tell two equal-length pages apart.
    pub hash: u64,
}

/// What a slot holds at its turn's end.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SlotLayout {
    /// The slot's depth in tokens.
    pub depth: usize,
    /// The tokens the model's per-position state covers.
    pub index_coverage: Option<usize>,
    pub index_pages: Vec<IndexPage>,
    pub chunks: Vec<ChunkLayout>,
}

/// A band's format tag by name.
pub fn format_name(tag: u8) -> String {
    format!("{:?}", ArenaFormatTag::from_u8(tag))
}

impl Scheduler {
    /// Describe `slot` as it stands ([`SlotLayout`]).
    pub(super) fn describe_slot(&self, slot: SequenceId) -> Result<SlotLayout, ConversationError> {
        let backings = self.session.backings();
        let mut per_layer = Vec::with_capacity(backings.len());
        for backing in backings.iter() {
            per_layer.push(
                backing
                    .record_turn(slot.0)
                    .map_err(ConversationError::Model)?,
            );
        }
        let first = per_layer
            .first()
            .ok_or_else(|| ConversationError::Channel("describe_slot: no layers".into()))?;
        let chunks = (0..first.chunks.len())
            .map(|i| {
                let c = &first.chunks[i];
                let layers_agree = per_layer.iter().all(|s| {
                    s.chunks
                        .get(i)
                        .is_some_and(|o| o.offset == c.offset && o.token_count == c.token_count)
                });
                ChunkLayout {
                    offset: c.offset,
                    tokens: c.token_count,
                    k: per_layer
                        .iter()
                        .map(|s| s.chunks.get(i).map_or(Vec::new(), |o| o.k_fmt.to_vec()))
                        .collect(),
                    v: per_layer
                        .iter()
                        .map(|s| s.chunks.get(i).map_or(Vec::new(), |o| o.v_fmt.to_vec()))
                        .collect(),
                    layers_agree,
                }
            })
            .collect();
        let depth = self.session.sequence_offset(slot.0).unwrap_or(0);
        let index_pages = self
            .model
            .seal_positional_tail_span(slot.0, depth)
            .map_err(ConversationError::Model)?
            .into_iter()
            .map(|(tokens, page)| {
                let mut h = DefaultHasher::new();
                page.hash(&mut h);
                IndexPage {
                    tokens,
                    bytes: page.len(),
                    hash: h.finish(),
                }
            })
            .collect();
        Ok(SlotLayout {
            depth,
            index_coverage: self.model.positional_coverage(slot.0),
            index_pages,
            chunks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A tag reads as its format's name.**
    #[test]
    fn a_tag_reads_as_its_formats_name() {
        assert_eq!(format_name(ArenaFormatTag::Q4_0.as_u8()), "Q4_0");
        assert_eq!(format_name(ArenaFormatTag::Q8_0.as_u8()), "Q8_0");
    }
}
