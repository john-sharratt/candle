//! Whether a section's recovered chunks cover the section whole.
//!
//! [`recover_section_cold_refs`](super::recover_section_cold_refs) lays a
//! stream's chunks out as a per-layer grid from the chunks it finds, and fills
//! an index it finds no chunk for with an empty placeholder. A stream short of
//! chunks therefore reads as a grid all the same: shorter, or with holes. Such
//! a stream is what a segment unlinked from under a section leaves — its
//! declaration and the chunks in other segments stand, the rest is gone — and
//! restoring it would install a section whose K/V stops short of its tokens.
//! A restore asks this first, and a gap sends the section to a fresh seal.

use crate::substrate::StoredSequence;

/// The first gap in `grid` against a section of `tokens` tokens, said in a
/// few words, or `None` when every layer holds every chunk and every token.
pub fn grid_gap(grid: &[StoredSequence], tokens: usize) -> Option<String> {
    for (layer, seq) in grid.iter().enumerate() {
        if let Some(chunk) = seq.chunks.iter().position(|c| c.record_len == 0) {
            return Some(format!("layer {layer} holds no chunk {chunk}"));
        }
        if seq.token_count != tokens {
            return Some(format!(
                "layer {layer} holds {} of {tokens} tokens",
                seq.token_count
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::StoredChunk;

    fn chunk(record_len: u64, token_count: u16) -> StoredChunk {
        StoredChunk {
            log_offset: if record_len == 0 { 0 } else { 4096 },
            record_len,
            token_count,
        }
    }

    fn layer(chunks: Vec<StoredChunk>) -> StoredSequence {
        let token_count = chunks.iter().map(|c| c.token_count as usize).sum();
        StoredSequence {
            chunks,
            token_count,
        }
    }

    /// **A whole grid has no gap.**
    #[test]
    fn a_whole_grid_has_no_gap() {
        let grid = vec![
            layer(vec![chunk(8192, 32), chunk(8192, 8)]),
            layer(vec![chunk(8192, 32), chunk(8192, 8)]),
        ];
        assert_eq!(grid_gap(&grid, 40), None);
    }

    /// **A chunk filled in as a placeholder is a gap**, whatever the token
    /// counts say.
    #[test]
    fn a_placeholder_chunk_is_a_gap() {
        let grid = vec![
            layer(vec![chunk(8192, 32), chunk(8192, 8)]),
            layer(vec![chunk(0, 32), chunk(8192, 8)]),
        ];
        assert_eq!(grid_gap(&grid, 40), Some("layer 1 holds no chunk 0".into()));
    }

    /// **A grid short of the section's tail on every layer is a gap** — the
    /// layers agree with one another and not with the section.
    #[test]
    fn a_grid_short_on_every_layer_is_a_gap() {
        let grid = vec![layer(vec![chunk(8192, 32)]), layer(vec![chunk(8192, 32)])];
        assert_eq!(
            grid_gap(&grid, 40),
            Some("layer 0 holds 32 of 40 tokens".into())
        );
    }
}
