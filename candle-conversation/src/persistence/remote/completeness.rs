//! When another member's turn is complete enough to adopt
//! (`docs/replicated_substrate.md` §6.5).
//!
//! A turn's records arrive in any order and across any number of batches, so
//! a turn is adopted only once everything its declaration calls for is held:
//! its `Tokens`, and every `Chunk` of its block range on every layer —
//! `n_layers × (block_end − block_start)`, the count a reload checks. Until
//! then it is pending: left out of every projection, never judged corrupt.
//!
//! A turn whose declaration claims no K/V (a summary node, a placeholder)
//! holds nothing more than its declaration, and a distilled timeline's turns
//! were shed on purpose, so either is complete as declared.

use crate::persistence::streams::TurnDecl;
use crate::substrate::StreamRuntime;

/// Whether `stream`, which `decl` declares, holds everything the turn calls
/// for — see the module docs.
pub fn turn_is_complete(
    decl: &TurnDecl,
    stream: &StreamRuntime,
    n_layers: usize,
    distilled: bool,
) -> bool {
    let blocks = decl.block_end.saturating_sub(decl.block_start) as usize;
    if distilled || blocks == 0 {
        return true;
    }
    stream.tokens.is_some() && stream.chunks.len() == n_layers * blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::manifest::{ChunkLoc, RecordLoc};
    use crate::persistence::segment::SegmentId;

    fn decl(block_start: u64, block_end: u64) -> TurnDecl {
        TurnDecl {
            timeline_id: 7,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 1,
            block_start,
            block_end,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        }
    }

    fn chunk() -> ChunkLoc {
        ChunkLoc {
            segment: SegmentId(1),
            offset: 0,
            payload_len: 1,
            record_size: 4096,
            token_count: 32,
            format: 4,
        }
    }

    fn tokens() -> RecordLoc {
        RecordLoc {
            segment: SegmentId(1),
            offset: 4096,
            payload_len: 8,
            record_size: 4096,
        }
    }

    /// Two layers of a two-block turn: complete with its tokens and all four
    /// chunks, and not one piece short of it.
    #[test]
    fn a_turn_is_complete_with_every_piece() {
        let d = decl(3, 5);
        let mut stream = StreamRuntime::default();
        for idx in 0..3 {
            stream.chunks.insert(idx, chunk());
        }
        stream.tokens = Some(tokens());
        assert!(
            !turn_is_complete(&d, &stream, 2, false),
            "three of four chunks"
        );
        stream.chunks.insert(3, chunk());
        assert!(turn_is_complete(&d, &stream, 2, false));
        stream.tokens = None;
        assert!(!turn_is_complete(&d, &stream, 2, false), "no tokens");
    }

    /// A turn that claims no K/V, and any turn of a distilled timeline, is
    /// complete as declared.
    #[test]
    fn a_turn_claiming_nothing_is_complete_as_declared() {
        let empty = StreamRuntime::default();
        assert!(turn_is_complete(&decl(4, 4), &empty, 2, false));
        assert!(turn_is_complete(&decl(0, 2), &empty, 2, true));
        assert!(!turn_is_complete(&decl(0, 2), &empty, 2, false));
    }
}
