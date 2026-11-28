//! Holes in a projection (`ProjectionSegment::Hole`): a piece of their own,
//! aligned as a forwarded piece, and counted in every position the index is
//! named by — the prompt's end, the pinned ranges and a record's ends — but in
//! no token count.

use std::sync::Arc;

use super::tests_fixtures::tests::{generated_seg, section_seg, turn_seg};
use super::*;
use crate::projection::{Pin, ResolvedSection, SectionId};
use crate::scheduler::piece_identity::{alignment_identities, chained_identities, PieceReads};
use crate::scheduler::piece_splice::PieceKind;

fn hole(positions: usize) -> ProjectionSegment {
    ProjectionSegment::Hole { positions }
}

/// Records of pieces with `(tokens, holes)` each, laid end to end.
fn records(widths: &[(u32, u32)]) -> Vec<PlacedPiece> {
    let mut out: Vec<PlacedPiece> = widths
        .iter()
        .map(|&(tokens, holes)| PlacedPiece {
            tokens,
            holes,
            blocks: 0,
            ..PlacedPiece::ending_at(0)
        })
        .collect();
    restack(&mut out);
    out
}

/// **A hole is its own piece, between the glue runs it parts.**
#[test]
fn a_hole_is_its_own_piece() {
    let pieces = assemble_pieces(&[
        generated_seg("a", 0, &[1, 2]),
        hole(100),
        section_seg(7),
        hole(0),
        generated_seg("b", 0, &[3]),
    ]);
    assert_eq!(
        pieces,
        vec![
            AssembledPiece::Glue(vec![1, 2]),
            AssembledPiece::Hole(100),
            AssembledPiece::Section(SectionId::new(7)),
            AssembledPiece::Glue(vec![3]),
        ],
        "an empty hole places nothing"
    );
}

/// **A hole aligns by its size alone, as a sealed piece by itself**: the piece
/// before it does not change it, another size does — so a rebuild keeps,
/// moves and resizes holes where they stand.
#[test]
fn a_hole_is_aligned_by_its_size_alone() {
    let reads = PieceReads {
        whole: &|_, _| false,
        section_stamp: &|id| (id.raw() as u64, 1),
        bridge: &|_| 0,
    };
    let ids = |pieces: &[AssembledPiece]| {
        let chained = chained_identities(pieces, &reads);
        alignment_identities(pieces, &chained, &reads)
    };
    let a = ids(&[
        AssembledPiece::Section(SectionId::new(7)),
        AssembledPiece::Hole(100),
    ]);
    let b = ids(&[
        AssembledPiece::Section(SectionId::new(8)),
        AssembledPiece::Hole(100),
    ]);
    let c = ids(&[
        AssembledPiece::Section(SectionId::new(7)),
        AssembledPiece::Hole(99),
    ]);
    assert_eq!(a[1].1, PieceKind::Sealed);
    assert_eq!(a[1], b[1], "a different piece before it is the same hole");
    assert_ne!(a[1].0, c[1].0, "a different size");
    assert_eq!(
        a[1],
        ids(&[
            AssembledPiece::Section(SectionId::new(7)),
            AssembledPiece::Hole(100)
        ])[1]
    );
}

/// **Records end at tokens and at positions apart.**
#[test]
fn records_end_at_their_tokens_and_their_positions() {
    let r = records(&[(10, 0), (0, 100), (5, 0)]);
    let ends: Vec<(u32, u32)> = r.iter().map(|p| (p.pos_after, p.at_after)).collect();
    assert_eq!(ends, [(10, 10), (10, 110), (15, 115)]);
}

/// **The prompt ends after its holes.**
#[test]
fn the_prompt_ends_past_its_holes() {
    let pieces = assemble_pieces(&[
        generated_seg("open", 0, &[1, 2]),
        hole(100),
        section_seg(7),
        turn_seg(1, 2, 3),
    ]);
    let r = records(&[(2, 0), (0, 100), (30, 0), (40, 0)]);
    assert_eq!(prompt_end_of(&pieces, &r), 132);
}

/// **A pinned section after a hole is pinned where it is read.**
#[test]
fn a_pinned_section_after_a_hole_is_pinned_at_its_position() {
    let pinned = |id: u32| {
        ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection {
            id: SectionId::new(id),
            pin: Pin::Whole,
        }))
    };
    let segments = vec![
        pinned(7),
        hole(50),
        pinned(8),
        ProjectionSegment::NewUserMessage {
            tokens: Arc::new(vec![9]),
        },
    ];
    let pieces = assemble_pieces(&segments);
    // s7 0..20, hole 20..70, s8 70..80.
    let r = records(&[(20, 0), (0, 50), (10, 0)]);
    assert_eq!(pin_ranges_of(&segments, &pieces, &r), vec![0..20, 70..80]);
}
