//! What each piece of an assembled slot IS — the key a rebuild aligns the
//! pieces the slot holds with the ones a projection wants by
//! (`super::piece_splice`), so it keeps every piece it already holds.
//!
//! A sealed piece is named by itself, since its K/V is the same wherever it
//! sits. A forwarded piece — glue, the live user message — is named by its
//! identity chained through every piece before it: two assemblies whose
//! chained identities agree through piece `i` built the same slot through
//! piece `i`, which is the only condition under which K/V the slot computed
//! there is still right.
//!
//! Every input the walk reads from a piece is in its identity:
//! - a section's id and its stored content;
//! - a turn's conversation, index, half and whether it goes in whole or with its
//!   reasoning windowed out (only the slot's own newest turn keeps it, so the
//!   same turn changes form when a newer one arrives);
//! - a glue run's tokens, its forward bridge, and — when it bridges — the piece
//!   it attends into, since its K/V then depends on that piece too.

use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

use super::piece_splice::PieceKind;
use super::projection_assembler::AssembledPiece;
use crate::projection::{SectionId, TimelineId, TurnIndex};

/// What the walk would inject for each piece, beyond the piece itself.
pub(super) struct PieceReads<'a> {
    /// Whether the turn goes in whole, reasoning included.
    pub whole: &'a dyn Fn(TimelineId, TurnIndex) -> bool,
    /// A section's stored content — see `projection_identity::section_content_stamp`.
    pub section_stamp: &'a dyn Fn(SectionId) -> (u64, usize),
    /// The forward bridge the glue run before `next` attends through.
    pub bridge: &'a dyn Fn(Option<&AssembledPiece>) -> u32,
}

/// Each piece's identity chained with every piece before it: entry `i` equals
/// another assembly's entry `i` exactly when both place the same slot through
/// piece `i`.
pub(super) fn chained_identities(pieces: &[AssembledPiece], reads: &PieceReads<'_>) -> Vec<u64> {
    let mut chain = DefaultHasher::new();
    let mut out = Vec::with_capacity(pieces.len());
    for (i, piece) in pieces.iter().enumerate() {
        write_piece(&mut chain, piece, reads);
        if let AssembledPiece::Glue(_) = piece {
            let next = pieces.get(i + 1);
            let fwd = (reads.bridge)(next);
            chain.write_u32(fwd);
            if fwd > 0 {
                if let Some(next) = next {
                    write_piece(&mut chain, next, reads);
                }
            }
        }
        out.push(chain.finish());
    }
    out
}

/// Each piece's identity for aligning one assembly with another
/// (`super::piece_splice`), and what its K/V is to the slot.
///
/// A sealed piece — section, turn, turn half — is borrowed K/V that may sit
/// anywhere, so it is named by what it is alone: the same piece wherever the
/// projection puts it. A forwarded piece — glue, the live user message — is
/// K/V the slot computed against everything before it, so it keeps its
/// `chained` identity and matches only where all that is unchanged.
pub(super) fn alignment_identities(
    pieces: &[AssembledPiece],
    chained: &[u64],
    reads: &PieceReads<'_>,
) -> Vec<(u64, PieceKind)> {
    pieces
        .iter()
        .zip(chained)
        .map(|(piece, &chain)| match piece {
            AssembledPiece::Glue(_) | AssembledPiece::DeferredUser(_) => {
                (chain, PieceKind::Forwarded)
            }
            AssembledPiece::Section(_)
            | AssembledPiece::Turn { .. }
            | AssembledPiece::TurnHalf { .. } => {
                let mut own = DefaultHasher::new();
                write_piece(&mut own, piece, reads);
                (own.finish(), PieceKind::Sealed)
            }
        })
        .collect()
}

/// A piece as a log names it: its kind and what identifies it.
pub(super) fn piece_label(piece: &AssembledPiece) -> String {
    let tl = |t: &Option<TimelineId>| t.map_or(0, |t| t.raw());
    match piece {
        AssembledPiece::Glue(tokens) => format!("glue[{}]", tokens.len()),
        AssembledPiece::Section(id) => format!("section {}", id.raw()),
        AssembledPiece::Turn {
            index, timeline, ..
        } => format!("turn {}:{}", tl(timeline), index.0),
        AssembledPiece::TurnHalf {
            index, timeline, ..
        } => format!("turn half {}:{}", tl(timeline), index.0),
        AssembledPiece::DeferredUser(tokens) => format!("user[{}]", tokens.len()),
    }
}

fn write_piece(h: &mut DefaultHasher, piece: &AssembledPiece, reads: &PieceReads<'_>) {
    match piece {
        AssembledPiece::Glue(tokens) => {
            h.write_u8(1);
            h.write_usize(tokens.len());
            for &t in tokens {
                h.write_u32(t);
            }
        }
        AssembledPiece::Section(id) => {
            let (content, len) = (reads.section_stamp)(*id);
            h.write_u8(2);
            h.write_u32(id.raw());
            h.write_u64(content);
            h.write_usize(len);
        }
        AssembledPiece::Turn {
            index,
            role,
            timeline,
            ..
        } => {
            h.write_u8(3);
            h.write_u64(timeline.map_or(0, |t| t.raw()));
            h.write_u32(index.0);
            h.write_u8(*role as u8);
            h.write_u8(timeline.is_some_and(|t| (reads.whole)(t, *index)) as u8);
        }
        AssembledPiece::TurnHalf {
            index, timeline, ..
        } => {
            h.write_u8(4);
            h.write_u64(timeline.map_or(0, |t| t.raw()));
            h.write_u32(index.0);
        }
        AssembledPiece::DeferredUser(tokens) => {
            h.write_u8(5);
            h.write_usize(tokens.len());
            for &t in tokens.iter() {
                h.write_u32(t);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::projection::GroupId;
    use crate::turn::Role;

    fn turn(timeline: u64, index: u32) -> AssembledPiece {
        AssembledPiece::Turn {
            group: GroupId::for_test(1),
            index: TurnIndex(index),
            role: Role::Assistant,
            timeline: TimelineId::from_raw(timeline),
        }
    }

    fn glue(tokens: &[u32]) -> AssembledPiece {
        AssembledPiece::Glue(tokens.to_vec())
    }

    fn section(id: u32) -> AssembledPiece {
        AssembledPiece::Section(SectionId::new(id))
    }

    fn user() -> AssembledPiece {
        AssembledPiece::DeferredUser(Arc::new(vec![9, 9]))
    }

    /// Glue bridges only into a turn; only turn 9:7 goes in whole.
    fn ids(pieces: &[AssembledPiece]) -> Vec<u64> {
        ids_with(pieces, &|tl, idx| tl.raw() == 9 && idx.0 == 7)
    }

    fn ids_with(
        pieces: &[AssembledPiece],
        whole: &dyn Fn(TimelineId, TurnIndex) -> bool,
    ) -> Vec<u64> {
        let stamp = |id: SectionId| (u64::from(id.raw()) * 31, 5usize);
        let bridge = |next: Option<&AssembledPiece>| match next {
            Some(AssembledPiece::Turn { .. }) => 16,
            _ => 0,
        };
        chained_identities(
            pieces,
            &PieceReads {
                whole,
                section_stamp: &stamp,
                bridge: &bridge,
            },
        )
    }

    fn base() -> Vec<AssembledPiece> {
        vec![
            section(1),
            turn(3, 0),
            turn(4, 0),
            glue(&[1, 2]),
            turn(9, 6),
            user(),
        ]
    }

    /// A change at piece `j` moves every identity from `j` on and none before
    /// it — which is what lets a rebuild keep exactly the shared prefix.
    #[test]
    fn a_change_moves_the_identities_from_there_on() {
        let before = ids(&base());
        let mut changed = base();
        changed[2] = turn(5, 0);
        let after = ids(&changed);
        assert_eq!(before[..2], after[..2]);
        for i in 2..before.len() {
            assert_ne!(before[i], after[i], "piece {i}");
        }
    }

    fn aligned(pieces: &[AssembledPiece]) -> Vec<(u64, PieceKind)> {
        let whole = |tl: TimelineId, idx: TurnIndex| tl.raw() == 9 && idx.0 == 7;
        let stamp = |id: SectionId| (u64::from(id.raw()) * 31, 5usize);
        let bridge = |next: Option<&AssembledPiece>| match next {
            Some(AssembledPiece::Turn { .. }) => 16,
            _ => 0,
        };
        let reads = PieceReads {
            whole: &whole,
            section_stamp: &stamp,
            bridge: &bridge,
        };
        alignment_identities(pieces, &chained_identities(pieces, &reads), &reads)
    }

    /// **A sealed piece is the same piece wherever it stands**, so a change in
    /// front of it does not make it new — what lets a rebuild keep it. Glue and
    /// the live message keep their chained identities: computed against what
    /// precedes them, they are new whenever that is.
    #[test]
    fn a_sealed_piece_aligns_by_itself_and_a_forwarded_one_by_its_chain() {
        let before = aligned(&base());
        let mut changed = base();
        changed[0] = section(2);
        let after = aligned(&changed);
        for i in [1, 2, 4] {
            assert_eq!(before[i], after[i], "sealed piece {i} kept its identity");
            assert_eq!(before[i].1, PieceKind::Sealed);
        }
        for i in [3, 5] {
            assert_ne!(before[i].0, after[i].0, "forwarded piece {i} is new");
            assert_eq!(before[i].1, PieceKind::Forwarded);
        }
        assert_ne!(before[0], after[0]);
    }

    /// A glue run that bridges into a turn attends that turn, so replacing the
    /// turn changes the glue too — the glue cannot be kept over a new target.
    #[test]
    fn a_bridging_glue_run_moves_with_the_turn_it_leads_into() {
        let before = ids(&base());
        let mut changed = base();
        changed[4] = turn(9, 5);
        let after = ids(&changed);
        assert_eq!(before[..3], after[..3]);
        assert_ne!(
            before[3], after[3],
            "the glue bridged into the replaced turn"
        );
    }

    /// The same turn is a different piece once it stops being the slot's newest
    /// and goes in with its reasoning windowed out.
    #[test]
    fn a_turn_changing_form_is_a_different_piece() {
        let pieces = [section(1), turn(9, 7)];
        let whole = ids_with(&pieces, &|_, _| true);
        let windowed = ids_with(&pieces, &|_, _| false);
        assert_eq!(whole[0], windowed[0]);
        assert_ne!(whole[1], windowed[1]);
    }

    #[test]
    fn a_label_names_the_kind_and_identity() {
        let labels: Vec<String> = base().iter().map(piece_label).collect();
        assert_eq!(
            labels,
            [
                "section 1",
                "turn 3:0",
                "turn 4:0",
                "glue[2]",
                "turn 9:6",
                "user[2]"
            ]
        );
    }
}
