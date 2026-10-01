//! What each piece of an assembled slot IS, chained through every piece before
//! it — the key a rebuild uses to keep the prefix it shares with the slot's
//! last assembly and re-inject only what follows.
//!
//! A slot is rebuilt whenever its projection's selection moves, and most moves
//! change a few pieces near the end: a working-set file dropping out of
//! provenance, the dialogue's newest turn. Truncating to zero and re-injecting
//! the whole prefix made every such rebuild cost the full prefix — measured at
//! ~1 s a rebuild with 230 turns ahead of the dialogue, every ~64 tokens. Two
//! assemblies whose chained identities agree through piece `i` built the same
//! slot through piece `i`, so a rebuild can keep exactly that much.
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

/// How many leading pieces a rebuild can keep: the longest run whose chained
/// identities match what the slot's last complete assembly placed. Stops at
/// the deferred user message, which is never part of a kept prefix.
pub(super) fn kept_prefix(placed: &[u64], identities: &[u64], pieces: &[AssembledPiece]) -> usize {
    placed
        .iter()
        .zip(identities)
        .zip(pieces)
        .take_while(|((was, now), piece)| {
            was == now && !matches!(piece, AssembledPiece::DeferredUser(_))
        })
        .count()
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
        assert_eq!(kept_prefix(&before, &after, &changed), 2);
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
        assert_eq!(kept_prefix(&before, &after, &changed), 3);
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

    /// Everything matching still stops at the deferred user message, which is
    /// prefilled fresh after every rebuild.
    #[test]
    fn the_kept_prefix_stops_at_the_deferred_user_message() {
        let same = ids(&base());
        assert_eq!(kept_prefix(&same, &same, &base()), 5);
        assert_eq!(kept_prefix(&same[..3], &same, &base()), 3);
        assert_eq!(kept_prefix(&[], &same, &base()), 0);
    }
}
