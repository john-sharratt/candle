//! The knowledge-base sections a slot keeps resident beside its range, and
//! how the hit declaration names them (`docs/placed_section_layout.md` §3.4,
//! §3.5).
//!
//! A resident section out of scope is one pooled round per index page, each
//! under the section's round ([`RoundKey::of_section`]), so the shadow scan
//! rates it at every step and the step's read sums its pages back into one
//! rate (`section_rates`).

use std::collections::HashSet;

use candle_transformers::models::hit_layers::PooledRound;
use candle_transformers::models::piece_key::PieceKey;

use crate::files::SectionKey;
use crate::projection::rounds::RoundKey;
use crate::projection::SectionId;

/// One resident section: its key, its member, its tokens and its index pages
/// — each page's token width and key — in token order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in super::super) struct KbResident {
    pub(in super::super) key: SectionKey,
    pub(in super::super) id: SectionId,
    pub(in super::super) tokens: usize,
    pub(in super::super) pages: Vec<(usize, PieceKey)>,
}

/// Every resident section `placed` does not hold, as pooled rounds for the
/// declaration and the keys they count under, in residency order.
pub(in super::super) fn kb_pooled_rounds(
    resident: &[KbResident],
    placed: &HashSet<SectionKey>,
) -> (Vec<RoundKey>, Vec<PooledRound>) {
    let mut keys = Vec::new();
    let mut rounds = Vec::new();
    for r in resident.iter().filter(|r| !placed.contains(&r.key)) {
        for &(width, piece) in &r.pages {
            keys.push(RoundKey::of_section(r.key));
            rounds.push(PooledRound {
                piece,
                positions: 0..width,
            });
        }
    }
    (keys, rounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A resident section out of scope is one pooled round per page, each
    /// under its section's round; a placed one is not pooled.**
    #[test]
    fn a_resident_section_is_one_pooled_round_per_page() {
        let (p, q, r) = (
            PieceKey::of(b"a page 0"),
            PieceKey::of(b"a page 1"),
            PieceKey::of(b"b page 0"),
        );
        let resident = [
            KbResident {
                key: SectionKey(1),
                id: SectionId::new(11),
                tokens: 70,
                pages: vec![(40, p), (30, q)],
            },
            KbResident {
                key: SectionKey(2),
                id: SectionId::new(12),
                tokens: 20,
                pages: vec![(20, r)],
            },
        ];
        let (keys, rounds) = kb_pooled_rounds(&resident, &HashSet::from([SectionKey(2)]));
        let a = RoundKey::of_section(SectionKey(1));
        assert_eq!(keys, vec![a, a]);
        assert_eq!(
            rounds,
            vec![
                PooledRound {
                    piece: p,
                    positions: 0..40
                },
                PooledRound {
                    piece: q,
                    positions: 0..30
                },
            ]
        );
    }
}
