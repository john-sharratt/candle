//! The knowledge base's sections as hit rounds, and a step's read of them
//! (`docs/placed_section_layout.md` §3.4).

use super::*;
use crate::files::SectionKey;

fn hits(hits: &[u32], blocks: &[u32], queries: u64) -> PieceHits {
    PieceHits {
        hits: hits.to_vec(),
        blocks: blocks.to_vec(),
        loaded: hits.len(),
        queries,
        shadow_queries: 0,
    }
}

fn turn_round(raw: u64) -> RoundKey {
    RoundKey {
        timeline: TimelineId::for_test(raw),
        head: TurnIndex(0),
        index: 0,
    }
}

/// **A placed section is a round of its own**: the section piece the map
/// names is declared over the positions it took, a section piece it does not
/// name is not.
#[test]
fn a_placed_section_is_a_round_of_its_own() {
    let pieces = [
        AssembledPiece::Section(SectionId::new(7)),
        AssembledPiece::Section(SectionId::new(8)),
    ];
    let placed: Vec<PlacedPiece> = [(10u32, 10u32), (6, 16)]
        .iter()
        .map(|&(tokens, at_after)| PlacedPiece {
            tokens,
            pos_after: at_after,
            at_after,
            ..PlacedPiece::ending_at(0)
        })
        .collect();
    let section = SectionKey(0x51);
    let sections = HashMap::from([(SectionId::new(8), section)]);
    let (keys, ranges) = hit_rounds_of(&pieces, &placed, &Substrate::default(), &sections);
    assert_eq!(keys, vec![RoundKey::of_section(section)]);
    assert_eq!(ranges, vec![10..16]);
}

/// **A section's rate is its share of blocks selected over the selection's
/// share of every placed block — a lift over chance; a turn's round is no
/// section's**: 2 of a's 10 blocks against 4 of the slot's 40 is a lift of 2,
/// b's none is 0; the same reads at ten times the queries lift the same. No
/// block selected anywhere, every lift is 0.
#[test]
fn a_sections_rate_is_its_lift_over_the_selection() {
    let a = SectionKey(1);
    let b = SectionKey(2);
    let keys = [
        RoundKey::of_section(a),
        turn_round(9),
        RoundKey::of_section(b),
    ];
    let kb = HashMap::from([(a.scan_id(), a), (b.scan_id(), b)]);
    let rates = section_rates(&keys, &hits(&[2, 2, 0], &[10, 10, 20], 1), &kb);
    assert_eq!(rates.len(), 2);
    assert_eq!(rates[&a], 2.0);
    assert_eq!(rates[&b], 0.0);
    let more = section_rates(&keys, &hits(&[20, 20, 0], &[10, 10, 20], 10), &kb);
    assert_eq!(more[&a], 2.0, "the lift does not move with the queries");
    let none = section_rates(&keys, &hits(&[0, 0, 0], &[10, 10, 20], 1), &kb);
    assert_eq!((none[&a], none[&b]), (0.0, 0.0));
}

/// **The selection's rate is every placed round's hits over all their blocks
/// and the queries**: 4 + 6 + 0 hits of 10 + 10 + 8 blocks over 2 queries is
/// 10 / 56 of the band, 178.57; a pooled round (9 hits of 8) is not counted,
/// 10 / 40, 250; no query, no rate.
#[test]
fn the_selection_rate_is_every_placed_rounds_hits_over_their_blocks() {
    assert_eq!(
        selection_rate(&hits(&[4, 6, 0], &[10, 10, 8], 2)),
        Some(178.57143)
    );
    let mut pooled = hits(&[4, 6, 9], &[10, 10, 8], 2);
    pooled.loaded = 2;
    assert_eq!(selection_rate(&pooled), Some(250.0));
    assert_eq!(selection_rate(&hits(&[4, 6], &[10, 10], 0)), None);
}

/// **A pooled section's rate sums its pages, out of the shadow scan's
/// queries, over the selection's**: the one loaded round, a, has 2 of 10
/// blocks over 4 queries — 50 on the band, the selection's own, a lift of 1;
/// c, declared as two pages after it, has 3 + 1 hits of 10 + 6 blocks over 2
/// shadow queries — 125, a lift of 2.5. With no shadow query counted c has no
/// rate.
#[test]
fn a_pooled_sections_rate_sums_its_pages_over_the_shadow_queries() {
    let (a, c) = (SectionKey(1), SectionKey(3));
    let keys = [
        RoundKey::of_section(a),
        RoundKey::of_section(c),
        RoundKey::of_section(c),
    ];
    let kb = HashMap::from([(a.scan_id(), a), (c.scan_id(), c)]);
    let read = |shadow_queries| PieceHits {
        hits: vec![2, 3, 1],
        blocks: vec![10, 10, 6],
        loaded: 1,
        queries: 4,
        shadow_queries,
    };
    let rates = section_rates(&keys, &read(2), &kb);
    assert_eq!(rates[&a], 1.0);
    assert_eq!(rates[&c], 2.5);
    let unscanned = section_rates(&keys, &read(0), &kb);
    assert_eq!(unscanned.get(&c), None);
    assert_eq!(unscanned[&a], 1.0);
}

/// **A section declared both placed and pooled is rated as placed, over the
/// selection's queries alone**: 2 of 10 blocks over 4 queries, the slot's only
/// placed round, is a lift of 1, whatever its pooled page counted over the
/// shadow scan's.
#[test]
fn a_section_declared_both_ways_is_rated_as_placed() {
    let a = SectionKey(1);
    let keys = [RoundKey::of_section(a), RoundKey::of_section(a)];
    let kb = HashMap::from([(a.scan_id(), a)]);
    let read = PieceHits {
        hits: vec![2, 9],
        blocks: vec![10, 10],
        loaded: 1,
        queries: 4,
        shadow_queries: 1,
    };
    assert_eq!(section_rates(&keys, &read, &kb)[&a], 1.0);
}

/// **A read every step keeps one part a declaration**: a read of the same
/// rounds is summed into the last part; a read of others is a part of its
/// own.
#[test]
fn a_read_every_step_keeps_one_part_a_declaration() {
    let keys = vec![RoundKey::of_section(SectionKey(1))];
    let mut parts: Vec<HitPart> = Vec::new();
    fold_part(&mut parts, &keys, hits(&[2], &[10], 1));
    fold_part(&mut parts, &keys, hits(&[3], &[10], 2));
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].1.hits, vec![5]);
    assert_eq!(parts[0].1.queries, 3);
    let other = vec![turn_round(4)];
    fold_part(&mut parts, &other, hits(&[1], &[4], 1));
    assert_eq!(parts.len(), 2);
}
