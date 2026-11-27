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

/// **A section's rate is its hits over its blocks and the queries, on the
/// band; a turn's round is no section's.**
#[test]
fn a_sections_rate_is_its_hits_over_its_blocks_and_queries() {
    let a = SectionKey(1);
    let b = SectionKey(2);
    let keys = [
        RoundKey::of_section(a),
        turn_round(9),
        RoundKey::of_section(b),
    ];
    let kb = HashMap::from([(a.scan_id(), a), (b.scan_id(), b)]);
    let rates = section_rates(&keys, &hits(&[4, 50, 0], &[10, 10, 8], 2), &kb);
    assert_eq!(rates.len(), 2);
    assert_eq!(rates[&a], 200.0);
    assert_eq!(rates[&b], 0.0);
}

/// **A pooled section's rate sums its pages, out of the shadow scan's
/// queries**: c, declared as two pages after the one loaded round, has 3 + 1
/// hits of 10 + 6 blocks over 2 shadow queries — 125 on the band; with no
/// shadow query counted it has no rate.
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
    assert_eq!(rates[&a], 50.0);
    assert_eq!(rates[&c], 125.0);
    let unscanned = section_rates(&keys, &read(0), &kb);
    assert_eq!(unscanned.get(&c), None);
    assert_eq!(unscanned[&a], 50.0);
}

/// **A section declared both placed and pooled is rated as placed, over the
/// selection's queries alone**: 2 of 10 blocks over 4 queries is 50, whatever
/// its pooled page counted over the shadow scan's.
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
    assert_eq!(section_rates(&keys, &read, &kb)[&a], 50.0);
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
