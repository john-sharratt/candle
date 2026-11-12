//! The round grammar (`docs/hit_driven_projection.md` §8.1).
//!
//! An exchange of tool use has one shape:
//!
//! ```text
//! U request · (A call · U result)* · A answer
//! ```
//!
//! Its turns are records in pairs — turn `n` is the result of call `n − 1` and
//! the reply that makes call `n` — so the **rounds** the slot places cut across
//! turns: the request is the first turn's user record, tool round `k` is turn
//! `k − 1`'s assistant record with turn `k`'s user record, and the answer is the
//! last turn's assistant record. Dropping any tool round leaves a request,
//! rounds that each answer the turn before them, and a closing answer: every
//! carving reads as a conversation that could have happened.
//!
//! A round's identity is its exchange — named by the exchange's first turn —
//! and its index in it, so the counter and momentum it carries follow it from
//! one projection to the next whatever else is placed around it.

use std::ops::Range;

use crate::projection::{TimelineId, TurnIndex};
use crate::turn_layout::TurnRecord;

/// One round of one exchange: `index` 0 is the request, `1..turns` the tool
/// rounds, `turns` the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RoundKey {
    pub timeline: TimelineId,
    /// The exchange's first turn.
    pub head: TurnIndex,
    pub index: u32,
}

/// What an exchange's carving places, in order: a whole turn where both its
/// records are kept, one record where only one is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Turn(TurnIndex),
    Record(TurnIndex, TurnRecord),
}

/// The rounds an exchange of `turns` turns has: its request, its tool rounds
/// and its answer.
pub fn round_count(turns: usize) -> usize {
    match turns {
        0 => 0,
        n => n + 1,
    }
}

/// The records round `index` of the exchange `turns` holds, in order.
pub fn records_of(turns: &[TurnIndex], index: usize) -> Vec<(TurnIndex, TurnRecord)> {
    let n = turns.len();
    match index {
        _ if n == 0 || index > n => Vec::new(),
        0 => vec![(turns[0], TurnRecord::User)],
        i if i == n => vec![(turns[n - 1], TurnRecord::Assistant)],
        k => vec![
            (turns[k - 1], TurnRecord::Assistant),
            (turns[k], TurnRecord::User),
        ],
    }
}

/// The round a turn's `record` belongs to in the exchange `turns`, or `None`
/// for a turn not in it.
pub fn round_of(turns: &[TurnIndex], turn: TurnIndex, record: TurnRecord) -> Option<usize> {
    let pos = turns.iter().position(|&t| t == turn)?;
    Some(match record {
        TurnRecord::User => pos,
        TurnRecord::Assistant => pos + 1,
    })
}

/// The placements of the exchange `turns` with the tool rounds `keep_tool`
/// keeps; the request and the answer are always kept, being the exchange's
/// floor.
pub fn carve(turns: &[TurnIndex], keep_tool: impl Fn(usize) -> bool) -> Vec<Placement> {
    let n = turns.len();
    let kept = |round: usize| round == 0 || round == n || keep_tool(round);
    let mut out = Vec::with_capacity(n);
    for (pos, &turn) in turns.iter().enumerate() {
        // The turn's user record is round `pos`, its assistant record round
        // `pos + 1`.
        match (kept(pos), kept(pos + 1)) {
            (true, true) => out.push(Placement::Turn(turn)),
            (true, false) => out.push(Placement::Record(turn, TurnRecord::User)),
            (false, true) => out.push(Placement::Record(turn, TurnRecord::Assistant)),
            (false, false) => {}
        }
    }
    out
}

/// Each round's span in the slot, given where the exchange's placed records
/// sit: `placed` names each record and its range, in slot order. A round
/// whose records are not all placed, or not contiguous, has no span.
pub fn round_spans(
    turns: &[TurnIndex],
    placed: &[(TurnIndex, TurnRecord, Range<usize>)],
) -> Vec<(usize, Range<usize>)> {
    let mut out = Vec::new();
    for round in 0..round_count(turns.len()) {
        let records = records_of(turns, round);
        let ranges: Option<Vec<&Range<usize>>> = records
            .iter()
            .map(|(t, r)| {
                placed
                    .iter()
                    .find(|(pt, pr, _)| pt == t && pr == r)
                    .map(|(_, _, range)| range)
            })
            .collect();
        let Some(ranges) = ranges else {
            continue;
        };
        let contiguous = ranges.windows(2).all(|w| w[0].end == w[1].start);
        if let (true, Some(first), Some(last)) = (contiguous, ranges.first(), ranges.last()) {
            out.push((round, first.start..last.end));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turns(n: u32) -> Vec<TurnIndex> {
        (0..n).map(|i| TurnIndex(10 + i)).collect()
    }

    /// The records of every placement, in order, as roles.
    fn roles(placed: &[Placement]) -> Vec<TurnRecord> {
        placed
            .iter()
            .flat_map(|p| match *p {
                Placement::Turn(_) => vec![TurnRecord::User, TurnRecord::Assistant],
                Placement::Record(_, r) => vec![r],
            })
            .collect()
    }

    /// The request is the first turn's user record, each tool round a call and
    /// the result after it, the answer the last turn's assistant record.
    #[test]
    fn rounds_cut_across_turns() {
        let t = turns(3);
        assert_eq!(round_count(3), 4);
        assert_eq!(records_of(&t, 0), vec![(t[0], TurnRecord::User)]);
        assert_eq!(
            records_of(&t, 1),
            vec![(t[0], TurnRecord::Assistant), (t[1], TurnRecord::User)]
        );
        assert_eq!(
            records_of(&t, 2),
            vec![(t[1], TurnRecord::Assistant), (t[2], TurnRecord::User)]
        );
        assert_eq!(records_of(&t, 3), vec![(t[2], TurnRecord::Assistant)]);
        assert!(records_of(&t, 4).is_empty());
        assert_eq!(round_of(&t, t[1], TurnRecord::User), Some(1));
        assert_eq!(round_of(&t, t[1], TurnRecord::Assistant), Some(2));
        assert_eq!(round_of(&t, TurnIndex(99), TurnRecord::User), None);
        // A plain question and answer is two rounds of one turn.
        assert_eq!(round_count(1), 2);
        assert_eq!(round_count(0), 0);
    }

    /// Keeping every round places every turn whole; dropping a tool round
    /// splits the two turns it crosses into the records the neighbours keep.
    #[test]
    fn a_carving_places_whole_turns_where_it_can() {
        let t = turns(3);
        assert_eq!(
            carve(&t, |_| true),
            vec![
                Placement::Turn(t[0]),
                Placement::Turn(t[1]),
                Placement::Turn(t[2])
            ]
        );
        assert_eq!(
            carve(&t, |r| r != 1),
            vec![
                Placement::Record(t[0], TurnRecord::User),
                Placement::Record(t[1], TurnRecord::Assistant),
                Placement::Turn(t[2]),
            ]
        );
        assert_eq!(
            carve(&t, |_| false),
            vec![
                Placement::Record(t[0], TurnRecord::User),
                Placement::Record(t[2], TurnRecord::Assistant),
            ]
        );
    }

    /// **Every carving reads `U (A U)* A`**: over every exchange of up to six
    /// turns and every choice of tool rounds, the placed records alternate
    /// roles, start with the request and end with the answer.
    #[test]
    fn every_carving_keeps_the_grammar() {
        for n in 1..=6u32 {
            let t = turns(n);
            let tools = n as usize - 1;
            for mask in 0..(1u32 << tools) {
                let placed = carve(&t, |r| mask & (1 << (r - 1)) != 0);
                let seq = roles(&placed);
                assert_eq!(seq.first(), Some(&TurnRecord::User), "n={n} mask={mask:b}");
                assert_eq!(
                    seq.last(),
                    Some(&TurnRecord::Assistant),
                    "n={n} mask={mask:b}"
                );
                assert!(
                    seq.windows(2).all(|w| w[0] != w[1]),
                    "two of one role side by side: n={n} mask={mask:b} {seq:?}"
                );
                let kept_tools = (0..tools).filter(|b| mask & (1 << b) != 0).count();
                assert_eq!(seq.len(), 2 + 2 * kept_tools, "n={n} mask={mask:b}");
            }
        }
    }

    /// A round's span runs from its first record's start to its last record's
    /// end; a round missing a record, or split by another, has none.
    #[test]
    fn a_round_spans_its_records() {
        let t = turns(2);
        // Turn 0 whole at [0, 10) with its edge at 4; turn 1 whole at [10, 18)
        // with its edge at 13.
        let placed = vec![
            (t[0], TurnRecord::User, 0..4),
            (t[0], TurnRecord::Assistant, 4..10),
            (t[1], TurnRecord::User, 10..13),
            (t[1], TurnRecord::Assistant, 13..18),
        ];
        assert_eq!(
            round_spans(&t, &placed),
            vec![(0, 0..4), (1, 4..13), (2, 13..18)]
        );
        // The tool round dropped: request and answer only.
        let carved = vec![
            (t[0], TurnRecord::User, 0..4),
            (t[1], TurnRecord::Assistant, 4..9),
        ];
        assert_eq!(round_spans(&t, &carved), vec![(0, 0..4), (2, 4..9)]);
        // A round whose records are not side by side has no span.
        let split = vec![
            (t[0], TurnRecord::Assistant, 0..6),
            (t[1], TurnRecord::User, 9..12),
        ];
        assert!(round_spans(&t, &split).is_empty());
    }
}
