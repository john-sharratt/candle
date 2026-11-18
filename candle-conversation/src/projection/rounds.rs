//! The round grammar (`docs/hit_driven_projection.md` §8.1).
//!
//! An exchange of tool use has one shape:
//!
//! ```text
//! U request · (A call · U result)* · A answer
//! ```
//!
//! Its turns are records in pairs — turn `n` is the result of call `n − 1` and
//! the reply that makes call `n` — and the **rounds** the slot places are its
//! whole turns: round 0 is the first turn (the request and the first call),
//! the last round is the last turn (the last result and the answer), and the
//! rounds between are its tool turns. Every turn is placed whole, so every
//! carving reads `U A U A …`, opening on the request and closing on the answer.
//!
//! **Whole turns, never records.** A turn's index pages always start and end
//! with the turn, but the page edge between its two records is only where the
//! seal cut one: a turn sealed before the cut landed on the record edge
//! (2026-11-15) has none there, and neither of its records can be placed alone
//! with rows of its own. A carving of whole turns is placeable whatever the
//! turn's inner pages. Dropping a tool turn leaves a call whose own result is
//! not placed beside the next kept turn's result; the result names its own
//! content in its fence.
//!
//! A round's identity is its exchange — named by the exchange's first turn —
//! and its index in it, so the counter and momentum it carries follow it from
//! one projection to the next whatever else is placed around it.

use std::ops::Range;

use crate::projection::{TimelineId, TurnIndex};

/// One round of one exchange: its `index`-th turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RoundKey {
    pub timeline: TimelineId,
    /// The exchange's first turn.
    pub head: TurnIndex,
    pub index: u32,
}

/// The round `turn` is in the exchange `turns`, or `None` for a turn not in it.
pub fn round_of(turns: &[TurnIndex], turn: TurnIndex) -> Option<usize> {
    turns.iter().position(|&t| t == turn)
}

/// The turns of the exchange `turns` a carving places, in order: its first and
/// last — the exchange's floor — and the tool turns `keep_tool` keeps.
pub fn carve(turns: &[TurnIndex], keep_tool: impl Fn(usize) -> bool) -> Vec<TurnIndex> {
    let last = turns.len().saturating_sub(1);
    turns
        .iter()
        .enumerate()
        .filter(|&(round, _)| round == 0 || round == last || keep_tool(round))
        .map(|(_, &turn)| turn)
        .collect()
}

/// Each round's span in the slot, given where the exchange's placed turns sit:
/// `placed` names each turn and its range. A round whose turn is not placed
/// has no span.
pub fn round_spans(
    turns: &[TurnIndex],
    placed: &[(TurnIndex, Range<usize>)],
) -> Vec<(usize, Range<usize>)> {
    turns
        .iter()
        .enumerate()
        .filter_map(|(round, t)| {
            placed
                .iter()
                .find(|(pt, _)| pt == t)
                .map(|(_, range)| (round, range.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turns(n: u32) -> Vec<TurnIndex> {
        (0..n).map(|i| TurnIndex(10 + i)).collect()
    }

    /// A round is a turn: its position in the exchange.
    #[test]
    fn a_round_is_a_turn() {
        let t = turns(3);
        assert_eq!(round_of(&t, t[0]), Some(0));
        assert_eq!(round_of(&t, t[2]), Some(2));
        assert_eq!(round_of(&t, TurnIndex(99)), None);
    }

    /// The first and the last turn are always placed; the tool turns between
    /// are placed as `keep_tool` says.
    #[test]
    fn a_carving_keeps_the_floor_and_the_chosen_tool_turns() {
        let t = turns(4);
        assert_eq!(carve(&t, |_| true), t);
        assert_eq!(carve(&t, |r| r == 2), vec![t[0], t[2], t[3]]);
        assert_eq!(carve(&t, |_| false), vec![t[0], t[3]]);
        // A plain question and answer is one turn, and it is its own floor.
        let one = turns(1);
        assert_eq!(carve(&one, |_| false), one);
        assert!(carve(&[], |_| true).is_empty());
    }

    /// **Every carving opens on the request and closes on the answer**: over
    /// every exchange of up to six turns and every choice of tool turns, the
    /// first and last turns are placed, in order, each once.
    #[test]
    fn every_carving_keeps_the_floor_in_order() {
        for n in 1..=6u32 {
            let t = turns(n);
            let tools = (n as usize).saturating_sub(2);
            for mask in 0..(1u32 << tools) {
                let placed = carve(&t, |r| mask & (1 << (r - 1)) != 0);
                assert_eq!(placed.first(), t.first(), "n={n} mask={mask:b}");
                assert_eq!(placed.last(), t.last(), "n={n} mask={mask:b}");
                assert!(
                    placed.windows(2).all(|w| w[0] < w[1]),
                    "n={n} mask={mask:b} {placed:?}"
                );
                let kept_tools = (0..tools).filter(|b| mask & (1 << b) != 0).count();
                let floor = if n == 1 { 1 } else { 2 };
                assert_eq!(placed.len(), floor + kept_tools, "n={n} mask={mask:b}");
            }
        }
    }

    /// A round's span is its turn's; a round whose turn is not placed has none.
    #[test]
    fn a_round_spans_its_turn() {
        let t = turns(3);
        let placed = vec![(t[0], 0..10), (t[2], 10..18)];
        assert_eq!(round_spans(&t, &placed), vec![(0, 0..10), (2, 10..18)]);
        assert!(round_spans(&t, &[]).is_empty());
    }
}
