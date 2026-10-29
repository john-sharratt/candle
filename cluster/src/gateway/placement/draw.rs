//! Where a draw goes (§5.3): the strongest card not drawing, and in turn the
//! rest.

use super::{capacity_of, guest_reach, Weigh};
use crate::gateway::registry::MemberView;

/// Where a draw goes among `pool` (§5.3): **the strongest card not drawing,
/// and in turn the rest** — the fewest draws in flight on its card, then the
/// card that draws furthest (`guest_reach`, its size), then the most live
/// power, then the lower node. One draw at a time lands on the strongest card;
/// a burst goes round every card that reaches it, strongest first each round.
///
/// Reach before power: `power` is a decode rate, or a card's size before it
/// has decoded, and says little of how a card draws; its reach is the card.
pub(super) fn draw_of(pool: Vec<&MemberView>, weigh: &Weigh<'_>) -> Option<u64> {
    let drawing = |m: &MemberView| weigh.card(m).drawing;
    let seen: Vec<String> = pool
        .iter()
        .map(|m| {
            format!(
                "{:016x} drawing={} reach={} power={}",
                m.node,
                drawing(m),
                guest_reach(m),
                capacity_of(m).power
            )
        })
        .collect();
    let pick = pool
        .into_iter()
        .min_by(|a, b| {
            drawing(a)
                .cmp(&drawing(b))
                .then(guest_reach(b).cmp(&guest_reach(a)))
                .then(capacity_of(b).power.cmp(&capacity_of(a).power))
                .then(a.node.cmp(&b.node))
        })
        .map(|m| m.node);
    tracing::info!(
        target: "cluster::gateway::placement",
        picked = %pick.map_or_else(|| "none".to_string(), |n| format!("{n:016x}")),
        by = "draw",
        candidates = %seen.join("; "),
        "placed"
    );
    pick
}
