//! Where a unit of work that draws nothing goes (`docs/static_placement.md`
//! §3.2): the deterministic fill to each machine's `min` in rank order, then a
//! round robin up to each one's `max`, then nobody — the unit waits.

use super::{has_room, Placed, Weigh};
use crate::gateway::registry::MemberView;

/// The member of `pool` a unit goes to.
///
/// 1. **The fill.** Among the members whose card has room ([`has_room`]), in
///    `rank` order, the first whose card holds fewer units than its machine's
///    `min`.
/// 2. **The round robin.** Every one at its `min`: the next after the
///    cursor, in `(rank, node)` order, wrapping — and the cursor moves to it.
/// 3. **None with room**: [`Placed::Full`].
pub(super) fn conversation_of(pool: Vec<&MemberView>, weigh: &Weigh<'_>) -> Placed {
    let mut open: Vec<(u32, u32, u32, &MemberView)> = pool
        .iter()
        .copied()
        .filter(|m| has_room(m, weigh, false))
        .filter_map(|m| {
            let c = weigh.machines.conversations(&m.host)?;
            Some((c.rank, c.min, weigh.card(m).work(), m))
        })
        .collect();
    open.sort_by_key(|(rank, _, _, m)| (*rank, m.node));
    let (by, pick) = match open.iter().find(|(_, min, units, _)| units < min) {
        Some((_, _, _, m)) => ("fill", Some(m.node)),
        None => {
            let after = weigh.cursor.get();
            let next = open
                .iter()
                .find(|(rank, _, _, m)| after.is_none_or(|a| (*rank, m.node) > a))
                .or_else(|| open.first());
            if let Some((rank, _, _, m)) = next {
                weigh.cursor.set(Some((*rank, m.node)));
            }
            ("round", next.map(|(_, _, _, m)| m.node))
        }
    };
    tracing::info!(
        target: "cluster::gateway::placement",
        picked = %pick.map_or_else(|| "none".to_string(), |n| format!("{n:016x}")),
        by = if pick.is_some() { by } else { "full" },
        candidates = %candidates(&pool, weigh),
        "placed"
    );
    pick.map_or(Placed::Full, Placed::To)
}

/// Every candidate as the pick saw it, for the placement line
/// (`docs/static_placement.md` §3.2): its machine, what its card holds, and
/// whether it had room.
fn candidates(pool: &[&MemberView], weigh: &Weigh<'_>) -> String {
    pool.iter()
        .map(|m| {
            let card = weigh.card(m);
            let c = weigh.machines.conversations(&m.host);
            format!(
                "{:016x} host={} rank={} min={} max={} units={} drawing={} reserved={} room={}",
                m.node,
                m.host,
                c.map_or_else(|| "-".to_string(), |c| c.rank.to_string()),
                c.map_or(0, |c| c.min),
                c.map_or(0, |c| c.max()),
                card.work(),
                card.drawing,
                weigh.reserved(m),
                has_room(m, weigh, false),
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}
