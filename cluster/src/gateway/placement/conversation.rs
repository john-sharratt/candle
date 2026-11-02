//! Where a unit of work that draws nothing goes (`docs/static_placement.md`
//! §3.2): the deterministic fill to each machine's `min` in rank order, then a
//! round robin up to each one's `max`, then nobody — the unit waits.

use super::{has_room, prefer_current, Placed, Weigh};
use crate::gateway::registry::MemberView;

/// The member of `pool` a unit goes to.
///
/// 1. **The fill.** Among the members whose card has room ([`has_room`]) —
///    those not lagging on replication, while any of them has room — in
///    `(rank, node)` order, the first whose card holds fewer units than its
///    machine's `min`.
/// 2. **The round robin.** Every one at its `min`: the next after the
///    cursor, in `(rank, node)` order, wrapping — and the cursor moves to it.
/// 3. **None with room**: [`Placed::Full`], the unit waits. **No machine of the
///    pool takes conversations at all** — named nowhere in the table, or with no
///    `conversations` — is [`Placed::Nobody`]: waiting would never end.
pub(super) fn conversation_of(pool: Vec<&MemberView>, weigh: &Weigh<'_>) -> Placed {
    if pool
        .iter()
        .all(|m| weigh.machines.conversations(&m.host).is_none())
    {
        log(&pool, weigh, None, "nobody");
        return Placed::Nobody;
    }
    let room: Vec<&MemberView> = pool
        .iter()
        .copied()
        .filter(|m| has_room(m, weigh, false))
        .collect();
    let mut open: Vec<(u32, u32, u32, &MemberView)> = prefer_current(room)
        .into_iter()
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
    log(&pool, weigh, pick, if pick.is_some() { by } else { "full" });
    pick.map_or(Placed::Full, Placed::To)
}

/// The placement line (`docs/static_placement.md` §3.2): how the unit was
/// placed, and every candidate as the pick saw it — its machine, what its card
/// holds, and whether it had room.
fn log(pool: &[&MemberView], weigh: &Weigh<'_>, pick: Option<u64>, by: &str) {
    let candidates = pool
        .iter()
        .map(|m| {
            let card = weigh.card(m);
            let c = weigh.machines.conversations(&m.host);
            format!(
                "{:016x} host={} rank={} min={} max={} units={} drawing={} reserved={} lagging={} room={}",
                m.node,
                m.host,
                c.map_or_else(|| "-".to_string(), |c| c.rank.to_string()),
                c.map_or(0, |c| c.min),
                c.map_or(0, |c| c.max()),
                card.work(),
                card.drawing,
                weigh.reserved(m),
                m.lagging,
                has_room(m, weigh, false),
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    tracing::info!(
        target: "cluster::gateway::placement",
        picked = %pick.map_or_else(|| "none".to_string(), |n| format!("{n:016x}")),
        by,
        candidates = %candidates,
        "placed"
    );
}
