//! Where a draw goes (`docs/static_placement.md` §4): a card that is not
//! drawing, one with no unit of work in flight before one with, then the
//! machine's picture preference.

use super::{guest_reach, has_room, Placed, Weigh};
use crate::gateway::registry::{CardKey, MemberView};

/// The member of `pool` — the cards that reach the draw — a draw goes to.
///
/// Among the cards that may draw now ([`has_room`]): **one with no unit in
/// flight first**, whatever the preferences — a draw on a card mid-turn stalls
/// that turn's stream for as long as it draws — then the lowest `pictures`,
/// then the lower node. When every card that may draw has a unit in flight,
/// the draw waits ([`Placed::Full`]); the gateway holds its card
/// ([`draw_target`]) so that it takes no new unit, and the draw starts once the
/// units there end.
pub(super) fn draw_of(pool: Vec<&MemberView>, weigh: &Weigh<'_>) -> Placed {
    let pick = best(&pool, weigh);
    let placed = match pick {
        Some(m) if weigh.card(m).work() == 0 => Placed::To(m.node),
        _ => Placed::Full,
    };
    tracing::info!(
        target: "cluster::gateway::placement",
        picked = %match placed {
            Placed::To(n) => format!("{n:016x}"),
            _ => "none".to_string(),
        },
        by = "draw",
        waits_for = %pick
            .filter(|_| placed == Placed::Full)
            .map_or_else(|| "-".to_string(), |m| format!("{:016x}", m.node)),
        candidates = %pool
            .iter()
            .map(|m| format!(
                "{:016x} host={} pictures={} drawing={} units={} reach={}",
                m.node,
                m.host,
                weigh
                    .machines
                    .pictures(&m.host)
                    .map_or_else(|| "-".to_string(), |p| p.to_string()),
                weigh.card(m).drawing,
                weigh.card(m).work(),
                guest_reach(m),
            ))
            .collect::<Vec<_>>()
            .join("; "),
        "placed"
    );
    placed
}

/// The card a draw of `pool`, waiting at the gateway, holds: the card it would
/// take once the units there end. Held for as long as the draw waits, whatever
/// is in flight there — let go the moment its last turn ended, the card went to
/// whichever conversation the queue released first. `None` when no card may
/// draw it at all (every one drawing, or none that draws).
pub(crate) fn draw_target(pool: Vec<&MemberView>, weigh: &Weigh<'_>) -> Option<CardKey> {
    best(&pool, weigh).map(CardKey::of)
}

fn best<'a>(pool: &[&'a MemberView], weigh: &Weigh<'_>) -> Option<&'a MemberView> {
    pool.iter()
        .copied()
        .filter(|m| has_room(m, weigh, true))
        .min_by_key(|m| {
            (
                weigh.card(m).work() > 0,
                weigh.machines.pictures(&m.host).unwrap_or(u32::MAX),
                m.node,
            )
        })
}
