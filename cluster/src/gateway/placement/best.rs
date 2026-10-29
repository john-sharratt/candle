//! `Best` (§5.3): the soonest finish among the members whose card has room, a
//! probe of one that has not learned what a turn costs it, or power per
//! request — and, when no card has room, nobody (rule 3 of
//! `docs/cooperative_admission.md` §4).

use std::cmp::{Ordering, Reverse};

use super::{capacity_of, finish_us, has_room, held, width, Placed, TurnSize, Weigh};
use crate::gateway::registry::MemberView;

/// The best of `ready` (§5.3): among those whose card has room ([`has_room`])
/// — the soonest finish among those that have said what a turn costs them
/// ([`finish_us`]). A member that has not learned that yet takes a turn while
/// its card is answering none: one at a time, which is what it learns from.
/// With none learned, the largest share by [`rank`]. With no card that has
/// room, [`Placed::Full`]: the unit waits at the gateway (§4.3).
///
/// **An unlearned member is probed, not shared into.** It learns only from the
/// turns it is given, so it must get some; but its share of a burst is a guess.
/// Ranking every candidate by power whenever one had not learned put 12 of run
/// 36's 48 heavy-decode turns on WhiteWorld, fresh from a restart, at 25–45 s a
/// turn against BlackWorld's 10–18 s, and halved the pattern's rate.
pub(super) fn best_of(ready: Vec<&MemberView>, weigh: &Weigh<'_>) -> Placed {
    let size = TurnSize::of(&ready);
    // Every candidate as the pick saw it, for the placement line below.
    let seen: Vec<String> = ready
        .iter()
        .map(|m| placement_view(m, size, weigh))
        .collect();
    let pool: Vec<&MemberView> = ready
        .iter()
        .copied()
        .filter(|m| has_room(m, weigh, false))
        .collect();
    if pool.is_empty() {
        tracing::info!(
            target: "cluster::gateway::placement",
            picked = "none",
            by = "full",
            prompt = size.prompt,
            reply = size.reply,
            candidates = %seen.join("; "),
            "placed"
        );
        return Placed::Full;
    }
    let rank = |a: &&MemberView, b: &&MemberView| rank(a, b, weigh);
    let finishes: Vec<(u64, &MemberView)> = pool
        .iter()
        .filter_map(|m| finish_us(m, weigh.card(m), size).map(|f| (f, *m)))
        .collect();
    let probe = pool
        .iter()
        .copied()
        .filter(|m| finish_us(m, weigh.card(m), size).is_none() && weigh.card(m).in_flight == 0)
        .max_by(rank);
    let (by, pick) = match (probe, finishes.is_empty()) {
        (_, true) => ("power", pool.into_iter().max_by(rank).map(|m| m.node)),
        (Some(m), false) => ("probe", Some(m.node)),
        (None, false) => (
            "finish",
            finishes
                .into_iter()
                .min_by(|(fa, a), (fb, b)| fa.cmp(fb).then(rank(b, a)))
                .map(|(_, m)| m.node),
        ),
    };
    tracing::info!(
        target: "cluster::gateway::placement",
        picked = %pick.map_or_else(|| "none".to_string(), |n| format!("{n:016x}")),
        by,
        prompt = size.prompt,
        reply = size.reply,
        candidates = %seen.join("; "),
        "placed"
    );
    pick.map_or(Placed::Nobody, Placed::To)
}

/// One candidate as a placement weighed it: its node, whether its card had
/// room, what is placed on the card, the figures it advertised, its width and
/// the rows the fence holds back there, and the finish they give a turn of
/// `size`.
fn placement_view(m: &MemberView, size: TurnSize, weigh: &Weigh<'_>) -> String {
    let c = capacity_of(m);
    let card = weigh.card(m);
    let width = width(c);
    format!(
        "{:016x} room={} card_in_flight={} card_pending={} drawing={} backlog={} prefill_tok_s={} prefill_wide={} step_next_us={} step_wide_us={} tok_per_step_milli={} width={} fence_held={} finish_ms={}",
        m.node,
        has_room(m, weigh, false),
        card.in_flight,
        card.pending(),
        card.drawing,
        c.prefill_backlog,
        c.prefill_tok_s,
        c.prefill_tok_s_wide,
        c.decode_step_next_us,
        c.decode_step_wide_us,
        c.tokens_per_step_milli,
        width.map_or_else(|| "-".to_string(), |w| w.to_string()),
        width.map_or(0, |w| held(w, weigh.fence(m))),
        finish_us(m, card, size).map_or_else(|| "-".to_string(), |f| (f / 1000).to_string()),
    )
}

/// `a` against `b` for `Best`: the larger share of power per unit on its card
/// — `power / (1 + in_flight)`, compared cross-multiplied so no share rounds —
/// then idle over busy, then the lower node.
fn rank(a: &MemberView, b: &MemberView, weigh: &Weigh<'_>) -> Ordering {
    let (ca, cb) = (capacity_of(a), capacity_of(b));
    let share_a = u64::from(ca.power) * (1 + u64::from(weigh.card(b).in_flight));
    let share_b = u64::from(cb.power) * (1 + u64::from(weigh.card(a).in_flight));
    share_a
        .cmp(&share_b)
        .then(ca.idle.cmp(&cb.idle))
        .then(Reverse(a.node).cmp(&Reverse(b.node)))
}
