//! Treats a tool round-trip as one unit, so a call is never selected without its
//! answer.
//!
//! Turns coupled into an exchange collapse to one candidate whose score is the
//! best of its members, so a hit on any member pulls in the whole run.
//! `close_under_exchange` applies the same rule to a turn named directly, such as
//! a group's default or a timeline's anchor. The budget prices an exchange as the
//! sum of its members, and emission expands a selected candidate back into its
//! turns in ascending order. Two statics report once per process that a working
//! set exceeds the trained window, or that a dialogue was projected unfilled.

use super::*;

/// Collapse `keys` into exchange candidates: one `(representative, score)` per
/// exchange, in first-appearance order, recording each representative's full
/// member list in `members`.
///
/// The representative is the exchange's first turn, so the candidate order the
/// rules see is still insertion order. The score is the best over the members:
/// the belief scan already stamps every member of a run with the shared
/// exchange score, and taking the best is what makes "a hit on any member pulls
/// in the whole run" true for every other scoring path too.
pub(super) fn exchange_candidates(
    resolver: &dyn ContentResolver,
    keys: &[TurnKey],
    members: &mut HashMap<TurnKey, Vec<TurnKey>>,
) -> Vec<(TurnKey, f32)> {
    resolver
        .group_exchanges(keys)
        .into_iter()
        .filter_map(|m| {
            let rep = *m.first()?;
            let score = m
                .iter()
                .map(|k| resolver.turn_score(*k))
                .fold(f32::MIN, f32::max);
            members.insert(rep, m);
            Some((rep, score))
        })
        .collect()
}

/// Take `key`'s exchange as one candidate, recording its membership, and return
/// the representative that stands for the whole run at every later step.
///
/// For the two places that name a turn directly rather than ranking candidates
/// — a group's `default:` fallback and the timeline anchor — so neither can
/// seat half a round-trip.
pub(super) fn close_under_exchange(
    resolver: &dyn ContentResolver,
    members: &mut HashMap<TurnKey, Vec<TurnKey>>,
    key: TurnKey,
) -> TurnKey {
    let Some(m) = resolver.group_exchanges(&[key]).into_iter().next() else {
        return key;
    };
    let Some(&rep) = m.first() else {
        return key;
    };
    members.insert(rep, m);
    rep
}

/// Whether a working set past the engine's trained window has been reported:
/// the configuration is fixed for the process, so it is said once.
pub(super) static SLOT_EXPANDED_WARNED: AtomicBool = AtomicBool::new(false);

/// Whether a dialogue with a working set has been projected without a fill —
/// its members placed whole, no cap binding — once for the process.
pub(super) static UNFILLED_REPORTED: AtomicBool = AtomicBool::new(false);

/// Emit a selected candidate: the exchange's turns in ascending order, right
/// where the representative sat. Selection ranks representatives; this is the
/// one place they expand back into the turns the projection carries.
pub(super) fn push_exchange(
    out: &mut Vec<(GroupId, TurnKey)>,
    group: GroupId,
    members: &HashMap<TurnKey, Vec<TurnKey>>,
    key: TurnKey,
) {
    match members.get(&key) {
        Some(m) => out.extend(m.iter().map(|k| (group, *k))),
        None => out.push((group, key)),
    }
}

/// Tokens one exchange candidate costs — the sum over its members, so every
/// budget decision is priced against what the projection actually emits.
///
/// A key with no recorded membership (an anchor head, a `default:` fallback)
/// prices as itself; the caller closes those under exchange before emitting.
pub(super) fn exchange_token_count(
    resolver: &dyn ContentResolver,
    members: &HashMap<TurnKey, Vec<TurnKey>>,
    key: TurnKey,
) -> usize {
    match members.get(&key) {
        Some(m) => m.iter().map(|k| resolver.turn_token_count(*k)).sum(),
        None => resolver.turn_token_count(key),
    }
}
