//! A draw's reach: which cards can draw a picture of so many pixels (§5.3).

use crate::dispatch::reaches;
use crate::gateway::registry::MemberView;

/// The side of the largest square `m` draws, as its last `CAPACITY` said;
/// zero before it has said one.
pub(crate) fn guest_reach(m: &MemberView) -> u32 {
    m.capacity.as_ref().map_or(0, |c| c.guest_reach)
}

/// The members of `pool` a draw of `draw` pixels may go to: those whose card
/// reaches it; when none does, those that reach furthest, whose own refusal
/// then names the largest draw the app serves. Every member for a request that
/// draws nothing.
pub(crate) fn within_reach(pool: Vec<&MemberView>, draw: Option<u64>) -> Vec<&MemberView> {
    let Some(area) = draw else {
        return pool;
    };
    let reaching: Vec<&MemberView> = pool
        .iter()
        .copied()
        .filter(|m| reaches(guest_reach(m), area))
        .collect();
    if !reaching.is_empty() {
        return reaching;
    }
    let furthest = pool.iter().map(|m| guest_reach(m)).max().unwrap_or(0);
    pool.into_iter()
        .filter(|m| guest_reach(m) == furthest)
        .collect()
}
