//! Where a unit of an app that runs no model goes (`docs/static_placement.md`
//! §3.1): the agents, keep, the MCP aggregator. The machine table places the
//! model's work; these go to the member with the fewest requests in flight, then
//! the lower node — a lagging one only when every one lags.

use super::{prefer_current, Placed};
use crate::gateway::registry::MemberView;

/// The member of `pool` with the fewest requests in flight.
pub(super) fn least_of(pool: Vec<&MemberView>) -> Placed {
    prefer_current(pool)
        .into_iter()
        .min_by_key(|m| (m.in_flight, m.node))
        .map_or(Placed::Nobody, |m| Placed::To(m.node))
}
