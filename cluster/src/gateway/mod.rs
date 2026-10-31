//! A gateway's cluster (§3–§5, §12). The state — who it admits, the members of
//! each app and their claims, where each request goes — has no I/O; [`Hub`]
//! runs member sockets over any transport and answers the `web` crate's
//! proxy and status page.

mod admission;
mod agent_row;
mod agents;
mod claims;
mod coord_log;
mod drain;
mod hub;
mod load;
mod log;
mod open_answers;
mod paxos_log;
mod placement;
mod planner;
mod queue;
#[cfg(test)]
mod queue_tests;
mod registry;
mod relay;
mod status;
mod tiebreak;
mod traffic;
pub mod upgrades;
mod world_log;
mod worlds_view;

pub use admission::{Admission, Admitted, MAX_SKEW_MS};
pub use agent_row::{AgentRepo, AgentRow, AGENTS_PATH};
pub use agents::{
    choose, choose_with_standby, Choice, Need, LINEAGE_HEADER, NODE_HEADER, PUSH_HEADER,
    REPO_HEADER, SPILL_AFTER, TOOLCHAIN_HEADER,
};
pub use claims::Claims;
pub use coord_log::changes;
pub use drain::DRAIN_WAIT;
pub use hub::{
    is_service, rolls_out, Hub, MemberStream, APPS, CATCHUP_CAP, KEEP, LAG_CAP, MCP, MEMBERS_EVERY,
    REFUSALS_KEPT, SERVICES,
};
pub use load::{Load, Picked};
pub use log::{CoordLog, Entry};
pub use paxos_log::paxos_step;
pub use queue::{Placing, RouteAt, Urgency, HOLD};
pub use registry::{MemberView, Pick, Registry};
pub use status::{
    state_name, AppView, ClaimView, FaultRow, HubStatus, MemberRow, Queues, Refusal, SocketReport,
    TiebreakLease,
};
pub use traffic::{Class, Failure, OpenStream, Traffic, TrafficView};
pub use worlds_view::{disconnected, proposer_key, worlds, DisconnectedRow, WorldView};
