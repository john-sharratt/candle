//! A member's side of the cluster (§3.2, §5.4, §5.8): how its link to each
//! app's clusters stands, what its own listener does with a request, the
//! claims it holds, and [`AppMember`], which runs one app's sockets.

mod app;
mod claims;
mod link;
mod listener;

pub use app::{AppMember, CapacitySource, ClusterView, Identity, APPLY_WAIT, CAPACITY_EVERY};
pub use claims::{Check, MemberClaims};
pub use link::{Link, Links, START_WAIT};
pub use listener::{decide, Arrival, Serve, DUMB_HOP, DUMB_HOP_VALUE, STATUS_PATH};
