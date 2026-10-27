//! A member's side of the cluster (§3.2, §5.4, §5.8): how its link to each
//! app's clusters stands, what its own listener does with a request, the
//! claims it holds, the node it is kept as, and [`AppMember`], which runs one
//! app's sockets.

mod app;
mod claims;
mod link;
mod listener;
mod node_file;

pub use app::{
    AppMember, CapacitySource, ClaimUse, ClusterView, Identity, KeyThroughHook, ReleaseHook,
    UnclaimHook, APPLY_WAIT, CAPACITY_EVERY, GROUP,
};
pub use claims::{Check, MemberClaims};
pub use link::{Link, Links, START_WAIT};
pub use listener::{
    decide, Arrival, Serve, DUMB_HOP, DUMB_HOP_VALUE, FROM_MEMBER, PREFER_PEER, STATUS_PATH,
};
pub use node_file::kept_node;
