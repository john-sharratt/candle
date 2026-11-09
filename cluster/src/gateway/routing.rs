//! How a request for an app is routed, as the hub decides it under one lock:
//! no member admitted, no member ready, or the table it is matched in.

use std::sync::Arc;

use super::registry::Registry;
use crate::dispatch::RouteTable;

/// How a request is routed ([`super::Hub::routing`]).
#[derive(Debug, Clone, PartialEq)]
pub enum Routing {
    /// No member of the app is admitted here: its configured upstream serves.
    Unadmitted,
    /// Members are admitted and none is ready: the one standing by answers, or
    /// nobody does when none is left to (§5.7).
    StandingBy(Option<u64>),
    /// A ready member's table, shared.
    Table(Arc<RouteTable>),
}

/// Whether an app is admitted here, and who stands by ([`super::Hub::standing`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standing {
    Unadmitted,
    By(Option<u64>),
}

/// [`Standing`] as `registry` has it.
pub fn standing_of(registry: &Registry, app: &str) -> Standing {
    if registry.admits(app) {
        Standing::By(registry.standing_by(app))
    } else {
        Standing::Unadmitted
    }
}
