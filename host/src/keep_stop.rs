//! `POST /v1/host/stop`: keep asks it before it stops this process for a
//! trial, an accept, a restart or a stop, so the process leaves the way a
//! `shutdown` command makes it leave (`docs/replicated_substrate.md` §12.4) —
//! hands its live work on, drains, sends every event it wrote, flushes, exits —
//! instead of being killed with events still unsent. keep waits for the exit
//! and kills only past its own limit.
//!
//! Only a caller on this machine is answered: keep, from loopback or this
//! machine's own address (as `agent_hold`). The answer comes at once — `202`
//! when this request began the stop, `409` when one was already under way —
//! and the stop runs after it.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::extract::{ConnectInfo, Request, State};
use axum::http::StatusCode;
use axum::routing::post;
use axum::Router;

use crate::agent_hold::on_this_machine;
use crate::stopping::{Ending, Stopping};

/// The route keep asks.
pub const STOP_PATH: &str = "/v1/host/stop";

#[derive(Clone)]
struct Stop {
    stopping: Arc<Stopping>,
    /// The address this process listens on: a caller there is on this machine.
    own: IpAddr,
}

/// `router` with the stop route, for callers on this machine, which listens on
/// `own`.
pub fn with_stop(router: Router, stopping: Arc<Stopping>, own: IpAddr) -> Router {
    router.merge(
        Router::new()
            .route(STOP_PATH, post(stop))
            .with_state(Stop { stopping, own }),
    )
}

/// The answer to a stop request from `peer`: refused off this machine, else
/// whether it began the stop.
fn answer(peer: Option<SocketAddr>, own: IpAddr, stopping: &Stopping) -> StatusCode {
    if !on_this_machine(peer, own) {
        return StatusCode::FORBIDDEN;
    }
    if stopping.ask(Ending::Exit) {
        tracing::warn!("keep asked this process to stop: draining, sending what it wrote, exiting");
        StatusCode::ACCEPTED
    } else {
        StatusCode::CONFLICT
    }
}

async fn stop(State(s): State<Stop>, req: Request) -> StatusCode {
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(a)| *a);
    answer(peer, s.own, &s.stopping)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **keep's stop begins the shutdown once**, as an exit; a second one finds
    /// it under way; a caller off this machine is refused and stops nothing.
    #[test]
    fn a_stop_from_this_machine_begins_the_shutdown_once() {
        let own: IpAddr = "192.168.1.2".parse().unwrap();
        let at = |s: &str| Some(s.parse::<SocketAddr>().unwrap());
        let s = Stopping::new();
        assert_eq!(
            answer(at("192.168.1.3:5000"), own, &s),
            StatusCode::FORBIDDEN
        );
        assert_eq!(answer(None, own, &s), StatusCode::FORBIDDEN);
        assert_eq!(s.ending(), None, "a refused stop stops nothing");
        assert_eq!(answer(at("127.0.0.1:5000"), own, &s), StatusCode::ACCEPTED);
        assert_eq!(s.ending(), Some(Ending::Exit));
        assert_eq!(
            answer(at("192.168.1.2:5000"), own, &s),
            StatusCode::CONFLICT
        );
    }
}
