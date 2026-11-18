//! `POST /v1/agent/hold` (`docs/build_agent.md` §5): keep asks it before it
//! stops this process, so the build agent in it holds its builds — the deploy
//! build running waited for, any other run set aside, the queue parked on keep
//! — and answers once it has. Only a caller on this machine is answered: keep,
//! from loopback or this machine's own address. A request over a cluster
//! socket carries no peer address, and is refused like any other.
//!
//! A hold keep asked for lifts by itself if the process is still running
//! [`LIFT_AFTER`] after it answered: the stop it was for did not happen.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use agent::jobs::Jobs;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::StatusCode;
use axum::routing::post;
use axum::Router;

/// The route keep asks.
pub const HOLD_PATH: &str = "/v1/agent/hold";

/// How long a hold keep asked for stands once it has answered.
pub const LIFT_AFTER: Duration = Duration::from_secs(5 * 60);

#[derive(Clone)]
struct Hold {
    jobs: Arc<Jobs>,
    /// The address this process listens on: a caller there is on this machine.
    own: IpAddr,
}

/// `router` with the hold route, holding `jobs`, for callers on this machine,
/// which listens on `own`.
pub fn with_hold(router: Router, jobs: Arc<Jobs>, own: IpAddr) -> Router {
    router.merge(
        Router::new()
            .route(HOLD_PATH, post(hold))
            .with_state(Hold { jobs, own }),
    )
}

/// Whether `peer` is on this machine, which listens on `own`.
fn on_this_machine(peer: Option<SocketAddr>, own: IpAddr) -> bool {
    peer.is_some_and(|p| p.ip().is_loopback() || p.ip() == own)
}

async fn hold(State(h): State<Hold>, req: Request) -> StatusCode {
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(a)| *a);
    if !on_this_machine(peer, h.own) {
        return StatusCode::FORBIDDEN;
    }
    let held = h.jobs.hold();
    tracing::warn!("keep asked to hold this process's builds before it stops it");
    held.until_drained().await;
    tracing::warn!("the builds are held: keep may stop this process");
    tokio::spawn(async move {
        tokio::time::sleep(LIFT_AFTER).await;
        tracing::warn!("keep held the builds and did not stop this process: the hold lifts");
        drop(held);
    });
    StatusCode::OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_caller_on_this_machine_is_answered() {
        let own: IpAddr = "192.168.1.2".parse().unwrap();
        let at = |s: &str| Some(s.parse::<SocketAddr>().unwrap());
        assert!(on_this_machine(at("127.0.0.1:5000"), own));
        assert!(on_this_machine(at("[::1]:5000"), own));
        assert!(on_this_machine(at("192.168.1.2:5000"), own));
        assert!(!on_this_machine(at("192.168.1.3:5000"), own));
        assert!(!on_this_machine(None, own));
    }
}
