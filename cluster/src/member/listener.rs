//! What a member's own listener does with a request (§5.8): forward it up to a
//! gateway while joined, serve it locally while cut off, and refuse it while
//! the member cannot serve — connecting, rejoining, reloading, or stopped.
//! Some requests are always served here:
//!
//! - one that arrived on a socket — the cluster already routed it, so
//!   forwarding it would only loop;
//! - one a gateway sent here in dumb mode, marked `x-tokera-hop: dumb` by an
//!   allowlisted gateway — forwarding it up would send it straight back;
//! - `GET /v1/status`, which always answers for this machine;
//! - every request to a local build, which serves its work in progress to
//!   whoever reaches it directly (§3.10).

use super::link::Link;

/// The mark a gateway sets on a request it proxies to a configured upstream.
pub const DUMB_HOP: &str = "x-tokera-hop";
pub const DUMB_HOP_VALUE: &str = "dumb";

/// The header naming the member a request came from, its node in hex — set by
/// the gateway on a request that arrived on a stream a member opened (§5.8),
/// and on nothing else.
pub const FROM_MEMBER: &str = "x-tokera-from-member";

/// The header a member sets on a request it sends up to ask that it be placed
/// on a peer — another member of its app, never a certifying canary — while
/// one can take it (`docs/guest_draw_orchestration.md` §3). Honoured only on a
/// request that came up from a member; a request without it is placed as a
/// client's is.
pub const PREFER_PEER: &str = "x-tokera-prefer-peer";

/// The one path that never forwards.
pub const STATUS_PATH: &str = "/v1/status";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Serve {
    /// Answer it here.
    Local,
    /// Open a stream to a gateway and answer from wherever it routes.
    ForwardUp,
    /// `503` with `Retry-After`, and why.
    Unavailable(String),
}

/// A request as the listener sees it.
#[derive(Debug, Clone, Copy)]
pub struct Arrival<'a> {
    pub path: &'a str,
    pub on_socket: bool,
    /// Marked dumb by a peer this listener believes `x-tokera-*` from.
    pub dumb_from_gateway: bool,
}

pub fn decide(link: &Link, local_build: bool, arrival: Arrival<'_>) -> Serve {
    if arrival.on_socket || arrival.dumb_from_gateway || arrival.path == STATUS_PATH || local_build
    {
        return Serve::Local;
    }
    match link {
        Link::Joined => Serve::ForwardUp,
        Link::CutOff => Serve::Local,
        Link::Starting => Serve::Unavailable("connecting to the cluster".into()),
        Link::Rejoining => Serve::Unavailable("rejoining the cluster".into()),
        Link::Reloading => Serve::Unavailable("reloading its substrate".into()),
        Link::Stopped(close) => Serve::Unavailable(close.reason.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::close::{Close, CloseCode};

    fn at(path: &str) -> Arrival<'_> {
        Arrival {
            path,
            on_socket: false,
            dumb_from_gateway: false,
        }
    }

    #[test]
    fn joined_forwards_up_and_cut_off_serves_locally() {
        assert_eq!(
            decide(&Link::Joined, false, at("/v1/chat")),
            Serve::ForwardUp
        );
        assert_eq!(decide(&Link::CutOff, false, at("/v1/chat")), Serve::Local);
        assert!(matches!(
            decide(&Link::Starting, false, at("/v1/chat")),
            Serve::Unavailable(_)
        ));
        assert_eq!(
            decide(&Link::Rejoining, false, at("/v1/chat")),
            Serve::Unavailable("rejoining the cluster".into())
        );
        assert_eq!(
            decide(&Link::Reloading, false, at("/v1/chat")),
            Serve::Unavailable("reloading its substrate".into())
        );
        let stopped = Link::Stopped(Close::new(
            CloseCode::Superseded,
            "rebuild: below the floor",
        ));
        assert_eq!(
            decide(&stopped, false, at("/")),
            Serve::Unavailable("rebuild: below the floor".into())
        );
    }

    /// A request off a socket, a dumb-mode request, the status route and a local
    /// build's requests are served here in every state.
    #[test]
    fn some_requests_are_always_served_here() {
        for link in [Link::Joined, Link::Starting] {
            let mut a = at("/v1/chat");
            a.on_socket = true;
            assert_eq!(decide(&link, false, a), Serve::Local);
            let mut a = at("/v1/chat");
            a.dumb_from_gateway = true;
            assert_eq!(decide(&link, false, a), Serve::Local);
            assert_eq!(decide(&link, false, at(STATUS_PATH)), Serve::Local);
            assert_eq!(decide(&link, true, at("/v1/chat")), Serve::Local);
        }
    }
}
