//! Server-side mock APIs, so a console can be developed with no daemon behind
//! it — `web --authoritative`.
//!
//! These live here, beside the files they serve, for the same reason
//! `content/npcd/lib/api.mock.js` does: a console and its fixtures ship
//! together. The client-side mock (`?mock=1`) replaces the network entirely and
//! is what the Playwright suite drives; this one replaces only the *daemon*, so
//! the browser still makes real requests over real sockets and the proxy, the
//! router, the error pages and the websocket tunnel are all exercised on the
//! way. Between them they cover both halves of "the GUI works without a GPU".
//!
//! A mock is a fixture, never a fallback. Nothing selects one automatically:
//! `--authoritative` is a deliberate flag, and without it every API route is
//! forwarded to a real daemon and a missing one is an error, not a silent
//! substitution.

use axum::Router;

pub mod npcd;

/// The mock API for a site, if this build carries one.
///
/// The single place that maps a site name to its mock — `--authoritative`
/// registers whatever this returns, and `--check` reports the same thing, so
/// the flag's summary can never claim a mock the server would not serve.
pub fn for_site(name: &str) -> Option<Router> {
    match name {
        "npcd" => Some(npcd::router()),
        // `zend` has a real daemon and no mock: its console is not split out of
        // that binary yet, so there is nothing here to stand in for it.
        _ => None,
    }
}
