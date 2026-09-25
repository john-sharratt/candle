//! `GET /v1/me` — what the caller may do here.
//!
//! The GUI asks once on load, so its tools dial offers exactly the modes the
//! caller's role permits and starts at the role's default. It is advice to the
//! page, not the enforcement: `/v1/chat/completions` resolves the same role and
//! holds a mode above it to Restricted whatever the page sent.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{ConnectInfo, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::Serialize;
use web::auth::Role;

use crate::access;
use crate::session::ZendSession;

#[derive(Serialize)]
pub struct Me {
    pub role: Role,
    /// The tools modes this caller may choose, by wire id, in dial order.
    pub tool_modes: Vec<&'static str>,
    /// The mode a turn runs in when the caller names none.
    pub default_tools: &'static str,
}

pub async fn me(
    State(session): State<Arc<ZendSession>>,
    peer: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
) -> Json<Me> {
    let peer = peer.map(|ConnectInfo(a)| a.ip());
    let role = access::role(
        &headers,
        peer,
        session.gateways(),
        session.roles(),
        session.local_signin(),
    );
    Json(Me {
        role,
        tool_modes: access::allowed_modes(role)
            .into_iter()
            .map(|m| m.id())
            .collect(),
        default_tools: access::default_mode(role).id(),
    })
}
