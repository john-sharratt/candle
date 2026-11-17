//! The gateway's forwarder of datagrams (`docs/battlecities_app.md` §5):
//! each UDP port `cluster.udp_forward` names is listened on here and carried
//! to the members of its app, a client's flow going to the member a hash of
//! its address and port picks.
//!
//! | Module | |
//! |---|---|
//! | [`pick`] | which member a client's flow is given to |
//! | [`table`] | a port's flows, and the limits on them |
//! | [`flow`] | the port's socket and each flow's task |
//!
//! The members are the app's as the hub has them ([`AppMembers`]): those that
//! serve, each reached at the address its cluster socket came from, and told
//! of a flow over that socket — `POST /api/v1/flows` with
//! `x-tokera-forwarder`, which the gateway removes from every request it
//! passes on, so the member believes it is the gateway's own.

pub mod flow;
pub mod pick;
pub mod table;

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use anyhow::Context;
use axum::body::Body;
use axum::http::{header, Method, Request};
use serde_json::json;
use tokio::net::UdpSocket;

use crate::cluster::transport::send;
use crate::cluster::Gateway;
use crate::config::UdpForward;
use crate::hooks::pushes::OWN;
use flow::{Members, Port, Told};

/// Where a member is told of a flow.
pub const FLOWS_PATH: &str = "/api/v1/flows";

/// The members of one app, as the gateway's hub has them.
pub struct AppMembers {
    pub gateway: Arc<Gateway>,
    pub app: String,
}

impl AppMembers {
    async fn tell(&self, node: u64, via: SocketAddr, client: IpAddr) -> Result<(), String> {
        let body = json!({"via": via.to_string(), "client": client.to_string()});
        let req = Request::builder()
            .method(Method::POST)
            .uri(FLOWS_PATH)
            .header(header::CONTENT_TYPE, "application/json")
            .header(OWN, "1")
            .body(Body::from(body.to_string()))
            .map_err(|e| format!("the announcement does not build: {e}"))?;
        let res = send(&self.gateway.hub, &self.app, node, req)
            .await
            .map_err(|e| format!("member {node:016x} was not told: {e}"))?;
        match res.status().is_success() {
            true => Ok(()),
            false => Err(format!(
                "member {node:016x} answered {} to the flow's announcement",
                res.status()
            )),
        }
    }
}

impl Members for AppMembers {
    fn serving(&self) -> Vec<(u64, IpAddr)> {
        self.gateway.hub.serving(&self.app)
    }

    fn has(&self, node: u64) -> bool {
        self.gateway.hub.has_member(&self.app, node)
    }

    fn announce(&self, node: u64, via: SocketAddr, client: IpAddr) -> Told<'_> {
        Box::pin(self.tell(node, via, client))
    }
}

/// Bind every port `forwards` names on `bind` and carry it from now on. A
/// port that is taken stops the start, as a taken entrance does.
pub async fn start(
    gateway: &Arc<Gateway>,
    forwards: &[UdpForward],
    bind: IpAddr,
) -> anyhow::Result<()> {
    for forward in forwards {
        let members: Arc<dyn Members> = Arc::new(AppMembers {
            gateway: Arc::clone(gateway),
            app: forward.app.clone(),
        });
        for port in &forward.ports {
            let at = SocketAddr::new(bind, port.listen);
            let socket = UdpSocket::bind(at)
                .await
                .with_context(|| format!("binding UDP {at} for `{}`", forward.app))?;
            tracing::info!(
                app = %forward.app,
                listen = %at,
                member_port = port.member,
                "udp: forwarding"
            );
            tokio::spawn(Port::new(socket, port.member, Arc::clone(&members)).run());
        }
    }
    Ok(())
}
