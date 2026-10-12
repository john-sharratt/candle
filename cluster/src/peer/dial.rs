//! This gateway's dial loop to one configured peer: the member's loop of §3.2
//! — connect, handshake, run, back off, again — with `app = gateway`. A link
//! a crossed dial set aside is closed 4409, which ends this loop: the peer's
//! own dial is the one kept, and the peer redials when it drops.

use std::sync::Arc;
use std::time::Duration;

use super::{link, Peering, GATEWAY_APP};
use crate::clock::now_ms;
use crate::dial::{dial, Attempt, Connect, Stopped};
use crate::frame::Hello;
use crate::handshake::PROTO;
use crate::session::Ended;

impl Peering {
    /// Dial the peer at `url` through `connector` until a close stops the
    /// loop. `secure` when the connector verifies the peer's certificate.
    pub async fn dial<C: Connect>(
        self: &Arc<Self>,
        url: &str,
        connector: C,
        secure: bool,
    ) -> Stopped {
        let row = self.dialling(url);
        let me = self.identity().clone();
        let name = self.hub.name().to_string();
        let hello = move || Hello {
            proto: PROTO,
            app: GATEWAY_APP.to_string(),
            node: me.node,
            host: name.clone(),
            build: me.build.clone(),
            rejected: false,
            model: None,
            power: 0,
            repos: Vec::new(),
            now_ms: now_ms(),
            // `join` draws a fresh nonce.
            nonce: [0; 32],
        };
        let this = self.clone();
        let serve = move |_admit, session| {
            let this = this.clone();
            async move {
                this.dial_state(row, "linked", None);
                link::run(this, session, None, true, secure).await
            }
        };
        let attempted = |attempt: &Attempt, wait: Option<Duration>| {
            let close = match attempt {
                Attempt::ConnectFailed(e) => format!("could not connect: {e}"),
                Attempt::Handshake(e) => e.to_string(),
                Attempt::Session(Ended::Closed(c) | Ended::Refused(c)) => c.to_string(),
                Attempt::Session(Ended::Dropped) => "the socket dropped".into(),
                Attempt::Session(Ended::Dead) => "no frame for 45 s".into(),
            };
            let state = if wait.is_some() {
                "retrying"
            } else {
                "stopped"
            };
            self.dial_state(row, state, Some(close));
        };
        dial(connector, self.key().clone(), hello, serve, attempted).await
    }
}
