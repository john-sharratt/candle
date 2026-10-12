//! A peer gateway's dial, taken: the §3.3 handshake as the gateway's side,
//! admitting only `app = gateway`, then the link.

use std::sync::Arc;

use super::{link, Peering, GATEWAY_APP};
use crate::close::{Close, CloseCode};
use crate::frame::Admit;
use crate::handshake::{admit, HandshakeError};
use crate::pipe::{PipeRx, PipeTx};
use crate::session::{start, Role};

/// What a peer's `ADMIT` carries: a link has no members, claims or floor.
fn peer_admit(session: u64) -> Admit {
    Admit {
        session,
        members: Vec::new(),
        claims: Vec::new(),
        floor_count: 0,
        floor_hash: String::new(),
        routes_wanted: false,
    }
}

impl Peering {
    /// Take a peer's dial on `tx`/`rx` and run the link until it ends.
    /// `secure` when it arrived on the TLS entrance, whose certificate the
    /// dialling peer verified.
    pub async fn accept<T: PipeTx, R: PipeRx>(
        self: &Arc<Self>,
        mut tx: T,
        mut rx: R,
        secure: bool,
    ) {
        let session = {
            let mut s = self.state();
            s.next_id += 1;
            s.next_id
        };
        let decided = admit(
            &mut tx,
            &mut rx,
            self.key(),
            |_| {},
            |hello| {
                if hello.app != GATEWAY_APP {
                    return Err(Close::new(
                        CloseCode::Malformed,
                        format!(
                            "a peer link's handshake names app `{GATEWAY_APP}`, not `{}`",
                            hello.app
                        ),
                    ));
                }
                Ok(peer_admit(session))
            },
        )
        .await;
        let hello = match decided {
            Ok((hello, _)) => hello,
            Err(HandshakeError::Refused(close) | HandshakeError::Closed(close)) => {
                self.refused(None, close.reason);
                return;
            }
            Err(HandshakeError::Dropped) => return,
        };
        let session = start(Role::Gateway, tx, rx);
        link::run(self.clone(), session, Some(hello.node), false, secure).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_peer_is_admitted_with_nothing_but_its_session() {
        let a = peer_admit(7);
        assert_eq!(
            (
                a.session,
                a.members.len(),
                a.claims.len(),
                a.floor_count,
                a.routes_wanted
            ),
            (7, 0, 0, 0, false)
        );
    }
}
