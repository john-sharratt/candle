//! One peer link, once its handshake has passed: `PEER_HELLO` each way, the
//! digests compared, the link registered — or refused, or set aside for the
//! one a crossed dial keeps — and then every frame handed to the gateway's
//! [`PeerHandler`](super::PeerHandler) until the session ends.

use std::sync::Arc;

use tokio::sync::mpsc::Receiver;
use tokio::sync::oneshot;

use super::crossing::{keep, Keep};
use super::{hex, Link, Peering};
use crate::clock::now_ms;
use crate::close::{Close, CloseCode};
use crate::frame::{name, Frame, PeerHello};
use crate::session::{Ended, Incoming, Session};

/// Why a peer's `PEER_HELLO` refuses the peering, if it does.
fn mismatch(me: &PeerHello, peer: &PeerHello, expect: Option<u64>) -> Option<Close> {
    if let Some(n) = expect.filter(|n| *n != peer.node) {
        return Some(Close::new(
            CloseCode::Malformed,
            format!(
                "PEER_HELLO names node {} but the handshake named {}",
                hex(peer.node),
                hex(n)
            ),
        ));
    }
    if peer.node == me.node {
        return Some(Close::new(
            CloseCode::Malformed,
            format!(
                "node {} is this gateway: it cannot peer with itself",
                hex(me.node)
            ),
        ));
    }
    if peer.session_key != me.session_key {
        return Some(Close::new(
            CloseCode::Proof,
            "the session keys differ: a session cookie one gateway signs would not verify on \
             the other",
        ));
    }
    if peer.oauth_client != me.oauth_client {
        return Some(Close::new(
            CloseCode::Proof,
            "the Google OAuth clients differ: a sign-in through one gateway would not be \
             accepted by the other",
        ));
    }
    None
}

/// The peer's `PEER_HELLO`, the first frame of the link.
async fn peer_hello(incoming: &mut Receiver<Incoming>) -> Result<PeerHello, Option<Close>> {
    loop {
        match incoming.recv().await {
            Some(Incoming::Frame(Frame::PeerHello(h))) => return Ok(h),
            Some(Incoming::Frame(other)) => {
                return Err(Some(Close::new(
                    CloseCode::Malformed,
                    format!("expected PEER_HELLO, got {}", name(other.kind())),
                )))
            }
            Some(_) => continue,
            None => return Err(None),
        }
    }
}

/// How the session ended, once it has.
async fn finish(ended: oneshot::Receiver<Ended>) -> Ended {
    ended.await.unwrap_or(Ended::Dropped)
}

/// Run an admitted peer session: `expect` is the peer's node when the
/// handshake named it (a dial taken); `dialled` when this gateway dialled.
pub(crate) async fn run(
    peering: Arc<Peering>,
    session: Session,
    expect: Option<u64>,
    dialled: bool,
    secure: bool,
) -> Ended {
    let Session {
        handle,
        mut incoming,
        ended,
    } = session;
    let me = peering.identity().hello();
    if handle.control(Frame::PeerHello(me.clone())).is_err() {
        return finish(ended).await;
    }
    let hello = match peer_hello(&mut incoming).await {
        Ok(h) => h,
        Err(close) => {
            if let Some(close) = close {
                peering.refused(expect, close.reason.clone());
                handle.close(close);
            }
            return finish(ended).await;
        }
    };
    if let Some(close) = mismatch(&me, &hello, expect) {
        peering.refused(Some(hello.node), close.reason.clone());
        handle.close(close);
        return finish(ended).await;
    }
    let peer = hello.node;
    let dialler = if dialled { peering.node() } else { peer };

    // Registered, or set aside for the link a crossed dial keeps.
    let registered: Result<u64, Close> = {
        let mut s = peering.state();
        let held = s.links.get(&peer).map(|h| (h.dialler, h.handle.clone()));
        match held.map(|(d, h)| (keep(d, dialler), d, h)) {
            Some((Keep::Held, held_dialler, _)) => Err(Close::new(
                CloseCode::Replaced,
                format!(
                    "a link to {} already runs, dialled by {}: the lower node's dial is kept",
                    hex(peer),
                    hex(held_dialler)
                ),
            )),
            replaced => {
                if let Some((_, _, old)) = replaced {
                    old.close(Close::new(
                        CloseCode::Replaced,
                        format!("replaced by the link {} dialled", hex(dialler)),
                    ));
                }
                s.next_id += 1;
                let id = s.next_id;
                s.links.insert(
                    peer,
                    Link {
                        id,
                        handle: handle.clone(),
                        dialler,
                        secure,
                        hello: hello.clone(),
                        since_ms: now_ms(),
                    },
                );
                Ok(id)
            }
        }
    };
    let id = match registered {
        Ok(id) => id,
        Err(close) => {
            handle.close(close);
            return finish(ended).await;
        }
    };
    peering.note(
        peer,
        format!(
            "peer linked: build {}-{}, dialled by {}, {}",
            hello.build.count,
            hello.build.hash,
            hex(dialler),
            if secure {
                "over verified TLS"
            } else {
                "not over verified TLS: no certificate key crosses it"
            }
        ),
    );
    if let Some(h) = peering.handler() {
        h.linked(peer, secure);
    }

    let mut ended = ended;
    let how = loop {
        tokio::select! {
            got = incoming.recv() => match got {
                Some(Incoming::Frame(f)) => {
                    if let Some(h) = peering.handler() {
                        h.frame(peer, f);
                    }
                }
                Some(_) => {}
                None => break None,
            },
            how = &mut ended => break Some(how.unwrap_or(Ended::Dropped)),
        }
    };
    let how = match how {
        Some(how) => how,
        None => finish(ended).await,
    };

    let still_ours = {
        let mut s = peering.state();
        let ours = s.links.get(&peer).is_some_and(|l| l.id == id);
        if ours {
            s.links.remove(&peer);
        }
        ours
    };
    if still_ours {
        peering.note(peer, format!("peer unlinked: {how:?}"));
        if let Some(h) = peering.handler() {
            h.unlinked(peer);
        }
    }
    how
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::BuildId;

    fn hello(node: u64, key: u8, client: u8) -> PeerHello {
        PeerHello {
            node,
            build: BuildId {
                count: 1,
                hash: "h".into(),
                local: String::new(),
            },
            session_key: [key; 32],
            oauth_client: [client; 32],
        }
    }

    /// Agreeing secrets link; a differing session key or OAuth client is
    /// refused 4401 naming which; a node other than the handshake's, or this
    /// gateway's own, is refused 4400.
    #[test]
    fn a_peer_hello_must_agree_on_the_shared_secrets() {
        let me = hello(1, 7, 9);
        assert_eq!(mismatch(&me, &hello(2, 7, 9), Some(2)), None);
        assert_eq!(mismatch(&me, &hello(2, 7, 9), None), None);
        let key = mismatch(&me, &hello(2, 8, 9), None).unwrap();
        assert_eq!(key.code, 4401);
        assert!(key.reason.starts_with("the session keys differ"));
        let client = mismatch(&me, &hello(2, 7, 1), None).unwrap();
        assert!(client.reason.starts_with("the Google OAuth clients differ"));
        assert_eq!(
            mismatch(&me, &hello(2, 7, 9), Some(3)).unwrap().reason,
            "PEER_HELLO names node 0000000000000002 but the handshake named 0000000000000003"
        );
        assert_eq!(mismatch(&me, &hello(1, 7, 9), None).unwrap().code, 4400);
    }
}
