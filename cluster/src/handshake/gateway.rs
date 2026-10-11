//! The gateway's side of the handshake.

use super::proof::{gateway_proof, nonce, verify_member};
use super::{next, refuse, send, unexpected, GroupKey, HandshakeError, PROTO};
use crate::close::{Close, CloseCode};
use crate::frame::{Admit, Challenge, Frame, Hello};
use crate::pipe::{PipeRx, PipeTx};

/// Take a member's `HELLO`, refuse an unsupported protocol (4426), challenge
/// it, check its proof (4401), and then ask `decide` — the build floor, the
/// model lock, the clock and the node's session (§3.3–§3.6) — for its `ADMIT`
/// or the close that refuses it. `seen` is shown the `HELLO` as it arrives,
/// before anything is checked, so a refusal can name who it refused. Returns
/// the member's `HELLO` and its `ADMIT`.
pub async fn admit<T, R, S, D>(
    tx: &mut T,
    rx: &mut R,
    key: &GroupKey,
    seen: S,
    decide: D,
) -> Result<(Hello, Admit), HandshakeError>
where
    T: PipeTx,
    R: PipeRx,
    S: FnOnce(&Hello),
    D: FnOnce(&Hello) -> Result<Admit, Close>,
{
    let hello = match next(tx, rx).await? {
        Frame::Hello(h) => *h,
        other => return Err(unexpected(tx, &other, "HELLO").await),
    };
    seen(&hello);
    if hello.proto != PROTO {
        return Err(refuse(
            tx,
            Close::new(
                CloseCode::Protocol,
                format!(
                    "protocol {} is not supported; this gateway speaks {PROTO}",
                    hello.proto
                ),
            ),
        )
        .await);
    }
    if hello.node == 0 {
        return Err(refuse(tx, Close::new(CloseCode::Malformed, "node 0 is no node")).await);
    }

    let nonce_g = nonce();
    let proof = gateway_proof(key, &hello.app, hello.node, &hello.nonce, &nonce_g);
    send(
        tx,
        &Frame::Challenge(Challenge {
            nonce: nonce_g,
            proof,
        }),
    )
    .await?;

    let prove = match next(tx, rx).await? {
        Frame::Prove(p) => p,
        other => return Err(unexpected(tx, &other, "PROVE").await),
    };
    if !verify_member(
        key,
        &hello.app,
        hello.node,
        &nonce_g,
        &hello.nonce,
        &prove.proof,
    ) {
        return Err(refuse(
            tx,
            Close::new(
                CloseCode::Proof,
                format!(
                    "node {:016x}'s proof failed: its group key differs from this gateway's",
                    hello.node
                ),
            ),
        )
        .await);
    }

    match decide(&hello) {
        Ok(admit) => {
            send(tx, &Frame::Admit(admit.clone())).await?;
            Ok((hello, admit))
        }
        Err(close) => Err(refuse(tx, close).await),
    }
}
