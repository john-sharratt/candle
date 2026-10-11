//! The member's side of the handshake.

use super::proof::{member_proof, nonce, verify_gateway};
use super::{next, refuse, send, unexpected, GroupKey, HandshakeError};
use crate::close::{Close, CloseCode};
use crate::frame::{Admit, Frame, Hello, Prove};
use crate::pipe::{PipeRx, PipeTx};

/// Send `hello` (its nonce is replaced with a fresh one), check the gateway's
/// proof, prove membership, and wait for `ADMIT`. A gateway whose proof fails
/// is not ours: the socket is closed 4401.
pub async fn join<T: PipeTx, R: PipeRx>(
    tx: &mut T,
    rx: &mut R,
    key: &GroupKey,
    mut hello: Hello,
) -> Result<Admit, HandshakeError> {
    hello.nonce = nonce();
    send(tx, &Frame::Hello(Box::new(hello.clone()))).await?;

    let challenge = match next(tx, rx).await? {
        Frame::Challenge(c) => c,
        other => return Err(unexpected(tx, &other, "CHALLENGE").await),
    };
    if !verify_gateway(
        key,
        &hello.app,
        hello.node,
        &hello.nonce,
        &challenge.nonce,
        &challenge.proof,
    ) {
        return Err(refuse(
            tx,
            Close::new(
                CloseCode::Proof,
                "the gateway's proof failed: this is not our gateway, or its group key differs",
            ),
        )
        .await);
    }
    let proof = member_proof(key, &hello.app, hello.node, &challenge.nonce, &hello.nonce);
    send(tx, &Frame::Prove(Prove { proof })).await?;

    match next(tx, rx).await? {
        Frame::Admit(admit) => Ok(admit),
        other => Err(unexpected(tx, &other, "ADMIT").await),
    }
}
