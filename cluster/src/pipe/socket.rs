//! A WebSocket as a message pipe. Generic over the socket type, so the
//! gateway's axum socket and a member's `tokio-tungstenite` client share it:
//! each says how its library's messages map to ours with a [`Codec`].
//!
//! Sends go through a pump task that writes and flushes one message at a
//! time; a message counts as unsent until its flush returns, which is when the
//! transport has taken it. That count gates the catch-up lane (§4.2).

use std::marker::PhantomData;

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use tokio::sync::mpsc;

use super::{Drain, Message, PipeClosed, PipeRx, PipeTx};
use crate::close::Close;

/// What a socket's message is, as a pipe reads it.
pub enum Classified {
    Frame(Vec<u8>),
    Close(Close),
    /// A ping, pong or anything else the transport answers itself.
    Skip,
}

/// How one WebSocket library's messages map to a pipe's.
pub trait Codec: Send + Sync + 'static {
    type Message: Send + 'static;
    fn binary(bytes: Vec<u8>) -> Self::Message;
    fn close(close: &Close) -> Self::Message;
    fn classify(message: Self::Message) -> Classified;
}

/// How many messages wait for the pump before a send waits.
const PUMP_DEPTH: usize = 64;

pub struct SocketTx {
    tx: mpsc::Sender<Message>,
    unsent: Drain,
}

pub struct SocketRx<S, C> {
    stream: SplitStream<S>,
    codec: PhantomData<C>,
}

/// Split `socket` into a pipe's two halves, spawning the send pump.
pub fn split<C, S, E>(socket: S) -> (SocketTx, SocketRx<S, C>)
where
    C: Codec,
    S: Sink<C::Message, Error = E> + Stream<Item = Result<C::Message, E>> + Send + Unpin + 'static,
    E: Send + 'static,
{
    let (sink, stream) = socket.split();
    let (tx, rx) = mpsc::channel(PUMP_DEPTH);
    let unsent = Drain::new();
    tokio::spawn(pump::<C, S, E>(sink, rx, unsent.clone()));
    (
        SocketTx { tx, unsent },
        SocketRx {
            stream,
            codec: PhantomData,
        },
    )
}

async fn pump<C, S, E>(
    mut sink: SplitSink<S, C::Message>,
    mut rx: mpsc::Receiver<Message>,
    unsent: Drain,
) where
    C: Codec,
    S: Sink<C::Message, Error = E> + Stream<Item = Result<C::Message, E>> + Send + Unpin + 'static,
{
    while let Some(message) = rx.recv().await {
        let len = message.len();
        let (wire, last) = match message {
            Message::Frame(bytes) => (C::binary(bytes), false),
            Message::Close(close) => (C::close(&close), true),
        };
        let sent = sink.send(wire).await;
        unsent.sub(len);
        if sent.is_err() || last {
            break;
        }
    }
    let _ = sink.close().await;
}

impl PipeTx for SocketTx {
    async fn send(&mut self, message: Message) -> Result<(), PipeClosed> {
        let len = message.len();
        self.unsent.add(len);
        if self.tx.send(message).await.is_err() {
            self.unsent.sub(len);
            return Err(PipeClosed);
        }
        Ok(())
    }

    fn drain(&self) -> &Drain {
        &self.unsent
    }
}

impl<S, C, E> PipeRx for SocketRx<S, C>
where
    C: Codec,
    S: Stream<Item = Result<C::Message, E>> + Send + Unpin + 'static,
    E: Send + 'static,
{
    async fn recv(&mut self) -> Option<Message> {
        loop {
            match self.stream.next().await? {
                Err(_) => return None,
                Ok(m) => match C::classify(m) {
                    Classified::Frame(bytes) => return Some(Message::Frame(bytes)),
                    Classified::Close(close) => return Some(Message::Close(close)),
                    Classified::Skip => continue,
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::duplex;
    use tokio_tungstenite::tungstenite::protocol::Role;
    use tokio_tungstenite::WebSocketStream;

    use super::*;
    use crate::close::CloseCode;
    use crate::pipe::tungstenite::Tungstenite;

    async fn sockets() -> ((SocketTx, impl PipeRx), (SocketTx, impl PipeRx)) {
        let (a, b) = duplex(1 << 16);
        let gateway = WebSocketStream::from_raw_socket(a, Role::Server, None).await;
        let member = WebSocketStream::from_raw_socket(b, Role::Client, None).await;
        (
            split::<Tungstenite, _, _>(gateway),
            split::<Tungstenite, _, _>(member),
        )
    }

    /// Frames cross a real WebSocket in order both ways, byte for byte; a
    /// close arrives with its code and reason; once everything is flushed
    /// nothing counts as unsent.
    #[tokio::test]
    async fn frames_and_a_close_cross_a_websocket() {
        let ((mut gtx, mut grx), (mut mtx, mut mrx)) = sockets().await;
        let big = vec![7u8; 300_000];
        mtx.send(Message::Frame(vec![1, 2, 3])).await.unwrap();
        mtx.send(Message::Frame(big.clone())).await.unwrap();
        assert_eq!(grx.recv().await, Some(Message::Frame(vec![1, 2, 3])));
        assert_eq!(grx.recv().await, Some(Message::Frame(big)));
        assert_eq!(mtx.unsent_bytes(), 0);

        gtx.send(Message::Frame(vec![9])).await.unwrap();
        let close = Close::new(CloseCode::Replaced, "a newer session");
        gtx.send(Message::Close(close.clone())).await.unwrap();
        assert_eq!(mrx.recv().await, Some(Message::Frame(vec![9])));
        assert_eq!(mrx.recv().await, Some(Message::Close(close)));
    }

    /// A peer whose halves are dropped closes the socket without a status, and
    /// the reader then ends.
    #[tokio::test]
    async fn a_dropped_peer_ends_the_pipe() {
        let ((gtx, grx), (_mtx, mut mrx)) = sockets().await;
        drop((gtx, grx));
        assert_eq!(
            mrx.recv().await,
            Some(Message::Close(Close {
                code: 1005,
                reason: String::new()
            }))
        );
        assert_eq!(mrx.recv().await, None);
    }
}
