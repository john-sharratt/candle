//! A tunnelled WebSocket to a member (§5.6): bytes spliced both ways, as the
//! proxy splices any upgrade — and when the member's side fails rather than
//! ends, the browser is closed `1011` instead of being cut off mid-stream.
//!
//! A close frame can only be written between frames, so the member's bytes
//! are read with the frame headers counted ([`Boundary`]): at a boundary the
//! browser gets the close; inside a frame it gets the connection's end, which
//! it reads as an abnormal close (`1006`) — never a frame spliced into the
//! middle of another.

use std::sync::Arc;

use cluster::clock::now_ms;
use cluster::gateway::{Failure, Hub};
use tokio::io::{split, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// A close frame from the server's side, unmasked: `1011`, no reason.
pub const CLOSE_1011: [u8; 4] = [0x88, 0x02, 0x03, 0xF3];

/// Where a byte stream of WebSocket frames stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Boundary {
    /// The current frame's header bytes so far.
    header: Vec<u8>,
    /// Payload bytes still to come in the current frame.
    payload: u64,
}

impl Default for Boundary {
    fn default() -> Self {
        Self::new()
    }
}

impl Boundary {
    pub fn new() -> Self {
        Self {
            header: Vec::new(),
            payload: 0,
        }
    }

    /// Whether the bytes fed so far end exactly between two frames.
    pub fn at_boundary(&self) -> bool {
        self.header.is_empty() && self.payload == 0
    }

    /// The header's full length once its first two bytes are known.
    fn header_len(&self) -> Option<usize> {
        let b1 = *self.header.get(1)?;
        let len = match b1 & 0x7F {
            126 => 2,
            127 => 8,
            _ => 0,
        };
        let mask = if b1 & 0x80 != 0 { 4 } else { 0 };
        Some(2 + len + mask)
    }

    fn payload_len(&self) -> u64 {
        let b1 = self.header[1];
        match b1 & 0x7F {
            126 => u16::from_be_bytes([self.header[2], self.header[3]]) as u64,
            127 => u64::from_be_bytes(self.header[2..10].try_into().expect("8 bytes")),
            n => n as u64,
        }
    }

    pub fn feed(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            if self.payload > 0 {
                let n = (self.payload as usize).min(bytes.len());
                self.payload -= n as u64;
                bytes = &bytes[n..];
                continue;
            }
            self.header.push(bytes[0]);
            bytes = &bytes[1..];
            if self.header_len() == Some(self.header.len()) {
                self.payload = self.payload_len();
                self.header.clear();
            }
        }
    }
}

/// Splice `client` and `member` until either ends. A failure on the member's
/// side closes the client `1011` when it can, and is counted.
pub async fn splice<C, M>(client: C, member: M, hub: Arc<Hub>)
where
    C: AsyncRead + AsyncWrite + Send + 'static,
    M: AsyncRead + AsyncWrite + Send + 'static,
{
    let (mut client_rx, mut client_tx) = split(client);
    let (mut member_rx, mut member_tx) = split(member);
    let up = async {
        let mut buf = vec![0u8; 16 << 10];
        loop {
            match client_rx.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if member_tx.write_all(&buf[..n]).await.is_err() {
                        break;
                    }
                }
            }
        }
        let _ = member_tx.shutdown().await;
    };
    let down = async {
        let mut buf = vec![0u8; 16 << 10];
        let mut boundary = Boundary::new();
        loop {
            match member_rx.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    boundary.feed(&buf[..n]);
                    if client_tx.write_all(&buf[..n]).await.is_err() {
                        return;
                    }
                }
                Err(e) => {
                    tracing::debug!(error = %e, "the member's side of a tunnel failed");
                    hub.traffic.failed(Failure::Tunnel, now_ms());
                    if boundary.at_boundary() {
                        let _ = client_tx.write_all(&CLOSE_1011).await;
                    }
                    break;
                }
            }
        }
        let _ = client_tx.shutdown().await;
    };
    tokio::join!(up, down);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames of each length form — 7-bit, 16-bit, 64-bit, and masked — are
    /// counted across any split of their bytes.
    #[test]
    fn frame_boundaries_are_found_across_any_split() {
        let mut stream = Vec::new();
        stream.extend([0x81, 0x03]);
        stream.extend(b"abc");
        stream.extend([0x82, 126, 0x01, 0x00]);
        stream.extend(vec![7u8; 256]);
        stream.extend([0x82, 127, 0, 0, 0, 0, 0, 0, 0x00, 0x02]);
        stream.extend([9u8; 2]);
        stream.extend([0x81, 0x82, 1, 2, 3, 4]);
        stream.extend([5u8; 2]);
        let ends = [5, 5 + 4 + 256, 5 + 260 + 12, 5 + 260 + 12 + 8];
        for chunk in [1, 2, 3, 7, 64, stream.len()] {
            let mut b = Boundary::new();
            let mut fed = 0;
            for piece in stream.chunks(chunk) {
                b.feed(piece);
                fed += piece.len();
                assert_eq!(
                    b.at_boundary(),
                    ends.contains(&fed),
                    "chunk {chunk}, at {fed}"
                );
            }
        }
    }

    /// The close frame is `1011`, unmasked, from the server.
    #[test]
    fn the_close_is_1011() {
        assert_eq!(u16::from_be_bytes([CLOSE_1011[2], CLOSE_1011[3]]), 1011);
        let mut b = Boundary::new();
        b.feed(&CLOSE_1011);
        assert!(b.at_boundary());
    }
}
