//! A transport stream that stamps every read: below TLS and the WebSocket, so
//! the bytes of a message still arriving count as hearing from the peer
//! ([`Heard`]). A member's dialled socket and a gateway's accepted one both run
//! over it.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

use super::Heard;

pub struct Stamped<T> {
    inner: T,
    heard: Heard,
}

impl<T> Stamped<T> {
    pub fn new(inner: T, heard: Heard) -> Self {
        Self { inner, heard }
    }
}

impl<T: AsyncRead + Unpin> AsyncRead for Stamped<T> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let before = buf.filled().len();
        let polled = Pin::new(&mut self.inner).poll_read(cx, buf);
        if matches!(polled, Poll::Ready(Ok(()))) && buf.filled().len() > before {
            self.heard.stamp();
        }
        polled
    }
}

impl<T: AsyncWrite + Unpin> AsyncWrite for Stamped<T> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::{duplex, AsyncReadExt, AsyncWriteExt};

    use super::*;

    /// Each read that brings bytes stamps; writing does not.
    #[tokio::test(start_paused = true)]
    async fn a_read_with_bytes_stamps_and_a_write_does_not() {
        let (a, mut b) = duplex(64);
        let heard = Heard::new();
        let mut s = Stamped::new(a, heard.clone());
        s.write_all(b"out").await.unwrap();
        assert_eq!(heard.at(), None);
        b.write_all(b"in").await.unwrap();
        let mut got = [0u8; 2];
        s.read_exact(&mut got).await.unwrap();
        assert_eq!(&got, b"in");
        assert!(heard.at().is_some());
    }
}
