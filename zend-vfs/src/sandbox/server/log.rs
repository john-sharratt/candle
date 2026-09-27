//! A job's log: the file its command's output goes to, and that file read
//! back as a stream while it is written.
//!
//! The writer publishes, after every write, how much the file holds and
//! whether the job is done writing. A write is on the file before it is
//! published, so a reader that has seen a count can read that much. The
//! stream reads what is there, then waits on the count, and ends once the job
//! is done and it has read everything — however late it started.

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::stream::{self, Stream};
use tokio::fs::File as AsyncFile;
use tokio::io::{AsyncReadExt, AsyncWrite};
use tokio::sync::watch;

/// How much a log holds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Written {
    pub bytes: u64,
    /// Every `\n` written.
    pub lines: u64,
    /// Whether the job is done writing.
    pub closed: bool,
}

/// A job's output as it is written: each item the next bytes of the log.
pub type OutputStream = Pin<Box<dyn Stream<Item = io::Result<Vec<u8>>> + Send>>;

/// The writing end of a log.
pub(super) struct LogWriter {
    file: File,
    written: watch::Sender<Written>,
}

impl LogWriter {
    /// A new, empty log at `path`, and what it holds as it is written.
    pub(super) fn create(path: &Path) -> io::Result<(Self, watch::Receiver<Written>)> {
        let file = File::create(path)?;
        let (written, watching) = watch::channel(Written::default());
        Ok((Self { file, written }, watching))
    }
}

impl Drop for LogWriter {
    /// However the job ended, nothing more is written: readers finish.
    fn drop(&mut self) {
        self.written.send_modify(|w| w.closed = true);
    }
}

impl AsyncWrite for LogWriter {
    /// Written straight to the file: a chunk of output is a page-cache copy,
    /// and a reader woken by the count finds the bytes already there.
    fn poll_write(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let n = self.file.write(buf)?;
        let lines = buf[..n].iter().filter(|&&b| b == b'\n').count() as u64;
        self.written.send_modify(|w| {
            w.bytes += n as u64;
            w.lines += lines;
        });
        Poll::Ready(Ok(n))
    }

    fn poll_flush(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(self.file.flush())
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.poll_flush(cx)
    }
}

/// The log at `path` as a stream, from its first byte, following `written`.
pub(super) fn tail(path: &Path, written: watch::Receiver<Written>) -> io::Result<OutputStream> {
    let file = AsyncFile::from_std(File::open(path)?);
    let reading = stream::unfold(Some((file, written)), |state| async move {
        let (mut file, mut written) = state?;
        loop {
            // What the log holds is taken before reading it: a write after
            // this is waited for below, never missed. A writer gone is a log
            // closed.
            let closed = written.borrow_and_update().closed || written.has_changed().is_err();
            let mut buf = vec![0u8; 64 * 1024];
            match file.read(&mut buf).await {
                Err(e) => return Some((Err(e), None)),
                Ok(0) if closed => return None,
                Ok(0) => {
                    let _ = written.changed().await;
                }
                Ok(n) => {
                    buf.truncate(n);
                    return Some((Ok(buf), Some((file, written))));
                }
            }
        }
    });
    Ok(Box::pin(reading))
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    use super::*;

    async fn collect(mut stream: OutputStream) -> Vec<u8> {
        let mut all = Vec::new();
        while let Some(chunk) = stream.next().await {
            all.extend(chunk.unwrap());
        }
        all
    }

    /// **The count follows every write**, lines and bytes, and dropping the
    /// writer marks the log closed.
    #[tokio::test]
    async fn the_count_follows_every_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.log");
        let (mut log, written) = LogWriter::create(&path).unwrap();
        log.write_all(b"one\ntwo").await.unwrap();
        assert_eq!(
            *written.borrow(),
            Written {
                bytes: 7,
                lines: 1,
                closed: false
            }
        );
        log.write_all(b"\nthree\n").await.unwrap();
        drop(log);
        assert_eq!(
            *written.borrow(),
            Written {
                bytes: 14,
                lines: 3,
                closed: true
            }
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"one\ntwo\nthree\n");
    }

    /// **A stream started before, during and after the writing all read the
    /// whole log**, and each ends once the writer is done.
    #[tokio::test]
    async fn a_stream_reads_everything_whenever_it_starts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.log");
        let (mut log, written) = LogWriter::create(&path).unwrap();
        let early = tokio::spawn(collect(tail(&path, written.clone()).unwrap()));
        log.write_all(b"first\n").await.unwrap();
        tokio::task::yield_now().await;
        let middle = tokio::spawn(collect(tail(&path, written.clone()).unwrap()));
        for n in 0..50 {
            log.write_all(format!("line {n}\n").as_bytes())
                .await
                .unwrap();
            tokio::task::yield_now().await;
        }
        drop(log);
        let late = collect(tail(&path, written).unwrap()).await;

        let whole = std::fs::read(&path).unwrap();
        assert_eq!(early.await.unwrap(), whole);
        assert_eq!(middle.await.unwrap(), whole);
        assert_eq!(late, whole);
        assert!(whole.starts_with(b"first\nline 0\n") && whole.ends_with(b"line 49\n"));
    }

    /// An empty log that is closed is an empty stream, not one that waits.
    #[tokio::test]
    async fn a_closed_empty_log_ends_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.log");
        let (log, written) = LogWriter::create(&path).unwrap();
        drop(log);
        assert!(collect(tail(&path, written).unwrap()).await.is_empty());
    }
}
