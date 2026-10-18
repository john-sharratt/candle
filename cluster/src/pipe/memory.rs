//! An in-memory message pipe with a byte capacity per direction, standing in
//! for a WebSocket in tests and harnesses. A full direction makes its sender
//! wait, as a socket's send buffer does.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;

use super::{Drain, Message, PipeClosed, PipeRx, PipeTx};

/// One direction of the pipe.
struct Direction {
    queue: Mutex<VecDeque<Message>>,
    /// The bytes queued and not yet received.
    bytes: Drain,
    cap: usize,
    /// The receiving half has something to read, or the pipe closed.
    readable: Notify,
    /// The sending half has room, or the pipe closed.
    writable: Notify,
    closed: AtomicBool,
}

impl Direction {
    fn new(cap: usize) -> Arc<Self> {
        Arc::new(Self {
            queue: Mutex::new(VecDeque::new()),
            bytes: Drain::new(),
            cap,
            readable: Notify::new(),
            writable: Notify::new(),
            closed: AtomicBool::new(false),
        })
    }

    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.readable.notify_waiters();
        self.readable.notify_one();
        self.writable.notify_waiters();
        self.writable.notify_one();
    }
}

/// The sending half of one direction.
pub struct MemoryTx {
    dir: Arc<Direction>,
}

/// The receiving half of one direction.
pub struct MemoryRx {
    dir: Arc<Direction>,
}

/// One end of the pipe: what it sends and what it receives.
pub struct End {
    pub tx: MemoryTx,
    pub rx: MemoryRx,
}

/// Two connected ends; each direction holds at most `cap` bytes in flight (a
/// single larger message is still accepted when the direction is empty).
pub fn pair(cap: usize) -> (End, End) {
    let a_to_b = Direction::new(cap);
    let b_to_a = Direction::new(cap);
    (
        End {
            tx: MemoryTx {
                dir: a_to_b.clone(),
            },
            rx: MemoryRx {
                dir: b_to_a.clone(),
            },
        },
        End {
            tx: MemoryTx { dir: b_to_a },
            rx: MemoryRx { dir: a_to_b },
        },
    )
}

impl PipeTx for MemoryTx {
    async fn send(&mut self, message: Message) -> Result<(), PipeClosed> {
        let len = message.len();
        loop {
            let notified = self.dir.writable.notified();
            if self.dir.closed.load(Ordering::SeqCst) {
                return Err(PipeClosed);
            }
            let held = self.dir.bytes.unsent();
            if held == 0 || held + len <= self.dir.cap {
                let close = matches!(message, Message::Close(_));
                self.dir.bytes.add(len);
                self.dir.queue.lock().unwrap().push_back(message);
                self.dir.readable.notify_one();
                if close {
                    // A close is the last message: nothing more is sent, but
                    // the peer still reads everything queued before it.
                    self.dir.closed.store(true, Ordering::SeqCst);
                    self.dir.writable.notify_waiters();
                }
                return Ok(());
            }
            notified.await;
        }
    }

    fn drain(&self) -> &Drain {
        &self.dir.bytes
    }
}

impl Drop for MemoryTx {
    fn drop(&mut self) {
        self.dir.close();
    }
}

impl PipeRx for MemoryRx {
    async fn recv(&mut self) -> Option<Message> {
        loop {
            let notified = self.dir.readable.notified();
            if let Some(m) = self.dir.queue.lock().unwrap().pop_front() {
                self.dir.bytes.sub(m.len());
                self.dir.writable.notify_one();
                return Some(m);
            }
            if self.dir.closed.load(Ordering::SeqCst) {
                return None;
            }
            notified.await;
        }
    }
}

impl Drop for MemoryRx {
    fn drop(&mut self) {
        self.dir.close();
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;

    use super::*;
    use crate::close::{Close, CloseCode};

    #[tokio::test]
    async fn messages_arrive_in_order_and_a_close_ends_the_pipe() {
        let (mut a, mut b) = pair(1 << 20);
        a.tx.send(Message::Frame(vec![1])).await.unwrap();
        a.tx.send(Message::Frame(vec![2, 3])).await.unwrap();
        assert_eq!(a.tx.unsent_bytes(), 3);
        let close = Close::new(CloseCode::GoingAway, "bye");
        a.tx.send(Message::Close(close.clone())).await.unwrap();
        assert_eq!(b.rx.recv().await, Some(Message::Frame(vec![1])));
        assert_eq!(b.rx.recv().await, Some(Message::Frame(vec![2, 3])));
        assert_eq!(b.rx.recv().await, Some(Message::Close(close)));
        assert_eq!(b.rx.recv().await, None);
        assert_eq!(a.tx.send(Message::Frame(vec![4])).await, Err(PipeClosed));
        assert_eq!(a.tx.unsent_bytes(), 0);
    }

    #[tokio::test]
    async fn a_full_direction_makes_the_sender_wait() {
        let (mut a, mut b) = pair(4);
        a.tx.send(Message::Frame(vec![0; 3])).await.unwrap();
        let mut send = Box::pin(a.tx.send(Message::Frame(vec![0; 3])));
        assert!(futures_poll_once(&mut send).await.is_none());
        b.rx.recv().await.unwrap();
        send.await.unwrap();
        assert_eq!(b.rx.recv().await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn dropping_one_end_ends_the_other() {
        let (a, mut b) = pair(16);
        drop(a);
        assert_eq!(b.rx.recv().await, None);
        assert_eq!(b.tx.send(Message::Frame(vec![1])).await, Err(PipeClosed));
    }

    /// Poll a future once, returning its output if it was ready.
    async fn futures_poll_once<F: Future + Unpin>(f: &mut F) -> Option<F::Output> {
        tokio::select! {
            biased;
            out = f => Some(out),
            _ = std::future::ready(()) => None,
        }
    }
}
