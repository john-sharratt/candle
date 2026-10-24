//! What a session has heard from its peer, two ways: the transport's last
//! read (`bytes`, when the transport can tell) and the last whole message
//! (`whole`). A WebSocket's messages do not interleave, so bytes newer than the
//! last whole message are a message part-way across — a reply still arriving
//! over a slow link — while an idle socket's `PING`s complete as whole messages
//! and leave none in progress.

use tokio::time::Instant;

use crate::pipe::Heard;

#[derive(Debug, Clone, Default)]
pub struct Hearing {
    bytes: Option<Heard>,
    whole: Heard,
}

impl Hearing {
    /// Hearing a transport whose reads stamp `bytes`, when it can.
    pub fn new(bytes: Option<Heard>) -> Self {
        Self {
            bytes,
            whole: Heard::new(),
        }
    }

    /// A whole message arrived.
    pub fn message(&self) {
        self.whole.stamp();
    }

    /// When the peer was last heard at all: a whole message, or a byte.
    pub fn last(&self) -> Option<Instant> {
        let bytes = self.bytes.as_ref().and_then(Heard::at);
        bytes.max(self.whole.at())
    }

    /// When a byte last arrived of a message not yet whole; `None` when no
    /// message is part-way across.
    pub fn mid_message(&self) -> Option<Instant> {
        let bytes = self.bytes.as_ref().and_then(Heard::at)?;
        match self.whole.at() {
            Some(whole) if whole >= bytes => None,
            _ => Some(bytes),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// Bytes after the last whole message are a message in progress; a whole
    /// message arriving ends it; a transport that cannot stamp its reads never
    /// has one.
    #[tokio::test(start_paused = true)]
    async fn bytes_after_the_last_whole_message_are_a_message_in_progress() {
        let bytes = Heard::new();
        let h = Hearing::new(Some(bytes.clone()));
        assert_eq!((h.last(), h.mid_message()), (None, None));

        // A ping: its bytes, then it is whole.
        bytes.stamp();
        h.message();
        assert_eq!(h.mid_message(), None, "an idle socket's ping is whole");
        assert_eq!(h.last(), Some(Instant::now()));

        // A large reply starts arriving.
        tokio::time::sleep(Duration::from_secs(20)).await;
        bytes.stamp();
        assert_eq!(h.mid_message(), Some(Instant::now()));
        assert_eq!(h.last(), Some(Instant::now()));

        // It completes.
        tokio::time::sleep(Duration::from_secs(1)).await;
        bytes.stamp();
        h.message();
        assert_eq!(h.mid_message(), None);

        let blind = Hearing::new(None);
        blind.message();
        assert_eq!(blind.mid_message(), None);
        assert!(blind.last().is_some());
    }
}
