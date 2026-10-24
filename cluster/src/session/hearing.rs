//! What a session has heard from its peer: the transport's last read (`bytes`,
//! when the transport can tell), the last whole message (`whole`), and the
//! last catch-up frame (`catchup`). A WebSocket's messages do not interleave,
//! so bytes newer than the last whole message are a message part-way across —
//! a reply still arriving over a slow link — while an idle socket's `PING`s
//! complete as whole messages and leave none in progress. A fetched record of
//! many fragments arrives as many whole messages, each a catch-up frame: a
//! catch-up's answer is still arriving while they do.

use tokio::time::Instant;

use crate::pipe::Heard;

#[derive(Debug, Clone, Default)]
pub struct Hearing {
    bytes: Option<Heard>,
    whole: Heard,
    catchup: Heard,
}

impl Hearing {
    /// Hearing a transport whose reads stamp `bytes`, when it can.
    pub fn new(bytes: Option<Heard>) -> Self {
        Self {
            bytes,
            whole: Heard::new(),
            catchup: Heard::new(),
        }
    }

    /// A whole message arrived.
    pub fn message(&self) {
        self.whole.stamp();
    }

    /// A catch-up frame arrived: a `HEADERS`, a `DONE`, or a fragment of a
    /// record on the catch-up lane.
    pub fn catchup(&self) {
        self.catchup.stamp();
    }

    /// When a catch-up's answer was last heard of: a catch-up frame, or a
    /// message part-way across that may be one.
    pub fn answer(&self) -> Option<Instant> {
        self.catchup.at().max(self.mid_message())
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

    /// **A record of many fragments is an answer arriving.** AlienWorld's
    /// planned admission was dropped two minutes in while BlackWorld's reply
    /// streamed at 137 KB/s: each 1 MiB fragment is a whole message, so between
    /// them none was part-way across, and the catch-up heard nothing of its own
    /// until a whole record landed. Each catch-up frame is heard as its answer.
    #[tokio::test(start_paused = true)]
    async fn catch_up_frames_are_an_answer_arriving_between_whole_messages() {
        let bytes = Heard::new();
        let h = Hearing::new(Some(bytes.clone()));
        assert_eq!(h.answer(), None);
        // A fragment: whole, and on the catch-up lane.
        let fragment = Instant::now();
        bytes.stamp();
        h.message();
        h.catchup();
        assert_eq!(h.mid_message(), None);
        assert_eq!(h.answer(), Some(fragment));
        // Something else part-way across: an answer may be arriving.
        tokio::time::sleep(Duration::from_secs(5)).await;
        bytes.stamp();
        assert_eq!(h.answer(), Some(Instant::now()));
        // It was a ping, whole, not a catch-up frame: the answer was last
        // heard at the fragment.
        tokio::time::sleep(Duration::from_secs(5)).await;
        bytes.stamp();
        h.message();
        assert_eq!(h.answer(), Some(fragment));
    }
}
