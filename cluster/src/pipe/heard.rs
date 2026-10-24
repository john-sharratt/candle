//! When bytes last arrived from the peer, partial messages included
//! (`docs/replicated_substrate.md` §3.2, keep-alive). A message of a mebibyte
//! over a link of a few kilobytes a second takes minutes to arrive whole, and a
//! peer still sending it is alive: the transport stamps every read, and the
//! session's reader counts it as hearing from the peer.

use std::sync::{Arc, Mutex};

use tokio::time::Instant;

/// The time of the last read, shared between the transport that stamps it and
/// the pipe that reports it.
#[derive(Debug, Clone, Default)]
pub struct Heard(Arc<Mutex<Option<Instant>>>);

impl Heard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes arrived now.
    pub fn stamp(&self) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
    }

    /// When bytes last arrived, if ever.
    pub fn at(&self) -> Option<Instant> {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test(start_paused = true)]
    async fn a_stamp_is_seen_by_every_clone() {
        let heard = Heard::new();
        let reader = heard.clone();
        assert_eq!(reader.at(), None);
        tokio::time::sleep(Duration::from_secs(5)).await;
        heard.stamp();
        assert_eq!(reader.at(), Some(Instant::now()));
    }
}
