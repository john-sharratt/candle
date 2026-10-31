//! The user's context being read for a conversation that has not had it yet
//! (`docs/mcp_user_context.md` §2.5): a first call reads it beside its own
//! work and never waits for it. A read its call answered ahead of is left
//! running here, and the conversation's next call — a first call again, since
//! the conversation has not had its context — takes it up instead of reading
//! again.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;

/// How many reads are left running before the oldest are let go.
const KEPT: usize = 4096;

/// How long a read left running is the conversation's: a call that comes
/// later reads afresh, since what the memory holds has moved on and the
/// conversation it was read for may have ended.
pub const FRESH: Duration = Duration::from_secs(120);

/// A read of the context: the composed context, or `None` when it could not
/// be read.
pub type Read = JoinHandle<Option<String>>;

/// The reads left running, by conversation, each with when it was left.
#[derive(Default)]
pub struct Readings(Mutex<HashMap<String, (Read, Instant)>>);

impl Readings {
    /// The read `who`'s conversation left running within [`FRESH`] of `now`;
    /// an older one is let go.
    pub fn take(&self, who: &str, now: Instant) -> Option<Read> {
        let (read, left) = self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(who)?;
        if now.saturating_duration_since(left) > FRESH {
            read.abort();
            return None;
        }
        Some(read)
    }

    /// Leave `read` running for `who`'s conversation's next call, from `now`.
    /// Past [`KEPT`], every read already finished is let go first: its
    /// conversation's next call reads again.
    pub fn leave(&self, who: &str, read: Read, now: Instant) {
        let mut reads = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if reads.len() >= KEPT {
            reads.retain(|_, (r, left)| {
                !r.is_finished() && now.saturating_duration_since(*left) <= FRESH
            });
        }
        if reads.len() < KEPT {
            reads.insert(who.to_string(), (read, now));
        }
    }

    /// Let `who`'s read go: the conversation has its context.
    pub fn forget(&self, who: &str) {
        let left = self.0.lock().unwrap_or_else(|e| e.into_inner()).remove(who);
        if let Some((read, _)) = left {
            read.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A read left running is taken up once, by the conversation's next
    /// call**; another conversation's is its own, one forgotten is gone, and
    /// one older than [`FRESH`] is never handed on.
    #[tokio::test]
    async fn a_read_left_running_is_taken_up_by_the_next_call() {
        let r = Readings::default();
        let now = Instant::now();
        r.leave(
            "a",
            tokio::spawn(async { Some("context".to_string()) }),
            now,
        );
        r.leave("b", tokio::spawn(async { None }), now);
        assert!(r.take("c", now).is_none());
        let read = r.take("a", now).unwrap();
        assert_eq!(read.await.unwrap().as_deref(), Some("context"));
        assert!(r.take("a", now).is_none(), "taken once");
        r.forget("b");
        assert!(r.take("b", now).is_none());
        r.leave("old", tokio::spawn(async { None }), now);
        assert!(
            r.take("old", now + FRESH + Duration::from_secs(1))
                .is_none(),
            "a stale read reads afresh"
        );
    }
}
