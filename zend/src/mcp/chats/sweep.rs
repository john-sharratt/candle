//! The sweep that disposes of the chats nobody is talking on any more
//! (`docs/zen_chat.md` §3): every member, every [`SWEEP_EVERY`].
//!
//! A chat is named by its id alone, never by the MCP session it was opened on:
//! claude.ai keeps one session across every conversation a person has through
//! it, so a session's end says nothing about a chat's. A chat is due when it is
//! closing, or when nothing has been said on it for [`CHAT_IDLE_MS`].

use std::sync::{Arc, Weak};
use std::time::Duration;

use tokio::runtime::Handle;

use crate::mcp::records::Records;
use crate::mcp::workers::now_ms;
use crate::session::ZendSession;

use super::dispose::{dispose, tombstone_with, Disposal, Tombstone, When};
use super::record::{every_chat, Chat};

/// How long a chat may go with nothing said on it before it is disposed of.
pub const CHAT_IDLE_MS: u64 = 12 * 60 * 60 * 1000;

/// How often the sweep runs.
pub const SWEEP_EVERY: Duration = Duration::from_secs(10 * 60);

/// Why a chat is disposed of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Due {
    /// Claude closed it while Zen was answering, or an earlier sweep found it
    /// due then.
    Closing,
    /// Nothing has been said on it for [`CHAT_IDLE_MS`].
    Idle,
}

/// Whether `chat` is due at `now`.
pub fn due(chat: &Chat, now: u64) -> Option<Due> {
    if chat.closing {
        return Some(Due::Closing);
    }
    (now.saturating_sub(chat.last_ms) > CHAT_IDLE_MS).then_some(Due::Idle)
}

/// One pass at `now`: every due chat disposed of, or closing while Zen
/// answers. How many were disposed of, and how many are left closing.
pub fn pass(records: &dyn Records, tombstone: Tombstone, now: u64) -> (usize, usize) {
    let (mut disposed, mut closing) = (0, 0);
    for (owner, chat) in every_chat(records) {
        let Some(why) = due(&chat, now) else {
            continue;
        };
        // Judged again under the lock a send takes: a chat spoken on since
        // this read is not disposed of under it.
        let still_due = |c: &Chat| due(c, now).is_some();
        match dispose(
            records,
            tombstone,
            &owner,
            &chat.id,
            now,
            When::If(&still_due),
        ) {
            Ok(Disposal::Disposed) => {
                tracing::info!(chat = %chat.id, why = ?why, "mcp: a chat was disposed of");
                disposed += 1;
            }
            Ok(Disposal::Answering) => closing += 1,
            Ok(Disposal::Gone | Disposal::NotDue) => {}
            Err(e) => {
                tracing::warn!(chat = %chat.id, "mcp: a chat could not be disposed of: {e}");
                closing += 1;
            }
        }
    }
    (disposed, closing)
}

/// Run the sweep every [`SWEEP_EVERY`] for as long as the session lives.
async fn sweeping(session: Weak<ZendSession>) {
    let mut every = tokio::time::interval(SWEEP_EVERY);
    loop {
        every.tick().await;
        let Some(session) = session.upgrade() else {
            return;
        };
        if !session.is_loaded() {
            continue;
        }
        // Disposing writes through the persistence owner, whose flow control
        // can wait for room: off the runtime's threads.
        let swept = tokio::task::spawn_blocking(move || {
            let records = session.shared_substrate()?;
            Some(pass(&records, &tombstone_with(&session), now_ms()))
        })
        .await;
        match swept {
            Ok(Some((0, 0))) | Ok(None) => {}
            Ok(Some((disposed, closing))) => {
                tracing::info!(disposed, closing, "mcp: swept the chats")
            }
            Err(e) => tracing::warn!("mcp: sweeping the chats failed: {e}"),
        }
    }
}

/// Start [`sweeping`] on `runtime` for `session`.
pub fn start_sweeping(session: &Arc<ZendSession>, runtime: &Handle) {
    runtime.spawn(sweeping(Arc::downgrade(session)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;

    use super::super::record::put_chat;

    fn chat(id: &str, last_ms: u64, closing: bool) -> Chat {
        Chat {
            id: id.into(),
            opened_ms: 0,
            last_ms,
            closing,
        }
    }

    /// **A chat is due when closing or idle twelve hours — and not
    /// otherwise.**
    #[test]
    fn a_chat_is_due_only_when_closing_or_idle() {
        let id = "0123456789abcdef";
        assert_eq!(due(&chat(id, 0, false), CHAT_IDLE_MS), None);
        assert_eq!(due(&chat(id, 0, false), CHAT_IDLE_MS + 1), Some(Due::Idle));
        assert_eq!(due(&chat(id, 0, true), 1), Some(Due::Closing));
        assert_eq!(
            due(&chat(id, CHAT_IDLE_MS, false), CHAT_IDLE_MS + 1),
            None,
            "something was said lately"
        );
    }

    /// **A pass disposes of the idle chat and keeps the one spoken on
    /// lately.**
    #[test]
    fn a_pass_disposes_of_the_idle_chat() {
        let r = MemRecords::default();
        let now = CHAT_IDLE_MS + 10;
        put_chat(&r, "o", &chat("aaaaaaaaaaaaaaaa", now - 5, false)).unwrap();
        put_chat(&r, "o", &chat("bbbbbbbbbbbbbbbb", 0, false)).unwrap();
        assert_eq!(pass(&r, &|_: &str| Ok(()), now), (1, 0));
        let left: Vec<String> = every_chat(&r).into_iter().map(|(_, c)| c.id).collect();
        assert_eq!(left, ["aaaaaaaaaaaaaaaa"]);
    }
}
