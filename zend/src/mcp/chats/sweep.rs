//! The sweep that disposes of the chats whose session is done
//! (`docs/zen_chat.md` §3): every member, every [`SWEEP_EVERY`].
//!
//! A client is not obliged to say its session has ended, and Claude Code
//! through claude.ai's connectors never does, so most chats end by idleness. A
//! chat is due when it is closing, when its session's record is gone or holds
//! another chat, or when nothing has been said on it for [`CHAT_IDLE_MS`].

use std::sync::{Arc, Weak};
use std::time::Duration;

use tokio::runtime::Handle;

use crate::mcp::records::Records;
use crate::mcp::state::{held_by, CHAT};
use crate::mcp::workers::now_ms;
use crate::session::ZendSession;

use super::dispose::{dispose, tombstone_with, Disposal, Tombstone, When};
use super::record::{every_chat, Chat};

/// How long a chat may go with nothing said on it before it is disposed of.
pub const CHAT_IDLE_MS: u64 = 12 * 60 * 60 * 1000;

/// How long after its opening a chat whose session record is not found is
/// still taken for one whose record is on its way to this member.
pub const SESSION_GRACE_MS: u64 = 10 * 60 * 1000;

/// How often the sweep runs.
pub const SWEEP_EVERY: Duration = Duration::from_secs(10 * 60);

/// Why a chat is disposed of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Due {
    /// Claude closed it, or an earlier sweep found it due.
    Closing,
    /// Its session's record is gone, or holds another chat.
    SessionGone,
    /// Nothing has been said on it for [`CHAT_IDLE_MS`].
    Idle,
}

/// Whether `chat` is due at `now`, its session's record holding `held` under
/// [`CHAT`].
pub fn due(chat: &Chat, held: Option<&str>, now: u64) -> Option<Due> {
    if chat.closing {
        return Some(Due::Closing);
    }
    if held != Some(chat.id.as_str()) && now.saturating_sub(chat.opened_ms) > SESSION_GRACE_MS {
        return Some(Due::SessionGone);
    }
    (now.saturating_sub(chat.last_ms) > CHAT_IDLE_MS).then_some(Due::Idle)
}

/// One pass at `now`: every due chat disposed of, or closing while Zen
/// answers. How many were disposed of, and how many are left closing.
pub fn pass(records: &dyn Records, tombstone: Tombstone, now: u64) -> (usize, usize) {
    let (mut disposed, mut closing) = (0, 0);
    for (owner, chat) in every_chat(records) {
        let due_now = |c: &Chat| due(c, held_by(records, &c.session, CHAT).as_deref(), now);
        let Some(why) = due_now(&chat) else {
            continue;
        };
        // Judged again under the lock a send takes: a chat spoken on since
        // this read is not disposed of under it.
        let still_due = |c: &Chat| due_now(c).is_some();
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
    use mcp_server::{Client, Store};

    use super::*;
    use crate::mcp::records::MemRecords;
    use crate::mcp::state::RecordStore;

    use super::super::record::put_chat;

    fn chat(opened_ms: u64, last_ms: u64, closing: bool) -> Chat {
        Chat {
            id: "0123456789abcdef".into(),
            session: "s1".into(),
            opened_ms,
            last_ms,
            closing,
        }
    }

    /// **A chat is due when closing, idle twelve hours, or its session gone
    /// past the grace — and not otherwise.**
    #[test]
    fn a_chat_is_due_only_when_its_session_is_done() {
        let id = Some("0123456789abcdef");
        assert_eq!(due(&chat(0, 0, false), id, CHAT_IDLE_MS), None);
        assert_eq!(
            due(&chat(0, 0, false), id, CHAT_IDLE_MS + 1),
            Some(Due::Idle)
        );
        assert_eq!(due(&chat(0, 0, true), id, 1), Some(Due::Closing));
        assert_eq!(
            due(&chat(0, 0, false), None, SESSION_GRACE_MS),
            None,
            "its session's record may be on its way"
        );
        assert_eq!(
            due(&chat(0, 0, false), None, SESSION_GRACE_MS + 1),
            Some(Due::SessionGone)
        );
        assert_eq!(
            due(
                &chat(0, 0, false),
                Some("fedcba9876543210"),
                SESSION_GRACE_MS + 1
            ),
            Some(Due::SessionGone),
            "its session holds another chat"
        );
        assert_eq!(
            due(&chat(0, CHAT_IDLE_MS, false), id, CHAT_IDLE_MS + 1),
            None,
            "something was said lately"
        );
    }

    /// **A pass disposes of the chat whose session ended and keeps the one
    /// whose session holds it.**
    #[test]
    fn a_pass_disposes_of_the_chat_whose_session_ended() {
        let r = MemRecords::default();
        let sessions = RecordStore::new(MemRecords::default(), Default::default());
        let live = sessions.open("o", Client::Other, "2025-11-25", 0).unwrap();
        let s = sessions.get(&live, 0).unwrap();
        sessions
            .hold(&live, &s, CHAT, "aaaaaaaaaaaaaaaa", 0)
            .unwrap();
        for (key, value) in sessions.records.under("mcp/") {
            r.put(&key, value).unwrap();
        }
        let kept = Chat {
            id: "aaaaaaaaaaaaaaaa".into(),
            session: live,
            ..chat(0, 0, false)
        };
        let ended = Chat {
            id: "bbbbbbbbbbbbbbbb".into(),
            session: "closed-session".into(),
            ..chat(0, 0, false)
        };
        put_chat(&r, "o", &kept).unwrap();
        put_chat(&r, "o", &ended).unwrap();
        let now = SESSION_GRACE_MS + 1;
        assert_eq!(pass(&r, &|_: &str| Ok(()), now), (1, 0));
        let left: Vec<String> = every_chat(&r).into_iter().map(|(_, c)| c.id).collect();
        assert_eq!(left, ["aaaaaaaaaaaaaaaa"]);
    }
}
