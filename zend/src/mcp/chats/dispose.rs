//! Disposing of a chat (`docs/zen_chat.md` §3): every conversation it ran on
//! tombstoned and every record it made ceased.
//!
//! The chat is read, judged and marked closing under the lock a turn begins
//! under ([`workers::holding`]). Once it is closing with nothing answering, no
//! turn can begin on it again — a send opens a new chat instead, and a lost
//! answer is not resumed — so its conversations are tombstoned and its records
//! ceased outside the lock, which no other call waits on meanwhile.
//!
//! A chat Zen is still answering is marked closing and its answer asked to stop
//! at its next tool round; it is disposed of as the answer settles
//! (`super::talk::settled`), or by the next sweep. A tombstone that fails leaves
//! every record in place, the chat closing, for the next sweep.

use crate::mcp::records::Records;
use crate::mcp::threads::{self, Worker};
use crate::mcp::{nudges, reported, stops, workers};
use crate::session::ZendSession;

use super::naming::thread_of;
use super::record::{self, chat_key, Chat};

/// Tombstone one conversation, by its `conv_id`.
pub type Tombstone<'a> = &'a dyn Fn(&str) -> Result<(), String>;

/// What a stopped answer is told.
pub const STOPPED: &str = "the chat was closed, so it is being deleted: end your answer here";

/// How disposing of a chat came out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Disposal {
    /// Every conversation it ran on tombstoned, every record it made ceased.
    Disposed,
    /// Zen is answering: the chat is closing, and the answer was asked to stop.
    Answering,
    /// Nothing of it is kept.
    Gone,
    /// Read again under the lock, it was no longer due: something was said on
    /// it since the sweep looked.
    NotDue,
}

/// When a chat is disposed of: now, or only if — read again under the lock —
/// it is still due, as the sweep found it.
#[derive(Clone, Copy)]
pub enum When<'a> {
    Now,
    If(&'a dyn Fn(&Chat) -> bool),
}

/// The session's own tombstone, as disposal calls it: refused before the model
/// is loaded, so no record is ceased for a conversation that was not
/// tombstoned.
pub fn tombstone_with(session: &ZendSession) -> impl Fn(&str) -> Result<(), String> + '_ {
    |conv_id| match session.tombstone_conversation(conv_id) {
        Some(Ok(())) => Ok(()),
        Some(Err(e)) => Err(e.to_string()),
        None => Err("the model is still loading".into()),
    }
}

/// Dispose of `owner`'s chat `id` at `now`, `when` it should be.
pub fn dispose(
    records: &dyn Records,
    tombstone: Tombstone,
    owner: &str,
    id: &str,
    now: u64,
    when: When,
) -> Result<Disposal, String> {
    let thread = thread_of(id);
    let (chat, workers) = {
        let _held = workers::holding();
        let chat = record::chat(records, owner, id);
        let worker_prefix = threads::worker_prefix(owner, &thread);
        // By key, read or not: a record that no longer decodes still names a
        // conversation, whose id its name gives.
        let workers: Vec<(String, Option<Worker>)> = records
            .under(&worker_prefix)
            .into_iter()
            .filter_map(|(key, value)| {
                let name = key.strip_prefix(&worker_prefix)?.to_string();
                let worker = serde_json::from_value::<Worker>(value)
                    .ok()
                    .map(|w| w.observed(now));
                Some((name, worker))
            })
            .collect();
        let kept_any = chat.is_some()
            || !workers.is_empty()
            || records.count_under(&record::said_prefix(owner, id)) > 0
            || records.count_under(&threads::note_prefix(owner, &thread)) > 0
            || threads::thread(records, owner, &thread).is_some();
        if !kept_any {
            return Ok(Disposal::Gone);
        }
        if let (When::If(due), Some(c)) = (when, chat.as_ref()) {
            if !c.closing && !due(c) {
                return Ok(Disposal::NotDue);
            }
        }
        if let Some(chat) = chat.as_ref().filter(|c| !c.closing) {
            let closing = Chat {
                closing: true,
                ..chat.clone()
            };
            record::put_chat(records, owner, &closing).map_err(|e| e.to_string())?;
        }
        let answering: Vec<&Worker> = workers
            .iter()
            .filter_map(|(_, w)| w.as_ref())
            .filter(|w| w.state.working())
            .collect();
        if !answering.is_empty() {
            for w in answering {
                stops::request(records, &w.conv_id, STOPPED).map_err(|e| e.to_string())?;
            }
            return Ok(Disposal::Answering);
        }
        (chat, workers)
    };
    for (name, _) in &workers {
        let conv_id = threads::conv_id(owner, &thread, name);
        tombstone(&conv_id).map_err(|e| format!("`{conv_id}` could not be tombstoned: {e}"))?;
    }
    let cease = |key: &str| records.cease(key).map_err(|e| e.to_string());
    for (name, _) in &workers {
        let conv_id = threads::conv_id(owner, &thread, name);
        nudges::take(records, &conv_id);
        stops::take(records, &conv_id);
        reported::forget(records, owner, &thread, name).map_err(|e| e.to_string())?;
        cease(&threads::worker_key(owner, &thread, name))?;
    }
    for (key, _) in records.under(&threads::note_prefix(owner, &thread)) {
        cease(&key)?;
    }
    if threads::thread(records, owner, &thread).is_some() {
        cease(&threads::thread_key(owner, &thread))?;
    }
    for (key, _) in record::said_keyed(records, owner, id) {
        cease(&key)?;
    }
    if chat.is_some() {
        cease(&chat_key(owner, id))?;
    }
    Ok(Disposal::Disposed)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;
    use crate::mcp::nudges::Sender;
    use crate::mcp::records::MemRecords;
    use crate::mcp::threads::{Note, State};

    use super::super::record::{Chat, Said, Speaker};

    const OWNER: &str = "o";
    const ID: &str = "0123456789abcdef";

    fn worker(name: &str, state: &str, beat_ms: u64) -> Worker {
        serde_json::from_value(json!({
            "thread": thread_of(ID), "name": name,
            "conv_id": threads::conv_id(OWNER, &thread_of(ID), name),
            "prompt": "p", "state": state, "role": "admin", "priority": "normal",
            "kind": "chat", "started_ms": 0, "turn_ms": 0, "beat_ms": beat_ms,
            "activity": null, "no_entry": null
        }))
        .unwrap()
    }

    /// A chat as `zen_chat_send` leaves it: its record, a thread, Zen's
    /// worker in `state`, a note, two messages, a nudge, a stop and a
    /// reported mark — beside a person's thread and another owner's chat,
    /// which disposal must not touch.
    fn kept(r: &MemRecords, state: &str) {
        let thread = thread_of(ID);
        record::put_chat(
            r,
            OWNER,
            &Chat {
                id: ID.into(),
                opened_ms: 0,
                last_ms: 5,
                closing: false,
            },
        )
        .unwrap();
        threads::attach(r, OWNER, &thread, 0).unwrap();
        threads::put_worker(r, OWNER, &worker("zen", state, 1_000)).unwrap();
        threads::put_worker(r, OWNER, &worker("zen.2", "done", 1_000)).unwrap();
        let note = Note {
            at_ms: 1,
            label: "Zen started:".into(),
            text: "t".into(),
            worker: None,
            client: None,
        };
        threads::add_note(r, OWNER, &thread, &note).unwrap();
        for (by, text) in [(Speaker::Caller, "hello"), (Speaker::Zen, "hi")] {
            let s = Said {
                at_ms: 2,
                by,
                text: text.into(),
                failed: false,
                client: None,
            };
            record::say(r, OWNER, ID, &s).unwrap();
        }
        let conv = threads::conv_id(OWNER, &thread, "zen");
        nudges::queue(r, &conv, Sender::Starter, "look again", 3).unwrap();
        reported::mark(r, OWNER, &thread, "zen", 4).unwrap();
        // Untouched: a person's thread, and the same chat id of another owner.
        threads::attach(r, OWNER, "review", 0).unwrap();
        threads::add_note(r, OWNER, "review", &note).unwrap();
        record::put_chat(
            r,
            "p",
            &Chat {
                id: ID.into(),
                opened_ms: 0,
                last_ms: 0,
                closing: false,
            },
        )
        .unwrap();
    }

    fn keys(r: &MemRecords) -> Vec<String> {
        r.under("mcp/").into_iter().map(|(k, _)| k).collect()
    }

    /// **Disposal tombstones every conversation the chat ran on and ceases
    /// every record it made** — and nothing else.
    #[test]
    fn disposal_ceases_every_record_the_chat_made_and_nothing_else() {
        let r = MemRecords::default();
        kept(&r, "done");
        let tombstoned = Mutex::new(Vec::new());
        let tombstone = |conv: &str| {
            tombstoned.lock().unwrap().push(conv.to_string());
            Ok(())
        };
        assert_eq!(
            dispose(&r, &tombstone, OWNER, ID, 10_000, When::Now),
            Ok(Disposal::Disposed)
        );
        assert_eq!(
            *tombstoned.lock().unwrap(),
            [
                format!("mcp:{OWNER}:~{ID}/zen"),
                format!("mcp:{OWNER}:~{ID}/zen.2")
            ]
        );
        let left = keys(&r);
        assert_eq!(left.len(), 3, "{left:?}");
        assert_eq!(left[0], format!("mcp/chat/p/{ID}"));
        assert!(left[1].starts_with("mcp/note/o/review/"), "{left:?}");
        assert_eq!(left[2], "mcp/thread/o/review");
        assert_eq!(
            dispose(&r, &tombstone, OWNER, ID, 10_001, When::Now),
            Ok(Disposal::Gone),
            "nothing is left to dispose of"
        );
    }

    /// **A chat Zen is answering is not disposed of yet**: it is marked
    /// closing, the answer is asked to stop, and nothing is tombstoned.
    #[test]
    fn a_chat_zen_is_answering_closes_and_its_answer_is_stopped() {
        let r = MemRecords::default();
        kept(&r, "running");
        let tombstone = |_: &str| -> Result<(), String> { panic!("nothing is tombstoned yet") };
        assert_eq!(
            dispose(&r, &tombstone, OWNER, ID, 1_500, When::Now),
            Ok(Disposal::Answering)
        );
        assert!(record::chat(&r, OWNER, ID).unwrap().closing);
        let conv = threads::conv_id(OWNER, &thread_of(ID), "zen");
        assert_eq!(stops::take(&r, &conv).as_deref(), Some(STOPPED));
        assert_eq!(record::said(&r, OWNER, ID).len(), 2, "the messages stay");
    }

    /// **An answer whose member stopped beating is not waited for**: nothing
    /// will end it, so the chat is disposed of.
    #[test]
    fn a_lost_answer_is_not_waited_for() {
        let r = MemRecords::default();
        kept(&r, "running");
        let late = 1_001 + threads::BEAT_LOST_MS;
        let lost = threads::worker(&r, OWNER, &thread_of(ID), "zen")
            .unwrap()
            .observed(late);
        assert_eq!(lost.state, State::Error);
        assert_eq!(
            dispose(&r, &|_: &str| Ok(()), OWNER, ID, late, When::Now),
            Ok(Disposal::Disposed)
        );
    }

    /// **A failed tombstone ceases nothing**: the chat stays, closing, for
    /// the next sweep to finish.
    #[test]
    fn a_failed_tombstone_ceases_nothing() {
        let r = MemRecords::default();
        kept(&r, "done");
        let before = keys(&r).len();
        let failing = |_: &str| -> Result<(), String> { Err("the model is still loading".into()) };
        let err = dispose(&r, &failing, OWNER, ID, 10_000, When::Now).unwrap_err();
        assert_eq!(
            err,
            format!("`mcp:{OWNER}:~{ID}/zen` could not be tombstoned: the model is still loading")
        );
        assert_eq!(keys(&r).len(), before);
        assert!(record::chat(&r, OWNER, ID).unwrap().closing);
    }

    #[test]
    fn a_chat_never_kept_is_gone() {
        let r = MemRecords::default();
        assert_eq!(
            dispose(&r, &|_: &str| Ok(()), OWNER, ID, 0, When::Now),
            Ok(Disposal::Gone)
        );
        assert!(keys(&r).is_empty(), "disposing of nothing writes nothing");
    }

    /// **A chat the sweep found due is judged again under the lock**: one
    /// spoken on since is kept as it is, and nothing is written.
    #[test]
    fn a_chat_no_longer_due_is_kept() {
        let r = MemRecords::default();
        kept(&r, "done");
        let before = r.under("mcp/");
        let tombstone = |_: &str| -> Result<(), String> { panic!("nothing is tombstoned") };
        let spoken_on = |c: &Chat| c.last_ms < 5;
        assert_eq!(
            dispose(&r, &tombstone, OWNER, ID, 10_000, When::If(&spoken_on)),
            Ok(Disposal::NotDue)
        );
        assert_eq!(r.under("mcp/"), before);
        let still_due = |c: &Chat| c.last_ms <= 5;
        assert_eq!(
            dispose(
                &r,
                &|_: &str| Ok(()),
                OWNER,
                ID,
                10_000,
                When::If(&still_due)
            ),
            Ok(Disposal::Disposed)
        );
    }
}
