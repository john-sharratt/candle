//! Talking on a chat (`docs/zen_chat.md` §1, §2): a send, a read, and Zen's
//! answer said as a message when its turn settles.
//!
//! A send runs under the lock a turn begins under ([`workers::holding`]), as a
//! task's start does, so two sends at once never begin two turns on one
//! conversation, and a disposal never ceases a chat under a turn it begins. Zen
//! answers on a worker of kind chat under the chat's thread: the same worker
//! for every message while it can be continued, and a fresh one — shown the
//! chat's last messages — after a turn that failed.

use std::sync::Arc;
use std::time::{Duration, Instant};

use mcp_server::{Session, Store};
use tokio::sync::mpsc::UnboundedSender;

use crate::mcp::archiving;
use crate::mcp::caller::Caller;
use crate::mcp::names;
use crate::mcp::records::Records;
use crate::mcp::roles::WorkerKind;
use crate::mcp::state::CHAT;
use crate::mcp::threads::{self, Effort, Priority, State, Worker};
use crate::mcp::workers::{self, now_ms};
use crate::session::ZendSession;

use super::dispose::{dispose, tombstone_with, When};
use super::naming::{id_of, new_id, thread_of, ZEN};
use super::record::{self, Chat, Said, Speaker};

/// How many of a chat's last messages a fresh worker on it is shown.
const RECAP: usize = 8;
/// How much of each it is shown.
const RECAP_QUOTE: usize = 600;
/// How often a waiting read looks again: an answer another member runs
/// changes its record without waking this one.
const LOOK_EVERY: Duration = Duration::from_secs(2);

/// What a send did.
#[derive(Debug)]
pub struct Sent {
    pub chat: Chat,
    pub worker: Worker,
    /// The sent message's number.
    pub number: usize,
    /// The chat the session held was disposed of, so this one is new.
    pub reopened: bool,
}

/// Which worker answers the next message on a chat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Next {
    /// The worker whose conversation holds the chat so far.
    Continue(Box<Worker>),
    /// A fresh one, by this name: the chat's first, or the one after a turn
    /// that failed.
    Fresh(String),
}

/// What a send is refused with while Zen answers.
pub const STILL_ANSWERING: &str = "Zen is still answering your last message: read its answer with \
    `zen_chat_read`, then send this one";

/// The worker that answers the next message on the chat whose thread is
/// `thread` and whose workers, as observed, are `workers`.
pub fn next(workers: &[Worker], thread: &str) -> Result<Next, String> {
    if workers.iter().any(|w| w.state.working()) {
        return Err(STILL_ANSWERING.into());
    }
    let newest = workers
        .iter()
        .filter(|w| !w.is_part())
        .max_by_key(|w| w.started_ms);
    match newest {
        Some(w)
            if w.state == State::Done && w.resumable && !archiving::being_archived(&w.conv_id) =>
        {
            Ok(Next::Continue(Box::new(w.clone())))
        }
        _ => {
            let taken: Vec<(String, bool)> = workers
                .iter()
                .map(|w| (w.name.clone(), w.state.live()))
                .collect();
            names::child_name(thread, "", ZEN, &taken).map(Next::Fresh)
        }
    }
}

/// `text`, at most `max` characters, marked where it was cut.
fn quote(text: &str, max: usize) -> String {
    let mut out: String = text.trim().chars().take(max).collect();
    if text.trim().chars().count() > max {
        out.push('…');
    }
    out
}

/// What a fresh worker on a chat is sent: `text`, after the chat's last
/// messages when there were any — a fresh worker's conversation holds none of
/// them.
pub fn recap(earlier: &[Said], text: &str) -> String {
    if earlier.is_empty() {
        return text.to_string();
    }
    let mut out = String::from(
        "This chat began before this conversation did: an earlier answer of yours failed, so you \
         are taking the chat up afresh. Its last messages:",
    );
    let skipped = earlier.len().saturating_sub(RECAP);
    for (i, s) in earlier.iter().enumerate().skip(skipped) {
        out.push_str(&format!(
            "\n#{} {}: {}",
            i + 1,
            s.by.name(),
            quote(&s.text, RECAP_QUOTE)
        ));
    }
    out.push_str("\n\nClaude's new message:\n");
    out.push_str(text);
    out
}

/// `w`, finished and kept, made ready to answer `text` as its next turn at
/// `effort`: its last answer and how it ended are cleared, so a reader never
/// takes them for this one's.
fn continued(
    session: &ZendSession,
    mut w: Worker,
    text: &str,
    caller: &Caller,
    effort: Effort,
    now: u64,
) -> Worker {
    workers::ready_for_turn(session, &mut w, caller.role, now);
    w.effort = effort;
    w.prompt = text.to_string();
    w.stopped = false;
    w.error = None;
    w.final_text = String::new();
    w
}

/// Send `text` to Zen on the chat the MCP session `mcp_id` (`mcp`, as the call
/// found it, kept in `store`) holds, opening one when it holds none, or one
/// since disposed of; Zen's answer runs at `effort`.
pub fn send(
    session: &Arc<ZendSession>,
    caller: &Caller,
    store: &dyn Store,
    (mcp_id, mcp): (&str, &Session),
    text: &str,
    effort: Effort,
) -> Result<Sent, String> {
    let records = session.shared_substrate().ok_or_else(workers::loading)?;
    if text.trim().is_empty() {
        return Err("the `text` is empty: say what to tell Zen".into());
    }
    let owner = caller.key();
    let now = now_ms();
    let _held = workers::holding();
    // What the session holds is read again under the lock, and the chat it
    // opens held under it, so two sends at once on a session that holds none
    // open one chat between them.
    let held = match store.get(mcp_id, now) {
        Some(kept) => kept.held(CHAT).map(str::to_string),
        None => mcp.held(CHAT).map(str::to_string),
    }
    .filter(|id| !id.is_empty());
    let found = held
        .as_deref()
        .and_then(|id| record::chat(&records, &owner, id))
        .filter(|c| !c.closing);
    let reopened = held.is_some() && found.is_none();
    let mut chat = found.unwrap_or_else(|| Chat {
        id: new_id(),
        session: mcp_id.to_string(),
        opened_ms: now,
        last_ms: now,
        closing: false,
    });
    chat.last_ms = now;
    // The chat's record first: whatever else is written after it, the sweep
    // finds the chat by its record, so nothing it made is left unfound.
    record::put_chat(&records, &owner, &chat).map_err(|e| e.to_string())?;
    if held.as_deref() != Some(chat.id.as_str()) {
        store
            .hold(mcp_id, mcp, CHAT, &chat.id, now)
            .map_err(|e| format!("this session could not keep its chat: {e}"))?;
    }
    let thread = thread_of(&chat.id);
    threads::attach(&records, &owner, &thread, now).map_err(|e| e.to_string())?;
    let all: Vec<Worker> = threads::workers(&records, &owner, &thread)
        .into_iter()
        .map(|w| w.observed(now))
        .collect();
    let number = records.count_under(&record::said_prefix(&owner, &chat.id)) + 1;
    let (worker, message) = match next(&all, &thread)? {
        Next::Continue(w) => (
            continued(session, *w, text, caller, effort, now),
            text.to_string(),
        ),
        Next::Fresh(name) => (
            workers::fresh(
                session,
                &owner,
                &thread,
                name,
                text,
                (caller.role, Priority::Normal, effort, WorkerKind::Chat),
                now,
            ),
            recap(&record::said(&records, &owner, &chat.id), text),
        ),
    };
    let said = Said {
        at_ms: now,
        by: Speaker::Claude,
        text: text.to_string(),
        failed: false,
    };
    record::say(&records, &owner, &chat.id, &said).map_err(|e| e.to_string())?;
    let worker = workers::begin(session, &records, &owner, worker, message)?;
    Ok(Sent {
        chat,
        worker,
        number,
        reopened,
    })
}

/// A chat as it stands: its record, its messages, and Zen's answer while it
/// runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub chat: Option<Chat>,
    pub said: Vec<Said>,
    pub answering: Option<Worker>,
}

/// `owner`'s chat `id` as it stands at `now`. Reads only.
pub fn view(records: &dyn Records, owner: &str, id: &str, now: u64) -> View {
    let answering = threads::workers(records, owner, &thread_of(id))
        .into_iter()
        .map(|w| w.observed(now))
        .filter(|w| !w.is_part() && w.state.working())
        .max_by_key(|w| w.started_ms);
    View {
        chat: record::chat(records, owner, id),
        said: record::said(records, owner, id),
        answering,
    }
}

/// The message a read starts after: `after`, or else Claude's last message,
/// so a read with none gives what Zen said since Claude last spoke.
pub fn start_after(said: &[Said], after: Option<usize>) -> usize {
    after.unwrap_or_else(|| {
        said.iter()
            .rposition(|s| s.by == Speaker::Claude)
            .map_or(0, |i| i + 1)
    })
}

/// Read `owner`'s chat `id` after its message `after` ([`start_after`]),
/// waiting up to `wait` while Zen answers and has said nothing new, each step
/// of its answer said on `progress`. The chat as it stands, and the message the
/// read started after.
pub async fn read(
    session: &ZendSession,
    owner: &str,
    id: &str,
    after: Option<usize>,
    wait: Duration,
    progress: Option<&UnboundedSender<String>>,
) -> Result<(View, usize), String> {
    let records = session.shared_substrate().ok_or_else(workers::loading)?;
    let deadline = Instant::now() + wait;
    let mut told = String::new();
    loop {
        // Subscribed before reading, so a change between the read and the
        // wait still wakes it.
        let mut changed = session.mcp().watch();
        let v = view(&records, owner, id, now_ms());
        let from = start_after(&v.said, after);
        let heard = v.said.iter().skip(from).any(|s| s.by == Speaker::Zen);
        let left = deadline.saturating_duration_since(Instant::now());
        let Some(w) = v.answering.as_ref().filter(|_| !heard && !left.is_zero()) else {
            return Ok((v, from));
        };
        if let Some(progress) = progress {
            let line = format!("Zen is answering — {}", w.doing(now_ms()));
            if line != told {
                let _ = progress.send(line.clone());
                told = line;
            }
        }
        let _ = tokio::time::timeout(left.min(LOOK_EVERY), changed.changed()).await;
    }
}

/// `v`'s messages after `from`, each under its number, and whether Zen is
/// still answering, as the chat's tools answer them.
pub fn render(v: &View, from: usize, now: u64) -> String {
    let mut out = format!(
        "Chat with Zen: {} message{}.",
        v.said.len(),
        if v.said.len() == 1 { "" } else { "s" }
    );
    let new: Vec<(usize, &Said)> = v
        .said
        .iter()
        .enumerate()
        .skip(from)
        .map(|(i, s)| (i + 1, s))
        .collect();
    if new.is_empty() {
        out.push_str(&format!(" Nothing after #{from}."));
    }
    for (n, s) in new {
        let failed = if s.failed {
            ", whose answer failed"
        } else {
            ""
        };
        out.push_str(&format!("\n\n#{n} {}{failed}:\n{}", s.by.name(), s.text));
    }
    if let Some(w) = &v.answering {
        out.push_str(&format!(
            "\n\nZen is still answering — {}. Read its answer with `zen_chat_read`.",
            w.doing(now)
        ));
    }
    out
}

/// Zen's side of a chat settled: what it answered, or the failure its turn
/// ended in, said as Zen's message — before its record says it ended, so a
/// reader woken by the record finds it.
pub fn answered(records: &dyn Records, owner: &str, worker: &Worker) {
    // A part's ending is its worker's to read, as its `sub_run` result.
    let Some(id) = id_of(&worker.thread).filter(|_| !worker.is_part()) else {
        return;
    };
    // Ordered against a disposal marking the chat closing, which this write
    // would otherwise undo.
    let _held = workers::holding();
    let Some(mut chat) = record::chat(records, owner, id) else {
        return;
    };
    let now = now_ms();
    let said = match worker.state {
        State::Error => Said {
            at_ms: now,
            by: Speaker::Zen,
            text: worker
                .error
                .clone()
                .unwrap_or_else(|| "the answer failed".into()),
            failed: true,
        },
        _ => Said {
            at_ms: now,
            by: Speaker::Zen,
            text: worker.final_text.clone(),
            failed: false,
        },
    };
    if let Err(e) = record::say(records, owner, id, &said) {
        tracing::warn!(chat = %id, "mcp: Zen's answer could not be kept on its chat: {e}");
    }
    chat.last_ms = now;
    if let Err(e) = record::put_chat(records, owner, &chat) {
        tracing::warn!(chat = %id, "mcp: a chat's last message could not be marked: {e}");
    }
}

/// Zen's side of a chat settled and its record kept: a chat closed while it
/// answered — or no longer kept at all — is disposed of now, off the
/// runtime's threads, since tombstoning writes through the persistence owner,
/// whose flow control can wait for room.
pub fn settled(session: &Arc<ZendSession>, owner: &str, worker: &Worker) {
    let Some(id) = id_of(&worker.thread).map(str::to_string) else {
        return;
    };
    let (session, owner) = (Arc::clone(session), owner.to_string());
    tokio::task::spawn_blocking(move || {
        let Some(records) = session.shared_substrate() else {
            return;
        };
        if record::chat(&records, &owner, &id).is_some_and(|c| !c.closing) {
            return;
        }
        let tombstone = tombstone_with(&session);
        if let Err(e) = dispose(&records, &tombstone, &owner, &id, now_ms(), When::Now) {
            tracing::warn!(chat = %id, "mcp: a closed chat could not be disposed of yet: {e}");
        }
    });
}

#[cfg(test)]
mod tests {
    use std::slice;

    use candle_conversation::projection::Conversation;
    use serde_json::json;
    use tempfile::TempDir;
    use web::auth::Role;

    use super::*;
    use crate::config::DaemonConfig;
    use crate::mcp::caller::Person;
    use mcp_server::Client;

    use crate::mcp::chats::dispose::{Disposal, STOPPED};
    use crate::mcp::records::MemRecords;
    use crate::mcp::state::{held_by, RecordStore};
    use crate::mcp::stops;
    use crate::workspace::single_repo;

    const THREAD: &str = "~0123456789abcdef";

    fn worker(name: &str, state: &str, started_ms: u64, resumable: bool) -> Worker {
        serde_json::from_value(json!({
            "thread": THREAD, "name": name, "conv_id": format!("mcp:o:{THREAD}/{name}"),
            "prompt": "p", "state": state, "role": "admin", "priority": "normal",
            "kind": "chat", "started_ms": started_ms, "turn_ms": started_ms,
            "beat_ms": started_ms, "resumable": resumable,
            "activity": null, "no_entry": null
        }))
        .unwrap()
    }

    fn said(by: Speaker, text: &str) -> Said {
        Said {
            at_ms: 0,
            by,
            text: text.into(),
            failed: false,
        }
    }

    /// **The first message opens Zen's side, a later one continues it, and a
    /// failed turn leaves the next message to a fresh `zen.2`.**
    #[test]
    fn the_next_message_goes_to_the_worker_that_can_answer_it() {
        assert_eq!(next(&[], THREAD), Ok(Next::Fresh("zen".into())));
        let done = worker("zen", "done", 1, true);
        assert_eq!(
            next(slice::from_ref(&done), THREAD),
            Ok(Next::Continue(Box::new(done)))
        );
        let failed = worker("zen", "error", 1, false);
        assert_eq!(next(&[failed], THREAD), Ok(Next::Fresh("zen.2".into())));
        let later = worker("zen.2", "done", 5, true);
        assert_eq!(
            next(&[worker("zen", "error", 1, false), later.clone()], THREAD),
            Ok(Next::Continue(Box::new(later)))
        );
    }

    /// **One message is answered at a time**: a send while Zen answers — or
    /// while a part of its answer runs — is refused.
    #[test]
    fn a_send_while_zen_answers_is_refused() {
        let now = now_ms();
        let running = worker("zen", "running", now, false);
        assert_eq!(next(&[running], THREAD), Err(STILL_ANSWERING.into()));
        let part = worker("zen/look", "running", now, false);
        assert_eq!(
            next(&[worker("zen", "done", 1, true), part], THREAD),
            Err(STILL_ANSWERING.into())
        );
    }

    #[test]
    fn a_fresh_worker_is_shown_the_chats_last_messages() {
        assert_eq!(recap(&[], "hello"), "hello");
        let earlier: Vec<Said> = (1..=10)
            .map(|i| {
                let by = if i % 2 == 1 {
                    Speaker::Claude
                } else {
                    Speaker::Zen
                };
                said(by, &format!("m{i}"))
            })
            .collect();
        assert_eq!(
            recap(&earlier, "and now?"),
            "This chat began before this conversation did: an earlier answer of yours failed, so \
             you are taking the chat up afresh. Its last messages:\n\
             #3 Claude: m3\n#4 Zen: m4\n#5 Claude: m5\n#6 Zen: m6\n\
             #7 Claude: m7\n#8 Zen: m8\n#9 Claude: m9\n#10 Zen: m10\n\n\
             Claude's new message:\nand now?"
        );
        assert_eq!(quote("abcdef", 3), "abc…");
        assert_eq!(quote(" abc ", 3), "abc");
    }

    /// **A read with no `after` gives what Zen said since Claude last spoke.**
    #[test]
    fn a_read_starts_after_claudes_last_message() {
        let s = [
            said(Speaker::Claude, "q1"),
            said(Speaker::Zen, "a1"),
            said(Speaker::Claude, "q2"),
            said(Speaker::Zen, "a2"),
            said(Speaker::Zen, "a2, more"),
        ];
        assert_eq!(start_after(&s, None), 3);
        assert_eq!(start_after(&s, Some(1)), 1);
        assert_eq!(start_after(&[], None), 0);
        assert_eq!(start_after(&s[..1], None), 1);
    }

    #[test]
    fn a_read_renders_each_message_under_its_number() {
        let mut failed = said(Speaker::Zen, "the model is still loading");
        failed.failed = true;
        let v = View {
            chat: None,
            said: vec![
                said(Speaker::Claude, "q1"),
                failed,
                said(Speaker::Zen, "a1"),
            ],
            answering: None,
        };
        assert_eq!(
            render(&v, 1, 0),
            "Chat with Zen: 3 messages.\n\n#2 Zen, whose answer failed:\nthe model is still \
             loading\n\n#3 Zen:\na1"
        );
        assert_eq!(
            render(&v, 3, 0),
            "Chat with Zen: 3 messages. Nothing after #3."
        );
        let one = View {
            said: vec![said(Speaker::Claude, "q1")],
            ..v
        };
        assert_eq!(
            render(&one, 1, 0),
            "Chat with Zen: 1 message. Nothing after #1."
        );
    }

    /// **A read writes nothing** (MCP request-path invariant 2).
    #[test]
    fn a_view_writes_no_record() {
        let r = MemRecords::default();
        record::say(&r, "o", "0123456789abcdef", &said(Speaker::Claude, "q")).unwrap();
        threads::put_worker(&r, "o", &worker("zen", "done", 1, true)).unwrap();
        let before = r.under("mcp/");
        let v = view(&r, "o", "0123456789abcdef", 10);
        assert_eq!(v.said.len(), 1);
        assert_eq!(v.answering, None);
        assert_eq!(r.under("mcp/"), before);
    }

    /// **Zen's answer becomes its message**, a failure said as one, and the
    /// chat's last message is marked; a part's ending is no message.
    #[test]
    fn zens_answer_is_said_as_its_message() {
        let r = MemRecords::default();
        let id = "0123456789abcdef";
        record::put_chat(
            &r,
            "o",
            &Chat {
                id: id.into(),
                session: "s".into(),
                opened_ms: 0,
                last_ms: 0,
                closing: false,
            },
        )
        .unwrap();
        let mut w = worker("zen", "done", 1, true);
        w.final_text = "about 30 GiB".into();
        answered(&r, "o", &w);
        let mut failed = worker("zen", "error", 1, false);
        failed.error = Some("the model is still loading".into());
        answered(&r, "o", &failed);
        answered(&r, "o", &worker("zen/look", "done", 1, false));
        let got = record::said(&r, "o", id);
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(
            (got[0].by, got[0].text.as_str(), got[0].failed),
            (Speaker::Zen, "about 30 GiB", false)
        );
        assert_eq!(
            (got[1].text.as_str(), got[1].failed),
            ("the model is still loading", true)
        );
        assert!(record::chat(&r, "o", id).unwrap().last_ms > 0);
    }

    fn caller() -> Caller {
        Caller {
            person: Person {
                provider: "google".into(),
                sub: "john".into(),
                email: "john@example.com".into(),
                name: "John".into(),
            },
            role: Role::Admin,
        }
    }

    /// **A chat runs on a real session**: the first send opens the chat and
    /// Zen's side of it, running; with no model loaded its turn waits for one,
    /// so a read waits out its `wait` and says Zen is still answering, and a
    /// second send is refused. The chat's thread keeps no journal line.
    /// Closing it while Zen answers marks it closing and asks the answer to
    /// stop, tombstoning nothing yet, and a send after that opens a new chat.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_chat_runs_on_a_real_session() {
        let work = TempDir::new().unwrap();
        let session = Arc::new(ZendSession::new(DaemonConfig::new(
            single_repo(work.path(), "project").unwrap(),
        )));
        session.hold_substrate(Conversation::new());
        let records = session.shared_substrate().unwrap();
        let (caller, owner) = (caller(), caller().key());
        let store = RecordStore::new(records.clone(), Arc::default());
        let mcp = Session::new(&owner, Client::Other, "2026-07-28", now_ms());
        let on = ("s1", &mcp);

        let first = send(
            &session,
            &caller,
            &store,
            on,
            "how big is the arena?",
            Effort::Balanced,
        )
        .unwrap();
        assert_eq!(held_by(&records, "s1", CHAT), Some(first.chat.id.clone()));
        assert_eq!((first.number, first.reopened), (1, false));
        assert_eq!(first.worker.name, ZEN);
        assert!(first.worker.kind.is_chat());
        assert_eq!(first.worker.effort, Effort::Balanced);
        assert_eq!(first.chat.session, "s1");
        let thread = thread_of(&first.chat.id);
        assert_eq!(first.worker.conv_id, format!("mcp:{owner}:{thread}/zen"));

        let started = Instant::now();
        let (v, from) = read(
            &session,
            &owner,
            &first.chat.id,
            None,
            Duration::from_millis(300),
            None,
        )
        .await
        .unwrap();
        assert!(started.elapsed() >= Duration::from_millis(300), "it waited");
        assert_eq!(from, 1, "a read starts after Claude's last message");
        assert_eq!(v.said.len(), 1);
        assert_eq!(
            (v.said[0].by, v.said[0].text.as_str()),
            (Speaker::Claude, "how big is the arena?")
        );
        let answering = v.answering.as_ref().expect("Zen is answering");
        assert_eq!(answering.name, ZEN);
        assert!(render(&v, from, now_ms()).starts_with(
            "Chat with Zen: 1 message. Nothing after #1.\n\nZen is still answering — "
        ));

        assert_eq!(
            send(
                &session,
                &caller,
                &store,
                on,
                "still there?",
                Effort::Balanced
            )
            .unwrap_err(),
            STILL_ANSWERING,
            "the session's chat is read from its record, not the call's snapshot"
        );
        assert_eq!(record::said(&records, &owner, &first.chat.id).len(), 1);
        assert_eq!(
            threads::note_count(&records, &owner, &thread),
            0,
            "a chat's thread keeps no copy of what was said"
        );

        let nothing = |_: &str| -> Result<(), String> { panic!("nothing is tombstoned yet") };
        assert_eq!(
            dispose(
                &records,
                &nothing,
                &owner,
                &first.chat.id,
                now_ms(),
                When::Now
            ),
            Ok(Disposal::Answering)
        );
        assert!(
            record::chat(&records, &owner, &first.chat.id)
                .unwrap()
                .closing
        );
        assert_eq!(
            stops::take(&records, &first.worker.conv_id).as_deref(),
            Some(STOPPED)
        );

        let after = send(&session, &caller, &store, on, "and now?", Effort::Deep).unwrap();
        assert!(after.reopened, "a closing chat takes no more messages");
        assert_ne!(after.chat.id, first.chat.id);
        assert_eq!((after.number, after.worker.name.as_str()), (1, ZEN));
        assert_eq!(after.worker.effort, Effort::Deep);
        assert_eq!(held_by(&records, "s1", CHAT), Some(after.chat.id.clone()));
    }
}
