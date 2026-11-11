//! The calls each conversation made that changed something, as the
//! aggregator records them (`user_episode_note`), held in memory until
//! [`write_noted`] writes each conversation's into its `journal.yaml` — at most
//! once every [`WRITE_EVERY`], off the request path, one commit per
//! conversation (`docs/zen_memory_and_guides.md` Phase 2). A call itself
//! only notes.

use std::collections::HashMap;
use std::mem;
use std::sync::Mutex;
use std::time::Duration;

use journals::actor::Actor;
use journals::files::activity::MAX_DONE;
use journals::ops::{Call, Op};
use journals::scope::Channel;

use super::caller::Caller;
use super::tools::mem::open_for;
use crate::session::ZendSession;

/// How often the noted calls are written.
pub const WRITE_EVERY: Duration = Duration::from_secs(60);

/// One conversation's calls not yet written, and who made them in which
/// conversation, by its ref's key — what opens the conversation's journal.
#[derive(Clone, Debug)]
struct Noted {
    caller: Caller,
    conversation: String,
    channel: Channel,
    calls: Vec<Call>,
}

/// The calls noted since the last write, by conversation.
#[derive(Default)]
pub struct NotedCalls(Mutex<HashMap<String, Noted>>);

/// What names a conversation's calls: who, in which conversation, through
/// which channel — as the conversation's journal is found.
fn key(caller: &Caller, conversation: &str, channel: Channel) -> String {
    format!("{}\0{conversation}\0{}", caller.key(), channel.name())
}

/// `calls` cut to the newest an activity record keeps.
fn newest(calls: &mut Vec<Call>) {
    let over = calls.len().saturating_sub(MAX_DONE);
    calls.drain(..over);
}

impl NotedCalls {
    /// Note `call`, made by `caller` through `channel` in `conversation`.
    pub fn note(&self, caller: &Caller, conversation: &str, channel: Channel, call: Call) {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let n = held
            .entry(key(caller, conversation, channel))
            .or_insert_with(|| Noted {
                caller: caller.clone(),
                conversation: conversation.to_string(),
                channel,
                calls: Vec::new(),
            });
        n.calls.push(call);
        newest(&mut n.calls);
    }

    fn take(&self) -> Vec<Noted> {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        mem::take(&mut *held).into_values().collect()
    }

    /// Hold `n` again, before whatever was noted for it since it was taken.
    fn keep(&self, mut n: Noted) {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(since) = held.remove(&key(&n.caller, &n.conversation, n.channel)) {
            n.calls.extend(since.calls);
        }
        newest(&mut n.calls);
        held.insert(key(&n.caller, &n.conversation, n.channel), n);
    }

    /// How many conversations have calls waiting.
    pub fn waiting(&self) -> usize {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

/// What became of one conversation's calls.
#[derive(Debug, PartialEq, Eq)]
enum Wrote {
    Written,
    /// Not written, and never will be as they are: the caller has no
    /// conversation, or the rules refuse the record.
    Dropped,
    /// Not written yet: the memory is still being read, or moved under it.
    Again,
}

/// Write every conversation's noted calls; how many were written, dropped and
/// held for the next write.
pub fn write_noted(session: &ZendSession, noted: &NotedCalls) -> (usize, usize, usize) {
    let (mut written, mut dropped, mut again) = (0, 0, 0);
    for n in noted.take() {
        match write_one(session, &n) {
            Wrote::Written => written += 1,
            Wrote::Dropped => dropped += 1,
            Wrote::Again => {
                again += 1;
                noted.keep(n);
            }
        }
    }
    (written, dropped, again)
}

fn write_one(session: &ZendSession, n: &Noted) -> Wrote {
    let m = match open_for(session, &n.caller, n.channel, Some(n.conversation.clone())) {
        Ok(m) => m,
        Err(e) => {
            tracing::info!("journals: noted calls wait for the memory: {e}");
            return Wrote::Again;
        }
    };
    let (Some(person), Some(path)) = (m.person(), m.here.clone()) else {
        tracing::info!(
            calls = n.calls.len(),
            "journals: noted calls not recorded: {}",
            m.why_no_here()
        );
        return Wrote::Dropped;
    };
    let server = Actor::Server {
        person: person.clone(),
        channel: m.channel,
    };
    let op = Op::Activity {
        path: path.clone(),
        calls: n.calls.clone(),
    };
    match m.memory.write(&server, &op) {
        Ok(_) => Wrote::Written,
        Err(e) if e.is_refusal() || e.is_broken() => {
            tracing::warn!(%path, calls = n.calls.len(), "journals: noted calls refused: {e}");
            Wrote::Dropped
        }
        Err(e) => {
            tracing::info!(%path, "journals: noted calls wait to be written: {e}");
            Wrote::Again
        }
    }
}

#[cfg(test)]
mod tests {
    use journals::time::Stamp;
    use web::auth::Role;

    use super::*;
    use crate::mcp::caller::Person;

    fn caller(sub: &str) -> Caller {
        Caller {
            person: Person {
                provider: "google".into(),
                sub: sub.into(),
                email: String::new(),
                name: String::new(),
            },
            role: Role::User,
        }
    }

    fn call(tool: &str, at: i64) -> Call {
        Call {
            at: Stamp::parse("2026-11-09T00:00:00Z").unwrap().plus(at),
            tool: tool.into(),
            subject: String::new(),
        }
    }

    /// **A conversation's calls are held together, newest last, at most as
    /// many as its record keeps**; another conversation's are held apart.
    #[test]
    fn calls_are_held_by_conversation_and_capped() {
        let noted = NotedCalls::default();
        let john = caller("john");
        for i in 0..(MAX_DONE as i64 + 3) {
            noted.note(&john, "s1", Channel::Grok, call("log_grep", i));
        }
        noted.note(&john, "s2", Channel::Grok, call("job_status", 0));
        noted.note(&john, "s1", Channel::ClaudeAi, call("job_status", 0));
        assert_eq!(noted.waiting(), 3);
        let taken = noted.take();
        assert_eq!(noted.waiting(), 0);
        let grok = taken
            .iter()
            .find(|n| n.conversation == "s1" && n.channel == Channel::Grok)
            .unwrap();
        assert_eq!(grok.calls.len(), MAX_DONE);
        assert_eq!(grok.calls[0], call("log_grep", 3), "the oldest let go");
    }

    /// **Calls held again go before those noted since**, so a write that
    /// waited keeps their order.
    #[test]
    fn calls_held_again_keep_their_order() {
        let noted = NotedCalls::default();
        let john = caller("john");
        noted.note(&john, "s1", Channel::Grok, call("a", 0));
        let first = noted.take().pop().unwrap();
        noted.note(&john, "s1", Channel::Grok, call("b", 1));
        noted.keep(first);
        let again = noted.take().pop().unwrap();
        assert_eq!(again.calls, [call("a", 0), call("b", 1)]);
    }
}
