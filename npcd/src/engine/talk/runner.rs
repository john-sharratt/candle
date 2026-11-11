//! Every conversation of the characters this member holds
//! (`docs/npc_conversations.md` §9): what has arrived for each, and whether
//! its turn is due.
//!
//! Lines arrive as `Talk` events and purposes as `Purpose` events, handed
//! here by the scheduler in place of the character's inbox: a conversation
//! answers them, never the mind. Each conversation gathers what arrives over
//! its burst window, takes one turn at a time, and keeps what arrives during
//! a turn for the next ([`clock::Phase`]).

use std::collections::HashMap;
use std::mem;
use std::sync::Mutex;

use tokio::task::AbortHandle;

use crate::engine::event::TalkLine;
use crate::engine::talk::clock::{Ended, Phase};

/// A conversation, as the runner keys it: the character and the party's
/// member id — a body id, or an account handle.
pub type Key = (u64, String);

/// What a turn takes in.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Inputs {
    /// The pair's thread, once a line has named it.
    pub thread: Option<String>,
    pub lines: Vec<TalkLine>,
    pub purposes: Vec<String>,
    /// How the previous turn ended, for the activation count.
    pub last: Option<Ended>,
}

#[derive(Default)]
struct Conv {
    phase: Phase,
    inputs: Inputs,
}

/// Every conversation on this member.
#[derive(Default)]
pub struct Talks {
    convs: Mutex<HashMap<Key, Conv>>,
    /// Each running turn, so a character put down cancels its turns before
    /// it is released.
    running: Mutex<HashMap<Key, AbortHandle>>,
}

impl Talks {
    pub fn new() -> Talks {
        Talks::default()
    }

    fn with<T>(&self, f: impl FnOnce(&mut HashMap<Key, Conv>) -> T) -> T {
        f(&mut self.convs.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Lines from `member` on `thread` arrived for `npc`'s conversation.
    pub fn lines(&self, npc: u64, member: &str, thread: &str, lines: Vec<TalkLine>, now_ms: u64) {
        self.with(|convs| {
            let c = convs.entry((npc, member.to_string())).or_default();
            c.inputs.thread = Some(thread.to_string());
            for l in lines {
                if !c.inputs.lines.iter().any(|h| h.seq == l.seq) {
                    c.inputs.lines.push(l);
                }
            }
            c.inputs.lines.sort_by_key(|l| l.seq);
            c.phase.arrive(now_ms);
        });
    }

    /// `npc` means to tell `member` something.
    pub fn purpose(&self, npc: u64, member: &str, intent: &str, now_ms: u64) {
        self.with(|convs| {
            let c = convs.entry((npc, member.to_string())).or_default();
            c.inputs.purposes.push(intent.trim().to_string());
            c.phase.arrive(now_ms);
        });
    }

    /// The conversations whose turn starts at `now_ms`; each is now turning.
    pub fn due(&self, now_ms: u64) -> Vec<Key> {
        self.with(|convs| {
            let mut due: Vec<Key> = convs
                .iter_mut()
                .filter_map(|(k, c)| c.phase.start_if_due(now_ms).then(|| k.clone()))
                .collect();
            due.sort();
            due
        })
    }

    /// What `key`'s turn takes in, taken out: what arrives from now waits for
    /// the next turn.
    pub fn take(&self, key: &Key) -> Inputs {
        self.with(|convs| {
            let Some(c) = convs.get_mut(key) else {
                return Inputs::default();
            };
            let last = c.inputs.last;
            let mut taken = mem::take(&mut c.inputs);
            c.inputs.thread = taken.thread.clone();
            c.inputs.last = last;
            taken.last = last;
            taken
        })
    }

    /// `key`'s turn ended, as `ended`, or took nothing: what arrived during it
    /// gathers for the next.
    pub fn finish(&self, key: &Key, ended: Option<Ended>) {
        self.with(|convs| {
            if let Some(c) = convs.get_mut(key) {
                if ended.is_some() {
                    c.inputs.last = ended;
                }
                c.phase.end_turn();
            }
        });
    }

    /// `key`'s turn could not run: what it took goes back ahead of what
    /// arrived since, and gathers again from `now_ms`.
    pub fn put_back(&self, key: &Key, mut inputs: Inputs, now_ms: u64) {
        self.with(|convs| {
            let c = convs.entry(key.clone()).or_default();
            inputs.lines.append(&mut c.inputs.lines);
            inputs.lines.sort_by_key(|l| l.seq);
            inputs.lines.dedup_by_key(|l| l.seq);
            inputs.purposes.append(&mut c.inputs.purposes);
            inputs.last = c.inputs.last.or(inputs.last);
            inputs.thread = inputs.thread.or(c.inputs.thread.take());
            c.inputs = inputs;
            c.phase.end_turn();
            c.phase.arrive(now_ms);
        });
    }

    /// How many lines wait for `npc`'s conversations to take them.
    pub fn waiting(&self, npc: u64) -> usize {
        self.with(|convs| {
            convs
                .iter()
                .filter(|((n, _), _)| *n == npc)
                .map(|(_, c)| c.inputs.lines.len() + c.inputs.purposes.len())
                .sum()
        })
    }

    /// Whether any of `npc`'s conversations is mid-turn.
    pub fn turning(&self, npc: u64) -> bool {
        self.with(|convs| {
            convs
                .iter()
                .any(|((n, _), c)| *n == npc && matches!(c.phase, Phase::Turning { .. }))
        })
    }

    /// `key`'s turn is running as `turn`.
    pub fn running(&self, key: Key, turn: AbortHandle) {
        self.running
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, turn);
    }

    /// `key`'s turn is over.
    pub fn ran(&self, key: &Key) {
        self.running
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(key);
    }

    /// Put `npc`'s conversations down: its running turns are cancelled, the
    /// character leaves this member, and whoever holds it next rescans what
    /// waits for it.
    pub fn forget(&self, npc: u64) {
        let mut running = self.running.lock().unwrap_or_else(|e| e.into_inner());
        running.retain(|(n, _), turn| {
            if *n == npc {
                turn.abort();
            }
            *n != npc
        });
        drop(running);
        self.with(|convs| convs.retain(|(n, _), _| *n != npc));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::talk::clock::BURST_MS;

    fn line(seq: u64, text: &str) -> TalkLine {
        TalkLine {
            seq,
            from: "u_mira".into(),
            name: "Mira".into(),
            text: text.into(),
            account: Some("u_mira".into()),
            at_ms: 0,
        }
    }

    fn mira() -> Key {
        (7, "u_mira".to_string())
    }

    #[test]
    fn a_burst_is_one_turn_that_takes_every_line() {
        let t = Talks::new();
        t.lines(7, "u_mira", "dm:npc-7:u_mira", vec![line(1, "hi")], 0);
        t.lines(
            7,
            "u_mira",
            "dm:npc-7:u_mira",
            vec![line(2, "are you there?")],
            1_000,
        );
        assert!(t.due(1_000 + BURST_MS - 1).is_empty());
        assert_eq!(t.due(1_000 + BURST_MS), [mira()]);
        assert!(t.turning(7));
        let taken = t.take(&mira());
        let seqs: Vec<u64> = taken.lines.iter().map(|l| l.seq).collect();
        assert_eq!(seqs, [1, 2]);
        assert_eq!(taken.thread.as_deref(), Some("dm:npc-7:u_mira"));
        t.finish(&mira(), Some(Ended::Spoke));
        assert!(!t.turning(7));
        assert!(t.due(u64::MAX).is_empty());
    }

    #[test]
    fn what_arrives_during_a_turn_waits_for_the_next() {
        let t = Talks::new();
        t.purpose(7, "u_mira", "that the parcel is in", 0);
        assert_eq!(t.due(BURST_MS), [mira()]);
        let first = t.take(&mira());
        assert_eq!(first.purposes, ["that the parcel is in"]);
        t.lines(
            7,
            "u_mira",
            "dm:npc-7:u_mira",
            vec![line(3, "thanks!")],
            3_000,
        );
        assert!(t.due(10_000).is_empty(), "one turn at a time");
        t.finish(&mira(), Some(Ended::Spoke));
        assert_eq!(t.due(3_000 + BURST_MS), [mira()]);
        let second = t.take(&mira());
        assert_eq!(second.lines.len(), 1);
        assert_eq!(second.last, Some(Ended::Spoke));
    }

    #[test]
    fn a_turn_that_could_not_run_keeps_what_it_took() {
        let t = Talks::new();
        t.lines(7, "u_mira", "dm:npc-7:u_mira", vec![line(1, "hi")], 0);
        t.due(BURST_MS);
        let taken = t.take(&mira());
        t.lines(7, "u_mira", "dm:npc-7:u_mira", vec![line(2, "hello?")], 100);
        t.put_back(&mira(), taken, 5_000);
        assert_eq!(t.due(5_000 + BURST_MS), [mira()]);
        let again = t.take(&mira());
        let seqs: Vec<u64> = again.lines.iter().map(|l| l.seq).collect();
        assert_eq!(seqs, [1, 2]);
    }

    #[test]
    fn a_character_put_down_forgets_its_conversations_here() {
        let t = Talks::new();
        t.purpose(7, "u_mira", "x", 0);
        t.purpose(8, "u_mira", "y", 0);
        t.forget(7);
        assert_eq!(t.due(BURST_MS), [(8, "u_mira".to_string())]);
    }
}
