//! A conversation's own journal (`docs/npc_conversations.md` §5.5): the same
//! record a zend dialogue keeps of itself (`docs/zend_journal.md`), written by
//! the same workflow (`journal_core::workflow`), for a character's
//! conversation with one person.
//!
//! **One journal a conversation, as a dialogue has.** After each turn a
//! person spoke on, the conversation is asked — on a fork of itself, writing
//! nothing back — for an entry: what was asked and answered in one sentence,
//! each claim citing the line it rests on, what the character means to do
//! next, and what it left open. A turn that added nothing is skipped. The
//! journal is kept in the conversation's own metadata, so it follows the
//! conversation and nobody else's.
//!
//! **A kept entry is news to the mind.** It is told to the character's main
//! loop as it is kept, and wakes it: what a conversation promised is what the
//! mind is for, and words said on the phone that nothing recorded were never
//! acted on.
//!
//! What is the conversation's is here — the stretch a turn makes, the wording,
//! the record. The draft itself is `Minds::talk_journal`.

use std::collections::BTreeMap;

use journal_core::entry::Kind;
use journal_core::journal::{Holds, Journal};
use journal_core::section;
use journal_core::span::SpanTurn;
use journal_core::style::Style;

use crate::engine::event::TalkLine;
use crate::engine::journal::style::NPC;

/// The metadata key a conversation's journal record is kept under.
pub const META_JOURNAL: &str = "talk.journal";

/// The wording a conversation's journal is kept in: a character's own, with
/// no facts — what was said to it is not something the world can read back.
pub const TALK: Style = Style {
    facts: false,
    ..NPC
};

/// One conversation's journal.
#[derive(Debug, Default)]
pub struct TalkJournal(Journal);

impl Holds for TalkJournal {
    fn journal(&self) -> &Journal {
        &self.0
    }

    fn journal_mut(&mut self) -> &mut Journal {
        &mut self.0
    }
}

impl TalkJournal {
    /// The journal a conversation's metadata holds, or an empty one. Its turn
    /// ids are the thread's own, so what it covered still means what it did.
    pub fn restore(meta: &BTreeMap<String, String>) -> Self {
        match meta.get(META_JOURNAL).and_then(|r| section::restore(r)) {
            Some(kept) => {
                let covered = kept.covered_to;
                TalkJournal(Journal::restore(kept, covered))
            }
            None => TalkJournal(Journal::new()),
        }
    }

    /// The record to keep in the conversation's metadata.
    pub fn record(&self) -> String {
        self.0.prompt(&TALK).record().to_string()
    }
}

/// The number a line of the thread is cited by: twice its seq, so the reply
/// to it has the number between it and the next line.
fn heard_id(seq: u64) -> u64 {
    seq * 2
}

/// The turns a conversation's turn makes, as a draft is shown them: each line
/// the party said, heard, then what the character replied, its own act. At
/// `at_ms`, the character's world time. Empty when the party said nothing —
/// a turn the character began itself is its mind's own doing, and its mind
/// already knows it.
pub fn stretch(lines: &[&TalkLine], me: &str, reply: &str, at_ms: u64) -> Vec<SpanTurn> {
    let Some(last) = lines.iter().map(|l| l.seq).max() else {
        return Vec::new();
    };
    let mut turns: Vec<SpanTurn> = lines
        .iter()
        .map(|l| {
            let text = format!("{}: {}", l.name, l.text.trim());
            SpanTurn {
                id: heard_id(l.seq),
                at_ms,
                evidence: Some(Kind::Heard),
                full: text.clone(),
                text,
            }
        })
        .collect();
    let reply = reply.trim();
    if !reply.is_empty() {
        let text = format!("{me} replied: {reply}");
        turns.push(SpanTurn {
            id: heard_id(last) + 1,
            at_ms,
            evidence: Some(Kind::Did),
            full: text.clone(),
            text,
        });
    }
    turns
}

#[cfg(test)]
mod tests {
    use journal_core::entry::Entry;

    use super::*;

    fn entry(summary: &str, from_turn: u64, to_turn: u64) -> Entry {
        Entry {
            id: 0,
            from_turn,
            to_turn,
            from_ms: 0,
            to_ms: 0,
            summary: summary.into(),
            claims: Vec::new(),
            intend: Vec::new(),
            opened: Vec::new(),
            resolved: Vec::new(),
            period: Default::default(),
            digest: String::new(),
        }
    }

    fn line(seq: u64, text: &str) -> TalkLine {
        TalkLine {
            name: "John".into(),
            seq,
            from: "u_john".into(),
            text: text.into(),
            account: None,
            at_ms: 0,
        }
    }

    /// A turn's stretch is what was heard, line by line, then what was said
    /// back, each citable and each numbered so the order holds across turns.
    #[test]
    fn a_turn_is_what_was_heard_then_what_was_replied() {
        let (a, b) = (line(7, "  Is the cluster up? "), line(8, "And the house?"));
        let turns = stretch(&[&a, &b], "Zen", "I will look and tell you.", 42);
        let shown: Vec<(u64, Option<Kind>, &str)> = turns
            .iter()
            .map(|t| (t.id, t.evidence, t.text.as_str()))
            .collect();
        assert_eq!(
            shown,
            [
                (14, Some(Kind::Heard), "John: Is the cluster up?"),
                (16, Some(Kind::Heard), "John: And the house?"),
                (
                    17,
                    Some(Kind::Did),
                    "Zen replied: I will look and tell you."
                ),
            ]
        );
        assert!(turns.iter().all(|t| t.at_ms == 42 && t.full == t.text));
        // The next turn's lines number past this one's reply.
        let next = line(9, "Thanks.");
        assert!(stretch(&[&next], "Zen", "", 43)[0].id > 17);
    }

    /// Nothing heard is nothing to journal, whatever was said.
    #[test]
    fn a_turn_nobody_spoke_on_has_no_stretch() {
        assert!(stretch(&[], "Zen", "The cluster is up.", 1).is_empty());
    }

    /// The record round-trips through the conversation's metadata with its
    /// coverage, and a conversation with none starts empty.
    #[test]
    fn the_journal_is_restored_from_the_conversations_metadata() {
        assert!(TalkJournal::restore(&BTreeMap::new())
            .journal()
            .entries()
            .next()
            .is_none());
        let mut held = TalkJournal::default();
        held.journal_mut().keep_closing(entry(
            "John asked whether the cluster is up; I said I would look.",
            14,
            17,
        ));
        let meta = BTreeMap::from([(META_JOURNAL.to_string(), held.record())]);
        let again = TalkJournal::restore(&meta);
        let summaries: Vec<&str> = again
            .journal()
            .entries()
            .map(|e| e.summary.as_str())
            .collect();
        assert_eq!(
            summaries,
            ["John asked whether the cluster is up; I said I would look."]
        );
        assert_eq!(again.journal().covered_to(), 17);
        assert_eq!(again.record(), held.record());
    }

    /// A conversation's claims carry no facts: the world cannot read back
    /// what somebody said.
    #[test]
    fn a_conversations_journal_checks_no_facts() {
        assert_eq!((TALK.facts, NPC.facts), (false, true));
        assert_eq!(TALK.entry, NPC.entry);
    }
}
