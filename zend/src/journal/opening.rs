//! The journal's opening: two entries a dialogue's journal begins with — what
//! it had read before its session began, then that the user asked their
//! opening question.
//!
//! A dialogue descends from the priming chain — every repository's root
//! listing and its anchor documents — and from the working set's files, and
//! all of it stands ahead of the user's first message as the dialogue's own
//! reading. Its drafted entries say nothing of that, so a dialogue asked what
//! it had read went and read it again. The opening says it plainly, as
//! ordinary entries written when the conversation opens: the prompt, the
//! journal tools and the journal's view read them as they read any other. The
//! first is built from the chain the member last primed.
//!
//! Neither has turns behind it: they are about what happened before the
//! conversation's first turn.

use std::time::Duration;

use journal_core::entry::Entry;
use journal_core::period::Period;

/// How long the round that wrote the opening waits for it to seal. It is two
/// short lines, so a seal past this is a GPU queue the round should not stand
/// behind: it answers without the opening, and the next round has it.
pub const OPENING_SEAL_WAIT: Duration = Duration::from_secs(2);

/// The opening's second entry.
pub const READING: &str = "Then read other files that seemed relevant to the work here, and \
the user asked their opening question of this session, which I must answer.";

/// What the priming chain read, in chain order: each repository whose root
/// listing it read, and each anchor document, as `{repo}/{path}`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SeedRead {
    pub listings: Vec<String>,
    pub documents: Vec<String>,
}

impl SeedRead {
    /// The opening's first entry's line, or `None` when the chain read nothing.
    pub fn line(&self) -> Option<String> {
        let mut read: Vec<String> = Vec::new();
        if !self.listings.is_empty() {
            let one = self.listings.len() == 1;
            read.push(format!(
                "the root listing{} of {}",
                if one { "" } else { "s" },
                listed(&self.listings)
            ));
        }
        if !self.documents.is_empty() {
            read.push(listed(&self.documents));
        }
        (!read.is_empty()).then(|| {
            format!(
                "Before this session began, read the workspace: {}.",
                read.join("; ")
            )
        })
    }

    /// The opening as entries written at `now_ms`, in order — both, or none
    /// when the chain read nothing. Their ids are the journal's to give.
    pub fn entries(&self, now_ms: u64) -> Vec<Entry> {
        let Some(line) = self.line() else {
            return Vec::new();
        };
        [line, READING.to_string()]
            .into_iter()
            .map(|summary| opening_entry(summary, now_ms))
            .collect()
    }
}

/// An entry of the opening: a step at `now_ms` over no turns.
fn opening_entry(summary: String, now_ms: u64) -> Entry {
    Entry {
        id: 0,
        from_turn: 0,
        to_turn: 0,
        from_ms: now_ms,
        to_ms: now_ms,
        summary,
        claims: Vec::new(),
        intend: Vec::new(),
        opened: Vec::new(),
        resolved: Vec::new(),
        period: Period::Step,
        digest: String::new(),
    }
}

/// `items` as prose: `a`, `a and b`, `a, b and c`, each in backticks.
fn listed(items: &[String]) -> String {
    let quoted: Vec<String> = items.iter().map(|i| format!("`{i}`")).collect();
    match quoted.split_last() {
        None => String::new(),
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    /// **The first entry names what the chain read** — the listings, then the
    /// documents, in chain order, by the paths the file tools take.
    #[test]
    fn the_first_entry_names_every_listing_and_document_the_chain_read() {
        let read = SeedRead {
            listings: strings(&["candle", "battle-cities", "yenbot"]),
            documents: strings(&["candle/README.md", "candle/AGENTS.md", "yenbot/README.md"]),
        };
        assert_eq!(
            read.line().unwrap(),
            "Before this session began, read the workspace: the root listings of `candle`, \
             `battle-cities` and `yenbot`; `candle/README.md`, `candle/AGENTS.md` and \
             `yenbot/README.md`."
        );
    }

    #[test]
    fn one_listing_and_one_document_read_as_one() {
        let read = SeedRead {
            listings: strings(&["candle"]),
            documents: strings(&["candle/README.md"]),
        };
        assert_eq!(
            read.line().unwrap(),
            "Before this session began, read the workspace: the root listing of `candle`; \
             `candle/README.md`."
        );
        let listings_only = SeedRead {
            documents: Vec::new(),
            ..read
        };
        assert_eq!(
            listings_only.line().unwrap(),
            "Before this session began, read the workspace: the root listing of `candle`."
        );
    }

    /// **Both entries, in order, or none** — steps written at the moment the
    /// conversation opened, over no turns.
    #[test]
    fn the_opening_is_both_entries_or_none() {
        assert!(SeedRead::default().entries(5).is_empty());
        let read = SeedRead {
            listings: strings(&["candle"]),
            documents: Vec::new(),
        };
        let entries = read.entries(1_700_000_000_000);
        assert_eq!(
            entries
                .iter()
                .map(|e| e.summary.as_str())
                .collect::<Vec<_>>(),
            [read.line().unwrap().as_str(), READING]
        );
        for e in &entries {
            assert_eq!((e.from_turn, e.to_turn), (0, 0));
            assert_eq!((e.from_ms, e.to_ms), (1_700_000_000_000, 1_700_000_000_000));
            assert_eq!(e.period, Period::Step);
            assert!(e.claims.is_empty());
        }
    }
}
