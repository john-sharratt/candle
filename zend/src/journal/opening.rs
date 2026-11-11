//! The journal's opening: what a dialogue had read before its session began,
//! then that the user asked their opening question.
//!
//! A dialogue descends from the priming chain — every repository's root
//! listing and its anchor documents — and from the working set's files, and
//! all of it stands ahead of the user's first message as the dialogue's own
//! reading. Its own entries say nothing of that, so a dialogue asked what it
//! had read went and read it again. The opening says it plainly, in the
//! journal the dialogue already answers "what have I done" from, and its first
//! line is built from the chain the member last primed: a repository or
//! anchor added or gone changes the text, and a carried section whose text
//! changed is sealed again.
//!
//! Neither line is numbered. An entry's number is what `journal_read` opens,
//! and these two have no turns behind them to open.

use std::time::Duration;

use journal_core::section::Section;

/// How long a round that would go without the opening waits for it to seal.
/// It is two short lines, so a seal past this is a GPU queue the round should
/// not stand behind: it answers without the opening, and the next round has it.
pub const OPENING_SEAL_WAIT: Duration = Duration::from_secs(2);

/// The section the chain's reading is carried under.
pub const SEED_SECTION: &str = "journal/seed";

/// The section the rest of the opening is carried under.
pub const READING_SECTION: &str = "journal/reading";

/// The opening's second line.
pub const READING: &str = "Then — I read other files that seemed relevant to the work here, \
and the user asked their opening question of this session, which I must answer.";

/// What the priming chain read, in chain order: each repository whose root
/// listing it read, and each anchor document, as `{repo}/{path}`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SeedRead {
    pub listings: Vec<String>,
    pub documents: Vec<String>,
}

impl SeedRead {
    /// The opening's first line, or `None` when the chain read nothing.
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
                "Before this session — I read the workspace: {}.",
                read.join("; ")
            )
        })
    }

    /// The opening's sections, in order: both lines, or none when the chain
    /// read nothing.
    pub fn sections(&self) -> Vec<Section> {
        let Some(line) = self.line() else {
            return Vec::new();
        };
        vec![
            Section {
                name: SEED_SECTION.to_string(),
                text: line,
            },
            Section {
                name: READING_SECTION.to_string(),
                text: READING.to_string(),
            },
        ]
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

    /// **The first line names what the chain read** — the listings, then the
    /// documents, in chain order, by the paths the file tools take.
    #[test]
    fn the_first_line_names_every_listing_and_document_the_chain_read() {
        let read = SeedRead {
            listings: strings(&["candle", "battle-cities", "yenbot"]),
            documents: strings(&["candle/README.md", "candle/AGENTS.md", "yenbot/README.md"]),
        };
        assert_eq!(
            read.line().unwrap(),
            "Before this session — I read the workspace: the root listings of `candle`, \
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
            "Before this session — I read the workspace: the root listing of `candle`; \
             `candle/README.md`."
        );
        let listings_only = SeedRead {
            documents: Vec::new(),
            ..read
        };
        assert_eq!(
            listings_only.line().unwrap(),
            "Before this session — I read the workspace: the root listing of `candle`."
        );
    }

    /// **Both lines, in order, or none.** A chain that read nothing has no
    /// opening: there is nothing the dialogue read before it began.
    #[test]
    fn the_opening_is_both_lines_or_none() {
        assert!(SeedRead::default().sections().is_empty());
        let read = SeedRead {
            listings: strings(&["candle"]),
            documents: Vec::new(),
        };
        let sections = read.sections();
        assert_eq!(
            sections.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [SEED_SECTION, READING_SECTION]
        );
        assert_eq!(sections[1].text, READING);
    }
}
