//! What kind of work a worker does, and the document that tells it how
//! (`docs/claude_zend_gateway.md` §5.1).
//!
//! A task names its kind when it starts and a part takes its worker's. The
//! kind's document is carried as the first section of the conversation's
//! journal collection, ahead of the inherited journals and outside their
//! budget, so a long lineage never cuts it. The documents are compiled in from
//! `prompts/roles/`, one file a kind, and ship with the behaviour they describe.

use serde::{Deserialize, Serialize};

use super::lineage::Inherited;

/// The section name a worker's document is carried under.
pub const SECTION: &str = "lineage/role";

/// The most characters one document may run to: about 1,000 tokens. It is read
/// every round of every worker of its kind.
pub const MAX_DOCUMENT_CHARS: usize = 4_000;

/// The kind of work a task does: `coder` unless the call asks otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerKind {
    #[default]
    Coder,
    Researcher,
    Reviewer,
}

impl WorkerKind {
    /// The kind a call names, `coder` when it names none.
    pub fn of(name: Option<&str>) -> Result<Self, String> {
        match name {
            None | Some("coder") => Ok(Self::Coder),
            Some("researcher") => Ok(Self::Researcher),
            Some("reviewer") => Ok(Self::Reviewer),
            Some(other) => Err(format!(
                "`role` is `coder`, `researcher` or `reviewer`, not `{other}`"
            )),
        }
    }

    fn document(self) -> &'static str {
        match self {
            Self::Coder => include_str!("../prompts/roles/coder.md"),
            Self::Researcher => include_str!("../prompts/roles/researcher.md"),
            Self::Reviewer => include_str!("../prompts/roles/reviewer.md"),
        }
    }

    /// The section that carries this kind's document.
    pub fn section(self) -> Inherited {
        Inherited {
            name: SECTION.to_string(),
            text: self.document().trim_end().to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [WorkerKind; 3] = [
        WorkerKind::Coder,
        WorkerKind::Researcher,
        WorkerKind::Reviewer,
    ];

    /// **A task codes unless asked**, and a kind nobody defined is refused
    /// rather than leaving a worker with no document.
    #[test]
    fn a_call_names_a_kind_or_gets_coder() {
        assert_eq!(WorkerKind::of(None).unwrap(), WorkerKind::Coder);
        assert_eq!(
            ["coder", "researcher", "reviewer"].map(|k| WorkerKind::of(Some(k)).unwrap()),
            KINDS
        );
        assert_eq!(
            WorkerKind::of(Some("tester")).unwrap_err(),
            "`role` is `coder`, `researcher` or `reviewer`, not `tester`"
        );
    }

    /// The record holds the name the tool takes.
    #[test]
    fn the_kind_is_stored_by_its_name() {
        assert_eq!(
            serde_json::to_string(&WorkerKind::Reviewer).unwrap(),
            "\"reviewer\""
        );
    }

    /// Every document opens with its title and stays short: it is carried
    /// every round.
    #[test]
    fn every_document_is_short_and_titled() {
        for kind in KINDS {
            let section = kind.section();
            assert_eq!(section.name, "lineage/role");
            assert!(section.text.starts_with("# How a "), "{kind:?}");
            assert!(
                section.text.len() <= MAX_DOCUMENT_CHARS,
                "{kind:?} runs to {}",
                section.text.len()
            );
        }
    }
}
