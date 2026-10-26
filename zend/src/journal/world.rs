//! The journals above a conversation, as its claims may rest on them.
//!
//! A conversation owned through `/mcp` inherits the journals above it — its
//! thread's notes, its project's rules, the worker that started it
//! (`docs/claude_zend_gateway.md` §5.2). A claim that rests on one of their
//! entries names it in `per`, and the call offers exactly the entries they
//! hold, by address (`T:review#4`), so a claim cannot rest on an entry that is
//! not there or not an ancestor's.
//!
//! zend asks the world nothing else: a coding session's claims are about what
//! files say and what results showed, which no check can read, so its journal
//! offers no typed facts ([`journal_core::style::Style::facts`]).

use std::collections::BTreeSet;

use journal_core::verify::World;

/// A journal above the conversation, as a citation names it: its path, as the
/// conversation's prompt heads it, and the entries a citation may name —
/// those it holds, and those a fold took, whose addresses still read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Citable {
    pub path: String,
    pub entries: Vec<u64>,
}

/// The journals above one conversation, nearest first.
pub struct JournalsAbove(pub Vec<Citable>);

impl World for JournalsAbove {
    fn read(&self, _subject: &str, _attribute: &str) -> Option<String> {
        None
    }

    /// Each entry above by its address, `<path>#<n>`, once.
    fn entries(&self) -> Vec<String> {
        let mut seen = BTreeSet::new();
        self.0
            .iter()
            .flat_map(|a| a.entries.iter().map(move |n| format!("{}#{n}", a.path)))
            .filter(|address| seen.insert(address.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use candle_conversation::stencil::{compile_tool_call_tree, ToolCallEnvelope, ToolSpec};
    use journal_core::entry::Item;
    use journal_core::tools::{draft_specs, Claims, WRITE};
    use journal_core::verify::MAX_CLAIMS;

    use super::*;

    /// The drafting grammar's node count at its largest stretch, with `n`
    /// entries above to rest on.
    fn nodes_with(n: usize) -> usize {
        let citable: Vec<u64> = (1..=64).collect();
        let open: Vec<Item> = (1..=6)
            .map(|id| Item {
                id,
                text: "something".into(),
            })
            .collect();
        let entries: Vec<String> = (1..=n).map(|i| format!("T:review-layers#{i}")).collect();
        let claims = Claims {
            facts: false,
            entries: &entries,
        };
        let specs: Vec<ToolSpec> = draft_specs(&citable, &open, true, claims)
            .unwrap()
            .into_iter()
            .map(|s| serde_json::from_value(s).unwrap())
            .collect();
        compile_tool_call_tree(&specs, &ToolCallEnvelope::qwen3())
            .unwrap()
            .nodes
            .len()
    }

    /// **The grammar does not grow with the lineage's history**: `per` is one
    /// choice whose values are a table, so a thread of 1,600 entries above
    /// compiles to the same nodes as one of 10 (measured: 155 nodes with no
    /// `per`, 179 with one at any size — each list unrolled to its own most,
    /// not the engine's bound of 64).
    #[test]
    fn the_grammar_stays_the_same_size_however_many_entries_are_above() {
        let few = nodes_with(10);
        assert_eq!(nodes_with(1_600), few);
        assert_eq!((nodes_with(0), few), (155, 179), "measured node counts");
    }

    /// **The grammar zend compiles closes the claims at the check's most**: the
    /// flat spec's `max_items` reaches the engine, so a writer cannot go past
    /// what `journal_core::verify` keeps and lose its entry for it.
    #[test]
    fn the_claims_close_at_what_the_check_keeps() {
        let claims = Claims {
            facts: false,
            entries: &[],
        };
        let specs: Vec<ToolSpec> = draft_specs(&[1, 2, 3], &[], true, claims)
            .unwrap()
            .into_iter()
            .map(|s| serde_json::from_value(s).unwrap())
            .collect();
        let write = specs.iter().find(|s| s.name == WRITE).unwrap();
        let listed = write.params.iter().find(|p| p.name == "claims").unwrap();
        assert_eq!(listed.max_items, Some(MAX_CLAIMS));
    }

    /// **A claim may rest on exactly the entries above**, each by its
    /// journal's path, nearest journal first; nothing is read as a fact.
    #[test]
    fn the_entries_above_are_offered_by_address_and_nothing_else_is_read() {
        let above = JournalsAbove(vec![
            Citable {
                path: "T:review/verify".into(),
                // A fold's entry that the journal also still holds is offered
                // once.
                entries: vec![2, 3, 3],
            },
            Citable {
                path: "T:review".into(),
                entries: vec![1, 12],
            },
        ]);
        assert_eq!(
            above.entries(),
            [
                "T:review/verify#2",
                "T:review/verify#3",
                "T:review#1",
                "T:review#12"
            ]
        );
        assert_eq!(above.read("file:r/a.txt", "exists"), None);
        assert!(JournalsAbove(Vec::new()).entries().is_empty());
    }
}
