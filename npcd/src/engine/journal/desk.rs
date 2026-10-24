//! The engine-backed [`Desk`]: the draft's question put to the character itself.
//!
//! The write runs on the character's live conversation through
//! [`Minds::ask_under`], projected under [`AskScope::Conversation`]: the
//! character reads its own system prompt, its journal and its own turns, and
//! nothing it retrieved from elsewhere, so the entry is written from what it
//! lived and not from what it only had beside it. The stretch is already in its
//! cache and the question costs only its own tokens. Nothing the question or its
//! answer says is written back to that conversation; the entry reaches the
//! character only as a section of its journal, once it has been kept
//! ([`Minds::keep_journal`]).
//!
//! The answer is held to one call of those the draft offers — `journal_write`,
//! and `journal_skip` once the journal holds an entry. Their grammar depends on
//! the stretch (the `cite` enum holds that stretch's turns), so it is compiled
//! the first time the write is asked and reused for the rest of the draft,
//! whose every round offers the same calls.

use std::sync::Arc;

use candle_conversation::stencil::{StencilTree, ToolSpec};
use candle_conversation::AskScope;
use journal_core::section::JournalPrompt;
use journal_core::transcript::Filed;
use journal_core::workflow::Desk;
use serde_json::Value;

use crate::engine::mind::{Minds, Put};
use crate::engine::prompt::Persona;
use crate::engine::tools::Within;

/// Room for a brief entry: a few claims, what is intended, what is open.
pub const WRITE_TOKENS: usize = 1024;

/// Room for a fold: a one-line summary and a digest of a few paragraphs, which
/// runs well past an entry. An answer cut off by its budget does not parse.
pub const FOLD_TOKENS: usize = 3072;

pub struct EngineDesk<'a> {
    minds: &'a Minds,
    npc_id: u64,
    persona: &'a Persona<'a>,
    within: &'a Within,
    /// The most an answer may run to: [`WRITE_TOKENS`] or [`FOLD_TOKENS`].
    max_tokens: usize,
    tree: Option<Arc<StencilTree>>,
}

impl<'a> EngineDesk<'a> {
    pub fn new(
        minds: &'a Minds,
        npc_id: u64,
        persona: &'a Persona<'a>,
        within: &'a Within,
        max_tokens: usize,
    ) -> Self {
        Self {
            minds,
            npc_id,
            persona,
            within,
            max_tokens,
            tree: None,
        }
    }
}

/// The core's `{name, params}` forms, as the engine reads them.
fn tool_specs(specs: &[Value]) -> anyhow::Result<Vec<ToolSpec>> {
    specs
        .iter()
        .map(|spec| Ok(serde_json::from_value(spec.clone())?))
        .collect()
}

impl Desk for EngineDesk<'_> {
    async fn write(&mut self, specs: &[Value], question: &str) -> anyhow::Result<String> {
        let tree = match &self.tree {
            Some(tree) => Arc::clone(tree),
            None => {
                let tree = self.minds.answer_tree(&tool_specs(specs)?)?;
                self.tree = Some(Arc::clone(&tree));
                tree
            }
        };
        let put = Put {
            scope: AskScope::Conversation,
            tree,
            max_tokens: self.max_tokens,
        };
        self.minds
            .ask_under(self.npc_id, self.persona, self.within, question, put)
            .await
    }

    /// A character keeps no text behind its entries: none of its tools reads
    /// it back, so the stretch it was written from is not filed.
    async fn keep(
        &mut self,
        prompt: &JournalPrompt,
        _filed: Option<&Filed>,
        _ceased: &[u64],
    ) -> anyhow::Result<()> {
        self.minds.keep_journal(self.npc_id, prompt).await
    }
}

#[cfg(test)]
mod tests {
    use candle_conversation::stencil::{compile_tool_call_tree, ToolCallEnvelope};
    use journal_core::entry::Item;
    use journal_core::tools::{draft_specs, write_spec, SKIP, WRITE};

    use super::*;
    use crate::engine::act::{self, Rejected};
    use crate::engine::tools::CATALOG;

    fn item(id: u64, text: &str) -> Item {
        Item {
            id,
            text: text.into(),
        }
    }

    /// The core's `{name, params}` form, as the engine reads it.
    fn spec(citable: &[u64], open: &[Item]) -> ToolSpec {
        serde_json::from_value(write_spec(citable, open).unwrap()).unwrap()
    }

    #[test]
    fn the_journal_calls_are_not_ones_the_world_accepts() {
        for call in [WRITE, SKIP] {
            assert!(
                CATALOG.iter().all(|c| c.name != call),
                "`{call}` must not be in the world catalog: `act::parse` would accept it live"
            );
            let parsed = act::parse(&format!(r#"{{"tool": "{call}"}}"#));
            assert!(parsed.acts.is_empty(), "`{call}` became an act");
            assert!(
                matches!(&parsed.rejected[..], [Rejected::UnknownTool { tool }] if tool == call),
                "`{call}` was not refused as unknown: {:?}",
                parsed.rejected
            );
        }
    }

    /// Nothing is lost between the core's JSON and the engine's spec: the
    /// parameters, which are required, the minimum claim count, and every enum.
    #[test]
    fn the_engine_reads_the_write_call_as_the_core_wrote_it() {
        let write = spec(&[7, 9, 12], &[item(3, "the valve")]);
        assert_eq!(write.name, WRITE);
        let names: Vec<(&str, bool, usize)> = write
            .params
            .iter()
            .map(|p| (p.name.as_str(), p.required, p.min_items))
            .collect();
        assert_eq!(
            names,
            [
                ("claims", true, 2),
                ("summary", true, 0),
                ("intend", false, 0),
                ("open", false, 0)
            ]
        );
        let claim = write.params[0].items.as_ref().unwrap();
        let props = claim.properties.as_ref().unwrap();
        let cite = props.iter().find(|p| p.name == "cite").unwrap();
        assert!(cite.required);
        assert_eq!(
            cite.enum_values.as_deref(),
            Some(&["7", "9", "12"].map(String::from)[..])
        );
        let open = write.params[3].items.as_ref().unwrap();
        let relates = open
            .properties
            .as_ref()
            .unwrap()
            .iter()
            .find(|p| p.name == "relates")
            .unwrap();
        assert_eq!(
            relates.enum_values.as_deref().unwrap(),
            ["new", "restates 3", "resolves 3"]
        );
    }

    #[test]
    fn the_engine_reads_the_skip_beside_the_write() {
        let specs = tool_specs(&draft_specs(&[1, 2], &[], true).unwrap()).unwrap();
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, [WRITE, SKIP]);
        let reason = &specs[1].params[0];
        assert_eq!((reason.name.as_str(), reason.required), ("reason", true));
    }

    #[test]
    fn the_drafting_grammar_compiles_under_both_call_shapes() {
        let citable: Vec<u64> = (1..=16).collect();
        let open: Vec<Item> = (1..=6).map(|i| item(i, "something")).collect();
        let specs = tool_specs(&draft_specs(&citable, &open, true).unwrap()).unwrap();
        for env in [ToolCallEnvelope::qwen3(), ToolCallEnvelope::qwen35()] {
            compile_tool_call_tree(&specs, &env).expect("the drafting grammar compiles");
        }
    }

    /// The drafting grammar is built once per draft, with the largest span and
    /// the most open items a draft can have. It has to stay a size that is cheap
    /// to compile — an array level is unrolled 64 times, so what one element
    /// costs is multiplied.
    #[test]
    fn the_drafting_grammar_stays_small_at_its_largest() {
        let citable: Vec<u64> = (1..=64).collect();
        let open: Vec<Item> = (1..=6).map(|i| item(i, "something")).collect();
        let specs = tool_specs(&draft_specs(&citable, &open, true).unwrap()).unwrap();
        let tree = compile_tool_call_tree(&specs, &ToolCallEnvelope::qwen3()).unwrap();
        let nodes = tree.nodes.len();
        assert!(
            nodes < 10_000,
            "{nodes} nodes: an array element that costs this much, times 64 levels, is a compile \
             that is not worth a journal entry"
        );
    }
}
