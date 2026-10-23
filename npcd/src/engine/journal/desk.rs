//! The engine-backed [`Desk`]: the draft's question put to the character itself.
//!
//! The write runs on the character's live conversation through
//! [`Minds::ask_under`]: the character reads its own system prompt and the history
//! it reads when it acts, so the stretch is already in its cache and the question
//! costs only its own tokens. Nothing the question or its answer says is written
//! back to that conversation; the entry reaches the character only as a section of
//! its journal, once it has been kept ([`Minds::keep_journal`]).
//!
//! The write is held to one `journal_write` call: its grammar depends on the
//! stretch (the `cite` enum holds that stretch's turns), so it is compiled the
//! first time the write is asked and reused for the rest of the draft.

use std::sync::Arc;

use candle_conversation::stencil::{StencilTree, ToolSpec};
use journal_core::section::JournalPrompt;
use journal_core::workflow::Desk;
use serde_json::Value;

use crate::engine::mind::Minds;
use crate::engine::prompt::Persona;
use crate::engine::tools::Within;

/// Room for a brief entry: a few claims, what is intended, what is open.
const WRITE_TOKENS: usize = 1024;

pub struct EngineDesk<'a> {
    minds: &'a Minds,
    npc_id: u64,
    persona: &'a Persona<'a>,
    within: &'a Within,
    tree: Option<Arc<StencilTree>>,
}

impl<'a> EngineDesk<'a> {
    pub fn new(
        minds: &'a Minds,
        npc_id: u64,
        persona: &'a Persona<'a>,
        within: &'a Within,
    ) -> Self {
        Self {
            minds,
            npc_id,
            persona,
            within,
            tree: None,
        }
    }
}

impl Desk for EngineDesk<'_> {
    async fn write(&mut self, spec: &Value, question: &str) -> anyhow::Result<String> {
        let tree = match &self.tree {
            Some(tree) => Arc::clone(tree),
            None => {
                let spec: ToolSpec = serde_json::from_value(spec.clone())?;
                let tree = self.minds.answer_tree(spec)?;
                self.tree = Some(Arc::clone(&tree));
                tree
            }
        };
        self.minds
            .ask_under(
                self.npc_id,
                self.persona,
                self.within,
                question,
                tree,
                WRITE_TOKENS,
            )
            .await
    }

    async fn keep(&mut self, prompt: &JournalPrompt) -> anyhow::Result<()> {
        self.minds.keep_journal(self.npc_id, prompt).await
    }
}

#[cfg(test)]
mod tests {
    use candle_conversation::stencil::{compile_tool_call_tree, ToolCallEnvelope};
    use journal_core::entry::Item;
    use journal_core::tools::{write_spec, WRITE};

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
    fn the_journal_call_is_not_one_the_world_accepts() {
        assert!(
            CATALOG.iter().all(|c| c.name != WRITE),
            "`{WRITE}` must not be in the world catalog: `act::parse` would accept it live"
        );
    }

    #[test]
    fn a_live_turn_that_makes_the_journal_call_acts_on_nothing() {
        let parsed = act::parse(&format!(r#"{{"tool": "{WRITE}"}}"#));
        assert!(parsed.acts.is_empty(), "`{WRITE}` became an act");
        assert!(
            matches!(&parsed.rejected[..], [Rejected::UnknownTool { tool }] if tool == WRITE),
            "`{WRITE}` was not refused as unknown: {:?}",
            parsed.rejected
        );
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
                ("claims", true, 1),
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
        let open = write.params[2].items.as_ref().unwrap();
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
    fn the_writing_grammar_compiles_under_both_call_shapes() {
        let citable: Vec<u64> = (1..=16).collect();
        let open: Vec<Item> = (1..=6).map(|i| item(i, "something")).collect();
        for env in [ToolCallEnvelope::qwen3(), ToolCallEnvelope::qwen35()] {
            compile_tool_call_tree(&[spec(&citable, &open)], &env)
                .expect("the writing grammar compiles");
        }
    }

    /// The writing grammar is built once per draft, with the largest span and the
    /// most open items a draft can have. It has to stay a size that is cheap to
    /// compile — an array level is unrolled 64 times, so what one element costs
    /// is multiplied.
    #[test]
    fn the_writing_grammar_stays_small_at_its_largest() {
        let citable: Vec<u64> = (1..=64).collect();
        let open: Vec<Item> = (1..=6).map(|i| item(i, "something")).collect();
        let tree =
            compile_tool_call_tree(&[spec(&citable, &open)], &ToolCallEnvelope::qwen3()).unwrap();
        let nodes = tree.nodes.len();
        assert!(
            nodes < 10_000,
            "{nodes} nodes: an array element that costs this much, times 64 levels, is a compile \
             that is not worth a journal entry"
        );
    }
}
