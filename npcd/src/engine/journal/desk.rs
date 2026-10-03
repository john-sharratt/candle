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

use crate::engine::journal::section::JournalPrompt;
use crate::engine::journal::workflow::Desk;
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
    async fn write(&mut self, spec: &ToolSpec, question: &str) -> anyhow::Result<String> {
        let tree = match &self.tree {
            Some(tree) => Arc::clone(tree),
            None => {
                let tree = self.minds.answer_tree(spec.clone())?;
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
