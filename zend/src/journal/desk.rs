//! The engine-backed desk: a draft's question put to the dialogue itself.
//!
//! Each question runs on an ephemeral fork of the dialogue's live slot
//! ([`UnsealedAsker`]) at the `Normal` band — beneath the user's own turns,
//! above the background layers — projected under [`AskScope::Conversation`]:
//! the dialogue's system prompt, its journal and its own turns, and from other
//! layers only the working-set members it was served. What it retrieved from
//! elsewhere is left out, so the entry is written from the session and not from
//! material the session only had beside it. Nothing a question or its answer
//! says is written back: an entry reaches the dialogue only as a section of its
//! journal, once kept.
//!
//! The answer is held to one call of those the draft offers — `journal_write`,
//! and `journal_skip` once the journal holds an entry — compiled from their
//! `{name, params}` form the first time it is asked and reused for the rest of
//! the draft, whose every round offers the same calls.

use std::future::Future;
use std::sync::Arc;

use candle_conversation::projection::DecodePriority;
use candle_conversation::stencil::StencilTree;
use candle_conversation::{AskScope, SamplingConfig, TurnOptions, UnsealedAsker};
use journal_core::section::JournalPrompt;
use journal_core::workflow::Desk;
use serde_json::Value;

/// Room for a brief entry: a few claims, what is intended, what is open.
const WRITE_TOKENS: usize = 1024;

/// Compiles the calls' `{name, params}` forms into the grammar one answer is
/// held to: exactly one call, of any of them.
pub type Compile = Arc<dyn Fn(&[Value]) -> anyhow::Result<Arc<StencilTree>> + Send + Sync>;

/// Writes the journal's record where a restart reads it back.
pub type Record = Box<dyn FnMut(&JournalPrompt) -> anyhow::Result<()> + Send>;

/// How every question of one draft is put.
pub struct Asking {
    pub asker: UnsealedAsker,
    pub compile: Compile,
    /// Opens the answer: the reasoning block written closed, then the call
    /// marker, so the grammar starts at the call.
    pub prefill: String,
    pub sampling: SamplingConfig,
}

impl Asking {
    async fn ask(&self, question: &str, tree: Arc<StencilTree>) -> anyhow::Result<String> {
        let options = TurnOptions {
            assistant_prefill: Some(self.prefill.clone()),
            turn_grammar: Some(tree),
            sampling: Some(self.sampling.clone()),
            max_tokens: Some(WRITE_TOKENS),
            selection: self.asker.selection(),
            ..Default::default()
        };
        let response = self
            .asker
            .ask(
                question,
                DecodePriority::Normal,
                AskScope::Conversation,
                options,
            )
            .await?;
        Ok(response.text)
    }
}

pub struct ZendDesk {
    asking: Asking,
    record: Record,
    tree: Option<Arc<StencilTree>>,
}

impl ZendDesk {
    pub fn new(asking: Asking, record: Record) -> Self {
        Self {
            asking,
            record,
            tree: None,
        }
    }
}

impl Desk for ZendDesk {
    async fn write(&mut self, specs: &[Value], question: &str) -> anyhow::Result<String> {
        let tree = match &self.tree {
            Some(tree) => Arc::clone(tree),
            None => {
                let tree = (self.asking.compile)(specs)?;
                self.tree = Some(Arc::clone(&tree));
                tree
            }
        };
        let answer = self.asking.ask(question, tree).await?;
        tracing::info!(
            target: "zend::journal",
            question = ?question,
            answer = ?answer,
            "journal draft answered",
        );
        Ok(answer)
    }

    fn keep(&mut self, prompt: &JournalPrompt) -> impl Future<Output = anyhow::Result<()>> + Send {
        let kept = (self.record)(prompt);
        async move { kept }
    }
}
