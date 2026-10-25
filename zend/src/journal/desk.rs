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
//! the draft, whose every round offers the same calls. A draft is one answer
//! ([`Desk::rounds`]).

use std::future::Future;
use std::sync::Arc;

use candle_conversation::projection::DecodePriority;
use candle_conversation::stencil::StencilTree;
use candle_conversation::{AskScope, SamplingConfig, TurnOptions, UnsealedAsker};
use journal_core::section::JournalPrompt;
use journal_core::transcript::Filed;
use journal_core::workflow::Desk;
use serde_json::Value;

/// Room for an entry: its claims, what is intended, what is open. An answer
/// that runs past it is read as far as it got
/// ([`journal_core::tools::parse`]).
pub const WRITE_TOKENS: usize = 2048;

/// Room for a fold: a one-line summary and a digest of a few paragraphs. A
/// month's digest runs well past an entry, and an answer cut off by its budget
/// does not parse.
pub const FOLD_TOKENS: usize = 3072;

/// Compiles the calls' `{name, params}` forms into the grammar one answer is
/// held to: exactly one call, of any of them.
pub type Compile = Arc<dyn Fn(&[Value]) -> anyhow::Result<Arc<StencilTree>> + Send + Sync>;

/// Files a stretch's turns, or a fold's digest, under its entry, ceases the
/// text of the entries a fold took, then writes the journal's record where a
/// restart reads it back.
pub type Record =
    Box<dyn FnMut(&JournalPrompt, Option<&Filed>, &[u64]) -> anyhow::Result<()> + Send>;

/// How every question of one draft is put.
pub struct Asking {
    pub asker: UnsealedAsker,
    pub compile: Compile,
    /// Opens the answer: the reasoning block written closed, then the call
    /// marker, so the grammar starts at the call.
    pub prefill: String,
    pub sampling: SamplingConfig,
    /// The most an answer may run to: [`WRITE_TOKENS`] for a draft,
    /// [`FOLD_TOKENS`] for a fold.
    pub max_tokens: usize,
}

impl Asking {
    async fn ask(&self, question: &str, tree: Arc<StencilTree>) -> anyhow::Result<String> {
        let options = TurnOptions {
            assistant_prefill: Some(self.prefill.clone()),
            turn_grammar: Some(tree),
            sampling: Some(self.sampling.clone()),
            max_tokens: Some(self.max_tokens),
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

    fn keep(
        &mut self,
        prompt: &JournalPrompt,
        filed: Option<&Filed>,
        ceased: &[u64],
    ) -> impl Future<Output = anyhow::Result<()>> + Send {
        let kept = (self.record)(prompt, filed, ceased);
        async move { kept }
    }

    /// One answer: a draft blocks whoever waits on the entry — a delegated
    /// task's caller above all — an answer asked again runs as long as the
    /// first, and one cut off is read as far as it got.
    fn rounds(&self) -> usize {
        1
    }
}
