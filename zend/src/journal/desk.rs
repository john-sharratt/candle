//! The engine-backed desk: a draft's questions put to the dialogue itself.
//!
//! Each question runs on an ephemeral fork of the dialogue's live slot
//! ([`UnsealedAsker`]) at the `Normal` band — beneath the user's own turns,
//! above the background layers. The fork reads the dialogue's own system prompt
//! and the history its last turn ran under, so the stretch is already in its
//! cache and a question costs only its own tokens. Nothing a question or its
//! answer says is written back: an entry reaches the dialogue only as a section
//! of its journal, once kept.
//!
//! Each answer is held to one call — `journal_gate` for the gate, the per-draft
//! `journal_write` for the write — compiled from the call's `{name, params}`
//! form the first time it is asked and reused for the rest of the draft.

use std::future::Future;
use std::sync::Arc;

use candle_conversation::projection::DecodePriority;
use candle_conversation::stencil::StencilTree;
use candle_conversation::{SamplingConfig, TurnOptions, UnsealedAsker};
use journal_core::section::JournalPrompt;
use journal_core::workflow::Desk;
use serde_json::Value;

use crate::journal::draft::Gatekeeper;
use crate::journal::gate;

/// Room for a brief entry: a few claims, what is intended, what is open.
const WRITE_TOKENS: usize = 1024;
/// Room for the gate's answer and its one-sentence reason.
const GATE_TOKENS: usize = 160;

/// Compiles a call's `{name, params}` form into the grammar one answer is held
/// to.
pub type Compile = Arc<dyn Fn(&Value) -> anyhow::Result<Arc<StencilTree>> + Send + Sync>;

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
    async fn ask(
        &self,
        question: &str,
        tree: Arc<StencilTree>,
        max_tokens: usize,
    ) -> anyhow::Result<String> {
        let options = TurnOptions {
            assistant_prefill: Some(self.prefill.clone()),
            turn_grammar: Some(tree),
            sampling: Some(self.sampling.clone()),
            max_tokens: Some(max_tokens),
            selection: self.asker.selection(),
            ..Default::default()
        };
        let response = self
            .asker
            .ask(question, DecodePriority::Normal, options)
            .await?;
        Ok(response.text)
    }
}

pub struct ZendDesk {
    asking: Asking,
    record: Record,
    write_tree: Option<Arc<StencilTree>>,
    gate_tree: Option<Arc<StencilTree>>,
}

impl ZendDesk {
    pub fn new(asking: Asking, record: Record) -> Self {
        Self {
            asking,
            record,
            write_tree: None,
            gate_tree: None,
        }
    }

    fn tree(
        compile: &Compile,
        held: &mut Option<Arc<StencilTree>>,
        spec: &Value,
    ) -> anyhow::Result<Arc<StencilTree>> {
        match held {
            Some(tree) => Ok(Arc::clone(tree)),
            None => {
                let tree = compile(spec)?;
                *held = Some(Arc::clone(&tree));
                Ok(tree)
            }
        }
    }
}

impl Desk for ZendDesk {
    async fn write(&mut self, spec: &Value, question: &str) -> anyhow::Result<String> {
        let tree = Self::tree(&self.asking.compile, &mut self.write_tree, spec)?;
        self.asking.ask(question, tree, WRITE_TOKENS).await
    }

    fn keep(&mut self, prompt: &JournalPrompt) -> impl Future<Output = anyhow::Result<()>> + Send {
        let kept = (self.record)(prompt);
        async move { kept }
    }
}

impl Gatekeeper for ZendDesk {
    async fn gate(&mut self, question: &str) -> anyhow::Result<String> {
        let tree = Self::tree(&self.asking.compile, &mut self.gate_tree, &gate::spec())?;
        self.asking.ask(question, tree, GATE_TOKENS).await
    }
}
