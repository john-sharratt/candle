//! Putting the questions to one conversation.
//!
//! Each question is an unsealed turn ([`UnsealedAsker::ask`]): it projects the
//! conversation's own history exactly as a real turn would, decodes, and writes
//! nothing back — the timeline's turn count does not move, so asking cannot
//! itself damage what it is judging. The four are asked side by side; each runs
//! on its own ephemeral fork.
//!
//! The answer is forced: the reasoning block is written closed, and a grammar
//! admits exactly [`YES`] or [`NO`] and then ends the turn. Decoding is greedy,
//! so the same conversation gets the same verdict on every run.

use std::sync::Arc;

use candle_conversation::stencil::{StencilTree, StencilTreeBuilder};
use candle_conversation::{
    ConversationEngine, OptionalState, SamplingConfig, Sequence, SequenceConfig, TurnOptions,
    NO_THINK_SELECTOR,
};
use futures::future::try_join_all;

use super::questions::{Answer, Findings, NO, QUESTIONS, YES};

/// Answer tokens a turn may decode: the one arm, and nothing after it.
const MAX_ANSWER_TOKENS: usize = 4;

/// What every question is asked with.
pub struct Asker {
    grammar: Arc<StencilTree>,
    prefill: String,
    sampling: SamplingConfig,
}

impl Asker {
    pub fn new(engine: &ConversationEngine, config: &SequenceConfig) -> anyhow::Result<Self> {
        for arm in [YES, NO] {
            let n = engine
                .tokenizer()
                .encode(arm, false)
                .map(|e| e.len())
                .map_err(|e| anyhow::anyhow!("tokenizing {arm:?}: {e}"))?;
            anyhow::ensure!(
                n == 1,
                "the answer {arm:?} is {n} tokens in this vocabulary — a branch decides on \
                 the first token of each arm, so the arms must be single tokens"
            );
        }
        let spec = StencilTreeBuilder::new("self_check")
            .root("answer")
            .branch("answer", &[(YES, "done"), (NO, "done")])
            .end("done")
            .build()
            .map_err(|e| anyhow::anyhow!("building the yes/no grammar: {e}"))?;
        let grammar = Arc::new(engine.compile_stencil(&spec)?);
        let mut sampling = config.sampling.clone();
        sampling.temperature = 0.0;
        Ok(Self {
            grammar,
            prefill: config.dialect.no_think_block.to_string(),
            sampling,
        })
    }

    /// Ask `conversation` every question.
    pub async fn check(&self, conversation: &Sequence) -> anyhow::Result<Findings> {
        let asker = conversation.unsealed_asker();
        let mut selection = asker.selection();
        selection.set_optional(NO_THINK_SELECTOR, OptionalState::Present);
        let replies = try_join_all(QUESTIONS.iter().map(|q| {
            let options = TurnOptions {
                max_tokens: Some(MAX_ANSWER_TOKENS),
                sampling: Some(self.sampling.clone()),
                selection: selection.clone(),
                turn_grammar: Some(Arc::clone(&self.grammar)),
                assistant_prefill: Some(self.prefill.clone()),
                ..Default::default()
            };
            let asker = asker.clone();
            async move { asker.ask(q.text, options).await }
        }))
        .await?;
        let mut answers = [Answer::Yes; QUESTIONS.len()];
        for ((slot, reply), q) in answers.iter_mut().zip(&replies).zip(QUESTIONS) {
            *slot = Answer::parse(&reply.text, &self.prefill)
                .map_err(|e| anyhow::anyhow!("question {}: {e}", q.name))?;
        }
        Ok(Findings(answers))
    }
}
