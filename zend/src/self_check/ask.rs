//! Putting the questions about one conversation's record.
//!
//! Each question is an unsealed turn ([`UnsealedAsker::ask`]) on the dialogue
//! base: an ephemeral fork of a conversation with no history of its own, so the
//! record the question carries is the only conversation in view. Nothing is
//! written back, so asking cannot itself damage what it is judging. A
//! conversation's questions are asked side by side; each runs on its own fork.
//!
//! The answer is forced: the reasoning block is written closed and a grammar
//! admits exactly [`YES`] or [`NO`] as the first token. A grammar's end node
//! hands decoding back to the model rather than closing the turn, so the model
//! goes on to give its reason ("No — the final reply summarises README.md");
//! [`Answer::parse`] reads the verdict from the arm alone and the reason is
//! kept for the report.
//!
//! [`Answer::parse`]: super::questions::Answer::parse

use std::sync::Arc;

use candle_conversation::stencil::{StencilTree, StencilTreeBuilder};
use candle_conversation::{
    ConversationEngine, SamplingConfig, SequenceConfig, TurnOptions, UnsealedAsker,
};
use futures::future::try_join_all;

use super::questions::{reply_text, Answer, Findings, Subject, Verdict, NO, YES};

/// The turn's decode budget: the arm, then a sentence of the model's own reason
/// for it, which the report carries beside each failed question. Each arm is a
/// single token ([`Asker::new`] refuses a vocabulary where it is not).
const MAX_ANSWER_TOKENS: usize = 48;

/// What every question is asked with.
pub struct Asker {
    base: UnsealedAsker,
    grammar: Arc<StencilTree>,
    prefill: String,
    sampling: SamplingConfig,
}

impl Asker {
    /// Questions put through `base`, an asker on a conversation with no history
    /// of its own.
    pub fn new(
        engine: &ConversationEngine,
        config: &SequenceConfig,
        base: UnsealedAsker,
    ) -> anyhow::Result<Self> {
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
            base,
            grammar,
            prefill: config.dialect.no_think_block.to_string(),
            sampling,
        })
    }

    /// Ask the questions for `subject` about `record`.
    ///
    /// Under the base's own selection, so a question projects exactly the
    /// pieces the base already holds plus itself, and its fork keeps the whole
    /// placed prefix. The reasoning block needs no selector to stay shut: the
    /// prefill writes it closed.
    pub async fn check(&self, subject: &Subject, record: &str) -> anyhow::Result<Findings> {
        let selection = self.base.selection();
        let questions = subject.questions(record);
        let replies = try_join_all(questions.iter().map(|q| {
            let options = TurnOptions {
                max_tokens: Some(MAX_ANSWER_TOKENS),
                sampling: Some(self.sampling.clone()),
                selection: selection.clone(),
                turn_grammar: Some(Arc::clone(&self.grammar)),
                assistant_prefill: Some(self.prefill.clone()),
                ..Default::default()
            };
            let asker = self.base.clone();
            async move { asker.ask(&q.text, options).await }
        }))
        .await?;
        let answers = questions
            .iter()
            .zip(&replies)
            .map(|(q, reply)| {
                Answer::parse(&reply.text, &self.prefill)
                    .map(|answer| Verdict {
                        question: q.name,
                        answer,
                        reply: reply_text(&reply.text, &self.prefill),
                    })
                    .map_err(|e| anyhow::anyhow!("question {}: {e}", q.name))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Findings(answers))
    }
}
