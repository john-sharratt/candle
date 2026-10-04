//! Putting a question to a character and forcing an answer.
//!
//! The question runs on an ephemeral fork of the character's live slot
//! ([`Sequence::ask_unsealed`]), so it reuses the K/V and projection caches the
//! live slot already holds: the character reads its own system prompt and the
//! same selected history it reads when it acts, then answers under a one-call
//! grammar. Nothing is written back —
//! not the question, not the answer — so asking never changes what the character
//! remembers. The question is put between the character's turns and never inside
//! one, but it does not wait for the character's lock to arrange that: the
//! scheduler parks the fork until the turn in flight ends.

use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use serde_json::{json, Value};

use candle_conversation::stencil::{Param, StencilTree, ToolSpec};
use candle_conversation::TurnOptions;

use super::{compile_calls, Minds};
use crate::engine::act;
use crate::engine::identity::Deliberation;
use crate::engine::prompt::Persona;
use crate::engine::tools::Within;

/// The tool a question is answered through.
pub const ANSWER_TOOL: &str = "answer";

/// Enough for a paragraph of free-text answer and a reason of similar length.
const ANSWER_TOKENS: usize = 640;

/// What a character said when asked.
#[derive(Debug, Clone, Serialize)]
pub struct Answer {
    /// The value it gave: one of the offered choices, or free text.
    pub answer: String,
    /// Why, in its own words, when it gave one.
    pub reason: Option<String>,
    /// The decode as it came back, for when the parse is the thing in doubt.
    pub raw: String,
    /// Wall time from asking to the answer landing, waiting for the character's
    /// turn included.
    pub ms: u128,
}

/// The one tool a question is answered through: `answer`, constrained to
/// `choices` when there are any and free text when there are not, then a
/// `reason` in the character's own words. The reason is required: left optional
/// the grammar closes the call straight after the value and never offers one.
pub fn answer_spec(choices: &[String]) -> anyhow::Result<ToolSpec> {
    let value = if choices.is_empty() {
        json!({ "name": "value", "type": "string", "required": true })
    } else {
        json!({ "name": "value", "type": "string", "required": true, "enum": choices })
    };
    let reason = json!({ "name": "reason", "type": "string", "required": true });
    let params: Vec<Param> = serde_json::from_value(json!([value, reason]))?;
    Ok(ToolSpec {
        name: ANSWER_TOOL.to_string(),
        params,
    })
}

/// Read an answer out of a decode. `None` when the decode holds no `answer` call.
pub fn read_answer(raw: &str) -> Option<(String, Option<String>)> {
    let (_, args) = act::raw_calls(raw)
        .into_iter()
        .find(|(name, _)| name == ANSWER_TOOL)?;
    let text = |key: &str| match args.get(key) {
        Some(Value::String(s)) => Some(s.trim().to_string()),
        Some(other) => Some(other.to_string()),
        None => None,
    };
    let value = text("value")?;
    let reason = text("reason").filter(|r| !r.is_empty());
    Some((value, reason))
}

impl Minds {
    /// Compile `spec` into the one-call grammar a question is answered under.
    ///
    /// Errors when the checkpoint has no tool-call grammar to force an answer
    /// with.
    pub fn answer_tree(&self, spec: ToolSpec) -> anyhow::Result<Arc<StencilTree>> {
        if !self.grammar_ok {
            anyhow::bail!(
                "this checkpoint has no tool-call grammar, so an answer cannot be forced"
            );
        }
        let engine = self
            .engine
            .lock()
            .map_err(|_| anyhow::anyhow!("engine lock poisoned"))?;
        compile_calls(&engine, &self.base_config, &[spec], 1, None)
    }

    /// Ask `npc_id` a question and return what it answered.
    ///
    /// `choices` empty means free text. Errors when the character has no live
    /// conversation yet (nothing has been said to it to ask about) or the
    /// checkpoint has no grammar to force the answer with.
    pub async fn ask(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        within: &Within,
        question: &str,
        choices: &[String],
    ) -> anyhow::Result<Answer> {
        let started = Instant::now();
        let tree = self.answer_tree(answer_spec(choices)?)?;
        let text = self
            .ask_under(npc_id, persona, within, question, tree, ANSWER_TOKENS)
            .await?;
        let (answer, reason) = read_answer(&text)
            .ok_or_else(|| anyhow::anyhow!("no answer call in the reply: {}", text.trim()))?;
        Ok(Answer {
            answer,
            reason,
            raw: text,
            ms: started.elapsed().as_millis(),
        })
    }

    /// Put `question` to `npc_id` on its live conversation and return the raw
    /// decode, held to `tree` and cut at `max_tokens`.
    ///
    /// The caller reads the reply: it is the call `tree` was compiled from, as the
    /// dialect writes it.
    pub async fn ask_under(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        within: &Within,
        question: &str,
        tree: Arc<StencilTree>,
        max_tokens: usize,
    ) -> anyhow::Result<String> {
        let conversation = self
            .live
            .lock()
            .unwrap()
            .get(&npc_id)
            .map(Arc::clone)
            .ok_or_else(|| anyhow::anyhow!("npc {npc_id} has no live conversation yet"))?;

        let asker = self
            .askers
            .lock()
            .unwrap()
            .get(&npc_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("npc {npc_id} has no live conversation yet"))?;

        let prepared = Instant::now();
        // An idle character has its mission and journal brought current first. A
        // character mid-turn is not waited for: the question is read through the
        // selection its last turn ran under, and the fork is taken when that turn
        // ends.
        let (selection, was_idle) = match conversation.try_lock() {
            Ok(mut live) => (self.prepare(&mut live, npc_id, persona, within).await, true),
            Err(_) => (asker.selection(), false),
        };
        let prepare_ms = prepared.elapsed().as_millis();
        let options = TurnOptions {
            assistant_prefill: self.opening(Deliberation::None, true),
            turn_grammar: Some(tree),
            sampling: Some(self.sampling_for(Deliberation::None)),
            max_tokens: Some(max_tokens),
            selection,
            ..Default::default()
        };
        let asked = Instant::now();
        let response = asker.ask(question, options).await?;
        let ask_ms = asked.elapsed().as_millis();
        let stats = &response.stats;
        tracing::info!(
            npc_id,
            was_idle,
            prepare_ms,
            ask_ms,
            fork_wait_ms = ask_ms.saturating_sub(stats.total_ms as u128),
            prefill_ms = stats.prefill_ms as u64,
            prefill_tokens = stats.turn_prefill_tokens,
            decode_ms = stats.decode_ms as u64,
            tokens = stats.tokens_generated,
            tokens_per_second = stats.tokens_per_second as u64,
            context_tokens = stats.context_tokens,
            "asked on a fork of the live slot"
        );
        Ok(response.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answer_spec_with_choices_is_an_enum_then_an_optional_reason() {
        let spec = answer_spec(&["yes".into(), "no".into()]).unwrap();
        assert_eq!(spec.name, "answer");
        assert_eq!(spec.params.len(), 2);
        assert_eq!(spec.params[0].name, "value");
        assert!(spec.params[0].required);
        assert_eq!(
            spec.params[0].enum_values.as_deref(),
            Some(&["yes".to_string(), "no".to_string()][..])
        );
        assert!(spec.params[1].required);
        assert!(spec.params[1].enum_values.is_none());
    }

    #[test]
    fn answer_spec_without_choices_is_free_text() {
        let spec = answer_spec(&[]).unwrap();
        assert!(spec.params[0].enum_values.is_none());
    }

    #[test]
    fn reads_the_answer_and_reason_out_of_a_call() {
        let raw = r#"<think>

</think>

<tool_call>
{"name": "answer", "arguments": {"value": "lost", "reason": "I do not know this room."}}
</tool_call>"#;
        let (value, reason) = read_answer(raw).expect("an answer");
        assert_eq!(value, "lost");
        assert_eq!(reason.as_deref(), Some("I do not know this room."));
    }

    #[test]
    fn an_answer_without_a_reason_has_none() {
        let raw = r#"<tool_call>
{"name": "answer", "arguments": {"value": "yes"}}
</tool_call>"#;
        assert_eq!(read_answer(raw), Some(("yes".to_string(), None)));
    }

    #[test]
    fn a_reply_with_no_answer_call_reads_as_none() {
        assert_eq!(read_answer("I am fine."), None);
    }
}
