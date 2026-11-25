//! A trial turn's request and answer: one turn put to a suspended character,
//! answered as one call under the grammar the caller supplies, and nothing of
//! it kept (`docs/npc_trials.md`).
//!
//! The body is checked here, before anything reaches the engine, so a trial
//! the grammar could not be built for is refused with what was wrong with it.
//! The call is read back out of the decode with the same specs the grammar was
//! compiled from.

use std::collections::HashSet;

use candle_conversation::stencil::{Param, ToolSpec};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::engine::act;
use crate::engine::prose::Exchange;

/// A trial's decode cap when its body names none: a call with a paragraph of
/// free text in it, as an `ask`'s answer is.
pub const DEFAULT_MAX_TOKENS: usize = 640;

/// The largest decode a trial may ask for.
pub const MAX_TOKENS: usize = 4096;

/// The most exchanges a given frame may lay down before its turn: each is a
/// prefill on the engine every other character shares.
pub const MAX_HISTORY: usize = 16;

/// The most text a trial may send, its turn, system prompt and history
/// together, in bytes.
pub const MAX_TEXT_BYTES: usize = 256 * 1024;

/// The most tools a trial may answer through.
pub const MAX_TOOLS: usize = 16;

/// The highest temperature a trial may decode at.
pub const MAX_TEMPERATURE: f32 = 2.0;

/// Why a trial was not answered.
#[derive(Debug)]
pub enum TrialRefused {
    /// The engine has not loaded yet.
    NotLoaded,
    /// The trial's tools make no grammar the engine can hold an answer to.
    BadTools(String),
    /// The engine took the trial and failed it.
    Failed(anyhow::Error),
}

impl From<anyhow::Error> for TrialRefused {
    fn from(e: anyhow::Error) -> Self {
        TrialRefused::Failed(e)
    }
}

/// What a trial reads under in place of the character's own frame.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GivenFrame {
    /// The system prompt.
    pub system: String,
    /// Completed exchanges laid down before the turn, oldest first.
    #[serde(default)]
    pub history: Vec<Exchange>,
}

/// The body of `POST /v1/npc/:nid/trial`, as it arrives.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialBody {
    turn: String,
    tools: Vec<ToolSpec>,
    #[serde(default)]
    frame: Option<GivenFrame>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
}

/// A trial whose body has been checked.
#[derive(Clone, Debug)]
pub struct Trial {
    /// The user message, verbatim.
    pub turn: String,
    /// The calls the answer may be: exactly one, of one of these.
    pub tools: Vec<ToolSpec>,
    /// `None` reads the character's own frame (`docs/npc_trials.md` §2).
    pub frame: Option<GivenFrame>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub seed: u64,
}

impl TrialBody {
    /// The trial this body asks for, or what is wrong with it.
    pub fn check(self) -> Result<Trial, String> {
        if self.turn.trim().is_empty() {
            return Err("a trial needs a `turn` to read".to_string());
        }
        if self.tools.is_empty() {
            return Err("a trial needs at least one tool in `tools` to answer through".to_string());
        }
        if self.tools.len() > MAX_TOOLS {
            return Err(format!(
                "a trial answers through at most {MAX_TOOLS} tools, not {}",
                self.tools.len()
            ));
        }
        let mut names = HashSet::new();
        for tool in &self.tools {
            if tool.name.trim().is_empty() {
                return Err("every tool in `tools` needs a name".to_string());
            }
            if !names.insert(tool.name.as_str()) {
                return Err(format!("two tools are named `{}`", tool.name));
            }
        }
        let mut bytes = self.turn.len() + self.tools.iter().map(spec_bytes).sum::<usize>();
        if let Some(frame) = &self.frame {
            if frame.system.trim().is_empty() {
                return Err("a given `frame` needs a `system` prompt".to_string());
            }
            if frame.history.len() > MAX_HISTORY {
                return Err(format!(
                    "a given `frame` lays down at most {MAX_HISTORY} exchanges, not {}",
                    frame.history.len()
                ));
            }
            bytes += frame.system.len();
            bytes += frame
                .history
                .iter()
                .map(|e| e.user.len() + e.assistant.len())
                .sum::<usize>();
        }
        if bytes > MAX_TEXT_BYTES {
            return Err(format!(
                "a trial sends at most {MAX_TEXT_BYTES} bytes of text, not {bytes}"
            ));
        }
        let max_tokens = self.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        if !(1..=MAX_TOKENS).contains(&max_tokens) {
            return Err(format!(
                "`max_tokens` is 1 to {MAX_TOKENS}, not {max_tokens}"
            ));
        }
        let temperature = self.temperature.unwrap_or(0.0);
        if !temperature.is_finite() || !(0.0..=MAX_TEMPERATURE).contains(&temperature) {
            return Err(format!(
                "`temperature` is 0 to {MAX_TEMPERATURE}, not {temperature}"
            ));
        }
        Ok(Trial {
            turn: self.turn,
            tools: self.tools,
            frame: self.frame,
            max_tokens,
            temperature,
            seed: self.seed.unwrap_or(0),
        })
    }
}

/// The text a tool's spec holds: its name, and every parameter's name, enum
/// values and pattern, nested ones included — what its grammar is compiled
/// from.
fn spec_bytes(tool: &ToolSpec) -> usize {
    tool.name.len() + tool.params.iter().map(param_bytes).sum::<usize>()
}

fn param_bytes(p: &Param) -> usize {
    p.name.len()
        + p.enum_values
            .iter()
            .flatten()
            .map(String::len)
            .sum::<usize>()
        + p.pattern.as_ref().map_or(0, String::len)
        + p.items.as_deref().map_or(0, param_bytes)
        + p.properties
            .iter()
            .flatten()
            .map(param_bytes)
            .sum::<usize>()
        + p.shapes
            .iter()
            .map(|(v, ps)| v.len() + ps.iter().map(param_bytes).sum::<usize>())
            .sum::<usize>()
}

/// The call a trial was answered with.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Call {
    pub name: String,
    pub arguments: Map<String, Value>,
}

/// What a trial came back with.
#[derive(Clone, Debug, Serialize)]
pub struct TrialAnswer {
    /// The decode as it came back.
    pub raw: String,
    /// The call in it, or `None` for a decode cut off before one closed.
    pub call: Option<Call>,
    /// The context the turn was read in, in tokens.
    pub prompt_tokens: usize,
    /// Tokens decoded.
    pub tokens: usize,
    /// Wall time from receipt to answer.
    pub ms: u128,
}

/// The first call in `raw` to one of `tools`, read with the specs the
/// grammar was compiled from.
pub fn read_call(raw: &str, tools: &[ToolSpec]) -> Option<Call> {
    act::raw_calls(raw, tools)
        .into_iter()
        .find(|(name, _)| tools.iter().any(|t| t.name == *name))
        .map(|(name, arguments)| Call { name, arguments })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn body(v: Value) -> Result<Trial, String> {
        serde_json::from_value::<TrialBody>(v)
            .map_err(|e| e.to_string())?
            .check()
    }

    fn answer_tool() -> Value {
        json!({ "name": "answer", "params": [
            { "name": "value", "type": "string", "required": true, "enum": ["north", "south"] },
            { "name": "reason", "type": "string", "required": true }
        ]})
    }

    /// **A bare trial reads the character's own frame, greedily**: no frame,
    /// the default cap, zero temperature, seed zero.
    #[test]
    fn a_bare_trial_reads_the_own_frame_greedily() {
        let t = body(json!({ "turn": "Where is the gate?", "tools": [answer_tool()] })).unwrap();
        assert_eq!(t.turn, "Where is the gate?");
        assert_eq!(t.tools.len(), 1);
        assert_eq!(t.tools[0].name, "answer");
        assert_eq!(
            t.tools[0].params[0].enum_values.as_deref(),
            Some(&["north".to_string(), "south".to_string()][..])
        );
        assert_eq!(t.frame, None);
        assert_eq!(t.max_tokens, 640);
        assert_eq!(t.temperature, 0.0);
        assert_eq!(t.seed, 0);
    }

    /// A given frame carries its system prompt and its history in order.
    #[test]
    fn a_given_frame_carries_its_history_in_order() {
        let t = body(json!({
            "turn": "And now?",
            "tools": [answer_tool()],
            "frame": { "system": "You read maps.", "history": [
                { "user": "first", "assistant": "one" },
                { "user": "second", "assistant": "two" }
            ]},
            "max_tokens": 64,
            "temperature": 0.7,
            "seed": 42
        }))
        .unwrap();
        assert_eq!(
            t.frame,
            Some(GivenFrame {
                system: "You read maps.".to_string(),
                history: vec![
                    Exchange {
                        user: "first".to_string(),
                        assistant: "one".to_string()
                    },
                    Exchange {
                        user: "second".to_string(),
                        assistant: "two".to_string()
                    },
                ],
            })
        );
        assert_eq!((t.max_tokens, t.temperature, t.seed), (64, 0.7, 42));
    }

    /// Every malformed body is refused with what was wrong with it.
    #[test]
    fn a_malformed_trial_is_refused_with_why() {
        let tools = || json!([answer_tool()]);
        let cases = [
            (
                json!({ "turn": "  ", "tools": tools() }),
                "a trial needs a `turn` to read",
            ),
            (
                json!({ "turn": "x", "tools": [] }),
                "a trial needs at least one tool in `tools` to answer through",
            ),
            (
                json!({ "turn": "x", "tools": [answer_tool(), answer_tool()] }),
                "two tools are named `answer`",
            ),
            (
                json!({ "turn": "x", "tools": [{ "name": " " }] }),
                "every tool in `tools` needs a name",
            ),
            (
                json!({ "turn": "x", "tools": tools(), "frame": { "system": "" } }),
                "a given `frame` needs a `system` prompt",
            ),
            (
                json!({ "turn": "x", "tools": tools(), "max_tokens": 0 }),
                "`max_tokens` is 1 to 4096, not 0",
            ),
            (
                json!({ "turn": "x", "tools": tools(), "max_tokens": 4097 }),
                "`max_tokens` is 1 to 4096, not 4097",
            ),
            (
                json!({ "turn": "x", "tools": tools(), "temperature": -0.5 }),
                "`temperature` is 0 to 2, not -0.5",
            ),
            (
                json!({ "turn": "x", "tools": tools(), "temperature": 3.0 }),
                "`temperature` is 0 to 2, not 3",
            ),
        ];
        for (v, why) in cases {
            assert_eq!(body(v).unwrap_err(), why);
        }
    }

    /// A field the route does not know is refused rather than ignored, at every
    /// level: a misspelt `frame` would run on the character's own, a misspelt
    /// `history` with no priming, a misspelt exchange field with half of one.
    #[test]
    fn an_unknown_field_is_refused_at_every_level() {
        let e = body(json!({ "turn": "x", "tools": [answer_tool()], "fram": {} })).unwrap_err();
        assert!(e.starts_with("unknown field `fram`"), "{e}");
        let e = body(json!({
            "turn": "x", "tools": [answer_tool()],
            "frame": { "system": "S", "histroy": [] }
        }))
        .unwrap_err();
        assert!(e.starts_with("unknown field `histroy`"), "{e}");
        let e = body(json!({
            "turn": "x", "tools": [answer_tool()],
            "frame": { "system": "S", "history": [{ "user": "u", "asistant": "a" }] }
        }))
        .unwrap_err();
        assert!(e.starts_with("unknown field `asistant`"), "{e}");
    }

    /// A trial too long for the shared engine is refused before it reaches it.
    #[test]
    fn a_trial_past_its_bounds_is_refused() {
        let history: Vec<Value> = (0..17)
            .map(|_| json!({ "user": "u", "assistant": "a" }))
            .collect();
        assert_eq!(
            body(json!({
                "turn": "x", "tools": [answer_tool()],
                "frame": { "system": "S", "history": history }
            }))
            .unwrap_err(),
            "a given `frame` lays down at most 16 exchanges, not 17"
        );
        // The tool's own text counts: "answer", "value", "north", "south" and
        // "reason" are 27 bytes, the turn 1.
        let long = "x".repeat(MAX_TEXT_BYTES - 28);
        assert!(body(json!({
            "turn": "y", "tools": [answer_tool()],
            "frame": { "system": long }
        }))
        .is_ok());
        let long = "x".repeat(MAX_TEXT_BYTES - 27);
        assert_eq!(
            body(json!({
                "turn": "y", "tools": [answer_tool()],
                "frame": { "system": long }
            }))
            .unwrap_err(),
            "a trial sends at most 262144 bytes of text, not 262145"
        );
        let many: Vec<Value> = (0..17)
            .map(|i| json!({ "name": format!("t{i}"), "params": [] }))
            .collect();
        assert_eq!(
            body(json!({ "turn": "x", "tools": many })).unwrap_err(),
            "a trial answers through at most 16 tools, not 17"
        );
    }

    fn specs() -> Vec<ToolSpec> {
        serde_json::from_value(json!([
            answer_tool(),
            { "name": "move", "params": [{ "name": "to", "type": "string", "required": true }] }
        ]))
        .unwrap()
    }

    /// **The call is read back by its own specs**, whichever of them it is.
    #[test]
    fn the_call_is_read_back_whichever_tool_it_is() {
        let raw =
            "<tool_call>\n{\"name\": \"move\", \"arguments\": {\"to\": \"12,40\"}}\n</tool_call>";
        assert_eq!(
            read_call(raw, &specs()),
            Some(Call {
                name: "move".to_string(),
                arguments: json!({ "to": "12,40" }).as_object().unwrap().clone(),
            })
        );
        let raw = "<think>\n\n</think>\n\n<tool_call>\n{\"name\": \"answer\", \"arguments\": \
                   {\"value\": \"north\", \"reason\": \"The gate is above me.\"}}\n</tool_call>";
        assert_eq!(
            read_call(raw, &specs()),
            Some(Call {
                name: "answer".to_string(),
                arguments: json!({ "value": "north", "reason": "The gate is above me." })
                    .as_object()
                    .unwrap()
                    .clone(),
            })
        );
    }

    /// A decode with no call in it, or only a call to a tool the trial did
    /// not offer, reads as none.
    #[test]
    fn a_decode_without_an_offered_call_reads_as_none() {
        assert_eq!(read_call("I would go north.", &specs()), None);
        let raw = "<tool_call>\n{\"name\": \"shoot\", \"arguments\": {}}\n</tool_call>";
        assert_eq!(read_call(raw, &specs()), None);
    }
}
