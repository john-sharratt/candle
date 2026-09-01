//! Reading acts out of what a character said.
//!
//! # The wire format, and why it is this one
//!
//! A character acts by emitting a call. The format is one JSON object per line,
//! each naming a tool and its arguments:
//!
//! ```text
//! {"tool":"speak","intent":"that he will not get the ledger"}
//! {"tool":"face","target":"the door"}
//! ```
//!
//! Not a function-calling API, and not XML. Three reasons, in order of how much
//! they cost when ignored:
//!
//! 1. **A partial line is detectable.** Generation can stop mid-token for a
//!    dozen reasons, and one-object-per-line means an interrupted emission
//!    loses exactly the last act rather than corrupting the parse of every act
//!    before it.
//! 2. **The model already writes it well.** ChatML-tuned models emit JSON
//!    objects reliably; the failure modes are known (trailing prose, a fenced
//!    block, a stray comma) and each is cheap to tolerate here.
//! 3. **It survives being wrong.** A line that is not JSON, or names a tool
//!    that does not exist, is *reported* rather than silently dropped — see
//!    [`Parsed::rejected`]. A parser that quietly discarded malformed acts would
//!    make a character that is failing to act indistinguishable from one
//!    choosing not to, and those need completely different fixes.
//!
//! # Prose around the calls is not an act
//!
//! A model will narrate. It will write "I turn to face him and say:" before the
//! call, or explain itself after. That text is kept — as [`Parsed::narration`],
//! so it is visible in Pulse and in the log — and it is *not* an act. The world
//! never sees it. This is the prompt's "if you write out what you are doing
//! instead of calling the tool, nothing happens", enforced rather than merely
//! requested.

use serde::Serialize;
use serde_json::{Map, Value};

use crate::engine::tools;

/// One act a character took.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Act {
    pub tool: &'static str,
    /// The call's arguments, minus the `tool` key.
    pub args: Map<String, Value>,
}

impl Act {
    /// How this act reads in the Pulse feed and in the character's own window.
    ///
    /// Not the narrator's rendering — that is a separate concern and needs the
    /// character's voice. This is the act as an act: what was done, and with
    /// what intent, in a form a person reading the feed can scan.
    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        // Ordered by the tool's own parameter list rather than by the map's
        // iteration order — `serde_json::Map` is a BTreeMap, so it would
        // otherwise render alphabetically and `manner` would precede `intent`.
        if let Some(t) = tools::by_name(self.tool) {
            for p in t.params {
                if let Some(v) = self.args.get(p.name) {
                    parts.push(render(v));
                }
            }
        }
        if parts.is_empty() {
            self.tool.to_string()
        } else {
            format!("{} — {}", self.tool, parts.join("; "))
        }
    }
}

fn render(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Why a line was not an act.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Rejected {
    /// Looked like a call and was not valid JSON.
    NotJson { line: String },
    /// Valid JSON, no `tool` key.
    NoTool { line: String },
    /// Named a tool that is not in the catalog. Carries the name so the log can
    /// say which — an invented tool is the single most useful signal that a
    /// character's prompt and its vocabulary have drifted apart.
    UnknownTool { tool: String },
    /// A required parameter was missing.
    MissingParam {
        tool: &'static str,
        param: &'static str,
    },
}

/// What one decode produced.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Parsed {
    pub acts: Vec<Act>,
    /// Text the model wrote around the calls. Kept and shown; never acted on.
    pub narration: String,
    /// Lines that tried to be acts and failed. Reported, never dropped.
    pub rejected: Vec<Rejected>,
}

impl Parsed {
    pub fn is_empty(&self) -> bool {
        self.acts.is_empty()
    }
}

/// Whether a line is plausibly a call rather than prose.
///
/// Deliberately loose: anything starting with `{` is *tried*, and failing the
/// try is reported. A tight test here would silently reclassify a malformed act
/// as narration, which is exactly the confusion this module exists to prevent.
fn looks_like_call(line: &str) -> bool {
    line.starts_with('{')
}

/// Parse a decode into acts.
pub fn parse(output: &str) -> Parsed {
    let mut out = Parsed::default();
    let mut narration: Vec<&str> = Vec::new();

    for raw in output.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        // A model asked for JSON will sometimes fence it. Strip the fence
        // markers rather than treating them as narration — they are neither.
        if line.starts_with("```") {
            continue;
        }
        if !looks_like_call(line) {
            narration.push(raw);
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Err(_) => out.rejected.push(Rejected::NotJson {
                line: line.to_string(),
            }),
            Ok(Value::Object(mut obj)) => {
                let Some(name) = obj.remove("tool").and_then(|v| match v {
                    Value::String(s) => Some(s),
                    _ => None,
                }) else {
                    out.rejected.push(Rejected::NoTool {
                        line: line.to_string(),
                    });
                    continue;
                };
                let Some(tool) = tools::by_name(&name) else {
                    out.rejected.push(Rejected::UnknownTool { tool: name });
                    continue;
                };
                // A missing required parameter is a rejection, not a
                // best-effort act. `speak` with no intent is not a quieter
                // `speak`; it is a call the character did not finish making.
                if let Some(p) = tool
                    .params
                    .iter()
                    .find(|p| p.required && !obj.contains_key(p.name))
                {
                    out.rejected.push(Rejected::MissingParam {
                        tool: tool.name,
                        param: p.name,
                    });
                    continue;
                }
                out.acts.push(Act {
                    tool: tool.name,
                    args: obj,
                });
            }
            // Valid JSON that is not an object — a bare string or number on its
            // own line. Not a call, and not prose either.
            Ok(_) => out.rejected.push(Rejected::NoTool {
                line: line.to_string(),
            }),
        }
    }

    out.narration = narration.join("\n").trim().to_string();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_call_parses() {
        let p = parse(r#"{"tool":"speak","intent":"that I will not"}"#);
        assert_eq!(p.acts.len(), 1);
        assert_eq!(p.acts[0].tool, "speak");
        assert_eq!(p.acts[0].args["intent"], "that I will not");
        assert!(p.rejected.is_empty());
        assert!(p.narration.is_empty());
    }

    #[test]
    fn several_calls_keep_their_order() {
        let p = parse(
            "{\"tool\":\"face\",\"target\":\"the door\"}\n\
             {\"tool\":\"speak\",\"intent\":\"that someone is coming\"}",
        );
        assert_eq!(
            p.acts.iter().map(|a| a.tool).collect::<Vec<_>>(),
            vec!["face", "speak"]
        );
    }

    /// **Narration is not an act.** The model will write "I turn and say:"
    /// around its calls; the world must never see it, and it must still be
    /// visible to a person reading the feed.
    #[test]
    fn prose_around_calls_is_kept_and_not_acted_on() {
        let p = parse(
            "I have had enough of this.\n\
             {\"tool\":\"refuse\",\"what\":\"handing over the ledger\"}\n\
             He will not like that.",
        );
        assert_eq!(p.acts.len(), 1);
        assert_eq!(p.acts[0].tool, "refuse");
        assert_eq!(
            p.narration,
            "I have had enough of this.\nHe will not like that."
        );
    }

    /// A model asked for JSON fences it about a third of the time. The fence is
    /// neither an act nor narration.
    #[test]
    fn a_fenced_block_is_unwrapped() {
        let p = parse("```json\n{\"tool\":\"wait\"}\n```");
        assert_eq!(p.acts.len(), 1);
        assert_eq!(p.acts[0].tool, "wait");
        assert!(
            p.narration.is_empty(),
            "the fence became narration: {:?}",
            p.narration
        );
    }

    /// **The failure this module exists for.** A character failing to act and a
    /// character choosing not to act look identical from the outside and need
    /// completely different fixes, so a malformed call is reported.
    #[test]
    fn a_malformed_call_is_rejected_rather_than_dropped() {
        let p = parse(r#"{"tool":"speak","intent":}"#);
        assert!(p.acts.is_empty());
        assert_eq!(p.rejected.len(), 1);
        assert!(matches!(p.rejected[0], Rejected::NotJson { .. }));
    }

    /// An invented tool is the clearest signal that the prompt and the
    /// vocabulary have drifted apart, so the name is carried.
    #[test]
    fn an_invented_tool_is_named_in_the_rejection() {
        let p = parse(r#"{"tool":"teleport","to":"the keep"}"#);
        assert!(p.acts.is_empty());
        assert_eq!(
            p.rejected[0],
            Rejected::UnknownTool {
                tool: "teleport".into()
            }
        );
    }

    /// `speak` with no intent is not a quieter `speak` — it is a call the
    /// character did not finish making.
    #[test]
    fn a_missing_required_parameter_is_a_rejection_not_a_best_effort_act() {
        let p = parse(r#"{"tool":"speak","manner":"flatly"}"#);
        assert!(p.acts.is_empty());
        assert_eq!(
            p.rejected[0],
            Rejected::MissingParam {
                tool: "speak",
                param: "intent"
            }
        );
    }

    /// An optional parameter's absence is fine — that is what optional means.
    #[test]
    fn an_absent_optional_parameter_is_not_a_rejection() {
        let p = parse(r#"{"tool":"speak","intent":"hello"}"#);
        assert_eq!(p.acts.len(), 1);
        assert!(p.rejected.is_empty());
    }

    #[test]
    fn an_object_without_a_tool_key_is_rejected() {
        let p = parse(r#"{"intent":"something"}"#);
        assert!(p.acts.is_empty());
        assert_eq!(p.rejected.len(), 1);
        assert!(matches!(p.rejected[0], Rejected::NoTool { .. }));
    }

    /// Only a line starting with `{` is *tried* as a call. A bare string or
    /// number is prose that happens to be valid JSON — a model writing `42` on
    /// its own line has narrated, not acted, and reporting it as a malformed
    /// call would fill the rejection list with things nobody meant as acts.
    #[test]
    fn json_that_is_not_an_object_is_narration() {
        for line in [r#""just a string""#, "42", "true"] {
            let p = parse(line);
            assert!(p.acts.is_empty(), "{line} produced an act");
            assert!(
                p.rejected.is_empty(),
                "{line} was reported as a failed call"
            );
            assert_eq!(p.narration, line);
        }
    }

    /// One bad line must not cost the good ones. Generation stopping mid-token
    /// is common, and losing the whole batch to a truncated last line would
    /// make every interrupted decode a total loss.
    #[test]
    fn a_truncated_last_line_costs_only_itself() {
        let p = parse(
            "{\"tool\":\"observe\",\"target\":\"the ridge\"}\n\
             {\"tool\":\"spea",
        );
        assert_eq!(p.acts.len(), 1);
        assert_eq!(p.acts[0].tool, "observe");
        assert_eq!(p.rejected.len(), 1);
    }

    /// The summary renders in the tool's declared parameter order.
    /// `serde_json::Map` is a BTreeMap, so without that it would come out
    /// alphabetically and `manner` would precede `intent`.
    #[test]
    fn a_summary_follows_the_tools_parameter_order_not_the_maps() {
        let p = parse(r#"{"tool":"speak","manner":"flatly","intent":"that I refuse","to":"Hess"}"#);
        let s = p.acts[0].summary();
        assert_eq!(s, "speak — that I refuse; Hess; flatly");
        let intent = s.find("that I refuse").unwrap();
        let manner = s.find("flatly").unwrap();
        assert!(
            intent < manner,
            "alphabetical order leaked into the summary: {s}"
        );
    }

    #[test]
    fn an_argumentless_act_summarises_as_its_name() {
        let p = parse(r#"{"tool":"wait"}"#);
        assert_eq!(p.acts[0].summary(), "wait");
    }

    /// A decode with nothing in it is not an error — a character can be handed
    /// a moment and have nothing to say about it.
    #[test]
    fn an_empty_decode_is_empty_not_broken() {
        let p = parse("   \n\n  ");
        assert!(p.is_empty());
        assert!(p.rejected.is_empty());
        assert!(p.narration.is_empty());
    }

    /// Every tool in the catalog must round-trip through its own example — the
    /// examples are what calibration prefills, so a format the parser rejects
    /// would be taught to the model as correct.
    #[test]
    fn every_catalog_example_parses_as_an_act() {
        for t in tools::CATALOG {
            for e in t.examples {
                let mut obj: Map<String, Value> = serde_json::from_str(e.call).unwrap();
                obj.insert("tool".into(), Value::String(t.name.into()));
                let line = serde_json::to_string(&Value::Object(obj)).unwrap();
                let p = parse(&line);
                assert_eq!(
                    p.acts.len(),
                    1,
                    "{}'s example does not parse as an act: {:?}",
                    t.name,
                    p.rejected
                );
                assert_eq!(p.acts[0].tool, t.name);
            }
        }
    }
}
