//! The tools a life story calls, and reading them back out of a document.
//!
//! # A second catalog, and why it must be separate
//!
//! [`crate::engine::tools`] is the **action plane** — what a character can do
//! with a thinking step. Nothing in it writes a belief, and a test enforces
//! that: the mind design is explicit that a character cannot decide to stop
//! believing something, any more than a person can.
//!
//! This is the **authoring plane**. An author *can* write a belief, because
//! somebody has to be able to say what a character holds true. These are the
//! tools a life story invokes as it plays out.
//!
//! Keeping them in separate catalogs is what makes the write-protection real
//! rather than a convention. The action catalog is what a character is offered
//! at run time; this one is never offered to a decode at all. A tool cannot
//! leak from here to there by being edited into the wrong table, because the
//! two tables are read by different code at different times.
//!
//! # Simulated, not decoded
//!
//! A life document is prefilled, never generated. Its `<tool_call>` blocks were
//! written by an author, and running the life means **executing** them — the
//! same calls the same way, so the mechanism that forms a belief during
//! authoring is the mechanism that forms one at run time.
//!
//! # The wire format is the model's own
//!
//! `<tool_call>{"name":…,"arguments":{…}}</tool_call>` — what a Qwen3 decode
//! emits and what `zend/src/tools.rs` parses. Using the model's own shape means
//! an author can lift a real trajectory into a life document unchanged, and
//! means the format is already what a future decode would produce if a character
//! ever earned the right to write its own history.

use serde::Serialize;
use serde_json::{Map, Value};

/// One authoring-plane tool.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct AuthoringTool {
    pub name: &'static str,
    /// The layer it writes into. Named so the catalog itself says which part of
    /// the character a call touches.
    pub writes: &'static str,
    pub description: &'static str,
    pub params: &'static [&'static str],
    pub required: &'static [&'static str],
    /// A worked call, for the author and for the tests.
    pub example: &'static str,
}

/// Everything a life story may do.
///
/// Small on purpose. A life produces convictions, relationships and the
/// occasional standing intention; anything larger belongs in prose, where it is
/// read rather than executed.
pub const CATALOG: &[AuthoringTool] = &[
    AuthoringTool {
        name: "form_belief",
        writes: "beliefs",
        description: "A conviction this episode produced. The statement is what the character now \
                      holds true, in their own voice — not a description of them holding it.",
        params: &["statement", "confidence", "threshold"],
        required: &["statement"],
        example: concat!(
            r#"{"name":"form_belief","arguments":{"statement":"Hess burned the east granary","#,
            r#""confidence":0.9,"threshold":0.6}}"#
        ),
    },
    AuthoringTool {
        name: "form_relationship",
        writes: "relationships",
        description: "Someone this episode put into the character's life, and how they stand with \
                      them afterwards. Dials are −1..1 except familiarity, which is 0..1.",
        params: &[
            "entity_id",
            "display",
            "trust",
            "affect",
            "familiarity",
            "notes",
        ],
        required: &["entity_id"],
        example: concat!(
            r#"{"name":"form_relationship","arguments":{"entity_id":"prof-lim","display":"Professor Lim","#,
            r#""trust":0.8,"affect":0.6,"familiarity":0.7,"notes":"Taught you to question assumptions."}}"#
        ),
    },
    AuthoringTool {
        name: "revise_relationship",
        writes: "relationships",
        description: "A relationship this episode moved. Only the named dials change; the rest \
                      keep whatever an earlier episode left them at.",
        params: &["entity_id", "trust", "affect", "familiarity", "notes"],
        required: &["entity_id"],
        example: r#"{"name":"revise_relationship","arguments":{"entity_id":"hess","trust":-0.7}}"#,
    },
    AuthoringTool {
        // `leave_intent`, not `set_intent` — the action catalog already has the
        // latter, and the two catalogs must stay disjoint or a character's own
        // decode could reach an authoring tool by name. The word is also more
        // accurate: an episode *leaves* a character set on something, where a
        // character *sets* an intent for itself in the moment.
        name: "leave_intent",
        writes: "agency",
        description: "A standing intention this episode left the character with — what they are \
                      set on afterwards, and what would end it.",
        params: &["intent", "until"],
        required: &["intent"],
        example: concat!(
            r#"{"name":"leave_intent","arguments":{"intent":"finish the doctorate on her own terms","#,
            r#""until":"submitted, or abandoned"}}"#
        ),
    },
];

pub fn by_name(name: &str) -> Option<&'static AuthoringTool> {
    CATALOG.iter().find(|t| t.name == name)
}

/// One call lifted out of a life document.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Call {
    pub tool: &'static str,
    pub args: Map<String, Value>,
}

/// Why a `<tool_call>` block was not executed.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Rejected {
    NotJson {
        block: String,
    },
    NoName {
        block: String,
    },
    /// Named a tool the authoring catalog does not have. Carries the name: an
    /// invented tool is the clearest sign an author is writing against a
    /// vocabulary that has moved.
    UnknownTool {
        tool: String,
    },
    MissingArgument {
        tool: &'static str,
        argument: &'static str,
    },
}

impl Rejected {
    pub fn message(&self) -> String {
        match self {
            Rejected::NotJson { block } => format!("not JSON: {block}"),
            Rejected::NoName { block } => format!("no `name`: {block}"),
            Rejected::UnknownTool { tool } => {
                format!("no authoring tool `{tool}` — a life story cannot call it")
            }
            Rejected::MissingArgument { tool, argument } => {
                format!("`{tool}` needs `{argument}`")
            }
        }
    }
}

/// What one life document asked for.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Parsed {
    pub calls: Vec<Call>,
    /// Blocks that tried to be calls and failed. **Reported, never dropped** —
    /// a life story whose belief silently did not form is a character missing a
    /// conviction with nothing to say why.
    pub rejected: Vec<Rejected>,
    /// The document with its call blocks removed: the prose a reader sees, and
    /// what is prefilled as the episode's text.
    pub prose: String,
}

/// Pull the calls out of a life document.
///
/// Returns both the calls and the prose without them. The prose is what gets
/// prefilled — the raw JSON of a call is machinery, and leaving it in the turn
/// would teach the character that its own history is written in tool syntax.
pub fn parse(body: &str) -> Parsed {
    let mut out = Parsed::default();
    let mut prose = String::with_capacity(body.len());
    let mut rest = body;

    while let Some(start) = rest.find("<tool_call>") {
        prose.push_str(&rest[..start]);
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            // Unterminated: everything after it is machinery of unknown extent,
            // so it is reported and dropped rather than prefilled as prose.
            out.rejected.push(Rejected::NotJson {
                block: after.trim().chars().take(80).collect(),
            });
            rest = "";
            break;
        };
        let block = after[..end].trim();
        admit(block, &mut out);
        rest = &after[end + "</tool_call>".len()..];
    }
    prose.push_str(rest);

    // Collapse the blank lines a removed call leaves behind, so the prose reads
    // as though the machinery was never in it.
    out.prose = squeeze(&prose);
    out
}

fn admit(block: &str, out: &mut Parsed) {
    let Ok(Value::Object(mut obj)) = serde_json::from_str::<Value>(block) else {
        out.rejected.push(Rejected::NotJson {
            block: block.chars().take(80).collect(),
        });
        return;
    };
    let Some(name) = obj.remove("name").and_then(|v| match v {
        Value::String(s) => Some(s),
        _ => None,
    }) else {
        out.rejected.push(Rejected::NoName {
            block: block.chars().take(80).collect(),
        });
        return;
    };
    let Some(tool) = by_name(&name) else {
        out.rejected.push(Rejected::UnknownTool { tool: name });
        return;
    };
    // `arguments` is the model's own nesting. A call with none is a call with no
    // arguments, not a malformed one — `set_intent` with nothing to set is still
    // caught below by the required check.
    let args = match obj.remove("arguments") {
        Some(Value::Object(a)) => a,
        _ => Map::new(),
    };
    if let Some(missing) = tool.required.iter().find(|r| !args.contains_key(**r)) {
        out.rejected.push(Rejected::MissingArgument {
            tool: tool.name,
            argument: missing,
        });
        return;
    }
    out.calls.push(Call {
        tool: tool.name,
        args,
    });
}

/// Three or more consecutive newlines become two.
fn squeeze(s: &str) -> String {
    // **CRLF is folded first.** `\r` is whitespace, so it neither counted as a
    // newline nor reset the run: on a document written in a Windows editor —
    // which is the common case, not the exotic one — the `\r`s were pushed
    // through while the `\n`s they belonged to were being dropped, so three
    // blank lines came out as `\n\r\n\r\r` and the prose reached the character
    // with stray carriage returns in it.
    let s = s.replace("\r\n", "\n");
    let mut out = String::with_capacity(s.len());
    let mut run = 0;
    for c in s.chars() {
        if c == '\n' {
            run += 1;
            if run > 2 {
                continue;
            }
        } else if !c.is_whitespace() {
            run = 0;
        }
        out.push(c);
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A document written in a Windows editor reaches the character clean.**
    ///
    /// `\r` is whitespace, so it neither counted as a newline nor reset the blank
    /// -line run: the `\n`s were dropped as duplicates while the `\r`s they
    /// belonged to were pushed through, and the prose arrived with bare carriage
    /// returns embedded in it. Nothing reported it — the text is still text.
    #[test]
    fn crlf_prose_squeezes_without_leaving_carriage_returns() {
        let doc = "You came back to the yard.\r\n\r\n\r\n\r\nThe granary was gone.\r\n";
        let p = parse(doc);
        assert!(
            !p.prose.contains('\r'),
            "carriage returns survived: {:?}",
            p.prose
        );
        assert_eq!(
            p.prose,
            "You came back to the yard.\n\nThe granary was gone."
        );
    }

    #[test]
    fn a_call_is_lifted_and_the_prose_is_left() {
        let doc = "You came back to the yard and the granary was gone.\n\n\
                   <tool_call>\n\
                   {\"name\":\"form_belief\",\"arguments\":{\"statement\":\"Hess burned it\"}}\n\
                   </tool_call>\n\n\
                   You did not say so out loud.";
        let p = parse(doc);
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].tool, "form_belief");
        assert_eq!(p.calls[0].args["statement"], "Hess burned it");
        assert!(p.rejected.is_empty());
        assert_eq!(
            p.prose,
            "You came back to the yard and the granary was gone.\n\nYou did not say so out loud."
        );
    }

    /// **The machinery must not reach the character.** Leaving the JSON in the
    /// prefilled turn would teach it that its own history is written in tool
    /// syntax.
    #[test]
    fn no_tool_syntax_survives_into_the_prose() {
        let p = parse(
            "Before.\n<tool_call>\n{\"name\":\"leave_intent\",\"arguments\":{\"intent\":\"go\"}}\n</tool_call>\nAfter.",
        );
        for leak in ["tool_call", "\"name\"", "arguments", "{"] {
            assert!(!p.prose.contains(leak), "leaked {leak:?}: {}", p.prose);
        }
        // A blank line, not a joined one. The call sat on its own line, so its
        // removal leaves a paragraph break — which is what the author's layout
        // meant. Collapsing further would run two separate beats together.
        assert_eq!(p.prose, "Before.\n\nAfter.");
    }

    #[test]
    fn several_calls_keep_their_order() {
        let doc = concat!(
            "<tool_call>{\"name\":\"form_relationship\",\"arguments\":{\"entity_id\":\"lim\"}}</tool_call>\n",
            "Years passed.\n",
            "<tool_call>{\"name\":\"revise_relationship\",\"arguments\":{\"entity_id\":\"lim\",\"trust\":0.9}}</tool_call>",
        );
        let p = parse(doc);
        assert_eq!(
            p.calls.iter().map(|c| c.tool).collect::<Vec<_>>(),
            vec!["form_relationship", "revise_relationship"]
        );
        assert_eq!(p.prose, "Years passed.");
    }

    /// **A belief that silently did not form is a character missing a conviction
    /// with nothing to say why.** Every failure is reported.
    #[test]
    fn a_malformed_call_is_reported_not_dropped() {
        let p = parse("<tool_call>{\"name\":\"form_belief\",}</tool_call>");
        assert!(p.calls.is_empty());
        assert_eq!(p.rejected.len(), 1);
        assert!(matches!(p.rejected[0], Rejected::NotJson { .. }));
    }

    #[test]
    fn a_missing_required_argument_is_reported_with_both_names() {
        let p = parse(
            "<tool_call>{\"name\":\"form_belief\",\"arguments\":{\"confidence\":0.9}}</tool_call>",
        );
        assert_eq!(
            p.rejected[0],
            Rejected::MissingArgument {
                tool: "form_belief",
                argument: "statement"
            }
        );
        assert!(p.rejected[0].message().contains("needs `statement`"));
    }

    /// An author writing against a vocabulary that has moved is the clearest
    /// case for naming the tool in the rejection.
    #[test]
    fn an_invented_tool_is_named() {
        let p = parse("<tool_call>{\"name\":\"implant_memory\",\"arguments\":{}}</tool_call>");
        assert_eq!(
            p.rejected[0],
            Rejected::UnknownTool {
                tool: "implant_memory".into()
            }
        );
        assert!(p.rejected[0].message().contains("cannot call it"));
    }

    /// **The action plane cannot reach these, and this is the assertion that
    /// keeps it so.** A tool edited into the wrong catalog would otherwise let a
    /// character's own decode write its beliefs.
    #[test]
    fn no_authoring_tool_is_offered_to_a_character() {
        use crate::engine::tools;
        for a in CATALOG {
            assert!(
                tools::by_name(a.name).is_none(),
                "{} is in BOTH catalogs — a character could write its own beliefs",
                a.name
            );
        }
        // And the reverse: nothing a character does at run time can be invoked
        // by a life story, so an author cannot make a character `speak` in its
        // own history and have it reach the world.
        for t in tools::CATALOG.iter() {
            assert!(by_name(t.name).is_none(), "{} is in both catalogs", t.name);
        }
    }

    /// Every catalog example must parse as the call it documents. An example
    /// that does not work is documentation that lies.
    #[test]
    fn every_documented_example_executes() {
        for t in CATALOG.iter() {
            let p = parse(&format!("<tool_call>{}</tool_call>", t.example));
            assert_eq!(
                p.calls.len(),
                1,
                "{}'s example does not parse: {:?}",
                t.name,
                p.rejected
            );
            assert_eq!(p.calls[0].tool, t.name);
            for r in t.required {
                assert!(
                    p.calls[0].args.contains_key(*r),
                    "{}'s example omits required `{r}`",
                    t.name
                );
            }
        }
    }

    /// Every argument an example passes must be declared, or the example teaches
    /// a field the executor will not read.
    #[test]
    fn examples_only_pass_declared_parameters() {
        for t in CATALOG.iter() {
            let p = parse(&format!("<tool_call>{}</tool_call>", t.example));
            for key in p.calls[0].args.keys() {
                assert!(
                    t.params.contains(&key.as_str()),
                    "{}: example passes undeclared `{key}`",
                    t.name
                );
            }
        }
    }

    /// An unterminated block is machinery of unknown extent — reported, and not
    /// prefilled as prose.
    #[test]
    fn an_unterminated_block_does_not_leak_into_the_prose() {
        let p = parse("Before.\n<tool_call>\n{\"name\":\"form_belief\"");
        assert_eq!(p.prose, "Before.");
        assert_eq!(p.rejected.len(), 1);
    }

    #[test]
    fn a_document_with_no_calls_is_all_prose() {
        let p = parse("She was born in the spring.\n\nIt rained.");
        assert!(p.calls.is_empty());
        assert!(p.rejected.is_empty());
        assert_eq!(p.prose, "She was born in the spring.\n\nIt rained.");
    }

    #[test]
    fn catalog_names_are_unique_and_each_writes_a_named_layer() {
        let mut n: Vec<&str> = CATALOG.iter().map(|t| t.name).collect();
        let len = n.len();
        n.sort_unstable();
        n.dedup();
        assert_eq!(n.len(), len, "two authoring tools share a name");
        for t in CATALOG.iter() {
            assert!(!t.writes.is_empty());
            assert!(!t.description.is_empty());
            assert!(!t.required.is_empty(), "{} requires nothing", t.name);
        }
    }
}
