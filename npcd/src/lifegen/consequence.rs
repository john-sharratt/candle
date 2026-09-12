//! What a day is required to produce, authored by the operator.
//!
//! # The inversion, and the entire class of failure it removes
//!
//! The obvious way to generate a life is to let the model write the prose *and*
//! the `<tool_call>` blocks, then parse them back. It does not work well. Over
//! fifteen episodes a model will invent tool names, drift argument names, and
//! spell one person four ways — `prof-lim`, `lim`, `professor-lim`,
//! `lim-wei-ming` — producing four relationships where the author meant one.
//! [`crate::engine::authoring`] catches the malformed ones, but it catches them
//! at *ingest*, and a rejected belief is a character quietly missing a
//! conviction.
//!
//! So the calls do not round-trip through the model at all. The operator
//! attaches them to a day in the console, picking entities from the cast that
//! already exists, and generation **injects** them. The prose is what the model
//! writes; the consequences are data the console already held.
//!
//! What that removes, completely rather than mostly: invented tool names,
//! drifted argument names, malformed JSON, ambiguous entity ids, and a `revise`
//! that lands before the `form` it revises. None of them are validated away —
//! none of them can occur.
//!
//! It also inverts the prompt, which is the part that reads better: not *write
//! a day and tell me what it produced*, but **write a day that earns these**.
//! A reviewer can then judge the thing that matters — whether the prose
//! actually earns the belief.
//!
//! # The catalog stays the single authority
//!
//! A consequence is a tool name and its arguments, checked against
//! [`crate::engine::authoring::CATALOG`] — not a parallel enum of the same
//! four shapes. An authoring tool added there becomes available here, and to
//! the console's form, with no change to this file. A second enumeration of the
//! same vocabulary is a second place for it to be wrong.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::engine::authoring::{by_name, AuthoringTool};

/// One consequence an operator has attached to a day.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Consequence {
    /// The authoring tool this calls.
    pub tool: String,
    /// Its arguments, as the console's form produced them.
    #[serde(default)]
    pub args: Map<String, Value>,
}

/// Why a consequence cannot be written into a document.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "problem", rename_all = "snake_case")]
pub enum BadConsequence {
    /// Names a tool the authoring catalog does not have.
    UnknownTool { tool: String },
    /// The catalog says this argument is required.
    MissingArgument { tool: String, argument: String },
    /// An argument the catalog does not declare. Refused rather than dropped:
    /// silently discarding it would leave the operator believing they had set
    /// something.
    UnknownArgument { tool: String, argument: String },
    /// A dial outside its range. Carries both bounds, because "out of range"
    /// without them is not actionable.
    OutOfRange {
        argument: String,
        low: f64,
        high: f64,
    },
    /// A revision of a relationship no earlier day established.
    Unestablished { entity_id: String },
    /// Required by the catalog and present, but empty — an empty belief is a
    /// record with nothing in it.
    Empty { tool: String, argument: String },
}

impl BadConsequence {
    pub fn message(&self) -> String {
        match self {
            BadConsequence::UnknownTool { tool } => {
                format!("no authoring tool `{tool}`")
            }
            BadConsequence::MissingArgument { tool, argument } => {
                format!("`{tool}` needs `{argument}`")
            }
            BadConsequence::UnknownArgument { tool, argument } => {
                format!("`{tool}` has no argument `{argument}`")
            }
            BadConsequence::OutOfRange {
                argument,
                low,
                high,
            } => format!("`{argument}` must be between {low} and {high}"),
            BadConsequence::Unestablished { entity_id } => format!(
                "`{entity_id}` is revised before any day establishes them — \
                 form the relationship first"
            ),
            BadConsequence::Empty { tool, argument } => {
                format!("`{tool}`'s `{argument}` is empty")
            }
        }
    }
}

/// The range a named dial must fall in.
///
/// Familiarity is how well you know someone, which has no negative half — you
/// cannot know a person less than not at all. Trust and affect are signed
/// because their negative halves are real positions rather than absences.
/// Confidence is a probability.
fn range(argument: &str) -> Option<(f64, f64)> {
    match argument {
        "trust" | "affect" => Some((-1.0, 1.0)),
        "familiarity" | "confidence" | "threshold" => Some((0.0, 1.0)),
        _ => None,
    }
}

/// Check one consequence against the catalog, reporting everything wrong.
pub fn check(c: &Consequence) -> Vec<BadConsequence> {
    let Some(tool) = by_name(&c.tool) else {
        return vec![BadConsequence::UnknownTool {
            tool: c.tool.clone(),
        }];
    };
    let mut bad = Vec::new();

    for r in tool.required {
        match c.args.get(*r) {
            None => bad.push(BadConsequence::MissingArgument {
                tool: tool.name.to_string(),
                argument: (*r).to_string(),
            }),
            Some(Value::String(s)) if s.trim().is_empty() => bad.push(BadConsequence::Empty {
                tool: tool.name.to_string(),
                argument: (*r).to_string(),
            }),
            Some(_) => {}
        }
    }

    for (key, value) in &c.args {
        if !tool.params.contains(&key.as_str()) {
            bad.push(BadConsequence::UnknownArgument {
                tool: tool.name.to_string(),
                argument: key.clone(),
            });
            continue;
        }
        if let (Some((low, high)), Some(n)) = (range(key), value.as_f64()) {
            if n < low || n > high {
                bad.push(BadConsequence::OutOfRange {
                    argument: key.clone(),
                    low,
                    high,
                });
            }
        }
    }
    bad
}

/// The entity a consequence names, when it names one.
pub fn entity(c: &Consequence) -> Option<&str> {
    c.args.get("entity_id").and_then(|v| v.as_str())
}

/// Check a whole life's consequences **in chronological order**, so a revision
/// that precedes the relationship it revises is caught.
///
/// This is the one rule that cannot be checked a day at a time, which is why it
/// lives here rather than in [`check`]: a `revise_relationship` is perfectly
/// well-formed on its own and wrong only in company.
///
/// `established` seeds the set with entities that exist before the life is
/// generated — the seed's cast — because those relationships were not formed by
/// any day and revising them on day one is legitimate.
pub fn check_ordered<'a>(
    days: impl IntoIterator<Item = &'a [Consequence]>,
    established: &[String],
) -> Vec<BadConsequence> {
    let mut known: Vec<String> = established.to_vec();
    let mut bad = Vec::new();
    for day in days {
        // Everything one day forms is available to everything later in the
        // same day: an episode may introduce somebody and then move the
        // relationship it just created.
        for c in day {
            bad.extend(check(c));
            match c.tool.as_str() {
                "form_relationship" => {
                    if let Some(e) = entity(c) {
                        if !known.iter().any(|k| k == e) {
                            known.push(e.to_string());
                        }
                    }
                }
                "revise_relationship" => {
                    if let Some(e) = entity(c) {
                        if !known.iter().any(|k| k == e) {
                            bad.push(BadConsequence::Unestablished {
                                entity_id: e.to_string(),
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }
    bad
}

/// Render one consequence as the `<tool_call>` block a life document carries.
///
/// The wire format is the model's own — what a Qwen3 decode emits and what
/// [`crate::engine::authoring::parse`] reads back. Using it here means an
/// injected call and a hand-written one are indistinguishable to everything
/// downstream, which is what keeps one ingest path instead of two.
pub fn render(c: &Consequence) -> String {
    let mut obj = Map::new();
    obj.insert("name".into(), Value::String(c.tool.clone()));
    obj.insert("arguments".into(), Value::Object(c.args.clone()));
    format!("<tool_call>\n{}\n</tool_call>", Value::Object(obj))
}

/// Every consequence for one day, as the block appended to its prose.
///
/// Appended **after** the prose rather than interleaved. An author writing by
/// hand puts a call at the moment in the episode where the belief formed, and
/// that reads well; a generator cannot know where in prose it did not write the
/// moment falls, and guessing would put the call in the wrong paragraph. The
/// order within the day is the operator's, and it is preserved because
/// `revise` after `form` on the same day depends on it.
pub fn render_all(cs: &[Consequence]) -> String {
    cs.iter().map(render).collect::<Vec<_>>().join("\n\n")
}

/// The catalog, for the console's form. Re-exported here so the console has one
/// place to ask, rather than knowing that authoring tools live in the engine.
pub fn catalog() -> &'static [AuthoringTool] {
    crate::engine::authoring::CATALOG
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::authoring::{parse, Call};

    fn c(tool: &str, args: Value) -> Consequence {
        Consequence {
            tool: tool.into(),
            args: match args {
                Value::Object(m) => m,
                _ => Map::new(),
            },
        }
    }

    fn belief() -> Consequence {
        c(
            "form_belief",
            serde_json::json!({"statement": "Hess burned the east granary", "confidence": 0.9}),
        )
    }

    /// **The property this whole module exists for.** What the console holds,
    /// rendered into a document and read back by the ingest parser, is the same
    /// call — so an injected consequence and a hand-written one are the same
    /// thing to everything downstream.
    #[test]
    fn a_rendered_consequence_parses_back_as_the_same_call() {
        let original = belief();
        let parsed = parse(&render(&original));
        assert!(parsed.rejected.is_empty(), "{:?}", parsed.rejected);
        assert_eq!(
            parsed.calls,
            vec![Call {
                tool: "form_belief",
                args: original.args.clone(),
            }]
        );
        // And the prose is empty, because the block was the whole document.
        assert_eq!(parsed.prose, "");
    }

    /// The exact bytes, not a shape. This is the wire format two modules agree
    /// on, and a change to it must break a test rather than a character.
    #[test]
    fn a_rendered_call_is_exactly_the_models_own_format() {
        let x = c("leave_intent", serde_json::json!({"intent": "go north"}));
        assert_eq!(
            render(&x),
            "<tool_call>\n{\"arguments\":{\"intent\":\"go north\"},\"name\":\"leave_intent\"}\n</tool_call>"
        );
    }

    /// Several consequences on one day survive as several calls, in the
    /// operator's order — which `revise` after `form` on the same day needs.
    #[test]
    fn a_days_consequences_render_in_order_and_parse_as_several_calls() {
        let day = vec![
            c(
                "form_relationship",
                serde_json::json!({"entity_id": "lim", "display": "Professor Lim"}),
            ),
            c(
                "revise_relationship",
                serde_json::json!({"entity_id": "lim", "trust": 0.9}),
            ),
        ];
        let parsed = parse(&render_all(&day));
        assert!(parsed.rejected.is_empty(), "{:?}", parsed.rejected);
        assert_eq!(
            parsed.calls.iter().map(|x| x.tool).collect::<Vec<_>>(),
            vec!["form_relationship", "revise_relationship"]
        );
    }

    #[test]
    fn a_consequence_naming_no_catalog_tool_is_refused() {
        let bad = check(&c("implant_memory", serde_json::json!({})));
        assert_eq!(
            bad,
            vec![BadConsequence::UnknownTool {
                tool: "implant_memory".into()
            }]
        );
    }

    #[test]
    fn a_missing_required_argument_is_refused() {
        let bad = check(&c("form_belief", serde_json::json!({"confidence": 0.5})));
        assert_eq!(
            bad,
            vec![BadConsequence::MissingArgument {
                tool: "form_belief".into(),
                argument: "statement".into()
            }]
        );
    }

    /// **Dropping an unknown argument silently is worse than refusing it** —
    /// the operator would leave the form believing they had set something.
    #[test]
    fn an_argument_the_catalog_does_not_declare_is_refused_not_dropped() {
        let bad = check(&c(
            "form_belief",
            serde_json::json!({"statement": "x", "certainty": 0.9}),
        ));
        assert_eq!(
            bad,
            vec![BadConsequence::UnknownArgument {
                tool: "form_belief".into(),
                argument: "certainty".into()
            }]
        );
    }

    /// The dials have different ranges and the difference is meaningful:
    /// familiarity has no negative half, because you cannot know somebody less
    /// than not at all.
    #[test]
    fn each_dial_is_checked_against_its_own_range() {
        let over = check(&c(
            "form_relationship",
            serde_json::json!({"entity_id": "x", "trust": 1.5}),
        ));
        assert!(
            matches!(over[0], BadConsequence::OutOfRange { low, high, .. } if low == -1.0 && high == 1.0)
        );

        let neg = check(&c(
            "form_relationship",
            serde_json::json!({"entity_id": "x", "familiarity": -0.2}),
        ));
        assert!(
            matches!(neg[0], BadConsequence::OutOfRange { low, high, .. } if low == 0.0 && high == 1.0)
        );

        // The signed dials accept their whole range, including the ends.
        assert!(check(&c(
            "form_relationship",
            serde_json::json!({"entity_id": "x", "trust": -1.0, "affect": 1.0, "familiarity": 0.0})
        ))
        .is_empty());
    }

    #[test]
    fn a_required_argument_present_but_empty_is_refused() {
        let bad = check(&c("form_belief", serde_json::json!({"statement": "   "})));
        assert_eq!(
            bad,
            vec![BadConsequence::Empty {
                tool: "form_belief".into(),
                argument: "statement".into()
            }]
        );
    }

    /// **The rule that cannot be checked one day at a time.** A revision is
    /// well-formed alone and wrong only in company.
    #[test]
    fn revising_a_relationship_no_day_established_is_refused() {
        let d1: Vec<Consequence> = vec![c(
            "revise_relationship",
            serde_json::json!({"entity_id": "hess", "trust": -0.7}),
        )];
        let bad = check_ordered([d1.as_slice()], &[]);
        assert_eq!(
            bad,
            vec![BadConsequence::Unestablished {
                entity_id: "hess".into()
            }]
        );
    }

    #[test]
    fn a_relationship_formed_on_an_earlier_day_may_be_revised_later() {
        let d1 = vec![c(
            "form_relationship",
            serde_json::json!({"entity_id": "hess"}),
        )];
        let d2 = vec![c(
            "revise_relationship",
            serde_json::json!({"entity_id": "hess", "trust": -0.7}),
        )];
        assert!(check_ordered([d1.as_slice(), d2.as_slice()], &[]).is_empty());
    }

    /// An episode may introduce somebody and then move the relationship it just
    /// created, so a form earlier in the same day counts.
    #[test]
    fn a_relationship_formed_earlier_the_same_day_may_be_revised() {
        let day = vec![
            c("form_relationship", serde_json::json!({"entity_id": "lim"})),
            c(
                "revise_relationship",
                serde_json::json!({"entity_id": "lim", "affect": 0.4}),
            ),
        ];
        assert!(check_ordered([day.as_slice()], &[]).is_empty());
    }

    /// **The seed's cast was never formed by any day**, so revising one on the
    /// first day is legitimate rather than an ordering fault.
    #[test]
    fn the_seeds_own_cast_counts_as_already_established() {
        let day = vec![c(
            "revise_relationship",
            serde_json::json!({"entity_id": "prof-lim", "trust": 0.9}),
        )];
        assert!(check_ordered([day.as_slice()], &["prof-lim".to_string()]).is_empty());
    }

    /// Every catalog example must survive the round trip, so the console's form
    /// can be built from the catalog and produce something that ingests.
    #[test]
    fn every_catalog_example_round_trips_as_a_consequence() {
        for t in catalog() {
            let parsed = parse(&format!("<tool_call>{}</tool_call>", t.example));
            let call = &parsed.calls[0];
            let x = Consequence {
                tool: call.tool.to_string(),
                args: call.args.clone(),
            };
            assert!(check(&x).is_empty(), "{}: {:?}", t.name, check(&x));
            let back = parse(&render(&x));
            assert_eq!(
                back.calls[0].args, call.args,
                "{} did not round trip",
                t.name
            );
        }
    }

    #[test]
    fn a_consequence_round_trips_through_json() {
        let x = belief();
        let back: Consequence = serde_json::from_str(&serde_json::to_string(&x).unwrap()).unwrap();
        assert_eq!(x, back);
    }
}
