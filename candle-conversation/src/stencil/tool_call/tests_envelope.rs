//! Tests that the envelope and the grammar agree on shape, and that every
//! closing delimiter is one the model can write.
//!
//! The tests assert the raw structure the function-block and JSON grammars emit,
//! the position of the assistant-turn terminator, and that a value closes only on
//! a tag the tree offers next, so a value that holds the tag still writes intact.
//! Action loops are tested for their bound and for a stop that is always
//! reachable, and a thinking turn for a node edge joining its block to its calls.
//! Other tests check that a tokenizer which merges bytes across a boundary still
//! compiles.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use crate::stencil::compile::compile;

    use crate::stencil::vocab::TestVocab;

    /// `render` writes the shape the grammar compiles — including the scalar
    /// split the grammar makes.
    ///
    /// The JSON arm used to quote every value, so a numeric argument rendered
    /// `"page": "1"` while `build_value`'s `Number` arm emits `1`. Only
    /// prose-valued callers existed, so nothing caught it until an ingest chain
    /// needed to prefill a page number.
    #[test]
    fn render_writes_numbers_bare_and_strings_quoted() {
        let json = ToolCallEnvelope::qwen3();
        let out = json.render("file_read", &[("path", "a/mod.rs"), ("page", "1")]);
        assert!(
            out.contains(r#""path": "a/mod.rs""#),
            "a string argument stays quoted and escaped: {out}"
        );
        assert!(
            out.contains(r#""page": 1"#),
            "a numeric argument is bare, as `build_value` emits it: {out}"
        );

        // A function block's values are raw whatever their type, so neither
        // gains quotes and the scalar question does not arise.
        let fb = ToolCallEnvelope::qwen35();
        let out = fb.render("file_read", &[("path", "a/mod.rs"), ("page", "1")]);
        assert!(
            out.contains("<parameter=path>\na/mod.rs</parameter>"),
            "{out}"
        );
        assert!(out.contains("<parameter=page>\n1</parameter>"), "{out}");
        assert!(!out.contains('"'), "raw values carry no quotes: {out}");
    }

    /// A quote inside a string value must not end the string early.
    #[test]
    fn render_escapes_a_quote_in_a_string_value() {
        let json = ToolCallEnvelope::qwen3();
        let out = json.render("say", &[("intent", r#"that "it" is mine"#)]);
        assert!(
            out.contains(r#""intent": "that \"it\" is mine""#),
            "the quote is escaped rather than terminating: {out}"
        );
    }

    /// **The assistant close must END on the turn terminator.** The rule and the
    /// measured consequence are on
    /// [`ToolCallEnvelope::for_assistant_turn`]; this is the guard.
    ///
    /// It exists because the regression that motivated it replaced a correct
    /// literal (`"}}\n</tool_call><|im_end|>"`) with `assistant_end` appended
    /// verbatim — right in intent, since the shape belongs to the dialect and
    /// not to a literal, but ChatML's `assistant_end` trails a newline. Nothing
    /// in the area asserted the terminator's POSITION, so every Qwen tool call
    /// ran past its own turn end until the substrate was read by hand.
    ///
    /// Asserted over every dialect, not just the broken one: the three that
    /// happen not to trail whitespace today are one edit away from doing so.
    #[test]
    fn the_assistant_close_ends_on_the_turn_terminator() {
        for d in [
            Dialect::chat_ml(),
            Dialect::qwen35(),
            Dialect::llama2(),
            Dialect::llama3(),
            Dialect::deepseek(),
        ] {
            let env = ToolCallEnvelope::for_assistant_turn(&d);
            let term = d.assistant_end.trim_end();
            assert!(
                !term.is_empty(),
                "{:?}: no assistant terminator to end on",
                d.dialect_type
            );
            assert!(
                env.close.ends_with(term),
                "{:?}: close {:?} must end on {:?} — a terminator in any earlier \
                 slot has its flag dropped and the decode never stops",
                d.dialect_type,
                env.close,
                term
            );
            // The marker the model already emitted must not be re-emitted.
            assert!(
                !env.open.starts_with(&env.marker),
                "{:?}: open {:?} still leads with the marker {:?}",
                d.dialect_type,
                env.open,
                env.marker
            );
        }
    }

    pub(in super::super) fn catalog() -> Vec<ToolSpec> {
        parse_tools(
            r#"[
              {"name":"read_file","params":[{"name":"path","type":"string","required":true}]},
              {"name":"write_file","params":[
                  {"name":"path","type":"string","required":true},
                  {"name":"append","type":"boolean","required":false}
              ]},
              {"name":"set_mode","params":[
                  {"name":"mode","type":"string","required":true,
                   "enum":["read","write","exec"]}
              ]}
            ]"#,
        )
        .unwrap()
    }

    #[test]
    fn parses_catalog() {
        let c = catalog();
        assert_eq!(c.len(), 3);
        assert_eq!(c[2].params[0].enum_values.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn compiles_to_a_valid_tree() {
        let tree = compile(
            &compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen3()).unwrap(),
            &TestVocab::new(),
        )
        .unwrap();
        assert!(tree.len() > 5);
        assert_eq!(tree.label(), "tool_call");
    }

    // ── the function-block shape ────────────────────────────────────────────

    /// Every static string the tree can emit, in id order — enough to see what
    /// shape a decode is being held to without walking edges.
    pub(in super::super) fn statics(spec: &TreeSpec) -> Vec<String> {
        spec.nodes
            .iter()
            .filter_map(|n| match n {
                NodeSpec::Static { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    /// Every branch arm the tree offers.
    pub(in super::super) fn arms(spec: &TreeSpec) -> Vec<String> {
        spec.nodes
            .iter()
            .flat_map(|n| match n {
                NodeSpec::Branch { arms } => arms.iter().map(|(t, _)| t.clone()).collect(),
                _ => Vec::new(),
            })
            .collect()
    }

    /// **The grammar emits Qwen3.5's own syntax**, not JSON wearing its
    /// markers. Nothing checked this before: the prompt described one shape,
    /// the grammar forced another, and the only test in the area compared the
    /// prompt against the *parser*.
    #[test]
    fn the_function_block_tree_emits_elements_not_json() {
        let spec = compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen35()).unwrap();
        let st = statics(&spec);
        let all = st.join("");

        assert!(all.contains("<tool_call>\n<function="), "{st:?}");
        assert!(all.contains("<parameter=path>\n"), "{st:?}");
        assert!(all.contains("</function>\n</tool_call>"), "{st:?}");

        // None of the JSON scaffolding survives — no quoted keys, no argument
        // object, no comma separators between arguments.
        assert!(!all.contains("\"arguments\""), "{st:?}");
        assert!(!all.contains("\"path\""), "{st:?}");
        assert!(!all.contains(", "), "a JSON separator leaked in: {st:?}");
    }

    /// The act name's closing `>` rides on each arm, so a name that prefixes
    /// another stays distinguishable in the trie — the same rule the JSON
    /// shape's closing quote follows.
    #[test]
    fn a_function_name_arm_carries_its_own_terminator() {
        let spec = compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen35()).unwrap();
        let a = arms(&spec);
        assert!(a.iter().any(|s| s == "read_file>"), "{a:?}");
        assert!(a.iter().any(|s| s == "write_file>"), "{a:?}");
    }

    /// An enumerated value is raw text with the element's end marker on it —
    /// no quotes, because there is no JSON string to close.
    #[test]
    fn an_enum_value_is_unquoted_and_carries_the_end_marker() {
        let spec = compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen35()).unwrap();
        let a = arms(&spec);
        assert!(a.iter().any(|s| s == "read</parameter>"), "{a:?}");
        assert!(
            !a.iter().any(|s| s.contains("\"")),
            "an enum arm was quoted: {a:?}"
        );
    }

    /// A free value ends at the element marker rather than at a quote, which
    /// is what lets a character's prose hold quotes and newlines untouched.
    ///
    /// The marker is the **bare tag**, with no layout newlines on either side.
    /// A marker of `"\n</parameter>\n"` only fires when the model happens to put
    /// the tag on its own line, and it does not always: measured live, an `ask`
    /// ended a value `...pass again.</parameter>` with the tag hard against the
    /// prose. The span stayed open, swallowed the two elements that followed —
    /// so the `ask` appeared to carry `reflect`'s arguments — and the injected
    /// close then wrote a second tag on top. The tag alone is the model's
    /// close; the newlines around it belong to the grammar.
    ///
    /// **And the tag alone is content.** A value closes where structure
    /// follows the tag — a separator and the next element's tag — so a file
    /// holding `</parameter>` writes intact (`element_close`). Each fork goes
    /// on after the tag the model wrote, and only tags the tree offers next
    /// can close.
    #[test]
    fn a_free_value_is_terminated_by_the_element_and_not_by_a_quote() {
        let spec = compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen35()).unwrap();
        assert!(
            !spec
                .nodes
                .iter()
                .any(|n| matches!(n, NodeSpec::FreeText { .. })),
            "every raw value is an element"
        );
        let elements: Vec<(&str, &[&str], &Vec<SpecId>)> = spec
            .nodes
            .iter()
            .filter_map(|n| match n {
                NodeSpec::Element {
                    marker,
                    tails,
                    forks,
                    ..
                } => Some((*marker, *tails, forks)),
                _ => None,
            })
            .collect();
        assert!(!elements.is_empty(), "no raw value at all");
        for (marker, tails, forks) in elements {
            assert_eq!(marker, "</parameter>");
            assert_eq!(tails.len(), forks.len());
            for (tail, fork) in tails.iter().zip(forks) {
                let text = match spec.node(*fork) {
                    NodeSpec::Static { text, .. } => text.clone(),
                    NodeSpec::Branch { arms } => arms[0].0.clone(),
                    other => panic!("{other:?}"),
                };
                if tail.ends_with("<parameter=") {
                    assert!(
                        text.ends_with(">\n") && !text.contains('<'),
                        "{tail:?} → {text:?}"
                    );
                } else {
                    assert!(text.starts_with("\n</tool_call>"), "{tail:?} → {text:?}");
                }
            }
        }
    }

    /// **The tree supplies the tags**: a value with a required element still
    /// owed may not close the function, and the last value may not open
    /// another element.
    #[test]
    fn a_value_closes_only_on_a_tag_the_tree_offers_next() {
        let catalog = parse_tools(
            r#"[{"name":"pair","params":[
                {"name":"a","type":"string","required":true},
                {"name":"b","type":"string","required":true}]}]"#,
        )
        .unwrap();
        let spec = compile_tool_call_tree(&catalog, &ToolCallEnvelope::qwen35()).unwrap();
        let mut follows: Vec<&[&str]> = spec
            .nodes
            .iter()
            .filter_map(|n| match n {
                NodeSpec::Element { tails, .. } => Some(*tails),
                _ => None,
            })
            .collect();
        // Sorted by first tail: `\n</function>` before `\n<parameter=`.
        follows.sort_by_key(|t| t[0]);
        assert_eq!(follows, [Follows::Close.tails(), Follows::Element.tails()]);
    }

    /// It still compiles against a real vocabulary — the shape being right is
    /// not the same as the trie accepting it.
    #[test]
    fn the_function_block_tree_compiles() {
        let tree = compile(
            &compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen35()).unwrap(),
            &TestVocab::new(),
        )
        .unwrap();
        assert!(tree.len() > 5);
        assert_eq!(tree.label(), "tool_call");
    }

    /// **The action loop is the tree npcd actually compiles**, and it builds its
    /// own name arms.
    ///
    /// It had a hardcoded `"` where the single-call tree had one too — fixing
    /// only the latter left the loop splicing a JSON quote into a function
    /// block, so every live act arrived named `reflect"<parameter=inner_thoughts"`
    /// and was rejected as an unknown tool. Twenty-two of twenty-two ticks.
    /// The single-call tree is not what runs; this is.
    #[test]
    fn the_action_loop_writes_the_same_shape_as_a_single_call() {
        let spec = compile_action_loop(
            &catalog(),
            &ToolCallEnvelope::qwen35(),
            2,
            "<|im_end|>",
            None,
        )
        .unwrap();
        let a = arms(&spec);
        assert!(
            a.iter().any(|s| s == "read_file>"),
            "the loop's name arms are not the envelope's: {a:?}"
        );
        assert!(
            !a.iter().any(|s| s.starts_with("read_file\"")),
            "a JSON quote leaked into a function block: {a:?}"
        );
        // And it still compiles for both styles at the bound npcd uses.
        for env in [ToolCallEnvelope::qwen3(), ToolCallEnvelope::qwen35()] {
            let spec = compile_action_loop(&catalog(), &env, 2, "<|im_end|>", None).unwrap();
            compile(&spec, &TestVocab::new())
                .unwrap_or_else(|e| panic!("{:?} will not compile: {e}", env.style));
        }
    }

    /// The JSON families are untouched — the whole point of the style axis is
    /// that adding one changes nothing for the others.
    #[test]
    fn the_json_tree_is_exactly_what_it_always_was() {
        let spec = compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen3()).unwrap();
        let all = statics(&spec).join("");
        assert!(all.contains("<tool_call>\n{\"name\": \""));
        assert!(all.contains(", \"arguments\": {"));
        assert!(all.contains("}}\n</tool_call>"));
        assert!(!all.contains("<parameter="), "{all}");
    }

    /// **A bounded envelope bounds every free string value**, in both call
    /// styles; the default leaves them at the file-sized guard.
    #[test]
    fn an_envelope_bounds_its_string_values() {
        let forced = |env: &ToolCallEnvelope| -> Vec<u32> {
            compile_tool_call_tree(&catalog(), env)
                .unwrap()
                .nodes
                .iter()
                .filter_map(|n| match n {
                    NodeSpec::FreeText {
                        term:
                            Terminator::JsonString
                            | Terminator::JsonStringValue
                            | Terminator::Until { .. },
                        limits,
                        ..
                    }
                    | NodeSpec::Element { limits, .. } => Some(limits.forced_after),
                    _ => None,
                })
                .collect()
        };
        for env in [ToolCallEnvelope::qwen3(), ToolCallEnvelope::qwen35()] {
            let open = forced(&env);
            assert!(!open.is_empty(), "the catalog has a free string");
            assert!(
                open.iter().all(|&f| f == MAX_STRING_VALUE_TOKENS),
                "{open:?}"
            );
            let bounded = forced(&env.clone().with_max_string_tokens(900));
            assert!(bounded.iter().all(|&f| f == 900), "{bounded:?}");
        }
    }

    /// **A field's own bound holds beneath the envelope's**, with room for the
    /// value's two delimiters, and never lifts a value above it.
    #[test]
    fn a_field_bounds_its_own_value() {
        let tools = parse_tools(
            r#"[{"name":"reading","params":[
                  {"name":"event","type":"string","required":true,"max_tokens":200},
                  {"name":"checked","type":"string","required":true},
                  {"name":"long","type":"string","required":true,"max_tokens":5000}
            ]}]"#,
        )
        .unwrap();
        // Sorted: the builder pushes a call's fields back to front.
        let forced = |env: &ToolCallEnvelope| -> Vec<u32> {
            let mut f: Vec<u32> = compile_tool_call_tree(&tools, env)
                .unwrap()
                .nodes
                .iter()
                .filter_map(|n| match n {
                    NodeSpec::FreeText {
                        term: Terminator::JsonStringValue | Terminator::Until { .. },
                        limits,
                        ..
                    }
                    | NodeSpec::Element { limits, .. } => Some(limits.forced_after),
                    _ => None,
                })
                .collect();
            f.sort();
            f
        };
        for env in [ToolCallEnvelope::qwen3(), ToolCallEnvelope::qwen35()] {
            assert_eq!(
                forced(&env.clone().with_max_string_tokens(900)),
                [202, 900, 900]
            );
            let long = (5000 + VALUE_DELIMITER_TOKENS).min(MAX_STRING_VALUE_TOKENS);
            let mut want = vec![202, long, MAX_STRING_VALUE_TOKENS];
            want.sort();
            assert_eq!(forced(&env), want);
        }
    }

    /// The envelope the action loop uses: `close` carries no turn terminator,
    /// because the finishing arm does.
    pub(in super::super) fn loop_env() -> ToolCallEnvelope {
        ToolCallEnvelope {
            close: "}}\n</tool_call>".to_string(),
            ..ToolCallEnvelope::qwen3()
        }
    }

    /// **The bound binds, and it is a bound rather than a target.**
    ///
    /// Every level offers the closing arm, so a character that has said what it
    /// means stops at one call. What the bound removes is the unbounded loop a
    /// model can sit in — the same runaway the reasoning block had.
    #[test]
    fn the_loop_admits_up_to_its_bound_and_can_always_stop() {
        for max in 1..=4 {
            let spec = compile_tool_call_loop(&catalog(), &loop_env(), max, "<|im_end|>").unwrap();

            // One name branch per level. Counted by an arm naming a tool rather
            // than by arm *count* — an enum parameter also compiles to a branch
            // with as many arms as the catalog has tools, which is a coincidence
            // that made the first version of this test pass for the wrong reason.
            let name_branches = spec
                .nodes
                .iter()
                .filter(|n| {
                    matches!(n, NodeSpec::Branch { arms }
                        if arms.iter().any(|(t, _)| t == "read_file\""))
                })
                .count();
            assert_eq!(name_branches, max, "expected {max} levels");

            // And a continuation branch after every call, each of which offers
            // the closing arm — so stopping is always reachable.
            let continuations: Vec<&NodeSpec> = spec
                .nodes
                .iter()
                .filter(|n| {
                    matches!(n, NodeSpec::Branch { arms }
                        if arms.iter().any(|(t, _)| t == "<|im_end|>"))
                })
                .collect();
            assert_eq!(continuations.len(), max);
            for c in continuations {
                let NodeSpec::Branch { arms } = c else {
                    unreachable!()
                };
                assert!(
                    arms.iter().any(|(text, _)| text == "<|im_end|>"),
                    "a level with no way to stop is an unbounded loop"
                );
            }

            compile(&spec, &TestVocab::new()).expect("the loop must compile");
        }
    }

    /// **Thinking flows into acting as a node edge, with no free join.**
    ///
    /// The reason it is one tree rather than two composed by a trigger: a
    /// trigger fires on a *decoded* token, so a marker the grammar injects as
    /// static text may not fire one — and the model would be free at exactly the
    /// join the grammar exists to close. Spliced, the block's closing tag walks
    /// straight into the first call.
    #[test]
    fn a_thinking_turn_walks_from_the_block_into_the_calls() {
        use crate::stencil::think::{compile_think_tree, ThinkMode, ThinkSteerEnvelope};

        let env = ThinkSteerEnvelope {
            think_open: 1,
            think_close: 2,
            eos: 3,
            // This test asserts the SPLICE closes the join, so the injected
            // marker must stay out of it.
            after_close: "",
        };
        let prelude = compile_think_tree(ThinkMode::Balanced, &env);
        let spec =
            compile_action_loop(&catalog(), &loop_env(), 2, "<|im_end|>", Some(&prelude)).unwrap();

        // The spliced prelude is present…
        assert!(
            spec.nodes
                .iter()
                .any(|n| matches!(n, NodeSpec::FreeText { .. })),
            "the reasoning span did not survive the splice"
        );
        // …and nothing still points at an `End` that would have handed control
        // back between the block and the calls. Every edge out of the prelude
        // leads onward.
        let ends: Vec<usize> = spec
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n, NodeSpec::End))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(ends.len(), 2, "one live End, one spliced-over remnant");

        // The turn still starts inside the block, not at a call.
        assert!(
            !matches!(&spec.nodes[spec.root.0], NodeSpec::Static { text, .. } if text.contains("name")),
            "a thinking turn must begin in the block"
        );
        compile(&spec, &TestVocab::new()).expect("the spliced tree must compile");
    }

    /// A turn that may make no calls is a turn that cannot act, which for an
    /// action loop is not a configuration — it is a mistake with no symptom.
    #[test]
    fn a_loop_of_zero_calls_is_refused() {
        assert!(compile_tool_call_loop(&catalog(), &loop_env(), 0, "<|im_end|>").is_err());
        assert!(compile_tool_call_loop(&[], &loop_env(), 3, "<|im_end|>").is_err());
    }

    #[test]
    fn empty_catalog_errors() {
        assert!(matches!(
            compile_tool_call_tree(&[], &ToolCallEnvelope::qwen3()),
            Err(BuildError::ToolSchema(_))
        ));
    }

    #[test]
    fn optional_only_tool_heals_brace_quote_merge() {
        // Reproduces the real-tokenizer failure: a `datetime`-style tool with no
        // required params and one optional string.  Its args object opens `{`
        // immediately followed by the optional-gate branch (`"timezone": …` or
        // the close `}…`).  With a tokenizer that merges `{"` and `{}` into one
        // token (as Qwen3 does), the `{`→branch boundary must HEAL rather than
        // error.
        let v = TestVocab::new()
            .with_special("{\"", 300)
            .with_special("{}", 301);
        let tools = parse_tools(
            r#"[{"name":"datetime","params":[
                  {"name":"timezone","type":"string","required":false}]}]"#,
        )
        .unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        // Must compile (no BoundaryMerge) and be walkable.
        let tree = compile(&spec, &v).unwrap();
        assert!(tree.len() > 3);
    }

    #[test]
    fn gate_after_string_value_compiles_with_quote_comma_merge() {
        // A tool with a required string field and an optional field: the optional
        // gate sits *after* the string value.  With a tokenizer that merges
        // `",` (closing quote + comma) into one token, the old in-context
        // lowering tried to retract the committed opening quote and errored.  The
        // clean-boundary lowering of free-text successors fixes it.
        let v = TestVocab::new().with_special("\",", 300);
        let tools = parse_tools(
            r#"[{"name":"write_file","params":[
                  {"name":"path","type":"string","required":true},
                  {"name":"create","type":"boolean","required":false}]}]"#,
        )
        .unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        let tree = compile(&spec, &v).unwrap();
        assert!(tree.len() > 5);
    }

    #[test]
    fn number_field_gate_compiles_with_digit_delimiter_merge() {
        // A number (lookahead) field with an optional → a gate follows the value.
        // A tokenizer that merges the value's last digit with the gate's
        // delimiter (`0,` / `0}`) must not break compilation: the gate is lowered
        // from a fresh boundary, not in the value's `…0` context.
        let v = TestVocab::new()
            .with_special("0,", 300)
            .with_special("0}", 301);
        let tools = parse_tools(
            r#"[{"name":"wait","params":[
                  {"name":"secs","type":"integer","required":true},
                  {"name":"unit","type":"string","required":false}]}]"#,
        )
        .unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        let tree = compile(&spec, &v).unwrap();
        assert!(tree.len() > 5);
    }
}
