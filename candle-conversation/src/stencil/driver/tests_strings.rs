//! Tests of string and free-text values: how their open and close tokens are written,
//! healed and dropped.
//!
//! A value with no lead-in is prefilled only up to its key, so the model writes its own
//! leading space. A string's opening quote is the model's to write. A closer where a
//! value belonged is not taken as string content, and an escaped quote does not close
//! the value. A value closes on its tag alone, with no newline before it, and the empty
//! string the model writes in one token closes the value cleanly.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::stencil::compile::compile;

    use crate::stencil::tool_call::{compile_tool_call_tree, parse_tools, ToolCallEnvelope};

    use crate::stencil::vocab::{TestVocab, Vocab};

    pub(in super::super) fn tool_tree(catalog: &str) -> Arc<StencilTree> {
        let tools = parse_tools(catalog).unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        Arc::new(compile(&spec, &TestVocab::new()).unwrap())
    }

    /// Drive the walk to follow `target` (the full materialized call), acting as
    /// a mask-respecting sampler.  Returns the emitted byte string.
    pub(in super::super) fn follow(
        tree: Arc<StencilTree>,
        target: &str,
        vocab: &TestVocab,
    ) -> String {
        let bytes = target.as_bytes();
        let mut driver = StencilDriver::new(tree);
        let mut pos = 0usize;
        let mut out: Vec<u8> = Vec::new();
        let mut guard = 0usize;
        loop {
            guard += 1;
            assert!(guard < 100_000, "runaway");
            match driver.step() {
                StepMask::Prefill(run) => {
                    // The whole run is fixed; it must match the target verbatim.
                    let rb = vocab.decode(&run);
                    assert_eq!(
                        &bytes[pos..pos + rb.len()],
                        &rb[..],
                        "prefill run mismatch at {pos} (target {target:?})"
                    );
                    out.extend_from_slice(&rb);
                    pos += rb.len();
                }
                StepMask::Branch(set) => {
                    let b = bytes[pos];
                    assert!(
                        set.contains(b as TokenId),
                        "byte {b} not in branch frontier"
                    );
                    out.push(b);
                    driver.accept(b as TokenId, &[b]);
                    pos += 1;
                }
                StepMask::Free { .. } => {
                    let b = bytes[pos];
                    out.push(b);
                    driver.accept(b as TokenId, &[b]);
                    pos += 1;
                }
                StepMask::Done => break,
            }
        }
        String::from_utf8(out).unwrap()
    }

    pub(in super::super) const COMMANDS: &str = r#"[{"name":"run_commands","params":[
        {"name":"commands","type":"array","required":true}]}]"#;

    /// An array argument is the model's from its leading space on: the call
    /// reads exactly as it would have been written unsteered.
    #[test]
    fn an_array_argument_is_written_by_the_model_from_its_space() {
        let v = TestVocab::new();
        let target = "<tool_call>\n{\"name\": \"run_commands\", \"arguments\": \
                      {\"commands\": [\"pwd\", \"ls\"]}}\n</tool_call>";
        assert_eq!(follow(tool_tree(COMMANDS), target, &v), target);
    }

    /// **The key stops where the model's own token begins.** A value with no
    /// lead-in is prefilled up to `":` and no further, so the model writes
    /// ` [` itself. Prefilled through a bare space, it was left mid-token: calls
    /// came out `"commands":  [` with the space doubled, and on one Cline turn
    /// the first token for the value was a space and a closer, which ended the
    /// value empty — `{"commands":  }}`, not a call.
    #[test]
    fn a_value_without_a_lead_in_is_keyed_up_to_the_colon() {
        let v = TestVocab::new();
        let mut driver = StencilDriver::new(tool_tree(COMMANDS));
        let mut text: Vec<u8> = Vec::new();
        loop {
            match driver.step() {
                StepMask::Prefill(run) => text.extend_from_slice(&v.decode(&run)),
                StepMask::Branch(set) => {
                    let t = set.tokens()[0];
                    text.extend_from_slice(&v.token_bytes(t));
                    driver.accept(t, &v.token_bytes(t));
                }
                StepMask::Free { .. } => break,
                StepMask::Done => panic!("the value was never decoded"),
            }
        }
        let text = String::from_utf8(text).unwrap();
        assert!(text.ends_with("{\"commands\":"), "{text:?}");
    }

    /// **A delimiter the grammar does not continue with reaches the decode loop
    /// as a drop**, so it is never committed, and the grammar's own structure is
    /// the next thing written. Here the model ends a number with `]` inside an
    /// object whose `}` comes next. (A `number`: an integer's digits and its
    /// ending are masked, so a stray `]` cannot be sampled there at all.)
    #[test]
    fn a_misplaced_delimiter_after_a_value_is_dropped() {
        let v = TestVocab::new();
        let tree = tool_tree(
            r#"[{"name":"seek","params":[{"name":"at","type":"number","required":true}]}]"#,
        );
        let mut driver = StencilDriver::new(tree);
        let mut text: Vec<u8> = Vec::new();
        let mut script = b" 7]".iter();
        let healed = loop {
            match driver.step() {
                StepMask::Prefill(run) => text.extend_from_slice(&v.decode(&run)),
                StepMask::Free { .. } => {
                    let b = *script.next().expect("script ran out");
                    match driver.accept(b as TokenId, &[b]) {
                        Healed::No => text.push(b),
                        other => break other,
                    }
                }
                other => panic!("unexpected {other:?}"),
            }
        };
        assert_eq!(healed, Healed::Drop);
        assert_eq!(driver.stats().dropped_delimiters, 1);
        assert!(!driver.stats().bailed);
        // The grammar writes the close the model did not.
        let StepMask::Prefill(run) = driver.step() else {
            panic!("the close is prefilled after the drop");
        };
        text.extend_from_slice(&v.decode(&run));
        assert_eq!(driver.step(), StepMask::Done);
        assert_eq!(
            String::from_utf8(text).unwrap(),
            "<tool_call>\n{\"name\": \"seek\", \"arguments\": {\"at\": 7}}\n</tool_call>"
        );
    }

    #[test]
    fn drives_a_multi_tool_call_with_prefilled_statics() {
        let v = TestVocab::new();
        let tree = tool_tree(
            r#"[
              {"name":"read_file","params":[{"name":"path","type":"string","required":true}]},
              {"name":"set_mode","params":[{"name":"mode","type":"string","required":true,
                 "enum":["read","write"]}]}
            ]"#,
        );
        let target =
            "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>";
        let out = follow(Arc::clone(&tree), target, &v);
        assert_eq!(out, target);
    }

    /// **The closing tag is the delimiter; the newline around it is not.**
    ///
    /// A character does not put `</parameter>` on its own line reliably. Live,
    /// one wrote `…to pass again.</parameter>` — tag hard against the prose —
    /// and against a `"\n</parameter>\n"` marker that closed nothing: the span
    /// stayed open, ate the elements that followed, and the call arrived
    /// carrying the *next* act's arguments while missing its own.
    ///
    /// Here the whole block is followed byte for byte with no newline before a
    /// single closing tag, and it round-trips.
    #[test]
    fn a_value_closes_on_the_tag_alone_with_no_newline_before_it() {
        let v = TestVocab::new();
        let tools = parse_tools(
            r#"[{"name":"say","params":[
                 {"name":"to","type":"string","required":true},
                 {"name":"words","type":"string","required":true}]}]"#,
        )
        .unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen35()).unwrap();
        let tree = Arc::new(compile(&spec, &v).unwrap());

        let target = "<tool_call>\n<function=say>\n<parameter=to>\nMira</parameter>\
                      \n<parameter=words>\nyou take the order now.</parameter>\
                      \n</function>\n</tool_call>";
        assert_eq!(follow(tree, target, &v), target);
    }

    #[test]
    fn emits_a_static_run_as_one_prefill() {
        // Single tool: the name folds into the open static run, so the whole
        // envelope up to the value arrives as one `Prefill` (not token-by-token).
        let v = TestVocab::new();
        let tree = tool_tree(
            r#"[{"name":"ping","params":[{"name":"n","type":"integer","required":true}]}]"#,
        );
        let mut driver = StencilDriver::new(Arc::clone(&tree));
        match driver.step() {
            StepMask::Prefill(run) => {
                let text = String::from_utf8(v.decode(&run)).unwrap();
                // One run carrying the entire opening envelope + name + arg key.
                assert!(
                    text.starts_with("<tool_call>\n{\"name\": \"ping\""),
                    "unexpected run: {text:?}"
                );
                // Through the key; in this byte vocabulary the space every
                // digit arm shares is prefilled with it.
                assert!(
                    text.trim_end().ends_with("\"n\":"),
                    "run should reach the value: {text:?}"
                );
            }
            other => panic!("expected a single Prefill run, got {other:?}"),
        }
        // After the run, the integer's first digit is a masked choice: a digit
        // or a sign, and nothing that could open a string.
        let StepMask::Branch(set) = driver.step() else {
            panic!("an integer value begins with a masked choice");
        };
        assert!(set.contains(b'7' as TokenId), "{set:?}");
        assert!(
            set.contains(b'-' as TokenId),
            "unsigned only with a minimum"
        );
        assert!(!set.contains(b'"' as TokenId), "{set:?}");
    }

    // ── Free-text exit healing (the merged exit-token edge cases) ───────────

    /// Build a driver positioned at the first field's free-text decode (after
    /// the envelope + name + key prefill).
    pub(in super::super) fn driver_at_first_value(catalog: &str, v: &TestVocab) -> StencilDriver {
        let tools = parse_tools(catalog).unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        let tree = Arc::new(compile(&spec, v).unwrap());
        let mut driver = StencilDriver::new(tree);
        assert!(matches!(driver.step(), StepMask::Prefill(_)));
        assert!(matches!(driver.step(), StepMask::Free { .. }));
        driver
    }

    pub(in super::super) const STR_OPT: &str = r#"[{"name":"write_file","params":[
        {"name":"path","type":"string","required":true},
        {"name":"create","type":"boolean","required":false}]}]"#;

    pub(in super::super) const STR_ONLY: &str = r#"[{"name":"read_file","params":[
        {"name":"path","type":"string","required":true}]}]"#;

    /// **An end-of-turn token is barred at a call value's first step, and only
    /// there**: once the value has begun, an EOS closes it as written.
    #[test]
    fn an_end_token_is_barred_only_at_a_values_first_step() {
        let v = TestVocab::new();
        let tools = parse_tools(STR_ONLY).unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        let tree = Arc::new(compile(&spec, &v).unwrap());
        let mut driver = StencilDriver::new(tree);
        let first = loop {
            match driver.step() {
                StepMask::Prefill(_) => continue,
                StepMask::Branch(set) => {
                    let t = set.tokens()[0];
                    driver.accept(t, &v.token_bytes(t));
                }
                other => break other,
            }
        };
        assert_eq!(
            first,
            StepMask::Free {
                close_boost: 0.0,
                opens_value: true
            },
            "the value's first token"
        );
        driver.accept(b' ' as TokenId, b" ");
        driver.accept(b'"' as TokenId, b"\"");
        driver.accept(b'a' as TokenId, b"a");
        assert_eq!(
            driver.step(),
            StepMask::Free {
                close_boost: 0.0,
                opens_value: false
            },
            "a value begun may end on EOS"
        );
    }

    #[test]
    fn tool_call_value_span_is_never_a_terminal_close_span() {
        // A tool-call value consumes its own close token (no suppress_close):
        // the sampler's closing-statement script must not play there.
        let v = TestVocab::new();
        let driver = driver_at_first_value(STR_ONLY, &v);
        assert!(!driver.in_terminal_close_span());
    }

    /// Write a string value's opening ` "` — the model's, not the grammar's.
    pub(in super::super) fn open_string(d: &mut StencilDriver) {
        assert_eq!(d.accept(b' ' as TokenId, b" "), Healed::No);
        assert_eq!(d.accept(b'"' as TokenId, b"\""), Healed::No);
    }

    #[test]
    fn clean_string_close_does_not_heal() {
        // Closing quote is its own byte token — a clean boundary, no heal.
        let v = TestVocab::new();
        let mut d = driver_at_first_value(STR_OPT, &v);
        open_string(&mut d);
        assert_eq!(d.accept(b'a' as TokenId, b"a"), Healed::No);
        assert_eq!(d.accept(b'"' as TokenId, b"\""), Healed::No);
    }

    /// **Bare text where the opening quote belonged is not the string's.** Let
    /// through, its first later `"` would open the string and the call's own
    /// close would be swallowed as content; instead it is replaced by a whole
    /// empty value, like any other skipped value.
    #[test]
    fn bare_text_before_the_opening_quote_is_written_empty() {
        let v = TestVocab::new();
        let mut d = driver_at_first_value(STR_OPT, &v);
        assert_eq!(
            d.accept(b'a' as TokenId, b"a"),
            Healed::Rewrite {
                bytes: b" \"\"".to_vec()
            }
        );
    }

    #[test]
    fn string_exit_merged_with_comma() {
        // `",` — quote exits at byte 0, the comma is leftover.
        let v = TestVocab::new().with_special("\",", 300);
        let mut d = driver_at_first_value(STR_OPT, &v);
        open_string(&mut d);
        assert_eq!(d.accept(b'a' as TokenId, b"a"), Healed::No);
        assert_eq!(
            d.accept(300, b"\","),
            Healed::Rewrite {
                bytes: b"\"".to_vec()
            }
        );
    }

    #[test]
    fn string_exit_merged_with_close_brace() {
        // No optional → the value is followed by the object close; the model
        // closes the string merged with the first `}` (`"}`).
        let v = TestVocab::new().with_special("\"}", 300);
        let mut d = driver_at_first_value(STR_ONLY, &v);
        open_string(&mut d);
        assert_eq!(d.accept(b'a' as TokenId, b"a"), Healed::No);
        assert_eq!(
            d.accept(300, b"\"}"),
            Healed::Rewrite {
                bytes: b"\"".to_vec()
            }
        );
    }

    #[test]
    fn string_exit_with_value_content_in_token() {
        // The exit token also carries the last value byte: `h",` → consumed=2
        // (the `h` value byte + the closing quote), `,` leftover.
        let v = TestVocab::new().with_special("h\",", 300);
        let mut d = driver_at_first_value(STR_OPT, &v);
        open_string(&mut d);
        assert_eq!(
            d.accept(300, b"h\","),
            Healed::Rewrite {
                bytes: b"h\"".to_vec()
            }
        );
    }

    /// **The empty string Qwen actually writes — ` ""`, one token — closes the
    /// value cleanly.** This is the token a prefilled opening quote made
    /// unreachable; with the quote the model's, it is an ordinary exit.
    #[test]
    fn the_empty_string_token_closes_the_value() {
        let v = TestVocab::new().with_special(" \"\"", 300);
        let mut d = driver_at_first_value(STR_OPT, &v);
        assert_eq!(d.accept(300, b" \"\""), Healed::No);
    }

    /// **The failure that motivated all of this no longer swallows the call.**
    /// A model that writes `}}` where the value belonged had it taken as string
    /// content, with the call's own close after it. Before any opening quote it
    /// is replaced by a whole empty value and the grammar carries on.
    #[test]
    fn a_closer_where_the_value_belonged_is_not_string_content() {
        let v = TestVocab::new().with_special("}}", 300);
        let mut d = driver_at_first_value(STR_OPT, &v);
        assert_eq!(
            d.accept(300, b"}}"),
            Healed::Rewrite {
                bytes: b" \"\"".to_vec()
            },
            "`}}` before the opening quote is replaced by an empty value"
        );
    }

    #[test]
    fn escaped_quote_does_not_exit() {
        // An escaped quote mid-value must not be treated as the close.
        let v = TestVocab::new();
        let mut d = driver_at_first_value(STR_OPT, &v);
        open_string(&mut d);
        assert_eq!(d.accept(b'\\' as TokenId, b"\\"), Healed::No);
        assert_eq!(d.accept(b'"' as TokenId, b"\""), Healed::No); // escaped — not a close
        assert_eq!(d.accept(b'b' as TokenId, b"b"), Healed::No);
        assert_eq!(d.accept(b'"' as TokenId, b"\""), Healed::No); // real close, clean
    }
}
