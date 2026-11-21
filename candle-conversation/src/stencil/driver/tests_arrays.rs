//! Tests that a string argument's close decides whether another element follows, and
//! that array elements are steered as the model writes them.
//!
//! The model's own token decides what follows a string close. A quote followed by a
//! comma means another element follows, and a bracket closes the array. A token whose
//! rest leads nowhere the branch offers narrows nothing. The tests also pin that an
//! end-of-turn token inside a value closes the call and not the turn, and that an
//! early stop moves on to the next argument, so the rest of the call is still written.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::stencil::compile::compile;

    use crate::stencil::named_arm::{arm_name, named_arm};

    use crate::stencil::tool_call::{compile_tool_call_tree, parse_tools, ToolCallEnvelope};

    use crate::stencil::vocab::{TestVocab, Vocab};

    use super::super::tests_strings::tests::tool_tree;

    pub(in super::super) const ARGS: &str = r#"[{"name":"run","params":[
        {"name":"args","type":"array","required":true,"items":{"type":"string"}}]}]"#;

    /// A driver inside `run`'s `args`, with the element `a` written and its
    /// closing quote still to come.
    pub(in super::super) fn driver_in_an_element(v: &TestVocab) -> StencilDriver {
        let tools = parse_tools(ARGS).unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        let mut driver = StencilDriver::new(Arc::new(compile(&spec, v).unwrap()));
        let target = b"<tool_call>\n{\"name\": \"run\", \"arguments\": {\"args\": [\"a";
        let mut pos = 0;
        while pos < target.len() {
            match driver.step() {
                StepMask::Prefill(run) => pos += v.decode(&run).len(),
                StepMask::Branch(_) | StepMask::Free { .. } => {
                    driver.accept(target[pos] as TokenId, &[target[pos]]);
                    pos += 1;
                }
                StepMask::Done => panic!("the call ended early"),
            }
        }
        driver
    }

    /// **A token that closes a string says what comes next, and that decision
    /// is the model's.** `",` commits as `"`, but the comma it carried means
    /// another element follows. Asked again from a bare quote, a model reached
    /// for the close instead: `["test", "-p", "web"]` came out `["test", "-p"]`
    /// on six calls of six, with `web` in its reasoning each time.
    #[test]
    fn a_string_closed_with_a_comma_goes_on_to_the_next_element() {
        let v = TestVocab::new().with_special("\",", 300);
        let mut d = driver_in_an_element(&v);
        assert_eq!(
            d.accept(300, b"\","),
            Healed::Rewrite {
                bytes: b"\"".to_vec()
            }
        );
        let StepMask::Branch(set) = d.step() else {
            panic!("an array goes on or closes by a masked choice");
        };
        assert!(set.contains(b',' as TokenId), "{set:?}");
        assert!(!set.contains(b']' as TokenId), "{set:?}");
    }

    /// **The same decision the other way.** `"]` closes the array; asked again,
    /// a model went on and wrote `--test-threads=1` a dozen times over.
    #[test]
    fn a_string_closed_with_a_bracket_closes_the_array() {
        let v = TestVocab::new().with_special("\"]", 300);
        let mut d = driver_in_an_element(&v);
        assert_eq!(
            d.accept(300, b"\"]"),
            Healed::Rewrite {
                bytes: b"\"".to_vec()
            }
        );
        let StepMask::Branch(set) = d.step() else {
            panic!("an array goes on or closes by a masked choice");
        };
        assert!(set.contains(b']' as TokenId), "{set:?}");
        assert!(!set.contains(b',' as TokenId), "{set:?}");
    }

    /// A token whose rest leads nothing the branch offers narrows nothing: the
    /// model chooses from every arm, and no call can be left with no way on.
    #[test]
    fn a_rest_that_leads_no_arm_leaves_the_choice_whole() {
        let v = TestVocab::new().with_special("\"x", 300);
        let mut d = driver_in_an_element(&v);
        d.accept(300, b"\"x");
        let StepMask::Branch(set) = d.step() else {
            panic!("an array goes on or closes by a masked choice");
        };
        assert!(set.contains(b',' as TokenId), "{set:?}");
        assert!(set.contains(b']' as TokenId), "{set:?}");
    }

    #[test]
    fn out_of_grammar_token_bails_and_terminates() {
        let v = TestVocab::new();
        let tree = tool_tree(
            r#"[{"name":"read_file","params":[{"name":"path","type":"string","required":true}]},
                {"name":"write_file","params":[{"name":"path","type":"string","required":true}]}]"#,
        );
        let mut driver = StencilDriver::new(Arc::clone(&tree));
        let mut out: Vec<u8> = Vec::new();

        // Prefill the open run until we reach the name branch.
        let branch = loop {
            match driver.step() {
                StepMask::Prefill(run) => out.extend_from_slice(&v.decode(&run)),
                StepMask::Branch(set) => break set,
                other => panic!("expected a prefill run then a branch, got {other:?}"),
            }
        };
        // Feed a token the branch forbids (simulating a mask that didn't hold).
        let bad = b'Z' as TokenId;
        assert!(!branch.contains(bad));
        out.push(b'Z');
        driver.accept(bad, b"Z");

        // The driver now yields the bail run, then finishes.
        loop {
            match driver.step() {
                StepMask::Prefill(run) => out.extend_from_slice(&v.decode(&run)),
                StepMask::Done => break,
                other => panic!("after bail expected the bail run then Done, got {other:?}"),
            }
        }
        assert!(driver.is_done());
        assert!(
            driver.stats().bailed,
            "the bail must be recorded in the path stats"
        );
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains('Z'));
        assert!(
            text.ends_with("</tool_call>"),
            "bail must terminate the block: {text:?}"
        );
    }

    /// **EOS inside a value does not end the turn — the stencil closes the
    /// call.**
    ///
    /// The hole this closes. A free-text span is the one place the stencil hands
    /// the sampler the whole vocabulary, so EOS is samplable there; and until
    /// this, a span with neither `eos_ends` nor a `close_token` had no reaction
    /// to it — the token fell through to a byte terminator that could never
    /// match it, the cursor stayed parked in the value, and the decode loop
    /// sealed the turn on the EOS it had just sampled. The structural statics
    /// after the span were never injected.
    ///
    /// Measured on the persisted substrate before the fix: **244 of 259 turns**
    /// wrote a complete, correct function block and never closed it. Every one
    /// was discarded as narration, so the character read as doing nothing at
    /// all.
    ///
    /// A stencil is a guarantee about what may be emitted. A token that ends
    /// the turn from inside one is a hole in that guarantee, not a style of
    /// ending.
    #[test]
    fn eos_inside_a_value_is_swallowed_and_the_call_still_closes() {
        let v = TestVocab::new();
        let tree = tool_tree(
            r#"[{"name":"read_file","params":[{"name":"path","type":"string","required":true}]}]"#,
        );
        let mut driver = StencilDriver::new(Arc::clone(&tree));
        let mut out: Vec<u8> = Vec::new();

        // Walk to the value span, writing whatever the grammar asks for.
        let value = b"a.rs";
        let mut wrote = 0usize;
        let mut dropped = false;
        loop {
            match driver.step() {
                StepMask::Prefill(run) => out.extend_from_slice(&v.decode(&run)),
                StepMask::Branch(_) => {
                    // The one-tool catalog still branches on the name; follow it.
                    let b = b"read_file\""[wrote.min(9)];
                    out.push(b);
                    driver.accept(b as TokenId, &[b]);
                    wrote += 1;
                }
                StepMask::Free { .. } => {
                    // Part of the value, then stop mid-string with EOS — the
                    // model deciding it has said enough.
                    if let Some(&b) = value.get(out.len() % value.len()) {
                        if !dropped && out.last() != Some(&b'"') && wrote < 40 {
                            out.push(b);
                            driver.accept(b as TokenId, &[b]);
                            wrote += 1;
                            continue;
                        }
                    }
                    // Intercepted: the EOS is never committed. What replaces it
                    // is the string's closing quote, which the decode loop
                    // commits in its place.
                    let eos = tree.eos();
                    match driver.accept(eos, b"") {
                        Healed::Rewrite { bytes } => out.extend_from_slice(&bytes),
                        Healed::Drop => {}
                        Healed::No => panic!("EOS in a byte-terminated span was not intercepted"),
                    }
                    dropped = true;
                }
                StepMask::Done => break,
            }
        }

        let text = String::from_utf8_lossy(&out).to_string();
        assert!(
            text.ends_with("</tool_call>"),
            "the stencil did not close the call after EOS: {text:?}"
        );
        assert!(
            !text.contains('\u{0}'),
            "the EOS was committed into the output: {text:?}"
        );
        assert!(driver.is_done());
    }

    /// **An intercepted EOS moves to the next argument, it does not end the
    /// call.**
    ///
    /// The cursor goes to the span's own `next`, which for anything but the
    /// last argument is the following `<parameter=…>` static. So a model that
    /// stops early does not truncate the call — the tree walks on and emits
    /// every remaining argument's scaffold, then closes.
    ///
    /// Asserted on a three-argument act because a one-argument one cannot tell
    /// "moved on" from "closed": they are the same node.
    #[test]
    fn eos_moves_to_the_next_argument_rather_than_ending_the_call() {
        let v = TestVocab::new();
        let tree = tool_tree(
            r#"[{"name":"reflect","params":[
                 {"name":"inner_thoughts","type":"string","required":true},
                 {"name":"feeling","type":"string","required":true},
                 {"name":"my_reflections","type":"string","required":true}]}]"#,
        );
        let mut driver = StencilDriver::new(Arc::clone(&tree));
        let mut out: Vec<u8> = Vec::new();
        let mut name = b"reflect\"".iter();
        // Stop dead on the very first thing the model gets to choose freely.
        loop {
            match driver.step() {
                StepMask::Prefill(run) => out.extend_from_slice(&v.decode(&run)),
                StepMask::Branch(_) => match name.next() {
                    Some(&b) => {
                        out.push(b);
                        driver.accept(b as TokenId, &[b]);
                    }
                    None => panic!("branch after the name was exhausted: {out:?}"),
                },
                StepMask::Free { .. } => match driver.accept(tree.eos(), b"") {
                    // What the decode loop commits in the EOS's place.
                    Healed::Rewrite { bytes } => out.extend_from_slice(&bytes),
                    Healed::Drop => {}
                    Healed::No => panic!("EOS was committed: {out:?}"),
                },
                StepMask::Done => break,
            }
        }
        let text = String::from_utf8_lossy(&out).to_string();
        // Every argument's scaffold was still written, in order.
        for arg in ["inner_thoughts", "feeling", "my_reflections"] {
            assert!(
                text.contains(arg),
                "EOS in the first value dropped `{arg}`: {text:?}"
            );
        }
        assert!(
            text.ends_with("</tool_call>"),
            "the call was not closed: {text:?}"
        );
        // **And what came out is a readable call.** The terminator never fired
        // — the model wrote nothing at all — so every closing delimiter here
        // was injected by the tree. Without them the values ran on unquoted and
        // the JSON was malformed, which loses the whole call rather than the one
        // argument the model stopped inside.
        let body = text
            .trim_start_matches("<tool_call>\n")
            .trim_end_matches("</tool_call>")
            .trim();
        let v: serde_json::Value = serde_json::from_str(body)
            .unwrap_or_else(|e| panic!("the interrupted call is not valid JSON ({e}): {body:?}"));
        let args = &v["arguments"];
        for arg in ["inner_thoughts", "feeling", "my_reflections"] {
            assert_eq!(args[arg], "", "`{arg}` did not survive as an empty value");
        }
    }

    #[test]
    fn path_stats_track_a_clean_call() {
        let v = TestVocab::new();
        let tree = tool_tree(
            r#"[{"name":"read_file","params":[{"name":"path","type":"string","required":true}]}]"#,
        );
        let target =
            "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>";
        let bytes = target.as_bytes();
        let mut driver = StencilDriver::new(Arc::clone(&tree));
        let mut pos = 0usize;
        loop {
            match driver.step() {
                StepMask::Prefill(run) => pos += v.decode(&run).len(),
                StepMask::Branch(_) | StepMask::Free { .. } => {
                    driver.accept(bytes[pos] as TokenId, &[bytes[pos]]);
                    pos += 1;
                }
                StepMask::Done => break,
            }
        }
        let s = driver.stats();
        assert!(!s.bailed, "clean call must not bail: {s:?}");
        assert!(
            s.prefills > 0 && s.prefill_tokens > 0,
            "envelope/close prefilled as static runs: {s:?}"
        );
        assert!(s.free_tokens > 0, "the path value was free-decoded: {s:?}");
    }

    /// The scaffold a turn can prefill, and where the model's first real choice
    /// is. `opening` exists so a caller can write the one and mask the other.
    #[test]
    fn opening_yields_the_static_scaffold_and_stops_at_the_first_choice() {
        let v = TestVocab::new();
        // Two tools, so the name is a genuine decision. With a one-tool catalog
        // there is nothing to choose and the scaffold correctly runs on through
        // the name to the first argument value.
        let tree = tool_tree(
            r#"[{"name":"read_file","params":[{"name":"path","type":"string","required":true}]},
                {"name":"say","params":[{"name":"intent","type":"string","required":true}]}]"#,
        );
        let (scaffold, action) = StencilDriver::new(tree).opening();

        // Everything up to the tool name is fixed text, so it is *written*
        // rather than sampled — which is the whole point: no sampling outcome
        // can decline to produce it or produce something else.
        assert_eq!(
            String::from_utf8(v.decode(&scaffold)).unwrap(),
            "<tool_call>\n{\"name\": \""
        );
        // And what stops the walk is the first genuine decision — masked, so an
        // invented name is not reachable rather than merely discouraged.
        let StepMask::Branch(set) = action else {
            panic!("the tool name must be a masked branch, got {action:?}");
        };
        assert!(
            set.contains(b'r' as TokenId),
            "`read_file` is in the catalog"
        );
        assert!(set.contains(b's' as TokenId), "`say` is in the catalog");
        assert!(
            !set.contains(b'l' as TokenId),
            "a name outside the catalog — `look` — must be unreachable"
        );
    }

    pub(in super::super) const THREE_TOOLS: &str = r#"[
        {"name":"list_dir","params":[{"name":"path","type":"string","required":true}]},
        {"name":"read_file","params":[{"name":"path","type":"string","required":true}]},
        {"name":"write_file","params":[{"name":"path","type":"string","required":true}]}]"#;

    /// Walk to the first value, answering every branch with the lowest token
    /// the mask allows; returns the text written.
    pub(in super::super) fn lowest_until_value(
        driver: &mut StencilDriver,
        v: &TestVocab,
    ) -> String {
        let mut text: Vec<u8> = Vec::new();
        loop {
            match driver.step() {
                StepMask::Prefill(run) => text.extend_from_slice(&v.decode(&run)),
                StepMask::Branch(set) => {
                    let t = set.tokens()[0];
                    text.extend_from_slice(&v.token_bytes(t));
                    driver.accept(t, &v.token_bytes(t));
                }
                StepMask::Free { .. } | StepMask::Done => break,
            }
        }
        String::from_utf8(text).unwrap()
    }

    /// **The tool the reasoning named is the tool written.** Unsteered, a
    /// sampler that takes the lowest token writes `list_dir`; steered to the
    /// arm `named_arm` picks from the reasoning, it writes `write_file`.
    #[test]
    fn the_first_branch_is_steered_to_the_named_arm() {
        let v = TestVocab::new();
        let tree = tool_tree(THREE_TOOLS);
        let unsteered = lowest_until_value(&mut StencilDriver::new(Arc::clone(&tree)), &v);
        assert!(unsteered.contains("\"list_dir\""), "{unsteered:?}");

        let arms = tree.first_branch().expect("the name is a branch").arms();
        let decoded: Vec<Vec<u8>> = arms.iter().map(|a| v.decode(a)).collect();
        let names: Vec<&str> = decoded.iter().map(|b| arm_name(b)).collect();
        let reasoning = "The file is new, so I will call write_file with the path.";
        let pick = named_arm(reasoning, &names).expect("one tool is named");

        let mut driver = StencilDriver::new(tree);
        driver.steer_first_branch(arms[pick].clone());
        let steered = lowest_until_value(&mut driver, &v);
        assert!(steered.contains("\"write_file\""), "{steered:?}");
        assert!(!driver.stats().bailed);
    }

    /// An arm the branch does not have steers nothing: the frontier stays whole.
    #[test]
    fn an_arm_outside_the_branch_releases_the_steer() {
        let v = TestVocab::new();
        let mut driver = StencilDriver::new(tool_tree(THREE_TOOLS));
        driver.steer_first_branch(v.encode("zebra\""));
        let text = lowest_until_value(&mut driver, &v);
        assert!(text.contains("\"list_dir\""), "{text:?}");
    }

    /// Replaying the walk gives the same scaffold and the same frontier. The
    /// turn path depends on it: the assistant prefill is built from one walk and
    /// the decode driver is armed by a second, and they must land on one node.
    #[test]
    fn opening_is_deterministic_so_two_walks_agree() {
        let tree = tool_tree(
            r#"[{"name":"read_file","params":[{"name":"path","type":"string","required":true}]}]"#,
        );
        let (a, a_action) = StencilDriver::new(Arc::clone(&tree)).opening();
        let (b, b_action) = StencilDriver::new(tree).opening();
        assert_eq!(a, b, "the scaffold moved between two walks of one tree");
        assert_eq!(a_action, b_action, "the frontier moved");
    }
}
