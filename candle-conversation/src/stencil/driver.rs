//! Online, sampler-driven walk of a stencil tree.
//!
//! Where [`simulate`](super::sim::simulate) drives a whole tree with an internal
//! oracle, `StencilDriver` is the *online* counterpart the decode loop uses: each
//! step it yields what to do next ([`StencilDriver::step`]) — prefill a static
//! run, mask a branch, free-decode a span, or finish — and a decoded token is fed
//! back via [`StencilDriver::accept`].
//!
//! A `Static` run is handed back whole as [`StepMask::Prefill`] so the scheduler
//! can inject it in one prefill pass; the bail failsafe's closing run flows the
//! same way and then the driver finishes.

use std::sync::Arc;

use super::mask::AllowedSet;
use super::session::{Observe, StencilAction, StencilSession};
use super::tree::StencilTree;
use super::vocab::TokenId;

/// What [`StencilDriver::accept`] decided about the sampled token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Healed {
    /// The token was accepted as-is — commit it normally.
    No,
    /// Commit the re-tokenized `bytes` in place of the sampled token.
    ///
    /// A free-text span ended *inside* this token (the model merged the value's
    /// close with the next node's delimiter — `",` — and `bytes` is the part
    /// that belongs to the value); or the token could not be committed as
    /// written, and `bytes` is its repair — a character escaped, a malformed
    /// value completed, an EOS inside a value replaced by the text that closes
    /// it, a value at its hard limit completed after the token. The session has
    /// already moved on as if the model had written `bytes`. Never empty: a
    /// token that contributes nothing is [`Healed::Drop`].
    Rewrite { bytes: Vec<u8> },
    /// Drop this sampled token entirely: it must NOT be committed to the
    /// sequence's KV.  A token-closed free-text span with `suppress_close` ended
    /// on this exact token (the close token, e.g. `</think>`), and the span
    /// suppresses it so the steering can retry — the session cursor has already
    /// advanced, so the next [`step`](StencilDriver::step) yields the
    /// continuation [`Prefill`](StepMask::Prefill) that prefills in its place.
    /// (The decode-loop side that skips the commit lands separately in
    /// `scheduler/decode.rs`; the stencil core only produces this signal.)
    Drop,
}

/// What the decode loop should do for this sequence next.
#[derive(Debug, Clone, PartialEq)]
pub enum StepMask {
    /// Inject this `Static` run (or the bail closing run) into the sequence's KV.
    /// The session cursor has already advanced past the run, so the next
    /// [`step`](StencilDriver::step) yields the node after it.
    Prefill(Vec<TokenId>),
    /// Mask the sampler to this allowed set (a branch frontier).
    Branch(AllowedSet),
    /// Free decode within a span (normal decode — EOS and any close token are
    /// intercepted by the session, never banned).  `close_boost` is the soft
    /// close-token logit ramp.
    Free { close_boost: f32 },
    /// The stencil finished — resume unconstrained decoding.
    Done,
}

/// A running tally of the path a [`StencilDriver`] walk took — emitted as the
/// steering's finish trace so a malformed call is diagnosable at a glance
/// (e.g. `bailed=true`, or `free_tokens=0` where a value was expected).
#[derive(Debug, Clone, Copy, Default)]
pub struct PathStats {
    /// Static runs prefilled atomically (structural scaffold: envelope, keys,
    /// punctuation).
    pub prefills: u32,
    /// Tokens across those prefilled runs.
    pub prefill_tokens: u32,
    /// Masked branch decodes (constrained choices: tool name, enum, bool,
    /// optional-field gates).
    pub branch_tokens: u32,
    /// Free decodes inside value spans (the model writing argument content).
    pub free_tokens: u32,
    /// Exit-token heals applied (the model merged a value's closing char with
    /// the next delimiter and was steered back).
    pub heals: u32,
    /// Close signals intercepted and dropped inside a span, of either kind.
    ///
    /// Two things land here and they are the same event to the driver: a
    /// thinking block's `</think>`, dropped so the successor can re-steer the
    /// model back into reasoning with a continuation phrase; and an **EOS
    /// sampled inside a tool-call value**, dropped so it never reaches the
    /// sequence and never seals the turn while the tree still has structure to
    /// emit.
    ///
    /// It was `think_continuations` and counted only the first, which stopped
    /// being true when EOS interception was extended to every span — the
    /// counter kept its name and quietly began totalling both.
    pub intercepted_closes: u32,
    /// Delimiters dropped at the end of a lookahead value because the grammar
    /// does not continue with them (a `]` where the element's `}` comes next);
    /// the grammar wrote the structure in their place.
    pub dropped_delimiters: u32,
    /// Free-span tokens committed as a repair instead of as written: a
    /// character escaped inside a string, or a malformed value ended and
    /// completed.
    pub repairs: u32,
    /// An out-of-grammar token escaped the mask and forced the bail failsafe.
    pub bailed: bool,
}

/// A live walk of one tree attached to a decoding sequence.
pub struct StencilDriver {
    session: StencilSession,
    done: bool,
    stats: PathStats,
}

impl StencilDriver {
    /// Begin a walk at the tree's root.
    pub fn new(tree: Arc<StencilTree>) -> Self {
        StencilDriver {
            session: StencilSession::new(tree),
            done: false,
            stats: PathStats::default(),
        }
    }

    /// Whether the walk has finished (the caller should drop the driver and
    /// resume free decode).
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// The path the walk took so far — for the steering finish trace.
    pub fn stats(&self) -> PathStats {
        self.stats
    }

    /// The compiled tree being walked.
    pub fn tree(&self) -> &StencilTree {
        self.session.tree()
    }

    /// Whether the cursor sits in a terminal free-text span — the only place
    /// the sampler's hard-cap closing-statement script may play — see
    /// [`StencilSession::in_terminal_close_span`].
    pub fn in_terminal_close_span(&self) -> bool {
        self.session.in_terminal_close_span()
    }

    /// Walk the tree's opening scaffold, for a turn that **begins** inside a
    /// grammar rather than entering one on a trigger token.
    ///
    /// Returns every leading `Static` run concatenated, plus the first action
    /// that actually needs the sampler.  The runs are deterministic — no token
    /// has been observed yet, so the walk from the root cannot branch — which is
    /// what lets the caller seed them into the turn's assistant prefill and then
    /// re-derive the same position here to keep the driver in step with the K/V.
    ///
    /// Prefilling that scaffold rather than decoding it is both cheaper and
    /// stronger: it costs one prefill pass instead of a decode step per token,
    /// and the tokens are *written* rather than chosen, so no sampling outcome
    /// can decline to produce them.  The first sampled token of the turn is then
    /// the first genuine decision the grammar leaves open — a masked branch.
    pub fn opening(&mut self) -> (Vec<TokenId>, StepMask) {
        let mut scaffold = Vec::new();
        loop {
            match self.step() {
                StepMask::Prefill(run) => scaffold.extend_from_slice(&run),
                action => return (scaffold, action),
            }
        }
    }

    /// What to do for the next decode step: prefill a static run, mask a branch,
    /// free-decode a span, or finish.  After a `Prefill` the caller injects the
    /// run and calls `step` again; after `Branch`/`Free` it samples a token under
    /// the constraint and feeds it back via [`accept`](Self::accept).
    pub fn step(&mut self) -> StepMask {
        loop {
            match self.session.next_action() {
                StencilAction::Prefill(toks) => {
                    // An empty run (possible after a fully-consumed push-back)
                    // carries no tokens — skip straight to the next node.
                    if toks.is_empty() {
                        continue;
                    }
                    self.stats.prefills += 1;
                    self.stats.prefill_tokens += toks.len() as u32;
                    return StepMask::Prefill(toks);
                }
                StencilAction::MaskedDecode(set) => {
                    self.stats.branch_tokens += 1;
                    return StepMask::Branch(set);
                }
                StencilAction::FreeDecode { close_boost } => {
                    self.stats.free_tokens += 1;
                    return StepMask::Free { close_boost };
                }
                StencilAction::Exit => {
                    self.done = true;
                    return StepMask::Done;
                }
            }
        }
    }

    /// Feed back the token sampled under the constraint from the preceding
    /// `Branch`/`Free` [`step`](Self::step).  `bytes` are its decoded bytes (used
    /// by free-text terminators).  An out-of-grammar token makes the session bail
    /// — its closing run is then returned as a `Prefill` on the next `step`.
    ///
    /// Returns [`Healed::Rewrite`] when the token is committed as other bytes —
    /// a span that closed strictly inside it, or a repair (see
    /// [`StencilSession::take_rewrite`]). Returns [`Healed::Drop`] when a
    /// `suppress_close` token-closed span ended on this token (the close token
    /// is dropped and the successor prefills a steering continuation), when an
    /// EOS was intercepted inside a span with nothing to close, when a
    /// lookahead value ended on a delimiter the grammar does not continue with,
    /// and when a repair leaves the token nothing to contribute.
    pub fn accept(&mut self, token: TokenId, bytes: &[u8]) -> Healed {
        let observed = self.session.observe(token, bytes);
        let rewrite = self.session.take_rewrite();
        let Ok(observe) = observed else {
            return Healed::No;
        };
        match observe {
            Observe::SpanClosed { leftover } if leftover > 0 && leftover < bytes.len() => {
                self.stats.heals += 1
            }
            // A close signal the span keeps to itself: the token never reaches
            // the sequence, and the tree writes whatever it owes — a steering
            // phrase, or the structure the EOS would otherwise have cut short.
            Observe::TokenClosedDrop => self.stats.intercepted_closes += 1,
            // A wrong delimiter after a value: dropped, and the successor
            // writes the right one.
            Observe::DelimiterDropped => self.stats.dropped_delimiters += 1,
            Observe::Repaired { .. } => self.stats.repairs += 1,
            Observe::Bailed => self.stats.bailed = true,
            _ => {}
        }
        healing(observe, rewrite, bytes)
    }
}

/// How a sampled token is committed, given what the session observed of it and
/// the rewrite it left.
///
/// Shared by [`StencilDriver::accept`] and the simulator, so the simulator
/// commits exactly what the decode loop does — a test that judged the session
/// on committing whole tokens would pass output the decode loop never produces.
pub(super) fn healing(observe: Observe, rewrite: Option<Vec<u8>>, bytes: &[u8]) -> Healed {
    match (observe, rewrite) {
        (_, Some(bytes)) if bytes.is_empty() => Healed::Drop,
        (_, Some(bytes)) => Healed::Rewrite { bytes },
        (Observe::SpanClosed { leftover }, None) if leftover > 0 && leftover < bytes.len() => {
            Healed::Rewrite {
                bytes: bytes[..bytes.len() - leftover].to_vec(),
            }
        }
        (Observe::TokenClosedDrop | Observe::DelimiterDropped, None) => Healed::Drop,
        _ => Healed::No,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stencil::compile::compile;
    use crate::stencil::tool_call::{compile_tool_call_tree, parse_tools, ToolCallEnvelope};
    use crate::stencil::vocab::{TestVocab, Vocab};

    fn tool_tree(catalog: &str) -> Arc<StencilTree> {
        let tools = parse_tools(catalog).unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        Arc::new(compile(&spec, &TestVocab::new()).unwrap())
    }

    /// Drive the walk to follow `target` (the full materialized call), acting as
    /// a mask-respecting sampler.  Returns the emitted byte string.
    fn follow(tree: Arc<StencilTree>, target: &str, vocab: &TestVocab) -> String {
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

    const COMMANDS: &str = r#"[{"name":"run_commands","params":[
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
    /// object whose `}` comes next.
    #[test]
    fn a_misplaced_delimiter_after_a_value_is_dropped() {
        let v = TestVocab::new();
        let tree = tool_tree(
            r#"[{"name":"seek","params":[{"name":"at","type":"integer","required":true}]}]"#,
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
                // Up to the colon: the value's leading space is the model's.
                assert!(
                    text.ends_with("\"n\":"),
                    "run should reach the value: {text:?}"
                );
            }
            other => panic!("expected a single Prefill run, got {other:?}"),
        }
        // After the run, the next step is the integer value's free decode.
        assert!(matches!(driver.step(), StepMask::Free { .. }));
    }

    // ── Free-text exit healing (the merged exit-token edge cases) ───────────

    /// Build a driver positioned at the first field's free-text decode (after
    /// the envelope + name + key prefill).
    fn driver_at_first_value(catalog: &str, v: &TestVocab) -> StencilDriver {
        let tools = parse_tools(catalog).unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        let tree = Arc::new(compile(&spec, v).unwrap());
        let mut driver = StencilDriver::new(tree);
        assert!(matches!(driver.step(), StepMask::Prefill(_)));
        assert!(matches!(driver.step(), StepMask::Free { .. }));
        driver
    }

    const STR_OPT: &str = r#"[{"name":"write_file","params":[
        {"name":"path","type":"string","required":true},
        {"name":"create","type":"boolean","required":false}]}]"#;
    const STR_ONLY: &str = r#"[{"name":"read_file","params":[
        {"name":"path","type":"string","required":true}]}]"#;
    const INT_ONLY: &str = r#"[{"name":"wait","params":[
        {"name":"secs","type":"integer","required":true}]}]"#;

    #[test]
    fn tool_call_value_span_is_never_a_terminal_close_span() {
        // A tool-call value consumes its own close token (no suppress_close):
        // the sampler's closing-statement script must not play there.
        let v = TestVocab::new();
        let driver = driver_at_first_value(STR_ONLY, &v);
        assert!(!driver.in_terminal_close_span());
    }

    #[test]
    fn clean_string_close_does_not_heal() {
        // Closing quote is its own byte token — a clean boundary, no heal.
        let v = TestVocab::new();
        let mut d = driver_at_first_value(STR_OPT, &v);
        assert_eq!(d.accept(b'a' as TokenId, b"a"), Healed::No);
        assert_eq!(d.accept(b'"' as TokenId, b"\""), Healed::No);
    }

    #[test]
    fn string_exit_merged_with_comma() {
        // `",` — quote exits at byte 0, the comma is leftover.
        let v = TestVocab::new().with_special("\",", 300);
        let mut d = driver_at_first_value(STR_OPT, &v);
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
        assert_eq!(
            d.accept(300, b"h\","),
            Healed::Rewrite {
                bytes: b"h\"".to_vec()
            }
        );
    }

    #[test]
    fn escaped_quote_does_not_exit() {
        // An escaped quote mid-value must not be treated as the close.
        let v = TestVocab::new();
        let mut d = driver_at_first_value(STR_OPT, &v);
        assert_eq!(d.accept(b'\\' as TokenId, b"\\"), Healed::No);
        assert_eq!(d.accept(b'"' as TokenId, b"\""), Healed::No); // escaped — not a close
        assert_eq!(d.accept(b'b' as TokenId, b"b"), Healed::No);
        assert_eq!(d.accept(b'"' as TokenId, b"\""), Healed::No); // real close, clean
    }

    #[test]
    fn lookahead_value_merged_with_delimiter() {
        // Integer value: lookahead terminator.  `30}` is one token — the `30`
        // is the value (consumed=2), the `}` is the lookahead delimiter.
        let v = TestVocab::new().with_special("30}", 300);
        let mut d = driver_at_first_value(INT_ONLY, &v);
        assert_eq!(
            d.accept(300, b"30}"),
            Healed::Rewrite {
                bytes: b"30".to_vec()
            }
        );
    }

    #[test]
    fn lookahead_clean_delimiter_does_not_heal() {
        // The delimiter arrives as its own token: a clean lookahead (consumed=0),
        // handled by push-back — not a heal.
        let v = TestVocab::new();
        let mut d = driver_at_first_value(INT_ONLY, &v);
        assert_eq!(d.accept(b'3' as TokenId, b"3"), Healed::No);
        assert_eq!(d.accept(b'0' as TokenId, b"0"), Healed::No);
        assert_eq!(d.accept(b'}' as TokenId, b"}"), Healed::No); // whole token = delimiter
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
