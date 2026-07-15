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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Healed {
    /// The token was accepted as-is — commit it normally.
    No,
    /// A consumed-close free-text span ended *inside* this token: the trailing
    /// `bytes.len() - consumed` bytes are the next node's delimiter, emitted as
    /// part of this token.  The decode loop must commit only the re-tokenized
    /// first `consumed` bytes and drop the rest; the successor re-emits the
    /// delimiter.
    Exit { consumed: usize },
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
    /// Suppressed thinking-block close tokens dropped (each one is a steering
    /// continuation retry: the model emitted `</think>` and was re-steered back
    /// into the reasoning block with a continuation phrase).
    pub think_continuations: u32,
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
    /// Returns [`Healed::Exit`] when a consumed-close span ended strictly inside
    /// this token (the model merged the closing char with the next delimiter);
    /// the caller heals by committing only the valid prefix.  Returns
    /// [`Healed::Drop`] when a `suppress_close` token-closed span ended on this
    /// token (the close token is dropped and the successor prefills a steering
    /// continuation).
    pub fn accept(&mut self, token: TokenId, bytes: &[u8]) -> Healed {
        match self.session.observe(token, bytes) {
            Ok(Observe::SpanClosed { leftover }) if leftover > 0 && leftover < bytes.len() => {
                self.stats.heals += 1;
                Healed::Exit {
                    consumed: bytes.len() - leftover,
                }
            }
            // A suppressed token close: drop the token, count the continuation,
            // and let the successor prefill the steering phrase.
            Ok(Observe::TokenClosedDrop) => {
                self.stats.think_continuations += 1;
                Healed::Drop
            }
            // A kept token close (the real, final close): commit it normally.
            Ok(Observe::TokenClosedKeep) => Healed::No,
            Ok(Observe::Bailed) => {
                self.stats.bailed = true;
                Healed::No
            }
            _ => Healed::No,
        }
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
                assert!(
                    text.ends_with("\"n\": "),
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
        assert_eq!(d.accept(300, b"\","), Healed::Exit { consumed: 1 });
    }

    #[test]
    fn string_exit_merged_with_close_brace() {
        // No optional → the value is followed by the object close; the model
        // closes the string merged with the first `}` (`"}`).
        let v = TestVocab::new().with_special("\"}", 300);
        let mut d = driver_at_first_value(STR_ONLY, &v);
        assert_eq!(d.accept(b'a' as TokenId, b"a"), Healed::No);
        assert_eq!(d.accept(300, b"\"}"), Healed::Exit { consumed: 1 });
    }

    #[test]
    fn string_exit_with_value_content_in_token() {
        // The exit token also carries the last value byte: `h",` → consumed=2
        // (the `h` value byte + the closing quote), `,` leftover.
        let v = TestVocab::new().with_special("h\",", 300);
        let mut d = driver_at_first_value(STR_OPT, &v);
        assert_eq!(d.accept(300, b"h\","), Healed::Exit { consumed: 2 });
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
        assert_eq!(d.accept(300, b"30}"), Healed::Exit { consumed: 2 });
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
}
