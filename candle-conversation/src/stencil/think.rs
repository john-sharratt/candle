//! Front-end D — a thinking-block steering tree.
//!
//! Qwen3 emits a `<think>…</think>` reasoning block before its answer.  This
//! front-end builds a stencil tree, *triggered by the `<think>` token* (exactly
//! like the tool-call tree is triggered by `<tool_call>`), that steers the
//! contents of that block by effort dial 0..4 ([`ThinkMode`]).  The trigger token
//! has already been emitted when the tree is entered, so the tree's content
//! starts immediately *after* `<think>`; the block ends when the model emits the
//! `</think>` close token (or a per-span hard limit fires).
//!
//! The dial selects how the block is driven:
//!
//! - [`ThinkMode::Off`] — no tree (the `/no_think` glue already yields the empty
//!   block, so `<think>` is not registered as a trigger).
//! - [`ThinkMode::Quick`] — a short primed thought, the model closes it.
//! - [`ThinkMode::Balanced`] — a pure free flow, the model closes when ready.
//! - [`ThinkMode::Deep`] — two continuation retries (the model's `</think>` is
//!   suppressed and re-steered with `"But wait — "` then steered to a conclusion).
//! - [`ThinkMode::Exhaustive`] — four continuation retries, the last concluding.
//!
//! The retry mechanism is the new token-closed free span: a [`Terminator::Never`]
//! span whose `close_token` is `</think>`.  With `suppress_close`, the sampled
//! `</think>` is dropped and the next static run prefills a continuation phrase,
//! coaxing the model to keep reasoning; the final span keeps the close so the
//! block actually terminates.

use super::spec::{NodeSpec, SpecId, TreeSpec};
use super::terminator::Terminator;
use super::tree::FreeTextLimits;
use super::vocab::TokenId;

/// The reasoning-effort dial (0..4) selecting how the thinking block is steered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkMode {
    /// 0 — no steering tree (the empty block is produced by the `/no_think` glue).
    Off,
    /// 1 — a short primed thought; the model closes it quickly.
    Quick,
    /// 2 — a pure free flow; the model closes when ready.
    Balanced,
    /// 3 — two continuation retries, the last steering to a conclusion.
    Deep,
    /// 4 — four continuation retries, the last steering to a conclusion.
    Exhaustive,
}

impl ThinkMode {
    /// The in-`<think>` reflection-marker suppression penalty for this dial — the
    /// *ceiling* lever (inverse of the injected pivots).  Subtracted from the
    /// `Wait`/`Hmm`/`Alternatively`/`Actually` family's logits while in a think
    /// block, so the lower tiers can't self-inject "Wait" and spiral:
    ///   - `Quick` — HARD (effectively bans the family; reasoning stays short)
    ///   - `Balanced` — SOFT (discourages, but ~one spontaneous correction survives)
    ///   - `Deep`/`Exhaustive`/`Off` — none (reconsideration is wanted / no block)
    ///
    /// These are starting magnitudes worth A/B-ing on the real checkpoint.
    pub fn suppress_penalty(self) -> f32 {
        match self {
            ThinkMode::Quick => 100.0,
            ThinkMode::Balanced => 6.0,
            ThinkMode::Deep | ThinkMode::Exhaustive | ThinkMode::Off => 0.0,
        }
    }

    /// The per-span thinking-token budget for this dial, as the EOT close ramp's
    /// `(graceful_segment_close_after, force_segment_close_after)` thresholds.  `segment_len` restarts
    /// at every steered span boundary, so this is a budget *per span*: `graceful`
    /// closes the block at the next clause boundary once passed, `force` is the
    /// hard token cap that rewrites the next token to `</think>`.  Higher dials get
    /// more room per span — and, because they also have more spans, far more total
    /// (Quick: 1×80; Balanced: 1×240; Deep: 3×360; Exhaustive: 5×560).
    ///
    /// Each `force` sits below that mode's tree `forced_after` (Quick 96, Balanced
    /// /Deep 512, Exhaustive 768 — the stencil's hard backstop) so the EOT ramp
    /// closes a span gracefully before the stencil force-cuts it.  `Off` has no
    /// steered spans, so its budget bounds the whole block.  Worth A/B-ing on the
    /// real checkpoint.
    pub fn eot_budget(self) -> (i32, i32) {
        match self {
            ThinkMode::Off => (220, 300),
            ThinkMode::Quick => (48, 80),
            ThinkMode::Balanced => (160, 240),
            ThinkMode::Deep => (240, 360),
            ThinkMode::Exhaustive => (384, 560),
        }
    }

    /// The number of free-decode spans this dial's steering tree produces: one per
    /// continuation phrase, plus the initial span.  Derived from the same phrase
    /// arrays the tree is built from, so it can't drift from the compiled tree.
    pub fn span_count(self) -> i32 {
        match self {
            ThinkMode::Off => 0,
            ThinkMode::Quick | ThinkMode::Balanced => 1,
            ThinkMode::Deep => DEEP_PHRASES.len() as i32 + 1,
            ThinkMode::Exhaustive => EXHAUSTIVE_PHRASES.len() as i32 + 1,
        }
    }

    /// The EOS (turn-ender) budget for this dial, as `(eos_ramp_start, graceful_eos,
    /// forced_eos)` in *total* generated tokens.  This is the whole-turn backstop on
    /// `current_len`, distinct from the per-span EOT close: it must clear the
    /// thinking budget (or it truncates what the EOT ramp may spend) and then bound
    /// the answer.
    ///
    /// It is **derived from two dials**, not tabled:
    ///   - the thinking budget — `span_count ×` the per-span EOT force cap (what the
    ///     steering tree actually permits) — fixes where the answer begins, so the
    ///     EOS ramp *starts as the think block ends* (`eos_ramp_start = thinking`)
    ///     and stays dormant during reasoning (the per-span EOT/EOS boost handles
    ///     that);
    ///   - `response_tokens` (mapped from the composer's `response_length` dial) is
    ///     the room for the answer above that — the ramp covers it, graceful closes
    ///     ~4/5 of the way in, forced caps the turn.
    ///
    /// So both knobs move the cap automatically: a deeper think pushes the answer
    /// window later, a longer `response_length` widens it.
    pub fn eos_budget(self, response_tokens: i32) -> (i32, i32, i32) {
        let thinking = self.span_count() * self.eot_budget().1;
        let ramp_start = thinking;
        let graceful = thinking + response_tokens * 4 / 5;
        let forced = thinking + response_tokens;
        (ramp_start, graceful, forced)
    }
}

/// The resolved token ids the thinking-block tree is built against.  The
/// front-end passes already-resolved ids (the compiler copies `close_token`
/// verbatim — it is never re-tokenized in context).
#[derive(Debug, Clone, Copy)]
pub struct ThinkSteerEnvelope {
    /// `<think>` id — the trigger; the tree resumes AFTER it.
    pub think_open: TokenId,
    /// `</think>` id — the close token that ends each span.
    pub think_close: TokenId,
    /// The model's end-of-sequence id.
    pub eos: TokenId,
}

/// Build the steering tree spec for `mode`.  Returns `None` for
/// [`ThinkMode::Off`] (no tree: `<think>` is not registered as a trigger, so the
/// block decodes freely / the `/no_think` glue yields the empty block).
///
/// The tree's content starts after `<think>` (the trigger).  Every span ends on
/// EITHER `</think>` (`env.think_close`) OR EOS — both intercepted by normal
/// decode and dropped (the span suppresses its close) — or its hard
/// `forced_after` runaway guard.  Because the model's own `</think>` is always
/// dropped, the FINAL span is followed by an injected `Static("</think>")` that
/// closes the block.
pub fn compile_think_tree(mode: ThinkMode, env: &ThinkSteerEnvelope) -> Option<TreeSpec> {
    match mode {
        ThinkMode::Off => None,
        ThinkMode::Quick => Some(quick(env)),
        ThinkMode::Balanced => Some(balanced(env)),
        ThinkMode::Deep => Some(deep(env)),
        ThinkMode::Exhaustive => Some(exhaustive(env)),
    }
}

/// A token-closed thinking span: no byte terminator, never ends on `eos_ends`
/// (EOS is instead intercepted as a second close trigger by the session), always
/// suppresses its close (the model's `</think>`/EOS is dropped), and is capped at
/// `forced_after`.
fn think_span(
    spec: &mut TreeSpec,
    env: &ThinkSteerEnvelope,
    forced_after: u32,
    next: SpecId,
) -> SpecId {
    spec.push(NodeSpec::FreeText {
        term: Terminator::Never,
        eos_ends: false,
        limits: FreeTextLimits::think_flow(forced_after),
        close_token: Some(env.think_close),
        suppress_close: true,
        next,
    })
}

/// The injected closing tag the block actually ends on (the model's own
/// `</think>` is always dropped), spliced to `End`.
fn close_tag_then_end(spec: &mut TreeSpec) -> SpecId {
    let end = spec.push(NodeSpec::End);
    spec.push(NodeSpec::Static {
        text: "</think>".to_string(),
        next: end,
    })
}

/// Per-span hard token backstops (`forced_after`) for each dial's free-text spans.
/// The EOT close ramp ([`ThinkMode::eot_budget`]) is tuned to fire below these, so
/// a span closes gracefully on a clause boundary before the stencil force-cuts it;
/// these are the last-resort caps that apply only if the ramp never fires.
const QUICK_SPAN_CAP: u32 = 96;
const BALANCED_SPAN_CAP: u32 = 512;
const DEEP_SPAN_CAP: u32 = 512;
const EXHAUSTIVE_SPAN_CAP: u32 = 768;

/// `Deep`'s continuation phrases (reconsider, then settle) — each prefilled after a
/// dropped close to re-steer the next span.  The dial therefore has `len() + 1`
/// free-decode spans; this array is the single source for both the tree and
/// [`ThinkMode::span_count`].
const DEEP_PHRASES: &[&str] = &["\n\nBut wait — ", "\n\nSo, where I land: "];

/// `Exhaustive`'s continuation phrases in order (reconsider, re-angle, check,
/// settle).  Single source for the tree and [`ThinkMode::span_count`].
const EXHAUSTIVE_PHRASES: &[&str] = &[
    "\n\nBut wait — ",
    "\n\nAlternatively — ",
    "\n\nWait, let me check that — ",
    "\n\nPutting it all together: ",
];

/// Quick: a short opener static, one tight span, then the injected closing tag.
///
/// `Static("\nLet me focus on what matters most. ")` → span(`QUICK_SPAN_CAP`) →
/// `Static("</think>")` → `End`.
fn quick(env: &ThinkSteerEnvelope) -> TreeSpec {
    let mut spec = TreeSpec::new("think_quick");
    let close = close_tag_then_end(&mut spec);
    let span = think_span(&mut spec, env, QUICK_SPAN_CAP, close);
    let opener = spec.push(NodeSpec::Static {
        text: "\nLet me focus on what matters most. ".to_string(),
        next: span,
    });
    spec.root = opener;
    spec
}

/// Balanced: a short opener static, one span, then the injected closing tag.
///
/// `Static("\nLet me think about this. ")` → span(`BALANCED_SPAN_CAP`) →
/// `Static("</think>")` → `End`.
fn balanced(env: &ThinkSteerEnvelope) -> TreeSpec {
    let mut spec = TreeSpec::new("think_balanced");
    let close = close_tag_then_end(&mut spec);
    let span = think_span(&mut spec, env, BALANCED_SPAN_CAP, close);
    let opener = spec.push(NodeSpec::Static {
        text: "\nLet me think about this. ".to_string(),
        next: span,
    });
    spec.root = opener;
    spec
}

/// Build `opener` → a chain of `phrases.len() + 1` spans separated by continuation
/// phrase statics, terminated by the injected `Static("</think>")` → `End`.  The
/// root is the `opener` static (which carries its own leading `"\n"`), then the
/// first span; `phrases[i]` (each carrying its own leading `"\n\n"`) is prefilled
/// after the i-th span's dropped close, re-steering the block.
fn continuation_chain(
    env: &ThinkSteerEnvelope,
    label: &str,
    opener: &str,
    phrases: &[&str],
    forced_after: u32,
) -> TreeSpec {
    let mut spec = TreeSpec::new(label);
    let close = close_tag_then_end(&mut spec);

    // The final span's successor is the injected closing tag.
    let mut next = think_span(&mut spec, env, forced_after, close);

    // Build backwards: for each phrase (last → first), a static that prefills the
    // phrase and feeds the following span, preceded by another span.
    for phrase in phrases.iter().rev() {
        let phrase_static = spec.push(NodeSpec::Static {
            text: (*phrase).to_string(),
            next,
        });
        next = think_span(&mut spec, env, forced_after, phrase_static);
    }

    // `next` is the first span; the opener static prefills ahead of it.
    spec.root = spec.push(NodeSpec::Static {
        text: opener.to_string(),
        next,
    });
    spec
}

/// Deep: an opener, then 2 continuation phrases (reconsider, then settle), each on
/// its own line.
fn deep(env: &ThinkSteerEnvelope) -> TreeSpec {
    continuation_chain(
        env,
        "think_deep",
        "\nLet me work this out. ",
        DEEP_PHRASES,
        DEEP_SPAN_CAP,
    )
}

/// Exhaustive: an opener, then 4 continuation phrases in order (reconsider,
/// re-angle, check, settle), each on its own line.
fn exhaustive(env: &ThinkSteerEnvelope) -> TreeSpec {
    continuation_chain(
        env,
        "think_exhaustive",
        "\nLet me really work through this. ",
        EXHAUSTIVE_PHRASES,
        EXHAUSTIVE_SPAN_CAP,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::stencil::compile::compile;
    use crate::stencil::driver::{Healed, StencilDriver, StepMask};
    use crate::stencil::tree::StencilTree;
    use crate::stencil::vocab::{TestVocab, Vocab};

    const THINK_OPEN_ID: TokenId = 151667;
    const THINK_CLOSE_ID: TokenId = 151668;

    fn vocab() -> TestVocab {
        TestVocab::new()
            .with_special("<think>", THINK_OPEN_ID)
            .with_special("</think>", THINK_CLOSE_ID)
    }

    fn env() -> ThinkSteerEnvelope {
        ThinkSteerEnvelope {
            think_open: THINK_OPEN_ID,
            think_close: THINK_CLOSE_ID,
            eos: vocab().eos(),
        }
    }

    /// Compile a mode's tree against the test vocab, asserting it builds cleanly.
    fn tree_for(mode: ThinkMode) -> Arc<StencilTree> {
        let spec = compile_think_tree(mode, &env()).expect("mode must produce a tree");
        Arc::new(compile(&spec, &vocab()).expect("think tree must compile cleanly"))
    }

    /// Step until the next decode point, prefilling static runs and recording
    /// their decoded text.  Returns the `StepMask` (`Free`/`Branch`/`Done`) and
    /// the concatenated text of any prefills consumed on the way.
    fn step_to_decode(driver: &mut StencilDriver, v: &TestVocab) -> (StepMask, String) {
        let mut text = String::new();
        loop {
            match driver.step() {
                StepMask::Prefill(run) => {
                    text.push_str(&String::from_utf8(v.decode(&run)).unwrap());
                }
                other => return (other, text),
            }
        }
    }

    /// Free-decode `n` ordinary filler tokens (an `'x'` the span never closes on),
    /// each its own token, asserting the span stays open (`StepMask::Free`,
    /// `Healed::No`).  Decode is normal — nothing is banned.
    fn free_decode(driver: &mut StencilDriver, n: u32) {
        for _ in 0..n {
            match driver.step() {
                StepMask::Free { .. } => {
                    assert_eq!(driver.accept(b'x' as TokenId, b"x"), Healed::No);
                }
                other => panic!("expected Free during free decode, got {other:?}"),
            }
        }
    }

    // ── One test per mode ────────────────────────────────────────────────────

    #[test]
    fn off_registers_no_tree() {
        assert!(compile_think_tree(ThinkMode::Off, &env()).is_none());
    }

    #[test]
    fn quick_primes_then_closes() {
        let v = vocab();
        let mut d = StencilDriver::new(tree_for(ThinkMode::Quick));

        // First step prefills the opener, then arrives at the free span.
        let (mask, primed) = step_to_decode(&mut d, &v);
        assert_eq!(primed, "\nLet me focus on what matters most. ");
        assert!(matches!(mask, StepMask::Free { .. }));

        // A few free tokens, then the close → DROP (suppressed): the injected
        // closing tag prefills in its place.
        free_decode(&mut d, 3);
        assert_eq!(
            d.accept(THINK_CLOSE_ID, &v.token_bytes(THINK_CLOSE_ID)),
            Healed::Drop
        );

        // The injected `</think>` static prefills, then End.
        let (mask, closed) = step_to_decode(&mut d, &v);
        assert_eq!(closed, "</think>");
        assert!(matches!(mask, StepMask::Done));
        assert!(d.is_done());
        assert_eq!(d.stats().think_continuations, 1);
    }

    #[test]
    fn balanced_free_flows_until_close() {
        let v = vocab();
        let mut d = StencilDriver::new(tree_for(ThinkMode::Balanced));

        // The opener prefills, then the free span.
        let (mask, primed) = step_to_decode(&mut d, &v);
        assert_eq!(primed, "\nLet me think about this. ");
        assert!(matches!(mask, StepMask::Free { .. }));

        // A few free tokens, then close → DROP (suppressed).
        free_decode(&mut d, 5);
        assert_eq!(
            d.accept(THINK_CLOSE_ID, &v.token_bytes(THINK_CLOSE_ID)),
            Healed::Drop
        );

        // The injected closing tag prefills, then done.
        let (mask, closed) = step_to_decode(&mut d, &v);
        assert_eq!(closed, "</think>");
        assert!(matches!(mask, StepMask::Done));
        assert!(d.is_done());
        assert_eq!(d.stats().think_continuations, 1);
    }

    /// `balanced_closes_on_eos_too`: an EOS sample closes the span exactly like
    /// `</think>` — intercepted by normal decode → `Healed::Drop`, then the
    /// injected tag prefills.
    #[test]
    fn balanced_closes_on_eos_too() {
        let v = vocab();
        let mut d = StencilDriver::new(tree_for(ThinkMode::Balanced));

        let (mask, _) = step_to_decode(&mut d, &v);
        assert!(matches!(mask, StepMask::Free { .. }));
        free_decode(&mut d, 5);

        // Feed EOS (not </think>): the token-closed span closes on it too → DROP.
        let eos = v.eos();
        assert_eq!(d.accept(eos, &v.token_bytes(eos)), Healed::Drop);

        let (mask, closed) = step_to_decode(&mut d, &v);
        assert_eq!(closed, "</think>");
        assert!(matches!(mask, StepMask::Done));
        assert!(d.is_done());
        assert_eq!(d.stats().think_continuations, 1);
    }

    #[test]
    fn deep_intercepts_two_closes() {
        let v = vocab();
        let mut d = StencilDriver::new(tree_for(ThinkMode::Deep));

        // The opener prefills, then the first span.
        let (mask, primed) = step_to_decode(&mut d, &v);
        assert_eq!(primed, "\nLet me work this out. ");
        assert!(matches!(mask, StepMask::Free { .. }));

        // span 1: a few tokens, then close → DROP; next prefill "\n\nBut wait — ".
        free_decode(&mut d, 4);
        assert_eq!(
            d.accept(THINK_CLOSE_ID, &v.token_bytes(THINK_CLOSE_ID)),
            Healed::Drop
        );
        let (mask, phrase) = step_to_decode(&mut d, &v);
        assert_eq!(phrase, "\n\nBut wait — ");
        assert!(matches!(mask, StepMask::Free { .. }));

        // span 2: close → DROP; next prefill "\n\nSo, where I land: ".
        free_decode(&mut d, 4);
        assert_eq!(
            d.accept(THINK_CLOSE_ID, &v.token_bytes(THINK_CLOSE_ID)),
            Healed::Drop
        );
        let (mask, phrase) = step_to_decode(&mut d, &v);
        assert_eq!(phrase, "\n\nSo, where I land: ");
        assert!(matches!(mask, StepMask::Free { .. }));

        // span 3 (final): close → DROP; the injected closing tag prefills.
        free_decode(&mut d, 4);
        assert_eq!(
            d.accept(THINK_CLOSE_ID, &v.token_bytes(THINK_CLOSE_ID)),
            Healed::Drop
        );
        let (mask, closed) = step_to_decode(&mut d, &v);
        assert_eq!(closed, "</think>");
        assert!(matches!(mask, StepMask::Done));
        assert!(d.is_done());
        assert_eq!(d.stats().think_continuations, 3);
    }

    /// `deep_phrases_have_leading_newlines`: the prefilled continuation runs decode
    /// to `"\n\nBut wait — "` then `"\n\nSo, where I land: "`, and the final prefill
    /// is `"</think>"`.
    #[test]
    fn deep_phrases_have_leading_newlines() {
        let v = vocab();
        let mut d = StencilDriver::new(tree_for(ThinkMode::Deep));

        let mut prefills: Vec<String> = Vec::new();
        // The opener prefills first; then decode each span a little, close it, and
        // collect every prefilled run between spans.
        let (mask, first) = step_to_decode(&mut d, &v);
        assert_eq!(first, "\nLet me work this out. ");
        assert!(matches!(mask, StepMask::Free { .. }));
        loop {
            free_decode(&mut d, 3);
            d.accept(THINK_CLOSE_ID, &v.token_bytes(THINK_CLOSE_ID));
            let (mask, text) = step_to_decode(&mut d, &v);
            prefills.push(text);
            match mask {
                StepMask::Done => break,
                StepMask::Free { .. } => {}
                other => panic!("unexpected mask {other:?}"),
            }
        }
        assert_eq!(
            prefills,
            vec![
                "\n\nBut wait — ".to_string(),
                "\n\nSo, where I land: ".to_string(),
                "</think>".to_string(),
            ]
        );
    }

    /// `deep_intercepts_close_and_eos`: close one span with `</think>` and the next
    /// with EOS — both `Healed::Drop`, both produce the next phrase;
    /// `think_continuations == 2` after two intercepts.
    #[test]
    fn deep_intercepts_close_and_eos() {
        let v = vocab();
        let mut d = StencilDriver::new(tree_for(ThinkMode::Deep));

        let (mask, _) = step_to_decode(&mut d, &v);
        assert!(matches!(mask, StepMask::Free { .. }));

        // span 1 closed by </think>.
        free_decode(&mut d, 4);
        assert_eq!(
            d.accept(THINK_CLOSE_ID, &v.token_bytes(THINK_CLOSE_ID)),
            Healed::Drop
        );
        let (mask, phrase1) = step_to_decode(&mut d, &v);
        assert_eq!(phrase1, "\n\nBut wait — ");
        assert!(matches!(mask, StepMask::Free { .. }));

        // span 2 closed by EOS.
        free_decode(&mut d, 4);
        let eos = v.eos();
        assert_eq!(d.accept(eos, &v.token_bytes(eos)), Healed::Drop);
        let (_, phrase2) = step_to_decode(&mut d, &v);
        assert_eq!(phrase2, "\n\nSo, where I land: ");

        assert_eq!(d.stats().think_continuations, 2);
    }

    #[test]
    fn exhaustive_intercepts_four_closes() {
        let v = vocab();
        let mut d = StencilDriver::new(tree_for(ThinkMode::Exhaustive));

        let phrases = [
            "\n\nBut wait — ",
            "\n\nAlternatively — ",
            "\n\nWait, let me check that — ",
            "\n\nPutting it all together: ",
        ];

        // The opener prefills, then the first span.
        let (mask, primed) = step_to_decode(&mut d, &v);
        assert_eq!(primed, "\nLet me really work through this. ");
        assert!(matches!(mask, StepMask::Free { .. }));

        for phrase in phrases {
            // A few tokens, suppressed close → DROP, then the phrase.
            free_decode(&mut d, 4);
            assert_eq!(
                d.accept(THINK_CLOSE_ID, &v.token_bytes(THINK_CLOSE_ID)),
                Healed::Drop
            );
            let (mask, got) = step_to_decode(&mut d, &v);
            assert_eq!(got, phrase);
            assert!(matches!(mask, StepMask::Free { .. }));
        }

        // Final span: close → DROP, the injected closing tag prefills.
        free_decode(&mut d, 4);
        assert_eq!(
            d.accept(THINK_CLOSE_ID, &v.token_bytes(THINK_CLOSE_ID)),
            Healed::Drop
        );
        let (mask, closed) = step_to_decode(&mut d, &v);
        assert_eq!(closed, "</think>");
        assert!(matches!(mask, StepMask::Done));
        assert!(d.is_done());
        assert_eq!(d.stats().think_continuations, 5);
    }

    // ── Invariant: every produced tree compiles cleanly and reaches End ──────

    #[test]
    fn all_modes_compile_clean() {
        for mode in [
            ThinkMode::Quick,
            ThinkMode::Balanced,
            ThinkMode::Deep,
            ThinkMode::Exhaustive,
        ] {
            let spec = compile_think_tree(mode, &env()).unwrap();
            // Compiling enforces: forced_after > 0 on every FreeText, every path
            // reaches End, no adjacent statics, acyclic.
            let tree = compile(&spec, &vocab())
                .unwrap_or_else(|e| panic!("{mode:?} failed to compile: {e}"));
            assert!(tree.len() >= 2, "{mode:?} tree too small");
        }
    }

    #[test]
    fn eot_budget_scales_and_stays_under_span_caps() {
        use ThinkMode::*;
        // graceful < force within each steered dial, both positive.
        for m in [Quick, Balanced, Deep, Exhaustive] {
            let (g, f) = m.eot_budget();
            assert!(
                g > 0 && g < f,
                "{m:?}: expected 0 < graceful({g}) < force({f})"
            );
        }
        // The force budget grows with the dial — exhaustive thinks longest per span.
        let forces: Vec<i32> = [Quick, Balanced, Deep, Exhaustive]
            .iter()
            .map(|m| m.eot_budget().1)
            .collect();
        assert!(
            forces.windows(2).all(|w| w[0] < w[1]),
            "force budget must increase with the dial: {forces:?}"
        );
        // Each force sits below that mode's tree span cap (the hard backstop), so the
        // EOT ramp closes a span gracefully before the stencil force-cuts it.
        assert!(Quick.eot_budget().1 < QUICK_SPAN_CAP as i32);
        assert!(Balanced.eot_budget().1 < BALANCED_SPAN_CAP as i32);
        assert!(Deep.eot_budget().1 < DEEP_SPAN_CAP as i32);
        assert!(Exhaustive.eot_budget().1 < EXHAUSTIVE_SPAN_CAP as i32);
    }

    #[test]
    fn eos_budget_is_derived_from_the_tree_and_clears_thinking() {
        use ThinkMode::*;
        // span_count is the tree's phrase arrays, not a separate table.
        assert_eq!(Quick.span_count(), 1);
        assert_eq!(Balanced.span_count(), 1);
        assert_eq!(Deep.span_count(), DEEP_PHRASES.len() as i32 + 1);
        assert_eq!(Exhaustive.span_count(), EXHAUSTIVE_PHRASES.len() as i32 + 1);

        let response = 1024; // a sample response_length budget
        for m in [Quick, Balanced, Deep, Exhaustive] {
            let (ramp, graceful, forced) = m.eos_budget(response);
            assert!(
                ramp < graceful && graceful < forced,
                "{m:?}: ramp<graceful<forced"
            );
            let thinking = m.span_count() * m.eot_budget().1;
            // The EOS ramp begins exactly as the think budget ends, and stays dormant
            // during reasoning (the per-span EOT/EOS boost handles that).
            assert_eq!(ramp, thinking, "{m:?}: EOS ramp must start at thinking-end");
            // The hard cap = think budget + response budget — never below thinking,
            // or EOS would truncate what the EOT ramp may spend.
            assert_eq!(
                forced,
                thinking + response,
                "{m:?}: forced = thinking + response"
            );
            assert!(
                forced > thinking,
                "{m:?}: EOS must clear the thinking budget"
            );
        }

        // With a fixed response budget, the EOS cap grows with the think dial —
        // exhaustive gets the largest turn.
        let forced: Vec<i32> = [Quick, Balanced, Deep, Exhaustive]
            .iter()
            .map(|m| m.eos_budget(response).2)
            .collect();
        assert!(
            forced.windows(2).all(|w| w[0] < w[1]),
            "EOS budget must increase with the dial: {forced:?}"
        );
    }
}
