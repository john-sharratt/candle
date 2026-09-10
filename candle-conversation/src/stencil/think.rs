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
//! **The tree supplies structure, never content.**  Every dial has the same
//! shape — `Static("\n")` → one free span → the injected `Static("</think>")` —
//! and the dials differ only in how much room that span gets.  The opening
//! static is a bare newline, so the block the model sees is exactly
//! `<think>\n` and the first *word* of the thought is always the model's own.
//!
//! That constraint is load-bearing, not stylistic.  Words prefilled into the
//! block do not read to the model as its own private scratch: they read as
//! something already said, and the model tries to interpret them.  Measured on
//! the daemon's greeting prompt at temp 0.75 over 20 seeds, priming the block
//! with `"Okay, "` gives 10/20 usable answers, 7/20 empty, ~5/20 confabulating
//! a question that was never asked — one sample quoted the primed opener back
//! as the user's own words — and replaces the Zen persona with the base
//! model's.  Priming only the newline gives 20/20 usable, 0 empty, 0 persona
//! breaks, and a correct one-sentence thought every time.  Both arms live in
//! `candle-transformers/tests/flat_prefill_probe.rs`; the structural guard that
//! keeps words out of the tree is `no_tree_injects_words_into_the_block`.
//!
//! The dial therefore buys thinking *room*, via [`ThinkMode::eot_budget`]:
//!
//! - [`ThinkMode::Off`] — the block is closed the instant it opens: the tree
//!   prefills `"\n\n</think>\n\n"` and ends, with no free span at all, so the
//!   rendered block is the empty `<think>\n\n</think>\n\n` the chat template
//!   produces for `enable_thinking: false`.
//! - [`ThinkMode::Quick`] — a tight budget; the model closes quickly.
//! - [`ThinkMode::Balanced`] — a little more room.
//! - [`ThinkMode::Deep`] — a long block.
//! - [`ThinkMode::Exhaustive`] — the longest block.
//!
//! The span is a token-closed free span: a [`Terminator::Never`] span whose
//! `close_token` is `</think>`.  It still runs with `suppress_close`, so the
//! model's own `</think>` is dropped and the block is closed by the tree's
//! injected tag — the block always terminates on the stencil's terms, and the
//! close is never duplicated.

use super::spec::{NodeSpec, SpecId, TreeSpec};
use super::terminator::Terminator;
use super::tree::FreeTextLimits;
use super::vocab::TokenId;

/// The reasoning-effort dial (0..4) selecting how the thinking block is steered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkMode {
    /// 0 — no steering tree (the empty block is produced by the `/no_think` glue).
    Off,
    /// 1 — a tight thinking budget; the model closes quickly.
    Quick,
    /// 2 — a free flow with a little more room.
    Balanced,
    /// 3 — a long block: room for the model to reconsider on its own.
    Deep,
    /// 4 — the longest block.
    Exhaustive,
}

impl ThinkMode {
    /// The in-`<think>` reflection-marker suppression penalty for this dial — the
    /// *ceiling* lever.  Subtracted from the `Wait`/`Hmm`/`Alternatively`/
    /// `Actually` family's logits while in a think block, so the lower tiers
    /// can't pivot into "Wait" and spiral:
    ///   - `Quick` — HARD (effectively bans the family; reasoning stays short)
    ///   - `Balanced` — SOFT (discourages, but ~one spontaneous correction survives)
    ///   - `Deep`/`Exhaustive`/`Off` — none (reconsideration is wanted / no block)
    ///
    /// On the upper dials this is the *whole* reconsideration mechanism: nothing
    /// injects a pivot phrase, so a "But wait" only appears where the model
    /// chose it. Leaving the family unpenalised, with the room `eot_budget`
    /// grants, is what makes `Deep` deep.
    ///
    /// These are starting magnitudes worth A/B-ing on the real checkpoint.
    pub fn suppress_penalty(self) -> f32 {
        match self {
            ThinkMode::Quick => 100.0,
            ThinkMode::Balanced => 6.0,
            ThinkMode::Deep | ThinkMode::Exhaustive | ThinkMode::Off => 0.0,
        }
    }

    /// The thinking-token budget for this dial, as the EOT close ramp's
    /// `(graceful_segment_close_after, force_segment_close_after)` thresholds.
    /// Every dial has exactly one span, so `segment_len` runs the length of the
    /// block and this is the whole block's budget: `graceful` closes it at the
    /// next clause boundary once passed, `force` is the hard token cap that
    /// rewrites the next token to `</think>`.
    ///
    /// **The budget is the dial.**  Nothing injects a continuation phrase, so
    /// room is the only thing separating `Deep` from `Balanced`.  The upper
    /// rungs are spent as one continuous thought — the model reconsiders inside
    /// its own block, with the reflection-marker family left unpenalised
    /// ([`Self::suppress_penalty`]) so it can.
    ///
    /// **A budget is a backstop, not a routine terminator.**  Each one sits
    /// well above what that rung's workload actually costs, so ordinary turns
    /// end because the model finished and the cutoff only catches a block that
    /// has gone wild.  Measured with the real tokenizer over live turns:
    ///
    /// | workload | think tokens |
    /// |---|---|
    /// | greeting / thanks / one-line fact | 33–58 |
    /// | arithmetic or a unit conversion | 59–196 |
    /// | ordinary engineering question | 155–348 |
    /// | hard design question, uncapped | 720–1155 |
    ///
    /// **`graceful` is the number that matters** — it is where a block actually
    /// lands, because the close fires at the next clause boundary once passed
    /// (measured: `graceful + ~10` tokens).  The ladder is therefore the
    /// thinking length each rung buys: **512 → 1024 → 2048 → 4096**.
    ///
    /// `Quick` 512 is an order of magnitude above a trivial turn's 33–58, so
    /// nothing ordinary is ever touched.  `Balanced` 1024 is the default path:
    /// it clears the 155–348 band by 3×, and clears a hard question's natural
    /// 720–1155 too, so a legitimate long thought completes rather than being
    /// cut — while an unbounded enumeration still gets closed.  `Deep` 2048 and
    /// `Exhaustive` 4096 bracket the 1024–4096 range where extra thinking is
    /// documented to pay, and 4096 is the checkpoint's own `thinking_budget`
    /// default (4000) to the nearest power of two.
    ///
    /// The two derived figures are held at fixed ratios so the three can never
    /// drift apart: `force = 1.5 × graceful` gives the clause-boundary close a
    /// real window to find a boundary in, and the span cap is `2 × force`
    /// (= `3 × graceful`), far enough above that the stencil's runaway guard
    /// stays a last resort rather than the routine terminator it once was.
    /// `Off` never reaches a decode point inside the block, so its numbers are
    /// inert.
    pub fn eot_budget(self) -> (i32, i32) {
        match self {
            ThinkMode::Off => (512, 768),
            ThinkMode::Quick => (512, 768),
            ThinkMode::Balanced => (1024, 1536),
            ThinkMode::Deep => (2048, 3072),
            ThinkMode::Exhaustive => (4096, 6144),
        }
    }

    /// This dial's hard runaway backstop — the `forced_after` its steered span
    /// is compiled with, and the figure [`Self::eos_budget`] reserves for
    /// thinking.  One accessor so the tree and the turn budget can never name
    /// different numbers.  `Off` has no span; its cap is 0, which is what keeps
    /// its thinking reservation honestly zero.
    pub fn span_cap(self) -> i32 {
        match self {
            ThinkMode::Off => 0,
            ThinkMode::Quick => QUICK_SPAN_CAP as i32,
            ThinkMode::Balanced => BALANCED_SPAN_CAP as i32,
            ThinkMode::Deep => DEEP_SPAN_CAP as i32,
            ThinkMode::Exhaustive => EXHAUSTIVE_SPAN_CAP as i32,
        }
    }

    /// The number of free-decode spans this dial's steering tree produces.  Every
    /// steered dial is a single span — depth lives in [`Self::eot_budget`], not
    /// in a chain of re-steered fragments — so this is 1 for every mode that
    /// builds a tree and 0 for `Off`.
    pub fn span_count(self) -> i32 {
        match self {
            ThinkMode::Off => 0,
            ThinkMode::Quick
            | ThinkMode::Balanced
            | ThinkMode::Deep
            | ThinkMode::Exhaustive => 1,
        }
    }

    /// The EOS (turn-ender) budget for this dial, as `(eos_ramp_start, graceful_eos,
    /// forced_eos)` in *total* generated tokens.  This is the whole-turn backstop on
    /// `current_len`, distinct from the per-span EOT close: it must clear the
    /// thinking budget (or it truncates what the EOT ramp may spend) and then bound
    /// the answer.
    ///
    /// It is **derived from two dials**, not tabled:
    ///   - the thinking reservation — `span_count ×` the EOT force cutoff, the
    ///     point at which the sampler rewrites the next token to `</think>` —
    ///     fixes where the answer begins, so the EOS ramp *starts as the think
    ///     block ends* (`eos_ramp_start = thinking`) and stays dormant during
    ///     reasoning (the per-span EOT/EOS boost handles that);
    ///   - `response_tokens` (mapped from the composer's `response_length` dial) is
    ///     the room for the answer above that — the ramp covers it, graceful closes
    ///     ~4/5 of the way in, forced caps the turn.
    ///
    /// So both knobs move the cap automatically: a deeper think pushes the answer
    /// window later, a longer `response_length` widens it.
    ///
    /// **The reservation is the EOT force cutoff, not the span cap.** The
    /// cutoff is a true upper bound on thinking: at `segment_len >= force` the
    /// sampler rewrites the next token to `</think>`, the steering tree drops it
    /// and plays its own closing tag, and the block is over. So reserving it can
    /// never under-reserve, while reserving the span cap (3× larger) would leave
    /// `response_length` unable to bound a runaway answer — at `Exhaustive` the
    /// answer window would exceed a `terse` request by more than an order of
    /// magnitude.
    ///
    /// This distinction is only safe because the cutoff actually fires. It did
    /// not for a long time — the dialogue sampling config was snapshotted before
    /// the `<think>` ids were resolved, so every tier of the close budget was
    /// unreachable and blocks ran to the span cap instead, overspending this
    /// reservation and taking the difference out of the answer. That is fixed at
    /// the source (the config is now taken after the engine build, and an
    /// unresolved id is a hard startup failure), which is what lets the tighter,
    /// more useful figure be correct here.
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

/// Build the steering tree spec for `mode`.  **Every** mode yields a tree,
/// `Off` included — suppression is structural here, not advisory.
///
/// The tree's content starts after `<think>` (the trigger).  Every span ends on
/// EITHER `</think>` (`env.think_close`) OR EOS — both intercepted by normal
/// decode and dropped (the span suppresses its close) — or its hard
/// `forced_after` runaway guard.  Because the model's own `</think>` is always
/// dropped, the FINAL span is followed by an injected `Static("</think>")` that
/// closes the block.  `Off` is the degenerate case of that shape: no span at
/// all, so the closing run follows the trigger immediately.
pub fn compile_think_tree(mode: ThinkMode, env: &ThinkSteerEnvelope) -> TreeSpec {
    match mode {
        ThinkMode::Off => off(),
        ThinkMode::Quick => quick(env),
        ThinkMode::Balanced => balanced(env),
        ThinkMode::Deep => deep(env),
        ThinkMode::Exhaustive => exhaustive(env),
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

/// Hard token backstops (`forced_after`) for each dial's free-text span — twice
/// that mode's EOT force cutoff ([`ThinkMode::eot_budget`]), so the ramp always
/// closes the span on a clause boundary long before these fire.  They are the
/// last resort, reached only if the ramp cannot run at all.
///
/// The 2× margin is deliberate and was bought the hard way: these caps once sat
/// at 512 for both `Quick` and `Balanced`, close enough to ordinary reasoning
/// that they became the *routine* terminator rather than a backstop — nine
/// measured turns ended at exactly 513 tokens, cut mid-word, because the EOT
/// ramp above them was inert and nothing else stopped the block.
const QUICK_SPAN_CAP: u32 = 1536;
const BALANCED_SPAN_CAP: u32 = 3072;
const DEEP_SPAN_CAP: u32 = 6144;
const EXHAUSTIVE_SPAN_CAP: u32 = 12288;

/// The tree's opening static: the newline after `<think>`, and nothing else.
///
/// The trigger token `<think>` has already been emitted when the tree is entered,
/// so this is the whole of what the stencil writes into the block before handing
/// over — the model sees `<think>\n` and picks its own first word.  Priming any
/// *words* here corrupts the turn; the module docs carry the measurement, and
/// `no_tree_injects_words_into_the_block` enforces it.
const BLOCK_OPEN: &str = "\n";

/// The shape every steered dial has: `Static("\n")` → one span capped at
/// `forced_after` → the injected `Static("</think>")` → `End`.  `label` names the
/// compiled tree and `forced_after` is the dial's runaway backstop; nothing else
/// differs between the dials at the tree level (their real separation is the EOT
/// budget the session programs from [`ThinkMode::eot_budget`]).
fn single_span_tree(env: &ThinkSteerEnvelope, label: &str, forced_after: u32) -> TreeSpec {
    let mut spec = TreeSpec::new(label);
    let close = close_tag_then_end(&mut spec);
    let span = think_span(&mut spec, env, forced_after, close);
    spec.root = spec.push(NodeSpec::Static {
        text: BLOCK_OPEN.to_string(),
        next: span,
    });
    spec
}

/// Off: close the block on the token after `<think>`, giving the model nowhere
/// to reason.
///
/// `Static("\n\n</think>")` → `End`, so the rendered block is the empty
/// `<think>\n\n</think>` the chat template emits for `enable_thinking: false`
/// (the trigger supplied the opening tag).
///
/// **The run must END on `</think>`, with no trailing text.** A static run's
/// last token is deliberately held back from the forward and rides the next
/// decode step, which commits it through `push_committed` and arms the index
/// page cut. A marker in any earlier slot is `push_forwarded` instead, whose
/// cut flag is dropped, and `run_prefill`'s `reasoning_split` cannot make the
/// cut either — it declines to split when the break token is last in the pass,
/// which it then is. The block would still record its `think_close_at`, so the
/// turn seals with a reasoning span that is not a union of whole pages, and
/// every LATER turn's projection fails
/// `Substrate::turn_sealed_without_thinking` — the conversation answers with
/// that error from then on. Measured: a five-turn `Off` conversation poisoned
/// itself from turn three (`reasoning span [21..24)` against boundaries
/// `[0, 21, 22, 57, 58]`). The trailing separator the template shows is the
/// model's to emit, exactly as it is for every other dial, whose closing static
/// is likewise bare `</think>`.
///
/// **This must be a tree, not an instruction.** Suppression used to ride on the
/// dialect's `/no_think` marker, and the Qwen3.5/3.8 family has none — its
/// template suppresses by rendering the closed block instead. With no tree
/// bound, `Off` was left with nothing but a system-prompt line asking the model
/// not to deliberate, which it ignored: every sampled `Off` turn opened a block
/// and reasoned in it, and because `Off` budgets zero thinking tokens
/// ([`ThinkMode::span_count`]) the turn was then cut off mid-answer. Binding a
/// tree closes the block whatever the model intends, and makes that zero true.
///
/// It carries no [`NodeSpec::FreeText`], which is what keeps `span_count` at 0.
fn off() -> TreeSpec {
    let mut spec = TreeSpec::new("think_off");
    let end = spec.push(NodeSpec::End);
    spec.root = spec.push(NodeSpec::Static {
        text: "\n\n</think>".to_string(),
        next: end,
    });
    spec
}

/// Quick: the block newline, one tight span, then the injected closing tag.
fn quick(env: &ThinkSteerEnvelope) -> TreeSpec {
    single_span_tree(env, "think_quick", ThinkMode::Quick.span_cap() as u32)
}

/// Balanced: the block newline, one span, then the injected closing tag.
fn balanced(env: &ThinkSteerEnvelope) -> TreeSpec {
    single_span_tree(env, "think_balanced", ThinkMode::Balanced.span_cap() as u32)
}

/// Deep: the same shape with room for a long thought — enough that the model can
/// reconsider within the one span rather than being re-steered into it.
fn deep(env: &ThinkSteerEnvelope) -> TreeSpec {
    single_span_tree(env, "think_deep", ThinkMode::Deep.span_cap() as u32)
}

/// Exhaustive: the same shape at the widest budget.
fn exhaustive(env: &ThinkSteerEnvelope) -> TreeSpec {
    single_span_tree(
        env,
        "think_exhaustive",
        ThinkMode::Exhaustive.span_cap() as u32,
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
        let spec = compile_think_tree(mode, &env());
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

    /// **`Off` closes the block structurally, on the very next step.**
    ///
    /// The trigger has already emitted `<think>`; the tree's whole content is
    /// the closing run, so the rendered block is `<think>\n\n</think>\n\n` — the
    /// chat template's `enable_thinking: false` form — and the model never
    /// reaches a decode point inside it.
    ///
    /// The regression this guards: `Off` used to bind NO tree, leaving
    /// suppression to the dialect's `/no_think` marker. This model family has
    /// none, so nothing enforced it and every `Off` turn reasoned anyway.
    #[test]
    fn off_closes_the_block_immediately() {
        let v = vocab();
        let mut d = StencilDriver::new(tree_for(ThinkMode::Off));

        let (mask, closed) = step_to_decode(&mut d, &v);
        assert_eq!(
            closed, "\n\n</think>",
            "Off must prefill the closing run and nothing else"
        );
        assert!(
            matches!(mask, StepMask::Done),
            "Off must reach Done without ever offering a free-decode step"
        );
        assert!(d.is_done());
    }

    /// **Every closing static must END on `</think>`, with nothing after it.**
    ///
    /// A static run's last token is held back from the forward and rides the
    /// next decode step, which commits it and arms the index page cut. Put the
    /// marker anywhere earlier and no cut is made — not by the run (the flag is
    /// dropped for non-final tokens) and not by `reasoning_split` (it declines
    /// when the break token is last in the pass, which it then is). The turn
    /// still records `think_close_at`, so it seals with a reasoning span that is
    /// not a union of whole pages, and every LATER turn's projection then fails
    /// `Substrate::turn_sealed_without_thinking` — the conversation answers with
    /// that error forever after.
    ///
    /// This is invisible to any single-turn test, which is how it shipped once:
    /// `Off`'s static was `"\n\n</think>\n\n"` and a five-turn conversation
    /// poisoned itself from turn three.
    #[test]
    fn every_closing_static_ends_on_the_marker() {
        for mode in [
            ThinkMode::Off,
            ThinkMode::Quick,
            ThinkMode::Balanced,
            ThinkMode::Deep,
            ThinkMode::Exhaustive,
        ] {
            let spec = compile_think_tree(mode, &env());
            for text in spec.nodes.iter().filter_map(|n| match n {
                NodeSpec::Static { text, .. } => Some(text.as_str()),
                _ => None,
            }) {
                if text.contains("</think>") {
                    assert!(
                        text.ends_with("</think>"),
                        "{mode:?}: the closing static {text:?} has text after the marker, so \
                         the index page cut is never armed"
                    );
                }
            }
        }
    }

    /// `Off` budgets zero thinking tokens, and [`off`] is what makes that true:
    /// a tree with no [`NodeSpec::FreeText`] cannot be reasoned in. If a span
    /// were ever added here, `eos_budget` would under-reserve and the turn would
    /// be cut off mid-answer.
    #[test]
    fn off_has_no_free_span() {
        let spec = compile_think_tree(ThinkMode::Off, &env());
        assert!(
            !spec
                .nodes
                .iter()
                .any(|n| matches!(n, NodeSpec::FreeText { .. })),
            "Off must contain no free span"
        );
        assert_eq!(ThinkMode::Off.span_count(), 0);
    }

    /// The hard-cap closer gate: every dial has one span and it retires straight
    /// into the injected close, so a close there ends the block and the
    /// sampler's closing-statement script applies. The gate must be shut while
    /// the cursor is outside free text (the block-opening static's prefill).
    #[test]
    fn every_dials_span_is_terminal() {
        let v = vocab();

        for mode in [
            ThinkMode::Quick,
            ThinkMode::Balanced,
            ThinkMode::Deep,
            ThinkMode::Exhaustive,
        ] {
            let mut d = StencilDriver::new(tree_for(mode));
            assert!(
                !d.in_terminal_close_span(),
                "{mode:?}: not in free text yet — the gate is closed during static prefill"
            );
            let (mask, _) = step_to_decode(&mut d, &v);
            assert!(matches!(mask, StepMask::Free { .. }));
            assert!(
                d.in_terminal_close_span(),
                "{mode:?}: the only span is terminal — the closer script applies"
            );
        }
    }

    /// **The regression guard for the injection defect.** Walk every mode's spec
    /// and assert that the only text the tree ever prefills is the block-opening
    /// newline and the closing tag. Any word put into the block reads to the
    /// model as conversation content, not private scratch (module docs), so a
    /// static carrying prose is the bug returning — whatever it says.
    #[test]
    fn no_tree_injects_words_into_the_block() {
        for mode in [
            ThinkMode::Off,
            ThinkMode::Quick,
            ThinkMode::Balanced,
            ThinkMode::Deep,
            ThinkMode::Exhaustive,
        ] {
            let spec = compile_think_tree(mode, &env());
            // Strip the structural tag and whatever remains must be whitespace:
            // that admits `Off`'s combined closing run without admitting a
            // single word anywhere in any tree.
            for text in spec.nodes.iter().filter_map(|n| match n {
                NodeSpec::Static { text, .. } => Some(text.as_str()),
                _ => None,
            }) {
                let bare = text.replace("</think>", "");
                assert!(
                    bare.chars().all(char::is_whitespace),
                    "{mode:?}: tree prefills {text:?} into the think block"
                );
            }
            let statics: Vec<&str> = spec
                .nodes
                .iter()
                .filter_map(|n| match n {
                    NodeSpec::Static { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            if mode == ThinkMode::Off {
                // One combined run — the opener has nothing to open into.
                assert_eq!(statics, vec!["\n\n</think>"]);
                continue;
            }
            assert_eq!(
                statics.len(),
                2,
                "{mode:?}: expected exactly the opener and the close, got {statics:?}"
            );
            for text in statics {
                assert!(
                    text == BLOCK_OPEN || text == "</think>",
                    "{mode:?}: tree prefills {text:?} into the think block"
                );
            }
        }
    }

    #[test]
    fn quick_opens_the_block_then_closes() {
        let v = vocab();
        let mut d = StencilDriver::new(tree_for(ThinkMode::Quick));

        // First step prefills the block-opening newline — and nothing else —
        // then arrives at the free span.
        let (mask, primed) = step_to_decode(&mut d, &v);
        assert_eq!(primed, "\n");
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

        // The block-opening newline prefills, then the free span.
        let (mask, primed) = step_to_decode(&mut d, &v);
        assert_eq!(primed, "\n");
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

    /// **The upper dials close on the model's FIRST close, like every other dial.**
    ///
    /// `Deep`'s extra depth is budget, not injection, so the first `</think>`
    /// the model samples ends the block — dropped and replaced by the tree's own
    /// tag, which is what keeps the close single.
    #[test]
    fn deep_closes_on_the_first_close() {
        let v = vocab();
        let mut d = StencilDriver::new(tree_for(ThinkMode::Deep));

        let (mask, primed) = step_to_decode(&mut d, &v);
        assert_eq!(primed, "\n");
        assert!(matches!(mask, StepMask::Free { .. }));

        free_decode(&mut d, 4);
        assert_eq!(
            d.accept(THINK_CLOSE_ID, &v.token_bytes(THINK_CLOSE_ID)),
            Healed::Drop
        );

        let (mask, closed) = step_to_decode(&mut d, &v);
        assert_eq!(closed, "</think>");
        assert!(matches!(mask, StepMask::Done));
        assert!(d.is_done());
        assert_eq!(d.stats().think_continuations, 1);
    }

    /// `deep_closes_on_eos_too`: an EOS sample closes `Deep`'s span exactly like
    /// `</think>` — intercepted, dropped, and the injected tag prefills.
    #[test]
    fn deep_closes_on_eos_too() {
        let v = vocab();
        let mut d = StencilDriver::new(tree_for(ThinkMode::Deep));

        let (mask, _) = step_to_decode(&mut d, &v);
        assert!(matches!(mask, StepMask::Free { .. }));
        free_decode(&mut d, 4);

        let eos = v.eos();
        assert_eq!(d.accept(eos, &v.token_bytes(eos)), Healed::Drop);
        let (mask, closed) = step_to_decode(&mut d, &v);
        assert_eq!(closed, "</think>");
        assert!(matches!(mask, StepMask::Done));
        assert_eq!(d.stats().think_continuations, 1);
    }

    /// The whole prefill sequence of the widest dial, in order: the block-opening
    /// newline, then the closing tag. Nothing between them — no `"But wait, "`,
    /// no `"Alternatively, "`, no closing statement.
    #[test]
    fn exhaustive_prefills_only_the_newline_and_the_close() {
        let v = vocab();
        let mut d = StencilDriver::new(tree_for(ThinkMode::Exhaustive));

        let mut prefills: Vec<String> = Vec::new();
        let (mask, first) = step_to_decode(&mut d, &v);
        prefills.push(first);
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
        assert_eq!(prefills, vec!["\n".to_string(), "</think>".to_string()]);
        assert!(d.is_done());
        assert_eq!(d.stats().think_continuations, 1);
    }

    // ── Invariant: every produced tree compiles cleanly and reaches End ──────

    #[test]
    fn all_modes_compile_clean() {
        for mode in [
            ThinkMode::Off,
            ThinkMode::Quick,
            ThinkMode::Balanced,
            ThinkMode::Deep,
            ThinkMode::Exhaustive,
        ] {
            let spec = compile_think_tree(mode, &env());
            // Compiling enforces: forced_after > 0 on every FreeText, every path
            // reaches End, no adjacent statics, acyclic.
            let tree = compile(&spec, &vocab())
                .unwrap_or_else(|e| panic!("{mode:?} failed to compile: {e}"));
            assert!(tree.len() >= 2, "{mode:?} tree too small");
        }
    }

    /// **The locked ladder.** These four numbers are the product decision, not
    /// an implementation detail: `graceful` is where a think block actually
    /// lands (the close fires at the next clause boundary once passed), so this
    /// table IS the thinking length each rung buys. Pinned literally, because a
    /// silent drift here changes how the assistant reasons on every turn and
    /// nothing else in the suite would notice.
    ///
    /// Derived columns are checked by
    /// [`eot_budget_scales_and_stays_under_span_caps`]; this test fixes the
    /// anchors they derive from.
    #[test]
    fn the_thinking_ladder_is_512_1024_2048_4096() {
        use ThinkMode::*;
        let graceful = |m: ThinkMode| m.eot_budget().0;
        assert_eq!(graceful(Quick), 512);
        assert_eq!(graceful(Balanced), 1024);
        assert_eq!(graceful(Deep), 2048);
        assert_eq!(graceful(Exhaustive), 4096);

        // Each rung doubles the one below: the dial is a power-of-two ladder,
        // so "one notch up" always means "twice the thinking".
        for (lo, hi) in [(Quick, Balanced), (Balanced, Deep), (Deep, Exhaustive)] {
            assert_eq!(
                graceful(lo) * 2,
                graceful(hi),
                "{hi:?} must be exactly twice {lo:?}"
            );
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
        // The force budget grows with the dial — exhaustive thinks longest.
        let forces: Vec<i32> = [Quick, Balanced, Deep, Exhaustive]
            .iter()
            .map(|m| m.eot_budget().1)
            .collect();
        assert!(
            forces.windows(2).all(|w| w[0] < w[1]),
            "force budget must increase with the dial: {forces:?}"
        );
        // The span cap is a BACKSTOP, so it must sit a long way above the EOT
        // cutoff that is meant to do the closing — not a hair above it, or the
        // cap becomes the routine terminator and blocks end mid-word. Held at
        // exactly 2x: `force = span_cap / 2`.
        for m in [Quick, Balanced, Deep, Exhaustive] {
            assert_eq!(
                m.eot_budget().1 * 2,
                m.span_cap(),
                "{m:?}: the runaway cap must be twice the EOT force cutoff"
            );
        }
        // force = 1.5x graceful, so the clause-boundary close gets a real
        // window to find a boundary in before the hard cutoff.
        for m in [Quick, Balanced, Deep, Exhaustive] {
            let (g, f) = m.eot_budget();
            assert_eq!(
                g * 3,
                f * 2,
                "{m:?}: expected force({f}) == 1.5 x graceful({g})"
            );
        }
        // `Off` never decodes inside the block, so it reserves nothing.
        assert_eq!(Off.span_cap(), 0);
    }

    #[test]
    fn eos_budget_is_derived_from_the_tree_and_clears_thinking() {
        use ThinkMode::*;
        // Every steered dial is a single span — the depth is budget, not spans.
        for m in [Quick, Balanced, Deep, Exhaustive] {
            assert_eq!(m.span_count(), 1, "{m:?} must be a single span");
        }
        assert_eq!(Off.span_count(), 0);

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
            // Reserving the cutoff over-reserves by exactly the gap between it
            // and where blocks actually land (`graceful`), and that slack falls
            // into the answer's budget — so `response_length` bounds a runaway
            // answer loosely, the more so the higher the dial. Held to half the
            // landing point, which is what `force = 1.5 x graceful` makes it;
            // reserving the SPAN CAP instead would make it 2x the landing point
            // and the dial close to meaningless. The tight fix is to rebuild the
            // budget from where the block actually closed rather than from a
            // prediction, which this derivation cannot do.
            let slack = thinking - m.eot_budget().0; // reserved minus where blocks land
            assert_eq!(
                slack * 2,
                m.eot_budget().0,
                "{m:?}: reservation slack must stay half the landing point"
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
