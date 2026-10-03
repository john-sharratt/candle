//! Where one sequence's drafted block has to stop.
//!
//! A speculative step samples a block's positions in order (the accept walk) and
//! then runs each accepted token through the per-token commit path. A few tokens
//! change the rules the NEXT position was sampled under:
//!
//! - a trigger token opens a grammar, whose first decode may be masked;
//! - inside a grammar's free-text span, a token that closes the span, is dropped
//!   or is committed as other bytes (a heal) moves the walk off the span;
//! - a token that follows a page-break token arms an index page cut, which has
//!   to land on the token's own step.
//!
//! A position sampled past one of those was drawn under rules that no longer
//! hold, so the commit path refuses it. The walk has to stop at the same token,
//! not merely have the tail discarded afterwards: the sampler records every token
//! it draws into the sequence's penalty history, so a walk that ran on would
//! leave that history ahead of what the sequence committed.
//!
//! The same guard decides whether a sequence may draft at all. Free decode may,
//! and so may a grammar sitting in a free-text span — nothing is masked there,
//! so a proposal is drawn under exactly the rules a plain step would apply. A
//! grammar at a branch, a static run or its exit may not: its next token is
//! constrained or written, and there is nothing to speculate about.

use std::sync::Arc;

use crate::stencil::{Healed, StencilDriver, StepMask, TriggerRegistry};

/// What the block walks through, which decides what ends it.
enum Walk {
    /// No grammar is running; a trigger token starts one.
    Open(Arc<TriggerRegistry>),
    /// A grammar in a free-text span. A copy of the sequence's driver, advanced
    /// along the block, so the sequence's own walk only moves at commit.
    Free(StencilDriver),
}

/// One sequence's stop rule for a speculative step.
pub(super) struct BlockGuard {
    walk: Walk,
    /// The token before the next one judged: the sequence's last committed
    /// token at the start of the step, then each token the guard has passed.
    prev: Option<u32>,
}

impl BlockGuard {
    /// The guard for a sequence about to decode, or `None` when it may not
    /// draft this step.
    ///
    /// `pending` is the action the sequence's grammar yielded for this step,
    /// and `last` its last committed token.
    pub(super) fn for_step(
        stencil: Option<&StencilDriver>,
        pending: Option<&StepMask>,
        triggers: &Arc<TriggerRegistry>,
        last: Option<u32>,
    ) -> Option<Self> {
        let walk = match (stencil, pending) {
            (None, _) => Walk::Open(Arc::clone(triggers)),
            (Some(driver), Some(StepMask::Free { .. })) if driver.mid_free_span() => {
                Walk::Free(driver.clone())
            }
            _ => return None,
        };
        Some(Self { walk, prev: last })
    }

    /// Whether the block may go on to the position after `token`.
    ///
    /// Called once per position, in order, with the token the walk accepted
    /// there. `bytes` decodes `token`; only a free-text span reads it.
    /// `breaks` are the page-break tokens.
    pub(super) fn continues_after(
        &mut self,
        token: u32,
        bytes: impl FnOnce() -> Vec<u8>,
        breaks: &[u32],
    ) -> bool {
        // The commit that follows a break token (or opens the turn) arms the
        // page cut, and the commit loop takes nothing after it this step.
        let opens_page = self.prev.is_none_or(|p| breaks.contains(&p));
        self.prev = Some(token);
        let stays = match &mut self.walk {
            Walk::Open(triggers) => !triggers.is_trigger(token),
            Walk::Free(driver) => {
                driver.accept(token, &bytes()) == Healed::No && driver.mid_free_span()
            }
        };
        stays && !opens_page
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stencil::{
        compile, compile_think_tree, FreeTextLimits, NodeSpec, StencilTree, Terminator, TestVocab,
        ThinkMode, ThinkSteerEnvelope, TokenId, TreeSpec, Vocab,
    };

    const THINK_OPEN: TokenId = 151667;
    const THINK_CLOSE: TokenId = 151668;
    const BREAKS: &[u32] = &[THINK_OPEN, THINK_CLOSE];

    fn vocab() -> TestVocab {
        TestVocab::new()
            .with_special("<think>", THINK_OPEN)
            .with_special("</think>", THINK_CLOSE)
    }

    fn think_tree() -> Arc<StencilTree> {
        let env = ThinkSteerEnvelope {
            think_open: THINK_OPEN,
            think_close: THINK_CLOSE,
            eos: vocab().eos(),
            after_close: "",
        };
        let spec = compile_think_tree(ThinkMode::Balanced, &env);
        Arc::new(compile(&spec, &vocab()).unwrap())
    }

    /// `{"` → a JSON string span → `}`: the shape of a tool-call string value.
    fn string_tree() -> Arc<StencilTree> {
        let mut spec = TreeSpec::new("string");
        let end = spec.push(NodeSpec::End);
        let close = spec.push(NodeSpec::Static {
            text: "}".into(),
            next: end,
        });
        let span = spec.push(NodeSpec::FreeText {
            term: Terminator::JsonString,
            eos_ends: false,
            limits: FreeTextLimits::json_string(),
            close_token: None,
            suppress_close: false,
            next: close,
        });
        spec.root = spec.push(NodeSpec::Static {
            text: "{\"".into(),
            next: span,
        });
        Arc::new(compile(&spec, &vocab()).unwrap())
    }

    /// A driver stepped past its opening static to its first decode, with the
    /// action that decode is taken under.
    fn at_first_decode(tree: Arc<StencilTree>) -> (StencilDriver, StepMask) {
        let mut d = StencilDriver::new(tree);
        loop {
            match d.step() {
                StepMask::Prefill(_) => continue,
                action => return (d, action),
            }
        }
    }

    fn byte(b: u8) -> (u32, Vec<u8>) {
        (b as u32, vec![b])
    }

    fn free_guard(tree: Arc<StencilTree>) -> BlockGuard {
        let (d, action) = at_first_decode(tree);
        let triggers = Arc::new(TriggerRegistry::new());
        BlockGuard::for_step(Some(&d), Some(&action), &triggers, Some(b'a' as u32))
            .expect("a grammar in a free-text span drafts")
    }

    #[test]
    fn a_think_span_runs_until_its_close_is_dropped() {
        let mut g = free_guard(think_tree());
        for b in *b"so the answer is" {
            let (t, bytes) = byte(b);
            assert!(g.continues_after(t, || bytes, BREAKS), "{:?}", b as char);
        }
        assert!(
            !g.continues_after(THINK_CLOSE, || b"</think>".to_vec(), BREAKS),
            "the model's own close is dropped and the tree writes its tag next"
        );
    }

    #[test]
    fn an_end_of_turn_inside_a_think_span_stops_the_block() {
        let mut g = free_guard(think_tree());
        let eos = vocab().eos();
        assert!(!g.continues_after(eos, Vec::new, BREAKS));
    }

    #[test]
    fn a_string_value_stops_at_its_closing_quote() {
        let mut g = free_guard(string_tree());
        let (a, ab) = byte(b'a');
        assert!(g.continues_after(a, || ab, BREAKS));
        let (q, qb) = byte(b'"');
        assert!(
            !g.continues_after(q, || qb, BREAKS),
            "the quote closes the span cleanly, and the static after it is next"
        );
    }

    #[test]
    fn a_healed_token_stops_the_block_even_inside_the_span() {
        let mut g = free_guard(string_tree());
        // A raw newline is committed as its escape; the span stays open, but
        // the token in the block is not the one the sequence commits.
        let (nl, nlb) = byte(b'\n');
        assert!(!g.continues_after(nl, || nlb, BREAKS));
    }

    #[test]
    fn free_decode_stops_on_a_trigger() {
        let mut triggers = TriggerRegistry::new();
        triggers.register(THINK_OPEN, think_tree());
        let mut g = BlockGuard::for_step(None, None, &Arc::new(triggers), Some(b'a' as u32))
            .expect("free decode drafts");
        let (h, hb) = byte(b'h');
        assert!(g.continues_after(h, || hb, BREAKS));
        assert!(!g.continues_after(THINK_OPEN, Vec::new, BREAKS));
    }

    #[test]
    fn the_token_after_a_break_stops_the_block() {
        let triggers = Arc::new(TriggerRegistry::new());
        let mut g = BlockGuard::for_step(None, None, &triggers, Some(THINK_CLOSE))
            .expect("free decode drafts");
        let (h, hb) = byte(b'h');
        assert!(
            !g.continues_after(h, || hb, BREAKS),
            "this token opens the answer's page, and the cut lands on its own step"
        );
    }

    #[test]
    fn the_first_token_of_a_turn_stops_the_block() {
        let triggers = Arc::new(TriggerRegistry::new());
        let mut g = BlockGuard::for_step(None, None, &triggers, None).expect("free decode drafts");
        let (h, hb) = byte(b'h');
        assert!(!g.continues_after(h, || hb, BREAKS));
    }

    #[test]
    fn a_grammar_outside_a_free_span_does_not_draft() {
        let triggers = Arc::new(TriggerRegistry::new());
        let d = StencilDriver::new(string_tree());
        // Nothing stepped: the walk is at its opening static.
        assert!(BlockGuard::for_step(Some(&d), None, &triggers, Some(1)).is_none());
        let (d, action) = at_first_decode(string_tree());
        assert!(matches!(action, StepMask::Free { .. }));
        assert!(
            BlockGuard::for_step(Some(&d), Some(&StepMask::Done), &triggers, Some(1)).is_none(),
            "an action other than a free decode is not drafted under"
        );
    }
}
