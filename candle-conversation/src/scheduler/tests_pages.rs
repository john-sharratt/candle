//! Tests for the index-page boundary: the cut that separates a turn's own tokens
//! from the prefix ahead of them, and the places where it must land.
//!
//! The cut is taken on the parent before a fork is carved, so the fork copies a
//! closed index and a turn's tokens never share a page with the prefix before
//! them. A second turn does not re-close a prefix the parent has already closed,
//! which would make the page list grow with every turn. Every break token ends a
//! page and stays inside it, and a suppressed token leaves nothing pending, so
//! nothing is forwarded twice. A commit only marks the cut and the flush takes it,
//! because a cut inside a speculative rewind's window would count rows twice.

use super::*;

// •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
// Scheduler unit tests
// •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
//
// The `DummyModel` is a minimal `ManagedBatchedModel` that returns dummy CPU
// tensors so no GPU is needed. `BatchedInferenceSession` is constructed on
// `Device::Cpu` with tiny arena dimensions so these tests run without CUDA.

#[cfg(test)]
pub(super) mod tests {
    // Expected token totals are written as the sum of the per-item terms they
    // came from, including the zero ones, so the arithmetic under test is
    // visible in the assertion.
    #![allow(clippy::identity_op)]

    use super::*;

    use candle_transformers::models::batched_inference::ManagedBatchedModel;

    use super::super::tests_fixtures::tests::{
        forward_on, make_test_scheduler_recurrent, DummyModel,
    };

    use super::super::tests_toy_model::tests::{DummyRecurrentModel, RecurrentProbe, ToyPages};

    /// **The cut lands on the parent, and it lands before the fork.**
    ///
    /// `fork_recurrent` copies the index wholesale, live tail included. A
    /// boundary taken after the carve leaves the parent's tail open — so it
    /// survives into the view as rows the turn's own tokens then extend, and the
    /// parent re-accumulates the same prefix for the next turn to close again.
    #[test]
    fn create_view_closes_the_parents_index_before_forking_it() {
        let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
        let parent = scheduler.session.create_sequence().unwrap();

        // A projected prefix, forwarded on the parent — the case that fails
        // silently, because borrowed prefix arrives as pages already.
        forward_on(&mut scheduler, parent, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            probe.get_pages(parent),
            ToyPages {
                closed: Vec::new(),
                tail: 8
            },
            "a forwarded prefix leaves its rows in the live tail"
        );

        let (view_id, _) = scheduler
            .create_view(SequenceId(parent), &[BlockRange::new(0, 1)])
            .expect("view creation");

        assert_eq!(
            probe.get_pages(parent),
            ToyPages {
                closed: vec![8],
                tail: 0
            },
            "the parent's prefix must be a page of its own once the turn begins"
        );
        assert_eq!(
            probe.get_pages(view_id.0),
            ToyPages {
                closed: vec![8],
                tail: 0
            },
            "the fork must copy an already-closed cache, or the turn extends the prefix's tail"
        );
    }

    /// The property the boundary exists for: a turn's own width is recoverable
    /// from the index, separately from the prefix ahead of it.
    ///
    /// Without the cut this is one undifferentiated run of 13, and the seal —
    /// which asks for "the last N tokens" — takes 5 tokens' worth of rows from a
    /// page spanning 13. That is the measured 376-vs-53 failure in miniature.
    #[test]
    fn a_turns_own_tokens_never_share_a_page_with_the_prefix_ahead_of_them() {
        let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
        let parent = scheduler.session.create_sequence().unwrap();
        forward_on(&mut scheduler, parent, &[1, 2, 3, 4, 5, 6, 7, 8]);

        let (view_id, _) = scheduler
            .create_view(SequenceId(parent), &[BlockRange::new(0, 1)])
            .expect("view creation");
        forward_on(&mut scheduler, view_id.0, &[9, 10, 11, 12, 13]);

        let pages = probe.get_pages(view_id.0);
        assert_eq!(
            pages,
            ToyPages {
                closed: vec![8],
                tail: 5
            },
            "the turn's 5 tokens must stand apart from the 8-token prefix"
        );
        assert_eq!(pages.indexed(), 13, "and nothing may be lost by the split");
    }

    /// A slot that accumulates across turns must not re-close the same prefix
    /// once per turn.
    ///
    /// This is why the cut is on the parent rather than the view. Closing on the
    /// view leaves the parent's tail open, so the *next* carve closes the whole
    /// accumulated prefix again — a page list that grows quadratically with turn
    /// count, and page boundaries that move under a turn already sealed against
    /// them.
    #[test]
    fn a_second_turn_does_not_re_close_the_first_turns_prefix() {
        let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
        let parent = scheduler.session.create_sequence().unwrap();
        forward_on(&mut scheduler, parent, &[1, 2, 3, 4, 5, 6, 7, 8]);

        let (first, _) = scheduler
            .create_view(SequenceId(parent), &[BlockRange::new(0, 1)])
            .expect("first view");
        forward_on(&mut scheduler, first.0, &[9, 10, 11, 12]);
        // The view finalizes: its pages move to the parent, as `move_recurrent`
        // does at the end of every turn.
        scheduler.model.move_recurrent(first.0, parent).unwrap();

        let (second, _) = scheduler
            .create_view(SequenceId(parent), &[BlockRange::new(0, 1)])
            .expect("second view");

        assert_eq!(
            probe.get_pages(second.0),
            ToyPages {
                closed: vec![8, 4],
                tail: 0
            },
            "each turn contributes exactly one page; the prefix is not re-closed"
        );
    }

    /// An empty tail closes nothing, so the cut is free on the path that was
    /// already correct — and a slot may pass the boundary twice without the
    /// second pass inventing a zero-width page.
    #[test]
    fn a_unit_boundary_on_an_empty_tail_adds_no_page() {
        let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
        let parent = scheduler.session.create_sequence().unwrap();
        forward_on(&mut scheduler, parent, &[1, 2, 3, 4]);

        scheduler.begin_unit(SequenceId(parent), "test");
        scheduler.begin_unit(SequenceId(parent), "test");
        scheduler.begin_unit(SequenceId(parent), "test");

        assert_eq!(
            probe.get_pages(parent),
            ToyPages {
                closed: vec![4],
                tail: 0
            },
            "closing an empty tail must be a no-op, not a zero-width page"
        );
    }

    // —— reasoning-boundary tests ————————————————————————————————————————————

    /// A `DecodeState` carrying nothing but the two fields the reasoning
    /// boundary is decided from.
    pub(in super::super) fn boundary_state() -> (DecodeState, Receiver<TurnEvent>) {
        let (tx, rx) = flume::unbounded();
        let state = DecodeState {
            event_tx: tx,
            generated_tokens: TokenBuffer::default(),
            think_close_at: None,
            forwarded_generated: 0,
            draft_depth: DraftDepth::default(),
            pending_page_cut: false,
            pending_page_cut_after: None,
            max_tokens: 64,
            sampling_config: SamplingConfig::default(),
            seal_action: SealAction::None,
            free_tool_calls_from_penalties: false,
            prefill_assistant_text: String::new(),
            finished: false,
            faulted: false,
            finish: FinishReason::Stop,
            decode_start: Instant::now(),
            decode_busy_us: 0,
            prefill_ms: 0.0,
            prefill_token_count: 0,
            turn_start: Instant::now(),
            health: crate::decode_health::DecodeHealthState::new(8, 8),
            reprojection: None,
            boundary: BoundaryState::default(),
            last_projection_end: 0,
            post_decode_tokens: TokenBuffer::default(),
            belief: PriorBelief::default(),
            prefill_tokens: TokenBuffer::default(),
            user_text: String::new(),
            tags: Vec::new(),
            user_content_start: 0,
            user_content_end: 0,
            assistant_content_start: 0,
            no_think: false,
            wrote_tool_call: false,
            triggers: Arc::new(TriggerRegistry::new()),
            stencil: None,
            pending_mask: None,
            recorded_reply: None,
        };
        (state, rx)
    }

    /// The turn's own tokens, walked through the one funnel: exactly two cuts,
    /// at the turn's first token and at the token after `</think>`.
    ///
    /// **Every break token ends a page; the marker stays in the page it ends.**
    ///
    /// The span and the page boundary are now separate questions.
    /// `think_close_at` still records only the FIRST `</think>`, because it
    /// locates the reasoning span the layout carries — while a page ends at
    /// *any* break token, including a second `</think>` a steering tree replays
    /// in the answer body. An extra boundary there costs nothing: the projection
    /// drops pages by position and width, so a finer split is still exact.
    #[test]
    fn every_break_token_ends_a_page_and_stays_inside_it() {
        const OPEN: u32 = 76;
        const CLOSE: u32 = 77;
        const BREAKS: &[u32] = &[OPEN, CLOSE];
        let probe = RecurrentProbe::default();
        let model = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe.clone(),
        };
        let (mut state, _rx) = boundary_state();
        let slot = SequenceId(0);

        // One token per step, each forwarded after it is committed — the decode
        // loop's shape. The tail advances on the forward, so a page's width is
        // the tokens between two cuts.
        // Commit, then flush, then forward — the decode loop's real order. The
        // cut is deferred so it cannot land inside a speculative rewind's
        // window; here nothing speculates, so the flush follows immediately.
        for &t in &[10u32, 11, CLOSE, 12, 13, CLOSE, 14] {
            state.push_committed(t, Some(CLOSE), BREAKS);
            flush_page_cut(&model, slot, &mut state);
            probe.advance_index(slot.0, 1);
        }

        assert_eq!(
            state.think_close_at,
            Some(2),
            "the SPAN is still fixed by the first close only"
        );
        assert_eq!(
            probe.get_pages(slot.0),
            ToyPages {
                // `[10, 11, CLOSE]` then `[12, 13, CLOSE]`, each closed by the
                // token that follows its marker; `14` is still open.
                closed: vec![3, 3],
                tail: 1
            },
            "a page ends after each close marker, with the marker inside it"
        );
    }

    /// **A suppressed token leaves nothing pending, so nothing is re-sent.**
    ///
    /// The decode loop takes each step's input from the trailing generated
    /// token, on the understanding that exactly one is always
    /// committed-but-unforwarded. A dropped token commits nothing, so that
    /// understanding fails and the previous token — already forwarded — is
    /// carried a second time. The K/V hides it (the offset never advanced, so
    /// the re-forward overwrites), while the index appends a row and the turn
    /// then seals with more rows than tokens.
    ///
    /// The count makes the question answerable: after a commit one token is
    /// pending; after that token is forwarded, none is; and a drop adds nothing,
    /// so none is pending then either.
    #[test]
    fn a_suppressed_token_leaves_nothing_pending_to_re_forward() {
        // No model here: this is pure `DecodeState` bookkeeping — which of the
        // recorded tokens have gone out — and involves neither the index nor a
        // page cut.
        const BREAKS: &[u32] = &[77];
        let (mut state, _rx) = boundary_state();

        state.push_committed(10, None, BREAKS);
        assert_eq!(
            state.pending_forward(),
            &[10],
            "a freshly committed token is waiting to be forwarded"
        );

        // The decode step carries it.
        state.mark_forwarded();
        assert!(
            state.pending_forward().is_empty(),
            "and stops being pending once it has gone out"
        );

        // Now the sampler produces a token that steering suppresses: it is
        // never committed, so `generated_tokens` does not grow.
        assert_eq!(
            state.generated_tokens.len(),
            1,
            "a dropped token commits nothing"
        );
        assert!(
            state.pending_forward().is_empty(),
            "so there is still nothing pending — the previous token must NOT be \
             carried again, which is exactly what the old `last()` rule did"
        );

        // The steering's own run then arrives; its last token rides the next step.
        state.push_forwarded(11, None, BREAKS);
        state.push_pending(12, None, BREAKS);
        assert_eq!(
            state.pending_forward(),
            &[12],
            "only the token that rides the next decode is pending"
        );
        assert_eq!(
            state.forwarded_generated, 2,
            "and the seal takes exactly the two that were actually forwarded"
        );
    }

    /// **A commit records the cut; it does not take it.**
    ///
    /// The separation is the whole fix for a speculative wave.
    /// `IndexCache::snapshot` covers the live tail and NOT the pages, so a cut
    /// taken between the snapshot and the rewind moves the tail's rows into a
    /// page and the rewind then restores the same rows into the tail — counting
    /// them twice, and leaving the index exactly one drafted block past the K/V
    /// for the rest of the sequence. Measured at a constant +5 against a 5-token
    /// block, on a full accept as much as a partial one.
    ///
    /// So `commit_generated` may only mark, and the cut lands when the caller
    /// says the K/V is final.
    #[test]
    fn a_commit_marks_the_cut_and_the_flush_takes_it() {
        const BREAKS: &[u32] = &[77];
        let probe = RecurrentProbe::default();
        let model = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe.clone(),
        };
        let (mut state, _rx) = boundary_state();
        let slot = SequenceId(0);
        probe.advance_index(slot.0, 9);

        state.push_committed(10, None, BREAKS);
        assert!(state.pending_page_cut, "the turn's first token earns a cut");
        assert_eq!(
            probe.get_pages(slot.0),
            ToyPages {
                closed: Vec::new(),
                tail: 9
            },
            "but the commit must NOT have taken it — a rewind may still be in flight"
        );

        flush_page_cut(&model, slot, &mut state);
        assert!(!state.pending_page_cut, "and the flush consumes it");
        assert_eq!(
            probe.get_pages(slot.0),
            ToyPages {
                closed: vec![9],
                tail: 0
            },
            "the page lands once the K/V is final"
        );

        // Idempotent: a second flush with nothing pending must not invent a page.
        flush_page_cut(&model, slot, &mut state);
        assert_eq!(
            probe.get_pages(slot.0),
            ToyPages {
                closed: vec![9],
                tail: 0
            },
            "a flush with nothing pending is a no-op"
        );
    }

    /// The list is read off the stream, so a marker the sampler never produced
    /// still ends a page — which is the case the previous design could not see.
    ///
    /// `think_close` is `None` here: no span is located at all, and the page
    /// boundaries are unaffected. That pairing is the whole point — a turn whose
    /// `<think>…</think>` was prefilled recorded no `think_close_at`, and under
    /// the old rule therefore cut nothing.
    #[test]
    fn a_page_ends_at_a_break_token_even_with_no_span_recorded() {
        const OPEN: u32 = 76;
        const TURN_END: u32 = 99;
        const BREAKS: &[u32] = &[OPEN, 77, TURN_END];
        let probe = RecurrentProbe::default();
        let model = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe.clone(),
        };
        let (mut state, _rx) = boundary_state();
        let slot = SequenceId(0);

        for &t in &[OPEN, 10, 11, TURN_END, 12] {
            state.push_committed(t, None, BREAKS);
            flush_page_cut(&model, slot, &mut state);
            probe.advance_index(slot.0, 1);
        }

        assert_eq!(state.think_close_at, None, "no span was located");
        assert_eq!(
            probe.get_pages(slot.0),
            ToyPages {
                // `[OPEN]`, then `[10, 11, TURN_END]`, then `12` still open.
                closed: vec![1, 3],
                tail: 1
            },
            "both markers ended a page despite no span being recorded"
        );
    }

    /// A token that has already been forwarded records its boundary but must not
    /// cut: its row is in the page the cut would have started after.
    #[test]
    fn an_already_forwarded_token_records_its_boundary_without_cutting() {
        const CLOSE: u32 = 77;
        let probe = RecurrentProbe::default();
        let model = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe.clone(),
        };
        let (mut state, _rx) = boundary_state();

        // A static run: forwarded as one span, then recorded token by token.
        probe.advance_index(0, 3);
        for &t in &[10u32, CLOSE, 11] {
            state.push_forwarded(t, Some(CLOSE), &[CLOSE]);
        }

        assert_eq!(
            state.think_close_at,
            Some(1),
            "the boundary is still recorded — a steered close is the only one \
             many turns ever emit"
        );
        assert_eq!(
            probe.get_pages(0),
            ToyPages {
                closed: Vec::new(),
                tail: 3
            },
            "but no cut, because the rows are already pooled"
        );
        // And the model was never asked, which is the point of the second name.
        assert!(
            model.close_positional_page(0).unwrap() == 3,
            "the tail is still open and closable by whoever owns the boundary"
        );
    }
}
