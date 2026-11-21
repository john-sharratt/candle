//! Tests for admission: that queued turns and sections are priced, offered at
//! their own band, and refused or answered correctly when ground is short.
//!
//! The admission tests run a stub model whose claims are either accepted or
//! refused. A refused claim is checked to wait while something is in flight and
//! to fail the turn when nothing is. Every figure admission compares is checked
//! to be in one currency, because mixing expert-cache occupancy with a zone
//! capacity once refused every offer on a healthy card. A store's cost is derived
//! without a division by zero, and a model with no planner still admits against
//! its width backstop.

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

    use candle::{DType, Tensor};

    use candle_nn::kv_cache::KV_DEVICE_OOM_MARKER;

    use candle_transformers::models::batched_inference::{
        BatchedInferenceSession, ManagedBatchedModel, WaveResult,
    };

    use candle_transformers::models::wave_driver::{StepOutcome, StepSpec};

    use super::super::tests_fixtures::tests::{
        make_dummy_tokenizer, make_test_session, stub_residual, stub_step, DummyModel,
    };

    /// A model whose every carried-state claim is refused — what admission meets
    /// when the ground below the weight floor is gone.
    #[derive(Clone)]
    pub(in super::super) struct RefusingClaim(pub(in super::super) DummyModel);

    impl ManagedBatchedModel for RefusingClaim {
        fn maybe_change_dtype(&self, dtype: DType) -> candle::Result<()> {
            self.0.maybe_change_dtype(dtype)
        }
        fn num_layers(&self) -> usize {
            self.0.num_layers()
        }
        fn n_kv_head(&self) -> usize {
            self.0.n_kv_head()
        }
        fn head_dim(&self) -> usize {
            self.0.head_dim()
        }
        fn wave_geometry(&self, act_dtype: DType) -> candle_nn::kv_cache::ModelGeometry {
            self.0.wave_geometry(act_dtype)
        }
        fn device(&self) -> &candle::Device {
            self.0.device()
        }
        #[allow(clippy::too_many_arguments)]
        fn forward_wave(
            &self,
            session: &mut BatchedInferenceSession,
            decode_seqs: &[usize],
            decode_inputs: &[Tensor],
            prefill_seqs: &[usize],
            prefill_inputs: &[Tensor],
            glue_seqs: &[usize],
            glue_inputs: &[Tensor],
            layer_start: usize,
            layer_end: usize,
            residual: Option<&mut CreepResidual>,
        ) -> candle::Result<WaveResult> {
            self.0.forward_wave(
                session,
                decode_seqs,
                decode_inputs,
                prefill_seqs,
                prefill_inputs,
                glue_seqs,
                glue_inputs,
                layer_start,
                layer_end,
                residual,
            )
        }
        fn forward_step(
            &self,
            session: &mut BatchedInferenceSession,
            spec: StepSpec<'_>,
        ) -> candle::Result<StepOutcome> {
            stub_step(self, session, spec)
        }
        fn claim_residual(
            &self,
            _session: &BatchedInferenceSession,
            rows: usize,
        ) -> candle::Result<CreepResidual> {
            stub_residual(rows)
        }
        fn prune(&self) -> candle::Result<()> {
            Ok(())
        }
        fn claim_carried_state(&self, _seq: usize) -> candle::Result<()> {
            candle::bail!("slot arena: no region for recurrent state")
        }
    }

    /// A scheduler over [`RefusingClaim`], sized as `admission_scheduler` is.
    pub(in super::super) fn refusing_claim_scheduler() -> Scheduler {
        let (_tx, rx) = flume::bounded(16);
        Scheduler::new(
            rx,
            Box::new(RefusingClaim(DummyModel::new())) as Box<dyn ManagedBatchedModel + Send>,
            make_test_session(),
            make_dummy_tokenizer(),
            vec![0u32].into(),
            64,
            8,
            false,
            None,
            DecodeHealthConfig::default(),
            512,
            PersistenceTrigger::noop(),
            SummariserTrigger::noop(),
            projection_assembler::BoundaryMarkers::default(),
            Arc::new(Guests::new()),
        )
    }

    /// **A refused claim waits while something is in flight to free ground.**
    /// The turn goes back to the head of the queue, unstarted.
    #[test]
    fn a_refused_claim_waits_while_something_is_in_flight() {
        let mut sched = refusing_claim_scheduler();
        let running = SequenceId(sched.session.create_sequence().unwrap());
        sched.begin_prefill(test_prefill_work(running));
        let waiting = SequenceId(sched.session.create_sequence().unwrap());
        sched.prefill_queue.push_back(test_prefill_work(waiting));

        sched.promote_new_prefills();

        assert_eq!(sched.prefill_queue.len(), 1, "back in the queue");
        assert_eq!(sched.prefill_queue[0].sequence_id, waiting);
        assert_eq!(
            sched.active_prefills.len(),
            1,
            "only the one already running"
        );
    }

    /// **With nothing in flight a refused claim fails the turn** — answered and
    /// cleaned up the way any failed prefill is, never dropped unanswered and
    /// never retried forever.
    #[test]
    fn a_refused_claim_with_nothing_in_flight_fails_the_turn() {
        let mut sched = refusing_claim_scheduler();
        let slot = SequenceId(sched.session.create_sequence().unwrap());
        sched.prefill_queue.push_back(test_prefill_work(slot));

        sched.promote_new_prefills();

        assert!(sched.prefill_queue.is_empty(), "not retried forever");
        assert_eq!(sched.active_prefills.len(), 1);
        assert!(
            sched.active_prefills[0].error.is_some(),
            "in flight already failed, so it leaves as a failed prefill does",
        );
    }

    /// A section whose claim is refused with nothing in flight is answered with
    /// the refusal, not left queued.
    #[test]
    fn a_refused_section_claim_with_nothing_in_flight_is_answered() {
        let mut sched = refusing_claim_scheduler();
        let (_, rx) = ingest_section(&mut sched, 64);

        sched.promote_new_prefills();

        assert!(sched.section_queue.is_empty());
        assert!(sched.active_section_ingests.is_empty());
        assert!(rx.try_recv().expect("answered").is_err());
    }

    // —— admission (`admit_ground::AdmitPass`) ———————————————————————————————

    /// A scheduler over the stub model, for the admission tests below.
    pub(in super::super) fn admission_scheduler() -> Scheduler {
        let (_tx, rx) = flume::bounded(16);
        Scheduler::new(
            rx,
            Box::new(DummyModel::new()) as Box<dyn ManagedBatchedModel + Send>,
            make_test_session(),
            make_dummy_tokenizer(),
            vec![0u32].into(),
            64,
            8,
            false,
            None,
            DecodeHealthConfig::default(),
            512,
            PersistenceTrigger::noop(),
            SummariserTrigger::noop(),
            projection_assembler::BoundaryMarkers::default(),
            Arc::new(Guests::new()),
        )
    }

    /// A minimal queued turn: the only fields admission reads are the slot and
    /// the token count it prices. Its caller holds it — the receiver is kept for
    /// the test's life — so the step does not withdraw it as let go
    /// ([`Scheduler::withdraw_abandoned_prefills`]).
    pub(in super::super) fn test_prefill_work(sequence_id: SequenceId) -> PrefillWork {
        let (event_tx, rx) = flume::bounded(16);
        std::mem::forget(rx);
        PrefillWork {
            sequence_id,
            tokens: TokenBuffer::from(vec![1u32; 64]),
            prefill_text: String::new(),
            user_text: String::new(),
            tags: Vec::new(),
            user_content_start: 0,
            user_content_end: 0,
            assistant_content_start: 0,
            no_think: false,
            prefill_assistant_text: String::new(),
            event_tx,
            max_decode_tokens: 16,
            sampling: SamplingConfig::default(),
            submitted_at: Instant::now(),
            reprojection: None,
            belief: PriorBelief::default(),
            seal_action: SealAction::Turn,
            post_decode_tokens: TokenBuffer::default(),
            projection_offsets: Vec::new(),
            staged_composition: None,
            triggers: Arc::new(TriggerRegistry::default()),
            turn_grammar: None,
            free_tool_calls_from_penalties: false,
            recorded_reply: None,
            recurrent_spans: Vec::new(),
            rereads: false,
        }
    }

    /// **A spent forward takes no more turns.** Turns of 64 tokens fill the
    /// pass's rows, and the pass ends there: the forward is filled in order
    /// (`gang::gang_shares`), so a turn past its rows would advance by
    /// nothing while holding its store. The rest wait in the queue for the next
    /// pass.
    #[test]
    fn a_spent_forward_takes_no_more_turns() {
        use super::admit::{Ground, Kind};
        use super::admit_ground::AdmitPass;
        use crate::projection::DecodePriority;

        let mut sched = admission_scheduler();
        let chunk = sched.prefill_pass_budget();
        let queued = chunk / CHUNK_SIZE + 4;
        for _ in 0..queued {
            let id = SequenceId(sched.session.create_sequence().unwrap());
            sched.prefill_queue.push_back(test_prefill_work(id));
        }

        let mut pass = AdmitPass::new(&mut sched);
        let mut offers = Vec::new();
        while let Some(cost) = pass.peek(Kind::Prefill, DecodePriority::High) {
            offers.push(cost.rows);
            assert!(pass.admit(Kind::Prefill, DecodePriority::High, cost));
        }
        assert!(
            offers.iter().all(|&r| r > 0),
            "every turn admitted rides the forward: {offers:?}"
        );
        assert_eq!(
            offers.iter().sum::<usize>(),
            chunk,
            "the rows go to the offers in order until the pass is spent"
        );
        assert!(offers.len() < queued, "the rest wait for the next pass");
        assert_eq!(sched.prefill_queue.len(), queued - offers.len());
    }

    /// **Every figure admission compares must be in one currency.**
    ///
    /// This integration has been wedged twice by mixing them — expert-cache
    /// occupancy against a zone capacity, then the zone's lagging extent against
    /// a live-region identity — and each time the symptom was every offer
    /// refused on a healthy card. The invariant that would have caught both is
    /// simply that `Budget::resident` is the same quantity `Headroom::zone` is,
    /// and that the floor is at or above the hold.
    #[test]
    fn the_headroom_and_the_budget_agree_on_one_currency() {
        let sched = admission_scheduler();
        let standing = sched.standing_tier_bytes();
        let room = sched.admit_headroom(standing);
        let budget = sched.admit_budget_terms(&room, 512);

        assert_eq!(
            budget.resident, room.zone,
            "residency and the zone must be the same measurement",
        );
        assert!(
            budget.decode_floor >= room.zone_min,
            "the decode floor may never sit under the hold: {} < {}",
            budget.decode_floor,
            room.zone_min,
        );
        // The prefill floor is deliberately lower — it is the ground a prefill may
        // spend for rows — but never below its own hold, and never above the
        // decode's.
        assert!(
            budget.prefill_floor >= room.zone_min_prefill
                && budget.prefill_floor <= budget.decode_floor,
            "the prefill floor sits between its own hold and the decode's: {} / {} / {}",
            room.zone_min_prefill,
            budget.prefill_floor,
            budget.decode_floor,
        );
        // The spendable ground is one subtraction, never the free list added
        // beside the zone — that double-counts the same regions.
        assert!(
            room.free_kv <= room.zone,
            "spendable ground cannot exceed the zone it comes out of",
        );
    }

    /// **The tier is held back from admission, so the floor clears the hold by
    /// a useful forward's worth.**
    ///
    /// Reserve nothing and K/V claims to the weight floor: the tier is the
    /// span's third tenant and gets what is left, which is nothing, so no
    /// forward can be planned at all — not a narrow one, none — and nothing
    /// completes to give the ground back.
    #[test]
    fn the_floor_holds_back_a_useful_forwards_tier() {
        let sched = admission_scheduler();
        let room = sched.admit_headroom(0);
        let budget = sched.admit_budget_terms(&room, 512);
        assert_eq!(
            budget.decode_floor,
            room.floor().saturating_add(sched.min_forward_tier_bytes()),
            "the floor is the hold, the eviction margin and a forward's tier",
        );
        // The prefill floor holds the tier back too: spending weights for rows is a
        // trade, leaving no ground for the tier is a stall.
        assert_eq!(
            budget.prefill_floor,
            room.prefill_floor()
                .saturating_add(sched.min_forward_tier_bytes()),
            "the prefill floor is the cache's minimum and a forward's tier",
        );
        // A wave already standing reserves its own tier, so the next forward
        // needs only what it does not already hold.
        let tier = sched.min_forward_tier_bytes() as usize;
        assert_eq!(sched.admit_headroom(tier).forward_reserve, 0);
        assert_eq!(
            sched.admit_headroom(tier / 2).forward_reserve,
            (tier - tier / 2) as u64,
        );
    }

    /// **A store cost is never derived from a division by zero.**
    ///
    /// The per-sequence figure is a whole-engine total over the live count, so
    /// the empty engine is the case that has to be stated rather than computed.
    #[test]
    fn the_recurrent_cost_is_zero_on_an_empty_engine() {
        let sched = admission_scheduler();
        assert!(sched.active_decodes.is_empty() && sched.active_prefills.is_empty());
        assert_eq!(sched.recurrent_cost(), 0);
    }

    /// **A stack that cannot describe its weight side still admits.**
    ///
    /// `weight_plan` is [`WeightPlanning::Dense`] for a model with no expert cache, and
    /// such a stack has no residency to trade a wave's rows against — so admission
    /// proceeds on its width backstop alone rather than waiting for a planner that will
    /// never arm.
    ///
    /// The stub is `Dense` specifically, not merely "no plan": a routed model reporting
    /// no plan is [`WeightPlanning::Incomplete`] and panics, because an unarmed planner
    /// narrows every wave to one row. This test covers the case that is legitimate.
    #[test]
    fn an_engine_with_no_planner_still_takes_its_queue_head() {
        let mut sched = admission_scheduler();
        assert!(
            matches!(
                sched.model.weight_plan(),
                candle_transformers::models::expert_lre::WeightPlanning::Dense
            ),
            "the stub reports no expert cache at all, which is the case under test",
        );
        let slot = sched.session.create_sequence().unwrap();
        sched
            .prefill_queue
            .push_back(test_prefill_work(SequenceId(slot)));

        sched.promote_new_prefills();

        assert!(
            sched.prefill_queue.is_empty(),
            "the head must leave the queue",
        );
        assert_eq!(
            sched.active_prefills.len(),
            1,
            "and must be in flight, or the engine never starts",
        );
    }

    /// A section request on a fresh slot, with a receiver for its answer.
    pub(in super::super) fn ingest_section(
        sched: &mut Scheduler,
        tokens: usize,
    ) -> (
        SequenceId,
        flume::Receiver<Result<SealResult, ConversationError>>,
    ) {
        ingest_section_at(sched, tokens, DecodePriority::Low)
    }

    /// [`ingest_section`] at `band`.
    pub(in super::super) fn ingest_section_at(
        sched: &mut Scheduler,
        tokens: usize,
        band: DecodePriority,
    ) -> (
        SequenceId,
        flume::Receiver<Result<SealResult, ConversationError>>,
    ) {
        let slot = SequenceId(sched.session.create_sequence().unwrap());
        let (response_tx, response_rx) = flume::bounded(1);
        assert!(sched.handle_request(SchedulerRequest::IngestSection {
            sequence_id: slot,
            section_id: SectionId::new(1),
            prefix_section_ids: Vec::new(),
            tokens: TokenBuffer::from(vec![1u32; tokens]),
            address: ContentAddress::default(),
            debug_name: "s".into(),
            in_collection: false,
            start: None,
            band,
            response_tx,
        }));
        (slot, response_rx)
    }

    /// **A section is offered at its own band.** Offered at `Low` whatever it
    /// carried, a dialogue's section stood behind every background one queued
    /// ahead of it; within a band the queue stays first come, first served.
    #[test]
    fn a_section_is_admitted_at_its_own_band() {
        let mut sched = admission_scheduler();
        ingest_section_at(&mut sched, 64, DecodePriority::Low);
        let (first, _rx) = ingest_section_at(&mut sched, 64, DecodePriority::High);
        let (second, _rx) = ingest_section_at(&mut sched, 64, DecodePriority::High);

        sched.promote_new_prefills();

        let order: Vec<SequenceId> = sched
            .active_section_ingests
            .iter()
            .map(|s| s.sequence_id)
            .collect();
        assert_eq!(&order[..2], &[first, second]);
    }

    /// **A live section joins a spent forward only while the pass can carry
    /// it.** Past that it would claim its store and advance by nothing.
    #[test]
    fn a_live_section_runs_ahead_while_the_pass_has_a_chunk_for_it() {
        let mut sched = admission_scheduler();
        assert!(sched.section_runs_ahead(DecodePriority::High, 149));

        let budget = sched.prefill_pass_budget();
        ingest_section_at(&mut sched, budget, DecodePriority::High);
        sched.promote_new_prefills();

        assert!(!sched.section_runs_ahead(DecodePriority::High, 149));
    }

    /// **A section waits for admission; it does not go straight into flight.**
    ///
    /// It used to: the request handler put it in flight as it drained, so no
    /// admission pass ever priced one, and a calibration's sections and cases
    /// filled the ground below the weight floor until no wave could be placed.
    #[test]
    fn an_ingested_section_waits_in_the_queue_until_admitted() {
        let mut sched = admission_scheduler();
        ingest_section(&mut sched, 64);
        assert_eq!(sched.section_queue.len(), 1, "queued on arrival");
        assert!(
            sched.active_section_ingests.is_empty(),
            "nothing in flight before admission",
        );
        assert!(sched.settled_since_admit, "an arrival opens admission");

        sched.promote_new_prefills();

        assert!(sched.section_queue.is_empty(), "admission took it");
        assert_eq!(sched.active_section_ingests.len(), 1, "and it is in flight");
        assert!(
            !sched.active_section_ingests[0].export_end,
            "a section given no start has no end state anyone asked for"
        );
    }

    /// **A section batch's device OOM is retried, any other error fails it.**
    /// The wave-group path retries an OOM's section members; the standalone
    /// batch carries the same sections and answers the same error the same way.
    #[test]
    fn a_section_batch_oom_is_retried_and_other_errors_fail_it() {
        let mut sched = admission_scheduler();
        ingest_section(&mut sched, 64);
        sched.promote_new_prefills();
        assert_eq!(sched.active_section_ingests.len(), 1);

        let oom = candle::Error::Msg(format!("{KV_DEVICE_OOM_MARKER}: no room"));
        sched.fail_section_batch(&[0], &[64], &oom);
        assert!(
            sched.active_section_ingests[0].error.is_none(),
            "an OOM failed the section instead of leaving it to retry"
        );

        let other = candle::Error::Msg("bad operand".into());
        sched.fail_section_batch(&[0], &[64], &other);
        assert!(sched.active_section_ingests[0].error.is_some());
    }

    /// **The width backstop bounds sections**, which it never did while they
    /// skipped admission: 88 were in flight on Qwen3.8-Flash-Next's calibration.
    /// A stack with no planner takes them up to the backstop and leaves the
    /// rest queued.
    #[test]
    fn sections_are_bounded_by_the_width_backstop() {
        let mut sched = admission_scheduler();
        for _ in 0..MAX_PREFILL_WIDTH + 3 {
            ingest_section(&mut sched, 64);
        }
        sched.promote_new_prefills();
        assert_eq!(sched.active_section_ingests.len(), MAX_PREFILL_WIDTH);
        assert_eq!(sched.section_queue.len(), 3);
    }

    /// An empty section has no forward to admit and seals on arrival.
    #[test]
    fn an_empty_section_is_never_queued() {
        let mut sched = admission_scheduler();
        ingest_section(&mut sched, 0);
        assert!(sched.section_queue.is_empty());
        assert!(sched.active_section_ingests.is_empty());
    }

    /// A queued section whose slot is freed is answered and dropped, not
    /// prepared later on whatever reuses the slot's id.
    #[test]
    fn a_freed_slot_drops_its_queued_section_with_an_answer() {
        let mut sched = admission_scheduler();
        let (slot, rx) = ingest_section(&mut sched, 64);
        sched.purge_freed_slot_scheduling_state(slot);
        assert!(sched.section_queue.is_empty());
        let answer = rx.try_recv().expect("the caller is answered");
        assert!(answer.is_err(), "with an error, not a seal");
    }

    /// **A retired section's queued ingest is withdrawn and answered**, on its
    /// own substrate only: the same section id queued for another substrate
    /// stays.
    #[test]
    fn a_retired_section_withdraws_its_queued_ingest() {
        let mut sched = admission_scheduler();
        let ours = Conversation::new();
        let theirs = Conversation::new();
        let (slot, rx) = ingest_section(&mut sched, 64);
        sched.slot_conversations.insert(slot, ours.clone());
        let (other, other_rx) = ingest_section(&mut sched, 64);
        sched.slot_conversations.insert(other, theirs.clone());

        assert!(sched.handle_request(SchedulerRequest::RetireSections {
            conversation: ours,
            sections: vec![SectionId::new(1)],
        }));

        let queued: Vec<SequenceId> = sched.section_queue.iter().map(|s| s.sequence_id).collect();
        assert_eq!(
            queued,
            vec![other],
            "only the retired substrate's ingest left"
        );
        let answer = rx.try_recv().expect("the caller is answered");
        assert!(answer.is_err(), "with an error, not a seal");
        assert!(other_rx.try_recv().is_err(), "the other is still waiting");
    }
}
