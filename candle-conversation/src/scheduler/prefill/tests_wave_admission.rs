//! Tests how a wave admits turns and sections: the boundaries they stop at, what
//! they wait behind, and what they share.
//!
//! A slot's chunk stops at each reserve stride, and a first reading stops one
//! token short so it is scored before anything is sampled. Admission closes while
//! a guest waits for room. A live turn that runs ahead of a full creep is admitted,
//! and one that would run behind it waits. Prefills share one pass budget and fill
//! it in order, a gang divides the group in proportion, and a prompt past the
//! model's RoPE reach fails alone rather than taking the forward with it.

use super::*;

#[cfg(test)]
pub(super) mod wave_chunk_tests {
    use std::sync::Arc;
    use std::time::Instant;

    use candle_transformers::models::rope_schedule::RungSelect;

    use super::super::super::admit::rate::{DecodeModel, ExpertGeometry, RateModel};
    use super::super::super::admit::{fill, Ground, Kind, WaveRate};
    use super::super::super::admit_ground::AdmitPass;
    use super::super::super::tests::{
        make_test_scheduler, make_test_scheduler_recurrent, ToyPages,
    };
    use super::super::super::*;

    /// A dialogue prefill carrying `tokens` and nothing else.
    pub(in super::super) fn dialogue_prefill(seq: SequenceId, tokens: Vec<u32>) -> ActivePrefill {
        let (event_tx, _event_rx) = flume::unbounded();
        ActivePrefill {
            work: PrefillWork {
                sequence_id: seq,
                tokens: TokenBuffer::from(tokens),
                prefill_text: String::new(),
                user_text: String::new(),
                tags: Vec::new(),
                user_content_start: 0,
                user_content_end: 0,
                assistant_content_start: 0,
                no_think: false,
                prefill_assistant_text: String::new(),
                event_tx,
                max_decode_tokens: 0,
                sampling: SamplingConfig::compression(),
                submitted_at: Instant::now(),
                reprojection: None,
                belief: PriorBelief::default(),
                seal_action: SealAction::None,
                post_decode_tokens: TokenBuffer::default(),
                projection_offsets: Vec::new(),
                staged_composition: None,
                triggers: Arc::new(TriggerRegistry::new()),
                turn_grammar: None,
                free_tool_calls_from_penalties: false,
                recorded_reply: None,
                recurrent_spans: Vec::new(),
                image_spans: Vec::new(),
                rereads: false,
            },
            offset: 0,
            next_projection: 0,
            finished: None,
            error: None,
            prefill_start: None,
            stops_every: None,
            awaiting: false,
        }
    }

    /// **An opening on a filled slot stops at every stride** (§7.4): a chunk
    /// advances no further than the next multiple of `reserve_tokens`, the
    /// opening stands at a boundary only on one with tokens still to come, and
    /// one with no stride runs to its end.
    #[test]
    fn an_opening_stops_at_every_stride() {
        let mut p = dialogue_prefill(SequenceId(1), vec![0; 40_000]);
        p.stops_every = Some(16_384);
        assert_eq!((p.advance_cap(), p.at_boundary()), (16_384, false));
        p.offset = 10_000;
        assert_eq!((p.advance_cap(), p.at_boundary()), (6_384, false));
        p.offset = 16_384;
        assert_eq!((p.advance_cap(), p.at_boundary()), (16_384, true));
        p.offset = 32_768;
        assert_eq!((p.advance_cap(), p.at_boundary()), (7_232, true));
        p.offset = 40_000;
        assert!(!p.at_boundary(), "the end of the opening is no boundary");
        p.stops_every = None;
        p.offset = 16_384;
        assert_eq!((p.advance_cap(), p.at_boundary()), (23_616, false));
    }

    /// **A first reading stops one token short of its end, and stands at a
    /// boundary there** (`first_reading`): 40 tokens advance 39 and no
    /// further, so the reading is scored before anything is sampled; short of
    /// that it is at no boundary.
    #[test]
    fn a_first_reading_stops_one_token_short() {
        let mut p = dialogue_prefill(SequenceId(1), vec![0; 40]);
        p.work.rereads = true;
        assert_eq!((p.advance_cap(), p.at_boundary()), (39, false));
        p.offset = 20;
        assert_eq!((p.advance_cap(), p.at_boundary()), (19, false));
        p.offset = 39;
        assert_eq!((p.advance_cap(), p.at_boundary()), (0, true));
    }

    /// **Only a prefill that leaves opens the next admission.** Promotion runs
    /// after every decode step; a prefill still advancing frees nothing, so it
    /// must not price a whole admission pass a step. One that leaves — here on
    /// an error — hands its slot back, and the pass is due.
    #[test]
    fn only_a_prefill_that_leaves_opens_the_next_admission() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let seq = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler
            .active_prefills
            .push(dialogue_prefill(seq, (0..64u32).collect()));
        scheduler.settled_since_admit = false;
        scheduler.promote_finished_prefills_to_decodes();
        assert_eq!(scheduler.active_prefills.len(), 1);
        assert!(
            !scheduler.settled_since_admit,
            "a prefill still advancing settled nothing"
        );

        scheduler.active_prefills[0].error = Some(ConversationError::Channel("lost".into()));
        scheduler.promote_finished_prefills_to_decodes();
        assert!(scheduler.active_prefills.is_empty());
        assert!(
            scheduler.settled_since_admit,
            "a prefill that left is an admission opportunity"
        );
    }

    /// **A guest waiting for room closes admission, the forced head included.**
    /// With nothing in flight and admission unsettled, the queue head is
    /// normally forced in so the engine makes progress; while a guest waits,
    /// that turn would take the ground the guest is waiting for. Once the wait
    /// clears, the head is forced in as before.
    #[test]
    fn a_guest_waiting_for_room_closes_admission_and_the_forced_head() {
        use crate::guest::work::{GuestRequest, MatteRequest};
        use crate::guest::Guest;
        use std::time::Duration;

        let (mut scheduler, _tx) = make_test_scheduler();
        let seq = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler
            .prefill_queue
            .push_back(dialogue_prefill(seq, (0..64u32).collect()).work);
        scheduler.settled_since_admit = false;

        let queue = Arc::clone(&scheduler.guests.queue);
        let _receipt = queue
            .submit(GuestRequest::Matte(MatteRequest {
                pixels: vec![0; 4 * 4 * 3],
                width: 4,
                height: 4,
            }))
            .unwrap();
        queue.requeue(queue.take(Guest::Matte), Duration::from_secs(60));

        scheduler.promote_new_prefills();
        assert_eq!(
            scheduler.prefill_queue.len(),
            1,
            "a turn was admitted while a guest waited for its ground"
        );
        assert!(scheduler.active_prefills.is_empty());

        // An urgent turn is held too: unjudged, it is work past the rate model
        // on a card already short of ground.
        let live = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler.slot_bands.insert(live, DecodePriority::Urgent);
        scheduler
            .prefill_queue
            .push_back(dialogue_prefill(live, (0..16u32).collect()).work);
        scheduler.promote_new_prefills();
        assert!(
            scheduler.active_prefills.is_empty(),
            "an urgent turn went in unjudged while a guest waited"
        );
        assert_eq!(scheduler.prefill_queue.len(), 2);

        let _ = queue.take(Guest::Matte);
        scheduler.promote_new_prefills();
        assert_eq!(
            scheduler.prefill_queue.len(),
            1,
            "with the wait cleared, the idle engine did not force its queue head"
        );
    }

    /// An idle slot's working set is not live: it is neither pinned nor kept
    /// from relief. A turn queued on the slot makes it live again.
    #[test]
    fn only_a_slot_with_a_turn_on_it_keeps_its_working_set_live() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let slot = SequenceId(scheduler.session.create_sequence().expect("create"));
        let key = TurnKey::new(
            crate::projection::TimelineId::for_test(999),
            crate::projection::TurnIndex(0),
        );
        scheduler
            .slot_projection_state
            .entry(slot)
            .or_default()
            .working_set
            .turns
            .push(key);

        let (_, turns) = scheduler.live_working_sets();
        assert!(turns.is_empty(), "an idle slot's working set was live");

        scheduler
            .prefill_queue
            .push_back(dialogue_prefill(slot, (0..16u32).collect()).work);
        let (_, turns) = scheduler.live_working_sets();
        assert_eq!(turns, vec![key]);
    }

    /// An urgent prompt queued behind a high prefill outranks it, and so is not
    /// held by the pressure gate; another high one beside it is.
    #[test]
    fn only_a_higher_band_outranks_the_work_in_flight() {
        let (mut scheduler, _tx) = make_test_scheduler();
        assert!(!scheduler.a_queued_prompt_outranks_the_work_in_flight());

        let running = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler.slot_bands.insert(running, DecodePriority::High);
        scheduler
            .active_prefills
            .push(dialogue_prefill(running, (0..4_000u32).collect()));

        let same = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler.slot_bands.insert(same, DecodePriority::High);
        scheduler
            .prefill_queue
            .push_back(dialogue_prefill(same, (0..64u32).collect()).work);
        assert!(
            !scheduler.a_queued_prompt_outranks_the_work_in_flight(),
            "a prompt of the running band is more of the same"
        );

        let urgent = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler.slot_bands.insert(urgent, DecodePriority::Urgent);
        scheduler
            .prefill_queue
            .push_back(dialogue_prefill(urgent, (0..16u32).collect()).work);
        assert!(scheduler.a_queued_prompt_outranks_the_work_in_flight());
    }

    /// **A dialogue prefill enters the wave group in pass-sized chunks.** It
    /// used to enter whole: a 15.2k-token Cline turn needed a 13.2 GiB transient
    /// tier on a 16 GB card, more than all the ground below the weight floor,
    /// and failed identically on every retry.
    #[test]
    fn a_long_dialogue_prefill_creeps_in_pass_sized_chunks() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let cap = scheduler.max_prefill_pass_tokens;
        let seq = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler
            .active_prefills
            .push(dialogue_prefill(seq, (0..15_000u32).collect()));

        scheduler.form_wave_group(false);
        let (members, _, inputs, _) = scheduler.build_wave_group_inputs();
        assert_eq!(members.len(), 1);
        assert_eq!(
            inputs[0].dims(),
            &[1, cap],
            "the first chunk is one pass wide"
        );

        // A later group resumes at the committed offset, never from token 0.
        scheduler.reset_wave_prefill();
        scheduler.active_prefills[0].offset = 14_800;
        scheduler.form_wave_group(false);
        let (_, _, inputs, _) = scheduler.build_wave_group_inputs();
        let rows = inputs[0].to_vec2::<u32>().expect("u32 rows");
        assert_eq!(rows[0].len(), 200, "the tail chunk is what remains");
        assert_eq!(rows[0][0], 14_800, "the chunk starts at the offset");
    }

    /// **A creeping turn closes a page at the edge between its records.** The
    /// npcd turn whose first page measured 109 against an edge of 91: a user
    /// record of 91 tokens ending on its `<|im_end|>`, then the assistant's
    /// 18-token opener. The group's first chunk ends at the edge, the page closes
    /// when that sweep commits, and the opener rides the next chunk, so the user
    /// record can be placed alone (`docs/hit_driven_projection.md` §8.1).
    #[test]
    fn a_creeping_turn_closes_a_page_at_its_user_records_edge() {
        const IM_END: u32 = 900;
        let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
        scheduler.page_break_tokens = vec![IM_END];
        let seq = SequenceId(scheduler.session.create_sequence().expect("create"));
        let mut tokens = vec![1u32; 90];
        tokens.push(IM_END);
        tokens.extend([2u32; 18]);
        scheduler
            .active_prefills
            .push(dialogue_prefill(seq, tokens));

        scheduler.form_wave_group(false);
        let (members, _, inputs, _) = scheduler.build_wave_group_inputs();
        assert_eq!(inputs[0].dims(), &[1, 91], "the chunk ends at the edge");
        // The sweep's forward pools the chunk's rows into the open page.
        probe.advance_index(seq.0, 91);
        scheduler.complete_wave_group(&members, &[]);
        assert_eq!(
            probe.get_pages(seq.0),
            ToyPages {
                closed: vec![91],
                tail: 0
            },
            "the first page is the user record, 91 wide"
        );

        scheduler.form_wave_group(false);
        let (_, _, inputs, _) = scheduler.build_wave_group_inputs();
        assert_eq!(
            inputs[0].dims(),
            &[1, 18],
            "the opener rides the next chunk"
        );
    }

    /// **An opening on a filled slot creeps to its next boundary and waits
    /// there** (§7.4): a chunk ends on the stride, and an opening waiting for
    /// its reprojection rides no group until the reprojection hands it back.
    #[test]
    fn an_opening_creeps_to_its_boundary_and_waits_there() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let cap = scheduler.max_prefill_pass_tokens;
        let seq = SequenceId(scheduler.session.create_sequence().expect("create"));
        let mut p = dialogue_prefill(seq, (0..15_000u32).collect());
        p.stops_every = Some(cap + 100);
        p.offset = cap;
        scheduler.active_prefills.push(p);

        scheduler.form_wave_group(false);
        let (_, _, inputs, _) = scheduler.build_wave_group_inputs();
        assert_eq!(inputs[0].dims(), &[1, 100], "the chunk ends on the stride");

        scheduler.reset_wave_prefill();
        scheduler.active_prefills[0].offset = cap + 100;
        assert!(scheduler.active_prefills[0].at_boundary());
        scheduler.active_prefills[0].awaiting = true;
        scheduler.form_wave_group(false);
        assert!(
            scheduler.wave_prefill_members.is_empty(),
            "an opening waiting at its boundary rides no group"
        );
        assert!(scheduler.awaiting_opening(seq).is_some());

        scheduler.resume_unreprojected(seq);
        scheduler.form_wave_group(false);
        assert_eq!(
            scheduler.wave_prefill_members.len(),
            1,
            "handed back, it rides"
        );
    }

    /// A queued turn of `len` tokens on a new slot at `band`.
    fn queue_turn(s: &mut Scheduler, band: DecodePriority, len: usize) -> SequenceId {
        let seq = SequenceId(s.session.create_sequence().expect("create"));
        s.slot_bands.insert(seq, band);
        s.prefill_queue
            .push_back(dialogue_prefill(seq, vec![1; len]).work);
        seq
    }

    /// A `band` creep `len` tokens long that fills the forward, held mid-sweep.
    fn creep_mid_sweep(s: &mut Scheduler, band: DecodePriority, len: usize) {
        let seq = SequenceId(s.session.create_sequence().expect("create"));
        s.slot_bands.insert(seq, band);
        s.active_prefills.push(dialogue_prefill(seq, vec![1; len]));
        s.form_wave_group(false);
        assert_eq!(s.wave_prefill_members.len(), 1, "the creep is held");
        s.wave_prefill_cursor = 5;
    }

    /// Whether admission offers the next queued turn of `band` at all.
    fn offered(s: &mut Scheduler, band: DecodePriority) -> bool {
        AdmitPass::new(s).peek(Kind::Prefill, band).is_some()
    }

    /// One admission pass through the rate model, as the scheduler runs it:
    /// the standing cohort charged first, then each band offered in turn.
    fn admission_pass(s: &mut Scheduler) {
        s.settled_since_admit = true;
        let mut rate = WaveRate::with_link_rate(
            25e9,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel::default(),
        )
        .with_min_gain(0.0);
        let mut admission = AdmitPass::new(s);
        fill(&mut admission, &mut rate);
        admission.conclude();
    }

    /// Whether `seq`'s turn is in flight.
    fn admitted(s: &Scheduler, seq: SequenceId) -> bool {
        s.active_prefills.iter().any(|p| p.work.sequence_id == seq)
    }

    /// **Work owes the decodes it outranks nothing.** A person's prompt beside
    /// background decodes repays none of them; beside a decode of its own band
    /// or above it does, and work of equal band still shares.
    #[test]
    fn work_owes_only_the_decodes_it_does_not_outrank() {
        use super::owes_decode;
        use DecodePriority::{High, Low, Normal, Urgent};
        assert!(!owes_decode(Urgent, [Low, Low, Normal, High]));
        assert!(!owes_decode(High, [Low, Normal]));
        assert!(owes_decode(Urgent, [Low, Urgent]));
        assert!(owes_decode(Low, [Low]));
        assert!(owes_decode(Normal, [Low, High]));
        assert!(!owes_decode(Low, []), "nothing decodes, nothing is owed");
    }

    /// **A person's prompt holds the background from the queue on.** Tool
    /// results are that prompt between two decodes; a `High` one is the cast,
    /// which the hold must never key on (`urgent_hold`).
    #[test]
    fn only_an_urgent_prompt_counts_as_prefilling_for_the_hold() {
        let (mut s, _tx) = make_test_scheduler();
        assert!(!s.an_urgent_prompt_prefills());
        queue_turn(&mut s, DecodePriority::High, 8);
        assert!(!s.an_urgent_prompt_prefills(), "the cast does not hold it");
        queue_turn(&mut s, DecodePriority::Urgent, 8);
        assert!(s.an_urgent_prompt_prefills(), "queued");

        s.prefill_queue.clear();
        assert!(!s.an_urgent_prompt_prefills());
        let seq = SequenceId(s.session.create_sequence().expect("create"));
        s.slot_bands.insert(seq, DecodePriority::Urgent);
        s.active_prefills.push(dialogue_prefill(seq, vec![1; 8]));
        assert!(s.an_urgent_prompt_prefills(), "in flight");

        s.active_prefills[0].offset = 8;
        assert!(
            !s.an_urgent_prompt_prefills(),
            "written: nothing to hold for"
        );
    }

    /// **A live turn that runs ahead of a full creep is admitted.** On
    /// WhiteWorld (2026-10-27 22:46–22:48) one `High` member 158k tokens long
    /// held all of every forward's rows, and nothing else was offered: four
    /// `Urgent` content checks waited ninety seconds and an 87-token `High`
    /// turn six minutes. Judged end to end — the creep charged first, as
    /// `admit::fill` charges the standing cohort — the short `High` turn joins,
    /// to lead the group re-formed at the creep's boundary, and the `Urgent`
    /// check joins, to pause the creep and run in a group of its own.
    #[test]
    fn a_live_turn_ahead_of_a_full_creep_is_admitted() {
        let (mut s, _tx) = make_test_scheduler();
        let pass = s.prefill_pass_ceiling();
        creep_mid_sweep(&mut s, DecodePriority::High, 8 * pass);
        let turn = queue_turn(&mut s, DecodePriority::High, 87);
        admission_pass(&mut s);
        assert!(admitted(&s, turn), "the short high turn was not admitted");

        let check = queue_turn(&mut s, DecodePriority::Urgent, 64);
        admission_pass(&mut s);
        assert!(admitted(&s, check), "the urgent check was not admitted");
        assert!(
            s.creep_outranked_by_a_waiting_prompt(),
            "the check pauses the creep at the next step"
        );
    }

    /// **The turn admitted ahead takes the rows it was admitted for.** At the
    /// creep's sweep boundary the group re-forms nearest-first, and the short
    /// `High` turn leads it with all of its rows.
    #[test]
    fn a_turn_admitted_ahead_leads_the_reformed_group() {
        let (mut s, _tx) = make_test_scheduler();
        let pass = s.prefill_pass_ceiling();
        creep_mid_sweep(&mut s, DecodePriority::High, 8 * pass);
        let turn = queue_turn(&mut s, DecodePriority::High, 87);
        admission_pass(&mut s);
        assert!(admitted(&s, turn));

        s.reset_wave_prefill();
        s.form_wave_group(false);
        assert!(
            matches!(
                s.wave_prefill_members.first(),
                Some(&WaveMember::Prefill { seq_id, advance: 87 }) if seq_id == turn.0
            ),
            "the turn admitted ahead leads the group: {:?}",
            s.wave_prefill_members
        );
    }

    /// **Every content check of a draw is admitted, not one a sweep.** Four
    /// `Urgent` checks beside a full `High` creep each take rows in the group
    /// they form apart from it, which has room for all four.
    #[test]
    fn every_urgent_check_beside_a_full_creep_is_admitted() {
        let (mut s, _tx) = make_test_scheduler();
        let pass = s.prefill_pass_ceiling();
        creep_mid_sweep(&mut s, DecodePriority::High, 8 * pass);
        let checks: Vec<SequenceId> = (0..4)
            .map(|_| queue_turn(&mut s, DecodePriority::Urgent, 64))
            .collect();
        admission_pass(&mut s);
        for check in checks {
            assert!(admitted(&s, check), "a check was held back");
        }
    }

    /// **Turns that would take no rows are not admitted to wait**, however
    /// many are nearer their first token than the creep they would lead: once
    /// the ones ahead of a turn fill the budget, it is held back.
    #[test]
    fn turns_ahead_of_a_creep_are_admitted_only_while_they_take_rows() {
        let (mut s, _tx) = make_test_scheduler();
        let pass = s.prefill_pass_ceiling();
        creep_mid_sweep(&mut s, DecodePriority::High, 8 * pass);
        let turns: Vec<SequenceId> = (0..3)
            .map(|_| queue_turn(&mut s, DecodePriority::High, pass - 100))
            .collect();
        admission_pass(&mut s);
        let taken: Vec<bool> = turns.iter().map(|&t| admitted(&s, t)).collect();
        assert_eq!(
            taken,
            vec![true, true, false],
            "the first takes most of a pass and the second the rest"
        );
    }

    /// **A live turn that would run behind every member still waits.** A
    /// `High` turn longer than the creep it would join takes no rows until
    /// that creep finishes, so it is not admitted to hold its store meanwhile.
    #[test]
    fn a_live_turn_behind_a_full_creep_waits() {
        let (mut s, _tx) = make_test_scheduler();
        let pass = s.prefill_pass_ceiling();
        creep_mid_sweep(&mut s, DecodePriority::High, 2 * pass);
        queue_turn(&mut s, DecodePriority::High, 16 * pass);
        assert!(!offered(&mut s, DecodePriority::High));
    }

    /// **Work below a live turn's band does not make it ahead.** Its group
    /// leaves that work out, so what decides is its own band: behind two
    /// `High` prompts nearer their first tokens, filling the forward, a long
    /// `High` turn would take no rows — and a `Low` prompt in flight beside
    /// them changes nothing.
    #[test]
    fn a_live_turn_behind_its_own_band_waits_whatever_runs_below() {
        let (mut s, _tx) = make_test_scheduler();
        let pass = s.prefill_pass_ceiling();
        for (band, len) in [
            (DecodePriority::High, pass),
            (DecodePriority::High, pass),
            (DecodePriority::Low, pass),
        ] {
            let seq = SequenceId(s.session.create_sequence().expect("create"));
            s.slot_bands.insert(seq, band);
            s.active_prefills.push(dialogue_prefill(seq, vec![1; len]));
        }
        queue_turn(&mut s, DecodePriority::High, 8 * pass);
        assert!(!offered(&mut s, DecodePriority::High));
    }

    /// **Below the live bands a full forward takes no more turns**, as it
    /// always has: there throughput matters more than latency.
    #[test]
    fn a_background_turn_beside_a_full_creep_waits() {
        let (mut s, _tx) = make_test_scheduler();
        let pass = s.prefill_pass_ceiling();
        creep_mid_sweep(&mut s, DecodePriority::Low, 8 * pass);
        queue_turn(&mut s, DecodePriority::Low, 64);
        assert!(!offered(&mut s, DecodePriority::Low));
    }

    /// **Prefills share one pass budget.** The two short prompts ride whole and the
    /// one a whole pass long takes what is left, so all three advance together in
    /// a forward of exactly one pass — rather than the long one waiting out a
    /// forward it could have shared.
    #[test]
    fn dialogue_prefills_share_one_pass_budget() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let cap = scheduler.max_prefill_pass_tokens;
        let a = SequenceId(scheduler.session.create_sequence().expect("create"));
        let b = SequenceId(scheduler.session.create_sequence().expect("create"));
        let c = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler
            .active_prefills
            .push(dialogue_prefill(a, vec![1; 100]));
        scheduler
            .active_prefills
            .push(dialogue_prefill(b, vec![1; 100]));
        scheduler
            .active_prefills
            .push(dialogue_prefill(c, vec![1; cap]));

        scheduler.form_wave_group(false);
        let budget = scheduler.prefill_pass_budget();
        let rode: Vec<(usize, usize)> = scheduler
            .wave_prefill_members
            .iter()
            .map(|m| match m {
                WaveMember::Prefill { seq_id, advance } => (*seq_id, *advance),
                WaveMember::Section { .. } => unreachable!("no sections were offered"),
            })
            .collect();
        assert_eq!(
            rode,
            vec![(a.0, 100), (b.0, 100), (c.0, budget - 200)],
            "all three ride, sharing exactly one pass"
        );
    }

    /// **A prompt past the model's RoPE reach fails alone.** Its headers would
    /// be refused and take the whole forward with them; instead it is errored
    /// before the group forms, and a prompt that fits rides as usual.
    #[test]
    fn a_prompt_past_the_rope_reach_fails_alone() {
        let (mut scheduler, _tx) = make_test_scheduler();
        scheduler
            .session
            .set_rope_select(RungSelect::new(vec![64]).expect("ceilings"));
        let fits = SequenceId(scheduler.session.create_sequence().expect("create"));
        let over = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler
            .active_prefills
            .push(dialogue_prefill(fits, vec![1; 64]));
        scheduler
            .active_prefills
            .push(dialogue_prefill(over, vec![1; 65]));

        scheduler.form_wave_group(false);
        let seqs: Vec<usize> = scheduler
            .wave_prefill_members
            .iter()
            .map(|m| m.seq_id())
            .collect();
        assert_eq!(
            seqs,
            vec![fits.0],
            "only the prompt that fits joins the group"
        );
        assert!(scheduler.active_prefills[0].error.is_none());
        assert!(
            matches!(
                &scheduler.active_prefills[1].error,
                Some(ConversationError::Other(m)) if m.contains("65") && m.contains("64")
            ),
            "the prompt past the reach carries its own error"
        );
    }

    /// The `(seq_id, advance)` of each prefill the group formed with.
    fn rode(scheduler: &Scheduler) -> Vec<(usize, usize)> {
        scheduler
            .wave_prefill_members
            .iter()
            .map(|m| match m {
                WaveMember::Prefill { seq_id, advance } => (*seq_id, *advance),
                WaveMember::Section { .. } => unreachable!("no sections were offered"),
            })
            .collect()
    }

    /// **A background gang rides between the shorts and the other longs**
    /// (`docs/prefill_gangs.md`): the short `Low` prompt first, the gang's two
    /// pages next sharing in proportion, and the long prompt that arrived
    /// first but is outside the gang behind them — the group divided as
    /// `gang::gang_shares` divides it.
    #[test]
    fn a_background_gang_rides_between_the_shorts_and_the_other_longs() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let cap = scheduler.max_prefill_pass_tokens;
        let outside = SequenceId(scheduler.session.create_sequence().expect("create"));
        let g1 = SequenceId(scheduler.session.create_sequence().expect("create"));
        let g2 = SequenceId(scheduler.session.create_sequence().expect("create"));
        let short = SequenceId(scheduler.session.create_sequence().expect("create"));
        for (seq, len) in [
            (outside, 4 * cap),
            (g1, 4 * cap),
            (g2, 4 * cap),
            (short, 100),
        ] {
            scheduler.slot_bands.insert(seq, DecodePriority::Low);
            scheduler
                .active_prefills
                .push(dialogue_prefill(seq, vec![1; len]));
        }
        scheduler.prefill_gang = vec![g1.0, g2.0];

        scheduler.form_wave_group(false);

        let budget = scheduler.prefill_group_budget(100 + 3 * cap);
        let shares = gang::gang_shares(
            &[100, cap, cap, cap],
            &[false, true, true, false],
            budget,
            CHUNK_SIZE,
        );
        let expected: Vec<(usize, usize)> = [short.0, g1.0, g2.0, outside.0]
            .into_iter()
            .zip(shares)
            .filter(|&(_, s)| s > 0)
            .collect();
        assert_eq!(rode(&scheduler), expected);
        assert_eq!(expected[1].1, expected[2].1, "the gang advances in step");
        assert_eq!(scheduler.prefill_gang, vec![g1.0, g2.0], "the gang holds");
    }

    /// **A live prompt leads, and the gang waits intact**: a `High` prompt's
    /// group carries `High` prompts only, filled in order, and the background
    /// gang is still the gang when the background resumes.
    #[test]
    fn a_live_prompt_leads_and_the_gang_waits_intact() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let cap = scheduler.max_prefill_pass_tokens;
        let g1 = SequenceId(scheduler.session.create_sequence().expect("create"));
        let g2 = SequenceId(scheduler.session.create_sequence().expect("create"));
        let live = SequenceId(scheduler.session.create_sequence().expect("create"));
        for seq in [g1, g2] {
            scheduler.slot_bands.insert(seq, DecodePriority::Low);
            scheduler
                .active_prefills
                .push(dialogue_prefill(seq, vec![1; 4 * cap]));
        }
        scheduler.slot_bands.insert(live, DecodePriority::High);
        scheduler
            .active_prefills
            .push(dialogue_prefill(live, vec![1; 64]));
        scheduler.prefill_gang = vec![g1.0, g2.0];

        scheduler.form_wave_group(false);

        assert_eq!(rode(&scheduler), vec![(live.0, 64)]);
        assert_eq!(scheduler.prefill_gang, vec![g1.0, g2.0]);
    }

    /// **A gang dissolves when its prefill is done**, not its decode — and with
    /// the decode model unlearned, no new one forms: the group is filled in
    /// order.
    #[test]
    fn a_gang_dissolves_when_its_prefill_is_done() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let cap = scheduler.max_prefill_pass_tokens;
        let g1 = SequenceId(scheduler.session.create_sequence().expect("create"));
        let g2 = SequenceId(scheduler.session.create_sequence().expect("create"));
        let next = SequenceId(scheduler.session.create_sequence().expect("create"));
        for seq in [g1, g2, next] {
            scheduler.slot_bands.insert(seq, DecodePriority::Low);
            scheduler
                .active_prefills
                .push(dialogue_prefill(seq, vec![1; 4 * cap]));
        }
        scheduler.active_prefills[0].offset = 4 * cap;
        scheduler.active_prefills[1].offset = 4 * cap;
        scheduler.prefill_gang = vec![g1.0, g2.0];

        scheduler.form_wave_group(false);

        assert!(
            scheduler.prefill_gang.is_empty(),
            "no gang outlives its prefill"
        );
        assert_eq!(rode(&scheduler)[0].0, next.0);
    }
}

#[cfg(test)]
pub(super) mod turn_view_release_tests {
    use super::super::super::tests::{make_test_scheduler, register_turn_view};
    use super::super::super::*;
    use super::wave_chunk_tests::dialogue_prefill;

    /// **An errored turn prefill releases its view.** The error reached the
    /// caller and the view stayed registered with the parent's prefix borrowed
    /// — nothing but a completed decode or a reprojection ever released it, and
    /// an errored turn has neither.
    #[test]
    fn an_errored_turn_prefill_releases_its_view() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let parent = SequenceId(scheduler.session.create_sequence().expect("create"));
        let view = register_turn_view(&mut scheduler, parent);
        let mut prefill = dialogue_prefill(view, vec![1; 8]);
        prefill.error = Some(ConversationError::Channel("the wave failed".into()));
        scheduler.active_prefills.push(prefill);

        scheduler.promote_finished_prefills_to_decodes();

        assert!(scheduler.turn_views.is_empty(), "the view is unregistered");
        assert!(
            scheduler.session.sequence_offset(view.0).is_none(),
            "the view's slot is released"
        );
        assert!(
            scheduler.session.sequence_offset(parent.0).is_some(),
            "the parent is untouched"
        );
    }
}

#[cfg(test)]
pub(super) mod wave_give_up_tests {
    use super::super::super::tests::make_test_scheduler;
    use super::super::super::*;
    use super::wave_chunk_tests::dialogue_prefill;

    /// **A wave given up on fails what it carries, and nothing else.** A
    /// prefill still owing tokens hears the error; one that finished, or that
    /// already failed for its own reason, keeps what it had; and the held group
    /// is dropped so the next wave forms afresh.
    #[test]
    fn giving_up_fails_only_the_prefills_still_owing_tokens() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let owing = SequenceId(scheduler.session.create_sequence().expect("create"));
        let done = SequenceId(scheduler.session.create_sequence().expect("create"));
        let failed = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler
            .active_prefills
            .push(dialogue_prefill(owing, vec![1; 64]));
        let mut finished = dialogue_prefill(done, vec![1; 8]);
        finished.offset = 8;
        scheduler.active_prefills.push(finished);
        let mut own = dialogue_prefill(failed, vec![1; 8]);
        own.error = Some(ConversationError::Other("its own".into()));
        scheduler.active_prefills.push(own);
        scheduler.form_wave_group(false);
        assert!(!scheduler.wave_prefill_members.is_empty());

        scheduler.fail_decode_less_wave("no ground");

        assert!(matches!(
            &scheduler.active_prefills[0].error,
            Some(ConversationError::Other(m)) if m == "no ground"
        ));
        assert!(
            scheduler.active_prefills[1].error.is_none(),
            "finished keeps no error"
        );
        assert!(matches!(
            &scheduler.active_prefills[2].error,
            Some(ConversationError::Other(m)) if m == "its own"
        ));
        assert!(
            scheduler.wave_prefill_members.is_empty(),
            "the group is dropped"
        );
    }
}
