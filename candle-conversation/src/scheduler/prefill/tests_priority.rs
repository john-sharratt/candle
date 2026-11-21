//! Tests that work is grouped, paced and admitted by the band it runs at, so a
//! person's prompt is not held behind background work.
//!
//! A dialogue's prompt forms its own group, and a `High` group carries only
//! `High` prompts, so its first token does not wait on ingest rows. A second
//! conversation's prompt creeps beside the first's decode at the shared ratio,
//! while ingest takes the dialogue's own ratio. A waiting prompt outranks an
//! ingest creep even when that creep is not throttled. A summary probe decodes
//! at `Normal`, and a decode whose priority does not resolve counts as `High`.

use super::*;

#[cfg(test)]
pub(super) mod priority_admission_tests {
    use std::sync::Arc;

    use super::super::super::tests::{boundary_state, make_test_scheduler};
    use super::super::super::*;
    use super::wave_chunk_tests::dialogue_prefill;
    use super::{throttle_ratio, HIGH_BESIDE_HIGH_RATIO};
    use crate::projection::{Conversation, DecodePriority};
    use crate::tenant_rank::TenantRank;

    const YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
layers:
  - name: dialogue
    window: 8000
    decode_priority: high
    summary:
      turns:
        max_tokens: 256
        user: { system_prompt: s, user_prompt: u }
        assistant: { system_prompt: s, user_prompt: u }
    score_formula: max
    budget: { priority: 40 }
    groups:
      - id: chat
        selection: { kind: top_k, k: 2 }
  - name: ingest
    window: 8000
    decode_priority: low
    summary:
      turns:
        max_tokens: 256
        user: { system_prompt: s, user_prompt: u }
        assistant: { system_prompt: s, user_prompt: u }
    score_formula: max
    budget: { priority: 10 }
    groups:
      - id: files
        selection: { kind: top_k, k: 2 }
"#;

    /// Bind `seq` to `layer`'s group on its own timeline, so its priority
    /// resolves through the schema the way a live slot's does.
    fn bind(sched: &mut Scheduler, builder: &Arc<Builder>, seq: SequenceId, layer: &str, tl: u64) {
        let group = if layer == "dialogue" { "chat" } else { "files" };
        let timeline = TimelineId::from_raw(tl).expect("timeline id");
        sched.slot_targets.insert(
            seq,
            ProjectionTarget {
                layer: builder.id_for_layer(layer).unwrap(),
                group: builder.id_for_group(group).unwrap(),
                timeline,
            },
        );
        sched
            .timeline_projections
            .insert(timeline, Arc::clone(builder));
    }

    /// **A dialogue prompt's group is its own.** An ingest prefill queued
    /// first waits for the next group: a `High` group carries `High` prompts
    /// only, so the dialogue's first token does not wait on the ingest's rows.
    /// The ingest still advances — it is not held out of the device, only out
    /// of this group.
    #[test]
    fn a_dialogue_prompts_group_is_its_own() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let dialogue = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, ingest, "ingest", 1);
        bind(&mut sched, &builder, dialogue, "dialogue", 2);
        sched
            .active_prefills
            .push(dialogue_prefill(ingest, vec![1; 8]));
        sched
            .active_prefills
            .push(dialogue_prefill(dialogue, vec![1; 8]));

        sched.form_wave_group(false);

        let members = sched.wave_prefill_members.clone();
        let seqs: Vec<usize> = members.iter().map(|m| m.seq_id()).collect();
        assert_eq!(seqs, vec![dialogue.0], "the dialogue alone");
        assert_eq!(sched.creep_priority(&members), DecodePriority::High);
        assert_eq!(sched.running_prefills(), 2, "both can advance");
    }

    /// **The forced head is the highest band's.** With admission closed and
    /// nothing in flight, a dialogue turn queued behind ingest is the one the
    /// keep-one-alive rule forces in.
    #[test]
    fn the_forced_head_is_the_highest_bands_first_turn() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let dialogue = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, ingest, "ingest", 1);
        bind(&mut sched, &builder, dialogue, "dialogue", 2);
        sched
            .prefill_queue
            .push_back(dialogue_prefill(ingest, vec![1; 8]).work);
        sched
            .prefill_queue
            .push_back(dialogue_prefill(dialogue, vec![1; 8]).work);

        let head = sched.pop_highest_band_head().expect("a head");

        assert_eq!(head.sequence_id, dialogue);
        assert_eq!(
            sched.prefill_queue.len(),
            1,
            "the ingest turn keeps its place"
        );
        assert_eq!(sched.prefill_queue[0].sequence_id, ingest);
    }

    /// **A second conversation's prompt creeps beside the first's decode** at
    /// `HIGH_BESIDE_HIGH_RATIO`, while an ingest prompt takes the dialogue's own
    /// ratio — and both run every layer in one wave again once that decode is
    /// gone.
    #[test]
    fn a_dialogue_prompt_creeps_beside_a_dialogue_decode_until_it_clears() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let decoding = SequenceId(sched.session.create_sequence().expect("create"));
        let second = SequenceId(sched.session.create_sequence().expect("create"));
        let ingest = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, decoding, "dialogue", 1);
        bind(&mut sched, &builder, second, "dialogue", 2);
        bind(&mut sched, &builder, ingest, "ingest", 3);
        let (state, _rx) = boundary_state();
        sched.active_decodes.insert(decoding, state);

        let creep = [WaveMember::Prefill {
            seq_id: second.0,
            advance: 26,
        }];
        let work = sched.creep_priority(&creep);
        assert_eq!(work, DecodePriority::High);
        assert_eq!(sched.airtime_ratio_over(work), HIGH_BESIDE_HIGH_RATIO);

        let ingest_creep = [WaveMember::Prefill {
            seq_id: ingest.0,
            advance: 26,
        }];
        let ingest_work = sched.creep_priority(&ingest_creep);
        assert_eq!(ingest_work, DecodePriority::Low);
        assert_eq!(sched.airtime_ratio_over(ingest_work), 64);

        sched.active_decodes.remove(&decoding);
        assert_eq!(sched.airtime_ratio_over(work), 1, "the decode cleared");
        assert_eq!(sched.airtime_ratio_over(ingest_work), 1);
    }

    /// **Tenants are equal; the band decides.** npcd's live conversation
    /// throttles zend's ingest exactly as zend's own would — its creep yields
    /// at the dialogue's ratio — and zend's ingest never holds npcd's
    /// conversation back. Two tenants' ingests run side by side, and keep
    /// running beside the conversation.
    #[test]
    fn a_second_tenants_dialogue_outranks_a_first_tenants_ingest() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let zend = Conversation::new();
        let npcd = || Conversation::new().with_tenant_rank(TenantRank::nth(1).unwrap());
        let zend_ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let npcd_dialogue = SequenceId(sched.session.create_sequence().expect("create"));
        let npcd_ingest = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, zend_ingest, "ingest", 1);
        bind(&mut sched, &builder, npcd_dialogue, "dialogue", 2);
        bind(&mut sched, &builder, npcd_ingest, "ingest", 3);
        sched.slot_conversations.insert(zend_ingest, zend);
        sched.slot_conversations.insert(npcd_dialogue, npcd());
        sched.slot_conversations.insert(npcd_ingest, npcd());

        // Both tenants ingesting, nobody talking: both creep at full speed.
        sched
            .active_prefills
            .push(dialogue_prefill(zend_ingest, vec![1; 8]));
        sched
            .active_prefills
            .push(dialogue_prefill(npcd_ingest, vec![1; 8]));
        assert_eq!(sched.airtime_ratio_over(DecodePriority::Low), 1);

        // npcd's conversation decodes: zend's ingest yields to it at the
        // dialogue's ratio, and both ingests still advance beside it.
        let (state, _rx) = boundary_state();
        sched.active_decodes.insert(npcd_dialogue, state);
        let creep = [WaveMember::Prefill {
            seq_id: zend_ingest.0,
            advance: 26,
        }];
        let work = sched.creep_priority(&creep);
        assert_eq!(work, DecodePriority::Low);
        assert_eq!(sched.airtime_ratio_over(work), 64);
        assert_eq!(sched.running_prefills(), 2, "neither ingest is held out");
        sched.form_wave_group(false);
        let seqs: Vec<usize> = sched
            .wave_prefill_members
            .iter()
            .map(|m| m.seq_id())
            .collect();
        assert_eq!(seqs, vec![zend_ingest.0, npcd_ingest.0]);
    }

    /// **A dialogue prompt waiting behind an ingest creep outranks it, throttled
    /// or not.** Every creep step is bounded in time, so an unthrottled group
    /// still takes many steps to reach its boundary. A waiting prompt that does
    /// not outrank it leaves it at its pace.
    #[test]
    fn a_waiting_dialogue_prompt_outranks_an_ingest_creep() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let cast = SequenceId(sched.session.create_sequence().expect("create"));
        let ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let other_ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let prompt = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, cast, "dialogue", 1);
        bind(&mut sched, &builder, ingest, "ingest", 2);
        bind(&mut sched, &builder, other_ingest, "ingest", 3);
        bind(&mut sched, &builder, prompt, "dialogue", 4);
        sched
            .active_prefills
            .push(dialogue_prefill(ingest, vec![1; 8]));
        sched.form_wave_group(false);
        sched.wave_prefill_cursor = 5;

        // Nothing decodes: the creep is unthrottled, and the prompt still
        // outranks it.
        sched
            .active_prefills
            .push(dialogue_prefill(prompt, vec![1; 8]));
        assert!(sched.creep_outranked_by_a_waiting_prompt());

        // The cast decodes: the ingest creep is throttled, the prompt waits.
        let (state, _rx) = boundary_state();
        sched.active_decodes.insert(cast, state);
        assert!(sched.creep_outranked_by_a_waiting_prompt());

        // Another ingest prompt waiting does not outrank it.
        sched.active_prefills.pop();
        sched
            .active_prefills
            .push(dialogue_prefill(other_ingest, vec![1; 8]));
        assert!(!sched.creep_outranked_by_a_waiting_prompt());
    }

    /// **The paused ingest creep waits out the dialogue prompt, and a `High`
    /// group carries `High` prompts only.** The group formed beside the paused
    /// creep holds the prompt and neither the paused member nor another
    /// ingest's; the paused creep resumes only once the prompt has run.
    #[test]
    fn a_paused_creep_waits_out_the_prompt_that_outranked_it() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let other_ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let prompt = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, ingest, "ingest", 1);
        bind(&mut sched, &builder, other_ingest, "ingest", 2);
        bind(&mut sched, &builder, prompt, "dialogue", 3);
        sched
            .active_prefills
            .push(dialogue_prefill(ingest, vec![1; 8]));
        sched.form_wave_group(false);
        sched.wave_prefill_cursor = 5;
        sched.pause_creep();

        sched
            .active_prefills
            .push(dialogue_prefill(other_ingest, vec![1; 8]));
        sched
            .active_prefills
            .push(dialogue_prefill(prompt, vec![1; 8]));
        assert!(!sched.resume_creep(), "the dialogue prompt is waiting");
        sched.form_wave_group(false);
        let seqs: Vec<usize> = sched
            .wave_prefill_members
            .iter()
            .map(|m| m.seq_id())
            .collect();
        assert_eq!(seqs, vec![prompt.0], "the prompt alone");

        // The prompt has run.
        sched.reset_wave_prefill();
        sched
            .active_prefills
            .retain(|p| p.work.sequence_id != prompt);
        assert!(sched.resume_creep());
        assert_eq!(sched.wave_prefill_cursor, 5);
        assert_eq!(sched.wave_prefill_members[0].seq_id(), ingest.0);
    }

    /// A section in flight on `seq`, at `band`, with `tokens` left to ingest.
    fn active_section(seq: SequenceId, band: DecodePriority, tokens: usize) -> ActiveSectionIngest {
        let (response_tx, _rx) = flume::bounded(1);
        ActiveSectionIngest {
            sequence_id: seq,
            section_id: SectionId::new(1),
            tokens: TokenBuffer::from(vec![1u32; tokens]),
            offset: 0,
            seal_block_from: 0,
            address: ContentAddress::default(),
            debug_name: "s".into(),
            in_collection: true,
            export_end: false,
            band,
            response_tx,
            error: None,
            timing: SectionTiming::new(Instant::now(), Instant::now(), Instant::now()),
        }
    }

    /// **A dialogue's own section outranks an ingest creep, and runs apart
    /// from ingest rows.** A journal entry its next turn reads was measured
    /// waiting out a 6,213-row ingest sweep: here it pauses the creep, the group
    /// formed beside it leaves the ingest prompt out, and the paused creep
    /// resumes only once the section has run. A `Low` section changes nothing.
    #[test]
    fn a_dialogues_section_outranks_an_ingest_creep_and_rides_apart() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let other_ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let section = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, ingest, "ingest", 1);
        bind(&mut sched, &builder, other_ingest, "ingest", 2);
        sched
            .active_prefills
            .push(dialogue_prefill(ingest, vec![1; 8]));
        sched.form_wave_group(true);
        sched.wave_prefill_cursor = 5;

        sched
            .active_section_ingests
            .push(active_section(section, DecodePriority::Low, 38));
        assert!(
            !sched.creep_outranked_by_a_waiting_prompt(),
            "a schema's section is background"
        );

        sched.active_section_ingests[0].band = DecodePriority::High;
        assert!(sched.creep_outranked_by_a_waiting_prompt());
        sched.pause_creep();
        sched
            .active_prefills
            .push(dialogue_prefill(other_ingest, vec![1; 8]));
        assert!(!sched.resume_creep(), "the section is waiting");
        sched.form_wave_group(true);
        assert!(
            sched.wave_prefill_members.is_empty(),
            "no ingest rides beside it; it runs in the standalone section pass"
        );

        // The section has sealed.
        sched.active_section_ingests.clear();
        assert!(sched.resume_creep());
        assert_eq!(sched.wave_prefill_members[0].seq_id(), ingest.0);
    }

    /// **A section chunk with no creep beside it rides at its own band.** A
    /// schema's sections alone are `Low`; a dialogue's own waiting puts the
    /// batch at `High`, and beside a `High` decode it then yields at `High`
    /// beside `High` rather than at background's 64. A section already creeping
    /// in a group is not counted.
    #[test]
    fn a_standalone_section_chunk_rides_at_its_own_band() {
        let (mut sched, _tx) = make_test_scheduler();
        let ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let journal = SequenceId(sched.session.create_sequence().expect("create"));
        assert_eq!(sched.section_batch_band(), None, "nothing to ride");

        sched
            .active_section_ingests
            .push(active_section(ingest, DecodePriority::Low, 512));
        assert_eq!(sched.section_batch_band(), Some(DecodePriority::Low));

        sched
            .active_section_ingests
            .push(active_section(journal, DecodePriority::High, 38));
        assert_eq!(sched.section_batch_band(), Some(DecodePriority::High));
        assert_eq!(
            throttle_ratio(DecodePriority::High, [DecodePriority::High]),
            HIGH_BESIDE_HIGH_RATIO
        );
        assert_eq!(
            throttle_ratio(DecodePriority::Low, [DecodePriority::High]),
            DecodePriority::High.ratio()
        );

        sched.wave_prefill_members = vec![WaveMember::Section {
            seq_id: journal.0,
            advance: 32,
        }];
        assert_eq!(
            sched.section_batch_band(),
            None,
            "the journal entry creeping in a group outranks the ingest: its \
             debt would hold the group"
        );
        sched.wave_prefill_members.clear();
        sched
            .active_section_ingests
            .retain(|s| s.sequence_id != journal);
        assert_eq!(sched.section_batch_band(), Some(DecodePriority::Low));

        // A dialogue's prompt that can advance holds the ingest section back:
        // it would owe decode 64 times its share, and the debt holds the
        // prompt's creep too.
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let prompt = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, prompt, "dialogue", 1);
        sched
            .active_prefills
            .push(dialogue_prefill(prompt, vec![1; 8]));
        assert_eq!(sched.section_batch_band(), None, "the prompt outranks it");
        sched.active_prefills.clear();
        assert_eq!(sched.section_batch_band(), Some(DecodePriority::Low));
    }

    /// A section chunk runs at the band it was submitted at: a schema's alone
    /// at `Low`, a dialogue's at `High`, and beside a dialogue prompt the group
    /// rides at the prompt's band.
    #[test]
    fn a_creep_runs_at_its_highest_prompt_and_sections_alone_are_low() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let prompt = SequenceId(sched.session.create_sequence().expect("create"));
        let ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let section = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, prompt, "dialogue", 1);
        bind(&mut sched, &builder, ingest, "ingest", 2);

        let alone = [WaveMember::Section {
            seq_id: section.0,
            advance: 32,
        }];
        assert_eq!(sched.creep_priority(&alone), DecodePriority::Low);
        sched
            .active_section_ingests
            .push(active_section(section, DecodePriority::High, 32));
        assert_eq!(sched.creep_priority(&alone), DecodePriority::High);
        sched.active_section_ingests[0].band = DecodePriority::Low;
        assert_eq!(sched.creep_priority(&alone), DecodePriority::Low);

        let mixed = [
            WaveMember::Prefill {
                seq_id: ingest.0,
                advance: 32,
            },
            WaveMember::Prefill {
                seq_id: prompt.0,
                advance: 32,
            },
            WaveMember::Section {
                seq_id: section.0,
                advance: 32,
            },
        ];
        assert_eq!(sched.creep_priority(&mixed), DecodePriority::High);
    }

    /// **A section chunk yields to any tenant's conversation.** zend's
    /// section ingest runs at `Low`, so npcd's dialogue decode holds it to the
    /// dialogue's ratio.
    #[test]
    fn a_section_chunk_yields_to_another_tenants_dialogue() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let zend = Conversation::new();
        let npcd = Conversation::new().with_tenant_rank(TenantRank::nth(1).unwrap());
        let zend_section = SequenceId(sched.session.create_sequence().expect("create"));
        let npcd_dialogue = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, npcd_dialogue, "dialogue", 2);
        sched.slot_conversations.insert(zend_section, zend);
        sched.slot_conversations.insert(npcd_dialogue, npcd);
        let (state, _rx) = boundary_state();
        sched.active_decodes.insert(npcd_dialogue, state);

        let creep = [WaveMember::Section {
            seq_id: zend_section.0,
            advance: 32,
        }];
        let work = sched.creep_priority(&creep);
        assert_eq!(work, DecodePriority::Low);
        assert_eq!(sched.airtime_ratio_over(work), 64);
    }

    /// A decode whose priority does not resolve (the titler, a summary probe)
    /// is protected as `High` — the band it is gated in — so it holds back
    /// ingest and a dialogue prompt alike.
    #[test]
    fn an_unresolvable_decode_counts_as_high() {
        let (mut sched, _tx) = make_test_scheduler();
        let probe = SequenceId(sched.session.create_sequence().expect("create"));
        let (state, _rx) = boundary_state();
        sched.active_decodes.insert(probe, state);

        assert_eq!(sched.airtime_ratio_over(DecodePriority::Low), 64);
        assert_eq!(sched.airtime_ratio_over(DecodePriority::High), 2);
    }

    /// **A summary probe decodes at `Normal`**, though its slot is bound to the
    /// dialogue it compresses: it holds ingest back at `Normal`'s ratio and
    /// never paces a dialogue prompt.
    #[test]
    fn a_summary_probe_decodes_at_normal() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let probe = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, probe, "dialogue", 1);
        let (mut state, _rx) = boundary_state();
        state.seal_action = SealAction::CompressionPass { job_id: 7 };
        sched.active_decodes.insert(probe, state);

        assert_eq!(sched.decode_priority_or_high(probe), DecodePriority::Normal);
        assert_eq!(
            sched.airtime_ratio_over(DecodePriority::High),
            1,
            "a Normal probe never holds a dialogue prompt back"
        );
    }

    /// **A `High` creep yields to a `High` decode at the shared ratio**, and
    /// runs at full speed the moment that decode is gone.
    #[test]
    fn a_high_creep_shares_with_a_high_decode_and_runs_whole_when_it_clears() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let decoding = SequenceId(sched.session.create_sequence().expect("create"));
        let prompt = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, decoding, "dialogue", 1);
        bind(&mut sched, &builder, prompt, "dialogue", 2);
        sched
            .active_prefills
            .push(dialogue_prefill(prompt, vec![1; 8]));
        sched.form_wave_group(false);
        let (state, _rx) = boundary_state();
        sched.active_decodes.insert(decoding, state);

        assert_eq!(sched.airtime_ratio_over(DecodePriority::High), 2);
        assert_eq!(sched.airtime_ratio_over(DecodePriority::Low), 64);

        sched.active_decodes.remove(&decoding);
        assert_eq!(
            sched.airtime_ratio_over(DecodePriority::High),
            1,
            "the decode cleared"
        );
    }

    /// A finished decode and a compression pass protect nothing: only a
    /// foreground decode still generating holds work back.
    #[test]
    fn only_a_running_foreground_decode_holds_work_back() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let done = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, done, "dialogue", 1);
        let (mut state, _rx) = boundary_state();
        state.finished = true;
        sched.active_decodes.insert(done, state);

        assert_eq!(sched.airtime_ratio_over(DecodePriority::Low), 1);
    }
}
