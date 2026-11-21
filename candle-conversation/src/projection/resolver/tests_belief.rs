//! Tests the belief scoring rules: what a tool turn teaches, how scopes are
//! segmented, and how gallery and vote policy parse.
//!
//! These pin the failures that cost the most measured accuracy. Dialogue teaching
//! the tool lens cut the correct-answer rate from 67% to 17% across 24
//! conversations, so a probe may teach only the scopes it sits inside. Gallery
//! keeps, vote law and normalization must parse from the policy, and a law the scan
//! cannot run is refused. Calibration questions must vote for the file they
//! calibrate, and a file's gallery is kept until something it reads changes.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::{Conversation, Observe, ScanPlans};

    use std::sync::Arc;

    use crate::persistence::content_hash::turn_stream_id;

    use crate::projection::TimelineId;

    use crate::substrate::TurnPartWrite;

    use crate::turn::Role;

    use crate::projection::Builder;

    use super::super::tests_turns::tests::tags;

    /// **A sealed tool turn is about the tool its tags name**, and a pass that
    /// teaches nothing, or names no member, is about none.
    #[test]
    fn a_tool_turn_is_about_the_tool_its_tags_name() {
        let members = ["datetime", "file_read", "git_grep"];
        let turn = tags(&["tool", "file_read"]);
        let seal = Observe::Yes {
            tags: &turn,
            source: 7,
        };
        assert_eq!(seal.answer(members), Some("file_read"));
        let dialogue = tags(&["tool"]);
        let none = Observe::Yes {
            tags: &dialogue,
            source: 8,
        };
        assert_eq!(none.answer(members), None);
        assert_eq!(Observe::No.answer(members), None);
    }

    /// The segmentation rule: a scope learns from a probe only when the probe is
    /// inside it. This is what stops dialogue traffic degrading the tool lens
    /// WITHOUT switching learning off — the failure mode that took the measured
    /// correct-answer rate from 67% to 17% across 24 conversations.
    #[test]
    fn a_scope_learns_only_from_probes_inside_it() {
        let tools = tags(&["tool"]);
        let unscoped: Vec<String> = Vec::new();

        // A tool-tagged turn teaches the tool lens.
        let from_tool = Observe::Yes {
            tags: &tags(&["tool", "datetime"]),
            source: 1,
        };
        assert!(from_tool.teaches(&tools));

        // An ordinary dialogue turn does NOT — this is the whole fix.
        let from_dialogue = Observe::Yes {
            tags: &[],
            source: 2,
        };
        assert!(
            !from_dialogue.teaches(&tools),
            "dialogue must not teach a tag-scoped lens",
        );

        // …but it still teaches the unscoped scopes, so learning continues.
        assert!(
            from_dialogue.teaches(&unscoped),
            "dialogue must still teach unscoped turn groups",
        );

        // A turn tagged for a DIFFERENT scope does not leak across.
        let from_other = Observe::Yes {
            tags: &tags(&["memory"]),
            source: 3,
        };
        assert!(!from_other.teaches(&tools));

        // And a read-only pass teaches nothing at all.
        assert!(!Observe::No.teaches(&tools));
        assert!(!Observe::No.teaches(&unscoped));
    }

    /// **A group's gallery keeps parse**: a count per code, `all`, the default
    /// when omitted, and `0` or another word refused.
    #[test]
    fn gallery_keeps_parse_from_the_policy() {
        use crate::provenance::gallery_arena::Keep;

        let yaml = |policy: &str| {
            format!(
                r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
layers:
  - name: files
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 40
    groups:
      - id: scopes
        selection: {{ kind: top_k, k: 2 }}
        policy: {{ {policy} }}
"#
            )
        };
        let keeps = |policy: &str| {
            Builder::from_yaml(&yaml(policy)).map(|b| {
                let scan = b.schema().layers[0].groups[0].policy.scan;
                (scan.gallery_keep, scan.calibration_gallery_keep)
            })
        };
        assert_eq!(
            keeps("gallery_keep: 1, calibration_gallery_keep: all").unwrap(),
            (Keep::FirstPerCode(1), Keep::All)
        );
        assert_eq!(
            keeps("cases: file").unwrap(),
            (Keep::All, Keep::All),
            "omitted keeps every token"
        );
        assert_eq!(
            keeps("calibration_gallery_keep: 2").unwrap(),
            (Keep::All, Keep::FirstPerCode(2))
        );
        assert!(keeps("gallery_keep: 0").is_err());
        assert!(keeps("gallery_keep: some").is_err());
    }

    /// **A group's vote law, normalization and lineage parse from its
    /// policy**, default to today's single vote against each file's background,
    /// and a law the scan cannot run is refused with its reason.
    #[test]
    fn a_vote_law_and_kind_normalization_parse_from_the_policy() {
        use crate::projection::ScanNorm;
        use crate::provenance::VoteLaw;

        let yaml = |policy: &str| {
            format!(
                r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
layers:
  - name: files
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 40
    groups:
      - id: scopes
        selection: {{ kind: top_k, k: 2 }}
        policy: {{ {policy} }}
"#
            )
        };
        let scan = |policy: &str| {
            Builder::from_yaml(&yaml(policy)).map(|b| b.schema().layers[0].groups[0].policy.scan)
        };
        let lab = scan(
            "cases: file, vote_top_k: 8, vote_power: 0.5, normalize: kind_z, \
             exclude_lineage: true",
        )
        .unwrap();
        assert_eq!(
            lab.law,
            VoteLaw {
                top_k: 8,
                power: 0.5
            }
        );
        assert_eq!(lab.normalize, ScanNorm::KindZ);
        assert!(lab.exclude_lineage);
        let plain = scan("cases: file").unwrap();
        assert_eq!(plain.law, VoteLaw::SINGLE);
        assert_eq!(plain.normalize, ScanNorm::Background);
        assert!(!plain.exclude_lineage);

        let reason = |policy: &str| scan(policy).err().unwrap().to_string();
        assert!(reason("vote_top_k: 9").contains("vote_top_k must be in 1..=8"));
        assert!(reason("vote_top_k: 0").contains("vote_top_k must be in 1..=8"));
        assert!(reason("vote_power: 0").contains("vote_power a positive number"));
        assert!(reason("vote_top_k: 2, fusion: { mode: content_gated }")
            .contains("needs fusion: additive"));
        assert!(reason("normalize: kind_z").contains("needs cases: file"));
        assert!(reason("normalize: kind").contains("unknown normalize"));
    }

    /// **A calibration's questions vote for the file they calibrate.** Two
    /// files whose content shares nothing with a question; a calibration of
    /// the second holds that question, and one of the first its opposite.
    /// Assembled, each file carries its calibration's question as a window
    /// apart from its own, and the question probe ranks the second first
    /// through the calibration competition, where the two questions compete.
    #[test]
    fn a_calibrations_questions_vote_for_their_file() {
        use super::super::super::belief_files::{assemble_file_scans, scan_file_scans};
        use crate::provenance::gallery_arena::Holder;
        use crate::provenance::{encode_wide_sigs, WideQSig};
        use crate::substrate::Substrate;

        const YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
layers:
  - name: files
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 40
    groups:
      - id: scopes
        selection: { kind: top_k, k: 2 }
        policy: { cases: file }
  - name: hidden
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 10
    groups:
      - id: calibration
        selection: { kind: top_k, k: 1 }
"#;
        let sig = |fill: u64| WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        };
        let builder = Builder::from_yaml(YAML).unwrap();
        let files_layer = builder.id_for_layer("files").unwrap();
        let scopes = builder.id_for_group("scopes").unwrap();
        let hidden = builder.id_for_layer("hidden").unwrap();
        let calibration = builder.id_for_group("calibration").unwrap();
        let conv = Conversation::ephemeral();
        let write = |tl: TimelineId, sigs: Vec<WideQSig>| {
            let idx = conv
                .record_turn(
                    tl,
                    Role::User,
                    TurnPartWrite {
                        token_count: sigs.len(),
                        ..Default::default()
                    },
                    |seqs| Ok(seqs.to_vec()),
                )
                .expect("record_turn");
            conv.persist_wide_q_sigs(turn_stream_id(tl.raw(), idx.0), &encode_wide_sigs(&sigs))
                .expect("persist sigs");
        };
        let (x, y, cal, cal_x) = (
            TimelineId::from_raw(51).unwrap(),
            TimelineId::from_raw(52).unwrap(),
            TimelineId::from_raw(53).unwrap(),
            TimelineId::from_raw(54).unwrap(),
        );
        conv.register_timeline(x, files_layer, scopes);
        conv.register_timeline(y, files_layer, scopes);
        conv.register_timeline(cal, hidden, calibration);
        conv.register_timeline(cal_x, hidden, calibration);
        write(
            x,
            vec![sig(0x0F0F_0F0F_0F0F_0F0F), sig(0x00FF_00FF_00FF_00FF)],
        );
        write(
            y,
            vec![sig(0x3333_3333_3333_3333), sig(0x0000_FFFF_0000_FFFF)],
        );
        let question = 0x5555_5555_5555_5555u64;
        write(cal, vec![sig(question)]);
        conv.set_conversation_metadata(cal, Substrate::CALIBRATES_KEY, &y.raw().to_string())
            .unwrap();
        write(cal_x, vec![sig(!question)]);
        conv.set_conversation_metadata(cal_x, Substrate::CALIBRATES_KEY, &x.raw().to_string())
            .unwrap();

        let files = {
            let sub = conv.inner.read();
            assemble_file_scans(&sub, vec![x, y]).0
        };
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].windows, vec![(0, 0, 2, 0)]);
        assert_eq!(files[0].cal_windows, vec![(1, 0, 1)]);
        assert_eq!(files[1].windows, vec![(0, 0, 2, 0)]);
        assert_eq!(files[1].cal_windows, vec![(1, 0, 1)]);
        assert_eq!(files[1].sid(1), turn_stream_id(cal.raw(), 0));

        let group = builder
            .schema()
            .layers
            .iter()
            .flat_map(|l| &l.groups)
            .find(|g| g.id == scopes)
            .unwrap();
        let probe = vec![sig(question)];
        let scanned = scan_file_scans(
            &files,
            group,
            &[probe.as_slice()],
            None,
            Holder::Shared,
            &ScanPlans::default(),
        );
        let (x_score, y_score) = (scanned[0][0].0[0], scanned[0][1].0[0]);
        assert!(y_score > x_score, "x {x_score} y {y_score}");

        // **A conversation's calibrations are one record, read again as long
        // as it stands.** Nothing changed, so the second pass walks no turn.
        let assemble = || {
            let sub = conv.inner.read();
            assemble_file_scans(&sub, vec![x, y]).0
        };
        let record = || assemble()[1].cal.clone().expect("y is calibrated");
        let first = record();
        let second = record();
        assert!(Arc::ptr_eq(&first, &second), "the record stands");

        // A question added shows in the calibration's length.
        write(cal, vec![sig(!0)]);
        let grown = assemble();
        assert_eq!(grown[1].cal_windows, vec![(1, 0, 1), (2, 0, 1)]);
        assert_eq!(grown[1].sid(2), turn_stream_id(cal.raw(), 1));

        // A stored signature replaced shows in nothing a length can: every
        // record goes, and the next read holds the new window.
        let before = record();
        conv.persist_wide_q_sigs(
            turn_stream_id(cal.raw(), 0),
            &encode_wide_sigs(&[sig(0x7777_7777_7777_7777)]),
        )
        .expect("persist sigs");
        let after = record();
        assert!(!Arc::ptr_eq(&before, &after), "a rewrite drops the record");
        assert_ne!(before.sigs[0].fingerprint(), after.sigs[0].fingerprint());
    }

    /// **A file's gallery is built once and kept on its timeline until
    /// something it reads changes, and only its own change rebuilds it.** Two
    /// files of one signed turn: a second assembly builds neither; a signature
    /// rewritten on one rebuilds that one, holding the new window; a turn
    /// added to the other rebuilds that one, holding both turns.
    #[test]
    fn a_files_gallery_is_kept_until_what_it_reads_changes() {
        use super::super::super::belief_files::assemble_file_scans;
        use super::super::super::belief_scan_split::GROUP_SCAN_SPLIT;
        use crate::provenance::{encode_wide_sigs, WideQSig};

        const YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
layers:
  - name: files
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 40
    groups:
      - id: scopes
        selection: { kind: top_k, k: 2 }
        policy: { cases: file }
"#;
        let sig = |fill: u64| WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        };
        let builder = Builder::from_yaml(YAML).unwrap();
        let files_layer = builder.id_for_layer("files").unwrap();
        let scopes = builder.id_for_group("scopes").unwrap();
        let conv = Conversation::ephemeral();
        let write = |tl: TimelineId, sigs: Vec<WideQSig>| {
            let idx = conv
                .record_turn(
                    tl,
                    Role::User,
                    TurnPartWrite {
                        token_count: sigs.len(),
                        ..Default::default()
                    },
                    |seqs| Ok(seqs.to_vec()),
                )
                .expect("record_turn");
            conv.persist_wide_q_sigs(turn_stream_id(tl.raw(), idx.0), &encode_wide_sigs(&sigs))
                .expect("persist sigs");
        };
        let (x, y) = (
            TimelineId::from_raw(61).unwrap(),
            TimelineId::from_raw(62).unwrap(),
        );
        conv.register_timeline(x, files_layer, scopes);
        conv.register_timeline(y, files_layer, scopes);
        write(x, vec![sig(1), sig(2)]);
        write(y, vec![sig(3), sig(4), sig(5)]);
        // Each assembly: the files, and how many galleries it built.
        let assemble = || {
            GROUP_SCAN_SPLIT.take();
            let files = assemble_file_scans(&conv.inner.read(), vec![x, y]).0;
            (files, GROUP_SCAN_SPLIT.take().galleries_built)
        };

        let (first, built) = assemble();
        assert_eq!(built, 2);
        assert_eq!(first[0].windows, vec![(0, 0, 2, 0)]);
        assert_eq!(first[1].windows, vec![(0, 0, 3, 0)]);
        let (again, built) = assemble();
        assert_eq!(built, 0, "nothing changed");
        assert!(Arc::ptr_eq(&first[0].gallery, &again[0].gallery));
        assert!(Arc::ptr_eq(&first[1].gallery, &again[1].gallery));

        conv.persist_wide_q_sigs(
            turn_stream_id(x.raw(), 0),
            &encode_wide_sigs(&[sig(7), sig(8), sig(9), sig(10)]),
        )
        .expect("persist sigs");
        let (resigned, built) = assemble();
        assert_eq!(built, 1, "x alone");
        assert_eq!(resigned[0].windows, vec![(0, 0, 4, 0)]);
        assert_ne!(
            resigned[0].arcs_kept[0].fingerprint(),
            first[0].arcs_kept[0].fingerprint()
        );
        assert!(Arc::ptr_eq(&first[1].gallery, &resigned[1].gallery));

        write(y, vec![sig(11)]);
        let (grown, built) = assemble();
        assert_eq!(built, 1, "y alone");
        assert_eq!(grown[1].n_slots, 2);
        assert_eq!(grown[1].windows, vec![(0, 0, 3, 0), (1, 0, 1, 1)]);
        assert!(Arc::ptr_eq(&resigned[0].gallery, &grown[0].gallery));
    }

    /// **The preload takes only what every conversation reads.** A shared
    /// layer's two turns are preloaded; a conversation-scoped layer's three are
    /// left for their own conversation's scans to upload while it is loaded.
    #[test]
    fn a_preload_takes_only_shared_layers() {
        use crate::provenance::{encode_wide_sigs, WideQSig};

        const YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
layers:
  - name: files
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 40
    groups:
      - id: scopes
        selection: { kind: top_k, k: 2 }
  - name: dialogue
    gather_scope: conversation
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 100
    groups:
      - id: chat
        selection: { kind: top_k, k: 2 }
"#;
        let builder = Builder::from_yaml(YAML).unwrap();
        let conv = Conversation::ephemeral();
        let write = |tl: TimelineId, n: u64| {
            for t in 0..n {
                let sigs = vec![WideQSig {
                    n_heads: 12,
                    words: vec![t; 24],
                }];
                let idx = conv
                    .record_turn(
                        tl,
                        Role::User,
                        TurnPartWrite {
                            token_count: 1,
                            ..Default::default()
                        },
                        |seqs| Ok(seqs.to_vec()),
                    )
                    .expect("record_turn");
                conv.persist_wide_q_sigs(turn_stream_id(tl.raw(), idx.0), &encode_wide_sigs(&sigs))
                    .expect("persist sigs");
            }
        };
        let (file, chat) = (
            TimelineId::from_raw(61).unwrap(),
            TimelineId::from_raw(62).unwrap(),
        );
        conv.register_timeline(
            file,
            builder.id_for_layer("files").unwrap(),
            builder.id_for_group("scopes").unwrap(),
        );
        conv.register_timeline(
            chat,
            builder.id_for_layer("dialogue").unwrap(),
            builder.id_for_group("chat").unwrap(),
        );
        write(file, 2);
        write(chat, 3);
        let preload = conv.preload_galleries(builder.schema(), None);
        assert_eq!(preload.turns, 2, "only the shared layer's turns");
    }
}
