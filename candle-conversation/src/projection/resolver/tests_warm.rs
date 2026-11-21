//! Tests which groups and collections the normalization warm-up may teach, and how
//! its probes are planned.
//!
//! Only an additive, exchange-scored group in a scanned layer is warmed by
//! self-match, since a file's self-match would teach its scope nothing but the
//! file's own ceiling. Each collection member's exemplars are scanned without that
//! member, grouped under it and capped deterministically. A corpus with no scope
//! tags yields no plan, so the caller reports zero rather than a silent gap.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::{
        collection_warm_plan, ingest_group, is_warmable, leave_out_rounds, subwindow_bounds,
        warmable, Conversation,
    };

    use crate::projection::{GroupId, TimelineId};

    use crate::projection::Builder;

    use std::collections::HashSet;

    /// A belief layer with two ranked groups beside a dialogue layer — enough
    /// to ask which group a warm reaches and what it scores under.
    pub(in super::super) const WARM_YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "You are a helpful assistant."
layers:
  - name: ground
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
      - id: facts
        selection:
          kind: top_k
          k: 3
      - id: rumours
        selection:
          kind: top_k
          k: 2
  - name: dialogue
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
      min_percent: 50
    groups:
      - id: conversation
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 8
"#;

    /// A ranked group is warmed where the schema declares it, and asking for it
    /// under any other layer finds nothing rather than a group of the same id
    /// somewhere else.
    #[test]
    fn a_ranked_group_is_warmable_in_its_own_layer_and_nowhere_else() {
        let b = Builder::from_yaml(WARM_YAML).unwrap();
        let ground = b.id_for_layer("ground").unwrap();
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let facts = b.id_for_group("facts").unwrap();

        let (layer, group) = warmable(b.schema(), ground, facts).expect("facts is warmable");
        assert_eq!(layer.name, "ground");
        assert_eq!(group.name, "facts");
        assert!(warmable(b.schema(), dialogue, facts).is_none());
    }

    /// Recency is not a belief, so there is no background to learn.
    #[test]
    fn a_recency_group_is_not_warmable() {
        let b = Builder::from_yaml(WARM_YAML).unwrap();
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let conversation = b.id_for_group("conversation").unwrap();
        let group = b
            .schema()
            .layers
            .iter()
            .find(|l| l.id == dialogue)
            .unwrap()
            .groups[0]
            .clone();
        assert!(!is_warmable(&group));
        assert!(warmable(b.schema(), dialogue, conversation).is_none());
    }

    /// **A layer out of retrieval is not warmed** through `warmable()` — the
    /// single-group and single-timeline warm-ups this gates. The BULK ingest
    /// warm-up (`warm_ingest_normalization`) does NOT go through `warmable()`
    /// and does not honour this flag for an append-only layer: see
    /// `turn_belief_scan.rs`'s `warm_ingest_normalization_still_warms_an_append_only_layer_when_not_gathered`
    /// for why — a `code_reading`/`repo_map`-shaped layer scores itself
    /// regardless of `gathered`, since that flag promises only that no OTHER
    /// conversation draws from it.
    #[test]
    fn a_group_in_a_layer_out_of_retrieval_is_not_warmable() {
        let mut b = Builder::from_yaml(WARM_YAML).unwrap();
        let ground = b.id_for_layer("ground").unwrap();
        let facts = b.id_for_group("facts").unwrap();
        assert!(warmable(b.schema(), ground, facts).is_some());
        assert!(b.set_layer_gathered("ground", false));
        assert!(warmable(b.schema(), ground, facts).is_none());
    }

    /// The bulk ingest warm-up finds an append-only layer's group whether or
    /// not the layer is gathered — the flag promises only that no OTHER
    /// conversation draws from it — while [`warmable`] refuses it.
    #[test]
    fn an_ingest_group_is_found_whether_or_not_its_layer_is_gathered() {
        let mut b = Builder::from_yaml(WARM_YAML).unwrap();
        let ground = b.id_for_layer("ground").unwrap();
        let facts = b.id_for_group("facts").unwrap();
        assert!(b.set_layer_gathered("ground", false));
        assert!(warmable(b.schema(), ground, facts).is_none());
        assert_eq!(
            ingest_group(b.schema(), ground, facts).map(|g| g.id),
            Some(facts)
        );
    }

    /// **A file-scored group is not warmed.** Its files are cases of its pool
    /// scope, whose levels live traffic teaches; a file's self-match would
    /// teach that scope nothing but the file's own ceiling. Beside it in one
    /// layer, an exchange-scored group is warmed. No warm-up entry point —
    /// the bulk ingest work, a single group, a single timeline — takes the
    /// file-scored one.
    #[test]
    fn a_file_scored_group_is_not_warmed() {
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
        policy:
          cases: file
      - id: exchanges
        selection: { kind: top_k, k: 2 }
"#;
        let builder = Builder::from_yaml(YAML).unwrap();
        let schema = builder.schema();
        let layer = builder.id_for_layer("files").unwrap();
        let scopes = builder.id_for_group("scopes").unwrap();
        let exchanges = builder.id_for_group("exchanges").unwrap();
        let conv = Conversation::ephemeral();
        let (file, exchange) = (
            TimelineId::from_raw(41).unwrap(),
            TimelineId::from_raw(42).unwrap(),
        );
        conv.register_timeline(file, layer, scopes);
        conv.register_timeline(exchange, layer, exchanges);
        conv.mark_layer_append_only(layer);

        assert!(warmable(schema, layer, exchanges).is_some());
        assert!(warmable(schema, layer, scopes).is_none());
        assert!(ingest_group(schema, layer, scopes).is_none());
        let work: Vec<GroupId> = conv
            .ingest_warm_work(schema)
            .into_iter()
            .map(|(_, g, _)| g)
            .collect();
        assert_eq!(work, vec![exchanges]);
        assert_eq!(conv.warm_group_normalization(schema, scopes), 0);
        assert_eq!(conv.warm_group_normalization(schema, exchanges), 1);
        assert!(!conv.warm_timeline_normalization(schema, file));
        assert!(conv.warm_timeline_normalization(schema, exchange));
    }

    /// **The warm-up scans once per member, each scan leaving that member out.**
    /// Probes are grouped under the case they are exemplars of, in the order
    /// each case first appears; a probe about no member is a round of its own.
    #[test]
    fn the_warm_up_scans_each_members_exemplars_without_that_member() {
        let rounds = leave_out_rounds([Some(2), Some(0), Some(2), None, Some(0), Some(2)]);
        assert_eq!(
            rounds,
            vec![
                (Some(2), vec![0, 2, 5]),
                (Some(0), vec![1, 4]),
                (None, vec![3]),
            ]
        );
        assert!(leave_out_rounds([]).is_empty());
    }

    /// `(tags, timeline, index)` rows in the shape `collection_warm_plan` reads.
    pub(in super::super) fn corpus(rows: &[(&[&str], u64, u32)]) -> Vec<(Vec<String>, u64, u32)> {
        rows.iter()
            .map(|(tags, tl, ix)| (tags.iter().map(|t| t.to_string()).collect(), *tl, *ix))
            .collect()
    }

    pub(in super::super) fn plan(
        members: &[&str],
        scope: &[&str],
        rows: &[(Vec<String>, u64, u32)],
        per_member: usize,
    ) -> Vec<(String, Vec<(u64, u32)>)> {
        let names: HashSet<&str> = members.iter().copied().collect();
        let scope: Vec<String> = scope.iter().map(|s| s.to_string()).collect();
        collection_warm_plan(
            &names,
            &scope,
            rows.iter().map(|(t, tl, ix)| (t.as_slice(), *tl, *ix)),
            per_member,
        )
    }

    #[test]
    fn warm_plan_groups_corpus_turns_by_the_member_they_teach() {
        let rows = corpus(&[
            (&["tool", "datetime"], 7, 1),
            (&["tool", "calculator"], 7, 0),
            (&["tool", "datetime"], 3, 5),
        ]);
        let got = plan(&["datetime", "calculator"], &["tool"], &rows, 8);
        assert_eq!(
            got,
            vec![
                ("calculator".to_string(), vec![(7, 0)]),
                // Members in name order, keys in (timeline, index) order — NOT the
                // order the rows arrived in, which is stream-map order live.
                ("datetime".to_string(), vec![(3, 5), (7, 1)]),
            ]
        );
    }

    #[test]
    fn warm_plan_skips_turns_outside_the_scope_or_naming_no_member() {
        let rows = corpus(&[
            // A dialogue turn: no scope tag at all.
            (&[], 1, 0),
            // Scoped, but names no member of THIS collection — must not be
            // attributed to an arbitrary member.
            (&["tool"], 1, 1),
            // A different collection's corpus.
            (&["memory", "datetime"], 1, 2),
            (&["tool", "datetime"], 1, 3),
        ]);
        let got = plan(&["datetime"], &["tool"], &rows, 8);
        assert_eq!(got, vec![("datetime".to_string(), vec![(1, 3)])]);
    }

    #[test]
    fn warm_plan_caps_probes_per_member_keeping_the_lowest_keys() {
        // Ten turns for one member, cap 3. The cap is per MEMBER, and which three
        // survive must be deterministic, not whichever three were seen first.
        let rows = corpus(
            &(0..10)
                .rev()
                .map(|i| (&["tool", "datetime"][..], 1u64, i as u32))
                .collect::<Vec<_>>(),
        );
        let got = plan(&["datetime"], &["tool"], &rows, 3);
        assert_eq!(
            got,
            vec![("datetime".to_string(), vec![(1, 0), (1, 1), (1, 2)])]
        );
    }

    #[test]
    fn warm_plan_is_empty_when_the_corpus_is_untagged() {
        // The regression this warm-up exists for: a workspace whose only turns are
        // dialogue produces no plan, and the caller's log line reports zero rather
        // than silently leaving levels cold.
        let rows = corpus(&[(&[], 1, 0), (&[], 1, 1)]);
        assert!(plan(&["datetime"], &["tool"], &rows, 8).is_empty());
    }

    #[test]
    fn subwindow_bounds_splits_and_degrades_gracefully() {
        // No seams → one whole-turn window (the prior behaviour).
        assert_eq!(subwindow_bounds(688, &[]), vec![(0, 688)]);
        // Interior seams → contiguous, gap-free intervals covering [0, len).
        assert_eq!(
            subwindow_bounds(688, &[100, 300]),
            vec![(0, 100), (100, 300), (300, 688)]
        );
        // Seams at 0, at/past len, and non-advancing duplicates are no-ops — never
        // an empty or inverted range.
        assert_eq!(subwindow_bounds(688, &[0]), vec![(0, 688)]);
        assert_eq!(subwindow_bounds(688, &[688, 900]), vec![(0, 688)]);
        assert_eq!(
            subwindow_bounds(688, &[100, 100]),
            vec![(0, 100), (100, 688)]
        );
        // Empty turn → a single empty window (the caller filters `e > s`).
        assert_eq!(subwindow_bounds(0, &[]), vec![(0, 0)]);
    }
}
