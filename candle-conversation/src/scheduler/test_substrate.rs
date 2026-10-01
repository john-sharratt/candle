//! A small on-disk substrate with a tool collection and one dialogue layer —
//! the fixture the scheduler's belief-scan tests score against.

use std::path::Path;

use crate::persistence::content_hash::turn_stream_id;
use crate::persistence::SubstratePersistence;
use crate::projection::{Builder, Conversation, ProjectionTarget, TimelineId};
use crate::provenance::{encode_wide_sigs, WideQSig};
use crate::substrate::{Substrate, TurnPartWrite};
use crate::turn::Role;

/// A tool-catalog-shaped collection scoped by the `tool` tag, members named by
/// each exemplar's second tag, beside one dialogue layer.
pub(super) const YAML: &str = r#"
system_prompt:
  items:
    - kind: section
      id: frame
      content: "frame"
    - kind: collection
      name: tools
      selection: { kind: top_k, k: 2 }
      policy:
        tags: ["tool"]
      sections:
        - id: alpha
          content: "alpha tool"
        - id: beta
          content: "beta tool"
layers:
  - name: mem
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
      - id: clusters
        selection: { kind: top_k, k: 2 }
"#;

/// A folded 12-head signature, every word `fill`.
pub(super) fn sig(fill: u64) -> WideQSig {
    WideQSig {
        n_heads: 12,
        words: vec![fill; 24],
    }
}

/// A fresh substrate under `dir`.
pub(super) fn open(dir: &Path) -> Conversation {
    let mut substrate = Substrate::new();
    let persistence = SubstratePersistence::open_in_with_substrate(dir, &mut substrate).unwrap();
    Conversation::from_parts(substrate, persistence)
}

/// Record one turn per `(tags, fill, len)` on timeline 31 of [`YAML`]'s
/// dialogue layer, each with a `len`-token signature of `fill`. Returns the
/// layer's target.
pub(super) fn record_turns(
    conv: &Conversation,
    builder: &Builder,
    turns: &[(&[&str], u64, usize)],
) -> ProjectionTarget {
    let layer = builder.id_for_layer("mem").unwrap();
    let group = builder.id_for_group("clusters").unwrap();
    let timeline = TimelineId::from_raw(31).expect("timeline id");
    conv.register_timeline(timeline, layer, group);
    for &(tags, fill, len) in turns {
        let idx = conv
            .record_turn(
                timeline,
                Role::User,
                TurnPartWrite {
                    token_count: 4,
                    tags: tags.iter().map(|t| t.to_string()).collect(),
                    ..Default::default()
                },
                |seqs| Ok(seqs.to_vec()),
            )
            .expect("record_turn");
        let window: Vec<WideQSig> = (0..len).map(|_| sig(fill)).collect();
        conv.persist_wide_q_sigs(
            turn_stream_id(timeline.raw(), idx.0),
            &encode_wide_sigs(&window),
        )
        .expect("persist sigs");
    }
    ProjectionTarget {
        layer,
        group,
        timeline,
    }
}
