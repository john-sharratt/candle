//! Multi-layer projection engine for unbounded-context LLM inference.
//!
//! # What this is
//!
//! `projection` compresses an **unbounded layered substrate** of conversation
//! content into a **fixed-size context window** by applying declared budget
//! and selection rules at every level of the hierarchy.
//!
//! The substrate grows without bound — turns accumulate across cognitive
//! cycles, conversations span sessions, every layer of the substrate produces
//! content over time. The window does not grow. Every cognitive step requires
//! a fresh projection: which substrate content survives the budget, in which
//! order, framed by which system prompt.
//!
//! # What this is NOT
//!
//! This module is a **pure structural reconciler**. It does not own:
//!
//! - **Turn or section content** — the caller stores content and resolves it
//!   from `(GroupId, TurnIndex)` or `SectionId` keys after projection emits
//!   them
//! - **Tokenization** — token counts are supplied through the resolver; no
//!   tokenizer touches this module
//! - **Scoring** — scores are computed externally (typically from a Binary
//!   Directional Provenance scan against a live query) and supplied through
//!   the resolver per projection
//!
//! # Conversation model (ASCII)
//!
//! ```text
//! Schema (immutable after construction)
//! └── Layers                                   [ordered, layer 0 first]
//!     ├── Layer { name, window, score_threshold, budget }
//!     │   │
//!     │   │   `window` = total turn-budget when THIS layer is the
//!     │   │             projection target.  Different targets get
//!     │   │             different pies.
//!     │   │   `budget` = flex weight when SOME OTHER layer is the
//!     │   │             target and this layer is visible.
//!     │   │
//!     │   ├── system_prompt                    [framing for THIS layer
//!     │   │   └── Section, Section, ...        as the projection target;
//!     │   │                                    sections emit in full,
//!     │   │                                    declaration order]
//!     │   │
//!     │   └── Groups                           [unordered structurally]
//!     │       ├── Group { selection, score_threshold, budget }
//!     │       │   └── Turns                    [append-only, opaque to crate]
//!     │       └── Group { ... }
//!     └── Layer { ... }
//! ```
//!
//! Each layer carries its own system prompt because each cognitive layer
//! (live dialogue, bug analysis, dream exploration, daily convergence …) is
//! a distinct conversation with its own framing. The `system_prompt` block
//! is **required** on every layer, with at least one section — a layer
//! without framing isn't a usable projection target. Section names are
//! per-layer scoped, so `dialogue` and `bug_analysis` may both declare a
//! `frame` section without conflict.
//!
//! Sections have no token budget — they always emit in declaration order,
//! in full. What the system prompt projects to is what it is.
//!
//! - **Layers are ordered** (layer 0 first) — used by masking and emission
//! - **Groups within a layer are unordered structurally** — at projection time
//!   they are sorted by their derived group score
//! - **Turns within a group are ordered by insertion** (which is time order,
//!   since appends are append-only)
//! - **Sections** within the system prompt are ordered by declaration
//!
//! # Identifiers
//!
//! Two kinds of stable identifiers are handed out by the crate:
//!
//! | Type        | Scope                        | Assigned at        |
//! |-------------|------------------------------|--------------------|
//! | [`LayerId`] | Whole schema                 | Construction       |
//! | [`GroupId`] | Whole schema (globally)      | Construction       |
//! | [`SectionId`] | Whole schema               | Construction       |
//! | [`TurnIndex`] | Single group               | [`Builder::append`](builder::Builder::append) |
//!
//! # Projection flow
//!
//! Calling [`Builder::project`](builder::Builder::project) runs this 12-step
//! pipeline (see [`project::run`] for the implementation):
//!
//! ```text
//!   ┌────────────────────────────────────────────────────────────────┐
//!   │ 1. Apply target mask                                           │
//!   │    → which layers/groups are visible from this target?         │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │ 2. Score every visible turn via resolver                       │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │ 3. Apply group score thresholds                                │
//!   │    → eligible turns per group                                  │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │ 4. Apply selection rules under unbounded budget                │
//!   │    → naturally-selected turns per group                        │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │ 5. Compute group scores via FIXED_FORMULA (Span α=2.0)         │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │ 6. Apply layer score thresholds → surviving groups             │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │ 7. Filter empty groups and empty layers                        │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │ 8. Emit target layer's system-prompt sections (no reconcile)   │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │ 9. Reconcile turn budget across layers, then groups, where the │
//!   │    total = target_layer.window.  Flexbox with natural-         │
//!   │    consumption caps redistributes between visible layers.      │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │ 10. Run bounded selection per group with allocated budget      │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │ 11. (Iterative redistribution if needed — see project.rs)      │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │ 12. Emit:                                                      │
//!   │     • sections in declaration order                            │
//!   │     • layers in declaration order, groups by ascending score   │
//!   │       (so highest-scored emit LAST, near attention sink)       │
//!   │     • turns within each group in insertion order               │
//!   └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Masking semantics (visualised)
//!
//! Masking is parameterised by the projection target — `(LayerId, GroupId)`.
//! For a target of `(motivational, active_mission)`:
//!
//! ```text
//!                                            visible?
//! SystemPrompt                                YES (always)
//! Layer perceptual_ground                     YES (lower than target)
//!   ├── group type_specialist                 YES
//!   └── group structure_specialist            YES
//! Layer motivational                          partial
//!   ├── group active_mission     [TARGET]     YES
//!   └── group goal_pressure                   NO  (same layer, sibling)
//! Layer dialogue                              NO  (higher than target)
//!   └── group primary_conversation            NO
//! ```
//!
//! Masked nodes are excluded from reconciliation entirely. Their declared
//! minimums do not reserve budget; the visible nodes split the full window
//! between themselves.
//!
//! # Variable substitution
//!
//! YAML templates may carry `{name}` placeholders for values that vary per
//! deployment (workspace name, user identity, deployment environment).
//! Substitution happens **once**, at [`Builder::from_yaml_with_vars`]
//! construction. The resulting schema is immutable and carries the
//! substituted content forever — there is no projection-time
//! substitution path. This is load-bearing: the system-prompt KV cache is
//! prefilled from the rendered system prompt; if substitution varied
//! across projections the cache would silently invalidate.
//!
//! # Quick-start
//!
//! ```rust,ignore
//! use candle_conversation::projection::{Builder, ProjectionTarget};
//!
//! let mut builder = Builder::from_yaml_with_vars(
//!     SCHEMA_YAML,
//!     &[("workspace", "candle")],
//! )?;
//!
//! // Resolve human-readable names from YAML to opaque ids.
//! let dialogue_layer = builder.id_for_layer("dialogue").unwrap();
//! let conv_group    = builder.id_for_group("primary_conversation").unwrap();
//!
//! // Append turns as they arrive (content is stored by the caller, keyed by
//! // (GroupId, TurnIndex)).
//! let idx = builder.append(conv_group);
//! my_storage.put((conv_group, idx), turn_content);
//!
//! // At inference time, project for a target group:
//! let target = ProjectionTarget { layer: dialogue_layer, group: conv_group };
//! let projection = builder.project(target, &my_resolver);
//! //   projection.system_prompt — sections in declaration order
//! //   projection.turns         — ordered list of (group, index) to feed the LLM
//! ```
//!
//! # Module map
//!
//! | Submodule       | Purpose                                                        |
//! |-----------------|----------------------------------------------------------------|
//! | [`builder`]     | Public [`Builder`] type, lifecycle, projection entry-point     |
//! | [`schema`]      | All schema types ([`Schema`], [`LayerSchema`], …)              |
//! | [`ids`]         | [`LayerId`], [`GroupId`], [`SectionId`], [`TurnIndex`] newtypes |
//! | [`yaml`]        | YAML deserialisation + parse-time validation                   |
//! | [`error`]       | [`ConstructionError`] enum (parse + construction errors)       |
//! | [`resolver`]    | [`ContentResolver`] trait — only dynamic input to projection   |
//! | [`score`]       | [`ScoreFormula::aggregate`] — turn scores → group score        |
//! | [`selection`]   | All four selection rules, with budget-bounded variant          |
//! | [`reconcile`]   | CSS-flexbox-style budget distribution                          |
//! | [`project`]     | Full projection pipeline orchestrator                          |

mod builder;
mod error;
mod event;
mod ids;
mod policy;
mod project;
mod reconcile;
mod resolver;
mod schema;
mod score;
mod selection;
mod yaml;

#[cfg(test)]
mod tests;

// ── Public surface ────────────────────────────────────────────────────────────

pub use crate::substrate::{ContentResolver, Substrate, SubstrateRead, SubstrateWrite};
pub use builder::Builder;
pub use error::ConstructionError;
pub use event::{
    aggregate, decode_events, encode_events, from_projection, BucketKind, ProjectionBucket,
    ProjectionEvent, ProjectionSelection, SelectedSection, SelectedTurn, SelectionScores,
    SystemItem,
};
pub use ids::{
    CollectionId, GroupId, LayerId, Reserved, SectionId, TimelineAllocator, TimelineId, TurnId,
    TurnIndex, TurnKey,
};
pub use policy::{PolicyConfig, PolicyPreset, SelectionPolicy};
pub use project::{
    GeneratedIdentity, OptionalState, PriorBelief, Projection, ProjectionMode, ProjectionSegment,
    ProjectionTarget, ResolvedSection, ResolvedSelection, ResolvedTurn, SealedKind, SelectionState,
    NO_THINK_SELECTOR,
};
pub use reconcile::{EPSILON_TOKENS, MAX_ITERATIONS};
pub use resolver::{Conversation, TargetedRead};
pub use schema::{
    Budget, CompressionPrompt, GatherScope, GroupSchema, GroupSummary, GroupSummaryStage,
    LayerSchema, LayerSummary, Schema, ScoreFormula, SectionCollection, SectionSchema, SectionTree,
    SelectionRule, SummaryMode, SystemPromptItem, SystemPromptSchema, TreeDim, TreeNode,
    TreeOption, TreeVariant, TurnSummary,
};
