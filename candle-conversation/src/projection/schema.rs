//! Static schema types — declared once, immutable thereafter.
//!
//! A schema declares **what content can exist where**. After construction it
//! never changes; content is appended into groups over time, but the schema
//! itself is frozen. Everything dynamic (token counts, scores) flows through
//! the [`super::ContentResolver`] at projection time.
//!
//! # Hierarchy
//!
//! ```text
//!  Schema
//!  └── layers: Vec<LayerSchema>
//!      └── LayerSchema { name, window, score_threshold,
//!                        budget, system_prompt, groups: Vec<GroupSchema> }
//!          ├── system_prompt: SystemPromptSchema  { sections: [SectionSchema] }
//!          │   └── (used when THIS layer is the projection target)
//!          └── GroupSchema { name, selection, score_threshold, budget }
//! ```
//!
//! Each layer carries its own system prompt because each cognitive layer
//! (dialogue, bug analysis, dream exploration, daily convergence …) is a
//! genuinely different conversation with its own framing. The system
//! prompt emitted by [`super::Builder::project`] is the **target** layer's
//! sections.
//!
//! # Budget model
//!
//! Every layer and group carries a [`Budget`] declaring its **priority**
//! (relative weight for proportional allocation), an optional **min** and an
//! optional **max** (as percentages of the parent budget). The reconciler in
//! [`super::reconcile`] treats this CSS-flexbox-style: priorities determine
//! ideal share, mins and maxes are bounds.
//!
//! Sections have no budget — they always emit in declaration order in full.
//!
//! Each [`LayerSchema`] also carries a `window` field — the total
//! turn-budget that flexbox slices when **this layer is the projection
//! target**. Different targets get different pies.
//!
//! # Defaults
//!
//! When the YAML omits a field, the crate fills in a default:
//!
//! | Field                 | Default                 |
//! |-----------------------|-------------------------|
//! | `Budget.priority`     | `50.0` (any positive — all defaults equal = even split) |
//! | `Budget.min_percent`  | `None`                  |
//! | `Budget.max_percent`  | `None`                  |
//! | `score_threshold`     | `0.0`                   |
//! | selection rule        | [`SelectionRule::AlwaysVisible`] |
//!
//! `LayerSchema.window` has no default — it must be declared.

use super::ids::{CollectionId, GroupId, LayerId, SectionId};

/// Schema for one layer's system-prompt content.
///
/// Holds an ordered list of [`SystemPromptItem`]s — each item is either a
/// single [`SectionSchema`] (always emits) or a [`SectionCollection`] (a
/// named bucket with its own selection rule, only the surviving subset
/// emits).  Items emit in declaration order; sections inside a collection
/// also emit in their declaration order, after the collection's selection
/// rule has filtered them.
///
/// This shape lets a single layer's system prompt mix static framing
/// (always-emit) with dynamic catalogs (top-k filtered) at well-defined
/// positions — typical use is a chat dialect prompt with a `<tools>`
/// envelope: static intro section, then a `tools` collection that picks
/// the K most relevant tool definitions, then a static outro section.
///
/// Lives on [`LayerSchema::system_prompt`]. When projection runs for a
/// target `(layer, group)`, the emitted system prompt is **the target
/// layer's** items in declaration order.
#[derive(Debug, Clone, Default)]
pub struct SystemPromptSchema {
    /// In declaration order — interleaves single sections with
    /// collections.
    pub items: Vec<SystemPromptItem>,
}

impl SystemPromptSchema {
    /// Iterate every section in this system_prompt, regardless of
    /// whether it's a top-level item or nested inside a collection.
    /// Yields in declaration order.  Used for diagnostic walks; the
    /// projection emitter walks `items` directly so it can apply each
    /// collection's selection rule.
    pub fn all_sections(&self) -> impl Iterator<Item = &SectionSchema> {
        self.items.iter().flat_map(|it| match it {
            SystemPromptItem::Section(s) => std::slice::from_ref(s).iter(),
            SystemPromptItem::Collection(c) => c.sections.iter(),
        })
    }

    /// Find a [`SectionCollection`] by name, walking only top-level
    /// items.  Returns `None` for unknown names.
    pub fn collection_named(&self, name: &str) -> Option<&SectionCollection> {
        self.items.iter().find_map(|it| match it {
            SystemPromptItem::Collection(c) if c.name == name => Some(c),
            _ => None,
        })
    }

    /// True when `section_id` belongs to any `SectionCollection` in
    /// this layer's prompt.  Used by the content-address chain in
    /// `Sequence::insert_section_collection` to avoid hashing
    /// collection-member tokens into the prefix — without this
    /// filter, every change to a collection member (e.g. installing
    /// or removing a single tool) would cascade into a new
    /// `prefix_hash` for every downstream section, force-invalidating
    /// the manifest entries that would otherwise have cold-loaded.
    /// Collection members are an approximation-rich prefix anyway —
    /// projection picks a subset at runtime, so the section's K/V
    /// already isn't a strict function of which specific members
    /// ingested.  Treating them as outside the content chain matches
    /// that existing approximation.
    pub fn is_collection_member(&self, section_id: super::ids::SectionId) -> bool {
        self.items.iter().any(|it| match it {
            SystemPromptItem::Collection(c) => c.sections.iter().any(|s| s.id == section_id),
            _ => false,
        })
    }
}

/// One entry in a layer's system-prompt list.  Either a single
/// always-emit section or a named collection with its own selection
/// rule.
#[derive(Debug, Clone)]
pub enum SystemPromptItem {
    /// Authored static framing.  Always emits in declaration order
    /// regardless of any resolver-supplied scores.
    Section(SectionSchema),
    /// A named bucket of sections with its own selection rule.  Only
    /// the surviving subset emits, but in declaration order.
    Collection(SectionCollection),
}

/// A named bucket of system-prompt sections with its own selection rule.
///
/// Sections inside a collection are individually scored (typically via
/// per-section BDP sigs in the substrate) and filtered by
/// [`Self::selection`].  The surviving subset emits in declaration order
/// at the position of the collection within the system_prompt's items.
///
/// Typical use is a `tools` collection embedded in a dialogue layer's
/// system prompt: 93 tool-definition sections, `selection: TopK { k: 3 }`,
/// driven by BDP scoring against the user's recent intent.
#[derive(Debug, Clone)]
pub struct SectionCollection {
    /// Crate-assigned id.
    pub id: CollectionId,
    /// Original declared name (from YAML or builder injection).
    /// Layer-scoped — collections in different layers may share a name.
    pub name: String,
    /// Sections that belong to this collection.  Each retains its own
    /// [`SectionId`] — the collection is purely a selection-time
    /// grouping.
    pub sections: Vec<SectionSchema>,
    /// Which sections survive selection.  Default
    /// [`SelectionRule::AlwaysVisible`] (collection acts as a label
    /// only); typical TopK to surface the most relevant subset.
    pub selection: SelectionRule,
    /// Sections below this score are filtered before selection.
    /// Default `0.0`.
    pub score_threshold: f32,
    /// Per-collection BDP depth weights for section scoring.  When
    /// `Some`, overrides the enclosing layer's `depth_weights` for this
    /// collection's selection.  When `None`, falls back to the layer's
    /// value.  Allows different collections within the same layer to
    /// weight the three BDP bands independently (e.g. a `tools`
    /// collection calibrated for pragmatic-only while turn scoring uses
    /// semantic-heavy weights).
    pub depth_weights: Option<DepthWeights>,
}

impl Default for SectionCollection {
    fn default() -> Self {
        Self {
            id: CollectionId::new(1),
            name: String::new(),
            sections: Vec::new(),
            selection: SelectionRule::AlwaysVisible,
            score_threshold: 0.0,
            depth_weights: None,
        }
    }
}

/// A single authored section of the system prompt.
///
/// Always emits in declaration order when used as a top-level
/// `SystemPromptItem::Section`.  When nested inside a
/// [`SectionCollection`], emission is gated by the collection's
/// selection rule; the section's `priority` then breaks score ties.
#[derive(Debug, Clone)]
pub struct SectionSchema {
    /// Crate-assigned id.
    pub id: SectionId,
    /// Original declared `id:` string. Kept for diagnostics and name lookup.
    pub name: String,
    /// Authored text. The crate does not tokenize; the caller resolves
    /// content from `id` after projection emits a [`super::ResolvedSection`].
    pub content: String,
    /// Static fallback priority used as a score-tie breaker inside a
    /// collection's selection.  Higher = preferred.  Default `50.0`.
    pub priority: f32,
    /// Conditional emission gate: when `Some(cid)`, this section only
    /// emits at projection time if the named [`SectionCollection`] in
    /// the same layer materialised ≥ 1 of its members. Ingested
    /// unconditionally — the substrate always has its bytes — so the
    /// emission check is purely a projection-time predicate.
    ///
    /// Used by the YAML schema to wrap collections in structural
    /// markers (e.g. `<tools>` / `</tools>`) that should only appear
    /// when the collection itself emits anything.
    pub depends_on: Option<CollectionId>,
    /// Marks this section as resolved from a dialect template (a
    /// `kind: template` YAML item that referenced a `DialectTemplate`
    /// catalog entry, e.g. `system_start`).  The scheduler's
    /// projection assembler routes template-kind items through live
    /// prefill against the current runtime left context rather than
    /// the substrate-backed sealed path, so structural envelope K/V
    /// stays attention-correct under whatever prefix the projection
    /// selected this turn.
    pub is_template: bool,
    /// Pre-tokenised template content, populated by
    /// [`super::Builder::tokenize_templates`] before the first
    /// projection.  `Some` only when `is_template == true`; the
    /// projection engine emits a [`super::ProjectionSegment::Generated`]
    /// carrying these tokens for the assembler to inject as a live-
    /// prefilled run.  `None` for `is_template == false` sections —
    /// their K/V comes from the substrate-pinned sealed path.
    pub template_tokens: Option<std::sync::Arc<Vec<u32>>>,
}

/// Per-layer weighting for the three Binary Directional Provenance depths.
///
/// The BDP scanner produces a separate per-turn score for each depth
/// (syntactic ~15%, semantic ~50%, pragmatic ~85%).  This struct says how
/// the three are combined into a single per-turn score, computed as the
/// normalised weighted sum:
///
/// ```text
///   combined = (w_syn * s_syn + w_sem * s_sem + w_prag * s_prag)
///            / (w_syn + w_sem + w_prag)
/// ```
///
/// All weights must be non-negative; at least one must be > 0.
/// Default is `(1.0, 1.0, 1.0)` — the simple mean of the three depths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthWeights {
    pub syntactic: f32,
    pub semantic: f32,
    pub pragmatic: f32,
}

impl Default for DepthWeights {
    fn default() -> Self {
        // Universal calibration optimum (cross-corpus BDP sweep, 2026-05-16):
        // syn:1 / sem:1 / prag:4 → 0.167 / 0.167 / 0.667 normalised.
        // MRR=0.854, Top-1=81.6% across 640 probes × 64 items (8 layers).
        Self {
            syntactic: 1.0,
            semantic: 1.0,
            pragmatic: 4.0,
        }
    }
}

impl DepthWeights {
    /// Combine three per-depth scores into a single per-turn score.
    ///
    /// Returns `0.0` if all weights are zero (defensive — validation rejects
    /// this at construction time, but the math should still be safe).
    pub fn combine(&self, syn: f32, sem: f32, prag: f32) -> f32 {
        let total = self.syntactic + self.semantic + self.pragmatic;
        if total <= 0.0 {
            return 0.0;
        }
        (self.syntactic * syn + self.semantic * sem + self.pragmatic * prag) / total
    }
}

/// Schema for one cognitive layer.
///
/// A layer aggregates multiple [`GroupSchema`]s. Turn scores aggregate
/// into a per-group score via the fixed [`super::project::FIXED_FORMULA`],
/// used both for the layer-level threshold gate and for emission ordering
/// within the layer.
///
/// # Per-target turn budget
///
/// `window` declares the **total turn-budget** that flexbox distributes
/// across all visible layers **when this layer is the projection target**.
/// Different targets get different budgets — projecting for a thin "lower"
/// layer at 6K behaves identically to projecting for a 16K dialogue layer,
/// just with a smaller pie to slice. The flex priorities on layers and
/// groups still control the relative cuts; only the absolute size of the
/// pie changes.
#[derive(Debug, Clone)]
pub struct LayerSchema {
    /// Crate-assigned id, used by [`super::ProjectionTarget::layer`].
    pub id: LayerId,
    /// Original YAML `name:` string.
    pub name: String,
    /// Free-form description. Not used by the engine.
    pub description: String,
    /// Groups whose derived score is below this threshold are filtered from
    /// the layer before reconciliation. Default `0.0` (no gate).
    pub score_threshold: f32,
    /// Total turn-budget (in tokens) distributed across all visible layers
    /// when this layer is the projection target.
    pub window: usize,
    /// Flex weight when *some other layer* is the projection target and
    /// this layer is visible (lower than the target). Determines how much
    /// of the target's `window` this layer receives.
    pub budget: Budget,
    /// System-prompt sections framing the cognitive activity for which
    /// this layer is the projection target. **Required** at construction
    /// — every layer must declare at least one section so the layer is
    /// always usable as a projection target.
    pub system_prompt: SystemPromptSchema,
    /// Groups in declaration order. At projection time they are sorted by
    /// derived group score for emission.
    pub groups: Vec<GroupSchema>,
    /// Weights for combining per-depth BDP scores into a single per-turn
    /// score.  Default is equal weighting across all three depths.
    pub depth_weights: DepthWeights,
}

/// Schema for one group within a layer.
#[derive(Debug, Clone)]
pub struct GroupSchema {
    /// Crate-assigned id, **globally unique** across all layers in the schema.
    pub id: GroupId,
    /// Original YAML `id:` string.
    pub name: String,
    /// Which turns survive into the projection.
    pub selection: SelectionRule,
    /// Turns whose score is below this threshold are invisible to selection.
    /// Default `0.0` (no gate).
    pub score_threshold: f32,
    pub budget: Budget,
}

/// Flexbox-style token budget descriptor.
///
/// All percentages are **of the parent's resolved token budget** at
/// projection time, not of the global window. A min_percent of 30 on a group
/// inside a layer that received 1000 tokens means "at least 300 tokens for
/// this group."
///
/// # Reconciliation rules
///
/// - **`priority`** — relative weight when distributing the parent's budget.
///   Higher = larger share of remainder. Must be > 0.
/// - **`min_percent`** — floor. The flexbox distributor reserves at least
///   this fraction (subject to dynamic-shortfall proportional shrink if
///   sibling mins exceed 100%).
/// - **`max_percent`** — ceiling. Once a node hits its max, it's saturated
///   and excess budget redistributes to unsaturated siblings.
#[derive(Debug, Clone)]
pub struct Budget {
    /// Relative weight for proportional allocation. Must be > 0.
    pub priority: f32,
    /// Floor as a percent of parent budget (0–100).
    pub min_percent: Option<f32>,
    /// Ceiling as a percent of parent budget (0–100).
    pub max_percent: Option<f32>,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            priority: 50.0,
            min_percent: None,
            max_percent: None,
        }
    }
}

/// Which turns from a group survive into the projection.
///
/// # Decision tree
///
/// ```text
///   Selection rule ─┬─ AlwaysVisible        → all turns above threshold
///                   ├─ TopK { k }           → k highest-scored above threshold
///                   ├─ Single               → 1 highest-scored above threshold
///                   └─ Sequence         → recent-N (inviolate) + top-K historical
/// ```
///
/// All rules **emit in insertion order** regardless of selection order
/// (selection picks by relevance; emission preserves dialogue coherence).
///
/// See [`super::selection`] for the implementation.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionRule {
    /// Every turn in the group survives selection (subject to score threshold
    /// and budget). Used for sections where ordering is structural, not
    /// salience-driven.
    AlwaysVisible,

    /// The `k` highest-scored turns survive. Ties broken by lower
    /// `TurnIndex` (earlier wins).
    TopK { k: usize },

    /// The single highest-scored turn survives. Used for groups where only
    /// one entry is ever relevant at a time (a single goal pressure, a
    /// single active threat).
    Single,

    /// Composite for the natural shape of an ongoing conversation: the most
    /// recent `recent` turns survive **unconditionally** (no score threshold,
    /// no budget eviction), plus the top `historical_top_k` from the rest of
    /// the group by score.
    ///
    /// ```text
    ///   group.turns = [t0  t1  t2  t3  t4  t5  t6  t7]   (insertion order)
    ///                  ──────────────  ────────────────
    ///                   historical          recent
    ///                  (top-K by score)   (inviolate, last `recent`)
    /// ```
    Sequence {
        recent: usize,
        historical_top_k: usize,
    },
}

/// How turn scores are aggregated into a single group score.
///
/// The group score is used for:
/// 1. The **layer-level threshold gate** — groups whose derived score falls
///    below the layer's threshold are dropped entirely.
/// 2. **Emission ordering** — within a layer, groups are sorted by ascending
///    score so higher-scored groups appear LAST (closer to the model's
///    recency bias).
///
/// | Variant         | Behaviour                                              |
/// |-----------------|--------------------------------------------------------|
/// | `Max`           | Maximum turn score. Default. Robust to noise from low-scoring tail content. |
/// | `Sum`           | Sum of all turn scores. Larger groups dominate.        |
/// | `Mean`          | Arithmetic mean. Penalises noisy groups.               |
/// | `TopKMean { k }` | Mean of the top-`k` scores. Smoothed peak.            |
/// | `Count`         | Number of eligible turns. Score-independent salience.  |
/// | `Span { alpha }` | Σ L^α over consecutive runs of above-threshold probe positions. Rewards sustained relevance. |
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoreFormula {
    Max,
    Sum,
    Mean,
    TopKMean {
        k: usize,
    },
    Count,
    /// Power-law span scoring: consecutive runs of probe tokens that each find
    /// an above-threshold corpus match score L^α (default α=2.0).  Isolated
    /// hits score 1.0; a run of 3 scores 9.0.  The group-level aggregate
    /// (turn scores → group score) uses Max of per-turn span scores.
    Span {
        alpha: f32,
    },
    /// Per-token excess: Σ over probe tokens of `max(0, best_agreement − 64)`.
    /// Recentered on the random XOR-popcount baseline and reduced per probe
    /// token, with no hit threshold.  Calibrated as the strongest
    /// **prefill-phase** section-scoring formula — it recovers the weak,
    /// sub-threshold signal that `Span` (run-based) and `Max` (extreme-value)
    /// miss when the model is *reading* a query rather than generating.
    PerTokenExcess,
}

/// The complete parsed, validated schema. Immutable after construction.
///
/// All structural state lives on individual layers — there is no top-level
/// system prompt and no top-level token budget. Each layer carries its own
/// `window` (per-target turn budget) and `system_prompt` (framing for when
/// it is the target).
///
/// The schema does **not** carry dialect-specific structural tokens
/// (turn-boundary markers, role openers/closers).  Those are runtime
/// concerns owned by the scheduler / projection assembler — see
/// `crate::scheduler::projection_assembler::BoundaryMarkers`.
#[derive(Debug, Clone)]
pub struct Schema {
    /// Ordered: layer 0 first. Order is meaningful for masking and emission.
    pub layers: Vec<LayerSchema>,
}
