//! Construction-time errors.
//!
//! Projection itself never errors — it is pure given a valid schema and a
//! working resolver. All failures surface at parse or construction time.
//!
//! # Error categories
//!
//! ```text
//!   ┌──────────────────────┬─────────────────────────────────────────────┐
//!   │ Phase                │ Errors                                      │
//!   ├──────────────────────┼─────────────────────────────────────────────┤
//!   │ Parse-time           │ YamlParse                                   │
//!   │   (yaml.rs)          │ DuplicateGroupName                          │
//!   │                      │ DuplicateSectionName                        │
//!   │                      │ UnknownSelectionKind                        │
//!   │                      │ InvalidPriority      (priority <= 0)        │
//!   │                      │ InvalidPercentage    (outside 0..=100)      │
//!   │                      │ InvalidTopK          (k missing or 0)       │
//!   │                      │ InvalidConversationK (recent=0 AND tk=0)    │
//!   ├──────────────────────┼─────────────────────────────────────────────┤
//!   │ Construction-time    │ MinPercentExceedsTotal                      │
//!   │   (builder.rs)       │ MaxLessThanMin                              │
//!   │                      │ NegativeScoreThreshold                      │
//!   │                      │ EmptyLayerSystemPrompt                      │
//!   ├──────────────────────┼─────────────────────────────────────────────┤
//!   │ Projection-time      │ (none — pure given valid inputs)            │
//!   └──────────────────────┴─────────────────────────────────────────────┘
//! ```

/// Errors from parsing YAML or constructing a [`super::Builder`].
///
/// All variants implement [`std::error::Error`] via thiserror. Diagnostics
/// include the offending node's name where applicable.
#[derive(Debug, thiserror::Error)]
pub enum ConstructionError {
    // ── Parse-time ────────────────────────────────────────────────────────────
    /// The YAML document was malformed or missing a required field.
    #[error("YAML parse error: {0}")]
    YamlParse(String),

    /// Two groups in the schema share the same string name. Group names must
    /// be unique across the whole schema (not per-layer).
    #[error("duplicate group name {0:?} — group ids must be unique across the whole schema")]
    DuplicateGroupName(String),

    /// Two sections share the same string name.
    #[error("duplicate section name {0:?} — section ids must be unique")]
    DuplicateSectionName(String),

    /// Two collections in the same layer share the same string name.
    #[error("duplicate section-collection name {0:?} within a layer")]
    DuplicateCollectionName(String),

    /// Selection rule kind is not in the closed set
    /// (`always_visible`, `top_k`, `single`, `conversation`).
    #[error("unknown selection kind {0:?}")]
    UnknownSelectionKind(String),

    /// Priority must be a positive float.
    #[error("priority for {name:?} must be positive, got {value}")]
    InvalidPriority { name: String, value: f32 },

    /// `min_percent` and `max_percent` are bounded to 0..=100.
    #[error("percentage for {name:?} must be in 0..=100, got {value}")]
    InvalidPercentage { name: String, value: f32 },

    // ── Construction-time ─────────────────────────────────────────────────────
    /// Sum of declared `min_percent` across siblings exceeds 100. Statically
    /// infeasible — the configured floors cannot all be honoured.
    #[error("sum of min_percent for siblings of {parent:?} is {sum:.1}, exceeds 100")]
    MinPercentExceedsTotal { parent: String, sum: f32 },

    /// `max_percent < min_percent` for a single node.
    #[error("max_percent < min_percent for {name:?}")]
    MaxLessThanMin { name: String },

    /// Negative `score_threshold`. Thresholds are eligibility floors.
    #[error("score_threshold for {name:?} must be >= 0, got {value}")]
    NegativeScoreThreshold { name: String, value: f32 },

    /// A layer declared a `system_prompt:` block with no sections. Every
    /// layer must declare at least one section — even an analysis phase
    /// that hasn't yet been wired up should have a stub frame so the
    /// projection target is always usable.
    #[error("layer {layer:?} must declare at least one system_prompt section")]
    EmptyLayerSystemPrompt { layer: String },

    /// A `{name}` placeholder remained in the YAML template after
    /// substitution — either the caller forgot to supply that variable, or
    /// the template has a typo (`{wokrspace}` vs `{workspace}`).
    ///
    /// All variable substitution happens **once** at builder construction
    /// time. After that the schema is immutable, so a missed substitution
    /// here is the difference between the system-prompt KV cache being
    /// stable across invocations and being silently broken.
    #[error(
        "unresolved {{{name}}} placeholder in YAML template — \
             pass it via Builder::from_yaml_with_vars or fix the template"
    )]
    UnresolvedVariable { name: String },

    /// `top_k` selection had a missing or zero `k`.
    #[error("top_k k must be >= 1 for {name:?}")]
    InvalidTopK { name: String },

    /// `conversation` selection had both `recent = 0` and
    /// `historical_top_k = 0` (no turns can ever survive).
    #[error("conversation rule for {name:?} must have recent >= 1 or historical_top_k >= 1")]
    InvalidConversationK { name: String },

    /// A depth weight in the `depth_weights` block was negative.
    #[error("depth_weights for {layer:?} must all be >= 0, got {value} for {depth}")]
    NegativeDepthWeight {
        layer: String,
        depth: &'static str,
        value: f32,
    },

    /// All three depth weights were zero. At least one must be > 0 so the
    /// combine formula has a non-zero denominator.
    #[error("depth_weights for {layer:?} must have at least one positive weight")]
    AllDepthWeightsZero { layer: String },

    /// Runtime mutator (e.g. [`super::Builder::add_section`]) was given an
    /// id that no layer in the schema owns.
    #[error("unknown layer {0:?}")]
    UnknownLayer(String),

    /// Runtime mutator (e.g. [`super::Builder::add_section_to_collection`])
    /// was given a collection id that no layer in the schema owns.
    #[error("unknown collection {0:?}")]
    UnknownCollection(String),

    /// A `kind: template` item referenced a `dialect:` name that
    /// [`candle_transformers::models::dialect::DialectTemplate::from_yaml_name`]
    /// could not resolve.
    #[error("unknown dialect template {name:?} on item {item:?}")]
    UnknownDialectTemplate { item: String, name: String },

    /// A `kind: template` item appeared in the YAML but no
    /// [`Dialect`](candle_transformers::models::dialect::Dialect) was
    /// supplied to the parser. Use
    /// [`super::Builder::from_yaml_with_vars_and_dialect`] when the
    /// schema uses template items.
    #[error(
        "schema contains `kind: template` item {item:?} but no dialect was \
         supplied to the parser — use Builder::from_yaml_with_vars_and_dialect"
    )]
    DialectRequired { item: String },
}
