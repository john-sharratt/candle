//! The YAML shadow types for a layer and a group, and the conversions of their
//! gather, decode and corrupt-turn settings.
//!
//! A layer names its persisted id, its turn budget, its summary and its place in
//! the visibility stack, which defaults to its declaration order. A group names
//! its persisted id, its score gate and its selection. An absent threshold is not
//! the same as an explicit zero. Working-set layers carry their share, reserve and
//! fill settings. The enums map the author's words, such as `drop_turn` and
//! `drop_conversation`, onto the engine's policies.

use super::*;

#[derive(Deserialize)]
pub(super) struct YamlLayer {
    pub(super) name: String,
    /// The id this layer's conversations are persisted under — see
    /// `persisted_ids::assign`.
    #[serde(default)]
    pub(super) persisted_id: Option<u32>,
    #[serde(default)]
    pub(super) description: String,
    /// Total turn-budget when this layer is the projection target.
    pub(super) window: usize,
    /// The already-ingested content a conversation targeting this layer carries
    /// ahead of its own turns — see [`LayerSchema::working_set`].
    #[serde(default)]
    pub(super) working_set: Option<YamlWorkingSet>,
    #[serde(default)]
    pub(super) score_threshold: f32,
    #[serde(default)]
    pub(super) budget: YamlBudget,
    /// Per-layer overrides of the unified `system_prompt`'s section-tree
    /// selectors, applied when this layer is the projection target. A map of
    /// `<selector_id>: <option_id>` (e.g. `thinking_effort: exhaustive`). Keys
    /// are the section-tree node names verbatim; a key matching no selector is
    /// inert at projection time and available to the host. Omitted → inherit the
    /// section-tree defaults.
    #[serde(default)]
    pub(super) dials: std::collections::BTreeMap<String, String>,
    /// Required — how this layer's turns are compressed across the summary tree
    /// (`turns` + optional `summaries`, each with question/answer halves).
    pub(super) summary: YamlLayerSummary,
    pub(super) groups: Vec<YamlGroup>,
    /// Selection policy for this layer's turn groups; inherits the schema
    /// default when absent. Collections/groups may override it.
    #[serde(default)]
    pub(super) policy: Option<YamlPolicy>,
    /// Gather-tree scope for this layer's turns. `shared` (default) → one tree
    /// across all conversations; `conversation` → one private tree per conversation
    /// (the dialogue layer).
    #[serde(default)]
    pub(super) gather_scope: YamlGatherScope,
    /// Continuous-fair-wave decode priority. `low` (default) → prefill drains at
    /// full speed; `normal` / `high` throttle a co-running prefill so more decode
    /// tokens land per completed prefill (the dialogue layer is `high`).
    #[serde(default)]
    pub(super) decode_priority: YamlDecodePriority,
    /// What to do with a turn that's unrecoverable on reload. `drop_conversation`
    /// (default) tombstones the whole conversation; `drop_turn` tombstones only the
    /// corrupt turn — set on the dialogue layer so one bad turn doesn't drop the chat.
    #[serde(default)]
    pub(super) on_corrupt_turn: YamlCorruptTurnPolicy,
    /// Noun for this layer's startup-ingest progress readout ("N / M <unit>" on
    /// the loading screen) — e.g. `files`, `folders`. Omitted → a mode-derived
    /// default in the ingest driver; non-ingest layers ignore it.
    #[serde(default)]
    pub(super) ingest_unit: Option<String>,
    /// The host ingest's own description of how the layer is filled — see
    /// [`LayerSchema::ingest`]. Passed through unread.
    #[serde(default)]
    pub(super) ingest: Option<serde_yaml::Value>,
    /// The host's calibration of the layer's conversations — see
    /// [`LayerSchema::calibrate`]. Passed through unread.
    #[serde(default)]
    pub(super) calibrate: Option<serde_yaml::Value>,
    /// Whether the layer's turns project into tool rounds — see
    /// [`LayerSchema::in_tool_rounds`]. Omitted → `true`.
    #[serde(default)]
    pub(super) in_tool_rounds: Option<bool>,
    /// The layer's place in the visibility stack — see [`LayerSchema::rank`].
    /// Omitted → its declaration index.
    #[serde(default)]
    pub(super) rank: Option<i32>,
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub(super) enum YamlGatherScope {
    #[default]
    Shared,
    Conversation,
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub(super) enum YamlDecodePriority {
    #[default]
    Low,
    Normal,
    High,
    Urgent,
}

impl From<YamlDecodePriority> for DecodePriority {
    fn from(y: YamlDecodePriority) -> Self {
        match y {
            YamlDecodePriority::Low => DecodePriority::Low,
            YamlDecodePriority::Normal => DecodePriority::Normal,
            YamlDecodePriority::High => DecodePriority::High,
            YamlDecodePriority::Urgent => DecodePriority::Urgent,
        }
    }
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub(super) enum YamlCorruptTurnPolicy {
    #[default]
    DropConversation,
    DropTurn,
}

impl From<YamlCorruptTurnPolicy> for CorruptTurnPolicy {
    fn from(y: YamlCorruptTurnPolicy) -> Self {
        match y {
            YamlCorruptTurnPolicy::DropConversation => CorruptTurnPolicy::DropConversation,
            YamlCorruptTurnPolicy::DropTurn => CorruptTurnPolicy::DropTurn,
        }
    }
}

impl From<YamlGatherScope> for GatherScope {
    fn from(s: YamlGatherScope) -> Self {
        match s {
            YamlGatherScope::Shared => GatherScope::Shared,
            YamlGatherScope::Conversation => GatherScope::Conversation,
        }
    }
}

#[derive(Deserialize)]
pub(super) struct YamlGroup {
    pub(super) id: String,
    /// The id this group's conversations are persisted under — see
    /// `persisted_ids::assign`.
    #[serde(default)]
    pub(super) persisted_id: Option<u32>,
    #[serde(default)]
    pub(super) selection: YamlSelection,
    /// Belief/selection gate. Absent is NOT the same as `0.0`: an absent
    /// threshold leaves the group's `policy:` score band in force, while an
    /// explicit `0.0` means "no gate" and overrides it.
    #[serde(default)]
    pub(super) score_threshold: Option<f32>,
    #[serde(default)]
    pub(super) budget: YamlBudget,
    /// Selection policy override; inherits the enclosing layer's when absent.
    #[serde(default)]
    pub(super) policy: Option<YamlPolicy>,
    /// Concept B: mass-driven member-budget extension.
    #[serde(default)]
    pub(super) budget_adaptive: Option<YamlMemberBudgetAdaptive>,
    /// Concept C: neighbor drag.
    #[serde(default)]
    pub(super) locality: Option<YamlLocality>,
    /// Concept D: timeline anchor member.
    #[serde(default)]
    pub(super) anchor: Option<YamlAnchor>,
    /// Newcomers a mid-decode reprojection may admit; [`DEFAULT_MAX_SWAPS`]
    /// when absent.
    #[serde(default)]
    pub(super) max_swaps: Option<usize>,
    /// The units of an earlier group of this layer this group places instead.
    #[serde(default)]
    pub(super) draws: Option<YamlDraws>,
}

/// YAML shadow of [`Draws`] — `draws: { from: scopes, paths: [...], except: [...] }`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct YamlDraws {
    pub(super) from: String,
    pub(super) paths: Vec<String>,
    #[serde(default)]
    pub(super) except: Vec<String>,
}

/// YAML shadow of [`SelectionDefault`] — `default: { tag: "…" }`.
#[derive(Deserialize)]
pub(super) struct YamlSelectionDefault {
    pub(super) tag: String,
}

#[derive(Deserialize)]
pub(super) struct YamlMemberBudgetAdaptive {
    pub(super) per_extra: f32,
    pub(super) absolute_max: usize,
}

#[derive(Deserialize)]
pub(super) struct YamlLocality {
    pub(super) seed_threshold: f32,
    #[serde(default = "default_locality_decay")]
    pub(super) decay: f32,
    #[serde(default = "default_locality_base_radius")]
    pub(super) base_radius: usize,
    #[serde(default)]
    pub(super) extend_per: f32,
    #[serde(default)]
    pub(super) extra_radius_max: usize,
}

pub(super) fn default_locality_decay() -> f32 {
    0.5
}

pub(super) fn default_locality_base_radius() -> usize {
    1
}

#[derive(Deserialize)]
pub(super) struct YamlAnchor {
    pub(super) member: String,
}

#[derive(Deserialize, Default)]
pub(super) struct YamlBudget {
    #[serde(default)]
    pub(super) priority: Option<f32>,
    #[serde(default)]
    pub(super) min_percent: Option<f32>,
    #[serde(default)]
    pub(super) max_percent: Option<f32>,
    /// Concept B: mass-modulated priority within the declared rails.
    #[serde(default)]
    pub(super) adaptive: Option<YamlBudgetAdaptive>,
}

#[derive(Deserialize)]
pub(super) struct YamlBudgetAdaptive {
    pub(super) gain: f32,
    #[serde(default)]
    pub(super) max_percent: Option<f32>,
}

#[derive(Deserialize, Default)]
pub(super) struct YamlSelection {
    #[serde(default)]
    pub(super) kind: String,
    #[serde(default)]
    pub(super) k: Option<usize>,
    #[serde(default)]
    pub(super) recent: Option<usize>,
    #[serde(default)]
    pub(super) historical_top_k: Option<usize>,
    /// `kind: birth` only — the tokens of past exchanges a request opens with.
    #[serde(default)]
    pub(super) window: Option<usize>,
    /// `kind: conversation` only — the score an inherited lineage turn is
    /// never ranked below; [`LINEAGE_BASE`] when absent.
    #[serde(default)]
    pub(super) lineage_base: Option<f32>,
    #[serde(default)]
    pub(super) selector: Option<String>,
    /// `kind: working_set` only — `folders` or `remainder`.
    #[serde(default)]
    pub(super) share: Option<String>,
    /// `kind: working_set` only — how its units fade with age.
    #[serde(default)]
    pub(super) recency: Option<YamlRecency>,
}

/// YAML shadow of [`Recency`] — `recency: { half_life_days, dated_by }`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct YamlRecency {
    pub(super) half_life_days: f32,
    pub(super) dated_by: String,
}

/// A dialogue layer's `working_set:` block — see [`WorkingSetConfig`].
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct YamlWorkingSet {
    pub(super) beta: f32,
    pub(super) min_momentum: f32,
    /// What the slot keeps back for what arrives between two projections:
    /// one cadence of decoded tokens or one prefill chunk.
    pub(super) reserve_tokens: usize,
    /// The most of the pool's tokens each type of document may hold; none
    /// when absent.
    #[serde(default)]
    pub(super) type_shares: Vec<YamlTypeShare>,
    /// Each kind's share of the slot's room, the fill's caps.
    pub(super) shares: YamlFillShares,
}

/// A `working_set.shares` block — see [`FillShares`].
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct YamlFillShares {
    pub(super) readings: f64,
    pub(super) folders: f64,
    pub(super) guides: f64,
    pub(super) journals: f64,
    pub(super) dialogue: f64,
}

/// One `working_set.type_shares` entry.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct YamlTypeShare {
    pub(super) name: String,
    /// Regexes over a unit's subject; the first type one matches is its type.
    pub(super) files: Vec<String>,
    pub(super) max_share: f32,
}
