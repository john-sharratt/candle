//! The schema of one cognitive layer: the turn budget it receives as a
//! projection target, the gates on its groups, and how its turns are compressed
//! and scheduled.
//!
//! A layer's `window` is the total turn budget its groups are flexed across when
//! it is the projection target, and its flex weight sets how much of another
//! target's window it takes when it is only visible. `rank` decides which layers
//! a projection can see. The gathered flag is an operator's runtime decision and
//! is not part of the document: a layer disabled for a boot is taken out of
//! retrieval altogether. `DecodePriority` sets the decode-to-prefill airtime
//! ratio the scheduler gives the layer, and `CorruptTurnPolicy` decides whether an
//! unrecoverable turn drops its whole conversation or only itself.

use super::*;

/// How a layer's turns are scoped into gather (provenance) trees.
///
/// - [`GatherScope::Shared`] (default): one tree per layer, shared across all
///   conversations — institutional memory / repo map / motivation.
/// - [`GatherScope::Conversation`]: one tree per conversation (keyed by its
///   `TimelineId`) — the live dialogue layer, private to each conversation.
///
/// Section groups are always per-collection regardless of this flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GatherScope {
    /// One shared tree across all conversations.
    #[default]
    Shared,
    /// One private tree per conversation.
    Conversation,
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
///
/// # Decode priority
///
/// `decode_priority` sets how strongly the continuous-fair-wave scheduler
/// favours this layer's *decode* over a co-running background prefill — see
/// `docs/continuous_fair_waves.md`. It is the decode-to-prefill airtime ratio
/// `R`: the number of decode tokens produced for each prefill forward that
/// completes. A conversation inherits its target layer's priority.
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
    /// The already-ingested content a conversation targeting this layer
    /// carries ahead of its own turns (`working_set:` in YAML,
    /// `docs/zend_working_set.md`). `None` — every layer but the dialogue — and
    /// its `working_set` groups select nothing.
    pub working_set: Option<WorkingSetConfig>,
    /// Flex weight when *some other layer* is the projection target and
    /// this layer is visible (lower than the target). Determines how much
    /// of the target's `window` this layer receives.
    pub budget: Budget,
    /// Per-layer overrides of the unified [`Schema::system_prompt`]'s
    /// section-tree selectors, applied when this layer is the projection target.
    /// Empty inherits the section-tree defaults. The layer does NOT carry its own
    /// system prompt — the framing is shared so its sealed K/V is reused across
    /// every layer; a layer differs only by which pre-sealed branch it selects.
    pub dials: LayerDials,
    /// How this layer's turns are compressed across the summary tree — the
    /// `turns`/`summaries` tree-levels, each splitting the question and answer
    /// halves into their own framing + instruction, with a per-level decode cap.
    pub summary: LayerSummary,
    /// Groups in declaration order. At projection time they are sorted by
    /// derived group score for emission.
    pub groups: Vec<GroupSchema>,
    /// Selection policy (belief-update + budget + tag scope) for this layer's
    /// turn groups, resolved from the layer's `policy:` or inherited from the
    /// schema default. Collections and groups may override it.
    pub policy: SelectionPolicy,
    /// Gather-tree scoping for this layer's turns (`gather_scope:` in YAML).
    /// Default [`GatherScope::Shared`].
    pub gather_scope: GatherScope,
    /// Whether this layer's turns compete in the provenance gather — and, since
    /// a `Sequence` group is never scored, whether the layer takes part in
    /// projection assembly at all (`project::run` skips a non-target layer that
    /// is not gathered, so a recency window cannot be emitted behind the
    /// gather's back). `true` for
    /// every layer the schema declares; set `false` at RUNTIME only, by
    /// [`super::Builder::set_layer_gathered`].
    ///
    /// **Not a YAML field.** A schema declares what a projection *is*; this
    /// records an operator's decision for one boot (`zend --disable-layer`), so
    /// it has no place in the document. It is the difference between the two
    /// flags: `--skip-layer` leaves a layer gathered and merely stops
    /// populating it, so its existing turns keep answering queries and its
    /// backgrounds keep being learned; `--disable-layer` takes it out of
    /// retrieval altogether.
    ///
    /// Read by [`super::Conversation::score_belief_groups`] and by both
    /// normalization warm-ups — a layer that cannot be selected must not teach
    /// backgrounds either, or it would spend the warm-up's probe budget learning
    /// denominators for candidates that can never be returned.
    pub gathered: bool,
    /// Continuous-fair-wave decode priority (`decode_priority:` in YAML).
    /// Default [`DecodePriority::Low`]. The dialogue layer is `Urgent`.
    pub decode_priority: DecodePriority,
    /// What to do when one of this layer's turns is unrecoverable on reload
    /// (`on_corrupt_turn:` in YAML). Default
    /// [`CorruptTurnPolicy::DropConversation`]; the dialogue layer sets `drop_turn`.
    pub on_corrupt_turn: CorruptTurnPolicy,
    /// Noun for this layer's startup-ingest progress readout (`ingest_unit:` in
    /// YAML) — e.g. `"files"`, `"folders"`. Shown on the loading screen as
    /// "N / M <unit>" while the layer ingests. `None` falls back to a mode-derived
    /// default in [`crate`]'s ingest driver; non-ingest layers ignore it.
    pub ingest_unit: Option<String>,
    /// Whether this layer's turns are projected into a **tool round** — a turn
    /// whose user message is the results of calls the turn before made, marked
    /// by [`super::TOOL_ROUND_SELECTOR`] (`in_tool_rounds:` in YAML, default
    /// `true`).
    ///
    /// A tool round's last user turn is a tool response, so the model looks back
    /// through the context for the question it is answering. An ingest layer's
    /// turns are requests too — "Summarize the root folder of this project in
    /// one or two complete sentences" — and a round that found one nearer than
    /// the user's answered it instead: a chat asked to list a crate's modules
    /// fetched the page and then described the workspace. `false` keeps such a
    /// layer out of every tool round; the turn that opened the round, where the
    /// question is the user's own last message, still sees it.
    pub in_tool_rounds: bool,
    /// Where this layer sits in the stack a projection sees down through
    /// (`rank:` in YAML): a projection targeting a layer sees every layer of
    /// **lower** rank, and none of equal or higher rank but its own.
    ///
    /// Defaults to the layer's declaration index, which is the stack every
    /// schema had before this existed. It is its own field because declaration
    /// order also fixes every layer's and group's id, and those ids are
    /// persisted with every timeline written under them — so a layer added to a
    /// live schema has to be appended, and without a rank an appended layer
    /// could only ever sit on top, invisible to everything declared before it.
    pub rank: i32,
    /// How the layer is populated by its host's ingest (`ingest:` in YAML),
    /// carried through unread: what a unit is, what it is asked and what it
    /// builds on are the host's to interpret, never the projection's. `None`
    /// for a layer the host fills some other way, or not at all.
    pub ingest: Option<serde_yaml::Value>,
    /// How the host calibrates the layer's conversations for provenance
    /// (`calibrate:` in YAML), carried through unread like [`Self::ingest`]:
    /// what each conversation is asked about itself and how many exemplars it
    /// lends are the host's. `None` for a layer that is not calibrated.
    pub calibrate: Option<serde_yaml::Value>,
}

/// How strongly a layer's decode is favoured over a co-running background
/// prefill in the continuous-fair-wave scheduler (`docs/continuous_fair_waves.md`).
///
/// The variant maps to the **decode-to-prefill airtime ratio** `R`
/// ([`Self::ratio`]) — decode tokens produced per completed prefill forward.
/// Higher priority ⇒ larger `R` ⇒ the prefill is throttled to creep further in
/// the background while more decode tokens land, protecting interactive latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodePriority {
    /// `R = 1`: prefill drains at full speed, least decode protection. Default —
    /// the right choice for bulk-ingest / background layers.
    #[default]
    Low,
    /// `R = 16`: prefill stretched ~16× while 16 decode tokens land.
    Normal,
    /// `R = 64`: prefill heavily throttled; decode latency maximally protected.
    /// Work somebody is waiting on that is not a live conversation — a Maker's
    /// table call, a task delegated at `high`.
    High,
    /// `R = 64`, and **above `High`**: a person is waiting on this turn right
    /// now — zend's live dialogue, a passthrough client's turn, the content
    /// check an image draw waits on. Paced like `High` in every other respect
    /// ([`Self::is_live`]); it only outranks it.
    Urgent,
}

impl DecodePriority {
    /// The conversation metadata key a conversation's own band is kept under,
    /// in place of its target layer's: a task delegated to a dialogue-layer
    /// conversation decodes below a live user unless someone waits on it.
    pub const METADATA_KEY: &'static str = "decode_band";

    /// The band's name, as YAML and the metadata spell it.
    pub fn name(self) -> &'static str {
        match self {
            DecodePriority::Low => "low",
            DecodePriority::Normal => "normal",
            DecodePriority::High => "high",
            DecodePriority::Urgent => "urgent",
        }
    }

    /// The band `name` spells; `None` for anything else.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "low" => Some(DecodePriority::Low),
            "normal" => Some(DecodePriority::Normal),
            "high" => Some(DecodePriority::High),
            "urgent" => Some(DecodePriority::Urgent),
            _ => None,
        }
    }

    /// Whether the band is one somebody waits on token by token — `High` or
    /// `Urgent`. Everywhere the scheduler protects latency rather than
    /// throughput (a group of live prompts only, the split by backlog, the
    /// prompts a placed turn waits behind) the two are the same; only their
    /// rank differs.
    pub fn is_live(self) -> bool {
        matches!(self, DecodePriority::High | DecodePriority::Urgent)
    }

    /// The decode-to-prefill airtime ratio `R`: how many decode tokens are
    /// produced for each prefill forward that completes. Always `>= 1`.
    pub fn ratio(self) -> u32 {
        match self {
            DecodePriority::Low => 1,
            DecodePriority::Normal => 16,
            DecodePriority::High | DecodePriority::Urgent => 64,
        }
    }
}

/// What to do with one of a layer's turns whose persisted state is unrecoverable
/// on substrate reload (e.g. a partial write that leaves a chunk-count mismatch).
/// Set per layer via `on_corrupt_turn:` in the projection YAML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CorruptTurnPolicy {
    /// Tombstone the whole **timeline** (conversation). Correct for ingest layers
    /// (code_read / repo_map), where one file is one timeline and the tombstone
    /// hands the file back to the background refresh to re-ingest. The default.
    #[default]
    DropConversation,
    /// Tombstone only the **corrupt turn**, leaving the rest of the conversation
    /// live. Correct for multi-turn dialogue, where one bad turn should not drop
    /// the whole chat.
    DropTurn,
}

impl LayerSchema {
    /// Every `SectionId` (as raw `u32`) this layer allocates for its own content:
    /// the `turns`/`summaries` × `question`/`answer` compression prompts. The
    /// unified system-prompt sections belong to the schema, not the layer — see
    /// [`Schema::all_section_ids`].
    pub fn all_section_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::new();
        self.summary.push_section_ids(&mut ids);
        ids
    }
}

/// Per-layer overrides for the unified [`Schema::system_prompt`]'s section-tree
/// selectors, applied when THIS layer is the projection target and *beneath* any
/// per-turn override the caller supplies.
///
/// Each entry is `(selector_id, option_id)`. The selector ids are exactly the
/// section-tree node names in the unified system prompt (e.g. `thinking_effort`,
/// `response_length`), so the set is data-driven — the engine hardcodes no dial
/// names. A key that matches no tree selector is inert at projection time and is
/// available to the host (e.g. zend reads a `tools` catalog dial here) via
/// [`Self::get`]. An empty set inherits the section-tree's authored defaults.
#[derive(Debug, Clone, Default)]
pub struct LayerDials {
    pub(super) dials: Vec<(String, String)>,
}

impl LayerDials {
    /// Build from `(selector_id, option_id)` pairs in declaration order.
    pub fn from_pairs(dials: Vec<(String, String)>) -> Self {
        Self { dials }
    }

    /// The option chosen for `selector`, if this layer overrides it.
    pub fn get(&self, selector: &str) -> Option<&str> {
        self.dials
            .iter()
            .find(|(k, _)| k == selector)
            .map(|(_, v)| v.as_str())
    }

    /// `(selector_id, option_id)` pairs in declaration order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> + '_ {
        self.dials.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// No overrides — every selector falls back to its section-tree default.
    pub fn is_empty(&self) -> bool {
        self.dials.is_empty()
    }
}
