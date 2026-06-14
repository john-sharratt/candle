//! Core identity and tag types: [`TurnType`], [`StorageTier`],
//! [`TurnId`], [`SegmentId`], [`NodeId`].
//!
//! # Temporal marker format `[T-{day}.{seq}]`
//!
//! The marker format is designed to be recognizable across the LLM training
//! distribution:
//!
//! - **Bracket notation `[...]`**: signals inline metadata, not conversation
//!   content. Chat logs, IRC transcripts, roleplay annotations
//!   (`[OOC: ...]`, `[Action: ...]`), and compiler diagnostics all use this
//!   convention. Angle brackets collide with HTML/XML and template tokens
//!   (`<|im_start|>`, `<think>`); curly braces appear in template engines;
//!   `#` reads as Markdown heading.
//!
//! - **`T-` prefix**: standard notation for elapsed time across the training
//!   distribution — countdown sequences (T-minus), statistical notation
//!   (T-test), time-series academic papers. The model interprets `T-` as
//!   "a temporal quantity." It is also the exact prefix used in the
//!   *Attention-Organized Sequence Trees* paper.
//!
//! - **`day` component (integer)**: elapsed UTF calendar days since the
//!   reference instant captured at tree construction. Day 0 = opening day.
//!   Stays compact indefinitely (3 digits covers 2.7 years). Combines with
//!   `T-` to read as "T-minus N days" — common elapsed-time framing.
//!
//! - **`seq` component (monotonic integer)**: global turn-pair counter, never
//!   resets across day boundaries. `T-3.47` = 48th turn overall, on day 3.
//!   Every marker is globally unique; the model can cross-reference:
//!   "as we discussed in T-3.47". Turn density per day is readable from
//!   adjacent markers: `T-3.47` → `T-4.52` implies ~5 turns on day 4.

// ────────────────────────────────────────────────────────────────────────────
// TurnType — cognitive mode
// ────────────────────────────────────────────────────────────────────────────

/// The cognitive mode / pipeline that produced a turn.
///
/// All turns created by the live user↔assistant request loop are `Reality`.
/// The remaining variants are reserved for async pipelines that run alongside
/// the main conversation and inject turns into the tree when they complete.
/// Defined now so those pipelines slot in without changing any struct.
///
/// `TurnType` determines how a turn is weighted in summarization, how it
/// appears in debug output, and which async pipeline owns it.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TurnType {
    /// Live interaction in the character's active world — the normal
    /// request/response loop. This is the default and the only variant
    /// currently used; the async-pipeline variants are TODO.
    #[default]
    Reality,

    /// End-of-day prospective simulation — the limbic layer's deepest process.
    ///
    /// Distinct from summarization: summarization is retrospective compression
    /// (takes what happened and makes it smaller). Sleep is prospective
    /// simulation (takes what happened and runs it forward, sideways, and into
    /// the extreme).
    ///
    /// A Sleep cycle generates a large batch of short parallel dreams from the
    /// day's [`Reality`](TurnType::Reality) turns as seed material — typically
    /// 20–50, running simultaneously via the engine's batched decode
    /// capability, each 100–200 tokens. Three categories:
    ///
    /// - **Counterfactual** — "What if I had done this instead?" Regret and
    ///   relief simulations.
    /// - **Prospective** — "Let's play this one forward." Simulated futures of
    ///   unresolved situations. These surface through the attention system and
    ///   colour future Reality responses.
    /// - **Extreme** — "Let's take some extremes." Worst-case, best-case, most
    ///   absurd implication. Visceral understanding of what is really at stake.
    ///
    /// After generation, each dream is scored against the character's core
    /// identity nodes using attention statistics. Dreams that resonate are
    /// inserted into the tree as `Sleep` turns; the rest are discarded. The
    /// survivors are emotionally indexed simulations the attention system can
    /// surface when similar situations arise in future Reality turns.
    ///
    /// Sleep batch size is configured via
    /// [`ConversationTreeConfig::sleep_batch_size`](super::config::ConversationTreeConfig::sleep_batch_size).
    ///
    /// TODO: Sleep pipeline not yet implemented.
    Sleep,

    /// Spontaneous associative flicker between Reality turns — the limbic
    /// layer's lightest process.
    ///
    /// **Trigger:** after each Reality turn completes, the last user message
    /// is probed against cold node representative K vectors in the tree. If
    /// any node scores above
    /// [`ConversationTreeConfig::daydream_resonance_threshold`](super::config::ConversationTreeConfig::daydream_resonance_threshold),
    /// the character's mind has snagged on something — a phrase, a word that
    /// rhymes with something buried in their history.
    ///
    /// **Generation:** minimal by design. A short context from the triggering
    /// phrase and the resonant node. 50–100 tokens, low temperature. Not
    /// elaborate narrative — a flicker: a memory fragment, an unresolved
    /// question resurfacing, a connection between two things never consciously
    /// linked before.
    ///
    /// **Latency gate:** the daydream runs in the background while the
    /// character waits for the player to respond.
    /// - If the player responds quickly, the daydream is aborted and
    ///   discarded. Nothing happened.
    /// - If the player takes long enough, the daydream completes: a `Thought`
    ///   turn is inserted into the tree, and a re-entry turn bridges it back
    ///   to current Reality context, creating an ancestor relationship future
    ///   attention can traverse.
    ///
    /// TODO: Daydream pipeline not yet implemented.
    Thought,

    /// Deliberate executive planning — the frontal layer.
    ///
    /// The character talks to themselves: user and assistant are the same
    /// voice. This is genuine internal dialogue — proposing and resisting,
    /// advocating and interrogating — arriving not at a logically correct
    /// answer but at something that feels like a decision they can act on.
    ///
    /// **Input:** a situational system prompt (current problem frame) plus
    /// limbic material — relevant [`Sleep`](TurnType::Sleep) and
    /// [`Thought`](TurnType::Thought) turns surfaced through the attention
    /// system. These arrive as questions, not instructions. The Reason turn
    /// wrestles with them.
    ///
    /// **Output — the Plan:** a short block written into
    /// [`ConversationTree::plan`](super::conversation_tree::ConversationTree::plan)
    /// in the character's own voice:
    /// - What they are trying to do
    /// - What they are wary of
    /// - What they are waiting for
    ///
    /// The Plan is injected into every subsequent Reality system prompt until
    /// a new Reason turn rewrites it. Significant Reality events, new Sleep
    /// material, or unexpected developments trigger a new Reason turn that
    /// replaces it.
    ///
    /// TODO: Reason pipeline not yet implemented.
    Reason,
}

// ────────────────────────────────────────────────────────────────────────────
// StorageTier
// ────────────────────────────────────────────────────────────────────────────

/// Storage tier for a node's KV cache — transient, determined by tree depth
/// or policy. Tracks where the KV actually lives in VRAM / staging / disk.
/// Applied independently to both turn nodes and segment nodes.
///
/// The actual [`KvFormat`](candle_nn::kv_cache::KvFormat) applied at each
/// tier is resolved via
/// [`ConversationTreeConfig::storage_tier_format`](super::config::ConversationTreeConfig::storage_tier_format).
///
/// Currently all nodes start and remain `Hot`. `Warm` and `Cold` are
/// defined for future tier management without struct changes (TODO).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StorageTier {
    /// VRAM-resident, full or quantized KV, immediate attention access.
    /// Default format: BF16 (configured via `hot_turn_format`).
    /// Currently all nodes stay here for the conversation lifetime.
    #[default]
    Hot,

    /// VRAM staging while KV is being regenerated ("warming").
    /// KV occupies a reserved slot in the staging arena during regeneration.
    /// Default format: BF16 (configured via `warm_turn_format`).
    Warm,

    /// Disk-resident; no KV materialized in VRAM.
    /// Ancestor metadata and representative K vectors stored on disk.
    /// KV is regenerated on demand (Cold → Warm warming pass).
    /// TODO: tier management deferred.
    Cold,
}

// ────────────────────────────────────────────────────────────────────────────
// Identity types: TurnId, SegmentId, NodeId
// ────────────────────────────────────────────────────────────────────────────

/// Identity of a turn node: temporal coordinates recorded at `finish_turn()`
/// time.
///
/// Both components are recorded regardless of whether temporal markers are
/// enabled; the marker string is derived on demand via
/// [`temporal_marker`](TurnId::temporal_marker).
///
/// # Coordinate semantics
///
/// - `day`: elapsed UTC calendar days since the tree's reference instant
///   (captured at `ConversationTree::new()` time). Day 0 = opening day.
///   Uses `i32` to allow negative values in testing and replay scenarios.
/// - `seq`: monotonic turn-pair counter scoped to this conversation. Starts
///   at 1 and never resets, even across day boundaries. The model can
///   cross-reference turns by seq: "as we discussed in T-3.47".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TurnId {
    /// Elapsed days since reference_instant. `i32` allows negative values for
    /// testing and replay scenarios.
    pub day: i32,
    /// Monotonic turn-pair counter across the entire conversation. Never
    /// resets, always ≥ 1.
    pub seq: u32,
}

impl TurnId {
    /// The temporal marker string derived from this turn's coordinates.
    ///
    /// Returns `"[T-{day}.{seq}]"`. Only injected into the token stream when
    /// `temporal_markers_enabled = true`, but always computable.
    pub fn temporal_marker(&self) -> String {
        format!("[T-{}.{}]", self.day, self.seq)
    }

    /// The `seq` value used for timeline ordering.
    pub fn ordering_seq(&self) -> u32 {
        self.seq
    }
}

/// Identity of a segment node: the inclusive [`TurnId`] range it summarizes.
///
/// Using `TurnId` boundaries rather than parallel seq-based coordinates
/// eliminates redundancy and ensures segment boundaries are always coherent
/// with actual turn identities. Segments are contiguous ranges; no gaps or
/// overlaps are allowed. The segment is identified by the actual identities
/// of the turns it covers, not by a separate index into a parallel sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentId {
    /// The identity of the first turn summarized (inclusive).
    pub start_turn: TurnId,
    /// The identity of the last turn summarized (inclusive).
    pub end_turn: TurnId,
}

impl SegmentId {
    /// The seq of the end_turn — used for timeline ordering alongside turns.
    pub fn ordering_seq(&self) -> u32 {
        self.end_turn.seq
    }
}

/// Globally stable identity of a tree node: either a turn or a segment.
///
/// The enum variant encodes the structural type; no separate type tag needed.
/// `NodeId` is `Copy` — safe to pass over the stack, store as a HashMap key,
/// or embed in log messages.
///
/// # Ordering
///
/// Nodes are ordered chronologically by their `ordering_seq()` value:
/// - Turns: the point-in-time `seq` from `TurnId`.
/// - Segments: the `seq` of `end_turn` from `SegmentId` (the last turn they cover).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeId {
    /// An actual user↔assistant exchange (turn-pair).
    Turn(TurnId),
    /// A compressed summary section (segment).
    Segment(SegmentId),
}

impl NodeId {
    /// Convenience constructor: `NodeId::Turn(TurnId { day, seq })`.
    pub fn turn(day: i32, seq: u32) -> Self {
        NodeId::Turn(TurnId { day, seq })
    }

    /// Convenience constructor: `NodeId::Segment(SegmentId { start_turn, end_turn })`.
    pub fn segment(start_turn: TurnId, end_turn: TurnId) -> Self {
        NodeId::Segment(SegmentId {
            start_turn,
            end_turn,
        })
    }

    /// Seq coordinate for timeline ordering.
    ///
    /// For turns: the point-in-time `seq` from `TurnId`.  
    /// For segments: the `seq` of `end_turn` from `SegmentId`.
    pub fn ordering_seq(&self) -> u32 {
        match self {
            NodeId::Turn(tid) => tid.ordering_seq(),
            NodeId::Segment(sid) => sid.ordering_seq(),
        }
    }

    /// Returns the `TurnId` if this is a `Turn` variant.
    pub fn as_turn_id(&self) -> Option<TurnId> {
        match self {
            NodeId::Turn(t) => Some(*t),
            NodeId::Segment(_) => None,
        }
    }

    /// Returns the `SegmentId` if this is a `Segment` variant.
    pub fn as_segment_id(&self) -> Option<SegmentId> {
        match self {
            NodeId::Turn(_) => None,
            NodeId::Segment(s) => Some(*s),
        }
    }
}
