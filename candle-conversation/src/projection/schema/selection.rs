//! The rule that decides which of a group's turns survive into a projection,
//! and the formula that reduces a group's turn scores to one number.
//!
//! Every rule selects by relevance and then emits its survivors in insertion
//! order, so selection picks what matters while emission keeps the dialogue
//! coherent. `AlwaysVisible`, `TopK`, `Single` and `Sequence` rank turns by
//! score, and `Sequence` also keeps the most recent turns unconditionally.
//! `Named` and `Offered` are collection-only and take their members from the
//! caller: `Named` shows exactly the members it names, and `Offered` keeps the
//! best-scoring of them, with a mandatory member always shown. An unset selector
//! offers nothing. `WorkingSet` fills a group from the target's working set and
//! emits it outside the flexbox. `ScoreFormula` gives the layer gate and the
//! emission order one group score to work from.

/// Which turns from a group survive into the projection.
///
/// # Decision tree
///
/// ```text
///   Selection rule ─┬─ AlwaysVisible        → all turns above threshold
///                   ├─ TopK { k }           → k highest-scored above threshold
///                   ├─ Single               → 1 highest-scored above threshold
///                   ├─ Named { selector }   → the one member named by a runtime selector
///                   ├─ Offered { selector, k } → k highest-scored of the members a selector names
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

    /// Collection-only: the members whose `name`s the named selector holds
    /// (resolved from the projection's `SelectionState`, set per turn via
    /// `TurnOptions::selection`), emitted in declaration order. This is an
    /// **explicit, score-independent** pick — it ignores provenance relevance
    /// and the score threshold entirely, selecting exactly the members the
    /// caller names (or nothing, if the selector is unset or names none).
    ///
    /// One name pins one member — a character's identity out of every
    /// character's. Several is a caller showing a set only it can decide: a
    /// character's acts are the ones its body can do here and now, with these
    /// people and these things in reach, which is a fact about the world rather
    /// than a score.
    ///
    /// On a turn group (which has no member names) this selects nothing.
    Named { selector: String },

    /// Collection-only: **what the runtime offers, ranked by provenance.** The
    /// members the named selector holds are the only candidates — a fact about
    /// the world, such as the acts a body can take in this room — and the
    /// belief-driven selection (provenance score, hysteresis, budget) then keeps
    /// the `k` most relevant of them. A member the selector does not name is
    /// never shown however well it scores; a named member that scores poorly is
    /// dropped unless it is `mandatory`, in which case it is shown on every
    /// projection that offers it.
    ///
    /// An unset selector offers nothing, so nothing is shown. The
    /// forced-member pin ([`FORCE_TOOL_SELECTOR`]) overrides both, as it does
    /// for `TopK`.
    Offered { selector: String, k: usize },

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
    ///
    /// The lineage a conversation inherited — its ancestors' turns, the
    /// priming chain — is scored like any history, from a base of
    /// `lineage_base` (`docs/hit_driven_projection.md` §8.2): it holds its
    /// seats while nothing outscores it and gives way to what does when the
    /// window runs short.
    Sequence {
        recent: usize,
        historical_top_k: usize,
        lineage_base: f32,
    },

    /// A conversation **by birth** (`docs/placed_section_layout.md` §3.7,
    /// `docs/npc_mind_layout.md` §2): its inherited lineage whole, as the
    /// window's floor, then its current exchange always and the newest past
    /// exchanges, whole, while they fit `window` tokens — none chosen by score
    /// and none shed for the budget. Past exchanges are sealed and the window
    /// is fixed, so every boundary of a request holds the same ones; between
    /// requests the oldest drop out of the far end.
    Birth { window: usize },

    /// A turn group of whole ingest conversations sized by the projection
    /// target's working set (`docs/zend_working_set.md` §4.4): the target's
    /// pinned conversations in this group, whole, then its provenance
    /// candidates by momentum until `share`'s budget is spent. Emitted
    /// verbatim, outside the flexbox — never trimmed. A target whose layer
    /// declares no working set selects nothing here.
    WorkingSet { share: WorkingSetShare },
}

/// Which kind of piece a [`SelectionRule::WorkingSet`] group holds: the kind
/// whose share of the room caps its rounds in the slot fill
/// (`docs/hit_driven_projection.md` §7.2 step 4, `working_set.shares`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkingSetShare {
    /// File readings. Members hold what `budget_tokens` leaves once the folder
    /// groups have selected.
    Readings,
    /// Folder summaries. Members hold up to `folder_tokens`, shared by every
    /// folder group.
    Folders,
    /// The workers' guides.
    Guides,
    /// Journal notes.
    Journals,
}

impl WorkingSetShare {
    /// Whether this share's members draw on the working set's guidance pool,
    /// `guidance_tokens`, a pool of its own beside `budget_tokens`: a guide or
    /// note never dislodges a file, and no burst of files crowds one out
    /// (`docs/guidance_layer.md` §1).
    pub fn is_guidance(self) -> bool {
        matches!(self, Self::Guides | Self::Journals)
    }
}

/// A fallback section injected when a **collection's** normal selection
/// (belief + challenger + rule) yields nothing, named by section name.
///
/// Collections only. A collection is a set of mutually exclusive variants of
/// one dial — a persona, a thinking effort, a response length — so exactly one
/// must always be present and "none qualified" has no valid rendering; the
/// default names which variant that is.
///
/// Turn **groups** deliberately have no equivalent. Their members are content,
/// not dial settings, and an empty selection there is a real answer: nothing in
/// the corpus was relevant. A group-level default was tried and removed — it
/// injected its member with a score synthesised to clear the threshold that
/// member had just failed, which put thousands of tokens of unrelated content
/// in front of the model on a cold probe.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionDefault {
    /// The section name identifying the fallback member. Must uniquely name one
    /// section of the collection.
    pub tag: String,
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
