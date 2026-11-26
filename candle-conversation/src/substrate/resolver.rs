//! Defines the `ContentResolver` interface the projection engine queries for turn
//! counts, sizes and scores, and the guards that answer it from a substrate.
//!
//! The resolver returns counts, sizes and scores, never content: the engine emits
//! opaque ids and the caller reads content from them. Turns are enumerated across
//! every timeline of a group, because indices repeat from one conversation to the
//! next, and tool round-trips are kept whole so a call is never selected without
//! its response. The trained window has no default, because a resolver that fell
//! back to none silently filled no dialogue slot.

use super::*;
use crate::files::FileRecord;

// ── ContentResolver trait ─────────────────────────────────────────────────────

/// Supplies dynamic content metadata to the projection engine.
///
/// The resolver never returns content itself — only the **turn count**,
/// **token counts**, and **scores** the engine needs to make budget and
/// selection decisions.  The engine emits a [`super::Projection`] of opaque
/// ids; the caller looks up content from those ids in its own store.
pub trait ContentResolver {
    /// Every turn visible in `group`, across **all** of its active timelines.
    ///
    /// A group is a *shape*, not a conversation: `code_reading` declares one
    /// conversation per file, so a group routinely holds many timelines. Turn
    /// indices are per-timeline, so `(group, index)` alone is ambiguous — index 3
    /// exists in every file's timeline. Enumerating [`TurnKey`]s (timeline +
    /// index) is what lets every conversation in a multi-timeline group be scored
    /// and projected, instead of only the first-registered one.
    ///
    /// Returned in a stable order: timelines in registration order, turns
    /// ascending within each.
    fn group_turns(&self, group: GroupId) -> Vec<TurnKey>;

    /// Close `keys` under **exchange** membership: one member list per
    /// exchange, in first-appearance order, each in ascending turn order.
    ///
    /// A tool round-trip is recorded as coupled turns — `user(request) →
    /// assistant(<tool_call>)`, then `user(<tool_response>) →
    /// assistant(answer)` — and the run is ONE indivisible projection unit.
    /// Selection ranks and the budget trims these units, never their halves: a
    /// projection that seats the call without the response hands the model a
    /// dangling `<tool_call>`, which it imitates. (Observed: a dialogue answered
    /// a question by emitting `<tool_call>{"name":"file_list",…}` for a folder
    /// nobody had asked about, because `top_k` had seated the first turn of a
    /// `repo_map` folder chain on its own.)
    ///
    /// Members absent from `keys` are pulled IN — closure, not just grouping —
    /// so a rule that ranked one half brings the whole run with it.
    ///
    /// Default: every key is its own exchange. A resolver that tracks no
    /// couplings has no round-trips to keep whole.
    fn group_exchanges(&self, keys: &[TurnKey]) -> Vec<Vec<TurnKey>> {
        keys.iter().map(|key| vec![*key]).collect()
    }

    /// Whether any layer is toggled out of projection assembly — the
    /// diagnostic kill switch the substrate holds
    /// ([`Substrate::toggle_layer`]). Default: a resolver with no substrate
    /// behind it has nothing toggled.
    fn any_layer_disabled(&self) -> bool {
        false
    }

    /// Whether layer `name` is toggled out of projection assembly.
    fn layer_disabled(&self, _name: &str) -> bool {
        false
    }

    /// Token count for a turn.  Stable across projection calls.
    fn turn_token_count(&self, turn: TurnKey) -> usize;

    /// The trained window of the engine the projection is for — the slot a
    /// dialogue's fill holds (`docs/hit_driven_projection.md` §4). `None` for
    /// an engine whose attention selects nothing, which is filled by its
    /// layers' windows instead.
    ///
    /// **No default.** `SubstrateRead` once took the trait's `None` while the
    /// substrate it reads held the engine's window, so `TargetedRead`, which
    /// every dialogue projection runs through, never filled a slot. Members
    /// were placed whole and no cap bound: measured 2026-11-18 on BlackWorld,
    /// every placed turn of 30 prompts was whole, with the working set ~75K
    /// against a 64K budget. Each resolver says which it is.
    fn trained_window(&self) -> Option<usize>;

    /// Relevance score for a turn — the wide-Q belief score the scheduler
    /// recorded for this projection. Zero when unscored.
    fn turn_score(&self, turn: TurnKey) -> f32;

    /// How much the model's own selection read a turn over the strides before
    /// this projection: its round's lift less one, `gain · h`
    /// (`docs/hit_driven_projection.md` §7.1). Zero when unread. Default zero.
    fn turn_reads(&self, _turn: TurnKey) -> f32 {
        0.0
    }

    /// Concept B: a turn group's attention mass this projection (0 = no mass /
    /// unscored — adaptivity then rests at the static budget).
    fn group_attention_mass(&self, _group: GroupId) -> f32 {
        0.0
    }

    /// The order statistics of a turn group's raw scores this projection;
    /// `None` when the group was not scanned (default: no scan behind it).
    fn group_scan(&self, _group: GroupId) -> Option<GroupScan> {
        None
    }

    /// Concept B: a collection's attention mass this projection.
    fn collection_attention_mass(&self, _collection: u32) -> f32 {
        0.0
    }

    /// Concept C: whether this turn's score was raised by locality drag —
    /// selection stamps such picks `SelectionOrigin::Locality`.
    fn turn_locality_boosted(&self, _turn: TurnKey) -> bool {
        false
    }

    /// The timelines of this member's seed (`Substrate::seed_chain`), so
    /// selection can name a turn that is there because the conversation was
    /// founded on it.
    fn seed_timelines(&self) -> HashSet<TimelineId> {
        HashSet::new()
    }

    /// Layer that produced a given turn.  Used to denormalise
    /// `layer_id` onto the emitted `TurnId` without a back-lookup
    /// through the schema.
    ///
    /// Default impl returns `None` (resolver doesn't track origins).
    /// When `None`, projection emit falls back to the layer-walk's
    /// `layer_id` — which is correct for tests using mock resolvers.
    fn turn_origin(&self, _turn: TurnKey) -> Option<LayerId> {
        None
    }

    /// Whether the projection TARGET is an append-only ingest layer — i.e. this
    /// projection is a conversation GENERATING its own content, not a dialogue
    /// RETRIEVING from a corpus.
    ///
    /// The distinction decides whether a turn on the target's own timeline has
    /// to earn its place. Retrieval must: the corpus is large and belief
    /// selection is the whole point. Generation must not: an ingest turn is
    /// inserted and immediately decoded against, so it has no wide-Q belief yet,
    /// scores zero, and is filtered out by its own group's band — leaving the
    /// decode unable to see the request it is answering. (Observed: a folder
    /// summary decode replying "the user hasn't asked a specific question yet"
    /// with the request one turn back in the same conversation.)
    ///
    /// Default `false` — an ordinary projection retrieves.
    fn target_is_ingest_self(&self) -> bool {
        false
    }

    /// The turn a dialogue holds in place of `turn` — its unit's face
    /// ([`Substrate::face_of`]) — or `None` when `turn` is held as it is.
    /// Default `None`: a resolver with no units has no faces.
    fn face_of(&self, _turn: TurnKey) -> Option<TurnKey> {
        None
    }

    /// The unit opening a face's `turn` stands for ([`Substrate::unit_of`]) —
    /// what a projection's record names in its place. Default `None`.
    fn unit_of(&self, _turn: TurnKey) -> Option<TurnKey> {
        None
    }

    /// Whether an ingest target draws on `group` across conversations — a
    /// group of a layer below it that it compounds on
    /// ([`Substrate::set_ingest_sees`]). Default `false`.
    fn compounds_on(&self, _group: GroupId) -> bool {
        false
    }

    /// Whether `turn` carries one of `tags` — the scope a tagged turn group
    /// admits (see [`crate::projection::Builder::set_group_tags`]).
    ///
    /// Asked only for a group whose tag list is non-empty. Default `true`:
    /// a resolver that tracks no tags cannot scope, and a mock that says every
    /// turn is out of scope would empty every tagged group in every test.
    fn turn_carries(&self, _turn: TurnKey, _tags: &[String]) -> bool {
        true
    }

    /// Forest kind of a projected turn — `Normal` (a raw conversation turn) vs
    /// `SummaryOfTurns` / `SummaryOfSummaries` (a summary node standing in for
    /// the turns beneath it). Lets the projection record (and the GUI / inspector
    /// that read it) show whether a slot was filled with a real turn or a summary
    /// — the distinction that is otherwise invisible once a node is materialized
    /// as a `Sealed(Turn)` segment.
    ///
    /// Default impl returns `Normal` (mock resolvers and non-summarised timelines
    /// have no forest, so every turn is raw).
    fn turn_kind(&self, _turn: TurnKey) -> TurnKind {
        TurnKind::Normal
    }

    /// The turn indices `index` transitively covers in `group`'s summary forest
    /// — every descendant node (SummaryOfSummaries / SummaryOfTurns / Normal)
    /// beneath it, at every level. Empty for a raw `Normal` turn (a leaf covers
    /// nothing). A summary only ever covers turns on its OWN timeline, so this
    /// is self-contained per timeline.
    ///
    /// Used by the rule-based selection's descendant-dedup: a summary that wins
    /// a slot on provenance score is dropped when any node it covers also wins,
    /// so the projection keeps the SPECIFIC over the coarse (never a summary
    /// stacked on top of the very turns it summarises).
    ///
    /// Default impl returns empty (mock resolvers / non-summarised timelines
    /// have no forest, so nothing is covered).
    fn node_covers(&self, _turn: TurnKey) -> Vec<TurnIndex> {
        Vec::new()
    }

    /// Whether a sealed turn was decoded with `/no_think` thinking suppression —
    /// the assembler re-renders the `/no_think` soft-switch glue after
    /// `user_start` for such turns, so the materialized-glue builder needs the
    /// same bit to reproduce the engine's boundary run exactly. Default `false`
    /// (mock resolvers / non-substrate timelines have no suppression record).
    fn turn_no_think(&self, _timeline: TimelineId, _index: TurnIndex) -> bool {
        false
    }

    /// Token count for a system-prompt section.  Returns `0` for
    /// sections the resolver has no record of.
    fn section_token_count(&self, _section: SectionId) -> usize {
        0
    }

    /// Score-density selection over a timeline's summary forest.
    /// Returns the chrono-
    /// logically ordered `(turn_index, effective_score)` list for the
    /// given timeline, fitted into `budget` tokens, or `None` when no
    /// summary tree exists yet for that timeline.
    ///
    /// When `Some(_)`, the projection's step-9 reconciler skips the
    /// flexbox / rule-based budget allocation for this timeline's
    /// target group and emits the returned list verbatim — score-
    /// density already picked turns inside `budget`.
    ///
    /// Default impl returns `None` so existing test resolvers and
    /// non-summary-tree groups keep their current behaviour.
    fn summary_tree_select(
        &self,
        _timeline: TimelineId,
        _budget: u32,
    ) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
        None
    }

    /// Number of turns currently awaiting the async summariser for
    /// `timeline`.  Used by the projection to populate the
    /// score-density backpressure metric inside its diagnostic sink.
    /// Default returns 0.
    fn pending_summary_len(&self, _timeline: TimelineId) -> usize {
        0
    }

    /// Relevance score for a system-prompt section — the wide-Q belief score
    /// the scheduler recorded. Default `0.0`; concrete resolvers override.
    fn section_score(&self, _section: SectionId) -> f32 {
        0.0
    }

    /// The projection target's working set as it bears on `group` — its
    /// pooled conversations that belong to that group
    /// (`docs/zend_working_set.md` §4.4). `None` when the target carries no
    /// working set, which is every ingest conversation. Default `None`.
    fn working_set_members(&self, _group: GroupId) -> Option<WorkingSetMembers> {
        None
    }

    /// What the target's slot holds from the boundary before, and whether
    /// this is the first boundary after a new user message
    /// (`docs/hit_driven_projection.md` §7.2 step 5). `None` for a target
    /// with no working set, whose fill is chosen whole. Default `None`.
    fn working_set_boundary(&self) -> Option<SlotBoundary> {
        None
    }

    /// The target's working set of the knowledge base's files
    /// (`docs/provenance_system_collection.md` §4). Default `None`.
    fn files_set(&self) -> Option<FilesSet> {
        None
    }

    /// Every live file record of the knowledge base. Default none.
    fn files_records(&self) -> Vec<FileRecord> {
        Vec::new()
    }

    /// The section of the knowledge base's member registered as `name`
    /// (`files::names`), when this process holds it. Default `None`.
    fn member_section(&self, _name: &str) -> Option<SectionId> {
        None
    }

    /// Where `timeline` stands in the target's pool — its momentum, the
    /// scan's fresh score, its hit momentum and strength, and the path it
    /// reads — for the projection record (`projection::fill_report`).
    /// Default: nothing known.
    fn working_set_standing(&self, _timeline: TimelineId) -> Standing {
        Standing::default()
    }

    /// The `n` strongest conversations the last scan scored that the
    /// target's pool does not hold, strongest fresh score first. Default
    /// empty.
    fn working_set_unpooled(&self, _n: usize) -> Vec<Unpooled> {
        Vec::new()
    }

    /// Every turn of `timeline`, ascending — how a working-set group takes a
    /// conversation whole. Default empty.
    fn timeline_turns(&self, _timeline: TimelineId) -> Vec<TurnKey> {
        Vec::new()
    }
}

/// Group-keyed [`ContentResolver`] over a bare [`Substrate`].
///
/// **The substrate owns no scoring state** — it is a directory of turns,
/// sections and timelines, not a score table. Without an attached
/// [`ProjectionScores`], every score lookup returns zero. Production
/// code attaches scores via [`SubstrateRead::scores`]; this impl exists
/// so structural-only callers (e.g. tests reading only turn counts and
/// sealed pointers) can still pass `&substrate` to `Builder::project`.
///
/// Enumerates **every** active timeline in a group, so a group holding many
/// conversations (`code_reading` declares one per file) is fully visible rather
/// than collapsing to the first-registered timeline.
///
/// This impl has **no projection target**, so it applies **no sibling-timeline
/// masking**: it surfaces every conversation in a group, including what would be
/// the target group. A target-masked projection (where a live slot must NOT see
/// sibling conversations of the same shape) MUST project through [`TargetedRead`]
/// — this bare impl is for structural / test callers only.
impl ContentResolver for Substrate {
    fn trained_window(&self) -> Option<usize> {
        Substrate::trained_window(self)
    }

    fn group_turns(&self, group: GroupId) -> Vec<TurnKey> {
        self.active_timelines_for_group(group)
            .flat_map(|tl| {
                (0..Substrate::turn_count(self, tl)).map(move |i| TurnKey::new(tl, TurnIndex(i)))
            })
            .collect()
    }

    fn any_layer_disabled(&self) -> bool {
        Substrate::any_layer_disabled(self)
    }

    fn layer_disabled(&self, name: &str) -> bool {
        self.is_layer_disabled(name)
    }

    fn turn_token_count(&self, turn: TurnKey) -> usize {
        self.turn(turn.timeline, turn.index)
            .map_or(0, |e| e.content.token_count)
    }

    fn turn_score(&self, _turn: TurnKey) -> f32 {
        // Bare substrate has no attached scores; pair via ScoredSubstrate
        // or read through Conversation::read_scored to see non-zero values.
        0.0
    }

    fn turn_origin(&self, turn: TurnKey) -> Option<LayerId> {
        let (layer, _) = self.timeline_target(turn.timeline)?;
        Some(layer)
    }

    fn face_of(&self, turn: TurnKey) -> Option<TurnKey> {
        Substrate::face_of(self, turn)
    }

    fn unit_of(&self, turn: TurnKey) -> Option<TurnKey> {
        Substrate::unit_of(self, turn)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        self.sections.get(&section).map_or(0, |e| e.token_count)
    }

    fn section_score(&self, _section: SectionId) -> f32 {
        0.0
    }
}

// ── Guards ────────────────────────────────────────────────────────────────────

/// Read guard over a [`Substrate`] inside a [`super::resolver::Conversation`].
///
/// Carries an optional [`ProjectionScores`] reference: when `Some`, the
/// read implements [`ContentResolver`] using those scores; when `None`,
/// scoring methods return zero (correct default for non-projection
/// callers reading structural fields like turn counts or sealed pointers).
///
/// Construct scored variants via
/// [`super::resolver::Conversation::read_scored`]; the bare
/// [`super::resolver::Conversation::read`] returns the unscored variant.
pub struct SubstrateRead<'a> {
    pub(in super::super) guard: SubstrateReadGuard,
    pub(in super::super) scores: Option<&'a ProjectionScores>,
}

impl<'a> SubstrateRead<'a> {
    /// Lookup helper: returns the attached scores, or an empty default
    /// when this is an unscored read.
    pub(crate) fn scores_or_empty(&self) -> &ProjectionScores {
        static EMPTY_SCORES: OnceLock<ProjectionScores> = OnceLock::new();
        self.scores
            .unwrap_or_else(|| EMPTY_SCORES.get_or_init(ProjectionScores::default))
    }

    /// Per-`TimelineId` variant of [`ContentResolver::turn_score`]. Used
    /// by [`super::projection::resolver::TargetedRead`] which already
    /// knows the target-corrected `TimelineId` for the queried group.
    /// Returns `0.0` when the turn is unknown.
    pub fn turn_score_for_timeline(&self, timeline: TimelineId, index: TurnIndex) -> f32 {
        if self.guard.turn(timeline, index).is_none() {
            return 0.0;
        }
        self.scores_or_empty().turn(timeline, index)
    }
}

impl<'a> std::ops::Deref for SubstrateRead<'a> {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        &self.guard
    }
}

/// Untargeted read guard: enumerates every active timeline in a group and applies
/// **no** sibling-timeline masking (it has no target). Wrap in [`TargetedRead`]
/// for a target-masked projection; this bare guard is for structural / test reads.
impl<'a> ContentResolver for SubstrateRead<'a> {
    fn trained_window(&self) -> Option<usize> {
        self.guard.trained_window()
    }

    fn group_turns(&self, group: GroupId) -> Vec<TurnKey> {
        self.guard
            .active_timelines_for_group(group)
            .flat_map(|tl| {
                (0..Substrate::turn_count(&self.guard, tl))
                    .map(move |i| TurnKey::new(tl, TurnIndex(i)))
            })
            .collect()
    }

    fn any_layer_disabled(&self) -> bool {
        self.guard.any_layer_disabled()
    }

    fn layer_disabled(&self, name: &str) -> bool {
        self.guard.is_layer_disabled(name)
    }

    fn turn_token_count(&self, turn: TurnKey) -> usize {
        self.guard
            .turn(turn.timeline, turn.index)
            .map_or(0, |e| e.content.token_count)
    }

    fn turn_score(&self, turn: TurnKey) -> f32 {
        if self.guard.turn(turn.timeline, turn.index).is_none() {
            return 0.0;
        }
        self.scores_or_empty().turn(turn.timeline, turn.index)
    }

    fn turn_reads(&self, turn: TurnKey) -> f32 {
        self.scores_or_empty().reads(turn.timeline, turn.index)
    }

    fn group_attention_mass(&self, group: GroupId) -> f32 {
        self.scores_or_empty().group_mass(group)
    }

    fn group_scan(&self, group: GroupId) -> Option<GroupScan> {
        self.scores_or_empty().group_scan(group)
    }

    fn collection_attention_mass(&self, collection: u32) -> f32 {
        self.scores_or_empty().collection_mass(collection)
    }

    fn turn_locality_boosted(&self, turn: TurnKey) -> bool {
        self.scores_or_empty()
            .is_locality_boosted(turn.timeline, turn.index)
    }

    fn seed_timelines(&self) -> HashSet<TimelineId> {
        self.guard.seed_chain().into_iter().collect()
    }

    fn turn_origin(&self, turn: TurnKey) -> Option<LayerId> {
        let (layer, _) = self.guard.timeline_target(turn.timeline)?;
        Some(layer)
    }

    fn face_of(&self, turn: TurnKey) -> Option<TurnKey> {
        self.guard.face_of(turn)
    }

    fn unit_of(&self, turn: TurnKey) -> Option<TurnKey> {
        self.guard.unit_of(turn)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        self.guard
            .sections
            .get(&section)
            .map_or(0, |e| e.token_count)
    }

    fn section_score(&self, section: SectionId) -> f32 {
        if !self.guard.sections.contains_key(&section) {
            return 0.0;
        }
        self.scores_or_empty().section(section)
    }

    fn summary_tree_select(
        &self,
        timeline: TimelineId,
        budget: u32,
    ) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
        select_summary_tree(&self.guard, self.scores_or_empty(), timeline, budget)
    }

    fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.guard.pending_summary_len(timeline)
    }
}
