//! [`Conversation`] — the workspace-shared substrate handle, and
//! [`TargetedRead`] — the target-aware [`ContentResolver`] wrapper.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock, RwLock};

use super::event::{decode_events, ProjectionSelection, SystemItem};
use super::ids::{GroupId, LayerId, SectionId, TimelineAllocator, TimelineId, TurnIndex, TurnKey};
use super::project::ProjectionTarget;
use super::schema::{LayerSchema, Schema, SystemPromptItem, SystemPromptSchema};
use crate::normalization::{ChildKey, NormalizationCache, ScopeKey};
use crate::persistence::record::{DistillMode, TreeMetadataPayload};
use crate::persistence::streams::{ContentAddress, SectionDecl, StreamDecl, StreamId, TurnDecl};
use crate::persistence::SubstratePersistence;
use crate::provenance::{decode_wide_sigs, score_slots_weighted, WideQSig};
use crate::substrate::{
    ContentResolver, ProjectionScores, StoredSequence, Substrate, SubstrateRead, SubstrateWrite,
    TurnPartWrite,
};
use crate::summary_tree::exchange::{exchanges, over_normals};
use crate::summary_tree::{SelectionDiagnostics, SelectionOrigin, TurnKind};
use crate::token_buffer::TokenBuffer;
use crate::turn::Role;
use crate::turn_layout::TurnLayout;
use candle_nn::kv_cache::SealedSequence;

/// Upper bound on how many recent dialogue turns the normalization warm-up
/// replays on load (`ensure_normalization_warm`). The asymmetric-EWMA hit levels
/// converge in a few dozen steps, so this caps the one-time cost without changing
/// the warmed levels materially.
const WARM_REPLAY_MAX_TURNS: usize = 512;

/// Contiguous `[start, end)` sub-window bounds over a `len`-token sig, split at
/// sorted, deduped `seams`. An empty `seams` yields one window `[0, len)` — the
/// prior whole-turn behaviour. Seams at 0, at/past `len`, or that don't advance
/// are ignored, so a malformed seam can never produce an empty or inverted range.
fn subwindow_bounds(len: usize, seams: &[usize]) -> Vec<(usize, usize)> {
    let mut bounds = Vec::with_capacity(seams.len() + 1);
    let mut prev = 0usize;
    for &s in seams {
        if s > prev && s < len {
            bounds.push((prev, s));
            prev = s;
        }
    }
    bounds.push((prev, len));
    bounds
}

/// The name of the section a projection selected inside `collection`, if any.
fn selected_in_collection(sel: &ProjectionSelection, collection: &str) -> Option<String> {
    sel.system.iter().find_map(|item| match item {
        SystemItem::Collection { name, sections } if name == collection => {
            sections.iter().find(|s| s.selected).map(|s| s.name.clone())
        }
        _ => None,
    })
}

// ── Conversation ──────────────────────────────────────────────────────────────

/// Workspace-shared, lock-protected handle to the per-turn record store.
///
/// Multiple conversations in the same workspace clone this handle; they all
/// see (and write into) the same underlying [`Substrate`].  Locking is
/// coarse-grained — one `RwLock` over the whole resolver — but scans and
/// mutations are short, so contention is minimal in practice.
///
/// # Phase 4 substrate semantics
///
/// - **Append (write)** — at seal time, each conversation appends its new
///   turn into the shared store.  Index allocation is per-group, monotonic,
///   under the resolver's lock.
/// - **Read** — projection takes a read guard for the duration of a single
///   `project()` call via [`Conversation::read`].  Returns a
///   [`SubstrateRead`] that implements [`ContentResolver`].
/// - **Reset** — does *not* clear the shared store (other conversations
///   would lose their history).  Sequence-level reset only drops local
///   KV state via the scheduler.
#[derive(Clone)]
pub struct Conversation {
    inner: Arc<RwLock<Substrate>>,
    allocator: Arc<TimelineAllocator>,
    /// The mandatory persistence layer — every turn is recorded into its
    /// redo log (`docs/kv_tier_migration.md` §13.6).
    persistence: Arc<Mutex<SubstratePersistence>>,
    /// Runtime, in-memory score normalization (per-scope hit levels). NOT
    /// persisted — rebuilt from the substrate's existing turns on first use, then
    /// evolved as new turns seal. Shared across clones of this handle so learning
    /// pools over all sessions. See `docs/provenance_score_normalization.md`.
    normalization: Arc<Mutex<NormalizationCache>>,
    /// Spawns the warm-from-substrate replay exactly once, on the first belief
    /// scan, onto a detached background thread (so it never blocks the first live
    /// reprojection on the decode hot path). Early reprojections may read a
    /// still-warming cache — best-effort by construction, since the levels are
    /// runtime-derived and seal-observe keeps warming them. See
    /// [`Self::ensure_normalization_warm`].
    normalization_warm: Arc<Once>,
    /// Cached `(segment_count, last_op)` for the `/v1/status` maintenance
    /// indicator — refreshed by the persistence thread after each maintenance
    /// pass. Read through this **separate** lock so the status endpoint never
    /// blocks on `persistence` (which a compaction holds across its I/O).
    /// GUI compaction-indicator cache: `(segment_count, last_op, running)`.
    /// `running` is `true` only while a maintenance op's relocation I/O is in
    /// flight, so the sidebar can show a live spinner (vs. a settled ✓ for the
    /// last completed op). Refreshed by the persistence thread each pass; read
    /// by `/v1/status` off this separate lock so it never blocks on compaction.
    maintenance: Arc<Mutex<(usize, Option<(String, u64)>, bool)>>,
}

impl Default for Conversation {
    /// An ephemeral conversation — see [`Conversation::ephemeral`].
    fn default() -> Self {
        Self::ephemeral()
    }
}

impl Conversation {
    /// Create a fresh ephemeral conversation (throwaway temp-dir log).
    pub fn new() -> Self {
        Self::ephemeral()
    }

    /// An ephemeral conversation: its persistence layer is backed by a
    /// throwaway log in a unique temp directory. Used by tests and by
    /// transient helper conversations (e.g. summarisation).
    pub fn ephemeral() -> Self {
        static EPHEMERAL_SEQ: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let seq = EPHEMERAL_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("zend_ephemeral_{nanos}_{seq}"));
        let mut substrate = Substrate::new();
        let persistence = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate)
            .expect("ephemeral SubstratePersistence");
        Self {
            inner: Arc::new(RwLock::new(substrate)),
            allocator: Arc::new(TimelineAllocator::new()),
            maintenance: Arc::new(Mutex::new((persistence.segment_count(), None, false))),
            persistence: Arc::new(Mutex::new(persistence)),
            normalization: Arc::new(Mutex::new(NormalizationCache::default())),
            normalization_warm: Arc::new(Once::new()),
        }
    }

    /// Create a conversation from a freshly-built `(Substrate,
    /// SubstratePersistence)` pair.  Callers that want the walker to
    /// dispatch into the substrate in one pass should use
    /// [`SubstratePersistence::open_in_with_substrate`] and pass both
    /// here.
    pub fn from_parts(substrate: Substrate, persistence: SubstratePersistence) -> Self {
        Self {
            inner: Arc::new(RwLock::new(substrate)),
            allocator: Arc::new(TimelineAllocator::new()),
            maintenance: Arc::new(Mutex::new((persistence.segment_count(), None, false))),
            persistence: Arc::new(Mutex::new(persistence)),
            normalization: Arc::new(Mutex::new(NormalizationCache::default())),
            normalization_warm: Arc::new(Once::new()),
        }
    }

    /// Create a conversation backed by a real [`SubstratePersistence`].
    /// Equivalent to `from_parts(Substrate::new(), persistence)` — for
    /// callers that already have a populated persistence and want an
    /// empty substrate.
    pub fn with_persistence(persistence: SubstratePersistence) -> Self {
        Self::from_parts(Substrate::new(), persistence)
    }

    /// Allocate a fresh [`TimelineId`] and register it against
    /// `(layer, group)` on the substrate.
    pub fn mint_timeline(&self, layer: LayerId, group: GroupId) -> TimelineId {
        let mut view = self.inner.write().unwrap();
        view.mint_timeline(layer, group, &self.allocator)
    }

    /// Look up `(layer, group)` for a previously-minted timeline.
    pub fn timeline_target(&self, timeline: TimelineId) -> Option<(LayerId, GroupId)> {
        self.inner.read().unwrap().timeline_target(timeline)
    }

    /// Register a specific [`TimelineId`] against `(layer, group)` —
    /// idempotent. Used by the resume path to bind a conversation to a
    /// timeline recovered from the redo log instead of minting a fresh one.
    pub fn register_timeline(&self, timeline: TimelineId, layer: LayerId, group: GroupId) {
        self.inner
            .write()
            .unwrap()
            .register_timeline(timeline, layer, group);
    }

    /// Set (or clear) the per-conversation KV-compression override for
    /// `timeline`. Called at conversation creation from the
    /// [`SequenceConfig`]; must run before the first turn seals so each
    /// turn residence inherits it. See [`crate::substrate::ConvCompression`].
    pub fn set_timeline_compression(
        &self,
        timeline: TimelineId,
        compression: Option<crate::substrate::ConvCompression>,
    ) {
        self.inner
            .write()
            .unwrap()
            .set_timeline_compression(timeline, compression);
    }

    /// Set whether `timeline`'s turns are summarised. `false` for append-only
    /// utility/reference layers (repo_map, code_reading) so their turns never
    /// enter the summariser. Called at conversation creation, before the first
    /// turn seals.
    pub fn set_timeline_summarize(&self, timeline: TimelineId, summarize: bool) {
        self.inner
            .write()
            .unwrap()
            .set_timeline_summarize(timeline, summarize);
    }

    /// Acquire an unscored read guard.  The returned guard implements
    /// [`ContentResolver`] but every score lookup returns zero —
    /// appropriate for callers reading structural fields (turn counts,
    /// sealed pointers) without projection.
    ///
    /// Use [`Self::read_scored`] when projecting against a freshly-built
    /// [`ProjectionScores`] from the wide-Q belief scan.
    pub fn read(&self) -> SubstrateRead<'_> {
        SubstrateRead {
            guard: self.inner.read().unwrap(),
            scores: None,
        }
    }

    /// Acquire a read guard bound to an externally-owned
    /// [`ProjectionScores`]. The scores are transient per-projection state —
    /// populated by the reprojection's wide-Q belief scan on the call site's
    /// stack and dropped at end of scope. They are **not** held by the
    /// substrate.
    pub fn read_scored<'a>(&'a self, scores: &'a ProjectionScores) -> SubstrateRead<'a> {
        SubstrateRead {
            guard: self.inner.read().unwrap(),
            scores: Some(scores),
        }
    }

    /// Acquire a target-aware read guard.  The returned [`TargetedRead`]
    /// implements [`ContentResolver`] with proper sibling-timeline masking
    /// for `target.group`, with score lookups returning zero (unscored).
    pub fn read_for(&self, target: ProjectionTarget) -> TargetedRead<'_> {
        TargetedRead::new(self.read(), target)
    }

    /// Target-aware variant of [`Self::read_scored`].
    pub fn read_for_scored<'a>(
        &'a self,
        target: ProjectionTarget,
        scores: &'a ProjectionScores,
    ) -> TargetedRead<'a> {
        TargetedRead::new(self.read_scored(scores), target)
    }

    /// Bounded rolling-window view ranges for an append-only ingest — the
    /// system-prompt blocks plus the most recent `window_turns` sealed turns of
    /// `timeline` (design `docs/unified_wave_inference_engine.md` §4.7). Returned
    /// as raw `(start_block, end_block)` pairs for the scheduler's view borrow.
    ///
    /// `window_turns == 0` (unbounded) or fewer sealed turns than the window
    /// returns `[(0, total_blocks)]` — the whole parent, byte-for-byte the
    /// unwindowed behaviour. Used only on the disable-reprojection ingest path
    /// and only when `CANDLE_CODEREAD_WINDOW_TURNS > 0`; the KV-windowing effect
    /// itself requires golden-token validation on a live model.
    pub fn windowed_ingest_ranges(
        &self,
        timeline: TimelineId,
        window_turns: usize,
        total_blocks: usize,
    ) -> Vec<(usize, usize)> {
        let sub = self.inner.read().unwrap();
        let turn_count = sub.turn_count(timeline);
        // Per-turn start block in the sealed block grid the view borrow indexes.
        // `block_range_of` is `(start, end)`; the system prompt is the blocks
        // before turn 0's start.
        let turn_starts: Vec<usize> = (0..turn_count)
            .map(|i| sub.block_range_of(timeline, TurnIndex(i)).0 as usize)
            .collect();
        let sys_end = turn_starts.first().copied().unwrap_or(0);
        crate::conversation::windowed_ingest_ranges_impl(
            sys_end,
            &turn_starts,
            total_blocks,
            window_turns,
        )
    }

    /// Acquire a write guard for mutating operations (append, set_*).
    pub fn write(&self) -> SubstrateWrite<'_> {
        SubstrateWrite {
            guard: self.inner.write().unwrap(),
        }
    }

    /// Assemble the belief gallery for a section `collection`: every in-scope
    /// turn's wide-Q window mapped to the belief slot it stands for.
    ///
    /// Scope is a **tag partition**, so a tagged corpus and untagged live
    /// conversation never bleed into each other:
    /// - a **tagged** policy (`tags` non-empty, e.g. `["tool"]`) admits only
    ///   turns carrying one of those tags — its fixed calibration corpus;
    /// - an **untagged** policy (`tags` empty) admits only *untagged* turns —
    ///   live conversation, the self-reinforcing case where a past turn that
    ///   scored high gets pulled back in (tagged calibration turns are excluded
    ///   so they don't leak into general memory).
    ///
    /// Label: a turn's tag that names a slot wins (so tagged calibration turns
    /// are labelled directly — no cold-start bootstrap); otherwise the turn is
    /// labelled by the section its own last projection selected in `collection`
    /// (self-reinforcing). Turns that resolve to no slot are skipped. Returns
    /// `(windows, slot_per_window)` for [`crate::provenance::score_slots`].
    pub fn belief_gallery(
        &self,
        collection: &str,
        tags: &[String],
        slot_of: impl Fn(&str) -> Option<usize>,
    ) -> (Vec<Arc<Vec<WideQSig>>>, Vec<usize>) {
        let sub = self.inner.read().unwrap();
        let mut windows: Vec<Arc<Vec<WideQSig>>> = Vec::new();
        let mut slots: Vec<usize> = Vec::new();
        for (sid, e) in sub.all_streams() {
            let Some(StreamDecl::Turn(d)) = &e.decl else {
                continue;
            };
            let in_scope = if tags.is_empty() {
                d.tags.is_empty()
            } else {
                d.tags.iter().any(|t| tags.contains(t))
            };
            if !in_scope {
                continue;
            }
            // Cached decode: the static gallery is decoded once per session
            // (invalidated when a turn's sig blob changes), not re-parsed on every
            // reprojection. `None` covers absent or empty windows.
            let Some(window) = sub.decoded_wide_sig(sid) else {
                continue;
            };
            let slot = d.tags.iter().find_map(|t| slot_of(t)).or_else(|| {
                e.projection_events
                    .as_ref()
                    .map(|b| decode_events(b))
                    .and_then(|evs| {
                        evs.iter()
                            .rev()
                            .find_map(|ev| selected_in_collection(&ev.selection, collection))
                    })
                    .and_then(|name| slot_of(&name))
            });
            let Some(slot) = slot else {
                continue;
            };
            windows.push(window);
            slots.push(slot);
        }
        (windows, slots)
    }

    /// Score every belief-driven collection in the shared system prompt against
    /// its tag-scoped gallery, using `probe` as the query window, into per-section
    /// [`ProjectionScores`].
    ///
    /// Shared by the two probes that drive selection: the scheduler's live
    /// reproject scan (probe = live wide-Q gather over the decode window) and
    /// the post-turn projection event (probe = the finished turn's stored
    /// signature). A collection with an empty gallery or no sections contributes
    /// nothing — its sections read `0.0`.
    pub fn score_belief_collections(
        &self,
        sp: &SystemPromptSchema,
        probe: &[WideQSig],
    ) -> ProjectionScores {
        let mut scores = ProjectionScores::new();
        if probe.is_empty() {
            return scores;
        }
        for item in &sp.items {
            let SystemPromptItem::Collection(coll) = item else {
                continue;
            };
            let n = coll.sections.len();
            if n == 0 {
                continue;
            }
            let slot_of = |name: &str| coll.sections.iter().position(|s| s.name == name);
            let (windows, slots) = self.belief_gallery(&coll.name, &coll.policy.tags, slot_of);
            if windows.is_empty() {
                // The belief loop has nothing to score against, so selection falls
                // back to declaration order (the first tool in the catalog) and
                // pins there. This is THE signature of "wrong tool every turn".
                // An empty gallery is also the persistent steady state of an
                // uncalibrated workspace — and during the calibration load-phase
                // the scan runs on every one of hundreds of per-tool example
                // decodes while the gallery is still being built, so the repeat
                // fires en masse. WARN once per collection (the actionable
                // signal), then demote repeats to TRACE so they don't bury the log
                // during calibration; steady-state behaviour is unchanged.
                static EMPTY_GALLERY_WARNED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
                let first = EMPTY_GALLERY_WARNED
                    .get_or_init(|| Mutex::new(HashSet::new()))
                    .lock()
                    .map(|mut seen| seen.insert(coll.name.clone()))
                    .unwrap_or(false);
                if first {
                    tracing::warn!(
                        target: "candle_conversation::belief",
                        collection = %coll.name,
                        tags = ?coll.policy.tags,
                        sections = n,
                        probe_windows = probe.len(),
                        "belief gallery EMPTY — no tag-scoped gallery turns; tool selection \
                         falls back to catalog order (repeats logged at trace)"
                    );
                } else {
                    tracing::trace!(
                        target: "candle_conversation::belief",
                        collection = %coll.name,
                        "belief gallery still empty"
                    );
                }
                continue;
            }
            let wref: Vec<&[WideQSig]> = windows.iter().map(|w| w.as_slice()).collect();
            // Per-layer-group weights from the collection's `policy.layer_weights`
            // (empty ⇒ uniform — the tool default). Configured in the schema YAML.
            let fresh = score_slots_weighted(probe, &wref, &slots, n, &coll.policy.layer_weights);
            if tracing::enabled!(tracing::Level::DEBUG) {
                let nonzero = fresh.iter().filter(|&&s| s != 0.0).count();
                let top = fresh
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, s)| format!("{}={:.1}", coll.sections[i].name, s))
                    .unwrap_or_default();
                tracing::debug!(
                    target: "candle_conversation::belief",
                    collection = %coll.name,
                    probe_windows = probe.len(),
                    gallery_windows = windows.len(),
                    nonzero_scores = nonzero,
                    top = %top,
                    "belief scan"
                );
            }
            for (s, &score) in coll.sections.iter().zip(&fresh) {
                scores.set_section(s.id, score);
            }
        }
        scores
    }

    /// Score every belief node the projection will consult: the **target
    /// layer's collections** (the tool catalog) plus **every layer's
    /// belief-driven turn groups** (repo_map clusters, code scopes, memory tiers
    /// — each in its own non-target layer). Returns the combined
    /// [`ProjectionScores`] and, for the scheduler's turn-boundary challenger,
    /// each scored turn group's candidate `(turn, fresh_score)` list.
    ///
    /// The projection materializes *all* visible layers and belief-selects each
    /// layer's groups, so the scan must cover all layers too — scoping it to the
    /// target layer alone leaves every non-target turn group on all-zero scores
    /// (a degenerate index tie-break instead of relevance). An empty probe scores
    /// nothing.
    /// `observe` folds this probe's raw turn-group scores into the normalization
    /// hit levels: `true` only on the once-per-turn seal scan
    /// ([`crate::conversation`]'s `last_turn_belief_scores`), `false` on every
    /// live reprojection (which only reads the levels to normalize). See
    /// `docs/provenance_score_normalization.md` §4.2.
    pub fn score_beliefs(
        &self,
        schema: &Schema,
        target: ProjectionTarget,
        probe: &[WideQSig],
        observe: bool,
    ) -> (ProjectionScores, Vec<(GroupId, Vec<(TurnIndex, f32)>)>) {
        let mut scores = ProjectionScores::new();
        let mut candidates = Vec::new();
        if probe.is_empty() {
            return (scores, candidates);
        }
        self.ensure_normalization_warm(schema, target);
        // Collections (the tool catalog) live in the shared system prompt.
        scores = self.score_belief_collections(&schema.system_prompt, probe);
        // Belief-driven turn groups live across every layer.
        for layer in &schema.layers {
            candidates.extend(self.score_belief_groups(layer, target, probe, &mut scores, observe));
        }
        (scores, candidates)
    }

    /// Kick the normalization warm-up — building the hit levels from the
    /// substrate's existing **sealed dialogue turns** so a freshly-loaded process
    /// starts WARM (the cache is runtime-only, so without this a restart would be
    /// cold until new traffic accrued). See `docs/provenance_score_normalization.md`
    /// §4.4.
    ///
    /// The replay is spawned **once** (via [`Once`]) onto a detached background
    /// thread, so its O(replayed turns × gallery) scan never blocks the FIRST live
    /// reprojection on the decode hot path (it previously ran inline under this
    /// call — a multi-second first-token stall on a large substrate). Early
    /// reprojections may read a still-warming cache: that is best-effort by
    /// construction — the levels are runtime-derived and the once-per-turn
    /// seal-observe keeps warming them — so correctness is unaffected; only how
    /// quickly generic candidates get discounted. The handle is all-`Arc`, so the
    /// clone moved into the thread shares the same substrate + normalization cache.
    fn ensure_normalization_warm(&self, schema: &Schema, target: ProjectionTarget) {
        self.normalization_warm.call_once(|| {
            let this = self.clone();
            let schema = schema.clone();
            std::thread::spawn(move || {
                this.warm_normalization_from_substrate(&schema, target);
            });
        });
    }

    /// The warm-replay body (see [`Self::ensure_normalization_warm`]). Scores the
    /// last [`WARM_REPLAY_MAX_TURNS`] dialogue turns as seal-style observes against
    /// every belief group, folding the hit levels. Turns are ordered by
    /// `(timeline, turn index)` so the warmed levels don't depend on `HashMap`
    /// iteration order, and bounded to the recent window (the asymmetric EWMA
    /// converges in a few dozen steps, so older turns barely move it — this caps
    /// the one-time cost on a huge substrate). It reuses the live turn-group scan
    /// with `observe = true`; the normalized scores it also computes are discarded.
    /// Only dialogue turns (empty-tagged — gallery turns are tagged) are probes; the
    /// still-decoding current turn is excluded (no sealed signature yet). Runs off
    /// the hot path, reading the substrate under a shared lock and briefly locking
    /// the normalization cache per group, so it coexists with live reprojections.
    fn warm_normalization_from_substrate(&self, schema: &Schema, target: ProjectionTarget) {
        let mut probes: Vec<(u64, u32, Vec<WideQSig>)> = {
            let sub = self.inner.read().unwrap();
            sub.all_streams()
                .filter_map(|(_sid, e)| {
                    let Some(StreamDecl::Turn(d)) = e.decl.as_ref() else {
                        return None;
                    };
                    if !d.tags.is_empty() {
                        return None;
                    }
                    let sig = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b))?;
                    (!sig.is_empty()).then_some((d.timeline_id, d.turn_index, sig))
                })
                .collect()
        };
        probes.sort_by_key(|(tl, idx, _)| (*tl, *idx));
        let start = probes.len().saturating_sub(WARM_REPLAY_MAX_TURNS);
        for (_, _, probe) in &probes[start..] {
            let mut throwaway = ProjectionScores::new();
            for layer in &schema.layers {
                let _ = self.score_belief_groups(layer, target, probe, &mut throwaway, true);
            }
        }
    }

    /// Score every belief-driven **turn group** in `layer` against its own turns
    /// (self-match), folding the fresh per-turn scores into `scores`.
    ///
    /// The turn-group analogue of [`Self::score_belief_collections`]: where a
    /// collection scans a tag-scoped gallery of *other* turns mapped to section
    /// slots, a turn group's retrieval target IS the turn itself, so each
    /// candidate turn is its own slot (identity map) and the probe scores against
    /// each turn's stored `WideQSig` window directly. A `Sequence` (recency) group
    /// is skipped — it isn't belief-driven. The group's timeline is resolved the
    /// same way the projection will (`resolve_turn_timeline(Some(target), …)`) and
    /// its turns enumerated `0..turn_count(timeline)` exactly as selection does,
    /// so the `(timeline, index)` keys line up with what selection reads back.
    pub fn score_belief_groups(
        &self,
        layer: &LayerSchema,
        target: ProjectionTarget,
        probe: &[WideQSig],
        scores: &mut ProjectionScores,
        observe: bool,
    ) -> Vec<(GroupId, Vec<(TurnIndex, f32)>)> {
        use crate::persistence::content_hash::turn_stream_id;
        let mut per_group: Vec<(GroupId, Vec<(TurnIndex, f32)>)> = Vec::new();
        if probe.is_empty() {
            return per_group;
        }
        let sub = self.inner.read().unwrap();
        for group in &layer.groups {
            if !group.is_belief_driven() {
                continue;
            }
            let Some(timeline) = sub.resolve_turn_timeline(Some(target), group.id) else {
                continue;
            };
            // Enumerate the group's turns exactly as selection does — the whole
            // resolved timeline, `0..turn_count`, fetched by stream id — instead
            // of scanning `all_streams()` per group (which is O(all timelines'
            // streams) on the reproject hot path).
            let count = sub.turn_count(timeline);
            // Per candidate turn: its full sig plus the self-referencing sub-window
            // seams recorded on it. A turn with no seams scores as one whole-turn
            // window (the prior behaviour); a turn with N seams scores as N+1
            // focused windows that all resolve back to it — so a query matching one
            // structural region of a prefilled listing surfaces the whole turn
            // without diluting against the rest.
            let mut arcs: Vec<Option<Arc<Vec<WideQSig>>>> = Vec::new();
            let mut arc_turn: Vec<TurnIndex> = Vec::new();
            let mut arc_bounds: Vec<Vec<(usize, usize)>> = Vec::new();
            for i in 0..count {
                let idx = TurnIndex(i);
                // Summary forest nodes are selected only by the score-density
                // path, never the belief/rule path — mirror project.rs and skip
                // them so a summary can't take a raw turn's belief slot.
                if sub
                    .tree_meta_of(timeline, idx)
                    .map(|m| m.kind.is_summary())
                    .unwrap_or(false)
                {
                    continue;
                }
                // Keep EVERY non-summary (Normal) turn in `arc_turn`, so it is the
                // COMPLETE Normal subsequence the couplings project onto. A turn that
                // has no wide-Q sig (e.g. a prefilled tool-response half) must still
                // hold its position, or `over_normals`/`exchanges` would fuse a
                // coupled call with the wrong later turn. A sig-less turn contributes
                // no gallery window (empty bounds) but still joins its exchange.
                arc_turn.push(idx);
                match sub.decoded_wide_sig(turn_stream_id(timeline.raw(), i)) {
                    Some(window) => {
                        // Self-referencing projection events mark sub-window seams.
                        // Read the memoized (sorted, deduped) seams — decoded from the
                        // events JSON once per session, not re-parsed for every gallery
                        // turn on every reprojection — and derive contiguous
                        // `[start, end)` bounds; no seams ⇒ one whole-turn window.
                        // `subwindow_bounds` ignores any seam at/past `len`, so the raw
                        // offsets are safe to pass straight through.
                        let len = window.len();
                        let seams = sub.decoded_seams(turn_stream_id(timeline.raw(), i));
                        arc_bounds.push(subwindow_bounds(len, &seams));
                        arcs.push(Some(window));
                    }
                    None => {
                        arc_bounds.push(Vec::new());
                        arcs.push(None);
                    }
                }
            }
            if arc_turn.is_empty() {
                continue;
            }
            // Group the kept turns into EXCHANGES before scoring. A code-read scope
            // (and any tool round-trip) is a *coupled pair* — a `<tool_call>` turn
            // and its `<tool_response>` turn joined by a `TurnCoupling` record — and
            // provenance must hit the pair as ONE unit (`exchange_of`): scoring the
            // two halves as separate candidates splits the scope's vote and lets the
            // generic call framing (near-identical across every scope) compete on
            // its own. `arc_turn` is the COMPLETE Normal subsequence in chronological
            // order, so `over_normals` maps the call-turn indices straight onto arc
            // positions and the response is always the next position. Uncoupled turns
            // are their own singleton exchange (no behaviour change).
            let couplings = over_normals(&sub.couplings_of(timeline), &arc_turn);
            let ex_ranges = exchanges(&couplings, arc_turn.len());
            let mut ex_slot = vec![0usize; arc_turn.len()];
            for (slot, r) in ex_ranges.iter().enumerate() {
                for ai in r.clone() {
                    ex_slot[ai] = slot;
                }
            }
            let n_slots = ex_ranges.len();
            // Flatten every turn's sub-windows into gallery windows, all tagged with
            // their EXCHANGE slot (`wslot[i] = exchange index`). So `score_slots`
            // aggregates a whole round-trip — every sub-window of the call AND the
            // response — into ONE case (best-token agreement across the pair),
            // rather than letting the halves (or a turn's own regions) compete as
            // separate cases and split the scope's vote. The seams still bound the
            // windows, ready for the diverse-window step, but never fight each
            // other. Then the L46-weighted vote (§83) decides the exchange.
            let mut wref: Vec<&[WideQSig]> = Vec::new();
            let mut wslot: Vec<usize> = Vec::new();
            for (ai, arc) in arcs.iter().enumerate() {
                let Some(arc) = arc else {
                    continue; // sig-less turn: no window, but keeps its exchange slot
                };
                for &(s, e) in &arc_bounds[ai] {
                    if e > s {
                        wref.push(&arc[s..e]);
                        wslot.push(ex_slot[ai]);
                    }
                }
            }
            if wref.is_empty() {
                continue;
            }
            // Per-layer-group vote weights from the group's `policy.layer_weights`
            // (empty ⇒ uniform). Repo_map peaks on L46 (§83); other groups inherit
            // uniform. Configured in the schema YAML, not hard-coded.
            let fresh =
                score_slots_weighted(probe, &wref, &wslot, n_slots, &group.policy.layer_weights);
            // Normalize the raw scores against each EXCHANGE's learned hit level so
            // selection compares candidates on a common 0-1000 band, not a shared
            // absolute scale (docs/provenance_score_normalization.md). Scope =
            // group@timeline — a re-scan mints a new timeline, hence a fresh scope,
            // resetting learning for the regenerated clusters; child = the
            // exchange's head-turn index (stable across scans).
            let scope = ScopeKey::turn_group(group.id.raw() as u64, timeline.raw());
            let raw_pairs: Vec<(ChildKey, f32)> = ex_ranges
                .iter()
                .enumerate()
                .map(|(slot, r)| (ChildKey::turn(arc_turn[r.start].0 as u64), fresh[slot]))
                .collect();
            // One lock for the read-then-(maybe)-write, so no other thread mutates
            // the levels between this turn's normalize and observe. Learning only
            // fires on the once-per-turn seal scan, not on every reprojection.
            let normed = {
                let mut cache = self.normalization.lock().unwrap();
                let normed = cache.normalize(&scope, &raw_pairs);
                if observe {
                    cache.observe(&scope, &raw_pairs);
                }
                normed
            };
            // Stamp EVERY member turn of an exchange with its (shared) normalized
            // score, so provenance selecting either half brings in the whole
            // round-trip — never half a tool call.
            let mut cands: Vec<(TurnIndex, f32)> = Vec::with_capacity(arc_turn.len());
            for (slot, r) in ex_ranges.iter().enumerate() {
                let sc = normed.get(slot).map(|(_, s)| *s).unwrap_or(0.0);
                for ai in r.clone() {
                    let idx = arc_turn[ai];
                    scores.set_turn(timeline, idx, sc);
                    cands.push((idx, sc));
                }
            }
            per_group.push((group.id, cands));
        }
        per_group
    }

    /// Atomically append a turn to the substrate.
    ///
    /// `write` carries the turn's text, token IDs, block range, and
    /// GPU-resident sealed K/V snapshot.  `migrate_to_cpu` is called
    /// to move the bytes to the warm (CPU) tier before storing.
    pub fn record_turn(
        &self,
        timeline: TimelineId,
        role: Role,
        write: TurnPartWrite,
        migrate_to_cpu: impl FnMut(&[SealedSequence]) -> candle::Result<Vec<SealedSequence>>,
    ) -> candle::Result<TurnIndex> {
        let block_start = write.block_start;
        let block_end = write.block_end;
        // Capture the segment layout and gather-scope tags before the write
        // moves into the substrate — the redo-log `TurnDecl` carries them
        // verbatim so reload can reconstruct `TurnPart::layout` (per-half text +
        // spans + `/no_think`) without re-tokenising, and re-tag the turn for the
        // provenance gallery's `tags:` scoping.
        let segments = write.layout.segments.clone();
        let tags = write.tags.clone();
        let idx = {
            let mut view = self.inner.write().unwrap();
            view.append_complete(timeline, write, migrate_to_cpu)?
        };
        // Record the turn's structure into the redo log.
        let (layer_id, group_id) = self
            .timeline_target(timeline)
            .map(|(l, g)| (l.raw(), g.raw()))
            .unwrap_or((0, 0));
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: timeline.raw(),
            turn_index: idx.0,
            turn_id_day: 0,
            turn_id_seq: idx.0 + 1,
            role: match role {
                Role::System => 0,
                Role::User => 1,
                Role::Assistant => 2,
            },
            block_start,
            block_end,
            layer_id,
            group_id,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments,
            tags,
        });
        self.declare_and_mirror(decl, "persist turn")?;
        Ok(idx)
    }

    /// Declare a stream in the persistence redo log AND mirror the decl into
    /// the LIVE substrate. The reload walker only installs decls at startup,
    /// but live readers — the tag-scoped belief gallery reading a turn's tags,
    /// section-by-name lookups — consult the in-memory decls, so a stream
    /// declared in the current session (every calibration turn or section on
    /// a fresh substrate) must be visible to them without a restart.
    fn declare_and_mirror(&self, decl: StreamDecl, err_ctx: &str) -> candle::Result<StreamId> {
        let stream_id = self
            .persistence
            .lock()
            .unwrap()
            .declare_stream(&decl)
            .map_err(|e| candle::Error::Msg(format!("{err_ctx}: {e}")))?;
        self.write().apply_stream_decl(stream_id, decl);
        Ok(stream_id)
    }

    /// Union of the gather-scope tags on `children`'s TurnDecls, dedup'd
    /// with child order preserved. Empty when every child is untagged
    /// (dialogue turns), so summaries of untagged content stay in the
    /// untagged partition. Used by the summariser to stamp a summary node
    /// with the tags of the turns it compresses — a code_read leaf inherits
    /// its scan turn's `["code", <path>]`, a summary-of-summaries the union
    /// of its children's.
    pub fn union_turn_tags(&self, timeline: TimelineId, children: &[TurnIndex]) -> Vec<String> {
        use crate::persistence::content_hash::turn_stream_id;
        let read = self.inner.read().unwrap();
        let mut tags: Vec<String> = Vec::new();
        for c in children {
            let sid = turn_stream_id(timeline.raw(), c.0);
            let Some(entry) = read.stream_of(sid) else {
                continue;
            };
            let Some(StreamDecl::Turn(d)) = &entry.decl else {
                continue;
            };
            for t in &d.tags {
                if !tags.contains(t) {
                    tags.push(t.clone());
                }
            }
        }
        tags
    }

    /// Append a summariser-allocated turn (SoT leaf or SoS internal)
    /// and persist its declaration to the redo log.
    ///
    /// The summariser allocates these turns to back tree nodes; they
    /// carry no KV chunks (`block_range = 0..0`) and `token_count` is
    /// the placeholder for the summary text.  Without persistence, on
    /// reopen the walker would replay [`TreeMetadata`] records for
    /// these indices but find no matching [`TurnDecl`], leaving
    /// orphan `tree_meta` entries that the score-density selector
    /// would then pick and elevate would fail to lift.
    ///
    /// Drops the auto-pending entry that [`append_with_blocks`] pushed
    /// — summary turns are not Normal and shouldn't loop back through
    /// the pending queue.
    pub fn record_summary_turn(
        &self,
        timeline: TimelineId,
        token_count: usize,
    ) -> candle::Result<TurnIndex> {
        let idx = {
            let mut view = self.inner.write().unwrap();
            let idx = view.append_with_blocks(timeline, token_count, 0, 0);
            view.pop_pending_summary(timeline);
            idx
        };
        let (layer_id, group_id) = self
            .timeline_target(timeline)
            .map(|(l, g)| (l.raw(), g.raw()))
            .unwrap_or((0, 0));
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: timeline.raw(),
            turn_index: idx.0,
            turn_id_day: 0,
            turn_id_seq: idx.0 + 1,
            role: 2,
            block_start: 0,
            block_end: 0,
            layer_id,
            group_id,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        self.declare_and_mirror(decl, "persist summary turn")?;
        Ok(idx)
    }

    /// Rebuild the in-RAM [`Substrate`] from the persistence redo log — the
    /// §5.6 / §16.12 substrate-reload path run on daemon restart.
    ///
    /// **Cold-only restart.** The substrate is, by design, the on-disk redo
    /// log; warm (RAM) and hot (VRAM) tiers belong to the inference engine
    /// and are demand-populated. Reload therefore:
    /// - Walks every persisted turn stream in `(timeline, turn_index)`
    ///   order.
    /// - Replays **tokens** (for text history) into the in-RAM substrate —
    ///   small (RAM-resident) and load-bearing for text display.
    /// - Records each turn's stream metadata (`block_start`/`block_end`,
    ///   role, timeline) so projection knows the turn exists and where its
    ///   KV lives on disk.
    /// - **Does not materialize KV into VRAM.** Each restored turn's
    ///   `sealed` is an empty `Vec<SealedSequence>` — a "cold" marker. The
    ///   inject path materializes through the warm pool on demand (see
    ///   the engine's `ensure_hot` orchestrator).
    ///
    /// Returns the number of turns restored.
    pub fn reconstruct_from_log(
        &self,
        n_layers: usize,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> candle::Result<usize> {
        // Substrate's per-stream / per-timeline state was populated
        // in one walker pass during `SubstratePersistence::open_in_with_substrate`
        // — no mirror step needed here.  This pass replays turn-decl
        // records into the substrate's per-turn KV residence slots
        // (the cold-load setup that demands knowing layer count) and
        // then runs the post-reload sweeps for the summary tree.
        let decls = {
            let substrate = self.read();
            crate::persistence::resume::recovered_turn_decls(&substrate)
        };
        let total = decls.len();
        tracing::info!(turns = total, "substrate reconstruct: begin replay loop");
        let mut restored = 0usize;
        let mut skipped_corrupt = 0usize;
        for (i, mut decl) in decls.into_iter().enumerate() {
            // Report turns processed so far (restored + skipped) so the daemon's
            // loading bar advances steadily even across corrupt-turn skips.
            if let Some(p) = progress {
                p(i, total);
            }
            // Per-turn fault isolation.  A single turn whose
            // persisted state is inconsistent (e.g. a daemon was
            // killed between chunk writes and the final TurnDecl
            // seal update, leaving a TurnDecl with
            // `block_end == block_start` but real chunks on disk)
            // should not poison the entire reload — every other turn
            // is independent.  Log the failure and tombstone the
            // corrupt turn's timeline so the next reload doesn't
            // re-encounter it.
            let recover_result = {
                let mut p = self.persistence.lock().unwrap();
                let substrate_read = self.read();
                // Metadata only — token ids, signatures, and the chunk-index
                // token count. The turn's KV payload bytes stay on disk; a
                // later projection reads them via the cold→hot elevation
                // (`recover_turn_grid`).
                let r =
                    crate::persistence::resume::recover_turn_meta(&mut p, &substrate_read, &decl);
                let cr = if r.is_ok() {
                    crate::persistence::resume::recover_turn_cold_refs(
                        &substrate_read,
                        &decl,
                        n_layers,
                    )
                } else {
                    Ok(Default::default())
                };
                r.and_then(|recovered| cr.map(|cold_refs| (recovered, cold_refs)))
            };
            let (recovered, cold_refs) = match recover_result {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!(
                        timeline_id = decl.timeline_id,
                        turn_index = decl.turn_index,
                        "skipping corrupt turn during substrate reload: {e}",
                    );
                    skipped_corrupt += 1;
                    if let Some(timeline) = TimelineId::from_raw(decl.timeline_id) {
                        self.write().tombstone_timeline(timeline);
                        // Best-effort durable mark — failures here
                        // just mean the next reload will encounter
                        // the same turn and skip it again, which is
                        // still correct, so we don't propagate.
                        if let Ok(mut p) = self.persistence.lock() {
                            let _ = p.write_tombstone(timeline.raw());
                        }
                    }
                    continue;
                }
            };
            let timeline = TimelineId::from_raw(decl.timeline_id).ok_or_else(|| {
                candle::Error::Msg("reconstruct: turn has zero timeline_id".into())
            })?;
            let token_count = recovered.token_count;
            let mut view = self.write();
            if let (Some(layer), Some(group)) = (
                LayerId::from_raw(decl.layer_id),
                GroupId::from_raw(decl.group_id),
            ) {
                view.register_timeline(timeline, layer, group);
            }
            // Cold-marker sealed: an empty `Vec<SealedSequence>` flags the
            // turn as on-disk-only. The runtime inject path detects the
            // empty sealed and routes through the engine's `ensure_hot`
            // orchestrator (cold → warm → hot) before borrowing into a
            // view slot.
            //
            // `cold_refs = Some(...)` lights up the residence's cold
            // tier so the new bulk `elevate_to_hot` classifier routes
            // the turn through cold_to_hot. Without it the residence
            // would be `(hot, warm, cold) = (None, None, None)` and
            // the classifier would tag the turn `missing` on the very
            // first projection that needs it. `cold_refs = None` is
            // a recoverable-token-only turn (no persisted chunks) —
            // the substrate keeps it discoverable but it stays unable
            // to materialise KV.
            view.restore_turn(
                timeline,
                TurnLayout::new(std::mem::take(&mut decl.segments)),
                TokenBuffer::from(recovered.token_ids),
                token_count,
                cold_refs,
                decl.block_start,
                decl.block_end,
            );
            restored += 1;
        }
        // Arm low-priority reconciliation per timeline. The summary forest is
        // immutable and its canonical ternary shape is a pure function of the
        // leaves (`docs/immutable_summary_forest.md`), so the reload doesn't
        // re-summarise anything — it just asks the summariser to rebuild any
        // internal node that's missing (a crash between sealing leaves and their
        // parent) or non-canonical (binary nodes from the superseded AVL, which
        // `mark_for_reconcile` purges). For a clean forest `reconcile_next`
        // returns `None` on the first pass and the hint clears immediately, so a
        // healthy restart does zero probe work. Reconcile is strictly
        // lower-priority than live turns, so it never floods startup the way the
        // old "re-enqueue the orphan tail" sweep did.
        {
            let timelines: Vec<TimelineId> = self.read().all_timeline_ids().collect();
            let mut view = self.write();
            for tl in timelines {
                view.mark_for_reconcile(tl);
            }
        }
        if let Some(p) = progress {
            p(total, total);
        }
        let read = self.read();
        let n_sections = read.section_count();
        let n_timelines = read.timeline_count();
        let n_conversations = read.conversation_count();
        drop(read);
        tracing::info!(
            sections = n_sections,
            timelines = n_timelines,
            conversations = n_conversations,
            turns = restored,
            skipped_corrupt = skipped_corrupt,
            "substrate reload complete",
        );
        Ok(restored)
    }

    /// Read a cold turn's per-layer chunk grid from the redo log so the
    /// caller can run the warm→hot leg (`load_stream` per layer) and
    /// install the resulting `Vec<SealedSequence>`s on the substrate
    /// via the `elevate_to_hot` orchestrator (`ColdRecall`).
    ///
    /// Returns `Ok(None)` when the turn doesn't have a recoverable chunk
    /// grid — e.g. its `Tokens` record is durable but `Chunks` records
    /// haven't yet landed (the async persist callback was still pending
    /// when the daemon shut down).
    pub fn recover_turn_chunks(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
        n_layers: usize,
    ) -> candle::Result<Option<crate::persistence::resume::TurnChunkGrid>> {
        use crate::persistence::resume::{recover_turn_grid, recovered_turn_decls};
        let stream_id = crate::persistence::content_hash::turn_stream_id(timeline.raw(), index.0);
        let mut p = self.persistence.lock().unwrap();
        // We need the turn's `StreamDecl` to drive `recover_turn_grid`.
        // Walk the substrate's persisted decls and pick the one matching
        // this (timeline, index). The decl set is small and rebuilt once
        // at restart, so a linear scan is fine.
        let substrate_read = self.read();
        let decls = recovered_turn_decls(&substrate_read);
        let decl = match decls
            .into_iter()
            .find(|d| d.timeline_id == timeline.raw() && d.turn_index == index.0)
        {
            Some(d) => d,
            None => return Ok(None),
        };
        let substrate = self.read();
        let grid = recover_turn_grid(&mut p, &substrate, &decl, n_layers)
            .map_err(|e| candle::Error::Msg(format!("recover_turn_chunks: {e}")))?;
        if grid.is_empty() {
            return Ok(None);
        }
        let _ = stream_id; // (computed for diagnostics if needed later)
        Ok(Some(grid))
    }

    /// Batched cold→hot load. Loads every key in `keys`, taking the
    /// persistence + substrate locks **once** and scanning the recovered
    /// turn-decl table **once** for the whole batch. `recovered_turn_decls`
    /// walks every stream (O(total streams)); doing that per cold turn would
    /// make the elevate cold-load loop O(cold_turns × total_streams), quadratic
    /// in a deep conversation. Returns one `(key, result)` per input key, in
    /// the same order, so the caller can pair each result with its plan entry.
    pub fn cold_load_turns_into_hot(
        &self,
        keys: &[TurnKey],
        backings: &[candle_nn::kv_cache::ChunkedKvBacking],
        device: &candle::Device,
        stager: &mut crate::persistence::cold_load::ColdLoadStager,
    ) -> Vec<(TurnKey, candle::Result<Option<(Vec<SealedSequence>, u64)>>)> {
        use crate::persistence::content_hash::turn_stream_id;
        use crate::persistence::streams::StreamDecl;
        use crate::persistence::transfer::load_turn_into_hot;

        let mut p = self.persistence.lock().unwrap();
        let substrate = self.read();
        keys.iter()
            .map(|&key| {
                // Resolve each turn's decl by its deterministic stream id — an
                // O(1) `streams` map hit — rather than scanning, cloning, and
                // sorting every turn decl in the whole substrate. The stream that
                // carries the decl is the same one whose chunks the load reads, so
                // one `stream_of` serves both. This keeps the cold-load O(keys),
                // not O(turns-ingested-so-far).
                let stream_id = turn_stream_id(key.timeline.raw(), key.index.0);
                let Some(stream) = substrate.stream_of(stream_id) else {
                    return (key, Ok(None));
                };
                let Some(StreamDecl::Turn(decl)) = &stream.decl else {
                    return (key, Ok(None));
                };
                let kv_bytes_total: u64 = stream.chunks.values().map(|loc| loc.payload_len).sum();
                let result = load_turn_into_hot(backings, device, &mut p, &substrate, decl, stager)
                    .map(|sealed| Some((sealed, kv_bytes_total)));
                (key, result)
            })
            .collect()
    }

    /// Clear a turn's hot sealed grid, releasing VRAM arena chunks via
    /// dropping its ChunkGid Arcs. Returns `true` if hot bytes were
    /// dropped (see [`Substrate::clear_turn_sealed`]).
    pub fn clear_turn_sealed(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        self.write().clear_turn_sealed(timeline, index)
    }

    /// Hot-tier VRAM byte snapshot (sum across every turn whose `sealed`
    /// carries an actual chunk grid).
    pub fn hot_turn_bytes(&self) -> usize {
        self.read().hot_turn_bytes()
    }

    /// Pinned-section byte snapshot.
    pub fn section_bytes(&self) -> usize {
        self.read().section_bytes()
    }

    /// Byte size of a single hot turn (for the pre-flight evict
    /// accounting). `None` if cold or unknown.
    pub fn turn_hot_bytes(&self, timeline: TimelineId, index: TurnIndex) -> Option<usize> {
        self.read().turn_hot_bytes(timeline, index)
    }

    /// FIFO-oldest hot-resident turn excluding `except`.
    pub fn oldest_hot_turn_except(&self, except: TurnKey) -> Option<TurnKey> {
        self.read().oldest_hot_turn_except(except)
    }

    /// The sidebar label for `timeline`, or `None` if no label has been
    /// recorded — either the conversation hasn't had its first user turn
    /// yet, or the recovered redo log carries no label for it.
    pub fn label_of(&self, timeline: TimelineId) -> Option<String> {
        self.read().label_of(timeline).map(|s| s.to_string())
    }

    /// The client-supplied `conv_id` string for `timeline`, or `None` if
    /// no submit has been recorded yet. Recovered from the redo log on
    /// substrate reload — drives the daemon's sidebar id field.
    pub fn conv_id_of(&self, timeline: TimelineId) -> Option<String> {
        self.read().conv_id_of(timeline).map(|s| s.to_string())
    }

    /// Persist a sidebar label for `timeline`. Last-write-wins on the
    /// underlying `RecordType::Label`; this writes the same record the
    /// titler writes, preserving whatever `conv_id` is already known
    /// for this timeline.
    pub fn set_conversation_label(&self, timeline: TimelineId, label: &str) -> candle::Result<()> {
        if label.is_empty() {
            return Ok(());
        }
        let conv_id = self.conv_id_of(timeline).unwrap_or_default();
        let custom = self.read().custom_of(timeline).cloned().unwrap_or_default();
        {
            let mut p = self.persistence.lock().unwrap();
            p.write_conv_meta(timeline.raw(), &conv_id, label, &custom)
                .map_err(|e| candle::Error::Msg(format!("write_conv_meta: {e}")))?;
        }
        self.write().set_label(timeline, label);
        Ok(())
    }

    /// Merge a single `(key, value)` into `timeline`'s `custom` metadata
    /// bag and persist the full ConvMeta. Used as a content-addressed
    /// cache tag by utility ingests (code_read / repo_map). Reads the
    /// sibling fields (conv_id, label, existing custom) so the Label
    /// record stays complete.
    pub fn set_conversation_metadata(
        &self,
        timeline: TimelineId,
        key: &str,
        value: &str,
    ) -> candle::Result<()> {
        if key.is_empty() {
            return Ok(());
        }
        let mut one = std::collections::BTreeMap::new();
        one.insert(key.to_string(), value.to_string());
        self.set_conversation_metadata_many(timeline, &one)
    }

    /// Merge several `(key, value)` pairs into `timeline`'s `custom`
    /// metadata in a single persisted Label record (cheaper than one
    /// `set_conversation_metadata` per key when tagging a conversation
    /// with several fields at once).
    pub fn set_conversation_metadata_many(
        &self,
        timeline: TimelineId,
        kv: &std::collections::BTreeMap<String, String>,
    ) -> candle::Result<()> {
        if kv.is_empty() {
            return Ok(());
        }
        let conv_id = self.conv_id_of(timeline).unwrap_or_default();
        let label = self.read().label_of(timeline).unwrap_or("").to_string();
        let mut custom = self.read().custom_of(timeline).cloned().unwrap_or_default();
        for (k, v) in kv {
            custom.insert(k.clone(), v.clone());
        }
        {
            let mut p = self.persistence.lock().unwrap();
            p.write_conv_meta(timeline.raw(), &conv_id, &label, &custom)
                .map_err(|e| candle::Error::Msg(format!("write_conv_meta: {e}")))?;
        }
        self.write().merge_custom(timeline, kv);
        Ok(())
    }

    /// `timeline`'s `custom` metadata bag, or `None` if unregistered.
    pub fn conversation_metadata(
        &self,
        timeline: TimelineId,
    ) -> Option<std::collections::BTreeMap<String, String>> {
        self.read().custom_of(timeline).cloned()
    }

    /// Every live timeline whose `custom` metadata contains `key == value`.
    /// The reload-time cache lookup utility ingests use to skip units
    /// already present in the substrate. Tombstoned timelines are excluded.
    pub fn find_timelines_by_metadata(&self, key: &str, value: &str) -> Vec<TimelineId> {
        self.read().timelines_with_metadata(key, value)
    }

    /// Distinct `custom[key]` values across live timelines — a one-pass
    /// snapshot for O(1) membership probing (e.g. the resume cache).
    pub fn metadata_values_for_key(&self, key: &str) -> std::collections::HashSet<String> {
        self.read().metadata_values_for_key(key)
    }

    /// Live timelines carrying `key`, paired with that key's value.
    /// Drives ingest reconciliation (tombstone units whose source is gone).
    pub fn timelines_with_metadata_key(&self, key: &str) -> Vec<(TimelineId, String)> {
        self.read().timelines_with_metadata_key(key)
    }

    /// Persist the client-supplied `conv_id` for `timeline`. Idempotent;
    /// the typical caller is the daemon's chat handler, invoking this on
    /// every submit so the conv_id reaches the redo log immediately
    /// (well before the titler completes). The current `label` is
    /// preserved, so this can be called freely at any point in the
    /// conversation's lifecycle.
    pub fn set_conversation_conv_id(
        &self,
        timeline: TimelineId,
        conv_id: &str,
    ) -> candle::Result<()> {
        if conv_id.is_empty() {
            return Ok(());
        }
        let label = self.read().label_of(timeline).unwrap_or("").to_string();
        let custom = self.read().custom_of(timeline).cloned().unwrap_or_default();
        {
            let mut p = self.persistence.lock().unwrap();
            p.write_conv_meta(timeline.raw(), conv_id, &label, &custom)
                .map_err(|e| candle::Error::Msg(format!("write_conv_meta: {e}")))?;
        }
        self.write().set_conv_id(timeline, conv_id);
        Ok(())
    }

    /// Every conversation the workspace substrate knows about —
    /// `(timeline, conv_id, label, archived, order)` tuples drawn from the
    /// in-RAM `Substrate::timelines` map (`order` = creation-order rank).
    /// Drives `GET /v1/conversations` directly; no sidecar involved.
    pub fn known_conversations(&self) -> Vec<(TimelineId, String, String, bool, u64)> {
        self.read().known_conversations()
    }

    /// Set a conversation's `archived` lifecycle flag and persist it
    /// as a `RecordType::ConvState` record. Idempotent: if the
    /// substrate already holds the requested state, the record is
    /// not written and the call returns `Ok(())` without touching the
    /// log.
    ///
    /// Last-write-wins on replay — toggling archive↔unarchive each
    /// appends one small record (~ 16 bytes payload + framing); a
    /// subsequent compaction collapses the chain to one record per
    /// timeline.
    pub fn set_conversation_archived(
        &self,
        timeline: TimelineId,
        archived: bool,
    ) -> candle::Result<()> {
        let changed = self.write().set_archived(timeline, archived);
        if !changed {
            return Ok(());
        }
        let state = crate::persistence::manifest::ConvState { archived };
        let mut p = self.persistence.lock().unwrap();
        p.write_conv_state(timeline.raw(), state)
            .map_err(|e| candle::Error::Msg(format!("write_conv_state: {e}")))?;
        Ok(())
    }

    /// Whether `timeline` is currently archived. Untouched / unknown
    /// timelines return `false`.
    pub fn is_conversation_archived(&self, timeline: TimelineId) -> bool {
        self.read().is_archived(timeline)
    }

    /// Tombstone `timeline` — marks it logically deleted both
    /// in-RAM (so projection retrieval stops surfacing its turns on
    /// the next query) and on disk (via a
    /// [`crate::persistence::record::RecordType::Tombstone`]
    /// record).  The compactor drops the underlying records on the
    /// next compaction pass; ordinary reads never see them.
    pub fn tombstone_timeline(&self, timeline: TimelineId) -> candle::Result<()> {
        self.write().tombstone_timeline(timeline);
        let mut p = self.persistence.lock().unwrap();
        p.write_tombstone(timeline.raw())
            .map_err(|e| candle::Error::Msg(format!("write_tombstone: {e}")))?;
        Ok(())
    }

    /// Couple `from_turn` to the tool response that follows it — in-RAM (so this
    /// session's summariser groups the exchange immediately) and on disk (a
    /// [`crate::persistence::record::RecordType::TurnCoupling`] record, so the
    /// grouping survives reload).
    ///
    /// Must be called before the response turn is submitted: that ordering is
    /// what stops the summariser observing half an exchange and freezing a leaf
    /// over it.
    pub fn couple_turn(&self, timeline: TimelineId, from_turn: u32) -> candle::Result<()> {
        self.write().couple_turn(timeline, from_turn);
        let mut p = self.persistence.lock().unwrap();
        p.write_turn_coupling(timeline.raw(), from_turn)
            .map_err(|e| candle::Error::Msg(format!("write_turn_coupling: {e}")))?;
        Ok(())
    }

    /// Whether `timeline` has been tombstoned.
    pub fn is_timeline_tombstoned(&self, timeline: TimelineId) -> bool {
        self.read().is_tombstoned(timeline)
    }

    /// Mark `timeline` for distillation at `mode` — in-RAM (so the compactor
    /// sees it this session) and on disk (a
    /// [`crate::persistence::record::RecordType::Distilled`] record, so it
    /// survives reload). Its turns shed content at the next compaction. A later
    /// call may upgrade the mode (e.g. provenance-only → text-only on archive).
    pub fn distill_timeline(&self, timeline: TimelineId, mode: DistillMode) -> candle::Result<()> {
        self.write().distill_timeline(timeline, mode);
        let mut p = self.persistence.lock().unwrap();
        p.write_distill(timeline.raw(), mode)
            .map_err(|e| candle::Error::Msg(format!("write_distill: {e}")))?;
        Ok(())
    }

    /// Whether `timeline` is marked for distillation.
    pub fn is_timeline_distilled(&self, timeline: TimelineId) -> bool {
        self.read().is_distilled(timeline)
    }

    /// Whether `timeline` still has KV chunks on any of its turns — i.e. content
    /// not yet reclaimed. Callers gate distill-marking on this so a
    /// content-reclaimed timeline is never re-marked (which would keep triggering
    /// compaction forever).
    pub fn timeline_has_kv(&self, timeline: TimelineId) -> bool {
        use crate::persistence::content_hash::turn_stream_id;
        let sub = self.inner.read().unwrap();
        // Scope to this timeline's own turn streams — O(turns), not a scan of the
        // whole stream table.
        for idx in 0..sub.turn_count(timeline) {
            if let Some(e) = sub.stream_of(turn_stream_id(timeline.raw(), idx)) {
                if !e.chunks.is_empty() {
                    return true;
                }
            }
        }
        false
    }

    /// Set the substrate-side resume key (`debug_id`) for `timeline`
    /// and persist a `RecordType::DebugId` record to the redo log.
    /// Last-write-wins on replay.  Idempotent: if the substrate
    /// already holds the requested key, the record is not written and
    /// the call returns `Ok(())` without touching the log.
    pub fn set_conversation_debug_id(
        &self,
        timeline: TimelineId,
        debug_id: &str,
    ) -> candle::Result<()> {
        if debug_id.is_empty() {
            return Ok(());
        }
        self.write().set_debug_id(timeline, debug_id);
        let mut p = self.persistence.lock().unwrap();
        p.write_debug_id(timeline.raw(), debug_id)
            .map_err(|e| candle::Error::Msg(format!("write_debug_id: {e}")))?;
        Ok(())
    }

    /// Look up a timeline by `debug_id`.  O(1).
    pub fn lookup_by_debug_id(&self, debug_id: &str) -> Option<TimelineId> {
        self.read().lookup_by_debug_id(debug_id)
    }

    /// Number of turns currently waiting on the summariser thread to
    /// absorb them into the summary tree (§9 backpressure metric).
    /// `0` means steady state: the background tempo is keeping up
    /// with the foreground turn rate.
    pub fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.read().pending_summary_len(timeline)
    }

    /// Most recent score-density [`SelectionDiagnostics`] for
    /// `timeline`, or `None` if no projection has run yet (or the
    /// projection used the rule-based path).  Pure test-harness
    /// instrumentation: the substrate retains only the latest
    /// selection per timeline, written by the scheduler at projection
    /// time.  Production daemons can ignore.
    pub fn last_selection_diagnostics(&self, timeline: TimelineId) -> Option<SelectionDiagnostics> {
        self.read().last_selection_of(timeline).cloned()
    }

    /// Persist a per-`(timeline, turn_index)` summary-tree metadata
    /// record to the redo log.  Idempotent: skips the append when the
    /// in-memory manifest already records the same payload.  Called
    /// by the summariser thread after every atomic tree mutation
    /// (§7.2).
    pub fn write_tree_metadata(&self, payload: TreeMetadataPayload) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        p.write_tree_metadata(payload)
            .map_err(|e| candle::Error::Msg(format!("write_tree_metadata: {e}")))
    }

    /// Persist a sealed turn's per-layer KV grid + token ids to the redo log
    /// — the seal-time half of the resume path (§16.12). All layers share
    /// one chunk count.
    pub fn persist_turn_kv(
        &self,
        stream_id: StreamId,
        layers: &crate::persistence::resume::TurnChunkGrid,
        token_ids: &[u32],
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        crate::persistence::resume::persist_turn_kv(&mut p, stream_id, layers, token_ids)
            .map_err(|e| candle::Error::Msg(format!("persist turn kv: {e}")))
    }

    /// Persist only a turn's per-layer chunk records — the post-quantization
    /// half of the async seal/persist chain. Called from inside the
    /// bg-quantizer callback once float→quant migrations have landed.
    pub fn persist_turn_chunks(
        &self,
        stream_id: StreamId,
        layers: &crate::persistence::resume::TurnChunkGrid,
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        crate::persistence::resume::persist_turn_chunks(&mut p, stream_id, layers)
            .map_err(|e| candle::Error::Msg(format!("persist turn chunks: {e}")))
    }

    /// Persist a turn's chunks and return the per-layer [`StoredSequence`]
    /// references — the warm→cold leg of the persistence thread's
    /// `run_pass`. The returned references go straight into the
    /// substrate via `Substrate::install_cold`.
    pub fn persist_turn_chunks_capture(
        &self,
        stream_id: StreamId,
        layers: &crate::persistence::resume::TurnChunkGrid,
    ) -> candle::Result<Vec<StoredSequence>> {
        let (stored, locs) = {
            let mut p = self.persistence.lock().unwrap();
            crate::persistence::resume::persist_turn_chunks_capture(&mut p, stream_id, layers)
                .map_err(|e| candle::Error::Msg(format!("persist turn chunks capture: {e}")))?
        };
        // Fold the freshly-written chunk locations into the substrate's
        // authoritative chunk index so an in-process cold→hot elevation
        // (eviction then re-access without a daemon restart) can locate them
        // via `plan_chunked_read`. The redo-log walker only repopulates
        // `stream.chunks` on reload; without this the index stays empty until
        // then, and the cold-load pipeline plans zero records. The persistence
        // lock is released above so we never hold it across the substrate write.
        {
            let mut view = self.write();
            for (flat, loc) in locs {
                view.apply_chunk_loc(stream_id, flat, loc);
            }
        }
        Ok(stored)
    }

    /// Persist a turn's `Tokens` record and the trailing `Commit` — always
    /// called synchronously on seal, regardless of compression policy.
    /// `layers` is only used to compute the highest chunk index; pass an
    /// empty grid when no chunks were persisted (compression `None` path).
    pub fn persist_turn_tokens(
        &self,
        stream_id: StreamId,
        token_ids: &[u32],
        layers: &crate::persistence::resume::TurnChunkGrid,
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        crate::persistence::resume::persist_turn_tokens(&mut p, stream_id, token_ids, layers)
            .map_err(|e| candle::Error::Msg(format!("persist turn tokens: {e}")))
    }

    /// Persist a turn's `Tokens` record only — no trailing `Commit`.
    /// Used by the seal path now that chunks (and the matching Commit)
    /// are written asynchronously by the persistence thread.
    pub fn persist_tokens_only(
        &self,
        stream_id: StreamId,
        token_ids: &[u32],
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        crate::persistence::resume::persist_tokens_only(&mut p, stream_id, token_ids)
            .map_err(|e| candle::Error::Msg(format!("persist tokens: {e}")))
    }

    /// Append a stream-level `Commit` record at the given chunk index — the
    /// post-chunks re-commit used by the async seal/persist chain to upgrade
    /// the manifest's `committed_through` once the heavy `Chunks` records
    /// have been written.
    pub fn commit_stream_through(
        &self,
        stream_id: StreamId,
        through_index: u64,
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        p.commit_stream(stream_id, through_index)
            .map_err(|e| candle::Error::Msg(format!("commit stream: {e}")))
    }

    /// Persist the projection schema/template into the substrate's
    /// `Template` record — compare-and-insert (only appends when it differs
    /// from what the log already holds), then commit if written. Lets the
    /// log carry the projection needed to reconstruct the substrate.
    pub fn set_template(&self, template: &[u8]) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        let wrote = p
            .set_template(template)
            .map_err(|e| candle::Error::Msg(format!("persist template: {e}")))?;
        if wrote {
            p.commit()
                .map_err(|e| candle::Error::Msg(format!("commit template: {e}")))?;
        }
        Ok(())
    }

    /// Persist a turn's projection-event timeline to the redo log — the
    /// `ProjectionEvents` record. `payload` is the
    /// [`crate::projection::encode_events`] JSON. Also mirrors the bytes into
    /// the in-RAM substrate so the current session reads them back without a
    /// restart.
    pub fn persist_projection_events(
        &self,
        stream_id: StreamId,
        payload: &[u8],
    ) -> candle::Result<()> {
        self.write()
            .set_projection_events_blob(stream_id, payload.to_vec());
        let mut p = self.persistence.lock().unwrap();
        p.append_projection_events(stream_id, payload)
            .map_err(|e| candle::Error::Msg(format!("persist projection events: {e}")))
    }

    /// Persist a turn's encoded wide-Q signature window to the redo log (`WideQSig`
    /// record, last-writer-wins per stream) and mirror it into the in-RAM substrate.
    pub fn persist_wide_q_sigs(&self, stream_id: StreamId, payload: &[u8]) -> candle::Result<()> {
        self.write()
            .set_wide_q_sigs_blob(stream_id, payload.to_vec());
        let mut p = self.persistence.lock().unwrap();
        p.append_wide_q_sigs(stream_id, payload)
            .map_err(|e| candle::Error::Msg(format!("persist wide-Q sigs: {e}")))
    }

    /// Declare a section stream — appends a `StreamDecl::PromptSection`
    /// record carrying the content address and debug name.  The
    /// derived stream id matches `section_stream_id(address)`.  Called
    /// by the scheduler at section seal time; pairs with later
    /// `Tokens` / `Signatures` / `Chunks` records keyed by the same id.
    pub fn declare_section_stream(
        &self,
        address: ContentAddress,
        debug_name: &str,
    ) -> candle::Result<StreamId> {
        let decl = StreamDecl::PromptSection(SectionDecl {
            address,
            debug_name: debug_name.to_string(),
        });
        self.declare_and_mirror(decl, "declare section stream")
    }

    /// True when the workspace's manifest already holds durable
    /// chunks for `stream_id` — i.e. a section under this content
    /// address has been persisted and can be cold-loaded back into
    /// hot without re-prefilling.  The check matches the ingest
    /// loop's skip-if-present gate.
    pub fn section_stream_is_persisted(&self, stream_id: StreamId) -> bool {
        drop(self.persistence.lock().unwrap());
        self.read()
            .stream_of(stream_id)
            .map(|s| s.committed_through.is_some() && !s.chunks.is_empty())
            .unwrap_or(false)
    }

    /// Snapshot a persisted section stream's `chunks_per_layer` for the
    /// cold-load path — `manifest.chunks.len() / n_layers` — when the
    /// stream is known and its chunk count divides evenly, otherwise `None`.
    pub fn section_stream_layout(&self, stream_id: StreamId, n_layers: usize) -> Option<usize> {
        drop(self.persistence.lock().unwrap());
        let substrate = self.read();
        let entry = substrate.stream_of(stream_id)?;
        if entry.chunks.is_empty() || n_layers == 0 {
            return None;
        }
        let total = entry.chunks.len();
        if total % n_layers != 0 {
            return None;
        }
        Some(total / n_layers)
    }

    /// Cold-load a persisted section's chunks back into hot VRAM via
    /// the shared `load_stream_into_hot` pipeline.  Returns the
    /// per-layer `SealedSequence` the substrate's residence slab
    /// installs as the section's hot tier.
    pub fn cold_load_section_into_hot(
        &self,
        stream_id: StreamId,
        chunks_per_layer: usize,
        backings: &[candle_nn::kv_cache::ChunkedKvBacking],
        device: &candle::Device,
        stager: &mut crate::persistence::cold_load::ColdLoadStager,
    ) -> candle::Result<Vec<SealedSequence>> {
        use crate::persistence::transfer::load_stream_into_hot;
        let mut p = self.persistence.lock().unwrap();
        let substrate = self.read();
        load_stream_into_hot(
            backings,
            device,
            &mut p,
            &substrate,
            stream_id,
            chunks_per_layer,
            stager,
        )
        .map_err(|e| candle::Error::Msg(format!("cold_load_section_into_hot: {e}")))
    }

    /// Resolve a section stream's per-chunk redo-log locations into
    /// per-layer cold references — what the substrate stores under
    /// `residence.cold = Some(...)`.  Returns `None` when the stream
    /// is unknown or has no chunks recorded.
    pub fn recover_section_cold_refs(
        &self,
        stream_id: StreamId,
        n_layers: usize,
    ) -> candle::Result<Option<Vec<StoredSequence>>> {
        let substrate = self.read();
        crate::persistence::resume::recover_section_cold_refs(&substrate, stream_id, n_layers)
            .map_err(|e| candle::Error::Msg(format!("recover_section_cold_refs: {e}")))
    }

    /// Look up a section by the human-readable `debug_name` recorded
    /// on its `SectionDecl`.  Wrapper around
    /// [`Substrate::section_id_for_debug_name`].  Used by calibration
    /// consumers that pick scenarios out of a loaded workspace by id.
    pub fn section_id_for_debug_name(&self, debug_name: &str) -> Option<SectionId> {
        self.read().section_id_for_debug_name(debug_name)
    }

    /// Durably flush the persistence redo log — the group-commit point.
    /// `fsync`s every staged record so an in-flight turn survives a crash.
    pub fn commit_persistence(&self) -> candle::Result<()> {
        self.persistence
            .lock()
            .unwrap()
            .commit()
            .map_err(|e| candle::Error::Msg(format!("persist commit: {e}")))
    }

    /// Like [`Self::commit_persistence`] but a no-op when nothing is
    /// staged. Returns `Ok(true)` when an `fsync` actually happened.
    /// Used by the daemon's 5-second flush task so a quiescent
    /// workspace doesn't issue pointless syscalls.
    pub fn commit_persistence_if_pending(&self) -> candle::Result<bool> {
        self.persistence
            .lock()
            .unwrap()
            .commit_if_pending()
            .map_err(|e| candle::Error::Msg(format!("persist commit_if_pending: {e}")))
    }

    /// Force a full redo-log compaction — the whole-file dead-record rewrite
    /// (§5.8). Ignores the dead-ratio threshold: the operator opted in
    /// explicitly via the daemon's startup flag. `progress` reports coarse
    /// phase progress (0..=5) for the loading screen.
    pub fn compact_substrate(&self, progress: Option<&dyn Fn(usize, usize)>) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        p.commit()
            .map_err(|e| candle::Error::Msg(format!("persist commit: {e}")))?;
        let mut substrate = self.write();
        p.compact(&mut substrate, progress)
            .map_err(|e| candle::Error::Msg(format!("persist compaction: {e}")))
    }

    /// Run one **background maintenance** op on the segmented redo log — drop a
    /// fully-dead segment, compact a mostly-dead one, or combine two small
    /// adjacent ones (`docs/segmented_substrate_log.md` §6). At most one op per
    /// call; the persistence thread polls this every pass. The common no-op
    /// path is a cheap per-segment liveness scan. Holds the persistence lock
    /// and the substrate write lock for the op's duration — cold-loads and
    /// persists queue behind it.
    ///
    /// Returns `true` when an op actually ran.
    pub fn compact_persistence_if_needed(&self) -> candle::Result<bool> {
        self.run_maintenance_pass(false)
    }

    /// Force one maintenance op **now**, waiving the age/ratio gates — the manual
    /// `POST /v1/debug/maintenance` trigger. Seals the active segment first, so a
    /// conversation just archived/distilled this session (whose now-dead records
    /// sit in the active, which maintenance never touches) moves into a sealed
    /// segment and gets compacted away. Uses the same phased locking, so it never
    /// holds the substrate write lock across the relocation I/O — background
    /// decode keeps running. Returns whether an op ran.
    pub fn force_compact_persistence(&self) -> candle::Result<bool> {
        {
            let mut p = self.persistence.lock().unwrap();
            p.commit()
                .map_err(|e| candle::Error::Msg(format!("substrate maintenance commit: {e}")))?;
            p.seal_active()
                .map_err(|e| candle::Error::Msg(format!("substrate maintenance seal: {e}")))?;
        }
        self.run_maintenance_pass(true)
    }

    /// One maintenance pass under phased locking so the slow relocation I/O never
    /// holds the substrate write lock (which would stall every decode `read()`),
    /// mirroring the persistence thread's hot→warm / warm→cold discipline. When
    /// `force` is set the age/ratio gates are waived (`pick_maintenance_op`).
    fn run_maintenance_pass(&self, force: bool) -> candle::Result<bool> {
        // 1. Plan under a brief read + persistence lock (snapshot only).
        let plan = {
            let p = self.persistence.lock().unwrap();
            let substrate = self.read();
            p.plan_maintenance(&substrate, force)
                .map_err(|e| candle::Error::Msg(format!("substrate maintenance plan: {e}")))?
        };
        let outcome = if let Some(plan) = plan {
            // Signal "in progress" so the GUI shows a live spinner across the I/O.
            self.maintenance.lock().unwrap().2 = true;
            // Run the I/O in a closure so its result is captured rather than
            // early-returned: `running` MUST be cleared below on both success and
            // failure — an `?` straight out of here would leave the spinner (and
            // `/v1/status.running`) stuck true until the next successful pass.
            (|| -> candle::Result<bool> {
                // 2. Relocation I/O under the persistence lock ONLY — the
                //    substrate lock is released, so decode's in-RAM projection
                //    proceeds during the read + re-append + fsync.
                let result = {
                    let mut p = self.persistence.lock().unwrap();
                    p.execute_maintenance(&plan).map_err(|e| {
                        candle::Error::Msg(format!("substrate maintenance exec: {e}"))
                    })?
                };
                // 3. Repoint the index at the relocated records under a brief write lock.
                {
                    let mut substrate = self.write();
                    result.apply_to_substrate(&mut substrate);
                }
                // 4. Unlink the drained source segments under the persistence lock.
                {
                    let mut p = self.persistence.lock().unwrap();
                    p.finish_maintenance(&plan).map_err(|e| {
                        candle::Error::Msg(format!("substrate maintenance drop: {e}"))
                    })?;
                }
                Ok(true)
            })()
        } else {
            Ok(false)
        };
        // Always refresh the status cache — this also clears `running`, even if
        // the op above errored. The persistence lock here is held only for two
        // O(1) reads (never across I/O); the status endpoint reads the separate
        // `maintenance` lock, so it never blocks on a compaction.
        let view = {
            let p = self.persistence.lock().unwrap();
            (p.segment_count(), p.last_maintenance(), false)
        };
        *self.maintenance.lock().unwrap() = view;
        outcome
    }

    /// The segmented redo log's maintenance state for the daemon status / GUI
    /// compaction indicator: `(segment_count, last_op, running)` where `last_op`
    /// is `(label, unix_secs)` of the most recent drop/compact/combine this
    /// session (or `None`) and `running` is `true` while an op's I/O is in
    /// flight. Read from the **cache** — never takes the persistence lock — so
    /// `/v1/status` stays responsive while a compaction runs.
    pub fn maintenance_status(&self) -> (usize, Option<(String, u64)>, bool) {
        self.maintenance.lock().unwrap().clone()
    }

    /// Non-blocking snapshot of the redo log's dead-byte ratio — superseded
    /// last-writer-wins records plus tombstoned-stream bytes over total record
    /// bytes, the same measure the auto-compaction trigger polls. `try_lock`s
    /// the persistence layer so a read endpoint (the substrate viewer) never
    /// stalls behind a compaction holding it across I/O; returns `None` when the
    /// lock is momentarily contended rather than blocking.
    pub fn dead_ratio(&self) -> Option<f32> {
        // Lock order MUST be `inner` then `persistence` — the same order the
        // write path takes (see `with_persistence_scan` below) — or a viewer
        // poll holding `persistence` while waiting on `inner` would deadlock a
        // seal holding `inner` while waiting on `persistence`. `try_lock` on
        // persistence keeps this non-blocking against an in-flight compaction.
        let substrate = self.inner.read().unwrap();
        let persistence = self.persistence.try_lock().ok()?;
        Some(persistence.dead_ratio(&substrate))
    }

    /// Run `f` against the persistence layer's current manifest snapshot.
    /// Read-only accessor for callers that need to inspect the redo
    /// log's stream/chunk locations (sizes, formats, offsets) without
    /// rebuilding the whole substrate boundary.
    pub fn with_persistence_manifest<R>(
        &self,
        f: impl FnOnce(&crate::persistence::manifest::Manifest) -> R,
    ) -> R {
        let p = self.persistence.lock().unwrap();
        f(p.manifest())
    }

    /// Run `f` with a read view of the in-memory substrate and exclusive access to
    /// the live persistence layer — for offline-style scans (e.g. the noise
    /// calibration) that read sealed stream chunks/tokens back from the log.
    /// Locks `inner` (read) then `persistence`, matching the write path's order.
    /// Callers should `checkpoint_persistence` first so the records being read are
    /// durable on disk at their recorded offsets.
    pub fn with_substrate_and_persistence<R>(
        &self,
        f: impl FnOnce(&Substrate, &mut SubstratePersistence) -> R,
    ) -> R {
        let substrate = self.inner.read().unwrap();
        let mut persistence = self.persistence.lock().unwrap();
        f(&substrate, &mut persistence)
    }
}

// ── TargetedRead ──────────────────────────────────────────────────────────────

/// Target-aware [`ContentResolver`] wrapper around a [`SubstrateRead`].
///
/// For `target.group`: only `target.timeline` is visible; sibling timelines
/// are masked.  For other groups: the first-registered timeline is used
/// (Phase 3 simplification for groups with a single shared timeline).
/// Sections are workspace singletons and pass straight through.
pub struct TargetedRead<'a> {
    read: SubstrateRead<'a>,
    target: ProjectionTarget,
}

impl<'a> TargetedRead<'a> {
    pub fn new(read: SubstrateRead<'a>, target: ProjectionTarget) -> Self {
        Self { read, target }
    }

    fn timeline_for(&self, group: GroupId) -> Option<TimelineId> {
        self.read.resolve_turn_timeline(Some(self.target), group)
    }
}

impl<'a> std::ops::Deref for TargetedRead<'a> {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        &self.read
    }
}

impl<'a> ContentResolver for TargetedRead<'a> {
    fn turn_count(&self, group: GroupId) -> u32 {
        let Some(timeline) = self.timeline_for(group) else {
            return 0;
        };
        Substrate::turn_count(&self.read, timeline)
    }

    fn turn_token_count(&self, group: GroupId, index: TurnIndex) -> usize {
        let Some(timeline) = self.timeline_for(group) else {
            return 0;
        };
        self.read.turn_token_count_of(timeline, index)
    }

    fn turn_score(&self, group: GroupId, index: TurnIndex) -> f32 {
        let Some(timeline) = self.timeline_for(group) else {
            return 0.0;
        };
        self.read.turn_score_for_timeline(timeline, index)
    }

    fn turn_origin(&self, group: GroupId, _index: TurnIndex) -> Option<LayerId> {
        let timeline = self.timeline_for(group)?;
        let (layer, _) = self.read.timeline_target(timeline)?;
        Some(layer)
    }

    fn turn_with_tag(&self, group: GroupId, tag: &str) -> Option<TurnIndex> {
        let timeline = self.timeline_for(group)?;
        // Call the Substrate inherent method (timeline-keyed) via deref — not the
        // trait method (group-keyed) on `SubstrateRead`.
        Substrate::turn_with_tag(&self.read, timeline, tag)
    }

    fn turn_timeline(&self, group: GroupId, _index: TurnIndex) -> Option<TimelineId> {
        self.timeline_for(group)
    }

    fn turn_kind(&self, group: GroupId, index: TurnIndex) -> TurnKind {
        let Some(timeline) = self.timeline_for(group) else {
            return TurnKind::Normal;
        };
        self.read
            .tree_meta_of(timeline, index)
            .map(|m| m.kind)
            .unwrap_or(TurnKind::Normal)
    }

    fn node_covers(&self, group: GroupId, index: TurnIndex) -> Vec<TurnIndex> {
        let Some(timeline) = self.timeline_for(group) else {
            return Vec::new();
        };
        // Walk the immutable forest downward from `index`, collecting every
        // transitive child. `tree_meta_of` gives a node's direct children;
        // Normal leaves have none, so a raw turn yields an empty cover set.
        let mut out = Vec::new();
        let mut stack = vec![index];
        while let Some(node) = stack.pop() {
            if let Some(meta) = self.read.tree_meta_of(timeline, node) {
                for &child in &meta.children {
                    out.push(child);
                    stack.push(child);
                }
            }
        }
        out
    }

    fn turn_no_think(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        self.read.turn_no_think(timeline, index)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        ContentResolver::section_token_count(&self.read, section)
    }

    fn section_score(&self, section: SectionId) -> f32 {
        ContentResolver::section_score(&self.read, section)
    }

    fn summary_tree_select(
        &self,
        timeline: TimelineId,
        budget: u32,
    ) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
        // Delegate to the inner scored read: score-density selection over the
        // timeline's summary forest (see `select_dense` in `select.rs`).
        self.read.summary_tree_select(timeline, budget)
    }
}

#[cfg(test)]
mod tests {
    use super::{selected_in_collection, subwindow_bounds};
    use crate::projection::{ProjectionSelection, SelectedSection, SystemItem};

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

    #[test]
    fn selected_in_collection_finds_the_selected_member() {
        let sel = ProjectionSelection {
            system: vec![SystemItem::Collection {
                name: "tools".into(),
                sections: vec![
                    SelectedSection {
                        name: "a".into(),
                        tokens: 1,
                        selected: false,
                        score: 2.0,
                    },
                    SelectedSection {
                        name: "b".into(),
                        tokens: 1,
                        selected: true,
                        score: 9.0,
                    },
                ],
            }],
            turns: vec![],
        };
        assert_eq!(selected_in_collection(&sel, "tools"), Some("b".to_string()));
        // A different collection name matches nothing.
        assert_eq!(selected_in_collection(&sel, "memory"), None);
    }
}
