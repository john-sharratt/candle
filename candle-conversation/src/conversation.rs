//! Client-side conversation handle.
//!
//! Held on the caller's thread. Manages turn history and submits GPU work
//! to the scheduler via channel. All CPU work (tokenization, formatting)
//! happens here; the scheduler only does GPU work.

use crate::config::{SamplingConfig, SequenceConfig};
use crate::error::ConversationError;
use crate::handle::{SealResult, TurnHandle, TurnResponse};
use crate::persistence::content_hash::{hash_tokens, ContentChain};
use crate::persistence::streams::ContentAddress;
use crate::projection::{
    Builder, Conversation, ProjectionTarget, SectionId, SystemPromptItem, TimelineId, TurnIndex,
};
use crate::provenance::ProvenanceFile;
use crate::scheduler::{ProjectionInputs, ReprojectionPolicy, SchedulerRequest};
use crate::sequence_handle::{BlockCount, SequenceId};
use crate::token_buffer::TokenBuffer;
use crate::tree::token_text::TokenizedText;
use crate::tree::{CognitiveTask, ConversationTree, TaskPoll, TurnType};
use crate::turn::{Role, Turn, TurnOptions};
use candle_nn::kv_cache::SealedSequence;
use candle_transformers::models::batched_inference::ModelCoreProperties;

/// Slice a per-layer sealing down to the chunk range `[from..to)`.
///
/// Each layer's `SealedSequence` is replaced with one whose `chunks`
/// covers just the requested block window, with the token count
/// recomputed from those chunks.  Used by the substrate seal path
/// (and by parallel section ingestion) to extract a unit's *delta*
/// bytes from a whole-sequence snapshot.
pub(crate) fn slice_per_layer_sealed(
    full: &[SealedSequence],
    from: usize,
    to: usize,
) -> Vec<SealedSequence> {
    full.iter()
        .map(|seq| {
            let chunks: Vec<_> = seq
                .chunks
                .get(from..to.min(seq.chunks.len()))
                .unwrap_or(&[])
                .to_vec();
            let token_count = chunks.iter().map(|c| c.token_count as usize).sum();
            SealedSequence {
                chunks,
                token_count,
                chunk_size: seq.chunk_size,
                location: seq.location,
            }
        })
        .collect()
}

use crossbeam::channel::Sender;
use std::sync::Arc;

/// A client-side conversation handle.
///
/// Manages turn history and submits GPU work to the scheduler via channel.
/// One turn at a time per conversation (`turn_in_flight` guard).
///
/// # Usage
///
/// ```ignore
/// // Blocking:
/// let response = conv.send("hello")?;
///
/// // Streaming:
/// let handle = conv.submit_turn("hello")?;
/// for event in handle.stream() { /* ... */ }
/// conv.finish_turn(&response);
/// ```
pub struct Sequence {
    /// Channel to submit GPU work to the scheduler.
    scheduler_tx: Sender<SchedulerRequest>,

    /// The sequence slot allocated by the scheduler.  This is the
    /// sequence's identifier — there is no separate logical id,
    /// since persistence is workspace-scoped (substrate-keyed) and
    /// every other use of "which sequence" is in-memory addressing
    /// that this `SequenceId` covers.
    id: SequenceId,

    /// Shared tokenizer.
    tokenizer: Arc<tokenizers::Tokenizer>,

    /// Sequence tree — the canonical turn history for this conversation.
    /// Holds system prompt (with token ids), paired user↔assistant exchanges
    /// (each with text + token ids), temporal markers, and phase 2+ segment nodes.
    tree: ConversationTree,

    /// Pending user turn set by `submit_turn_with_options`, consumed by
    /// `finish_turn`. Holds raw user text + token ids for commit to the tree.
    pending_user: Option<TokenizedText>,

    /// Monotonic counter tracking the total number of individual entries
    /// (system, user, assistant) appended to cold store.  Also used by
    /// `turn_count()` for informational display.

    /// Monotonic turn counter.
    turn_counter: u64,

    /// Per-conversation config.
    config: SequenceConfig,

    /// True while a submitted turn is being processed by the scheduler.
    turn_in_flight: bool,

    /// True after `close()` has been called (prevents double-free in Drop).
    freed: bool,

    /// High-water mark: cumulative sealed blocks at the end of the
    /// most recent seal.  Per-turn block ranges live on the substrate
    /// (one entry per appended turn); this field is just the running
    /// tail pointer used for fork-origin and snapshot bookkeeping.
    current_blocks: BlockCount,

    /// KV chunk size in tokens (mirrors `BatchedConfig::chunk_size`).
    chunk_size: usize,

    /// Shared mmap-backed provenance file.
    ///
    /// The seal step writes chunk-group triplets (syntactic / semantic /
    /// pragmatic) here inline and stashes the resulting `SigEntry`
    /// values in the workspace substrate per `(group, turn)` key.
    provenance: Arc<ProvenanceFile>,

    /// Static model properties captured at engine construction.
    model_core: ModelCoreProperties,

    /// Number of 32-token KV blocks already indexed in `ProvenanceFile`.
    /// Passed to the seal step so extraction starts from the first
    /// unprocessed block rather than re-indexing the entire sequence.
    sig_blocks_processed: usize,

    /// Projection schema for this conversation.
    ///
    /// Held as an `Arc` so the scheduler can hold a reference for
    /// continuous re-projection without repeatedly deep-cloning the
    /// underlying `Schema`.
    pub(crate) projection: Arc<Builder>,
    /// Workspace-shared substrate handle (Arc-cloneable).  Holds per-turn
    /// metadata (token counts, scores, sig entries, restoration sources)
    /// for every conversation in the engine.  Read-locked during projection,
    /// write-locked at seal time.
    pub(crate) substrate: Conversation,
    /// `(layer, group)` this conversation is for.  Determines which group
    /// `seal_and_register_turn` appends into and which target is passed to
    /// `projection.project()`.  Different cognitive activities (dialogue,
    /// bug analysis, dream log, …) point at different targets.
    pub(crate) target: ProjectionTarget,
    /// Persistent BDP scanner — its score map lives for the lifetime of
    /// the conversation, refreshed each time `run_bdp_scan` is called.
    pub(crate) bdp_scanner: crate::provenance::BdpScanner,
}

impl Sequence {
    /// Create a new conversation backed by a full projection [`Builder`].
    ///
    /// `system_prompt` must already be formatted for the model's chat template
    /// (e.g. ChatML-wrapped).  `target` names the `(layer, group)` this
    /// conversation is for — turns are appended into `target.group`, and
    /// `target` is what gets passed to `projection.project()`.
    ///
    /// Eagerly ingests the conversation's plain `system_prompt` and the
    /// schema's declared sections / collections into the workspace
    /// substrate before returning, so the first `submit_turn` finds
    /// every static system-side section already pinned.
    pub(crate) fn new_with_projection(
        scheduler_tx: Sender<SchedulerRequest>,
        sequence_id: SequenceId,
        tokenizer: Arc<tokenizers::Tokenizer>,
        system_prompt: &str,
        projection: Builder,
        target: ProjectionTarget,
        config: SequenceConfig,
        chunk_size: usize,
        provenance: Arc<ProvenanceFile>,
        model_core: ModelCoreProperties,
        substrate: Conversation,
        section_progress: Option<&dyn Fn(u64, u64)>,
    ) -> crate::Result<Self> {
        // Persistence is now a property of the workspace `Conversation`
        // (the substrate handle), wired in by the engine via
        // `Conversation::open(path)`.  The Sequence has nothing to do
        // with on-disk records anymore; turn_counter is informational
        // only — exposed via `Sequence::turn_count` for tests.  Count
        // the system prompt as turn 0 when present, matching the
        // pre-substrate behaviour (system + user + assistant = 3).
        let prior_turns = substrate.read().turn_count(target.timeline) as u64;
        let initial_turn_counter: u64 = if system_prompt.is_empty() {
            prior_turns
        } else {
            prior_turns + 1
        };

        let tree_config = config.tree.clone();
        let conv = Self {
            scheduler_tx,
            id: sequence_id,
            tokenizer,
            tree: ConversationTree::with_config(system_prompt, tree_config),
            pending_user: None,
            turn_counter: initial_turn_counter,
            config,
            turn_in_flight: false,
            freed: false,
            current_blocks: BlockCount(0),
            chunk_size,
            provenance,
            model_core,
            sig_blocks_processed: 0,
            projection: Arc::new(projection),
            substrate,
            target,
            bdp_scanner: crate::provenance::BdpScanner::new(),
        };

        // Set the in-memory tree's system prompt tokens so the tree's
        // own per-turn formatting can reference them.  Persistence
        // for the system prompt has moved to the substrate side via
        // `insert_section` below — no need for a separate cold-store
        // write.
        let mut conv = conv;
        if !system_prompt.is_empty() {
            let text = conv.tree.system_prompt_text().to_string();
            let token_ids = conv.tokenize(&text).unwrap_or_default();
            conv.tree.set_system_prompt_tokens(token_ids);
            let _ = text;
        }

        // Eagerly seed every static system-side section into the
        // workspace substrate.  The slot itself stays empty — the
        // first `submit_turn` runs `apply_projection` which
        // materialises the relevant sections onto it from the
        // substrate's CPU-pinned bytes via the upload cache.
        //
        // The schema's declared sections + collections for this
        // target's layer.  Each `Section` becomes one substrate
        // section under its declared id; each `Collection` runs
        // all its sections in parallel via
        // `insert_section_collection`.  The whole system prompt is
        // composed from these schema items — no separate monolithic
        // "system_section_id" pre-pinning, which used to double the
        // system content with an unwrapped fragment copy.
        let layer_items: Vec<SystemPromptItem> = conv
            .projection
            .schema()
            .layers
            .iter()
            .find(|l| l.id == target.layer)
            .map(|l| l.system_prompt.items.clone())
            .unwrap_or_default();

        // Cumulative-prefix ingest builds each content section's K/V
        // conditioned on the chain of previously-ingested content
        // sections.  Templates do not appear in this chain — their K/V
        // is live-prefilled at projection-apply time under the actual
        // runtime context, so substrate carries no entries for them
        // and they contribute nothing to the prefix here.
        let mut linear_prefix: Vec<SectionId> = Vec::new();
        // Mirrors `linear_prefix` but excludes Collection members and
        // `depends_on`-gated sections — feeds the priming projection
        // so the first-turn slot mirrors the empty-collection layout.
        let mut fixed_prefix: Vec<SectionId> = Vec::new();

        // ── Progress accounting ────────────────────────────────────────
        // Byte totals across every section we're about to ingest. The
        // frontend's "Sections" step progress fills as we walk the
        // ingest loop below, so the load screen actually moves between
        // 0 % and 100 % instead of sitting at 0 % until the final
        // `set_step` transition.
        // Templates are filtered from the ingest loop and contribute
        // nothing to the byte-count progress — they don't enter the
        // substrate, so there's no work to count for them.
        let total_bytes: u64 = layer_items
            .iter()
            .map(|item| match item {
                SystemPromptItem::Section(s) => {
                    if s.is_template {
                        0
                    } else {
                        s.content.len() as u64
                    }
                }
                SystemPromptItem::Collection(c) => c
                    .sections
                    .iter()
                    .filter(|s| !s.is_template)
                    .map(|s| s.content.len() as u64)
                    .sum::<u64>(),
            })
            .sum::<u64>();
        let mut done_bytes: u64 = 0;
        let report = |done: u64| {
            if let Some(cb) = section_progress {
                cb(done, total_bytes);
            }
        };
        // Initial 0 / total tick so the bar appears immediately —
        // otherwise the UI shows nothing until the first section
        // finishes, which can be seconds for a heavy schema.
        report(0);

        // Cumulative ingest: walk schema items in declaration order,
        // ingesting each non-template `Section`-kind item with all
        // previously-ingested content sections Arc-injected onto the
        // scratch slot as its prefix.  The prefill forward pass for
        // section N therefore attends to content sections 0..N-1 just
        // like it would at projection time, producing K_raw values
        // conditioned on the real preceding content context.
        //
        // Template-kind sections (`is_template = true`) are skipped:
        // their K/V is live-prefilled at projection-apply time under
        // the actual runtime left context, so the substrate never sees
        // them and they contribute nothing to other sections' ingest
        // prefix here.
        //
        // `Collection` items: every member ingests in parallel with
        // the *same* pre-collection prefix (members do not attend to
        // each other). Sections declared after a Collection attend to
        // the full set of members during ingest — the maximal-context
        // approximation for runtime selection.
        for item in &layer_items {
            match item {
                SystemPromptItem::Section(s) => {
                    if s.is_template {
                        continue;
                    }
                    conv.insert_section_with_prefix(s.id, s.content.as_str(), &linear_prefix)?;
                    linear_prefix.push(s.id);
                    // Sections with `depends_on` are conditional — they
                    // only emit when the named collection materialises —
                    // so they must NOT contribute to the priming
                    // projection's `fixed_prefix`.
                    if s.depends_on.is_none() {
                        fixed_prefix.push(s.id);
                    }
                    done_bytes += s.content.len() as u64;
                    report(done_bytes);
                }
                SystemPromptItem::Collection(coll) => {
                    let batch: Vec<(SectionId, &str)> = coll
                        .sections
                        .iter()
                        .filter(|sec| !sec.is_template)
                        .map(|sec| (sec.id, sec.content.as_str()))
                        .collect();
                    if !batch.is_empty() {
                        // Per-section progress: tick `done_bytes`
                        // and call the outer report() as each member
                        // completes its ingest / restore / skip.
                        // Collection members complete in submission
                        // order at the wait-phase boundary inside
                        // `insert_section_collection_with_progress`,
                        // so the bar advances ~once per ~prefill of
                        // a single tool description rather than
                        // jumping at the end of the whole collection.
                        let done_bytes_ref = &mut done_bytes;
                        let report_ref = &report;
                        conv.insert_section_collection_with_progress(
                            &batch,
                            &linear_prefix,
                            true, // schema Collection: members get the
                            // aggressive turn-level quantize policy
                            |_sid, content_len| {
                                *done_bytes_ref += content_len as u64;
                                report_ref(*done_bytes_ref);
                            },
                        )?;
                    }
                    // Extend the live linear_prefix with every
                    // collection member so subsequent sections see
                    // them at projection time, but do NOT add them
                    // to fixed_prefix — the priming projection must
                    // be coherent for the empty-collection case.
                    for sec in &coll.sections {
                        if sec.is_template {
                            continue;
                        }
                        linear_prefix.push(sec.id);
                    }
                }
            }
        }

        // Pre-warm the slot: inject all system-prompt sections now so the
        // first `submit_turn` sees an already-populated slot and skips
        // `apply_projection` entirely, removing that injection from the
        // first-turn critical path.
        //
        // Priming uses `fixed_prefix` — the collections-as-empty
        // projection — so the slot's starting state mirrors what
        // `emit_system_prompt_items` would produce when no collection
        // members are selected and no `depends_on`-gated sections fire.
        // Any per-turn projection that selects ≥ 1 collection member
        // will materialise those (and the gated sections) via
        // `apply_projection` at submit_turn time.
        if !fixed_prefix.is_empty() {
            let (tx, rx) = crossbeam::channel::bounded(1);
            conv.scheduler_tx
                .send(crate::scheduler::SchedulerRequest::PrimingProjection {
                    sequence_id: conv.id,
                    section_ids: fixed_prefix,
                    response_tx: tx,
                })
                .map_err(|_| ConversationError::SchedulerGone)?;
            rx.recv().map_err(|_| ConversationError::SchedulerGone)??;
        }

        Ok(conv)
    }

    /// Install an observer channel for cognitive task events (summarization,
    /// daydream, etc.). When set, streaming events — `Token`, `Prefill`,
    /// `PrefillProgress`, `HealthWarning` — are forwarded to `tx` in real
    /// time as the task generates tokens. This allows callers to monitor
    /// background inference (e.g. print tokens as they stream).
    ///
    /// Pass `None` to remove the observer.
    pub fn set_task_observer(
        &mut self,
        tx: Option<crossbeam::channel::Sender<crate::handle::TurnEvent>>,
    ) {
        self.tree.task_event_observer = tx;
    }

    /// Ingest one system-prompt section into the substrate.
    ///
    /// Forks the conversation, prefills the section's content onto
    /// the fork (no decode), seals the fork to CPU, and pins the
    /// resulting `Arc<Vec<SealedSequence>>` in the substrate via
    /// [`Substrate::set_section_sealed`].  The section's BDP
    /// sig entries and absolute block range are recorded by
    /// [`Substrate::set_section_data`].  The fork is freed.
    ///
    /// The conversation's main sequence is unchanged: section bytes
    /// live in CPU arenas, and the scheduler's `apply_projection`
    /// step re-injects them onto the parent at every turn.
    ///
    /// Used by [`Self::preemptive_prefill`] for sequentially-ingested
    /// sections (those before the first
    /// [`SectionCollection`](SectionCollection)
    /// in a layer's items).  Parallel collection ingestion goes
    /// through [`Self::insert_section_collection`].
    ///
    /// **Architectural note (planned migration):** under the
    /// substrate-as-parent model this operation conceptually lives
    /// on the workspace [`Conversation`], not on
    /// a `Sequence` — section ingestion is a substrate operation,
    /// not a sequence operation.  The current `&mut self` signature
    /// is retained because the section-fork pipeline pulls
    /// scheduler_tx / tokenizer / provenance / chunk_size off the
    /// Sequence; lifting it to a workspace method requires threading
    /// those dependencies through and restructuring callers.
    pub fn insert_section(&mut self, section: SectionId, content: &str) -> crate::Result<()> {
        if content.is_empty() {
            return Ok(());
        }
        let mut results = self.insert_section_collection(&[(section, content)], &[], false)?;
        results.pop();
        Ok(())
    }

    /// Cumulative-prefix variant of [`Self::insert_section`].  Every
    /// section in `prefix_section_ids` is Arc-injected onto the
    /// scratch slot before this section's prefill, so the model's
    /// forward pass attends to the real preceding context.  Used by
    /// the schema-driven system_prompt loop to walk
    /// declaration-order sections cumulatively.
    pub fn insert_section_with_prefix(
        &mut self,
        section: SectionId,
        content: &str,
        prefix_section_ids: &[SectionId],
    ) -> crate::Result<()> {
        if content.is_empty() {
            return Ok(());
        }
        let mut results =
            self.insert_section_collection(&[(section, content)], prefix_section_ids, false)?;
        results.pop();
        Ok(())
    }

    /// Ingest a collection of sections into the substrate in parallel.
    ///
    /// Forks the conversation once per section, submits each fork's
    /// section content as a no-decode prefill (the scheduler batches
    /// the concurrent prefills into shared forward passes), waits for
    /// every fork to complete, then for each fork seals to CPU and
    /// pins the section in the substrate.  Forks are freed.
    ///
    /// Replaces the older fork-and-merge path: section bytes are
    /// never merged back into the conversation's main sequence —
    /// they live in CPU arenas pinned by the substrate, and
    /// `apply_projection` re-injects the selected ones per turn.
    /// Returns the list of `(SectionId, block_count)` for caller
    /// diagnostics; the empty `Vec` is also a valid return for an
    /// empty input.
    pub fn insert_section_collection(
        &mut self,
        sections: &[(SectionId, &str)],
        prefix_section_ids: &[SectionId],
        in_collection: bool,
    ) -> crate::Result<Vec<(SectionId, usize)>> {
        self.insert_section_collection_with_progress(
            sections,
            prefix_section_ids,
            in_collection,
            |_, _| {},
        )
    }

    /// Same as [`Self::insert_section_collection`] but fires
    /// `on_section_done(section_id, content_len_bytes)` after every
    /// single section completes — skipped, restored from disk, or
    /// freshly ingested — so the surrounding ingest loop can update
    /// its progress bar at section granularity instead of waiting for
    /// the whole batch.
    ///
    /// `in_collection = true` when the caller is materialising a
    /// schema `Collection` (tools in a tool catalog, hits in a
    /// retrieval list, …): every section in `sections` becomes a
    /// collection member, and the scheduler picks the more
    /// aggressive turn-level compression policy for them.  `false`
    /// is for boundary sections — `insert_section` and
    /// `insert_section_with_prefix` always pass `false` because the
    /// model needs near-lossless K/V on the role markers / opening
    /// and closing tags every later token attends back over.
    pub fn insert_section_collection_with_progress<F>(
        &mut self,
        sections: &[(SectionId, &str)],
        prefix_section_ids: &[SectionId],
        in_collection: bool,
        mut on_section_done: F,
    ) -> crate::Result<Vec<(SectionId, usize)>>
    where
        F: FnMut(SectionId, usize),
    {
        if sections.is_empty() {
            return Ok(Vec::new());
        }
        // Tokenise every section up-front, dropping empty entries so
        // we never spawn a pointless slot.  Carrying the token count
        // through means the substrate's `set_section_data` records
        // a real value.
        struct Pending<'a> {
            section_id: SectionId,
            content: &'a str,
            tokens: TokenBuffer,
            token_count: usize,
            address: ContentAddress,
            debug_name: String,
        }
        // Rebuild a `ContentChain` snapshot from the supplied prefix
        // section ids.  Sections in this call all share the same
        // prefix snapshot (collection-member semantics: members don't
        // attend to each other, so each member's `prefix_hash` is the
        // chain state *before* the collection).  The chain reads each
        // prefix section's tokens from the substrate — they were
        // tokenised and pinned on a previous `insert_section_*` call.
        //
        // **Collection members are excluded from the chain.**  Without
        // this, every change to a collection member (installing a new
        // tool, removing one, editing a tool's description) would
        // produce a fresh `prefix_hash` for every section *after* the
        // collection, cascading stream-id invalidation through the
        // whole downstream tail and forcing a re-prefill of sections
        // whose own content didn't change.  Collection members are
        // already an approximation in the post-collection prefix
        // (projection selects a subset at runtime, so the cached K/V
        // is never a strict function of the specific members that
        // ingested) — treating them as outside the content chain
        // matches that approximation and keeps minor catalog changes
        // local.
        let prefix_hash = {
            let mut chain = ContentChain::new();
            let view = self.substrate.read();
            let schema = self.projection.schema();
            for &pid in prefix_section_ids {
                // Skip collection members from any layer's prompt —
                // they don't advance the content chain.
                if schema
                    .layers
                    .iter()
                    .any(|l| l.system_prompt.is_collection_member(pid))
                {
                    continue;
                }
                let pre_tokens = view.section_tokens_of(pid);
                if pre_tokens.is_empty() {
                    // A prefix section with no recorded tokens — this
                    // happens for template-kind items that don't enter
                    // the substrate.  Skip; they contribute nothing to
                    // the cumulative content prefix anyway.
                    continue;
                }
                chain.push_section(&pre_tokens);
            }
            chain.prefix()
        };
        // Walk every requested section and triage into three buckets:
        //   - Already hot in the substrate → skip entirely (a prior
        //     `insert_section_*` call in this same daemon run already
        //     pinned it).  Block-count contribution is its
        //     `section_sealed_of` chunk count.
        //   - Persisted in the redo log under its content-addressed
        //     stream id → restore from disk (`RestoreSection`).
        //   - Otherwise → ingest with a fresh prefill (`IngestSection`).
        let n_layers = self.model_core.num_layers;
        let mut to_ingest: Vec<Pending<'_>> = Vec::with_capacity(sections.len());
        let mut to_restore: Vec<Pending<'_>> = Vec::with_capacity(sections.len());
        let mut out_skip: Vec<(SectionId, usize)> = Vec::new();
        for &(section_id, content) in sections {
            if content.is_empty() {
                continue;
            }
            let tokens = self.tokenize(content)?;
            if tokens.is_empty() {
                continue;
            }
            let token_count = tokens.len();
            let address = ContentAddress {
                prefix_hash,
                section_hash: hash_tokens(&tokens),
            };
            let debug_name = self.section_debug_name(section_id);
            // Already-present check first — cheapest, no lock
            // contention on the persistence mutex.  `section_exists`
            // is broader than `section_is_hot`: it returns true for
            // cold-marker sections that were restored on substrate
            // reload but haven't been elevated yet.  Either way the
            // substrate already knows about this section; the elevate
            // path will lift it on the next projection that needs it.
            if self.substrate.read().section_exists(section_id) {
                let block_count = self
                    .substrate
                    .read()
                    .section_block_count(section_id)
                    .unwrap_or(0);
                out_skip.push((section_id, block_count));
                on_section_done(section_id, content.len());
                continue;
            }
            // Manifest check.  Only meaningful when the model's
            // layer count is known; without backings (test harnesses
            // that don't register a session) we can't compute
            // `chunks_per_layer`, so we fall through to ingest.
            let stream_id = crate::persistence::content_hash::section_stream_id(address);
            if n_layers > 0 && self.substrate.section_stream_is_persisted(stream_id) {
                if let Some((_, _, _)) = self.substrate.section_stream_layout(stream_id, n_layers) {
                    to_restore.push(Pending {
                        section_id,
                        content,
                        tokens,
                        token_count,
                        address,
                        debug_name,
                    });
                    continue;
                }
            }
            to_ingest.push(Pending {
                section_id,
                content,
                tokens,
                token_count,
                address,
                debug_name,
            });
        }
        let total = to_ingest.len() + to_restore.len();
        if total == 0 {
            return Ok(out_skip);
        }
        let n_sections = total;
        let t0 = std::time::Instant::now();

        // Dispatch restores first — they short-circuit the
        // prefill batcher entirely and the scheduler handles them
        // synchronously per request.  Wait for each to complete before
        // moving to ingest so the restored sections are visible in the
        // substrate if the upcoming ingest happens to need them in a
        // prefix (it won't in the cumulative-ingest loop, but it's
        // cheap insurance).
        let mut restore_out: Vec<(SectionId, usize)> = Vec::with_capacity(to_restore.len());
        for item in to_restore.into_iter() {
            let stream_id = crate::persistence::content_hash::section_stream_id(item.address);
            let chunks_per_layer = self
                .substrate
                .section_stream_layout(stream_id, n_layers)
                .map(|(c, _, _)| c)
                .unwrap_or(0);
            // Capture the content length for the progress callback
            // before `item.tokens` / `item.content` get consumed by
            // the request payload below.
            let item_section_id = item.section_id;
            let item_content_len = item.content.len();
            let (tx, rx) = crossbeam::channel::bounded(1);
            self.scheduler_tx
                .send(SchedulerRequest::RestoreSection {
                    conversation: self.substrate.clone(),
                    section_id: item.section_id,
                    stream_id,
                    address: item.address,
                    chunks_per_layer,
                    tokens: item.tokens.clone(),
                    response_tx: tx,
                })
                .map_err(|_| ConversationError::SchedulerGone)?;
            match rx.recv().map_err(|_| ConversationError::SchedulerGone)? {
                Ok(()) => {
                    // Block count for diagnostics — the section is
                    // now a cold-marker; the actual hot grid lands
                    // when the next projection elevates it.  Use the
                    // manifest's chunks_per_layer as the count.
                    restore_out.push((item.section_id, chunks_per_layer));
                    on_section_done(item_section_id, item_content_len);
                }
                Err(e) => {
                    tracing::warn!(
                        section_id = ?item.section_id,
                        "section restore failed ({e}) — falling back to ingest",
                    );
                    to_ingest.push(item);
                }
            }
            let _ = item.token_count;
        }

        // Bulk-allocate-then-fire: allocate one scratch slot per
        // section first (cheap, no timeline minting), then fire every
        // IngestSection request in a tight burst so the scheduler can
        // process them back-to-back without round-tripping fork
        // responses between each one.  All scratch slots stay open
        // until the wait phase finishes — section seals don't share
        // state across slots, so concurrency is safe.
        //
        // Why scratch slots (not `fork()`): section ingestion under
        // the substrate-as-parent model produces context-independent
        // KV that gets re-injected onto any parent layout at
        // projection time.  A scratch slot — fresh empty, no
        // projection target, no `TimelineId` minted — is exactly the
        // right shape.  `fork()` would also mint a timeline (and write
        // a manifest record per section under any future persistence
        // layer), which is wasted effort here.
        let mut slot_ids: Vec<SequenceId> = Vec::with_capacity(to_ingest.len());
        for _ in 0..to_ingest.len() {
            slot_ids.push(self.alloc_scratch_slot()?);
        }
        let t_alloc = std::time::Instant::now();

        let mut pending: Vec<(
            SequenceId,
            SectionId,
            usize, // content_len for the progress callback
            crossbeam::channel::Receiver<crate::Result<SealResult>>,
        )> = Vec::with_capacity(to_ingest.len());
        for (slot_id, item) in slot_ids.into_iter().zip(to_ingest.into_iter()) {
            let (tx, rx) = crossbeam::channel::bounded(1);
            let content_len = item.content.len();
            self.scheduler_tx
                .send(SchedulerRequest::IngestSection {
                    sequence_id: slot_id,
                    section_id: item.section_id,
                    prefix_section_ids: prefix_section_ids.to_vec(),
                    tokens: item.tokens,
                    address: item.address,
                    debug_name: item.debug_name,
                    in_collection,
                    response_tx: tx,
                })
                .map_err(|_| ConversationError::SchedulerGone)?;
            let _ = item.token_count;
            pending.push((slot_id, item.section_id, content_len, rx));
        }
        let t_fire = std::time::Instant::now();

        let mut out: Vec<(SectionId, usize)> =
            Vec::with_capacity(pending.len() + out_skip.len() + restore_out.len());
        out.extend(out_skip);
        out.extend(restore_out);
        // Drain pending in **completion order** rather than submission
        // order.  Sections in a collection ingest in parallel under
        // the same scheduler batch; shorter sections finish earlier
        // than longer ones, but the scheduler doesn't reorder
        // responses — each section's reply lands on its own bounded(1)
        // channel as soon as its forward passes complete.  Iterating
        // `pending` in order with blocking `rx.recv()` would stall on
        // the longest section's channel, drain every shorter section's
        // channel in microseconds afterwards, and produce one big
        // GUI jump.  `crossbeam::channel::Select` lets us pick up the
        // next ready receiver instead, firing `on_section_done`
        // progressively as sections actually complete.
        while !pending.is_empty() {
            // Pick the next-ready receiver via `Select`, drop the
            // `Select` borrow before mutating `pending`.
            let (idx, recv_result) = {
                let mut select = crossbeam::channel::Select::new();
                for (_, _, _, rx) in &pending {
                    select.recv(rx);
                }
                let op = select.select();
                let idx = op.index();
                let rx = &pending[idx].3;
                let r = op.recv(rx).map_err(|_| ConversationError::SchedulerGone)?;
                (idx, r)
            };
            let (slot_id, section_id, content_len, _rx) = pending.swap_remove(idx);
            let seal = recv_result?;
            let block_count = seal.block_to.saturating_sub(seal.block_from);
            let _ = self.scheduler_tx.send(SchedulerRequest::FreeSequence {
                sequence_id: slot_id,
            });
            out.push((section_id, block_count));
            on_section_done(section_id, content_len);
        }
        let t_done = std::time::Instant::now();

        tracing::debug!(
            target: "candle_conversation::insert_section_collection",
            n = n_sections,
            alloc_ms = (t_alloc - t0).as_millis() as u64,
            fire_ms = (t_fire - t_alloc).as_millis() as u64,
            wait_ms = (t_done - t_fire).as_millis() as u64,
            "section_collection",
        );
        Ok(out)
    }

    /// Look up a section's symbolic name from the projection schema.
    /// Used as the `debug_name` field on the persisted SectionDecl
    /// record so manifest dumps surface human-readable section ids
    /// rather than just `SectionId(u32)` raw values.  Falls back to a
    /// formatted `section_<raw>` string when the section isn't found
    /// in any layer (e.g. throwaway sigs-probe sections).
    fn section_debug_name(&self, section_id: SectionId) -> String {
        for layer in &self.projection.schema().layers {
            for s in layer.system_prompt.all_sections() {
                if s.id == section_id {
                    return s.name.clone();
                }
            }
        }
        format!("section_{}", section_id.raw())
    }

    /// Allocate a fresh GPU slot bound to the workspace substrate
    /// **without** minting a [`crate::projection::TimelineId`] or
    /// registering a projection target.
    ///
    /// Used by section ingestion: each section gets a scratch slot for
    /// its IngestSection request, and the slot is freed once the seal
    /// completes.  Cheaper than [`Self::fork`] (which mints a timeline
    /// + writes a `TimelineManifest` record to disk per call) because
    /// section ingestion produces context-independent KV that doesn't
    /// belong to any conversation timeline.
    fn alloc_scratch_slot(&self) -> crate::Result<SequenceId> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::NewSequence {
                conversation: self.substrate.clone(),
                target: None,
                response_tx: tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        rx.recv().map_err(|_| ConversationError::SchedulerGone)?
    }

    /// Submit a user turn for processing.
    ///
    /// Returns immediately with a [`TurnHandle`] that can be waited on
    /// or streamed from. Returns `Err(TurnInFlight)` if a turn is
    /// already in progress.
    ///
    /// After consuming the handle via streaming, call [`finish_turn`](Self::finish_turn)
    /// to record the assistant response and clear the in-flight guard.
    pub fn submit_turn(&mut self, user_message: &str) -> crate::Result<TurnHandle> {
        self.submit_turn_with_options(user_message, TurnOptions::default())
    }

    /// Submit with per-turn options (sampling, max tokens).
    pub fn submit_turn_with_options(
        &mut self,
        user_message: &str,
        options: TurnOptions,
    ) -> crate::Result<TurnHandle> {
        if self.turn_in_flight {
            return Err(ConversationError::TurnInFlight {
                sequence_id: self.id,
            });
        }

        // Pin the system prompt into the substrate (idempotent).
        // Subsequent SubmitTurn calls re-inject it onto the slot via
        // `apply_projection`; the upload cache amortises after the
        // first turn.

        // Format: [/no_think] + user_message + user_end + assistant_start.
        let no_think_prefix = if self.config.suppress_thinking {
            self.config.dialect.no_think
        } else {
            ""
        };
        // The persisted turn shape is
        //     [no_think_prefix] + user_message + user_end + assistant_start [+ think_block] + decoded_body
        // — i.e. `user_start` is **not** in the prefill and `assistant_end`
        // is **not** appended after decode.  Both boundary markers are
        // emitted as live `Generated` segments by the projection engine
        // (the trailing `Generated(UserStart)` the scheduler appends
        // before this prefill, and the `Generated(AssistantEnd)` that
        // closes every past-turn injection on the next projection that
        // includes this turn).  Their K vectors are computed under the
        // actual runtime causal prefix on every projection, so the
        // attention-pivot K/V stays correct as projection-selected
        // context changes.  The intra-turn `user_end` and
        // `assistant_start` markers stay baked — their hidden state is
        // dominated by the turn's own (invariant) content.
        let formatted = format!(
            "{}{}{}{}",
            no_think_prefix,
            user_message,
            self.config.dialect.user_end,
            self.config.dialect.active_assistant_start(
                self.config.suppress_thinking,
                self.config.thinking_capable
                    && (!self.config.suppress_thinking || self.config.inject_no_think_block)
                    && self.config.dialect.supports_no_think(),
            ),
        );
        let prefill_tokens = self.tokenize(&formatted)?;

        // No post-decode tail in the Phase 5 layout.  The model's
        // `<|im_end|>` EOS doesn't get forwarded (sampling stops on
        // EOS without a follow-up forward pass), and the trailing
        // `\n` that used to come from `assistant_end` is now the live
        // `Generated(AssistantEnd)` segment the projection emits
        // after the sealed turn on every future projection.
        let post_decode_tokens = TokenBuffer::new();

        // Record the pending user turn (text + raw tokens for the tree).
        let user_tokens = self.tokenize(user_message)?;
        self.pending_user = Some(TokenizedText::new(user_message, user_tokens));

        let sampling = options
            .sampling
            .unwrap_or_else(|| self.config.sampling.clone());
        let max_tokens = options
            .max_tokens
            .unwrap_or(self.config.max_response_tokens);

        tracing::debug!(
            max_decode_tokens = max_tokens,
            forced_eos_after = sampling.forced_eos_after,
            graceful_eos_after = sampling.graceful_eos_after,
            eos_ramp_start = sampling.eos_ramp_start,
            "turn submitted"
        );

        let reprojection = self.build_reprojection_policy();
        let handle = self.submit_prefill_unit(
            self.id,
            Some(self.projection_inputs()),
            formatted,
            prefill_tokens,
            user_message.to_string(),
            post_decode_tokens,
            max_tokens,
            sampling,
            reprojection,
        )?;
        self.turn_in_flight = true;
        Ok(handle)
    }

    /// Send a `SubmitTurn` request to the scheduler, returning the
    /// streaming handle.
    ///
    /// The shared projection-then-prefill primitive that backs every
    /// content-into-KV path — turn submission, no-decode turn
    /// insertion, and section ingestion (one section onto a fork per
    /// parallel slot).  Callers control:
    ///
    /// - `parent_id` — which sequence the scheduler materialises the
    ///   projection into (the conversation's slot for turns, a
    ///   fork's slot for a parallel section).  The scheduler resets
    ///   it to empty before applying the projection — the substrate
    ///   is the canonical parent.
    /// - `projection_inputs` — when `Some`, the scheduler runs
    ///   `Builder::project()` itself against its substrate; when
    ///   `None`, no projection is applied.
    /// - `max_decode_tokens` — `0` for no-decode prefill (insert
    ///   paths); `> 0` to follow with a decode loop.
    /// - `reprojection` — `Some` enables continuous mid-decode
    ///   re-projection; `None` runs a single-shot prefill+decode.
    fn submit_prefill_unit(
        &self,
        sequence_id: SequenceId,
        projection_inputs: Option<ProjectionInputs>,
        prefill_text: String,
        prefill_tokens: TokenBuffer,
        user_text: String,
        post_decode_tokens: TokenBuffer,
        max_decode_tokens: usize,
        sampling: SamplingConfig,
        reprojection: Option<ReprojectionPolicy>,
    ) -> crate::Result<TurnHandle> {
        let disable_reprojection = self.config.disable_reprojection;
        // Append-only ingests skip the per-turn projection rebuild and also
        // suppress continuous mid-decode reprojection.
        let reprojection = if disable_reprojection {
            None
        } else {
            reprojection
        };
        let (event_tx, event_rx) = crossbeam::channel::unbounded();
        self.scheduler_tx
            .send(SchedulerRequest::SubmitTurn {
                sequence_id,
                projection_inputs,
                prefill_tokens,
                prefill_text,
                user_text,
                post_decode_tokens,
                max_decode_tokens,
                sampling,
                event_tx,
                reprojection,
                disable_reprojection,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        Ok(TurnHandle::new(event_rx))
    }

    /// Build the projection inputs the scheduler needs to run
    /// `Builder::project()` for this conversation's `target`.  The
    /// target itself is pinned on the slot (via the scheduler's
    /// `slot_targets` map at `NewSequence` time) and read from there
    /// at handler entry.
    fn projection_inputs(&self) -> ProjectionInputs {
        ProjectionInputs {
            projection: Arc::clone(&self.projection),
        }
    }

    /// Insert a preformed turn into the conversation **without model inference**.
    ///
    /// Prefills the full user-and-assistant exchange into the KV
    /// cache in one forward pass, then runs the normal post-turn
    /// logic (summarisation tasks fire as usual with real inference).
    ///
    /// Goes through the same projection-then-prefill path as
    /// [`submit_turn`](Self::submit_turn): the parent is set to
    /// `system + projected_turns` before the exchange is prefilled,
    /// so the model's hidden states for the prefilled tokens reflect
    /// the same context the conversation would see for a normal
    /// `submit_turn`.  The only difference is `max_decode_tokens=0`
    /// — the assistant half is provided rather than generated.
    ///
    /// Useful for:
    /// - Injecting known exchanges cheaply in tests.
    /// - Seeding a conversation with prior context without re-running
    ///   the original decodes.
    ///
    /// Returns `Err(TurnInFlight)` if a turn is currently in progress.
    pub fn insert_turn(&mut self, user_message: &str, assistant_text: &str) -> crate::Result<()> {
        if self.turn_in_flight {
            return Err(ConversationError::TurnInFlight {
                sequence_id: self.id,
            });
        }

        // Format the full exchange (user + assistant_start prefix +
        // assistant_text) and tokenize as a single prefill payload.
        let no_think_prefix = if self.config.suppress_thinking {
            self.config.dialect.no_think
        } else {
            ""
        };
        // Format: {user_start}[/no_think]{user}{user_end}{assistant_start}{assistant_text}.
        // The assistant role-end comes through `post_decode_tokens`
        // — there is no decode in this path so the EOS isn't emitted
        // by the model; we append the full `assistant_end` (EOS +
        // structural tail) ourselves so the seal captures a
        // structurally-complete turn.
        // Same Phase 5 shape as `submit_turn`: `user_start` and
        // `assistant_end` are stripped from the persisted bytes; the
        // projection's live `Generated` segments re-emit them around
        // the sealed turn on every future projection.  The trailing
        // `Generated(UserStart)` the scheduler appends ahead of this
        // prefill supplies the live `<|im_start|>user\n` opener.
        let formatted = format!(
            "{}{}{}{}{}",
            no_think_prefix,
            user_message,
            self.config.dialect.user_end,
            self.config.dialect.active_assistant_start(
                self.config.suppress_thinking,
                self.config.thinking_capable
                    && (!self.config.suppress_thinking || self.config.inject_no_think_block)
                    && self.config.dialect.supports_no_think(),
            ),
            assistant_text,
        );
        let prefill_tokens = self.tokenize(&formatted)?;
        let post_decode_tokens = TokenBuffer::new();

        // Build the TokenizedText for both halves now — assistant
        // text is supplied directly, not decoded, so we can fill it
        // in without waiting on event_rx for token chunks.
        let user_tokens = self.tokenize(user_message)?;
        let user_tt = TokenizedText::new(user_message, user_tokens);
        let asst_tokens = self.tokenize(assistant_text)?;
        let asst_tt = TokenizedText::new(assistant_text, asst_tokens);

        // Submit through the shared projection-then-prefill path with
        // `max_decode_tokens = 0` — the scheduler applies the
        // projection, prefills the full exchange, appends
        // `post_decode_tokens`, and emits Done without entering
        // decode.
        let handle = self.submit_prefill_unit(
            self.id,
            Some(self.projection_inputs()),
            formatted,
            prefill_tokens,
            user_message.to_string(),
            post_decode_tokens,
            0,
            self.config.sampling.clone(),
            None,
        )?;

        // Drain events synchronously to Done.  The handle's event_rx
        // receives Prefill (text echo) and Done; we just need to
        // observe Done so we know the parent's KV is fully populated
        // before we register the turn.
        let response = handle.wait()?;

        // Run the same post-Done finalize as a regular turn.
        self.finalize_turn_post_done(user_tt, asst_tt, response.seal.as_ref())?;
        Ok(())
    }

    /// Blocking convenience: submit + wait.
    ///
    /// Automatically records the assistant response, prefills the next
    /// user header, and clears in-flight.
    pub fn send_turn(&mut self, user_message: &str) -> crate::Result<TurnResponse> {
        let handle = self.submit_turn(user_message)?;
        let response = handle.wait()?;

        // Record the assistant turn and prefill next user header.
        self.finish_turn(handle, &response)?;

        Ok(response)
    }

    /// Record the assistant response after streaming, prefill the next
    /// user turn header (`<|im_end|>\n<|im_start|>user\n`) into the KV
    /// cache, and clear the in-flight guard.
    ///
    /// Takes the handle returned by [`submit_turn`](Self::submit_turn) so it
    /// can finalize any windowed-context view sequence that was created for
    /// this turn.
    ///
    /// Does **not** return a handle — the next-user-header prefill fires
    /// fire-and-forget; FIFO ordering in the scheduler guarantees it completes
    /// before the next `submit_turn` starts.
    ///
    /// Returns the exact boundary text that was prefilled into the KV cache
    /// (e.g. `"<|im_end|>\n<|im_start|>user\n"`).  Callers that need to
    /// display the full token stream can print this string immediately after
    /// `finish_turn` returns — at which point the seal has completed and the
    /// tokens are guaranteed to be in the cache.
    pub fn finish_turn(
        &mut self,
        handle: TurnHandle,
        response: &TurnResponse,
    ) -> crate::Result<String> {
        // Take the pending user turn recorded at submit time.
        let user_tt = self.pending_user.take().unwrap_or_default();
        let assistant_tt = TokenizedText::new(response.text.as_str(), response.token_ids.clone());

        self.turn_in_flight = false;

        // The view sequence is owned by the scheduler and was already
        // auto-finalized in `cleanup_finished` before this `Done` event was
        // sent — so the parent now holds all of the turn's KV blocks.  We
        // drop the handle here purely to release the event channel.
        drop(handle);

        self.finalize_turn_post_done(user_tt, assistant_tt, response.seal.as_ref())
    }

    /// Shared post-prefill turn-completion path.
    ///
    /// After a `SubmitTurn` request emits `Done` (whether decoded by
    /// the model or fully prefilled inline), both `finish_turn` and
    /// `insert_turn` need the same fold-down work:
    ///
    /// 1. Apply the scheduler's seal payload (cold store register +
    ///    sig-blocks counter advance + BDP scan probe).  The
    ///    substrate write itself happened scheduler-side before
    ///    `Done` arrived.
    /// 2. Persist the user/assistant entries to the cold store.
    /// 3. Commit the exchange to the conversation tree (which may
    ///    queue cognitive tasks like summarisation).
    /// 4. Drain any cognitive tasks the tree launched.
    /// 5. Prefill the next user-turn header into the KV cache.
    ///
    /// Returns the boundary text prefilled in step 5 so callers can
    /// echo it for streaming displays.
    fn finalize_turn_post_done(
        &mut self,
        user_tt: TokenizedText,
        assistant_tt: TokenizedText,
        seal: Option<&crate::handle::SealResult>,
    ) -> crate::Result<String> {
        // Apply the seal payload that came back on Done.  The
        // scheduler already wrote the turn into the substrate; we
        // just register the bytes with the cold-store and advance
        // local counters.
        if let Some(seal) = seal {
            self.sig_blocks_processed = seal.sig_blocks_processed;
            self.current_blocks = BlockCount(seal.block_count);
            // The scheduler already wrote the turn's sealed bytes,
            // per-block sig entries, role/text/token_ids, AND the
            // workspace persistence layer (when configured) into the
            // substrate before sending `Done` — see
            // `Conversation::record_turn`.  Persistence is per-
            // workspace now, not per-Sequence; nothing for this
            // method to forward.
            // Refresh BDP scores using the just-finished turn's
            // chunk-group as probe.  The substrate already has the
            // turn (the scheduler wrote it before sending Done), so
            // the new turn's index is the current timeline tail minus 1.
            let timeline_count = self.substrate.read().turn_count(self.target.timeline);
            if timeline_count > 0 {
                let last_idx = TurnIndex(timeline_count - 1);
                if let Err(e) = self.run_bdp_scan(self.target.timeline, last_idx) {
                    tracing::warn!("BDP scan failed: {e}");
                }
            }
        }

        // Bump the local turn counter for diagnostics; the canonical
        // per-group append index lives on the substrate now.
        self.turn_counter += 2;

        // Commit the completed exchange to the conversation tree.
        self.tree.finish_turn(
            user_tt,
            assistant_tt,
            TurnType::Reality,
            vec![],
            Some((&self.scheduler_tx, &self.tokenizer)),
        );

        // Drain pending cognitive tasks launched by the tree during
        // finish_turn() and spin-poll each to completion before returning.
        self.drain_cognitive_tasks();

        // Each turn opens its own user role marker via the
        // `prefill_tokens` of the next `submit_turn` and closes the
        // assistant tail via `post_decode_tokens` after decode.
        // No inter-turn header prefill is needed.
        Ok(String::new())
    }

    /// Fork this sequence: allocate a fresh slot on the same
    /// workspace `Conversation` (substrate handle), with its own
    /// fresh [`crate::projection::TimelineId`] so the fork's turns
    /// don't interleave with the parent's in the substrate.  The
    /// fork's initial slot is empty — the next `submit_turn`
    /// materialises the parent's history onto it via
    /// `apply_projection` from the shared substrate.  No GPU CoW,
    /// no chunk copying; substrate is the canonical parent.
    pub fn fork(&self) -> crate::Result<Sequence> {
        // Mint a fresh timeline within the parent's (layer, group)
        // shape — each fork is its own conversation thread.
        let fork_timeline = self
            .substrate
            .mint_timeline(self.target.layer, self.target.group);
        self.fork_onto(fork_timeline)
    }

    /// Fork onto a **specific** timeline rather than a freshly minted one —
    /// the daemon resume path (§16.12 of `docs/kv_tier_migration.md`).
    ///
    /// `timeline` is registered against the parent's `(layer, group)`
    /// (idempotent). If the workspace substrate already holds turns under
    /// `timeline` — reconstructed from the redo log on startup — the next
    /// `submit_turn` materialises that recovered history onto the fork's
    /// slot, exactly as an in-process fork inherits its parent's turns.
    /// For an unknown `timeline` the fork simply starts empty.
    pub fn fork_resuming(&self, timeline: TimelineId) -> crate::Result<Sequence> {
        self.substrate
            .register_timeline(timeline, self.target.layer, self.target.group);
        self.fork_onto(timeline)
    }

    /// Shared body of [`Self::fork`] / [`Self::fork_resuming`]: allocate a
    /// fresh scheduler slot bound to the workspace substrate and `timeline`.
    fn fork_onto(&self, fork_timeline: TimelineId) -> crate::Result<Sequence> {
        let fork_target = ProjectionTarget {
            layer: self.target.layer,
            group: self.target.group,
            timeline: fork_timeline,
        };

        // Allocate a fresh slot bound to the same workspace handle and
        // the fork's target.
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::NewSequence {
                conversation: self.substrate.clone(),
                target: Some(fork_target),
                response_tx: tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;

        let new_seq_id = rx.recv().map_err(|_| ConversationError::SchedulerGone)??;

        let fork_conv = Sequence {
            scheduler_tx: self.scheduler_tx.clone(),
            id: new_seq_id,
            tokenizer: Arc::clone(&self.tokenizer),
            tree: self.tree.clone(),
            pending_user: None,
            turn_counter: self.turn_counter,
            config: self.config.clone(),
            turn_in_flight: false,
            freed: false,
            current_blocks: self.current_blocks,
            chunk_size: self.chunk_size,
            provenance: self.provenance.clone(),
            model_core: self.model_core,
            // The fork inherits all parent blocks already indexed.  Sig extraction
            // during seal_and_detach_into uses this as the range start, so only the
            // delta (blocks added after the fork) gets scanned — not the entire
            // inherited parent history.
            sig_blocks_processed: self.current_blocks.0,
            projection: self.projection.clone(),
            // Forks share the same substrate (Arc clone) so cross-fork
            // history aggregation continues to work.
            substrate: self.substrate.clone(),
            target: fork_target,
            // Forks start with a fresh scanner state — scoring will refresh
            // on the next BDP scan.  No need to clone the parent's scores.
            bdp_scanner: crate::provenance::BdpScanner::new(),
        };
        Ok(fork_conv)
    }

    /// Reset this conversation to its initial state.
    ///
    /// Clears the GPU KV cache and resets client-side state.  The next
    /// `submit_turn` will re-seed the system into the substrate via
    /// `ensure_system_ingested` and apply_projection will materialise
    /// it back onto the slot.
    ///
    /// The system prompt, config, and tokenizer are preserved; all turn
    /// history is cleared.
    ///
    /// Returns an error if a turn is currently in flight.
    pub fn reset(&mut self) -> crate::Result<()> {
        if self.turn_in_flight {
            return Err(ConversationError::TurnInFlight {
                sequence_id: self.id,
            });
        }

        let (response_tx, response_rx) = crossbeam::channel::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::ResetSequence {
                sequence_id: self.id,
                response_tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        response_rx
            .recv()
            .map_err(|_| ConversationError::SchedulerGone)??;

        self.pending_user = None;
        // The substrate is workspace-shared — don't reset its turn store
        // (other conversations would lose their history).  Only the local
        // KV state and BDP score cache are cleared here.
        self.bdp_scanner.clear();
        self.current_blocks = BlockCount(0);

        // Clear turn history (keeps system prompt, config, beliefs).
        self.tree.clear_turns();

        Ok(())
    }

    /// Extract raw K/V/Q float data from the KV cache for a set of layer
    /// indices, synchronously on the scheduler thread.
    ///
    /// Call this after [`Self::finish_turn`] and before dropping the sequence —
    /// the parent slot's KV cache is live in that window.
    ///
    /// Returns one entry per requested layer in the same order:
    /// `(layer_idx, Vec<(block_idx, k_flat, v_flat, q_flat)>)`.
    pub fn extract_raw_kvq(
        &self,
        layer_indices: Vec<usize>,
        block_range: Option<(usize, usize)>,
    ) -> crate::Result<Vec<(usize, Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>)>> {
        let (tx, rx) = crossbeam::channel::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::ExtractRawKvq {
                sequence_id: self.id,
                layer_indices,
                block_range,
                response_tx: tx,
            })
            .map_err(|_| crate::error::ConversationError::SchedulerGone)?;
        rx.recv()
            .map_err(|_| crate::error::ConversationError::SchedulerGone)?
    }

    /// Close this conversation, releasing the sequence slot.
    ///
    /// Consumes `self`. If not called, the slot is freed on drop (best-effort).
    pub fn close(mut self) -> crate::Result<()> {
        self.freed = true;
        self.scheduler_tx
            .send(SchedulerRequest::FreeSequence {
                sequence_id: self.id,
            })
            .map_err(|_| ConversationError::SchedulerGone)
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// Get the sequence ID.  This is the slot index allocated by the
    /// scheduler — the sequence's only identifier under the
    /// substrate-as-parent model (persistence is workspace-scoped).
    pub fn id(&self) -> SequenceId {
        self.id
    }

    /// Number of turns in this conversation.
    pub fn turn_count(&self) -> u64 {
        self.turn_counter
    }

    /// Get all turns in chronological order.
    ///
    /// Synthesizes a flat `[System?, User, Assistant, User, Assistant, ...]`
    /// view from the conversation tree. Useful for inspection and testing;
    /// the tree's own `tree().turns()` iterator gives richer metadata.
    pub fn turns(&self) -> Vec<Turn> {
        let mut result: Vec<Turn> = Vec::new();
        let mut id: u64 = 0;

        // System prompt (if any).
        let sys_text = self.tree.system_prompt_text();
        if !sys_text.is_empty() {
            result.push(Turn {
                id,
                role: Role::System,
                text: sys_text.to_string(),
                token_ids: TokenBuffer::from(self.tree.system_prompt_token_ids()),
            });
            id += 1;
        }

        // Paired user↔assistant exchanges.
        for ct in self.tree.turns() {
            let inner = ct.inner();
            result.push(Turn {
                id,
                role: Role::User,
                text: inner.user.text().to_string(),
                token_ids: TokenBuffer::from(inner.user.token_ids()),
            });
            id += 1;
            result.push(Turn {
                id,
                role: Role::Assistant,
                text: inner.assistant.text().to_string(),
                token_ids: TokenBuffer::from(inner.assistant.token_ids()),
            });
            id += 1;
        }

        result
    }

    /// The raw system prompt text (with any temporal marker suffix, or empty).
    pub fn system_prompt(&self) -> &str {
        self.tree.system_prompt_text()
    }

    /// Every recovered turn in the given timeline, split into a
    /// `User` half and an `Assistant` half — for re-populating a
    /// sidebar after restart.
    ///
    /// Each turn surfaces as two `(Role, String)` entries in order:
    /// the user's message (exactly what `submit_turn` received) then
    /// the assistant's reply (the decoded body).  Both strings come
    /// straight off `TurnPart::user_text` and `assistant_text` — no
    /// re-tokenising, no marker scanning, no decoding.
    pub fn recovered_history(&self, timeline: TimelineId) -> Vec<(Role, String)> {
        let read = self.substrate.read();
        let mut out: Vec<(Role, String)> = Vec::new();
        for idx in read.turn_indices(timeline) {
            let user_text = read.user_text_of(timeline, idx);
            let assistant_text = read.assistant_text_of(timeline, idx);
            if !user_text.is_empty() {
                out.push((Role::User, user_text));
            }
            if !assistant_text.is_empty() {
                out.push((Role::Assistant, assistant_text));
            }
        }
        out
    }

    /// Every timeline with at least one recovered turn, paired with the
    /// turn count.
    pub fn recovered_timelines(&self) -> Vec<(TimelineId, u32)> {
        let read = self.substrate.read();
        let mut counts: std::collections::HashMap<TimelineId, u32> =
            std::collections::HashMap::new();
        for key in read.all_turns() {
            *counts.entry(key.timeline).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }

    /// The substrate-backed sidebar label for `timeline`, or `None`.
    pub fn conversation_label_of(&self, timeline: TimelineId) -> Option<String> {
        self.substrate.label_of(timeline)
    }

    /// This sequence's timeline id.
    pub fn timeline_id(&self) -> TimelineId {
        self.target.timeline
    }

    /// Set this conversation's sidebar label, persisting it to the redo
    /// log. First-write-wins.
    pub fn set_conversation_label(&self, label: &str) -> crate::Result<()> {
        self.substrate
            .set_conversation_label(self.target.timeline, label)
            .map_err(ConversationError::Model)
    }

    /// Merge a `(key, value)` into this conversation's free-form `custom`
    /// metadata bag and persist it (see [`ConvMeta::custom`]). Utility
    /// ingests tag conversations with a content hash + descriptive fields
    /// for the restart-resume cache.
    pub fn set_metadata(&self, key: &str, value: &str) -> crate::Result<()> {
        self.substrate
            .set_conversation_metadata(self.target.timeline, key, value)
            .map_err(ConversationError::Model)
    }

    /// Merge several `(key, value)` pairs into this conversation's
    /// `custom` metadata in one persisted record.
    pub fn set_metadata_many(
        &self,
        kv: &std::collections::BTreeMap<String, String>,
    ) -> crate::Result<()> {
        self.substrate
            .set_conversation_metadata_many(self.target.timeline, kv)
            .map_err(ConversationError::Model)
    }

    /// This conversation's `custom` metadata bag, or `None` if the
    /// timeline isn't registered yet.
    pub fn metadata(&self) -> Option<std::collections::BTreeMap<String, String>> {
        self.substrate.conversation_metadata(self.target.timeline)
    }

    /// Every timeline (across the whole substrate) whose `custom`
    /// metadata contains `key == value`. Used by utility ingests to skip
    /// re-building units already present after substrate load.
    pub fn find_conversations_by_metadata(&self, key: &str, value: &str) -> Vec<TimelineId> {
        self.substrate.find_timelines_by_metadata(key, value)
    }

    /// Whether a turn is currently in flight.
    pub fn is_in_flight(&self) -> bool {
        self.turn_in_flight
    }

    // ── Test-introspection helpers ────────────────────────────────────────────

    /// Returns `(completed_turns, window_size)` where `completed_turns` is the
    /// number of entries in `turn_block_counts` (one per sealed `send_turn` /
    /// `insert_turn` call) and `window_size` is `config.context_window_turns`.
    ///
    /// Context masking activates when `completed_turns > window_size`; at that
    /// point the model only attends to the system prompt plus the most recent
    /// `window_size` turns.  Use this in tests to assert that the window has
    /// not (or has) become active after a given number of turns.
    pub fn window_state(&self) -> (usize, usize) {
        (
            self.substrate.read().turn_count(self.target.timeline) as usize,
            self.config.context_window_turns,
        )
    }

    /// Read-only view of the conversation tree.
    ///
    /// The tree mirrors the flat `turns()` slice but carries the full Phase 1
    /// type system: `TurnId` temporal coordinates, `TurnType`, children links,
    /// and the system prompt as a first-class field. Use `tree().nodes()` to
    /// iterate over `ConversationNode`s and inspect per-turn metadata.
    pub fn tree(&self) -> &ConversationTree {
        &self.tree
    }

    /// Writable access to the conversation tree
    pub fn tree_mut(&mut self) -> &mut ConversationTree {
        &mut self.tree
    }

    /// Spin-poll a cognitive task to completion, applying the resulting
    /// [`TreePatch`](crate::tree::TreePatch) to the tree if one arrives.
    ///
    /// After each [`TreePatch`] is applied, checks whether a recursive
    /// segment-of-segments summarization should fire and, if so, queues the
    /// new task onto the tree's `pending_tasks`. The outer drain loop in
    /// `send()` then picks it up automatically.
    ///
    /// This is the "crude blocking" variant from the design doc — acceptable
    /// for infrequent summarization events. Upgrade to async polling later.
    fn run_task_blocking_inner(
        tree: &mut ConversationTree,
        task: &mut dyn CognitiveTask,
        inference: Option<(
            &crossbeam::channel::Sender<SchedulerRequest>,
            &std::sync::Arc<tokenizers::Tokenizer>,
        )>,
    ) {
        loop {
            match task.poll() {
                TaskPoll::Ready(patch) => {
                    tree.apply_patch(patch);
                    tree.check_and_trigger_segment_summarize(inference);
                    return;
                }
                TaskPoll::Aborted => return,
                TaskPoll::Failed(e) => {
                    tracing::warn!("cognitive task failed: {}", e);
                    return;
                }
                TaskPoll::Pending => std::thread::yield_now(),
            }
        }
    }

    /// Build the canonical token sequence for one completed turn.
    ///
    /// The format is: `[no_think] user_text user_end asst_start asst_text turn_end user_start`
    /// This matches what is prefilled during a normal `submit_turn` / `finish_turn` cycle and
    /// is used to reconstruct the context window during the pre-submit phase.
    fn tokenize(&self, text: &str) -> crate::Result<TokenBuffer> {
        self.tokenizer
            .encode(text, false)
            .map(|enc| TokenBuffer::from(enc.get_ids()))
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))
    }

    // ── Turn-lifecycle helpers ────────────────────────────────────────────────

    /// Build the per-turn [`ReprojectionPolicy`] when continuous
    /// re-projection is enabled in config.  Returns `None` when both
    /// the cadence and punctuation triggers are disabled — the
    /// scheduler then runs the turn as a single-shot prefill+decode
    /// without mid-decode view swaps.
    fn build_reprojection_policy(&self) -> Option<ReprojectionPolicy> {
        let n = self.config.reproject_every_n_tokens;
        let trigger_ids = Arc::new(self.build_reproject_trigger_token_ids());
        if n == 0 && trigger_ids.is_empty() {
            return None;
        }
        let probe_filter = Arc::new(self.build_reproject_probe_filter_token_ids());
        Some(ReprojectionPolicy {
            target: self.target,
            projection: Arc::clone(&self.projection),
            substrate: self.substrate.clone(),
            provenance: Arc::clone(&self.provenance),
            provenance_layer_indices: self.model_core.provenance_layer_indices,
            every_n_tokens: n,
            max_probe_tokens: self.config.reproject_max_probe_tokens.max(1),
            probe_filter_token_ids: probe_filter,
            trigger_token_ids: trigger_ids,
            span_alpha: self.projection.span_alpha(),
        })
    }

    /// Encode every token that's structural rather than content into a
    /// deduplicated set to drop from the BDP probe.
    ///
    /// Three categories — anything in any of them appears in (almost)
    /// every turn's KV, so keeping them in the probe inflates each
    /// historical turn's score by roughly the same constant amount and
    /// dilutes the actual content signal.
    ///
    /// 1. **Whitespace** — plain ASCII (`\n`, ` `, `\t`, `\n\n`) plus
    ///    GPT-2 byte-level BPE encodings (`Ċ` = U+010A, `ċ` = U+0109,
    ///    `Ġ` = U+0120, `ĊĊ` for double-newline).
    /// 2. **Markdown punctuation** — `*` `-` `#` `` ` `` `|` `>` `\` `/`
    ///    (bullets, list dashes, headings, code fences, table pipes,
    ///    blockquotes, escapes, path separators).  Plus their
    ///    space-prefixed BPE forms (`Ġ*`, `Ġ-`, …) which most
    ///    tokenizers emit when these characters open a word.
    /// 3. **Chat-template scaffolding** from the dialect — turn-boundary
    ///    markers (`<|im_start|>`, `<|im_end|>`, role labels,
    ///    think-block markers, etc.).
    ///
    /// Mirrors [`DecodeHealthConfig::resolve_structural_tokens`] for
    /// categories 1 and 2 — kept as a parallel list rather than a
    /// shared call because the conversation only has
    /// [`SequenceConfig`], not [`EngineConfig`].
    fn build_reproject_probe_filter_token_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::new();
        let add = |ids: &mut Vec<u32>, s: &str| {
            if s.is_empty() {
                return;
            }
            if let Ok(enc) = self.tokenizer.encode(s, false) {
                for &id in enc.get_ids() {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
            }
        };

        // Category 1: whitespace.
        for s in ["\n", " ", "\t", "\n\n"] {
            add(&mut ids, s);
        }
        for s in [
            "\u{010A}",         // Ċ — newline
            "\u{0109}",         // ċ — tab
            "\u{0120}",         // Ġ — space
            "\u{010A}\u{010A}", // ĊĊ — double newline
        ] {
            add(&mut ids, s);
        }

        // Category 2: markdown / formatting punctuation, plus the
        // space-prefixed BPE forms most tokenizers emit at word starts.
        const MARKDOWN: &[char] = &['*', '-', '#', '`', '|', '>', '\\', '/'];
        for &c in MARKDOWN {
            let s = c.to_string();
            add(&mut ids, &s);
            let with_space = format!("\u{0120}{c}");
            add(&mut ids, &with_space);
        }

        // Category 3: chat-template scaffolding from the dialect.
        let d = &self.config.dialect;
        for s in [
            d.document_start,
            d.document_end,
            d.marker_start,
            d.marker_end,
            d.turn_start,
            d.turn_begin,
            d.turn_end,
            d.system_start,
            d.system_end,
            d.user_start,
            d.user_end,
            d.assistant_start,
            d.assistant_end,
            d.recent_start,
            d.recent_end,
            d.no_think_block,
            d.no_think,
            d.think_block,
        ] {
            add(&mut ids, s);
        }

        // Category 4: JSON structural punctuation.  Tool-definition
        // sections are prefilled as JSON blobs and tool calls are emitted
        // as JSON; braces, brackets, quotes and separators carry no
        // semantic signal and match across every section purely on shared
        // JSON structure — pure BDP noise.  Both bare forms and the BPE
        // merges a tokenizer emits inside compact JSON are added.
        for s in [
            "{", "}", "[", "]", ":", ",", "\"", "{\"", "\"}", "\"}}", "\":\"", "\",\"", "\":",
            "\",", "[{", "}]",
        ] {
            add(&mut ids, s);
        }
        ids
    }

    /// Encode each `reproject_trigger_texts` entry via the tokenizer
    /// and collect their IDs.  Multi-token texts contribute all their
    /// IDs (any one matching fires the trigger) — typically the
    /// caller passes single-character strings like `"\n"` so this is
    /// one ID per entry.
    fn build_reproject_trigger_token_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::new();
        for text in &self.config.reproject_trigger_texts {
            if text.is_empty() {
                continue;
            }
            if let Ok(enc) = self.tokenizer.encode(text.as_str(), false) {
                for &id in enc.get_ids() {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
            }
        }
        ids
    }

    /// Drain all pending cognitive tasks from the tree to completion,
    /// re-draining after each batch to handle recursive tasks queued by
    /// segment-of-segments summarization.
    fn drain_cognitive_tasks(&mut self) {
        let inference = Some((&self.scheduler_tx, &self.tokenizer));
        loop {
            let tasks = self.tree.drain_pending_tasks();
            if tasks.is_empty() {
                break;
            }
            for mut task in tasks {
                Self::run_task_blocking_inner(&mut self.tree, task.as_mut(), inference);
            }
        }
    }

    /// Run a BDP scan using the sig entries of `(probe_group, probe_index)`
    /// as the probe and every other tracked `(group, idx)` as the corpus.
    /// Updates the session resolver's per-turn `PerDepthScores` in place.
    ///
    /// Pure no-op when the probe turn has no sig entries (e.g. the seal
    /// produced zero new chunks).  Errors from the mmap path bubble up.
    pub(crate) fn run_bdp_scan(
        &mut self,
        probe_timeline: crate::projection::TimelineId,
        probe_index: TurnIndex,
    ) -> crate::Result<()> {
        // Snapshot probe + corpus under a single read lock, then drop the
        // guard before doing the (CPU-heavy) BDP scan and the write phase.
        //
        // BdpScanner is keyed by `TurnKey` natively — no group/timeline
        // translation needed.
        let probe_key = crate::projection::TurnKey::new(probe_timeline, probe_index);
        let (probe_entries, turn_corpus, section_corpus): (
            Vec<crate::provenance::SigEntry>,
            Vec<(crate::projection::TurnKey, Vec<crate::provenance::SigEntry>)>,
            Vec<(SectionId, Vec<crate::provenance::SigEntry>)>,
        ) = {
            let view = self.substrate.read();
            let probe = view.sig_entries_of(probe_timeline, probe_index).to_vec();
            if probe.is_empty() {
                return Ok(()); // nothing to probe with
            }
            let turn_corpus: Vec<_> = view
                .all_turns()
                .filter_map(|key| {
                    if key == probe_key {
                        return None;
                    }
                    let entries = view.sig_entries_of(key.timeline, key.index).to_vec();
                    if entries.is_empty() {
                        None
                    } else {
                        Some((key, entries))
                    }
                })
                .collect();
            let section_corpus: Vec<_> = view
                .all_sections()
                .map(|sid| (sid, view.section_sig_entries(sid).to_vec()))
                .filter(|(_, e)| !e.is_empty())
                .collect();
            (probe, turn_corpus, section_corpus)
        };

        // Read probe TokenSignatures back from the ProvenanceFile, concatenated
        // across all of the probe turn's chunks so each probe-token contributes.
        let mut probe_syn = Vec::new();
        let mut probe_sem = Vec::new();
        let mut probe_prag = Vec::new();
        for entry in &probe_entries {
            let (syn, sem, prag) = self.provenance.read_entry(*entry)?;
            probe_syn.extend(syn);
            probe_sem.extend(sem);
            probe_prag.extend(prag);
        }

        self.bdp_scanner.scan(
            &self.provenance,
            &probe_syn,
            &probe_sem,
            &probe_prag,
            &turn_corpus,
        )?;
        self.bdp_scanner.scan_sections(
            &self.provenance,
            &probe_syn,
            &probe_sem,
            &probe_prag,
            &section_corpus,
        )?;

        // Scanner output stays on `self.bdp_scanner` — it is **not**
        // pushed into the substrate. Scores are transient, per-projection
        // state: a downstream consumer wanting scored projection on this
        // Sequence's substrate view builds a [`ProjectionScores`] from
        // `self.bdp_scanner.scores()` / `.section_scores()` at the call
        // site (see `BdpScanner::to_projection_scores`) and reads with
        // [`Conversation::read_scored`]. The substrate's persistent
        // identity does not include scoring state.
        Ok(())
    }
}

impl Drop for Sequence {
    fn drop(&mut self) {
        if !self.freed {
            // Best-effort: free the sequence slot.
            let _ = self.scheduler_tx.send(SchedulerRequest::FreeSequence {
                sequence_id: self.id,
            });
        }
    }
}
