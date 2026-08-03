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
    from_projection_with_origins, Builder, Conversation, ProjectionEvent, ProjectionMode,
    ProjectionTarget, SectionId, SelectionState, SystemPromptItem, TimelineId, TurnIndex,
};
use crate::provenance::WideQSig;
use crate::scheduler::projection_assembler::{materialize_conversation, BoundaryMarkers};
use crate::scheduler::{ProjectionInputs, ReprojectionPolicy, SchedulerRequest};
use crate::sequence_handle::{BlockCount, SequenceId};
use crate::token_buffer::TokenBuffer;
use crate::tree::token_text::TokenizedText;
use crate::tree::{CognitiveTask, ConversationTree, TaskPoll, TurnType};
use crate::turn::{Role, Turn, TurnOptions};
use crate::turn_layout::TurnLayout;
use crate::TurnEvent;
use candle_nn::kv_cache::{SealedChunk, SealedSequence};
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

/// Window a turn's per-layer sealed K/V down to the token range
/// `[start_tok, end_tok)` as a zero-copy view over the *same* physical
/// chunks.
///
/// A turn is sealed once as a single contiguous unit, with the chat
/// template's role markers baked into the K/V grid:
/// `[user_start][no_think][user_msg][user_end][assistant_start][response]`.
/// The compressor injects *content-only* halves of a turn — the user
/// message body or the assistant response body — without those markers,
/// so each half is derived here by windowing the sealed grid to the
/// content's token span.
///
/// Each returned `SealedSequence` references the same `ChunkGid`s the
/// turn already owns, windowing the partially-covered boundary chunks by
/// `offset` / `token_count` so no K/V bytes are copied.  The kernel
/// honours `offset` (it packs into the high 16 bits of `ChunkMeta`), so
/// a window can be injected at any tail position and read correctly.
///
/// Walking each layer's chunks with a running token start `acc` and the
/// chunk's own `c = token_count`, the chunk's slot span is `[acc, acc + c)`.
/// Intersecting it with `[start_tok, end_tok)`:
/// - empty intersection → the chunk is skipped.
/// - the intersection equals the whole chunk → the chunk is cloned as-is.
/// - otherwise the chunk straddles a window edge → a clone is taken with
///   `offset += (overlap_start - acc)` and `token_count = overlap_len`.
///   The clone shares the same `ChunkGid`s as the source chunk.
///
/// The result's `token_count == end_tok - start_tok` (clamped to the
/// sequence's own length).  `start_tok >= end_tok` yields an empty window.
pub(crate) fn window_sealed_tokens(
    sealed: &[SealedSequence],
    start_tok: usize,
    end_tok: usize,
) -> Vec<SealedSequence> {
    let mut layers: Vec<SealedSequence> = Vec::with_capacity(sealed.len());

    for seq in sealed {
        let total = seq.token_count;
        let win_start = start_tok.min(total);
        let win_end = end_tok.min(total).max(win_start);
        let mut chunks: Vec<SealedChunk> = Vec::new();

        let mut acc = 0usize;
        for chunk in &seq.chunks {
            let c = chunk.token_count as usize;
            let chunk_start = acc;
            let chunk_end = acc + c;
            acc = chunk_end;

            // Intersect the chunk's slot span with the window.
            let overlap_start = chunk_start.max(win_start);
            let overlap_end = chunk_end.min(win_end);
            if overlap_start >= overlap_end {
                // No overlap — chunk lies wholly outside the window.
                continue;
            }
            let overlap_len = (overlap_end - overlap_start) as u16;
            if overlap_start == chunk_start && overlap_len as usize == c {
                // Whole chunk inside the window — clone as-is.
                chunks.push(chunk.clone());
            } else {
                // Partial chunk: window it, sharing the same physical
                // chunk via the cloned `ChunkGid`s.
                let mut w = chunk.clone();
                w.offset = chunk.offset + (overlap_start - chunk_start) as u16;
                w.token_count = overlap_len;
                chunks.push(w);
            }
        }

        layers.push(SealedSequence {
            chunks,
            token_count: win_end - win_start,
            chunk_size: seq.chunk_size,
            location: seq.location,
        });
    }

    layers
}

use crate::stencil::{ThinkMode, TriggerRegistry};
use crossbeam::channel::Sender;
use std::sync::Arc;

/// One code scope to ingest in parallel — the `(user, assistant)` pair a single
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

    /// Current section-tree selection (e.g. the composer dials).  Set from each
    /// turn's [`TurnOptions::selection`] and used by every projection until the
    /// next turn changes it.  Empty = the schema's authored defaults.
    selection: SelectionState,

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

    /// Static model properties captured at engine construction.
    model_core: ModelCoreProperties,

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
}

/// The dialect's framing markers (`<|im_start|>system`, `<|im_end|>`, …) — the
/// glue the projection assembler wraps around the system prompt and each turn.
/// Returned by [`Sequence::glue_markers`] so the projection panel can render the
/// framing verbatim.
#[derive(Debug, Clone)]
pub struct GlueMarkers {
    pub system_start: String,
    pub system_end: String,
    pub user_start: String,
    pub user_end: String,
    pub assistant_start: String,
    pub assistant_end: String,
    /// The `/no_think` soft-switch, emitted as live glue right after `user_start`
    /// on a suppressed (effort-off) turn — see the scheduler's `no_think_current`
    /// segment. Empty for non-thinking dialects. The panel renders it so its view
    /// matches the actual prefill.
    pub no_think: String,
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
        model_core: ModelCoreProperties,
        substrate: Conversation,
        section_progress: Option<&dyn Fn(u64, u64)>,
        // When `false`, skip the [`SchedulerRequest::PrimingProjection`] slot
        // pre-warm (a synchronous scheduler round-trip). Priming is a
        // first-turn latency optimization — it pre-injects the empty-collection
        // layout so the first `submit_turn` skips `apply_projection`. Callers
        // that create MANY sequences in a pipelined burst (calibration) pass
        // `false`: the per-sequence round-trip would serialise the burst back
        // to one-create-per-wave-latency, defeating the batching, and the
        // per-turn `apply_projection` at submit materialises the projection
        // just the same (it must anyway, to select the pinned tool).
        prime_slot: bool,
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
            selection: SelectionState::default(),
            pending_user: None,
            turn_counter: initial_turn_counter,
            config,
            turn_in_flight: false,
            freed: false,
            current_blocks: BlockCount(0),
            chunk_size,
            model_core,
            projection: Arc::new(projection),
            substrate,
            target,
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
        let layer_items: Vec<SystemPromptItem> =
            conv.projection.schema().system_prompt.items.clone();

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
                // Tree nodes are sealed content; the ingest loop seals every
                // option's branch variants — count content × variant-count over
                // every option of every node.
                SystemPromptItem::SectionTree(t) => t
                    .nodes
                    .iter()
                    .map(|n| {
                        let opt_bytes: u64 = n
                            .options
                            .iter()
                            .map(|o| o.content.len() as u64 * o.variants.len() as u64)
                            .sum();
                        // A collection node seals each member ×branch.
                        let coll_bytes: u64 = n.collection.as_ref().map_or(0, |tc| {
                            tc.collection
                                .sections
                                .iter()
                                .zip(tc.variants.iter())
                                .map(|(s, vs)| s.content.len() as u64 * vs.len() as u64)
                                .sum()
                        });
                        opt_bytes + coll_bytes
                    })
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
                SystemPromptItem::SectionTree(tree) => {
                    // Seal EVERY option's branch variants up front — the full
                    // cross-product prefill that makes selector switching free.
                    // Each variant is sealed against `[outer content prefix |
                    // that branch's in-tree ancestor variants]`.  Sealing in
                    // (node order, option order, variant order) is
                    // dependency-safe: a variant's in-tree prefix references only
                    // earlier nodes' variants, which are already sealed.
                    for node in &tree.nodes {
                        // A prefix-transparent embedded collection: seal each
                        // member ONCE PER ancestor branch (the ×outer-selector
                        // fan-out), batching all members of a branch under that
                        // branch's prefix with the aggressive collection quantize
                        // policy.  It never extends `linear_prefix` — nodes below
                        // anchor on the next mandatory node, not these members.
                        if let Some(tc) = &node.collection {
                            let members = &tc.collection.sections;
                            // The branch set is identical across members; read it
                            // off member 0's per-branch variant list.
                            if let Some(branch_list) = tc.variants.first() {
                                for (bi, bvar) in branch_list.iter().enumerate() {
                                    let mut prefix = linear_prefix.clone();
                                    prefix.extend_from_slice(&bvar.in_tree_prefix);
                                    let batch: Vec<(SectionId, &str)> = members
                                        .iter()
                                        .enumerate()
                                        .map(|(mi, sec)| {
                                            (tc.variants[mi][bi].id, sec.content.as_str())
                                        })
                                        .collect();
                                    if !batch.is_empty() {
                                        let done_bytes_ref = &mut done_bytes;
                                        let report_ref = &report;
                                        conv.insert_section_collection_with_progress(
                                            &batch,
                                            &prefix,
                                            true,
                                            |_sid, content_len| {
                                                *done_bytes_ref += content_len as u64;
                                                report_ref(*done_bytes_ref);
                                            },
                                        )?;
                                        // Offload-as-we-go: quantize + persist +
                                        // evict THIS branch's members before the
                                        // next branch prefills, so the native
                                        // catalog never exceeds one branch.  Safe
                                        // because members are prefix-transparent —
                                        // nothing attends back over them in the
                                        // build — and the per-turn elevate reloads
                                        // the projection's top-k on demand.
                                        let (tx, rx) = crossbeam::channel::bounded(1);
                                        conv.scheduler_tx
                                            .send(SchedulerRequest::OffloadCollectionMembers {
                                                conversation: conv.substrate.clone(),
                                                response_tx: tx,
                                            })
                                            .map_err(|_| ConversationError::SchedulerGone)?;
                                        rx.recv()
                                            .map_err(|_| ConversationError::SchedulerGone)??;
                                    }
                                }
                            }
                            continue;
                        }
                        for option in &node.options {
                            for v in &option.variants {
                                let mut prefix = linear_prefix.clone();
                                prefix.extend_from_slice(&v.in_tree_prefix);
                                conv.insert_section_with_prefix(
                                    v.id,
                                    option.content.as_str(),
                                    &prefix,
                                )?;
                                done_bytes += option.content.len() as u64;
                                report(done_bytes);
                            }
                        }
                    }
                    // Sections declared after the tree (and the priming
                    // projection) attend to the default branch — the fan-out is
                    // bounded to the tree itself.
                    for id in &tree.default_present_ids {
                        linear_prefix.push(*id);
                        fixed_prefix.push(*id);
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
        if prime_slot && !fixed_prefix.is_empty() {
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
    /// [`Substrate::set_section_sealed`].  The section's provenance
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
    /// scheduler_tx / tokenizer / chunk_size off the
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
                // Skip collection members of the shared prompt — they don't
                // advance the content chain.
                if schema.system_prompt.is_collection_member(pid) {
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
                if self
                    .substrate
                    .section_stream_layout(stream_id, n_layers)
                    .is_some()
                {
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
        for s in self.projection.schema().system_prompt.all_sections() {
            if s.id == section_id {
                return s.name.clone();
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

        // Adopt this turn's section-tree selection (the composer dials).  It
        // becomes the conversation's current selection and drives every
        // projection — initial prefill and decode reprojection — until the next
        // turn changes it.
        self.selection = options.selection.clone();

        // Pin the system prompt into the substrate (idempotent).
        // Subsequent SubmitTurn calls re-inject it onto the slot via
        // `apply_projection`; the upload cache amortises after the
        // first turn.

        // The persisted turn shape is
        //     user_message + user_end + assistant_start + decoded_body
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
        //
        // Thinking suppression is PER-TURN, driven by the composer effort dial.
        // Qwen3's `/no_think` soft-switch is only honoured from the user turn (not
        // the system prompt), so when suppressed it is emitted as live GLUE right
        // before this user message by the scheduler (`no_think_current`, gated on
        // the same selector — see `reproject` + `BoundaryMarkers`), never baked
        // into the turn.  The assistant header itself is never modified: a
        // suppressed turn decodes its own empty `<think></think>`, a thinking turn
        // opens its own `<think>`.
        let assistant_start_marker = self.config.dialect.assistant_start;
        // Optional assistant prefill: text seeded as the start of the response so
        // the decode is forced to continue from it (e.g. `<tool_call>` commits to
        // the tool-call grammar). It is pinned into the prefill grid and sealed
        // as assistant content (see `assistant_content_start` below); the model
        // decodes the continuation. Empty when unset — the path is then identical
        // to an ordinary turn.
        let assistant_prefill = options.assistant_prefill.as_deref().unwrap_or("");
        let assistant_head = format!(
            "{}{}{}",
            user_message, self.config.dialect.user_end, assistant_start_marker,
        );
        let formatted = format!("{assistant_head}{assistant_prefill}");
        let prefill_tokens = self.tokenize(&formatted)?;

        // No post-decode tail in the turn layout.  The model's
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

        tracing::trace!(
            max_decode_tokens = max_tokens,
            forced_eos_after = sampling.forced_eos_after,
            graceful_eos_after = sampling.graceful_eos_after,
            eos_ramp_start = sampling.eos_ramp_start,
            "turn submitted"
        );

        let reprojection = self.build_reprojection_policy();
        // Content boundaries inside the sealed grid.  The prefill grid is
        // `[user_msg][user_end][assistant_start]` — the leading `user_start` (and,
        // when suppressing, `/no_think`) are live `Generated` glue segments emitted
        // by the scheduler, NOT part of the prefill — so the user body spans
        // `[0, len(user_msg))`.
        let user_content_start = 0;
        let user_content_end = self.tokenize(user_message)?.len();
        // Assistant content begins at the `assistant_start` boundary — before any
        // prefilled prefix — so the prefix's K/V seals as part of the assistant
        // turn. With no prefill this is exactly `prefill_tokens.len()` (the head
        // IS the whole prefill), preserving the ordinary-turn layout byte-for-byte.
        let assistant_content_start = if assistant_prefill.is_empty() {
            prefill_tokens.len()
        } else {
            self.tokenize(&assistant_head)?.len()
        };
        // Clamp to the prefill length and force monotonic so a tokenizer that
        // merges across a join can never invert the windows at seal time.
        let total = prefill_tokens.len();
        let user_content_start = (user_content_start.min(total)) as u32;
        let user_content_end =
            (user_content_end.min(total).max(user_content_start as usize)) as u32;
        let assistant_content_start = (assistant_content_start
            .min(total)
            .max(user_content_end as usize)) as u32;
        // Record whether the composer's `/no_think` dial is active for this
        // turn, so the projection re-injects the soft-switch into this turn's
        // user opener when it is later re-rendered as history.
        let no_think = matches!(
            options
                .selection
                .optional(crate::projection::NO_THINK_SELECTOR),
            Some(crate::projection::OptionalState::Present)
        );
        let handle = self.submit_prefill_unit(
            self.id,
            Some(self.projection_inputs()),
            formatted,
            prefill_tokens,
            user_message.to_string(),
            user_content_start,
            user_content_end,
            assistant_content_start,
            no_think,
            options.tags.clone(),
            // Decode path: reprojection fires from the decode loop, not staged
            // prefill offsets.
            Vec::new(),
            // Decode path: the assistant half is produced by the model, so the
            // seal stores its decoded text — nothing to pre-supply here.
            String::new(),
            post_decode_tokens,
            max_tokens,
            sampling,
            reprojection,
            options.triggers,
        )?;
        self.turn_in_flight = true;
        Ok(handle)
    }

    /// Submit a calibration turn whose assistant trajectory is **supplied
    /// verbatim** and prefilled in a single batched forward pass instead of
    /// decoded — the fast path that reproduces a decode-built calibration turn's
    /// KV (and thus its seal-captured wide-Q signature) without the per-token
    /// decode cost.
    ///
    /// `assistant_trajectory` is the model's real reply body (`<think>…</think>`
    /// + `<tool_call>…`) exactly as a decode produced it — NOT think-suppressed
    /// like [`insert_turn_tagged`](Self::insert_turn_tagged). The grid is
    /// `user_message + user_end + assistant_start + assistant_trajectory`, byte-
    /// identical to the decode's persisted turn, so re-tokenizing it reproduces
    /// the same tokens (modulo rare non-canonical BPE runs). `max_decode_tokens =
    /// 0`, so the scheduler prefills then seals — `perform_seal_and_write`
    /// captures the whole turn's wide-Q from KV exactly as for a decoded turn.
    ///
    /// `selection` pins the tool (via `CALIB_TOOL_SELECTOR`) so the projection
    /// marks exactly that catalog member selected; `tags` scope the belief
    /// gallery. Returns the streaming handle — drain it to `Done` and seal it
    /// with [`finish_turn`](Self::finish_turn) exactly like a decoded turn.
    pub fn submit_prefilled_turn(
        &mut self,
        user_message: &str,
        assistant_trajectory: &str,
        projection_marker: &str,
        selection: SelectionState,
        tags: Vec<String>,
    ) -> crate::Result<TurnHandle> {
        if self.turn_in_flight {
            return Err(ConversationError::TurnInFlight {
                sequence_id: self.id,
            });
        }
        self.selection = selection;

        // Thinking turn: no forced `no_think_block` — the trajectory carries its
        // own `<think>` in the body, matching the decode grid exactly.
        let assistant_start_marker = self.config.dialect.assistant_start;
        let assistant_head = format!(
            "{}{}{}",
            user_message, self.config.dialect.user_end, assistant_start_marker,
        );
        // The trajectory carries `projection_marker`s at the points a real decode
        // reprojected. Strip them from the prefilled text (they are not model
        // tokens), and record each one's token offset so the staged prefill wave
        // fires a projection there — reproducing the decode's per-segment
        // projection sequence. The markers are token-aligned by construction (the
        // exporter verifies the split round-trips), so tokenizing the cumulative
        // prefix up to each marker yields its exact grid offset.
        let clean_trajectory = assistant_trajectory.replace(projection_marker, "");
        let formatted = format!("{assistant_head}{clean_trajectory}");
        let prefill_tokens = self.tokenize(&formatted)?;
        // No post-decode tail — `assistant_end` is a live `Generated` segment,
        // same as a decoded turn.
        let post_decode_tokens = TokenBuffer::new();

        // Record the pending user turn (text + raw tokens for the tree).
        let user_tokens = self.tokenize(user_message)?;
        self.pending_user = Some(TokenizedText::new(user_message, user_tokens));

        // Content boundaries: user body `[0, len(user_msg))`; the supplied
        // assistant trajectory begins right after the head. Clamp/monotonise so a
        // tokenizer that merges across a join can never invert the windows.
        let total = prefill_tokens.len();
        let user_content_start = 0u32;
        let user_content_end = self.tokenize(user_message)?.len().min(total) as u32;
        let assistant_content_start = self
            .tokenize(&assistant_head)?
            .len()
            .min(total)
            .max(user_content_end as usize) as u32;

        // Grid-token offset of each projection marker: tokenize `head + trajectory
        // up to the marker`. `split` yields one segment per marker plus a trailing
        // remainder, so the boundaries between segments are exactly the marker
        // positions. Two markers are dropped from the wave's emit list: the
        // initial one at generation start (index 0 — the handler already applied
        // that projection, and its span is carried by the next event) and the seal
        // marker flush with the trajectory end (the final projection is appended at
        // seal). The intermediate reprojections remain.
        let segments: Vec<&str> = assistant_trajectory.split(projection_marker).collect();
        let mut projection_offsets: Vec<u32> = Vec::new();
        if segments.len() > 1 {
            let end = prefill_tokens.len() as u32;
            let mut prefix = assistant_head.clone();
            for (i, seg) in segments[..segments.len() - 1].iter().enumerate() {
                prefix.push_str(seg);
                let off = self.tokenize(&prefix)?.len() as u32;
                if i > 0 && off < end {
                    projection_offsets.push(off);
                }
            }
        }

        let handle = self.submit_prefill_unit(
            self.id,
            Some(self.projection_inputs()),
            formatted,
            prefill_tokens,
            user_message.to_string(),
            user_content_start,
            user_content_end,
            assistant_content_start,
            // Thinking calibration turn: the trajectory carries its own `<think>`.
            false,
            tags,
            projection_offsets,
            // Prefill path: the assistant half is supplied — stored verbatim
            // (markers stripped) as the turn's assistant_text.
            clean_trajectory,
            post_decode_tokens,
            // No decode — prefill then seal.
            0,
            self.config.sampling.clone(),
            // No reprojection policy: there is no decode loop to trigger it.
            None,
            // No tool stencils on a calibration prefill.
            Arc::new(TriggerRegistry::new()),
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
    #[allow(clippy::too_many_arguments)]
    fn submit_prefill_unit(
        &self,
        sequence_id: SequenceId,
        projection_inputs: Option<ProjectionInputs>,
        prefill_text: String,
        prefill_tokens: TokenBuffer,
        user_text: String,
        user_content_start: u32,
        user_content_end: u32,
        assistant_content_start: u32,
        no_think: bool,
        tags: Vec<String>,
        projection_offsets: Vec<u32>,
        prefill_assistant_text: String,
        post_decode_tokens: TokenBuffer,
        max_decode_tokens: usize,
        sampling: SamplingConfig,
        reprojection: Option<ReprojectionPolicy>,
        triggers: Arc<TriggerRegistry>,
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
                user_content_start,
                user_content_end,
                assistant_content_start,
                no_think,
                tags,
                projection_offsets,
                prefill_assistant_text,
                post_decode_tokens,
                max_decode_tokens,
                sampling,
                event_tx,
                reprojection,
                disable_reprojection,
                triggers,
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
            selection: self.selection.clone(),
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
        self.insert_turn_tagged(user_message, assistant_text, Vec::new())
    }

    /// [`insert_turn`](Self::insert_turn) with gather-scope tags — used to seed
    /// tagged calibration turns (e.g. `["tool"]`) that a projection policy's
    /// `tags:` filter scopes its provenance gallery to.
    pub fn insert_turn_tagged(
        &mut self,
        user_message: &str,
        assistant_text: &str,
        tags: Vec<String>,
    ) -> crate::Result<()> {
        self.insert_turn_inner(user_message, assistant_text, tags)?;
        Ok(())
    }

    /// [`insert_turn_tagged`](Self::insert_turn_tagged) + staged provenance
    /// linkage for ingest turns (repo_map clusters, code_read scopes): after
    /// the turn seals, synthesizes two [`ProjectionEvent`]s — one for the
    /// user half (`start_token: 0`), one for the assistant half — whose
    /// `selection.turns` reference the turn itself and its immediate
    /// predecessor by `(timeline, index)`, and persists them keyed to the
    /// turn's stream id. Together with the turn's seal-time wide-Q signature
    /// this gives a later provenance scan the resolvable chain
    /// sig hit → event → turn.
    ///
    /// Returns the number of tokens prefilled (the full formatted grid), which
    /// the prefill-only ingest paths (repo map, code read) surface as the "tokens
    /// ingested" metric.
    pub fn insert_turn_staged(
        &mut self,
        user_message: &str,
        assistant_text: &str,
        tags: Vec<String>,
    ) -> crate::Result<usize> {
        let (assistant_content_start, turn_index, tokens) =
            self.insert_turn_inner(user_message, assistant_text, tags)?;
        let Some(idx) = turn_index else {
            // No substrate seal (no registered target) — nothing to key
            // events to; the turn itself was still prefilled.
            return Ok(tokens);
        };
        self.persist_staged_ingest_events(idx, assistant_content_start, 0.0, &[])?;
        Ok(tokens)
    }

    /// Like [`Self::insert_turn_staged`], but `assistant_with_seams` carries
    /// `seam_marker`s at structural boundaries (e.g. subdirectory headers in a
    /// repo_map cluster listing). Each marker becomes a **self-referencing**
    /// projection event, so the belief scan scores the interval between seams as an
    /// independent retrieval sub-window that resolves back to this turn — a query
    /// matching one region of the listing surfaces the whole cluster without
    /// diluting against the rest. The markers are stripped before prefill (they are
    /// not model tokens); the wide-Q sig is captured over the clean text, and the
    /// seam offsets index it directly (the sig is dense, one entry per grid token).
    pub fn insert_turn_staged_windowed(
        &mut self,
        user_message: &str,
        assistant_with_seams: &str,
        seam_marker: &str,
        tags: Vec<String>,
    ) -> crate::Result<()> {
        let segments: Vec<&str> = assistant_with_seams.split(seam_marker).collect();
        let clean_assistant = segments.concat();
        let (assistant_content_start, turn_index, _tokens) =
            self.insert_turn_inner(user_message, &clean_assistant, tags)?;
        let Some(idx) = turn_index else {
            return Ok(());
        };
        // Grid-token offset of each seam: tokenize the grid prefix up to it in the
        // SAME format `insert_turn_inner` seals, so the offset indexes the sig.
        let mut seams: Vec<u32> = Vec::new();
        if segments.len() > 1 {
            let mut prefix = format!(
                "{}{}{}{}",
                self.config.dialect.no_think,
                user_message,
                self.config.dialect.user_end,
                self.config.dialect.assistant_start,
            );
            for seg in &segments[..segments.len() - 1] {
                prefix.push_str(seg);
                seams.push(self.tokenize(&prefix)?.len() as u32);
            }
        }
        self.persist_staged_ingest_events(idx, assistant_content_start, 0.0, &seams)
    }

    /// Shared body of [`insert_turn_tagged`] / [`insert_turn_staged`]: format,
    /// tokenize, prefill through the shared projection path, drain to `Done`,
    /// finalize. Returns the assistant half's grid-token start and the sealed
    /// substrate `TurnIndex` (from the seal result — never derived by
    /// counting, which races the async summariser).
    fn insert_turn_inner(
        &mut self,
        user_message: &str,
        assistant_text: &str,
        tags: Vec<String>,
    ) -> crate::Result<(u32, Option<u32>, usize)> {
        if self.turn_in_flight {
            return Err(ConversationError::TurnInFlight {
                sequence_id: self.id,
            });
        }

        // Format the full exchange (user + assistant_start prefix +
        // assistant_text) and tokenize as a single prefill payload.
        //
        // Prefilled content turns (repo_map, tool ingests, …) carry SUPPLIED
        // content — they never run a decoded reasoning pass — so there is nothing
        // to suppress at ingest. The `/no_think` soft-switch is NOT baked into the
        // grid: like `submit_turn`, it is a live `Generated` glue segment the
        // projection re-emits around the sealed turn on every future projection
        // (gated on the turn's `no_think()` flag, set below). Baking it here — as
        // this path used to — sealed the switch into the turn K/V AND left the
        // assembler re-emitting a second one, so every reconstructed turn opened
        // `[user_start][/no_think][/no_think][user]…` (a doubled soft-switch, a
        // model-degrading malformed opener). The role markers follow the same
        // rule: `user_start` / `assistant_end` are stripped from the persisted
        // bytes and re-emitted as glue; only the intra-turn `user_end` /
        // `assistant_start` stay baked.
        // Format: {user}{user_end}{assistant_start}{assistant_text}. The assistant
        // role-end comes through `post_decode_tokens` — no decode here, so the EOS
        // isn't emitted by the model; the projection's live `Generated` segments
        // supply the surrounding `user_start` / `/no_think` / `assistant_end`.
        let formatted = format!(
            "{}{}{}{}",
            user_message,
            self.config.dialect.user_end,
            self.config.dialect.assistant_start,
            assistant_text,
        );
        let prefill_tokens = self.tokenize(&formatted)?;
        let post_decode_tokens = TokenBuffer::new();

        // Content boundaries inside the sealed grid. The prefill grid is
        // `[user_msg][user_end][assistant_start][assistant_text]` (no leading
        // `user_start`, no baked `/no_think` — both are live `Generated`
        // segments). The user body spans `[0, len(user_msg))`; the assistant
        // content begins after the `[user_end][assistant_start]` markers. Tokenise
        // each prefix against the SAME strings the prefill is built from so the
        // indices land on the real grid; clamp/monotonise so a tokenizer that
        // merges across a join can never invert the windows.
        let assistant_start_marker = self.config.dialect.assistant_start;
        let user_content_start = 0usize;
        let user_content_end = self.tokenize(user_message)?.len();
        let assistant_content_start = self
            .tokenize(&format!(
                "{}{}{}",
                user_message, self.config.dialect.user_end, assistant_start_marker,
            ))?
            .len();
        let total = prefill_tokens.len();
        let user_content_start = (user_content_start.min(total)) as u32;
        let user_content_end =
            (user_content_end.min(total).max(user_content_start as usize)) as u32;
        let assistant_content_start = (assistant_content_start
            .min(total)
            .max(user_content_end as usize)) as u32;

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
            user_content_start,
            user_content_end,
            assistant_content_start,
            // `no_think()` = true: the turn is a no-reasoning content turn, so the
            // projection re-emits `/no_think` as glue before it (NOT baked into the
            // grid — see above). Exactly one soft-switch on reconstruction.
            true,
            tags,
            // Structured ingest: one projection, no staged prefill segments.
            Vec::new(),
            // Prefill path: the assistant half is supplied (not decoded), so
            // hand it through to be stored verbatim as the turn's assistant_text.
            assistant_text.to_string(),
            post_decode_tokens,
            0,
            self.config.sampling.clone(),
            None,
            // A no-decode insert never samples, so no stencil can fire.
            Arc::new(TriggerRegistry::new()),
        )?;

        // Drain events synchronously to Done.  The handle's event_rx
        // receives Prefill (text echo) and Done; we just need to
        // observe Done so we know the parent's KV is fully populated
        // before we register the turn.
        let response = handle.wait()?;
        let turn_index = response.seal.as_ref().and_then(|s| s.turn_index);

        // Run the same post-Done finalize as a regular turn.
        self.finalize_turn_post_done(user_tt, asst_tt, response.seal.as_ref())?;
        Ok((assistant_content_start, turn_index, total))
    }

    /// Ingest one code scope as a TOOL ROUND-TRIP of two coupled turns — the
    /// correct shape for a simulated `read_file` call. Recording it as two turns
    /// (not one baked four-segment exchange) is what keeps the inter-turn role
    /// seams as REGENERATED live glue on every projection: a seam baked into K/V
    /// carries a hidden state computed against its ingest-time prefix (an empty
    /// scratch slot) and goes stale when the scope is re-injected at a different
    /// position mid-dialogue — the exact degradation provenance-selected attention
    /// forbids. Mirrors the dialogue's own tool-call recording via
    /// [`couple_turn`](Self::couple_turn):
    ///
    /// - Turn A (call): `user(request)` → `assistant(<tool_call>…)`, prefilled.
    /// - Turn B (response): `user(<tool_response>…code…)` → `assistant(summary)`,
    ///   DECODED under `/no_think` (short) — so `<think>` is stripped by the
    ///   normal clean re-prefill and `/no_think` rides as live glue, never baked.
    ///
    /// The two turns are coupled, so provenance selecting either injects both,
    /// adjacent, with the seam regenerated between them. Serial by construction:
    /// Turn B's summary must decode with Turn A in its projected prefix. Returns
    /// tokens ingested (call prefill + decoded summary).
    pub fn ingest_scope_roundtrip(
        &mut self,
        call_user: &str,
        call_assistant: &str,
        response_user: &str,
        tags: Vec<String>,
        max_summary_tokens: usize,
    ) -> crate::Result<usize> {
        let (call_idx, _resp_idx, tokens) = self.ingest_scope_roundtrip_indices(
            call_user,
            call_assistant,
            response_user,
            tags,
            max_summary_tokens,
        )?;
        // Serial path: couple the pair here (the parallel splice couples on the
        // file timeline instead, after adopting both turns — see `adopt_turn`).
        self.couple_turn(call_idx)?;
        Ok(tokens)
    }

    /// Core of [`Self::ingest_scope_roundtrip`] that returns the sealed
    /// `(call_turn_index, response_turn_index, tokens_ingested)` **without**
    /// coupling — so the parallel code_read path can run this on a per-scope FORK
    /// and then splice both turns onto the file timeline (coupling there). The
    /// two-turn shape and its live-glue seams are identical to the serial path.
    pub fn ingest_scope_roundtrip_indices(
        &mut self,
        call_user: &str,
        call_assistant: &str,
        response_user: &str,
        tags: Vec<String>,
        max_summary_tokens: usize,
    ) -> crate::Result<(u32, u32, usize)> {
        // A scope round-trip is a SUMMARIZATION task, not the dialogue agent. Drive
        // the shared system prompt into its summarizer mode via selection — the
        // generic, per-mode section-toggling design rather than a bespoke per-layer
        // prompt string:
        //   - tools ON, force-pinned to `file_read`: the round-trip PREFILLS a
        //     `read_file` tool_call and its tool_response, so the projection must
        //     present a coherent tool context or the model can't connect the
        //     prefill to any capability and degrades (refusals, off-language,
        //     hallucinated tool chatter). Enable the tool block and force-select
        //     exactly the one tool the prefill uses (see `FORCE_TOOL_SELECTOR`) —
        //     one present, coherent tool, no belief-driven catalog noise.
        //   - `persona = summarize`: swaps the "You are Zen, pair programming…"
        //     dialogue frame for the terse code-summarizer frame (content-provided,
        //     English, summary-only) — the fix for the reasoning/refusal/off-language
        //     summaries the conversational persona produced.
        //   - `response_length = terse`: the default `standard` length section says
        //     "a short paragraph or two", which fights the two-sentence goal.
        self.selection.set_optional(
            crate::projection::TOOLS_ENABLED_SELECTOR,
            crate::projection::OptionalState::Present,
        );
        self.selection
            .select(crate::projection::FORCE_TOOL_SELECTOR, "file_read");
        self.selection.select("persona", "summarize");
        self.selection.select("response_length", "terse");
        //   - `summarize_examples = present`: stuff a few worked example turns
        //     (unrelated sample files) between the system prompt and this scope's
        //     turns, so the model imitates the exact request→summary shape — the
        //     strongest lever against reasoning/refusal/off-language drift.
        self.selection.set_optional(
            "summarize_examples",
            crate::projection::OptionalState::Present,
        );
        // Turn A — the call: prefill `[request][tool_call]` + staged provenance.
        let (call_acs, call_idx, call_tokens) =
            self.insert_turn_inner(call_user, call_assistant, tags.clone())?;
        let call_idx = call_idx.ok_or_else(|| {
            ConversationError::Channel("scope round-trip: call turn produced no index".into())
        })?;
        self.persist_staged_ingest_events(call_idx, call_acs, 0.0, &[])?;
        // Turn B — the response: submit the tool_response and DECODE the summary,
        // `/no_think` + short + low-temp nucleus. The decode sees Turn A (just recorded).
        //
        // `/no_think` alone does NOT reliably stop this hybrid 30B MoE from opening
        // a real `<think>` block: the empty block is *decoded*, not baked (see
        // `DecodeState::prefill_tokens` doc), so under sampling the model often opens
        // `<think>` and burns the whole `max_summary_tokens` budget on runaway —
        // frequently off-language (Chinese/Japanese) — reasoning, leaving a truncated
        // "thought" as the stored summary. Route the summary through the SAME
        // canonical think-close steering the dialogue path uses (`apply_think_mode`),
        // as `ThinkMode::Off`: with the short summary budget it collapses to a forced
        // empty block, so the budget goes to the summary, not the reasoning.
        let mut summary_sampling = SamplingConfig::compression();
        summary_sampling.apply_think_mode(ThinkMode::Off, &self.tokenizer, max_summary_tokens);
        let mut opts = TurnOptions {
            max_tokens: Some(max_summary_tokens),
            sampling: Some(summary_sampling),
            tags,
            ..Default::default()
        };
        opts.selection.set_optional(
            crate::projection::NO_THINK_SELECTOR,
            crate::projection::OptionalState::Present,
        );
        // Same tools-ON, single-tool pin as the conversation-level selection above:
        // the decode's own projection must also carry the coherent `file_read`
        // context the tool_response turn refers to.
        opts.selection.set_optional(
            crate::projection::TOOLS_ENABLED_SELECTOR,
            crate::projection::OptionalState::Present,
        );
        opts.selection
            .select(crate::projection::FORCE_TOOL_SELECTOR, "file_read");
        let handle = self.submit_turn_with_options(response_user, opts)?;
        let response = handle.wait()?;
        let resp_tokens = response.token_ids.len();
        let resp_idx = response
            .seal
            .as_ref()
            .and_then(|s| s.turn_index)
            .ok_or_else(|| {
                ConversationError::Channel(
                    "scope round-trip: response turn produced no index".into(),
                )
            })?;
        // Records Turn B + its staged provenance events (the decoded-ingest path).
        self.finish_turn_staged(handle, &response)?;
        Ok((call_idx, resp_idx, call_tokens + resp_tokens))
    }

    /// Fork this conversation onto a fresh timeline for one parallel scope
    /// ingest. The returned [`Sequence`] shares the substrate (so a later
    /// `adopt_turn` can reference its sealed K/V) but has its own scheduler slot
    /// + timeline, so scopes ingest concurrently without ordering conflicts.
    pub fn fork_scope(&self) -> crate::Result<Sequence> {
        // Do NOT mark the fork append-only / evict_when_cold: `adopt_turn` requires
        // the fork's turns to still be HOT at splice, so auto-evicting them would
        // race the splice ("source K/V not hot"). The fork's orphaned hot is freed
        // instead at `tombstone_timeline` (below), where the file timeline's cloned
        // chunk handles keep the shared KV alive.
        self.fork()
    }

    /// Splice a per-scope fork's two coupled turns onto THIS (file) timeline in
    /// order, then couple them — the ordered-merge step of parallel code_read.
    /// The fork's sealed K/V is referenced (not copied); its turns must still be
    /// HOT (splice right after the fork completes). Records at the next two file
    /// indices and returns them.
    pub fn splice_scope_turns(
        &self,
        fork_timeline: TimelineId,
        call_idx: u32,
        resp_idx: u32,
        tags: Vec<String>,
    ) -> crate::Result<(u32, u32)> {
        let file_tl = self.target.timeline;
        let new_call = self
            .substrate
            .adopt_turn(
                fork_timeline,
                TurnIndex(call_idx),
                file_tl,
                Role::Assistant,
                tags.clone(),
            )
            .map_err(ConversationError::Model)?;
        let new_resp = self
            .substrate
            .adopt_turn(
                fork_timeline,
                TurnIndex(resp_idx),
                file_tl,
                Role::Assistant,
                tags,
            )
            .map_err(ConversationError::Model)?;
        self.substrate
            .couple_turn(file_tl, new_call.0)
            .map_err(ConversationError::Model)?;
        // The fork's turns now live on the file timeline (their K/V shared by
        // reference); tombstone the fork so its orphaned timeline isn't reloaded
        // or scored. The shared chunks stay alive via the file timeline's clones.
        let _ = self.substrate.tombstone_timeline(fork_timeline);
        Ok((new_call.0, new_resp.0))
    }

    /// Tombstone a per-scope fork that will NOT be spliced onto the file
    /// timeline — its chunk failed before its ordered [`Self::splice_scope_turns`]
    /// ran, so its round-trip either errored or completed-but-unadopted.
    /// [`Sequence`]'s `Drop` only frees the scheduler slot; without this the
    /// fork's registered timeline and any sealed round-trip turns linger in the
    /// substrate as an orphaned, path-less "(untitled)" scope conversation.
    /// Mirrors the tombstone `splice_scope_turns` performs after adopting.
    pub fn tombstone_fork(&self, fork_timeline: TimelineId) {
        let _ = self.substrate.tombstone_timeline(fork_timeline);
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

    /// [`finish_turn`](Self::finish_turn) + staged provenance linkage for a
    /// DECODED ingest turn (the code_read per-file summary): synthesizes and
    /// persists the same two staged [`ProjectionEvent`]s as
    /// [`insert_turn_staged`](Self::insert_turn_staged), with the decode's
    /// wall-clock on the assistant-half event. The assistant half's grid
    /// start comes from the sealed turn's own [`TurnLayout`] rather than
    /// re-tokenizing.
    pub fn finish_turn_staged(
        &mut self,
        handle: TurnHandle,
        response: &TurnResponse,
    ) -> crate::Result<String> {
        let turn_index = response.seal.as_ref().and_then(|s| s.turn_index);
        let seconds = response.stats.decode_ms / 1000.0;
        let text = self.finish_turn(handle, response)?;
        if let Some(idx) = turn_index {
            let assistant_start = {
                let read = self.substrate.read();
                read.turn_layout(self.target.timeline, TurnIndex(idx))
                    .map(|l| l.assistant_content_start())
                    .unwrap_or(0)
            };
            self.persist_staged_ingest_events(idx, assistant_start, seconds, &[])?;
        }
        Ok(text)
    }

    /// Synthesize + persist the two staged [`ProjectionEvent`]s for the
    /// just-sealed ingest turn `turn_index`: one governing the user half
    /// (`start_token: 0`), one governing the assistant half. `selection`
    /// carries the layer's fixed system sections and — the mandatory
    /// provenance linkage — the turn itself plus its immediate predecessor as
    /// `(timeline, index)`-resolvable [`SelectedTurn`]s, keyed to the turn's
    /// stream id (the same key its wide-Q signature persists under).
    fn persist_staged_ingest_events(
        &self,
        turn_index: u32,
        assistant_content_start: u32,
        seconds: f64,
        seam_offsets: &[u32],
    ) -> crate::Result<()> {
        use crate::persistence::content_hash::turn_stream_id;
        use crate::persistence::streams::StreamDecl;
        use crate::projection::event::group_name_of;
        use crate::projection::{
            encode_events, staged_ingest_event, ProjectionEvent, SelectedTurn, SystemItem,
        };
        use crate::substrate::ContentResolver;
        use crate::summary_tree::TurnKind;

        let timeline = self.target.timeline;
        let schema = self.projection.schema();
        let layer = schema.layers.iter().find(|l| l.id == self.target.layer);
        let layer_name = layer.map(|l| l.name.clone()).unwrap_or_default();
        let group_name = group_name_of(schema, self.target.group)
            .unwrap_or_default()
            .to_string();

        let (system, turns) = {
            let read = self.substrate.read();
            // The trunk's fixed system sections — utility-layer system
            // prompts are fixed-only, so this is the complete composition.
            let mut system: Vec<SystemItem> = Vec::new();
            for item in &schema.system_prompt.items {
                if let SystemPromptItem::Section(s) = item {
                    let tokens = ContentResolver::section_token_count(&read, s.id) as u32;
                    system.push(SystemItem::Section {
                        name: s.name.clone(),
                        tokens,
                    });
                }
            }
            let count = crate::substrate::Substrate::turn_count(&read, timeline);
            let selected = |idx: u32| -> Option<SelectedTurn> {
                let t = TurnIndex(idx);
                if idx >= count {
                    return None;
                }
                let role = read
                    .stream_of(turn_stream_id(timeline.raw(), idx))
                    .and_then(|s| s.decl.as_ref())
                    .and_then(|d| match d {
                        StreamDecl::Turn(t) => Some(t.role),
                        _ => None,
                    })
                    .map(|r| match r {
                        0 => "system",
                        2 => "assistant",
                        _ => "user",
                    })
                    .unwrap_or("user")
                    .to_string();
                let kind = read
                    .tree_meta_of(timeline, t)
                    .map(|m| m.kind)
                    .unwrap_or(TurnKind::Normal);
                Some(SelectedTurn {
                    layer: layer_name.clone(),
                    group: group_name.clone(),
                    index: idx,
                    role,
                    tokens: read.turn_token_count_of(timeline, t) as u32,
                    kind,
                    reason: None,
                    timeline: Some(timeline.raw()),
                    selected: true,
                    score: 0.0,
                    qualified: false,
                })
            };
            let mut turns = Vec::with_capacity(2);
            if let Some(prev) = turn_index.checked_sub(1).and_then(selected) {
                turns.push(prev);
            }
            if let Some(own) = selected(turn_index) {
                turns.push(own);
            }
            (system, turns)
        };

        let mut events = vec![
            staged_ingest_event(0, 0.0, system.clone(), turns.clone()),
            staged_ingest_event(assistant_content_start, seconds, system, turns),
        ];
        // Self-referencing sub-window seams: one marker event per structural
        // boundary. The belief scan reads their `start_token`s and scores each
        // `[seam_i, seam_{i+1})` interval as a focused window that resolves back to
        // this turn (see `Substrate::score_belief_groups`). Empty ⇒ the turn scores
        // as one whole-turn window, unchanged.
        for &off in seam_offsets {
            events.push(ProjectionEvent {
                start_token: off,
                self_reference: true,
                ..Default::default()
            });
        }
        self.substrate
            .persist_projection_events(
                turn_stream_id(timeline.raw(), turn_index),
                &encode_events(&events),
            )
            .map_err(ConversationError::Model)?;
        Ok(())
    }

    /// Belief scores for the just-sealed turn, scored against each belief node
    /// (the target layer's collections AND every layer's belief-driven turn
    /// groups) using the turn's own persisted wide-Q signature as the probe. The
    /// post-turn counterpart of the scheduler's live reproject scan — same
    /// galleries + scorer, but the probe is the finished turn's stored signature
    /// rather than a live gather. Empty when the turn has no signature (nothing
    /// to score against).
    fn last_turn_belief_scores(&self) -> crate::substrate::ProjectionScores {
        use crate::provenance::decode_wide_sigs;
        let empty = crate::substrate::ProjectionScores::new();
        let timeline = self.target.timeline;
        let (probe, q_span) = {
            let read = self.substrate.read();
            let count = read.turn_count(timeline);
            if count == 0 {
                return empty;
            }
            let idx = TurnIndex(count - 1);
            let probe = match read
                .wide_q_sigs_blob(timeline, idx)
                .and_then(decode_wide_sigs)
            {
                // Same head+tail window the live reproject scans (see
                // `scheduler`'s `QUERY_HEAD_CHUNKS` / `max_probe_tokens`). Without
                // the cap this scan probes with the WHOLE turn, and a turn can be
                // arbitrarily large — a single `file_list` result ran to 5.7k
                // tokens, taking ~7s of scan and scoring every tool in the catalog
                // nonzero (the listing happened to name every tool's own
                // definition file), which then carried into the next turn's
                // opening belief.
                Some(p) => cap_probe_window(p, self.config.reproject_max_probe_tokens),
                None => return empty,
            };
            // Concept F: the sealed turn's question window is its persisted
            // user span — the sig grid is 1:1 with the real-KV layout.
            let q_span = read.user_sig_span(timeline, idx);
            (probe, q_span)
        };
        let probe_q = q_span.and_then(|r| probe.get(r)).filter(|q| !q.is_empty());
        let (scores, _) = self
            .substrate
            // observe = true: the seal scan is the once-per-turn learning
            // point for the score-normalization hit levels. No arena here (the
            // scheduler owns it) → the CPU per-file scan; the hot reproject path
            // runs the paged GPU scan over the resident arena.
            .score_beliefs(
                self.projection.schema(),
                self.target,
                &probe,
                probe_q,
                true,
                None,
            );
        scores
    }

    /// Recompute the materialized projection for this conversation and pair it
    /// with the just-finished decode's throughput into a [`ProjectionEvent`].
    ///
    /// Call this immediately after [`finish_turn`](Self::finish_turn): the turn
    /// is sealed, so its wide-Q signature is persisted and serves as the belief
    /// probe. Scoring that stored signature against each collection's tag-scoped
    /// gallery gives the same per-section belief the live reproject scan
    /// produces, so the persisted event reflects what provenance selects — not a
    /// zero-score min-budget fill. Returns `None` for layers that don't reproject
    /// (`disable_reprojection` — utility/reference ingestion), where a projection
    /// event carries no meaning.
    ///
    /// The composition (system / section groups / turns), per-category token
    /// counts, substrate total, and decode throughput are all real; the only
    /// approximation is the probe — the recompute scores against the
    /// just-finished turn's whole signature rather than replaying each live
    /// decode-step Q vector.
    pub fn projection_event(&self, stats: &crate::stats::TurnStats) -> Option<ProjectionEvent> {
        if self.config.disable_reprojection {
            return None;
        }
        let scores = self.last_turn_belief_scores();
        let resolver = self.substrate.read_for_scored(self.target, &scores);
        let projection = self.projection.project_with_selection(
            self.target,
            &resolver,
            ProjectionMode::Decode,
            &self.selection,
        );
        let substrate_total = resolver.total_token_count(self.target.timeline) as u32;
        // A projection is a POINT on the timeline, not a span. This is the
        // post-seal projection: the turn has finished decoding and its answer is
        // now materialized, so the point sits at the FINAL generated position
        // (`tokens_generated`), governing the closing interval `[last_reproj, end]`.
        // `seconds` is the full decode elapsed — the wall-clock at this final point.
        let mut ev = from_projection_with_origins(
            &projection.segments,
            &projection.selection_origins,
            self.projection.schema(),
            &resolver,
            &projection.selection_scores,
            substrate_total,
            stats.tokens_generated as u32,
            stats.decode_ms / 1000.0,
        );
        // Materialized dialogue glue, from the SAME `assemble_pieces` decision the
        // engine injects from, so the persisted panel shows the real boundary
        // markers. Best-effort: if the markers can't be tokenised, leave it empty
        // and the panel reconstructs the framing.
        if let Ok(markers) = BoundaryMarkers::from_dialect(&self.config.dialect, |s| {
            self.tokenizer
                .encode(s, false)
                .map(|e| e.get_ids().to_vec())
        }) {
            ev.materialized = materialize_conversation(
                &projection.segments,
                &markers,
                &projection.selection_origins,
                &resolver,
                self.projection.schema(),
                |toks| self.tokenizer.decode(toks, false).unwrap_or_default(),
            );
        }
        Some(ev)
    }

    /// Snapshot the handles the interactive projection probe needs into a
    /// lock-free [`ProbeCtx`]. Cheap (Arc/handle clones), so the caller drops the
    /// `base_conv` mutex immediately and runs the GPU-bound [`ProbeCtx::probe`]
    /// unlocked — the probe (a full turn round-trip) must not hold the fork source
    /// every new conversation and several status reads contend on.
    pub fn probe_ctx(&self) -> ProbeCtx {
        ProbeCtx {
            scheduler_tx: self.scheduler_tx.clone(),
            substrate: self.substrate.clone(),
            projection: Arc::clone(&self.projection),
            target: self.target,
            tokenizer: Arc::clone(&self.tokenizer),
            selection: self.selection.clone(),
        }
    }

    /// `(section name, authored content)` for every system-prompt section in the
    /// schema (bare sections + every collection member). Backs the projection
    /// panel's expandable section text — fetched on demand, never stored in the
    /// projection event.
    pub fn section_contents(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for item in &self.projection.schema().system_prompt.items {
            match item {
                crate::projection::SystemPromptItem::Section(s) => {
                    out.push((s.name.clone(), s.content.clone()));
                }
                crate::projection::SystemPromptItem::Collection(c) => {
                    for s in &c.sections {
                        out.push((s.name.clone(), s.content.clone()));
                    }
                }
                crate::projection::SystemPromptItem::SectionTree(t) => {
                    // One entry per (node, option): name `node` or `node:option`
                    // for selector options, so the panel can resolve each.
                    for n in &t.nodes {
                        for o in &n.options {
                            let name = if n.options.len() > 1 {
                                format!("{}:{}", n.name, o.id)
                            } else {
                                n.name.clone()
                            };
                            out.push((name, o.content.clone()));
                        }
                        // An embedded collection node lists its members.
                        if let Some(tc) = &n.collection {
                            for s in &tc.collection.sections {
                                out.push((s.name.clone(), s.content.clone()));
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The conversation's default sampling config (with the model's resolved
    /// thinking / reflection-marker token IDs).  Callers clone this to derive a
    /// per-turn config — e.g. to set the dial's `segment_suppress_penalty` — and
    /// pass it back via `TurnOptions::sampling`.
    pub fn default_sampling(&self) -> SamplingConfig {
        self.config.sampling.clone()
    }

    /// The dialect's framing markers — the "glue" the assembler wraps around the
    /// system prompt and each turn (`BoundaryMarkers` tokenises exactly these).
    /// Surfaced verbatim so the projection panel can show the glue between
    /// sections/turns without re-tokenising or re-projecting.
    pub fn glue_markers(&self) -> GlueMarkers {
        let d = &self.config.dialect;
        GlueMarkers {
            system_start: d.system_start.to_string(),
            system_end: d.system_end.to_string(),
            user_start: d.user_start.to_string(),
            user_end: d.user_end.to_string(),
            assistant_start: d.assistant_start.to_string(),
            assistant_end: d.assistant_end.to_string(),
            no_think: d.no_think.to_string(),
        }
    }

    /// The YAML name of this conversation's target layer (e.g. `dialogue`) — the
    /// memory tier the dialogue exchange itself lives in. The panel prefixes the
    /// conversation messages with it, derived from the schema rather than guessed.
    pub fn target_layer_name(&self) -> String {
        self.projection
            .schema()
            .layers
            .iter()
            .find(|l| l.id == self.target.layer)
            .map(|l| l.name.clone())
            .unwrap_or_default()
    }

    /// Shared post-prefill turn-completion path.
    ///
    /// After a `SubmitTurn` request emits `Done` (whether decoded by
    /// the model or fully prefilled inline), both `finish_turn` and
    /// `insert_turn` need the same fold-down work:
    ///
    /// 1. Apply the scheduler's seal payload (cold store register +
    ///    sig-blocks counter advance + provenance scan probe).  The
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
            self.current_blocks = BlockCount(seal.block_count);
            // The scheduler already wrote the turn's sealed bytes,
            // per-block sig entries, role/text/token_ids, AND the
            // workspace persistence layer (when configured) into the
            // substrate before sending `Done` — see
            // `Conversation::record_turn`.  Persistence is per-
            // workspace now, not per-Sequence; nothing for this
            // method to forward.
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

    /// Swap the projection schema this sequence projects + reprojects with.
    ///
    /// The new builder must share the current one's section ids (e.g. a clone
    /// with one collection's selection overridden) — its sealed-section KV is
    /// reused as-is. Takes effect on the next submitted turn (the prefill
    /// projection and the reprojection policy both read this on submit).
    pub fn set_projection(&mut self, projection: Arc<Builder>) {
        self.projection = projection;
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
            selection: self.selection.clone(),
            pending_user: None,
            turn_counter: self.turn_counter,
            config: self.config.clone(),
            turn_in_flight: false,
            freed: false,
            current_blocks: self.current_blocks,
            chunk_size: self.chunk_size,
            model_core: self.model_core,
            projection: self.projection.clone(),
            // Forks share the same substrate (Arc clone) so cross-fork
            // history aggregation continues to work.
            substrate: self.substrate.clone(),
            target: fork_target,
            // Forks start with a fresh scanner state — scoring will refresh
            // on the next provenance scan.  No need to clone the parent's scores.
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
        // KV state and provenance score cache are cleared here.
        self.current_blocks = BlockCount(0);

        // Clear turn history (keeps system prompt, config, beliefs).
        self.tree.clear_turns();

        Ok(())
    }

    /// Abandon an in-flight turn whose stream ended without a `Done` (e.g. the
    /// scheduler shut down mid-decode, or a scheduler error). Clears the local
    /// in-flight state so the next [`Self::submit_turn`]/[`Self::reset`] is not
    /// permanently rejected by the `turn_in_flight` guard. Best-effort on the
    /// scheduler side — a gone scheduler is ignored, since the only caller is
    /// already tearing the turn down.
    pub fn abort_turn(&mut self) {
        // Best-effort scheduler-side view cleanup; ignore a gone scheduler.
        let (response_tx, response_rx) = crossbeam::channel::bounded(1);
        if self
            .scheduler_tx
            .send(SchedulerRequest::ResetSequence {
                sequence_id: self.id,
                response_tx,
            })
            .is_ok()
        {
            let _ = response_rx.recv();
        }

        // Clear local turn state unconditionally so the sequence is reusable.
        self.turn_in_flight = false;
        self.pending_user = None;
        self.current_blocks = BlockCount(0);
        self.tree.clear_turns();
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
    /// Recovered turn history for `timeline`. When `include_ghost_summaries` is
    /// false, the ghost summary turns the summariser appends to the timeline
    /// (`SummaryOfTurns` / `SummaryOfSummaries` tree nodes) are skipped — they
    /// exist for provenance/projection, not for the conversation view. Pass
    /// `true` for substrate-level views that legitimately surface them.
    /// Recover the conversation as `(role, text, no_think)` bubbles — one per
    /// non-empty half of each turn, in order.  `no_think` is the turn's recorded
    /// thinking-suppressed flag, set on the USER bubble so the GUI can re-render
    /// the `/no_think` soft-switch on prior turns exactly as the assembler does
    /// for the model (see `turn_no_think`); the assistant bubble carries `false`.
    pub fn recovered_history(
        &self,
        timeline: TimelineId,
        include_ghost_summaries: bool,
    ) -> Vec<(Role, String, bool)> {
        let read = self.substrate.read();
        let mut out: Vec<(Role, String, bool)> = Vec::new();
        for idx in read.turn_indices(timeline) {
            if !include_ghost_summaries
                && read
                    .tree_meta_of(timeline, idx)
                    .is_some_and(|m| m.kind.is_summary())
            {
                continue;
            }
            let user_text = read.user_text_of(timeline, idx);
            let assistant_text = read.assistant_text_of(timeline, idx);
            let no_think = read.turn_no_think(timeline, idx);
            if !user_text.is_empty() {
                out.push((Role::User, user_text, no_think));
            }
            if !assistant_text.is_empty() {
                out.push((Role::Assistant, assistant_text, false));
            }
        }
        out
    }

    /// The two verbatim halves — `(user_text, assistant_text)` — of turn `index`
    /// in `timeline`, read from the substrate. Returned UNFRAMED (no dialect
    /// markers): the caller places the glue around and between them. Both halves
    /// are stored verbatim (decoded for dialogue turns, supplied for prefill
    /// ingests like repo_map/code_reading; see `insert_turn` → the seal).
    ///
    /// `timeline` is the turn's resolved identity, stamped by the projection at
    /// selection time (`SelectedTurn::timeline`) — it is NEVER re-derived from a
    /// group here, because the shared substrate registers many conversations under
    /// one group and that re-derivation is non-deterministic. `None` if the turn
    /// isn't found or is entirely empty.
    pub fn resolve_turn_text(&self, timeline: TimelineId, index: u32) -> Option<(String, String)> {
        let read = self.substrate.read();
        let idx = TurnIndex(index);
        let user = read.user_text_of(timeline, idx);
        let assistant = read.assistant_text_of(timeline, idx);
        if user.is_empty() && assistant.is_empty() {
            return None;
        }
        Some((user, assistant))
    }

    /// The ENTIRE turn `index` in `timeline` as one continuous string: the full
    /// sealed token range — user content, the baked intra-turn boundary
    /// (`user_end` / `assistant_start`), and assistant content — decoded verbatim,
    /// exactly as it sits in the KV. `timeline` is the turn's resolved identity
    /// (`SelectedTurn::timeline`); see [`Self::resolve_turn_text`]. `None` if the
    /// turn isn't found or has no tokens.
    pub fn resolve_turn_full_text(&self, timeline: TimelineId, index: u32) -> Option<String> {
        let read = self.substrate.read();
        let ids = read.token_ids_of(timeline, TurnIndex(index));
        if ids.is_empty() {
            return None;
        }
        self.tokenizer.decode(&ids, false).ok()
    }

    /// The segment-vector [`TurnLayout`] for turn `index` in `timeline` — the
    /// complete, validated description of its K/V (user / thinking / assistant /
    /// boundary glue). Built at seal time and stored on the turn, so this is a
    /// direct fetch. `timeline` is the turn's resolved identity
    /// (`SelectedTurn::timeline`); see [`Self::resolve_turn_text`]. `None` if the
    /// turn isn't found.
    pub fn turn_layout(&self, timeline: TimelineId, index: u32) -> Option<TurnLayout> {
        let read = self.substrate.read();
        read.turn_layout(timeline, TurnIndex(index))
    }

    /// Couple `from_turn` — the sealed **call turn** — to the tool response about
    /// to be submitted as the next turn, the two halves of one tool round-trip.
    ///
    /// Call this **after** the tools have returned real output and **before**
    /// submitting the response turn. In that window the round-trip is certain:
    /// the call turn is sealed (so its index is final) and the response turn is
    /// guaranteed to follow. Writing it here — rather than at either turn's seal
    /// — is what makes the record authoritative (a call that is never executed,
    /// or yields nothing, simply never couples).
    ///
    /// `from_turn` must be the call turn's **own** index, captured from its seal
    /// (`TurnResponse::seal`). Do not infer it from the current turn count: the
    /// async summariser can seal a `SummaryOfTurns`/`SummaryOfSummaries` turn
    /// into the same index space between the call turn's seal and this call, so
    /// "the last turn" may be a summary node, not the call — and coupling that
    /// index is silently dropped when the forest is derived (it maps to no
    /// `Normal`), leaving the exchange unformed.
    pub fn couple_turn(&self, from_turn: u32) -> crate::Result<()> {
        self.substrate
            .couple_turn(self.target.timeline, from_turn)
            .map_err(ConversationError::Model)
    }

    /// Persist this conversation's projection-event timeline for the most
    /// recently sealed turn. Called by the caller after `finish_turn` with the
    /// events streamed during (and at the end of) that turn's decode; survives
    /// a daemon restart via the `ProjectionEvents` redo-log record.
    pub fn persist_projection_events(&self, events: &[ProjectionEvent]) -> crate::Result<()> {
        let timeline = self.target.timeline;
        let count = self.substrate.read().turn_count(timeline);
        if count == 0 {
            return Ok(());
        }
        let stream_id = crate::persistence::content_hash::turn_stream_id(timeline.raw(), count - 1);
        let payload = crate::projection::encode_events(events);
        self.substrate
            .persist_projection_events(stream_id, &payload)
            .map_err(ConversationError::Model)
    }

    /// Recovered projection-event timelines for a conversation, one `Vec` per
    /// turn in turn order (empty for turns that have none). Mirrors
    /// [`recovered_history`](Self::recovered_history); backs the GUI timeline
    /// replay after reload.
    pub fn recovered_projection_events(&self, timeline: TimelineId) -> Vec<Vec<ProjectionEvent>> {
        let read = self.substrate.read();
        let indices: Vec<TurnIndex> = read.turn_indices(timeline).collect();
        indices
            .into_iter()
            // One bucket per assistant bubble — filter to turns with a non-empty
            // assistant reply, and skip ghost summary turns, so this lines up
            // index-for-index with the assistant entries the conversation view
            // (`recovered_history(.., false)`) produces.
            .filter(|&idx| {
                !read
                    .tree_meta_of(timeline, idx)
                    .is_some_and(|m| m.kind.is_summary())
                    && !read.assistant_text_of(timeline, idx).is_empty()
            })
            .map(|idx| {
                read.projection_events_blob(timeline, idx)
                    .map(crate::projection::decode_events)
                    .unwrap_or_default()
            })
            .collect()
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
        Some(ReprojectionPolicy {
            target: self.target,
            projection: Arc::clone(&self.projection),
            selection: self.selection.clone(),
            substrate: self.substrate.clone(),
            every_n_tokens: n,
            max_probe_tokens: self.config.reproject_max_probe_tokens.max(1),
            trigger_token_ids: trigger_ids,
            tool_call_open_id: self.single_token_id("<tool_call>"),
            tool_call_close_id: self.single_token_id("</tool_call>"),
        })
    }

    /// The token id of `text` iff the tokenizer encodes it to exactly one token.
    /// `<tool_call>` / `</tool_call>` are registered as single added tokens in the
    /// Qwen3 tokenizer, so this pins the reprojection-suppression boundary; a
    /// tokenizer without them yields `None` and the gate is simply inactive.
    fn single_token_id(&self, text: &str) -> Option<u32> {
        let enc = self.tokenizer.encode(text, false).ok()?;
        match enc.get_ids() {
            [id] => Some(*id),
            _ => None,
        }
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

/// Pure block-range windowing for a bounded rolling-window ingest (design
/// `docs/unified_wave_inference_engine.md` §4.7): given the system-prompt end
/// block `sys_end`, the ascending per-turn start blocks `turn_starts` (one per
/// sealed turn), the current `total` block count, and `window_turns`, return the
/// parent block ranges the next prefill should attend — the system prompt plus
/// the most recent `window_turns` sealed turns.
///
/// `window_turns == 0` (unbounded) or fewer sealed turns than the window returns
/// the whole parent `[(0, total)]` — a no-op, byte-for-byte the unwindowed
/// behaviour. Split out from [`Conversation::windowed_ingest_ranges`] so it is
/// unit-testable without a live substrate.
pub(crate) fn windowed_ingest_ranges_impl(
    sys_end: usize,
    turn_starts: &[usize],
    total: usize,
    window_turns: usize,
) -> Vec<(usize, usize)> {
    let whole = || {
        if total == 0 {
            Vec::new()
        } else {
            vec![(0, total)]
        }
    };
    if window_turns == 0 || turn_starts.len() <= window_turns {
        return whole();
    }
    let keep_from = turn_starts[turn_starts.len() - window_turns];
    // If the window already reaches back into (or to) the system prompt, the two
    // pieces are contiguous — borrow the whole parent.
    if keep_from <= sys_end {
        return whole();
    }
    let mut ranges = Vec::with_capacity(2);
    if sys_end > 0 {
        ranges.push((0, sys_end));
    }
    ranges.push((keep_from, total));
    ranges
}

/// A lock-free snapshot of the handles the interactive projection probe needs
/// ([`Sequence::probe_ctx`]), so the probe — a full GPU turn round-trip — runs
/// WITHOUT holding the `base_conv` mutex, which is the fork source every new
/// conversation and several status reads contend on.
pub struct ProbeCtx {
    scheduler_tx: Sender<SchedulerRequest>,
    substrate: Conversation,
    projection: Arc<Builder>,
    target: ProjectionTarget,
    tokenizer: Arc<tokenizers::Tokenizer>,
    selection: SelectionState,
}

impl ProbeCtx {
    /// Project a typed `text` against the current substrate as if it were a new
    /// turn under this conversation's system prompt — the interactive
    /// "what would this query retrieve" probe. Warms the query under the full
    /// system prompt, scores its wide-Q read-only (`observe = false`), and returns
    /// the resulting [`ProjectionEvent`] (selected turns + sections with belief
    /// scores) plus the query token count. Persists NOTHING to the substrate.
    pub fn probe(&self, text: &str) -> crate::Result<(ProjectionEvent, usize)> {
        let probe = self.probe_wide_q(text)?;
        let query_tokens = probe.len();
        // The typed probe IS the question, so a separate Q-window would be
        // identical to the tail scan — `None` scans it once.
        let (scores, _) = self.substrate.score_beliefs(
            self.projection.schema(),
            self.target,
            &probe,
            None,
            false,
            None,
        );
        let resolver = self.substrate.read_for_scored(self.target, &scores);
        let projection = self.projection.project_with_selection(
            self.target,
            &resolver,
            ProjectionMode::Prefill,
            &self.selection,
        );
        let substrate_total = resolver.total_token_count(self.target.timeline) as u32;
        let ev = from_projection_with_origins(
            &projection.segments,
            &projection.selection_origins,
            self.projection.schema(),
            &resolver,
            &projection.selection_scores,
            substrate_total,
            0,
            0.0,
        );
        Ok((ev, query_tokens))
    }

    /// Capture the WARM folded wide-Q signature of `text` as if it were a new turn
    /// under the system prompt, WITHOUT writing anything to the substrate. An
    /// ephemeral projection slot materializes the full system prompt (warming the
    /// query's Q exactly like a real turn) and decodes one throwaway token so the
    /// completion path finalizes the KV and gathers the query wide-Q; the slot
    /// resolves to `SealAction::None`, so no turn is sealed and no belief learned.
    fn probe_wide_q(&self, text: &str) -> crate::Result<Vec<WideQSig>> {
        let query_tokens = match self.tokenizer.encode(text, false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(_) => return Ok(Vec::new()),
        };
        if query_tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Ephemeral slot bound to this conversation's target: it projects (warm)
        // but never seals.
        let slot = {
            let (tx, rx) = crossbeam::channel::bounded(1);
            self.scheduler_tx
                .send(SchedulerRequest::NewEphemeralSequence {
                    conversation: self.substrate.clone(),
                    target: self.target,
                    response_tx: tx,
                })
                .map_err(|_| ConversationError::SchedulerGone)?;
            rx.recv().map_err(|_| ConversationError::SchedulerGone)??
        };
        let free = |tx: &Sender<SchedulerRequest>| {
            let _ = tx.send(SchedulerRequest::FreeSequence { sequence_id: slot });
        };

        // Submit the query as a projected turn (full warm system prompt via
        // `apply_projection`), decoding one token so the completion path finalizes
        // the KV onto the slot and gathers the query's warm wide-Q.
        let (event_tx, event_rx) = crossbeam::channel::unbounded();
        if self
            .scheduler_tx
            .send(SchedulerRequest::SubmitTurn {
                sequence_id: slot,
                projection_inputs: Some(ProjectionInputs {
                    projection: Arc::clone(&self.projection),
                    selection: self.selection.clone(),
                }),
                prefill_tokens: TokenBuffer::from(query_tokens),
                prefill_text: String::new(),
                user_text: String::new(),
                user_content_start: 0,
                user_content_end: 0,
                assistant_content_start: 0,
                no_think: false,
                tags: Vec::new(),
                projection_offsets: Vec::new(),
                prefill_assistant_text: String::new(),
                post_decode_tokens: TokenBuffer::new(),
                max_decode_tokens: 1,
                sampling: SamplingConfig::argmax(),
                event_tx,
                reprojection: None,
                disable_reprojection: false,
                triggers: Arc::new(TriggerRegistry::new()),
            })
            .is_err()
        {
            free(&self.scheduler_tx);
            return Err(ConversationError::SchedulerGone);
        }

        // Drain to Done/Error. A channel closed before `Done` (the turn was aborted
        // without a terminal event) is a failure — surface it rather than returning
        // an empty window that reads as "query retrieved nothing".
        let mut err: Option<ConversationError> = None;
        let mut done = false;
        while let Ok(ev) = event_rx.recv() {
            match ev {
                TurnEvent::Done(_) => {
                    done = true;
                    break;
                }
                TurnEvent::Error(e) => {
                    err = Some(e);
                    break;
                }
                _ => {}
            }
        }
        if let Some(e) = err {
            free(&self.scheduler_tx);
            return Err(e);
        }
        if !done {
            free(&self.scheduler_tx);
            return Err(ConversationError::SchedulerGone);
        }

        // Drain the warm wide-Q the ephemeral turn stashed, then free.
        let (tx, rx) = crossbeam::channel::bounded(1);
        let sigs = if self
            .scheduler_tx
            .send(SchedulerRequest::ProbeWideSigs {
                sequence_id: slot,
                response_tx: tx,
            })
            .is_ok()
        {
            rx.recv()
                .map_err(|_| ConversationError::SchedulerGone)
                .and_then(|r| r)
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        free(&self.scheduler_tx);

        // Keep the WHOLE turn window `[query .. decoded]`, not just the query — the
        // discriminating belief signal for a tool lives in the DECODE-Q (the token
        // where the model commits to the call), not the query's prefill-Q, which
        // sits near the noise floor (the call↔definition domain gap). Dropping the
        // decoded tail collapsed scores ~10× (600 → 60).
        Ok(sigs)
    }
}

/// Cap a whole-turn probe to the same head + trailing window the live
/// reproject scan uses: the turn's first `HEAD` signatures (the user query —
/// the strongest intent signal in the gallery domain) followed by its most
/// recent `max_tail`.
///
/// A turn is not bounded by anything the engine controls — a tool result goes
/// into it verbatim — so an uncapped probe is both a cost and a correctness
/// problem: the scan is O(probe × gallery), and the tail of a large tool
/// payload is not a query, so scoring it dilutes the turn's actual intent
/// across whatever the payload happens to mention.
fn cap_probe_window(probe: Vec<WideQSig>, max_tail: usize) -> Vec<WideQSig> {
    /// Head signatures kept regardless of how far the tail window has slid —
    /// mirrors the scheduler's `QUERY_HEAD_CHUNKS` (2 chunks of 32).
    const HEAD: usize = 64;
    let max_tail = max_tail.max(1);
    if probe.len() <= HEAD + max_tail {
        return probe;
    }
    let tail_lo = probe.len() - max_tail;
    let mut out = Vec::with_capacity(HEAD + max_tail);
    out.extend_from_slice(&probe[..HEAD]);
    out.extend_from_slice(&probe[tail_lo..]);
    out
}

#[cfg(test)]
mod windowed_ingest_tests {
    use super::windowed_ingest_ranges_impl as w;

    #[test]
    fn window_bounds_to_system_prompt_plus_last_n_turns() {
        // system prompt occupies [0,2); five sealed turns start at 2,5,9,12,16;
        // current total is 20 blocks.
        let sys = 2;
        let starts = [2usize, 5, 9, 12, 16];
        // Unbounded (0) or window >= turn count → whole parent (no-op).
        assert_eq!(w(sys, &starts, 20, 0), vec![(0, 20)]);
        assert_eq!(w(sys, &starts, 20, 5), vec![(0, 20)]);
        assert_eq!(w(sys, &starts, 20, 9), vec![(0, 20)]);
        // Keep last 2 turns → system [0,2) + [starts[3]=12, 20).
        assert_eq!(w(sys, &starts, 20, 2), vec![(0, 2), (12, 20)]);
        // Keep last 1 → [0,2) + [16, 20).
        assert_eq!(w(sys, &starts, 20, 1), vec![(0, 2), (16, 20)]);
        // No system prompt → single tail range.
        assert_eq!(w(0, &starts, 20, 2), vec![(12, 20)]);
        // Empty parent → no ranges.
        assert_eq!(w(0, &[], 0, 3), Vec::<(usize, usize)>::new());
        // Window reaches into the system-prompt region (keep_from <= sys_end) →
        // contiguous → whole parent.
        assert_eq!(w(13, &starts, 20, 2), vec![(0, 20)]);
    }
}

#[cfg(test)]
mod window_sealed_tokens_tests {
    use super::cap_probe_window;
    use super::window_sealed_tokens;
    use crate::provenance::WideQSig;
    use candle_nn::kv_cache::{ArenaLocation, SealedChunk, SealedSequence};

    /// Build a one-layer sealed turn with the given per-chunk token
    /// counts.  Each chunk gets a distinct detached gid (1000 + idx) so
    /// the boundary-sharing assertion can compare raw ids.
    fn one_layer(counts: &[u16]) -> Vec<SealedSequence> {
        let chunks: Vec<SealedChunk> = counts
            .iter()
            .enumerate()
            .map(|(i, &c)| SealedChunk::for_test(1000 + i as i64, c))
            .collect();
        let token_count = counts.iter().map(|&c| c as usize).sum();
        vec![SealedSequence {
            chunks,
            token_count,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }]
    }

    #[test]
    fn window_spanning_both_boundaries_windows_first_and_last_chunk() {
        // Four chunks [32, 32, 32, 20] = 116 tokens. Window [20, 90) starts
        // mid-chunk-0 (offset 20) and ends mid-chunk-2 (offset 90 -> the
        // chunk spanning [64,96) is cut at local index 26).
        let sealed = one_layer(&[32, 32, 32, 20]);
        let first_gid = sealed[0].chunks[0].gids.as_slice()[0].raw();
        let last_gid = sealed[0].chunks[2].gids.as_slice()[0].raw();

        let win = window_sealed_tokens(&sealed, 20, 90);

        // Chunk-0 partial (20..32 -> 12 tokens), chunk-1 whole (32),
        // chunk-2 partial (64..90 -> 26 tokens). Chunk-3 wholly outside.
        assert_eq!(win[0].chunks.len(), 3);

        // First chunk: windowed at the window start — offset advanced to 20,
        // token_count = 32 - 20 = 12.
        let first = &win[0].chunks[0];
        assert_eq!(first.offset, 20);
        assert_eq!(first.token_count, 12);

        // Middle chunk: wholly inside — cloned as-is.
        assert_eq!(win[0].chunks[1].offset, 0);
        assert_eq!(win[0].chunks[1].token_count, 32);

        // Last chunk: windowed at the window end — offset unchanged (its
        // span begins inside the window), token_count = 90 - 64 = 26.
        let last = &win[0].chunks[2];
        assert_eq!(last.offset, 0);
        assert_eq!(last.token_count, 26);

        // Sequence-level token count equals the window width, and the
        // summed chunk token counts reconstruct it.
        assert_eq!(win[0].token_count, 70);
        let sum: usize = win[0].chunks.iter().map(|c| c.token_count as usize).sum();
        assert_eq!(sum, 90 - 20);

        // The boundary chunks are physically SHARED: the windowed clones
        // carry the same raw gid as their source chunk.
        assert_eq!(first.gids.as_slice()[0].raw(), first_gid);
        assert_eq!(last.gids.as_slice()[0].raw(), last_gid);
    }

    #[test]
    fn empty_window_yields_no_chunks() {
        let sealed = one_layer(&[32, 10]);
        // start >= end -> empty window.
        let win = window_sealed_tokens(&sealed, 20, 20);
        assert_eq!(win[0].chunks.len(), 0);
        assert_eq!(win[0].token_count, 0);
    }

    #[test]
    fn window_beyond_total_clamps_to_sequence_length() {
        let sealed = one_layer(&[32, 10]);
        let win = window_sealed_tokens(&sealed, 0, 1000);
        assert_eq!(win[0].chunks.len(), 2);
        assert_eq!(win[0].token_count, 42);
        assert_eq!(win[0].chunks[0].token_count, 32);
        assert_eq!(win[0].chunks[1].token_count, 10);
    }

    #[test]
    fn chunk_aligned_window_does_not_window_chunks() {
        // Aligned window [32, 64): keeps only the second of three chunks.
        let sealed = one_layer(&[32, 32, 32]);
        let win = window_sealed_tokens(&sealed, 32, 64);
        assert_eq!(win[0].chunks.len(), 1);
        assert_eq!(win[0].chunks[0].token_count, 32);
        assert_eq!(win[0].chunks[0].offset, 0);
        assert_eq!(win[0].token_count, 32);
    }

    #[test]
    fn windowing_the_user_and_assistant_content_halves() {
        // A turn laid out as [markers | user_msg | markers | response]
        // across chunks [32, 32, 20]. Content bounds: user [4, 50),
        // assistant content starts at 56. The two halves carry only the
        // content span, never the leading/trailing template markers.
        let sealed = one_layer(&[32, 32, 20]);
        let user = window_sealed_tokens(&sealed, 4, 50);
        let asst = window_sealed_tokens(&sealed, 56, 84);

        assert_eq!(user[0].token_count, 46);
        // user-half: chunk-0 windowed (4..32 -> 28), chunk-1 windowed (32..50 -> 18).
        assert_eq!(user[0].chunks.len(), 2);
        assert_eq!(user[0].chunks[0].offset, 4);
        assert_eq!(user[0].chunks[0].token_count, 28);
        assert_eq!(user[0].chunks[1].offset, 0);
        assert_eq!(user[0].chunks[1].token_count, 18);

        assert_eq!(asst[0].token_count, 28);
        // assistant-half: chunk-1 windowed (56..64 -> 8), chunk-2 whole (20).
        assert_eq!(asst[0].chunks.len(), 2);
        assert_eq!(asst[0].chunks[0].offset, 24);
        assert_eq!(asst[0].chunks[0].token_count, 8);
        assert_eq!(asst[0].chunks[1].token_count, 20);
    }

    /// A probe within the window is untouched — the common case is a normal turn.
    #[test]
    fn cap_probe_window_leaves_a_normal_turn_whole() {
        let probe = vec![WideQSig::default(); 200];
        assert_eq!(cap_probe_window(probe.clone(), 256).len(), 200);
        // Exactly at the boundary (HEAD 64 + tail 256) is still whole.
        let probe = vec![WideQSig::default(); 320];
        assert_eq!(cap_probe_window(probe, 256).len(), 320);
    }

    /// The failure case: a turn carrying a large tool payload. The probe becomes
    /// head + tail rather than the whole 5.7k-token turn.
    #[test]
    fn cap_probe_window_bounds_an_oversized_turn() {
        let mut probe: Vec<WideQSig> = (0..5702)
            .map(|i| WideQSig {
                n_heads: i as u16,
                words: Vec::new(),
            })
            .collect();
        probe.truncate(5702);
        let capped = cap_probe_window(probe, 256);
        assert_eq!(capped.len(), 64 + 256);
        // The head is the turn's opening — the query — not the payload's tail.
        assert_eq!(capped[0].n_heads, 0);
        assert_eq!(capped[63].n_heads, 63);
        // …followed by the most recent window, ending at the turn's last token.
        assert_eq!(capped[64].n_heads, (5702 - 256) as u16);
        assert_eq!(capped.last().unwrap().n_heads, 5701);
    }

    /// A zero cap must not panic or produce an empty probe.
    #[test]
    fn cap_probe_window_tolerates_a_zero_cap() {
        let probe = vec![WideQSig::default(); 1000];
        assert_eq!(cap_probe_window(probe, 0).len(), 65);
    }
}
