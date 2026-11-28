//! Ingests system-prompt sections into the substrate as sealed content that later
//! turns are projected from.
//!
//! A section is prefilled on a scratch slot, sealed and pinned in the substrate,
//! so projections re-inject it rather than the conversation carrying it.
//! Collection members ingest in parallel against the same prefix and stay out of
//! the content chain, so changing one catalog member does not invalidate every
//! section after it. Sections already present are triaged by id before anything
//! is tokenised, and a shutdown ends the wait instead of holding the process.

use super::*;
use crate::scheduler::exported_state::SharedState;
use crate::SectionSeal;

/// Allocate a scratch slot on `scheduler_tx` — see
/// [`Sequence::alloc_scratch_slot`].
pub(super) fn alloc_scratch_slot_on(
    scheduler_tx: &Sender<SchedulerRequest>,
    substrate: &Conversation,
) -> crate::Result<SequenceId> {
    let (tx, rx) = flume::bounded(1);
    scheduler_tx
        .send(SchedulerRequest::NewSequence {
            conversation: substrate.clone(),
            target: None,
            // Scratch: a GPU resource, not a continuation of anything.
            parent: None,
            response_tx: tx,
        })
        .map_err(|_| ConversationError::SchedulerGone)?;
    replied(&rx).map_err(|_| ConversationError::SchedulerGone)?
}

impl Sequence {
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
        let mut results =
            self.insert_section_collection(&[(section, content)], &[], SectionSeal::Boundary)?;
        results.pop();
        Ok(())
    }

    /// Cumulative-prefix variant of [`Self::insert_section`].  Every
    /// section in `prefix` is Arc-injected onto the scratch slot before this
    /// section's prefill, and every hole in it left empty, so the model's
    /// forward pass attends to the real preceding context at the positions it
    /// is read at.  Used by the schema-driven system_prompt loop to walk
    /// declaration-order sections cumulatively.
    pub fn insert_section_with_prefix(
        &mut self,
        section: SectionId,
        content: &str,
        prefix: &[PrefixItem],
    ) -> crate::Result<()> {
        if content.is_empty() {
            return Ok(());
        }
        let mut results =
            self.insert_section_collection(&[(section, content)], prefix, SectionSeal::Boundary)?;
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
        prefix: &[PrefixItem],
        seal: SectionSeal,
    ) -> crate::Result<Vec<(SectionId, usize)>> {
        self.insert_section_collection_with_progress(sections, prefix, None, &[], seal, |_, _| {})
    }

    /// Same as [`Self::insert_section_collection`] but fires
    /// `on_section_done(section_id, content_len_bytes)` after every
    /// single section completes — skipped, restored from disk, or
    /// freshly ingested — so the surrounding ingest loop can update
    /// its progress bar at section granularity instead of waiting for
    /// the whole batch.
    ///
    /// `seal` is [`SectionSeal::Member`] at the collection's level when the
    /// caller is materialising a schema `Collection` (tools in a tool catalog,
    /// the knowledge base's sections, …): every section in `sections` becomes
    /// a collection member, quantized adaptively at that level.
    /// [`SectionSeal::Boundary`] is for boundary sections — `insert_section`
    /// and `insert_section_with_prefix` always pass it because the model needs
    /// near-lossless K/V on the role markers / opening and closing tags every
    /// later token attends back over.
    ///
    /// `range` is the collection's reserved range, in positions, when it has
    /// one: each section is then sealed at the range's top, behind `prefix` and
    /// the part of the range above its own tokens left empty
    /// (`docs/placed_section_layout.md` §3.1b), so its address names that hole
    /// too.
    ///
    /// `lead` is the collection's lead (`docs/placed_section_layout.md` §3.5f),
    /// already sealed: each section is sealed straight behind it, the range's
    /// hole short by its positions.
    pub fn insert_section_collection_with_progress<F>(
        &mut self,
        sections: &[(SectionId, &str)],
        prefix: &[PrefixItem],
        range: Option<usize>,
        lead: &[SectionId],
        seal: SectionSeal,
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
        // ── Pass 1: already-present sections, by id alone ──────────────
        // `section_exists` is broader than `section_is_hot`: it returns true for
        // cold-marker sections that were restored on substrate reload but
        // haven't been elevated yet.  Either way the substrate already knows
        // about this section; the elevate path will lift it on the next
        // projection that needs it.
        //
        // This runs BEFORE tokenising anything, because the test needs only the
        // section id — not the tokens, the section hash, or the prefix hash. A
        // caller that constructs many sequences against one shared schema (the
        // calibration phase builds ~700 against the same tool catalog) re-walks
        // that whole catalog per sequence, and every walk after the first is
        // entirely skips. Tokenising first made each of those walks re-tokenise
        // the full catalog to arrive at a hash it then threw away: measured at
        // ~294 ms per sequence, 206 s of a 341 s phase, all of it on the calling
        // thread with the GPU idle behind it.
        let mut out_skip: Vec<(SectionId, usize)> = Vec::new();
        let mut candidates: Vec<(SectionId, &str)> = Vec::with_capacity(sections.len());
        // ONE read guard for the whole triage, not one per section. The substrate
        // RwLock is writer-priority, so each reader taken while the persistence
        // thread is queued to write blocks until that write lands — and a
        // conversation opening against a large catalog runs this loop once per
        // section. Profiled as the single largest source of contended lock time
        // in the daemon (`RwLock::read_contended` under `insert_section_collection`).
        //
        // Holding one guard across the loop is also *less* blocking than
        // re-taking it: a writer waits for one reader span instead of racing a
        // rapid re-acquire cycle it can never win outright.
        //
        // `on_section_done` is caller-supplied, so it is deliberately NOT called
        // under the guard: the substrate lock is writer-priority, and a callback
        // that took a read while a writer was queued would deadlock against the
        // guard we are already holding. The skips are collected here and reported
        // below, after the guard drops — same callbacks, same order.
        let mut skipped: Vec<(SectionId, usize, usize)> = Vec::new();
        {
            let view = self.substrate.read();
            for &(section_id, content) in sections {
                if content.is_empty() {
                    continue;
                }
                if view.section_exists(section_id) {
                    let block_count = view.section_block_count(section_id).unwrap_or(0);
                    skipped.push((section_id, block_count, content.len()));
                    continue;
                }
                candidates.push((section_id, content));
            }
        }
        for (section_id, block_count, content_len) in skipped {
            out_skip.push((section_id, block_count));
            on_section_done(section_id, content_len);
        }
        // Every section was already present — nothing left needs a content
        // address, so the prefix chain below (which reads each prefix section's
        // tokens out of the substrate) is never built.
        if candidates.is_empty() {
            return Ok(out_skip);
        }

        let (base_chain, lead_tokens) = {
            let mut chain = ContentChain::new();
            let view = self.substrate.read();
            let lead_tokens: Vec<_> = lead.iter().map(|&id| view.section_tokens_of(id)).collect();
            let schema = self.projection.schema();
            for &item in prefix {
                let pid = match item {
                    PrefixItem::Section(pid) => pid,
                    // Everything after a hole is read past it.
                    PrefixItem::Hole(positions) => {
                        chain.push_hole(positions);
                        continue;
                    }
                };
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
            (chain, lead_tokens)
        };
        // A member of a range is sealed at its top: behind the prefix, the
        // part of the range above the lead and its own tokens, then the lead.
        // Without a range, behind the prefix and the lead alone.
        let lead_positions: usize = lead_tokens.iter().map(|t| t.len()).sum();
        let member_hole =
            |tokens: usize| range.map_or(0, |r| r.saturating_sub(tokens + lead_positions));
        let behind = MembersPrefix {
            prefix: prefix.to_vec(),
            range,
            level: seal.policy().compression_level,
            lead: lead.to_vec(),
            lead_tokens: lead_positions,
        };
        let member_prefix = |tokens: usize| behind.behind(tokens, &[]);
        let prefix_hash_of = |tokens: usize| {
            let mut chain = base_chain.clone();
            chain.push_hole(member_hole(tokens));
            for t in &lead_tokens {
                chain.push_section(t);
            }
            chain.prefix()
        };
        // ── Pass 2: triage what's left into two buckets ────────────────
        //   - Durable in the redo log under its content-addressed stream id →
        //     restore from disk (`RestoreSection`). Whether those chunks form a
        //     whole grid is the scheduler's call, not this one's: it knows how
        //     many KV backings the session holds, and on a hybrid that is the
        //     attention layers, not the model's transformer depth.
        //   - Otherwise → ingest with a fresh prefill (`IngestSection`).
        let mut to_ingest: Vec<Pending<'_>> = Vec::with_capacity(candidates.len());
        let mut to_restore: Vec<Pending<'_>> = Vec::with_capacity(candidates.len());
        for (section_id, content) in candidates {
            let tokens = self.tokenize(content)?;
            if tokens.is_empty() {
                continue;
            }
            let token_count = tokens.len();
            let address = ContentAddress {
                prefix_hash: prefix_hash_of(token_count),
                section_hash: hash_tokens(&tokens),
            };
            let debug_name = self.section_debug_name(section_id);
            let pending = Pending {
                section_id,
                content,
                tokens,
                token_count,
                address,
                debug_name,
            };
            if self
                .substrate
                .section_stream_is_persisted(section_stream_id(address))
            {
                to_restore.push(pending);
            } else {
                to_ingest.push(pending);
            }
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
            // Capture the content length for the progress callback
            // before `item.tokens` / `item.content` get consumed by
            // the request payload below.
            let item_section_id = item.section_id;
            let item_content_len = item.content.len();
            let (tx, rx) = flume::bounded(1);
            self.scheduler_tx
                .send(SchedulerRequest::RestoreSection {
                    conversation: self.substrate.clone(),
                    section_id: item.section_id,
                    stream_id: section_stream_id(item.address),
                    tokens: item.tokens.clone(),
                    response_tx: tx,
                })
                .map_err(|_| ConversationError::SchedulerGone)?;
            match replied(&rx).map_err(|_| ConversationError::SchedulerGone)? {
                Ok(chunks_per_layer) => {
                    // Block count for diagnostics — the section is now a
                    // cold-marker; the actual hot grid lands when the next
                    // projection elevates it.
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

        // Every section still in `to_ingest` — a refused restore included —
        // costs a prefill from here on. `out_skip` (pass 1, above) is counted
        // into `restored` too: a section that `section_exists` already found
        // resident — a cold-marker reconstructed when the substrate was
        // opened — needed no fresh prefill just as surely as one this call
        // explicitly restored, so excluding it from `SectionLoads` undercounts
        // a restart that mostly restores through substrate-open reconstruction
        // rather than this triage's own `RestoreSection` round trip.
        self.substrate
            .record_section_loads(restore_out.len() + out_skip.len(), to_ingest.len());

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
        //
        // A collection's members all start their recurrence from the state the
        // prompt before the collection leaves — one state for the batch, kept
        // under that prefix for every conversation on the prompt — so a member
        // is computed as it is read, straight after that prefix. They do not
        // attend to one another, and do not continue one another's state: a
        // projection picks among them.
        //
        // In a range every member is read across a hole of its own, so the
        // state it starts from is computed behind that hole; the holes do not
        // change what the recurrence reads, only where the attention it feeds
        // reads it from.
        let start_of = |tokens: usize| -> crate::Result<Option<SharedState>> {
            if !seal.is_member() {
                return Ok(None);
            }
            let system = &self.projection.schema().system_prompt;
            let prompt: Vec<PrefixItem> = member_prefix(tokens)
                .into_iter()
                .filter(|item| {
                    item.section()
                        .is_none_or(|id| !system.is_collection_member(id))
                })
                .collect();
            state_after(&self.scheduler_tx, &self.substrate, &prompt)
        };
        let band = self.band();
        let mut slot_ids: Vec<SequenceId> = Vec::with_capacity(to_ingest.len());
        for _ in 0..to_ingest.len() {
            match self.alloc_scratch_slot() {
                Ok(slot) => slot_ids.push(slot),
                Err(e) => {
                    // The slots already allocated have no ingest in flight yet
                    // — free them before propagating, or a failed 12th alloc
                    // leaks the first 11 for the life of the process.
                    for slot in slot_ids {
                        self.free_scratch_slot(slot);
                    }
                    return Err(e);
                }
            }
        }
        let t_alloc = std::time::Instant::now();

        let mut pending: Vec<(
            SequenceId,
            SectionId,
            usize, // content_len for the progress callback
            Receiver<crate::Result<SealResult>>,
        )> = Vec::with_capacity(to_ingest.len());
        let mut slots = slot_ids.into_iter();
        for (slot_id, item) in slots.by_ref().zip(to_ingest) {
            let (tx, rx) = flume::bounded(1);
            let content_len = item.content.len();
            let start = match start_of(item.token_count) {
                Ok(start) => start,
                Err(e) => {
                    self.free_scratch_slot(slot_id);
                    for slot in slots {
                        self.free_scratch_slot(slot);
                    }
                    self.drain_and_free(&mut pending);
                    return Err(e);
                }
            };
            if self
                .scheduler_tx
                .send(SchedulerRequest::IngestSection {
                    sequence_id: slot_id,
                    section_id: item.section_id,
                    prefix: member_prefix(item.token_count),
                    tokens: item.tokens,
                    address: item.address,
                    debug_name: item.debug_name,
                    seal,
                    start,
                    // At the conversation's own band: whoever waits on its
                    // first turn waits on its frame sealing first.
                    band,
                    response_tx: tx,
                })
                .is_err()
            {
                // A failed send means the scheduler is gone, so the frees are
                // best-effort too — but the drain below still owes each fired
                // section its wait, and the unfired slots owe nothing.
                self.free_scratch_slot(slot_id);
                for slot in slots {
                    self.free_scratch_slot(slot);
                }
                self.drain_and_free(&mut pending);
                return Err(ConversationError::SchedulerGone);
            }
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
        // GUI jump.  `flume::Selector` lets us pick up the
        // next ready receiver instead, firing `on_section_done`
        // progressively as sections actually complete.
        //
        // **The wait looks at the ingest-cancel latch every 100 ms.** A whole
        // tool catalog is minutes of prefill, and a load blocked here past a
        // process stop kept the process alive — drained from its cluster, never
        // exiting. A stop abandons the stragglers to the scheduler's teardown
        // (`drain_and_free`) and the collection ends `IngestCancelled`; what
        // sealed before it stands, and the next load restores it.
        let poll = std::time::Duration::from_millis(100);
        while !pending.is_empty() {
            if self.substrate.ingest_cancelled() {
                self.drain_and_free(&mut pending);
                return Err(ConversationError::IngestCancelled);
            }
            let mut selector = flume::Selector::new();
            for (i, (_, _, _, rx)) in pending.iter().enumerate() {
                selector = selector.recv(rx, move |r| (i, r));
            }
            // `wait_timeout` consumes the selector, so its borrow of `pending`
            // ends here.
            let Ok((idx, r)) = selector.wait_timeout(poll) else {
                continue;
            };
            let settled = r
                .map_err(|_| ConversationError::SchedulerGone)
                .and_then(|seal| seal);
            let (slot_id, section_id, content_len, _rx) = pending.swap_remove(idx);
            let seal = match settled {
                Ok(seal) => seal,
                Err(e) => {
                    // A failed section must not strand the collection's
                    // scratch slots — without this, a 20-section ingest that
                    // failed at section 3 leaked the other slots for the life
                    // of the process.
                    self.free_scratch_slot(slot_id);
                    self.drain_and_free(&mut pending);
                    return Err(e);
                }
            };
            let block_count = seal.block_to.saturating_sub(seal.block_from);
            self.free_scratch_slot(slot_id);
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
    pub(super) fn section_debug_name(&self, section_id: SectionId) -> String {
        for s in self.projection.schema().system_prompt.all_sections() {
            if s.id == section_id {
                return s.name.clone();
            }
        }
        format!("section_{}", section_id.raw())
    }

    /// Release a scratch slot — [`Self::alloc_scratch_slot`]'s other half,
    /// named so every exit path frees a slot the same way. Best-effort: a
    /// gone scheduler has already reclaimed everything.
    pub(super) fn free_scratch_slot(&self, slot: SequenceId) {
        let _ = self
            .scheduler_tx
            .send(SchedulerRequest::FreeSequence { sequence_id: slot });
    }

    /// Wait out every still-ingesting scratch slot and free it — the failure
    /// path's cleanup. A slot freed under an active ingest is a scheduler
    /// hazard, so each straggler is drained to completion first (immediate
    /// when the scheduler itself is gone: the channels are already dead).
    /// The wait honours the ingest-cancel latch the way the turn waits do:
    /// on a graceful shutdown the remaining stragglers are abandoned to the
    /// scheduler's own teardown rather than waited out.
    pub(super) fn drain_and_free(
        &self,
        pending: &mut Vec<(
            SequenceId,
            SectionId,
            usize,
            Receiver<crate::Result<SealResult>>,
        )>,
    ) {
        let poll = std::time::Duration::from_millis(100);
        for (slot, _, _, rx) in pending.drain(..) {
            loop {
                match rx.recv_timeout(poll) {
                    Ok(_) | Err(flume::RecvTimeoutError::Disconnected) => break,
                    Err(flume::RecvTimeoutError::Timeout) => {
                        if self.substrate.ingest_cancelled() {
                            return;
                        }
                    }
                }
            }
            self.free_scratch_slot(slot);
        }
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
    pub(super) fn alloc_scratch_slot(&self) -> crate::Result<SequenceId> {
        alloc_scratch_slot_on(&self.scheduler_tx, &self.substrate)
    }
}
