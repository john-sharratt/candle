use super::*;
use crate::token_buffer::TokenBuffer;
use std::collections::HashSet;

/// Default pool-budget headroom we keep free by offloading hot KV — see
/// [`vram_budget_band`]. 2 GiB: sized **above** a wide ragged prefill forward's
/// transient allocation peak (per-sequence activations × batch width + MoE
/// expert gather), which on a memory-tight card is far larger than a lone
/// decode's. Relieving only near ~1 GiB let a 20-wide upload forward's peak tip
/// the card into WDDM host-memory spill — tens of seconds per forward — so we now
/// keep a wider margin and shed hot KV earlier to defend it.
const DEFAULT_VRAM_BUDGET_BAND_MB: usize = 2048;

/// Pool-budget headroom kept free (bytes), overridable at process start via
/// `ZEND_VRAM_BUDGET_BAND_MB` so it can be tuned to a specific card/model without
/// a rebuild — the right value depends on the model's per-token activation
/// footprint and the prefill batch width, which vary per deployment. Cached on
/// first read. `0`/unparseable falls back to [`DEFAULT_VRAM_BUDGET_BAND_MB`].
fn vram_budget_band() -> usize {
    static BAND: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BAND.get_or_init(|| {
        let mb = std::env::var("ZEND_VRAM_BUDGET_BAND_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .unwrap_or(DEFAULT_VRAM_BUDGET_BAND_MB);
        mb * 1024 * 1024
    })
}
/// Pool reuse headroom (reserved-but-free pool bytes) below which the
/// stream-ordered pool can no longer absorb a new allocation without growing
/// our OS footprint — only then does a low driver `free` count as pressure
/// (mirrors `vram_has_room`'s `os_needed`). A fresh arena is small, so unlike
/// the budget band this stays tight; it is the OS-safety floor, not the
/// keep-forwards-fast headroom.
const VRAM_REUSE_BAND: usize = 512 * 1024 * 1024;
/// Bytes of hot KV to shed per pressure episode. The eviction overshoots the
/// trigger by this much, so the pool budget oscillates in
/// `[band, band + VRAM_EVICT_BAND]` (band = [`vram_budget_band`]) and we don't
/// re-trip on the very next wave.
const VRAM_EVICT_BAND: u64 = 1024 * 1024 * 1024;
/// Safety cap on the synchronous substrate-offload flush under pressure. The
/// pass migrates hot→warm *before* its cold-disk writes, so the warm copies
/// the eviction needs exist well before this fires — a timeout only clips the
/// tail of the cold-write wait (turns are already evictable) and guards against
/// a wedged persistence thread; it is not the expected path.
const VRAM_OFFLOAD_FLUSH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

impl Scheduler {
    /// Promote up to `MAX_ACTIVE_PREFILLS - active_prefills.len()` newly
    /// submitted PrefillWorks from the FIFO queue into the in-flight
    /// `active_prefills` set. Emits the initial `Prefill` and
    /// `PrefillProgress(0, total)` events so callers see their submission
    /// was picked up.
    /// Under VRAM pressure, shed hot KV to the substrate to reopen budget, and
    /// report whether pressure **survived** the attempt. Three steps:
    ///
    ///  1. Give substrate offload complete priority: synchronously drain the
    ///     pending hot→warm migration so just-sealed turns gain a warm (RAM) copy
    ///     — only warm-backed turns can be evicted hot→warm.
    ///  2. Evict the least-recently-used hot turns across the resident
    ///     conversations (drop the hot copy, keep warm), returning their VRAM
    ///     chunks to the pool free-list.
    ///  3. Release the arenas the eviction emptied back to the pool, which is what
    ///     actually lowers `pool_used` and restores budget. The cheap release only
    ///     frees fully-empty arenas; if that leaves pressure unrelieved and a
    ///     forced defrag could reclaim a fragmented arena, fall back to the
    ///     chunk-moving compaction (guarded by `can_reclaim_arena`) rather than let
    ///     an admitted prefill spill to host memory and thrash to death.
    ///
    /// Returns `true` if VRAM is **still** under pressure afterward — the caller's
    /// signal to narrow the admission window and stop admitting this pass.
    /// `whence` tags the log line with the calling gate (`promote` / `pump`).
    pub(super) fn relieve_vram_pressure(&mut self, whence: &str) -> bool {
        let t = std::time::Instant::now();
        let flushed = self
            .persist_trigger
            .flush_blocking(VRAM_OFFLOAD_FLUSH_TIMEOUT);
        let evicted = self.evict_cold_tail(VRAM_EVICT_BAND);
        let mut released = self.session.release_empty_arenas().unwrap_or(0);
        let mut still = self.vram_under_pressure();
        if still && self.session.can_reclaim_arena() {
            released += self.session.compact_forced().unwrap_or(0);
            still = self.vram_under_pressure();
        }
        if let Some((free, total)) = self.session.vram_free_total() {
            let (pool_used, pool_reserved) = self.session.vram_pool_stats().unwrap_or((0, 0));
            let offload_ms = t.elapsed().as_millis() as u64;
            let evicted_mib = evicted.bytes / (1 << 20);
            // Our footprint vs the OS-reserved high-water: this, not `vram_free`,
            // says what's actually consuming the card.
            let pool_used_mib = (pool_used / (1 << 20)) as u64;
            let pool_reserved_mib = (pool_reserved / (1 << 20)) as u64;
            let vram_free_mib = (free / (1 << 20)) as u64;
            let vram_total_mib = (total / (1 << 20)) as u64;
            let relieved = !still;
            // INFO only when the pass actually freed something — an eviction that
            // reclaimed hot turns or arenas is a real event worth surfacing. When it
            // was a no-op (nothing evictable, pressure persists) log at DEBUG: this
            // runs from both the promote and pump gates every scheduler loop, so
            // under a sustained upload burst an unconditional INFO floods the log.
            macro_rules! emit {
                ($lvl:ident) => {
                    tracing::$lvl!(
                        target: "candle_conversation::scheduler::timing",
                        whence,
                        offload_ms,
                        warm_flushed = flushed,
                        turns_evicted = evicted.count,
                        evicted_mib,
                        arenas_released = released,
                        pool_used_mib,
                        pool_reserved_mib,
                        vram_free_mib,
                        vram_total_mib,
                        relieved,
                        "offloaded hot KV to substrate under VRAM pressure"
                    )
                };
            }
            if evicted.count > 0 || released > 0 {
                emit!(info);
            } else {
                emit!(debug);
            }
        }
        still
    }

    pub(super) fn promote_new_prefills(&mut self) {
        // Ragged prefill forward width — how many in-flight prefills coalesce into
        // one forward: a burst of small parallel scopes (code_read's worker count),
        // a bulk collection ingest's per-section prefills (`insert_section_collection`
        // fires one prefill per section), or a batch of calibration cases. Capped by
        // the AIMD `admit_window`, which narrows the batch under VRAM pressure so the
        // forward's transient peak (which scales with width) can't OOM a busy card;
        // `MIN_PREFILL_WIDTH` keeps ≥1 in flight regardless. (Decode-side waves are
        // bounded separately, e.g. calibration's `CALIBRATION_BATCH`.)
        let cap = Self::MAX_PREFILL_WIDTH.min(self.admit_window.max(Self::MIN_PREFILL_WIDTH));
        while self.active_prefills.len() < cap {
            // VRAM-pressure backpressure (wave budgeting). Each admitted prefill
            // pins its conversation's KV in VRAM, so under pressure we shed hot KV
            // to the substrate rather than piling on more concurrent prefills; if
            // that doesn't clear it, narrow the window and leave the rest queued
            // this pass. We always keep ≥1 prefill in flight so the engine makes
            // progress — a single oversized turn is then bounded by the per-arena
            // VRAM budget gate (which compacts/fails fast rather than spilling).
            if !self.active_prefills.is_empty() && self.vram_under_pressure() {
                if self.relieve_vram_pressure("promote") {
                    // Pressure survived eviction — back off the admission window
                    // (so subsequent waves run narrower forwards) and stop piling
                    // on. The `!is_empty` guard already kept ≥1 in flight.
                    self.shrink_admit_window();
                    break;
                }
            }
            let work = match self.prefill_queue.pop_front() {
                Some(w) => w,
                None => break,
            };
            let total = work.tokens.len();
            let _ = work
                .event_tx
                .send(TurnEvent::Prefill(work.prefill_text.clone()));
            let _ = work.event_tx.send(TurnEvent::PrefillProgress {
                tokens_done: 0,
                tokens_total: total,
            });
            let error = if total == 0 {
                Some(ConversationError::Channel(
                    "prefill received zero tokens".into(),
                ))
            } else {
                None
            };
            self.active_prefills.push(ActivePrefill {
                work,
                offset: 0,
                next_projection: 0,
                final_logits: None,
                error,
                prefill_start: None,
            });
        }
    }

    /// True when VRAM is under pressure — the signal to offload hot KV to the
    /// substrate (and, failing that, to stop admitting more concurrent
    /// prefills).
    ///
    /// Two complementary gates, pressure if **either** trips:
    /// - **Pool budget low** — [`vram_budget_available`] (`init_free -
    ///   pool_used - reserve`) drops below [`vram_budget_band`]. Pool-aware, so
    ///   it doesn't false-fire when KV is freed back into our stream-ordered
    ///   pool (which the driver `free` can't see), robust to WDDM's polluted
    ///   driver free, and the gate hot-tier eviction can actually relieve
    ///   (dropping a hot copy lowers `pool_used`). The band sits above the
    ///   per-forward transient peak so we relieve *before* forwards stall.
    /// - **Driver free below the reserve floor *and* the pool can't absorb the
    ///   next allocation by reuse** — `free < max(10% total, 1 GiB)` while the
    ///   pool's reserved-but-free headroom (`pool_reserved - pool_used`) is
    ///   under [`VRAM_REUSE_BAND`]. The reuse-headroom qualifier mirrors
    ///   [`vram_has_room`]'s `os_needed` gate: a low driver free while the pool
    ///   still holds freed blocks to reuse is *not* pressure (reusing them
    ///   costs zero new OS memory, so `free` never moves) — without this
    ///   qualifier the floor false-fires on WDDM, where the pool's own
    ///   reservation pins driver free low, and needlessly throttles admission
    ///   while gigabytes of pool budget remain.
    ///
    /// `false` on non-CUDA / when the queries are unavailable.
    ///
    /// [`vram_budget_available`]: super::super::BatchedInferenceSession::vram_budget_available
    /// [`vram_has_room`]: candle_nn::kv_cache
    pub(super) fn vram_under_pressure(&self) -> bool {
        let pool_low = self
            .session
            .vram_budget_available()
            .is_some_and(|avail| avail < vram_budget_band());
        let driver_below_floor = match (
            self.session.vram_free_total(),
            self.session.vram_pool_stats(),
        ) {
            (Some((free, total)), Some((used, reserved))) => {
                let reuse_headroom = reserved.saturating_sub(used);
                free < (total / 10).max(1usize << 30) && reuse_headroom < VRAM_REUSE_BAND
            }
            _ => false,
        };
        pool_low || driver_below_floor
    }

    /// Shed least-recently-used hot turn KV to the warm (RAM) tier across the
    /// resident conversations, freeing up to `target_bytes` of pool VRAM.
    /// Oldest-first and reversible (a reselected turn reloads from RAM). Only
    /// turns that already hold a warm copy are evictable, so callers should
    /// first [`PersistenceTrigger::flush_blocking`] to make the just-sealed
    /// turns qualify. The `target_bytes` budget caps total bytes freed, so a
    /// conversation reached via several slots is naturally not over-evicted
    /// (and `evict_hot_to_free` is per-conversation scoped — it can never touch
    /// a parallel conversation's selected working set).
    fn evict_cold_tail(&mut self, target_bytes: u64) -> crate::substrate::EvictionReport {
        let mut report = crate::substrate::EvictionReport { count: 0, bytes: 0 };
        let mut remaining = target_bytes;
        let convs: Vec<Conversation> = self.slot_conversations.values().cloned().collect();
        for conv in convs {
            if remaining == 0 {
                break;
            }
            let r = conv.write().evict_hot_to_free(&[], &[], remaining);
            remaining = remaining.saturating_sub(r.bytes);
            report.count += r.count;
            report.bytes += r.bytes;
        }
        report
    }

    /// Number of in-flight prefills that still have tokens left to process
    /// and have not errored.
    pub(super) fn prefill_width(&self) -> usize {
        self.active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.offset < p.work.tokens.len())
            .count()
    }

    /// Number of in-flight section ingests with tokens remaining (not errored).
    pub(super) fn section_ingest_width(&self) -> usize {
        self.active_section_ingests
            .iter()
            .filter(|s| s.error.is_none() && s.offset < s.tokens.len())
            .count()
    }

    /// Run one prefill chunk across all active section ingests, batching them
    /// into a single `forward_batched` call so collection members sharing the
    /// same prefix context attend in parallel rather than serially.
    pub(super) fn run_one_section_ingest_chunk(&mut self) {
        let active: Vec<usize> = (0..self.active_section_ingests.len())
            .filter(|&i| {
                let s = &self.active_section_ingests[i];
                s.error.is_none() && s.offset < s.tokens.len()
            })
            .collect();
        if active.is_empty() {
            return;
        }

        // Ragged batch: each section advances by its OWN min(remaining, cap).
        // The varlen forward packs the heterogeneous lengths flat, so one
        // near-finished section no longer collapses the whole wave to the batch
        // minimum — the bug that dragged a 93-wide tool-catalog ingest down to
        // ~1 token/seq/forward. Mirrors `run_one_prefill_pass`.
        //
        // Bound the TOTAL tokens per forward to the same per-forward budget a
        // normal prefill targets (`max_prefill_pass_tokens`). Without this the
        // whole active set coalesces into one forward: the 93-section tool
        // catalog (~21k tokens) packed into a single pass whose transient
        // activation spiked VRAM to the card ceiling and paged (one forward took
        // minutes on a 16 GB card). Sections beyond the budget ride the next
        // `run_one_section_ingest_chunk` pass — the wave loop calls this until
        // every section seals — so throughput is unchanged (each forward still
        // fills to the expert-amortization target) while the peak stays bounded.
        // At least one section is always admitted so the wave makes progress.
        let cap = self.max_prefill_pass_tokens;
        let mut seq_ids: Vec<usize> = Vec::with_capacity(active.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(active.len());
        let mut group_idxs: Vec<usize> = Vec::with_capacity(active.len());
        let mut advances: Vec<usize> = Vec::with_capacity(active.len());
        let mut batch_tokens = 0usize;
        for &i in &active {
            let s = &mut self.active_section_ingests[i];
            let off = s.offset;
            let advance = (s.tokens.len() - off).min(cap);
            // Stop packing once this forward has reached the per-forward budget
            // (but never emit an empty forward).
            if !seq_ids.is_empty() && batch_tokens + advance > cap {
                break;
            }
            let tokens = &s.tokens[off..off + advance];
            match Tensor::new(tokens, &self.device).and_then(|t| t.unsqueeze(0)) {
                Ok(t) => {
                    seq_ids.push(s.sequence_id.0);
                    inputs.push(t);
                    group_idxs.push(i);
                    advances.push(advance);
                    batch_tokens += advance;
                }
                Err(e) => {
                    s.error = Some(ConversationError::Model(e));
                }
            }
        }
        if seq_ids.is_empty() {
            return;
        }

        let total_tokens: usize = advances.iter().sum();
        tracing::debug!(
            target: "sched",
            "section_ingest batch={} tokens={} seq_ids={:?}",
            seq_ids.len(),
            total_tokens,
            seq_ids
        );

        // Attended-KV length swept, summed over the batch, before the forward
        // advances the sequences.
        let kv_len: usize = seq_ids
            .iter()
            .map(|&sid| self.session.sequence_offset(sid).unwrap_or(0))
            .sum();
        let t_fwd = Instant::now();
        let logits_vec = match self
            .model
            .forward_batched(&mut self.session, &seq_ids, &inputs)
        {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("batched section ingest forward failed: {e}");
                for &i in &group_idxs {
                    self.active_section_ingests[i].error =
                        Some(ConversationError::Channel(msg.clone()));
                }
                return;
            }
        };
        // Section ingest batch = n_seqs sequences, Σ advances tokens (ragged).
        let fwd_ms = t_fwd.elapsed().as_millis() as u64;
        self.wave_stats
            .record_section(seq_ids.len(), total_tokens, kv_len, fwd_ms);

        // Logits are produced but not used — section ingests have no decode.
        // We only need to advance each section by ITS OWN advance and record
        // its slot tokens.
        for ((_logits, &i), &advance) in logits_vec
            .into_iter()
            .zip(group_idxs.iter())
            .zip(advances.iter())
        {
            let s = &mut self.active_section_ingests[i];
            if let Err(e) = self.session.advance_sequence(s.sequence_id.0, advance) {
                s.error = Some(ConversationError::Model(e));
                continue;
            }
            let seq_id = s.sequence_id;
            let off = s.offset;
            let chunk_tokens = s.tokens[off..off + advance].to_vec();
            super::Scheduler::record_slot_tokens(&mut self.slot_tokens, seq_id, &chunk_tokens);
            s.offset += advance;
        }
    }

    /// Drain completed or errored section ingest entries. Errored entries send
    /// `Err`; finished entries call `finalize_section_ingest` (seal + write)
    /// and send the `SealResult`.
    pub(super) fn finalize_done_section_ingests(&mut self) {
        let mut i = 0;
        while i < self.active_section_ingests.len() {
            let done = {
                let s = &self.active_section_ingests[i];
                s.error.is_some() || s.offset >= s.tokens.len()
            };
            if !done {
                i += 1;
                continue;
            }
            let s = self.active_section_ingests.swap_remove(i);
            if let Some(e) = s.error {
                let _ = s.response_tx.send(Err(e));
                continue;
            }
            let result = self.finalize_section_ingest(
                s.sequence_id,
                s.section_id,
                s.seal_block_from,
                std::sync::Arc::new(s.tokens.to_vec()),
                s.address,
                s.debug_name,
                s.in_collection,
            );
            let _ = s.response_tx.send(result);
            // swap_remove pulled the last element into i; don't increment.
        }
    }

    /// Run **one** prefill pass across every still-active in-flight prefill.
    /// Each prefill advances by its own `min(remaining, max_prefill_pass_tokens)`
    /// tokens; the varlen forward packs the ragged lengths flat into a single
    /// batched call.
    ///
    /// Sequences whose offset reaches `tokens.len()` after this pass have
    /// their `final_logits` recorded; they will be promoted to decode by
    /// `promote_finished_prefills_to_decodes`.
    pub(super) fn run_one_prefill_pass(&mut self) {
        // Collect indices of still-active prefills.
        let active: Vec<usize> = (0..self.active_prefills.len())
            .filter(|&i| {
                let p = &self.active_prefills[i];
                p.error.is_none() && p.final_logits.is_none() && p.offset < p.work.tokens.len()
            })
            .collect();
        if active.is_empty() {
            return;
        }

        // Ragged batch: each prefill advances by its OWN min(remaining, cap).
        // The varlen forward packs the heterogeneous lengths flat, so a short
        // scope no longer collapses the whole wave to the batch minimum.
        let cap = self.max_prefill_pass_tokens;
        let mut seq_ids: Vec<usize> = Vec::with_capacity(active.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(active.len());
        let mut group_idxs: Vec<usize> = Vec::with_capacity(active.len());
        let mut advances: Vec<usize> = Vec::with_capacity(active.len());
        for &i in &active {
            let p = &mut self.active_prefills[i];
            if p.prefill_start.is_none() {
                p.prefill_start = Some(Instant::now());
            }
            let off = p.offset;
            let mut advance = (p.work.tokens.len() - off).min(cap);
            // Staged calibration prefill: don't advance past the next projection
            // point, so the wave stops exactly on it and the advance loop below can
            // emit that segment's projection. Normal prefills carry no offsets and
            // advance by the full cap, co-batching in this same ragged forward.
            if let Some(&next_off) = p.work.projection_offsets.get(p.next_projection) {
                let seg_remaining = (next_off as usize).saturating_sub(off);
                if seg_remaining > 0 {
                    advance = advance.min(seg_remaining);
                }
            }
            let tokens = &p.work.tokens[off..off + advance];
            match Tensor::new(tokens, &self.device).and_then(|t| t.unsqueeze(0)) {
                Ok(t) => {
                    seq_ids.push(p.work.sequence_id.0);
                    inputs.push(t);
                    group_idxs.push(i);
                    advances.push(advance);
                }
                Err(e) => {
                    p.error = Some(ConversationError::Model(e));
                }
            }
        }
        if seq_ids.is_empty() {
            return;
        }

        let total_tokens: usize = advances.iter().sum();
        tracing::debug!(
            target: "sched",
            "prefill batch={} tokens={} decode_active={} seq_ids={:?}",
            seq_ids.len(),
            total_tokens,
            self.active_decodes.len(),
            seq_ids
        );

        let n_seqs = seq_ids.len();
        // Clear the per-op pipeline profile so the snapshot after the forward
        // covers only THIS prefill pass (attn:core / ffn / qkv / out_proj / the
        // paged-prefill kernel, summed over layers) — the code-read prefill is
        // the dominant wave cost and this is the only place its internal split
        // is exposed.
        #[cfg(feature = "profile")]
        let _ = candle_transformers::models::profile::pipeline_snapshot_and_reset();
        // Capture each sequence's attended context length (prefix + new tokens)
        // BEFORE the forward advances it, so the breakdown can tie the
        // attention-kernel time to the kv_len it actually sweeps — the deciding
        // number for prefix-bound vs kernel-inefficiency.
        #[cfg(feature = "profile")]
        let kv_prefixes: Vec<usize> = seq_ids
            .iter()
            .map(|&sid| self.session.sequence_offset(sid).unwrap_or(0))
            .collect();
        // Total attended-KV length this prefill sweeps (prefix/context summed
        // over the batch), captured before the forward advances the sequences.
        // Surfaced on the wave line so a growing prefix (the prefill-slowing
        // growth area) is visible vs. a flat paged-glue prefix.
        let kv_len: usize = seq_ids
            .iter()
            .map(|&sid| self.session.sequence_offset(sid).unwrap_or(0))
            .sum();
        let t_fwd = Instant::now();
        let logits_vec = match self
            .model
            .forward_batched(&mut self.session, &seq_ids, &inputs)
        {
            Ok(v) => v,
            Err(e) => {
                if candle_nn::kv_cache::is_device_oom(&e) {
                    // The batch outgrew the card's VRAM. Narrow the admission
                    // window so following waves run smaller forwards, and requeue
                    // the batch's scope-ingest prefills rather than dropping their
                    // content — a freed slot re-pumps them at the narrower width.
                    self.handle_prefill_oom(&group_idxs, &e);
                } else {
                    let msg = format!("batched prefill forward failed: {e}");
                    for &i in &group_idxs {
                        self.active_prefills[i].error =
                            Some(ConversationError::Channel(msg.clone()));
                    }
                }
                return;
            }
        };
        // Prefill batch = n_seqs sequences, Σ advances tokens (ragged).
        let fwd_ms = t_fwd.elapsed().as_millis() as u64;
        self.wave_stats
            .record(true, n_seqs, total_tokens, kv_len, fwd_ms);
        #[cfg(feature = "profile")]
        {
            let snap = candle_transformers::models::profile::pipeline_snapshot_and_reset();
            let mut parts: Vec<String> = snap
                .entries
                .iter()
                .map(|(n, ms, c)| format!("{n}={ms:.1}ms({c})"))
                .collect();
            parts.sort_by(|a, b| b.cmp(a));
            let max_prefix = kv_prefixes.iter().copied().max().unwrap_or(0);
            let max_kv = kv_prefixes
                .iter()
                .zip(advances.iter())
                .map(|(&p, &a)| p + a)
                .max()
                .unwrap_or(0);
            let sum_kv: usize = kv_prefixes
                .iter()
                .zip(advances.iter())
                .map(|(&p, &a)| p + a)
                .sum();
            tracing::info!(
                target: "candle_conversation::scheduler::timing",
                n_seqs,
                total_tokens,
                fwd_ms,
                max_prefix,
                max_kv,
                sum_kv,
                "code-read prefill forward op breakdown: {}",
                parts.join("  ")
            );
        }

        for ((logits, &i), &advance) in logits_vec
            .into_iter()
            .zip(group_idxs.iter())
            .zip(advances.iter())
        {
            let p = &mut self.active_prefills[i];
            if let Err(e) = self.session.advance_sequence(p.work.sequence_id.0, advance) {
                p.error = Some(ConversationError::Model(e));
                continue;
            }
            // Mirror the just-prefilled tokens into the diagnostic
            // log so the turn-complete dump can reconstruct the
            // exact context the kernel saw — `run_one_prefill_pass`
            // is the SubmitTurn prefill path, parallel to
            // `run_prefill`'s synchronous path.
            let seq_id = p.work.sequence_id;
            let off = p.offset;
            let advance_tokens = p.work.tokens[off..off + advance].to_vec();
            super::Scheduler::record_slot_tokens(&mut self.slot_tokens, seq_id, &advance_tokens);
            p.offset += advance;
            // Staged prefill: if this pass landed on the next projection point,
            // emit that segment's projection (the pinned composition, spanned to
            // this segment's generated tokens) and move to the next point. The
            // client collects these `TurnEvent::Projection`s and persists them, so
            // the sealed turn carries the same per-segment projection sequence a
            // real decode produced.
            if let Some(&next_off) = p.work.projection_offsets.get(p.next_projection) {
                if p.offset >= next_off as usize {
                    if let Some(comp) = &p.work.staged_composition {
                        let gen_start = p.work.assistant_content_start;
                        let prev_off = if p.next_projection == 0 {
                            gen_start
                        } else {
                            p.work.projection_offsets[p.next_projection - 1]
                        };
                        // A projection is a POINT: this segment's projection was
                        // selected at `prev_off` and governs forward to `next_off`
                        // (the next event's point). Emit it at its start position.
                        let mut ev = comp.clone();
                        ev.start_token = prev_off.saturating_sub(gen_start);
                        let _ = p.work.event_tx.send(TurnEvent::Projection(ev));
                    }
                    p.next_projection += 1;
                }
            }
            let total = p.work.tokens.len();
            let _ = p.work.event_tx.send(TurnEvent::PrefillProgress {
                tokens_done: p.offset,
                tokens_total: total,
            });
            if p.offset >= total {
                p.final_logits = Some(logits);
            }
        }
    }

    /// Handle a device-OOM from the ragged prefill forward: the batch was too
    /// wide for the card. Narrow the admission window (so subsequent waves run
    /// smaller forwards), then — rather than failing the whole batch — **requeue**
    /// its scope-ingest prefills (upload / code-read) so their content isn't lost;
    /// a freed slot re-pumps them later at the narrower width, and the process
    /// self-tunes toward a sustainable batch size. Non-scope prefills (live
    /// dialogue turns) can't be transparently retried, so they surface the error
    /// on their caller channel as before.
    ///
    /// `group_idxs` are the `active_prefills` positions that were in this forward;
    /// they're still valid because nothing mutates `active_prefills` between the
    /// forward returning and this call.
    fn handle_prefill_oom(&mut self, group_idxs: &[usize], err: &candle::Error) {
        self.shrink_admit_window();
        let in_batch: HashSet<usize> = group_idxs.iter().copied().collect();
        let msg = format!("batched prefill forward failed: {err}");
        let drained = std::mem::take(&mut self.active_prefills);
        let mut kept = Vec::with_capacity(drained.len());
        for (i, mut p) in drained.into_iter().enumerate() {
            if in_batch.contains(&i) {
                if matches!(p.work.seal_action, SealAction::ScopeIngest) {
                    // Requeued via scope_pending — dropped from active_prefills.
                    self.requeue_scope_slot(p.work.sequence_id);
                    continue;
                }
                p.error = Some(ConversationError::Channel(msg.clone()));
            }
            kept.push(p);
        }
        self.active_prefills = kept;
    }

    /// Return a scope-ingest scratch slot to the pending queue so a later pump
    /// retries it (at the current, narrower admission window). Frees the slot's
    /// partial KV, undoes the fairness increment applied at admission, and pushes
    /// the scope back to the front of its file's queue. Mirrors the
    /// sequence-capacity requeue path in [`Scheduler::pump_scope_prefills`].
    fn requeue_scope_slot(&mut self, slot: SequenceId) {
        self.active_scope_slots = self.active_scope_slots.saturating_sub(1);
        if let Some(p) = self.pending_scope_prefills.remove(&slot) {
            if let Some(n) = self.scope_submitted.get_mut(&p.timeline) {
                *n = n.saturating_sub(1);
            }
            self.scope_pending
                .entry(p.timeline)
                .or_default()
                .push_front(QueuedScope {
                    scope_index: p.scope_index,
                    tokens: p.token_ids,
                    layout: p.layout,
                    token_count: p.token_count,
                    tags: p.tags.clone(),
                });
        }
        self.free_summary_slot(slot);
    }

    /// Drain finished or errored entries from `active_prefills`. Errored
    /// entries emit `TurnEvent::Error`; finished entries are passed to
    /// `finalise_prefill` (which samples the first token and inserts into
    /// `active_decodes`).
    pub(super) fn promote_finished_prefills_to_decodes(&mut self) {
        // Use swap_remove for efficiency; iterate from the back.
        let mut i = 0;
        while i < self.active_prefills.len() {
            let done = {
                let p = &self.active_prefills[i];
                p.error.is_some() || (p.final_logits.is_some() && p.offset >= p.work.tokens.len())
            };
            if !done {
                i += 1;
                continue;
            }
            let p = self.active_prefills.swap_remove(i);
            let ActivePrefill {
                work,
                offset: _,
                next_projection: _,
                final_logits,
                error,
                prefill_start,
            } = p;
            // A compression-turn re-prefill carries no decode and reports to the
            // summariser, not a caller. Seal it directly off the wave (snapshot
            // the role-coherent K/V + record the turn) instead of running
            // `finalise_prefill`.
            if let SealAction::CompressionTurn { job_id } = &work.seal_action {
                let job_id = *job_id;
                let slot = work.sequence_id;
                match error {
                    Some(e) => {
                        if let Some(p) = self.pending_compression_seals.remove(&job_id) {
                            let _ = p
                                .response_tx
                                .send(Err(crate::summary_tree::ProbeError::Soft(format!(
                                    "SubmitSummaryProbe: reproject prefill: {e}"
                                ))));
                        }
                        self.free_summary_slot(slot);
                    }
                    None => self.complete_compression_turn(slot, job_id),
                }
                continue;
            }
            // A code-scope ingest re-prefill finished on the wave. Snapshot its
            // K/V into its file batch (no decode, reports to the ingest caller via
            // the batch's per-scope channel); on error, fail just that scope so its
            // siblings still flush.
            if let SealAction::ScopeIngest = &work.seal_action {
                let slot = work.sequence_id;
                match error {
                    Some(e) => self.fail_scope_ingest(slot, e),
                    None => self.complete_scope_ingest(slot),
                }
                continue;
            }
            // A dialogue turn's reasoning-free re-prefill finished on the wave.
            // Seal the clean K/V + fire the deferred `Done` (no decode, reports to
            // the caller, not the summariser). On prefill error, surface it on the
            // caller channel and drop the slot's chunks.
            if let SealAction::TurnReprefill { pending_id } = &work.seal_action {
                let pending_id = *pending_id;
                match error {
                    Some(e) => {
                        if let Some(p) = self.pending_turn_seals.remove(&pending_id) {
                            let _ = p.event_tx.send(TurnEvent::Error(e));
                            let _ = self.session.truncate_sequence_to_blocks(p.parent_id.0, 0);
                        }
                    }
                    None => self.complete_turn_reprefill(pending_id),
                }
                continue;
            }
            if let Some(e) = error {
                let _ = work.event_tx.send(TurnEvent::Error(e));
                continue;
            }
            let logits = match final_logits {
                Some(l) => l,
                None => {
                    let _ = work
                        .event_tx
                        .send(TurnEvent::Error(ConversationError::Channel(
                            "prefill produced no final logits".into(),
                        )));
                    continue;
                }
            };
            let prefill_ms = prefill_start
                .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let turn_start = work.submitted_at;
            let token_count = work.tokens.len();
            self.finalise_prefill(work, logits, prefill_ms, turn_start, token_count);
            // swap_remove pulled the last element into i; don't increment.
        }
    }

    /// Post-forward path shared by both single and batched prefill: sample
    /// the first token, emit it, and either transition to decode or close
    /// the turn out immediately on EOS / max_decode_tokens == 0.
    fn finalise_prefill(
        &mut self,
        work: PrefillWork,
        logits: Tensor,
        prefill_ms: f64,
        turn_start: Instant,
        token_count: usize,
    ) {
        // Total KV position after this prefill.
        let context_depth = self
            .session
            .sequence_offset(work.sequence_id.0)
            .unwrap_or(token_count);

        // Decode-start line: the effective sampling config this conversation turn
        // will decode under. Confirms empirically whether a turn is stochastic
        // (temp>0 + top_k/top_p) or greedy (temp≈0 → argmax), and at what context
        // depth. Enable with
        // `RUST_LOG=candle_conversation::scheduler::decode=debug`.
        tracing::debug!(
            target: "candle_conversation::scheduler::decode",
            seq = work.sequence_id.0,
            context_depth,
            prefill_tokens = token_count,
            max_decode_tokens = work.max_decode_tokens,
            temperature = work.sampling.temperature,
            top_k = work.sampling.top_k,
            top_p = work.sampling.top_p,
            repeat_penalty = work.sampling.repeat_penalty,
            segment_temp_boost = work.sampling.segment_temp_boost,
            dry = work.sampling.dry.is_some(),
            greedy = work.sampling.temperature <= 0.01,
            seed = work.sampling.seed,
            "conversation decode start",
        );

        let mut sampling_state = self
            .sampling_states
            .remove(&work.sequence_id)
            .expect("sampling state must exist for active sequence");
        sampling_state.end_turn();
        sampling_state.record_context_tokens(&work.tokens, self.sampler.max_recent_len());

        // Send prefill progress: complete (single-prefill path needs this;
        // batched path already streams progress per-chunk, but a final
        // tokens_done==tokens_total event is always benign).
        let _ = work.event_tx.send(TurnEvent::PrefillProgress {
            tokens_done: token_count,
            tokens_total: token_count,
        });

        let first_token = match self.sample_single(&logits, &work.sampling, &mut sampling_state) {
            Ok(t) => t,
            Err(e) => {
                self.sampling_states
                    .insert(work.sequence_id, sampling_state);
                let _ = work.event_tx.send(TurnEvent::Error(e));
                return;
            }
        };

        // Detect think-mode entry: the model opens its OWN `<think>` as the first
        // decoded token (we never prefill one). The `work.tokens` check covers a
        // caller-supplied assistant prefill that itself opens a think block.
        let initial_inside_think_block = {
            let tid = work.sampling.segment_open_token_id;
            if tid >= 0 {
                let tok = tid as u32;
                let prefill_has_think = work.tokens.iter().rev().take(5).any(|&t| t == tok);
                // The block opens either way: the common case is the model
                // sampling its OWN `<think>` as the first token; the rarer case is
                // a caller-supplied assistant prefill that already opens one.  In
                // BOTH cases the sampler's `in_segment` must flip — it gates the
                // reflection-marker suppression, the thinking temperature boost,
                // and the `</think>` EOT ramp (all keyed off `segment_len`, which
                // only advances while `in_segment`).  (DRY is no longer gated
                // here — it has its own `dry_span_len`/`dry_suppressed` scope,
                // reset at `<think>`/`</think>` via `enter_segment`/`exit_segment`.)
                // Flipping it only for the prefilled case left the sampler's flag
                // stuck false for a model-opened block, silently disabling every
                // one of those controls for its whole duration even though the
                // health flag (`inside_think_block`) correctly tracked it.
                let opens_think = prefill_has_think || first_token == tok;
                if opens_think && !sampling_state.in_segment {
                    sampling_state.enter_segment();
                }
                opens_think
            } else {
                false
            }
        };

        self.sampling_states
            .insert(work.sequence_id, sampling_state);

        // Per-token trace for the prefill-emitted first token.  Enable
        // with `RUST_LOG=candle_conversation::scheduler::sampling=trace`.
        // This is the canonical "what did the model say first?" diag —
        // an early-EOS bug very often shows up as the first sampled
        // token already being EOS, meaning the model's K/V context is
        // pushing logits onto the EOS column straight out of prefill.
        if tracing::enabled!(
            target: "candle_conversation::scheduler::sampling",
            tracing::Level::TRACE,
        ) {
            let decoded = self
                .tokenizer
                .decode(&[first_token], false)
                .unwrap_or_else(|_| "<?>".to_string());
            let first_token_is_eos = self.is_eos(first_token);
            tracing::trace!(
                target: "candle_conversation::scheduler::sampling",
                seq_id = work.sequence_id.0,
                step = 0,
                token_id = first_token,
                is_eos = first_token_is_eos,
                decoded = %decoded,
                "sampled token (prefill first)",
            );
            if first_token_is_eos {
                tracing::debug!(
                    target: "candle_conversation::scheduler::sampling",
                    seq_id = work.sequence_id.0,
                    token_id = first_token,
                    "EOS fired on the very first sampled token — model is \
                     producing EOS immediately after prefill; check K/V \
                     context coherence",
                );
            }
        }

        let sampling_temperature = work.sampling.temperature;

        if self.is_eos(first_token) || work.max_decode_tokens == 0 {
            // View sequences (SubmitTurn path): the prefill already wrote KV
            // blocks that must be finalized onto the parent and sealed into
            // the substrate.  Insert as a finished DecodeState so
            // cleanup_finished runs finalize_view + perform_seal_and_write.
            //
            // Non-view sequences (raw RULER / summarisation): no parent to
            // finalize and seal=None is correct — use the fast path.
            if self.turn_views.contains_key(&work.sequence_id) {
                self.active_decodes.insert(
                    work.sequence_id,
                    DecodeState {
                        event_tx: work.event_tx,
                        generated_tokens: TokenBuffer::from(vec![first_token]),
                        max_tokens: work.max_decode_tokens,
                        sampling_config: work.sampling,
                        seal_action: work.seal_action,
                        post_decode_tokens: work.post_decode_tokens,
                        belief: work.belief,
                        prefill_tokens: work.tokens,
                        user_text: work.user_text,
                        tags: work.tags,
                        user_content_start: work.user_content_start,
                        user_content_end: work.user_content_end,
                        assistant_content_start: work.assistant_content_start,
                        no_think: work.no_think,
                        prefill_assistant_text: work.prefill_assistant_text,
                        finished: true,
                        decode_start: Instant::now(),
                        prefill_ms,
                        prefill_token_count: context_depth,
                        turn_start,
                        health: {
                            let mut hs = crate::decode_health::DecodeHealthState::new(
                                self.health_config.repetition_window,
                                self.health_config.health_log_capacity,
                            );
                            hs.apply_baseline_config(
                                self.health_config.entropy_baseline_window,
                                self.health_config.entropy_trend_relative_factor,
                                self.health_config.entropy_trend_absolute_min_nats,
                            );
                            hs.inside_think_block = initial_inside_think_block;
                            hs.skip_entropy_checks = sampling_temperature <= 0.01;
                            hs
                        },
                        reprojection: work.reprojection,
                        non_punct_since_reproject: 0,
                        last_projection_end: 0,
                        in_tool_call: false,
                        triggers: work.triggers,
                        stencil: None,
                        pending_mask: None,
                    },
                );
            } else {
                self.finish_immediately(
                    work.sequence_id,
                    first_token,
                    &work.event_tx,
                    prefill_ms,
                    turn_start,
                    context_depth,
                );
            }
            return;
        }

        let _ = work.event_tx.send(TurnEvent::Token(first_token));

        // The first sampled token can itself be a stencil trigger — e.g. the
        // model emits `<tool_call>` as its very first response token, the common
        // case under /no_think (the think block is prefilled, so the model goes
        // straight to the call). The decode-loop trigger check runs only on
        // tokens sampled in `batch_decode_step`, never this one, so check it here
        // too — otherwise steering silently never engages for those calls.
        let stencil = work.triggers.driver_for(first_token);
        if let Some(d) = &stencil {
            tracing::debug!(
                target: "candle_conversation::stencil",
                seq_id = work.sequence_id.0,
                tree = d.tree().label(),
                trigger = first_token,
                "stencil steering started (trigger on the first decoded token)",
            );
        }
        // A first-token `<tool_call>` trigger enters the call immediately, so the
        // in-call state must be set HERE — the decode loop's `is_tool_open` scan
        // (which normally sets it) only sees tokens sampled in `batch_decode_step`,
        // never this one. Without it the in-call reprojection freeze never engages
        // for these turns and cadence/punctuation triggers re-orient the selection
        // mid-call. The early first-reprojection push below still fires once — it
        // is this turn's lock-in reprojection, exactly like the one `is_tool_open`
        // fires before freezing.
        let first_token_opens_call = stencil
            .as_ref()
            .is_some_and(|d| d.tree().label() == super::TOOL_CALL_TREE_LABEL);
        // Captured before `work.reprojection` moves into the DecodeState: the
        // early first-reprojection below fires only for turns whose target
        // layer runs belief-driven selection — a plain-prompt layer (the
        // titler's single-section schema) gains nothing from the extra swap.
        let wants_early_reprojection = work
            .reprojection
            .as_ref()
            .is_some_and(|p| p.has_belief_collections());

        self.active_decodes.insert(
            work.sequence_id,
            DecodeState {
                event_tx: work.event_tx,
                generated_tokens: TokenBuffer::from(vec![first_token]),
                max_tokens: work.max_decode_tokens,
                sampling_config: work.sampling,
                seal_action: work.seal_action,
                post_decode_tokens: work.post_decode_tokens,
                belief: work.belief,
                prefill_tokens: work.tokens,
                user_text: work.user_text,
                tags: work.tags,
                user_content_start: work.user_content_start,
                user_content_end: work.user_content_end,
                assistant_content_start: work.assistant_content_start,
                no_think: work.no_think,
                prefill_assistant_text: work.prefill_assistant_text,
                finished: false,
                decode_start: Instant::now(),
                prefill_ms,
                prefill_token_count: context_depth,
                turn_start,
                health: {
                    let mut hs = crate::decode_health::DecodeHealthState::new(
                        self.health_config.repetition_window,
                        self.health_config.health_log_capacity,
                    );
                    hs.apply_baseline_config(
                        self.health_config.entropy_baseline_window,
                        self.health_config.entropy_trend_relative_factor,
                        self.health_config.entropy_trend_absolute_min_nats,
                    );
                    hs.inside_think_block = initial_inside_think_block;
                    hs.skip_entropy_checks = sampling_temperature <= 0.01;
                    hs
                },
                reprojection: work.reprojection,
                non_punct_since_reproject: 0,
                last_projection_end: 0,
                in_tool_call: first_token_opens_call,
                triggers: work.triggers,
                stencil,
                pending_mask: None,
            },
        );
        // Fire the turn's FIRST reprojection immediately (drained right after
        // the next decode step, ~token 1). The prefill just wrote the user
        // query's wide-Q into R16, so the belief scan can score it and
        // materialize the right sections BEFORE the model's plan forms in the
        // early <think> tokens — waiting for the 64-token cadence lets a
        // wrong-tool prefix anchor the reasoning first (the submit-time
        // projection only carries the PREVIOUS turn's belief; it cannot see
        // this turn's query). For a first-token tool call this is the turn's
        // lock-in reprojection: `in_tool_call` is already set above, so the
        // call body stays frozen afterwards.
        if wants_early_reprojection {
            Self::queue_reprojection(&mut self.pending_reprojections, work.sequence_id);
        }
    }

    pub(super) fn run_prefill(
        &mut self,
        sequence_id: SequenceId,
        tokens: &[u32],
    ) -> Result<Tensor, ConversationError> {
        // Chunked prefill: split large prompts into bounded chunks to keep
        // intermediate activation buffers from growing unboundedly.
        let logits = if tokens.len() > self.max_prefill_pass_tokens {
            let mut last_logits: Option<Tensor> = None;
            for chunk in tokens.chunks(self.max_prefill_pass_tokens) {
                let input = Tensor::new(chunk, &self.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(ConversationError::Model)?;
                let logits_vec = self
                    .model
                    .forward_batched(&mut self.session, &[sequence_id.0], &[input])
                    .map_err(ConversationError::Model)?;
                self.session
                    .advance_sequence(sequence_id.0, chunk.len())
                    .map_err(ConversationError::Model)?;
                super::Scheduler::record_slot_tokens(&mut self.slot_tokens, sequence_id, chunk);
                last_logits = logits_vec.into_iter().next();
            }
            last_logits.ok_or_else(|| {
                ConversationError::Channel("no logits returned from chunked prefill".into())
            })?
        } else {
            let input = Tensor::new(tokens, &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(ConversationError::Model)?;

            let logits_vec = self
                .model
                .forward_batched(&mut self.session, &[sequence_id.0], &[input])
                .map_err(ConversationError::Model)?;

            self.session
                .advance_sequence(sequence_id.0, tokens.len())
                .map_err(ConversationError::Model)?;

            // Mirror these tokens into the slot's diagnostic log so the
            // turn-complete dump can reconstruct the exact context the
            // kernel saw (compiled out without the `context-dump` feature).
            super::Scheduler::record_slot_tokens(&mut self.slot_tokens, sequence_id, tokens);

            logits_vec.into_iter().next().ok_or_else(|| {
                ConversationError::Channel("no logits returned from prefill".into())
            })?
        };

        // Single exit for every prefill path: the forward wrote KV without the
        // decode kernel's self-increment, so refresh the cached decode
        // slot-state's writer slice with the advanced tail length. Without
        // this, a mid-decode injection (a stencil static run, a think-steer
        // continuation) is INVISIBLE to the following decode steps — the
        // kernel attends the tail chunk at its stale pre-prefill length and
        // the model decodes as if the injected tokens were never written.
        // No-op for slots that haven't decoded yet.
        self.session
            .refresh_decode_slot_state(sequence_id.0)
            .map_err(ConversationError::Model)?;

        Ok(logits)
    }
}
