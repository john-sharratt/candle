//! The scheduler's per-window wave statistics: forward and phase counters,
//! aggregated over a window and flushed as one log line and one telemetry sample.
//!
//! The phase bands of a window are disjoint and sum to its wall-clock. Sub-slices
//! such as reprojection, sealing and eviction are carved out of the timed quanta
//! that contain them rather than added on top. A forward is recorded inside the
//! forward, but the window is flushed only by the loop once per iteration, because
//! a flush mid-forward would leak the forward's time into the unaccounted
//! remainder. The accumulators hold microseconds so that a short request is not
//! truncated to zero, and idle waits are timed so that an idle card reads as idle
//! and not as blocked.

use super::*;

#[derive(Default)]
pub(super) struct WaveChannel {
    pub(super) fwds: u64,
    pub(super) seq_sum: u64,
    pub(super) seq_max: usize,
    pub(super) tok_sum: u64,
    /// Σ attended-KV length (each forward's prefix/context length, summed over
    /// the batch) over the window. `kv_sum/fwds` is the avg KV a forward swept
    /// — the number to watch for a context-growth slowdown.
    pub(super) kv_sum: u64,
    pub(super) ms_sum: u64,
}

/// Periodic aggregator for forward-pass batch ("wave") sizes. Lets us see
/// whether the scheduler is actually batching wide under load without
/// per-forward log spam — it emits a single INFO line at most every 2 s and
/// resets. Single-threaded (scheduler thread only), so no synchronization.
pub(super) struct WaveStats {
    pub(super) window_start: std::time::Instant,
    pub(super) prefill: WaveChannel,
    pub(super) decode: WaveChannel,
    /// Section-ingest forwards (startup code-read / repo-map prefill). These run
    /// through `forward_batched` like prefills but on the section-ingest path, so
    /// they get their own channel rather than being invisible behind `section_ms`.
    pub(super) section: WaveChannel,
    /// Per-phase wall-clock spent on the scheduler thread this window, in ms.
    /// These account for where the wall-clock goes when it is NOT a forward:
    /// `drain` includes the synchronous per-turn SubmitTurn handling
    /// (projection + elevate + apply_segments gap-fill + view create), `reproj`
    /// is the mid-decode continuous reprojection drain, the rest are the
    /// forward quanta themselves. A wave whose elapsed ≫ Σ(phase) was blocked
    /// off-thread (e.g. waiting on the persistence thread / a lock).
    pub(super) drain_ms: u64,
    pub(super) promote_ms: u64,
    pub(super) decode_ms: u64,
    pub(super) prefill_ms: u64,
    pub(super) section_ms: u64,
    pub(super) reproj_ms: u64,
    /// Per-scope code-read seal sub-slices of the Prefill phase (microseconds,
    /// summed over the window). These are NOT separate phases — they live inside
    /// `prefill_ms` — but split the non-forward seal cost so we can see whether
    /// the wave-dominating overhead is the GPU snapshot, the wide-Q provenance
    /// sig gather, or the record/compress/persist flush. `seal_count` is scopes
    /// sealed this window.
    pub(super) seal_snapshot_us: u64,
    pub(super) seal_sig_us: u64,
    pub(super) seal_flush_us: u64,
    pub(super) seal_count: u64,
    /// Eviction phase this window: resident KV bytes freed by the relief ladder
    /// (cold-tail evict + ingest demote + footprint reclaim), the residence count,
    /// and the wall-clock spent doing it. Fed the GUI's phase timeline.
    pub(super) evict_bytes: u64,
    pub(super) evict_count: u64,
    pub(super) evict_ms: u64,
    /// Idle phase this window: wall-clock the loop spent blocked on `rx.recv()`
    /// with NO work to run — waiting for the next request. Carved out of the
    /// window remainder so it isn't mislabeled as "blocked" (which is reserved for
    /// off-thread stalls *during* active work).
    pub(super) idle_ms: u64,
    /// Requests phase this window: wall-clock the loop spent INSIDE
    /// [`Scheduler::handle_request`] — slot creation (with its snapshot/belief
    /// restore), branch-state installs, section restores, turn submissions.
    ///
    /// This is scheduler-thread work between forwards, not a stall, and it used
    /// to be invisible: only the `rx.recv()` block around it was timed, so every
    /// request the scheduler serviced fell into the unattributed remainder and
    /// the dashboard drew it as "blocked". A phase that ingests thousands of
    /// scopes services thousands of requests, so that remainder was large and
    /// unexplained — one `InstallRecurrentState` alone measured 17 ms of device
    /// scatter, ×792 in a single calibration pass.
    pub(super) requests_us: u64,
    /// Housekeeping phase this window: the per-wave work that runs between the
    /// timed quanta and the next iteration — promoting finished prefills,
    /// regulating ingest admission, demoting cold ingest under pressure, walking
    /// the AIMD admit budget, and harvesting GPU spans.
    ///
    /// Real scheduler work, none of it inside a quantum, and none of it timed
    /// before — so it landed in the unattributed remainder and drew as "blocked".
    pub(super) housekeeping_us: u64,
    /// Quantize phase this window: the K/V quantization the loop runs between
    /// forwards — the last gap's catch-up, this gap's launch, and the native
    /// residence quantized a gap (`docs/kv_quantize_in_the_gap.md`). Outside
    /// every quantum, so untimed it drew as "blocked". Microseconds, as
    /// [`Self::requests_us`].
    pub(super) quantize_us: u64,
    /// Compaction phase this window: KV pool compaction between forwards,
    /// drawn apart from the rest of the housekeeping. Microseconds.
    pub(super) compaction_us: u64,
    /// Warm phase this window: the background scan the loop runs a pass
    /// (`step_norm_warm`), outside every quantum. Microseconds.
    pub(super) warm_us: u64,
    /// What the warm band was spent on, by kind of launch, logged at the
    /// window's close.
    pub(super) warm_split: warm_split::WarmSplit,
    /// What the compaction band was spent on, logged at the window's close.
    pub(super) compaction_split: compaction_split::CompactionSplit,
    /// What the allocation band was spent on, logged at the window's close.
    pub(super) admission_split: admission_split::AdmissionSplit,
    /// Telemetry phase this window: the previous window's flush, which runs
    /// after that window closed and so lands in this one. Microseconds.
    pub(super) telemetry_us: u64,
    /// The admission pass run mid-quantum, inside the decode quantum: a
    /// sub-slice of decode's time, carved out of it into the allocation band.
    /// Microseconds.
    pub(super) mid_promote_us: u64,
    /// Sync phase this window: deliberate GPU/persistence waits ([`WAIT_US`],
    /// swapped in at flush). Carved out of the compute phases + blocked so a
    /// device-sync / hot→warm flush stall is its own band, not hidden decode/prefill.
    pub(super) wait_ms: u64,
    /// Persistence-maintenance stall this window ([`MAINT_US`], swapped in at
    /// flush) — a segment compaction blocking the scheduler's seal writes. Folded
    /// into the Sync band but carved blocked-only (never spills into compute).
    pub(super) maint_ms: u64,
    /// Drain sub-timers (ms) + prefilled tokens, swapped from the `DRAIN_*` atoms.
    /// `prefill` (time + tokens) is re-attributed OUT of the Projection/drain band
    /// INTO the Prefill phase so the ingest prefill shows real throughput; `elevate`
    /// and `glue` decompose the remaining drain for the projection panel.
    pub(super) drain_prefill_ms: u64,
    pub(super) drain_prefill_tokens: u64,
    pub(super) drain_elevate_ms: u64,
    pub(super) drain_glue_ms: u64,
    /// The prefill and section shares of the co-batched forwards inside the
    /// decode quantum, in microseconds — see [`Self::close_decode_step`].
    pub(super) cobatched_prefill_us: u64,
    pub(super) cobatched_section_us: u64,
    /// The `(decode, prefill, section)` rows of the decode step in flight's
    /// co-batched forwards, split by [`Self::close_decode_step`].
    pub(super) step_rows: (usize, usize, usize),
    /// Tokens the window's decode steps committed — the decode phase's tokens
    /// ([`Self::add_decode_committed`]).
    pub(super) decode_committed: u64,
    /// Set for the span of the decode quantum, so a co-batched split is taken
    /// only from a forward whose time decode's timer holds — the same wave step
    /// also runs from the prefill quantum.
    pub(super) in_decode_quantum: bool,
    /// The expert pipeline's counters at the previous flush, so each wave line
    /// reports this window's own — `None` on a dense model.
    pub(super) experts_prev: Option<ExpertCounters>,
    /// The device's wave-capture counters at the previous flush, so the wave
    /// line reports this window's own — `None` off CUDA.
    pub(super) graphs_prev: Option<CaptureStats>,
    /// This window's waves: how many, their wall-clock and its overhead.
    pub(super) timing: WaveTiming,
}

impl WaveStats {
    pub(super) fn new() -> Self {
        Self {
            window_start: std::time::Instant::now(),
            prefill: WaveChannel::default(),
            decode: WaveChannel::default(),
            section: WaveChannel::default(),
            drain_ms: 0,
            promote_ms: 0,
            decode_ms: 0,
            prefill_ms: 0,
            section_ms: 0,
            reproj_ms: 0,
            seal_snapshot_us: 0,
            seal_sig_us: 0,
            seal_flush_us: 0,
            seal_count: 0,
            evict_bytes: 0,
            evict_count: 0,
            evict_ms: 0,
            idle_ms: 0,
            requests_us: 0,
            housekeeping_us: 0,
            quantize_us: 0,
            compaction_us: 0,
            warm_us: 0,
            warm_split: warm_split::WarmSplit::default(),
            compaction_split: compaction_split::CompactionSplit::default(),
            admission_split: admission_split::AdmissionSplit::default(),
            telemetry_us: 0,
            mid_promote_us: 0,
            wait_ms: 0,
            maint_ms: 0,
            drain_prefill_ms: 0,
            drain_prefill_tokens: 0,
            drain_elevate_ms: 0,
            drain_glue_ms: 0,
            cobatched_prefill_us: 0,
            cobatched_section_us: 0,
            step_rows: (0, 0, 0),
            decode_committed: 0,
            in_decode_quantum: false,
            experts_prev: None,
            graphs_prev: None,
            timing: WaveTiming::default(),
        }
    }

    /// Count one wave of the run loop — see [`WaveTiming`].
    pub(super) fn add_wave(&mut self, wall_us: u64, overhead_us: u64) {
        self.timing.add(wall_us, overhead_us);
    }

    /// Note the rows one co-batched forward carried, for the decode step it
    /// rode to be split by ([`Self::close_decode_step`]). Only a forward inside
    /// the decode quantum is noted: the same wave step also runs from the
    /// prefill quantum with no decode rows, its time already prefill's, and
    /// sharing that one too counted it twice — a 5.2 s window drew a 9.4 s
    /// prefill band.
    pub(super) fn record_cobatched_split(
        &mut self,
        decode_rows: usize,
        prefill_rows: usize,
        section_rows: usize,
    ) {
        if !self.in_decode_quantum {
            return;
        }
        let (d, p, s) = self.step_rows;
        self.step_rows = (d + decode_rows, p + prefill_rows, s + section_rows);
    }

    /// Split one decode step's wall-clock between the classes that rode it
    /// (`phase_split::cobatched_step_shares`), given what a decode-only step
    /// takes.
    ///
    /// The step ran inside the decode quantum, so all of it is already in
    /// `decode_ms`; this records how much of it was prefill's and section's
    /// work, which the phase window moves out of decode. A step cannot be spent
    /// twice: a 3-row decode beside an 8K-token prefill shared one 4.6 s sweep,
    /// and charging that sweep to both drew a decode bar the length of the whole
    /// window at ~12 tok/s.
    pub(super) fn close_decode_step(&mut self, step_us: u64, decode_only_us: Option<u64>) {
        let (d, p, s) = std::mem::take(&mut self.step_rows);
        let (pf, sc) = cobatched_step_shares(step_us, decode_only_us, d, p, s);
        self.cobatched_prefill_us += pf;
        self.cobatched_section_us += sc;
    }

    /// Count the tokens a decode step committed — what the decode phase
    /// delivered, which a step's rows overstate by every draft rejected.
    pub(super) fn add_decode_committed(&mut self, tokens: u64) {
        self.decode_committed += tokens;
    }

    /// Accumulate one eviction event (relief-ladder shed of resident KV).
    pub(super) fn add_evict(&mut self, bytes: u64, count: u64, ms: u64) {
        self.evict_bytes += bytes;
        self.evict_count += count;
        self.evict_ms += ms;
    }

    /// Accumulate one idle wait — the loop blocked on `rx.recv()` with no work.
    pub(super) fn add_idle(&mut self, ms: u64) {
        self.idle_ms += ms;
    }

    /// Accumulate one request-handling span — see [`Self::requests_us`].
    /// Accumulated in MICROSECONDS. A single request is routinely sub-millisecond,
    /// so millisecond spans truncated to 0 and thousands of them summed to
    /// nothing — the band stayed empty and its time kept reading as `Blocked`,
    /// which is the exact defect it was added to fix.
    pub(super) fn add_requests(&mut self, us: u64) {
        self.requests_us += us;
    }

    /// Accumulate one per-wave housekeeping span, in MICROSECONDS — same
    /// truncation hazard as [`Self::add_requests`].
    pub(super) fn add_housekeeping(&mut self, us: u64) {
        self.housekeeping_us += us;
    }

    /// Accumulate one gap-quantize span — see [`Self::quantize_us`].
    pub(super) fn add_quantize(&mut self, us: u64) {
        self.quantize_us += us;
    }

    /// Accumulate one compaction pass's span — see [`Self::compaction_us`].
    pub(super) fn add_compaction(&mut self, us: u64) {
        self.compaction_us += us;
    }

    /// Accumulate one background scan's span — see [`Self::warm_us`].
    pub(super) fn add_warm(&mut self, us: u64) {
        self.warm_us += us;
    }

    /// Charge the window just begun with the flush that closed the last one —
    /// see [`Self::telemetry_us`].
    pub(super) fn add_telemetry(&mut self, us: u64) {
        self.telemetry_us += us;
    }

    /// Accumulate one mid-quantum admission pass — see [`Self::mid_promote_us`].
    pub(super) fn add_mid_promote(&mut self, us: u64) {
        self.mid_promote_us += us;
    }

    /// Accumulate one scope-ingest seal's sub-step timings (microseconds).
    pub(super) fn add_seal(&mut self, snapshot_us: u64, sig_us: u64, flush_us: u64) {
        self.seal_snapshot_us += snapshot_us;
        self.seal_sig_us += sig_us;
        self.seal_flush_us += flush_us;
        self.seal_count += 1;
    }

    /// Record one forward. `n_tokens` is the total tokens in the batch
    /// (seqs × per-seq chunk for prefill, seqs × 1 for decode). `kv_len` is the
    /// total attended-KV length the forward swept (Σ per-seq prefix/context
    /// length, captured before the forward advanced the sequences).
    pub(super) fn record(
        &mut self,
        prefill: bool,
        n_seqs: usize,
        n_tokens: usize,
        kv_len: usize,
        fwd_ms: u64,
    ) {
        let ch = if prefill {
            &mut self.prefill
        } else {
            &mut self.decode
        };
        ch.fwds += 1;
        ch.seq_sum += n_seqs as u64;
        ch.seq_max = ch.seq_max.max(n_seqs);
        ch.tok_sum += n_tokens as u64;
        ch.kv_sum += kv_len as u64;
        ch.ms_sum += fwd_ms;
        // NB: do NOT flush here. Flushing mid-forward (record runs inside the
        // forward) would fire before the enclosing `timed_*` wrapper attributes
        // the phase ms, leaking the forward's wall-clock into `unaccounted`.
        // The run loop calls `flush_if_due` once per iteration, after all phase
        // attribution for that iteration is complete.
    }

    /// Record one co-batched section-ingest advance (its own throughput channel so
    /// the section panel isn't invisible behind decode). `fwd_ms` is the shared
    /// co-batch wave-step time — sections ride decode's sweep, so this reflects the
    /// concurrent throughput, not a serial section forward.
    pub(super) fn record_section(
        &mut self,
        n_seqs: usize,
        n_tokens: usize,
        kv_len: usize,
        fwd_ms: u64,
    ) {
        let ch = &mut self.section;
        ch.fwds += 1;
        ch.seq_sum += n_seqs as u64;
        ch.seq_max = ch.seq_max.max(n_seqs);
        ch.tok_sum += n_tokens as u64;
        ch.kv_sum += kv_len as u64;
        ch.ms_sum += fwd_ms;
    }

    /// Accumulate wall-clock spent in a named scheduler-loop phase.
    pub(super) fn add_phase(&mut self, phase: WavePhase, ms: u64) {
        match phase {
            WavePhase::Drain => self.drain_ms += ms,
            WavePhase::Promote => self.promote_ms += ms,
            WavePhase::Decode => self.decode_ms += ms,
            WavePhase::Prefill => self.prefill_ms += ms,
            WavePhase::Section => self.section_ms += ms,
            WavePhase::Reproject => self.reproj_ms += ms,
        }
    }

    /// Whether the [`WAVE_WINDOW`] has elapsed and [`Self::flush`] should run.
    /// Split from `flush` so the caller only pays for the VRAM query (a couple
    /// of CUDA driver calls) on the wave it actually emits, not every loop
    /// iteration.
    pub(super) fn due(&self) -> bool {
        self.window_start.elapsed() >= WAVE_WINDOW
    }

    /// How long until [`Self::due`] — zero once it is. What an idle wait blocks
    /// for at most, so an idle card's windows close on time too.
    pub(super) fn until_due(&self) -> Duration {
        WAVE_WINDOW.saturating_sub(self.window_start.elapsed())
    }
}

/// Named scheduler-loop phases for [`WaveStats::add_phase`].
#[derive(Clone, Copy)]
pub(super) enum WavePhase {
    Drain,
    Promote,
    Decode,
    Prefill,
    Section,
    Reproject,
}
