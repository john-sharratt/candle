//! The AIMD admission budget that bounds how much prefill a wave may admit,
//! and the per-pass token caps derived from it.
//!
//! The budget is cut multiplicatively on every throttle signal and raised one
//! quantum at a time when VRAM is not under pressure, so a transient episode's
//! backoff heals gradually instead of snapping back and re-tripping on the next
//! wide wave. A standing condition such as host RAM under its floor is cut at
//! most once per cooldown, so each cut's effect reaches the signal before the
//! next is decided, while a device OOM is an event and cuts at once. The pass
//! helpers never let a cap fall below one chunk, and the ground a wave's tier
//! can use is the frontier's regions rather than every free region, since holes
//! below the frontier are KV ground and cannot hold a tier.

use super::*;

impl Scheduler {
    /// Multiplicative-decrease the admission budget: halve it toward one quantum.
    /// Called from every throttle signal — VRAM pressure surviving an eviction
    /// pass, a prefill forward reporting device-OOM, host RAM under the ingest
    /// floor, the hot→warm drain falling behind. `reason` names which, and is
    /// carried into the throttle log so a collapsed budget is attributable.
    pub(super) fn cut_admit_budget(&mut self, reason: admission::ThrottleReason) {
        // Any cut invalidates the sustainable-budget evidence.
        self.admit_grow_streak = 0;
        let before = self.admit_budget;
        self.admit_budget = admission::cut_budget(self.admit_budget, admission::admit_quantum());
        if self.admit_budget != before {
            let ceiling = self.admit_budget_ceiling();
            self.log_throttle(reason, before, ceiling);
        }
    }

    /// Cut the budget for a STANDING CONDITION — host RAM under its floor, the
    /// hot→warm drain behind — rather than a discrete failure.
    ///
    /// Level signals are re-evaluated every wave and stay true until the
    /// condition clears, so an unguarded multiplicative cut collapses the
    /// setpoint to the floor in as many waves as it takes to halve: measured at
    /// 6144→256 MiB in **two seconds**, against a host-RAM condition that only
    /// moves on the scale of a drain pass. Rate-limiting to one cut per
    /// [`admission::LEVEL_CUT_COOLDOWN`] lets each cut's effect reach the signal before the
    /// next is decided.
    ///
    /// Device-OOM and relief-that-failed are EVENTS — they name a forward that
    /// already broke — and go straight to [`Self::cut_admit_budget`].
    pub(super) fn cut_admit_budget_leveled(&mut self, reason: admission::ThrottleReason) {
        let due = self
            .last_level_cut
            .is_none_or(|t| t.elapsed() >= admission::LEVEL_CUT_COOLDOWN);
        if !due {
            return;
        }
        self.last_level_cut = Some(std::time::Instant::now());
        self.cut_admit_budget(reason);
    }

    /// The setpoint's static upper bound — [`MAX_PREFILL_WIDTH`] quanta.
    /// A pure constant, so the reopen paths can cheaply ask "is there anything to
    /// reopen?" before paying for the live ceiling, which costs a device query
    /// (`QueryVideoMemoryInfo`) and a walk of the registered relievers.
    pub(super) fn max_admit_budget() -> u64 {
        MAX_PREFILL_WIDTH as u64 * admission::admit_quantum()
    }

    /// Additive-increase the admission budget by one quantum, toward the live
    /// ceiling. Called when VRAM is not under pressure, or on proven
    /// out-of-memory-free throughput under chronic nominal pressure, so a
    /// transient episode's backoff heals gradually (AIMD) rather than snapping
    /// straight back to full width and re-tripping on the next wide wave.
    pub(super) fn raise_admit_budget(&mut self, reason: admission::ThrottleReason) {
        let before = self.admit_budget;
        let ceiling = self.admit_budget_ceiling();
        self.admit_budget =
            admission::raise_budget(self.admit_budget, admission::admit_quantum(), ceiling);
        if self.admit_budget != before {
            self.log_throttle(reason, before, ceiling);
        }
    }

    /// Emit the setpoint-change event, in BOTH directions.
    ///
    /// Growth used to be silent while only shrinks logged, which made the
    /// controller's trajectory unreconstructable from a run: a budget that had
    /// climbed and collapsed several times was indistinguishable from one wedged
    /// at the floor the whole run. Every move is logged, with the reason that
    /// caused it and the live availability it was clamped against. Callers pass
    /// the `ceiling` they already measured rather than have this re-query it.
    pub(super) fn log_throttle(&self, reason: admission::ThrottleReason, was: u64, ceiling: u64) {
        const MIB: u64 = 1 << 20;
        tracing::debug!(
            target: "candle_conversation::scheduler::throttle",
            reason = reason.as_str(),
            budget_mib = self.admit_budget / MIB,
            was_mib = was / MIB,
            ceiling_mib = ceiling / MIB,
            prefills = self.active_prefills.len(),
            queued = self.prefill_queue.len(),
            "admission budget moved"
        );
    }

    /// Tokens one prefill forward may carry — see [`admission::prefill_pass_budget`].
    ///
    /// Read per forward rather than once at construction: the model's cap includes
    /// what the KV side can still hold, which moves with every claim. Bounded by
    /// the width the rate model chose for the cohort in flight
    /// ([`Self::rate_width`]).
    pub(in super::super) fn prefill_pass_budget(&self) -> usize {
        self.pass_budget_under(self.composition_target())
    }

    /// The pass budget **before** the rate model's choice: what an admission
    /// pass offers rows against, so that a pass can still find a width wider
    /// than the last one chose.
    pub(in super::super) fn prefill_pass_ceiling(&self) -> usize {
        self.pass_budget_under(self.max_prefill_pass_tokens)
    }

    /// Record the width the rate model chose — see [`Self::rate_width`].
    pub(in super::super) fn set_rate_width(&mut self, rows: usize) {
        self.rate_width = Some(RateWidth {
            rows: rows.max(CHUNK_SIZE),
            members: self.running_prefills(),
        });
    }

    /// Forget the rate model's width: a pass offered rows and every one paid.
    pub(in super::super) fn clear_rate_width(&mut self) {
        self.rate_width = None;
    }

    /// Forget the rate model's width once the cohort it was judged for has
    /// shrunk — see [`Self::rate_width`].
    pub(in super::super) fn settle_rate_width(&mut self) {
        if self
            .rate_width
            .is_some_and(|w| w.lapsed(self.running_prefills()))
        {
            self.rate_width = None;
        }
    }

    /// The target the forwards are composed against: the deployment's, or the
    /// rate model's narrower choice for the cohort in flight, and never wider
    /// than a width the partition has refused since ground last freed.
    pub(super) fn composition_target(&self) -> usize {
        self.composition_target_at(self.rate_width.map(|w| w.rows))
    }

    /// [`Self::composition_target`] had the rate model chosen `width` (`None`:
    /// no choice) — what admission prices a cohort at when its own pass is
    /// about to set or clear that choice.
    pub(in super::super) fn composition_target_at(&self, width: Option<usize>) -> usize {
        let target = width.map_or(self.max_prefill_pass_tokens, |w| {
            w.min(self.max_prefill_pass_tokens)
        });
        self.refused_width.map_or(target, |w| target.min(w.rows))
    }

    /// A wave of `rows` prefill rows was refused its tier, `short` bytes over
    /// the ground it had: compose the next narrower until that ground comes back. `false` when `rows` was already the least chunk, so a narrower
    /// wave would make no progress and the wave's members must be failed
    /// instead of retried.
    pub(in super::super) fn note_refused_width(&mut self, rows: usize, short: usize) -> bool {
        let ground = self.frontier_tier_bytes().unwrap_or(0);
        match RefusedWidth::after(rows, ground, short, CHUNK_SIZE) {
            Some(w) => {
                tracing::warn!(
                    refused_rows = rows,
                    next_rows = w.rows,
                    ground_mib = ground / (1024 * 1024),
                    short_mib = short / (1024 * 1024),
                    "the partition refused a wave's tier — composing the next one narrower",
                );
                self.refused_width = Some(w);
                true
            }
            None => false,
        }
    }

    /// Forget a refused width once the ground a tier can be placed in has grown
    /// by what the refused tier was short — see [`RefusedWidth::lapsed`].
    pub(in super::super) fn settle_refused_width(&mut self) {
        if let Some(w) = self.refused_width {
            if self.frontier_tier_bytes().is_none_or(|g| w.lapsed(g)) {
                self.refused_width = None;
            }
        }
        if let Some(d) = self.refused_depth {
            if self.frontier_tier_bytes().is_none_or(|g| d.lapsed(g)) {
                self.refused_depth = None;
            }
        }
    }

    /// A decode step drafting `depth` was refused its working memory: draft
    /// the next one shallower until the ground comes back. `false` when it
    /// drafted nothing, so there is no shallower step and it retries as it is.
    ///
    /// A refusal that names no shortfall (a full span, an outgrown arena) waits
    /// for a region back: a byte freed is not the step fitting.
    pub(in super::super) fn note_refused_depth(
        &mut self,
        depth: usize,
        err: &candle::Error,
    ) -> bool {
        let ground = self.frontier_tier_bytes().unwrap_or(0);
        let short = candle_nn::kv_cache::tier_refusal_shortfall(err)
            .unwrap_or(candle_nn::kv_cache::REGION_BYTES);
        match RefusedDepth::after(depth, ground, short) {
            Some(d) => {
                if self.refused_depth.is_none_or(|was| was.depth != d.depth) {
                    tracing::warn!(
                        refused_depth = depth,
                        next_depth = d.depth,
                        ground_mib = ground / (1024 * 1024),
                        short_mib = short / (1024 * 1024),
                        "the partition refused a speculative step — drafting the next one shallower",
                    );
                }
                self.refused_depth = Some(d);
                true
            }
            None => false,
        }
    }

    /// The model's draft ceiling at `width`, bounded by a depth the partition
    /// has refused since ground last freed.
    pub(in super::super) fn draft_ceiling(&self, width: usize) -> usize {
        let ceiling = self.model.draft_budget(width);
        self.refused_depth.map_or(ceiling, |d| ceiling.min(d.depth))
    }

    /// The ground a wave's tier can stand on: the regions between the arena
    /// frontier and the weight floor, and what the weight side would concede
    /// past it.
    ///
    /// **Not [`ManagedBatchedModel::prefill_ground_bytes`]**, which counts every
    /// free region — holes below the frontier too. Those are KV ground, never a
    /// tier's: the tier stands at the frontier. A refused width recorded and
    /// lapsed against them lapsed with the tier no closer to fitting. Measured on
    /// the RTX 3090 (2026-10-18): 3,450 rows refused, 2,588 refused, then 3,450
    /// composed again, 4,210 times over 27 s, the frontier never moving.
    pub(in super::super) fn frontier_tier_bytes(&self) -> Option<usize> {
        let stats = self.session.kv_region_stats()?;
        Some(admit::pass_budget::frontier_tier_ground(
            stats.total,
            stats.live_watermark,
            self.model.reclaimable_kv_bytes(),
            candle_nn::kv_cache::REGION_BYTES,
        ))
    }

    pub(super) fn pass_budget_under(&self, target: usize) -> usize {
        admit::pass_budget::prefill_pass_budget(
            target,
            self.model
                .prefill_width_cap(self.session.activation_dtype()),
            self.prefill_kv_token_cap(),
            // A tier budget that prices to a single row would make no progress,
            // so the cap never falls below one chunk.
            CHUNK_SIZE,
        )
    }

    /// Tokens a prefill group of `total` queued tokens shares — the pass
    /// budget, or all of `total` when it passes the model's cap by no more than
    /// the model's own slack. See [`admit::pass_budget::prefill_group_budget`].
    ///
    /// **Priced for the whole sweep.** The group is a creep cohort whose width
    /// holds until its head, and every window places its tier on the ground
    /// left then — so the cap is sized against the ground less the KV the
    /// running decodes will add over a sweep ([`sweep_forecast`]), not against
    /// the ground standing now.
    pub(in super::super) fn prefill_group_budget(&self, total: usize) -> usize {
        self.prefill_group_budget_at(total, self.composition_target())
    }

    /// [`Self::prefill_group_budget`] against an explicit composition target —
    /// see [`Self::composition_target_at`].
    pub(in super::super) fn prefill_group_budget_at(&self, total: usize, target: usize) -> usize {
        let dtype = self.session.activation_dtype();
        let withheld = self
            .model
            .kv_bytes_per_row(dtype)
            .map_or(0, |row| self.sweep_forecast.withheld_bytes(row));
        admit::pass_budget::prefill_group_budget(
            total,
            target,
            self.model.prefill_width_cap_withholding(dtype, withheld),
            self.prefill_kv_token_cap(),
            CHUNK_SIZE,
        )
    }

    /// Tokens the KV side's free ground can still back.
    ///
    /// The KV side's own bound: the admit phase claims every chunk a forward
    /// will write before it computes anything, so a chunk wider than the free
    /// ground can back fails part way through claiming.
    ///
    /// Priced through `kv_token_cap`, which is the only thing here that knows
    /// the units. The KV side counts 16 MiB regions and this budget is in
    /// tokens; `vram_budget_available` is that same free count in bytes
    /// (`(free + blocked) × REGION_BYTES`, so it reads the ground a standing
    /// tier releases before these claims run, exactly as `kv_region_state`
    /// does), and the block price turns bytes into tokens.
    pub(super) fn prefill_kv_token_cap(&self) -> Option<usize> {
        self.session
            .vram_budget_available()
            .and_then(|free| admit::pass_budget::kv_token_cap(free, self.per_block_kv_bytes()))
    }
}
