//! Prices a wave's buffers against the model geometry, so admission can size a
//! wave before any of it is built.
//!
//! `WavePlan` is cheap to copy and holds no state, so admission can call it for
//! each candidate. A phase costs the sum of each chain's buffers and the
//! maximum across chains, with preludes walked first, and each buffer is
//! charged one alignment because each is a separate aligned range. The tier is
//! priced the way the placement buys it, in whole regions with the launch
//! scratch included, since sizing by the largest single phase under-priced the
//! forward span that stays live across every layer. `ensure_fits` refuses a
//! wave that does not fit before any GPU work starts.

use super::*;

/// Prices a wave against the model's geometry.
///
/// Cheap to copy and free of interior state, so admission can hold one and call
/// it per candidate without synchronising anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavePlan {
    pub(super) geometry: ModelGeometry,
}

impl WavePlan {
    pub fn new(geometry: ModelGeometry) -> Self {
        Self { geometry }
    }

    /// Bytes for one named buffer — what a call site asks before allocating, so
    /// that what it takes is what it was priced for.
    pub fn bytes(&self, buffer: WaveBuffer, w: WaveWidth) -> usize {
        buffer.bytes(&self.geometry, w)
    }

    /// What one phase needs: the **sum** of every buffer it allocates.
    ///
    /// The sum, not the peak, because a generation hands out memory from a bump
    /// cursor that only rewinds when the guard drops. A buffer that dies early
    /// still holds its bytes until the end of the phase, so the high-water mark
    /// of *live* buffers would under-price the span and the first wide wave
    /// would exhaust it.
    ///
    /// (Reusing a dead buffer's bytes needs offsets assigned from declared
    /// lifetimes rather than a cursor. That was built, measured at ~1.75x on a
    /// MoE layer, and removed: it required buffer to hand-declare which step it
    /// died on, with nothing tying the declaration to the code — and operand
    /// provenance made the whole planned-slot path unnecessary, since a site now
    /// inherits its arena from its input instead of looking up an offset.)
    ///
    /// Each buffer is charged one alignment, since each is a separately aligned
    /// range.
    ///
    /// **Summed within a chain, maximised across them.** A phase can be opened
    /// by more than one kind of layer — on a hybrid, an attention mixer or a
    /// DeltaNet one — and a generation holds exactly one of them, so the span is
    /// sized by the largest rather than by their total. See [`Chain`].
    ///
    /// **Preludes first.** A prelude chain runs ahead of whichever other chain
    /// the layer opens the phase for, so the cursor walks the preludes and then
    /// each alternative from where they left it; the phase costs the largest of
    /// those walks.
    ///
    /// **Two alternatives walk differently.** The draft head's pass carves its
    /// input assembly, then the pre-mix, then the attention chain — its block
    /// is a trunk attention layer behind an assembly of its own. The PLE block
    /// opens its generation alone, with no pre-mix.
    pub fn phase_bytes(&self, phase: LayerPhase, w: WaveWidth) -> usize {
        let preludes = |cursor: usize| {
            Chain::iter()
                .filter(|c| c.phase() == phase && c.is_prelude())
                .fold(cursor, |cursor, c| self.walk_chain(cursor, c, w))
        };
        let prelude = preludes(0);
        Chain::iter()
            .filter(|c| c.phase() == phase && !c.is_prelude())
            .map(|c| match c {
                Chain::HeadPass => {
                    let assembled = self.walk_chain(0, Chain::HeadPass, w);
                    if assembled == 0 {
                        0
                    } else {
                        self.walk_chain(preludes(assembled), Chain::Attention, w)
                    }
                }
                Chain::Ple => self.walk_chain(0, Chain::Ple, w),
                _ => self.walk_chain(prelude, c, w),
            })
            .max()
            .unwrap_or(prelude)
    }

    /// What one chain's buffers cost, summed — see [`Self::phase_bytes`] for
    /// why a chain sums and a phase maximises.
    ///
    /// A buffer the geometry prices at zero is one this stack never allocates,
    /// so it takes no aligned range either and is charged nothing at all.
    ///
    /// **Walked as the cursor walks, not summed with each buffer rounded up.**
    /// A bump range starts at `aligned_start(base, cursor, align)` and then
    /// advances by its own length, so the alignment is paid on a buffer's
    /// *start* — which means the last buffer in a chain never pays for its
    /// tail, and a buffer whose length is already a multiple costs nothing
    /// extra at all.
    ///
    /// Rounding each length up instead is right for every chain whose shapes
    /// are 256-multiples — which is most of them, and why the census reports
    /// `0 B lost to alignment` there — and wrong wherever one is not. A
    /// DeltaNet wave of two multi-row spans carves span tables of 64 B and
    /// 16 B: charged rounded they cost 512, and the cursor spends 272.
    pub fn chain_bytes(&self, chain: Chain, w: WaveWidth) -> usize {
        self.walk_chain(0, chain, w)
    }

    /// The bump cursor after `chain`'s buffers are carved starting at `cursor`.
    pub(super) fn walk_chain(&self, cursor: usize, chain: Chain, w: WaveWidth) -> usize {
        WaveBuffer::iter()
            .filter(|b| b.chain() == chain)
            .map(|b| b.bytes(&self.geometry, w))
            .filter(|&len| len > 0)
            .fold(cursor, |cursor, len| {
                cursor.div_ceil(BUMP_ALIGNMENT) * BUMP_ALIGNMENT + len
            })
    }

    pub fn wave_bytes(&self, w: WaveWidth) -> usize {
        LayerPhase::iter()
            .map(|p| self.phase_bytes(p, w))
            .max()
            .unwrap_or(0)
    }

    /// What the transient tier will actually cost for `rows` — the
    /// [`tier_footprint`] of the phase spans, which is what `plan_wave_transient`
    /// buys: the attention and FFN spans share one region, the larger of the two,
    /// and the forward span stands beside it.
    ///
    /// **Not [`Self::wave_bytes`].** `wave_bytes` is the *max* over all three
    /// phases, the span one phase needs at a time. The forward span is live for
    /// the whole forward, across every layer phase, so it cannot share their
    /// ground; only the two layer phases, which never run at once, can. Sizing
    /// waves with the max under-priced the tier by the forward span and more:
    /// measured, waves were composed whose tier wanted 6.5 GiB against a
    /// 912 MiB guarantee, and were refused 62 times in one run.
    ///
    /// **Mirrors the arithmetic in the forward exactly** — the two must agree
    /// or the wave admitted is not the wave priced. That includes the absence
    /// of padding: the forward buys `phase_bytes` for each of the three phases
    /// and nothing more, so a `TARGET_ARENA_BYTES` pad here (there were two,
    /// matching a `+ REGION_BYTES` the forward has since dropped) would refuse
    /// 32 MiB of waves the placement would have taken.
    ///
    /// **Rounded up to a whole region, because the placement is.** The tier is
    /// carved in regions (`place_transient` rounds its length to one), so a
    /// plan whose phases sum to 160.5 MiB stands 176 MiB tall. Priced unrounded,
    /// the scheduler saw a 163 MiB gap as enough for the least chunk, bought
    /// nothing, and the placement was refused by one region — on every wave,
    /// 29,000 times in seven minutes, the same wave re-formed each time because
    /// nothing in its price had changed.
    ///
    /// **With the launchers' scratch, because the placement carries it.**
    /// `launch_scratch` is the reserve the forward span is widened by
    /// ([`launch_plan`]); priced without it, admission fitted a 32-row creep
    /// window whose 48 MiB tier the placement refused one region short, with
    /// the weight side at its floor, four times over (WhiteWorld, 2026-10-27).
    pub fn tier_bytes(&self, w: WaveWidth, launch_scratch: usize) -> usize {
        // `TARGET_ARENA_BYTES` is the region size read from the ungated source,
        // as `span_geometry` does — `region_pool::REGION_BYTES` is CUDA-only and
        // this plan is not.
        let raw = tier_footprint(launch_plan(
            [
                self.phase_bytes(LayerPhase::Attention, w),
                self.phase_bytes(LayerPhase::Ffn, w),
                self.phase_bytes(LayerPhase::Forward, w),
            ],
            launch_scratch,
        ));
        raw.div_ceil(TARGET_ARENA_BYTES) * TARGET_ARENA_BYTES
    }

    /// The most **prefill** rows that can ride on top of `head` and still leave
    /// the whole tier inside `budget`.
    ///
    /// The bound a wave is actually composed against. `head` is what the caller
    /// has already put in the wave — its decode rows and its sequence count —
    /// because those price the tier too, and a search that ignored them handed
    /// back a width the placement then refused by the head.
    ///
    /// Bisected rather than divided: the `div_ceil` steps make the cost a
    /// staircase, so dividing the budget by a per-row average lands inside a
    /// step and over-admits.
    pub fn max_prefill_rows_for_tier(
        &self,
        budget: usize,
        head: WaveWidth,
        launch_scratch: usize,
    ) -> usize {
        max_rows_within_cost(budget, |rows| {
            self.tier_bytes(head.with_one_more_sequence(rows), launch_scratch)
        })
    }

    /// The most **prefill** rows on top of `head` whose tier **and** the KV they
    /// write fit in `budget` together, at `kv_row_bytes` of KV per row.
    ///
    /// The bound a wave must be composed against when the tier and the KV are
    /// paid from the same ground — the KV side's free regions plus what the
    /// weight side would concede. Admit claims the wave's KV first and the tier
    /// is placed in what is left, so pricing each against the whole budget and
    /// taking the smaller admits a wave whose tier comes up short by exactly its
    /// own KV: measured on Qwen3.8-Flash-Next at BF16 ×8 on a 16 GB card, a
    /// 4,160-row wave whose tier fell 8 regions short — the ~108 MB its prompts
    /// wrote to the KV side.
    ///
    /// The KV is priced in whole regions, as arenas claim it.
    pub fn max_prefill_rows_for_tier_and_kv(
        &self,
        budget: usize,
        head: WaveWidth,
        kv_row_bytes: usize,
        launch_scratch: usize,
    ) -> usize {
        max_rows_within_cost(budget, |rows| {
            let kv = rows
                .saturating_mul(kv_row_bytes)
                .div_ceil(TARGET_ARENA_BYTES)
                .saturating_mul(TARGET_ARENA_BYTES);
            self.tier_bytes(head.with_one_more_sequence(rows), launch_scratch)
                .saturating_add(kv)
        })
    }

    /// The widest wave that fits in `budget` bytes — a **single phase's** bound.
    ///
    /// Not what a wave is sized by: the tier costs every phase together, which
    /// is [`Self::max_prefill_rows_for_tier`]. Kept for callers asking the
    /// narrower question of whether one phase's span holds a given width.
    ///
    /// Returns `0` when not even a single row fits, which admission must treat
    /// as a configuration error rather than as an empty wave: a budget that
    /// cannot price one token will never make progress, and silently admitting
    /// nothing would present as a hang.
    ///
    /// [`Self::wave_bytes`] is non-decreasing in `rows` (every term is a
    /// `div_ceil` of a product of it), so the largest fitting width is found by
    /// doubling to a bound that does not fit and bisecting. Exact rather than a
    /// closed form because the `div_ceil` steps make the cost a staircase, and
    /// dividing the budget by a per-row average would land inside a step and
    /// over-admit.
    pub fn max_rows_within(&self, budget: usize, head: WaveWidth) -> usize {
        if self.wave_bytes(head.with_one_more_sequence(1)) > budget {
            return 0;
        }
        let mut lo = 1usize;
        let mut hi = 2usize;
        while hi < ROW_SEARCH_CEILING && self.wave_bytes(head.with_one_more_sequence(hi)) <= budget
        {
            lo = hi;
            hi = hi.saturating_mul(2);
        }
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if self.wave_bytes(head.with_one_more_sequence(mid)) <= budget {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Whether a wave of width `w` fits in `budget`.
    pub fn fits(&self, w: WaveWidth, budget: usize) -> bool {
        self.wave_bytes(w) <= budget
    }

    /// Refuse a wave that does not fit, before any of it is assembled.
    ///
    /// The transient span already refuses an over-long bump, but by then the
    /// wave is part-built: earlier layers have launched, the half holds live
    /// ranges, and the failure names one buffer rather than the wave that was
    /// too wide. Pricing the whole wave up front turns that into one refusal at
    /// the gate, before any GPU work, with the width and the overage in the
    /// message.
    ///
    /// This is a hard error and it aborts the inference request. It is not a
    /// signal to trim the wave and continue: admission is supposed to have
    /// priced this wave against the same budget, so reaching here means the two
    /// disagree, and silently running a narrower wave would hide that
    /// disagreement for as long as it took to become a correctness bug.
    pub fn ensure_fits(&self, w: WaveWidth, budget: usize) -> candle::Result<()> {
        let cost = self.wave_bytes(w);
        if cost <= budget {
            return Ok(());
        }
        let head = WaveWidth {
            prefill_rows: 0,
            prefill_spans: 0,
            ..w
        };
        let widest = self.max_rows_within(budget, head);
        candle::bail!(
            "wave over budget: {} rows ({} prefill + {} decode, {} scored) need \
             {cost} B of transient span but the half holds {budget} B (over by {} B). \
             The widest prefill this budget admits beside that head is {widest} rows. \
             Admission priced this wave against the same plan, so this is an \
             accounting disagreement, not a wave to trim.\n{}",
            w.rows(),
            w.prefill_rows,
            w.decode_rows,
            w.scored_rows,
            cost - budget,
            self.describe(w)
        )
    }

    /// One line per phase and one per buffer within it — what to print when a
    /// span refuses, so an overflow names a shape rather than a number.
    pub fn describe(&self, w: WaveWidth) -> String {
        let mut out = format!(
            "wave plan @ {} prefill + {} decode rows, {} scored: {} B\n",
            w.prefill_rows,
            w.decode_rows,
            w.scored_rows,
            self.wave_bytes(w)
        );
        // Grouped by chain within the phase, because the phase is their max and
        // a flat list of buffers would not say which of them are alternatives.
        for phase in LayerPhase::iter() {
            out.push_str(&format!("  {phase:?}: {} B\n", self.phase_bytes(phase, w)));
            for chain in Chain::iter().filter(|c| c.phase() == phase) {
                out.push_str(&format!(
                    "    {chain:?}: {} B\n",
                    self.chain_bytes(chain, w)
                ));
                for buffer in WaveBuffer::iter().filter(|b| b.chain() == chain) {
                    out.push_str(&format!(
                        "      {buffer:?}: {} B\n",
                        buffer.bytes(&self.geometry, w)
                    ));
                }
            }
        }
        out
    }
}
