//! Creates the device reservation and lays out its regions and fixed blocks.
//!
//! The reservation is claimed once, before the model loads, so the weights can
//! be carved from its left side and the KV regions fill what remains. The load's
//! scratch comes from the top of the span. That headroom is conceded only from a
//! measured card, and never below the span minimum. The reservation records the
//! device's memory in use before it was claimed, so the memory report can
//! separate the driver's own working set from an unexplained residual. Epochs
//! start at one, so that zero can mean a region has never been dirtied.

use super::*;

impl RegionPool {
    /// The span, with its top `load_headroom` bytes opened as the load's scratch.
    ///
    /// **The headroom exists because loading a weight costs more than storing
    /// it.** `repack_ko` reads the quantized source and writes the KO twin
    /// through an F32 intermediate. Claiming the span before the model loads is
    /// what puts the weights *inside* it, and it would otherwise leave the pool
    /// with only its cushion and fail on the largest repack.
    ///
    /// So the caller measures the peak from the checkpoint — the largest source
    /// tensor plus the repack's two f32 bands, knowable from the GGUF header
    /// before a byte is read — and that much of the span's top serves the load's
    /// transients (`load_scratch`) until `freeze_dense` gives it to the weight
    /// side. Not an estimate with a safety factor: an arithmetic bound on the
    /// allocations that dominate.
    ///
    /// The band is what makes this a small number. The intermediate used to be
    /// the whole tensor in F32 — 4,850 MiB for the 27B's head, permanently out
    /// of a span that cannot be grown — and banding the repack made it constant
    /// in tensor size. It is bounded, not zero: `peak_load_pool_bytes` carries
    /// the `2 × REPACK_BAND_BYTES` term, and dropping it OOMs the first large
    /// repack.
    pub(super) fn create_with_headroom(
        stream: &std::sync::Arc<CudaStream>,
        load_headroom: usize,
    ) -> Result<Self> {
        let ordinal = stream.context().ordinal();
        // Read before the claim: mapping granules consumes headroom, so asking
        // afterwards reports what is left rather than what was available.
        let usable_before = candle::vram::get(ordinal)
            .and_then(|g| g.usable().ok())
            .unwrap_or(0) as usize;
        // **Headroom is conceded only from a MEASURED card.** Without a VRAM
        // governor `span_target` answers with the fixed `TEST_SPAN_BYTES`
        // constant, which is not a measurement of anything — there is no
        // capacity the pool could be left room in, so subtracting from it takes
        // KV regions to buy nothing.
        //
        // Both halves of getting this wrong were observed within one test.
        // Subtracting unconditionally took Qwen3-8B's 2.4 GiB `output.weight`
        // peak off the test constant and left 640 MiB against a 976 MiB
        // minimum, refusing the reservation outright. Clamping the subtraction
        // at the minimum stopped the refusal and produced the same failure one
        // step later — a span of exactly the floor, `0 regions`, and a model
        // that could not claim a single KV block.
        //
        // The clamp stays as well, for the measured path: a card whose capacity
        // is genuinely close to the floor must keep its floor rather than
        // concede into it. The headroom prevents a *possible* pool OOM during
        // load and is never worth a *certain* refusal at startup.
        // What the executable library may hold outside the span — read on this
        // card, so the library and the span agree (`graph_budget_bytes`).
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| candle::Error::Msg(format!("reservation: bind: {e}")))?;
        let graph_reserve = graph_budget_bytes()? as usize;
        let held_bytes = BOOTSTRAP_HELD_BYTES;
        let persist_bytes = PERSIST_SPAN_BYTES;
        let span_minimum = span_minimum_bytes(persist_bytes);
        let target = span_target(ordinal, graph_reserve)?;
        let measured = candle::vram::get(ordinal).is_some();
        let concedable = target.saturating_sub(span_minimum);
        // **The load's scratch is the top of the span, not ground beside it.** The span is
        // one device allocation that never grows, so headroom left outside it is the
        // driver's for the life of the process. Inside it, the top `scratch` bytes are
        // nobody's while the model loads — the dense block grows from the left, and the
        // weight side is placed from the right only once the block is frozen — so they
        // serve the load's transients (`load_scratch`) and go to the weight side at
        // `freeze_dense` (`docs/driver_model.md` §4).
        let scratch = if measured {
            load_headroom
                .min(concedable)
                .next_multiple_of(REGION_BYTES)
                .min((concedable / REGION_BYTES) * REGION_BYTES)
        } else {
            0
        };
        let want = (target / REGION_BYTES) * REGION_BYTES;
        // **Everything on the card that is not ours, measured while that is
        // still true.** Taken before the reservation and therefore before the
        // weight load: the CUDA context, the driver's working set, the module
        // images, and any other process. After this line the card holds our span
        // and the two can never be separated again by subtraction.
        //
        // The memory report otherwise has to publish `in_use − span − pool` and
        // call it "unaccounted", which buries a real leak inside a large
        // constant — it read 1.4 GB on a 16 GB card while being almost entirely
        // this baseline.
        // **`cuMemGetInfo`, because that is what the residual subtracts it
        // from.** The memory report derives `device_in_use_bytes` as
        // `driver_total − driver_free` from this same call, so the baseline has
        // to come from the same scale or the subtraction is meaningless.
        //
        // The obvious alternative — `vram::get(ordinal).measure()` — is wrong
        // here and was: on Windows that reads DXGI, whose `headroom` is relative
        // to a driver *budget* rather than to installed VRAM, and which the
        // governor's own docs forbid subtracting for exactly this reason.
        // Mixing the two drove `unaccounted_bytes` persistently negative, the
        // opposite of what the field exists to show.
        // Bound to this thread first, exactly as `CudaDevice::mem_get_info` does.
        // The driver call reads the CURRENT context, so without the bind it either
        // answers for whichever context the thread last touched or fails — and
        // either way it leaves the thread in a state the reservation below then
        // has to work from.
        let pre_reservation_in_use = stream
            .context()
            .bind_to_thread()
            .ok()
            .and_then(|()| candle::cuda_backend::cudarc::driver::result::mem_get_info().ok())
            .map(|(free, total)| (total as u64).saturating_sub(free as u64));
        let mut reservation = Reservation::reserve(stream, want)?;

        // Claim from the left. An allocation the card granted short of `want`
        // ends the span there, and because every other block is positioned
        // relative to the right edge, the shortfall lands on the region count —
        // the one tenant that degrades gracefully.
        let claimed = reservation.map_range(0, want)?;
        let span_bytes = (claimed / REGION_BYTES) * REGION_BYTES;
        let span_base = reservation.base();
        let span_end = span_base + span_bytes as u64;
        if span_bytes < span_minimum {
            candle::bail!(
                "reservation: the card backed only {span_bytes} B of the {want} B asked \
                 for, below the {span_minimum} B of fixed staging and transient \
                 span a forward needs — there is not enough VRAM left to run one"
            )
        }

        // No weight side yet: an expert cache installs one at load
        // (`set_weight_floor`), and a process without experts leaves the whole
        // span to KV. Until the load freezes the dense block the floor stands at
        // the scratch's base, so neither the dense block nor a region reaches it.
        // No dense block yet: the weights are claimed out of it by `claim_dense`
        // as the loader walks the checkpoint, and until then it is empty.
        let scratch =
            scratch.min((span_bytes.saturating_sub(span_minimum) / REGION_BYTES) * REGION_BYTES);
        let scratch_base = span_end - scratch as u64;
        if scratch > 0 {
            load_scratch::open(ordinal, stream.cu_stream() as usize, scratch_base, scratch)?;
        }
        let layout = layout_span(span_base, persist_bytes, span_bytes, 0, scratch_base);
        let total = layout.total;
        // The partition, on the one channel that survives a test binary (which
        // installs no tracing subscriber, so the log lines below are invisible
        // there).
        //
        // `residual` is `capacity − weights − usable` — CUDA pool growth, the
        // gallery arena, per-class overhead — and is the term to chase when the
        // span comes out smaller than the card suggests it should. `weights` is
        // reported next to it precisely so that anyone reading this can confirm
        // it was subtracted **once**: it is inside `usable`, not applied on top
        // of it.
        let gov = candle::vram::get(ordinal);
        let weights = gov
            .as_ref()
            .map_or(0, |g| g.class_reserved(AllocClass::Weights) as usize);
        let capacity = gov.as_ref().map_or(0, |g| g.capacity() as usize);
        let residual = capacity
            .saturating_sub(weights)
            .saturating_sub(usable_before);
        if super::super::alloc::arena_stats_enabled() {
            let mib = |b: usize| b / (1024 * 1024);
            eprintln!(
                "[reservation] capacity_c={}MiB usable={}MiB pool_cushion={}MiB \
                 graph_reserve={}MiB weights={}MiB residual={}MiB | asked={}MiB span={}MiB \
                 (shortfall {}MiB) = persist {}MiB + {total} regions ({}MiB) \
                 + transient {}MiB + weights-side {}MiB, slack {}B",
                mib(capacity),
                mib(usable_before),
                gov.as_ref().map_or(0, |g| mib(g.pool_cushion() as usize)),
                mib(graph_reserve),
                mib(weights),
                mib(residual),
                mib(want),
                mib(span_bytes),
                mib(want.saturating_sub(span_bytes)),
                mib(persist_bytes),
                mib(total * REGION_BYTES),
                mib(MAX_WAVE_TRANSIENT_BYTES),
                mib((span_end - layout.weight_floor) as usize),
                layout.slack,
            );
        }
        if span_bytes < want {
            log::warn!(
                "reservation: the card backed {} MiB of the {} MiB span asked for. \
                 The shortfall lands on the KV side ({total} regions) — the fixed \
                 staging, the transient tier and the weight side are positioned \
                 from the span's right edge and are unaffected.",
                span_bytes / (1024 * 1024),
                want / (1024 * 1024),
            );
        } else {
            log::info!(
                "reservation: {} MiB span = {} MiB persist (held ground {} MiB) + {total} \
                 regions ({} MiB), transient placed per forward (≤{} MiB), in {} KiB \
                 granules; {} MiB left outside it for graph executables",
                span_bytes / (1024 * 1024),
                persist_bytes / (1024 * 1024),
                held_bytes / (1024 * 1024),
                (total * REGION_BYTES) / (1024 * 1024),
                MAX_WAVE_TRANSIENT_BYTES / (1024 * 1024),
                reservation.granularity() / 1024,
                graph_reserve / (1024 * 1024),
            );
        }

        // The held ground's bootstrap stands at the very base of the span, before
        // the persistence staging is carved after it, and is live from here on;
        // its arenas are claimed once the load has frozen the dense block.
        let held = SpanHeldGround::new(span_base, held_bytes);
        install_held_ground(ordinal, held.clone());

        Ok(Self {
            reservation,
            pre_reservation_in_use,
            span_base,
            span_bytes,
            held,
            region_base: layout.region_base,
            dense_base: layout.dense_base,
            dense_bytes: 0,
            dense_capable: measured,
            dense_frozen: false,
            weight_floor: scratch_base,
            total,
            next: 0,
            free: BinaryHeap::new(),
            live: 0,
            span_live: 0,
            peak_live: 0,
            // **Epochs start at one**, so that zero can mean "never dirtied" in
            // `dirt` without colliding with a real epoch. A pristine
            // region is zero from the mapping touch and has no readers, and that
            // has to be distinguishable from one dirtied before the first
            // quiesce — otherwise every region on the card claims as dirty once.
            quiesce_epoch: 1,
            dirt: vec![Dirt::default(); total],
            owned: vec![false; total],
            transient_base: None,
            transient_bytes: 0,
            transient_high_water: 0,
            recent_tiers: [0; RECENT_FORWARDS],
            recent_at: 0,
            prompts_waiting: false,
            waiting_tier_bytes: 0,
            tier_peak: 0,
            tiers_placed: 0,
            persist_carved: held_bytes,
            persist_bytes,
            held_bytes,
            fresh_claims_during_wave: 0,
            refusals_during_wave: 0,
            // Empty because nothing has been observed yet. The weight side must
            // watch the KV side stay small before it may take anything, and what
            // enforces that is `observing_until` below — not a fabricated peak,
            // which the window roll would carry for a second window.
            growth: GrowthPolicy::new(),
            kv_target: FreeRegionTarget::default(),
        })
    }

    pub(super) fn span_end(&self) -> u64 {
        self.span_base + self.span_bytes as u64
    }

    /// Span the KV and weight sides may divide between them — everything above
    /// the dense block.
    ///
    /// **Not `span_bytes`.** The persistence block and the model's dense weights
    /// are carved off the left and belong to neither side; only the ground from
    /// `region_base` up is theirs to trade. Sizing the weight side from the whole
    /// span instead lets the expert cache commit slots for ground the weights
    /// already hold, and it discovers the shortfall as *"Expert cache full,
    /// cannot evict (all pinned)"* — a message about eviction, several steps from
    /// the arithmetic that overcommitted. Qwen3-30B-A3B failed exactly this way
    /// the first time its loader placed weights in the span.
    pub(super) fn tradeable_bytes(&self) -> usize {
        (self.span_end() - self.region_base) as usize
    }

    /// Re-derive the region grid after the dense block grew.
    ///
    /// Only reachable from [`claim_dense`], which has already established that
    /// Widen the ownership bitmap to cover `total` regions.
    ///
    /// **Grow-only, unlike `dirt`.** A shrink that truncated this would
    /// throw away the record that a high region is still held, and the very next
    /// grow would restore it as `false` — so the pool would have forgotten the
    /// owner of a live region and the double-issue guard would wave it through.
    /// `set_weight_floor` refuses to cut below the live watermark, so an index
    /// past `total` should never be owned; keeping the flag is what makes that a
    /// checked claim rather than an assumed one.
    pub(super) fn cover_regions(&mut self, total: usize) {
        if total > self.owned.len() {
            self.owned.resize(total, false);
        }
    }

    /// Re-derive the region grid after the dense block grew.
    ///
    /// Only reachable from [`claim_dense`], which has already established that
    /// no region has been handed out — so there is nothing to relocate and
    /// `dirt` can simply be resized. Every region in the new grid is still
    /// pristine (epoch zero), because the span was mapped and never written.
    pub(super) fn relayout_for_dense(&mut self) {
        let layout = layout_span(
            self.span_base,
            self.persist_bytes,
            self.span_bytes,
            self.dense_bytes,
            self.weight_floor,
        );
        self.region_base = layout.region_base;
        self.total = layout.total;
        self.dirt.resize(layout.total, Dirt::default());
        self.cover_regions(layout.total);
    }

    /// The lowest address the wave transient tier may not reach below: one past
    /// the last byte any live arena holds.
    ///
    /// A region is not preemptible — an arena keeps its address for as long as
    /// it lives — so this is a hard floor for the tier, not a preference.
    pub(super) fn live_end(&self) -> u64 {
        self.region_base + (self.live_watermark() * REGION_BYTES) as u64
    }

    /// Regions the KV side may hand out **right now**.
    ///
    /// `total` is what fits below the weight floor. While a forward is running,
    /// the transient tier sits on top of the live arenas, so the ceiling drops to
    /// where it starts — a region stamped past that would be inside the tier.
    ///
    /// Between forwards the tier is gone and the ceiling **is** the boundary.
    /// Nothing is withheld against the next placement: [`place_transient`] buys
    /// the ground it needs when it needs it, which is both exact and current
    /// where a withheld reserve was a one-sample guess at a quantity that varies
    /// eightfold between consecutive waves.
    pub(super) fn region_ceiling(&self) -> usize {
        ceiling_regions(
            self.transient_base,
            self.weight_floor,
            self.region_base,
            self.total,
        )
    }

    pub(super) fn region_base(&self, index: usize) -> u64 {
        self.region_base + (index * REGION_BYTES) as u64
    }

    /// Regions [`claim_region`] would actually hand out right now.
    ///
    /// **Not "regions nobody owns".** The two differ whenever a wave's transient
    /// tier is placed, because the tier caps the pool at
    /// [`Self::region_ceiling`] and everything above that is unclaimable however
    /// free it looks. Counting the difference matters because this number
    /// answers "what would `claim_region` hand out right now" — a claim made
    /// while the tier stands must be refused at the ceiling, and reporting the
    /// tier's own ground as claimable is a promise the allocator then refuses
    /// to keep. (A wide section prefill once did exactly that: a 496 MiB tier
    /// capped the ceiling at 293 regions, the 31 above it were reported free,
    /// and admission planned against them — 12,488 refusals in four seconds
    /// with no forward completing.)
    ///
    /// Callers whose decision lands *after* the tier releases — admission
    /// budgets, scheduler pressure, the weight-side negotiation — must add
    /// [`Self::ceiling_blocked`] back on top; [`RegionStats`] carries both
    /// numbers so each consumer picks the horizon it is actually deciding for.
    pub(super) fn free_count(&self) -> usize {
        // The heap is unordered under iteration, so this counts rather than
        // short-circuits — a few hundred entries on a path that runs per wave,
        // not per claim.
        let below = self
            .free
            .iter()
            .filter(|Reverse(i)| *i < self.region_ceiling())
            .count();
        claimable(below, self.next, self.region_ceiling())
    }

    /// The widest tier of the last [`RECENT_FORWARDS`] forwards.
    pub(super) fn recent_tier_bytes(&self) -> usize {
        self.recent_tiers.iter().copied().max().unwrap_or(0)
    }

    /// Regions that are unowned but sit above the ceiling, so cannot be claimed
    /// while the tier stands. Zero between waves; the gap between "free" and
    /// "free to use" when a wave is placed.
    pub(super) fn ceiling_blocked(&self) -> usize {
        let ceiling = self.region_ceiling();
        let above = self.free.iter().filter(|Reverse(i)| *i >= ceiling).count();
        blocked(above, self.next, self.total, ceiling)
    }

    /// Mark every region overlapping `[base, base + len)` as holding someone
    /// else's bytes that only stream `on` may still be touching, so the next
    /// claim zeroes it — after a quiesce, unless it claims on `on`.
    ///
    /// Rounds **outward**: a region half-covered is still dirty, and a byte of
    /// stale data is as bad as a region of it.
    pub(super) fn mark_dirty_span(&mut self, base: u64, len: usize, on: usize) {
        let start = base.saturating_sub(self.region_base) as usize / REGION_BYTES;
        let end = (base
            .saturating_add(len as u64)
            .saturating_sub(self.region_base) as usize)
            .div_ceil(REGION_BYTES);
        let epoch = self.quiesce_epoch;
        for slot in self.dirt.iter_mut().take(end.min(self.total)).skip(start) {
            *slot = slot.left(epoch, on);
        }
    }

    /// One past the highest region currently held by an arena.
    ///
    /// The KV side cannot shrink below this, and it is meaningfully smaller than
    /// `next` because the free list is lowest-first: live arenas stay
    /// left-packed, which is the property that makes the boundary movable at all.
    pub(super) fn live_watermark(&self) -> usize {
        watermark_of(self.next, self.free.iter().map(|Reverse(i)| *i))
    }

    /// Move the weight side's left edge and re-derive the region count.
    ///
    /// Runs at model load to place the boundary, and between waves to move it.
    /// Refuses rather than corrupts (principle 7): a floor past
    /// [`MIN_ELASTIC_RESERVE`], past the span end, or one that would strand a
    /// live region is an error, not a clamp.
    ///
    /// The fixed blocks do not move with it — the transient tier is a constant
    /// size at a constant address (see [`layout_span`]) — so this changes exactly
    /// two things: how many regions exist, and where the weight side starts.
    pub(super) fn set_weight_floor(&mut self, floor: u64) -> Result<usize> {
        let span_end = self.span_end();
        // **Measured from `region_base`, not `span_base`.** The reserve is what
        // must be left to the *elastic middle* — the KV regions and the wave tier
        // that stands among them — and the persistence block and the dense weight
        // block sit below `region_base` and are neither. Measuring from
        // `span_base` counted both inside the reserve, so the guard permitted a
        // region area smaller than the number it was enforcing.
        //
        // It mattered once the reserve came down from 2,048 MiB to 1,296. On the
        // 27B streamed-dense path `region_base − span_base` is ~1,064 MiB (64 MiB
        // persist plus the dense block), so the old arithmetic let the KV area
        // shrink to ~230 MiB — a quarter of the 912 MiB the transient tier alone
        // requires, and `place_transient` then fails with nothing able to concede.
        // At 2,048 the same slip left ~984 MiB, just above the tier, which is why
        // it had never bitten. `weight_capacity_bytes` has always measured this
        // from `tradeable_bytes()` and was already right.
        let min_floor = self.region_base + MIN_ELASTIC_RESERVE as u64;
        if floor < min_floor {
            candle::bail!(
                "weight floor {floor:#x} would leave the elastic middle below the \
                 {} MiB reserve (floor must be at least {min_floor:#x})",
                MIN_ELASTIC_RESERVE / (1024 * 1024),
            )
        }
        if floor > span_end {
            candle::bail!("weight floor {floor:#x} is past the span end {span_end:#x}")
        }
        // **The tier is the other occupant of the middle, and it was unguarded.**
        //
        // The check below refuses a floor that would cut live KV regions. It
        // said nothing about the wave transient tier, which stands in the same
        // ground between the arenas and the weights — so the weight side could
        // grow DOWN onto a tier that was already placed. `tier_fits` had
        // approved that tier against the floor as it stood a moment earlier;
        // lowering the floor underneath it retroactively puts its top inside
        // the weight zone, and the FFN span at the top of the tier then writes
        // activations over resident expert slots.
        //
        // Measured, at the moment of the fault: tier `[0x4cf9800000, +0x502400000)`
        // topping out at `0x51fbc00000` against a floor of `0x51f9c00000` — 32 MiB
        // of the tier inside expert ground, with the `wave-ffn` span 275 KB past
        // the line.
        //
        // A standing tier is not movable here (its ranges are named by live
        // pointers), so the only sound answer is to refuse the floor and let the
        // weight side ask again once the tier has gone — which it does between
        // forwards, every forward.
        if let (Some(base), bytes) = (self.transient_base, self.transient_bytes) {
            let tier_top = base.saturating_add(bytes as u64);
            if floor < tier_top {
                candle::bail!(
                    "weight floor {floor:#x} would fall below the standing wave transient \
                     tier's top {tier_top:#x} (tier at {base:#x}, {bytes} B). That ground \
                     is holding this forward's activations — taking it would put the FFN \
                     span inside the expert weight zone. The tier is released between \
                     forwards; ask again then."
                )
            }
        }
        let layout = layout_span(
            self.span_base,
            self.persist_bytes,
            self.span_bytes,
            self.dense_bytes,
            floor,
        );
        let watermark = self.live_watermark();
        if layout.total < watermark {
            candle::bail!(
                "weight floor would cut the KV side to {} regions, but region {} is \
                 live. The weight side may only take regions the KV side is not \
                 using; ask it to release them first.",
                layout.total,
                watermark - 1,
            )
        }
        let gained = layout.total.saturating_sub(self.total);
        self.weight_floor = floor;
        self.total = layout.total;
        // **Ground arriving from the weight side is dirty.** It was holding
        // expert weights a moment ago, and `resize(.., 0)` would declare it
        // pristine — the state reserved for a region untouched since the mapping
        // touch, which is the one case a claim may skip cleaning. A KV arena
        // would then be handed a region full of expert bytes.
        //
        // Stamped one epoch *behind* the current one, which is the exact truth:
        // the handover already quiesced (`quiesce_before_handover`), so there are
        // no readers left to wait for — but the bytes are still there, so it
        // still needs zeroing.
        self.dirt.resize(layout.total, Dirt::default());
        self.cover_regions(layout.total);
        if gained > 0 {
            let stale = self.quiesce_epoch.saturating_sub(1);
            for slot in self.dirt.iter_mut().skip(layout.total - gained) {
                *slot = Dirt::any(stale);
            }
            // **Every concession is the KV side's demand, whoever asked for it.**
            // A purchase through the broker said so already; one asked for
            // between forwards (`request_kv_ground` — admission, relief) did not,
            // and the next forward's growth negotiation read the ground it had
            // just conceded as spare and took it straight back.
            self.growth.note_demand();
        }
        // Regions past the new total no longer exist: drop them from the free
        // list and pull `next` back so the fresh-region path cannot hand one out.
        self.free.retain(|Reverse(i)| *i < layout.total);
        self.next = self.next.min(layout.total);
        Ok(layout.total)
    }

    /// Regions the weight side may take: the ground the KV side is **not using
    /// now**, less a slack margin, and nothing else.
    ///
    /// # Why the present, and not a forecast of it
    ///
    /// This answered against a sliding-window maximum of past KV demand for as
    /// long as the boundary has existed, on the reasoning that the weight side
    /// should be fast to concede and slow to take: being short of KV **fails a
    /// forward**, being short of experts is a slowdown, and the two are not
    /// worth trading symmetrically. The mark was insurance, and the asymmetry was the
    /// premium.
    ///
    /// **Being short of KV no longer fails a forward.** [`set_ground_broker`]
    /// means a claim that runs the KV side out buys exactly the ground it needs
    /// at the moment it needs it, and the weight side concedes on contact.
    /// Measured over a full 27B gate: thirty purchases, **zero refused**. The
    /// insurance now covers a loss that cannot occur, and it is not free.
    ///
    /// What it cost was a **ratchet**. Shrink reads the present exactly —
    /// admission grows KV rightward and evicts weights on contact, no estimate
    /// involved — while grow consulted a forecast. So a wide cohort drove the
    /// weight zone down in seconds and no amount of idle time brought it back:
    /// the mark remembered a peak the workload had left behind while occupancy
    /// said the ground was free. On the 27B that held 34 of 64 layers streaming
    /// through two configs needing a quarter of the KV. Removing it took the
    /// zone to full residency on every single-context config and 26 → 35 layers
    /// on the widest.
    ///
    /// # Why the present is knowable here
    ///
    /// This is reachable only from `reclaim_spare_ground`, which every caller
    /// invokes on the line *after* `end_wave_transient`: the tier is released, no
    /// wave generation is open, and empty arenas have just been swept. `live` is
    /// what the KV side actually holds and [`Self::free_count`] is what it does
    /// not. There is no estimate left to make.
    ///
    /// The one quantity genuinely in the future — persistence's quantize
    /// destinations — is what `slack` is for, and it stays. Replacing *that*
    /// with an exact figure was tried and refuted, for the good reason that the
    /// thing it estimates has not happened yet.
    ///
    /// The two guards below survive because they are statements about now rather
    /// than predictions: a workload that has never run, and a KV side visibly on
    /// its way up.
    pub(super) fn spare_regions(&mut self, slack: usize) -> usize {
        // **The decision lives in [`super::growth_policy`]; this gathers what it
        // reads.** One transcription, so the soak tests in
        // `docs/vram_partition_behavioural_tests.md` exercise the policy the
        // engine actually runs rather than a model of it — the partition's
        // defects have all been trajectory defects, and a trajectory is only
        // testable if it can be run without a device.
        let occ = Occupancy {
            total: self.total,
            kv_target: self.kv_target,
            live: self.live,
            free_below_ceiling: self.free_count(),
            ceiling_blocked: self.ceiling_blocked(),
            free_above_live: self.total.saturating_sub(self.live_watermark()),
            tier_bytes: self.transient_bytes,
            tier_high_water: self.transient_high_water,
            recent_tier_bytes: self.recent_tier_bytes(),
            waiting_tier_bytes: self.waiting_tier_bytes,
        };
        match self.growth.spare(occ, slack, REGION_BYTES) {
            Ok(spare) => {
                SPARE_TALLY[3].fetch_add(spare as u64, Ordering::Relaxed);
                spare
            }
            Err(why) => {
                let idx = match why {
                    Refusal::Observing => 0,
                    Refusal::Pressure => 1,
                    Refusal::Occupied => 2,
                    Refusal::Fragmented => 6,
                };
                SPARE_TALLY[idx].fetch_add(1, Ordering::Relaxed);
                0
            }
        }
    }
}
