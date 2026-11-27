//! Compacts the KV pools and the span tenants so free regions gather at the
//! arena frontier, paced by a time budget and gated on what a pass would gain.
//!
//! Fragmentation strands air below the frontier. A pass moves live chunks into
//! lower arenas so emptied arenas can be handed back to the region free list.
//! A pass costs a device sync and a copy per relocated chunk, so it runs only
//! when there is ground to recover, and it counts as progress only when it
//! lowered the frontier or gained arenas. Each pass is bounded by a wall-clock
//! budget, and passes repeat after a clipped one until the pools settle. Every
//! holder that names a chunk is rewritten in the same sweep, because a holder
//! left behind keeps naming a slot the pass has vacated.

use super::*;

/// [`Scheduler::frontier_holder`]'s answer when a KV arena stands on the top region.
pub(in super::super) const KV_POOLS: &str = "the KV pools";

/// [`Scheduler::frontier_holder`]'s answer when held ground stands on it.
pub(in super::super) const HELD_GROUND: &str = "held ground";

/// What a KV pack is for, which decides when one is worth running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum PackGoal {
    /// Lower the frontier: the wave loop's pass, run only with a KV arena on top
    /// ([`Scheduler::frontier_pass`]).
    Frontier,
    /// Recover free regions wherever they lie: relief and the settle, which pack
    /// under a span tenant too ([`Scheduler::ground_pass`]).
    Ground,
}

impl PackGoal {
    pub(super) fn pass(self, sched: &Scheduler) -> Option<(PoolShape, FrontierPass)> {
        match self {
            Self::Frontier => sched.frontier_pass(),
            Self::Ground => sched.ground_pass(),
        }
    }

    /// Wall-clock one pass may spend **planning and claiming**; the pass plans in
    /// proportion (`compact.rs`, `planned_moves_cap`).
    ///
    /// A budget rather than a move cap: the ladder spans 320 B to 16 KiB, so the
    /// same "one move" is fifty times the bandwidth at one rung and a count cannot
    /// bound a duration. The census and the claims are separately bounded inside it
    /// (see `compact_backings`).
    ///
    /// **The wave loop's pass is 80 ms.** At 8 ms every pass clipped and handed back
    /// a single arena; at 20 ms the census alone spent the budget and 37 of 40
    /// attempts claimed nothing; at 40 ms 31 of 42 passes clipped and a mass eviction
    /// left the pools at 71%. The loop's own housekeeping already spends 40–66 ms on
    /// the promote pass at this point, so a pass of this order is in proportion to
    /// the gap it runs in.
    ///
    /// **Relief's is three of those, because its passes ran back to back and each
    /// paid the sweep.** The sweep walks every residence of every substrate whatever
    /// the pass moved: on BlackWorld (2026-11-10) 111–158 ms a pass, ~75% of the
    /// pass, against a census of ~6 ms and a plan of ~11. A relief's settle ran up to
    /// three passes to absorb a burst, so one pass of the three budgets claims the
    /// same moves and sweeps once — ~280 ms off a large relief. The settle loop still
    /// follows a pass that clips.
    pub(super) fn budget(self) -> Duration {
        match self {
            Self::Frontier => Duration::from_millis(80),
            Self::Ground => Duration::from_millis(240),
        }
    }
}

/// What one KV compaction pass did: the arenas it emptied and handed back, and whether
/// its budget stopped it with work left.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in super::super) struct PackPass {
    pub(in super::super) released: usize,
    pub(in super::super) clipped: bool,
    /// Whether the pass handed ground back: the frontier fell, or it released more
    /// arenas than it created. A pass creates destination arenas ahead of its moves
    /// and releases whichever it did not fill, so `released` alone counts its own
    /// churn as progress.
    pub(in super::super) gained: bool,
}

/// Run `pass` back to back until the pools are settled, answering the arenas the
/// passes released.
///
/// **Ends on the first pass that stops short of more work or gains nothing.** A pass
/// that was turned away (`None`) or finished its plan inside the budget leaves no
/// more to do now. A pass that clipped but handed no ground back is not followed
/// either: the next would plan the same moves from the same ground. It is judged on
/// ground, not on `released`, because a pass releases the fresh arenas it created
/// and never filled — on Qwen3-30B-A3B a dozen passes in a row each "released" 3–40
/// arenas with the frontier pinned at 400 regions, at ~250 ms a pass. Every pass the
/// loop continues past lowered the frontier or the pools' arena count, so the loop
/// ends within as many passes as there are KV arenas.
pub(super) fn settle_passes(mut pass: impl FnMut() -> Option<PackPass>) -> usize {
    let mut released = 0;
    while let Some(p) = pass() {
        released += p.released;
        if !p.clipped || !p.gained {
            break;
        }
    }
    released
}

impl Scheduler {
    /// Pack the KV pools if they are fragmented enough to be worth a pass, then ask
    /// the weight side to take what packing released.
    ///
    /// **Gated on the gain, not run unconditionally.** A pass costs a device-wide
    /// sync and a copy per relocated chunk, so it is only worth making when there is
    /// real ground to recover: `MIN_FREEABLE_ARENAS` regions, which at 16 MiB each is
    /// the point where the expert cache can actually do something with the result.
    ///
    /// **And the second half is not optional.** Lowering the frontier only makes it
    /// *possible* for `weight_floor` to move left; `reclaim_spare_ground` is what
    /// moves it. Without that call a pass hands back regions nobody claims, decode is
    /// exactly as slow as before, and the work is invisible in every metric except
    /// the one that matters — which is why the two are one method and not two.
    ///
    /// **Considered every wave, run at most every `MIN_INTERVAL`.** Relief, which runs
    /// only under pressure and whose alternatives cost a turn or an expert, packs the KV
    /// pools directly and leaves the span tenants to this path.
    ///
    /// Answers the arenas the pass emptied and handed back to the region free list —
    /// zero when it did not run.
    /// **Lower the arena frontier for a tier refused `short` bytes**, by the
    /// frontier holder's own pass, until the ground above the frontier has grown
    /// by that much or a round moves it no further. Returns the bytes gained.
    ///
    /// The relief a refused tier needs, which [`Self::relieve_vram_pressure`]
    /// cannot see: relief measures pressure as free regions, and a span with
    /// hundreds of holes reads as having plenty — while the tier, which stands
    /// only above every live arena, is short. Measured on WhiteWorld
    /// (2026-10-19, 872957a58): cohorts refused one to six regions short with
    /// 346 regions (5.5 GiB) of holes below the frontier, retried at once and
    /// lost, because nothing moved the frontier down into them. The passes this
    /// drives are the ones the wave loop already paces (`compact_kv_if_fragmented`)
    /// — a K/V pack when a K/V arena holds the top, the span tenants' pass when
    /// a tenant does — run here for as long as they make headway against a
    /// shortfall that is known, rather than once per interval against none.
    pub(in super::super) fn lower_frontier_for(&mut self, short: usize) -> usize {
        /// Rounds one refusal may drive. Each is one pass of the holder's kind,
        /// a region or two for a tenant and the pools' worth for K/V, so this
        /// covers the one-to-six-region shortfalls a cohort is refused by.
        const ROUNDS: usize = 16;
        let gap = |s: &Self| {
            s.session.kv_region_stats().map_or(0, |st| {
                st.total.saturating_sub(st.live_watermark) * candle_nn::kv_cache::REGION_BYTES
            })
        };
        let start = gap(self);
        let target = start.saturating_add(short);
        for _ in 0..ROUNDS {
            let now = gap(self);
            if now >= target {
                break;
            }
            match self.frontier_pass() {
                Some((_, FrontierPass::Kv)) => {
                    self.pack_until_settled(PackGoal::Frontier);
                }
                Some((shape, FrontierPass::SpanTenants)) => self.pack_span_tenants_judged(shape),
                None => break,
            }
            if gap(self) <= now {
                break;
            }
        }
        let gained = gap(self).saturating_sub(start);
        // Who stands on the frontier's top region after the pass — the tenant a
        // pass that gained nothing could not move — and the holes under it.
        let (frontier, holes) = self.session.kv_region_stats().map_or((0, 0), |s| {
            (s.live_watermark, s.live_watermark.saturating_sub(s.live))
        });
        let holder = self.frontier_holder(frontier);
        let ordinal = match self.session.device().location() {
            candle::DeviceLocation::Cuda { gpu_id } => Some(gpu_id),
            _ => None,
        };
        // Held ground is raw ranges, not slots a pass can move: what holds it is
        // the call sites, the largest three.
        let held_sites = if holder == HELD_GROUND {
            ordinal
                .and_then(candle_nn::kv_cache::held_ground_sites)
                .unwrap_or_default()
                .iter()
                .take(3)
                .map(|s| format!("{}={}MiB/{}", s.site, s.live_bytes >> 20, s.live_count))
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            String::new()
        };
        tracing::info!(
            target: "candle_conversation::scheduler::vram_relief",
            short_mib = short >> 20,
            gained_mib = gained >> 20,
            frontier_regions = frontier,
            hole_regions = holes,
            holder,
            held_sites = %held_sites,
            "lowered the arena frontier for a refused tier"
        );
        gained
    }

    /// Who stands on the region just below `frontier` — the one whose ground holds
    /// the frontier where it is: the KV pools, held ground, or a span tenant by its
    /// label. `"unaccounted"` names ground none of them reports, which is a tenant
    /// missing from this list.
    pub(in super::super) fn frontier_holder(&self, frontier: usize) -> &'static str {
        let Some(top) = frontier.checked_sub(1) else {
            return "nothing";
        };
        if self.session.kv_top_rank() == Some(top) {
            return KV_POOLS;
        }
        let held_top = match self.session.device().location() {
            candle::DeviceLocation::Cuda { gpu_id } => {
                candle_nn::kv_cache::held_ground_top_rank(gpu_id)
            }
            _ => None,
        };
        if held_top == Some(top) {
            return HELD_GROUND;
        }
        candle_nn::kv_cache::SlotTenant::ALL
            .iter()
            .find(|&&t| candle_nn::kv_cache::arena_top_rank(self.session.device(), t) == Some(top))
            .map_or("unaccounted", |t| t.label())
    }

    /// The span ground held ground's arenas stand on now — zero off CUDA or
    /// before a reservation exists.
    pub(in super::super) fn held_arena_bytes(&self) -> u64 {
        match self.session.device().location() {
            candle::DeviceLocation::Cuda { gpu_id } => {
                candle_nn::kv_cache::held_ground_stats(gpu_id).map_or(0, |h| h.arena_bytes as u64)
            }
            _ => 0,
        }
    }

    pub(in super::super) fn compact_kv_if_fragmented(&mut self) -> usize {
        /// Shortest interval between passes.
        ///
        /// The gate in `pack_kv_pools` is cheap, but the pass behind it is not: its
        /// census walks every arena's occupancy bitmap once per rung of the ladder.
        /// Running it on every wave-loop iteration would put that walk between every
        /// pair of forwards. This is what makes "every wave, cheap-signal gated" mean
        /// *considered* every wave rather than *run* every wave.
        const MIN_INTERVAL: std::time::Duration = std::time::Duration::from_millis(150);

        // **The interval before either counter.** This runs on every iteration of the
        // wave loop: the interval is two loads, and the gate behind it takes a read
        // lock per rung of the ladder — cheap next to a census, not cheap next to
        // nothing, and pointless on an iteration that has already decided not to run.
        //
        // Returns without standing down deliberately: the pass it is deferring is
        // still coming, so a refusal's hold on the persistence thread must survive it.
        if self
            .last_kv_compaction
            .is_some_and(|t| t.elapsed() < MIN_INTERVAL)
        {
            return 0;
        }
        // Each part is charged to the window's compaction split, so the band
        // says which pass it went to.
        let t_gate = Instant::now();
        let pass = self.frontier_pass();
        self.wave_stats.compaction_split.gate(t_gate.elapsed());
        match pass {
            None => 0,
            Some((shape, FrontierPass::SpanTenants)) => {
                let t = Instant::now();
                self.pack_span_tenants_judged(shape);
                self.wave_stats.compaction_split.tenants(t.elapsed());
                0
            }
            Some((_, FrontierPass::Kv)) => {
                let t = Instant::now();
                let packed = self.pack_until_settled(PackGoal::Frontier);
                self.wave_stats.compaction_split.kv(t.elapsed());
                packed
            }
        }
    }

    /// The pass that could lower the frontier now, and the pools' shape it would
    /// start from — `None` when no pass could, or when the last one on this ground
    /// already showed it could not.
    ///
    /// **Who stands at the frontier decides it** ([`PoolShape::frontier_pass`]). A KV
    /// arena there is the KV pass's to move, given enough air below it; a span tenant's
    /// region is the span tenants' pass, given enough holes to move into. Packing the KV pools
    /// beneath a region another tenant holds lowers nothing — on Flash-Next's 128K
    /// prefill every such pass moved 4,560 chunks for no region — and its quiesce
    /// drained the device ahead of every forward.
    ///
    /// **A cheap gate, because this is consulted every wave.** `kv_ground_lost` is not
    /// cheap — it sums `packed_arenas` across the ladder, which is the census — so it
    /// cannot be the thing that decides whether to census. The hole count is a handful
    /// of field reads behind one lock; the sparsity sum and the KV pools' top rank take
    /// a read lock per rung of the ladder and walk that pool's arena map.
    ///
    /// Both halves of the KV side's loss count, because either alone misses the other's
    /// regime. Holes — free regions stranded below the frontier — are self-correcting
    /// under allocation, so a steadily-loaded pool reads zero holes with tens of sparse
    /// arenas beneath it; gating on holes alone ran 16 passes over 110 s of churn and
    /// left the pools at 82%. Sparsity alone would miss the burst, where a mass eviction
    /// strands the frontier over ground that is genuinely free.
    ///
    /// **And not again over ground the same pass could not improve** — until something
    /// is freed beneath the frontier. See `compaction_stall`.
    ///
    /// **A gate that will not run a KV pass stands down** the hold a previous refusal
    /// put on the persistence thread. A refused pass tells migrates to step aside for
    /// it; if no KV pass is coming, that promise has to be withdrawn or hot→warm defers
    /// for a pass that never runs.
    pub(in super::super) fn frontier_pass(&self) -> Option<(PoolShape, FrontierPass)> {
        self.pass_by(PoolShape::frontier_pass)
    }

    /// [`Self::frontier_pass`], for a caller short of free regions rather than of
    /// frontier — relief and the settle — which packs the KV pools under a span
    /// tenant too ([`PoolShape::ground_pass`]).
    ///
    /// A KV pack the stall gate holds falls back to the frontier's pass, so the
    /// span tenants still move into the holes the last one opened.
    pub(in super::super) fn ground_pass(&self) -> Option<(PoolShape, FrontierPass)> {
        self.pass_by(PoolShape::ground_pass)
            .or_else(|| self.frontier_pass())
    }

    pub(super) fn pass_by(
        &self,
        choose: impl Fn(&PoolShape, Option<usize>) -> Option<FrontierPass>,
    ) -> Option<(PoolShape, FrontierPass)> {
        let pass = self.session.kv_region_stats().and_then(|stats| {
            let shape = PoolShape {
                live: stats.live,
                frontier: stats.live_watermark,
                sparse: self.session.kv_sparse_arenas(),
                tenant_sparse: candle_nn::kv_cache::tenant_freeable_arenas(self.session.device()),
            };
            let pass = choose(&shape, self.session.kv_top_rank())?;
            (!self.kv_compaction_stall.holds(pass, shape)).then_some((shape, pass))
        });
        if !matches!(pass, Some((_, FrontierPass::Kv))) {
            candle_nn::kv_cache::clear_compaction_waiting();
        }
        pass
    }

    /// [`Self::compact_span_tenants`], judged like a KV pass: one that left the
    /// frontier where it found it stalls the gate on `shape` until something is freed
    /// beneath it, so a tenant that cannot move is not asked again every wave.
    pub(in super::super) fn pack_span_tenants_judged(&mut self, shape: PoolShape) {
        self.compact_span_tenants();
        let t_judge = Instant::now();
        let after = self
            .session
            .kv_region_stats()
            .map_or(shape.frontier, |s| s.live_watermark);
        self.wave_stats
            .compaction_split
            .tenant_part(TenantPart::Judge, t_judge.elapsed());
        self.wave_stats
            .compaction_split
            .tenants_lowered(shape.frontier, after);
        self.kv_compaction_stall.record(
            FrontierPass::SpanTenants,
            shape,
            PassOutcome {
                reclaimed_regions: shape.frontier.saturating_sub(after),
                frontier_before: shape.frontier,
                frontier_after: after,
                clipped: false,
            },
        );
        self.last_kv_compaction = Some(std::time::Instant::now());
    }

    /// Pack the span tenants that are not KV pools — the recurrent state store, the
    /// QSA index pages, the creeps' residuals and the provenance gallery — when one
    /// of them holds the frontier over enough holes to be worth moving into
    /// ([`super::compaction_stall::MIN_FREEABLE_ARENAS`]).
    ///
    /// **Never in the follow-ups or relief.** Each is a batch of device copies of whole
    /// blocks (a recurrent state is ~3 MiB) bounded by a move count rather than the KV
    /// pass's clock, and repeating it does not converge faster. Run from every relief
    /// episode and every follow-up pass as well, it ran twice a second on
    /// Qwen3.8-Flash-Next, moving hundreds of states a time for a region or none, beside
    /// forwards already slowed by streamed experts — and the probe's efficiency gate
    /// fell from 98–99% to 76–87%.
    pub(in super::super) fn compact_span_tenants(&mut self) {
        /// Layer states one recurrent pass may move. Each is one device copy of a few
        /// MiB, so this bounds the pass at a few hundred MiB of copy — well under the
        /// KV pass's budget — while letting a badly scattered arena set converge in a
        /// handful of passes.
        const MAX_STATE_MOVES: usize = 256;

        /// Gallery pages a single pass may move.
        ///
        /// A page is 6 KiB against a recurrent state block's 3 MiB, so this is a
        /// budget in the same class of copy — ~24 MiB — while being enough moves
        /// that a badly scattered gallery converges in a handful of passes rather
        /// than hundreds. The gallery's own ceiling is 512 MiB, or ~87k pages, so
        /// this is deliberately a fraction of the worst case.
        const MAX_GALLERY_MOVES: usize = 4096;

        /// Creep residual blocks a single pass may move. A block is up to a
        /// region of whole rows, so this is up to 256 MiB of copy — the same
        /// class as the recurrent pass — and covers both halves of a creep of a
        /// few thousand rows.
        const MAX_CREEP_MOVES: usize = 16;

        // **Recurrent state first.** Its arenas are regions of the span like the KV
        // side's, and on a hybrid stack they are most of what stands above the holes —
        // the KV pools can be packed to a region while the frontier stays pinned by a
        // state arena near the top. Its pass moves each state block with one device
        // copy and repoints the one store that holds it, so it needs no holder sweep
        // and no quiesce; a failure part-way leaves every store consistent (each move
        // is whole) and is reported loudly rather than retried silently.
        let t_recurrent = Instant::now();
        let recurrent = {
            let _g = super::super::profile::span("compact:recurrent");
            self.model.compact_recurrent(MAX_STATE_MOVES)
        };
        self.wave_stats
            .compaction_split
            .tenant_part(TenantPart::Recurrent, t_recurrent.elapsed());
        match recurrent {
            // Each compaction pass is routine housekeeping between forwards —
            // every few seconds under pressure — so its report is debug; a
            // failure stays an error.
            Ok(r) if r.planned > 0 => {
                tracing::debug!(
                    target: "candle_conversation::scheduler::vram_relief",
                    planned = r.planned,
                    moved = r.moved,
                    regions_before = r.regions_before,
                    regions_after = r.regions_after,
                    "recurrent state compaction packed the state arenas",
                );
                if r.regions_released() > 0 {
                    let _g = super::super::profile::span("compact:reclaim_weights");
                    self.model.reclaim_spare_ground();
                }
            }
            Ok(_) => {}
            Err(e) => tracing::error!(
                target: "candle_conversation::scheduler::vram_relief",
                "recurrent state compaction failed: {e}",
            ),
        }

        // **A creep's arena stands for the whole of its sweep**, which beside a
        // busy decode is many seconds — long enough for the KV side to free the
        // ground under it and leave its blocks holding the frontier. Its blocks
        // move like any tenant's: the one in flight and a paused one, which are
        // every creep arena there is, between forwards (`docs/wave_step.md` §3).
        let t_creep = Instant::now();
        let creeps = {
            let _g = super::super::profile::span("compact:creep");
            let held = self.wave_prefill_residual.iter_mut().chain(
                self.paused_creep
                    .iter_mut()
                    .filter_map(|p| p.residual.as_mut()),
            );
            compact_creeps(held, MAX_CREEP_MOVES)
        };
        self.wave_stats
            .compaction_split
            .tenant_part(TenantPart::Creep, t_creep.elapsed());
        match creeps {
            Ok(r) if r.planned > 0 => {
                tracing::debug!(
                    target: "candle_conversation::scheduler::vram_relief",
                    planned = r.planned,
                    moved = r.moved,
                    regions_before = r.regions_before,
                    regions_after = r.regions_after,
                    "creep residual compaction packed the creep arenas",
                );
                if r.regions_released() > 0 {
                    let _g = super::super::profile::span("compact:reclaim_weights");
                    self.model.reclaim_spare_ground();
                }
            }
            Ok(_) => {}
            Err(e) => tracing::error!(
                target: "candle_conversation::scheduler::vram_relief",
                "creep residual compaction failed: {e}",
            ),
        }

        // **The gallery's pages fragment from churn, not from size.** Its runs are
        // variable length and are freed out of claim order — LRU eviction, and a
        // re-seal freeing a turn's old run mid-corpus — so a corpus that reads
        // nearly full while it is still growing ends up with live pages scattered
        // across arenas, a few of them pinning the frontier above everything below.
        // Pages a scan is reading are skipped, so this cannot move ground out from
        // under an in-flight launch.
        if let Some(arena) = self.gallery_arena.as_ref() {
            let t_gallery = Instant::now();
            let gallery = {
                let _g = super::super::profile::span("compact:gallery");
                arena.compact(MAX_GALLERY_MOVES)
            };
            self.wave_stats
                .compaction_split
                .tenant_part(TenantPart::Gallery, t_gallery.elapsed());
            match gallery {
                Ok(r) if r.planned > 0 => {
                    tracing::debug!(
                        target: "candle_conversation::scheduler::vram_relief",
                        planned = r.planned,
                        moved = r.moved,
                        regions_before = r.regions_before,
                        regions_after = r.regions_after,
                        "gallery compaction packed the page arenas",
                    );
                    if r.regions_released() > 0 {
                        let _g = super::super::profile::span("compact:reclaim_weights");
                        self.model.reclaim_spare_ground();
                    }
                }
                Ok(_) => {}
                Err(e) => tracing::error!(
                    target: "candle_conversation::scheduler::vram_relief",
                    "gallery compaction failed: {e}",
                ),
            }
        }
    }

    /// Pass after pass until one finishes inside its budget — the pass both the wave
    /// loop and relief run.
    ///
    /// **One pass is not enough after a burst.** Relief's compression rewrites a dozen
    /// or more float turns as quantized chunks in one go, and a mass retirement frees
    /// chunks across every arena, so 50–80 arenas of air arrive at once on the 30B probe
    /// and a single budgeted pass clips after 20–50 of them. The next chance to finish
    /// is the next wave-loop iteration, which spans a decode quantum and its
    /// housekeeping — 1–3 s there — and the probe measured the remainder standing that
    /// long, at 70%, while the forwards in between bought ground from the weight side
    /// against it. Back to back, the passes cost up to ~80 ms each, and only when a
    /// pass clipped; an iteration with nothing to pack still costs one gate.
    ///
    /// **Bounded by what the passes free, not by a count of them.** Each pass plans
    /// at most `PLANNED_MOVES_PER_POOL` moves a pool (`compact.rs`), which on the 30B
    /// probe is ~75,000 moves and 12–16 arenas a pass. A fixed four passes therefore
    /// stopped ~52 regions into a burst of 75–120, and the remainder stood through the
    /// next forward and the publish after it: worst sustained efficiency read 66–75%
    /// with every pass reporting `plan_capped` and none clipped by the clock. See
    /// [`settle_passes`] for why the loop ends.
    pub(in super::super) fn pack_until_settled(&mut self, goal: PackGoal) -> usize {
        settle_passes(|| self.pack_kv_pools(goal))
    }

    /// One KV compaction pass if the pools are fragmented enough to be worth it, then
    /// the weight side's reclaim — what [`Self::compact_kv_if_fragmented`] paces and
    /// relief runs directly. The span tenants are not packed here; see
    /// [`Self::compact_span_tenants`].
    ///
    /// Answers what the pass did — nothing when the pools were not worth a pass — or
    /// `None` when it was turned away by a holder of the partition or of chunk
    /// locations (a forward, a hot→warm migrate group, a cold-writer free), which is
    /// the one answer that changes by asking again shortly.
    pub(in super::super) fn pack_kv_pools(&mut self, goal: PackGoal) -> Option<PackPass> {
        let Some((shape, FrontierPass::Kv)) = goal.pass(self) else {
            return Some(PackPass::default());
        };
        // Free first what other threads let go of: a chunk freed while the pass
        // relocates would leave its slot neither where the pass found it nor
        // where it put it.
        drain_reclaimed();

        // **Every registered conversation, not one.** Conversations in a workspace
        // share a substrate, so sweeping one sweeps all its residences — but the
        // scheduler can host conversations on more than one substrate, and a
        // workspace left out keeps residences naming vacated slots. Sweeping them all
        // is safe because a pass is idempotent under one `Sweep`: after a rewrite the
        // residence holds NEW gids and the map is keyed on the old ones, so a second
        // visit matches nothing.
        let substrates = self.distinct_substrates();
        let mut swept = 0usize;
        // **The projection caches are holders too.** Taken out for the duration so
        // the closure can rewrite them while `self.session` is borrowed — two
        // disjoint fields of `self`, which the borrow checker cannot split across a
        // method call on one of them — and put back below whatever the pass returns.
        //
        // Without this they kept naming the slots a pass had vacated: a cached glue
        // island or a pending user part is Arc-injected into a slot by a later
        // projection instead of being recomputed, so the stale K/V is read as if it
        // were the span's own. That is the incomplete holder set the pass's
        // completeness check found — 129 of 6767 relocated slots named by nobody it
        // reached — and the reason a compaction could poison a sequence that was not
        // even resident when it ran.
        let mut projections = std::mem::take(&mut self.slot_projection_state);
        // **So are the described slots' captures**, held past their seals for
        // transplants (`transplant.rs`) — taken out and put back the same way.
        let mut captures = std::mem::take(&mut self.captures);
        // A detached tenant's substrate, out of the attached set but still
        // holding its resident KV until its last handle goes — held alive for
        // the pass and let go after it (`Detached::alive`).
        let detached = self.detached.alive();
        // The closure receives the pass's OWN `Sweep`, already carrying whatever the
        // backings rewrote. Sharing it is what makes an allocation held by both a
        // block table and a residence come back as one replacement.
        // The substrates no live slot is bound to — on a shared engine, the tenant
        // between conversations — timed apart, so the cost of walking them reads
        // beside the pass it is part of.
        let idle: Vec<bool> = substrates
            .iter()
            .map(|s| {
                !self
                    .slot_conversations
                    .values()
                    .any(|c| c.same_substrate(s))
            })
            .collect();
        let mut idle_us = 0u128;
        // The sweep's holders by kind, so a slow sweep names the one it was
        // spent on; the backings' own block tables are the rest of `sweep_us`.
        let (mut substrates_us, mut projections_us, mut detached_us, mut reclaimed_us) =
            (0u128, 0u128, 0u128, 0u128);
        let outcome = {
            let _g = super::super::profile::span("compact:kv");
            self.session.compact_kv(goal.budget(), &mut |sweep| {
                for (s, idle) in substrates.iter().zip(&idle) {
                    let t = std::time::Instant::now();
                    swept += s.rewrite_for_compaction(sweep)?;
                    let us = t.elapsed().as_micros();
                    substrates_us += us;
                    if *idle {
                        idle_us += us;
                    }
                }
                let t = std::time::Instant::now();
                for state in projections.values_mut() {
                    swept += state.rewrite_for_compaction(sweep)?;
                }
                for capture in captures.values_mut() {
                    swept += capture.rewrite_for_compaction(sweep)?;
                }
                projections_us += t.elapsed().as_micros();
                let t = std::time::Instant::now();
                for substrate in &detached {
                    swept += substrate.rewrite_for_compaction(sweep)?;
                }
                detached_us += t.elapsed().as_micros();
                // Last, so a hot copy any thread let go of while the residences
                // were walked is still named by the sweep.
                let t = std::time::Instant::now();
                swept += rewrite_reclaimed(sweep)?;
                reclaimed_us += t.elapsed().as_micros();
                Ok(())
            })
        };
        // The pass is over: a detached substrate whose last handle went while it
        // ran is freed here, outside it.
        drop(detached);
        self.slot_projection_state = projections;
        self.captures = captures;
        // **The interval is stamped by a pass that ran, never by one that was
        // refused.** `MIN_INTERVAL` exists to bound the census, and a contended pass
        // pays no census — it refuses on a `try_` acquisition and returns. Stamping
        // ahead of the call made every refusal cost a quarter second of not trying
        // again, and against a persistence thread that held its migrate for the
        // length of a hot→warm batch that meant 15 passes out of 52 attempts and the
        // pools left at 69%. Now the next wave picks up the gap the moment the
        // migrate's writes have landed.
        if !matches!(
            outcome,
            Err(candle_nn::kv_cache::CompactionRefused::WaveInFlight)
                | Err(candle_nn::kv_cache::CompactionRefused::MigrateInFlight)
        ) {
            self.last_kv_compaction = Some(std::time::Instant::now());
        }
        // **The pass's own phase breakdown, filed where every other wave span lands.**
        // `candle-nn` sits below the profiler, so the pass measures itself and this
        // records it — see `CompactionReport::timings`. Recorded for a refusal too when
        // one got far enough to plan, because "the census cost 30 ms and then the claims
        // lost" is a diagnosis and "compaction was refused" is not.
        if let Ok(report) = &outcome {
            self.kv_compaction_stall.record(
                FrontierPass::Kv,
                shape,
                PassOutcome {
                    reclaimed_regions: report.regions_reclaimed(),
                    frontier_before: report.frontier_before,
                    frontier_after: report.frontier_after,
                    clipped: report.clipped,
                },
            );
            let t = report.timings;
            // A sweep long enough to be felt, by holder — one short line, so a
            // log reader shows it whole.
            if t.sweep >= SLOW_SWEEP {
                let ms = |us: u128| us / 1000;
                tracing::info!(
                    target: "candle_conversation::scheduler::compaction",
                    "compaction sweep split sweep_ms={} substrates_ms={} projections_ms={} \
                     detached_ms={} reclaimed_ms={} backings_ms={} mint_ms={} minted={} \
                     moves={} allocations={} clean_reused={} residences={swept}",
                    t.sweep.as_millis(),
                    ms(substrates_us),
                    ms(projections_us),
                    ms(detached_us),
                    ms(reclaimed_us),
                    ms(t.sweep.as_micros().saturating_sub(
                        substrates_us + projections_us + detached_us + reclaimed_us
                    )),
                    report.mint_us / 1000,
                    report.records_minted,
                    report.moves,
                    report.allocations_rewritten,
                    report.clean_reused,
                );
            }
            self.wave_stats
                .compaction_split
                .kv_lowered(report.frontier_before, report.frontier_after);
            self.wave_stats.compaction_split.kv_stage(
                [
                    t.quiesce,
                    t.plan,
                    t.claim,
                    t.copy,
                    t.barrier,
                    t.sweep,
                    t.mint,
                    t.invalidate,
                    t.publish,
                ],
                report.moves,
            );
            super::super::profile::record("compact:quiesce", t.quiesce);
            super::super::profile::record("compact:plan", t.plan);
            super::super::profile::record("compact:claim", t.claim);
            super::super::profile::record("compact:copy", t.copy);
            super::super::profile::record("compact:barrier", t.barrier);
            super::super::profile::record("compact:sweep", t.sweep);
            super::super::profile::record("compact:mint", t.mint);
            super::super::profile::record("compact:invalidate", t.invalidate);
            super::super::profile::record("compact:publish", t.publish);
        }
        let pass = match &outcome {
            Ok(r) => {
                let gained =
                    r.frontier_after < r.frontier_before || r.arenas_released > r.fresh_arenas;
                Some(PackPass {
                    released: r.arenas_released,
                    // Another pass right away when this one ran out of budget, or capped
                    // a pool's plan and still handed ground back. A capped pass that
                    // gained nothing is not pursued: re-running it would plan the same
                    // moves.
                    clipped: r.clipped || (r.plan_capped && gained),
                    gained,
                })
            }
            Err(
                candle_nn::kv_cache::CompactionRefused::WaveInFlight
                | candle_nn::kv_cache::CompactionRefused::MigrateInFlight,
            ) => None,
            Err(_) => Some(PackPass::default()),
        };
        match outcome {
            Ok(report) if !report.is_empty() => {
                // The weight side, immediately, while no wave generation is live.
                let _g = super::super::profile::span("compact:reclaim_weights");
                self.model.reclaim_spare_ground();
                tracing::debug!(
                    target: "candle_conversation::scheduler::vram_relief",
                    moves = report.moves,
                    allocations = report.allocations_rewritten,
                    // Fresh records for the relocated chunks, one batched launch. Lower
                    // than `allocations` by however many were writer windows, which carry
                    // no record and are addressed from their gids.
                    minted = report.records_minted,
                    // Non-zero means the up-front reservation under-provisioned and those
                    // chunks' sources were not reclaimed. Correct, but degrading.
                    records_declined = report.records_declined,
                    record_arenas_reserved = report.record_arenas_reserved,
                    // Arenas created ahead of demand; the unused ones, which the pass
                    // hands back itself, are among `arenas_released`.
                    fresh_arenas = report.fresh_arenas,
                    // `KvHead` records copied lower (included in `moves`), and the
                    // copies no holder took — expected where the chunk was minted anew.
                    records_moved = report.records_moved,
                    records_unfollowed = report.records_unfollowed,
                    // The pass's downstream cost: each cleared buffer is a host
                    // re-serialisation and an upload on its slot's next sync.
                    buffers_cleared = report.decode_buffers_cleared,
                    arenas_released = report.arenas_released,
                    frontier_before = report.frontier_before,
                    frontier_after = report.frontier_after,
                    // What holds the frontier up once the pass is done — a pass that
                    // moved a lot and lowered nothing is explained here or nowhere.
                    top_arena = ?report.top_arena,
                    reclaimed_mib =
                        report.regions_reclaimed() * (candle_nn::kv_cache::REGION_BYTES >> 20),
                    substrate_sequences = swept,
                    clipped = report.clipped,
                    plan_capped = report.plan_capped,
                    // **What the pass waited for, beside what it did.** The pre-plan
                    // drain is the one phase whose cost belongs to other work — it
                    // waits for whatever was in flight when the pass began — and it
                    // is outside the budget, so without it here a pass that spent
                    // 60 ms waiting and 20 ms working is indistinguishable from one
                    // that spent 80 ms working. Only visible through the profile
                    // spans otherwise, and those compile to nothing by default.
                    quiesce_us = report.timings.quiesce.as_micros(),
                    // The two halves the budget bounds: the census and plan, then the
                    // claim walk. A pass that clips after one batch spent its budget in
                    // the first.
                    plan_us = report.timings.plan.as_micros(),
                    // The plan's three steps, so a slow plan names its step.
                    census_us = report.timings.census.as_micros(),
                    provision_us = report.timings.provision.as_micros(),
                    walk_us = report.timings.walk.as_micros(),
                    planned_moves = report.planned_moves,
                    // Moves stepped around because a migrate group holds the chunk;
                    // they pack once the group has installed.
                    held = report.held,
                    pools_censused = report.pools_censused,
                    pools_ranked = report.pools_ranked,
                    first_source_rank = ?report.first_source_rank,
                    claim_us = report.timings.claim.as_micros(),
                    // The host walk over every slot's decode-buffer pins, outside the
                    // budget like the quiesce. Named because it scales with slots ×
                    // chunks × layers, not with what the pass moved.
                    invalidate_us = report.timings.invalidate.as_micros(),
                    // Rewriting every holder, and of that the substrates no live
                    // slot is bound to.
                    sweep_us = report.timings.sweep.as_micros(),
                    idle_substrates_us = idle_us,
                    substrates_us,
                    projections_us,
                    detached_us,
                    reclaimed_us,
                    // **The figure that says whether the pass was complete.** A relocated
                    // slot no holder the sweep reached names: its claim is wasted, its
                    // source is not reclaimed, and because nothing names it the census
                    // will plan it again next pass. Not a correctness problem — every
                    // party naming a slot keeps it alive — but the holder list is
                    // maintained by hand and this is how a missing one shows up.
                    unwitnessed = report.unwitnessed,
                    // Read/write collisions declined before the launch. Non-zero is
                    // the guard working: a collision suffered corrupts a chunk
                    // silently, and this is the only line that says it was there.
                    source_collisions = report.source_collisions,
                    "kv compaction packed the pools and handed the ground back",
                );
            }
            Ok(_) => {}
            Err(candle_nn::kv_cache::CompactionRefused::WaveInFlight) => {
                // Ordinary contention. The next pass tries again.
            }
            // **A fault is reported, not declined.** Every other arm here is "not
            // now" and belongs at `debug`, which is why this one cannot share the
            // arm: a partition invariant that broke would have been a debug line
            // nobody reads, on a pass that runs every wave.
            //
            // Every active turn fails, not one. The pass runs in the wave loop
            // between forwards, so there is no turn whose call stack this is on —
            // and the fault is not one turn's anyway: the arenas are pooled across
            // sessions, so a plan that named ground no arena owns implicates
            // whatever any of them is holding. Failing the turns that exist is what
            // makes it visible at the only place a caller is watching.
            Err(candle_nn::kv_cache::CompactionRefused::Fault(msg)) => {
                let ids: Vec<_> = self.active_decodes.keys().copied().collect();
                tracing::error!(
                    target: "candle_conversation::scheduler::vram_relief",
                    turns = ids.len(),
                    "kv compaction found a broken partition invariant and abandoned \
                     the pass: {msg}",
                );
                self.fail_all_decodes(
                    &ids,
                    &format!("kv compaction found a broken partition invariant: {msg}"),
                );
            }
            Err(e) => {
                tracing::debug!(
                    target: "candle_conversation::scheduler::vram_relief",
                    "kv compaction declined: {e:?}",
                );
            }
        }
        pass
    }
}
