//! Runs one compaction pass over every KV pool, from census to publish.
//!
//! A pass plans, claims and copies its moves within a time budget, then rewrites
//! every holder of a moved chunk: this backing's block tables, the caller's
//! residences, and the device records, which are minted fresh rather than
//! patched. Nothing is published until the holders are rewritten, and the device
//! is fenced before any emptied arena is released. A sweep that fails part way
//! still fills, invalidates and fences what it published, but releases nothing.

use super::*;

/// Pack every GPU KV pool toward the lowest addresses, within `budget`.
///
/// **Takes every backing, because one plan must reach every block table.** Arenas
/// pool globally across same-config layers, so the pool being packed is shared —
/// but each backing owns its *own* `sequences` block tables, one per layer. A pass
/// driven from a single backing would relocate a chunk and rewrite one layer's view
/// of it, leaving the other forty-seven pointing at the vacated slot. That is not a
/// degraded pass, it is wrong attention on every layer but one.
///
/// `sweep` is called **once**, after every backing has rewritten its own block
/// tables and before anything is published, and is handed **the same [`Sweep`]**
/// those rewrites used. That sharing is required, not convenient: holders across the
/// backings, the substrate and the projection caches routinely share one
/// `Arc<Vec<ChunkGid>>` allocation, and a second `Sweep` would give each its own
/// equal-but-distinct replacement — leaving refcounts that disagree with the sharing
/// the cache believes exists. The caller applies it with
/// [`super::compact_map::rewrite_sealed`]. Returning `Err` aborts the pass with
/// nothing published.
///
/// A caller with provably no holders of its own (a batched session, whose only
/// block tables are these backings') passes a closure that does nothing. That is a
/// statement about that caller, not a default.
///
/// # The pass rewrites gids, and MINTS a record — it never rewrites one
///
/// A chunk's location is written down twice: in the `ChunkGid`, which holds the arena
/// slot's refcount, and in the `KvHead` record's band-pointer word, which is the
/// address the paged kernels dereference. Both are published, and neither is *patched*.
///
/// The gids are rewritten through their holders, because a gid is an owning handle. A
/// relocated chunk is then given a **freshly minted** record built from its new gids
/// ([`RecordMint`](super::compact_mint::RecordMint)), and the record it had is left
/// exactly as it is for whoever still holds it. Two properties make each side sound on
/// its own:
///
/// * A record holds a clone of the `HeadGids` its words were serialized from (see
///   [`MetaGid`](super::meta_pool::MetaGid)), so **a record cannot outlive the bands it
///   names** — the allocator will not reissue a band slot while a record points at it,
///   because the record is one of that slot's refcount holders.
/// * So the rewritten holders name the destination and read a record naming the
///   destination; the holders the sweep did not reach name the source and read a record
///   naming the source, which their own record keeps alive. Neither had to be *found*.
///
/// Installing the minted record drops the old one, and when that was the chunk's last
/// holder the old record's slot frees — taking with it the clone of the *old* gids it was
/// holding, which is the last reference to the source bands. That is what lets
/// `release_empty_arenas` hand the region back, and it is the whole reclaim.
///
/// A **live writer window** carries no record (`meta: None`), is addressed from its own
/// gids, and has its cached decode buffer thrown away — so it reads and writes the
/// destination, consistently, and is never given a record. `sealed ⇒ has a record ⇒
/// immutable` and `live ⇒ meta: None` is the invariant that keeps a written chunk and a
/// record-read chunk from being the same chunk; `set_block_gids` enforces it from the
/// other side by clearing `meta` on any gid mutation, and
/// `a_writer_owned_chunk_with_a_record_is_left_alone` pins the pass's own half.
///
/// ## What a mint moves, and what must therefore be dropped
///
/// A mint changes a **record's** address. A band's address is owned — by the record — but
/// a record's own address is cached as a bare `kvheads_ptr` word in every `TokenSlice`
/// header of every slot's decode buffer, and that buffer is reused whenever the chunk
/// *count* agrees, which a compaction never changes. `rewrite_for_compaction` clears the
/// buffer of each slot whose own chunks moved, which is the set whose *band* addresses
/// changed and is not provably the set whose *record* addresses changed. So a pass that
/// minted anything clears every slot's buffer in every backing.
///
/// ## Why the previous design corrupted K/V
///
/// The pass used to rewrite the records too, and with a record owning nothing it could
/// not be made correct either way round. **A chunk has exactly one record, shared by
/// every holder of that chunk**, so a pass that rewrites some of a chunk's holders to
/// the destination and leaves the rest on the source has no correct value to put in
/// that word: whichever slot it names is kept alive only by the holders naming the same
/// slot, and when those go the others are still reading through the record into
/// re-tenanted ground. Patching to the destination, patching to the source and leaving
/// it alone were all wrong — leaving it alone only became right once the record started
/// holding a refcount.
///
/// Measured on Qwen3.8-Flash-Next with the `tensor-assert` harness: zero orphaned gids
/// against 148 orphaned records already present when a pass *began*, 4,320 pointer/gid
/// disagreements on two slots, and an engine probe answering 8/8, 5/8, 8/8, 5/8 against
/// 8/8 twice with the pass off. It was fatal on a recurrent model and survivable
/// elsewhere: wrong K/V puts a NaN in the first full-attention layer, a DeltaNet layer
/// computes its *persisted* recurrent state from it, and every later wave's logits are
/// NaN, so the model emits `!!!!!!!!` forever. The 30B ate the identical wrong K/V,
/// because it has no recurrent state to keep it in.
///
/// Three more faults were paid for on the way, each found by correlating a log line
/// against the story gate rather than by inspection, and each worth knowing because the
/// shape recurs: a record claim **promoted into a band arena** and filled over live K/V
/// (`stamp_region_promoting` widens the size class, and `ArenaKey::new` is always a band
/// key); a record claim **creating or releasing an arena mid-sweep**, which re-tenants an
/// `arena_idx` that half-rewritten holders still name — `arena slot: 2176 B requested from
/// a 1152 B slot`, 198/92/114 times on corrupt runs against 0 on clean ones; and
/// `Sweep::memo` keyed on `Arc::as_ptr` while minting made originals die mid-sweep, so a
/// replacement could land on a dead original's address and a second visit was handed
/// another chunk's gids.
///
/// `kv_integrity::report_boundary` is wired at both boundaries below under
/// `tensor-assert` and remains the standing check — but **it cannot see most of this
/// class**: it reported zero orphans and zero content changes on a run that answered 2/8,
/// because nothing was orphaned and its hash covers only bands that *live* slots' block
/// tables name. The engine probe's story gate is the oracle, and one clean run means
/// nothing: require several consecutive.
#[cfg(feature = "cuda")]
pub fn compact_backings(
    backings: &[super::super::backing::ChunkedKvBacking],
    budget: Duration,
    sweep: &mut dyn FnMut(&mut Sweep<'_>) -> Result<()>,
) -> std::result::Result<CompactionReport, CompactionRefused> {
    note(0, 1);
    let Some(first) = backings.first() else {
        note(4, 1);
        return Err(CompactionRefused::NoDevice);
    };
    // The pass takes its own clock, after it has drained the device — see the note
    // on the budget in `compact_with`.
    let outcome = first.compact_with(backings, budget, sweep);
    match &outcome {
        Ok(r) => {
            note(1, 1);
            // Bumped only for a pass that actually moved something: a pass that
            // relocated nothing invalidates nobody's captured gids, and a counter
            // that ticked anyway would make every holder's snapshot look stale.
            if r.moves > 0 {
                EPOCH.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            }
            note(5, r.moves as u64);
            note(6, r.arenas_released as u64);
            note(7, r.regions_reclaimed() as u64);
            if r.clipped {
                note(8, 1);
            }
        }
        Err(CompactionRefused::WaveInFlight) => note(2, 1),
        Err(CompactionRefused::MigrateInFlight) => note(10, 1),
        Err(CompactionRefused::AlreadyPacked) => note(3, 1),
        Err(CompactionRefused::NoDevice) => note(4, 1),
        Err(
            CompactionRefused::ClaimsLost
            | CompactionRefused::CopyFailed
            | CompactionRefused::BarrierFailed
            | CompactionRefused::SweepFailed(_)
            | CompactionRefused::Fault(_),
        ) => note(9, 1),
    }
    outcome
}

impl super::super::backing::ChunkedKvBacking {
    /// The body of [`compact_backings`], on the backing that owns the shared pool.
    #[cfg(feature = "cuda")]
    pub(super) fn compact_with(
        &self,
        backings: &[super::super::backing::ChunkedKvBacking],
        budget: Duration,
        sweep: &mut dyn FnMut(&mut Sweep<'_>) -> Result<()>,
    ) -> std::result::Result<CompactionReport, CompactionRefused> {
        let candle::Device::Cuda(cuda) = self.device() else {
            return Err(CompactionRefused::NoDevice);
        };

        // **Nothing off-thread may be writing a slot this pass could copy.** The arena
        // window below excludes forwards, which run on this thread; it says nothing
        // about the persistence thread, whose hot→warm migrate claims quantize
        // destinations and writes them. Copied before the convert lands, a slot hands
        // its holders a copy of nothing — which surfaces, much later, as a sequence
        // answering from another sequence's KV. The migrate holds locations only while
        // it writes; the rest of its group the pass steps around (`retain_unheld`
        // below). See `migrate_flight`.
        let _locations = super::super::migrate_flight::try_freeze_chunk_locations()
            .ok_or(CompactionRefused::MigrateInFlight)?;

        // One window for the whole pass — see the module note on 199,888 refusals.
        // It holds the stream's lease for the pass with it.
        let _window =
            super::super::bump_arena::enter_arena_window(cuda.cuda_stream(), "a KV compaction")
                .map_err(|_| CompactionRefused::WaveInFlight)?;

        // **Quiesce the whole device before reading anything, not just before
        // publishing.**
        //
        // The pass had a barrier after its copy and another after its patch, and
        // none before it began — so every source slot it read, and every
        // destination it wrote, was read and written against whatever was still in
        // flight when it started. The two exclusions above are host-side: the arena
        // window keeps a *new* forward from opening and the location freeze keeps a
        // *new* migrate from starting, and neither waits for device work that was
        // already launched to retire. The persistence thread's hot→warm migrate is
        // the one that matters — it runs on its own copy stream, so stream-order
        // FIFO says nothing about it, and it reads exactly the band addresses this
        // pass is about to move.
        //
        // Device-wide rather than per stream, for the reason in the module note: one
        // barrier at between-forwards cadence removes the whole overlap domain,
        // including side streams this module does not know about. Ordered after both
        // exclusions deliberately — quiescing first would fence work that the
        // unexcluded window could then replace before the census ran.
        //
        // Refused rather than continued if it fails: nothing has been planned, so
        // this is the cheapest possible place to abandon a pass.
        let quiesce = Instant::now();
        if let Err(e) = self.device().synchronize() {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                "compaction could not quiesce the device before planning, \
                 abandoning the pass: {e}",
            );
            return Err(CompactionRefused::BarrierFailed);
        }

        // **The budget starts after the wait, which is why this is taken here and
        // not by the caller.** Every deadline below is the pass's own work — the
        // census it chooses to run and the claims it chooses to make — and the
        // quiesce is neither: it waits for whatever the pass arrived behind. Timed
        // from before the drain, a 50 ms wait against an 80 ms budget would put the
        // planning deadline in the past before the first pool was censused, so every
        // pass would clip having moved nothing, and the harder the device was working
        // the more completely compaction would stop.
        let started = Instant::now();
        // From the census to the end of the sweep: a slot freed on another thread
        // inside this window is what the report's gaps are made of.
        #[cfg(feature = "tensor-assert")]
        let window = super::super::release_watch::PassWindow::open();

        let frontier_before = self.frontier_regions().ok_or(CompactionRefused::NoDevice)?;

        // ── Plan every pool, and claim + copy within the budget ──────────────
        let mut map = CompactionMap::new();
        let mut report = CompactionReport {
            frontier_before,
            timings: CompactionTimings {
                quiesce: quiesce.elapsed(),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut pending: Vec<ChunkMove> = Vec::new();
        let mut pool_of: Vec<ArenaKey> = Vec::new();
        // **The census is the expensive half, so the clock is consulted between
        // pools.** `compaction_census` walks every arena's occupancy bitmap — at a
        // frontier of 900 arenas and 50k slots per arena that is tens of millions of
        // bits and a `Vec<u32>` per arena — and it runs once per rung of the ladder.
        // Planning the whole ladder unconditionally spent the entire budget before a
        // single chunk was claimed, and the pass then refused with nothing done: 37 of
        // 40 attempts, reported as claims lost, with the pools at 60% efficiency the
        // whole time.
        //
        // A quarter of the budget for planning, so three quarters reach the claims. A
        // pool left unplanned is not lost — it is where the next pass starts from, and
        // the order below puts the pool holding the frontier first, so what gets
        // planned is what the pass most needs to move. The split was half and half
        // while the ladder was walked in class order and the planner's choice of pool
        // was therefore arbitrary; with the order deliberate, spending less of the
        // budget deciding and more of it moving is the better trade.
        let plan_deadline = budget / 4;
        // **Highest arena first, because address is the quantity under attack.**
        //
        // Walked in ladder order instead, a budgeted pass spends its census and its
        // claims on whichever rungs happen to come first, and the pool that owns the
        // topmost arena — the only pool whose position costs the weight side anything
        // — can be last and never reached. Measured: nine consecutive seconds at 52–59%
        // efficiency with the frontier pinned at 346 and 64 free regions under it,
        // while every pass ran, clipped, and packed the low rungs it could already
        // reach.
        //
        // The order comes from `pool_top_rank`, which answers without the census, so
        // deciding where to spend the budget does not spend it.
        //
        // **The `KvHead` record pool is ranked beside the band pools.** Its arenas are
        // regions like any other, so a sparse one high in the span holds the frontier up
        // exactly as a sparse band arena does. The same walk packs it; what differs is how
        // a holder follows a moved record (`Sweep::follow_record`), not how it is planned.
        let record_key = if self.records_are_resident() {
            self.record_layout().ok().map(|l| l.key)
        } else {
            None
        };
        let mut by_rank: Vec<(usize, ArenaKey)> = SizeClass::all()
            .map(|class| ArenaKey::new(class, ArenaLocation::Gpu))
            .chain(record_key)
            .filter_map(|key| self.pool_top_rank(key).map(|(rank, _)| (rank, key)))
            .collect();
        by_rank.sort_unstable_by_key(|a| std::cmp::Reverse(a.0));
        report.pools_ranked = by_rank.len();
        let mut census_by_pool: Vec<(ArenaKey, Vec<ArenaSlots>)> = Vec::new();
        for (_, key) in by_rank {
            if started.elapsed() >= plan_deadline && !census_by_pool.is_empty() {
                report.clipped = true;
                break;
            }
            let Ok(census) = self.compaction_census(key) else {
                continue;
            };
            census_by_pool.push((key, census));
        }
        report.timings.census = started.elapsed();
        // Every arena the pass creates ahead of demand goes through `fresh`, which keeps
        // the empty sweep off it while the pass needs it and releases the ones nothing
        // landed in itself — see [`FreshArenas`].
        let mut fresh = FreshArenas::new(&*self.inner);
        let provision_started = Instant::now();
        self.provision_low_arenas(&mut census_by_pool, &mut fresh);
        report.timings.provision = provision_started.elapsed();
        let walk_started = Instant::now();
        // **One list over every pool, highest source region first** — see
        // [`by_source_rank`]. The budget clips this list, so its order decides which
        // regions a clipped pass empties.
        let rank_of: HashMap<usize, usize> = census_by_pool
            .iter()
            .flat_map(|(_, census)| census.iter().map(|a| (a.arena_idx, a.rank)))
            .collect();
        let plans: Vec<CompactPlan> = census_by_pool
            .iter()
            .filter_map(|(key, census)| plan_pool(census, *key, planned_moves_cap(*key, budget)))
            .collect();
        // A pool cut short has work left, which the next pass resumes — reported apart
        // from the budget's clip; see `CompactionReport::plan_capped`.
        report.plan_capped = plans.iter().any(|p| p.clipped);
        let mut ranked = by_source_rank(&plans, |idx| rank_of.get(&idx).copied().unwrap_or(0));
        if ranked.is_empty() {
            return Err(CompactionRefused::AlreadyPacked);
        }
        // **Step around what a migrate group holds** — see `HeldChunks`. Its handles
        // keep each source alive whatever this pass does, so a move of one claims a
        // destination nothing this pass reaches would take up. The group packs once
        // it has installed and let go.
        report.held = retain_unheld(&mut ranked, |(m, _)| {
            (m.from.0 * GID_STRIDE + m.from.1 as usize) as i64
        });
        if ranked.is_empty() {
            return Err(CompactionRefused::MigrateInFlight);
        }
        for (m, key) in ranked {
            pending.push(m);
            pool_of.push(key);
        }
        report.planned_moves = pending.len();
        report.pools_censused = census_by_pool.len();
        report.first_source_rank = pending
            .first()
            .and_then(|m| rank_of.get(&m.from.0).copied());
        report.timings.walk = walk_started.elapsed();
        report.timings.plan = started.elapsed();
        let mut phase = Instant::now();

        // **Claim on the host, copy in one launch.** The budget bounds the claim
        // walk — a host loop over the plan, one free-list pop and two address
        // computations per move — and the copies it produced then go to the device
        // as a single batched record set. Checking the clock per batch rather than
        // per move keeps the branch off the inner path.
        let mut records = CopyRecords::default();
        // Indexes the PLAN, not the records: a claim that loses a race is skipped,
        // so the record count drifts from the plan position and using it to slice
        // `pool_of` would pair later moves with the wrong pool's size class — a
        // wrong `byte_lens` for the copy and a wrong stride for the address.
        // Every slot this pass plans to read, so a claim cannot hand one of them back
        // as a destination — see the note in `claim_moves`. Built from the whole plan
        // rather than per batch, because the collision is across batches: the plan is
        // complete before the first claim, and a destination claimed in the last batch
        // can name a source planned in the first.
        let mut planned_sources: AHashSet<(usize, u32)> = AHashSet::with_capacity(pending.len());
        planned_sources.extend(pending.iter().map(|m| (m.from.0, m.from.1)));
        // Room for every move up front: a pass of half a million moves rehashed
        // the map's tables as it grew, inside the claim walk.
        map.reserve(pending.len());
        let mut planned = 0usize;
        for batch in pending.chunks(MOVES_PER_BATCH) {
            // **The first batch is unconditional.** Planning has already spent part of
            // the budget, and a pass that plans and then claims nothing is worse than
            // no pass at all: it pays the census, the window and the refusal, and
            // leaves the pool exactly as it found it. One batch guarantees every
            // attempt converges by at least 512 slots.
            if planned > 0 && started.elapsed() >= budget {
                report.clipped = true;
                break;
            }
            let keys = &pool_of[planned..planned + batch.len()];
            planned += batch.len();
            // **Passed up, not absorbed.** This used to warn, set `clipped` and
            // carry on to the sweep and the publish — written for a lost claim,
            // where keeping what landed is exactly right. But a lost claim never
            // arrives here: `claim_moves` skips one with a `continue`, because the
            // census is a snapshot and a slot it believed free may have gone. The
            // only things that come out of it as `Err` are an arena table that
            // would not resolve and a plan naming ground no arena owns — faults
            // that recur on the next pass and whose implied addresses land in
            // another tenant's region. So the soft handling caught none of the
            // cases it was for and absorbed every case it was not.
            if let Err(e) = self.claim_moves(batch, keys, &planned_sources, &mut map, &mut records)
            {
                return Err(CompactionRefused::Fault(e.to_string()));
            }
        }
        if map.is_empty() {
            return Err(CompactionRefused::ClaimsLost);
        }
        // **Every provisioned arena the plan put nothing in goes back now**, before the
        // record reservation below looks for a hole to claim. The per-pool arenas took
        // one hole each, and a reservation that found none would leave the mints nowhere
        // to land. Safe here and only here: no holder has been rewritten yet, so nothing
        // names an empty arena. See [`FreshArenas::release_unused`].
        if let Err(e) = fresh.release_unused() {
            return Err(CompactionRefused::Fault(e.to_string()));
        }
        // The record slots this pass will read, which no mint may be handed — see
        // `ChunkedKvBacking::try_alloc_record_slot`.
        let record_sources: HashSet<(usize, u32)> = pending
            .iter()
            .zip(&pool_of)
            .filter(|(_, key)| matches!(key.kind, ArenaKind::Record { .. }))
            .map(|(m, _)| (m.from.0, m.from.1))
            .collect();
        // From here until the pass ends, the slots it is about to move are declared
        // immutable and any instrumented write into them names its writer. See
        // [`ReadonlySources`] for why it is the sources and not the whole band.
        #[cfg(feature = "tensor-assert")]
        let _sources = ReadonlySources::declare(self.device(), &records);

        // **Everything here that costs a readback is behind `tensor-assert`, and the
        // reason is a measurement it invalidated.**
        //
        // Left unconditional, these checks read from the device per chunk and compared
        // both ends of every copied record — megabytes off the device per pass. A run
        // then completed ONE compaction pass where an uninstrumented run completed
        // twenty, so the workload under observation was not the workload being
        // debugged, and "no faults found" meant "no passes ran". That is the harness's
        // own first danger: an instrument that fences suppresses the race it hunts.
        //
        // What stays unconditional is what costs nothing to be right about: the
        // read/write overlap refusal (host arithmetic), the band-pointer verify (one
        // four-byte readback), the source-collision rejection (a host set lookup) and
        // the pool's ownership guard (one indexed bool).
        //
        // **Every reference every live slot holds, and one content hash per slot.**
        //
        // Complete rather than sampled. A gid or a record pointer naming a slot the
        // refcount tables call free is ground the allocator will reissue while that band
        // still points there; reported here so the first boundary at which a slot's
        // references go bad names the operation that broke them. The hash is what the
        // closing boundary compares against: this pass rewrites addresses and must leave
        // every slot's bytes exactly as they are, so any slot whose hash moves across the
        // pass has a party naming another chunk's K/V.
        //
        // Not free — it walks every band of every slot of every backing, reads each
        // chunk's resident record back, and launches the hash — which is why it is behind
        // the harness rather than standing in the pass.
        #[cfg(feature = "tensor-assert")]
        let entering = super::super::kv_integrity::report_boundary(
            backings,
            "entering a compaction",
            0,
            0,
            None,
        );

        report.timings.claim = phase.elapsed();
        phase = Instant::now();
        // One launch for every relocated slot, whatever the pool or the rung —
        // `byte_lens` is per record, so a pass spanning the whole ladder is still
        // one grid. A per-move `memcpy_dtod_async` measured ~8 µs of launch
        // overhead each, which put 1,024 moves in an 8 ms budget and left every
        // pass clipped with the frontier exactly where it started.
        if let Err(e) = self.copy_records(&records) {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                "compaction copy launch failed, abandoning the pass: {e}",
            );
            // Nothing is rewritten yet, so every holder still names its source
            // slot and the claimed destinations are merely wasted.
            return Err(CompactionRefused::CopyFailed);
        }
        report.moves = records.srcs.len();
        report.source_collisions = records.source_collisions;
        report.timings.copy = phase.elapsed();
        phase = Instant::now();

        // **Bytes before pointers.** The records rewritten below name the
        // destinations; nothing may read through them until the copies landed.
        //
        // Device-wide, not `stream.synchronize()`. It used to be the stream's, with
        // a comment saying it was written that way so "a future side stream cannot
        // silently break it" — which is the opposite of what a stream-scoped
        // barrier does. FIFO covers this pass's own copy-then-patch ordering because
        // both are on `stream`; it says nothing about the persistence thread's copy
        // stream, which is a reader of the very slots being vacated.
        if let Err(e) = self.device().synchronize() {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                "compaction barrier failed, abandoning the pass: {e}",
            );
            // Nothing has been rewritten, so the old slots are still every
            // holder's truth and the destinations are merely wasted.
            return Err(CompactionRefused::BarrierFailed);
        }

        // **Did the copy reproduce the bytes?** Sampled, host-side, after the fence.
        //
        // Every other check in this pass is about *addressing* — that a gid names the
        // slot it should, that a record holds the pointer it was given, that a holder
        // was reached. All of them can pass while the bytes themselves are wrong, and
        // the reader cannot tell: quantised KV is a byte string, so a destination
        // holding the wrong bytes decodes to finite, plausibly-shaped, wrong values
        // and surfaces as a NaN several layers later.
        //
        // Behind the feature because it reads BOTH ENDS OF EVERY RECORD — megabytes
        // off the device per pass, which is what starved a run down to a single
        // compaction. Full coverage is the point (a 16-slot sample cannot tell "the
        // copy is correct" from "the corruption missed my sample"), and full coverage
        // is exactly what makes it too expensive to leave on.
        #[cfg(feature = "tensor-assert")]
        if let Err(e) = self.verify_copied_bytes(&records) {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                "compaction copy did not reproduce the source bytes: {e}",
            );
            // Nothing is rewritten yet — the holders still name their sources, which
            // still hold the data — so this is still the safe point to abandon.
            return Err(CompactionRefused::Fault(e.to_string()));
        }

        report.timings.barrier = phase.elapsed();
        phase = Instant::now();

        // ── Rewrite every holder: ours, then the caller's ────────────────────
        // Gids, and a freshly minted record for each rewritten chunk that had one. No
        // record is ever rewritten — see the header.
        let mut mint = match super::super::compact_mint::RecordMint::new(self) {
            Ok(m) => m,
            Err(e) => {
                tracing::error!(
                    target: "candle_nn::kv_cache::compact",
                    "compaction cannot mint records at this geometry, abandoning the pass \
                     before anything is published: {e}",
                );
                return Err(CompactionRefused::Fault(e.to_string()));
            }
        };
        // **Every record arena this pass needs is created HERE, before a single holder is
        // touched.** A record claim that had to create an arena mid-sweep would register
        // one, and a registration can re-tenant an `arena_idx` that holders the sweep has
        // not reached are still naming — after which their gids resolve against an arena
        // of the wrong stride. That is what `arena slot: 2176 B requested from a 1152 B
        // slot` was, 198 of them on a run that answered 1/8 against none on the runs that
        // answered 8/8. `map.len()` is the upper bound on mints, so this over-provisions
        // by design.
        if let Some(m) = mint.as_mut() {
            // **Bounded by the move count: one record per move is the most a pass can mint.**
            // A record is minted per relocated *chunk*, and a chunk covers up to
            // `n_kv_head · n_palette · 2` bands — but only some of them move when a pass
            // drains the arenas at the frontier, so dividing the moves by that factor
            // under-counts the chunks. Sized that way on Qwen3-30B-A3B, passes of ~115,000
            // moves touched ~8,000 chunks against a reservation for ~3,600, declined
            // thousands of records each, and left the frontier where it stood for a dozen
            // passes in a row.
            //
            // Over-provisioning cannot raise the frontier: `reserve_record_slots` claims a
            // region only while a hole stands below it, so a surplus arena lands in ground
            // already free, and an arena no mint reaches is released when the pass ends.
            let want = map.len().max(1);
            if let Err(e) = m.reserve(self, want, record_sources, &mut fresh) {
                tracing::error!(
                    target: "candle_nn::kv_cache::compact",
                    "compaction could not provision record slots, abandoning the pass \
                     before anything is published: {e}",
                );
                return Err(CompactionRefused::Fault(e.to_string()));
            }
        }
        let mut sweep_state = match mint.as_mut() {
            Some(m) => Sweep::with_mint(&map, m, self),
            None => Sweep::new(&map),
        };
        // **Every backing, one sweep.** Sharing the `Sweep` across them is not an
        // optimisation: layers routinely hold the SAME `HeadGids` allocation, so a
        // per-backing sweep would give each its own equal-but-distinct replacement
        // and the refcounts would then disagree with the sharing the cache believes
        // exists.
        // **A sweep that fails part-way has PUBLISHED part-way, and the records it
        // minted still need their bytes.**
        //
        // `rewrite_own_holders` mutates each backing's block tables in place, and the
        // caller's closure installs per residence, so an error on backing *k* or
        // residence *k* leaves everything before it holding fresh `MetaGid`s — whose
        // slots are filled only by the batched launch below. Returning here without
        // that launch would leave live chunks pointing at record slots holding the
        // previous tenant's record: well-formed band pointers into ground that has
        // since been freed and reissued, which is the original corruption exactly.
        //
        // So the outcome is *captured* and the fill, the invalidation and the fence all
        // run regardless; only the release of the emptied arenas is skipped. What that
        // leaves is a partially-applied pass, and a partially-applied pass is sound for
        // the same reason an unreached holder is: the holders that were rewritten name
        // destinations and carry records naming destinations, the holders that were not
        // name sources and carry records naming sources, and each side keeps its own
        // ground alive. The comment this replaces claimed "nothing published", which was
        // true before a sweep minted anything.
        let swept_ok = backings
            .iter()
            .try_fold((), |_, b| {
                b.rewrite_own_holders(&mut sweep_state).map(|_| ())
            })
            .and_then(|_| sweep(&mut sweep_state));
        if let Err(e) = &swept_ok {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                "compaction's holder sweep failed part-way; publishing what it installed \
                 and releasing nothing: {e}",
            );
        }
        // **Relocated slots no holder this sweep reached names.**
        //
        // Pure reclaim accounting now, not a correctness proxy. Nothing is freed by
        // this pass unless every party naming it let go, and the records let go only
        // when their chunk does — so an unreached holder costs the ground it is sitting
        // on and cannot cost correctness. Counted because the holder list is maintained
        // by hand and a new holder nobody sweeps is otherwise invisible: a rising
        // reading is how one is discovered.
        report.unwitnessed = sweep_state.unwitnessed(&map).len();
        #[cfg(feature = "tensor-assert")]
        {
            let foreign = window.foreign();
            if foreign != 0 {
                tracing::warn!(
                    target: "candle_nn::kv_cache::compact",
                    foreign,
                    unwitnessed = report.unwitnessed,
                    source_collisions = report.source_collisions,
                    "chunk slots were freed on other threads inside this pass",
                );
            }
        }
        if report.unwitnessed != 0 {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                moved = map.len(),
                unwitnessed = report.unwitnessed,
                "compaction relocated slots no holder it reached names — the sweep's \
                 holder set is incomplete, so those claims are wasted and their \
                 sources are not reclaimed",
            );
        }
        report.allocations_rewritten = sweep_state.allocations_rewritten();
        report.clean_reused = sweep_state.clean_reused();
        report.records_moved = map.records_len();
        report.records_unfollowed = sweep_state.records_unfollowed();
        report.timings.sweep = phase.elapsed();
        phase = Instant::now();

        // ── Drop every cached decode buffer serialised from replaced ground ──
        // A `kvheads_ptr` is a record's address and the band pointers behind it are the
        // bands', both cached in a slice header and owned by nothing. The live slots are
        // already covered: `rewrite_for_compaction` clears the buffer of every slot whose
        // gids moved or whose record was followed to a copy. This catches a buffer that
        // names replaced ground through a chunk *not* in its slot's own block table — a
        // holder the sweep does not reach — and names the slot, so the finding points at
        // it. Decided from each buffer's own pins, so a buffer that names nothing replaced
        // is kept and not miscounted.
        //
        // **Before the sweep drops**, because the replaced set is identified by allocation
        // address and record raw id, which only the sweep's own pins keep from being
        // reused. The minted records' bytes are not needed for this: it is host-only.
        //
        // Fatal on a poisoned lock: every holder has been installed, so there is nothing to
        // return to, and an unchecked buffer may be naming ground the pass is about to hand
        // back.
        let replaced = sweep_state.replaced();
        // A pass whose claims all landed and whose holders all turned out to be elsewhere
        // replaced nothing; there is no buffer to check against it.
        let checked: &[ChunkedKvBacking] = if replaced.is_empty() { &[] } else { backings };
        for (layer, b) in checked.iter().enumerate() {
            match b.invalidate_decode_buffers_naming(&replaced) {
                Ok(slots) if slots.is_empty() => {}
                Ok(slots) => {
                    tracing::warn!(
                        target: "candle_nn::kv_cache::compact",
                        layer,
                        ?slots,
                        "a cached decode buffer named ground this compaction replaced, but \
                         its slot's own rewrite did not clear it — a holder of those chunks \
                         the sweep does not reach. Cleared here; its old ground stays pinned \
                         until then",
                    );
                    report.decode_buffers_cleared += slots.len();
                }
                Err(e) => panic!(
                    "compaction published {} band and {} record relocations and then could \
                     not check the cached decode buffers against them: {e}. A buffer naming \
                     replaced ground would go on being read.",
                    map.len(),
                    map.records_len(),
                ),
            }
        }
        drop(replaced);
        report.timings.invalidate = phase.elapsed();
        phase = Instant::now();

        // ── Fill the minted records, in one launch ───────────────────────────
        // **Every rewritten chunk already holds its new record; this writes the bytes.**
        // The slots were claimed during the sweep (host-only) so the holder could install
        // the handle in the same visit, and the fill is batched to here because it is a
        // device launch and there is exactly one per pass however many records were
        // minted.
        //
        // Ordered before the barrier below deliberately: that barrier is what makes these
        // writes visible to every other stream, and `release_empty_arenas` after it hands
        // back the regions the old records' deaths just emptied. Nothing reads a minted
        // record in between — the pass holds the arena window, so no forward is in flight.
        //
        // Fatal rather than a refusal, for the same reason the barrier is: every holder's
        // gids and record have already been installed, so there is nothing to return to.
        // A record left unfilled holds whatever its slot held before, which is another
        // chunk's record or uninitialised ground.
        drop(sweep_state);
        if let Some(m) = mint.as_mut() {
            report.records_declined = m.declined();
            report.record_arenas_reserved = m.reserved_arenas();
            if report.records_declined != 0 {
                // Correct but degraded: those chunks stayed whole on their sources —
                // gids and record together (`Sweep::remint`) — so nothing is reclaimed
                // for them this pass and a pass with record room moves them. A standing
                // non-zero reading means the reservation is under-provisioning.
                tracing::warn!(
                    target: "candle_nn::kv_cache::compact",
                    declined = report.records_declined,
                    minted = m.len(),
                    "compaction ran out of pre-provisioned record slots mid-sweep; those \
                     chunks stay on their sources until a later pass",
                );
            }
            match m.flush(self) {
                Ok(n) => report.records_minted = n,
                Err(e) => panic!(
                    "compaction published {} relocations and then could not fill the {} \
                     records it minted for them: {e}. Those holders now name records whose \
                     bytes were never written, so the paged kernels would dereference \
                     whatever the slots held before.",
                    map.len(),
                    m.len(),
                ),
            }
        }

        report.timings.mint = phase.elapsed();
        phase = Instant::now();

        // Device-wide, not per-stream: one barrier at between-forwards cadence
        // buys the removal of the entire overlap failure domain, side streams this
        // module does not know about included.
        // **Fatal, not a refusal: the gid rewrite above is already published.** This
        // barrier is what makes the copies visible to every other stream before the
        // next line hands any emptied region back. Carrying on without it releases
        // regions while kernels may still be reading them, and the pool's next tenant
        // then zeroes ground a live reader is in — which reports as an illegal address
        // in some unrelated kernel, or as nothing at all.
        if let Err(e) = self.device().synchronize() {
            panic!(
                "compaction published {} relocations and then could not fence them: \
                 {e}. The copies are not known to have retired, and the next step \
                 returns emptied regions to the pool — releasing ground that may still \
                 have readers.",
                map.len(),
            );
        }

        // **Every reference the pass leaves behind, re-checked now that it has
        // published, and every slot's content against what it held on the way in.** An
        // orphan count that rose across the pass means some party names a slot the
        // refcount tables call free — which a record now cannot, because it holds one of
        // those refcounts, so a rise here is a *gid* holder nobody swept. A content hash
        // that moved means it has already been cashed in: something is reading ground
        // another chunk now owns.
        #[cfg(feature = "tensor-assert")]
        super::super::kv_integrity::report_boundary(
            backings,
            "after a compaction",
            report.moves,
            report.unwitnessed,
            Some(&entering),
        );

        // **No emptied source is released after a part-way sweep.** Everything above has
        // been published and fenced, so the pool is consistent — but the holders the sweep
        // did not reach still name sources, and handing regions back is the one step that
        // depends on having reached all of them. Refusing here costs the reclaim and
        // nothing else. The arenas this pass provisioned are still handed back when
        // `fresh` drops on the way out: an empty one is named by nothing, because every
        // destination and every mint in it keeps it occupied.
        if let Err(e) = swept_ok {
            return Err(CompactionRefused::SweepFailed(e.to_string()));
        }

        // Whatever the replacements above emptied is tombstoneable now. A source a
        // surviving holder or its record still names is NOT empty, so this reclaims only
        // the chunks nothing is left pointing at.
        //
        // The arenas this pass provisioned go first, and are counted: until the holder
        // sweep was over a released one was an `arena_idx` open to re-tenancy under
        // holders it had not reached, so they were held until here.
        report.fresh_arenas = fresh.created();
        let fresh_released = fresh.finish();
        report.arenas_released = fresh_released + self.release_empty_arenas().unwrap_or(0);
        report.frontier_after = self.frontier_regions().unwrap_or(frontier_before);
        report.top_arena = SizeClass::all()
            .map(|class| ArenaKey::new(class, ArenaLocation::Gpu))
            .chain(record_key)
            .filter_map(|key| {
                self.pool_top_rank(key)
                    .map(|(rank, live)| TopArena { key, rank, live })
            })
            .max_by_key(|t| t.rank);
        report.timings.publish = phase.elapsed();
        Ok(report)
    }
}
