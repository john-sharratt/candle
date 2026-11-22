//! The gate that decides when a KV arena may be created, and what happens to a
//! pass it refuses.
//!
//! Creating an arena moves the frontier a running wave's transient tier was
//! placed against, so arenas are created only between forwards. A creation
//! refused mid-wave is recorded as a demand, and the next gap creates the arena
//! so the retry only fills it. A CPU arena claims no region and is not gated.
//! An operation that cannot be refused part-way proceeds and records its class.

use super::*;

/// Gaps an arena made ahead of its demand stays protected while the demand
/// has not come: long enough for the refused pass to return, short enough
/// that a demand abandoned does not hold a region for good.
#[cfg(feature = "cuda")]
pub(super) const FRESH_GAPS: u32 = 64;

/// Most arenas one refused pass may owe a class: a mis-sized demand cannot
/// owe the whole span.
#[cfg(feature = "cuda")]
pub(super) const MAX_OWED_PER_CLASS: usize = 64;

/// Arenas of `per`-slot capacity a class needing `slots` slots, with `free`
/// slots free and `owed` arenas already owed, is owed more — the shortfall in
/// whole arenas, less what is already owed, capped (`owe_capacity`).
#[cfg(feature = "cuda")]
pub(super) fn more_arenas_owed(slots: usize, free: usize, per: usize, owed: usize) -> usize {
    let short = slots.saturating_sub(free);
    let arenas = short.div_ceil(per.max(1)).min(MAX_OWED_PER_CLASS);
    arenas.saturating_sub(owed)
}

impl BackingInner {
    /// Where this backing's *hot* tier is: the device it was built on.
    ///
    /// A CPU-only backing has one tier, so lifting out of warm lands back in
    /// host memory. Asking for `Gpu` there would name a region of a reservation
    /// that does not exist.
    pub(in super::super) fn hot_location(&self) -> ArenaLocation {
        if self.device.is_cpu() {
            ArenaLocation::Cpu
        } else {
            ArenaLocation::Gpu
        }
    }

    /// Elements one chunk slot holds at this backing's geometry — one
    /// `(head, palette-band, side)` of `CHUNK_SIZE` tokens.
    ///
    /// The single authority for the number every size-class question is asked
    /// with, so a model whose `head_dim / n_palette` is not 32 cannot end up
    /// with half its arenas sized from the assumption that it is (audit A9).
    ///
    /// Uses the backing's OWN band count (`n_palette()`): a single-latent
    /// (DeepSeek) backing carries [`LATENT_N_BANDS`] bands of
    /// `head_dim / LATENT_N_BANDS` dims, so sizing its slots at the GQA
    /// `head_dim / N_PALETTE` width would quadruple every payload (512/4=128
    /// vs 512/16=32 dims) and push R16 past the size-class ladder.
    pub(in super::super) fn elems_per_chunk(&self) -> usize {
        elems_per_chunk((self.head_dim / self.n_palette()).max(1))
    }

    /// The arena key a chunk of `format` allocates from at this geometry.
    pub(in super::super) fn arena_key_for(
        &self,
        format: KvFormat,
        location: ArenaLocation,
    ) -> Result<ArenaKey> {
        ArenaKey::for_format(format, self.elems_per_chunk(), location)
    }

    /// Claim the gap between forwards for one arena, or refuse.
    ///
    /// `None` — no gate at all — in the two cases where creating an arena cannot
    /// disturb a wave:
    ///
    /// - **a CPU arena**, which is a host `Tensor::zeros` ([`Self::claim_slab`]).
    ///   It claims no region, sits nowhere in the reservation, and cannot move
    ///   the arena frontier the transient tier was placed against. Gating it was
    ///   a bug, not caution: the hot→warm migration allocates its warm copies
    ///   exactly this way, under the sequence-state write lock, and blocking
    ///   there deadlocked the daemon against a forward that wanted the same lock
    ///   mid-layer.
    /// - **a CPU device**, which has no reservation and no tier.
    ///
    /// Every caller of [`Self::create_arena`] must hold one of these. The one
    /// that cannot take it here — [`ChunkedKvBacking::alloc_chunk_with_arenas`]
    /// creates arenas with the storage write lock already held — takes it at its
    /// own call site instead, above that lock.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn arena_window(&self, key: ArenaKey) -> Result<Option<ArenaWindow>> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(None);
        };
        if key.location == ArenaLocation::Cpu {
            return Ok(None);
        }
        enter_arena_window(cd.cuda_stream(), "a KV arena").map(Some)
    }

    /// The same gate for an operation that **cannot be refused part-way**.
    ///
    /// A fork is per *layer* — `KvCache::fork` runs once per layer and the caller
    /// loops over all forty-eight — and there is no transaction across them. So a
    /// refusal on layer N leaves N forked layers and 48−N unforked ones, which is
    /// not a deferred fork: it is a sequence whose token windows differ by layer,
    /// which no single decode position map can describe. It surfaced as
    /// `chunked decode layout diverged across layers … could not be reconciled`
    /// on the first message of a new conversation.
    ///
    /// Refusing costs more than proceeding here, so this one proceeds — but
    /// only past the *wave-in-flight* refusal. That refusal it absorbs and
    /// **records the class**, so the next inter-forward gap creates an arena
    /// for it and the case stops arising; the region claim is still counted by
    /// `fresh_claims_during_wave`, so an engine that takes this path often says
    /// so in the `in-wave-arenas` tripwire rather than hiding it. Every other
    /// error — a poisoned gate, a driver fault — is not a deferral and
    /// propagates: proceeding past a broken gate would fork against arenas
    /// whose state can no longer be trusted.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn arena_window_uninterruptible(
        &self,
        key: ArenaKey,
    ) -> Result<Option<ArenaWindow>> {
        match self.arena_window(key) {
            Ok(w) => Ok(w),
            Err(e) if Self::is_wave_deferral(&e) => {
                self.note_deferred_arena(key);
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// Whether an allocation failure is the wave-in-flight deferral.
    ///
    /// A deferral means "the ground exists, come back between forwards"; every
    /// other failure here means "there is no ground". They call for opposite
    /// responses, so no path may fold one into the other.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn is_wave_deferral(e: &candle::Error) -> bool {
        e.to_string().contains(KV_ARENA_MID_WAVE)
    }

    /// Remember that a wave-in-flight refusal wanted an arena of this class.
    ///
    /// Idempotent per class: one arena of a class serves every chunk that fits
    /// it, so a pass refused twenty times for `1088 B` needs one arena, not
    /// twenty. Recording the count would over-create on the very workload the
    /// refusal is most common on.
    #[cfg(feature = "cuda")]
    pub(super) fn note_deferred_arena(&self, key: ArenaKey) {
        if let Ok(mut pending) = self.deferred_arenas.lock() {
            if !pending.contains(&key) {
                pending.push(key);
            }
        }
    }

    /// **Owe what a refused pass will need, not one arena of each class it
    /// asked for.** `demand` is `(class, slots)` for the work the pass has yet
    /// to place; each class is owed the arenas its free slots fall short of,
    /// so the gap between forwards creates them all and the retry only fills.
    ///
    /// One arena a class was owed before. A hot→warm demotion group needs many
    /// regions of one class — a few hundred MB of a size class is twenty
    /// arenas — so each arena cost a full round trip: refused mid-wave, owed,
    /// created in the gap, retried, refused again at the next. On `.5`
    /// (2026-10-18, sustained load) the sixteen retries a pass gets could not
    /// keep pace with decode KV growth, demotion fell behind, and the span
    /// ended 100% hot with every turn failing.
    ///
    /// Free slots are counted, not runs, so a class whose free slots are
    /// scattered can still be refused once more; the retry covers that.
    /// Capped per class, so a mis-sized demand cannot owe the whole span.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn owe_capacity(&self, demand: &[(ArenaKey, usize)]) {
        let Ok(mut pending) = self.deferred_arenas.lock() else {
            return;
        };
        for &(key, slots) in demand {
            let free = self.pool.free_slots(key);
            let owed = pending.iter().filter(|k| **k == key).count();
            let more = more_arenas_owed(slots, free, key.chunks(), owed);
            for _ in 0..more {
                pending.push(key);
            }
            if slots <= free {
                continue;
            }
            // A shortfall the one-arena reserve would have covered is a
            // restoring bug; one past it is demand the gap had to buy.
            let reserve = self.pool.reserve_target(key);
            let covered = reserve.is_some_and(|t| slots <= t);
            macro_rules! short {
                ($level:ident) => {
                    tracing::$level!(
                        target: "candle_conversation::kv_reserve",
                        class = %key.label(),
                        needed_slots = slots,
                        free_slots = free,
                        reserve_slots = reserve.unwrap_or(0),
                        arenas_owed = more,
                        "kv_reserve: quantize deferred short of {} slots of {}{}",
                        slots - free,
                        key.label(),
                        if covered { " — inside its reserve" } else { "" },
                    )
                };
            }
            if covered {
                short!(warn);
            } else {
                short!(info);
            }
        }
    }

    /// The reserve classes whose free slots have fallen below their one-arena
    /// reserve, each once: the arenas a gap makes to restore them
    /// (`docs/kv_quantize_in_the_gap.md` §3.2). A reserve is the guard against
    /// a card full of native K/V — the gap's first conversion has somewhere to
    /// write — and sized by nothing else: the gap's lease buys whatever more a
    /// table needs, between forwards, where creating an arena is legal. A
    /// class already owed an arena by a refusal is not owed a second.
    #[cfg(feature = "cuda")]
    pub(super) fn reserves_short(&self) -> Vec<ArenaKey> {
        let hot = self.hot_location();
        let owed: Vec<ArenaKey> = self
            .deferred_arenas
            .lock()
            .map(|p| p.clone())
            .unwrap_or_default();
        self.pool
            .reserve_keys()
            .into_iter()
            .filter(|&key| {
                key.location == hot
                    && !owed.contains(&key)
                    && self.pool.free_slots(key) < key.chunks()
            })
            .collect()
    }

    /// Lift the protection of each arena made ahead of demand that the
    /// demand has since filled, that is gone, or that has waited
    /// [`FRESH_GAPS`] gaps for a demand that never came.
    #[cfg(feature = "cuda")]
    pub(super) fn settle_fresh(&self) {
        let Ok(mut fresh) = self.fresh_arenas.lock() else {
            return;
        };
        fresh.retain_mut(|(key, idx, gaps)| {
            *gaps += 1;
            let load = self.pool.pool_arena_load(*key);
            let standing = load.iter().find(|&&(i, _, _)| i == *idx);
            let settled = match standing {
                None => true,
                Some(&(_, _, live)) => live > 0 || *gaps > FRESH_GAPS,
            };
            if settled {
                self.pool.unprotect_arena(*idx);
            }
            !settled
        });
    }

    /// Create the arenas that mid-wave refusals asked for, and answer with how
    /// many were made.
    ///
    /// **Call this between forwards, with no KV lock held** — the top of a
    /// sealing pass is the intended site. It is the other half of the refusal:
    /// the pass that was turned away recorded what it wanted, and this creates
    /// it while the partition is idle, so the pass that follows finds the arena
    /// already there and only ever *fills* it. Filling is legal at any time; it
    /// is creation that moves the arena frontier, and this is what keeps the two
    /// apart without either side waiting on the other.
    ///
    /// Refusals here are ordinary and silent: if a forward has started, the
    /// demand stays recorded and the next gap gets it. Only a genuine allocation
    /// failure — the reservation truly full — propagates.
    ///
    /// It lifts the protection of the arenas it made earlier whose demand has
    /// arrived ([`Self::settle_fresh`]), makes what refusals owe — each
    /// protected until its demand fills it, so the sweep this same gap runs
    /// does not take it first — and then restores every reserve fallen below
    /// its one arena ([`Self::reserves_short`]). A reserve's arena is not
    /// protected: the sweep already holds a reserve class at its arena.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn create_deferred_arenas(&self) -> Result<usize> {
        self.settle_fresh();
        let reserves = self.reserves_short();
        let pending: Vec<ArenaKey> = match self.deferred_arenas.lock() {
            Ok(mut p) => std::mem::take(&mut *p),
            Err(_) => return Ok(0),
        };
        if pending.is_empty() && reserves.is_empty() {
            return Ok(0);
        }
        let mut made = 0usize;
        // `true` for a refusal's demand, `false` for a reserve's.
        let mut remaining = pending
            .into_iter()
            .map(|k| (k, true))
            .chain(reserves.into_iter().map(|k| (k, false)));
        while let Some((key, owed)) = remaining.next() {
            // Every exit that does not finish this key puts back the key **and
            // every refusal's demand after it** — the list was taken whole, so
            // leaving with only the current one restored would silently drop
            // the rest and the classes behind it would never be created. That
            // holds for the benign "a wave started" refusal and for genuine
            // errors alike: an error aborts this drain, not the demand, and the
            // next gap must still see it. A reserve's is found short again at
            // the next gap.
            let requeue =
                |first: (ArenaKey, bool), rest: &mut dyn Iterator<Item = (ArenaKey, bool)>| {
                    for (key, owed) in std::iter::once(first).chain(rest) {
                        if owed {
                            self.note_deferred_arena(key);
                        }
                    }
                };
            // One window per arena rather than one for the batch: a forward that
            // arrives mid-drain then takes the partition at the next arena
            // instead of waiting for the whole list.
            let window = match self.arena_window(key) {
                Ok(w) => w,
                Err(e) if Self::is_wave_deferral(&e) => {
                    requeue((key, owed), &mut remaining);
                    break;
                }
                Err(e) => {
                    requeue((key, owed), &mut remaining);
                    return Err(e);
                }
            };
            let idx = self.pool.register_arena(key);
            match self.create_arena(key, idx).and_then(|arena| {
                self.storage.try_write(|s| {
                    if !s.has_arena(idx) {
                        s.push_arena(arena, idx);
                    }
                    Ok(())
                })
            }) {
                Ok(()) => {
                    // The slab is in storage and this creator is done with the
                    // index, so the window that keeps a tombstoner off it
                    // closes here. There is no first `occupy` to close it the
                    // way an allocate-on-demand creator's does: this arena was
                    // stamped ahead of the demand that will fill it, and an
                    // arena whose window never closes is counted as reclaimable
                    // and then refused by the sweep for the life of the process.
                    self.pool.finish_creation(key, idx);
                    // A refusal's arena is made for a demand that has not
                    // filled it yet: protected until it does, or the sweep this
                    // same gap runs reclaims it first and the pass that wanted
                    // it is refused again. A reserve's needs no protection —
                    // the sweep holds a reserve class at its arena.
                    if owed {
                        self.pool.protect_arena(idx);
                        if let Ok(mut fresh) = self.fresh_arenas.lock() {
                            fresh.push((key, idx, 0));
                        }
                    }
                    made += 1;
                }
                Err(e) => {
                    // The slab (if it was even created) never reached storage:
                    // release the registration so the index is not leaked, and
                    // put the demand back so the next gap retries it.
                    self.pool.force_release_arena(idx);
                    drop(window);
                    requeue((key, owed), &mut remaining);
                    return Err(e);
                }
            }
        }
        Ok(made)
    }

    /// Create one arena: a `chunks x class_bytes` slab of raw bytes.
    ///
    /// **Only ever between forwards.** The caller holds an
    /// [`Self::arena_window`], because this claims a region and so moves the
    /// arena frontier a running wave's transient tier was placed against.
    ///
    /// There is no float/quantized fork any more. A slot is a fixed number of
    /// bytes and its tenant is whatever the owning chunk's tag says it is, so
    /// the allocator has exactly one thing to decide — how wide the slots are —
    /// and the class carries that.
    ///
    /// A GPU arena **carves a region** out of the device reservation: no
    /// `cuMemAlloc`, no pool growth, no chance of the driver spilling the slab
    /// to host memory, and a base pointer that stays valid until the process
    /// ends. Running out is a `KV_DEVICE_OOM_MARKER` error the way exceeding
    /// the old budget was — the difference is that it now means "every region
    /// is occupied", an exact count rather than an estimate of driver headroom.
    pub(in super::super) fn create_arena(&self, key: ArenaKey, index: usize) -> Result<Arena> {
        let t0 = Instant::now();
        let chunks = key.chunks();
        let stride = key.slot_stride();
        let arena_bytes = chunks.saturating_mul(stride);
        let out = self.claim_slab(key, index, arena_bytes);
        record_arena_create("slab", key.location, index, t0.elapsed().as_nanos() as u64);
        out
    }

    #[cfg(feature = "cuda")]
    pub(super) fn claim_slab(
        &self,
        key: ArenaKey,
        index: usize,
        arena_bytes: usize,
    ) -> Result<Arena> {
        if key.location == ArenaLocation::Cpu {
            return Ok(Arena::new(cpu_slab(arena_bytes)?, key, index));
        }
        let Device::Cuda(cuda) = &self.device else {
            candle::bail!("a GPU arena needs a CUDA device, not {:?}", self.device)
        };
        let stream = cuda.cuda_stream();
        let Some(region) = region_pool::claim_region(&stream)? else {
            // **Name which of the two refusals this is.** A claim that runs out
            // of ground buys more from the weight side, so reaching here means
            // one of exactly two things: a wave's tier stands over the free
            // regions (no concession can reach them — the wave must narrow), or
            // the purchase itself was refused because the weight zone is at its
            // floor. They want opposite responses and the message has to say
            // which. It used to report only `live` and assert the reservation
            // was occupied, which sent the first investigation of this looking
            // for a KV leak while the real answer was 31 free regions standing
            // under a 496 MiB tier.
            let s = region_pool::region_stats(stream.context().ordinal());
            let (live, total, blocked, ceiling, tier) = s
                .map(|s| {
                    (
                        s.live,
                        s.total,
                        s.blocked,
                        s.transient_ceiling,
                        s.transient_bytes,
                    )
                })
                .unwrap_or((0, 0, 0, 0, 0));
            if blocked > 0 {
                candle::bail!(
                    "{KV_DEVICE_OOM_MARKER}: no region is claimable for class {} B — the wave's \
                     {} MiB transient tier caps the pool at {ceiling} of {total} regions, all \
                     {live} of which are live. {blocked} regions above the tier are unowned but \
                     {TIER_STANDS_OVER_FREE}. This is a wave too wide for the ground \
                     below the tier, not an exhausted reservation.",
                    key.class.bytes(),
                    tier / (1 << 20),
                )
            }
            // A region under an arena that holds no chunk is occupied in name only, so
            // say how many there are and what kept the sweep off them.
            let empty = self.pool.empty_census();
            let holding: Vec<String> = self
                .pool
                .occupancy()
                .iter()
                .map(|o| {
                    format!(
                        "{:?} class {} B: {} arenas, {} of {} slots live",
                        o.key.kind,
                        o.key.class.bytes(),
                        o.arenas,
                        o.live,
                        o.arenas * o.arena_chunks,
                    )
                })
                .collect();
            candle::bail!(
                "{KV_DEVICE_OOM_MARKER}: no region is claimable for class {} B — every one of \
                 the KV reservation's {total} regions is occupied ({live} live), and the weight \
                 side would not sell any. It is at its floor: the fewest expert slots the cache \
                 can serve a token with. The partition has nothing left to trade. This \
                 backing's arenas holding no chunk: {} ({} inside their creation window, {} \
                 protected). Held by: {}.",
                key.class.bytes(),
                empty.empty,
                empty.in_creation,
                empty.protected,
                holding.join("; "),
            )
        };
        // A lease over the region: writes through it land in the reservation,
        // and dropping the tensor frees nothing. What returns the bytes is the
        // region handle the arena carries (§3.7).
        //
        // SAFETY: the region is `REGION_BYTES` of mapped, read/write device
        // memory that only this arena holds, and `arena_bytes` never exceeds a
        // region (asserted by the size-class ladder).
        let data = unsafe {
            Tensor::from_leased_cuda_ptr(
                region.base(),
                DType::U8,
                arena_bytes,
                &self.device,
                LeaseOrigin::Foreign,
            )?
        };
        // The arena's place in every allocation and release choice is where its
        // region sits, known only now.
        self.pool.set_arena_rank(key, index, region.index());
        Ok(Arena::new(data, key, index).in_region(region))
    }

    #[cfg(not(feature = "cuda"))]
    pub(super) fn claim_slab(
        &self,
        key: ArenaKey,
        index: usize,
        arena_bytes: usize,
    ) -> Result<Arena> {
        let data = match key.location {
            ArenaLocation::Gpu => Tensor::zeros(arena_bytes, DType::U8, &self.device)?,
            ArenaLocation::Cpu => cpu_slab(arena_bytes)?,
        };
        Ok(Arena::new(data, key, index))
    }
}

/// A warm-tier (host RAM) arena's storage, **uninitialised**.
///
/// Every byte of it that is ever read was written first: a slot is filled by
/// the hot→warm migration over exactly its band's payload, and read back —
/// by an elevate to hot or the warm→cold gather — over that same payload
/// length, taken from the chunk's own band tag. The tail of a
/// slot past its payload is never read. Nor does the slab change hands: it is a
/// fresh heap block freed when the arena is released, never another tenant's
/// ground the way a device region is (`region_pool::claim_region` zeroes those).
/// Zeroing it was a full-width write of up to an arena per migration, over
/// bytes the migration then overwrote.
pub(super) fn cpu_slab(arena_bytes: usize) -> Result<Tensor> {
    Tensor::empty(arena_bytes, DType::U8, &Device::Cpu)
}
