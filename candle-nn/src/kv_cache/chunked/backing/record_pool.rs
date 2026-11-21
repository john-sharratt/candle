//! Allocates and fills the device-resident KvHead records that paged attention
//! reads.
//!
//! A chunk's record holds the band pointers attention dereferences, so its address
//! must stay fixed for the life of its handle and must come from the arena the
//! record is actually in. Records are filled on the device in one launch from a
//! descriptor table, with no host serialization, and tests hold that fill against
//! the host serializer byte for byte. A compaction reserves record arenas before
//! its sweep, because creating one mid-sweep can re-tenant an index that a holder
//! still names.

use super::*;

impl ChunkedKvBacking {
    /// Device address of a resident KV-head record, for embedding into a slice's
    /// `kvheads_ptr`. Returns 0 for a CPU/host-only pool (no device residence).
    /// The address is cached on the handle at allocation, so this is a field read.
    pub fn meta_device_addr(&self, meta: &super::super::meta_pool::MetaGid) -> u64 {
        meta.device_addr()
    }

    /// Whether this backing's records live in device memory, and so whether there is
    /// anything to build or mint at all.
    pub(in super::super) fn records_are_resident(&self) -> bool {
        self.inner.meta_pool.is_device_resident()
    }

    /// Claim one record slot **from an arena that already exists**, or `None`.
    ///
    /// Host-only and, crucially, **non-creating**: a free-list pop and nothing else.
    ///
    /// # Why a compaction may not use the promoting claim
    ///
    /// [`Self::alloc_chunk_for_key`] widens to `claim_slot_promoting`, which on a miss
    /// calls `release_empty_arenas()` and stamps a fresh region. Both are safe at a seal
    /// boundary and **not** safe inside a compaction's holder sweep: releasing an arena
    /// frees its `arena_idx`, and the next registration re-tenants that index for a
    /// different key — `ChunkGidPool::register_arena` calls it out as the
    /// index-re-tenancy signature. Half-way through a sweep, gids naming the old arena at
    /// that index are still installed in holders the pass has not reached, so they resolve
    /// against whatever now sits there. When that is a *record* arena the stride is wrong,
    /// which surfaces as `arena slot: 2176 B requested from a 1152 B slot` out of the
    /// hot→warm migrate — 198 of them on the run that answered 1/8, and none on the runs
    /// that answered 8/8.
    ///
    /// So the pass provisions the pool up front ([`Self::reserve_record_slots`]) and takes
    /// only what already exists while it is sweeping.
    ///
    /// # Where the slot comes from
    ///
    /// `placement.order` is tried first, arena by arena — the pool's arenas lowest in the
    /// span first — so a mint lands as low as the pool allows rather than wherever the
    /// lowest arena *index* is, which is unrelated to address. A mint left to index order
    /// can land in the very arena the same pass is draining, and keep it alive.
    ///
    /// No slot in `placement.avoid` is ever returned: those are this pass's planned record
    /// sources. One is on the free list only if its record died after the census, and a
    /// mint written there would land in ground the pass has declared it is reading.
    #[allow(dead_code)] // the only caller is `compact_mint`, which is cuda-gated
    pub(in super::super) fn try_alloc_record_slot(
        &self,
        key: ArenaKey,
        placement: &RecordPlacement,
    ) -> Option<ChunkGid> {
        // **Held, not dropped, until a materialised one is found.** The pool can hold a
        // registration whose storage was never materialised, and a record in one has no
        // address — but dropping such a gid returns the slot to the head of the same free
        // list, so the next call pops it again and declines again. One unmaterialised
        // registration would turn *every* remaining mint in the pass into a decline while
        // the reservation reported success, because `reserve_record_slots` counts that
        // arena's capacity as available.
        //
        // Holding the rejects keeps them out of the way for this claim and releases them
        // on return, which is the right lifetime: the pass is inside the arena window, so
        // nothing else is claiming, and a registration that is unmaterialised now will
        // still be unmaterialised on the next mint.
        //
        // A planned source is held the same way, for the same reason: dropped, it would be
        // popped again by the next claim.
        let usable = |gid: &ChunkGid| {
            !placement
                .avoid
                .contains(&(gid.arena_idx(), gid.chunk_idx() as u32))
                && self
                    .inner
                    .storage
                    .read(|s| s.has_arena(gid.arena_idx()))
                    .unwrap_or(false)
        };
        let mut rejected: Vec<ChunkGid> = Vec::new();
        for &arena in &placement.order {
            while let Some(gid) = self.inner.pool.allocate_from_arena(key, arena) {
                if usable(&gid) {
                    return Some(gid);
                }
                rejected.push(gid);
            }
        }
        // Arenas the order does not know of: any the pool registered after it was taken.
        let found = loop {
            let Some(gid) = self.inner.pool.allocate_for(key) else {
                break None;
            };
            if usable(&gid) {
                break Some(gid);
            }
            rejected.push(gid);
        };
        drop(rejected);
        found
    }

    /// The pool's arena indices, lowest in the span first — the order a mint fills them
    /// in. Arenas with no region (host arenas) have no position and are left out.
    #[allow(dead_code)] // the only caller is `compact_mint`, which is cuda-gated
    pub(in super::super) fn arenas_by_rank(&self, key: ArenaKey) -> Vec<usize> {
        let load = self.inner.pool.pool_arena_load(key);
        let mut ranked: Vec<(usize, usize)> = self
            .inner
            .storage
            .read(|s| {
                let arenas = s.arenas();
                load.iter()
                    .filter_map(|(idx, _, _)| {
                        arenas
                            .get(idx)
                            .and_then(|a| a.region_rank())
                            .map(|r| (r, *idx))
                    })
                    .collect()
            })
            .unwrap_or_default();
        ranked.sort_unstable();
        ranked.into_iter().map(|(_, idx)| idx).collect()
    }

    /// Make sure the record pool for `key` can satisfy `want` claims without creating an
    /// arena, and answer how many arenas that took.
    ///
    /// **Called before the sweep, which is the only safe place to create one.** See
    /// [`Self::try_alloc_record_slot`] for what goes wrong if an arena is created or
    /// released while holders are half-rewritten.
    ///
    /// Every arena it creates is noted in `fresh`, which keeps the empty sweep off it
    /// until the pass ends and then releases any the mints never reached. See
    /// [`FreshArenas`].
    #[allow(dead_code)] // the only caller is `compact_mint`, which is cuda-gated
    pub(in super::super) fn reserve_record_slots(
        &self,
        key: ArenaKey,
        want: usize,
        fresh: &mut FreshArenas<'_>,
    ) -> Result<usize> {
        let free = |b: &Self| -> usize {
            b.inner
                .pool
                .pool_arena_load(key)
                .iter()
                .map(|&(_, capacity, live)| capacity.saturating_sub(live))
                .sum()
        };
        let mut created = 0usize;
        // Each turn either adds an arena's worth of free slots or gives up, so this
        // terminates on the free count alone. A refused region is not an error: it says
        // the KV side is full, and the caller then mints fewer records — recorded as
        // `records_declined` — rather than failing a pass that has published nothing.
        //
        // **And a claim is only taken while a hole exists below the frontier.** The region
        // free list is lowest-index-first, so with holes a fresh arena lands *under* the
        // frontier and costs nothing; with none it lands above and raises the very number
        // the pass exists to lower. `provision_low_arenas` stops for the same reason,
        // and this is the same trade: declining a mint costs one chunk's reclaim, raising
        // the frontier costs the weight side a region.
        while free(self) < want && self.has_region_hole() {
            if fresh.claim(&self.inner, key).is_err() {
                break;
            }
            created += 1;
        }
        Ok(created)
    }

    /// Write a batch of already-claimed records' bytes with one launch.
    ///
    /// The seal path reaches the same kernel through [`Self::build_meta_records`], which
    /// claims the slots itself; a compaction has already claimed them during its holder
    /// sweep and only needs the fill.
    #[allow(dead_code)] // the only caller is `compact_mint`, which is cuda-gated
    pub(in super::super) fn fill_record_batch(
        &self,
        handles: &[super::super::meta_pool::MetaGid],
        srcs: &[super::super::meta_pool::ChunkRecordSrc<'_>],
        arena_info: &[ResolvedArenaInfo],
        layout: RecordLayout,
    ) -> Result<()> {
        self.fill_records_on_device(
            handles,
            srcs,
            arena_info,
            layout.n_kv_head,
            layout.head_dim,
            layout.n_palette,
        )
    }

    /// Read every freshly filled record back and compare it against what
    /// [`serialize_kv_heads`](super::meta_pool::serialize_kv_heads) would have written.
    ///
    /// The fill kernel is held against that serializer byte-for-byte by unit test over six
    /// geometries; this asserts the same thing for a compaction's batch, against the live
    /// arena table and real relocated chunks. It exists to tell a **wrong record** apart
    /// from a **wrong reader of a right record**, which no other check here can do.
    ///
    /// One synchronous readback per record, so it is behind the harness and never in a
    /// production pass.
    #[cfg(all(feature = "cuda", feature = "tensor-assert"))]
    pub(in super::super) fn verify_minted_records(
        &self,
        handles: &[super::super::meta_pool::MetaGid],
        srcs: &[super::super::meta_pool::ChunkRecordSrc<'_>],
        arena_info: &[ResolvedArenaInfo],
        layout: RecordLayout,
    ) -> Result<()> {
        let mut want = vec![0u8; layout.record_bytes];
        for (i, (h, src)) in handles.iter().zip(srcs).enumerate() {
            if h.device_addr() == 0 {
                continue;
            }
            want.fill(0);
            super::super::meta_pool::serialize_kv_heads(
                &mut want,
                src,
                layout.n_kv_head,
                layout.head_dim,
                layout.n_palette,
                arena_info,
            );
            let Some(got) = self.read_record(h.device_addr(), layout.record_bytes) else {
                continue;
            };
            if got != want {
                let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
                candle::bail!(
                    "compaction minted record {i} of {} at {:#x} and the device bytes \
                     differ from the serializer's at byte {at} (device {:#04x}, expected \
                     {:#04x}). Every kernel reaches this chunk's K/V through these words, \
                     so a wrong one reads another chunk's bands — finite, plausibly shaped \
                     and wrong, with nothing to fault.",
                    handles.len(),
                    h.device_addr(),
                    got[at],
                    want[at],
                )
            }
        }
        Ok(())
    }

    /// The record arena key and geometry this backing's records are built at, with the
    /// two guards that must hold before a slot is claimed.
    ///
    /// **Derived per call, not held.** The record size follows the backing's *live*
    /// `n_palette()` — the single latent carries four times GQA's bands — so there is
    /// nothing cached to go stale against the geometry. See `docs/vram_span_partition.md`
    /// §8.
    ///
    /// Both checks run **before** anything allocates. They used to sit inside
    /// `fill_records_on_device`, after the claims, so a non-conforming geometry allocated
    /// and then freed N record slots on every seal for the life of the process. The size
    /// check is here rather than inside `ArenaKey::for_records` because that function
    /// *panics*, and this runs on the seal and persistence threads — a panic there poisons
    /// the locks the rest of this module `expect`s on, turning one bad geometry into
    /// unrelated failures everywhere.
    ///
    /// One function because both the seal path ([`Self::build_meta_records`]) and a
    /// compaction's mint ([`super::compact_mint::RecordMint`]) ask the same question, and
    /// two copies of these guards would be two opinions about what a record is.
    pub(in super::super) fn record_layout(&self) -> Result<RecordLayout> {
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;
        let n_palette = self.inner.n_palette();
        let record_bytes =
            super::super::meta_pool::chunk_record_bytes(n_kv_head, head_dim, n_palette);
        if record_bytes
            > super::super::arena::RECORD_STRIDES[super::super::arena::RECORD_STRIDES.len() - 1]
        {
            candle::bail!(
                "a {record_bytes} B KvHead record ({n_kv_head} heads x head_dim \
                 {head_dim} x {n_palette} bands) exceeds the largest record stride. Add a \
                 rung to RECORD_STRIDES, checking it against GID_STRIDE."
            );
        }
        // The fill kernel stores band pointers as `uint64_t` at `head_dim / 2 + p * 8`
        // within heads `head_sz` apart, so both must be 8-aligned or the store faults.
        // `head_dim % 16 == 0` gives the first, `n_palette % 4 == 0` the second. The host
        // serializer never had to care — unaligned writes are legal there — which is why
        // this was invisible until the fill moved to the device.
        let head_sz = head_dim / 2 + n_palette * 26;
        if !head_dim.is_multiple_of(16) || !head_sz.is_multiple_of(8) {
            candle::bail!(
                "KvHead record geometry is not 8-byte aligned: head_dim {head_dim} \
                 (needs % 16 == 0) and per-head size {head_sz} (needs % 8 == 0) at \
                 n_palette {n_palette}. The fill kernel stores band pointers as uint64_t, \
                 which faults on an unaligned address."
            );
        }
        Ok(RecordLayout {
            key: ArenaKey::for_records(ArenaLocation::Gpu, record_bytes),
            record_bytes,
            n_kv_head,
            head_dim,
            n_palette,
        })
    }

    /// Build device-resident KV-head metadata records for a batch of chunks and
    /// return one handle per input. Serializes each `KvHead[n_kv_head]` record
    /// (pal/scale/fmt + the per-palette pointers resolved against `arena_info` at the
    /// chunk's current placement) and writes them all with **one device launch** rather
    /// than a tiny copy per chunk. Called at the finalization sites — quantize,
    /// cold-load, and warm→hot elevate — so a resident record always matches the
    /// bytes it describes.
    ///
    /// Returns `None` for every input when the pool has no device residence
    /// (CPU / host-only tier): there is no readable record address, so the
    /// caller must keep `meta = None` and fall back to per-forward scratch heads.
    /// This preserves the invariant `meta.is_some() ⇒ device_addr != 0` that the
    /// prefill/glue serializer relies on (it builds no scratch heads for resident
    /// chunks). Each handle is stored on the `SealedChunk`/`ChunkWindow` and
    /// shared by every slot that references the chunk, and each holds a clone of the
    /// gids its pointer words name — see [`MetaGid`](super::meta_pool::MetaGid).
    ///
    /// A compaction does not come through here: it mints a record for an *existing*
    /// chunk whose slots it has already claimed, which is
    /// [`super::compact_mint::RecordMint`].
    #[allow(dead_code)] // callers are cuda-gated; a pure-CPU build sees none
    pub(crate) fn build_meta_records(
        &self,
        chunks: &[super::super::meta_pool::ChunkRecordSrc<'_>],
        arena_info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
    ) -> Result<Vec<Option<super::super::meta_pool::MetaGid>>> {
        if !self.inner.meta_pool.is_device_resident() {
            return Ok(vec![None; chunks.len()]);
        }
        let RecordLayout {
            key: record_key,
            record_bytes: rb,
            n_kv_head,
            head_dim,
            n_palette,
        } = self.record_layout()?;
        // Claim every slot first, then resolve all their addresses in **one** pass.
        // Resolving per record cost a `HashSet`, a `storage.read()` lock and a dense
        // `Vec<ResolvedArenaInfo>` each time, so a 4,096-record cold load paid 4,096 lock
        // round-trips and rebuilt the whole arena table 4,096 times to read one entry.
        let mut gids: Vec<ChunkGid> = Vec::with_capacity(chunks.len());
        for _ in chunks {
            // One arena slot per record, through the same allocator the bands use, so
            // the slot's refcount is what keeps the record alive and a clone of the
            // handle shares it.
            gids.push(self.alloc_chunk_for_key(record_key)?);
        }
        // Re-resolved rather than reusing the caller's `arena_info`: these slots did not
        // exist when that snapshot was taken, and the claims above can create an arena.
        let needed: HashSet<usize> = gids.iter().map(|g| g.arena_idx()).collect();
        let slot_info = self.resolve_arena_info_for(&needed)?;
        let mut handles: Vec<super::super::meta_pool::MetaGid> = Vec::with_capacity(chunks.len());
        // **Each handle takes a clone of the very gids its pointer words are about to be
        // filled from**, so the record cannot outlive the bands it names: the allocator
        // will not reissue a band slot while a record still holds a refcount on it. One
        // `Arc` bump per record and nothing per band — see [`MetaGid`].
        for (gid, src) in gids.into_iter().zip(chunks) {
            let addr = self.record_slot_addr(&gid, &slot_info, rb)?;
            handles.push(super::super::meta_pool::MetaGid::from_slot(
                gid,
                src.gids.clone(),
                addr,
            ));
        }
        // The records are written **on the device**, from a descriptor table — no host
        // serialization and no record bytes over the bus. See §8 of
        // `docs/vram_span_partition.md` and `fill_records_on_device`.
        self.fill_records_on_device(&handles, chunks, arena_info, n_kv_head, head_dim, n_palette)?;
        // **A filled record is write-once, and that is enforced by nothing having a
        // reason to write one** — not by `readonly_regions`, which was tried here and
        // does not fit. Its only release is `release_below(base)`, a whole-span clear at
        // the weight boundary; a record slot frees and is reissued constantly, so a
        // per-record declaration would go stale on the first free and then name the
        // slot's next legitimate tenant as a violator. That is the harness's own second
        // danger — a stale declaration blames an innocent allocation — and a guard that
        // cries wolf is worse here than no guard, because this is the instrument the
        // compaction work is validated with. What stands in its place is that the only
        // code that ever rewrote a record has been deleted, and the design note on
        // `compact_backings` says why nothing should write one again.
        Ok(handles.into_iter().map(Some).collect())
    }

    /// Write every record's bytes with one kernel launch.
    ///
    /// **Nothing is serialized on the host and no record byte crosses the bus.** The
    /// host ships a descriptor table plus the arrays it already holds; the kernel
    /// computes each band's absolute address from the arena extent table, derives the
    /// identity palette map arithmetically, and defaults an absent scale to unity — so
    /// the upload is strictly smaller than the records it produces, and for the common
    /// float chunk (identity map, unity scales) dramatically so.
    ///
    /// Asynchronous throughout: the uploads and the launch all go on the backing's
    /// stream and nothing is read back, so a batch of records costs no synchronisation.
    ///
    /// Held byte-for-byte against [`serialize_kv_heads`](super::meta_pool::serialize_kv_heads)
    /// by test — the layout is encoded in four places already
    /// (`docs/vram_span_partition.md` §8) and this must not become a fifth opinion.
    #[allow(unused_variables)]
    pub(super) fn fill_records_on_device(
        &self,
        handles: &[super::super::meta_pool::MetaGid],
        chunks: &[super::super::meta_pool::ChunkRecordSrc<'_>],
        arena_info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
        n_kv_head: usize,
        head_dim: usize,
        n_palette: usize,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            use candle::cuda_backend::kernels::simple::kv_record_fill as krf;
            use candle::cuda_backend::staged::segment;

            if handles.is_empty() {
                return Ok(());
            }
            // **Not a silent `Ok`.** A record slot may be recycled, and unlike
            // `alloc_chunk_with_arenas` the promoting claim does not zero what it hands
            // back — so skipping the fill leaves the *previous* tenant's record standing:
            // well-formed pointers into bands that have since been freed and reissued.
            // The caller only reaches here having found the pool device-resident, so a
            // non-CUDA device at this point is a broken assumption, not a configuration.
            let Device::Cuda(cuda) = &self.inner.device else {
                candle::bail!(
                    "fill_records_on_device: {} records were allocated on a non-CUDA \
                     device. Leaving them unwritten would publish the previous tenant's \
                     band pointers, which the attention kernels dereference unchecked.",
                    handles.len(),
                );
            };
            let pal_bytes = head_dim / 4;
            let bands = n_kv_head * n_palette * 2;
            let tags = n_kv_head * n_palette;

            // The record geometry's 8-byte alignment is checked by `build_meta_records`
            // before it claims a slot, and again inside `run_kv_record_fill` for the
            // callers that launch directly.

            // The descriptor and the packed inputs. Appending a chunk's existing slice is
            // a bulk copy, not a walk over bands — the per-band work is the kernel's.
            // Flat `i64` words rather than a slice of `KvRecordDesc`: `DeviceRepr` is
            // not implemented for it, every field is eight bytes, and
            // `krf::DESC_WORDS`' assertions hold the two forms bit-identical.
            let mut descs: Vec<i64> = Vec::with_capacity(handles.len() * krf::DESC_WORDS);
            let mut gids: Vec<i64> = Vec::with_capacity(handles.len() * bands);
            let (mut k_pal, mut v_pal) = (Vec::<u8>::new(), Vec::<u8>::new());
            let (mut k_fmt, mut v_fmt) = (Vec::<u8>::new(), Vec::<u8>::new());
            let (mut k_scale, mut v_scale) = (Vec::<f32>::new(), Vec::<f32>::new());
            for (h, src) in handles.iter().zip(chunks) {
                let gid_off = gids.len() as i64;
                gids.extend(src.gids.as_slice().iter().map(|g| g.raw()));
                // **Empty means "derive it"; SHORT is refused.**
                //
                // The descriptor carries one offset per record, so its fallback is
                // all-or-nothing, while `serialize_kv_heads` falls back per head and per
                // band independently (`k_pal.get(h * pal_bytes ..)`,
                // `k_fmt.get(tag_base + p)`, `k_scale.get(..).unwrap_or(1.0)`). For an
                // empty slice the two agree exactly — everything defaults. For a *short*
                // one they do not: the reference honours the populated prefix and defaults
                // the remainder, where an offset of -1 defaults the lot.
                //
                // The sharpest case is the format tags, and it is reachable: `compress`
                // sizes its fmt vectors `n_kv_head * N_PALETTE` (4), so against a backing
                // whose `n_palette()` is `LATENT_N_BANDS` (16) every tag would silently
                // become `Invalid` — a band with no known layout — where the reference
                // writes four real tags per head. Refused rather than reinterpreted,
                // because a wrong format tag decodes quantized bytes as floats.
                let short = |have: usize, need: usize| have != 0 && have < need;
                let pal_need = n_kv_head * pal_bytes;
                if short(src.k_pal.len(), pal_need) || short(src.v_pal.len(), pal_need) {
                    candle::bail!(
                        "palette maps are short: k {} v {} against {pal_need} needed for \
                         {n_kv_head} heads at head_dim {head_dim}. The device fill cannot \
                         express a per-head fallback, so this would default every head's \
                         map instead of only the absent ones.",
                        src.k_pal.len(),
                        src.v_pal.len(),
                    );
                }
                if short(src.k_fmt.len(), tags) || short(src.v_fmt.len(), tags) {
                    candle::bail!(
                        "band format tags are short: k {} v {} against {tags} needed for \
                         {n_kv_head} heads x {n_palette} bands. Defaulting the remainder \
                         would set every band to Invalid, so the record is refused.",
                        src.k_fmt.len(),
                        src.v_fmt.len(),
                    );
                }
                if short(src.k_scale.len(), tags) || short(src.v_scale.len(), tags) {
                    candle::bail!(
                        "outer scales are short: k {} v {} against {tags} needed. \
                         Defaulting the remainder would set every band's scale to unity.",
                        src.k_scale.len(),
                        src.v_scale.len(),
                    );
                }
                let pal_off = if src.k_pal.is_empty() || src.v_pal.is_empty() {
                    -1
                } else {
                    let off = k_pal.len() as i64;
                    k_pal.extend_from_slice(&src.k_pal[..pal_need]);
                    v_pal.extend_from_slice(&src.v_pal[..pal_need]);
                    off
                };
                let fmt_off = if src.k_fmt.is_empty() || src.v_fmt.is_empty() {
                    -1
                } else {
                    let off = k_fmt.len() as i64;
                    k_fmt.extend_from_slice(&src.k_fmt[..tags]);
                    v_fmt.extend_from_slice(&src.v_fmt[..tags]);
                    off
                };
                let scale_off = if src.k_scale.is_empty() || src.v_scale.is_empty() {
                    -1
                } else {
                    let off = k_scale.len() as i64;
                    k_scale.extend_from_slice(&src.k_scale[..tags]);
                    v_scale.extend_from_slice(&src.v_scale[..tags]);
                    off
                };
                // Through the struct, so the field order is the compiler's business and
                // the size assertions on both sides of the FFI cover these bytes.
                descs.extend_from_slice(
                    &krf::KvRecordDesc {
                        dst: h.device_addr(),
                        gid_off,
                        pal_off,
                        fmt_off,
                        scale_off,
                    }
                    .to_words(),
                );
            }

            // Extents indexed by arena index, so the kernel resolves a band the same way
            // the reference does: `base_ptr + chunk_idx * chunk_byte_stride`.
            let n_extents = arena_info.len();
            let mut extents: Vec<i64> = Vec::with_capacity(n_extents * krf::EXTENT_WORDS);
            for r in arena_info {
                extents.extend_from_slice(
                    &krf::KvArenaExtent {
                        base: r.base_ptr,
                        stride: r.chunk_byte_stride,
                    }
                    .to_words(),
                );
            }

            // All nine arrays go up as one copy into the device's staging scratch, so a
            // batch of records allocates nothing. A derived-only input (an empty array)
            // still gets a valid address inside the scratch; the kernel never reads it,
            // because every descriptor's offset is -1 — and `arena_info` is legitimately
            // empty when no arena is resolved, with every extent lookup bounded against
            // `n_extents`.
            let stream = cuda.cuda_stream();
            cuda.with_staged_segments(
                &[
                    segment(&descs),
                    segment(&gids),
                    &k_pal,
                    &v_pal,
                    &k_fmt,
                    &v_fmt,
                    segment(&k_scale),
                    segment(&v_scale),
                    segment(&extents),
                ],
                |p| {
                    candle::set_kernel_breadcrumb("run_kv_record_fill", file!(), line!());
                    // SAFETY: every array is device-resident and at least as long as the
                    // offsets the descriptors name; each `dst` is a record slot this call
                    // just allocated, of exactly `chunk_record_bytes` bytes. The scratch
                    // is next written by a later copy on this stream, ordered after the
                    // kernel's read.
                    unsafe {
                        krf::run_kv_record_fill(
                            p[0] as *const std::ffi::c_void,
                            p[1] as *const i64,
                            p[2] as *const u8,
                            p[3] as *const u8,
                            p[4] as *const u8,
                            p[5] as *const u8,
                            p[6] as *const f32,
                            p[7] as *const f32,
                            p[8] as *const std::ffi::c_void,
                            n_extents as i32,
                            handles.len() as i32,
                            n_kv_head as i32,
                            head_dim as i32,
                            n_palette as i32,
                            super::super::size_class::GID_STRIDE as i32,
                            ArenaFormatTag::Invalid.as_u8() as i32,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                    Ok(())
                },
            )?;
        }
        Ok(())
    }

    /// Absolute device address of the record slot `gid` names.
    ///
    /// **Stable for the handle's life**, which every paged kernel depends on: this
    /// value is written raw into each slice header's `kvheads_ptr` word and
    /// dereferenced by `reinterpret_cast` with no indirection. Resolved once, at
    /// allocation, and cached on the handle — nothing may relocate a held record.
    /// **The stride comes from the arena the slot is actually in, never from the key the
    /// caller asked for.** `alloc_chunk_for_key` may promote: when no region can be
    /// claimed, `stamp_region_promoting` walks the size-class ladder building *band*
    /// keys and allocates from one of those, returning the gid without saying which key
    /// it came from. Computing `base + chunk_idx * record_stride` against a band arena's
    /// base then lands far outside it — a 2 KiB record stride against a 320 B class's
    /// 52,428 slots reaches ~100 MiB past a 16 MiB arena, into another tenant's ground,
    /// mapped and silent. That is invariant 7 exactly, and the fill would write a record
    /// there for thirteen kernels to dereference.
    ///
    /// So the slot is bounds-checked against the arena's own capacity and addressed with
    /// the arena's own stride, and a slot whose stride cannot hold a record is refused
    /// rather than written. `Arena::slot_ptr` makes the same check for band slots and
    /// its doc gives the same reason.
    pub(super) fn record_slot_addr(
        &self,
        gid: &ChunkGid,
        info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
        record_bytes: usize,
    ) -> Result<u64> {
        let arena_idx = gid.arena_idx();
        let r = info.get(arena_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "record arena {arena_idx} vanished between claim and address resolution"
            ))
        })?;
        let slot = gid.chunk_idx();
        if r.base_ptr == 0 || r.chunk_byte_stride <= 0 {
            candle::bail!(
                "record slot {slot} landed in arena {arena_idx}, which is not device \
                 resident (base {:#x}, stride {}). A record's address is written raw into \
                 every slice header and dereferenced unchecked, so there is no safe \
                 address to return here.",
                r.base_ptr,
                r.chunk_byte_stride,
            );
        }
        if slot >= r.chunk_capacity as usize {
            candle::bail!(
                "record slot {slot} is past arena {arena_idx}'s capacity {} — addressing \
                 it would write into the next arena's bytes",
                r.chunk_capacity,
            );
        }
        if (r.chunk_byte_stride as usize) < record_bytes {
            candle::bail!(
                "record slot {slot} is in arena {arena_idx} whose stride is {} B, too \
                 small for a {record_bytes} B record. The allocator promoted this claim \
                 into a band arena (see `stamp_region_promoting`), and writing a record \
                 into a slot that cannot hold it overruns into the next slot.",
                r.chunk_byte_stride,
            );
        }
        Ok(r.base_ptr + (slot * r.chunk_byte_stride as usize) as u64)
    }
}
