//! Plans and issues the copies that move chunk slots to lower addresses.
//!
//! Each move claims its destination on the host, and every copy goes in one
//! kernel launch in which each record is a concurrent block. A destination that
//! is also a source, two writers of one slot, or a partial overlap would race
//! and leave a holder naming another sequence's bytes, so the plan is checked as
//! address ranges before the launch. Copy lengths come from the same arena
//! table as the addresses. A claim lost to a stale census is skipped, while a
//! slot past its arena's capacity is reported as an error.

use super::*;

impl super::super::backing::ChunkedKvBacking {
    /// Claim each move's destination slot and record the copy it implies.
    ///
    /// Host-only: no device work happens here. Every claim appends one record to
    /// `records`, which [`Self::copy_records`] then issues as a single launch.
    ///
    /// A claim that fails is skipped rather than fatal: the census is a snapshot
    /// (its occupancy bitmaps are sampled word by word), so a slot it believed free
    /// can have gone. Skipping costs reclaim and nothing else — the source is
    /// untouched and every holder still names it.
    #[cfg(feature = "cuda")]
    pub(super) fn claim_moves(
        &self,
        moves: &[ChunkMove],
        keys: &[ArenaKey],
        planned_sources: &AHashSet<(usize, u32)>,
        map: &mut CompactionMap,
        records: &mut CopyRecords,
    ) -> Result<()> {
        // Resolve only the arenas this batch touches, both ends.
        let needed: HashSet<usize> = moves.iter().flat_map(|m| [m.from.0, m.to.0]).collect();
        let info = self.resolve_arena_info_for(&needed)?;

        for (m, key) in moves.iter().zip(keys) {
            let Some(new_gid) = self.inner.pool.allocate_from_arena(*key, m.to.0) else {
                continue;
            };
            // The allocator is asked for a NAMED arena, never "wherever you
            // like": left to itself it hands back the leftmost arena with room,
            // and draining an arena is precisely what makes it the arena with the
            // most room — so the better a pass worked the more likely each next
            // chunk would land back where it came from.
            debug_assert_eq!(new_gid.arena_idx(), m.to.0);
            // **A destination may not be one of this pass's own sources.**
            //
            // The census is a snapshot, and this is the mirror of the case the
            // comment above records. A slot it saw OCCUPIED is planned as a source;
            // if its chunk is freed before the claims run, the slot is genuinely free
            // and the allocator hands it out — correctly — as some other move's
            // destination. Nothing downstream notices, because both halves are
            // individually legitimate.
            //
            // What it produces is a read/write race *inside one launch*. Every record
            // is a concurrent block, so the record copying INTO the slot runs against
            // the record copying OUT of it, and the second one's chunk arrives as a
            // mixture or as the first one's data entire. Its holder is then rewritten
            // to name ground holding another sequence's K/V — finite, plausibly
            // shaped, and wrong, which is why it surfaces as a NaN in the first
            // attention layer rather than as a fault.
            //
            // The byte check after the copy cannot see it: by the time it compares,
            // the source has been overwritten with the same bytes the destination
            // holds, so the two agree.
            //
            // Skipped rather than repaired, because there is nothing to repair. The
            // claim is dropped with `new_gid` at the `continue`, which frees the slot
            // again, and the move is simply left for the next pass — by which time
            // the census will see the slot as free and never plan it as a source.
            if planned_sources.contains(&(new_gid.arena_idx(), new_gid.chunk_idx() as u32)) {
                note(11, 1);
                records.source_collisions += 1;
                continue;
            }
            // **A slot index past its arena's capacity is an invariant breach, and
            // must be told apart from the ordinary skips below.**
            //
            // `slot_addr` returns `None` for three different things: no such arena,
            // a CPU arena with no device slab, and an index past the arena's end.
            // The first two are ordinary — the census is a snapshot — and skipping
            // them costs reclaim and nothing else. The third is not ordinary: it
            // means the plan named ground this arena does not own, and the address
            // it would have produced lands in the next region, whose tenant may be
            // another arena, the wave transient tier, or a sequence's recurrent
            // store. Sharing one silent `continue` with the benign cases would
            // hide it completely, which is how it stayed hidden.
            for (which, arena_idx, chunk_idx) in [
                ("source", m.from.0, m.from.1 as usize),
                ("destination", new_gid.arena_idx(), new_gid.chunk_idx()),
            ] {
                if let Some(a) = info.get(arena_idx) {
                    if chunk_idx >= a.chunk_capacity as usize {
                        candle::bail!(
                            "compaction planned a {which} slot {chunk_idx} in arena \
                             {arena_idx}, which holds only {} slots. The address that \
                             implies is {} B past the arena's end, inside whatever \
                             tenant holds the next region — and every address in the \
                             reservation is mapped, so writing it would not fault.",
                            a.chunk_capacity,
                            (chunk_idx + 1 - a.chunk_capacity as usize)
                                * a.chunk_byte_stride as usize,
                        )
                    }
                }
            }
            let (Some((src, src_stride)), Some((dst, dst_stride))) = (
                slot_addr(&info, m.from.0, m.from.1 as usize),
                slot_addr(&info, new_gid.arena_idx(), new_gid.chunk_idx()),
            ) else {
                continue;
            };
            // **The length comes from the arenas, and both ends must agree.**
            //
            // It used to be `key.class.bytes()` — the size class's stride — while
            // the two addresses above stepped by the *arena's* stride. Those are
            // equal by convention rather than by construction, and
            // `ResolvedArenaInfo::chunk_byte_stride` says in as many words that
            // the class figure "is generally larger than the payload" and must
            // never be used as a copy length.
            //
            // A length longer than the destination slot does not fault. It writes
            // into the slot above, and from the arena's last slot it writes into
            // the next region — whose tenant may be another arena, the wave
            // transient tier, or a sequence's recurrent store, none of which will
            // notice until something reads a wrong number out of it. Taking the
            // length from the same table that produced the addresses removes the
            // second source of truth, and comparing the two ends catches a plan
            // that paired slots of unequal classes, which no in-bounds check on
            // either end alone would see.
            if src_stride != dst_stride {
                candle::bail!(
                    "compaction planned a copy between slots of unequal extent: \
                     arena {} slot {} holds {src_stride} B, arena {} slot {} holds \
                     {dst_stride} B. A pool is one size class, so this is a planner \
                     or arena-table fault, and copying either length would write \
                     outside one of the two slots.",
                    m.from.0,
                    m.from.1,
                    new_gid.arena_idx(),
                    new_gid.chunk_idx(),
                )
            }
            records.srcs.push(src as i64);
            records.dsts.push(dst as i64);
            records.lens.push(dst_stride);
            let old_raw = (m.from.0 * super::super::types::GID_STRIDE + m.from.1 as usize) as i64;
            // A band is followed through a holder's gids, a record through its `meta` —
            // the same copy, published to the field that names it.
            match key.kind {
                ArenaKind::Band => map.insert(old_raw, new_gid, dst),
                ArenaKind::Record { .. } => map.insert_record(old_raw, new_gid, dst),
            }
        }
        Ok(())
    }

    /// Copy every claimed slot in one launch of the migration scatter/gather
    /// kernel.
    ///
    /// One CUDA block per record, a 16-byte vectorised body when the record's ends
    /// and length are all aligned — which for a size-class slot they are, the ladder
    /// being powers of two from 320 B up. Three `i64` arrays cross the bus and
    /// nothing else; the kernel resolves no tables and reads no metadata, because
    /// the addresses were computed by the claim walk that produced the records.
    #[cfg(feature = "cuda")]
    pub(super) fn copy_records(&self, records: &CopyRecords) -> Result<()> {
        if records.srcs.is_empty() {
            return Ok(());
        }
        // Before the launch, because after it the evidence is gone: the racing pair
        // leaves the source holding the destination's bytes, so every after-the-fact
        // comparison agrees.
        disjoint_ends(records)?;
        let candle::Device::Cuda(cuda) = self.device() else {
            return Ok(());
        };
        use candle::cuda_backend::kernels;
        use candle::cuda_backend::staged::segment;

        let stream = cuda.cuda_stream();
        // Every destination this launch will write, against ground declared
        // immutable after load. The kernel is plan-driven and has no bound of its
        // own, and the weight zone shares the reservation with the arenas — so a
        // destination computed from the wrong base writes expert weights and
        // raises nothing. Host-side, so the offender is named rather than the
        // victim discovered downstream.
        #[cfg(feature = "tensor-assert")]
        for (d, l) in records.dsts.iter().zip(records.lens.iter()) {
            candle::readonly_regions::forbid_write(
                "kv compaction destination",
                *d as u64,
                (*l).max(0) as usize,
            );
        }
        // The three arrays go up in one copy into the device's held staging
        // scratch, read by this launch and reused behind it on the same stream.
        cuda.with_staged_segments(
            &[
                segment(&records.srcs),
                segment(&records.dsts),
                segment(&records.lens),
            ],
            |at| {
                // SAFETY: each record names one slot of the reservation, `lens[r]`
                // bytes long, in arenas of the same class and the same memory (the
                // plan is per `ArenaKey`, so location cannot differ). Each
                // destination is held by this pass's own claim; each source is
                // read-only here and stays live until the holders that name it are
                // rewritten below. `at` names the three uploaded arrays.
                unsafe {
                    candle::set_kernel_breadcrumb("run_kv_migrate_copy", file!(), line!());
                    kernels::simple::kv_migrate::run_kv_migrate_copy(
                        at[0] as *const i64,
                        at[1] as *const i64,
                        at[2] as *const i64,
                        std::ptr::null::<i64>(),
                        std::ptr::null::<i64>(),
                        std::ptr::null::<i64>(),
                        records.srcs.len() as i32,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
                Ok(())
            },
        )
    }

    /// One resident `KvHead` record, as the device holds it.
    ///
    /// The whole record in one transfer rather than a word per band: the integrity
    /// check reads every band pointer of every chunk, and a chunk's record is a couple
    /// of kilobytes against `n_kv_head * n_palette * 2` separate synchronous reads.
    #[cfg(all(feature = "cuda", feature = "tensor-assert"))]
    pub(in super::super) fn read_record(&self, addr: u64, bytes: usize) -> Option<Vec<u8>> {
        let candle::Device::Cuda(cuda) = self.device() else {
            return None;
        };
        cuda.bind_to_thread().ok()?;
        let mut out = vec![0u8; bytes];
        // SAFETY: `addr` is a resident record's base address, taken from the
        // chunk's own `MetaGid`, and `bytes` is that record's serialized length
        // computed from the same geometry the serializer used.
        unsafe { cuda.read_raw(addr, &mut out) }.ok()?;
        Some(out)
    }
}

/// Refuse a copy plan in which any slot is both read and written.
///
/// **The kernel is one concurrent block per record, so this is the whole safety
/// condition.** A record writing a slot another record reads races it, and the loser
/// copies a mixture or the winner's chunk entire — then has its holder rewritten to
/// name that ground. Nothing faults and the bytes are plausible; it arrives as a NaN
/// in the first attention layer many launches later.
///
/// `plan_pool` guarantees disjointness and a test pins it, but the guarantee does not
/// survive the claim: `claim_moves` asks the allocator for *any* free slot in the
/// destination arena rather than the one the plan named, and a slot the census saw
/// occupied — and therefore planned as a source — is genuinely free by then if its
/// chunk was released in between. Both halves are legitimate, which is why this was
/// invisible.
///
/// Checked on address *ranges*, not on `(arena, slot)` or on bare addresses. Ranges
/// are what the kernel dereferences, so a length is part of the collision: two slots
/// of different classes can intersect without sharing a start address, and a wrong
/// stride makes that the normal case rather than the exotic one.
///
/// Every written range must be disjoint from every other range in the plan. Two
/// *reads* of the same bytes are harmless — both records copy the same content out —
/// so only a write makes a pair a collision. That covers three distinct faults with
/// one comparison:
///
/// - a destination that is also a source (the census-snapshot case above),
/// - two records writing one slot, where the loser's holder ends up naming the
///   winner's chunk,
/// - a destination overlapping a *different* slot part-way, which is what a wrong
///   length or stride produces.
pub(super) fn disjoint_ends(records: &CopyRecords) -> Result<()> {
    // (start, end, record, is_write)
    let mut ranges: Vec<(i64, i64, usize, bool)> =
        Vec::with_capacity(records.srcs.len() + records.dsts.len());
    for i in 0..records.srcs.len() {
        let len = records.lens[i].max(0);
        if len == 0 {
            continue;
        }
        ranges.push((records.srcs[i], records.srcs[i] + len, i, false));
        ranges.push((records.dsts[i], records.dsts[i] + len, i, true));
    }
    ranges.sort_unstable_by_key(|&(start, _, _, _)| start);
    for pair in ranges.windows(2) {
        let (a_start, a_end, a_rec, a_write) = pair[0];
        let (b_start, _b_end, b_rec, b_write) = pair[1];
        // Sorted by start, so the only way two ranges meet is the earlier one
        // reaching into the later one.
        if b_start >= a_end {
            continue;
        }
        // Two reads of the same bytes are fine: both records copy the same content.
        if !a_write && !b_write {
            continue;
        }
        let what = match (a_write, b_write) {
            (true, true) => {
                "two records WRITE overlapping ranges, so one chunk lands \
                             on top of the other and the loser's holder is rewritten \
                             to name it"
            }
            _ => {
                "one record READS a range another WRITES, so they race and whichever \
                  loses copies the other's chunk"
            }
        };
        candle::bail!(
            "compaction planned overlapping ranges: record {a_rec} \
             [{a_start:#x}, {a_end:#x}) and record {b_rec} at {b_start:#x} — {what}. \
             The migrate kernel runs one concurrent block per record, and every \
             address involved is mapped, so nothing faults and the bytes decode to \
             plausible, wrong values."
        )
    }
    Ok(())
}

/// Device address of one chunk slot, and the slot's byte extent, from a resolved
/// arena table.
///
/// The same two lines `serialize_kv_heads` uses to fill a record's band pointers,
/// named once here so a compaction's copy and the record describing it cannot
/// disagree about where a slot is. The stride comes from the resolved arena rather
/// than from the size class: they agree, and taking it from the arena is what makes
/// a disagreement impossible rather than merely unlikely.
///
/// **`chunk_idx` is bounded against the arena's capacity, not against the stride.**
/// The raw-gid namespace is one fixed power of two (`GID_STRIDE`) while an arena
/// holds only `chunks_per_region` slots for its class, so an index can be perfectly
/// valid as a gid and still name ground past this arena's end —
/// [`ResolvedArenaInfo::chunk_capacity`] exists to say exactly that, and says
/// validators must compare against it. Unbounded, the address this returns walks
/// into the next region, which belongs to another tenant: the next arena, the wave
/// transient tier, or a sequence's recurrent store. Nothing faults, because every
/// address in the reservation is mapped.
///
/// Returning the extent alongside the address is what lets a caller copy without
/// inventing a length from a second source. That mattered: the copy planner took
/// its length from the size class while its addresses stepped by the arena's
/// stride, and the two are only equal by convention — see
/// [`ChunkedKvBacking::claim_moves`].
pub(in super::super) fn slot_addr(
    info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
    arena_idx: usize,
    chunk_idx: usize,
) -> Option<(u64, i64)> {
    let a = info.get(arena_idx)?;
    if a.base_ptr == 0 {
        return None;
    }
    if chunk_idx >= a.chunk_capacity as usize {
        return None;
    }
    Some((
        a.base_ptr + chunk_idx as u64 * a.chunk_byte_stride as u64,
        a.chunk_byte_stride,
    ))
}
