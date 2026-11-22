//! Reads and writes one chunk's payload bytes, and moves a chunk between tiers
//! with its bytes unchanged.
//!
//! A slot's payload is the band's own length, and its address step is the arena's
//! class stride. The two are kept apart, because using one for the other addresses
//! the wrong slot. A tier move copies the whole slot verbatim and refuses a change
//! of class, since a different stride would truncate the band or spill it into a
//! neighbour. A source missing from storage fails loudly, rather than producing an
//! untracked alias of a freed chunk.

use super::*;

/// Read one chunk-slot's payload bytes from a CPU arena into `dst`. The
/// reverse of [`ChunkedKvBacking::write_chunk_from_pinned_bytes`].
///
/// `dst.len()` is the band's **payload** — how many bytes of the slot are the
/// chunk — while the slot's address step is the arena's class stride. They are
/// different numbers, so the offset must never be derived from `dst.len()`:
/// doing so addresses slot `n` at `n * payload` and walks into a neighbour.
pub(super) fn read_chunk_into_pinned_bytes(
    arena: &Arena,
    chunk_idx: usize,
    dst: &mut [u8],
) -> Result<()> {
    arena.copy_slot_bytes(chunk_idx, dst)
}

impl ChunkedKvBacking {
    /// Read the raw bytes of one chunk slot into `dst`. Works for both
    /// `Arena::Float` (any supported dtype) and `Arena::Quantized`
    /// (CPU or GPU).
    ///
    /// `dst` must be exactly the chunk's **payload**
    /// ([`ResolvedArenaInfo::chunk_payload_bytes`]); `slot_stride` is the
    /// arena's **address step** ([`ResolvedArenaInfo::chunk_byte_stride`]).
    /// Passing the same value for both is correct only while arenas are
    /// per-format.
    ///
    /// [`ResolvedArenaInfo::chunk_payload_bytes`]: crate::kv_cache::arena_table::ResolvedArenaInfo::chunk_payload_bytes
    /// [`ResolvedArenaInfo::chunk_byte_stride`]: crate::kv_cache::arena_table::ResolvedArenaInfo::chunk_byte_stride
    pub fn read_chunk_into_bytes(
        &self,
        arena_idx: usize,
        chunk_idx: usize,
        dst: &mut [u8],
    ) -> candle::Result<()> {
        self.inner.storage.read(|s| {
            let arena = s.arenas().get(&arena_idx).ok_or_else(|| {
                candle::Error::Msg(format!("read_chunk_into_bytes: arena {arena_idx} missing"))
            })?;
            read_chunk_into_pinned_bytes(arena, chunk_idx, dst)
        })?
    }

    /// Relocate one chunk slot to a fresh slot of the **same size class**.
    ///
    /// This is the tier move — hot↔warm — and nothing more. It copies the
    /// slot's bytes verbatim; it does not, and cannot, change a chunk's
    /// format. An arena is a run of fixed-stride byte slots that says nothing
    /// about what occupies them, so
    /// a per-chunk *conversion* has no source format to read here. Format
    /// changes happen where the formats are known: the quantize / dequantize
    /// passes in `compress.rs`, which work from the chunk's own band tags.
    ///
    /// Requiring the classes to match is what makes the verbatim copy safe:
    /// equal strides mean the destination slot is exactly as wide as the
    /// source, so no tenant's bytes can be truncated or spill into a
    /// neighbour.
    ///
    /// # Arguments
    /// * `source_gid` - Raw id of the source chunk slot
    /// * `target_key` - Class (must equal the source's) + destination location
    ///
    /// # Returns
    /// The `ChunkGid` of the newly allocated slot.
    pub fn migrate_chunk(&self, source_gid: i64, target_key: ArenaKey) -> Result<ChunkGid> {
        if source_gid < 0 {
            candle::bail!("migrate_chunk: invalid source chunk ID {}", source_gid);
        }
        let source_gid = source_gid as usize;
        let source_arena_idx = source_gid / GID_STRIDE;
        let source_chunk_idx = source_gid % GID_STRIDE;

        // Allocate destination GID through the pool — this is the single allocation path.
        let new_gid = {
            let _state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            self.alloc_chunk_for_key(target_key)?
        };

        let arena_idx = new_gid.arena_idx();
        let chunk_idx = new_gid.chunk_idx();
        let copy_result = self.inner.storage.try_write(|s| {
            // Re-validate the source by MAP LOOKUP, not an index-vs-len bound:
            // storage `arenas` is a keyed map with holes once any arena has been
            // tombstoned, so `arena_count()` (the map LEN) undercounts and a
            // perfectly valid high-index source would be misjudged "tombstoned".
            // That misjudgement made this path skip the copy and hand back a
            // detached pseudo-gid aliasing the live source (see below) — the
            // stale-gid corruption at the root of the CFW crash/drift.
            let Some(source_key) = s.arena_key(source_arena_idx) else {
                return Ok(false);
            };
            if source_key.class != target_key.class {
                candle::bail!(
                    "migrate_chunk: cannot relocate a {} B slot into a {} B slot —                      a slot move is byte-verbatim and never converts formats",
                    source_key.slot_stride(),
                    target_key.slot_stride(),
                );
            }
            Self::copy_slot_bytes(
                s.arenas_mut(),
                source_arena_idx,
                source_chunk_idx,
                arena_idx,
                chunk_idx,
            )?;
            Ok(true)
        });

        match copy_result {
            Ok(true) => Ok(new_gid),
            Ok(false) => {
                // The source arena is genuinely absent from storage. The caller
                // holds the source gid's Arc across this call, so a live source
                // cannot be tombstoned under us (`try_tombstone` requires
                // `live == 0`) — absence here means an arena was freed while
                // chunks still referenced it: the exact invariant violation
                // behind the stale-gid KV corruption. Fail loudly. The old
                // fallback returned `ChunkGid::detached(source_gid)` — an
                // UNTRACKED alias of the freed id that let its arena index be
                // re-tenanted under a live chunk window (illegal-address when
                // strides differ, silent cross-context KV reads when not).
                // new_gid drops here, returning the dest GID to the pool.
                Err(candle::Error::Msg(format!(
                    "migrate_chunk: source arena {source_arena_idx} (gid {source_gid}) vanished                      while its chunk is still referenced — arena freed with live KV"
                )))
            }
            Err(e) => {
                // Data copy failed — new_gid drops here, returning GID to pool.
                Err(e)
            }
        }
    }

    /// Copy one whole slot's bytes between arenas of the same class, crossing
    /// devices if their locations differ.
    ///
    /// The **full stride** moves, not just some payload: this is the one place
    /// with no chunk in hand to name a format, and the extra bytes are the
    /// slot's zeroed pad (invariant 4). Copying the whole slot also keeps the
    /// destination's pad clean without a second write.
    pub(in super::super) fn copy_slot_bytes(
        arenas: &mut ahash::AHashMap<usize, Arena>,
        src_arena: usize,
        src_chunk: usize,
        dst_arena: usize,
        dst_chunk: usize,
    ) -> Result<()> {
        // Read the source slot out first: it releases the immutable borrow
        // before the destination's mutable one, and it is also what carries the
        // bytes across a device boundary when the tiers differ.
        let (bytes, src_stride) = {
            let src = arenas.get(&src_arena).ok_or_else(|| {
                candle::Error::Msg(format!("copy_slot_bytes: src arena {src_arena} not found"))
            })?;
            let stride = src.slot_stride();
            (src.slot_bytes(src_chunk, stride)?.copy()?, stride)
        };
        let dst = arenas.get_mut(&dst_arena).ok_or_else(|| {
            candle::Error::Msg(format!("copy_slot_bytes: dst arena {dst_arena} not found"))
        })?;
        if dst.slot_stride() != src_stride {
            candle::bail!(
                "copy_slot_bytes: {src_stride} B source slot into a {} B destination slot",
                dst.slot_stride()
            );
        }
        let bytes = if bytes.device().location() != dst.byte_data().device().location() {
            bytes.to_device(dst.byte_data().device())?
        } else {
            bytes
        };
        dst.write_slot_bytes(dst_chunk, &bytes)
    }

    /// Write one chunk-slot's worth of raw bytes into a CPU `Arena::Float`
    /// at `(arena_idx, chunk_idx)`, copied straight into the slab
    /// ([`Arena::write_slot_from_bytes`]).
    ///
    /// `Arena::Quantized` is not supported here — see the module-level
    /// caveat on [`Self::migrate_sealed_to_cpu_batch_async`].
    ///
    /// `bytes` is the chunk's **payload**; `slot_stride` is the arena's
    /// **address step** (see [`Self::read_chunk_into_bytes`] for why they must
    /// be passed separately).
    ///
    /// Gated behind the cuda feature to match the only caller
    /// (`migrate_sealed_to_cpu_batch_async`).
    #[cfg(feature = "cuda")]
    pub(super) fn write_chunk_from_pinned_bytes(
        arenas: &mut ahash::AHashMap<usize, Arena>,
        arena_idx: usize,
        chunk_idx: usize,
        bytes: &[u8],
    ) -> candle::Result<()> {
        let arena = arenas.get_mut(&arena_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "write_chunk_from_pinned_bytes: arena {arena_idx} missing"
            ))
        })?;
        // The gather pulled exactly one PAYLOAD per slot into pinned scratch,
        // and slot `n` starts at `n * slot_stride` — `write_slot_from_bytes`
        // keeps those two lengths apart (invariant 8).
        arena.write_slot_from_bytes(chunk_idx, bytes)
    }
}
