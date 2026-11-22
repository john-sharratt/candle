//! Builds the per-head selection table the select kernels read, after checking
//! its gids.
//!
//! The kernel reads each band's base and stride with no bounds knowledge, so every
//! gid is validated against the current arenas, including its capacity, before
//! launch. The table is built under one storage read, and each band's row carries
//! its own arena's base and stride. An arena absent from storage resolves to a zero
//! row, which validation rejects before the launch.

use super::*;

impl BackingInner {
    /// Validate a selection batch's gids against the CURRENT storage arenas
    /// before any per-head table is uploaded and the selection kernel launched.
    ///
    /// The kernel addresses `row[gid / stride].base + (gid % stride) *
    /// row.chunk_byte_stride` with no bounds knowledge, so two host-state
    /// corruptions become silent OOB reads (or garbage-KV reads) on the GPU:
    ///   - a gid whose arena is ABSENT from storage (freed while referenced;
    ///     its table row is zeroed → near-null deref), and
    ///   - a gid whose `chunk_idx` exceeds the arena's per-FORMAT capacity
    ///     (`arena_chunks_for_format`): the arena index was re-tenanted with a
    ///     different-capacity format between gid mint and table build, so
    ///     `old_chunk_idx × new_stride` walks past the slab (the sanitizer-
    ///     confirmed read at exactly slab end).
    ///
    /// Failing here converts the corrupt launch into a named, recoverable
    /// error carrying both sides of the mismatch.
    #[cfg(feature = "cuda")]
    pub fn validate_selection_gids<B: Borrow<ChunkBands>>(
        &self,
        chunks: &[B],
        label: &str,
    ) -> Result<()> {
        let stride = GID_STRIDE;
        let n_kv_head = self.n_kv_head;
        self.storage.read(|s| -> Result<()> {
            for (ci, hg) in chunks
                .iter()
                .map(|c| &Borrow::<ChunkBands>::borrow(c).gids)
                .enumerate()
            {
                for h in 0..n_kv_head {
                    for p in 0..N_PALETTE {
                        for (side, raw) in [
                            ("K", hg.k_gid_pal(h, p).raw()),
                            ("V", hg.v_gid_pal(h, p).raw()),
                        ] {
                            if raw < 0 {
                                continue; // sentinel / absent palette slot
                            }
                            let arena_idx = raw as usize / stride;
                            let chunk_idx = raw as usize % stride;
                            let Some(arena) = s.arena(arena_idx) else {
                                candle::bail!(
                                    "selection gid validation ({label}): chunk {ci} head {h} \
                                     palette {p} {side}-gid {raw} → arena {arena_idx} ABSENT \
                                     from storage (chunk_idx {chunk_idx}) — arena freed under \
                                     a live gid"
                                );
                            };
                            // Capacity is a property of the arena's SIZE CLASS,
                            // and a region re-stamped to a different class has a
                            // different stride and a different slot count. This
                            // reads like format bookkeeping and is actually a
                            // memory-safety net: it caught a sanitizer-confirmed
                            // OOB read at slab end (audit A14).
                            let cap = arena.chunks();
                            if chunk_idx >= cap {
                                candle::bail!(
                                    "selection gid validation ({label}): chunk {ci} head {h} \
                                     palette {p} {side}-gid {raw} → arena {arena_idx} is class \
                                     {} B (capacity {cap}) but chunk_idx is {chunk_idx} — \
                                     pool/storage class mismatch: the gid was minted under a \
                                     different-capacity class (arena index re-tenanted under \
                                     a live gid)",
                                    arena.slot_stride()
                                );
                            }
                        }
                    }
                }
            }
            Ok(())
        })?
    }

    /// Host-side build of the selection table: one [`PerHeadEntry`] row per
    /// `(chunk, head)` of `chunks`, in that order, each `PerHeadEntry::COLS`
    /// wide. The kernel indexes it as `chunk_idx * n_kv_head + head_idx`.
    ///
    /// **Every palette sub-entry is populated from that band's own gid** — its
    /// arena's base pointer and slot stride, and the *chunk's* own format tag.
    /// This is what lets bands of one head live in different arenas and
    /// different formats: the address and the layout both travel per band,
    /// so nothing about the row depends on the bands sharing an arena.
    ///
    /// Sizing follows the job list rather than storage, which is also what
    /// makes the cross-layer persist selection cheap: layers concatenate with
    /// no arena-index rebasing, and the table grows with chunks actually being
    /// selected instead of with the arena count. An empty job list yields an empty table.
    ///
    /// The whole build happens under ONE storage read — never two. A split read
    /// (bound under one lock, pointers under another) lets the scheduler
    /// free or relocate an arena in between, so the rows no longer describe the
    /// arenas the caller's frozen gids address, and the select kernel
    /// dereferences a stale base pointer
    /// (`run_select_kv_format_palette4_paged` → `CUDA_ERROR_ILLEGAL_ADDRESS`).
    /// **The single-lock capture is now the whole of that guarantee.** This
    /// used to credit a migrate-in-flight guard with blocking arena
    /// free/relocate/trim across the build→launch→readback window;
    /// `migrate_flight` is an advisory counter with no mutual exclusion, so it
    /// blocks nothing. What actually holds is that every pointer here comes from
    /// a pinned gid — the gid keeps its chunk's arena alive for the window — and
    /// the reservation makes an arena base permanently valid for the process,
    /// so there is no relocation to race with in the first place.
    // Host-side table builder: the selection kernels consume it under `cuda`,
    // and `selection_table_tests` pins its row layout with or without one.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn per_head_table_host<B: Borrow<ChunkBands>>(&self, chunks: &[B]) -> Result<Vec<i64>> {
        use crate::kv_cache::arena_table::{PaletteSubEntry, N_PALETTE};

        let n_kv_head = self.n_kv_head;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        self.storage.read(|s| {
            let arenas = s.arenas();

            // Base pointer + slot stride per referenced arena, resolved once
            // up front. A chunk's eight bands usually hit one or two arenas,
            // but a whole job list hits each repeatedly — and resolving here
            // rather than inside the row loop is what lets a stride error
            // propagate instead of being swallowed into a zero.
            //
            // An arena that is absent from storage resolves to `(0, 0)`, the
            // same all-zero row the dense-over-storage build produced for a
            // hole. `validate_selection_gids` rejects that case before the
            // launch; this keeps the failure a named host error rather than a
            // near-null device deref if it ever slips through.
            let mut resolved: HashMap<usize, (u64, i64)> = HashMap::new();
            for chunk in chunks {
                for gid in Borrow::<ChunkBands>::borrow(chunk).gids.as_slice() {
                    let ai = gid.arena_idx();
                    if let std::collections::hash_map::Entry::Vacant(e) = resolved.entry(ai) {
                        let v = match arenas.get(&ai) {
                            // k_ptr == v_ptr: one combined K+V buffer per arena.
                            Some(a) => (a.base_ptr().unwrap_or(0), a.slot_stride() as i64),
                            None => (0, 0),
                        };
                        e.insert(v);
                    }
                }
            }
            let addr = |arena_idx: usize| -> (u64, i64) {
                resolved.get(&arena_idx).copied().unwrap_or((0, 0))
            };

            let mut data: Vec<i64> = vec![0i64; chunks.len() * n_kv_head * PerHeadEntry::COLS];
            for (ci, chunk) in chunks.iter().enumerate() {
                let chunk = Borrow::<ChunkBands>::borrow(chunk);
                for h in 0..n_kv_head {
                    let palette: [PaletteSubEntry; N_PALETTE] = std::array::from_fn(|p| {
                        let (k_base, k_stride) = addr(chunk.gids.k_gid_pal(h, p).arena_idx());
                        let (v_base, v_stride) = addr(chunk.gids.v_gid_pal(h, p).arena_idx());
                        let (k_tag, v_tag) = chunk.band_tags(h, p);
                        PaletteSubEntry {
                            k_ptr: k_base,
                            v_ptr: v_base,
                            // The band's chunk index is applied kernel-side from
                            // its own gid, so the row carries only the base and
                            // the step. Offsets stay zero: a chunk slot *is* one
                            // (head, palette, side) band.
                            k_byte_offset: 0,
                            v_byte_offset: 0,
                            k_chunk_byte_stride: k_stride,
                            v_chunk_byte_stride: v_stride,
                            k_format_tag: ArenaFormatTag::from_u8(k_tag),
                            v_format_tag: ArenaFormatTag::from_u8(v_tag),
                            k_outer_scale: 1.0,
                            v_outer_scale: 1.0,
                        }
                    });
                    let off = (ci * n_kv_head + h) * PerHeadEntry::COLS;
                    data[off..off + PerHeadEntry::COLS]
                        .copy_from_slice(&PerHeadEntry { palette }.to_tensor_row());
                }
            }

            Ok(data)
        })?
    }

    /// Build a lightweight per-arena info snapshot for constructing persistent
    /// GPU slot buffers without building a full GPU tensor.
    ///
    /// Returns one [`ResolvedArenaInfo`] per arena index (dense, 0..num_arenas).
    /// The `num_arenas` count is determined by scanning all active GIDs.
    pub fn resolve_arena_info(&self) -> Result<Vec<ResolvedArenaInfo>> {
        self.resolve_arena_info_filtered(None)
    }

    /// Like [`Self::resolve_arena_info`] but only builds entries for the arena
    /// indices in `needed` (others stay at the zero default). The provenance
    /// pointer-resolve touches only a couple of arenas per scope, so skipping
    /// `to_arena_entry` for every other arena is the bulk of that path's cost.
    /// `None` resolves all arenas (the original behaviour).
    pub fn resolve_arena_info_for(
        &self,
        needed: &std::collections::HashSet<usize>,
    ) -> Result<Vec<ResolvedArenaInfo>> {
        self.resolve_arena_info_filtered(Some(needed))
    }

    pub(super) fn resolve_arena_info_filtered(
        &self,
        needed: Option<&std::collections::HashSet<usize>>,
    ) -> Result<Vec<ResolvedArenaInfo>> {
        use crate::kv_cache::arena_table::ResolvedArenaInfo;

        self.storage.read(|s| {
            let arenas = s.arenas();
            if arenas.is_empty() {
                return Ok(Vec::new());
            }
            let num_arenas = arenas.keys().max().copied().unwrap_or(0) + 1;
            let mut info = vec![
                ResolvedArenaInfo {
                    base_ptr: 0,
                    chunk_byte_stride: 0,
                    chunk_capacity: 0,
                };
                num_arenas
            ];

            for (&arena_idx, arena) in arenas.iter() {
                if let Some(needed) = needed {
                    if !needed.contains(&arena_idx) {
                        continue;
                    }
                }
                // Address stride and capacity only. A band's payload length is
                // the chunk's business, not the arena's — see
                // `ResolvedArenaInfo`'s "what is deliberately absent".
                info[arena_idx] = ResolvedArenaInfo {
                    base_ptr: arena.base_ptr().unwrap_or(0),
                    chunk_byte_stride: arena.slot_stride() as i64,
                    chunk_capacity: arena.chunks() as u32,
                };
            }

            Ok(info)
        })?
    }
}
