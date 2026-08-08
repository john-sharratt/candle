//! KV tier-migration primitive — the Rust side of the `kv_pack` /
//! `kv_unpack` scatter/gather kernel (`docs/kv_tier_migration.md` §9).
//!
//! A migration plan is a flat list of `(src, dst, byte_len)` records, each
//! carrying device addresses the caller has already resolved. [`kv_migrate`]
//! executes the whole plan in a single kernel launch — it serves both
//! directions:
//!
//! - **kv_pack** (evict / gather) — `src` are scattered arena chunks,
//!   `dst` offsets into a contiguous staging buffer.
//! - **kv_unpack** (load / scatter) — `src` is the contiguous staging
//!   buffer, `dst` are freshly-allocated arena chunks.
//!
//! The kernel itself is `candle-kernels` `simple/kv_migrate.cu`.

#[cfg(feature = "cuda")]
use super::chunk_ops::BlockAllocSpec;
#[cfg(feature = "cuda")]
use crate::kv_cache::arena_table::N_PALETTE;
#[cfg(feature = "cuda")]
use crate::kv_cache::KvFormat;
#[cfg(feature = "cuda")]
use candle::cuda::cudarc::driver::CudaStream;

/// Bytes one band occupies, from the band's own format tag.
///
/// The migration plan's copy lengths come from here rather than from the
/// arena: a size-class arena holds whatever fits its slots, so its stride is an
/// upper bound on a band's bytes, never the bytes themselves
/// (`docs/archived/arena_unification.md` invariant 8). A tag that names no storage
/// format has no length, and a plan built on a guessed length would move the
/// wrong bytes — so it is an error, not a default.
#[cfg(feature = "cuda")]
fn band_payload(
    tag: crate::kv_cache::ArenaFormatTag,
    elems_per_chunk: usize,
) -> candle::Result<i64> {
    super::size_class::payload_bytes_for_tag(tag, elems_per_chunk)
        .map(|b| b as i64)
        .ok_or_else(|| {
            candle::Error::Msg(format!(
                "migration plan: band tag {tag:?} names no storage format, so its byte \
                 length is unknown"
            ))
        })
}

/// One copy in a migration plan: `byte_len` bytes from `src_ptr` to
/// `dst_ptr`, both raw device addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MigrationRecord {
    pub src_ptr: i64,
    pub dst_ptr: i64,
    pub byte_len: i64,
}

/// A migration plan — the flat record list spanning every sub-chunk of a
/// migration batch. One [`kv_migrate`] launch covers the whole plan.
#[derive(Clone, Debug, Default)]
pub struct MigrationPlan {
    pub records: Vec<MigrationRecord>,
}

impl MigrationPlan {
    /// An empty plan.
    pub fn new() -> MigrationPlan {
        MigrationPlan {
            records: Vec::new(),
        }
    }

    /// Append one copy record.
    pub fn push(&mut self, src_ptr: i64, dst_ptr: i64, byte_len: i64) {
        self.records.push(MigrationRecord {
            src_ptr,
            dst_ptr,
            byte_len,
        });
    }

    /// Number of copy records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the plan has no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Total bytes the plan moves.
    pub fn total_bytes(&self) -> i64 {
        self.records.iter().map(|r| r.byte_len).sum()
    }
}

/// Execute a migration plan on the device's default stream and
/// synchronise before returning. Equivalent to
/// `kv_migrate_on(device, plan, None)`.
#[cfg(feature = "cuda")]
pub fn kv_migrate(device: &candle::Device, plan: &MigrationPlan) -> candle::Result<()> {
    kv_migrate_on(device, plan, None)
}

/// Execute a migration plan on a caller-chosen stream. After this
/// returns, the stream has been synchronised — every record's bytes
/// have been copied and visible to subsequent work on that stream.
///
/// `stream = None` uses the device's default stream (same as
/// [`kv_migrate`]). Passing a dedicated copy stream lets the caller
/// overlap migration DMAs with compute submitted to other streams —
/// the persistence thread uses this to keep its hot→warm gather off
/// the default decode stream.
#[cfg(feature = "cuda")]
pub fn kv_migrate_on(
    device: &candle::Device,
    plan: &MigrationPlan,
    stream: Option<&CudaStream>,
) -> candle::Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::kernels;

    let dev = match device {
        candle::Device::Cuda(d) => d,
        _ => {
            return Err(candle::Error::Msg(
                "kv_migrate requires a CUDA device".into(),
            ))
        }
    };
    if plan.records.is_empty() {
        return Ok(());
    }

    let src: Vec<i64> = plan.records.iter().map(|r| r.src_ptr).collect();
    let dst: Vec<i64> = plan.records.iter().map(|r| r.dst_ptr).collect();
    let lens: Vec<i64> = plan.records.iter().map(|r| r.byte_len).collect();

    let src_gpu = dev
        .memcpy_stod(&src)
        .map_err(|e| candle::Error::Msg(format!("kv_migrate: src plan HtoD: {e}")))?;
    let dst_gpu = dev
        .memcpy_stod(&dst)
        .map_err(|e| candle::Error::Msg(format!("kv_migrate: dst plan HtoD: {e}")))?;
    let len_gpu = dev
        .memcpy_stod(&lens)
        .map_err(|e| candle::Error::Msg(format!("kv_migrate: len plan HtoD: {e}")))?;

    let default_stream = dev.cuda_stream();
    let used_stream = stream.unwrap_or(&default_stream);
    // Cross-stream ordering fence. The three plan uploads above — and the
    // record sources themselves when they are freshly written (e.g. Q arenas
    // filled by the primary-stream convert moments earlier) — are allocated
    // and copied in PRIMARY-stream order (`memcpy_stod` + the async pool's
    // stream-ordered validity). When the caller passes a dedicated copy
    // stream, the kernel below consumes them on THAT stream with no recorded
    // dependency: under load the primary stream queues seconds deep, so the
    // copy stream reaches the plan/source bytes long before their allocation
    // and fill retire — CUDA_ERROR_ILLEGAL_ADDRESS inside
    // `run_kv_migrate_copy` (the bulk-ingest decode-onset crash). Drain the
    // device once before the cross-stream launch; the default-stream path
    // needs no fence (same-stream FIFO).
    if stream.is_some() {
        device
            .synchronize()
            .map_err(|e| candle::Error::Msg(format!("kv_migrate: pre-launch fence: {e}")))?;
    }
    let (sp, _sg) = src_gpu.device_ptr(used_stream);
    let (dp, _dg) = dst_gpu.device_ptr(used_stream);
    let (lp, _lg) = len_gpu.device_ptr(used_stream);
    unsafe {
        candle::set_kernel_breadcrumb("run_kv_migrate_copy", file!(), line!());
        kernels::simple::kv_migrate::run_kv_migrate_copy(
            sp as *const i64,
            dp as *const i64,
            lp as *const i64,
            plan.records.len() as i32,
            used_stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    used_stream
        .synchronize()
        .map_err(|e| candle::Error::Msg(format!("kv_migrate: stream sync: {e}")))?;
    Ok(())
}

/// Host-side migration plan-builder (`docs/kv_tier_migration.md` §8).
#[cfg(feature = "cuda")]
impl super::ChunkedKvBacking {
    /// Resolve every unique physical sub-chunk of a sealed sequence to its
    /// device address and byte length.
    ///
    /// Returns `(device_ptr, byte_len)` per unique `(arena, chunk)` pair, in
    /// chunk order — the input from which `transfer.rs` builds the gather /
    /// scatter [`MigrationPlan`]s. Errors if a chunk's arena is not
    /// GPU-resident.
    pub fn resolve_sealed_chunk_ptrs(
        &self,
        chunks: &[super::SealedChunk],
    ) -> candle::Result<Vec<(i64, i64)>> {
        let arena_info = self.resolve_arena_info()?;
        let elems = self.inner.elems_per_chunk();
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for chunk in chunks {
            for (gid, tag) in chunk.bands() {
                let arena_idx = gid.arena_idx();
                let chunk_idx = gid.chunk_idx();
                if !seen.insert((arena_idx, chunk_idx)) {
                    continue;
                }
                let arena = arena_info.get(arena_idx).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "resolve_sealed_chunk_ptrs: arena index {arena_idx} out of range"
                    ))
                })?;
                if arena.base_ptr == 0 {
                    return Err(candle::Error::Msg(
                        "resolve_sealed_chunk_ptrs: chunk arena is not GPU-resident".into(),
                    ));
                }
                let ptr = arena.base_ptr as i64 + chunk_idx as i64 * arena.chunk_byte_stride;
                out.push((ptr, band_payload(tag, elems)?));
            }
        }
        Ok(out)
    }

    /// Resolve EVERY gid of each sealed chunk to its device `(ptr, byte_len)`,
    /// in gid order and WITHOUT deduping shared physical slots — one inner Vec
    /// per chunk. Capture needs this rather than the deduped
    /// [`Self::resolve_sealed_chunk_ptrs`]: [`Self::load_sealed_from_host`]
    /// bulk-allocates one fresh, distinct chunk per sub-band gid, so a faithful
    /// capture must carry one slot's bytes per gid — replicating any aliased
    /// physical slot — for the replay scatter to line up byte-for-byte.
    fn resolve_sealed_chunk_ptrs_per_gid(
        &self,
        chunks: &[super::SealedChunk],
    ) -> candle::Result<Vec<Vec<(i64, i64)>>> {
        let arena_info = self.resolve_arena_info()?;
        let elems = self.inner.elems_per_chunk();
        let mut out = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let mut per = Vec::with_capacity(chunk.gids.0.len());
            for (gid, tag) in chunk.bands() {
                let arena_idx = gid.arena_idx();
                let chunk_idx = gid.chunk_idx();
                let arena = arena_info.get(arena_idx).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "resolve_sealed_chunk_ptrs_per_gid: arena index {arena_idx} out of range"
                    ))
                })?;
                if arena.base_ptr == 0 {
                    return Err(candle::Error::Msg(
                        "resolve_sealed_chunk_ptrs_per_gid: chunk arena is not GPU-resident".into(),
                    ));
                }
                let ptr = arena.base_ptr as i64 + chunk_idx as i64 * arena.chunk_byte_stride;
                per.push((ptr, band_payload(tag, elems)?));
            }
            out.push(per);
        }
        Ok(out)
    }

    /// Gather a sealed sequence's chunks off the GPU into host
    /// [`HostSealedChunk`]s — the portable, GPU-pointer-free form for
    /// capture/replay fixtures. Mirrors the persistence cold-tier gather but
    /// lives next to its building blocks so capture tooling needn't depend on
    /// the persistence crate. The resident `meta` record is NOT captured; a
    /// replay rebuilds the per-forward header/slice pointers from the chunks.
    #[cfg(feature = "cuda")]
    pub fn dump_sealed_to_host(
        &self,
        chunks: &[super::SealedChunk],
        device: &candle::Device,
    ) -> candle::Result<Vec<HostSealedChunk>> {
        // Resolve + gather per-gid WITHOUT deduping shared physical slots, so the
        // captured byte stream carries one slot's bytes per sub-band gid — exactly
        // what `load_sealed_from_host`'s fresh, distinct per-gid chunks expect on
        // scatter. (The deduped `resolve_sealed_chunk_ptrs` is for the persistence
        // cold tier, which must not write an aliased physical slot twice.)
        let per_chunk_ptrs = self.resolve_sealed_chunk_ptrs_per_gid(chunks)?;
        let flat: Vec<(i64, i64)> = per_chunk_ptrs.iter().flatten().copied().collect();
        let blob = gather_chunks_to_host(device, &flat)?;
        let mut cursor = 0usize;
        let mut out = Vec::with_capacity(chunks.len());
        for (sc, dests) in chunks.iter().zip(&per_chunk_ptrs) {
            let n: usize = dests.iter().map(|&(_, len)| len as usize).sum();
            if cursor + n > blob.len() {
                return Err(candle::Error::Msg(format!(
                    "dump_sealed_to_host: blob underrun (need {n} at {cursor}, have {})",
                    blob.len()
                )));
            }
            let kv_bytes = blob[cursor..cursor + n].to_vec();
            cursor += n;
            let (k_formats, v_formats) = sc.format_tags()?;
            out.push(HostSealedChunk {
                offset: sc.offset,
                token_count: sc.token_count,
                k_formats: k_formats.to_vec(),
                v_formats: v_formats.to_vec(),
                k_pal: (*sc.k_pal).clone(),
                v_pal: (*sc.v_pal).clone(),
                k_scale: (*sc.k_scale).clone(),
                v_scale: (*sc.v_scale).clone(),
                kv_bytes,
            });
        }
        Ok(out)
    }

    /// Rebuild fresh arena chunks for `slot` from captured host bytes and
    /// return the rebuilt [`SealedSequence`] — the symmetric inverse of
    /// [`Self::dump_sealed_to_host`]. The slot must already be allocated
    /// (`alloc_sequence`). Each [`HostSealedChunk`] becomes one block: its
    /// per-`(head, palette)` formats (decoded from the `*_formats` tags) drive a
    /// fresh bulk arena allocation, the captured `kv_bytes` are scattered onto
    /// the freshly-claimed device chunks, and `record_turn` snapshots the grid.
    #[cfg(feature = "cuda")]
    pub fn load_sealed_from_host(
        &self,
        slot: usize,
        chunks: &[HostSealedChunk],
        device: &candle::Device,
    ) -> candle::Result<super::SealedSequence> {
        if chunks.is_empty() {
            return self.record_turn(slot);
        }
        let want = self.n_kv_head() * N_PALETTE;

        let mut specs: Vec<BlockAllocSpec> = Vec::with_capacity(chunks.len());
        for (block_idx, hc) in chunks.iter().enumerate() {
            if hc.k_formats.len() != want || hc.v_formats.len() != want {
                return Err(candle::Error::Msg(format!(
                    "load_sealed_from_host: block {block_idx} expected {want} sub-band formats, \
                     got k={} v={}",
                    hc.k_formats.len(),
                    hc.v_formats.len()
                )));
            }
            let decode = |tags: &[u8], side: &str| -> candle::Result<Vec<KvFormat>> {
                tags.iter()
                    .map(|&t| {
                        KvFormat::from_tag(t).ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "load_sealed_from_host: unknown {side} format tag {t} in block {block_idx}"
                            ))
                        })
                    })
                    .collect()
            };
            specs.push(BlockAllocSpec {
                block_idx,
                k_formats: decode(&hc.k_formats, "K")?,
                v_formats: decode(&hc.v_formats, "V")?,
                k_pal: std::sync::Arc::new(hc.k_pal.clone()),
                v_pal: std::sync::Arc::new(hc.v_pal.clone()),
                k_scale: std::sync::Arc::new(hc.k_scale.clone()),
                v_scale: std::sync::Arc::new(hc.v_scale.clone()),
                offset: hc.offset,
                usage: hc.token_count as u32,
            });
        }

        // Reserve the slot's block-table capacity up front (bulk-alloc requires
        // the slot to already cover max(block_idx)+1 chunks).
        self.ensure_capacity_for_blocks(slot, specs.len())?;

        // Bulk-allocate fresh arena chunks. `arena_info` is resolved INSIDE this
        // call, after the allocs — use exactly this returned pair for the
        // pointer resolve below, with no arena-mutating call in between.
        let mut pool_us = 0u64;
        let mut register_us = 0u64;
        let mut gpu_push_us = 0u64;
        let (hgids, arena_info) = self.alloc_sealed_blocks_bulk(
            slot,
            &specs,
            &mut pool_us,
            &mut register_us,
            &mut gpu_push_us,
        )?;

        // Resolve per-chunk device (ptr, len) destinations in the same gid-walk
        // order the dump used, then scatter each chunk's captured bytes onto its
        // freshly-claimed device slots.
        let block_ptrs = self.resolve_block_ptrs_from_hgids(&hgids, &specs, &arena_info)?;
        if block_ptrs.len() != chunks.len() {
            return Err(candle::Error::Msg(format!(
                "load_sealed_from_host: resolved {} block ptr sets for {} chunks",
                block_ptrs.len(),
                chunks.len()
            )));
        }
        for (block_idx, (dests, hc)) in block_ptrs.iter().zip(chunks).enumerate() {
            let total: i64 = dests.iter().map(|&(_, len)| len).sum();
            if hc.kv_bytes.len() as i64 != total {
                return Err(candle::Error::Msg(format!(
                    "load_sealed_from_host: block {block_idx} captured {} bytes, fresh chunks need {total}",
                    hc.kv_bytes.len()
                )));
            }
            scatter_chunks_to_device(device, dests, &hc.kv_bytes)?;
        }

        self.record_turn(slot)
    }

    /// Resolve device `(ptr, byte_stride)` pairs per block, walking
    /// the supplied [`HeadGids`] directly.
    ///
    /// The cold-load pipeline calls this immediately after
    /// [`super::ChunkedKvBacking::alloc_sealed_blocks_bulk`] using the
    /// `hgids` and `arena_info` that call just produced — no
    /// `state.read()` lock, no slot lookup. Fresh CAS-claimed gids are
    /// unique by construction, so there is no per-block `(arena_idx,
    /// chunk_idx)` dedup either (the legacy `resolve_block_ptrs_in_slot`
    /// did that with a `HashSet` per block — pure overhead on freshly
    /// allocated chunks).
    ///
    /// The outer Vec is in `hgids` order; the inner Vec is in GID-walk
    /// order. `arena_info` must cover every arena_idx referenced — pass
    /// the snapshot returned by `alloc_sealed_blocks_bulk` (still valid
    /// as long as no arena-mutating call runs between the two).
    pub fn resolve_block_ptrs_from_hgids(
        &self,
        hgids: &[super::head_gids::HeadGids],
        specs: &[BlockAllocSpec],
        arena_info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
    ) -> candle::Result<Vec<Vec<(i64, i64)>>> {
        if specs.len() != hgids.len() {
            return Err(candle::Error::Msg(format!(
                "resolve_block_ptrs_from_hgids: {} specs for {} blocks",
                specs.len(),
                hgids.len()
            )));
        }
        let elems = self.inner.elems_per_chunk();
        let mut out: Vec<Vec<(i64, i64)>> = Vec::with_capacity(hgids.len());
        for (block, spec) in hgids.iter().zip(specs) {
            // The block was allocated from these formats, so they are also the
            // authority on how many bytes each of its bands holds. Reading the
            // length off the arena instead would give the class stride, which
            // is generally larger (invariant 8).
            let k_fmt: Vec<u8> = spec.k_formats.iter().map(|f| f.to_tag()).collect();
            let v_fmt: Vec<u8> = spec.v_formats.iter().map(|f| f.to_tag()).collect();
            let mut block_ptrs: Vec<(i64, i64)> = Vec::with_capacity(block.0.len());
            for (gid, tag) in super::head_gids::band_tags(block, &k_fmt, &v_fmt) {
                let arena_idx = gid.arena_idx();
                let chunk_idx = gid.chunk_idx();
                let arena = arena_info.get(arena_idx).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "resolve_block_ptrs_from_hgids: arena index {arena_idx} out of range"
                    ))
                })?;
                if arena.base_ptr == 0 {
                    return Err(candle::Error::Msg(
                        "resolve_block_ptrs_from_hgids: chunk arena is not GPU-resident".into(),
                    ));
                }
                let ptr = arena.base_ptr as i64 + chunk_idx as i64 * arena.chunk_byte_stride;
                block_ptrs.push((ptr, band_payload(tag, elems)?));
            }
            out.push(block_ptrs);
        }
        Ok(out)
    }
}

/// Host snapshot of one sealed chunk's KV bytes + dequant metadata, fully
/// GPU-pointer-free. The portable form produced by
/// [`ChunkedKvBacking::dump_sealed_to_host`] for capture/replay fixtures.
/// `k_formats`/`v_formats` are `KvFormat::to_tag()` values per `(head,
/// palette)`; `kv_bytes` is the raw (possibly quantized) arena data, un-rotated
/// (RoPE is applied in-kernel).
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct HostSealedChunk {
    pub offset: u16,
    pub token_count: u16,
    pub k_formats: Vec<u8>,
    pub v_formats: Vec<u8>,
    pub k_pal: Vec<u8>,
    pub v_pal: Vec<u8>,
    pub k_scale: Vec<f32>,
    pub v_scale: Vec<f32>,
    pub kv_bytes: Vec<u8>,
}

/// Gather scattered VRAM `chunks` — `(device_ptr, byte_len)` pairs — into one
/// contiguous host buffer, in order (a `kv_pack` gather followed by a D2H copy).
/// Mirrors `persistence::transfer::gather_chunks`, kept here so capture tooling
/// in higher crates can reach it via [`ChunkedKvBacking::dump_sealed_to_host`].
#[cfg(feature = "cuda")]
fn gather_chunks_to_host(
    device: &candle::Device,
    chunks: &[(i64, i64)],
) -> candle::Result<Vec<u8>> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let dev = match device {
        candle::Device::Cuda(d) => d,
        _ => {
            return Err(candle::Error::Msg(
                "gather_chunks_to_host requires a CUDA device".into(),
            ))
        }
    };
    let total: i64 = chunks.iter().map(|&(_, len)| len).sum();
    if total == 0 {
        return Ok(Vec::new());
    }
    let staging = unsafe {
        dev.alloc::<u8>(total as usize)
            .map_err(|e| candle::Error::Msg(format!("gather_chunks_to_host: staging alloc: {e}")))?
    };
    let staging_base = {
        let stream = dev.cuda_stream();
        let base = staging.device_ptr(&stream).0 as i64;
        base
    };
    let mut plan = MigrationPlan::new();
    let mut offset = 0i64;
    for &(ptr, len) in chunks {
        plan.push(ptr, staging_base + offset, len);
        offset += len;
    }
    kv_migrate(device, &plan)?;
    dev.memcpy_dtov(&staging)
        .map_err(|e| candle::Error::Msg(format!("gather_chunks_to_host: staging DtoH: {e}")))
}

/// Scatter a contiguous host buffer back into scattered VRAM `chunks` —
/// `(device_ptr, byte_len)` pairs — in order (a HtoD copy of the whole blob
/// followed by a `kv_unpack` / `kv_migrate` into the destinations). The
/// symmetric inverse of [`gather_chunks_to_host`]; `host` must be exactly the
/// summed `byte_len` of `chunks`.
#[cfg(feature = "cuda")]
fn scatter_chunks_to_device(
    device: &candle::Device,
    chunks: &[(i64, i64)],
    host: &[u8],
) -> candle::Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let dev = match device {
        candle::Device::Cuda(d) => d,
        _ => {
            return Err(candle::Error::Msg(
                "scatter_chunks_to_device requires a CUDA device".into(),
            ))
        }
    };
    let total: i64 = chunks.iter().map(|&(_, len)| len).sum();
    if host.len() as i64 != total {
        return Err(candle::Error::Msg(format!(
            "scatter_chunks_to_device: host buffer is {} bytes, chunks need {total}",
            host.len()
        )));
    }
    if total == 0 {
        return Ok(());
    }
    let staging = dev
        .memcpy_stod(host)
        .map_err(|e| candle::Error::Msg(format!("scatter_chunks_to_device: HtoD: {e}")))?;
    let staging_base = {
        let stream = dev.cuda_stream();
        let base = staging.device_ptr(&stream).0 as i64;
        base
    };
    let mut plan = MigrationPlan::new();
    let mut offset = 0i64;
    for &(ptr, len) in chunks {
        plan.push(staging_base + offset, ptr, len);
        offset += len;
    }
    kv_migrate(device, &plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_accumulates_records_and_totals() {
        let mut plan = MigrationPlan::new();
        assert!(plan.is_empty());
        plan.push(0x1000, 0x2000, 256);
        plan.push(0x4000, 0x2100, 512);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.total_bytes(), 768);
        assert_eq!(
            plan.records[1],
            MigrationRecord {
                src_ptr: 0x4000,
                dst_ptr: 0x2100,
                byte_len: 512,
            }
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn kv_migrate_gather_then_scatter_is_byte_identical() {
        use candle::cuda_backend::cudarc::driver::DevicePtr;

        let device = match candle::Device::cuda_if_available(0) {
            Ok(d @ candle::Device::Cuda(_)) => d,
            _ => return, // no GPU — skip
        };
        let dev = match &device {
            candle::Device::Cuda(d) => d,
            _ => unreachable!(),
        };

        // Three scattered source chunks with distinct byte patterns.
        let chunks: Vec<Vec<u8>> = vec![
            (0..256u32).map(|i| (i % 256) as u8).collect(),
            (0..512u32).map(|i| ((i * 7 + 3) % 256) as u8).collect(),
            (0..176u32).map(|i| ((i * 13 + 1) % 256) as u8).collect(),
        ];
        let total: usize = chunks.iter().map(|c| c.len()).sum();

        let src_gpus: Vec<_> = chunks.iter().map(|c| dev.memcpy_stod(c).unwrap()).collect();
        let staging = unsafe { dev.alloc::<u8>(total).unwrap() };

        let stream = dev.cuda_stream();
        let staging_base = staging.device_ptr(&stream).0 as i64;
        let src_ptrs: Vec<i64> = src_gpus
            .iter()
            .map(|g| g.device_ptr(&stream).0 as i64)
            .collect();
        drop(stream);

        // kv_pack: gather the scattered chunks into the contiguous staging buffer.
        let mut gather = MigrationPlan::new();
        let mut offset = 0i64;
        for (i, c) in chunks.iter().enumerate() {
            gather.push(src_ptrs[i], staging_base + offset, c.len() as i64);
            offset += c.len() as i64;
        }
        kv_migrate(&device, &gather).unwrap();

        let staging_cpu = dev.memcpy_dtov(&staging).unwrap();
        let concatenated: Vec<u8> = chunks.iter().flatten().copied().collect();
        assert_eq!(staging_cpu, concatenated, "gather concatenates the chunks");

        // kv_unpack: scatter the staging buffer back into fresh chunks.
        let dst_gpus: Vec<_> = chunks
            .iter()
            .map(|c| unsafe { dev.alloc::<u8>(c.len()).unwrap() })
            .collect();
        let stream = dev.cuda_stream();
        let dst_ptrs: Vec<i64> = dst_gpus
            .iter()
            .map(|g| g.device_ptr(&stream).0 as i64)
            .collect();
        drop(stream);

        let mut scatter = MigrationPlan::new();
        let mut offset = 0i64;
        for (i, c) in chunks.iter().enumerate() {
            scatter.push(staging_base + offset, dst_ptrs[i], c.len() as i64);
            offset += c.len() as i64;
        }
        kv_migrate(&device, &scatter).unwrap();

        for (i, c) in chunks.iter().enumerate() {
            let back = dev.memcpy_dtov(&dst_gpus[i]).unwrap();
            assert_eq!(&back, c, "chunk {i} round-trips byte-identical");
        }
    }
}
