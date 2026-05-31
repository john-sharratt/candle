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
use candle::cuda::cudarc::driver::CudaStream;

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
        seq: &super::SealedSequence,
    ) -> candle::Result<Vec<(i64, i64)>> {
        let arena_info = self.resolve_arena_info()?;
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for chunk in &seq.chunks {
            for gid in chunk.gids.0.iter() {
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
                out.push((ptr, arena.chunk_byte_stride));
            }
        }
        Ok(out)
    }

    /// Resolve device pointers for specific block positions in a
    /// scratch slot, without materialising a [`super::SealedSequence`].
    ///
    /// For each `block_idx` in `block_indices`, returns the
    /// `(device_ptr, byte_len)` list for that block's unique sub-band
    /// arena chunks (deduped by `(arena_idx, chunk_idx)` *within the
    /// block*, matching what `kv_migrate` needs as scatter
    /// destinations). The returned outer Vec is in `block_indices`
    /// order; the inner Vec is in GID-walk order.
    ///
    /// This is the cold-load chunked path's fast equivalent of
    /// `record_turn` + `resolve_sealed_chunk_ptrs`: it skips the
    /// allocation and clone-per-chunk overhead of building a
    /// `SealedSequence` snapshot when the caller only needs the
    /// device pointers for a known subset of blocks.
    /// Caller is responsible for providing a `arena_info` that covers
    /// every arena_idx referenced by the requested blocks. The
    /// canonical source is the value returned by
    /// [`super::ChunkedKvBacking::alloc_sealed_blocks_bulk`] on the
    /// same backing — captured **after** that call's allocs, so it
    /// reflects every arena the freshly-allocated chunks point into.
    /// Reusing it here avoids a redundant `storage.read()` walk per
    /// call (~50 µs × 82 calls ≈ 4 ms on the 1824-chunk turn).
    ///
    /// If the caller doesn't have a fresh `arena_info` handy, pass
    /// `&self.resolve_arena_info()?`.
    pub fn resolve_block_ptrs_in_slot(
        &self,
        slot: usize,
        block_indices: &[usize],
        arena_info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
    ) -> candle::Result<Vec<Vec<(i64, i64)>>> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let slot_state = state
            .sequences
            .get(slot)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "resolve_block_ptrs_in_slot: slot {slot} is not allocated"
                ))
            })?;
        let chunks = slot_state.chunks_slice();

        let mut out: Vec<Vec<(i64, i64)>> = Vec::with_capacity(block_indices.len());
        for &block_idx in block_indices {
            let chunk = chunks.get(block_idx).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "resolve_block_ptrs_in_slot: block_idx {block_idx} out of range \
                     (slot has {} chunks)",
                    chunks.len()
                ))
            })?;
            let mut seen = std::collections::HashSet::new();
            let mut block_ptrs: Vec<(i64, i64)> = Vec::new();
            for gid in chunk.gids.0.iter() {
                let arena_idx = gid.arena_idx();
                let chunk_idx = gid.chunk_idx();
                if !seen.insert((arena_idx, chunk_idx)) {
                    continue;
                }
                let arena = arena_info.get(arena_idx).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "resolve_block_ptrs_in_slot: arena index {arena_idx} out of range"
                    ))
                })?;
                if arena.base_ptr == 0 {
                    return Err(candle::Error::Msg(
                        "resolve_block_ptrs_in_slot: chunk arena is not GPU-resident".into(),
                    ));
                }
                let ptr = arena.base_ptr as i64 + chunk_idx as i64 * arena.chunk_byte_stride;
                block_ptrs.push((ptr, arena.chunk_byte_stride));
            }
            out.push(block_ptrs);
        }
        Ok(out)
    }
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
