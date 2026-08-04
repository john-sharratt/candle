//! Per-chunk KV-head metadata record pool.
//!
//! A sealed chunk's KV-head metadata (palette maps, formats, outer scales, and
//! the resolved per-palette device pointers — the `KvHead[n_kv_head]` record the
//! attention kernels read) is constant for as long as the chunk is resident, and
//! is *shared* by every slot that references the chunk. Rather than rebuild and
//! re-upload it per layer per forward, the record lives once in a device-resident
//! slab and travels with the chunk.
//!
//! This module owns the pool: a reference-counted, free-on-drop handle
//! ([`MetaGid`]) into a slab of fixed-size records, modeled on
//! [`super::gid_pool::ChunkGid`] but at structural-event cadence (seal / inject /
//! migrate), not the decode hot path — so the allocator is a simple guarded
//! forward-scan rather than the lock-free bitmap the chunk pool needs.
//!
//! A [`MetaGid`] is an `i64` id plus an `Arc` to its slab's refcount table. The
//! id packs `(slab_idx, record_idx)` against [`META_SLAB_STRIDE`] exactly as a
//! `ChunkGid` packs `(arena_idx, chunk_idx)` against `arena_gid_stride()`, so the
//! record's device address resolves as `slab_base_ptr + record_idx * record_bytes`.
//! Clone bumps the slot refcount (every slot that references the chunk shares one
//! record); drop releases it; the record's slot frees when the last holder drops.
//!
//! Records are built at quantize / cold-load / warm→hot elevate and read by the
//! attention kernels via each slice's `kvheads_ptr`. A few helpers
//! (`device_addr`, `live_records`, `slab_count`) are CPU-build- or test-only and
//! carry their own `#[allow(dead_code)]`.

use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle::Device;

use crate::kv_cache::arena_table::{ArenaFormatTag, ResolvedArenaInfo, N_PALETTE};
use crate::kv_cache::chunked::head_gids::HeadGids;

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
#[cfg(feature = "cuda")]
use candle::cuda_backend::WrapErr;

/// Records per metadata slab. One slab holds this many `KvHead[n_kv_head]`
/// records; the pool grows by appending slabs. Chosen so a slab is a modest
/// fixed allocation and growth is rare.
pub(crate) const META_SLAB_RECORDS: usize = 4096;

/// Stride that packs `(slab_idx, record_idx)` into one `i64` id:
/// `id = slab_idx * META_SLAB_STRIDE + record_idx`. Must exceed
/// [`META_SLAB_RECORDS`] so record indices never collide across slabs. Mirrors
/// the role of `arena_gid_stride()` for the chunk-gid namespace.
pub(crate) const META_SLAB_STRIDE: usize = 1 << 20;

const _: () = assert!(
    META_SLAB_STRIDE >= META_SLAB_RECORDS,
    "META_SLAB_STRIDE must cover a slab's record count",
);

/// Bytes of one head's serialized `KvHead` record at `head_dim`, matching the
/// CUDA layout in `slot_types.cuh` (`kv_head_byte_size`): `head_dim/2 + 104`.
///
/// Layout: `k_pal[head_dim/4] + v_pal[head_dim/4] + k_ptr[4]·8 + v_ptr[4]·8 +
/// k_fmt[4] + v_fmt[4] + k_scale[4]·4 + v_scale[4]·4`.
pub(crate) fn kv_head_record_bytes(head_dim: usize) -> usize {
    (head_dim / 4) * 2 + 32 + 32 + 4 + 4 + 16 + 16
}

/// Bytes of one chunk's full `KvHead[n_kv_head]` record.
pub(crate) fn chunk_record_bytes(n_kv_head: usize, head_dim: usize) -> usize {
    n_kv_head * kv_head_record_bytes(head_dim)
}

/// Serialize a chunk's `KvHead[n_kv_head]` record into `dst` (length must equal
/// [`chunk_record_bytes`]). This is the resident-record body — identical
/// byte-for-byte to the per-head portion of the decode/prefill inline-head
/// serialization, just lifted out of the per-slice `TokenSlice` header.
///
/// `k_pal`/`v_pal` are `n_kv_head·(head_dim/4)` bytes (empty ⇒ identity routing);
/// `k_scale`/`v_scale` are `n_kv_head·N_PALETTE` f32s (empty ⇒ unity). The 8
/// pointers per head resolve each `(head, palette, K/V)` GID against `arena_info`
/// as `base_ptr + chunk_idx·chunk_byte_stride` — the location-dependent bytes a
/// migration/defrag re-patches.
pub(crate) fn serialize_kv_heads(
    dst: &mut [u8],
    gids: &HeadGids,
    k_pal: &[u8],
    v_pal: &[u8],
    k_scale: &[f32],
    v_scale: &[f32],
    n_kv_head: usize,
    head_dim: usize,
    arena_info: &[ResolvedArenaInfo],
) {
    debug_assert!(
        head_dim >= 4,
        "head_dim must be >= 4 for 2-bit pal_map packing"
    );
    debug_assert_eq!(
        dst.len(),
        chunk_record_bytes(n_kv_head, head_dim),
        "record dst must be exactly chunk_record_bytes"
    );
    let pal_bytes = head_dim / 4;
    let sub_hd = (head_dim / N_PALETTE).max(1);
    let mut pos = 0usize;

    macro_rules! put {
        ($b:expr) => {{
            let b: &[u8] = $b;
            dst[pos..pos + b.len()].copy_from_slice(b);
            pos += b.len();
        }};
    }

    for h in 0..n_kv_head {
        // Palette maps: populated slice when present, else identity routing
        // (matches `KvHeadHost::from_gids` / live ChunkWindow identity bytes).
        let k_pal_head = k_pal.get(h * pal_bytes..(h + 1) * pal_bytes);
        let v_pal_head = v_pal.get(h * pal_bytes..(h + 1) * pal_bytes);
        match k_pal_head {
            Some(s) => put!(s),
            None => {
                // Identity routing ORs into dst, so the target bytes must start
                // clean — `dst` may be a reused buffer (the decode pinned buffer
                // preserves bytes across forwards).
                dst[pos..pos + pal_bytes].fill(0);
                for d in 0..head_dim {
                    let pal_idx = ((d / sub_hd).min(N_PALETTE - 1)) as u8;
                    dst[pos + d / 4] |= pal_idx << ((d % 4) * 2);
                }
                pos += pal_bytes;
            }
        }
        match v_pal_head {
            Some(s) => put!(s),
            None => {
                dst[pos..pos + pal_bytes].fill(0);
                for d in 0..head_dim {
                    let pal_idx = ((d / sub_hd).min(N_PALETTE - 1)) as u8;
                    dst[pos + d / 4] |= pal_idx << ((d % 4) * 2);
                }
                pos += pal_bytes;
            }
        }

        let mut k_ptr = [0u64; N_PALETTE];
        let mut v_ptr = [0u64; N_PALETTE];
        let mut k_fmt = [ArenaFormatTag::BF16.as_u8(); N_PALETTE];
        let mut v_fmt = [ArenaFormatTag::BF16.as_u8(); N_PALETTE];
        for p in 0..N_PALETTE {
            let k_gid = gids.k_gid_pal(h, p);
            let v_gid = gids.v_gid_pal(h, p);
            if let Some(ai) = arena_info.get(k_gid.arena_idx()) {
                k_ptr[p] = ai.base_ptr + k_gid.chunk_idx() as u64 * ai.chunk_byte_stride as u64;
                k_fmt[p] = ai.k_format_tag.as_u8();
            }
            if let Some(ai) = arena_info.get(v_gid.arena_idx()) {
                v_ptr[p] = ai.base_ptr + v_gid.chunk_idx() as u64 * ai.chunk_byte_stride as u64;
                v_fmt[p] = ai.v_format_tag.as_u8();
            }
        }
        for &ptr in &k_ptr {
            put!(&ptr.to_le_bytes());
        }
        for &ptr in &v_ptr {
            put!(&ptr.to_le_bytes());
        }
        put!(&k_fmt);
        put!(&v_fmt);
        let scale_base = h * N_PALETTE;
        for p in 0..N_PALETTE {
            let s = k_scale.get(scale_base + p).copied().unwrap_or(1.0);
            put!(&s.to_le_bytes());
        }
        for p in 0..N_PALETTE {
            let s = v_scale.get(scale_base + p).copied().unwrap_or(1.0);
            put!(&s.to_le_bytes());
        }
    }
}

/// Per-slab refcount table. Lives behind an `Arc` shared by every [`MetaGid`]
/// allocated from this slab. `counts[i] == 0` means the record slot is free;
/// `≥ 1` means it is held by that many references (slots sharing the chunk).
#[derive(Debug)]
struct MetaSlabRefcounts {
    counts: Vec<AtomicU16>,
    first_free: AtomicUsize,
    live: AtomicUsize,
    slab_idx: usize,
}

impl MetaSlabRefcounts {
    fn new(slab_idx: usize) -> Self {
        let mut counts = Vec::with_capacity(META_SLAB_RECORDS);
        for _ in 0..META_SLAB_RECORDS {
            counts.push(AtomicU16::new(0));
        }
        Self {
            counts,
            first_free: AtomicUsize::new(0),
            live: AtomicUsize::new(0),
            slab_idx,
        }
    }

    fn try_claim_one(&self) -> Option<usize> {
        let start = self.first_free.load(Ordering::Acquire);
        for i in start..META_SLAB_RECORDS {
            if self.counts[i]
                .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                let _ = self.first_free.compare_exchange(
                    start,
                    i + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                self.live.fetch_add(1, Ordering::Relaxed);
                return Some(i);
            }
        }
        None
    }

    #[inline]
    fn inc(&self, record_idx: usize) {
        self.counts[record_idx].fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn dec(&self, record_idx: usize) {
        let prev = self.counts[record_idx].fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.live.fetch_sub(1, Ordering::Relaxed);
            let mut cur = self.first_free.load(Ordering::Acquire);
            while record_idx < cur {
                match self.first_free.compare_exchange_weak(
                    cur,
                    record_idx,
                    Ordering::Release,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(actual) => cur = actual,
                }
            }
        } else if prev == 0 {
            panic!(
                "MetaSlabRefcounts::dec: refcount underflow at slab {} record {}",
                self.slab_idx, record_idx
            );
        }
    }

    #[inline]
    fn load(&self, record_idx: usize) -> u16 {
        self.counts[record_idx].load(Ordering::Relaxed)
    }

    #[inline]
    #[allow(dead_code)] // diagnostics + test assertions
    fn live_count(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }
}

/// Backing of a [`MetaGid`]: either a real pooled slot (shares the slab's
/// refcount table) or a detached test/diagnostic record with no pool.
#[derive(Clone, Debug)]
enum MetaBacking {
    Pooled(Arc<MetaSlabRefcounts>),
    Detached(Arc<AtomicU16>),
}

/// RAII handle to one per-chunk KV-head metadata record.
///
/// Carries the record's `id` (packing `(slab_idx, record_idx)`) and a shared
/// reference to its slab's refcount table. Cloning bumps the slot refcount so
/// every slot referencing the same physical chunk resolves to the **same**
/// record; dropping the last clone frees the slot. Stored alongside `HeadGids`
/// on `ChunkWindow` / `SealedChunk` so a chunk's record shares the chunk's
/// lifetime automatically through `#[derive(Clone)]`.
#[derive(Debug)]
pub struct MetaGid {
    id: i64,
    /// Cached device address of this record (`slab_base_ptr + record_idx ·
    /// record_bytes`), computed at allocation. The slab's device buffer never
    /// moves, so this is stable for the handle's life. `0` ⇒ no device residence
    /// (CPU/host-only pool). Lets a chunk's `kvheads_ptr` be resolved straight
    /// from the handle without a pool lookup.
    device_addr: u64,
    backing: MetaBacking,
}

impl MetaGid {
    #[inline]
    pub fn raw(&self) -> i64 {
        self.id
    }

    /// Cached device address of this record (0 if not device-resident).
    #[inline]
    pub fn device_addr(&self) -> u64 {
        self.device_addr
    }

    #[inline]
    pub fn slab_idx(&self) -> usize {
        self.id as usize / META_SLAB_STRIDE
    }

    #[inline]
    pub fn record_idx(&self) -> usize {
        self.id as usize % META_SLAB_STRIDE
    }

    /// Current number of holders of this record's slot. `1` = uniquely owned
    /// (safe to rewrite in place); `> 1` = shared across slots. Detached
    /// records always report their own refcount.
    #[inline]
    pub fn strong_count(&self) -> u16 {
        match &self.backing {
            MetaBacking::Pooled(t) => t.load(self.record_idx()),
            MetaBacking::Detached(c) => c.load(Ordering::Relaxed),
        }
    }

    /// A detached record with no pool backing, for tests and diagnostic chunks
    /// that never resolve a real device record. Mirrors `ChunkGid::detached`.
    pub fn detached(id: i64) -> Self {
        Self {
            id,
            device_addr: 0,
            backing: MetaBacking::Detached(Arc::new(AtomicU16::new(1))),
        }
    }
}

impl Clone for MetaGid {
    fn clone(&self) -> Self {
        match &self.backing {
            MetaBacking::Pooled(t) => {
                t.inc(self.record_idx());
                Self {
                    id: self.id,
                    device_addr: self.device_addr,
                    backing: MetaBacking::Pooled(Arc::clone(t)),
                }
            }
            MetaBacking::Detached(c) => {
                c.fetch_add(1, Ordering::Relaxed);
                Self {
                    id: self.id,
                    device_addr: self.device_addr,
                    backing: MetaBacking::Detached(Arc::clone(c)),
                }
            }
        }
    }
}

impl Drop for MetaGid {
    fn drop(&mut self) {
        match &self.backing {
            MetaBacking::Pooled(t) => t.dec(self.record_idx()),
            MetaBacking::Detached(c) => {
                c.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

/// One slab's parallel host-refcount + device-buffer pair, kept index-aligned
/// (`refs[i]` describes the same slab as `device[i]`).
struct SlabSet {
    refs: Vec<Arc<MetaSlabRefcounts>>,
    #[cfg(feature = "cuda")]
    device: Vec<DeviceSlab>,
}

/// Device-resident bytes of one slab and its cached base pointer (`base_ptr` is
/// read by `device_addr` when resolving a record's address for a `kvheads_ptr`).
#[cfg(feature = "cuda")]
struct DeviceSlab {
    gpu: CudaSlice<u8>,
    base_ptr: u64,
}

/// Host-side allocator + device residence for per-chunk KV-head records.
///
/// Owns growable slabs: each slab is a host refcount table plus (on CUDA) a
/// device byte buffer of `META_SLAB_RECORDS · record_bytes`. Allocation scans
/// existing slabs for a free record slot and appends a new slab when full —
/// the same growth model the KV arenas use, at far lower cadence. The record's
/// device address is `slab_base_ptr + record_idx · record_bytes`.
#[derive(Debug)]
pub struct MetaPool {
    slabs: Mutex<SlabSet>,
    // record_bytes/device are only read on CUDA (device addressing + upload).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    record_bytes: usize,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    device: Device,
}

impl std::fmt::Debug for SlabSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlabSet")
            .field("n_slabs", &self.refs.len())
            .finish()
    }
}

impl MetaPool {
    /// Create an empty pool whose records are `record_bytes` each, resident on
    /// `device`. Slabs are allocated lazily on first `allocate`.
    pub fn new(record_bytes: usize, device: Device) -> Self {
        Self {
            slabs: Mutex::new(SlabSet {
                refs: Vec::new(),
                #[cfg(feature = "cuda")]
                device: Vec::new(),
            }),
            record_bytes,
            device,
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_stream(&self) -> Option<Arc<CudaStream>> {
        match &self.device {
            Device::Cuda(cd) => Some(cd.cuda_stream()),
            _ => None,
        }
    }

    /// Allocate a fresh record slot, returning its RAII handle (with the record's
    /// device address cached). Scans existing slabs first; appends a new slab
    /// (host refcounts + device buffer) when all are full. At structural cadence,
    /// so a single lock for the whole claim is fine.
    pub fn allocate(&self) -> candle::Result<MetaGid> {
        let mut s = self.slabs.lock().expect("meta pool lock poisoned");
        for slab_idx in 0..s.refs.len() {
            if let Some(record_idx) = s.refs[slab_idx].try_claim_one() {
                let addr = self.slab_record_addr(&s, slab_idx, record_idx);
                return Ok(Self::handle_for(&s.refs[slab_idx], record_idx, addr));
            }
        }
        // All slabs full (or none): append one.
        let slab_idx = s.refs.len();
        let slab = Arc::new(MetaSlabRefcounts::new(slab_idx));
        let record_idx = slab
            .try_claim_one()
            .expect("fresh meta slab must have a free record");
        #[cfg(feature = "cuda")]
        {
            // Back the slab with device memory only on a real CUDA device. A CPU
            // device (unit tests; warm/cold-tier pools) keeps host refcounts only:
            // records are never read on the CPU, so device_addr stays 0.
            if self.cuda_stream().is_some() {
                let dev_slab = self.alloc_device_slab()?;
                s.device.push(dev_slab);
                debug_assert_eq!(s.device.len(), slab_idx + 1);
            }
        }
        let addr = self.slab_record_addr(&s, slab_idx, record_idx);
        let handle = Self::handle_for(&slab, record_idx, addr);
        s.refs.push(slab);
        Ok(handle)
    }

    /// Device address of record `record_idx` in slab `slab_idx`: the slab's
    /// device base pointer plus the record's byte offset. `0` when the slab has
    /// no device buffer (CPU/host-only pool).
    #[allow(unused_variables)]
    fn slab_record_addr(&self, s: &SlabSet, slab_idx: usize, record_idx: usize) -> u64 {
        #[cfg(feature = "cuda")]
        {
            if let Some(dev) = s.device.get(slab_idx) {
                return dev.base_ptr + (record_idx * self.record_bytes) as u64;
            }
        }
        0
    }

    /// True when this pool backs records with device memory (a real CUDA
    /// device). A CPU/host-only pool keeps refcounts only; its records have no
    /// readable address, so a handle from it must be treated as non-resident.
    pub fn is_device_resident(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.cuda_stream().is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Upload many records, coalescing the device copies. Records that occupy
    /// consecutive slots in a slab (the common case — a batch allocated together)
    /// collapse to a single `memcpy_htod` per run, versus one tiny copy per
    /// record. No-op without CUDA / on a host-only pool.
    #[allow(unused_variables)]
    pub fn write_records_batched(&self, items: &[(MetaGid, Vec<u8>)]) -> candle::Result<()> {
        #[cfg(feature = "cuda")]
        {
            if items.is_empty() {
                return Ok(());
            }
            let stream = match self.cuda_stream() {
                Some(s) => s,
                None => return Ok(()),
            };
            let rb = self.record_bytes;
            use std::collections::HashMap;
            let mut by_slab: HashMap<usize, Vec<(usize, &[u8])>> = HashMap::new();
            for (gid, bytes) in items {
                debug_assert_eq!(bytes.len(), rb, "record byte-size mismatch");
                by_slab
                    .entry(gid.slab_idx())
                    .or_default()
                    .push((gid.record_idx(), bytes.as_slice()));
            }
            // Build every run's staging buffer BEFORE taking the lock. Coalescing
            // records into runs, allocating the staging vec and gathering the
            // bytes into it is pure host-side CPU work that needs no exclusion —
            // only reaching `s.device[slab_idx]` does. The lock previously
            // covered all of it, so every concurrent meta-record write serialised
            // behind another writer's memcpy *and* its buffer construction.
            struct Run {
                slab_idx: usize,
                off: usize,
                staging: Vec<u8>,
            }
            let mut runs: Vec<Run> = Vec::new();
            for (slab_idx, mut recs) in by_slab {
                recs.sort_unstable_by_key(|(i, _)| *i);
                let mut i = 0;
                while i < recs.len() {
                    let run_start = recs[i].0;
                    let mut j = i;
                    while j + 1 < recs.len() && recs[j + 1].0 == recs[j].0 + 1 {
                        j += 1;
                    }
                    let run_len = j - i + 1;
                    let mut staging = vec![0u8; run_len * rb];
                    for (k, (_, bytes)) in recs[i..=j].iter().enumerate() {
                        staging[k * rb..(k + 1) * rb].copy_from_slice(bytes);
                    }
                    runs.push(Run {
                        slab_idx,
                        off: run_start * rb,
                        staging,
                    });
                    i = j + 1;
                }
            }

            // The copies themselves stay under the lock: issuing one needs
            // `&mut` on the destination slab. Their ORDER and stream are
            // unchanged — this moves host work out of the critical section, it
            // does not reorder or defer any GPU operation. Each `staging` now
            // outlives its copy by construction (it lives in `runs` until the
            // function returns), which is strictly safer than the previous
            // per-iteration temporary.
            let mut s = self.slabs.lock().expect("meta pool lock poisoned");
            for run in &runs {
                let Some(dev) = s.device.get_mut(run.slab_idx) else {
                    continue;
                };
                let len = run.staging.len();
                stream
                    .memcpy_htod(&run.staging, &mut dev.gpu.slice_mut(run.off..run.off + len))
                    .w()?;
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn live_records(&self) -> usize {
        let s = self.slabs.lock().expect("meta pool lock poisoned");
        s.refs.iter().map(|r| r.live_count()).sum()
    }

    #[allow(dead_code)]
    pub fn slab_count(&self) -> usize {
        self.slabs
            .lock()
            .expect("meta pool lock poisoned")
            .refs
            .len()
    }

    #[cfg(feature = "cuda")]
    fn alloc_device_slab(&self) -> candle::Result<DeviceSlab> {
        let stream = self
            .cuda_stream()
            .ok_or_else(|| candle::Error::Msg("meta pool: no cuda stream".into()))?;
        let byte_len = META_SLAB_RECORDS * self.record_bytes;
        // SAFETY: untyped alloc; zeroed below so an unwritten slot reads as
        // null pointers / zero scales rather than garbage.
        let mut gpu = unsafe { stream.alloc::<u8>(byte_len).w()? };
        let zeros = vec![0u8; byte_len];
        stream.memcpy_htod(&zeros, &mut gpu).w()?;
        let base_ptr = {
            let (p, _g) = gpu.device_ptr(&stream);
            p
        };
        Ok(DeviceSlab { gpu, base_ptr })
    }

    fn handle_for(slab: &Arc<MetaSlabRefcounts>, record_idx: usize, device_addr: u64) -> MetaGid {
        let id = (slab.slab_idx * META_SLAB_STRIDE + record_idx) as i64;
        MetaGid {
            id,
            device_addr,
            backing: MetaBacking::Pooled(Arc::clone(slab)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::arena_table::ArenaFormatTag;
    use crate::kv_cache::chunked::gid_pool::ChunkGid;

    const REC_BYTES: usize = 2 * 168; // 2 heads, HD128

    fn cpu_pool() -> MetaPool {
        MetaPool::new(REC_BYTES, Device::Cpu)
    }

    #[test]
    fn allocate_returns_distinct_ids() {
        let pool = cpu_pool();
        let a = pool.allocate().unwrap();
        let b = pool.allocate().unwrap();
        let c = pool.allocate().unwrap();
        assert_ne!(a.raw(), b.raw());
        assert_ne!(b.raw(), c.raw());
        assert_eq!((a.record_idx(), b.record_idx(), c.record_idx()), (0, 1, 2));
        assert_eq!(a.slab_idx(), 0);
        assert_eq!(pool.live_records(), 3);
    }

    #[test]
    fn clone_shares_one_record_slot() {
        let pool = cpu_pool();
        let a = pool.allocate().unwrap();
        let a2 = a.clone();
        let a3 = a.clone();
        assert_eq!(a.raw(), a2.raw());
        assert_eq!(a.raw(), a3.raw());
        assert_eq!(a.strong_count(), 3);
        assert_eq!(pool.live_records(), 1);
    }

    #[test]
    fn slot_frees_only_on_last_drop() {
        let pool = cpu_pool();
        let a = pool.allocate().unwrap();
        let id = a.raw();
        let a2 = a.clone();
        drop(a);
        assert_eq!(pool.live_records(), 1);
        assert_eq!(a2.strong_count(), 1);
        drop(a2);
        assert_eq!(pool.live_records(), 0);
        let reused = pool.allocate().unwrap();
        assert_eq!(reused.raw(), id);
    }

    #[test]
    fn grows_across_slabs_when_full() {
        let pool = cpu_pool();
        let mut held = Vec::new();
        for _ in 0..META_SLAB_RECORDS {
            held.push(pool.allocate().unwrap());
        }
        assert_eq!(pool.slab_count(), 1);
        let overflow = pool.allocate().unwrap();
        assert_eq!(pool.slab_count(), 2);
        assert_eq!(overflow.slab_idx(), 1);
        assert_eq!(overflow.record_idx(), 0);
        assert_eq!(overflow.raw(), META_SLAB_STRIDE as i64);
    }

    #[test]
    fn detached_record_has_no_pool() {
        let d = MetaGid::detached(-1);
        assert_eq!(d.raw(), -1);
        let d2 = d.clone();
        assert_eq!(d.strong_count(), 2);
        drop(d2);
        assert_eq!(d.strong_count(), 1);
    }

    /// Byte-exact golden for the record body. HD=4, n_kv_head=1, single arena
    /// (one base_ptr/stride), identity palette (empty ⇒ identity), unity scales.
    /// Asserts the exact 168-bytes-at-HD4 = `4/2 + 104 = 106`-byte layout.
    #[test]
    fn serialize_kv_heads_golden_single_palette() {
        let head_dim = 4usize;
        let n_kv_head = 1usize;
        let rec = chunk_record_bytes(n_kv_head, head_dim);
        assert_eq!(rec, head_dim / 2 + 104); // 106

        // One arena, base_ptr=0x1000, stride=512. All 8 sub-band GIDs point at
        // arena 0, chunk_idx 0 → every pointer == base_ptr.
        let gids = HeadGids::uniform(ChunkGid::detached(0), n_kv_head);
        let arena_info = vec![ResolvedArenaInfo {
            base_ptr: 0x1000,
            chunk_byte_stride: 512,
            k_format_tag: ArenaFormatTag::Q8_0,
            v_format_tag: ArenaFormatTag::Q4_0,
            chunk_capacity: u32::MAX,
        }];

        let mut dst = vec![0u8; rec];
        serialize_kv_heads(
            &mut dst,
            &gids,
            &[],
            &[],
            &[],
            &[],
            n_kv_head,
            head_dim,
            &arena_info,
        );

        let mut exp: Vec<u8> = Vec::new();
        // k_pal[1]: identity for HD4 → dims 0,1,2,3 → palettes 0,1,2,3 (sub_hd=1)
        // packed: (3<<6)|(2<<4)|(1<<2)|0 = 0xE4
        exp.push(0xE4);
        // v_pal[1]: same
        exp.push(0xE4);
        // k_ptr[4]: all base_ptr 0x1000 (chunk_idx 0)
        for _ in 0..4 {
            exp.extend_from_slice(&0x1000u64.to_le_bytes());
        }
        // v_ptr[4]: same
        for _ in 0..4 {
            exp.extend_from_slice(&0x1000u64.to_le_bytes());
        }
        // k_fmt[4] = Q8_0 tag, v_fmt[4] = Q4_0 tag
        for _ in 0..4 {
            exp.push(ArenaFormatTag::Q8_0.as_u8());
        }
        for _ in 0..4 {
            exp.push(ArenaFormatTag::Q4_0.as_u8());
        }
        // k_scale[4]=1.0, v_scale[4]=1.0
        for _ in 0..4 {
            exp.extend_from_slice(&1.0f32.to_le_bytes());
        }
        for _ in 0..4 {
            exp.extend_from_slice(&1.0f32.to_le_bytes());
        }
        assert_eq!(exp.len(), rec);
        assert_eq!(dst, exp, "record body must match exact KvHead byte layout");
    }

    /// Two sub-bands in two different arenas resolve to two different pointers
    /// within one head (the multi-arena reality the design must support).
    #[test]
    fn serialize_kv_heads_multi_arena_pointers() {
        let head_dim = 4usize;
        let n_kv_head = 1usize;
        let rec = chunk_record_bytes(n_kv_head, head_dim);

        // GID layout per head: slot = palette*2 + is_value, over N_PALETTE=4.
        // Put K-palette-0 in arena 0 chunk 1, K-palette-1 in arena 1 chunk 2.
        use crate::kv_cache::chunked::head_gids::GIDS_PER_HEAD;
        let stride = crate::kv_cache::chunked::types::arena_gid_stride() as i64;
        let mut raw = vec![0i64; GIDS_PER_HEAD * n_kv_head];
        // k_gid_pal(0,0) is slot 0; k_gid_pal(0,1) is slot 2 (palette*2+0).
        raw[0] = stride * 0 + 1; // arena 0, chunk 1
        raw[2] = stride * 1 + 2; // arena 1, chunk 2
        let gids = HeadGids::from_vec(raw.iter().map(|&r| ChunkGid::detached(r)).collect());
        let arena_info = vec![
            ResolvedArenaInfo {
                base_ptr: 0x1000,
                chunk_byte_stride: 256,
                k_format_tag: ArenaFormatTag::Q8_0,
                v_format_tag: ArenaFormatTag::Q8_0,
                chunk_capacity: u32::MAX,
            },
            ResolvedArenaInfo {
                base_ptr: 0x9000,
                chunk_byte_stride: 128,
                k_format_tag: ArenaFormatTag::Q4_0,
                v_format_tag: ArenaFormatTag::Q4_0,
                chunk_capacity: u32::MAX,
            },
        ];
        let mut dst = vec![0u8; rec];
        serialize_kv_heads(
            &mut dst,
            &gids,
            &[],
            &[],
            &[],
            &[],
            n_kv_head,
            head_dim,
            &arena_info,
        );
        // k_ptr[0] = 0x1000 + 1*256 = 0x1100; k_ptr[1] = 0x9000 + 2*128 = 0x9100.
        let kptr0 = u64::from_le_bytes(dst[2..10].try_into().unwrap());
        let kptr1 = u64::from_le_bytes(dst[10..18].try_into().unwrap());
        assert_eq!(kptr0, 0x1000 + 256);
        assert_eq!(kptr1, 0x9000 + 2 * 128);
    }
}
