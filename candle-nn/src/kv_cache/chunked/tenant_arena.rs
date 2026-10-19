//! Fixed-stride slot arenas for the span's per-sequence tenants: recurrent state,
//! provenance gallery pages, QSA index pages and creep residuals.
//!
//! # Why an arena
//!
//! An arena is one region cut into slots of one **stride**, and a slot is the unit a
//! holder takes — the same shape as a KV band arena or a `KvHead` record arena. A
//! slot has an address, its arena has a position in the span, and the set of arenas
//! per stride is enumerable: the census a compaction plans from. A region held whole
//! by one holder has none of that — its position belongs to whichever holder claimed
//! it, and only that holder's end can give it back.
//!
//! # Dedicated per tenant
//!
//! Pools are keyed by `(device, tenant, stride)`, so two tenants never share an arena
//! even when their strides coincide. Each tenant's arenas are then its own to count
//! (`arena_regions`) and to pack, and one tenant's churn cannot scatter another's
//! slots across the span.
//!
//! # Why these arenas are not the KV backing's
//!
//! `KvHead` records live in the KV backing's gid pool. These tenants cannot: their
//! holders outlive every session (a model's recurrent and index maps, the scheduler's
//! gallery), while a KV backing and the arena storage behind it belong to a session
//! and go with it. So the arenas here are device-global, and each one holds its region
//! as a [`SpanRegion`] — a reservation region claimed between forwards and returned on
//! drop.
//!
//! # Stride and capacity
//!
//! The stride is the block size rounded up to [`SLOT_ALIGN`], exactly — there is no
//! ladder, because pools are looked up by stride and nothing needs the set of strides
//! to be finite. An arena holds `REGION_BYTES / stride` slots: four recurrent-state
//! slots on Flash-Next — `S`, its conv tail and its timeline, 3,859,968 B (8.0 % of
//! the region unused) — and 2,730 gallery pages of 6 KiB. A block
//! larger than one region is refused. The free set is a bitmap, so taking and returning
//! a slot stay cheap at thousands of slots per arena.
//!
//! # Lifetime
//!
//! An [`ArenaSlot`] is RAII. Dropping it returns the slot, and the drop that frees an
//! arena's last slot releases the arena's region back to the span — so the ground a
//! tenant holds tracks what it is holding, without an eviction path anyone has to
//! remember to call.

// The pool below claims reservation regions, which only exist with the feature.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use candle::Result;

#[cfg(feature = "cuda")]
use std::collections::hash_map::Entry;
#[cfg(feature = "cuda")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::result::{memcpy_dtod_async, memset_d8_async};
#[cfg(feature = "cuda")]
use candle::{DType, Device, DeviceLocation, LeaseAnchor, Shape, Tensor};

#[cfg(feature = "cuda")]
use super::region_pool::{
    regions_to_relocate, span_region_refusal, SpanClaims, SpanRegion, REGION_BYTES,
};

/// Alignment of every slot base: what a fresh CUDA allocation guarantees and what the
/// kernels' vectorised loads assume of a base pointer.
pub const SLOT_ALIGN: usize = 256;

/// The slot stride for a block of `bytes`.
pub fn slot_stride(bytes: usize) -> usize {
    bytes.next_multiple_of(SLOT_ALIGN)
}

/// What one slot of a block of `bytes` costs the span: its share of the region its
/// arena claims, the region's unused tail included.
///
/// Not [`slot_stride`]. An arena is a whole region and holds `REGION_BYTES / stride`
/// slots, so the tail the last slot leaves is ground the tenant holds as surely as
/// the slots — and priced at the stride, nobody paid for it. On Flash-Next that tail
/// is 8.0 % of every recurrent-state region: on WhiteWorld (98860c583, 2026-10-19) a
/// nine-member cohort's prefill scratch was priced at 1,192 MiB, claimed 81 regions
/// (1,296 MiB) as it swept, and its second window was refused two regions short.
#[cfg(feature = "cuda")]
pub fn slot_share_bytes(bytes: usize) -> usize {
    let per_arena = (REGION_BYTES / slot_stride(bytes)).max(1);
    REGION_BYTES.div_ceil(per_arena)
}

/// How many of the topmost arenas a pass can empty, given each arena's `live`
/// slots in span order and `capacity` slots per arena: the most `k` whose live
/// slots fit in the free slots of the arenas below them, within `max_moves`
/// (zero for none) — the first arena always whole. An arena already empty is
/// not counted; it has no region to give back here.
fn arenas_to_empty(live: &[usize], capacity: usize, max_moves: usize) -> usize {
    let mut moving = 0usize;
    let mut k = 0usize;
    for (i, &l) in live.iter().enumerate().rev() {
        // The destinations are the arenas below this one, less what the arenas
        // above already moved into them.
        let free_below: usize = live[..i].iter().map(|&b| capacity - b).sum();
        if l == 0 || moving + l > free_below {
            break;
        }
        if k > 0 && max_moves != 0 && moving + l > max_moves {
            break;
        }
        moving += l;
        k += 1;
    }
    k
}

/// Who a slot arena belongs to. Arenas are never shared across tenants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlotTenant {
    /// One DeltaNet layer state of one sequence — `s`, its conv tail and its
    /// timeline — or the prefill scratch beside it.
    RecurrentState,
    /// One page of the provenance gallery's resident signatures.
    Gallery,
    /// One page of a QSA index cache's block keys, or its open block.
    QsaIndex,
    /// One block of whole rows of a creep's residual, held between the steps of
    /// its sweep (`docs/wave_step.md` §3).
    CreepResidual,
}

impl SlotTenant {
    /// Every tenant, for the census that says which of them holds the span's
    /// ground. Exhaustive by construction — a new variant that is not added here
    /// goes missing from the accounting, and ground nothing can total is ground
    /// that goes missing.
    pub const ALL: [SlotTenant; 4] = [
        Self::RecurrentState,
        Self::Gallery,
        Self::QsaIndex,
        Self::CreepResidual,
    ];

    /// The tenant's name, for the arena window and for errors.
    pub fn label(self) -> &'static str {
        match self {
            Self::RecurrentState => "recurrent state",
            Self::Gallery => "the provenance gallery",
            Self::QsaIndex => "a QSA index cache",
            Self::CreepResidual => "a creep's residual",
        }
    }
}

/// What an arena stands on: a region with an address and a position in the span.
pub(crate) trait Ground {
    /// Address of the first byte.
    fn base(&self) -> u64;
    /// Position in the span, ascending with address.
    fn rank(&self) -> usize;
}

/// One region cut into slots of one stride.
struct Arena<R> {
    region: R,
    /// One bit per slot, set when the slot is free.
    free: Vec<u64>,
    /// Set bits in `free`.
    n_free: u32,
}

impl<R> Arena<R> {
    fn new(region: R, capacity: u32) -> Self {
        let words = (capacity as usize).div_ceil(64);
        let mut free = vec![u64::MAX; words];
        let tail = capacity as usize % 64;
        if tail != 0 {
            free[words - 1] = (1u64 << tail) - 1;
        }
        Self {
            region,
            free,
            n_free: capacity,
        }
    }

    fn is_free(&self, index: u32) -> bool {
        self.free[index as usize / 64] & (1u64 << (index % 64)) != 0
    }

    /// The lowest free slot, taken.
    fn take_lowest(&mut self) -> Option<u32> {
        let (w, word) = self.free.iter().enumerate().find(|(_, w)| **w != 0)?;
        let index = (w * 64) as u32 + word.trailing_zeros();
        self.claim(index);
        Some(index)
    }

    fn claim(&mut self, index: u32) {
        debug_assert!(self.is_free(index));
        self.free[index as usize / 64] &= !(1u64 << (index % 64));
        self.n_free -= 1;
    }

    fn release(&mut self, index: u32) {
        self.free[index as usize / 64] |= 1u64 << (index % 64);
        self.n_free += 1;
    }

    /// Occupied slot indices, ascending.
    fn occupied(&self, capacity: u32) -> Vec<u32> {
        (0..capacity).filter(|&i| !self.is_free(i)).collect()
    }
}

/// Every arena of one tenant and stride on one device: the bookkeeping, with no device
/// in it.
///
/// Kept apart from the global pool so its rules — lowest arena first, lowest slot
/// first, an emptied arena gives its region back — are testable on the host.
pub(crate) struct StrideArenas<R> {
    stride: usize,
    capacity: u32,
    /// Ascending by [`Ground::rank`].
    arenas: Vec<Arena<R>>,
}

impl<R: Ground> StrideArenas<R> {
    /// Arenas for `stride`-byte slots in `region_bytes`-byte regions. Refuses a
    /// stride that fits no slot in a region, and one that is not aligned.
    pub(crate) fn new(stride: usize, region_bytes: usize) -> Result<Self> {
        if stride == 0 || !stride.is_multiple_of(SLOT_ALIGN) {
            candle::bail!(
                "slot arena: stride {stride} B is not a positive multiple of {SLOT_ALIGN}"
            );
        }
        let capacity = region_bytes / stride;
        if capacity == 0 {
            candle::bail!(
                "slot arena: a {stride} B block exceeds the {region_bytes} B region, so no \
                 arena can hold one"
            );
        }
        Ok(Self {
            stride,
            capacity: capacity as u32,
            arenas: Vec::new(),
        })
    }

    /// Slots one arena holds.
    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Regions these arenas hold.
    pub(crate) fn regions(&self) -> usize {
        self.arenas.len()
    }

    /// The span positions of the regions these arenas hold.
    #[cfg(feature = "cuda")]
    pub(crate) fn ranks(&self) -> impl Iterator<Item = usize> + '_ {
        self.arenas.iter().map(|a| a.region.rank())
    }

    /// Slots held across these arenas.
    pub(crate) fn held(&self) -> usize {
        self.arenas
            .iter()
            .map(|a| (self.capacity - a.n_free) as usize)
            .sum()
    }

    /// Take `region` as a new arena, every slot free.
    pub(crate) fn adopt(&mut self, region: R) {
        let at = self
            .arenas
            .partition_point(|a| a.region.rank() < region.rank());
        self.arenas.insert(at, Arena::new(region, self.capacity));
    }

    /// A free slot — in the arena lowest in the span that has one, at the lowest
    /// index in it — as `(arena base, slot index, slot address)`.
    pub(crate) fn take(&mut self) -> Option<(u64, u32, u64)> {
        let arena = self.arenas.iter_mut().find(|a| a.n_free > 0)?;
        let index = arena.take_lowest()?;
        let base = arena.region.base();
        Some((base, index, base + index as u64 * self.stride as u64))
    }

    /// The moves that empty this stride's topmost arenas into free slots below
    /// them, each destination **already claimed**: its slot is taken off the free
    /// set here, so nothing else can be handed it before the copy lands. Answers
    /// `(source address, destination)` per move, the destination as `(arena base,
    /// slot index, slot address)`.
    ///
    /// **Whole arenas or nothing.** A region goes back only when its arena's last
    /// slot does, so a move that fills a hole below without finishing an arena
    /// above frees nothing and is pure copy. The two-cursor walk planned exactly
    /// those: on blackworld (2026-10-18) a pass every ~4 s moved 71–119 recurrent
    /// states — up to ~460 MB of device copies — with the region count unchanged.
    /// So the pass takes, from the top, the most arenas whose live slots fit in
    /// the free slots of the arenas left below them, and moves only those slots,
    /// highest first into the lowest free slot. Within `max_moves` (zero for none),
    /// except that the first arena is always taken whole: a budget smaller than
    /// one arena would otherwise never empty any. The sources stay occupied — they
    /// are the holders' to give back once they have moved onto the destinations.
    pub(crate) fn plan_moves(&mut self, max_moves: usize) -> Vec<(u64, (u64, u32, u64))> {
        let cap = self.capacity as usize;
        let live: Vec<usize> = self
            .arenas
            .iter()
            .map(|a| cap - a.n_free as usize)
            .collect();
        let emptied = arenas_to_empty(&live, cap, max_moves);
        let keep = self.arenas.len() - emptied;
        let stride = self.stride as u64;
        let mut moves = Vec::new();
        let mut below = 0usize;
        for from in (keep..self.arenas.len()).rev() {
            let src_base = self.arenas[from].region.base();
            for index in self.arenas[from].occupied(self.capacity).into_iter().rev() {
                while below < keep && self.arenas[below].n_free == 0 {
                    below += 1;
                }
                let dst = &mut self.arenas[below];
                let slot = dst
                    .take_lowest()
                    .expect("arenas_to_empty counted a free slot here");
                let dst_base = dst.region.base();
                moves.push((
                    src_base + index as u64 * stride,
                    (dst_base, slot, dst_base + slot as u64 * stride),
                ));
            }
        }
        moves
    }

    /// Remove the arena at `base` if none of its slots is held, handing its region
    /// back for the caller to drop. For an arena a pass provisioned and then did not
    /// use.
    pub(crate) fn release_if_empty(&mut self, base: u64) -> Option<R> {
        let pos = self.arenas.iter().position(|a| a.region.base() == base)?;
        if self.arenas[pos].n_free == self.capacity {
            Some(self.arenas.remove(pos).region)
        } else {
            None
        }
    }

    /// Return slot `index` of the arena at `base`. When that frees the arena's last
    /// slot the arena is removed and its region handed back, for the caller to drop.
    ///
    /// # Panics
    ///
    /// On a slot this set never handed out, or one returned twice. Either means two
    /// holders believe they own one block, which is two holders writing each other's
    /// memory — nothing to carry on from.
    pub(crate) fn give_back(&mut self, base: u64, index: u32) -> Option<R> {
        let pos = self
            .arenas
            .iter()
            .position(|a| a.region.base() == base)
            .unwrap_or_else(|| {
                panic!("slot arena: slot {index} returned to arena {base:#x}, which is not held")
            });
        let arena = &mut self.arenas[pos];
        assert!(
            index < self.capacity && !arena.is_free(index),
            "slot arena: slot {index} of arena {base:#x} returned twice or never handed out"
        );
        arena.release(index);
        if arena.n_free == self.capacity {
            Some(self.arenas.remove(pos).region)
        } else {
            None
        }
    }
}

#[cfg(feature = "cuda")]
impl Ground for SpanRegion {
    fn base(&self) -> u64 {
        SpanRegion::base(self)
    }

    fn rank(&self) -> usize {
        self.index()
    }
}

#[cfg(feature = "cuda")]
type PoolKey = (usize, SlotTenant, usize);

#[cfg(feature = "cuda")]
type Pools = HashMap<PoolKey, StrideArenas<SpanRegion>>;

#[cfg(feature = "cuda")]
static POOLS: OnceLock<Mutex<Pools>> = OnceLock::new();

#[cfg(feature = "cuda")]
fn pools() -> &'static Mutex<Pools> {
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// One slot of a tenant's arena, held for as long as the block lives in it.
///
/// Dropping it returns the slot, and releases the arena's region if it was the last
/// one in use. The holder must not keep a pointer into the slot past that — the slot
/// is handed to the next holder of the same tenant and stride.
///
/// **Returned on the host, while kernels may still be reading it.** Two cases, both
/// sound:
///
/// - **The slot goes to another holder of the tenant.** Every reader and every next
///   tenant of a slot works on the device's primary stream, so the next holder's
///   writes queue behind whatever was still in flight. A reader on a second stream
///   would need a fence before the slot is dropped.
/// - **The slot empties its arena, and the region goes back to the span.** The next
///   tenant there can be anything — a KV arena, the weight side — on any stream. The
///   region pool covers that exactly as it does for a KV arena: a released region is
///   stamped dirty, and its next claim synchronises the device before zeroing it.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct ArenaSlot {
    gpu: usize,
    tenant: SlotTenant,
    stride: usize,
    arena: u64,
    index: u32,
    ptr: u64,
}

#[cfg(feature = "cuda")]
impl ArenaSlot {
    /// Device address of the slot's first byte, [`SLOT_ALIGN`]-aligned.
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Bytes the slot spans — what it costs the span.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The tenant whose arena this slot is in.
    pub fn tenant(&self) -> SlotTenant {
        self.tenant
    }

    /// A tensor of `shape` viewing this slot from `offset` bytes in, **anchored** to
    /// the slot: every view, clone and re-lease of it shares the anchor, so the slot
    /// cannot go back to its arena while any of them exists.
    ///
    /// Refuses a view that would run past the slot or start off [`SLOT_ALIGN`] — the
    /// guarantee a kernel's vectorised loads rely on for a base pointer.
    pub fn tensor(
        self: &Arc<Self>,
        offset: usize,
        dtype: DType,
        shape: impl Into<Shape>,
        device: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let bytes = shape.elem_count() * dtype.size_in_bytes();
        if !offset.is_multiple_of(SLOT_ALIGN) || offset + bytes > self.stride {
            candle::bail!(
                "slot arena: a {bytes} B view at +{offset} of a {} B slot of {}",
                self.stride,
                self.tenant.label()
            );
        }
        // SAFETY: the range lies inside this slot, which the anchor holds for as long
        // as the tensor or any view of it exists.
        unsafe {
            Tensor::from_anchored_cuda_ptr(
                self.ptr + offset as u64,
                dtype,
                shape,
                device,
                LeaseAnchor::new(Arc::clone(self)),
            )
        }
    }

    /// Zero `bytes` of the slot from its base, on the device's primary stream — for a
    /// holder whose buffer is genuinely read before it is written (hot-path invariant
    /// 6's exemption). A slot is recycled: it last held some other holder's bytes.
    pub fn zero(&self, bytes: usize, device: &Device) -> Result<()> {
        let Device::Cuda(cuda) = device else {
            candle::bail!("slot arena: a slot is zeroed on a CUDA device");
        };
        if bytes > self.stride {
            candle::bail!(
                "slot arena: zeroing {bytes} B of a {} B slot of {}",
                self.stride,
                self.tenant.label()
            );
        }
        // SAFETY: the range lies inside this slot, and the stream orders the fill
        // ahead of every reader.
        unsafe { memset_d8_async(self.ptr, 0, bytes, cuda.cuda_stream().cu_stream()) }.map_err(
            |e| candle::Error::Msg(format!("zeroing a slot of {}: {e}", self.tenant.label())),
        )
    }

    /// Copy `bytes` of this slot into `dst`, on the device's primary stream — the
    /// move a compaction pass applies for a holder whose slot carries raw bytes
    /// rather than a tensor (a gallery page).
    ///
    /// The source may be dropped on the host as soon as this returns even though
    /// the copy is still queued: the slot's next tenant works on the same stream,
    /// so its writes are ordered behind this, and a slot that empties its arena
    /// releases a region the pool stamps dirty and synchronises before reissuing.
    pub fn copy_into(&self, dst: &ArenaSlot, bytes: usize, device: &Device) -> Result<()> {
        let Device::Cuda(cuda) = device else {
            candle::bail!("slot arena: a slot is copied on a CUDA device");
        };
        if bytes > self.stride || bytes > dst.stride {
            candle::bail!(
                "slot arena: copying {bytes} B between a {} B and a {} B slot of {}",
                self.stride,
                dst.stride,
                self.tenant.label()
            );
        }
        if self.ptr == dst.ptr {
            candle::bail!("slot arena: a slot cannot be copied onto itself");
        }
        // SAFETY: both ranges are `bytes` inside distinct live slots — the source
        // held by the caller, the destination claimed for this move — so they do
        // not overlap.
        unsafe { memcpy_dtod_async(dst.ptr, self.ptr, bytes, cuda.cuda_stream().cu_stream()) }
            .map_err(|e| {
                candle::Error::Msg(format!("relocating a slot of {}: {e}", self.tenant.label()))
            })
    }
}

/// Move `t` onto its planned destination if the slot it views is the source of a
/// move in `moves`, taking that destination out of the map; answers whether it
/// moved.
///
/// For a holder whose slot is viewed as a tensor — a QSA key page. One device copy
/// of the tensor's own bytes, then `t` is rebuilt as an anchored view of the
/// destination, so every later reader resolves the new address and the old slot
/// goes back when its last view drops.
///
/// A tensor on a device with no reservation, or whose slot is not a planned
/// source, is left exactly as it was.
#[cfg(feature = "cuda")]
pub fn relocate_tensor(t: &mut Tensor, moves: &mut HashMap<u64, ArenaSlot>) -> Result<bool> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let device = t.device().clone();
    let Device::Cuda(cuda) = &device else {
        return Ok(false);
    };
    // **A holder that is not dense is not relocatable by this.** The copy below
    // moves `elem_count × size` contiguous bytes and the tensor is rebuilt as a
    // dense view of the destination, so a strided or offset holder would come back
    // with different values under the same shape — silent, and exactly the class of
    // corruption a compaction must never introduce. Refused rather than tolerated:
    // every tenant's buffers are dense by construction, so this firing means a
    // holder changed shape, not that a fallback is wanted.
    if !t.layout().is_contiguous() {
        candle::bail!(
            "slot arena: a {:?} holder is not contiguous, so it cannot be relocated",
            t.dims()
        );
    }
    let (storage, layout) = t.storage_and_layout();
    let candle::Storage::Cuda(c) = &*storage else {
        return Ok(false);
    };
    let stream = cuda.cuda_stream();
    // The base address through the storage's own slice variant rather than a fixed
    // element type: the slot arenas are dtype-agnostic, and a holder is free to be
    // whatever its kernels read. Reading an F16 holder through an `f32` slice would
    // both mis-scale the start offset and refuse outright.
    let at = {
        use candle::cuda_backend::CudaStorageSlice as S;
        let start = layout.start_offset();
        macro_rules! base_of {
            ($s:expr) => {{
                let slice = $s.slice(start..);
                let (ptr, _guard) = slice.device_ptr(&stream);
                ptr
            }};
        }
        match &c.slice {
            S::U8(s) => base_of!(s),
            S::U32(s) => base_of!(s),
            S::I64(s) => base_of!(s),
            S::BF16(s) => base_of!(s),
            S::F16(s) => base_of!(s),
            S::F32(s) => base_of!(s),
            S::F64(s) => base_of!(s),
            S::F8E4M3(s) => base_of!(s),
            // The tombstone a leased storage's drop leaves behind, and never
            // observable from a live tensor.
            _ => return Ok(false),
        }
    };
    drop(storage);
    let Some(dst) = moves.remove(&at) else {
        return Ok(false);
    };
    let dims = t.dims().to_vec();
    let bytes = t.elem_count() * t.dtype().size_in_bytes();
    // SAFETY: `at` is this tensor's own base inside a live slot of `bytes`, and
    // `dst` is a distinct slot claimed for this move.
    unsafe { memcpy_dtod_async(dst.ptr(), at, bytes, stream.cu_stream()) }
        .map_err(|e| candle::Error::Msg(format!("relocating a tenant tensor: {e}")))?;
    let dtype = t.dtype();
    *t = Arc::new(dst).tensor(0, dtype, dims, &device)?;
    Ok(true)
}

#[cfg(feature = "cuda")]
impl Drop for ArenaSlot {
    fn drop(&mut self) {
        let emptied = {
            let key = (self.gpu, self.tenant, self.stride);
            let mut map = pools().lock().unwrap_or_else(|e| e.into_inner());
            let region = map
                .get_mut(&key)
                .and_then(|set| set.give_back(self.arena, self.index));
            // **A pool that holds no arenas is removed, not left behind.** A stride
            // is not a fixed set: a creep residual's follows its row's width and
            // dtype, so a long run meets a new one whenever a session's
            // activation width does, and entries that are never removed grow
            // without bound. They also skew the census
            // — `pools` is meant to say how many strides a tenant is spread over,
            // and an emptied entry is a stride it no longer occupies.
            if map.get(&key).is_some_and(|set| set.regions() == 0) {
                map.remove(&key);
            }
            region
        };
        // Released outside this module's lock: the region's own drop takes the
        // region pool's.
        drop(emptied);
    }
}

/// `n` slots of `tenant`'s arenas for blocks of `bytes` on `device`, from the lowest
/// arenas that have room, claiming regions for new arenas only when those run out.
///
/// **Between forwards.** A new arena is a region claim, which takes the arena window
/// ([`SpanClaims`]) and refuses inside a forward — the same rule every tenant of the
/// span keeps. The window is opened once, and only when a region is actually
/// needed, so a holder whose slots all come from existing arenas never touches it.
///
/// All or nothing: on a refusal the slots already taken are returned before the
/// error is.
///
/// **This module's lock is never held across the window or a region claim.** Opening
/// the window can hand back a standing tier and quiesce the device, and a claim takes
/// the region pool's lock; every `ArenaSlot` drop on every thread takes this one. So
/// the free slots are taken under it, the lock is let go to buy a region, and it is
/// taken again only to adopt the region as an arena — which also means a slot drop
/// never waits on a device quiesce, and this lock is never taken inside the region
/// pool's.
#[cfg(feature = "cuda")]
pub fn claim_arena_slots(
    device: &Device,
    tenant: SlotTenant,
    bytes: usize,
    n: usize,
) -> Result<Vec<ArenaSlot>> {
    let DeviceLocation::Cuda { gpu_id } = device.location() else {
        candle::bail!("slot arena: the reservation is a CUDA allocation");
    };
    let stride = slot_stride(bytes);
    let key = (gpu_id, tenant, stride);
    let mut out: Vec<ArenaSlot> = Vec::with_capacity(n);
    let mut claims: Option<SpanClaims> = None;
    loop {
        {
            let mut map = pools().lock().unwrap_or_else(|e| e.into_inner());
            let set = match map.entry(key) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => e.insert(StrideArenas::new(stride, REGION_BYTES)?),
            };
            while out.len() < n {
                let Some((arena, index, ptr)) = set.take() else {
                    break;
                };
                out.push(ArenaSlot {
                    gpu: gpu_id,
                    tenant,
                    stride,
                    arena,
                    index,
                    ptr,
                });
            }
        }
        if out.len() == n {
            return Ok(out);
        }
        if claims.is_none() {
            claims = Some(SpanClaims::open(device, tenant.label())?);
        }
        let claimed = claims.as_ref().expect("opened just above").claim()?;
        let Some(region) = claimed else {
            candle::bail!(
                "slot arena: no region for {} after {} of {n} slots of {stride} B — {}",
                tenant.label(),
                out.len(),
                span_region_refusal(device),
            );
        };
        pools()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(&key)
            .expect("a pool's arenas are never removed")
            .adopt(region);
    }
}

/// One planned relocation of a block: the address it is at now, and the slot it is
/// going to — already claimed, so nothing else can be handed it.
///
/// The holder of `src` copies the block and moves onto `dst`; dropping a `SlotMove`
/// unapplied returns `dst` to its arena, which is the right outcome for a source no
/// holder was found for.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct SlotMove {
    pub src: u64,
    pub dst: ArenaSlot,
}

/// Plan a compaction of `tenant`'s arenas for blocks of `bytes` on `device`: the
/// moves that pack their live slots toward the low end of the span, destinations
/// claimed.
///
/// **Fresh low arenas only for the arenas that hold the frontier**
/// ([`regions_to_relocate`]): this pool's topmost regions, above every other tenant's
/// live region, one per hole below them. The region free list is lowest-index first,
/// so new arenas land in the lowest holes, rank lowest among this pool's arenas, and
/// the walk fills them first; the arenas at the top empty and their regions go back
/// when their last holders move off, lowering the frontier. An arena under another
/// tenant's region is left where it is — emptying it would only move a hole. An arena
/// the walk puts nothing in is released before this returns. Claiming takes the arena
/// window, so this runs **between forwards**, like every other claim on the span.
///
/// `max_moves` (zero for none) bounds the pass; a clipped pass leaves the arenas
/// strictly better packed, and the next one resumes from there.
#[cfg(feature = "cuda")]
pub fn plan_slot_moves(
    device: &Device,
    tenant: SlotTenant,
    bytes: usize,
    max_moves: usize,
) -> Result<Vec<SlotMove>> {
    let DeviceLocation::Cuda { gpu_id } = device.location() else {
        candle::bail!("slot arena: the reservation is a CUDA allocation");
    };
    let stride = slot_stride(bytes);
    let key = (gpu_id, tenant, stride);
    let (held, capacity) = pools()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&key)
        .map_or((HashSet::new(), 1), |set| {
            (
                set.ranks().collect::<HashSet<usize>>(),
                set.capacity() as usize,
            )
        });
    if held.is_empty() {
        return Ok(Vec::new());
    }
    // **As many low arenas as there are frontier regions to empty, and no more.** Each
    // fresh arena lets the walk empty one arena at the top; a pass that provisions one
    // lowers the frontier by at most a region, slower than holders churn (measured:
    // Flash-Next held at 84 % with one). But only this pool's regions that hold the
    // frontier are worth emptying — below another tenant's, an emptied arena is a hole
    // the next pass refills from the next arena down, and Flash-Next's passes moved 912
    // slots each that way with the region count unchanged. The pass can also move only
    // `max_moves` blocks, which fill `max_moves / capacity` arenas.
    //
    // Gathered under this module's lock and asked of the region pool outside it, for
    // the reason on `claim_arena_slots`.
    let relocate = regions_to_relocate(gpu_id, &held).unwrap_or_default();
    let want = if max_moves == 0 {
        relocate.len()
    } else {
        max_moves.div_ceil(capacity).min(relocate.len())
    };
    // A fresh arena has to land below the lowest region it is meant to empty; one the
    // free list placed at or above it (another claim took the hole) receives nothing
    // and goes back below.
    let cutoff = relocate.get(want.saturating_sub(1)).copied().unwrap_or(0);
    let mut fresh: Vec<SpanRegion> = Vec::new();
    if want > 0 {
        let claims = SpanClaims::open(device, tenant.label())?;
        while fresh.len() < want {
            let Some(region) = claims.claim()? else {
                break;
            };
            if region.index() >= cutoff {
                drop(region);
                break;
            }
            fresh.push(region);
        }
    }
    let (moves, unused) = {
        let mut map = pools().lock().unwrap_or_else(|e| e.into_inner());
        let set = map
            .get_mut(&key)
            .expect("a pool's arenas are never removed");
        let bases: Vec<u64> = fresh
            .into_iter()
            .map(|region| {
                let base = region.base();
                set.adopt(region);
                base
            })
            .collect();
        let moves = set.plan_moves(max_moves);
        let unused: Vec<SpanRegion> = bases
            .into_iter()
            .filter_map(|b| set.release_if_empty(b))
            .collect();
        (moves, unused)
    };
    // Released outside the lock: the region's own drop takes the region pool's.
    drop(unused);
    Ok(moves
        .into_iter()
        .map(|(src, (arena, index, ptr))| SlotMove {
            src,
            dst: ArenaSlot {
                gpu: gpu_id,
                tenant,
                stride,
                arena,
                index,
                ptr,
            },
        })
        .collect())
}

/// The highest region `tenant`'s arenas on `device` stand on, every stride
/// together — whether it is the tenant holding the arena frontier, and so the one
/// a pass must move to lower it.
#[cfg(feature = "cuda")]
pub fn arena_top_rank(device: &Device, tenant: SlotTenant) -> Option<usize> {
    let DeviceLocation::Cuda { gpu_id } = device.location() else {
        return None;
    };
    pools()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|((g, t, _), _)| *g == gpu_id && *t == tenant)
        .filter_map(|(_, set)| set.ranks().max())
        .max()
}

/// Regions `tenant`'s arenas on `device` hold, every stride together — what that
/// tenant denies the rest of the span.
#[cfg(feature = "cuda")]
pub fn arena_regions(device: &Device, tenant: SlotTenant) -> usize {
    let DeviceLocation::Cuda { gpu_id } = device.location() else {
        return 0;
    };
    pools()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|((g, t, _), _)| *g == gpu_id && *t == tenant)
        .map(|(_, set)| set.regions())
        .sum()
}

/// Bytes of slots `tenant`'s arenas on `device` hold — what its holders are using,
/// which is less than [`arena_regions`] by every arena's free slots and unused tail.
#[cfg(feature = "cuda")]
pub fn arena_held_bytes(device: &Device, tenant: SlotTenant) -> usize {
    let DeviceLocation::Cuda { gpu_id } = device.location() else {
        return 0;
    };
    pools()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|((g, t, _), _)| *g == gpu_id && *t == tenant)
        .map(|((_, _, stride), set)| set.held() * stride)
        .sum()
}

/// One tenant's ground: the regions its arenas hold, what its holders actually
/// occupy, and how many pools (strides) it is spread over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenantArenas {
    pub tenant: SlotTenant,
    /// Whole regions the tenant's arenas stand on — what it denies the rest of
    /// the span.
    pub regions: usize,
    /// Bytes of slots actually held. The shortfall against `regions` is free
    /// slots plus every arena's unused tail.
    pub held_bytes: usize,
    /// Distinct strides. A tenant spread over several pools pays at least one
    /// region per pool however small its slots are, which no packing walk can
    /// recover — only giving it fewer strides can.
    pub pools: usize,
    /// One past the highest region the tenant holds — set against the pool's
    /// frontier, whether this tenant is the one standing on it.
    pub top: usize,
}

/// Every tenant's ground on `device`, in one pass under one lock.
///
/// **The question "which tenant holds the span" had no answer before this.** The
/// fragmentation probe reports the span tenants' regions as a single figure and
/// counts all of it as legitimately in use, so a tenant holding thirty
/// mostly-empty arenas and one holding thirty full ones read identically — and a
/// pass that reclaimed nothing looked the same as one that reclaimed everything.
/// Per-tenant is the granularity an optimisation can be aimed at or attributed
/// to.
///
/// Tenants with no arenas are included at zero, so the census always names every
/// tenant and a reader can tell "holds nothing" from "was never counted".
#[cfg(feature = "cuda")]
pub fn arena_census(device: &Device) -> Vec<TenantArenas> {
    let mut out: Vec<TenantArenas> = SlotTenant::ALL
        .iter()
        .map(|&tenant| TenantArenas {
            tenant,
            regions: 0,
            held_bytes: 0,
            pools: 0,
            top: 0,
        })
        .collect();
    let DeviceLocation::Cuda { gpu_id } = device.location() else {
        return out;
    };
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    for ((g, tenant, stride), set) in map.iter() {
        if *g != gpu_id {
            continue;
        }
        let Some(row) = out.iter_mut().find(|r| r.tenant == *tenant) else {
            continue;
        };
        row.regions += set.regions();
        row.held_bytes += set.held() * stride;
        row.pools += 1;
        row.top = set.ranks().map(|r| r + 1).fold(row.top, usize::max);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGION: usize = 16 * 1024 * 1024;

    /// A region of the fake span: rank `i`, based at `i` regions in.
    struct Fake(usize);

    impl Ground for Fake {
        fn base(&self) -> u64 {
            (self.0 * REGION) as u64
        }
        fn rank(&self) -> usize {
            self.0
        }
    }

    /// Flash-Next's recurrent block: a 3 MiB `s` plus a 120 KiB conv tail, which is
    /// already 256-aligned, and the five of them a region holds.
    #[test]
    fn the_flash_next_block_packs_five_to_a_region() {
        let s = 48 * 128 * 128 * 4;
        let tail = 10240 * 3 * 4;
        assert_eq!((s, tail), (3_145_728, 122_880));
        let stride = slot_stride(s + tail);
        assert_eq!(stride, 3_268_608);
        let set = StrideArenas::<Fake>::new(stride, REGION).unwrap();
        assert_eq!(set.capacity(), 5);
    }

    /// With its timeline the block is 3,859,968 B and four fit a region, so each
    /// slot claims a quarter of it — 4 MiB, not the 3.68 MiB stride — and a store
    /// priced at the stride ran 8 % short of the regions it claimed.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_slot_claims_its_share_of_the_region() {
        let block = 3_859_968;
        assert_eq!(slot_stride(block), block);
        assert_eq!(slot_share_bytes(block), REGION / 4);
        // A region the stride divides exactly leaves no tail to share.
        assert_eq!(slot_share_bytes(REGION / 8), REGION / 8);
    }

    /// A gallery page — 32 tokens of 24 folded words — is 6 KiB, and a region holds
    /// 2,730 of them: well past one bitmap word, which the free set has to handle.
    #[test]
    fn a_gallery_page_packs_2730_to_a_region() {
        let page = 32 * 24 * 8;
        assert_eq!(slot_stride(page), 6144);
        let mut set = StrideArenas::new(6144, REGION).unwrap();
        assert_eq!(set.capacity(), 2730);
        set.adopt(Fake(0));
        let slots: Vec<_> = (0..2730).map(|_| set.take().unwrap()).collect();
        assert_eq!(
            slots[2729].1, 2729,
            "the last slot, in the partial last word"
        );
        assert_eq!(set.take(), None, "all 2,730 taken, and no phantom 2,731st");
        assert_eq!(set.held(), 2730);
        assert!(set.give_back(slots[100].0, slots[100].1).is_none());
        assert_eq!(
            set.take(),
            Some((0, 100, 100 * 6144)),
            "the lowest free slot again"
        );
    }

    #[test]
    fn a_stride_is_rounded_to_the_alignment() {
        assert_eq!(slot_stride(1), 256);
        assert_eq!(slot_stride(256), 256);
        assert_eq!(slot_stride(257), 512);
    }

    #[test]
    fn a_block_larger_than_a_region_is_refused() {
        assert!(StrideArenas::<Fake>::new(slot_stride(REGION + 1), REGION).is_err());
        assert!(StrideArenas::<Fake>::new(REGION, REGION).is_ok());
    }

    /// Slots come from the arena lowest in the span first, lowest index first,
    /// whatever order the arenas were adopted in.
    #[test]
    fn slots_are_taken_lowest_arena_and_lowest_index_first() {
        let mut set = StrideArenas::new(REGION / 2, REGION).unwrap();
        set.adopt(Fake(7));
        set.adopt(Fake(3));
        let half = (REGION / 2) as u64;
        let r3 = (3 * REGION) as u64;
        let r7 = (7 * REGION) as u64;
        assert_eq!(set.take(), Some((r3, 0, r3)));
        assert_eq!(set.take(), Some((r3, 1, r3 + half)));
        assert_eq!(set.take(), Some((r7, 0, r7)));
        assert_eq!(set.take(), Some((r7, 1, r7 + half)));
        assert_eq!(set.take(), None, "both arenas full");

        // A slot given back is the next one handed out.
        assert!(set.give_back(r3, 1).is_none());
        assert_eq!(set.take(), Some((r3, 1, r3 + half)));
    }

    /// The give-back that empties an arena hands its region over, and the arena is
    /// no longer offered.
    #[test]
    fn an_emptied_arena_gives_back_its_region() {
        let mut set = StrideArenas::new(REGION / 2, REGION).unwrap();
        set.adopt(Fake(2));
        let a = set.take().unwrap();
        let b = set.take().unwrap();
        assert!(set.give_back(a.0, a.1).is_none(), "one slot still held");
        let region = set
            .give_back(b.0, b.1)
            .expect("the last slot frees the arena");
        assert_eq!(region.rank(), 2);
        assert_eq!(set.regions(), 0);
        assert_eq!(set.take(), None);
    }

    /// **The walk packs a pool's live slots into the lowest arena, claiming each
    /// destination as it plans it.** Two half-empty arenas, the lower one fresh: the
    /// upper arena's live slots move into the lower one, lowest slots first, and those
    /// slots are no longer offered to anyone else.
    #[test]
    fn a_pass_packs_live_slots_into_the_lowest_arena() {
        let quarter = REGION / 4;
        let mut set = StrideArenas::new(quarter, REGION).unwrap();
        set.adopt(Fake(9));
        let r9 = (9 * REGION) as u64;
        // Slots 0, 1 and 3 live, slot 2 given back: a hole inside the arena.
        let held: Vec<_> = (0..4).map(|_| set.take().unwrap()).collect();
        assert!(set.give_back(held[2].0, held[2].1).is_none());
        // A fresh arena below it, as the pass provisions one.
        set.adopt(Fake(1));
        let r1 = (REGION) as u64;

        let moves = set.plan_moves(0);
        let q = quarter as u64;
        assert_eq!(
            moves,
            vec![
                (r9 + 3 * q, (r1, 0, r1)),
                (r9 + q, (r1, 1, r1 + q)),
                (r9, (r1, 2, r1 + 2 * q)),
            ],
            "the highest live slot goes first, into the lowest free slot",
        );
        assert_eq!(
            set.take(),
            Some((r1, 3, r1 + 3 * q)),
            "the destinations are claimed; only the slot nobody was moved into is free",
        );
    }

    /// A pass over arenas that are already a gapless prefix plans nothing.
    #[test]
    fn a_packed_pool_plans_no_moves() {
        let mut set = StrideArenas::new(REGION / 2, REGION).unwrap();
        set.adopt(Fake(0));
        set.adopt(Fake(1));
        let _a = set.take().unwrap();
        let _b = set.take().unwrap();
        let _c = set.take().unwrap();
        assert!(set.plan_moves(0).is_empty());
    }

    /// `max_moves` stops the pass at a whole arena, never inside one — and the
    /// first arena is taken whole even past the budget, or an arena larger than
    /// it would never empty.
    #[test]
    fn max_moves_clips_the_pass_at_whole_arenas() {
        let live = [0, 2, 2];
        assert_eq!(
            arenas_to_empty(&live, 4, 0),
            2,
            "unbudgeted: both upper arenas"
        );
        assert_eq!(
            arenas_to_empty(&live, 4, 3),
            1,
            "the second would pass the budget"
        );
        assert_eq!(arenas_to_empty(&live, 4, 1), 1, "the first is always whole");

        let quarter = REGION / 4;
        let mut set = StrideArenas::new(quarter, REGION).unwrap();
        set.adopt(Fake(5));
        let _held: Vec<_> = (0..4).map(|_| set.take().unwrap()).collect();
        set.adopt(Fake(0));
        assert_eq!(set.plan_moves(2).len(), 4, "one arena, whole");
    }

    /// **A pass that cannot empty an arena plans nothing.** Blackworld's shape
    /// (2026-10-18): every arena one hole short of full. The two-cursor walk
    /// moved the top arena's states into the holes below without ever emptying
    /// one — 71–119 moves a pass, every ~4 s, regions unchanged. Here the top
    /// arena's three states fit the three holes below, so it empties; with only
    /// two holes below it, nothing moves.
    #[test]
    fn a_pass_that_cannot_empty_an_arena_moves_nothing() {
        assert_eq!(
            arenas_to_empty(&[3, 3, 3, 3], 4, 0),
            1,
            "three holes below hold it"
        );
        assert_eq!(
            arenas_to_empty(&[3, 3, 3], 4, 0),
            0,
            "two holes below do not"
        );
        assert_eq!(arenas_to_empty(&[4, 4, 4], 4, 0), 0, "nowhere to go");
        assert_eq!(arenas_to_empty(&[], 4, 0), 0);
    }

    /// An arena the pass provisioned and put nothing in is handed back; one that took
    /// a slot is not.
    #[test]
    fn an_unused_provisioned_arena_is_released() {
        let mut set = StrideArenas::new(REGION / 2, REGION).unwrap();
        set.adopt(Fake(3));
        let r3 = (3 * REGION) as u64;
        assert_eq!(set.release_if_empty(r3).map(|r| r.rank()), Some(3));
        assert_eq!(set.regions(), 0);

        set.adopt(Fake(4));
        let r4 = (4 * REGION) as u64;
        let _held = set.take().unwrap();
        assert!(set.release_if_empty(r4).is_none());
    }

    #[test]
    #[should_panic(expected = "returned twice")]
    fn a_slot_returned_twice_is_refused() {
        let mut set = StrideArenas::new(REGION / 4, REGION).unwrap();
        set.adopt(Fake(0));
        let a = set.take().unwrap();
        let _b = set.take().unwrap();
        set.give_back(a.0, a.1);
        set.give_back(a.0, a.1);
    }

    /// **The census names every tenant, including the ones holding nothing.**
    /// A tenant missing from the list is indistinguishable from one at zero, and
    /// ground nothing can total is ground that goes missing.
    #[test]
    fn the_census_covers_every_tenant() {
        use std::collections::HashSet;
        let named: HashSet<_> = SlotTenant::ALL.iter().copied().collect();
        assert_eq!(named.len(), SlotTenant::ALL.len(), "no duplicates in ALL");
        // Off CUDA the census still answers, one row per tenant at zero, rather
        // than an empty vec a reader could mistake for "not measured". The census
        // itself is CUDA-only (it reads the region pools), so this half of the
        // test is too — the `ALL` check above holds on any build.
        #[cfg(feature = "cuda")]
        {
            let rows = arena_census(&candle::Device::Cpu);
            assert_eq!(rows.len(), SlotTenant::ALL.len());
            for row in &rows {
                assert!(named.contains(&row.tenant));
                assert_eq!((row.regions, row.held_bytes, row.pools), (0, 0, 0));
            }
        }
    }

    /// Every tenant has its own name for the arena window and for errors.
    #[test]
    fn every_tenant_is_named() {
        use std::collections::HashSet;
        let labels: HashSet<_> = SlotTenant::ALL.into_iter().map(SlotTenant::label).collect();
        assert_eq!(labels.len(), SlotTenant::ALL.len());
    }
}
