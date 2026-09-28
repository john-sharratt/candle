//! Fixed-stride arenas for per-sequence recurrent state.
//!
//! A DeltaNet layer's state — the `s` accumulator and its conv tail — is a
//! fixed-size block whose size is model geometry: 3.12 MiB on Flash-Next
//! (48 V heads × 128², plus a 120 KiB tail), 2.09 MiB on Qwen3.5. Each sequence
//! holds two per recurrent layer, the live state and the half a wave writes.
//!
//! # Why an arena
//!
//! An arena is one region cut into slots of one **stride**, shared by every
//! sequence of that geometry, and a slot is the unit a sequence holds — the same
//! shape as a KV band arena or a `KvHead` record arena. A slot has an address, its
//! arena has a position in the span, and the set of arenas per stride is
//! enumerable: the census a compaction plans from. A region held whole by one
//! sequence has none of that — its position belongs to whichever sequence claimed
//! it, and only that sequence's end can give it back.
//!
//! # Why these arenas are not the KV backing's
//!
//! `KvHead` records live in the KV backing's gid pool. Recurrent state cannot:
//! a model's stores outlive every session (`qwen35::batched` keeps them in a
//! model-owned map), while a KV backing and the arena storage behind it belong to a
//! session and go with it. So the arenas here are device-global, keyed by
//! `(device, stride)`, and each one holds its region as a [`SpanRegion`] — a
//! reservation region claimed between forwards and returned on drop.
//!
//! # Stride and capacity
//!
//! The stride is the block size rounded up to [`STATE_ALIGN`], exactly — there is
//! no ladder, because pools are looked up by stride and nothing needs the set of
//! strides to be finite. An arena holds `REGION_BYTES / stride` slots: five on
//! Flash-Next (2.6 % of the region unused), seven on Qwen3.5 (8.4 %). A block larger
//! than one region is refused, as is a geometry with an empty half — there is no
//! address to give a buffer of no bytes that is not some other buffer's.
//!
//! # Lifetime
//!
//! A [`StateSlot`] is RAII. Dropping it returns the slot, and the drop that frees an
//! arena's last slot releases the arena's region back to the span — so the ground a
//! geometry holds tracks the number of live sequences of that geometry, without an
//! eviction path anyone has to remember to call.

// The pool below claims reservation regions, which only exist with the feature.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use candle::Result;

#[cfg(feature = "cuda")]
use std::collections::hash_map::Entry;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
use candle::{Device, DeviceLocation};

#[cfg(feature = "cuda")]
use super::region_pool::{span_region_refusal, SpanClaims, SpanRegion, REGION_BYTES};

/// Alignment of every slot base: what a fresh CUDA allocation guarantees and what
/// the recurrent kernels' vectorised loads assume of a base pointer.
pub const STATE_ALIGN: usize = 256;

/// The slot stride for a block of `bytes`.
pub fn state_stride(bytes: usize) -> usize {
    bytes.next_multiple_of(STATE_ALIGN)
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
    /// Free slot indices, highest first, so `pop` hands out the lowest.
    free: Vec<u32>,
}

/// Every arena of one stride on one device: the bookkeeping, with no device in it.
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
        if stride == 0 || !stride.is_multiple_of(STATE_ALIGN) {
            candle::bail!(
                "state arena: stride {stride} B is not a positive multiple of {STATE_ALIGN}"
            );
        }
        let capacity = region_bytes / stride;
        if capacity == 0 {
            candle::bail!(
                "state arena: a {stride} B recurrent state exceeds the {region_bytes} B \
                 region, so no arena can hold one"
            );
        }
        Ok(Self {
            stride,
            capacity: capacity as u32,
            arenas: Vec::new(),
        })
    }

    /// Slots one arena holds.
    #[cfg(test)]
    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Regions these arenas hold.
    pub(crate) fn regions(&self) -> usize {
        self.arenas.len()
    }

    /// Take `region` as a new arena, every slot free.
    pub(crate) fn adopt(&mut self, region: R) {
        let at = self
            .arenas
            .partition_point(|a| a.region.rank() < region.rank());
        self.arenas.insert(
            at,
            Arena {
                region,
                free: (0..self.capacity).rev().collect(),
            },
        );
    }

    /// A free slot — in the arena lowest in the span that has one, at the lowest
    /// index in it — as `(arena base, slot index, slot address)`.
    pub(crate) fn take(&mut self) -> Option<(u64, u32, u64)> {
        let arena = self.arenas.iter_mut().find(|a| !a.free.is_empty())?;
        let index = arena.free.pop()?;
        let base = arena.region.base();
        Some((base, index, base + index as u64 * self.stride as u64))
    }

    /// Return slot `index` of the arena at `base`. When that frees the arena's last
    /// slot the arena is removed and its region handed back, for the caller to drop.
    ///
    /// # Panics
    ///
    /// On a slot this set never handed out, or one returned twice. Either means two
    /// holders believe they own one block of recurrent state, which is two sequences
    /// writing each other's memory — nothing to carry on from.
    pub(crate) fn give_back(&mut self, base: u64, index: u32) -> Option<R> {
        let pos = self
            .arenas
            .iter()
            .position(|a| a.region.base() == base)
            .unwrap_or_else(|| {
                panic!("state arena: slot {index} returned to arena {base:#x}, which is not held")
            });
        let arena = &mut self.arenas[pos];
        assert!(
            index < self.capacity && !arena.free.contains(&index),
            "state arena: slot {index} of arena {base:#x} returned twice or never handed out"
        );
        // Kept highest-first so the next `take` hands out the lowest index.
        let at = arena.free.partition_point(|&f| f > index);
        arena.free.insert(at, index);
        if arena.free.len() == self.capacity as usize {
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
type Pools = HashMap<(usize, usize), StrideArenas<SpanRegion>>;

#[cfg(feature = "cuda")]
static POOLS: OnceLock<Mutex<Pools>> = OnceLock::new();

#[cfg(feature = "cuda")]
fn pools() -> &'static Mutex<Pools> {
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// One slot of a state arena, held for as long as the state lives in it.
///
/// Dropping it returns the slot, and releases the arena's region if it was the last
/// one in use. The holder must not keep a pointer into the slot past that — the
/// slot is handed to the next sequence of the same geometry.
///
/// **Returned on the host, while kernels may still be reading it.** Two cases, both
/// sound:
///
/// - **The slot goes to another sequence of the geometry.** Every reader and every
///   next tenant of a state slot works on the device's primary stream, so the next
///   tenant's zeroing and kernels queue behind whatever was still in flight. A
///   reader on a second stream would need a fence before the slot is dropped.
/// - **The slot empties its arena, and the region goes back to the span.** The next
///   tenant there can be anything — a KV arena, the weight side — on any stream. The
///   region pool covers that exactly as it does for a KV arena: a released region is
///   stamped dirty, and its next claim synchronises the device before zeroing it.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct StateSlot {
    gpu: usize,
    stride: usize,
    arena: u64,
    index: u32,
    ptr: u64,
}

#[cfg(feature = "cuda")]
impl StateSlot {
    /// Device address of the slot's first byte, [`STATE_ALIGN`]-aligned.
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Bytes the slot spans — what it costs the span.
    pub fn stride(&self) -> usize {
        self.stride
    }
}

#[cfg(feature = "cuda")]
impl Drop for StateSlot {
    fn drop(&mut self) {
        let emptied = {
            let mut map = pools().lock().unwrap_or_else(|e| e.into_inner());
            map.get_mut(&(self.gpu, self.stride))
                .and_then(|set| set.give_back(self.arena, self.index))
        };
        // Released outside this module's lock: the region's own drop takes the
        // region pool's.
        drop(emptied);
    }
}

/// `n` slots for blocks of `bytes` on `device`, from the lowest arenas that have
/// room, claiming regions for new arenas only when those run out.
///
/// **Between forwards.** A new arena is a region claim, which takes the arena window
/// ([`SpanClaims`]) and refuses inside a forward — the same rule every tenant of the
/// span keeps. The window is opened once, and only when a region is actually
/// needed, so a sequence whose slots all come from existing arenas never touches it.
///
/// All or nothing: on a refusal the slots already taken are returned before the
/// error is.
///
/// **This module's lock is never held across the window or a region claim.** Opening
/// the window can hand back a standing tier and quiesce the device, and a claim takes
/// the region pool's lock; every `StateSlot` drop on every thread takes this one. So
/// the free slots are taken under it, the lock is let go to buy a region, and it is
/// taken again only to adopt the region as an arena — which also means a slot drop
/// never waits on a device quiesce, and this lock is never taken inside the region
/// pool's.
#[cfg(feature = "cuda")]
pub fn claim_state_slots(
    device: &Device,
    bytes: usize,
    n: usize,
    tenant: &'static str,
) -> Result<Vec<StateSlot>> {
    let DeviceLocation::Cuda { gpu_id } = device.location() else {
        candle::bail!("state arena: the reservation is a CUDA allocation");
    };
    let stride = state_stride(bytes);
    let mut out: Vec<StateSlot> = Vec::with_capacity(n);
    let mut claims: Option<SpanClaims> = None;
    loop {
        {
            let mut map = pools().lock().unwrap_or_else(|e| e.into_inner());
            let set = match map.entry((gpu_id, stride)) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => e.insert(StrideArenas::new(stride, REGION_BYTES)?),
            };
            while out.len() < n {
                let Some((arena, index, ptr)) = set.take() else {
                    break;
                };
                out.push(StateSlot {
                    gpu: gpu_id,
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
            claims = Some(SpanClaims::open(device, tenant)?);
        }
        let claimed = claims.as_ref().expect("opened just above").claim()?;
        let Some(region) = claimed else {
            candle::bail!(
                "state arena: no region for {tenant} after {} of {n} slots of {stride} B — {}",
                out.len(),
                span_region_refusal(device),
            );
        };
        pools()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(&(gpu_id, stride))
            .expect("the stride's arenas are never removed")
            .adopt(region);
    }
}

/// Regions every state arena on `device` holds, for the span accounting.
#[cfg(feature = "cuda")]
pub fn state_arena_regions(device: &Device) -> usize {
    let DeviceLocation::Cuda { gpu_id } = device.location() else {
        return 0;
    };
    pools()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|((g, _), _)| *g == gpu_id)
        .map(|(_, set)| set.regions())
        .sum()
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

    /// Flash-Next's block: a 3 MiB `s` plus a 120 KiB conv tail, which is already
    /// 256-aligned, and the five of them a region holds.
    #[test]
    fn the_flash_next_block_packs_five_to_a_region() {
        let s = 48 * 128 * 128 * 4;
        let tail = 10240 * 3 * 4;
        assert_eq!((s, tail), (3_145_728, 122_880));
        let stride = state_stride(s + tail);
        assert_eq!(stride, 3_268_608);
        let set = StrideArenas::<Fake>::new(stride, REGION).unwrap();
        assert_eq!(set.capacity(), 5);
    }

    #[test]
    fn a_stride_is_rounded_to_the_alignment() {
        assert_eq!(state_stride(1), 256);
        assert_eq!(state_stride(256), 256);
        assert_eq!(state_stride(257), 512);
    }

    #[test]
    fn a_block_larger_than_a_region_is_refused() {
        assert!(StrideArenas::<Fake>::new(state_stride(REGION + 1), REGION).is_err());
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
}
