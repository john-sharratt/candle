//! Region-tier slabs for the per-`(layer, batch slot)` decode slot-state
//! buffers.
//!
//! # Why these are arena-managed and not bump-side
//!
//! The decode slot-state buffer is **cached across waves** —
//! `sync_decode_gpu_chunks` returns `Reuse` with the existing device pointer
//! whenever the chunk count still matches — so a per-wave cursor reset would
//! recycle it under a live sequence. Anything outliving a wave is
//! arena-managed (`docs/archived/arena_unification.md` principle 2's corollary, audit
//! A13).
//!
//! # What this replaces
//!
//! Every structural mutation of a sequence — including `push_chunk`, which
//! fires each time a sequence crosses a 32-token boundary — used to drop the
//! buffer and rebuild it: a full `stream.synchronize()`, a `cuMemFreeHost`, a
//! `cuMemFree`, then a fresh `cuMemHostAlloc` and `stream.alloc`. At batch 64
//! across 48 layers that is **≈3,000 alloc/free/sync cycles per 32 decoded
//! tokens**, with the syncs serialising the pipeline (audit A13).
//!
//! Here a buffer instead holds a slot from a doubling class family, and growth
//! is a **promotion**: claim the next class up, copy, release the old slot. No
//! allocator call, and no sync.
//!
//! # Promotion copies nothing
//!
//! A wider slot is claimed and the old one released, with no data moved
//! between them: the only caller of `resize` is `rebuild_decode`, which
//! rewrites every entry immediately afterwards. The buffer's two sections —
//! `[ slice headers | records ]` — mean the records half *moves* whenever the
//! entry count changes, so a forward copy would have produced bytes nobody
//! reads. A `copy_slot` helper was written for this and then deleted when the
//! caller made it unreachable.
//!
//! # Why releasing needs no fence
//!
//! Every copy into these slots runs on the device's compute stream, or is
//! recorded into a wave's segment and launched there in issue order, so every
//! copy into and out of them is FIFO-ordered against every other. A released
//! slot handed straight back out cannot be written before the copy that
//! drained it has run, because that write is issued behind it. This is the
//! property that lets the `stream.synchronize()` disappear rather than merely
//! move.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::Held;
use candle::{CudaDevice, Result};

use super::types::TARGET_ARENA_BYTES;

/// Slot sizes, ascending, each twice the last.
///
/// Sized in **entries**, where one entry is a 16 B slice header plus a
/// `KvHead` record per head — 688 B at the production geometry (4 KV heads,
/// head_dim 128), **not** the 16 B the header alone suggests. That factor of 43
/// is the whole reason this ladder reaches 16 MiB: computed from the header the
/// old 1 MiB top looked like 2 M tokens, and was really **48.7 K** — a hard
/// ceiling on an unbounded-context engine, since past the top rung `claim`
/// errors and the forward fails (the `stream.alloc` this tier replaced had no
/// cap at all).
///
/// The top rung now holds ~24.4 K chunks ≈ **781 K tokens** for one sequence in
/// one layer. Extending it is nearly free because a slab is bounded in *bytes*,
/// not slots — `SLOTS_PER_SLAB.min(TARGET_ARENA_BYTES / slot_bytes).max(1)` —
/// so every rung from 1 MiB up gets a 16 MiB slab holding 16, 8, 4, 2 or 1
/// slots. Rungs are claimed lazily, so a deployment that never runs deep never
/// allocates them.
///
/// Doubling rather than the KV ladder's fitted rungs because this buffer
/// *grows monotonically within a turn* — the number of promotions to reach
/// size N is what matters, and doubling makes it logarithmic.
/// Slots per slab.
///
/// A slab is sized by slot COUNT, not by bytes: 512 covers a wave's worth of
/// `(layer, batch slot)` pairs at the shapes this engine runs, while keeping
/// the small classes' slabs to a couple of MiB. Sizing every class to a whole
/// region instead put tens of MiB of mostly-empty slabs in front of KV on a
/// 16 GiB card, and the compaction that triggered cost more than the allocator
/// churn this tier exists to remove.
const SLOTS_PER_SLAB: usize = 512;

pub(crate) const SLOT_STATE_LADDER: [usize; 13] = [
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
];

/// A claimed slot: where it is, how wide, and which free list it returns to.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SlotStateSlot {
    /// Device address of the slot's first byte.
    pub ptr: u64,
    /// Rung of [`SLOT_STATE_LADDER`].
    class: usize,
    /// Index of the slab this slot was carved from.
    slab: usize,
    /// Index of the slot within that slab.
    index: usize,
}

impl SlotStateSlot {
    /// Bytes this slot can hold.
    pub(crate) fn capacity(&self) -> usize {
        SLOT_STATE_LADDER[self.class]
    }
}

/// The smallest rung that holds `bytes`, or `None` if it exceeds the ladder.
fn class_for(bytes: usize) -> Option<usize> {
    SLOT_STATE_LADDER.iter().position(|&b| b >= bytes)
}

/// The width of the rung that holds `bytes`.
///
/// Exposed so the *host* staging buffer can follow the same ladder as the
/// device slot: a capacity that grows logarithmically instead of on every
/// entry keeps the pinned-page alloc/free pair off the per-32-token path.
pub(crate) fn class_bytes_for(bytes: usize) -> Result<usize> {
    class_for(bytes)
        .map(|c| SLOT_STATE_LADDER[c])
        .ok_or_else(|| {
            candle::Error::Msg(format!(
                "slot-state buffer of {bytes} B exceeds the {} B top class",
                SLOT_STATE_LADDER[SLOT_STATE_LADDER.len() - 1]
            ))
        })
}

struct ClassPool {
    /// One entry per slab: the whole device allocation, held so it stays live;
    /// `None` for a slab given back, whose index the next slab reuses.
    slabs: Vec<Option<Held<u8>>>,
    /// Base device address of each slab, resolved once at creation.
    bases: Vec<u64>,
    /// Slots each slab holds, and how many of them are free.
    capacity: Vec<usize>,
    free_in: Vec<usize>,
    /// Free `(slab, index)` pairs, most recently released first.
    free: Vec<(usize, usize)>,
}

impl ClassPool {
    fn new() -> Self {
        Self {
            slabs: Vec::new(),
            bases: Vec::new(),
            capacity: Vec::new(),
            free_in: Vec::new(),
            free: Vec::new(),
        }
    }

    /// Install `slab` of `per_slab` slots at `base`, every slot free: in the
    /// index a given-back slab left, or a new one. Returns the index.
    fn add_slab(&mut self, slab: Held<u8>, base: u64, per_slab: usize) -> usize {
        let at = match self.slabs.iter().position(Option::is_none) {
            Some(at) => {
                self.slabs[at] = Some(slab);
                self.bases[at] = base;
                self.capacity[at] = per_slab;
                self.free_in[at] = per_slab;
                at
            }
            None => {
                self.slabs.push(Some(slab));
                self.bases.push(base);
                self.capacity.push(per_slab);
                self.free_in.push(per_slab);
                self.slabs.len() - 1
            }
        };
        self.free.extend((0..per_slab).rev().map(|i| (at, i)));
        at
    }

    /// A slot of `slab` came back. When that empties the slab and the class
    /// still has a slab's worth of free slots elsewhere, the slab is taken out
    /// of the pool and returned — its slots off the free list — for the caller
    /// to give back. The spare slab's worth is what keeps a 32-token boundary,
    /// where many pairs promote at once, from freeing and re-carving.
    fn returned(&mut self, slab: usize, index: usize) -> Option<Held<u8>> {
        self.free.push((slab, index));
        self.free_in[slab] += 1;
        if self.free_in[slab] < self.capacity[slab] {
            return None;
        }
        let elsewhere = self.free.len() - self.free_in[slab];
        if elsewhere < self.capacity[slab] {
            return None;
        }
        self.free.retain(|&(s, _)| s != slab);
        self.free_in[slab] = 0;
        self.capacity[slab] = 0;
        self.slabs[slab].take()
    }
}

/// Per-device slab pools, one set of class free lists per device ordinal.
struct Inner {
    classes: Vec<ClassPool>,
    /// Slots handed out and not yet returned, for the diagnostic.
    live: usize,
    /// Slabs allocated, for the diagnostic.
    slabs: usize,
    /// Bytes those slabs hold, for the diagnostic.
    reserved: usize,
}

static ARENAS: OnceLock<Mutex<HashMap<usize, Inner>>> = OnceLock::new();

fn arenas() -> &'static Mutex<HashMap<usize, Inner>> {
    ARENAS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Claim a slot of at least `bytes` on `device`.
///
/// Allocates a fresh slab only when the class has no free slot — the same
/// scarcity rule the KV classes follow. Errors if `bytes` exceeds the top
/// rung, which at 16 MiB and 688 B per entry means a single sequence's
/// slot-state in one layer has passed ~781 K tokens.
///
/// A slab is held ([`Held`]) — a range of the reservation's held ground, or a
/// pool allocation when the ground has no room — and is taken inside
/// [`CudaDevice::pause_capture`], behind the launches recorded so far. Only
/// then: a claim from a class with a free slot — every claim once the slabs
/// have settled — touches no device at all.
pub(crate) fn claim(device: &CudaDevice, bytes: usize) -> Result<SlotStateSlot> {
    let class = class_for(bytes).ok_or_else(|| {
        candle::Error::Msg(format!(
            "slot-state buffer of {bytes} B exceeds the {} B top class — a single \
             sequence's slot-state in one layer has outgrown the ladder",
            SLOT_STATE_LADDER[SLOT_STATE_LADDER.len() - 1]
        ))
    })?;
    let slot_bytes = SLOT_STATE_LADDER[class];
    let device_id = device.cuda_stream().context().ordinal();

    // Paused before the arena lock is taken, never under it: a pause launches
    // the recorded segment and frees what it retired, and nothing it frees may
    // find this lock held. So an empty class drops the lock, pauses, and looks
    // again — another thread may have refilled or drained it meanwhile.
    let mut paused = None;
    let mut map = loop {
        let map = arenas().lock().unwrap();
        let empty = map
            .get(&device_id)
            .is_none_or(|inner| inner.classes[class].free.is_empty());
        if !empty || paused.is_some() {
            break map;
        }
        drop(map);
        paused = Some(device.pause_capture()?);
    };
    let inner = map.entry(device_id).or_insert_with(|| Inner {
        classes: (0..SLOT_STATE_LADDER.len())
            .map(|_| ClassPool::new())
            .collect(),
        live: 0,
        slabs: 0,
        reserved: 0,
    });

    if inner.classes[class].free.is_empty() {
        // Carve a fresh slab.
        //
        // Sized to a fixed SLOT COUNT rather than a fixed byte count, capped at
        // the region size. A whole 16 MiB region per class is far more than
        // this tier needs — the entire slot-state working set is a few MiB —
        // and on a memory-tight card those slabs compete with KV for the same
        // pool, pushing `ensure_vram_budget` into compaction that costs far
        // more than the churn this tier removed.
        let per_slab = SLOTS_PER_SLAB.min(TARGET_ARENA_BYTES / slot_bytes).max(1);
        // SAFETY: the slab is untyped storage; every slot is written before it
        // is read (`rebuild_decode` fills it, `update_chunk` overwrites in
        // place) and the kernel only reads through a pointer we hand it.
        let slab_bytes = per_slab * slot_bytes;
        let stream = device.cuda_stream();
        let slab = unsafe { Held::<u8>::alloc_on(&stream, slab_bytes)? };
        let base = {
            let (base, _guard) = slab.device_ptr(&stream);
            base
        };
        drop(stream);
        inner.classes[class].add_slab(slab, base, per_slab);
        inner.slabs += 1;
        inner.reserved += slab_bytes;
    }

    let pool = &mut inner.classes[class];
    let (slab, index) = pool.free.pop().expect("just refilled the class");
    pool.free_in[slab] -= 1;
    let ptr = pool.bases[slab] + (index * slot_bytes) as u64;
    inner.live += 1;
    drop(map);
    drop(paused);
    Ok(SlotStateSlot {
        ptr,
        class,
        slab,
        index,
    })
}

/// Return a slot to its class's free list.
///
/// No fence: see the module header. The slot's bytes are left as they are —
/// the next tenant writes every byte it reads, and nothing reads past what it
/// wrote (`n_chunks` bounds the kernel's walk).
///
/// **A slab this empties goes back**, once its class has a slab's worth of free
/// slots elsewhere ([`ClassPool::returned`]): a sequence that went deep, or a wide
/// cohort, no longer holds its peak for the life of the process. It is retired
/// through the device, not dropped — a segment recorded and not yet launched may
/// still read it.
pub(crate) fn release(device: &CudaDevice, slot: SlotStateSlot) {
    let device_id = device.cuda_stream().context().ordinal();
    let freed = {
        let mut map = arenas().lock().unwrap();
        let Some(inner) = map.get_mut(&device_id) else {
            return;
        };
        inner.live = inner.live.saturating_sub(1);
        let freed = inner.classes[slot.class].returned(slot.slab, slot.index);
        if freed.is_some() {
            inner.slabs = inner.slabs.saturating_sub(1);
            inner.reserved = inner.reserved.saturating_sub(
                SLOT_STATE_LADDER[slot.class]
                    * SLOTS_PER_SLAB
                        .min(TARGET_ARENA_BYTES / SLOT_STATE_LADDER[slot.class])
                        .max(1),
            );
        }
        freed
    };
    // Outside the lock: retiring may free at once, which gives the range back to
    // the held ground under its own lock.
    if let Some(slab) = freed {
        device.retire(slab);
    }
}

/// `(live slots, slabs allocated, bytes reserved)`.
///
/// The evidence for step 1's gate condition "decode-path allocator traffic
/// falls to ~zero": `slabs` should settle within the first few waves and then
/// stop moving, however deep the sequences get.
pub fn stats() -> (usize, usize, usize) {
    let map = arenas().lock().unwrap();
    map.values().fold((0, 0, 0), |acc, i| {
        (acc.0 + i.live, acc.1 + i.slabs, acc.2 + i.reserved)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slabs standing in `class` on device `ordinal`.
    fn slabs_in(ordinal: usize, class: usize) -> usize {
        arenas().lock().unwrap().get(&ordinal).map_or(0, |inner| {
            inner.classes[class]
                .slabs
                .iter()
                .filter(|s| s.is_some())
                .count()
        })
    }

    /// **A slab whose slots all come back is given back** — all but one slab's
    /// worth of spare, which a 32-token boundary's promotions land in. On the
    /// 8 MiB rung, two slots a slab, which no other test reaches.
    #[test]
    fn emptied_slabs_go_back_and_one_stays_spare() -> Result<()> {
        use candle::backend::BackendDevice;
        let dev = CudaDevice::new(0)?;
        let ordinal = dev.cuda_stream().context().ordinal();
        let bytes = 8 * 1024 * 1024;
        let class = class_for(bytes).expect("a rung");
        let before = slabs_in(ordinal, class);
        let slots: Vec<SlotStateSlot> =
            (0..5).map(|_| claim(&dev, bytes)).collect::<Result<_>>()?;
        assert_eq!(
            slabs_in(ordinal, class),
            before + 3,
            "five slots, two a slab"
        );
        for s in slots {
            release(&dev, s);
        }
        assert_eq!(
            slabs_in(ordinal, class),
            before.max(1),
            "every emptied slab but one went back"
        );
        dev.synchronize()?;
        Ok(())
    }

    #[test]
    fn the_ladder_doubles_and_is_strictly_increasing() {
        for w in SLOT_STATE_LADDER.windows(2) {
            assert_eq!(
                w[1],
                w[0] * 2,
                "the ladder must double: {SLOT_STATE_LADDER:?}"
            );
        }
    }

    /// Every rung divides the slab size, so a slab carves into whole slots
    /// with nothing stranded at its end.
    ///
    /// Slot *count* per slab is deliberately not held to a floor. The shallow
    /// rungs get many slots because a wave has many `(layer, batch slot)` pairs
    /// of similar depth; the deep rungs exist for a single very long sequence
    /// and legitimately carve 8, 4, 2 or 1 slot from a region. Requiring ≥ 16
    /// everywhere is what capped this ladder at 1 MiB — and so at 48.7 K
    /// tokens — in the first place.
    #[test]
    fn every_class_tiles_a_slab_exactly() {
        for &bytes in SLOT_STATE_LADDER.iter() {
            assert_eq!(
                TARGET_ARENA_BYTES % bytes,
                0,
                "class {bytes} B leaves a partial slot at the end of a slab"
            );
        }
        // The rungs a wave actually shares still hold a useful number.
        for &bytes in SLOT_STATE_LADDER.iter().filter(|&&b| b <= 1024 * 1024) {
            assert!(
                TARGET_ARENA_BYTES / bytes >= 16,
                "class {bytes} B: too few slots per slab for a shared rung"
            );
        }
    }

    #[test]
    fn class_for_picks_the_smallest_that_fits() {
        let top = SLOT_STATE_LADDER[SLOT_STATE_LADDER.len() - 1];
        assert_eq!(class_for(1), Some(0));
        assert_eq!(class_for(4096), Some(0));
        assert_eq!(class_for(4097), Some(1));
        assert_eq!(class_for(1024 * 1024), Some(8));
        assert_eq!(class_for(top), Some(SLOT_STATE_LADDER.len() - 1));
        assert_eq!(
            class_for(top + 1),
            None,
            "past the top rung must not silently land in it"
        );
    }

    /// The top rung's capacity, stated in tokens, so a future geometry change
    /// has to confront whether it is still enough.
    ///
    /// **This is a real ceiling, not a formality.** A slot-state entry is a
    /// 16 B slice header *plus* one `KvHead` record per head — at the
    /// production geometry `4 * kv_head_record_bytes(128)` = 672 B, so 688 B
    /// per chunk, not the 16 B the header alone suggests. Past the top rung
    /// `claim` errors and the forward fails; the `stream.alloc` this tier
    /// replaced had no cap at all.
    ///
    /// Asserted against the real entry width so the number cannot drift back
    /// into looking like 2 M tokens.
    #[test]
    fn the_top_class_holds_a_deep_sequence() {
        use super::super::meta_pool::chunk_record_bytes;
        const SLICE_HEADER_BYTES: usize = 16;
        let per_chunk = SLICE_HEADER_BYTES
            + chunk_record_bytes(4, 128, crate::kv_cache::arena_table::N_PALETTE);
        assert_eq!(per_chunk, 688, "production (n_kv_head 4, head_dim 128)");

        let top = SLOT_STATE_LADDER[SLOT_STATE_LADDER.len() - 1];
        let tokens = (top / per_chunk) * 32;
        assert!(
            tokens >= 750_000,
            "top rung holds {tokens} tokens of ONE sequence in ONE layer; \
             an unbounded-context engine must not wall at less"
        );
    }

    /// Extending the ladder must not put huge slabs in front of KV: a slab is
    /// bounded in **bytes**, not slots, so every rung above 1 MiB carves at
    /// most one region however few slots that is. This is what makes the deep
    /// rungs cheap enough to exist at all.
    #[test]
    fn no_rung_carves_a_slab_larger_than_a_region() {
        for (i, &slot_bytes) in SLOT_STATE_LADDER.iter().enumerate() {
            let per_slab = SLOTS_PER_SLAB.min(TARGET_ARENA_BYTES / slot_bytes).max(1);
            assert!(per_slab >= 1, "rung {i} must hold at least one slot");
            let slab = per_slab * slot_bytes;
            assert!(
                slab <= TARGET_ARENA_BYTES,
                "rung {i} ({slot_bytes} B) carves a {slab} B slab, past the \
                 {TARGET_ARENA_BYTES} B region"
            );
        }
    }
}
