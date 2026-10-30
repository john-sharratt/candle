//! A turn's or section's KV residence, held outside the substrate's versioned
//! root and changed by atomic operations alone (`docs/substrate_lock_free.md`
//! §5).
//!
//! The substrate's root is replaced by compare-and-swap on every commit; a
//! residence is not part of it. Its slot in [`ResidenceSlab`] is handed out
//! before any commit and never moves, and every tier transition is a step on
//! the residence's own cells, so moving KV between VRAM, RAM and disk never
//! commits a substrate version and never waits on another thread.
//!
//! # What makes a transition race-free
//!
//! - **Payload cells are swapped, never edited.** `hot`, `warm` and `cold` are
//!   `ArcSwapOption`s: a reader loads one `Arc` and holds exactly the bytes that
//!   were installed, however the residence changes after.
//! - **A kill and an install are ordered by a sequentially consistent pair.** A
//!   kill sets [`DEAD`] and then takes the payloads; an install stores its
//!   payload and then reads [`DEAD`], taking its own payload back when it is set.
//!   One of the two always sees the other, so a dead residence never keeps a
//!   payload, whichever order they ran in.
//! - **Membership is re-checked after a removal.** A residence leaves the slab's
//!   hot or warm set only when its cell is empty, and is put back when an
//!   install landed between the swap and the removal — so a residence holding a
//!   payload is always found by the passes that walk the sets.
//! - **Device memory is freed on the device's own thread.** A hot payload whose
//!   last reference drops anywhere else is handed to the reclaim queue
//!   ([`drain_reclaimed`]), which that thread empties at each wave.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::ops::Deref;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwapOption;
use candle_nn::kv_cache::{rewrite_sealed, SealedSequence, Sweep};
use crossbeam::queue::SegQueue;

use crate::persistence::streams::StreamId;

use super::{ConvCompression, ResidenceIndex, StoredSequence};

/// The scheduler still owes this section a quantize: it was ingested with its
/// native prefill K/V in `hot`, which the next turn-seal boundary replaces with
/// the quantized form. Until then the hot copy is interim, and a cold copy of
/// it would diverge from the final in-memory form after the swap.
pub const PENDING_QUANTIZE: u32 = 1 << 0;
/// Offload as the KV becomes durable: hot goes the moment a warm or cold copy
/// exists, warm the moment cold lands, leaving the residence cold-only until
/// `elevate_to_hot` pulls it back. Set for collection-member sections (nothing
/// attends back over them during the build) and completed-ingest turns (not
/// attended again until retrieval). Live dialogue turns never carry it.
pub const EVICT_WHEN_COLD: u32 = 1 << 1;
/// A warm→cold write is queued on the writer and has not landed, so the next
/// persistence pass must not gather and write the same KV again. Distinct from
/// a cold copy, which exists only once the write lands.
pub const COLD_PENDING: u32 = 1 << 2;
/// Never written to cold: a scratch fork's KV, spliced by reference onto a
/// durable file timeline, where a cold copy would only strand an orphan. Hot and
/// warm are still allowed, and warm is freed at the tombstone since no cold copy
/// will ever reclaim it.
pub const NO_COLD_PERSIST: u32 = 1 << 3;
/// A turn of a splice-source timeline (a code_read scope fork) until the fork's
/// tombstone. `adopt_turn` splices from the hot copy and nothing else, so no
/// automatic drop may take it — not the migrate's install, the cold-land
/// offload, the idle demote, turnover, relief or the ingest demotes. Without
/// it, a fork whose warm copy had landed lost its hot copy to relief between
/// its seal and its splice, and the splice failed with "source K/V not hot".
pub const SPLICE_SOURCE: u32 = 1 << 4;
/// The residence's owner is gone. No payload may be installed on it again.
pub const DEAD: u32 = 1 << 5;
/// A prompt section's residence, not a turn's. Sections are shared prompt
/// content a projection re-reads constantly: no recency-ordered eviction ranks
/// them against turns, and only the idle demote takes their hot copy.
pub const SECTION: u32 = 1 << 6;

thread_local! {
    /// Whether this thread frees device memory itself — the thread that owns the
    /// device's allocations and compacts them.
    static OWNS_DEVICE_FREES: Cell<bool> = const { Cell::new(false) };
    /// The residences allocated by the version edit this thread is attempting,
    /// while one is — see [`begin_attempt`].
    static ATTEMPT: RefCell<Option<Vec<usize>>> = const { RefCell::new(None) };
}

/// A version edit is starting on this thread: every residence allocated until
/// [`end_attempt`] belongs to it.
///
/// An edit names the residences it allocates in the version it builds, and that
/// version may lose its commit to another writer's and be rebuilt. The losing
/// attempt's residences are named by nothing that will ever be published, so
/// whatever was installed on them is discarded rather than left to hold memory
/// no pass can reach.
pub fn begin_attempt() {
    ATTEMPT.with(|a| {
        let mut a = a.borrow_mut();
        debug_assert!(a.is_none(), "version edits do not nest");
        *a = Some(Vec::new());
    });
}

/// The attempt [`begin_attempt`] started is over: `committed` says whether the
/// version it built was published. A lost attempt's residences are discarded.
pub fn end_attempt(slab: &ResidenceSlab, committed: bool) {
    let allocated = ATTEMPT.with(|a| a.borrow_mut().take()).unwrap_or_default();
    if committed {
        return;
    }
    for i in allocated {
        slab.discard(ResidenceIndex(i));
    }
}

/// Hot payloads whose last reference dropped off the owning thread, waiting to
/// be freed on it.
static RECLAIM: SegQueue<Vec<SealedSequence>> = SegQueue::new();

/// Mark the calling thread as the one that frees device memory. Called once by
/// the scheduler thread as it starts.
pub fn own_device_frees() {
    OWNS_DEVICE_FREES.with(|o| o.set(true));
}

/// Free every hot payload handed back since the last call, on the calling
/// thread — which must be the device's own ([`own_device_frees`]). Returns how
/// many payloads were freed.
///
/// Called at each wave boundary and before a KV compaction plans, so a pass
/// never runs with frees it cannot see still pending from an earlier wave.
pub fn drain_reclaimed() -> usize {
    debug_assert!(
        OWNS_DEVICE_FREES.with(Cell::get),
        "reclaimed device memory drained off the device's own thread"
    );
    let mut n = 0usize;
    while let Some(payload) = RECLAIM.pop() {
        drop(payload);
        n += 1;
    }
    n
}

/// Rewrite every payload waiting in the reclaim queue through a KV compaction's
/// sweep, on the thread that compacts. Returns how many payloads were rewritten.
///
/// A queued payload is a holder: its chunks stay allocated until the next
/// drain. A hot copy dropped off this thread while a pass relocates lands here
/// after its residence stopped naming it, so a sweep that visits only the
/// residences leaves the pass's copies of its moved slots named by nothing —
/// claims spent on ground no holder takes up. Called after every residence has
/// been visited: a copy dropped before its residence was visited is found here,
/// and one dropped after is already the rewritten copy.
pub fn rewrite_reclaimed(sweep: &mut Sweep<'_>) -> candle::Result<usize> {
    debug_assert!(
        OWNS_DEVICE_FREES.with(Cell::get),
        "reclaimed device memory rewritten off the device's own thread"
    );
    rewrite_queue(&RECLAIM, sweep)
}

/// [`rewrite_reclaimed`] over `queue`. Every payload taken out goes back — its
/// replacement when it moved, itself otherwise or once a rewrite has failed —
/// so a failed sweep loses nothing a drain would have freed.
fn rewrite_queue(
    queue: &SegQueue<Vec<SealedSequence>>,
    sweep: &mut Sweep<'_>,
) -> candle::Result<usize> {
    let mut queued = Vec::with_capacity(queue.len());
    while let Some(payload) = queue.pop() {
        queued.push(payload);
    }
    let mut rewritten = 0usize;
    let mut failed = None;
    for payload in queued {
        if failed.is_none() {
            match rewrite_sealed(&payload, sweep) {
                Ok(Some(next)) => {
                    rewritten += 1;
                    queue.push(next);
                    continue;
                }
                Ok(None) => {}
                Err(e) => failed = Some(e),
            }
        }
        queue.push(payload);
    }
    match failed {
        Some(e) => Err(e),
        None => Ok(rewritten),
    }
}

/// Hot payloads waiting for [`drain_reclaimed`].
pub fn reclaim_pending() -> usize {
    RECLAIM.len()
}

/// A residence's VRAM-resident sealed chunks. Dropped off the device's own
/// thread, the chunks are handed to the reclaim queue instead of freed there.
pub struct HotKv {
    seqs: Vec<SealedSequence>,
}

impl HotKv {
    pub fn new(seqs: Vec<SealedSequence>) -> Self {
        Self { seqs }
    }

    /// The sequences, cloned — chunk handles, not bytes.
    pub fn to_vec(&self) -> Vec<SealedSequence> {
        self.seqs.clone()
    }
}

impl Deref for HotKv {
    type Target = [SealedSequence];

    fn deref(&self) -> &[SealedSequence] {
        &self.seqs
    }
}

impl std::fmt::Debug for HotKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HotKv")
            .field("layers", &self.seqs.len())
            .finish()
    }
}

impl Drop for HotKv {
    fn drop(&mut self) {
        let seqs = std::mem::take(&mut self.seqs);
        if seqs.is_empty() || OWNS_DEVICE_FREES.with(Cell::get) {
            return;
        }
        RECLAIM.push(seqs);
    }
}

/// Total VRAM bytes of `seqs`' chunks.
pub fn sealed_bytes(seqs: &[SealedSequence]) -> u64 {
    seqs.iter()
        .flat_map(|s| s.chunks.iter())
        .map(|c| c.byte_size)
        .sum()
}

/// One turn's or section's KV across hot (VRAM), warm (RAM) and cold (disk).
pub struct Residence {
    pub stream_id: StreamId,
    /// The owning timeline's compression override at allocation.
    pub compression: Option<ConvCompression>,
    flags: AtomicU32,
    /// Working sets holding it — see [`ResidenceSlab::publish_pins`].
    pins: AtomicU32,
    hot: ArcSwapOption<HotKv>,
    warm: ArcSwapOption<Vec<SealedSequence>>,
    cold: ArcSwapOption<Vec<StoredSequence>>,
    /// Bytes of the sealed payload, per tier. Constant across transitions.
    byte_size: AtomicU64,
    /// Wave epoch of the last forward that resolved its hot KV.
    last_used_epoch: AtomicU64,
    /// Recency stamp from the slab's clock: higher is more recent.
    touched: AtomicU64,
}

impl std::fmt::Debug for Residence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Residence")
            .field("stream_id", &self.stream_id)
            .field("flags", &self.flags())
            .field("pins", &self.pins())
            .field("hot", &self.has_hot())
            .field("warm", &self.has_warm())
            .field("cold", &self.has_cold())
            .finish()
    }
}

impl Residence {
    fn new(stream_id: StreamId, compression: Option<ConvCompression>, epoch: u64) -> Self {
        Self {
            stream_id,
            compression,
            flags: AtomicU32::new(0),
            pins: AtomicU32::new(0),
            hot: ArcSwapOption::empty(),
            warm: ArcSwapOption::empty(),
            cold: ArcSwapOption::empty(),
            byte_size: AtomicU64::new(0),
            // **Born now, not at zero.** A residence is installed before the wave
            // that will attend it publishes its pins, so a zero stamp would make it
            // stale from birth and let the idle demote take it before its first use.
            last_used_epoch: AtomicU64::new(epoch),
            touched: AtomicU64::new(0),
        }
    }

    pub fn flags(&self) -> u32 {
        self.flags.load(Ordering::SeqCst)
    }

    pub fn has_flag(&self, flag: u32) -> bool {
        self.flags() & flag != 0
    }

    pub fn set_flag(&self, flag: u32) {
        self.flags.fetch_or(flag, Ordering::SeqCst);
    }

    pub fn clear_flag(&self, flag: u32) {
        self.flags.fetch_and(!flag, Ordering::SeqCst);
    }

    pub fn is_dead(&self) -> bool {
        self.has_flag(DEAD)
    }

    pub fn pins(&self) -> u32 {
        self.pins.load(Ordering::SeqCst)
    }

    pub fn is_pinned(&self) -> bool {
        self.pins() > 0
    }

    pub fn hot(&self) -> Option<Arc<HotKv>> {
        self.hot.load_full()
    }

    pub fn warm(&self) -> Option<Arc<Vec<SealedSequence>>> {
        self.warm.load_full()
    }

    pub fn cold(&self) -> Option<Arc<Vec<StoredSequence>>> {
        self.cold.load_full()
    }

    pub fn has_hot(&self) -> bool {
        self.hot.load().is_some()
    }

    pub fn has_warm(&self) -> bool {
        self.warm.load().is_some()
    }

    pub fn has_cold(&self) -> bool {
        self.cold.load().is_some()
    }

    pub fn byte_size(&self) -> u64 {
        self.byte_size.load(Ordering::Relaxed)
    }

    /// Set the accounted payload size directly — for tests whose payloads carry
    /// no chunks but whose budget passes count bytes.
    #[cfg(test)]
    pub fn set_byte_size(&self, bytes: u64) {
        self.byte_size.store(bytes, Ordering::Relaxed);
    }

    pub fn last_used_epoch(&self) -> u64 {
        self.last_used_epoch.load(Ordering::Relaxed)
    }

    pub fn stamp_used(&self, epoch: u64) {
        self.last_used_epoch.fetch_max(epoch, Ordering::Relaxed);
    }

    pub fn touched(&self) -> u64 {
        self.touched.load(Ordering::Relaxed)
    }
}

/// Every residence of a substrate, and the tier sets the passes walk.
///
/// Shared by every version of the substrate's root: a residence allocated under
/// one version is the same residence under every later one.
pub struct ResidenceSlab {
    slots: boxcar::Vec<Residence>,
    hot: papaya::HashSet<usize>,
    warm: papaya::HashSet<usize>,
    /// The idle demote's clock: one tick per persistence pass.
    wave_epoch: AtomicU64,
    /// Recency clock behind [`Residence::touched`].
    clock: AtomicU64,
    /// Each pin publisher's last published set — see [`Self::publish_pins`].
    published: papaya::HashMap<u64, Arc<HashSet<usize>>>,
}

impl std::fmt::Debug for ResidenceSlab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidenceSlab")
            .field("residences", &self.slots.count())
            .field("hot", &self.hot.len())
            .field("warm", &self.warm.len())
            .finish()
    }
}

impl Default for ResidenceSlab {
    fn default() -> Self {
        Self {
            slots: boxcar::Vec::new(),
            hot: papaya::HashSet::new(),
            warm: papaya::HashSet::new(),
            wave_epoch: AtomicU64::new(0),
            clock: AtomicU64::new(0),
            published: papaya::HashMap::new(),
        }
    }
}

impl ResidenceSlab {
    /// A fresh residence, holding nothing, carrying `flags`. Its index is
    /// stable for the slab's life, and is handed out before any substrate
    /// commit names it.
    pub fn alloc(
        &self,
        stream_id: StreamId,
        compression: Option<ConvCompression>,
        flags: u32,
    ) -> ResidenceIndex {
        let epoch = self.wave_epoch();
        let r = Residence::new(stream_id, compression, epoch);
        r.flags.store(flags, Ordering::SeqCst);
        let i = self.slots.push(r);
        ATTEMPT.with(|a| {
            if let Some(a) = a.borrow_mut().as_mut() {
                a.push(i);
            }
        });
        ResidenceIndex(i)
    }

    /// A residence nothing names: dead, holding nothing in any tier.
    fn discard(&self, idx: ResidenceIndex) {
        let r = self.get(idx);
        r.set_flag(DEAD | NO_COLD_PERSIST);
        self.take_hot(idx);
        self.take_warm(idx);
        r.cold.store(None);
    }

    /// The residence at `idx`. Panics on an index no allocation returned.
    pub fn get(&self, idx: ResidenceIndex) -> &Residence {
        self.slots
            .get(idx.0)
            .unwrap_or_else(|| panic!("residence {} was never allocated", idx.0))
    }

    /// The residence at `idx`, if one was allocated there.
    pub fn try_get(&self, idx: ResidenceIndex) -> Option<&Residence> {
        self.slots.get(idx.0)
    }

    /// Residences allocated so far.
    pub fn len(&self) -> usize {
        self.slots.count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every residence, with its index.
    pub fn iter(&self) -> impl Iterator<Item = (ResidenceIndex, &Residence)> + '_ {
        self.slots.iter().map(|(i, r)| (ResidenceIndex(i), r))
    }

    pub fn wave_epoch(&self) -> u64 {
        self.wave_epoch.load(Ordering::SeqCst)
    }

    /// Advance the idle demote's clock by one pass, returning the new epoch.
    pub fn tick_epoch(&self) -> u64 {
        self.wave_epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn touch(&self, r: &Residence) {
        let t = self.clock.fetch_add(1, Ordering::Relaxed) + 1;
        r.touched.store(t, Ordering::Relaxed);
    }

    /// Mark `idx` most recently used.
    pub fn touch_index(&self, idx: ResidenceIndex) {
        self.touch(self.get(idx));
    }

    // ── hot ─────────────────────────────────────────────────────────────────

    /// Install `seqs` as `idx`'s hot copy, replacing any. Returns `false`, and
    /// installs nothing, on a dead residence — its payload goes to the reclaim
    /// path with the dropped `Arc`.
    ///
    /// **An install is a use.** Whatever put KV hot — a seal, a lift for a
    /// projection, a section's ingest — puts it there to be read, so it is
    /// stamped used at the current epoch: the idle demote leaves it for the
    /// whole grace window, however long ago the residence was last read.
    pub fn install_hot(&self, idx: ResidenceIndex, seqs: Vec<SealedSequence>) -> bool {
        debug_assert!(!seqs.is_empty(), "install_hot with an empty payload");
        let r = self.get(idx);
        if r.is_dead() {
            return false;
        }
        r.byte_size.store(sealed_bytes(&seqs), Ordering::Relaxed);
        r.stamp_used(self.wave_epoch());
        r.hot.store(Some(Arc::new(HotKv::new(seqs))));
        self.touch(r);
        self.hot.pin().insert(idx.0);
        // The kill/install pair: a kill that set DEAD after the check above
        // either took this payload already or is seen here.
        if r.is_dead() {
            self.take_hot(idx);
            return false;
        }
        true
    }

    /// Install `seqs` as hot only if `idx` holds none — what a lift does, so
    /// two lifts of one residence never stack and a lift never lands on a dead
    /// one. Returns whether it was installed. Stamped used, as
    /// [`Self::install_hot`] is.
    pub fn install_hot_if_absent(&self, idx: ResidenceIndex, seqs: Vec<SealedSequence>) -> bool {
        let r = self.get(idx);
        if r.is_dead() || r.has_hot() {
            return false;
        }
        let bytes = sealed_bytes(&seqs);
        // Stamped before the cell is published: a demote that finds the lift
        // also finds it fresh.
        r.stamp_used(self.wave_epoch());
        let fresh = Some(Arc::new(HotKv::new(seqs)));
        let prev = r.hot.compare_and_swap(&None::<Arc<HotKv>>, fresh);
        if prev.is_some() {
            return false;
        }
        r.byte_size.store(bytes, Ordering::Relaxed);
        self.touch(r);
        self.hot.pin().insert(idx.0);
        if r.is_dead() {
            self.take_hot(idx);
            return false;
        }
        true
    }

    /// Replace `idx`'s hot copy only if it is still `expected` — a quantize or
    /// a migrate installing the copy it produced from that one. Returns whether
    /// it was replaced; a residence whose hot copy changed or went keeps what it
    /// has, and `seqs` is dropped.
    pub fn replace_hot(
        &self,
        idx: ResidenceIndex,
        expected: &Arc<HotKv>,
        seqs: Vec<SealedSequence>,
    ) -> bool {
        let r = self.get(idx);
        if r.is_dead() {
            return false;
        }
        let fresh = Some(Arc::new(HotKv::new(seqs)));
        let prev = r.hot.compare_and_swap(expected, fresh);
        let replaced = prev.as_ref().is_some_and(|p| Arc::ptr_eq(p, expected));
        if replaced && r.is_dead() {
            self.take_hot(idx);
            return false;
        }
        replaced
    }

    /// Take `idx`'s hot copy, leaving none. The returned `Arc` frees its chunks
    /// on the device's thread when the last holder lets go.
    pub fn take_hot(&self, idx: ResidenceIndex) -> Option<Arc<HotKv>> {
        let r = self.get(idx);
        let old = r.hot.swap(None);
        if old.is_some() {
            let set = self.hot.pin();
            set.remove(&idx.0);
            if r.has_hot() {
                set.insert(idx.0);
            }
        }
        old
    }

    /// Indices whose hot cell may hold a payload — a superset; the caller reads
    /// the cell.
    pub fn hot_members(&self) -> Vec<ResidenceIndex> {
        self.hot.pin().iter().map(|&i| ResidenceIndex(i)).collect()
    }

    // ── warm ────────────────────────────────────────────────────────────────

    /// Install `seqs` as `idx`'s warm copy. Returns `false` on a dead residence.
    pub fn install_warm(&self, idx: ResidenceIndex, seqs: Vec<SealedSequence>) -> bool {
        debug_assert!(!seqs.is_empty(), "install_warm with an empty payload");
        let r = self.get(idx);
        if r.is_dead() && r.has_flag(NO_COLD_PERSIST) {
            return false;
        }
        if r.byte_size() == 0 {
            r.byte_size.store(sealed_bytes(&seqs), Ordering::Relaxed);
        }
        r.warm.store(Some(Arc::new(seqs)));
        self.touch(r);
        self.warm.pin().insert(idx.0);
        if r.is_dead() && r.has_flag(NO_COLD_PERSIST) {
            self.take_warm(idx);
            return false;
        }
        true
    }

    /// Replace `idx`'s warm copy only if it is still `expected` — a compaction
    /// installing the copy it rewrote from that one. Returns whether it was
    /// replaced.
    pub fn replace_warm(
        &self,
        idx: ResidenceIndex,
        expected: &Arc<Vec<SealedSequence>>,
        seqs: Vec<SealedSequence>,
    ) -> bool {
        let r = self.get(idx);
        let prev = r.warm.compare_and_swap(expected, Some(Arc::new(seqs)));
        prev.as_ref().is_some_and(|p| Arc::ptr_eq(p, expected))
    }

    /// Take `idx`'s warm copy, leaving none.
    pub fn take_warm(&self, idx: ResidenceIndex) -> Option<Arc<Vec<SealedSequence>>> {
        let r = self.get(idx);
        let old = r.warm.swap(None);
        if old.is_some() {
            let set = self.warm.pin();
            set.remove(&idx.0);
            if r.has_warm() {
                set.insert(idx.0);
            }
        }
        old
    }

    /// Indices whose warm cell may hold a payload — a superset.
    pub fn warm_members(&self) -> Vec<ResidenceIndex> {
        self.warm.pin().iter().map(|&i| ResidenceIndex(i)).collect()
    }

    // ── cold ────────────────────────────────────────────────────────────────

    /// Install `idx`'s cold references and clear its in-flight cold write.
    pub fn install_cold(&self, idx: ResidenceIndex, cold: Vec<StoredSequence>) {
        let r = self.get(idx);
        if r.byte_size() == 0 {
            let bytes: u64 = cold
                .iter()
                .flat_map(|s| s.chunks.iter())
                .map(|c| c.record_len)
                .sum();
            r.byte_size.store(bytes, Ordering::Relaxed);
        }
        r.cold.store(Some(Arc::new(cold)));
        r.clear_flag(COLD_PENDING);
    }

    /// Replace `idx`'s cold references wholesale.
    pub fn set_cold(&self, idx: ResidenceIndex, cold: Option<Vec<StoredSequence>>) {
        self.get(idx).cold.store(cold.map(Arc::new));
    }

    /// Replace `idx`'s cold references only if they are still `expected` — a
    /// relocation's refresh of the references it read. Returns whether they
    /// were replaced.
    pub fn replace_cold(
        &self,
        idx: ResidenceIndex,
        expected: &Arc<Vec<StoredSequence>>,
        cold: Vec<StoredSequence>,
    ) -> bool {
        let prev = self
            .get(idx)
            .cold
            .compare_and_swap(expected, Some(Arc::new(cold)));
        prev.as_ref().is_some_and(|p| Arc::ptr_eq(p, expected))
    }

    // ── kill ────────────────────────────────────────────────────────────────

    /// The residence's owner is gone: mark it dead and take its hot copy, and
    /// its warm one too when no cold copy will ever land to reclaim it. Returns
    /// the payloads taken, freed when the caller drops them.
    pub fn kill(
        &self,
        idx: ResidenceIndex,
    ) -> (Option<Arc<HotKv>>, Option<Arc<Vec<SealedSequence>>>) {
        let r = self.get(idx);
        r.clear_flag(SPLICE_SOURCE);
        r.set_flag(EVICT_WHEN_COLD | DEAD);
        let hot = self.take_hot(idx);
        let warm = if r.has_flag(NO_COLD_PERSIST) {
            self.take_warm(idx)
        } else {
            None
        };
        (hot, warm)
    }

    // ── pins ────────────────────────────────────────────────────────────────

    /// Publish `keep` as the residences `owner`'s working sets hold, replacing
    /// what `owner` published before: each residence entering the set gains a
    /// pin and each leaving it loses one, and every residence in it is stamped
    /// `epoch` as used. Publishers do not overwrite each other — pins are
    /// counts — and a publisher's own calls come from its one thread.
    pub fn publish_pins(&self, owner: u64, keep: HashSet<usize>) {
        let epoch = self.wave_epoch();
        let map = self.published.pin();
        let before = map.get(&owner).cloned().unwrap_or_default();
        for &i in keep.difference(&before) {
            if let Some(r) = self.slots.get(i) {
                r.pins.fetch_add(1, Ordering::SeqCst);
            }
        }
        for &i in before.difference(&keep) {
            if let Some(r) = self.slots.get(i) {
                let _ = r
                    .pins
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |p| p.checked_sub(1));
            }
        }
        for &i in &keep {
            if let Some(r) = self.slots.get(i) {
                r.stamp_used(epoch);
            }
        }
        map.insert(owner, Arc::new(keep));
    }

    /// Every publisher's pins released — the shutdown drain's final pass, which
    /// must reach every turn.
    pub fn release_all_pins(&self) {
        let map = self.published.pin();
        let owners: Vec<u64> = map.keys().copied().collect();
        drop(map);
        for owner in owners {
            self.publish_pins(owner, HashSet::new());
        }
    }

    /// Residences any publisher holds.
    pub fn pinned(&self) -> Vec<ResidenceIndex> {
        let map = self.published.pin();
        let mut all: HashSet<usize> = HashSet::new();
        for set in map.values() {
            all.extend(set.iter().copied());
        }
        all.into_iter().map(ResidenceIndex).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::kv_cache::{
        ArenaLocation, ChunkGid, CompactionMap, HeadGids, SealedChunk, GID_STRIDE,
    };

    /// A sealed sequence with no chunks — the protocol under test is the cell,
    /// not the bytes in it.
    fn seq() -> SealedSequence {
        SealedSequence {
            chunks: Vec::new(),
            token_count: 0,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }
    }

    /// A one-layer sequence of one chunk at raw gid `raw`.
    fn seq_at(raw: i64) -> SealedSequence {
        SealedSequence {
            chunks: vec![SealedChunk {
                gids: HeadGids::uniform(ChunkGid::detached(raw), 1),
                offset: 0,
                token_count: 32,
                k_pal: Arc::new(Vec::new()),
                v_pal: Arc::new(Vec::new()),
                k_scale: Arc::new(Vec::new()),
                v_scale: Arc::new(Vec::new()),
                k_fmt: Arc::new(Vec::new()),
                v_fmt: Arc::new(Vec::new()),
                byte_size: 0,
                meta: None,
            }],
            token_count: 32,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }
    }

    /// The raw gid of each queued payload's one chunk, in queue order.
    fn queued_raws(queue: &SegQueue<Vec<SealedSequence>>) -> Vec<i64> {
        let mut raws = Vec::new();
        let mut kept = Vec::new();
        while let Some(p) = queue.pop() {
            raws.push(p[0].chunks[0].gids.as_slice()[0].raw());
            kept.push(p);
        }
        for p in kept {
            queue.push(p);
        }
        raws
    }

    /// **A payload waiting to be freed is a holder the sweep reaches.** One that
    /// moved comes back as its replacement, naming the destination; one that did
    /// not stays as it was; and nothing taken out of the queue is lost.
    #[test]
    fn a_queued_payload_is_rewritten_by_the_sweep() {
        let src = (5 * GID_STRIDE + 2) as i64;
        let dst = (GID_STRIDE + 1) as i64;
        let still = (7 * GID_STRIDE + 7) as i64;
        let queue = SegQueue::new();
        queue.push(vec![seq_at(src)]);
        queue.push(vec![seq_at(still)]);

        let mut map = CompactionMap::new();
        map.insert(src, ChunkGid::detached(dst), 0x4000);
        let mut sweep = Sweep::new(&map);

        assert_eq!(rewrite_queue(&queue, &mut sweep).unwrap(), 1);
        assert_eq!(queued_raws(&queue), [dst, still]);
        assert_eq!(sweep.allocations_rewritten(), 1);
    }

    fn slab_with(n: usize) -> (ResidenceSlab, Vec<ResidenceIndex>) {
        let slab = ResidenceSlab::default();
        let idx = (0..n)
            .map(|i| slab.alloc(StreamId(i as u64 + 1), None, 0))
            .collect();
        (slab, idx)
    }

    /// A residence never holds a payload once dead: an install after the kill
    /// is refused, and a kill after an install takes it.
    #[test]
    fn a_dead_residence_keeps_no_hot_payload() {
        let (slab, idx) = slab_with(1);
        let r = idx[0];
        assert!(slab.install_hot(r, vec![seq()]));
        assert!(slab.get(r).has_hot());
        let (hot, warm) = slab.kill(r);
        assert!(hot.is_some() && warm.is_none());
        assert!(!slab.get(r).has_hot());
        assert!(!slab.install_hot(r, vec![seq()]));
        assert!(!slab.install_hot_if_absent(r, vec![seq()]));
        assert!(!slab.get(r).has_hot());
        assert!(slab.get(r).is_dead());
    }

    /// A lift never stacks on a residence already hot.
    #[test]
    fn a_lift_lands_only_on_an_empty_cell() {
        let (slab, idx) = slab_with(1);
        assert!(slab.install_hot_if_absent(idx[0], vec![seq()]));
        assert!(!slab.install_hot_if_absent(idx[0], vec![seq()]));
        assert_eq!(slab.hot_members(), vec![idx[0]]);
    }

    /// A replace succeeds only against the copy it was made from.
    #[test]
    fn a_replace_expects_the_copy_it_replaces() {
        let (slab, idx) = slab_with(1);
        slab.install_hot(idx[0], vec![seq()]);
        let seen = slab.get(idx[0]).hot().unwrap();
        slab.install_hot(idx[0], vec![seq()]);
        assert!(!slab.replace_hot(idx[0], &seen, vec![seq()]));
        let now = slab.get(idx[0]).hot().unwrap();
        assert!(slab.replace_hot(idx[0], &now, vec![seq()]));
    }

    /// Membership follows the cell through take and re-install.
    #[test]
    fn membership_follows_the_cells() {
        let (slab, idx) = slab_with(3);
        slab.install_hot(idx[0], vec![seq()]);
        slab.install_hot(idx[2], vec![seq()]);
        slab.install_warm(idx[1], vec![seq()]);
        let mut hot = slab.hot_members();
        hot.sort_by_key(|r| r.0);
        assert_eq!(hot, vec![idx[0], idx[2]]);
        assert_eq!(slab.warm_members(), vec![idx[1]]);
        assert!(slab.take_hot(idx[0]).is_some());
        assert_eq!(slab.hot_members(), vec![idx[2]]);
        assert!(slab.take_hot(idx[0]).is_none());
        assert!(slab.take_warm(idx[1]).is_some());
        assert!(slab.warm_members().is_empty());
    }

    /// Pins are counts per publisher: two publishers hold one residence twice,
    /// and a publisher's next set releases only what it held.
    #[test]
    fn pins_are_counted_per_publisher() {
        let (slab, idx) = slab_with(3);
        slab.publish_pins(1, [0usize, 1].into_iter().collect());
        slab.publish_pins(2, [1usize, 2].into_iter().collect());
        assert_eq!(
            (0..3).map(|i| slab.get(idx[i]).pins()).collect::<Vec<_>>(),
            vec![1, 2, 1]
        );
        slab.publish_pins(1, [2usize].into_iter().collect());
        assert_eq!(
            (0..3).map(|i| slab.get(idx[i]).pins()).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let mut pinned = slab.pinned();
        pinned.sort_by_key(|r| r.0);
        assert_eq!(pinned, vec![idx[1], idx[2]]);
        slab.release_all_pins();
        assert!((0..3).all(|i| slab.get(idx[i]).pins() == 0));
        assert!(slab.pinned().is_empty());
    }

    /// A publish stamps every residence it holds as used this epoch.
    #[test]
    fn a_publish_stamps_the_epoch() {
        let (slab, idx) = slab_with(2);
        assert_eq!(slab.tick_epoch(), 1);
        assert_eq!(slab.tick_epoch(), 2);
        slab.publish_pins(1, [1usize].into_iter().collect());
        assert_eq!(slab.get(idx[0]).last_used_epoch(), 0);
        assert_eq!(slab.get(idx[1]).last_used_epoch(), 2);
        let born = slab.alloc(StreamId(9), None, 0);
        assert_eq!(slab.get(born).last_used_epoch(), 2, "born now, not at zero");
    }

    /// **A lift is a use** (R5): a residence last read long ago and lifted now
    /// is stamped at the epoch it was lifted in, so the idle demote leaves it
    /// for the read the lift was for.
    #[test]
    fn a_lift_stamps_the_epoch_it_lands_in() {
        let (slab, idx) = slab_with(2);
        for _ in 0..40 {
            slab.tick_epoch();
        }
        assert_eq!(slab.get(idx[0]).last_used_epoch(), 0, "idle since birth");
        assert!(slab.install_hot_if_absent(idx[0], vec![seq()]));
        assert_eq!(slab.get(idx[0]).last_used_epoch(), 40);
        assert!(slab.install_hot(idx[1], vec![seq()]));
        assert_eq!(slab.get(idx[1]).last_used_epoch(), 40);
    }

    /// A kill takes warm too when no cold copy will land to reclaim it.
    #[test]
    fn a_kill_takes_warm_only_when_nothing_will_reclaim_it() {
        let (slab, idx) = slab_with(2);
        slab.get(idx[1]).set_flag(NO_COLD_PERSIST);
        for &r in &idx {
            slab.install_warm(r, vec![seq()]);
        }
        assert!(slab.kill(idx[0]).1.is_none());
        assert!(slab.get(idx[0]).has_warm());
        assert!(slab.kill(idx[1]).1.is_some());
        assert!(!slab.get(idx[1]).has_warm());
    }

    /// **A hot cell is always a member**, however installs and takes of one
    /// residence interleave (`docs/substrate_lock_free.md` §5.2): a residence
    /// leaves the hot set only when its cell is empty, and is put back when an
    /// install lands between a take's swap and its removal. Checked once both
    /// threads have stopped — a take mid-flight removes the member before it
    /// re-reads the cell and puts it back — over many rounds, each of which can
    /// end on a different interleaving.
    ///
    /// The set is a superset and no more ([`ResidenceSlab::hot_members`]): an
    /// install whose insert lands after a take's removal leaves a member whose
    /// cell is empty, which every reader of the set tolerates by reading the
    /// cell. Asserted exact, this failed one run in a few.
    #[test]
    fn concurrent_install_and_take_never_lose_a_hot_member() {
        let slab = Arc::new(ResidenceSlab::default());
        let r = slab.alloc(StreamId(1), None, 0);
        for round in 0..200 {
            let installer = {
                let slab = Arc::clone(&slab);
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        slab.install_hot(r, vec![seq()]);
                    }
                })
            };
            let taker = {
                let slab = Arc::clone(&slab);
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        slab.take_hot(r);
                    }
                })
            };
            installer.join().unwrap();
            taker.join().unwrap();
            if slab.get(r).has_hot() {
                assert!(
                    slab.hot_members().contains(&r),
                    "round {round}: a hot cell is in the hot set"
                );
            }
        }
    }
}
