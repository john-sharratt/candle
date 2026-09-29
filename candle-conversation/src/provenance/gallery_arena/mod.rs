//! Paged VRAM arena for the wide-Q provenance gallery.
//!
//! Keeps each turn's folded signatures **resident on the GPU** between
//! reprojections so the belief scan stops re-uploading the whole corpus every
//! scan. A turn's `N` tokens occupy `ceil(N/32)` fixed 6 KiB group-major pages,
//! each one slot of the gallery's own slot arenas in the device reservation
//! ([`page_io`] moves the words); a re-seal or eviction drops the turn's slots,
//! and an arena whose last page goes hands its region back to the span. The scan
//! kernel reads a *paged gallery* — an array of page device addresses plus a
//! per-token page map — modelled on the paged-KV pointer interface. See
//! `docs/archived/paged_gallery_arena.md`.
//!
//! The arena is a **device-level** resource (one per GPU, scheduler-owned): its
//! residency map is keyed by the global [`StreamId`] of a turn, so every
//! conversation on the device shares it. The warm/cold tiers already exist — the
//! substrate `wide_q_sigs` blob and its `decoded_wide_sig` `Arc` memo — so the
//! arena owns only the hot VRAM tier and rebuilds an evicted turn on demand.

mod page_io;
mod pages;
mod scan;

pub use pages::{page_u64, pages_for, transpose_to_pages, PAGE_TOKENS};
pub use scan::{PagedSegment, PagedWindow};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use candle::{Device, Result};
use candle_nn::kv_cache::{
    arena_held_bytes, arena_regions, claim_arena_slots, plan_slot_moves, ArenaSlot, SlotTenant,
    SpanRegion,
};

use crate::persistence::streams::StreamId;

use super::WideQSig;

/// Max distinct segment-set indices cached at once — a handful of belief groups
/// per reprojection, so this comfortably covers a whole reproject's scans.
const INDEX_CACHE_CAP: usize = 16;

/// What one gallery compaction pass did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GalleryCompaction {
    /// Moves planned, each with its destination claimed.
    pub planned: usize,
    /// Pages that moved. Short of `planned` when a source belonged to a pinned
    /// turn, or to one evicted between the plan and the apply.
    pub moved: usize,
    pub regions_before: usize,
    pub regions_after: usize,
}

impl GalleryCompaction {
    /// Regions the pass handed back to the span.
    pub fn regions_released(&self) -> usize {
        self.regions_before.saturating_sub(self.regions_after)
    }
}

/// A turn's owned run of page slots, in page order. Dropping it returns every slot
/// to the gallery's arenas — so evicting a turn is just dropping its
/// [`ResidentTurn`].
struct PageRun {
    slots: Vec<ArenaSlot>,
}

impl PageRun {
    /// Device addresses of the run's pages, in page order.
    fn addrs(&self) -> Vec<u64> {
        self.slots.iter().map(ArenaSlot::ptr).collect()
    }
}

struct ResidentTurn {
    fingerprint: u64,
    run: PageRun,
    n_tokens: usize,
    lru: u64,
    /// Non-zero while a scan is reading this turn's pages — the governor's
    /// eviction skips pinned turns so it can never free a page an in-flight launch
    /// dereferences (a scan pins every turn it touches, then unpins after launch).
    pinned: u32,
}

/// Paged VRAM gallery arena. Clone-free; share via `Arc<GalleryArena>`.
///
/// **Lock order:** `residency` → the slot arenas' own lock. A page claim and a
/// `PageRun` drop take the slot arenas' lock under `residency`; nothing in the slot
/// arenas ever calls back into the gallery, so the two never deadlock.
pub struct GalleryArena {
    residency: Mutex<HashMap<StreamId, ResidentTurn>>,
    lru_clock: AtomicU64,
    /// Bumped on every residency mutation (insert / evict / drop). The per-scan
    /// index cache is valid only while this is unchanged — any mutation could move
    /// a page, so an unchanged generation guarantees the cached device addresses
    /// still hold.
    residency_gen: AtomicU64,
    /// This arena's device tensor capabilities `(b1 BMMA, INT8 IMMA)`, queried
    /// once on first scan. Cached PER ARENA (not per process) so heterogeneous
    /// multi-GPU setups — e.g. mixed Ada/Blackwell — resolve each arena's
    /// backend ladder against its own device.
    tensor_caps: OnceLock<(bool, bool)>,
    /// Per-scan indices (page_ptr / pos_map / case / seg prefixes) keyed by segment
    /// fingerprint, reused when the same segment set is rescanned under an
    /// unchanged residency generation — skipping the O(scanned-tokens) rebuild each
    /// reprojection. Keyed (not a single slot) so the several belief groups scanned
    /// per reprojection don't evict each other; bounded by [`INDEX_CACHE_CAP`].
    index_cache: Mutex<HashMap<u64, scan::CachedIndex>>,
    device: Device,
    wpt: usize,
    n_groups: usize,
    page_bytes: u64,
}

impl GalleryArena {
    /// A gallery arena on `device` for the locked folded-signature geometry
    /// (`wpt` words per token, `n_groups` layer-groups). For the production fold
    /// that is `wpt = 24`, `n_groups = 3`.
    pub fn new(device: &Device, wpt: usize, n_groups: usize) -> Result<Self> {
        assert!(wpt > 0 && n_groups > 0 && wpt.is_multiple_of(n_groups));
        if !matches!(device, Device::Cuda(_)) {
            candle::bail!("gallery arena requires a CUDA device");
        }
        let pu64 = page_u64(wpt);
        Ok(Self {
            residency: Mutex::new(HashMap::new()),
            lru_clock: AtomicU64::new(0),
            residency_gen: AtomicU64::new(0),
            index_cache: Mutex::new(HashMap::new()),
            tensor_caps: OnceLock::new(),
            device: device.clone(),
            wpt,
            n_groups,
            page_bytes: (pu64 * std::mem::size_of::<u64>()) as u64,
        })
    }

    /// The CUDA device the arena's slabs live on.
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Words per token (signature width).
    #[inline]
    pub fn wpt(&self) -> usize {
        self.wpt
    }

    /// Layer-groups per signature.
    #[inline]
    pub fn n_groups(&self) -> usize {
        self.n_groups
    }

    /// Reservation bytes the gallery's arenas hold — whole regions, whatever is in
    /// them. What the gallery denies the rest of the span, and so what the memory
    /// report accounts to it.
    pub fn resident_bytes(&self) -> u64 {
        (arena_regions(&self.device, SlotTenant::Gallery) * SpanRegion::bytes()) as u64
    }

    /// Bytes of the pages resident turns hold — what eviction can give back. Less
    /// than [`Self::resident_bytes`] by every arena's free slots and unused tail.
    ///
    /// **The ceiling is measured in this, not in regions.** Evicting a turn frees
    /// its pages, but a region goes back only when every page in its arena has
    /// gone, so a ceiling on regions can stay breached however many turns are
    /// evicted — and would then evict the whole gallery, working set and all, to
    /// satisfy a figure eviction does not directly move.
    pub fn page_bytes(&self) -> u64 {
        arena_held_bytes(&self.device, SlotTenant::Gallery) as u64
    }

    /// Number of turns currently resident.
    pub fn resident_turns(&self) -> usize {
        self.residency
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Pages this arena's resident turns hold — its own share, where
    /// [`Self::page_bytes`] counts every gallery page on the device.
    #[cfg(test)]
    fn live_pages(&self) -> usize {
        self.residency
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|rt| rt.run.slots.len())
            .sum()
    }

    /// Claim and upload a turn's pages, returning the run.
    ///
    /// One claim for the whole turn, so the arena window opens at most once however
    /// many new arenas it needs. A refusal means the span itself is full, and it
    /// **is** an error: routine growth is bounded well before this by the gallery's
    /// own ceiling (`evict_to_cap_locked`, run before anything is admitted), so
    /// reaching a refusal is the KV side having no region to spare for a gallery
    /// that is already inside its budget.
    fn alloc_and_upload(&self, sigs: &[WideQSig]) -> Result<PageRun> {
        let host_pages = transpose_to_pages(sigs, self.wpt, self.n_groups);
        if host_pages.is_empty() {
            return Ok(PageRun { slots: Vec::new() });
        }
        let slots = claim_arena_slots(
            &self.device,
            SlotTenant::Gallery,
            self.page_bytes as usize,
            host_pages.len(),
        )?;
        for (slot, page_words) in slots.iter().zip(&host_pages) {
            page_io::write_page(&self.device, slot, page_words)?;
        }
        Ok(PageRun { slots })
    }

    /// Ensure a turn is resident under `fingerprint` (holding the residency
    /// guard), returning its pages' device **addresses** (page order). A matching
    /// fingerprint is a hit; a mismatch (or absence) frees any stale pages and
    /// re-uploads. When `pin`, the turn's pin count is bumped **atomically with
    /// residency** so a concurrent [`evict_lru`](Self::evict_lru) can never free it
    /// between here and the launch that reads its pages. The addresses are
    /// resolved while the residency lock is still held, so the gids cannot be
    /// recycled out from under them.
    fn ensure_locked(
        &self,
        res: &mut HashMap<StreamId, ResidentTurn>,
        sid: StreamId,
        sigs: &[WideQSig],
        fingerprint: u64,
        pin: bool,
    ) -> Result<Vec<u64>> {
        debug_assert!(
            sigs.first().map(|s| s.words.len()).unwrap_or(self.wpt) == self.wpt,
            "gallery sig width {} != arena wpt {} — folded-geometry mismatch (wrong \
             head_dim?); every token would be dropped and the scan silently zeroed",
            sigs.first().map(|s| s.words.len()).unwrap_or(0),
            self.wpt
        );
        // Keep the arena under its own ceiling before admitting anything new.
        // This is where gallery growth is bounded: `alloc_and_upload` adds a
        // slab whenever the page pool is empty, and nothing else ever shrinks
        // the arena. Doing it here — under `res`, before `inner` is taken —
        // respects the residency→inner lock order and skips pins, so an active
        // scan's working set is never evicted out from under it.
        self.evict_to_cap_locked(res);
        let lru = self.lru_clock.fetch_add(1, Ordering::Relaxed);
        // Addresses are resolved while `res` is still held: evicting or replacing
        // a run needs the residency lock, so its slots can't be freed in this
        // window.
        match res.get_mut(&sid) {
            Some(rt) if rt.fingerprint == fingerprint => {
                rt.lru = lru;
                if pin {
                    rt.pinned += 1;
                }
                Ok(rt.run.addrs())
            }
            _ => self.replace_locked(res, sid, sigs, fingerprint, pin, lru),
        }
    }

    /// The miss / stale-fingerprint branch of [`ensure_locked`](Self::ensure_locked):
    /// free the old run and upload fresh pages, returning the new pages' addresses.
    fn replace_locked(
        &self,
        res: &mut HashMap<StreamId, ResidentTurn>,
        sid: StreamId,
        sigs: &[WideQSig],
        fingerprint: u64,
        pin: bool,
        lru: u64,
    ) -> Result<Vec<u64>> {
        if let Some(old) = res.remove(&sid) {
            // The scan thread always unpins before the next ensure on that thread,
            // so a replaced entry is never pinned. If this ever fires, an in-flight
            // scan's pages are about to be recycled — the arena would need a
            // multi-version turn to stay safe under concurrent re-seal + scan.
            debug_assert!(
                old.pinned == 0,
                "re-seal of a turn pinned by an active scan — pages would be freed \
                 under an in-flight kernel"
            );
            drop(old); // frees the old run's pages before the fresh upload
        }
        // Bump the generation NOW — the pages are freed even if the upload below
        // fails, so the index cache (keyed on the generation) must invalidate
        // regardless of success.
        self.residency_gen.fetch_add(1, Ordering::Relaxed);
        let run = self.alloc_and_upload(sigs)?;
        let addrs = run.addrs();
        res.insert(
            sid,
            ResidentTurn {
                fingerprint,
                run,
                n_tokens: sigs.len(),
                lru,
                pinned: u32::from(pin),
            },
        );
        Ok(addrs)
    }

    /// Ensure a turn is resident under `fingerprint`, returning its pages' device
    /// addresses (page order). A matching fingerprint is a hit (no upload); a
    /// mismatch (or absence) frees any stale pages and re-uploads exactly this
    /// turn — the delta a seal produces. `sigs` must be the turn's full window.
    /// Does NOT pin — used off the scan hot path (and by tests).
    pub fn ensure_resident(
        &self,
        sid: StreamId,
        sigs: &[WideQSig],
        fingerprint: u64,
    ) -> Result<Vec<u64>> {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        self.ensure_locked(&mut res, sid, sigs, fingerprint, false)
    }

    /// Like [`ensure_resident`](Self::ensure_resident) but **pins** the turn for
    /// the duration of a scan. Every pin must be balanced by an
    /// [`unpin`](Self::unpin). Used by the paged scan's index builder.
    pub(super) fn scan_ensure(
        &self,
        sid: StreamId,
        sigs: &[WideQSig],
        fingerprint: u64,
    ) -> Result<Vec<u64>> {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        self.ensure_locked(&mut res, sid, sigs, fingerprint, true)
    }

    /// Release one pin taken by [`scan_ensure`](Self::scan_ensure).
    pub(super) fn unpin(&self, sid: StreamId) {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(rt) = res.get_mut(&sid) {
            rt.pinned = rt.pinned.saturating_sub(1);
        }
    }

    /// Drop a turn's residency if present (e.g. its timeline was dropped). Frees
    /// its pages. No-op if absent or pinned by an active scan.
    pub fn drop_turn(&self, sid: StreamId) {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        if res.get(&sid).map(|rt| rt.pinned == 0).unwrap_or(false) {
            res.remove(&sid);
            self.residency_gen.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The current residency generation. The per-scan index cache reuses its
    /// built index only while this is unchanged (see [`scan::CachedIndex`]).
    #[inline]
    pub(super) fn residency_gen(&self) -> u64 {
        self.residency_gen.load(Ordering::Relaxed)
    }

    /// Reuse the cached index if it matches `fingerprint` AND its residency
    /// generation still holds AND every referenced turn is still resident — in
    /// which case this **pins** them (bumping their LRU) for the scan and returns
    /// the shared index. Returns `None` on any miss (caller rebuilds).
    fn reuse_index(&self, fingerprint: u64) -> Option<Arc<scan::PagedIndex>> {
        let (gen, idx) = {
            let cache = self.index_cache.lock().unwrap_or_else(|e| e.into_inner());
            let ci = cache.get(&fingerprint)?;
            (ci.gen, ci.idx.clone())
        };
        // Generation check + pin, all under the residency lock so a concurrent
        // eviction can neither slip in nor free a page the launch will read.
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        if self.residency_gen.load(Ordering::Relaxed) != gen {
            return None;
        }
        // A matching generation guarantees every referenced turn is still resident
        // at the same pages, but verify BEFORE pinning any — so a violated
        // invariant degrades to a safe rebuild rather than launching a kernel on
        // stale page addresses (never a partial pin over a missing turn).
        if idx.pinned_sids.iter().any(|sid| !res.contains_key(sid)) {
            return None;
        }
        let lru = self.lru_clock.fetch_add(1, Ordering::Relaxed);
        for &sid in &idx.pinned_sids {
            if let Some(rt) = res.get_mut(&sid) {
                rt.pinned += 1;
                rt.lru = lru;
            }
        }
        Some(idx)
    }

    /// Cache `idx` (built at generation `gen`) under `fingerprint` for reuse.
    /// Bounded: a full flush on overflow is fine (rare, and entries invalidate
    /// wholesale whenever the generation moves anyway).
    fn store_index(&self, fingerprint: u64, gen: u64, idx: Arc<scan::PagedIndex>) {
        let mut cache = self.index_cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() >= INDEX_CACHE_CAP && !cache.contains_key(&fingerprint) {
            cache.clear();
        }
        cache.insert(fingerprint, scan::CachedIndex { gen, idx });
    }

    /// Pack the gallery's pages toward the low end of the span — the two-cursor
    /// pass the KV pools and the recurrent state run, applied to page slots.
    ///
    /// # Why this is needed even when the gallery reads nearly full
    ///
    /// A snapshot of a *growing* gallery is dense: pages are claimed in runs and
    /// nothing has been freed yet. The scattered state is what churn produces —
    /// runs are **variable length** (a short turn is one page, a long one
    /// hundreds) and they are freed in a different order than they were claimed
    /// (LRU eviction, and a re-seal freeing a turn's old run mid-corpus). That is
    /// the classic external-fragmentation generator, and its effect is not lost
    /// capacity but a **held frontier**: a handful of live pages in a high arena
    /// denies the weight side every region below them. Measuring a young corpus
    /// and concluding there is nothing to pack is measuring the wrong phase.
    ///
    /// # What makes the move safe
    ///
    /// - **A page copy is exact.** Its bytes are folded signature words and encode
    ///   nothing about their own address — unlike a `KvHead` record, whose bytes
    ///   *are* addresses — so there is no fill step and no minting.
    /// - **Pinned turns are never moved.** A scan pins every turn it reads and
    ///   unpins only after its launch has synchronised, so a pinned turn's pages
    ///   are being dereferenced right now. Their sources stay put and the
    ///   destinations claimed for them go back unused.
    /// - **The generation is bumped when anything moved.** [`scan::PagedIndex`]
    ///   caches raw page addresses and `reuse_index` revalidates them only by an
    ///   unchanged `residency_gen`. A page that moved without the bump would hand
    ///   the scan kernel a stale address — the one failure here that surfaces as
    ///   quietly wrong retrieval rather than as a fault.
    ///
    /// Planning happens **before** the residency lock is taken, because a claim
    /// opens the arena window and quiesces the device; holding `residency` across
    /// that would block every scan. A turn evicted in that gap simply loses its
    /// move.
    pub fn compact(&self, max_moves: usize) -> Result<GalleryCompaction> {
        let regions_before = arena_regions(&self.device, SlotTenant::Gallery);
        let moves = plan_slot_moves(
            &self.device,
            SlotTenant::Gallery,
            self.page_bytes as usize,
            max_moves,
        )?;
        let planned = moves.len();
        let mut by_src: HashMap<u64, ArenaSlot> =
            moves.into_iter().map(|m| (m.src, m.dst)).collect();
        let mut moved = 0usize;
        let outcome = {
            let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
            // **The generation is bumped on the way out, failure included.** A copy
            // that errors part-way has already reassigned every slot before it, so
            // returning straight to the caller would leave a cached `PagedIndex`
            // vouching for addresses those turns no longer occupy — the one failure
            // here that surfaces as quietly wrong retrieval rather than as a fault.
            // The result is carried past the bump instead of propagated through it.
            let mut outcome = Ok(());
            'pass: for rt in res.values_mut() {
                if rt.pinned != 0 {
                    continue;
                }
                for slot in &mut rt.run.slots {
                    let Some(dst) = by_src.remove(&slot.ptr()) else {
                        continue;
                    };
                    if let Err(e) = slot.copy_into(&dst, self.page_bytes as usize, &self.device) {
                        outcome = Err(e);
                        break 'pass;
                    }
                    *slot = dst;
                    moved += 1;
                }
            }
            if moved > 0 {
                // Under the residency lock, with the moves: a scan taking the lock
                // after this sees both the new addresses and the new generation.
                self.residency_gen.fetch_add(1, Ordering::Relaxed);
            }
            outcome
        };
        outcome?;
        // Destinations nothing claimed go back to their arenas.
        drop(by_src);
        Ok(GalleryCompaction {
            planned,
            moved,
            regions_before,
            regions_after: arena_regions(&self.device, SlotTenant::Gallery),
        })
    }

    /// Evict least-recently-used turns until at least `want` bytes are freed.
    /// Returns the bytes freed. This is the governor's cheap-rung relief: the
    /// dropped pages recycle and the turns rebuild on demand from the substrate
    /// blob. Pinned turns (an active scan's working set) are never evicted.
    pub fn evict_lru(&self, want: u64) -> u64 {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        self.evict_lru_locked(&mut res, want)
    }

    /// [`evict_lru`](Self::evict_lru) under a residency guard the caller holds.
    ///
    /// Split out so admission can bound the arena *before* claiming pages
    /// (see [`Self::cap_bytes`]); the residency mutex is not reentrant, so
    /// `ensure_locked` cannot call the public entry point.
    fn evict_lru_locked(&self, res: &mut HashMap<StreamId, ResidentTurn>, want: u64) -> u64 {
        // Order candidates by LRU ascending (oldest first), skipping pins.
        let mut cands: Vec<(u64, StreamId, usize)> = res
            .iter()
            .filter(|(_, rt)| rt.pinned == 0)
            .map(|(sid, rt)| (rt.lru, *sid, rt.n_tokens))
            .collect();
        cands.sort_by_key(|(lru, _, _)| *lru);
        let mut freed = 0u64;
        for (_, sid, n_tokens) in cands {
            if freed >= want {
                break;
            }
            res.remove(&sid); // drops the run → frees pages
            self.residency_gen.fetch_add(1, Ordering::Relaxed);
            freed += pages_for(n_tokens) as u64 * self.page_bytes;
        }
        freed
    }

    /// The arena's own VRAM ceiling, in bytes (`ZEN_GALLERY_CAP_MB`, default
    /// 512 MiB).
    ///
    /// **This is what bounds gallery growth.** `alloc_and_upload` claims new
    /// arenas whenever the gallery's run out, and the arena never evicts itself,
    /// so without a ceiling here the only limit was an outside `evict_lru` call
    /// from the scheduler's KV-pressure relief — which shed belief-scan
    /// residency the next scan had to rebuild from the substrate, on every
    /// pressure episode. The cheap limit is the one that never discards a
    /// working set.
    ///
    /// Measured against [`Self::page_bytes`] — the pages eviction frees — for the
    /// reason given there. Enforced at admission in `ensure_locked`, where no
    /// lock is held that eviction needs.
    pub fn cap_bytes(&self) -> u64 {
        static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        *CAP.get_or_init(|| {
            let mb = std::env::var("ZEN_GALLERY_CAP_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(512);
            tracing::info!(cap_mb = mb, "gallery arena VRAM ceiling");
            mb * 1024 * 1024
        })
    }

    /// Evict oldest turns until the arena is back under [`Self::cap_bytes`].
    ///
    /// Returns bytes freed. Pinned turns are skipped, so a scan's working set
    /// larger than the cap is served rather than refused — the cap bounds
    /// *growth*, it does not fail requests.
    fn evict_to_cap_locked(&self, res: &mut HashMap<StreamId, ResidentTurn>) -> u64 {
        let cap = self.cap_bytes();
        let held = self.page_bytes();
        if held <= cap {
            return 0;
        }
        self.evict_lru_locked(res, held - cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: (0..24)
                .map(|w| fill.wrapping_add((w as u64) << 8))
                .collect(),
        }
    }

    fn sid(n: u64) -> StreamId {
        crate::persistence::content_hash::turn_stream_id(1, n as u32)
    }

    /// Round-trip: upload a turn, read its pages back, verify the group-major
    /// transpose survived the H2D exactly (raw bytes, not a threshold).
    #[test]
    fn resident_pages_roundtrip_group_major() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU — skip
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs: Vec<WideQSig> = (0..40).map(|t| sig((t as u64) << 40)).collect(); // 2 pages
        let fp = 0xDEADBEEF;
        let addrs = arena.ensure_resident(sid(0), &sigs, fp).unwrap();
        assert_eq!(addrs.len(), 2, "40 tokens → 2 pages");
        assert_eq!(arena.resident_turns(), 1);

        // Read pages back and check the transpose token-by-token.
        let expect = transpose_to_pages(&sigs, 24, 3);
        let res = arena.residency.lock().unwrap();
        let rt = res.get(&sid(0)).unwrap();
        assert_eq!(
            rt.run.addrs(),
            addrs,
            "the addresses handed out are the run's"
        );
        for (p, slot) in rt.run.slots.iter().enumerate() {
            let got = page_io::read_page(&device, slot, page_u64(24)).unwrap();
            assert_eq!(got, expect[p], "page {p} bytes differ after H2D");
        }
    }

    /// A page is one slot of the gallery's own arenas, a 6 KiB stride: 2,730 to
    /// a region, and none of them in another tenant's arena.
    #[test]
    fn pages_are_gallery_tenant_slots() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs: Vec<WideQSig> = (0..64).map(|t| sig(t as u64)).collect(); // 2 pages
        arena.ensure_resident(sid(0), &sigs, 1).unwrap();
        let res = arena.residency.lock().unwrap();
        let run = &res.get(&sid(0)).unwrap().run;
        assert_eq!(run.slots.len(), 2);
        for slot in &run.slots {
            assert_eq!(slot.stride(), 6144);
            assert_eq!(slot.tenant(), SlotTenant::Gallery);
        }
        assert!(arena.resident_bytes() >= SpanRegion::bytes() as u64);
        assert!(arena.page_bytes() >= 2 * 6144);
    }

    /// A matching fingerprint is a hit: same addresses, no new pages allocated.
    #[test]
    fn fingerprint_hit_reuses_pages() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs: Vec<WideQSig> = (0..10).map(|t| sig(t as u64)).collect();
        let a1 = arena.ensure_resident(sid(0), &sigs, 7).unwrap();
        let live1 = arena.live_pages();
        let a2 = arena.ensure_resident(sid(0), &sigs, 7).unwrap();
        let live2 = arena.live_pages();
        assert_eq!(a1, a2, "hit must return identical page addresses");
        assert_eq!(live1, live2, "hit must not allocate new pages");
        assert_eq!(arena.resident_turns(), 1);
    }

    /// A changed fingerprint (a re-seal) frees the old pages and re-uploads.
    #[test]
    fn fingerprint_miss_reuploads_and_recycles() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs: Vec<WideQSig> = (0..10).map(|t| sig(t as u64)).collect();
        let _ = arena.ensure_resident(sid(0), &sigs, 1).unwrap();
        let live1 = arena.live_pages();
        // Re-seal to a bigger window under a new fingerprint.
        let sigs2: Vec<WideQSig> = (0..40).map(|t| sig((t + 100) as u64)).collect();
        let _ = arena.ensure_resident(sid(0), &sigs2, 2).unwrap();
        let live2 = arena.live_pages();
        // Old 1 page freed, 2 new pages live → net live == 2.
        assert_eq!(live1, 1);
        assert_eq!(live2, 2);
        assert_eq!(arena.resident_turns(), 1);
    }

    /// Eviction frees LRU turns and skips the pinned working set.
    #[test]
    fn evict_lru_skips_pins() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        for t in 0..4u64 {
            let sigs: Vec<WideQSig> = (0..32).map(|k| sig(t * 1000 + k)).collect(); // 1 page each
            arena.ensure_resident(sid(t), &sigs, t + 1).unwrap();
        }
        assert_eq!(arena.resident_turns(), 4);
        // Pin turns 2 and 3 via a scan-style ensure (hit → pins; fp = t+1).
        let dummy: Vec<WideQSig> = (0..32).map(sig).collect();
        arena.scan_ensure(sid(2), &dummy, 3).unwrap();
        arena.scan_ensure(sid(3), &dummy, 4).unwrap();
        // Evict everything possible — the pinned two must survive.
        let freed = arena.evict_lru(u64::MAX);
        assert!(freed > 0);
        assert_eq!(
            arena.resident_turns(),
            2,
            "only the two pinned turns survive"
        );
        {
            let res = arena.residency.lock().unwrap();
            assert!(res.contains_key(&sid(2)) && res.contains_key(&sid(3)));
        }
        // Unpin → they become evictable again.
        arena.unpin(sid(2));
        arena.unpin(sid(3));
        let freed2 = arena.evict_lru(u64::MAX);
        assert!(freed2 > 0);
        assert_eq!(arena.resident_turns(), 0, "unpinned turns now evict");
    }

    /// **A compaction moves pages and changes nothing a reader can see.**
    ///
    /// Scatter first — upload several turns, then drop alternate ones, which is
    /// the churn shape the pass exists for (variable-length runs freed out of
    /// claim order) — then pack, and read every surviving page back. Compared
    /// against the expected group-major transpose as **raw words**, not a
    /// tolerance: a page is bytes, and a copy that alters one is a corrupted
    /// signature, not an approximation.
    #[test]
    fn compaction_moves_pages_without_changing_them() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        // Runs of different lengths, so the freed holes are ragged.
        let sigs = |t: u64, n: usize| -> Vec<WideQSig> {
            (0..n).map(|k| sig((t << 40) + k as u64)).collect()
        };
        let plan = [(0u64, 40usize), (1, 80), (2, 33), (3, 96), (4, 64)];
        for &(t, n) in &plan {
            arena.ensure_resident(sid(t), &sigs(t, n), t + 1).unwrap();
        }
        // Free every other turn: holes below live pages.
        arena.drop_turn(sid(1));
        arena.drop_turn(sid(3));
        let live_before = arena.live_pages();

        let report = arena.compact(0).unwrap();
        // **The test is worthless unless the pass actually moved something.** The
        // holes above are below live pages by construction, so a pass that plans
        // nothing here means the walk never saw them.
        assert!(
            report.moved > 0 && report.moved == report.planned,
            "expected real moves, got {report:?}",
        );
        assert_eq!(arena.live_pages(), live_before, "a move frees no page");

        // Every surviving turn still reads back exactly its own transpose.
        let res = arena.residency.lock().unwrap();
        for &(t, n) in &plan {
            if t == 1 || t == 3 {
                assert!(!res.contains_key(&sid(t)), "dropped turn is gone");
                continue;
            }
            let expect = transpose_to_pages(&sigs(t, n), 24, 3);
            let run = &res.get(&sid(t)).expect("survivor").run;
            assert_eq!(run.slots.len(), expect.len());
            for (p, slot) in run.slots.iter().enumerate() {
                let got = page_io::read_page(&device, slot, page_u64(24)).unwrap();
                assert_eq!(got, expect[p], "turn {t} page {p} after compaction");
            }
        }
    }

    /// A pinned turn is never moved — a scan is dereferencing its pages — and the
    /// generation moves only when something actually did.
    #[test]
    fn compaction_skips_pins_and_bumps_the_generation_only_on_a_move() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let rows: Vec<WideQSig> = (0..64).map(sig).collect();
        for t in 0..4u64 {
            arena.ensure_resident(sid(t), &rows, t + 1).unwrap();
        }
        arena.drop_turn(sid(0));
        arena.drop_turn(sid(2));
        // Pin one survivor; record where its pages sit.
        arena.scan_ensure(sid(1), &rows, 2).unwrap();
        let pinned_addrs: Vec<u64> = {
            let res = arena.residency.lock().unwrap();
            res.get(&sid(1)).unwrap().run.addrs()
        };

        let gen_before = arena.residency_gen();
        let report = arena.compact(0).unwrap();
        let res = arena.residency.lock().unwrap();
        assert_eq!(
            res.get(&sid(1)).unwrap().run.addrs(),
            pinned_addrs,
            "a pinned turn's pages must not move under an in-flight scan",
        );
        drop(res);
        if report.moved > 0 {
            assert!(
                arena.residency_gen() > gen_before,
                "a cached index must be invalidated when a page moves",
            );
        } else {
            assert_eq!(arena.residency_gen(), gen_before, "no move, no bump");
        }
    }

    /// A gallery with nothing to pack plans nothing and bumps nothing — the pass
    /// has to be free to run every relief episode.
    #[test]
    fn compaction_of_a_packed_gallery_is_a_no_op() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let rows: Vec<WideQSig> = (0..32).map(sig).collect();
        for t in 0..3u64 {
            arena.ensure_resident(sid(t), &rows, t + 1).unwrap();
        }
        let gen_before = arena.residency_gen();
        let report = arena.compact(0).unwrap();
        assert_eq!((report.planned, report.moved), (0, 0));
        assert_eq!(arena.residency_gen(), gen_before);
    }
}
