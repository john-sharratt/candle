//! Paged VRAM arena for the wide-Q provenance gallery.
//!
//! Keeps each turn's folded signatures **resident on the GPU** between
//! reprojections so the belief scan stops re-uploading the whole corpus every
//! scan. A turn's `N` tokens occupy `ceil(N/32)` fixed 6 KiB group-major pages
//! allocated from a free-list ([`pool`]); the pages live in persistent VRAM
//! slabs ([`storage`]); a re-seal or eviction returns the pages to the pool. The
//! scan kernel reads a *paged gallery* — an array of page device addresses plus a
//! per-token page map — modelled on the paged-KV pointer interface. See
//! `docs/paged_gallery_arena.md`.
//!
//! The arena is a **device-level** resource (one per GPU, scheduler-owned): its
//! residency map is keyed by the global [`StreamId`] of a turn, so every
//! conversation on the device shares it. The warm/cold tiers already exist — the
//! substrate `wide_q_sigs` blob and its `decoded_wide_sig` `Arc` memo — so the
//! arena owns only the hot VRAM tier and rebuilds an evicted turn on demand.

mod pages;
mod pool;
mod scan;
mod storage;

pub use pages::{page_u64, pages_for, transpose_to_pages, PAGE_TOKENS};
pub use scan::{PagedSegment, PagedWindow};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use candle::{Device, Result};

use crate::persistence::streams::StreamId;

use super::WideQSig;
use pool::{slab_pages, PagePool};
use storage::GalleryStorage;

/// Max distinct segment-set indices cached at once — a handful of belief groups
/// per reprojection, so this comfortably covers a whole reproject's scans.
const INDEX_CACHE_CAP: usize = 16;

/// The physical arena state behind one lock: the free-list and the device slabs.
struct ArenaInner {
    pool: PagePool,
    storage: GalleryStorage,
}

impl ArenaInner {
    /// Device addresses of a run's pages, in page order.
    fn addrs(&self, gids: &[u32]) -> Vec<u64> {
        gids.iter()
            .map(|&g| {
                let (slab, pis) = self.pool.locate(g);
                self.storage.page_addr(slab, pis)
            })
            .collect()
    }
}

/// A turn's owned run of page slots. Dropping it returns the slots to the pool
/// (RAII free) — so evicting a turn is just dropping its [`ResidentTurn`].
struct PageRun {
    inner: Arc<Mutex<ArenaInner>>,
    gids: Vec<u32>,
}

impl Drop for PageRun {
    fn drop(&mut self) {
        if self.gids.is_empty() {
            return;
        }
        // Never nested under a held `inner` lock — see the lock-order note on
        // `GalleryArena`.
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.pool.free_run(&self.gids);
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
/// **Lock order:** `residency` → `inner`. `PageRun::drop` and every address
/// lookup take only `inner`; `ensure_resident`/`evict_lru` take `residency` and
/// may then take `inner`. No path ever locks `residency` while holding `inner`,
/// so the two never deadlock.
pub struct GalleryArena {
    inner: Arc<Mutex<ArenaInner>>,
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
        let pu64 = page_u64(wpt);
        let sp = slab_pages(pu64);
        let storage = GalleryStorage::new(device, pu64, sp)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(ArenaInner {
                pool: PagePool::new(sp),
                storage,
            })),
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

    /// VRAM currently held by the arena's slabs (for the governor's `evictable`).
    pub fn resident_bytes(&self) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .storage
            .resident_bytes()
    }

    /// Number of turns currently resident.
    pub fn resident_turns(&self) -> usize {
        self.residency
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Allocate and upload a turn's pages, returning the run. Holds `inner` for
    /// the alloc + H2D; the transpose is done first, off the lock.
    fn alloc_and_upload(&self, sigs: &[WideQSig]) -> Result<PageRun> {
        let host_pages = transpose_to_pages(sigs, self.wpt, self.n_groups);
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let inner = &mut *guard;
        let mut gids = Vec::with_capacity(host_pages.len());
        for page_words in &host_pages {
            let gid = loop {
                if let Some(g) = inner.pool.try_alloc_one() {
                    break g;
                }
                inner.storage.add_slab()?;
                inner.pool.grow_one_slab();
            };
            let (slab, pis) = inner.pool.locate(gid);
            inner.storage.write_page(slab, pis, page_words)?;
            gids.push(gid);
        }
        Ok(PageRun {
            inner: self.inner.clone(),
            gids,
        })
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
        let gids = if let Some(rt) = res.get_mut(&sid) {
            if rt.fingerprint == fingerprint {
                rt.lru = lru;
                if pin {
                    rt.pinned += 1;
                }
                rt.run.gids.clone()
            } else {
                self.replace_locked(res, sid, sigs, fingerprint, pin, lru)?
            }
        } else {
            self.replace_locked(res, sid, sigs, fingerprint, pin, lru)?
        };
        // Resolve addresses while `res` is still held: `evict_lru`/`PageRun::drop`
        // both need the residency lock, so the gids can't be freed in this window.
        Ok(self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .addrs(&gids))
    }

    /// The miss / stale-fingerprint branch of [`ensure_locked`](Self::ensure_locked):
    /// free the old run and upload fresh pages, returning the new gids.
    fn replace_locked(
        &self,
        res: &mut HashMap<StreamId, ResidentTurn>,
        sid: StreamId,
        sigs: &[WideQSig],
        fingerprint: u64,
        pin: bool,
        lru: u64,
    ) -> Result<Vec<u32>> {
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
        let gids = run.gids.clone();
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
        Ok(gids)
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
    /// Split out so admission can bound the arena *before* growing a slab
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
    /// **This is what bounds gallery growth.** `alloc_and_upload` adds a slab
    /// whenever the page pool is empty and the arena never evicts itself, so
    /// without a ceiling here the only limit was an outside `evict_lru` call
    /// from the scheduler's KV-pressure relief — a signal this arena cannot
    /// move (its slabs come from the CUDA pool and are never returned), so that
    /// call fired on every pressure episode and shed belief-scan residency the
    /// next scan had to rebuild from the substrate.
    ///
    /// Enforced at admission in `ensure_locked`, where no lock is held that
    /// eviction needs.
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
        let resident = self.resident_bytes();
        if resident <= cap {
            return 0;
        }
        self.evict_lru_locked(res, resident - cap)
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
        let guard = arena.inner.lock().unwrap();
        let expect = transpose_to_pages(&sigs, 24, 3);
        for (p, _addr) in addrs.iter().enumerate() {
            // recover the gid from the resident run for this page index
            let got = {
                let res = arena.residency.lock().unwrap();
                let rt = res.get(&sid(0)).unwrap();
                let (slab, pis) = guard.pool.locate(rt.run.gids[p]);
                guard.storage.read_page(slab, pis).unwrap()
            };
            assert_eq!(got, expect[p], "page {p} bytes differ after H2D");
        }
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
        let live1 = arena.inner.lock().unwrap().pool.live();
        let a2 = arena.ensure_resident(sid(0), &sigs, 7).unwrap();
        let live2 = arena.inner.lock().unwrap().pool.live();
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
        let live1 = arena.inner.lock().unwrap().pool.live();
        // Re-seal to a bigger window under a new fingerprint.
        let sigs2: Vec<WideQSig> = (0..40).map(|t| sig((t + 100) as u64)).collect();
        let _ = arena.ensure_resident(sid(0), &sigs2, 2).unwrap();
        let live2 = arena.inner.lock().unwrap().pool.live();
        // Old 1 page freed, 2 new pages live → net live == 2, and the freed
        // page recycled (so capacity didn't grow by the full 3).
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
}
