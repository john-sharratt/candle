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
//! conversation on the device shares it. The warm/cold tier already exists — the
//! substrate's encoded `wide_q_sigs` blob, read through a
//! [`SigWindow`](super::SigWindow) — so the arena owns only the hot VRAM tier
//! and rebuilds an evicted turn on demand, decoding it for the upload only.

pub(crate) mod eviction;
mod page_io;
mod pages;
mod probe_rows;
mod scan;
pub mod scan_split;
mod token_votes;
mod turn_sigs;

pub use pages::{page_u64, pages_for, transpose_to_pages, PAGE_TOKENS};
pub use scan::{PagedSegment, PagedWindow};
pub use token_votes::TokenVotes;
pub use turn_sigs::TurnSigs;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use std::result::Result as StdResult;

use candle::{Device, Result};
use candle_nn::kv_cache::{
    arena_held_bytes, arena_regions, claim_arena_slots, plan_slot_moves, ArenaSlot, SlotTenant,
    SpanRegion,
};

use crate::persistence::streams::StreamId;

use eviction::{index_cache_evictions, past_keep, Resident};

/// Max distinct segment-set indices cached at once — a handful of belief groups
/// per reprojection, so this comfortably covers a whole reproject's scans.
const INDEX_CACHE_CAP: usize = 16;

/// Device bytes the cached indices may hold together. An index's arrays live in
/// the driver pool — outside the span and the gallery's own ceiling — at about
/// 8 bytes per scanned token, so the 7.4M-token `code_reading` scan alone holds
/// ~60 MB; sixteen entries left unbounded could hold a gigabyte nothing else
/// accounts for. This covers a working set of a couple of conversations' large
/// groups plus the small ones.
const INDEX_CACHE_BYTES: u64 = 256 << 20;

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
    /// Identifies `run`'s current page addresses: fresh on every upload and
    /// every compaction move. A cached scan index is valid exactly while every
    /// turn it references still carries the run id it was built against.
    run_id: u64,
    n_tokens: usize,
    lru: u64,
    /// When a scan last used it.
    used: Instant,
    /// How long between the scans that use it, averaged — with `used`, how
    /// often keeping it resident spares a re-upload (see [`eviction`]). `None`
    /// until a second scan has used it.
    interval: Option<Duration>,
    /// Non-zero while a scan is reading this turn's pages — the governor's
    /// eviction skips pinned turns so it can never free a page an in-flight launch
    /// dereferences (a scan pins every turn it touches, then unpins after launch).
    pinned: u32,
}

impl ResidentTurn {
    /// A scan used it at `now`: its interval takes in the time since the last.
    fn touch(&mut self, now: Instant) {
        let since = now.saturating_duration_since(self.used);
        self.interval = Some(match self.interval {
            Some(i) => i.mul_f64(1.0 - INTERVAL_WEIGHT) + since.mul_f64(INTERVAL_WEIGHT),
            None => since,
        });
        self.used = now;
    }
}

/// Weight of the newest gap between scans in a turn's interval.
const INTERVAL_WEIGHT: f64 = 0.5;

/// Why a scan could not reuse a cached index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IndexMiss {
    /// No index was cached for the segment set — never built, evicted from
    /// the cache, or too large to cache.
    Absent,
    /// The cached index named a turn since evicted, moved or re-uploaded.
    Stale,
}

impl IndexMiss {
    pub(super) fn reason(self) -> &'static str {
        match self {
            IndexMiss::Absent => "no index cached for this segment set",
            IndexMiss::Stale => "a turn it holds was evicted, moved or re-uploaded",
        }
    }
}

/// A turn made resident: its pages' device addresses (page order), the run id
/// they belong to ([`ResidentTurn::run_id`]), and whether they were uploaded
/// now rather than found resident.
pub(super) struct Ensured {
    pub addrs: Vec<u64>,
    pub run_id: u64,
    pub uploaded: bool,
}

/// Paged VRAM gallery arena. Clone-free; share via `Arc<GalleryArena>`.
///
/// **Lock order:** `residency` → the slot arenas' own lock. A page claim and a
/// `PageRun` drop take the slot arenas' lock under `residency`; nothing in the slot
/// arenas ever calls back into the gallery, so the two never deadlock.
pub struct GalleryArena {
    residency: Mutex<HashMap<StreamId, ResidentTurn>>,
    lru_clock: AtomicU64,
    /// Bumped on every residency mutation (insert / evict / drop / move) — a
    /// count of churn, reported per scan. Not what validates a cached index:
    /// that is per turn ([`ResidentTurn::run_id`]), so an upload for one
    /// conversation does not invalidate another's index.
    residency_gen: AtomicU64,
    /// Source of [`ResidentTurn::run_id`]s.
    next_run_id: AtomicU64,
    /// `T*` in nanoseconds, `u64::MAX` for none — see [`Self::set_keep_within`].
    keep_within_ns: AtomicU64,
    /// Measured upload cost per byte, microseconds, as `f64` bits.
    upload_us_per_byte: AtomicU64,
    /// This arena's device tensor capabilities `(b1 BMMA, INT8 IMMA)`, queried
    /// once on first scan. Cached PER ARENA (not per process) so heterogeneous
    /// multi-GPU setups — e.g. mixed Ada/Blackwell — resolve each arena's
    /// backend ladder against its own device.
    tensor_caps: OnceLock<(bool, bool)>,
    /// Per-scan indices (page_ptr / pos_map / case / seg prefixes) keyed by segment
    /// fingerprint, reused when the same segment set is rescanned and every turn
    /// it references still holds the run it was built against — skipping the
    /// O(scanned-tokens) rebuild each reprojection. Keyed (not a single slot) so
    /// the several belief groups scanned per reprojection don't evict each other;
    /// bounded by [`INDEX_CACHE_CAP`].
    ///
    /// Validated per turn, not by a device-wide generation: with ingest sealing
    /// turns concurrently, a global generation moved on every scan, and a
    /// dialogue's 7.4M-token index was rebuilt from scratch each reprojection
    /// (1.8–2.2 s) for uploads that touched none of its turns.
    index_cache: Mutex<HashMap<u64, scan::CachedIndex>>,
    device: Device,
    wpt: usize,
    n_groups: usize,
    page_bytes: u64,
}

/// Weight of the newest upload in the arena's per-byte upload cost.
const UPLOAD_WEIGHT: f64 = 0.25;

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
            next_run_id: AtomicU64::new(0),
            keep_within_ns: AtomicU64::new(u64::MAX),
            upload_us_per_byte: AtomicU64::new(0),
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
    /// own keep rule (`evict_past_keep_locked`, run once before a scan's index build,
    /// a preload, and a single turn's miss), so
    /// reaching a refusal is the KV side having no region to spare for a gallery
    /// that is already inside its budget.
    fn alloc_and_upload(&self, sigs: TurnSigs<'_>) -> Result<PageRun> {
        let n_pages = pages_for(sigs.len());
        if n_pages == 0 {
            return Ok(PageRun { slots: Vec::new() });
        }
        let slots = claim_arena_slots(
            &self.device,
            SlotTenant::Gallery,
            self.page_bytes as usize,
            n_pages,
        )?;
        let stride_words = slots[0].stride() / std::mem::size_of::<u64>();
        let host_pages = sigs
            .with_decoded(|sigs| transpose_to_pages(sigs, self.wpt, self.n_groups, stride_words));
        page_io::write_pages(&self.device, &slots, &host_pages)?;
        Ok(PageRun { slots })
    }

    /// Ensure a turn is resident under `fingerprint` (holding the residency
    /// guard), returning its pages' device **addresses** (page order) and the
    /// run id they belong to ([`ResidentTurn::run_id`]). A matching
    /// fingerprint is a hit; a mismatch (or absence) frees any stale pages and
    /// re-uploads. When `pin`, the turn's pin count is bumped **atomically with
    /// residency** so a concurrent [`evict_past_keep`](Self::evict_past_keep) can never free it
    /// between here and the launch that reads its pages. The addresses are
    /// resolved while the residency lock is still held, so the gids cannot be
    /// recycled out from under them.
    ///
    /// **Makes no room.** The keep eviction walks every resident turn, so it is
    /// the caller's to run once before what it admits: a single turn's
    /// [`ensure_resident`](Self::ensure_resident) before its miss, a scan's
    /// index build and a preload once before their every turn. Its cost per
    /// ensure is what this avoids: over a 27,217-turn `code_reading` index it
    /// came to 20.6 s a build (BlackWorld, 2026-10-26), stalling every
    /// reprojection.
    fn ensure_locked(
        &self,
        res: &mut HashMap<StreamId, ResidentTurn>,
        sid: StreamId,
        sigs: TurnSigs<'_>,
        fingerprint: u64,
        pin: bool,
    ) -> Result<Ensured> {
        debug_assert!(
            sigs.first_width().unwrap_or(self.wpt) == self.wpt,
            "gallery sig width {} != arena wpt {} — folded-geometry mismatch (wrong \
             head_dim?); every token would be dropped and the scan silently zeroed",
            sigs.first_width().unwrap_or(0),
            self.wpt
        );
        let lru = self.lru_clock.fetch_add(1, Ordering::Relaxed);
        // Addresses are resolved while `res` is still held: evicting or replacing
        // a run needs the residency lock, so its slots can't be freed in this
        // window.
        match res.get_mut(&sid) {
            Some(rt) if rt.fingerprint == fingerprint => {
                rt.lru = lru;
                rt.touch(Instant::now());
                if pin {
                    rt.pinned += 1;
                }
                Ok(Ensured {
                    addrs: rt.run.addrs(),
                    run_id: rt.run_id,
                    uploaded: false,
                })
            }
            _ => self
                .replace_locked(res, sid, sigs, fingerprint, pin, lru)
                .map(|(addrs, run_id)| Ensured {
                    addrs,
                    run_id,
                    uploaded: true,
                }),
        }
    }

    /// The miss / stale-fingerprint branch of [`ensure_locked`](Self::ensure_locked):
    /// free the old run and upload fresh pages, returning the new pages' addresses.
    fn replace_locked(
        &self,
        res: &mut HashMap<StreamId, ResidentTurn>,
        sid: StreamId,
        sigs: TurnSigs<'_>,
        fingerprint: u64,
        pin: bool,
        lru: u64,
    ) -> Result<(Vec<u64>, u64)> {
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
        // Counted NOW — the pages are freed even if the upload below fails. The
        // old run id left with the removed entry, so any cached index over it is
        // already invalid.
        self.residency_gen.fetch_add(1, Ordering::Relaxed);
        let run = self.alloc_and_upload(sigs)?;
        let addrs = run.addrs();
        let run_id = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        res.insert(
            sid,
            ResidentTurn {
                fingerprint,
                run,
                run_id,
                n_tokens: sigs.len(),
                lru,
                used: Instant::now(),
                interval: None,
                pinned: u32::from(pin),
            },
        );
        Ok((addrs, run_id))
    }

    /// Ensure a turn is resident under `fingerprint`, returning its pages' device
    /// addresses (page order). A matching fingerprint is a hit (no upload); a
    /// mismatch (or absence) frees any stale pages and re-uploads exactly this
    /// turn — the delta a seal produces. `sigs` must be the turn's full window;
    /// a stored window is decoded only on the miss.
    /// Does NOT pin — used off the scan hot path (and by tests).
    ///
    /// A miss first gives up the turns past their keep time: this is where
    /// gallery growth is bounded outside a scan, since `alloc_and_upload` adds
    /// a slab whenever the page pool is empty and nothing else ever shrinks the
    /// arena. A hit admits nothing, so it makes no room.
    pub fn ensure_resident<'a>(
        &self,
        sid: StreamId,
        sigs: impl Into<TurnSigs<'a>>,
        fingerprint: u64,
    ) -> Result<Vec<u64>> {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        if res.get(&sid).is_none_or(|rt| rt.fingerprint != fingerprint) {
            self.evict_past_keep_locked(&mut res);
        }
        self.ensure_locked(&mut res, sid, sigs.into(), fingerprint, false)
            .map(|ensured| ensured.addrs)
    }

    /// [`ensure_resident`](Self::ensure_resident) for many turns at once — a
    /// gallery preload — giving up the turns past their keep time once, before
    /// the first, rather than before each miss: the pass walks every resident
    /// turn. One result per turn, in order.
    pub fn ensure_all_resident<'a>(
        &self,
        turns: impl IntoIterator<Item = (StreamId, TurnSigs<'a>, u64)>,
    ) -> Vec<Result<()>> {
        self.evict_past_keep();
        turns
            .into_iter()
            .map(|(sid, sigs, fingerprint)| {
                let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
                self.ensure_locked(&mut res, sid, sigs, fingerprint, false)
                    .map(|_| ())
            })
            .collect()
    }

    /// Like [`ensure_resident`](Self::ensure_resident) but **pins** the turn for
    /// the duration of a scan, and makes no room: the index builder runs the
    /// keep eviction once, before its first turn. Every pin must be balanced by
    /// an [`unpin`](Self::unpin). Used by the paged scan's index builder, which
    /// records the run id to validate a later reuse of its index.
    pub(super) fn scan_ensure<'a>(
        &self,
        sid: StreamId,
        sigs: impl Into<TurnSigs<'a>>,
        fingerprint: u64,
    ) -> Result<Ensured> {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        self.ensure_locked(&mut res, sid, sigs.into(), fingerprint, true)
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

    /// The residency mutation count — churn, reported per scan (see
    /// [`GalleryArena::residency_gen`]'s field).
    #[inline]
    pub(super) fn residency_gen(&self) -> u64 {
        self.residency_gen.load(Ordering::Relaxed)
    }

    /// Reuse the cached index if it matches `fingerprint` AND every turn it
    /// references is still resident on the run it was built against — in which
    /// case this **pins** them (bumping their LRU) for the scan and returns the
    /// shared index. Otherwise says why it missed — no index cached, or a turn
    /// it holds evicted, moved or re-uploaded — and the caller rebuilds.
    ///
    /// A turn whose run id still matches has not been re-uploaded, evicted or
    /// moved since the build, so every address the index holds for it is still
    /// its page. That is checked per referenced turn — O(turns), not
    /// O(tokens) — so another conversation's uploads leave this index valid.
    fn reuse_index(&self, fingerprint: u64) -> StdResult<Arc<scan::PagedIndex>, IndexMiss> {
        let idx = {
            let mut cache = self.index_cache.lock().unwrap_or_else(|e| e.into_inner());
            let entry = cache.get_mut(&fingerprint).ok_or(IndexMiss::Absent)?;
            entry.used = self.lru_clock.fetch_add(1, Ordering::Relaxed);
            entry.idx.clone()
        };
        // Check + pin, all under the residency lock so a concurrent eviction can
        // neither slip in nor free a page the launch will read. Verified BEFORE
        // pinning any, so a stale index degrades to a rebuild rather than a
        // partial pin.
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        let current = idx
            .pinned_sids
            .iter()
            .zip(&idx.run_ids)
            .all(|(sid, run)| res.get(sid).is_some_and(|rt| rt.run_id == *run));
        if !current {
            // It can never be valid again — a moved turn keeps its new run id —
            // so drop it now and hand its device arrays back, rather than
            // holding them until something overwrites the key.
            drop(res);
            self.index_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&fingerprint);
            return Err(IndexMiss::Stale);
        }
        let lru = self.lru_clock.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();
        for &sid in &idx.pinned_sids {
            if let Some(rt) = res.get_mut(&sid) {
                rt.pinned += 1;
                rt.lru = lru;
                rt.touch(now);
            }
        }
        Ok(idx)
    }

    /// Cache `idx` under `fingerprint` for reuse, evicting the least recently
    /// used entries until the cache is inside both [`INDEX_CACHE_CAP`] entries
    /// and [`INDEX_CACHE_BYTES`] of device arrays. An index larger than the
    /// byte bound on its own is not cached: it would evict everything and still
    /// not fit.
    fn store_index(&self, fingerprint: u64, idx: Arc<scan::PagedIndex>) {
        let bytes = idx.device_bytes();
        let mut cache = self.index_cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.remove(&fingerprint);
        if bytes > INDEX_CACHE_BYTES {
            return;
        }
        let entries: Vec<(u64, u64, u64)> = cache
            .iter()
            .map(|(&fp, e)| (fp, e.used, e.idx.device_bytes()))
            .collect();
        for fp in index_cache_evictions(&entries, bytes, INDEX_CACHE_CAP, INDEX_CACHE_BYTES) {
            cache.remove(&fp);
        }
        let used = self.lru_clock.fetch_add(1, Ordering::Relaxed);
        cache.insert(fingerprint, scan::CachedIndex { idx, used });
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
    /// - **A turn with a moved page gets a fresh run id.** [`scan::PagedIndex`]
    ///   caches raw page addresses and `reuse_index` revalidates them only by
    ///   each turn's unchanged run id. A page that moved without the new id
    ///   would hand the scan kernel a stale address — the one failure here that
    ///   surfaces as quietly wrong retrieval rather than as a fault.
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
            // **A moved turn's run id is renewed failure included.** A copy that
            // errors part-way has already reassigned every slot before it, so
            // leaving that turn's id alone would let a cached `PagedIndex` vouch
            // for addresses it no longer occupies — the one failure here that
            // surfaces as quietly wrong retrieval rather than as a fault. The
            // result is carried past the renewal instead of propagated through it.
            let mut outcome = Ok(());
            for rt in res.values_mut() {
                if rt.pinned != 0 {
                    continue;
                }
                let mut moved_here = 0usize;
                for slot in &mut rt.run.slots {
                    let Some(dst) = by_src.remove(&slot.ptr()) else {
                        continue;
                    };
                    if let Err(e) = slot.copy_into(&dst, self.page_bytes as usize, &self.device) {
                        outcome = Err(e);
                        break;
                    }
                    *slot = dst;
                    moved_here += 1;
                }
                if moved_here > 0 {
                    // Under the residency lock, with the moves: a scan taking the
                    // lock after this sees both the new addresses and the new id.
                    rt.run_id = self.next_run_id.fetch_add(1, Ordering::Relaxed);
                    moved += moved_here;
                }
                if outcome.is_err() {
                    break;
                }
            }
            if moved > 0 {
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

    /// Evict least-recently-used unpinned turns until at least `want` bytes are
    /// freed, regardless of recent use. Returns the bytes freed. The tests use it
    /// to empty the arena and prove a rebuild reproduces the scan; nothing in the
    /// engine sheds a working set this way.
    #[cfg(test)]
    pub fn evict_lru(&self, want: u64) -> u64 {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        self.evict_lru_locked(&mut res, want)
    }

    /// [`evict_lru`](Self::evict_lru) under a residency guard the caller holds.
    #[cfg(test)]
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

    /// How often a turn must be scanned to keep its ground — `T*`, set by the
    /// scheduler from the weight side's measured marginal value
    /// ([`eviction::keep_within`]); `None` keeps everything.
    pub fn set_keep_within(&self, keep: Option<Duration>) {
        let ns = keep.map_or(u64::MAX, |d| d.as_nanos().min(u64::MAX as u128 - 1) as u64);
        self.keep_within_ns.store(ns, Ordering::Relaxed);
    }

    fn keep_within(&self) -> Option<Duration> {
        match self.keep_within_ns.load(Ordering::Relaxed) {
            u64::MAX => None,
            ns => Some(Duration::from_nanos(ns)),
        }
    }

    /// What a scan waits per byte of its working set when that set is not all
    /// resident — the uploads of the turns it finds evicted and the index over
    /// them, in microseconds, timed on the scans that rebuilt — and so what a
    /// turn's residency spares per byte at each scan. Zero until one has.
    pub fn upload_us_per_byte(&self) -> f64 {
        f64::from_bits(self.upload_us_per_byte.load(Ordering::Relaxed))
    }

    /// An upload of `bytes` took `elapsed`: it joins the per-byte average.
    fn note_upload(&self, bytes: u64, elapsed: Duration) {
        if bytes == 0 {
            return;
        }
        let sample = elapsed.as_secs_f64() * 1e6 / bytes as f64;
        let was = self.upload_us_per_byte();
        let next = if was > 0.0 {
            was + UPLOAD_WEIGHT * (sample - was)
        } else {
            sample
        };
        self.upload_us_per_byte
            .store(next.to_bits(), Ordering::Relaxed);
    }

    /// Resident turns, and of them how many a scan holds pinned now — what the
    /// keep trade may not take whatever their age.
    pub fn residency_counts(&self) -> (usize, usize) {
        let res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        (res.len(), res.values().filter(|rt| rt.pinned != 0).count())
    }

    /// Give up every turn not worth its ground ([`eviction::past_keep`]) — the
    /// governor's relief rung, a scan's index build, and a preload. Returns
    /// the bytes freed. Pinned turns never go, so an in-flight scan's pages
    /// stay.
    pub fn evict_past_keep(&self) -> u64 {
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        self.evict_past_keep_locked(&mut res)
    }

    fn evict_past_keep_locked(&self, res: &mut HashMap<StreamId, ResidentTurn>) -> u64 {
        let keep = self.keep_within();
        if keep.is_none() {
            return 0;
        }
        let now = Instant::now();
        let (sids, turns): (Vec<StreamId>, Vec<Resident>) = res
            .iter()
            .map(|(sid, rt)| {
                (
                    *sid,
                    Resident {
                        lru: rt.lru,
                        idle: now.saturating_duration_since(rt.used),
                        interval: rt.interval,
                        bytes: pages_for(rt.n_tokens) as u64 * self.page_bytes,
                        pinned: rt.pinned != 0,
                    },
                )
            })
            .unzip();
        let mut freed = 0u64;
        let gone = past_keep(&turns, keep);
        for &i in &gone {
            res.remove(&sids[i]); // drops the run → frees pages
            self.residency_gen.fetch_add(1, Ordering::Relaxed);
            freed += turns[i].bytes;
        }
        if !gone.is_empty() {
            tracing::info!(
                target: "candle_conversation::provenance",
                turns = gone.len(),
                freed_mib = freed >> 20,
                keep_ms = keep.map_or(0, |k| k.as_millis() as u64),
                upload_ns_per_kib = (self.upload_us_per_byte() * 1024.0 * 1000.0) as u64,
                kept = res.len(),
                "gallery gave up the turns past its keep time"
            );
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::WideQSig;
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

    /// Held by every test that builds a gallery on the card.
    ///
    /// **The galleries on one device share one slot pool**, and a compaction
    /// plans over the whole pool: run beside another test's gallery, a packed
    /// gallery found that gallery's holes and planned moves, and a scattered
    /// one found its holes filled. One gallery at a time makes each test's
    /// pool its own.
    pub(super) fn gallery_pool() -> std::sync::MutexGuard<'static, ()> {
        static POOL: Mutex<()> = Mutex::new(());
        POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Round-trip: upload a turn, read its pages back, verify the group-major
    /// transpose survived the H2D exactly (raw bytes, not a threshold).
    #[test]
    fn resident_pages_roundtrip_group_major() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU — skip
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs: Vec<WideQSig> = (0..40).map(|t| sig((t as u64) << 40)).collect(); // 2 pages
        let fp = 0xDEADBEEF;
        let addrs = arena.ensure_resident(sid(0), &sigs, fp).unwrap();
        assert_eq!(addrs.len(), 2, "40 tokens → 2 pages");
        assert_eq!(arena.resident_turns(), 1);

        // Read pages back and check the transpose token-by-token.
        let pw = page_u64(24);
        let expect = transpose_to_pages(&sigs, 24, 3, pw);
        let res = arena.residency.lock().unwrap();
        let rt = res.get(&sid(0)).unwrap();
        assert_eq!(
            rt.run.addrs(),
            addrs,
            "the addresses handed out are the run's"
        );
        for (p, slot) in rt.run.slots.iter().enumerate() {
            let got = page_io::read_page(&device, slot, pw).unwrap();
            assert_eq!(
                got,
                expect[p * pw..(p + 1) * pw],
                "page {p} bytes differ after H2D"
            );
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
        let _pool = gallery_pool();
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
        let _pool = gallery_pool();
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
        let _pool = gallery_pool();
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

    /// **A resident hit evicts nothing; an upload makes room first.** With a
    /// keep time every turn is past, ensuring a resident turn leaves the other
    /// in place, and only admitting a new one gives the idle turns up.
    #[test]
    fn only_an_upload_runs_the_keep_eviction() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs = |t: u64| -> Vec<WideQSig> { (0..32).map(|k| sig(t * 1000 + k)).collect() };
        arena.ensure_resident(sid(0), &sigs(0), 1).unwrap();
        arena.ensure_resident(sid(1), &sigs(1), 2).unwrap();
        arena.set_keep_within(Some(Duration::from_nanos(1)));
        std::thread::sleep(Duration::from_millis(2));
        arena.ensure_resident(sid(0), &sigs(0), 1).unwrap();
        assert_eq!(arena.resident_turns(), 2, "a hit admits nothing");
        std::thread::sleep(Duration::from_millis(2));
        arena.ensure_resident(sid(2), &sigs(2), 3).unwrap();
        let res = arena.residency.lock().unwrap();
        assert_eq!(res.len(), 1, "the upload gave both idle turns up");
        assert!(res.contains_key(&sid(2)));
    }

    /// **A preload makes room once, before its first turn**, then admits every
    /// turn it names, in order: the idle turns past the keep time go, and the
    /// batch's own turns — one re-uploaded, one new — are all resident after.
    #[test]
    fn a_preload_gives_up_idle_turns_once_then_admits_its_own() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs = |t: u64| -> Vec<WideQSig> { (0..32).map(|k| sig(t * 1000 + k)).collect() };
        arena.ensure_resident(sid(0), &sigs(0), 1).unwrap();
        arena.ensure_resident(sid(1), &sigs(1), 2).unwrap();
        arena.set_keep_within(Some(Duration::from_nanos(1)));
        std::thread::sleep(Duration::from_millis(2));
        let (s0, s2) = (sigs(0), sigs(2));
        let done = arena.ensure_all_resident([
            (sid(0), TurnSigs::from(&s0), 1),
            (sid(2), TurnSigs::from(&s2), 3),
        ]);
        assert_eq!(done.len(), 2);
        assert!(done.iter().all(|r| r.is_ok()));
        let res = arena.residency.lock().unwrap();
        let mut held: Vec<StreamId> = res.keys().copied().collect();
        held.sort_by_key(|s| s.0);
        let mut want = vec![sid(0), sid(2)];
        want.sort_by_key(|s| s.0);
        assert_eq!(held, want, "the idle turn 1 went; the batch's turns stay");
    }

    /// Eviction frees LRU turns and skips the pinned working set.
    #[test]
    fn evict_lru_skips_pins() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
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

    /// **A compaction empties an arena and changes nothing a reader can see.**
    ///
    /// Scatter across two arenas — fill the first, spill a few turns into the
    /// second, then drop turns from the first — so the second's pages fit the
    /// holes below and the pass can give its region back. A pass only moves
    /// pages it can empty an arena with (`plan_slot_moves`); holes inside a
    /// single arena free nothing and are left alone. Then read every surviving
    /// page back, compared against the expected group-major transpose as **raw
    /// words**, not a tolerance: a page is bytes, and a copy that alters one is
    /// a corrupted signature, not an approximation.
    #[test]
    fn compaction_moves_pages_without_changing_them() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let sigs = |t: u64, n: usize| -> Vec<WideQSig> {
            (0..n).map(|k| sig((t << 40) + k as u64)).collect()
        };
        // Turns of 64 pages, enough to fill an arena and spill two into the next.
        let per_arena = candle_nn::kv_cache::REGION_BYTES
            / candle_nn::kv_cache::slot_stride(arena.page_bytes as usize);
        let turn_tokens = 64 * PAGE_TOKENS;
        let n_turns = (per_arena / 64 + 3) as u64;
        for t in 0..n_turns {
            arena
                .ensure_resident(sid(t), &sigs(t, turn_tokens), t + 1)
                .unwrap();
        }
        let regions_before = arena_regions(&device, SlotTenant::Gallery);
        assert!(regions_before >= 2, "the turns spill into a second arena");
        // Holes in the first arena, more than the second holds.
        let dropped = 0..4u64;
        for t in dropped.clone() {
            arena.drop_turn(sid(t));
        }
        let live_before = arena.live_pages();

        let report = arena.compact(0).unwrap();
        assert!(
            report.moved > 0 && report.moved == report.planned,
            "expected the second arena's pages to move, got {report:?}",
        );
        assert_eq!(arena.live_pages(), live_before, "a move frees no page");
        assert!(
            arena_regions(&device, SlotTenant::Gallery) < regions_before,
            "the emptied arena's region went back",
        );

        // Every surviving turn still reads back exactly its own transpose.
        let res = arena.residency.lock().unwrap();
        for t in 0..n_turns {
            if dropped.contains(&t) {
                assert!(!res.contains_key(&sid(t)), "dropped turn is gone");
                continue;
            }
            let pw = page_u64(24);
            let expect = transpose_to_pages(&sigs(t, turn_tokens), 24, 3, pw);
            let run = &res.get(&sid(t)).expect("survivor").run;
            assert_eq!(run.slots.len() * pw, expect.len());
            for (p, slot) in run.slots.iter().enumerate() {
                let got = page_io::read_page(&device, slot, pw).unwrap();
                assert_eq!(
                    got,
                    expect[p * pw..(p + 1) * pw],
                    "turn {t} page {p} after compaction"
                );
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
        let _pool = gallery_pool();
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
        let _pool = gallery_pool();
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
