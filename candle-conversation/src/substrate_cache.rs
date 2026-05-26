//! [`SubstrateCache`] — tier-management layer for the substrate.
//!
//! Owns all knowledge about *where* entries live across the three storage tiers
//! and *how recently* they were used.  [`Substrate`](super::substrate::Substrate)
//! is the data store (warm tier); this module is the VRAM accounting and LRU
//! eviction brain.
//!
//! # Tiers
//!
//! ```text
//!  Hot  (VRAM)  — tracked here: byte accounting + LRU per SubstrateKey
//!  Warm (RAM)   — Substrate::turns / sections — single source of truth for data
//!  Cold (NVMe)  — not yet implemented
//! ```
//!
//! The hot tier does NOT duplicate entry data.  It only records how many VRAM
//! bytes each entry occupies and when it was last accessed.  All actual entry
//! data lives in the warm tier (`Substrate`).  When the scheduler prefills a
//! turn's KV blocks onto the GPU, it calls [`SubstrateCache::mark_hot`] to
//! register the VRAM cost; when those blocks are evicted it calls
//! [`SubstrateCache::mark_cold`].
//!
//! # Shared ownership
//!
//! `SubstrateCache` wraps `Arc<Mutex<SubstrateCacheInner>>` and is cheaply
//! `Clone`-able.  The [`ConversationEngine`](super::engine::ConversationEngine)
//! creates one instance after model load, then clones it into every
//! [`Substrate`](super::substrate::Substrate) so VRAM accounting and the
//! budget cap are shared across all concurrent sessions.
//!
//! # Hot-tier accounting and budget
//!
//! `hot_bytes` tracks the total VRAM occupied by GPU-resident entries.
//! `hot_budget` caps that total; once set, every `mark_hot` call evicts the
//! least-recently-used hot entries before admitting the new one.
//!
//! The budget should be set **after** model weights are loaded and resident —
//! use [`SubstrateCache::new`] with the free-VRAM figure from a post-load
//! CUDA memory query so model weights are automatically excluded from the
//! calculation.

use std::sync::{Arc, Mutex};

use crate::projection::{SectionId, TurnKey};

// ── SubstrateKey ──────────────────────────────────────────────────────────────

/// Uniform handle for any entry in the substrate, used by the eviction manager
/// to address turns and sections without knowing which map they live in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubstrateKey {
    Turn(TurnKey),
    Section(SectionId),
}

// ── SubstrateCacheInner ───────────────────────────────────────────────────────

/// All mutable state for the hot tier.  Plain fields — no atomics, no nested
/// locks.  Always accessed under the single `Mutex` in [`SubstrateCache`].
///
/// Only accounting data lives here (byte totals, LRU timestamps).  Entry data
/// lives in `Substrate::turns` / `Substrate::sections` (warm tier).
#[derive(Debug)]
struct SubstrateCacheInner {
    /// Running VRAM total for all hot-tier entries.
    hot_bytes: u64,
    /// Per-entry VRAM byte count for precise subtraction on removal.
    hot_entry_bytes: ahash::AHashMap<SubstrateKey, u64>,
    /// VRAM budget. `None` = unlimited.
    hot_budget: Option<u64>,
    /// Monotonic counter; incremented on every `record_access` call.
    access_clock: u64,
    /// Last-access tick per hot-tier entry.
    hot_last_used: ahash::AHashMap<SubstrateKey, u64>,
    /// Hot-tier lookup hits.
    hit_count: u64,
    /// Hot-tier lookup misses.
    miss_count: u64,
}

impl SubstrateCacheInner {
    fn new(hot_budget: Option<u64>) -> Self {
        Self {
            hot_bytes: 0,
            hot_entry_bytes: ahash::AHashMap::new(),
            hot_budget,
            access_clock: 0,
            hot_last_used: ahash::AHashMap::new(),
            hit_count: 0,
            miss_count: 0,
        }
    }

    fn record_access(&mut self, key: SubstrateKey) {
        self.access_clock += 1;
        self.hot_last_used.insert(key, self.access_clock);
    }

    fn lru_n(&self, n: usize) -> Vec<SubstrateKey> {
        let mut candidates: Vec<(u64, SubstrateKey)> = self
            .hot_last_used
            .iter()
            .map(|(&key, &ts)| (ts, key))
            .collect();
        candidates.sort_unstable_by_key(|&(ts, _)| ts);
        candidates.into_iter().take(n).map(|(_, key)| key).collect()
    }

    /// Evict LRU hot entries until `self.hot_bytes + needed_bytes <= budget`.
    /// Returns the evicted keys so callers can act on them (e.g. move GPU
    /// blocks to warm tier).
    fn evict_to_budget(&mut self, needed_bytes: u64) -> Vec<SubstrateKey> {
        let Some(budget) = self.hot_budget else { return vec![] };
        if self.hot_bytes.saturating_add(needed_bytes) <= budget {
            return vec![];
        }
        let n = self.hot_entry_bytes.len();
        let candidates = self.lru_n(n);
        let mut evicted = Vec::new();
        for key in candidates {
            if self.hot_bytes.saturating_add(needed_bytes) <= budget {
                break;
            }
            self.mark_cold(key);
            evicted.push(key);
        }
        evicted
    }

    fn mark_hot(&mut self, key: SubstrateKey, byte_size: u64) {
        // Remove old accounting for this key if re-marking.
        if let Some(old) = self.hot_entry_bytes.remove(&key) {
            self.hot_bytes = self.hot_bytes.saturating_sub(old);
        }
        if byte_size > 0 {
            self.hot_entry_bytes.insert(key, byte_size);
            self.hot_bytes = self.hot_bytes.saturating_add(byte_size);
        }
        self.hot_last_used.entry(key).or_insert(0);
    }

    fn mark_cold(&mut self, key: SubstrateKey) {
        if let Some(bytes) = self.hot_entry_bytes.remove(&key) {
            self.hot_bytes = self.hot_bytes.saturating_sub(bytes);
        }
        self.hot_last_used.remove(&key);
    }

    fn clear(&mut self) {
        self.hot_bytes = 0;
        self.hot_entry_bytes.clear();
        self.hot_last_used.clear();
        // hot_budget intentionally preserved across clear()
    }

    fn purge(&mut self) {
        self.clear();
        self.hit_count = 0;
        self.miss_count = 0;
    }
}

// ── SubstrateCache ────────────────────────────────────────────────────────────

/// Shared hot-tier VRAM accounting cache.  Cheaply cloneable — every clone
/// shares the same inner state via `Arc<Mutex<>>`.
///
/// Does **not** store entry data.  Call [`SubstrateCache::mark_hot`] when GPU
/// blocks are prefilled for an entry, and [`SubstrateCache::mark_cold`] when
/// they are evicted.  All entry data lives in
/// [`Substrate`](super::substrate::Substrate).
#[derive(Clone, Debug)]
pub struct SubstrateCache {
    inner: Arc<Mutex<SubstrateCacheInner>>,
}

impl SubstrateCache {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create a cache with a VRAM budget derived from the **post-model-load**
    /// free-VRAM figure.
    ///
    /// Call this after model weights are fully resident so the free-VRAM query
    /// already excludes weight consumption.  Two reserves are subtracted:
    ///
    /// - `abs_reserve_bytes`: fixed floor for decode activations, attention
    ///   scratch space, and OS overhead.
    /// - `rel_reserve_frac`: fraction of `free_vram_bytes` kept as headroom
    ///   (e.g. `0.05` = 5 %).
    pub fn new(free_vram_bytes: u64, abs_reserve_bytes: u64, rel_reserve_frac: f64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SubstrateCacheInner::new(Some(
                Self::compute_budget(free_vram_bytes, abs_reserve_bytes, rel_reserve_frac),
            )))),
        }
    }

    /// Create a cache with no VRAM budget (unlimited).
    ///
    /// Intended for pre-load bootstrapping and tests.  Call
    /// [`SubstrateCache::activate_budget`] after the model is resident to
    /// enable eviction.
    pub fn unbounded() -> Self {
        Self {
            inner: Arc::new(Mutex::new(SubstrateCacheInner::new(None))),
        }
    }

    // ── Budget ────────────────────────────────────────────────────────────────

    fn compute_budget(free_vram_bytes: u64, abs_reserve_bytes: u64, rel_reserve_frac: f64) -> u64 {
        let rel_reserve = (free_vram_bytes as f64 * rel_reserve_frac.clamp(0.0, 1.0)) as u64;
        free_vram_bytes.saturating_sub(abs_reserve_bytes.saturating_add(rel_reserve))
    }

    /// Activate (or replace) the VRAM budget on an already-constructed cache.
    ///
    /// Mirrors the calculation in [`SubstrateCache::new`].  Does not
    /// immediately evict; eviction is demand-driven on the next `mark_hot`.
    pub fn activate_budget(
        &self,
        free_vram_bytes: u64,
        abs_reserve_bytes: u64,
        rel_reserve_frac: f64,
    ) {
        let budget = Self::compute_budget(free_vram_bytes, abs_reserve_bytes, rel_reserve_frac);
        if let Ok(mut inner) = self.inner.lock() {
            inner.hot_budget = Some(budget);
        }
    }

    /// Current VRAM budget. `None` = unlimited.
    pub fn hot_budget(&self) -> Option<u64> {
        self.inner.lock().ok().and_then(|g| g.hot_budget)
    }

    // ── Hot-tier byte total ───────────────────────────────────────────────────

    /// Total VRAM currently occupied by all GPU-resident (hot-tier) entries.
    pub fn hot_bytes(&self) -> u64 {
        self.inner.lock().map_or(0, |g| g.hot_bytes)
    }

    // ── LRU ──────────────────────────────────────────────────────────────────

    /// Stamp `key` as just-used in inference.
    pub fn record_access(&self, key: SubstrateKey) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.record_access(key);
        }
    }

    /// Return the `n` least-recently-used hot-tier keys.
    ///
    /// Entries with `last_used == 0` (registered but never accessed) sort to
    /// the front and are evicted first.
    pub fn lru_entries(&self, n: usize) -> Vec<SubstrateKey> {
        self.inner.lock().map(|g| g.lru_n(n)).unwrap_or_default()
    }

    // ── Hot-tier accounting ───────────────────────────────────────────────────

    /// Register `key` as GPU-resident with `byte_size` VRAM bytes.
    ///
    /// If the budget would be exceeded, evicts the LRU entries first and
    /// returns their keys so the caller can move those entries' GPU blocks
    /// to warm/cold storage.  Returns `vec![]` when no eviction was needed.
    pub fn mark_hot(&self, key: SubstrateKey, byte_size: u64) -> Vec<SubstrateKey> {
        self.inner.lock().map_or(vec![], |mut inner| {
            let evicted = inner.evict_to_budget(byte_size);
            inner.mark_hot(key, byte_size);
            evicted
        })
    }

    /// Deregister `key` from hot tier (GPU blocks evicted or freed).
    pub fn mark_cold(&self, key: SubstrateKey) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.mark_cold(key);
        }
    }

    // ── Hit/miss counters (used by callers that do hot-first lookups) ─────────

    /// Increment the hot-tier hit counter.
    pub fn record_hit(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.hit_count += 1;
        }
    }

    /// Increment the hot-tier miss counter.
    pub fn record_miss(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.miss_count += 1;
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /// Clear all hot-tier accounting.  Does not reset `hot_budget`.
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.clear();
        }
    }

    /// Clear all hot-tier accounting and reset hit/miss counters.
    /// Preserves `hot_budget`.
    pub fn purge(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.purge();
        }
    }

    // ── Hit/miss statistics ───────────────────────────────────────────────────

    /// Total hot-tier lookup hits since last `purge()`.
    pub fn hit_count(&self) -> u64 {
        self.inner.lock().map_or(0, |g| g.hit_count)
    }

    /// Total hot-tier lookup misses since last `purge()`.
    pub fn miss_count(&self) -> u64 {
        self.inner.lock().map_or(0, |g| g.miss_count)
    }

    /// Hit rate `hits / (hits + misses)`, or `0.0` if no lookups recorded.
    pub fn hit_rate(&self) -> f64 {
        self.inner.lock().map_or(0.0, |g| {
            let total = g.hit_count + g.miss_count;
            if total == 0 { 0.0 } else { g.hit_count as f64 / total as f64 }
        })
    }
}

impl Default for SubstrateCache {
    fn default() -> Self {
        Self::unbounded()
    }
}
