//! [`SubstrateCache`] — tier-management layer for the substrate.
//!
//! Owns all knowledge about *where* entries live across the three storage tiers
//! and *how recently* they were used.  [`Substrate`](super::substrate::Substrate)
//! is a dumb content store; this module is the eviction brain.
//!
//! # Tiers
//!
//! ```text
//!  Hot  (VRAM)  — SubstrateCache::hot_*   — GPU-resident ChunkGids
//!  Warm (RAM)   — Substrate::turns/sections — CPU-resident ChunkGids  ← main map
//!  Cold (NVMe)  — not yet implemented
//! ```
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
//! `hot_budget` caps that total; once set, every `insert_*` call evicts the
//! least-recently-used hot entries before admitting the new one.
//!
//! The budget should be set **after** model weights are loaded and resident —
//! use [`SubstrateCache::new`] with the free-VRAM figure from a post-load
//! CUDA memory query so model weights are automatically excluded from the
//! calculation.

use std::sync::{Arc, Mutex};

use ahash::AHashMap;

use crate::projection::{SectionId, TimelineId, TurnIndex};
use crate::substrate::{SectionEntryData, TurnEntryData};

// ── SubstrateKey ──────────────────────────────────────────────────────────────

/// Uniform handle for any entry in the substrate, used by the eviction manager
/// to address turns and sections without knowing which map they live in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubstrateKey {
    Turn(TimelineId, TurnIndex),
    Section(SectionId),
}

// ── SubstrateCacheInner ───────────────────────────────────────────────────────

/// All mutable state for the hot tier.  Plain fields — no atomics, no nested
/// locks.  Always accessed under the single `Mutex` in [`SubstrateCache`].
#[derive(Debug)]
struct SubstrateCacheInner {
    hot_sections: AHashMap<SectionId, SectionEntryData>,
    hot_turns: AHashMap<(TimelineId, TurnIndex), TurnEntryData>,
    /// Running VRAM total for all hot-tier entries.
    hot_bytes: u64,
    /// Per-entry VRAM byte count for precise subtraction on removal.
    hot_entry_bytes: AHashMap<SubstrateKey, u64>,
    /// VRAM budget. `None` = unlimited.
    hot_budget: Option<u64>,
    /// Monotonic counter; incremented on every `record_access` call.
    access_clock: u64,
    /// Last-access tick per hot-tier entry.
    hot_last_used: AHashMap<SubstrateKey, u64>,
    /// Hot-tier lookup hits (entry found in hot map).
    hit_count: u64,
    /// Hot-tier lookup misses (entry absent from hot map).
    miss_count: u64,
}

impl SubstrateCacheInner {
    fn new(hot_budget: Option<u64>) -> Self {
        Self {
            hot_sections: AHashMap::new(),
            hot_turns: AHashMap::new(),
            hot_bytes: 0,
            hot_entry_bytes: AHashMap::new(),
            hot_budget,
            access_clock: 0,
            hot_last_used: AHashMap::new(),
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

    fn entry_bytes(sealed: &Arc<Vec<candle_nn::kv_cache::SealedSequence>>) -> u64 {
        sealed
            .iter()
            .flat_map(|seq| seq.chunks.iter())
            .map(|c| c.byte_size)
            .sum()
    }

    fn evict_to_budget(&mut self, needed_bytes: u64) {
        let Some(budget) = self.hot_budget else { return };
        if needed_bytes == 0 || self.hot_bytes.saturating_add(needed_bytes) <= budget {
            return;
        }
        let n = self.hot_turns.len() + self.hot_sections.len();
        let candidates = self.lru_n(n);
        for key in candidates {
            if self.hot_bytes.saturating_add(needed_bytes) <= budget {
                break;
            }
            match key {
                SubstrateKey::Turn(tl, idx) => { self.remove_turn(tl, idx); }
                SubstrateKey::Section(sid) => { self.remove_section(sid); }
            }
        }
    }

    fn insert_turn(&mut self, timeline: TimelineId, index: TurnIndex, entry: TurnEntryData) {
        let key = SubstrateKey::Turn(timeline, index);
        if let Some(old) = self.hot_entry_bytes.remove(&key) {
            self.hot_bytes = self.hot_bytes.saturating_sub(old);
        }
        let bytes = Self::entry_bytes(&entry.sealed);
        self.evict_to_budget(bytes);
        if bytes > 0 {
            self.hot_entry_bytes.insert(key, bytes);
            self.hot_bytes = self.hot_bytes.saturating_add(bytes);
        }
        self.hot_last_used.entry(key).or_insert(0);
        self.hot_turns.insert((timeline, index), entry);
    }

    fn remove_turn(&mut self, timeline: TimelineId, index: TurnIndex) -> Option<TurnEntryData> {
        let key = SubstrateKey::Turn(timeline, index);
        if let Some(bytes) = self.hot_entry_bytes.remove(&key) {
            self.hot_bytes = self.hot_bytes.saturating_sub(bytes);
        }
        self.hot_last_used.remove(&key);
        self.hot_turns.remove(&(timeline, index))
    }

    fn insert_section(&mut self, section: SectionId, entry: SectionEntryData) {
        let key = SubstrateKey::Section(section);
        if let Some(old) = self.hot_entry_bytes.remove(&key) {
            self.hot_bytes = self.hot_bytes.saturating_sub(old);
        }
        let bytes = Self::entry_bytes(&entry.sealed);
        self.evict_to_budget(bytes);
        if bytes > 0 {
            self.hot_entry_bytes.insert(key, bytes);
            self.hot_bytes = self.hot_bytes.saturating_add(bytes);
        }
        self.hot_last_used.entry(key).or_insert(0);
        self.hot_sections.insert(section, entry);
    }

    fn remove_section(&mut self, section: SectionId) -> Option<SectionEntryData> {
        let key = SubstrateKey::Section(section);
        if let Some(bytes) = self.hot_entry_bytes.remove(&key) {
            self.hot_bytes = self.hot_bytes.saturating_sub(bytes);
        }
        self.hot_last_used.remove(&key);
        self.hot_sections.remove(&section)
    }

    fn clear(&mut self) {
        self.hot_sections.clear();
        self.hot_turns.clear();
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

/// Shared hot-tier cache.  Cheaply cloneable — every clone shares the same
/// inner state via `Arc<Mutex<>>`.
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
    /// immediately evict; eviction is demand-driven on the next `insert_*`.
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
    /// Entries with `last_used == 0` (inserted but never accessed) sort to
    /// the front and are evicted first.
    pub fn lru_entries(&self, n: usize) -> Vec<SubstrateKey> {
        self.inner.lock().map(|g| g.lru_n(n)).unwrap_or_default()
    }

    // ── Hot-tier operations ───────────────────────────────────────────────────

    pub fn insert_turn(&self, timeline: TimelineId, index: TurnIndex, entry: TurnEntryData) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.insert_turn(timeline, index, entry);
        }
    }

    /// Apply `f` to the hot-tier turn entry if present.  Increments hit/miss counter.
    pub fn with_turn_mut<F>(&self, timeline: TimelineId, index: TurnIndex, f: F)
    where
        F: FnOnce(&mut TurnEntryData),
    {
        if let Ok(mut inner) = self.inner.lock() {
            if inner.hot_turns.contains_key(&(timeline, index)) {
                inner.hit_count += 1;
                if let Some(entry) = inner.hot_turns.get_mut(&(timeline, index)) {
                    f(entry);
                }
            } else {
                inner.miss_count += 1;
            }
        }
    }

    pub fn remove_turn(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<TurnEntryData> {
        self.inner.lock().ok()?.remove_turn(timeline, index)
    }

    pub fn insert_section(&self, section: SectionId, entry: SectionEntryData) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.insert_section(section, entry);
        }
    }

    /// Apply `f` to the hot-tier section entry if present.  Increments hit/miss counter.
    pub fn with_section_mut<F>(&self, section: SectionId, f: F)
    where
        F: FnOnce(&mut SectionEntryData),
    {
        if let Ok(mut inner) = self.inner.lock() {
            if inner.hot_sections.contains_key(&section) {
                inner.hit_count += 1;
                if let Some(entry) = inner.hot_sections.get_mut(&section) {
                    f(entry);
                }
            } else {
                inner.miss_count += 1;
            }
        }
    }

    pub fn remove_section(&self, section: SectionId) -> Option<SectionEntryData> {
        self.inner.lock().ok()?.remove_section(section)
    }

    /// Clear all hot-tier entries, byte totals, and LRU state.
    /// Does not reset `hot_budget`.
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.clear();
        }
    }

    /// Clear all hot-tier entries and reset hit/miss counters.
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
