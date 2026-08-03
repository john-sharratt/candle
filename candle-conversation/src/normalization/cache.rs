//! The runtime cache of per-scope normalization state.
//!
//! In-memory and **not persisted** — it is derived from query traffic and rebuilt
//! empty each process load (see the module docs). `normalize` is the read path
//! (every reprojection); `observe` is the sole writer (once per turn at seal).

use std::collections::HashMap;

use super::scope::ScopeState;
use super::{ChildKey, NormConfig, ScopeKey};

/// Per-scope hit-level normalization, keyed by [`ScopeKey`]. Starts empty; each
/// scope fills in lazily as it is first observed.
#[derive(Default)]
pub struct NormalizationCache {
    cfg: NormConfig,
    scopes: HashMap<ScopeKey, ScopeState>,
}

impl NormalizationCache {
    pub fn new(cfg: NormConfig) -> Self {
        NormalizationCache {
            cfg,
            scopes: HashMap::new(),
        }
    }

    pub fn config(&self) -> &NormConfig {
        &self.cfg
    }

    /// Read path: normalized 0–1000 scores for a scope's children. A scope never
    /// observed yet normalizes every child against the cold-start prior. Pure.
    pub fn normalize(&self, scope: &ScopeKey, raw: &[(ChildKey, f32)]) -> Vec<(ChildKey, f32)> {
        self.normalize_with_floors(scope, raw, &[])
    }

    /// [`Self::normalize`] with a caller-supplied per-child denominator floor —
    /// the Concept A.4 size-aware level prior, constants owned by the caller's
    /// policy. Pure.
    pub fn normalize_with_floors(
        &self,
        scope: &ScopeKey,
        raw: &[(ChildKey, f32)],
        floors: &[f32],
    ) -> Vec<(ChildKey, f32)> {
        match self.scopes.get(scope) {
            Some(s) => s.normalize_with_floors(raw, floors, &self.cfg),
            None => ScopeState::default().normalize_with_floors(raw, floors, &self.cfg),
        }
    }

    /// Write path: fold a turn's raw scores into the scope's hit levels, creating
    /// the scope on first observation. Call once per turn at seal, with the scope's
    /// full current membership.
    ///
    /// Every `(group, timeline)` scope is retained independently. A belief group
    /// like `code_reading` has MANY simultaneously-active timelines — one per
    /// ingested file — and each needs its own learned hit levels so a cross-file
    /// query can rescale them onto the common 0–1000 band and compare them fairly.
    /// (An earlier version evicted a group's *other* timelines on every observe,
    /// assuming one active timeline per group. That holds for a re-scanned single
    /// cluster but is catastrophic for code_read: it wiped every file but the last,
    /// leaving the cache empty for all the others, so normalization degenerated to a
    /// flat `scale/prior` multiple of the raw score and a promiscuous low-entropy
    /// file won every query.) Stale scopes from a repo_map re-scan are left in place
    /// — dead once their timeline is inactive, bounded by the re-scan count, and a
    /// single small `ScopeState` each.
    pub fn observe(&mut self, scope: &ScopeKey, raw: &[(ChildKey, f32)]) {
        self.scopes
            .entry(scope.clone())
            .or_default()
            .observe(raw, &self.cfg);
    }

    #[cfg(test)]
    pub(super) fn level_of(&self, scope: &ScopeKey, child: &ChildKey) -> Option<f32> {
        self.scopes.get(scope).and_then(|s| s.level_of(child))
    }

    #[cfg(test)]
    pub(super) fn count_of(&self, scope: &ScopeKey, child: &ChildKey) -> Option<u32> {
        self.scopes.get(scope).and_then(|s| s.count_of(child))
    }
}
