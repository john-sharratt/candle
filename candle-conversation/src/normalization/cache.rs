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
        match self.scopes.get(scope) {
            Some(s) => s.normalize(raw, &self.cfg),
            None => ScopeState::default().normalize(raw, &self.cfg),
        }
    }

    /// Write path: fold a turn's raw scores into the scope's hit levels, creating
    /// the scope on first observation. Call once per turn at seal, with the scope's
    /// full current membership.
    ///
    /// A turn group's gallery lives on one timeline; a re-scan mints a new
    /// `(group, timeline)` scope. After observing one, drop that group's
    /// stale-timeline scopes so the cache stays bounded at one scope per active
    /// group instead of leaking a `ScopeState` per historical re-scan. The scan is
    /// once per turn and over a handful of scopes, so this is cheap.
    pub fn observe(&mut self, scope: &ScopeKey, raw: &[(ChildKey, f32)]) {
        self.scopes
            .entry(scope.clone())
            .or_default()
            .observe(raw, &self.cfg);
        if let ScopeKey::TurnGroup { group, timeline } = scope {
            self.scopes.retain(|k, _| match k {
                ScopeKey::TurnGroup {
                    group: g,
                    timeline: t,
                } => g != group || t == timeline,
                _ => true,
            });
        }
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
