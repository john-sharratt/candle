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

    /// [`Self::normalize_with_floors`] against `scope`, falling back to `parent`
    /// for any child `scope` has not learned enough about.
    ///
    /// Subdividing a scope — by phase, say — multiplies the scopes and splits the
    /// traffic between them, so some will be cold for a long time and some
    /// (a tool's `response` phase) effectively forever. Dividing by a
    /// learning-starved level is not a smaller version of normalizing, it
    /// INVERTS the ranking: the A.4 note measured normalization dropping code
    /// Top-1 57.1% → 47.5% on cold levels, and a cold tool lens scored 0/6 here
    /// against 6/6 warm. So a child the subdivision has not taught is normalized
    /// on the undivided parent instead of on noise, and the subdivision can only
    /// ever add resolution where it has evidence.
    pub fn normalize_with_fallback(
        &self,
        scope: &ScopeKey,
        parent: &ScopeKey,
        raw: &[(ChildKey, f32)],
        floors: &[f32],
        min_observations: u32,
    ) -> Vec<(ChildKey, f32)> {
        let Some(state) = self.scopes.get(scope) else {
            return self.normalize_with_floors(parent, raw, floors);
        };
        let (mut warm, mut cold): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        for (i, (child, _)) in raw.iter().enumerate() {
            if state.count_of(child).unwrap_or(0) >= min_observations {
                warm.push(i);
            } else {
                cold.push(i);
            }
        }
        if cold.is_empty() {
            return state.normalize_with_floors(raw, floors, &self.cfg);
        }
        if warm.is_empty() {
            return self.normalize_with_floors(parent, raw, floors);
        }
        // Mixed: normalize each half against the scope that actually knows the
        // child, then reassemble in the caller's order.
        let pick = |idx: &[usize]| -> (Vec<(ChildKey, f32)>, Vec<f32>) {
            (
                idx.iter().map(|&i| raw[i].clone()).collect(),
                idx.iter()
                    .map(|&i| floors.get(i).copied().unwrap_or(0.0))
                    .collect(),
            )
        };
        let (warm_raw, warm_floors) = pick(&warm);
        let (cold_raw, cold_floors) = pick(&cold);
        let warm_out = state.normalize_with_floors(&warm_raw, &warm_floors, &self.cfg);
        let cold_out = self.normalize_with_floors(parent, &cold_raw, &cold_floors);
        let mut out = raw.to_vec();
        for (slot, v) in warm.iter().zip(warm_out) {
            out[*slot] = v;
        }
        for (slot, v) in cold.iter().zip(cold_out) {
            out[*slot] = v;
        }
        out
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
    /// **Idempotent per observation.** `source` identifies the evidence — the
    /// turn this scoring pass came from — and a scope folds each source exactly
    /// once. Re-observing it is a no-op.
    ///
    /// That is what lets the levels be (re)built on EVERY load, from empty, even
    /// against a substrate already on disk, while the same scopes keep learning
    /// from live traffic afterwards. Without it the two paths fight: a replay
    /// drags every level toward whatever it re-feeds, so the levels a load
    /// reproduces differ from the ones it originally learned, and the ranking
    /// they drive drifts with nothing but a restart.
    ///
    /// Deduplicating on the SCORE instead would be wrong: a promiscuous child
    /// scores about the same on everything, so it would be recorded once and
    /// never learn that it is loud across all traffic — which is exactly what
    /// the hit level exists to discount.
    ///
    /// Fast on the hot path: one hash lookup rejects an already-folded source
    /// before touching any child.
    pub fn observe(&mut self, scope: &ScopeKey, source: u64, raw: &[(ChildKey, f32)]) {
        let state = self.scopes.entry(scope.clone()).or_default();
        if !state.mark_observed(source) {
            return;
        }
        state.observe(raw, &self.cfg);
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
