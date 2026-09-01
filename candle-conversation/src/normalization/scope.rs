//! One scope's normalization state: a hit level per child, plus the read
//! (`normalize_with_floors`) and write (`observe`) operations over them.

use std::collections::{HashMap, HashSet};

use super::hit_level::HitLevel;
use super::{ChildKey, NormConfig};

/// Per-child hit levels for a single budget scope.
#[derive(Default)]
pub(super) struct ScopeState {
    children: HashMap<ChildKey, HitLevel>,
    /// Evidence already folded into this scope, so folding it again is a no-op
    /// (see [`super::cache::NormalizationCache::observe`]). One `u64` per
    /// distinct observation — a turn stream id — which for the corpora here is
    /// hundreds to low thousands per scope.
    observed: HashSet<u64>,
}

impl ScopeState {
    /// Record `source` as folded into this scope. Returns `false` when it was
    /// already present, i.e. the caller should skip the fold.
    pub(super) fn mark_observed(&mut self, source: u64) -> bool {
        self.observed.insert(source)
    }

    /// Normalize each child's raw score to the 0–1000 band:
    /// `scale × raw / max(hit_level, floor)`. A child never observed in this
    /// scope normalizes against the cold-start prior. Pure — does not mutate.
    ///
    /// `floors` is a caller-supplied per-child denominator floor —
    /// the Concept A.4 size-aware level prior
    /// (`docs/provenance_adaptive_projection.md` §3). A positive `floors[i]`
    /// switches child `i` to **traffic-peak normalization**: its denominator is
    /// `max(observed_traffic_peak, floors[i])` — the prior seed and the
    /// scope-percentile floor do not apply. A promiscuous child's peak is high
    /// (it hits on everything), so its cross-hits mute; a quiet-but-large
    /// child's small size floor lets a rare genuine hit stand out — while a
    /// tiny fragment's (large) size floor mutes chance matches. The caller's
    /// policy owns the constants; a floor of `0.0` (or a missing entry) keeps
    /// that child on the standard hit-level path. The mechanism here stays
    /// constant-free.
    pub(super) fn normalize_with_floors(
        &self,
        raw: &[(ChildKey, f32)],
        floors: &[f32],
        cfg: &NormConfig,
    ) -> Vec<(ChildKey, f32)> {
        let floor = self.floor(cfg);
        raw.iter()
            .enumerate()
            .map(|(i, (k, r))| {
                let child = self.children.get(k);
                let child_floor = floors.get(i).copied().unwrap_or(0.0);
                let denom = if child_floor > 0.0 {
                    child.map(|h| h.peak()).unwrap_or(0.0).max(child_floor)
                } else {
                    child.map(|h| h.level()).unwrap_or(cfg.hit_prior).max(floor)
                };
                (k.clone(), cfg.scale * r / denom)
            })
            .collect()
    }

    /// Fold a turn's raw scores into each child's hit-level EWMA, creating an
    /// unseen child at the prior. Additive: existing children not named here keep
    /// their level (call [`Self::retain`] to prune after a membership change).
    pub(super) fn observe(&mut self, raw: &[(ChildKey, f32)], cfg: &NormConfig) {
        for (k, r) in raw {
            self.children
                .entry(k.clone())
                .or_insert_with(|| HitLevel::new(cfg.hit_prior))
                .observe(*r, cfg);
        }
    }

    /// Denominator floor for this scope: the `floor_pctl` percentile of current
    /// hit levels, hard-floored at `floor_min`. Empty scope ⇒ `floor_min`.
    fn floor(&self, cfg: &NormConfig) -> f32 {
        let mut levels: Vec<f32> = self
            .children
            .values()
            .map(|h| h.level())
            .filter(|l| *l > 0.0)
            .collect();
        if levels.is_empty() {
            return cfg.floor_min;
        }
        levels.sort_by(f32::total_cmp);
        let i = ((levels.len() as f32 * cfg.floor_pctl) as usize).min(levels.len() - 1);
        levels[i].max(cfg.floor_min)
    }

    #[cfg(test)]
    pub(super) fn level_of(&self, k: &ChildKey) -> Option<f32> {
        self.children.get(k).map(|h| h.level())
    }

    /// How many distinct observations have shaped this child's level here — the
    /// measure of whether this scope knows enough about it to be its
    /// denominator. `None` when the scope has never seen it at all.
    pub(super) fn count_of(&self, k: &ChildKey) -> Option<u32> {
        self.children.get(k).map(|h| h.count())
    }
}
