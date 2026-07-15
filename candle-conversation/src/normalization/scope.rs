//! One scope's normalization state: a hit level per child, plus the read
//! (`normalize`) and write (`observe`) operations over them.

use std::collections::HashMap;

use super::hit_level::HitLevel;
use super::{ChildKey, NormConfig};

/// Per-child hit levels for a single budget scope.
#[derive(Default)]
pub(super) struct ScopeState {
    children: HashMap<ChildKey, HitLevel>,
}

impl ScopeState {
    /// Normalize each child's raw score to the 0–1000 band:
    /// `scale × raw / max(hit_level, floor)`. A child never observed in this scope
    /// normalizes against the cold-start prior. Pure — does not mutate.
    pub(super) fn normalize(
        &self,
        raw: &[(ChildKey, f32)],
        cfg: &NormConfig,
    ) -> Vec<(ChildKey, f32)> {
        let floor = self.floor(cfg);
        raw.iter()
            .map(|(k, r)| {
                let hit = self
                    .children
                    .get(k)
                    .map(|h| h.level())
                    .unwrap_or(cfg.hit_prior);
                (k.clone(), cfg.scale * r / hit.max(floor))
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

    #[cfg(test)]
    pub(super) fn count_of(&self, k: &ChildKey) -> Option<u32> {
        self.children.get(k).map(|h| h.count())
    }
}
