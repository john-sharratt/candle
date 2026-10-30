//! One scope's normalization state: a hit level per child, plus the read
//! (`normalize_with_floors`) and write (`observe`) operations over them.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use super::hit_level::HitLevel;
use super::{ChildKey, NormConfig};

/// One raw score, restated as the score it would have reached over a
/// `cfg.probe_t_ref`-token probe.
///
/// The single place the band's unit is fixed, applied identically on the read
/// and the write path — which is the whole point: a level learned at one probe
/// length and a query measured at another were previously divided against each
/// other directly. See [`NormConfig::probe_t_ref`] for the measured
/// consequences and for why this is linear and uncapped.
///
/// `probe_tokens == 0` would be a scan with no probe, which produces no scores
/// to scale; treating it as one token keeps the function total rather than
/// letting a division by zero reach the band as an infinity.
fn to_ref(raw: f32, probe_tokens: usize, cfg: &NormConfig) -> f32 {
    raw * (cfg.probe_t_ref as f32 / probe_tokens.max(1) as f32)
}

/// Per-child hit levels for a single budget scope. Immutable once shared: an
/// observe edits a copy ([`super::cache::NormalizationCache::observe`]).
#[derive(Default, Clone)]
pub(super) struct ScopeState {
    children: HashMap<ChildKey, HitLevel>,
    /// Evidence already folded into this scope, so folding it again is a no-op
    /// (see [`super::cache::NormalizationCache::observe`]). One `u64` per
    /// distinct observation — a turn stream id — which for the corpora here is
    /// hundreds to low thousands per scope.
    observed: HashSet<u64>,
    /// [`Self::floor`], kept until a child's level changes. The floor is a
    /// percentile over every child's level — a sort of the whole scope — and
    /// a pooled scope is normalized once per document in it, so recomputing it
    /// per call sorted ~1,000 levels ~1,000 times a scan: 9.7 ms of a live
    /// reprojection's belief scan (`.5`, 2026-10-17). Only [`Self::observe`]
    /// changes a level, and it clears this.
    floor: OnceLock<f32>,
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
        probe_tokens: usize,
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
                (k.clone(), cfg.scale * to_ref(*r, probe_tokens, cfg) / denom)
            })
            .collect()
    }

    /// Fold a turn's raw scores into each child's hit-level EWMA, creating an
    /// unseen child at the prior. Additive: existing children not named here keep
    /// their level (call [`Self::retain`] to prune after a membership change).
    pub(super) fn observe(
        &mut self,
        raw: &[(ChildKey, f32)],
        cfg: &NormConfig,
        probe_tokens: usize,
    ) {
        self.floor = OnceLock::new();
        for (k, r) in raw {
            self.children
                .entry(k.clone())
                .or_insert_with(|| HitLevel::new(cfg.hit_prior))
                .observe(to_ref(*r, probe_tokens, cfg), cfg);
        }
    }

    /// Denominator floor for this scope: the `floor_pctl` percentile of current
    /// hit levels, hard-floored at `floor_min`. Empty scope ⇒ `floor_min`.
    ///
    /// Floors compare against reference-scaled scores (see [`to_ref`]) because
    /// that is what the levels now hold, so nothing here needs its own scaling.
    fn floor(&self, cfg: &NormConfig) -> f32 {
        *self.floor.get_or_init(|| self.compute_floor(cfg))
    }

    /// [`Self::floor`] from the levels as they stand.
    fn compute_floor(&self, cfg: &NormConfig) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(scores: &[(u64, f32)]) -> Vec<(ChildKey, f32)> {
        scores
            .iter()
            .map(|&(k, v)| (ChildKey::turn(k), v))
            .collect()
    }

    /// **The kept floor is the floor the levels give**, before and after an
    /// observe moves them: a normalize after an observe divides by the new
    /// percentile, never a stale one, and repeated normalizes agree exactly.
    #[test]
    fn the_kept_floor_follows_every_observe() {
        let cfg = NormConfig::default();
        let mut scope = ScopeState::default();
        scope.observe(
            &raw(&[(1, 40.0), (2, 900.0), (3, 120.0)]),
            &cfg,
            cfg.probe_t_ref,
        );
        let probe = raw(&[(1, 50.0), (2, 50.0), (3, 50.0), (4, 50.0)]);
        let first = scope.normalize_with_floors(&probe, &[], &cfg, cfg.probe_t_ref);
        assert_eq!(scope.floor.get().copied(), Some(scope.compute_floor(&cfg)));
        let again = scope.normalize_with_floors(&probe, &[], &cfg, cfg.probe_t_ref);
        assert_eq!(
            first, again,
            "a kept floor normalizes as the computed one did"
        );

        // Raise every level well past the old floor: the next normalize must
        // divide by the new one.
        for _ in 0..20 {
            scope.observe(
                &raw(&[(1, 5_000.0), (2, 9_000.0), (3, 7_000.0)]),
                &cfg,
                cfg.probe_t_ref,
            );
        }
        assert_eq!(scope.floor.get(), None, "an observe drops the kept floor");
        let after = scope.normalize_with_floors(&probe, &[], &cfg, cfg.probe_t_ref);
        assert_eq!(scope.floor.get().copied(), Some(scope.compute_floor(&cfg)));
        let fresh = ScopeState {
            children: scope.children.clone(),
            ..ScopeState::default()
        }
        .normalize_with_floors(&probe, &[], &cfg, cfg.probe_t_ref);
        assert_eq!(
            after, fresh,
            "the same levels, the same scores, kept or not"
        );
    }
}
