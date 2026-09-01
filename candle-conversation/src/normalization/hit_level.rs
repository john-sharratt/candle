//! One candidate's learned **hit level** — the score it reaches when it is the
//! answer — tracked as an asymmetric EWMA.

use super::NormConfig;

/// A child's hit level plus how many observations shaped it, and the raw PEAK
/// it has ever reached in observed traffic.
#[derive(Clone, Copy, Debug)]
pub(super) struct HitLevel {
    level: f32,
    count: u32,
    peak: f32,
}

impl HitLevel {
    /// A fresh child, seeded at the cold-start prior. The peak starts at zero —
    /// it reflects only REAL observed traffic, never the prior.
    pub(super) fn new(prior: f32) -> Self {
        HitLevel {
            level: prior,
            count: 0,
            peak: 0.0,
        }
    }

    pub(super) fn level(&self) -> f32 {
        self.level
    }

    /// The highest raw score ever observed for this child — the
    /// traffic-relative denominator the Concept A.4 floored path normalizes
    /// against (`docs/provenance_adaptive_projection.md` §3): a promiscuous
    /// child's peak is high (it hits on everything), a quiet child's stays at
    /// its best genuine hit, so a fresh hit near that peak stands out at ~1000
    /// regardless of the child's absolute loudness.
    pub(super) fn peak(&self) -> f32 {
        self.peak
    }

    /// Distinct observations folded in — how much this level is worth trusting,
    /// which is what a subdivided scope consults before using it as a
    /// denominator (see `NormalizationCache::normalize_with_fallback`).
    pub(super) fn count(&self) -> u32 {
        self.count
    }

    /// Fold one raw observation in. **Asymmetric:** a raw above the current level
    /// pulls it up fast (`alpha_up`); anything else lets it decay slowly
    /// (`alpha_dn`). So the level settles at the child's *strong-match* magnitude
    /// rather than its mean, and a rarely-queried child does not collapse toward
    /// zero between hits.
    ///
    /// Idempotency is enforced one level up, per OBSERVATION rather than per
    /// value — see [`super::cache::NormalizationCache::observe`]. Deduplicating
    /// on the value here would be wrong: a promiscuous child scores about the
    /// same on everything, so value-dedup would record it once and never learn
    /// that it is loud on *all* traffic, which is precisely the thing the hit
    /// level exists to discount.
    /// **The count follows scoring children only, while the level follows all
    /// of them.** They answer different questions: the level is "how loud is
    /// this child here", which a zero legitimately drags down, and the count is
    /// "how much has this scope actually taught me about this child", which a
    /// zero says nothing about.
    ///
    /// Counting every child in the slice made them the same question.
    /// `ScopeState::observe` folds the whole raw slice and production always
    /// passes every section, so each fold bumped every child equally and the
    /// per-child `min_observations` gate in `normalize_with_fallback` degenerated
    /// into a per-SCOPE one: after 8 folds every member read warm, including the
    /// ones the subdivision had never seen score. That is exactly the
    /// learning-starved denominator the fallback exists to avoid — measured at
    /// code Top-1 57.1% → 47.5%, and 0/6 against 6/6 for a cold tool lens.
    pub(super) fn observe(&mut self, raw: f32, cfg: &NormConfig) {
        let alpha = if raw > self.level {
            cfg.alpha_up
        } else {
            cfg.alpha_dn
        };
        self.level += alpha * (raw - self.level);
        self.peak = self.peak.max(raw);
        if raw > 0.0 {
            self.count = self.count.saturating_add(1);
        }
    }
}
