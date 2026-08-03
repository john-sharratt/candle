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

    #[cfg(test)]
    pub(super) fn count(&self) -> u32 {
        self.count
    }

    /// Fold one raw observation in. **Asymmetric:** a raw above the current level
    /// pulls it up fast (`alpha_up`); anything else lets it decay slowly
    /// (`alpha_dn`). So the level settles at the child's *strong-match* magnitude
    /// rather than its mean, and a rarely-queried child does not collapse toward
    /// zero between hits.
    pub(super) fn observe(&mut self, raw: f32, cfg: &NormConfig) {
        let alpha = if raw > self.level {
            cfg.alpha_up
        } else {
            cfg.alpha_dn
        };
        self.level += alpha * (raw - self.level);
        self.peak = self.peak.max(raw);
        self.count = self.count.saturating_add(1);
    }
}
