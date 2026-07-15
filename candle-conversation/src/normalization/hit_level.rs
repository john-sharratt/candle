//! One candidate's learned **hit level** — the score it reaches when it is the
//! answer — tracked as an asymmetric EWMA.

use super::NormConfig;

/// A child's hit level plus how many observations shaped it.
#[derive(Clone, Copy, Debug)]
pub(super) struct HitLevel {
    level: f32,
    count: u32,
}

impl HitLevel {
    /// A fresh child, seeded at the cold-start prior.
    pub(super) fn new(prior: f32) -> Self {
        HitLevel {
            level: prior,
            count: 0,
        }
    }

    pub(super) fn level(&self) -> f32 {
        self.level
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
        self.count = self.count.saturating_add(1);
    }
}
