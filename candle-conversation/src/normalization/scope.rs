//! One scope's normalization state: a background per child and their pool,
//! plus the read (`normalize`) and write (`observe`) operations over them.

use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};

use super::background::Background;
use super::{ChildKey, NormConfig};

/// One raw score, restated as the score it would have reached over a
/// `cfg.probe_t_ref`-token probe.
///
/// The single place the band's unit is fixed, applied identically on the read
/// and the write path: a background learned at one probe length and a query
/// measured at another would otherwise be compared in different units. See
/// [`NormConfig::probe_t_ref`] for the measured consequences and for why this
/// is linear and uncapped.
///
/// `probe_tokens == 0` would be a scan with no probe, which produces no scores
/// to scale; treating it as one token keeps the function total rather than
/// letting a division by zero reach the band as an infinity.
fn to_ref(raw: f32, probe_tokens: usize, cfg: &NormConfig) -> f32 {
    raw * (cfg.probe_t_ref as f32 / probe_tokens.max(1) as f32)
}

/// Per-child backgrounds for a single score-competition scope. Immutable once
/// shared: an observe edits a copy ([`super::cache::NormalizationCache::observe`]).
#[derive(Default, Clone)]
pub(super) struct ScopeState {
    children: HashMap<ChildKey, Background>,
    /// Every sample any child here was taught: what a child with too little
    /// background of its own is read against.
    pooled: Background,
    /// Evidence already folded into this scope, so folding it again is a no-op
    /// (see [`super::cache::NormalizationCache::observe`]). One `u64` per
    /// distinct observation — a turn stream id.
    observed: HashSet<u64>,
}

impl ScopeState {
    /// Record `source` as folded into this scope. Returns `false` when it was
    /// already present, i.e. the caller should skip the fold.
    pub(super) fn mark_observed(&mut self, source: u64) -> bool {
        self.observed.insert(source)
    }

    /// Each child's raw score on the band: `scale × surprise / ln(1 /
    /// full_hit_tail)`, its tail surprise against its own background — so a
    /// score beyond all but `full_hit_tail` of that background lands at
    /// `scale`. A child with fewer than `min_background` samples here is read
    /// against the scope's pool; a scope whose pool is that thin too, against
    /// the probe's own scores across `raw`. Pure — does not mutate.
    pub(super) fn normalize(
        &self,
        raw: &[(ChildKey, f32)],
        cfg: &NormConfig,
        probe_tokens: usize,
    ) -> Vec<(ChildKey, f32)> {
        let scaled: Vec<f32> = raw
            .iter()
            .map(|(_, r)| to_ref(*r, probe_tokens, cfg))
            .collect();
        let probe_only: OnceCell<Background> = OnceCell::new();
        let unit = cfg.scale / (1.0 / cfg.full_hit_tail).ln();
        raw.iter()
            .zip(&scaled)
            .map(|((k, _), &x)| {
                let background = match self.children.get(k) {
                    Some(b) if b.total() >= cfg.min_background => b,
                    _ if self.pooled.total() >= cfg.min_background => &self.pooled,
                    _ => probe_only.get_or_init(|| Background::of(scaled.iter().copied())),
                };
                (k.clone(), unit * background.surprise(x))
            })
            .collect()
    }

    /// Fold one probe's raw scores into the backgrounds: every child's, but the
    /// probe's `answer` when it is known — a probe of a child's own exemplar is
    /// a hit, not the background a hit is measured against. Additive: children
    /// not named here keep what they hold.
    pub(super) fn observe(
        &mut self,
        raw: &[(ChildKey, f32)],
        answer: Option<&ChildKey>,
        cfg: &NormConfig,
        probe_tokens: usize,
    ) {
        for (k, r) in raw {
            if answer == Some(k) {
                continue;
            }
            let x = to_ref(*r, probe_tokens, cfg);
            self.children.entry(k.clone()).or_default().add(x);
            self.pooled.add(x);
        }
    }

    /// How many background samples `k` holds here; `None` when the scope has
    /// never been taught it.
    #[cfg(test)]
    pub(super) fn background_of(&self, k: &ChildKey) -> Option<u32> {
        self.children.get(k).map(Background::total)
    }
}
