//! The schema's `hits:` block (`docs/hit_driven_projection.md` §6.1, §7.1).
//!
//! A selecting model counts, per placed piece, how often its indexer chose
//! the piece's blocks — or would have, for a piece with no K/V. The block says
//! where those counts are taken (percentiles over the selecting layers) and how
//! they move a piece's score: a hit momentum `h ← (1 − β)·h + d` over each
//! stride's hit density `d`, and a score of `base · (1 + gain · h)`.

use std::collections::HashMap;
use std::hash::Hash;

use candle_transformers::models::hit_layers::HitPercentiles;

/// Hit momentum below which a key leaves its map: read so little, for so
/// long, that it lifts nothing.
const HIT_FLOOR: f32 = 1e-4;

/// Hit momentum per key — a working-set conversation, a placed turn — folded
/// once per stride by [`HitsConfig::fold`] and lifting a score by
/// [`HitsConfig::score`].
#[derive(Debug, Clone, PartialEq)]
pub struct HitMomentum<K: Eq + Hash> {
    h: HashMap<K, f32>,
}

impl<K: Eq + Hash> Default for HitMomentum<K> {
    fn default() -> Self {
        Self { h: HashMap::new() }
    }
}

impl<K: Eq + Hash + Copy> HitMomentum<K> {
    /// Fold one stride in: every key leaks, each key of `density` adds its
    /// density, and a key that lifts nothing leaves.
    pub fn fold(&mut self, density: impl IntoIterator<Item = (K, f32)>, cfg: &HitsConfig) {
        for h in self.h.values_mut() {
            *h = cfg.fold(*h, 0.0);
        }
        for (k, d) in density {
            if d > 0.0 {
                *self.h.entry(k).or_insert(0.0) += d;
            }
        }
        self.h.retain(|_, h| *h > HIT_FLOOR);
    }

    /// `k`'s hit momentum — `0` for one not read.
    pub fn get(&self, k: K) -> f32 {
        self.h.get(&k).copied().unwrap_or(0.0)
    }

    /// `k`'s lift: what a base of 1 scores at its momentum, `1 + gain · h`.
    pub fn lift(&self, k: K, cfg: &HitsConfig) -> f32 {
        cfg.score(1.0, self.get(k))
    }

    /// Every key with momentum and its lift.
    pub fn lifts(&self, cfg: &HitsConfig) -> Vec<(K, f32)> {
        self.h
            .iter()
            .map(|(&k, &h)| (k, cfg.score(1.0, h)))
            .collect()
    }
}

/// How hits are counted and how they move a piece's score.
#[derive(Debug, Clone, PartialEq)]
pub struct HitsConfig {
    /// The selecting layers that count hits, as percentiles over them.
    pub layers: HitPercentiles,
    /// How far a piece's hit momentum lifts its score: `base · (1 + gain · h)`.
    pub gain: f32,
    /// The hit momentum's leak: `h ← (1 − β)·h + d`.
    pub beta: f32,
}

impl HitsConfig {
    /// `momentum` after one more stride at `density`.
    pub fn fold(&self, momentum: f32, density: f32) -> f32 {
        (1.0 - self.beta) * momentum + density
    }

    /// A piece's score from its base and its hit momentum.
    pub fn score(&self, base: f32, momentum: f32) -> f32 {
        base * (1.0 + self.gain * momentum)
    }

    /// `layers`, `gain` and `beta`, refused when the sample is empty or past
    /// 100, the gain is negative or not finite, or the leak is outside
    /// `(0, 1]`.
    pub fn new(layers: Vec<u8>, gain: f32, beta: f32) -> Result<Self, String> {
        let layers = HitPercentiles::new(layers)?;
        if !gain.is_finite() || gain < 0.0 {
            return Err(format!("hits.gain must be a finite gain >= 0, got {gain}"));
        }
        if !beta.is_finite() || beta <= 0.0 || beta > 1.0 {
            return Err(format!("hits.beta must be in (0, 1], got {beta}"));
        }
        Ok(Self { layers, gain, beta })
    }
}

impl Default for HitsConfig {
    /// The standard sample, a gain of 1 — a piece read in full on every query
    /// settles at six times its base — and the working set's own leak.
    fn default() -> Self {
        Self {
            layers: HitPercentiles::standard(),
            gain: 1.0,
            beta: 0.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HitMomentum, HitsConfig};

    /// A key's momentum folds its density each stride and leaks without one;
    /// its lift is `1 + gain · h`; a key that has decayed to nothing leaves.
    #[test]
    fn momentum_folds_lifts_and_leaves() {
        let cfg = HitsConfig::default();
        let mut m: HitMomentum<u32> = HitMomentum::default();
        m.fold([(7, 1.0), (8, 0.0)], &cfg);
        m.fold([(7, 1.0)], &cfg);
        assert!((m.get(7) - 1.8).abs() < 1e-6);
        assert!((m.lift(7, &cfg) - 2.8).abs() < 1e-6);
        assert_eq!(m.get(8), 0.0, "a zero density adds nothing");
        assert_eq!(m.lift(9, &cfg), 1.0);
        for _ in 0..60 {
            m.fold([], &cfg);
        }
        assert!(m.lifts(&cfg).is_empty(), "0.8^60 is under the floor");
    }

    /// A piece read in full every stride settles at `1 / β` = 5, and so at
    /// six times its base under the default gain; one never read keeps its
    /// base.
    #[test]
    fn a_piece_read_every_stride_settles_at_six_times_its_base() {
        let c = HitsConfig::default();
        let mut h = 0.0;
        for _ in 0..200 {
            h = c.fold(h, 1.0);
        }
        assert!((h - 1.0 / c.beta).abs() < 1e-4, "{h}");
        assert!((c.score(100.0, h) - 600.0).abs() < 1e-2);
        assert_eq!(c.score(100.0, 0.0), 100.0);
    }

    /// The leak carries about the last five strides: one stride's density is
    /// down to a third after five more without hits.
    #[test]
    fn one_strides_density_fades_over_about_five_strides() {
        let c = HitsConfig::default();
        let mut h = c.fold(0.0, 1.0);
        for _ in 0..5 {
            h = c.fold(h, 0.0);
        }
        assert_eq!(h, 0.8f32.powi(5));
        assert!(h < 0.34 && h > 0.32);
    }

    #[test]
    fn a_bad_block_is_refused() {
        assert!(HitsConfig::new(vec![], 1.0, 0.2).is_err());
        assert!(HitsConfig::new(vec![101], 1.0, 0.2).is_err());
        assert!(HitsConfig::new(vec![50], -1.0, 0.2).is_err());
        assert!(HitsConfig::new(vec![50], f32::NAN, 0.2).is_err());
        assert!(HitsConfig::new(vec![50], 1.0, 0.0).is_err());
        assert!(HitsConfig::new(vec![50], 1.0, 1.5).is_err());
        assert!(HitsConfig::new(vec![50], 0.0, 1.0).is_ok());
    }
}
