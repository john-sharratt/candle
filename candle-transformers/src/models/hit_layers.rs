//! Which selecting layers count hits (`docs/hit_driven_projection.md` §6.1).
//!
//! A selecting attention ranks its candidates at every layer that has an
//! index. Counting what each ranking chose, and scoring the pieces that have no
//! K/V against the index, at every one of them would cost every layer's index
//! rows for every unloaded piece. A sample does the work: the caller names
//! percentiles over the selecting layers, and only the layers they land on keep
//! index rows for unloaded pieces and count hits.
//!
//! Like [`super::selection_strata::StrataTokens`], this names no architecture:
//! a model whose attention reads every position has no selection to count, and
//! the setting is nothing to it.

use std::ops::Range;

use crate::models::piece_key::PieceKey;

/// What one sequence's selecting layers chose since the last read: one count
/// per declared round — the loaded rounds first, then the pooled ones the
/// shadow scan reads (`docs/hit_driven_projection.md` §6.3) — and how many
/// queries could have hit each kind, what a hit density divides by (§6.4).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PieceHits {
    /// Hits per declared round, in declaration order.
    pub hits: Vec<u32>,
    /// Blocks each declared round owns — what its hits are out of.
    pub blocks: Vec<u32>,
    /// How many of the declared rounds, from the first, are loaded; the rest
    /// are pooled.
    pub loaded: usize,
    /// Query rows counted against the loaded rounds, summed over the sampled
    /// layers.
    pub queries: u64,
    /// Kept queries the shadow scan scored against the pooled rounds, summed
    /// over the sampled layers.
    pub shadow_queries: u64,
}

impl PieceHits {
    /// The queries round `i` could have been hit by: the selection's for a
    /// loaded round, the shadow scan's for a pooled one.
    pub fn queries_of(&self, i: usize) -> u64 {
        if i < self.loaded {
            self.queries
        } else {
            self.shadow_queries
        }
    }
}

/// A pooled round the projection did not place, named for the shadow scan
/// (`docs/hit_driven_projection.md` §6.3): the piece whose index pages hold
/// it, and its positions within the piece.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PooledRound {
    pub piece: PieceKey,
    pub positions: Range<usize>,
}

/// Percentiles over a model's selecting layers, in `0..=100`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HitPercentiles(Vec<u8>);

impl HitPercentiles {
    /// The 40th, 65th and 90th percentiles: one layer past the middle of the
    /// stack and two in its upper part, where the selection has the most
    /// context to judge by.
    pub fn standard() -> Self {
        Self(vec![40, 65, 90])
    }

    /// `percentiles`, refused when it is empty or names one past 100.
    pub fn new(percentiles: Vec<u8>) -> Result<Self, String> {
        if percentiles.is_empty() {
            return Err("hits.layers names no percentile".into());
        }
        if let Some(p) = percentiles.iter().find(|&&p| p > 100) {
            return Err(format!("hits.layers: percentile {p} is past 100"));
        }
        Ok(Self(percentiles))
    }

    /// The percentiles as given.
    pub fn percentiles(&self) -> &[u8] {
        &self.0
    }

    /// The selecting layers these land on, as indices into a model's `n`
    /// selecting layers in layer order, ascending and each once: percentile
    /// `p` is index `round(p / 100 · (n − 1))`. Empty for a model with none.
    pub fn indices(&self, n: usize) -> Vec<usize> {
        if n == 0 {
            return Vec::new();
        }
        let mut out: Vec<usize> = self
            .0
            .iter()
            .map(|&p| ((p as f64 / 100.0) * (n - 1) as f64).round() as usize)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

impl Default for HitPercentiles {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::HitPercentiles;

    /// Flash-Next has twelve selecting layers (model layers 3, 7, … 47): the
    /// standard sample is its 5th, 8th and 11th — model layers 19, 31 and 43.
    #[test]
    fn the_standard_sample_on_twelve_selecting_layers() {
        let idx = HitPercentiles::standard().indices(12);
        assert_eq!(idx, vec![4, 7, 10]);
        let model_layers: Vec<usize> = idx.iter().map(|i| 3 + 4 * i).collect();
        assert_eq!(model_layers, vec![19, 31, 43]);
    }

    /// Two percentiles that land on one layer use it once, and the ends are
    /// the first and last selecting layer.
    #[test]
    fn landing_twice_counts_once_and_the_ends_are_the_ends() {
        let p = HitPercentiles::new(vec![0, 1, 100, 99]).unwrap();
        assert_eq!(p.indices(12), vec![0, 11]);
        assert_eq!(p.indices(1), vec![0]);
        assert_eq!(p.indices(0), Vec::<usize>::new());
    }

    #[test]
    fn an_empty_or_out_of_range_sample_is_refused() {
        assert!(HitPercentiles::new(Vec::new()).is_err());
        assert!(HitPercentiles::new(vec![50, 101]).is_err());
        assert!(HitPercentiles::new(vec![100]).is_ok());
    }
}
