//! What a forward reads of a routed model's experts: the distinct experts its
//! tokens select, layer by layer, and how many bytes that is.
//!
//! A token picks `k` of `E` experts per MoE layer. Over `t` tokens drawn
//! independently, an expert goes unpicked with probability `(1 - k/E)^t`, so a
//! layer reads `E · (1 - (1 - k/E)^t)` distinct experts on average. The real
//! routing is correlated — tokens of one sequence share experts — so this is
//! the upper bound a batch of unrelated rows approaches; the fit's
//! coefficients absorb the difference.

/// The routed part of a model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertGeometry {
    /// MoE layers.
    pub layers: u32,
    /// Experts per layer.
    pub experts: u32,
    /// Experts each token selects per layer.
    pub per_token: u32,
    /// Bytes of one expert's weights in one layer.
    pub slot_bytes: u64,
}

impl ExpertGeometry {
    /// Qwen3.8-Flash-Next at the Q4_KOEXP pack every member serves.
    pub const FLASH_NEXT: ExpertGeometry = ExpertGeometry {
        layers: 49,
        experts: 512,
        per_token: 10,
        slot_bytes: 2_611_200,
    };

    /// Distinct expert slots a forward of `tokens` reads, over every layer.
    pub fn distinct(&self, tokens: f64) -> f64 {
        let e = f64::from(self.experts);
        let unpicked = (1.0 - f64::from(self.per_token) / e).powf(tokens.max(0.0));
        f64::from(self.layers) * e * (1.0 - unpicked)
    }

    /// Bytes of expert weights a forward of `tokens` reads.
    pub fn bytes(&self, tokens: f64) -> f64 {
        self.distinct(tokens) * self.slot_bytes as f64
    }

    /// Every expert slot, every layer: what a fully resident zone holds.
    pub fn all_bytes(&self) -> f64 {
        f64::from(self.layers) * f64::from(self.experts) * self.slot_bytes as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: ExpertGeometry = ExpertGeometry::FLASH_NEXT;

    /// One token reads exactly its `k` per layer; nothing reads nothing; a
    /// huge batch reads every expert.
    #[test]
    fn distinct_experts_run_from_k_to_all() {
        assert!((G.distinct(1.0) - 490.0).abs() < 1e-9);
        assert_eq!(G.distinct(0.0), 0.0);
        assert!((G.distinct(1e6) - 49.0 * 512.0).abs() < 1e-6);
    }

    /// Sixteen rows read about 13.9× one row's experts, not 16×: they share.
    #[test]
    fn rows_share_experts() {
        let ratio = G.distinct(16.0) / G.distinct(1.0);
        assert!((ratio - 13.8569).abs() < 1e-4, "{ratio}");
    }

    /// The whole zone is 65.5 GB, what the pinned host copy holds.
    #[test]
    fn the_whole_zone_is_the_pack() {
        assert_eq!(G.all_bytes(), 65_509_785_600.0);
    }
}
