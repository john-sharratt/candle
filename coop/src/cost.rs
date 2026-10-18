//! What one forward costs a member, as a sum of the things it does: a fixed
//! cost per sweep, compute per token, attention per attended KV, expert bytes
//! read from the card, and expert bytes that crossed the link because they
//! were not resident.
//!
//! Every coefficient is in milliseconds per unit and never negative — each
//! term is work, and work does not make a forward faster.

use crate::experts::ExpertGeometry;

/// A member's forward cost on one channel.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ForwardCost {
    /// Per forward, whatever its width: launches, norms, the dense layers'
    /// weight reads.
    pub base_ms: f64,
    /// Per token.
    pub token_ms: f64,
    /// Per million attended KV positions, summed over the rows.
    pub mkv_ms: f64,
    /// Per GB of expert weights read from resident slots.
    pub resident_gb_ms: f64,
    /// Per GB of expert weights that were not resident.
    pub missed_gb_ms: f64,
}

/// What a forward does, in the cost's units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Work {
    pub tokens: f64,
    pub kv: f64,
    /// Expert hit rate, 0..=1.
    pub hit: f64,
}

impl Work {
    /// `[1, tokens, MKV, resident GB, missed GB]` over `geometry`.
    pub fn features(&self, geometry: &ExpertGeometry) -> [f64; 5] {
        let gb = geometry.bytes(self.tokens) / 1e9;
        [
            1.0,
            self.tokens,
            self.kv / 1e6,
            self.hit * gb,
            (1.0 - self.hit) * gb,
        ]
    }
}

impl ForwardCost {
    pub fn coefficients(&self) -> [f64; 5] {
        [
            self.base_ms,
            self.token_ms,
            self.mkv_ms,
            self.resident_gb_ms,
            self.missed_gb_ms,
        ]
    }

    pub fn from_coefficients(c: [f64; 5]) -> ForwardCost {
        ForwardCost {
            base_ms: c[0],
            token_ms: c[1],
            mkv_ms: c[2],
            resident_gb_ms: c[3],
            missed_gb_ms: c[4],
        }
    }

    /// Milliseconds a forward doing `work` takes.
    pub fn ms(&self, work: &Work, geometry: &ExpertGeometry) -> f64 {
        self.coefficients()
            .iter()
            .zip(work.features(geometry))
            .map(|(c, x)| c * x)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: ExpertGeometry = ExpertGeometry::FLASH_NEXT;

    /// A decode row at full residency pays the base, one token, its KV and
    /// its experts read from the card — nothing for the link.
    #[test]
    fn a_resident_decode_row_pays_no_link() {
        let c = ForwardCost {
            base_ms: 4.0,
            token_ms: 0.5,
            mkv_ms: 2.0,
            resident_gb_ms: 1.0,
            missed_gb_ms: 80.0,
        };
        let w = Work {
            tokens: 1.0,
            kv: 500_000.0,
            hit: 1.0,
        };
        let gb = G.bytes(1.0) / 1e9;
        assert!((c.ms(&w, &G) - (4.0 + 0.5 + 1.0 + gb)).abs() < 1e-12);
        let missing = Work { hit: 0.0, ..w };
        assert!((c.ms(&missing, &G) - (4.0 + 0.5 + 1.0 + 80.0 * gb)).abs() < 1e-12);
    }
}
