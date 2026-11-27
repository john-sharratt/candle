//! The compression a sealed section's K/V takes.
//!
//! **A boundary section** — a role marker, an opening or closing tag, a fixed
//! section of the system prompt, anything outside a collection — is read back
//! over by every later token, so it is sealed near-lossless: K at Q8_KS, V at
//! Q8_0, every block 8-bit, with no adaptive choice. Measured, a tool catalog
//! sealed adaptively at C5 dropped the model's tool-detection heads below their
//! decision threshold; sealed this way it did not.
//!
//! **A collection member** — a tool's signature, a knowledge-base section — is
//! one of many placed by relevance, so the noise of its K/V is averaged by the
//! selection that chose it, and it is sealed adaptively at its collection's
//! level ([`crate::projection::SectionCollection::kv_level`]): C4 unless the
//! collection says otherwise, C5 for the knowledge base.

use candle_nn::kv_cache::{CompressionPolicy, QuantFormat};

/// How a section is sealed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SectionSeal {
    /// Near-lossless: Q8 K and V.
    Boundary,
    /// Adaptive, at a collection's level.
    Member { level: u8 },
}

impl SectionSeal {
    /// Whether the section is a collection member.
    pub fn is_member(self) -> bool {
        matches!(self, SectionSeal::Member { .. })
    }

    /// The policy its K/V is quantized under.
    pub fn policy(self) -> CompressionPolicy {
        match self {
            SectionSeal::Boundary => CompressionPolicy::new(0)
                .with_override_k_quant(Some(QuantFormat::Q8_KS))
                .with_override_v_quant(Some(QuantFormat::Q8_0)),
            SectionSeal::Member { level } => CompressionPolicy::new(level),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A boundary is Q8 on both sides at C0; a member is adaptive at its
    /// level, neither side pinned.**
    #[test]
    fn each_seal_names_its_policy() {
        let b = SectionSeal::Boundary.policy();
        assert_eq!(b.compression_level, 0);
        assert_eq!(b.override_k_quant, Some(QuantFormat::Q8_KS));
        assert_eq!(b.override_v_quant, Some(QuantFormat::Q8_0));
        let m = SectionSeal::Member { level: 5 }.policy();
        assert_eq!(m.compression_level, 5);
        assert_eq!(m.override_k_quant, None);
        assert_eq!(m.override_v_quant, None);
        assert!(SectionSeal::Member { level: 4 }.is_member());
        assert!(!SectionSeal::Boundary.is_member());
    }
}
