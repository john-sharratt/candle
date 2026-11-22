//! How large a dialogue's pool is (`docs/hit_driven_projection.md` §7.2
//! step 1): the larger of a token budget and a count of pieces of each kind,
//! by the card alone. No configuration overrides it.

/// The pool's size on one card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolSize {
    /// Tokens the pool holds by strength across kinds, once each kind has its
    /// [`Self::per_kind`].
    pub tokens: usize,
    /// Pieces of each kind — readings, folders, guides, journal notes — the
    /// pool holds by that kind's own strength, whatever they cost.
    pub per_kind: usize,
}

impl PoolSize {
    /// The pool a card of `total_bytes` of device memory holds: 2M tokens and
    /// 6 of each kind on the 72 GB card, 500K and 4 on the 24 GB card, 256K and
    /// 4 on the 16 GB card.
    ///
    /// **The smaller cards' pools are a quarter and an eighth of the 72 GB
    /// card's, not a half and a quarter.** A pooled piece's index rows stand in
    /// device memory beside the expert zone, and on the 24 GB card the QSA
    /// index tenant held 3.7 GiB while the zone sat on its 1,537-slot floor —
    /// under 7% of the experts resident, 1,000–2,000 cold reads a wave
    /// (WhiteWorld, 2026-11-22). Every MiB the pool gives back is about a third
    /// of an expert slot.
    pub fn for_card(total_bytes: u64) -> Self {
        const GIB: u64 = 1 << 30;
        match total_bytes {
            b if b >= 48 * GIB => Self {
                tokens: 2_000_000,
                per_kind: 6,
            },
            b if b >= 20 * GIB => Self {
                tokens: 500_000,
                per_kind: 4,
            },
            _ => Self {
                tokens: 256_000,
                per_kind: 4,
            },
        }
    }
}

impl Default for PoolSize {
    /// The smallest card's, until the engine names its own.
    fn default() -> Self {
        Self::for_card(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_card_has_its_pool() {
        assert_eq!(
            PoolSize::for_card(73_415 << 20),
            PoolSize {
                tokens: 2_000_000,
                per_kind: 6
            }
        );
        assert_eq!(
            PoolSize::for_card(24_576 << 20),
            PoolSize {
                tokens: 500_000,
                per_kind: 4
            }
        );
        assert_eq!(
            PoolSize::for_card(16_376 << 20),
            PoolSize {
                tokens: 256_000,
                per_kind: 4
            }
        );
        assert_eq!(PoolSize::default(), PoolSize::for_card(16_376 << 20));
    }
}
