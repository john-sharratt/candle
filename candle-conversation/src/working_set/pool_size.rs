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
    /// 6 of each kind on the 72 GB card, 1M and 4 on the 24 GB card, 512K and
    /// 4 on the 16 GB card.
    pub fn for_card(total_bytes: u64) -> Self {
        const GIB: u64 = 1 << 30;
        match total_bytes {
            b if b >= 48 * GIB => Self {
                tokens: 2_000_000,
                per_kind: 6,
            },
            b if b >= 20 * GIB => Self {
                tokens: 1_000_000,
                per_kind: 4,
            },
            _ => Self {
                tokens: 512_000,
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
                tokens: 1_000_000,
                per_kind: 4
            }
        );
        assert_eq!(
            PoolSize::for_card(16_376 << 20),
            PoolSize {
                tokens: 512_000,
                per_kind: 4
            }
        );
        assert_eq!(PoolSize::default(), PoolSize::for_card(16_376 << 20));
    }
}
