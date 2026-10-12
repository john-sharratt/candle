//! Where a tenant stands in line against the others sharing one engine.
//!
//! Not scheduling priority: the scheduler ranks work by its band alone, and
//! tenants are equal there (`docs/shared_model_host.md` §3.6). The rank orders
//! the tenants where an order is needed and nothing is being prioritised —
//! which of them builds the shared engine with every tenant's merged requests,
//! the order those requests merge in, and the order they stop in. The host
//! assigns ranks in the order it is given tenants.

/// A tenant's place in line: [`TenantRank::FIRST`] comes first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct TenantRank(u8);

impl TenantRank {
    /// The highest rank — the first tenant, and every engine with only one.
    pub const FIRST: TenantRank = TenantRank(0);

    /// How many tenants one engine ranks.
    pub const MAX: usize = 4;

    /// The lowest rank one engine gives.
    pub const LAST: TenantRank = TenantRank((Self::MAX - 1) as u8);

    /// The rank of the `index`-th tenant, counting from zero. Refused past
    /// [`Self::MAX`].
    pub fn nth(index: usize) -> Option<TenantRank> {
        (index < Self::MAX).then_some(TenantRank(index as u8))
    }

    /// The rank as a zero-based position, `0` highest.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_count_from_the_first_tenant_and_stop_at_the_maximum() {
        assert_eq!(TenantRank::nth(0), Some(TenantRank::FIRST));
        assert_eq!(TenantRank::nth(1).map(TenantRank::index), Some(1));
        assert_eq!(TenantRank::nth(TenantRank::MAX), None);
        assert_eq!(TenantRank::default(), TenantRank::FIRST);
    }
}
