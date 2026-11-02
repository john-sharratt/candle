//! What a store's persistence layer holds in memory for the memory census
//! (`docs/memory_census.md` §5): the replica location index origin by origin,
//! the live-record accounting, and what it has applied from other members.

use serde::Serialize;

use super::applied_mix::AppliedLine;
use super::replica::OriginGauge;
use super::SubstratePersistence;

/// The persistence layer's share.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct PersistenceGauge {
    pub location_index: Vec<OriginGauge>,
    /// The location index's capacity across every origin, in bytes.
    pub location_index_bytes: u64,
    /// Live records the accounting tracks.
    pub accounting_entries: u64,
    /// Records applied from other members since open, by type, largest first.
    pub applied: Vec<AppliedLine>,
}

impl SubstratePersistence {
    /// What this layer holds now. O(index slots).
    pub fn gauge(&self) -> PersistenceGauge {
        let location_index = self.locations.gauge();
        PersistenceGauge {
            location_index_bytes: location_index.iter().map(|o| o.bytes).sum(),
            location_index,
            accounting_entries: self.accounting.entries() as u64,
            applied: self.applied_mix.lines(),
        }
    }
}
