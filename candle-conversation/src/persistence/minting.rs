//! Mints the identity of each record this substrate originates: its node, next origin sequence number, clock and generation.
//!
//! A runtime write mints its identity when the value changes in memory, so the value holds the identity its record will carry from the start. A read-only handle over a directory that never had a node has none and mints nothing. The marker records a rewrite writes back keep the identities they already held, so the rewrite does not change which records a marker names.

use super::*;

impl SubstratePersistence {
    /// This substrate's node — the origin of every record it writes (§3.7).
    /// Zero on a read-only handle over a directory that never had one.
    pub fn node(&self) -> u64 {
        self.minter.node()
    }

    /// How this handle came by its node; `None` on a read-only handle.
    pub fn node_source(&self) -> Option<NodeSource> {
        self.node_source
    }

    /// The writer's identity state as it stands: the next `oseq`, the clock,
    /// the generation.
    pub fn origination(&self) -> Origination {
        self.minter.origination()
    }

    /// The shared handle the runtime write paths mint through, so a value set
    /// in memory holds from the start the identity its record will carry.
    pub fn minter(&self) -> Minter {
        self.minter.clone()
    }

    /// The identity of the next record this substrate originates.
    pub fn mint(&self) -> RecordIdentity {
        self.minter.mint(0)
    }

    /// The identity of the next record this substrate originates under the
    /// claim or lease epoch `epoch` (§6.2, §9.2).
    pub fn mint_under(&self, epoch: u64) -> RecordIdentity {
        self.minter.mint(epoch)
    }

    /// The identity of the next record of type `rt` this substrate writes
    /// under stream `stream_id` — carrying the section's generation when the
    /// stream is a section this node is writing ([`Minter::mint_for`]).
    pub fn mint_for(&self, rt: RecordType, stream_id: StreamId) -> RecordIdentity {
        self.minter.mint_for(rt, stream_id.0)
    }

    /// The identity of each marker record a rewrite writes back.
    pub fn marker_ids(&self) -> &MarkerIds {
        &self.marker_ids
    }
}
