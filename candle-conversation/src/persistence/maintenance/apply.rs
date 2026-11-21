//! Runs one maintenance op inline, for callers that hold the substrate lock.
//!
//! The daemon runs an op in phases that release the lock between them, and it
//! unlinks retired files off the log's owner. These helpers run every phase in
//! one call under the caller's lock, so a test can force a specific op. A
//! read-only handle refuses, and a file that will not unlink is reported as the
//! op's error.

use super::*;

impl SubstratePersistence {
    /// Test/forced-op convenience: run all phases inline for a specific op under
    /// the caller's already-held substrate lock. A read-only handle refuses
    /// with [`PersistenceError::ReadOnly`].
    pub fn apply_maintenance_op(
        &mut self,
        substrate: &mut Substrate,
        op: &MaintenanceOp,
    ) -> Result<()> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        let ix = self.live_indexes();
        let resident = ix.resident_to_carry(substrate, &op.targets());
        let (
            chunk_relocs,
            token_relocs,
            snapshot_relocs,
            branch_checkpoint_relocs,
            singleton_relocs,
        ) = ix.gather_relocations(substrate, &op.targets());
        let plan = MaintenancePlan {
            op: *op,
            resident,
            chunk_relocs,
            token_relocs,
            index_page_relocs: ix.gather_index_page_relocs(substrate, &op.targets()),
            snapshot_relocs,
            branch_checkpoint_relocs,
            singleton_relocs,
            npc_relocs: ix.npc_relocations(&op.targets()),
            vfs_relocs: ix.vfs_relocations(substrate, &op.targets()),
            blob_relocs: ix.blob_relocations(&op.targets()),
        };
        let result = self.execute_maintenance(&plan)?;
        let stranded = result.apply_to_substrate(&plan, substrate);
        unlink_now(self.finish_maintenance(&plan, &stranded)?)
    }
}

/// Unlink `retired` on this thread, for the single-threaded callers that run
/// an op end to end; a file that will not go is the op's error.
pub(super) fn unlink_now(retired: RetiredSegments) -> Result<()> {
    match retired.unlink().failed.into_iter().next() {
        Some((_, e)) => Err(e.into()),
        None => Ok(()),
    }
}
