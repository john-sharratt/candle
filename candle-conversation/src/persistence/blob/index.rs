//! Where every blob is.
//!
//! Locations only — no blob's bytes are held — keyed by the first 128 bits of
//! each blob's hash, which its record header carries. Built by the open walk,
//! kept current by every append and every remote record applied, rebuilt
//! after compaction; compaction and maintenance read it to know what to
//! carry, and a read follows it to the bytes.

use std::collections::BTreeMap;

use super::payload::BlobKey;
use crate::persistence::manifest::RecordLoc;

/// Every blob's location. See the module.
#[derive(Debug, Default)]
pub struct BlobIndex {
    blobs: BTreeMap<BlobKey, RecordLoc>,
}

impl BlobIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Blob `key` is at `loc`. Returns whether it is live: the first record
    /// of its hash. A second is the same bytes again, and dead.
    pub fn record(&mut self, key: BlobKey, loc: RecordLoc) -> bool {
        match self.blobs.contains_key(&key) {
            true => false,
            false => {
                self.blobs.insert(key, loc);
                true
            }
        }
    }

    pub fn get(&self, key: BlobKey) -> Option<RecordLoc> {
        self.blobs.get(&key).copied()
    }

    /// Every blob, in key order — the order a rewrite carries them in.
    pub fn blobs(&self) -> &BTreeMap<BlobKey, RecordLoc> {
        &self.blobs
    }

    /// Blob `key`, at `from`, now lives at `to`. A blob no longer at `from` is
    /// left where the index has it.
    pub fn repoint(&mut self, key: BlobKey, from: RecordLoc, to: RecordLoc) {
        if let Some(at) = self.blobs.get_mut(&key) {
            if *at == from {
                *at = to;
            }
        }
    }

    pub fn clear(&mut self) {
        self.blobs.clear();
    }
}
