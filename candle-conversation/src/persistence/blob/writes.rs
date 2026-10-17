//! Appending a blob, and reading one back.

use super::index::BlobIndex;
use super::payload::{BlobKey, BlobPayload};
use crate::persistence::record::RecordType;
use crate::persistence::{Result, SubstratePersistence};

impl SubstratePersistence {
    /// Stage `bytes` as a blob, unless one of their hash is already held;
    /// returns their SHA-256 either way. Staged, not made durable: the next
    /// group commit makes it so.
    ///
    /// A read-only handle refuses with
    /// [`PersistenceError::ReadOnly`](crate::persistence::PersistenceError::ReadOnly).
    pub fn write_blob(&mut self, bytes: Vec<u8>) -> Result<[u8; 32]> {
        let blob = BlobPayload::of(bytes);
        let key = blob.key();
        if self.blob_index.get(key).is_none() {
            let payload = blob.encode();
            let loc = self
                .append_record(RecordType::Blob, 0, key.hi, key.lo, 0, 0, &payload)?
                .loc(payload.len() as u64);
            self.blob_index.record(key, loc);
        }
        Ok(blob.sha256)
    }

    /// The bytes named `sha256`, or `None` when no blob of that hash is held.
    /// Whatever is staged is committed first, so a blob reads back the moment
    /// after it was written.
    pub fn blob(&mut self, sha256: &[u8; 32]) -> Result<Option<Vec<u8>>> {
        self.commit_if_pending()?;
        let Some(loc) = self.blob_index.get(BlobKey::of(sha256)) else {
            return Ok(None);
        };
        let record = self
            .segments
            .read_record_at(loc.segment, loc.offset, loc.record_size)?;
        let blob = BlobPayload::decode(&record.payload)?;
        Ok((&blob.sha256 == sha256).then_some(blob.bytes))
    }

    /// Whether a blob of `sha256` is held, without reading it.
    pub fn has_blob(&self, sha256: &[u8; 32]) -> bool {
        self.blob_index.get(BlobKey::of(sha256)).is_some()
    }

    /// Where every blob is.
    pub fn blob_index(&self) -> &BlobIndex {
        &self.blob_index
    }
}
