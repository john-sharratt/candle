//! Appending a blob, reading one back, and retiring one.

use super::index::BlobIndex;
use super::payload::{BlobKey, BlobPayload};
use super::{forget_if_retired, forget_retired};
use crate::persistence::record::{CustomObjectPayload, RecordType};
use crate::persistence::{Result, SubstratePersistence};
use crate::substrate::Substrate;

impl SubstratePersistence {
    /// Stage `bytes` as a blob, unless one of their hash is already held;
    /// returns their SHA-256 either way. Staged, not made durable: the next
    /// group commit makes it so. A blob retired before is written again, and
    /// its new record outranks the tombstone.
    ///
    /// A read-only handle refuses with
    /// [`PersistenceError::ReadOnly`](crate::persistence::PersistenceError::ReadOnly).
    pub fn write_blob(&mut self, bytes: Vec<u8>) -> Result<[u8; 32]> {
        let blob = BlobPayload::of(bytes);
        let key = blob.key();
        if self.blob_index.get(key).is_none() {
            let payload = blob.encode();
            let at = self.append_record(RecordType::Blob, 0, key.hi, key.lo, 0, 0, &payload)?;
            self.blob_index
                .record(key, at.loc(payload.len() as u64), at.identity);
        }
        Ok(blob.sha256)
    }

    /// Retire the blob `sha256` names, durably: its tombstone
    /// ([`BlobKey::tombstone_key`], ceased) is written and applied to
    /// `substrate`, the blob reads as absent from now on, and the next
    /// compaction or maintenance pass over its segment lets its bytes go. The
    /// same bytes written afterwards live again. Retiring a blob not held
    /// writes the tombstone all the same, so a copy still on its way dies too.
    pub fn retire_blob(&mut self, substrate: &mut Substrate, sha256: &[u8; 32]) -> Result<()> {
        let key = BlobKey::of(sha256);
        let tombstone = CustomObjectPayload::ceased(key.tombstone_key());
        let id = self.mint();
        self.write_custom_object(&tombstone, id)?;
        self.commit()?;
        substrate.apply_custom_object(tombstone, id);
        self.forget_retired_blob(substrate, key);
        Ok(())
    }

    /// Forget blob `key` when the tombstone standing on it in `substrate`
    /// outranks the record held, and count that record dead.
    pub(crate) fn forget_retired_blob(&mut self, substrate: &Substrate, key: BlobKey) {
        forget_if_retired(&mut self.blob_index, &mut self.accounting, substrate, key);
    }

    /// Forget every blob a tombstone in `substrate` outranks — once an open's
    /// walk has met every blob and every tombstone, in log order.
    pub(crate) fn forget_retired_blobs(&mut self, substrate: &Substrate) {
        forget_retired(&mut self.blob_index, &mut self.accounting, substrate);
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
