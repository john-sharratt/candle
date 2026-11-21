//! Binds a substrate to the model and tokenizer its turns were sealed under, so token ids keep their meaning across every reopen.
//!
//! The tokenizer is stored whole in the log as a `Tokenizer` record, so the log is a complete image with no companion files. Once bound, a different vocabulary is refused on the handle that bound it and on any reopen, because accepting one would leave every stored turn meaning something else while the manifest reports success. Recovery reads the tokenizer only to compute its hash, so the large record costs one read at open.

use super::*;

/// The first eight bytes of a digest, hex — enough to name which tokenizer a
/// substrate is bound to in an error a person has to act on.
pub(super) fn hex16(digest: &[u8; 32]) -> String {
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 of `bytes` — the tokenizer identity digest.
pub(super) fn sha256(bytes: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

impl SubstratePersistence {
    /// Set the projection template — last-writer-wins, and never written
    /// through a read-only handle.
    pub fn set_template(&mut self, template: &[u8]) -> Result<bool> {
        if self.template.as_deref() == Some(template) || self.is_read_only() {
            return Ok(false);
        }
        self.append_record(RecordType::Template, 0, 0, 0, 0, 0, template)?;
        self.template = Some(template.to_vec());
        Ok(true)
    }

    /// Bind the substrate to a model's `tokenizer.json` bytes.
    ///
    /// Writes once, on a substrate that has no tokenizer yet, and thereafter
    /// **refuses** anything that does not hash identically: the tokenizer is
    /// the vocabulary every turn already in this log was sealed under. Token
    /// ids are only meaningful against it — accept a second one and every
    /// stored turn silently means something else, while the manifest points at
    /// the new record and reports success.
    ///
    /// [`SubstratePersistence::bind_model`] refuses a different vocabulary
    /// first, as part of the model's key; this is the embedded copy's own
    /// guard, so the bytes the log carries can never disagree with the key.
    ///
    /// The full bytes are the `Tokenizer` record's payload, so the log is
    /// a self-contained substrate image (§5.7) — no companion files.
    /// Recovery never reads the payload (the walk skips it and the digest
    /// chain carries only its header), so the ~11 MB record costs one
    /// on-demand read at open to compute the identity hash and nothing else.
    pub fn set_tokenizer(&mut self, tokenizer: &[u8]) -> Result<bool> {
        let hash = sha256(tokenizer);
        if let Some(existing) = self.tokenizer_sha256 {
            if existing == hash {
                return Ok(false);
            }
            return Err(PersistenceError::Corrupt(format!(
                "this substrate is bound to a different tokenizer (recorded {}, offered {}). \
                 Every turn in the log was sealed under the recorded vocabulary, so adopting \
                 another would silently change what all of them say. Point this model at a \
                 fresh substrate, or wipe this one.",
                hex16(&existing),
                hex16(&hash),
            )));
        }
        // A read-only handle refuses a mismatch like any other (above) but
        // never binds a store that has no tokenizer recorded.
        if self.is_read_only() {
            return Ok(false);
        }
        self.append_record(RecordType::Tokenizer, 0, 0, 0, 0, 0, tokenizer)?;
        self.tokenizer_sha256 = Some(hash);
        Ok(true)
    }

    /// Read the tokenizer bytes embedded in the active log's `Tokenizer`
    /// record. `Ok(None)` when the log has no such record yet (no model
    /// has ever called `set_tokenizer` against this substrate). The read
    /// CRC-verifies the payload, so bit rot surfaces as `BadChecksum`.
    pub fn read_tokenizer_bytes(&mut self) -> Result<Option<Vec<u8>>> {
        let Some(loc) = self.manifest.tokenizer else {
            return Ok(None);
        };
        let r = self
            .segments
            .read_record_at(loc.segment, loc.offset, loc.record_size)?;
        Ok(Some(r.payload))
    }

    /// The latest projection template payload, if any.
    pub fn template(&self) -> Option<&[u8]> {
        self.template.as_deref()
    }

    /// SHA-256 of the embedded tokenizer, if a `Tokenizer` record is present.
    pub fn tokenizer_sha256(&self) -> Option<[u8; 32]> {
        self.tokenizer_sha256
    }
}
