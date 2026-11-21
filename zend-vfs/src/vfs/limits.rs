//! The size limits on what a store holds and what it returns to a tool.
//!
//! The session layer holds at most 10 MiB, counted over every chain, and a change
//! past that is refused with the layer left exactly as it was, deletions
//! included. Workspace files are read on demand and never retained, so they cost
//! nothing against that cap. A workspace file above `MAX_LOWER_FILE_BYTES` still
//! lists with its true size, but is refused when read into a tool response.

use super::*;

/// The most a store's session layer holds, every chain counted: past it a
/// change is refused [`VfsError::Full`].
pub const MAX_BYTES: usize = 10 * 1024 * 1024;

// 10 MiB

/// Largest workspace file the lower layer will read into a tool response.
/// Listing is unaffected — an oversize file still shows up with its true size.
pub const MAX_LOWER_FILE_BYTES: u64 = 4 * 1024 * 1024;

impl VfsStore {
    /// Refusal of a lower-layer file above [`MAX_LOWER_FILE_BYTES`].
    pub(super) fn too_large(norm: &str, bytes: u64) -> VfsError {
        VfsError::Unreadable(format!(
            "{norm} is {bytes} bytes, above the {MAX_LOWER_FILE_BYTES}-byte workspace read limit",
        ))
    }

    /// Bytes the session's deltas hold. Workspace files cost nothing — they
    /// are read on demand and never retained — and neither does the unchanged
    /// part of a file the session edited.
    pub fn total_bytes(&self) -> usize {
        self.upper.read().unwrap().bytes()
    }

    // ── Upper-layer helpers ──────────────────────────────────────────────────

    /// Install `chain` as `norm`'s, unless the session layer would then hold
    /// more than [`MAX_BYTES`] — refused as [`VfsError::Full`] with the layer
    /// left exactly as it was, a deletion included.
    pub(super) fn set_capped(
        upper: &mut Upper,
        norm: String,
        chain: Chain,
    ) -> Result<(), VfsError> {
        let old = upper.chains.get(&norm).map_or(0, Chain::bytes);
        if upper.bytes() - old + chain.bytes() > MAX_BYTES {
            return Err(VfsError::Full);
        }
        upper.chains.insert(norm, chain);
        Ok(())
    }
}
