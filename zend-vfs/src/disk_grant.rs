//! The proof a checkout run is built from.

/// Proof that the caller was granted the right to change files on disk.
///
/// A checkout run ([`materialize`](crate::checkout::materialize())) and the
/// sandbox that makes one take it, so the right is decided once, where the
/// capability is checked, and carried from there rather than assumed.
///
/// [`DiskWriteGrant::issue`] is public because the check that decides it lives
/// in the tool layer, outside this crate: `zend_tools::Grants::disk_write` is
/// its one caller, and the tool layer's source scan holds every tool to that.
#[derive(Debug)]
pub struct DiskWriteGrant {
    _private: (),
}

impl DiskWriteGrant {
    /// A grant, for the capability check that has just decided the caller
    /// holds the right. Nothing else calls this.
    pub fn issue() -> Self {
        Self { _private: () }
    }
}
