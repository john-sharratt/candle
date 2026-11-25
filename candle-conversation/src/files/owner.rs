//! Whose the knowledge base's members are in the substrate's owned partition.
//!
//! A member is no conversation's: it is shared by every conversation that
//! places it. Its section id still comes from the owned partition, which
//! records an owner for every id, so the members are registered under one
//! owner that no conversation is — a timeline derived from a fixed name, never
//! opened, never tombstoned.

use crate::persistence::content_hash::hash_bytes;
use crate::projection::TimelineId;

/// The name the members' owner is derived from.
const OWNER_NAME: &str = "knowledge base members";

/// The owner every member's section is registered under.
pub fn files_owner() -> TimelineId {
    let bytes = hash_bytes(OWNER_NAME.as_bytes()).to_bytes();
    let mut first = [0u8; 8];
    first.copy_from_slice(&bytes[..8]);
    TimelineId::from_raw(u64::from_le_bytes(first).max(1))
        .expect("an owner derived with max(1) is never the reserved 0")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The owner is one fixed timeline, the same in every process.
    #[test]
    fn the_members_owner_is_one_fixed_timeline() {
        assert_eq!(files_owner(), files_owner());
        let bytes = hash_bytes(b"knowledge base members").to_bytes();
        assert_eq!(
            files_owner().raw(),
            u64::from_le_bytes(bytes[..8].try_into().unwrap()).max(1)
        );
    }
}
