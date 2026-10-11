//! Whose store over the mind repository (`docs/replicated_substrate.md`
//! §10.3): an operation's, carried through its workflow's steps, or an
//! author's, holding what an admin has written and not yet published.

use candle_conversation::persistence::content_hash::hash_bytes;
use candle_conversation::projection::TimelineId;

/// The owner of one store.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Owner {
    /// Operation `op` of `world`.
    Operation { world: String, op: u64 },
    /// The account whose public id is this (`accounts::user_id_of`) — what
    /// its characters are owned by: one store per account, whatever it writes.
    Author(String),
}

impl Owner {
    pub fn operation(world: &str, op: u64) -> Owner {
        Owner::Operation {
            world: world.to_string(),
            op,
        }
    }

    pub fn author(account: &str) -> Owner {
        Owner::Author(account.to_string())
    }

    /// The timeline the store's events live on: a hash of what owns it, so
    /// every member names it the same way.
    pub fn timeline(&self) -> TimelineId {
        let key = match self {
            Owner::Operation { world, op } => {
                let mut key = b"operation\0".to_vec();
                key.extend_from_slice(world.as_bytes());
                key.push(0);
                key.extend_from_slice(&op.to_le_bytes());
                key
            }
            Owner::Author(account) => {
                let mut key = b"author\0".to_vec();
                key.extend_from_slice(account.as_bytes());
                key
            }
        };
        let h = hash_bytes(&key);
        TimelineId::from_raw(h.lo.max(1)).expect("a timeline id is non-zero")
    }

    /// The world whose running decides whether the store writes events: an
    /// operation's. An author's store belongs to no world, so it always does.
    pub fn world(&self) -> Option<&str> {
        match self {
            Owner::Operation { world, .. } => Some(world),
            Owner::Author(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The timeline is the hash of the owner's key bytes**, so every member
    /// names a store the same way. The operation's key is unchanged from when
    /// operations were the only owners: the stores already written keep their
    /// timelines.
    #[test]
    fn each_owner_hashes_its_own_key() {
        let mut op_key = b"operation\0w\0".to_vec();
        op_key.extend_from_slice(&7u64.to_le_bytes());
        assert_eq!(
            Owner::operation("w", 7).timeline(),
            TimelineId::from_raw(hash_bytes(&op_key).lo.max(1)).unwrap()
        );
        assert_eq!(
            Owner::author("ab12").timeline(),
            TimelineId::from_raw(hash_bytes(b"author\0ab12").lo.max(1)).unwrap()
        );
        assert_ne!(
            Owner::author("w").timeline(),
            Owner::operation("w", 0).timeline()
        );
    }

    #[test]
    fn only_an_operation_belongs_to_a_world() {
        assert_eq!(Owner::operation("w", 1).world(), Some("w"));
        assert_eq!(Owner::author("a").world(), None);
    }
}
