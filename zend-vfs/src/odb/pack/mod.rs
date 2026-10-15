//! Git's pack format, read and written in this process
//! (`docs/substrate_git.md` §4.1, §4.2).
//!
//! A pack is `PACK`, a version, an object count, the entries, and a SHA-1 of
//! everything before it. An entry is a header — its kind and inflated size —
//! then a zlib stream: a whole object, or a delta against a base named by its
//! offset in the pack or by its id. Its index (`.idx`, version 2) lists every
//! object's id, in order, with the entry's CRC-32 and offset.
//!
//! | Module | Concern |
//! |---|---|
//! | [`entry`] | An entry's header: its kind, its size, its base |
//! | [`inflate`] | An entry's zlib stream |
//! | [`delta`] | Git's delta format |
//! | [`idx`] | The index, read and written |
//! | [`indexer`] | A pack's bytes indexed, a thin pack made whole |
//! | [`reader`] | An object read from a pack, its delta chain resolved |
//! | [`base_cache`] | The decoded bases a pack's chains run through |

pub(crate) mod base_cache;
pub(crate) mod delta;
pub(crate) mod entry;
#[cfg(test)]
pub(crate) mod fixture;
pub(crate) mod idx;
pub(crate) mod indexer;
pub(crate) mod inflate;
pub(crate) mod reader;

use std::fmt;

use sha1::{Digest, Sha1};

use crate::error::GitError;
use crate::types::Oid;

/// A pack's first four bytes.
pub(crate) const PACK_MAGIC: &[u8; 4] = b"PACK";

/// The size of a pack's header: magic, version, object count.
pub(crate) const PACK_HEADER: usize = 12;

/// The size of a SHA-1, a pack's trailer and every id in it.
pub(crate) const HASH: usize = 20;

/// The four kinds of object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ObjectType {
    Commit,
    Tree,
    Blob,
    Tag,
}

impl ObjectType {
    /// The kind a pack entry's type number names; `None` for a delta or an
    /// unused number.
    pub(crate) fn from_pack(number: u8) -> Option<Self> {
        match number {
            1 => Some(Self::Commit),
            2 => Some(Self::Tree),
            3 => Some(Self::Blob),
            4 => Some(Self::Tag),
            _ => None,
        }
    }

    /// The type number a pack entry carries for this kind.
    pub(crate) fn pack_number(self) -> u8 {
        match self {
            Self::Commit => 1,
            Self::Tree => 2,
            Self::Blob => 3,
            Self::Tag => 4,
        }
    }

    /// The name an object's header carries: what its id is hashed over.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Commit => "commit",
            Self::Tree => "tree",
            Self::Blob => "blob",
            Self::Tag => "tag",
        }
    }
}

/// A SHA-1 object id as its 20 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct RawId(pub(crate) [u8; HASH]);

impl RawId {
    /// The id of an object of `kind` holding `data`: the SHA-1 of
    /// `<kind> <length>\0` and the data, as git computes it.
    pub(crate) fn of(kind: ObjectType, data: &[u8]) -> Self {
        let mut h = Sha1::new();
        h.update(format!("{} {}\0", kind.name(), data.len()).as_bytes());
        h.update(data);
        Self(h.finalize().into())
    }

    /// The 20 bytes at the front of `bytes`.
    pub(crate) fn from_slice(bytes: &[u8]) -> Result<Self, GitError> {
        let raw: [u8; HASH] = bytes
            .get(..HASH)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| GitError::malformed("pack", "an object id cut short"))?;
        Ok(Self(raw))
    }

    /// The id as this layer names it.
    pub(crate) fn to_oid(self) -> Oid {
        Oid::parse(&self.to_string()).expect("40 lowercase hex characters")
    }

    /// The id `oid` names; a SHA-256 id is refused, since no pack this layer
    /// reads holds one.
    pub(crate) fn from_oid(oid: &Oid) -> Result<Self, GitError> {
        let hex = oid.as_str();
        if hex.len() != HASH * 2 {
            return Err(GitError::invalid(format!(
                "{oid} is not a SHA-1 id, and a repository read from the substrate holds only SHA-1"
            )));
        }
        let mut raw = [0u8; HASH];
        for (i, byte) in raw.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .map_err(|_| GitError::invalid(format!("{oid} is not hex")))?;
        }
        Ok(Self(raw))
    }
}

impl fmt::Display for RawId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for RawId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RawId({self})")
    }
}

/// An object's kind and bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Object {
    pub(crate) kind: ObjectType,
    pub(crate) data: Vec<u8>,
}

impl Object {
    /// This object's id.
    pub(crate) fn id(&self) -> RawId {
        RawId::of(self.kind, &self.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **An id is git's**: `hello\n` as a blob is the well-known `ce0136…`,
    /// and the empty tree `4b825d…`.
    #[test]
    fn an_object_id_is_the_one_git_computes() {
        assert_eq!(
            RawId::of(ObjectType::Blob, b"hello\n").to_string(),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
        assert_eq!(
            RawId::of(ObjectType::Tree, b"").to_string(),
            "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
        );
    }

    #[test]
    fn an_id_round_trips_through_the_layers_oid() {
        let id = RawId::of(ObjectType::Blob, b"x");
        assert_eq!(RawId::from_oid(&id.to_oid()).unwrap(), id);
        let sha256 = Oid::parse(&"a".repeat(64)).unwrap();
        assert!(RawId::from_oid(&sha256).is_err());
    }

    #[test]
    fn type_numbers_are_the_packs() {
        for kind in [
            ObjectType::Commit,
            ObjectType::Tree,
            ObjectType::Blob,
            ObjectType::Tag,
        ] {
            assert_eq!(ObjectType::from_pack(kind.pack_number()), Some(kind));
        }
        assert_eq!(ObjectType::from_pack(6), None, "an offset delta is no kind");
        assert_eq!(ObjectType::from_pack(7), None, "a ref delta is no kind");
    }
}
