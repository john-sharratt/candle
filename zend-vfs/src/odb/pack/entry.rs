//! A pack entry's header: what the entry is, how large it inflates to, and —
//! for a delta — where its base is.
//!
//! The first byte holds a continuation bit, a three-bit type and the size's
//! low four bits; each following byte, while the continuation bit is set,
//! seven more bits of size. An offset delta then names its base by a distance
//! back from the entry's own start, in git's offset encoding; a ref delta, by
//! the base's 20-byte id.

use super::{ObjectType, RawId, HASH};
use crate::error::GitError;

/// The type number of an offset delta.
const OFS_DELTA: u8 = 6;
/// The type number of a ref delta.
const REF_DELTA: u8 = 7;

/// What an entry holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntryKind {
    /// A whole object.
    Whole(ObjectType),
    /// A delta against the entry starting at this offset in the same pack.
    OfsDelta { base: u64 },
    /// A delta against the object with this id.
    RefDelta { base: RawId },
}

/// An entry's header, read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EntryHeader {
    pub(crate) kind: EntryKind,
    /// The inflated size: the object's, or the delta's.
    pub(crate) size: u64,
    /// Where the zlib stream starts.
    pub(crate) data: usize,
}

fn cut_short() -> GitError {
    GitError::malformed("pack", "an entry header cut short")
}

/// The header of the entry starting at `offset` in `pack`.
pub(crate) fn read_header(pack: &[u8], offset: usize) -> Result<EntryHeader, GitError> {
    let mut at = offset;
    let mut byte = *pack.get(at).ok_or_else(cut_short)?;
    at += 1;
    let number = (byte >> 4) & 0x7;
    let mut size = u64::from(byte & 0x0f);
    let mut shift = 4;
    while byte & 0x80 != 0 {
        byte = *pack.get(at).ok_or_else(cut_short)?;
        at += 1;
        if shift > 57 {
            return Err(GitError::malformed("pack", "an entry size over 64 bits"));
        }
        size |= u64::from(byte & 0x7f) << shift;
        shift += 7;
    }
    let kind = match number {
        OFS_DELTA => {
            let mut byte = *pack.get(at).ok_or_else(cut_short)?;
            at += 1;
            let mut distance = u64::from(byte & 0x7f);
            while byte & 0x80 != 0 {
                byte = *pack.get(at).ok_or_else(cut_short)?;
                at += 1;
                distance = distance
                    .checked_add(1)
                    .and_then(|d| d.checked_mul(128))
                    .ok_or_else(|| GitError::malformed("pack", "a delta base offset overflows"))?
                    | u64::from(byte & 0x7f);
            }
            let base = (offset as u64)
                .checked_sub(distance)
                .filter(|_| distance > 0)
                .ok_or_else(|| GitError::malformed("pack", "a delta base before the pack"))?;
            EntryKind::OfsDelta { base }
        }
        REF_DELTA => {
            let base = RawId::from_slice(pack.get(at..).ok_or_else(cut_short)?)?;
            at += HASH;
            EntryKind::RefDelta { base }
        }
        n => EntryKind::Whole(
            ObjectType::from_pack(n)
                .ok_or_else(|| GitError::malformed("pack", format!("entry type {n}")))?,
        ),
    };
    Ok(EntryHeader {
        kind,
        size,
        data: at,
    })
}

/// The header of a whole object of `kind` and `size`, as an entry begins.
pub(crate) fn whole_header(kind: ObjectType, size: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    let mut byte = (kind.pack_number() << 4) | (size & 0x0f) as u8;
    let mut rest = size >> 4;
    while rest != 0 {
        out.push(byte | 0x80);
        byte = (rest & 0x7f) as u8;
        rest >>= 7;
    }
    out.push(byte);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A small blob's header is one byte**: type 3, size 6 → `0x36`.
    #[test]
    fn a_small_whole_header_is_one_byte() {
        assert_eq!(whole_header(ObjectType::Blob, 6), vec![0x36]);
        let h = read_header(&[0x36], 0).unwrap();
        assert_eq!(
            h,
            EntryHeader {
                kind: EntryKind::Whole(ObjectType::Blob),
                size: 6,
                data: 1
            }
        );
    }

    /// **A larger size continues into following bytes**, seven bits each:
    /// 1000 = 0b1111101000 → low four `1000`, then `111110` → `0xb8 0x3e`
    /// for a commit (type 1).
    #[test]
    fn a_larger_size_continues_across_bytes() {
        let bytes = whole_header(ObjectType::Commit, 1000);
        assert_eq!(bytes, vec![0x98, 0x3e]);
        assert_eq!(read_header(&bytes, 0).unwrap().size, 1000);
        for size in [0, 15, 16, 127, 128, 1 << 20, (1 << 35) + 7] {
            let bytes = whole_header(ObjectType::Tree, size);
            let h = read_header(&bytes, 0).unwrap();
            assert_eq!((h.size, h.data), (size, bytes.len()), "{size}");
        }
    }

    /// **An offset delta's base distance uses git's encoding**: `0x91 0x00`
    /// is ((0x11 + 1) << 7) | 0 = 2304 back.
    #[test]
    fn an_offset_delta_names_its_base_by_distance() {
        // Type 6, size 3: 0x63; distance bytes 0x91 0x00.
        let mut pack = vec![0u8; 3000];
        pack.extend_from_slice(&[0x63, 0x91, 0x00]);
        let h = read_header(&pack, 3000).unwrap();
        assert_eq!(h.kind, EntryKind::OfsDelta { base: 3000 - 2304 });
        assert_eq!((h.size, h.data), (3, 3003));
        // A distance reaching before the pack is refused.
        assert!(read_header(&[0x63, 0x05], 0).is_err());
    }

    #[test]
    fn a_ref_delta_names_its_base_by_id() {
        let base = RawId([7; 20]);
        let mut pack = vec![0x75];
        pack.extend_from_slice(&base.0);
        let h = read_header(&pack, 0).unwrap();
        assert_eq!(h.kind, EntryKind::RefDelta { base });
        assert_eq!((h.size, h.data), (5, 21));
    }

    #[test]
    fn a_header_cut_short_or_of_no_type_is_refused() {
        assert!(
            read_header(&[0xb6], 0).is_err(),
            "continuation with no next byte"
        );
        assert!(read_header(&[0x05], 0).is_err(), "type 0");
        assert!(read_header(&[0x55], 0).is_err(), "type 5");
        assert!(read_header(&[0x75, 1, 2], 0).is_err(), "an id cut short");
    }
}
