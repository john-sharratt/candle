//! A pack's index, version 2, as `git index-pack` writes it.
//!
//! `\377tOc`, version 2, a fan-out of 256 counts — how many ids begin with a
//! byte at most each value — then every id in order, each entry's CRC-32, each
//! entry's offset (an offset of 2 GiB or more is an index into a table of
//! 64-bit offsets that follows), the pack's checksum, and a SHA-1 of the index
//! itself. Held as its bytes, which is about 28 per object.

use std::cmp::Ordering;
use std::fmt;
use std::ops::Range;

use sha1::{Digest, Sha1};

use super::{RawId, HASH};
use crate::error::GitError;

const MAGIC: [u8; 4] = [0xff, b't', b'O', b'c'];
const VERSION: u32 = 2;
const FANOUT_AT: usize = 8;
const IDS_AT: usize = FANOUT_AT + 256 * 4;
/// An offset with this bit set indexes the 64-bit table.
const LARGE: u32 = 0x8000_0000;

fn bad(detail: &str) -> GitError {
    GitError::malformed("pack index", detail.to_string())
}

fn be32(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

/// A pack's index, read.
#[derive(Clone)]
pub(crate) struct PackIndex {
    bytes: Vec<u8>,
    count: usize,
}

impl fmt::Debug for PackIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PackIndex({} objects)", self.count)
    }
}

impl PackIndex {
    /// Read `bytes` as an index, checking its shape and its own checksum.
    pub(crate) fn parse(bytes: Vec<u8>) -> Result<Self, GitError> {
        if bytes.len() < IDS_AT + 2 * HASH || bytes[..4] != MAGIC {
            return Err(bad("not a version 2 index"));
        }
        if be32(&bytes, 4) != VERSION {
            return Err(bad("not a version 2 index"));
        }
        let mut last = 0;
        for i in 0..256 {
            let n = be32(&bytes, FANOUT_AT + i * 4);
            if n < last {
                return Err(bad("a fan-out that goes down"));
            }
            last = n;
        }
        let count = last as usize;
        let offsets_at = IDS_AT + count * HASH + count * 4;
        let small_end = offsets_at + count * 4;
        if bytes.len() < small_end + 2 * HASH {
            return Err(bad("cut short"));
        }
        let large = (0..count)
            .filter(|i| be32(&bytes, offsets_at + i * 4) & LARGE != 0)
            .count();
        if bytes.len() != small_end + large * 8 + 2 * HASH {
            return Err(bad("not the length its counts say"));
        }
        let body = bytes.len() - HASH;
        if Sha1::digest(&bytes[..body])[..] != bytes[body..] {
            return Err(bad("its checksum does not match"));
        }
        Ok(Self { bytes, count })
    }

    /// How many objects the pack holds.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.count
    }

    /// The `i`th id, in order.
    pub(crate) fn id(&self, i: usize) -> RawId {
        let at = IDS_AT + i * HASH;
        RawId(self.bytes[at..at + HASH].try_into().expect("twenty bytes"))
    }

    /// The `i`th object's entry CRC-32.
    #[cfg(test)]
    pub(crate) fn crc(&self, i: usize) -> u32 {
        be32(&self.bytes, IDS_AT + self.count * HASH + i * 4)
    }

    /// Where the `i`th object's entry starts in the pack.
    pub(crate) fn offset(&self, i: usize) -> u64 {
        let offsets_at = IDS_AT + self.count * HASH + self.count * 4;
        let small = be32(&self.bytes, offsets_at + i * 4);
        if small & LARGE == 0 {
            return u64::from(small);
        }
        let at = offsets_at + self.count * 4 + (small & !LARGE) as usize * 8;
        u64::from_be_bytes(self.bytes[at..at + 8].try_into().expect("eight bytes"))
    }

    /// The pack's checksum, as the index records it.
    #[cfg(test)]
    pub(crate) fn pack_checksum(&self) -> RawId {
        let at = self.bytes.len() - 2 * HASH;
        RawId(self.bytes[at..at + HASH].try_into().expect("twenty bytes"))
    }

    /// The positions of the ids beginning with `first`.
    fn bucket(&self, first: u8) -> Range<usize> {
        let end = be32(&self.bytes, FANOUT_AT + usize::from(first) * 4) as usize;
        let start = match first {
            0 => 0,
            n => be32(&self.bytes, FANOUT_AT + (usize::from(n) - 1) * 4) as usize,
        };
        start..end
    }

    /// The position of `id`, when the pack holds it.
    pub(crate) fn find(&self, id: &RawId) -> Option<usize> {
        let range = self.bucket(id.0[0]);
        let (mut lo, mut hi) = (range.start, range.end);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.id(mid).cmp(id) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// The positions of the ids whose first `hex` hex digits are those of
    /// `prefix`, in order.
    pub(crate) fn prefixed(&self, prefix: &RawId, hex: usize) -> Range<usize> {
        let matches = |id: &RawId| starts_with(id, prefix, hex);
        let range = self.bucket(prefix.0[0]);
        if hex < 2 {
            // The first byte's top nibble only: every bucket it covers.
            let top = prefix.0[0] & 0xf0;
            let start = self.bucket(top).start;
            let end = self.bucket(top | 0x0f).end;
            return start..end;
        }
        let first = (range.start..range.end).find(|&i| matches(&self.id(i)));
        match first {
            None => range.start..range.start,
            Some(first) => {
                let end = (first..range.end)
                    .find(|&i| !matches(&self.id(i)))
                    .unwrap_or(range.end);
                first..end
            }
        }
    }

    /// Every id, in order.
    pub(crate) fn ids(&self) -> impl Iterator<Item = RawId> + '_ {
        (0..self.count).map(|i| self.id(i))
    }

    /// The index as its bytes.
    #[cfg(test)]
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Whether the first `hex` hex digits of `id` are `prefix`'s.
pub(crate) fn starts_with(id: &RawId, prefix: &RawId, hex: usize) -> bool {
    let whole = hex / 2;
    if id.0[..whole] != prefix.0[..whole] {
        return false;
    }
    hex.is_multiple_of(2) || (id.0[whole] & 0xf0) == (prefix.0[whole] & 0xf0)
}

/// One object of a pack, as its index records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexEntry {
    pub(crate) id: RawId,
    pub(crate) crc: u32,
    pub(crate) offset: u64,
}

/// The index of a pack holding `entries`, whose checksum is `pack_checksum`:
/// exactly the bytes `git index-pack` writes.
pub(crate) fn write(mut entries: Vec<IndexEntry>, pack_checksum: &RawId) -> Vec<u8> {
    entries.sort_by_key(|e| e.id);
    let mut out = Vec::with_capacity(IDS_AT + entries.len() * 28 + 2 * HASH);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_be_bytes());
    let mut fanout = [0u32; 256];
    for e in &entries {
        fanout[usize::from(e.id.0[0])] += 1;
    }
    let mut running = 0u32;
    for n in fanout {
        running += n;
        out.extend_from_slice(&running.to_be_bytes());
    }
    for e in &entries {
        out.extend_from_slice(&e.id.0);
    }
    for e in &entries {
        out.extend_from_slice(&e.crc.to_be_bytes());
    }
    let mut large: Vec<u64> = Vec::new();
    for e in &entries {
        let small = if e.offset < u64::from(LARGE) {
            e.offset as u32
        } else {
            large.push(e.offset);
            LARGE | (large.len() as u32 - 1)
        };
        out.extend_from_slice(&small.to_be_bytes());
    }
    for offset in large {
        out.extend_from_slice(&offset.to_be_bytes());
    }
    out.extend_from_slice(&pack_checksum.0);
    let digest = Sha1::digest(&out);
    out.extend_from_slice(&digest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(first: u8, rest: u8) -> RawId {
        let mut raw = [rest; HASH];
        raw[0] = first;
        RawId(raw)
    }

    fn sample() -> PackIndex {
        let entries = vec![
            IndexEntry {
                id: id(0xab, 2),
                crc: 7,
                offset: 12,
            },
            IndexEntry {
                id: id(0x01, 1),
                crc: 8,
                offset: 40,
            },
            IndexEntry {
                id: id(0xab, 1),
                crc: 9,
                offset: 3 << 31,
            },
        ];
        PackIndex::parse(write(entries, &RawId([5; HASH]))).unwrap()
    }

    /// **An index written reads back**: ids in order, each with its CRC and
    /// offset, a 2 GiB offset through the large table.
    #[test]
    fn an_index_round_trips() {
        let idx = sample();
        assert_eq!(idx.len(), 3);
        let ids: Vec<RawId> = idx.ids().collect();
        assert_eq!(ids, vec![id(0x01, 1), id(0xab, 1), id(0xab, 2)]);
        let i = idx.find(&id(0xab, 1)).unwrap();
        assert_eq!((idx.crc(i), idx.offset(i)), (9, 3 << 31));
        let j = idx.find(&id(0xab, 2)).unwrap();
        assert_eq!((idx.crc(j), idx.offset(j)), (7, 12));
        assert_eq!(idx.find(&id(0xab, 3)), None);
        assert_eq!(idx.find(&id(0x02, 1)), None);
        assert_eq!(idx.pack_checksum(), RawId([5; HASH]));
    }

    #[test]
    fn prefixes_find_every_match_and_only_those() {
        let idx = sample();
        assert_eq!(idx.prefixed(&id(0xab, 0), 2).len(), 2);
        assert_eq!(idx.prefixed(&id(0xab, 1), 4).len(), 1);
        assert_eq!(
            idx.prefixed(&id(0xab, 2), 3).len(),
            2,
            "ab01… and ab02… share ab0"
        );
        assert_eq!(idx.prefixed(&id(0xa0, 0), 1).len(), 2, "one nibble");
        assert_eq!(idx.prefixed(&id(0xcd, 0), 2).len(), 0);
    }

    /// **A damaged index is refused**: a wrong magic, a flipped byte the
    /// checksum catches, a cut.
    #[test]
    fn a_damaged_index_is_refused() {
        let good = sample().as_bytes().to_vec();
        let mut magic = good.clone();
        magic[0] = 0;
        assert!(PackIndex::parse(magic).is_err());
        let mut flipped = good.clone();
        flipped[IDS_AT] ^= 1;
        assert!(PackIndex::parse(flipped).is_err());
        assert!(PackIndex::parse(good[..good.len() - 1].to_vec()).is_err());
    }
}
