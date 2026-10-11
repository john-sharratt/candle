//! The generations fixed for what a member wrote while cut off
//! (`docs/replicated_substrate.md` §6.7).
//!
//! A member with no admitted socket writes its records **detached**, under the
//! provisional generation [`GEN_DETACHED`], which sorts above every number.
//! When it rejoins, the group decides a generation `k` for it, and every record
//! it wrote while cut off takes `k` — one `oseq` range of its own origin per
//! rejoin. The records themselves are not rewritten: the assignment is kept
//! here, as one [`RebaseEntry`] per rejoin, and applied wherever the member
//! reads its own records back — at open, before a record reaches the
//! substrate, and when it sends one — so every record carries the generation
//! the group fixed for it, on this member and every other.
//!
//! Persisted whole as the `Rebase` record (tag 30): local housekeeping, never
//! replicated, with no identity, rewritten each time an entry is added and
//! carried by every rewrite.

use super::identity::{RecordIdentity, GEN_DETACHED};
use super::le_reader::LeReader;
use super::Result;

/// The generation fixed for `origin`'s records `from..=through`, which it
/// wrote detached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RebaseEntry {
    pub origin: u64,
    pub from: u64,
    pub through: u64,
    pub gen: u64,
}

/// Every generation fixed so far. See the module documentation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RebaseMap {
    entries: Vec<RebaseEntry>,
}

/// One entry's size in the payload: four `u64`s.
const ENTRY_BYTES: usize = 32;

impl RebaseMap {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[RebaseEntry] {
        &self.entries
    }

    /// Record `entry`. Answers whether it was new.
    pub fn add(&mut self, entry: RebaseEntry) -> bool {
        if self.entries.contains(&entry) {
            return false;
        }
        self.entries.push(entry);
        true
    }

    /// Take in every entry of `other` — what reading another `Rebase` record
    /// does; the map only grows.
    pub fn merge(&mut self, other: &RebaseMap) {
        for &entry in &other.entries {
            self.add(entry);
        }
    }

    /// `id` with the generation fixed for it, when it is a detached record an
    /// entry covers; otherwise `id` itself.
    pub fn fix(&self, id: RecordIdentity) -> RecordIdentity {
        if id.gen != GEN_DETACHED {
            return id;
        }
        match self
            .entries
            .iter()
            .find(|e| e.origin == id.origin && (e.from..=e.through).contains(&id.oseq))
        {
            Some(e) => RecordIdentity { gen: e.gen, ..id },
            None => id,
        }
    }

    /// The payload: the entry count, then each entry's `origin`, `from`,
    /// `through` and `gen`, all little-endian.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.entries.len() * ENTRY_BYTES);
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for e in &self.entries {
            for v in [e.origin, e.from, e.through, e.gen] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    pub fn decode(buf: &[u8]) -> Result<RebaseMap> {
        let mut r = LeReader::new(buf, "Rebase");
        let n = r.u32()? as usize;
        let mut entries = Vec::with_capacity(n);
        for _ in 0..n {
            entries.push(RebaseEntry {
                origin: r.u64()?,
                from: r.u64()?,
                through: r.u64()?,
                gen: r.u64()?,
            });
        }
        r.finish()?;
        Ok(RebaseMap { entries })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detached(origin: u64, oseq: u64) -> RecordIdentity {
        RecordIdentity {
            origin,
            oseq,
            gen: GEN_DETACHED,
            epoch: 0,
            hlc: 77,
        }
    }

    fn map() -> RebaseMap {
        let mut m = RebaseMap::default();
        m.add(RebaseEntry {
            origin: 7,
            from: 10,
            through: 12,
            gen: 3,
        });
        m
    }

    /// The payload is its exact bytes, and decodes back.
    #[test]
    fn the_payload_is_exact_bytes() {
        let m = map();
        let bytes = m.encode();
        assert_eq!(
            bytes,
            vec![
                1, 0, 0, 0, //
                7, 0, 0, 0, 0, 0, 0, 0, //
                10, 0, 0, 0, 0, 0, 0, 0, //
                12, 0, 0, 0, 0, 0, 0, 0, //
                3, 0, 0, 0, 0, 0, 0, 0,
            ]
        );
        assert_eq!(RebaseMap::decode(&bytes).unwrap(), m);
        assert!(RebaseMap::decode(&bytes[..10]).is_err());
        assert_eq!(
            RebaseMap::decode(&RebaseMap::default().encode()).unwrap(),
            RebaseMap::default()
        );
    }

    /// A detached record in a covered range takes the fixed generation; one of
    /// another origin, outside the range, or not detached keeps its own.
    #[test]
    fn fix_applies_only_to_covered_detached_records() {
        let m = map();
        assert_eq!(m.fix(detached(7, 10)).gen, 3);
        assert_eq!(m.fix(detached(7, 12)).gen, 3);
        assert_eq!(
            m.fix(detached(7, 12)).hlc,
            77,
            "only the generation changes"
        );
        assert_eq!(m.fix(detached(7, 13)).gen, GEN_DETACHED);
        assert_eq!(m.fix(detached(8, 11)).gen, GEN_DETACHED);
        let joined = RecordIdentity {
            gen: 2,
            ..detached(7, 11)
        };
        assert_eq!(m.fix(joined), joined);
    }

    /// Adding an entry twice keeps one.
    #[test]
    fn an_entry_is_kept_once() {
        let mut m = map();
        assert!(!m.add(m.entries()[0]));
        m.merge(&map());
        assert_eq!(m.entries().len(), 1);
    }
}
