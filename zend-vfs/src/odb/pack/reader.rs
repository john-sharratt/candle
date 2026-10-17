//! An object read from a pack: the entry at an offset, its delta chain walked
//! back to a whole object and applied forward.
//!
//! An offset delta's base is in the same pack. A ref delta's base is found in
//! the same pack by its id, and only otherwise asked of `external` — which no
//! pack in the log needs, since every one is self-contained (§4.2), but a pack
//! being indexed from a thin bundle does.

use std::sync::Arc;

use super::base_cache::BaseCache;
use super::delta;
use super::entry::{read_header, EntryKind};
use super::idx::PackIndex;
use super::inflate::{inflate, inflate_prefix};
use super::{Object, ObjectType, RawId};
use crate::error::GitError;

/// The longest delta chain followed; git's own packs stay under 50 deep, so a
/// longer one is a cycle in a damaged pack.
const MAX_CHAIN: usize = 10_000;

/// Enough of a delta's front to hold its two sizes.
const DELTA_SIZES: usize = 20;

/// A whole object asked of what lies outside the pack, by id.
pub(crate) type External<'a> = &'a dyn Fn(&RawId) -> Result<Option<Arc<Object>>, GitError>;

/// An object's kind and size asked of what lies outside the pack, by id.
pub(crate) type ExternalHeader<'a> =
    &'a dyn Fn(&RawId) -> Result<Option<(ObjectType, u64)>, GitError>;

/// A pack as an object is read from it: its bytes, and where in it an object
/// of a given id starts — its index, or while it is being indexed the ids
/// resolved so far.
pub(crate) struct Pack<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) find: &'a dyn Fn(&RawId) -> Option<u64>,
}

/// Where `index` says the object `id` starts.
pub(crate) fn offset_in(index: &PackIndex, id: &RawId) -> Option<u64> {
    index.find(id).map(|i| index.offset(i))
}

fn missing_base(id: &RawId) -> GitError {
    GitError::malformed("pack", format!("a delta's base {id} is not held"))
}

impl Pack<'_> {
    /// The object whose entry starts at `offset`.
    pub(crate) fn read_at(
        &self,
        offset: u64,
        cache: &mut BaseCache,
        external: External<'_>,
    ) -> Result<Arc<Object>, GitError> {
        let mut deltas: Vec<(u64, Vec<u8>)> = Vec::new();
        let mut at = offset;
        let mut base = loop {
            if let Some(found) = cache.get(at) {
                break found;
            }
            if deltas.len() > MAX_CHAIN {
                return Err(GitError::malformed(
                    "pack",
                    "a delta chain that does not end",
                ));
            }
            let start = usize::try_from(at)
                .map_err(|_| GitError::malformed("pack", "an offset past the pack"))?;
            let header = read_header(self.bytes, start)?;
            let (data, _) = inflate(self.bytes, header.data, header.size)?;
            match header.kind {
                EntryKind::Whole(kind) => break Arc::new(Object { kind, data }),
                EntryKind::OfsDelta { base } => {
                    deltas.push((at, data));
                    at = base;
                }
                EntryKind::RefDelta { base } => {
                    deltas.push((at, data));
                    match (self.find)(&base) {
                        Some(found) => at = found,
                        None => break external(&base)?.ok_or_else(|| missing_base(&base))?,
                    }
                }
            }
        };
        // The base itself is worth keeping only when deltas were applied to it.
        if !deltas.is_empty() && at != offset {
            cache.put(at, Arc::clone(&base));
        }
        while let Some((entry, d)) = deltas.pop() {
            let data = delta::apply(&base.data, &d)?;
            base = Arc::new(Object {
                kind: base.kind,
                data,
            });
            // Intermediate results are the next read's likely bases; the
            // object asked for is the caller's to keep.
            if entry != offset {
                cache.put(entry, Arc::clone(&base));
            }
        }
        Ok(base)
    }

    /// The kind and size of the object whose entry starts at `offset`, without
    /// inflating more of it than a delta's sizes.
    pub(crate) fn header_at(
        &self,
        offset: u64,
        external: ExternalHeader<'_>,
    ) -> Result<(ObjectType, u64), GitError> {
        let start = usize::try_from(offset)
            .map_err(|_| GitError::malformed("pack", "an offset past the pack"))?;
        let header = read_header(self.bytes, start)?;
        let size = match header.kind {
            EntryKind::Whole(kind) => return Ok((kind, header.size)),
            EntryKind::OfsDelta { .. } | EntryKind::RefDelta { .. } => {
                let front = inflate_prefix(self.bytes, header.data, DELTA_SIZES)?;
                delta::sizes(&front)?.1
            }
        };
        let mut kind_at = header.kind;
        for _ in 0..MAX_CHAIN {
            match kind_at {
                EntryKind::Whole(kind) => return Ok((kind, size)),
                EntryKind::OfsDelta { base } => {
                    let start = usize::try_from(base)
                        .map_err(|_| GitError::malformed("pack", "an offset past the pack"))?;
                    kind_at = read_header(self.bytes, start)?.kind;
                }
                EntryKind::RefDelta { base } => match (self.find)(&base) {
                    Some(found) => {
                        let start = usize::try_from(found)
                            .map_err(|_| GitError::malformed("pack", "an offset past the pack"))?;
                        kind_at = read_header(self.bytes, start)?.kind;
                    }
                    None => {
                        let (kind, _) = external(&base)?.ok_or_else(|| missing_base(&base))?;
                        return Ok((kind, size));
                    }
                },
            }
        }
        Err(GitError::malformed(
            "pack",
            "a delta chain that does not end",
        ))
    }
}

#[cfg(test)]
mod tests;
