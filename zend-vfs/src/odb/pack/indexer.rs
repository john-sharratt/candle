//! A pack's bytes indexed in this process, as `git index-pack` indexes them
//! (`docs/substrate_git.md` §4.2).
//!
//! 1. Every entry is walked in order: its header read, its stream inflated to
//!    find where the next begins, its CRC-32 taken over its raw bytes, and a
//!    whole object hashed for its id.
//! 2. Every delta is resolved to its object — an offset delta through its base
//!    in the pack, a ref delta through the id it names, found among the ids
//!    resolved so far or else asked of `external` — and hashed. Rounds repeat
//!    while any resolves, since a ref delta's base may itself be a delta later
//!    in the pack.
//! 3. A base that came from `external` — a thin pack's — is appended to the
//!    pack whole, the header's count and the trailer rewritten, so the pack
//!    stands alone.
//! 4. The index is written from every id, CRC and offset.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::sync::Arc;

use flate2::write::ZlibEncoder;
use flate2::Compression;
use sha1::{Digest, Sha1};

use super::base_cache::BaseCache;
use super::entry::{read_header, whole_header, EntryKind};
use super::idx::{self, IndexEntry};
use super::inflate::inflate;
use super::reader::{External, Pack};
use super::{Object, RawId, HASH, PACK_HEADER, PACK_MAGIC};
use crate::error::GitError;

/// What the bases of deltas being resolved may hold in memory.
const RESOLVE_BUDGET: usize = 256 * 1024 * 1024;

/// A pack, indexed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Indexed {
    /// The pack: the bytes given, or those with a thin pack's bases appended.
    pub(crate) pack: Vec<u8>,
    /// Its `.idx`, version 2.
    pub(crate) index: Vec<u8>,
    /// Its trailing checksum, which names it.
    pub(crate) checksum: RawId,
    /// How many objects it holds.
    pub(crate) objects: u32,
    /// How many of those were bases appended to make a thin pack whole.
    pub(crate) appended: u32,
}

fn bad(detail: impl Into<String>) -> GitError {
    GitError::malformed("pack", detail.into())
}

/// One entry, as the walk finds it.
struct Walked {
    offset: u64,
    crc: u32,
    kind: EntryKind,
    id: Option<RawId>,
}

/// The pack and the version and object count its header says, its trailer
/// checked.
fn check(pack: &[u8]) -> Result<u32, GitError> {
    if pack.len() < PACK_HEADER + HASH || &pack[..4] != PACK_MAGIC {
        return Err(bad("not a pack"));
    }
    let version = u32::from_be_bytes(pack[4..8].try_into().expect("four bytes"));
    if version != 2 && version != 3 {
        return Err(bad(format!("pack version {version}")));
    }
    let body = pack.len() - HASH;
    if Sha1::digest(&pack[..body])[..] != pack[body..] {
        return Err(bad("its trailing checksum does not match"));
    }
    Ok(u32::from_be_bytes(
        pack[8..12].try_into().expect("four bytes"),
    ))
}

/// Index `pack`, any ref delta's base it does not hold asked of `external`
/// and appended.
pub(crate) fn index_pack(mut pack: Vec<u8>, external: External<'_>) -> Result<Indexed, GitError> {
    let count = check(&pack)?;
    let end = pack.len() - HASH;

    // 1. The walk.
    let mut walked: Vec<Walked> = Vec::with_capacity(count as usize);
    let mut at = PACK_HEADER;
    for _ in 0..count {
        if at >= end {
            return Err(bad("fewer entries than its header says"));
        }
        let header = read_header(&pack, at)?;
        let (data, used) = inflate(&pack, header.data, header.size)?;
        let next = header.data + used;
        if next > end {
            return Err(bad("an entry runs into the trailer"));
        }
        let id = match header.kind {
            EntryKind::Whole(kind) => Some(RawId::of(kind, &data)),
            EntryKind::OfsDelta { .. } | EntryKind::RefDelta { .. } => None,
        };
        walked.push(Walked {
            offset: at as u64,
            crc: crc32fast::hash(&pack[at..next]),
            kind: header.kind,
            id,
        });
        at = next;
    }
    if at != end {
        return Err(bad("bytes after its last entry"));
    }

    // 2. The deltas resolved.
    let mut known: HashMap<RawId, u64> = walked
        .iter()
        .filter_map(|w| w.id.map(|id| (id, w.offset)))
        .collect();
    let thin: RefCell<BTreeMap<RawId, Arc<Object>>> = RefCell::new(BTreeMap::new());
    let mut cache = BaseCache::new(RESOLVE_BUDGET);
    let mut pending: Vec<usize> = (0..walked.len())
        .filter(|&i| walked[i].id.is_none())
        .collect();
    // A round in which nothing resolved without asking `external` lets the
    // next one ask it.
    let mut ask_outside = false;
    while !pending.is_empty() {
        let mut left = Vec::new();
        let mut resolved = 0usize;
        for i in pending {
            // Whether the read missed a base: deferred to a later round, where
            // any other failure is the pack's and fails the indexing.
            let missed = Cell::new(false);
            let outside = |id: &RawId| -> Result<Option<Arc<Object>>, GitError> {
                let found = match ask_outside {
                    true => external(id)?,
                    false => None,
                };
                match &found {
                    Some(object) => {
                        thin.borrow_mut().insert(*id, Arc::clone(object));
                    }
                    None => missed.set(true),
                }
                Ok(found)
            };
            let find = |id: &RawId| known.get(id).copied();
            let reader = Pack {
                bytes: &pack,
                find: &find,
            };
            match reader.read_at(walked[i].offset, &mut cache, &outside) {
                Ok(object) => {
                    let id = object.id();
                    walked[i].id = Some(id);
                    known.insert(id, walked[i].offset);
                    resolved += 1;
                }
                Err(_) if missed.get() => left.push(i),
                Err(e) => return Err(e),
            }
        }
        if left.is_empty() {
            break;
        }
        if resolved == 0 {
            if ask_outside {
                let base = match walked[left[0]].kind {
                    EntryKind::RefDelta { base } => base.to_string(),
                    _ => "an offset in the pack".to_string(),
                };
                return Err(bad(format!(
                    "{} deltas whose bases are held nowhere, the first against {base}",
                    left.len()
                )));
            }
            ask_outside = true;
        }
        pending = left;
    }

    // 3. A thin pack's bases appended.
    let thin = thin.into_inner();
    if !thin.is_empty() {
        pack.truncate(end);
        for (id, object) in &thin {
            if known.contains_key(id) {
                continue;
            }
            let offset = pack.len() as u64;
            let start = pack.len();
            pack.extend_from_slice(&whole_header(object.kind, object.data.len() as u64));
            let mut z = ZlibEncoder::new(Vec::new(), Compression::default());
            z.write_all(&object.data).map_err(GitError::Io)?;
            pack.extend_from_slice(&z.finish().map_err(GitError::Io)?);
            walked.push(Walked {
                offset,
                crc: crc32fast::hash(&pack[start..]),
                kind: EntryKind::Whole(object.kind),
                id: Some(*id),
            });
            known.insert(*id, offset);
        }
        let total = walked.len() as u32;
        pack[8..12].copy_from_slice(&total.to_be_bytes());
        let digest = Sha1::digest(&pack);
        pack.extend_from_slice(&digest);
    }

    // 4. The index.
    let checksum = RawId::from_slice(&pack[pack.len() - HASH..])?;
    let entries: Vec<IndexEntry> = walked
        .iter()
        .map(|w| IndexEntry {
            id: w.id.expect("every entry resolved above"),
            crc: w.crc,
            offset: w.offset,
        })
        .collect();
    let objects = entries.len() as u32;
    Ok(Indexed {
        index: idx::write(entries, &checksum),
        pack,
        checksum,
        objects,
        appended: objects - count,
    })
}

#[cfg(test)]
mod tests;
