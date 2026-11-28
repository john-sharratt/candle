//! Cutting a stopped store's log back to the moment before one origin's
//! writes from a given wall-clock time on, so that every record written in a
//! bad run is gone and every record before it stands.
//!
//! The log is append-only and its segments are in append order, so "written
//! after the run began" is "after the run's first record" in file order. The
//! cut finds that record — the first one `origin` wrote at or after `from_ms`
//! by its identity's clock — and keeps everything before it:
//!
//! - every segment wholly after it is set aside under `<name>.cut`, which no
//!   open reads as a segment;
//! - the segment holding it is truncated at its offset. Its superblock hint then
//!   points past the end, so the next open forward-walks it (`recovery`), and as
//!   the highest-id segment left it is the active one appended to.
//!
//! A record re-emitted by relocation keeps its original identity, so a run's
//! relocations carry clocks from before it began. They come after the run's
//! first record in file order all the same, and the cut takes them with it.
//!
//! The store must be stopped: the cut works on the files directly.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};

use super::identity::wall_ms;
use super::log_file::{LogFile, SUPERBLOCK_SIZE};
use super::record::{CustomObjectPayload, RecordType};
use super::segment::SegmentId;
use super::segmented_log::scan_segment_files;
use super::walker;
use super::Result;

/// What a segment set aside by a cut is renamed with: `seg-<id>.log.cut`.
pub const CUT_SUFFIX: &str = ".cut";

/// A cut of the log, planned and not yet applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TailCut {
    /// The segment truncated, and the offset it keeps up to. `None` when the
    /// first record cut opens a segment, so no segment is truncated.
    pub truncate: Option<(SegmentId, PathBuf, u64)>,
    /// Every segment wholly after the cut, ascending.
    pub set_aside: Vec<(SegmentId, PathBuf)>,
    /// The records the cut removes, by record type.
    pub cut: BTreeMap<String, usize>,
    /// The latest clock reading, in wall-clock milliseconds, of a record
    /// `origin` wrote that the cut keeps — what confirms the cut lands between
    /// two runs rather than inside one.
    pub last_kept_ms: u64,
    /// The keys of the custom objects the cut removes that carry a clock from
    /// before the run: copies relocation re-emitted during it, of objects
    /// whose earlier copies may be gone with the segments it dropped.
    pub carried_objects: Vec<String>,
}

impl TailCut {
    /// How many records the cut removes.
    pub fn records_cut(&self) -> usize {
        self.cut.values().sum()
    }
}

/// Plan the cut of the store in `dir` at the first record `origin` wrote at or
/// after `from_ms`. `None` when `origin` wrote nothing that late.
///
/// Segments are walked newest first and the walk stops at the first one with
/// nothing to cut, so a cut near the end reads only the segments it touches.
pub fn plan(dir: &Path, origin: u64, from_ms: u64) -> Result<Option<TailCut>> {
    let segments = scan_segment_files(dir)?;
    let mut set_aside = Vec::new();
    let mut cut: BTreeMap<String, usize> = BTreeMap::new();
    let mut truncate = None;
    let mut last_kept_ms = 0u64;
    let mut carried_objects = Vec::new();
    for (id, path) in segments.into_iter().rev() {
        let mut log = LogFile::open_read_only(&path)?;
        let mut first: Option<u64> = None;
        let mut here: Vec<String> = Vec::new();
        let mut carried_here: Vec<String> = Vec::new();
        let mut kept_ms = 0u64;
        walker::walk_filtered(
            &mut log,
            id,
            SUPERBLOCK_SIZE,
            |rt| rt == RecordType::CustomObject,
            |e| {
                let h = &e.record.header;
                if first.is_none() && h.origin == origin && wall_ms(h.hlc) >= from_ms {
                    first = Some(e.offset);
                }
                if first.is_some() {
                    here.push(format!("{:?}", h.record_type));
                    if h.record_type == RecordType::CustomObject && wall_ms(h.hlc) < from_ms {
                        carried_here.push(
                            CustomObjectPayload::decode(&e.record.payload)
                                .map_or_else(|err| format!("<undecodable: {err}>"), |o| o.key),
                        );
                    }
                } else if h.origin == origin {
                    kept_ms = kept_ms.max(wall_ms(h.hlc));
                }
            },
        )?;
        for ty in here {
            *cut.entry(ty).or_insert(0) += 1;
        }
        carried_objects.extend(carried_here);
        match first {
            Some(SUPERBLOCK_SIZE) => set_aside.push((id, path)),
            Some(offset) => {
                truncate = Some((id, path, offset));
                last_kept_ms = kept_ms;
                break;
            }
            None => {
                last_kept_ms = kept_ms;
                break;
            }
        }
    }
    if truncate.is_none() && set_aside.is_empty() {
        return Ok(None);
    }
    set_aside.reverse();
    carried_objects.sort();
    Ok(Some(TailCut {
        truncate,
        set_aside,
        cut,
        last_kept_ms,
        carried_objects,
    }))
}

/// The name a segment set aside by a cut goes by.
pub fn set_aside_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(CUT_SUFFIX);
    PathBuf::from(name)
}

/// Apply `cut`: set the later segments aside, newest first, then truncate the
/// segment the cut falls in. Each step leaves a store a fresh [`plan`] cuts
/// the rest of, so an interrupted cut is finished by planning and applying it
/// again.
pub fn apply(cut: &TailCut) -> Result<()> {
    for (_, path) in cut.set_aside.iter().rev() {
        fs::rename(path, set_aside_path(path))?;
    }
    if let Some((_, path, offset)) = &cut.truncate {
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(*offset)?;
        file.sync_all()?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "tail_cut_tests.rs"]
mod tests;
