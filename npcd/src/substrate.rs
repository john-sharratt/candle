//! `GET /v1/substrate/storage` — the redo log on disk.
//!
//! The counterpart of the `Storage` block in `zend/src/api/substrate.rs`, and
//! the one part of the substrate view this daemon can answer truthfully today:
//! a segmented append-only log is a directory of files, and a directory of
//! files can be read without opening the substrate that wrote them.
//!
//! Everything else on that page — layer occupancy, the summary forest, what a
//! projection selected — lives inside `candle-conversation`, which means
//! `candle`, CUDA and a twenty-minute build. Those stay behind the engine. This
//! does not, because it is genuinely just `readdir` plus `len()`.
//!
//! # Absent is a normal answer
//!
//! No substrate has been written yet on a daemon that has never run an engine,
//! so the ordinary case is a directory that does not exist. That reports as
//! `open: false` with an empty segment list — not an error, and never a
//! fabricated segment.

use std::path::{Path, PathBuf};

use serde::Serialize;

/// Segment files are named `seg-<id>.log`; the highest id is the one being
/// appended to. Matching the name zend's persistence layer writes.
const PREFIX: &str = "seg-";
const SUFFIX: &str = ".log";

/// A cap on how many segment entries are listed. A substrate that has been
/// running for months has more segments than any page can draw, and the totals
/// below stay correct regardless — see [`Storage::listed`].
const MAX_LISTED: usize = 512;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Segment {
    pub id: u64,
    /// The highest-numbered segment: the one currently being appended to.
    pub active: bool,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Storage {
    /// Whether a substrate directory exists at all. `false` is the ordinary
    /// state for a daemon that has not run an engine, not a failure.
    pub open: bool,
    /// Where this daemon would keep it, whether or not it is there. Shown so an
    /// operator can tell "looked in the wrong place" from "nothing written yet".
    pub path: String,
    /// Segments, ascending by id; the last is active. Truncated to the newest
    /// [`MAX_LISTED`] when there are more.
    pub segments: Vec<Segment>,
    /// How many segments exist, which is not `segments.len()` once truncated.
    pub segment_count: usize,
    /// Whether the list above is the whole set.
    pub listed: bool,
    /// Sum over *every* segment, including any not listed.
    pub total_bytes: u64,
    /// Live KV chunk records indexed in RAM, and the reclaimable fraction.
    /// Both require the substrate's in-memory index, so both are absent until
    /// an engine holds one open — distinct from a measured zero.
    pub live_chunks: Option<u64>,
    pub dead_ratio: Option<f32>,
}

/// Where the substrate lives, resolved once from the daemon's data directory.
pub struct SubstrateDir(PathBuf);

impl SubstrateDir {
    pub fn new(data_dir: &Path) -> Self {
        Self(data_dir.join(".substrate"))
    }

    /// Read the segment files.
    ///
    /// Per request rather than cached: a running engine appends continuously,
    /// and a cached answer on a page about storage growth would be the one
    /// number guaranteed to be stale. A `readdir` over a few hundred entries is
    /// cheaper than the JSON it produces.
    pub fn read(&self) -> Storage {
        let path = self.0.display().to_string();
        let Ok(entries) = std::fs::read_dir(&self.0) else {
            return Storage {
                open: false,
                path,
                segments: Vec::new(),
                segment_count: 0,
                listed: true,
                total_bytes: 0,
                live_chunks: None,
                dead_ratio: None,
            };
        };

        let mut segs: Vec<(u64, u64)> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let id = segment_id(&name)?;
                // A directory named like a segment is not a segment; only a
                // regular file has a length worth summing.
                let meta = e.metadata().ok()?;
                meta.is_file().then_some((id, meta.len()))
            })
            .collect();
        segs.sort_unstable();

        let total_bytes = segs.iter().map(|(_, b)| b).sum();
        let segment_count = segs.len();
        let listed = segment_count <= MAX_LISTED;
        let newest = segs.last().map(|(id, _)| *id);

        // Truncate from the FRONT: the recent segments are the ones anybody is
        // looking at, and the active one must always be in the list.
        if !listed {
            segs.drain(..segment_count - MAX_LISTED);
        }

        Storage {
            open: true,
            path,
            segments: segs
                .into_iter()
                .map(|(id, bytes)| Segment {
                    id,
                    active: Some(id) == newest,
                    bytes,
                })
                .collect(),
            segment_count,
            listed,
            total_bytes,
            live_chunks: None,
            dead_ratio: None,
        }
    }
}

/// `seg-000017.log` → `17`. Anything else is not a segment.
fn segment_id(name: &str) -> Option<u64> {
    name.strip_prefix(PREFIX)?
        .strip_suffix(SUFFIX)?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    /// A unique directory per test. A timestamp is not unique enough on
    /// Windows, whose clock granularity is ~15ms — two tests in the same tick
    /// collided and one deleted the other's fixture.
    fn tmp() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-substrate-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seg(dir: &Path, name: &str, bytes: usize) {
        fs::write(dir.join(name), vec![0u8; bytes]).unwrap();
    }

    /// The ordinary case on a daemon that has never run an engine.
    #[test]
    fn a_substrate_that_was_never_written_reports_closed_not_broken() {
        let root = tmp();
        let s = SubstrateDir::new(&root).read();
        assert!(!s.open);
        assert!(s.segments.is_empty());
        assert_eq!(s.total_bytes, 0);
        assert_eq!(s.segment_count, 0);
        // The path is reported anyway, so "looked in the wrong place" is
        // distinguishable from "nothing written yet".
        assert!(s.path.ends_with(".substrate"));
    }

    #[test]
    fn segments_are_ordered_and_the_newest_is_the_active_one() {
        let root = tmp();
        let sub = root.join(".substrate");
        fs::create_dir_all(&sub).unwrap();
        seg(&sub, "seg-000002.log", 200);
        seg(&sub, "seg-000010.log", 50);
        seg(&sub, "seg-000001.log", 100);

        let s = SubstrateDir::new(&root).read();
        assert!(s.open);
        assert_eq!(
            s.segments.iter().map(|x| x.id).collect::<Vec<_>>(),
            vec![1, 2, 10],
            "ids sort numerically, not as the zero-padded strings they are named with"
        );
        assert_eq!(s.total_bytes, 350);
        assert_eq!(s.segments.iter().filter(|x| x.active).count(), 1);
        assert!(s.segments.last().unwrap().active);
    }

    /// The index lives in the engine's memory, so neither of these can be known
    /// from the filesystem. Absent, never a plausible zero.
    #[test]
    fn the_in_memory_figures_stay_absent() {
        let root = tmp();
        fs::create_dir_all(root.join(".substrate")).unwrap();
        let s = SubstrateDir::new(&root).read();
        assert!(s.live_chunks.is_none());
        assert!(s.dead_ratio.is_none());
        let j = serde_json::to_value(&s).unwrap();
        assert_eq!(j["live_chunks"], serde_json::Value::Null);
        assert_eq!(j["dead_ratio"], serde_json::Value::Null);
    }

    #[test]
    fn files_that_are_not_segments_are_ignored() {
        let root = tmp();
        let sub = root.join(".substrate");
        fs::create_dir_all(&sub).unwrap();
        seg(&sub, "seg-000001.log", 10);
        seg(&sub, "substrate.log", 999);
        seg(&sub, "seg-abc.log", 999);
        seg(&sub, "seg-000002.txt", 999);
        seg(&sub, "notes.md", 999);
        // A directory named exactly like a segment must not be counted either.
        fs::create_dir_all(sub.join("seg-000003.log")).unwrap();

        let s = SubstrateDir::new(&root).read();
        assert_eq!(s.segments.len(), 1);
        assert_eq!(
            s.total_bytes, 10,
            "only the real segment's bytes are summed"
        );
    }

    /// A long-running substrate has more segments than a page can draw. The
    /// list truncates but the totals must not, or the page understates the
    /// footprint — the single number it exists to report.
    #[test]
    fn a_huge_substrate_truncates_the_list_but_not_the_totals() {
        let root = tmp();
        let sub = root.join(".substrate");
        fs::create_dir_all(&sub).unwrap();
        let n = MAX_LISTED + 7;
        for i in 0..n {
            seg(&sub, &format!("seg-{i:06}.log"), 3);
        }

        let s = SubstrateDir::new(&root).read();
        assert_eq!(s.segment_count, n);
        assert_eq!(s.total_bytes, (n * 3) as u64);
        assert!(!s.listed);
        assert_eq!(s.segments.len(), MAX_LISTED);
        // The newest survives truncation — dropping the active segment would
        // hide the only one currently changing.
        assert!(s.segments.last().unwrap().active);
        assert_eq!(s.segments.last().unwrap().id, (n - 1) as u64);
    }

    #[test]
    fn segment_ids_parse_only_in_the_exact_form() {
        assert_eq!(segment_id("seg-000017.log"), Some(17));
        assert_eq!(segment_id("seg-0.log"), Some(0));
        assert_eq!(segment_id("seg-.log"), None);
        assert_eq!(segment_id("seg-12.log.bak"), None);
        assert_eq!(segment_id("xseg-12.log"), None);
        assert_eq!(segment_id("seg--1.log"), None);
    }
}
