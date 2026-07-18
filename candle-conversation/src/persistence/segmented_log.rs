//! The segmented redo log — the on-disk file set under `.substrate/`.
//!
//! The redo log is a set of ~4 GB **segment** files, not one monolithic
//! file (`docs/segmented_substrate_log.md`). Every segment is a `seg-<id>.log`
//! file; the **active** append target is simply the **highest-id** one, and the
//! rest are immutable **sealed** segments. A record's in-RAM location is
//! `(SegmentId, offset)`, so a read routes to the file that physically holds it.
//!
//! ## No manifest, single namespace
//!
//! The segment set, order, and active/sealed split are all derived from the
//! directory listing (§4.1): the segments are the `seg-<id>.log` files sorted by
//! id, the active is the highest, and **id order = append order = recency
//! order** (relocation always appends into the active, so a key's live record is
//! its highest-id occurrence). There is no `.active` extension and no
//! rename-on-seal — a segment becomes sealed simply by a higher-id one existing,
//! which removes the two-active crash window entirely. (A legacy `seg-<id>.active`
//! from before this simplification is adopted — renamed to `.log` — on open.)
//! Recovery replays segments by ascending id (active last) into one shared
//! substrate; a later (higher-id) record overwrites an earlier one, landing on
//! the newest per key.
//!
//! ## Migration
//!
//! On open a bare legacy `substrate.log` is renamed to `seg-0000000001.log`
//! (sealed) and a fresh active is minted — O(1), no byte rewrite (§13). The
//! big legacy blob is then reclaimed by ordinary background maintenance.
//!
//! ## I/O model
//!
//! Each segment is exactly today's [`LogFile`] (buffered writes + small
//! reads) plus its [`DirectFile`] (`O_DIRECT` stripe reads). The active is
//! always open; sealed-segment read handles are cached in an LRU
//! [`SealedPool`] bounded to [`OPEN_SEALED_SEGMENTS`], so the open-handle
//! count stays bounded regardless of how many segments exist.

use std::fs;
use std::path::{Path, PathBuf};

use super::direct_io::DirectFile;
use super::header_index::IndexEntry;
use super::log_file::{read_record_at, LogFile, LogSource, Superblock, SUPERBLOCK_SIZE};
use super::manifest::Manifest;
use super::record::Record;
use super::recovery;
use super::segment::{SegmentId, FIRST_SEGMENT};
use super::walker::WalkEntry;
use super::Result;

/// Soft size the active segment grows to before the next commit seals it and
/// mints a fresh active. ~4 GB (§12). A single commit is never split across
/// segments, so the active can overshoot by up to one commit's records.
pub const SEGMENT_TARGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Max sealed segments kept open in the read pool at once. Steady-state
/// open-handle count = (1 active + `OPEN_SEALED_SEGMENTS`) × the per-segment
/// `DirectFile` handle count.
pub const OPEN_SEALED_SEGMENTS: usize = 8;

const SEGMENT_PREFIX: &str = "seg-";
const SEGMENT_EXT: &str = "log";
/// Pre-single-namespace active extension. A `seg-<id>.active` file is a legacy
/// active from before the active/sealed rename was retired; it is **adopted**
/// (renamed to `.log`) on open — see [`scan_segments`]. The active is now simply
/// the highest-id `.log` file.
const LEGACY_ACTIVE_EXT: &str = "active";
const LEGACY_LOG_NAME: &str = "substrate.log";

/// The on-disk name of a segment — `seg-<id>.log`. There is one naming scheme:
/// the active segment is just the highest-id `.log` file (no `.active`
/// extension), and a segment is sealed simply by a higher-id one existing. The
/// id is zero-padded so a lexical directory sort matches numeric id order.
fn segment_name(id: SegmentId) -> String {
    format!("{SEGMENT_PREFIX}{:010}.{SEGMENT_EXT}", id.raw())
}

/// Parse a `seg-<id>.log` name into its id. `None` for any other name
/// (including a legacy `seg-<id>.active` or the legacy `substrate.log`).
fn parse_segment(name: &str) -> Option<SegmentId> {
    let rest = name.strip_prefix(SEGMENT_PREFIX)?;
    let (num, ext) = rest.rsplit_once('.')?;
    if ext != SEGMENT_EXT {
        return None;
    }
    Some(SegmentId(num.parse().ok()?))
}

/// Parse a legacy `seg-<id>.active` name into its id — used only by the one-time
/// adopt in [`scan_segments`].
fn parse_legacy_active(name: &str) -> Option<SegmentId> {
    let rest = name.strip_prefix(SEGMENT_PREFIX)?;
    let (num, ext) = rest.rsplit_once('.')?;
    if ext != LEGACY_ACTIVE_EXT {
        return None;
    }
    Some(SegmentId(num.parse().ok()?))
}

/// LRU pool of open read handles on sealed segments. Each entry is a
/// [`LogFile`] (its own buffered handle + `DirectFile` handle set), so the
/// pool bounds the total open-handle count. The active segment is never
/// pooled — it is always open on [`SegmentedLog::active`].
struct SealedPool {
    /// `(id, reader)` in LRU order — least-recently-used first, most-recently
    /// used last. Length is bounded to `cap`.
    open: Vec<(SegmentId, LogFile)>,
    cap: usize,
}

impl SealedPool {
    fn new(cap: usize) -> SealedPool {
        SealedPool {
            open: Vec::new(),
            cap,
        }
    }

    /// An open read handle for sealed `id`, opening it (and evicting the LRU
    /// entry if at capacity) on a miss. The returned entry becomes MRU.
    fn get_or_open(&mut self, dir: &Path, id: SegmentId) -> Result<&mut LogFile> {
        if let Some(pos) = self.open.iter().position(|(sid, _)| *sid == id) {
            let entry = self.open.remove(pos);
            self.open.push(entry);
            return Ok(&mut self.open.last_mut().unwrap().1);
        }
        let log = LogFile::open(&dir.join(segment_name(id)))?;
        if self.open.len() >= self.cap {
            self.open.remove(0);
        }
        self.open.push((id, log));
        Ok(&mut self.open.last_mut().unwrap().1)
    }

    /// Drop any cached read handle for `id` — called when the segment is
    /// dropped by background maintenance so its file becomes unlinkable.
    fn forget(&mut self, id: SegmentId) {
        self.open.retain(|(sid, _)| *sid != id);
    }
}

/// The segmented redo log: the active segment plus the set of sealed
/// segments, addressed by [`SegmentId`].
pub struct SegmentedLog {
    dir: PathBuf,
    active_id: SegmentId,
    active: LogFile,
    /// Sealed segment ids present on disk, ascending. Excludes the active.
    sealed: Vec<SegmentId>,
    pool: SealedPool,
}

/// Everything [`SegmentedLog::open_with_sink`] hands back to the persistence
/// layer alongside the log itself.
pub struct OpenedSegments {
    pub segments: SegmentedLog,
    /// Combined singleton manifest — the highest-id occurrence of each
    /// singleton wins (id-order recency).
    pub manifest: Manifest,
    /// The **active** segment's `HeaderIndex` chain head — the live chain the
    /// next flush extends. Sealed segments' chains are frozen.
    pub last_index: Option<(u64, u64)>,
    /// The active segment's un-indexed tail digests — seed the writer's
    /// accumulator so the next flush covers them.
    pub tail_digests: Vec<IndexEntry>,
    /// Total records replayed across every segment (diagnostic).
    pub recovered_records: usize,
}

/// Overlay `from`'s present singletons onto `into` — used to fold each
/// segment's singletons in ascending id order so the highest-id occurrence
/// wins (last-writer-wins recency).
fn merge_singletons(into: &mut Manifest, from: &Manifest) {
    if from.model_spec.is_some() {
        into.model_spec = from.model_spec;
    }
    if from.template.is_some() {
        into.template = from.template;
    }
    if from.tokenizer.is_some() {
        into.tokenizer = from.tokenizer;
    }
}

/// Delete stray compaction scratch files — a `.compact` left by an interrupted
/// whole-store compaction, or an older `*.tmp`-suffixed scratch on a store
/// written before the rename. Neither is authoritative; recovery ignores them
/// and open removes them.
fn cleanup_scratch(dir: &Path) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name.ends_with(".compact") || name.ends_with(".tmp") {
            let _ = fs::remove_file(entry.path());
        }
    }
    Ok(())
}

/// Auto-split a legacy monolithic `substrate.log` into the segmented layout
/// (§13): seal it as `seg-0000000001.log` and mint a fresh empty active
/// (`seg-0000000002.log`) above it, so new records append to the small active
/// rather than the big legacy blob (which is then reclaimed by background
/// maintenance). O(1) — a rename + a create, no byte rewrite. If any segment
/// file already exists the store is already segmented; the legacy file (if any)
/// is left untouched.
fn migrate_legacy(dir: &Path) -> Result<()> {
    let legacy = dir.join(LEGACY_LOG_NAME);
    if !legacy.exists() {
        return Ok(());
    }
    let already_segmented = fs::read_dir(dir)?.filter_map(|e| e.ok()).any(|e| {
        e.file_name()
            .to_str()
            .is_some_and(|n| parse_segment(n).is_some() || parse_legacy_active(n).is_some())
    });
    if already_segmented {
        return Ok(());
    }
    let sealed = FIRST_SEGMENT;
    let active = sealed.next();
    fs::rename(&legacy, &dir.join(segment_name(sealed)))?;
    LogFile::create(&dir.join(segment_name(active)))?;
    tracing::info!(
        "migrated legacy {LEGACY_LOG_NAME} -> {} (sealed) + {} (active)",
        segment_name(sealed),
        segment_name(active)
    );
    Ok(())
}

/// Scan `.substrate/` for segment files. First **adopts** any legacy
/// `seg-<id>.active` into the single `.log` naming (a `.active` file predates
/// the active/sealed rename being retired — the active is now just the
/// highest-id `.log`). Returns every segment id, ascending; the caller takes
/// the highest as the active and the rest as sealed.
fn scan_segments(dir: &Path) -> Result<Vec<SegmentId>> {
    // One-time adopt of legacy `.active` files.
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some(id) = parse_legacy_active(name) {
            let target = dir.join(segment_name(id));
            if !target.exists() {
                fs::rename(entry.path(), &target)?;
                tracing::info!("adopted legacy active {name} -> {}", segment_name(id));
            }
        }
    }
    // Collect every segment id, ascending.
    let mut ids = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if let Some(id) = entry.file_name().to_str().and_then(parse_segment) {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

impl SegmentedLog {
    /// Open (creating if absent) the segment set in `dir`, recovering every
    /// segment's records through `sink` in ascending id order (active last).
    ///
    /// Runs migration and rotation-heal first, then walks each segment via
    /// its own `HeaderIndex` chain (full-walk fallback), truncating a torn
    /// tail in place. The active segment's chain state is returned as the
    /// live writer state; sealed segments' chains are frozen.
    pub fn open_with_sink<F>(dir: &Path, mut sink: F) -> Result<OpenedSegments>
    where
        F: FnMut(&WalkEntry),
    {
        fs::create_dir_all(dir)?;
        cleanup_scratch(dir)?;
        migrate_legacy(dir)?;
        // Every segment id, ascending. The highest is the active; the rest are
        // sealed (id-order = recency, so the active is always the newest).
        let mut ids = scan_segments(dir)?;
        let active_opt = ids.pop();
        let sealed = ids;

        let mut manifest = Manifest::new();
        let mut recovered_records = 0usize;

        for &id in &sealed {
            let mut log = LogFile::open(&dir.join(segment_name(id)))?;
            let hint = log.superblock().last_index;
            let rec = recovery::recover_with_sink(&mut log, id, hint, |e| {
                recovered_records += 1;
                sink(e);
            })?;
            if rec.torn {
                // A migrated legacy file (or a segment sealed across a crash)
                // may carry a torn tail — heal it once, in place.
                log.truncate_to(rec.tail_offset)?;
            }
            merge_singletons(&mut manifest, &rec.manifest);
            // The read handle is dropped; later reads reopen via the pool.
        }

        let (active_id, active, last_index, tail_digests) = match active_opt {
            Some(id) => {
                let mut log = LogFile::open(&dir.join(segment_name(id)))?;
                let hint = log.superblock().last_index;
                let rec = recovery::recover_with_sink(&mut log, id, hint, |e| {
                    recovered_records += 1;
                    sink(e);
                })?;
                if rec.torn {
                    log.truncate_to(rec.tail_offset)?;
                }
                log.set_write_offset(rec.tail_offset);
                merge_singletons(&mut manifest, &rec.manifest);
                (id, log, rec.last_index, rec.tail_digests)
            }
            None => {
                // Empty store: create the first segment (which is the active).
                let id = FIRST_SEGMENT;
                let log = LogFile::create(&dir.join(segment_name(id)))?;
                (id, log, None, Vec::new())
            }
        };

        let segments = SegmentedLog {
            dir: dir.to_path_buf(),
            active_id,
            active,
            sealed,
            pool: SealedPool::new(OPEN_SEALED_SEGMENTS),
        };
        Ok(OpenedSegments {
            segments,
            manifest,
            last_index,
            tail_digests,
            recovered_records,
        })
    }

    /// The active (append target) segment's id.
    pub fn active_id(&self) -> SegmentId {
        self.active_id
    }

    /// Sealed segment ids on disk, ascending. Excludes the active.
    pub fn sealed_ids(&self) -> &[SegmentId] {
        &self.sealed
    }

    /// Stage one already-encoded record into the active segment. Returns the
    /// `(segment, offset)` it will occupy once flushed.
    pub fn stage(&mut self, bytes: &[u8]) -> (SegmentId, u64) {
        let offset = self.active.stage(bytes);
        (self.active_id, offset)
    }

    /// Bytes staged into the active but not yet flushed.
    pub fn pending_len(&self) -> usize {
        self.active.pending_len()
    }

    /// Flush + fsync the active segment.
    pub fn commit(&mut self) -> Result<()> {
        self.active.commit()
    }

    /// Publish the active segment's newest `HeaderIndex` chain head.
    pub fn set_last_index(&mut self, last_index: (u64, u64)) -> Result<()> {
        self.active.set_last_index(last_index)
    }

    /// The active segment's superblock.
    pub fn superblock(&self) -> Superblock {
        self.active.superblock()
    }

    /// The active segment's durable logical end.
    pub fn write_offset(&self) -> u64 {
        self.active.write_offset()
    }

    /// Whether the active segment has reached the seal threshold.
    pub fn should_rotate(&self) -> bool {
        self.active.write_offset() >= SEGMENT_TARGET_BYTES
    }

    /// Seal the active segment and mint a fresh one at the next id. The caller
    /// must have completed the active's index chain and committed it (no
    /// un-flushed records) first.
    ///
    /// **No rename:** the old segment is sealed simply by a higher-id one
    /// existing (the active is always the highest id). We create the fresh
    /// active (`seg-<new_id>.log`) and close the old segment's write handle.
    /// A crash at any point leaves the old (durable) segment plus possibly an
    /// empty new one; the highest id is unambiguously the active on the next
    /// open, so there is no two-active state to heal.
    pub fn seal_and_rotate(&mut self) -> Result<()> {
        debug_assert_eq!(
            self.active.pending_len(),
            0,
            "seal_and_rotate with un-flushed records"
        );
        let old_id = self.active_id;
        let new_id = old_id.next();

        let new_active = LogFile::create(&self.dir.join(segment_name(new_id)))?;
        let old_active = std::mem::replace(&mut self.active, new_active);
        // Close the now-sealed segment's write handle; future reads reopen it
        // read-only via the pool.
        drop(old_active);
        self.sealed.push(old_id);
        self.active_id = new_id;
        Ok(())
    }

    /// Read one record, routing to the segment that physically holds it.
    pub fn read_record_at(
        &mut self,
        segment: SegmentId,
        offset: u64,
        record_size: u64,
    ) -> Result<Record> {
        if segment == self.active_id {
            read_record_at(&mut self.active, offset, record_size)
        } else {
            let src = self.pool.get_or_open(&self.dir, segment)?;
            read_record_at(src, offset, record_size)
        }
    }

    /// Read `dest.len()` bytes at `(segment, offset)` — the CPU stripe read
    /// (`read_stream_chunks_batched`). Routes to the owning segment.
    pub fn read_into(&mut self, segment: SegmentId, offset: u64, dest: &mut [u8]) -> Result<()> {
        if segment == self.active_id {
            self.active.read_into(offset, dest)
        } else {
            let src = self.pool.get_or_open(&self.dir, segment)?;
            src.read_into(offset, dest)
        }
    }

    /// The active segment's direct-I/O handle set — the cold-load reader pool
    /// for records that live in the active segment.
    pub fn active_direct_file(&self) -> &DirectFile {
        self.active.direct_file()
    }

    /// Open a **fresh** direct-I/O handle set on a sealed segment for the
    /// duration of one cold-load. Unlike the CPU read path (which borrows a
    /// pooled handle via `&mut self`), the GPU cold-load pipeline holds
    /// `&self` shared across its reader threads and needs several sealed
    /// segments' handles live at once, so it owns them directly rather than
    /// borrowing from the LRU pool.
    pub fn open_sealed_direct(&self, segment: SegmentId) -> Result<DirectFile> {
        Ok(DirectFile::open(&self.dir.join(segment_name(segment)))?)
    }

    /// Record bytes (excluding the superblock) in sealed `segment` — the
    /// denominator of that segment's dead-weight ratio for the maintenance
    /// triggers.
    pub fn sealed_record_bytes(&self, segment: SegmentId) -> Result<u64> {
        let len = fs::metadata(self.dir.join(segment_name(segment)))?.len();
        Ok(len.saturating_sub(SUPERBLOCK_SIZE))
    }

    /// Seconds since sealed `segment`'s file was last written — the settle
    /// timer for the compact trigger. Survives restarts (it is the file's
    /// mtime), so a segment sealed long ago is immediately compactable.
    pub fn sealed_age_secs(&self, segment: SegmentId, now: std::time::SystemTime) -> Result<u64> {
        let mtime = fs::metadata(self.dir.join(segment_name(segment)))?.modified()?;
        Ok(now.duration_since(mtime).unwrap_or_default().as_secs())
    }

    /// Drop sealed `segment` — close its pooled read handle and unlink the
    /// file. The caller must have already relocated its live records into the
    /// active (background maintenance), so nothing references it.
    pub fn drop_sealed(&mut self, segment: SegmentId) -> Result<()> {
        self.pool.forget(segment);
        fs::remove_file(self.dir.join(segment_name(segment)))?;
        self.sealed.retain(|&s| s != segment);
        Ok(())
    }

    /// The `.substrate/` directory this segment set lives in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Total live+dead record bytes across every segment (excluding the
    /// per-segment superblocks) — the denominator of the store-wide
    /// dead-weight ratio. Sealed sizes come from the file metadata; the
    /// active from its write offset + staged bytes.
    pub fn total_record_bytes(&self) -> Result<u64> {
        let mut total = self.active.write_offset().saturating_sub(SUPERBLOCK_SIZE)
            + self.active.pending_len() as u64;
        for &id in &self.sealed {
            let len = fs::metadata(self.dir.join(segment_name(id)))?.len();
            total += len.saturating_sub(SUPERBLOCK_SIZE);
        }
        Ok(total)
    }

    /// Replace the whole segment set with a freshly-written compacted segment.
    /// `new_active` is an open handle on `scratch` (a non-segment scratch file
    /// the compactor wrote the live records into).
    ///
    /// **Crash-safe ordering:** the scratch is renamed into place as a **fresh
    /// highest-id** active segment *first*, then the old segments are deleted.
    /// A crash between the two leaves the old segments plus the new one; the new
    /// (highest-id) wins by id-order on recovery — its compacted records are a
    /// superset of the live set — and the stale old segments are reclaimed by
    /// the next maintenance pass. So the compacted data is never the only copy
    /// under a name that recovery would discard. The `new_active` handle follows
    /// the rename (Rust opens with `FILE_SHARE_DELETE`).
    pub fn adopt_compacted(&mut self, new_active: LogFile, scratch: &Path) -> Result<()> {
        // A fresh id above every existing segment (the current active is the
        // highest), so the compacted segment wins by id-order over the old ones.
        let new_id = self.active_id.next();

        // Swap in the compacted active + close old handles so the old files are
        // deletable (Windows).
        let old_active = std::mem::replace(&mut self.active, new_active);
        drop(old_active);
        self.pool.open.clear();

        // 1. Rename the scratch into place as the fresh active FIRST — now the
        //    compacted data is a real, highest-id segment. A crash here leaves
        //    old + new; recovery keeps new, maintenance reclaims old.
        let keep = segment_name(new_id);
        fs::rename(scratch, &self.dir.join(&keep))?;
        self.active_id = new_id;

        // 2. Delete every OTHER segment file (the old ones). The just-renamed new
        //    active is skipped.
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name_str) = name.to_str() else {
                continue;
            };
            if name_str != keep
                && (parse_segment(name_str).is_some() || parse_legacy_active(name_str).is_some())
            {
                let _ = fs::remove_file(entry.path());
            }
        }
        self.sealed.clear();
        Ok(())
    }

    /// Re-walk the active segment through `sink`, rebuilding the caller's
    /// index state after an [`Self::adopt_compacted`] (the store is a single
    /// segment at that point). Truncates a torn tail and sets the write
    /// offset. Returns the recovered chain head and un-indexed tail digests.
    pub fn recover_active_with_sink<F>(
        &mut self,
        mut sink: F,
    ) -> Result<(Option<(u64, u64)>, Vec<IndexEntry>)>
    where
        F: FnMut(&WalkEntry),
    {
        let id = self.active_id;
        let hint = self.active.superblock().last_index;
        let rec = recovery::recover_with_sink(&mut self.active, id, hint, |e| sink(e))?;
        if rec.torn {
            self.active.truncate_to(rec.tail_offset)?;
        }
        self.active.set_write_offset(rec.tail_offset);
        Ok((rec.last_index, rec.tail_digests))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::log_file::SUPERBLOCK_SIZE;
    use crate::persistence::record::{encode_record, RecordHeader, RecordType};

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("seglog_{tag}_{nanos}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn chunk_bytes(stream_id: u64, chunk_index: u64, payload: &[u8]) -> Vec<u8> {
        encode_record(
            &RecordHeader {
                record_type: RecordType::Chunk,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id,
                chunk_index,
                token_count: 32,
            },
            payload,
        )
    }

    #[test]
    fn segment_name_round_trips() {
        assert_eq!(segment_name(SegmentId(1)), "seg-0000000001.log");
        assert_eq!(segment_name(SegmentId(42)), "seg-0000000042.log");
        assert_eq!(parse_segment("seg-0000000001.log"), Some(SegmentId(1)));
        // A legacy `.active` is NOT a segment name — it's adopted separately.
        assert_eq!(parse_segment("seg-0000000042.active"), None);
        assert_eq!(parse_segment("substrate.log"), None);
        assert_eq!(parse_segment("seg-.log"), None);
        assert_eq!(parse_segment("random.txt"), None);
        // The legacy-active parser recognises the retired `.active` extension.
        assert_eq!(
            parse_legacy_active("seg-0000000042.active"),
            Some(SegmentId(42))
        );
        assert_eq!(parse_legacy_active("seg-0000000042.log"), None);
    }

    #[test]
    fn fresh_store_mints_first_active() {
        let dir = tmp_dir("fresh");
        let opened = SegmentedLog::open_with_sink(&dir, |_| {}).unwrap();
        assert_eq!(opened.segments.active_id(), FIRST_SEGMENT);
        assert!(opened.segments.sealed_ids().is_empty());
        assert!(dir.join("seg-0000000001.log").exists());
        assert_eq!(opened.recovered_records, 0);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn legacy_substrate_log_auto_splits() {
        let dir = tmp_dir("migrate");
        // Write a legacy monolithic substrate.log with two chunks.
        {
            let legacy = dir.join(LEGACY_LOG_NAME);
            let mut log = LogFile::create(&legacy).unwrap();
            log.stage(&chunk_bytes(7, 0, b"alpha"));
            log.stage(&chunk_bytes(7, 1, b"beta"));
            log.commit().unwrap();
        }
        let mut records = 0usize;
        let opened = SegmentedLog::open_with_sink(&dir, |_| records += 1).unwrap();
        // Legacy became sealed seg 1; a fresh active seg 2 was minted.
        assert!(dir.join("seg-0000000001.log").exists());
        assert!(!dir.join(LEGACY_LOG_NAME).exists());
        assert_eq!(opened.segments.active_id(), SegmentId(2));
        assert_eq!(opened.segments.sealed_ids(), &[SegmentId(1)]);
        assert_eq!(records, 2, "both legacy chunks replay");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn migration_skipped_when_already_segmented() {
        let dir = tmp_dir("nomigrate");
        // A store that already has a segment plus a stray legacy file: the
        // legacy file must be left untouched (never silently sealed over an
        // existing segment set).
        LogFile::create(&dir.join(segment_name(SegmentId(1)))).unwrap();
        LogFile::create(&dir.join(LEGACY_LOG_NAME)).unwrap();
        SegmentedLog::open_with_sink(&dir, |_| {}).unwrap();
        assert!(dir.join(LEGACY_LOG_NAME).exists(), "legacy file untouched");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn seal_and_rotate_preserves_records_across_segments() {
        let dir = tmp_dir("rotate");
        // Write two records to seg 1, seal it, write one to seg 2.
        let (s1_off, s2_off);
        {
            let mut opened = SegmentedLog::open_with_sink(&dir, |_| {}).unwrap();
            let seg = &mut opened.segments;
            let (id, off) = seg.stage(&chunk_bytes(1, 0, b"first"));
            assert_eq!(id, FIRST_SEGMENT);
            s1_off = off;
            seg.stage(&chunk_bytes(1, 1, b"second"));
            seg.commit().unwrap();
            seg.seal_and_rotate().unwrap();
            assert_eq!(seg.active_id(), SegmentId(2));
            assert_eq!(seg.sealed_ids(), &[SegmentId(1)]);
            let (id2, off2) = seg.stage(&chunk_bytes(2, 0, b"third"));
            assert_eq!(id2, SegmentId(2));
            s2_off = off2;
            seg.commit().unwrap();

            // Read routing: seg 1 record via the pool, seg 2 via the active.
            let r1 = seg.read_record_at(SegmentId(1), s1_off, 4096).unwrap();
            assert_eq!(r1.header.stream_id, 1);
            let r2 = seg.read_record_at(SegmentId(2), s2_off, 4096).unwrap();
            assert_eq!(r2.header.stream_id, 2);
        }
        // Reopen: seg 1 sealed + seg 2 active, all three records replay.
        let mut replayed = Vec::new();
        let opened = SegmentedLog::open_with_sink(&dir, |e| {
            replayed.push((
                e.segment,
                e.record.header.stream_id,
                e.record.header.chunk_index,
            ));
        })
        .unwrap();
        assert_eq!(opened.segments.active_id(), SegmentId(2));
        assert_eq!(opened.segments.sealed_ids(), &[SegmentId(1)]);
        assert!(replayed.contains(&(SegmentId(1), 1, 0)));
        assert!(replayed.contains(&(SegmentId(1), 1, 1)));
        assert!(replayed.contains(&(SegmentId(2), 2, 0)));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn legacy_active_file_is_adopted_on_open() {
        let dir = tmp_dir("adopt");
        // A pre-single-namespace store: the active was `seg-<id>.active`, with a
        // lower-id sealed `.log`. Open must adopt the `.active` into `.log`.
        {
            let mut active = LogFile::create(&dir.join(format!("seg-{:010}.active", 3))).unwrap();
            active.stage(&chunk_bytes(9, 0, b"legacy-active"));
            active.commit().unwrap();
            let mut sealed = LogFile::create(&dir.join(segment_name(SegmentId(2)))).unwrap();
            sealed.stage(&chunk_bytes(8, 0, b"sealed"));
            sealed.commit().unwrap();
        }
        let mut replayed = 0usize;
        let opened = SegmentedLog::open_with_sink(&dir, |_| replayed += 1).unwrap();
        // The `.active` was adopted to `.log`; as the highest id it is the active.
        assert_eq!(opened.segments.active_id(), SegmentId(3));
        assert_eq!(opened.segments.sealed_ids(), &[SegmentId(2)]);
        assert!(dir.join("seg-0000000003.log").exists());
        assert!(!dir.join("seg-0000000003.active").exists());
        assert_eq!(replayed, 2, "the adopted-active + sealed records replay");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compact_scratch_file_is_removed_on_open() {
        let dir = tmp_dir("scratch");
        fs::write(&dir.join("substrate.log.compact"), b"garbage").unwrap();
        SegmentedLog::open_with_sink(&dir, |_| {}).unwrap();
        assert!(!dir.join("substrate.log.compact").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pool_evicts_lru_beyond_capacity() {
        let dir = tmp_dir("pool");
        // Create OPEN_SEALED_SEGMENTS + 2 sealed segments, then touch each.
        let n = OPEN_SEALED_SEGMENTS + 2;
        for i in 1..=n {
            let id = SegmentId(i as u64);
            let mut log = LogFile::create(&dir.join(segment_name(id))).unwrap();
            log.stage(&chunk_bytes(i as u64, 0, b"x"));
            log.commit().unwrap();
        }
        let mut pool = SealedPool::new(OPEN_SEALED_SEGMENTS);
        for i in 1..=n {
            pool.get_or_open(&dir, SegmentId(i as u64)).unwrap();
        }
        assert_eq!(
            pool.open.len(),
            OPEN_SEALED_SEGMENTS,
            "pool never exceeds its capacity"
        );
        // The oldest-touched ids (1, 2) were evicted; the newest is present.
        assert!(pool.open.iter().any(|(id, _)| *id == SegmentId(n as u64)));
        assert!(!pool.open.iter().any(|(id, _)| *id == SegmentId(1)));
        fs::remove_dir_all(&dir).ok();
    }

    /// The reader `read_into` routes a stripe read to the correct segment
    /// file (regression guard for the offset-vs-segment routing).
    #[test]
    fn read_into_routes_by_segment() {
        let dir = tmp_dir("readinto");
        let mut opened = SegmentedLog::open_with_sink(&dir, |_| {}).unwrap();
        let seg = &mut opened.segments;
        let rec = chunk_bytes(3, 0, b"routed-bytes");
        let (_, off) = seg.stage(&rec);
        seg.commit().unwrap();
        let mut buf = vec![0u8; rec.len()];
        seg.read_into(FIRST_SEGMENT, off, &mut buf).unwrap();
        assert_eq!(&buf[..], &rec[..]);
        // Offset region starts at the superblock boundary for the first record.
        assert_eq!(off, SUPERBLOCK_SIZE);
        fs::remove_dir_all(&dir).ok();
    }
}
