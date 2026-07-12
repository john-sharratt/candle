//! Multi-log inheritance — read-only inherited substrates and the
//! process-wide shared cache (§13.5 of `docs/kv_tier_migration.md`).
//!
//! A child substrate is opened over an ordered list of logs: the last is
//! its own writable active log, the earlier ones are inherited and
//! read-only. Inherited logs are loaded through a process-wide cache keyed
//! by canonical path, so a common base is loaded **once** and shared by
//! `Arc` across every child that inherits it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use super::direct_io::DirectFile;
use super::log_file::LogSource;
use super::log_file::{read_record_at, LogFile};
use super::manifest::Manifest;
use super::record::Record;
use super::recovery;
use super::streams::StreamId;
use super::Result;
use crate::substrate::Substrate;

/// A loaded, read-only inherited substrate log.
pub struct InheritedSubstrate {
    /// Canonical path of the inherited log file.
    path: PathBuf,
    /// The recovered manifest — carries only the singleton offsets
    /// (`ModelSpec`, `Template`, `Tokenizer`, `ToolSummary`).
    manifest: Manifest,
    /// In-RAM substrate populated by walking the inherited log during
    /// `load`.  Holds per-stream / per-timeline state (chunks, tokens,
    /// signatures, decls, labels, tree metadata, etc.) — the same
    /// state the active substrate gets from its log, just for the
    /// inherited (read-only) ancestor.  Cold-load consumers query
    /// `inherited.substrate()` instead of `inherited.manifest().streams`.
    substrate: Substrate,
    /// The open file, for reading inherited records on demand. Behind a
    /// `Mutex` because reads seek and the substrate is shared.
    file: Mutex<LogFile>,
    /// Cache-bypassing handle for the cold-load fast path. Independent
    /// of `file` — positioned reads via `pread`/`ReadFile`+`OVERLAPPED`
    /// are thread-safe so this is shared without a lock.
    direct: DirectFile,
}

impl InheritedSubstrate {
    /// The canonical path of the inherited log.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The inherited log's manifest.  Carries only singleton offsets;
    /// per-entity state lives on [`Self::substrate`].
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// In-RAM substrate populated by walking the inherited log.
    /// Cold-load reads stream / timeline state from here.
    pub fn substrate(&self) -> &Substrate {
        &self.substrate
    }

    /// Whether this inherited log declares `stream_id`.
    pub fn has_stream(&self, stream_id: StreamId) -> bool {
        self.substrate.has_stream(stream_id)
    }

    /// Read a record at `offset` from the inherited log. `record_size`
    /// is the padded on-disk size from the stream-index entry —
    /// captured at walk time. Single read.
    pub fn read_record(&self, offset: u64, record_size: u64) -> Result<Record> {
        let mut file = self.file.lock().unwrap();
        read_record_at(&mut *file, offset, record_size)
    }

    /// Read `dest.len()` bytes at `offset` from the inherited log
    /// directly into `dest`. Used by the batched cold-read path so a
    /// single caller-provided scratch absorbs a whole turn's records.
    pub fn read_into(&self, offset: u64, dest: &mut [u8]) -> Result<()> {
        let mut file = self.file.lock().unwrap();
        LogSource::read_into(&mut *file, offset, dest)
    }

    /// The cache-bypassing read handle on this inherited log. Used by
    /// the cold-load fast path to submit stripe reads in parallel via
    /// [`DirectFile::read_stripes_concurrent`].
    pub fn direct_file(&self) -> &DirectFile {
        &self.direct
    }
}

fn cache() -> &'static Mutex<HashMap<PathBuf, Arc<InheritedSubstrate>>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<InheritedSubstrate>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

impl InheritedSubstrate {
    /// Load an inherited log, recovering its manifest. Loads are shared:
    /// repeated calls for the same file return the *same* `Arc`, so a
    /// common base substrate exists once in memory.
    pub fn load(path: &Path) -> Result<Arc<InheritedSubstrate>> {
        let canonical = path.canonicalize()?;
        {
            let cache = cache().lock().unwrap();
            if let Some(existing) = cache.get(&canonical) {
                return Ok(Arc::clone(existing));
            }
        }
        // Recover outside the cache lock so a slow load does not block
        // other paths.  The walker drives both the manifest's
        // singleton offsets AND the substrate's per-entity state in a
        // single pass — recovery::recover_with_sink dispatches each
        // record through `substrate.apply_walker_entry`.
        let mut file = LogFile::open(&canonical)?;
        let hint = file.superblock().last_index;
        let mut substrate = Substrate::new();
        let recovered = recovery::recover_with_sink(&mut file, hint, |entry| {
            substrate.apply_walker_entry(entry)
        })?;
        let direct = DirectFile::open(&canonical)?;
        let loaded = Arc::new(InheritedSubstrate {
            path: canonical.clone(),
            manifest: recovered.manifest,
            substrate,
            file: Mutex::new(file),
            direct,
        });

        let mut cache = cache().lock().unwrap();
        // Another thread may have loaded it while we recovered — honour the
        // first one to win so the `Arc` stays unique per path.
        Ok(Arc::clone(cache.entry(canonical).or_insert(loaded)))
    }

    /// Drop a path from the shared cache (used by tests).
    #[cfg(test)]
    pub fn forget(path: &Path) {
        if let Ok(canonical) = path.canonicalize() {
            cache().lock().unwrap().remove(&canonical);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::record::{encode_record, RecordHeader, RecordType};
    use crate::persistence::streams::{ContentAddress, SectionDecl, StreamDecl};

    fn tmp_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("kvtier_inherit_{tag}_{nanos}.log"));
        p
    }

    fn section_decl(name: &str) -> StreamDecl {
        StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress::default(),
            debug_name: name.to_string(),
        })
    }

    /// Build a base log on disk with one declared section stream.
    fn build_base(path: &Path, decl: &StreamDecl) -> StreamId {
        let mut log = LogFile::create(path).unwrap();
        let sid = decl.stream_id();
        let header = RecordHeader {
            record_type: RecordType::StreamDecl,
            format: 0,
            payload_len: decl.encode().len() as u64,
            crc: 0,
            stream_id: sid.0,
            chunk_index: 0,
            token_count: 0,
        };
        log.stage(&encode_record(&header, &decl.encode()));
        log.commit().unwrap();
        sid
    }

    #[test]
    fn load_recovers_the_inherited_manifest() {
        let path = tmp_path("recover");
        let decl = section_decl("base_section");
        let sid = build_base(&path, &decl);

        InheritedSubstrate::forget(&path);
        let inherited = InheritedSubstrate::load(&path).unwrap();
        assert!(inherited.has_stream(sid));
        assert_eq!(inherited.substrate().all_stream_ids().count(), 1);

        InheritedSubstrate::forget(&path);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn repeated_load_shares_one_arc() {
        let path = tmp_path("shared");
        build_base(&path, &section_decl("shared_base"));

        InheritedSubstrate::forget(&path);
        let a = InheritedSubstrate::load(&path).unwrap();
        let b = InheritedSubstrate::load(&path).unwrap();
        // The common base is loaded once and shared — not duplicated.
        assert!(Arc::ptr_eq(&a, &b));
        assert!(Arc::strong_count(&a) >= 2);

        InheritedSubstrate::forget(&path);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn distinct_paths_are_distinct_substrates() {
        let p1 = tmp_path("d1");
        let p2 = tmp_path("d2");
        build_base(&p1, &section_decl("one"));
        build_base(&p2, &section_decl("two"));

        InheritedSubstrate::forget(&p1);
        InheritedSubstrate::forget(&p2);
        let a = InheritedSubstrate::load(&p1).unwrap();
        let b = InheritedSubstrate::load(&p2).unwrap();
        assert!(!Arc::ptr_eq(&a, &b));

        InheritedSubstrate::forget(&p1);
        InheritedSubstrate::forget(&p2);
        std::fs::remove_file(&p1).ok();
        std::fs::remove_file(&p2).ok();
    }
}
