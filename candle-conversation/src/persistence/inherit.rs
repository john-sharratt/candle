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

use super::checkpoint;
use super::log_file::{read_record_at, LogFile};
use super::manifest::Manifest;
use super::record::Record;
use super::streams::StreamId;
use super::Result;

/// A loaded, read-only inherited substrate log.
pub struct InheritedSubstrate {
    /// Canonical path of the inherited log file.
    path: PathBuf,
    /// The recovered manifest — the inherited log's stream index.
    manifest: Manifest,
    /// The open file, for reading inherited records on demand. Behind a
    /// `Mutex` because reads seek and the substrate is shared.
    file: Mutex<LogFile>,
}

impl InheritedSubstrate {
    /// The canonical path of the inherited log.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The inherited log's manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Whether this inherited log declares `stream_id`.
    pub fn has_stream(&self, stream_id: StreamId) -> bool {
        self.manifest.streams.contains_key(&stream_id)
    }

    /// Read a record at `offset` from the inherited log.
    pub fn read_record(&self, offset: u64) -> Result<Record> {
        let mut file = self.file.lock().unwrap();
        read_record_at(&mut *file, offset)
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
        // other paths.
        let mut file = LogFile::open(&canonical)?;
        let hint = file.superblock().latest_checkpoint_offset;
        let recovered = checkpoint::recover(&mut file, hint)?;
        let loaded = Arc::new(InheritedSubstrate {
            path: canonical.clone(),
            manifest: recovered.manifest,
            file: Mutex::new(file),
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
        assert_eq!(inherited.manifest().streams.len(), 1);

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
        assert_eq!(Arc::strong_count(&a) >= 2, true);

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
