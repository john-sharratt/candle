//! Shares one writable substrate handle per directory, so a host that writes records of its own and the engine that runs on them append through one cursor.
//!
//! Two writable handles on one directory each choose offsets the other does not know about, and a compaction then drops every record the other appended after it opened, so a second writable open is refused. A host that writes its own records opens the substrate once and hands the pair to the engine. Custom objects are durable and last-writer-wins: the greater stamp is the object on replay, and the in-RAM substrate holds only the winner, so compaction re-emits it.

use super::*;

/// A substrate the host process opened, for the engine to adopt rather than
/// open a second time.
///
/// **One writable handle per `substrate/`.** A second [`SubstratePersistence`]
/// over the same directory would be a second append cursor *and* a second
/// [`SubstratePersistence::npc_locs`] view. The two writers would choose offsets
/// neither told the other about, and — because
/// [`compaction::collect_live_records`] can only carry forward what the
/// compacting handle's own walk recorded — a compaction would drop every
/// header-keyed record the other handle appended after it opened. A writable
/// open therefore holds [`writer_lock::WriterLock`], and a second one is refused.
///
/// That is not theoretical: `npcd` opened one handle for its character registry
/// and let the engine open another for conversation turns, and every character
/// created while the daemon ran vanished at the next restart. The ones that
/// predated both opens survived, which made the loss look intermittent instead
/// of mechanical. See
/// [`tests::a_second_writable_handle_is_refused`].
///
/// So a host that writes records of its own — `npcd` writes
/// [`RecordType::Npc`] — opens the substrate **once** and hands this pair over,
/// instead of naming a directory and letting the engine open its own. Both
/// sides then append through the same cursor and compact against the same view.
#[derive(Clone)]
pub struct SharedSubstrate {
    pub substrate: Arc<SubstrateLock>,
    pub persistence: Arc<PersistenceOwner>,
}

/// Names the directory rather than the contents. A substrate's interesting
/// state is megabytes of index and arena bookkeeping, and this appears inside
/// the `Debug` for a whole engine config — the useful fact at that scale is
/// *which* log it is.
impl std::fmt::Debug for SharedSubstrate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SharedSubstrate")
            .field(&self.persistence.dir())
            .finish()
    }
}

impl SharedSubstrate {
    /// Wrap a freshly-opened pair: the substrate's versioned root, and the log
    /// handed to a thread of its own ([`PersistenceOwner`]). The caller does the
    /// walk — typically through
    /// [`SubstratePersistence::open_in_with_substrate_and_sink`], so its own
    /// record classes are collected in the same pass.
    pub fn new(substrate: Substrate, persistence: SubstratePersistence) -> Self {
        let substrate = Arc::new(SubstrateLock::new(substrate));
        let persistence = PersistenceOwner::spawn(persistence, Arc::clone(&substrate));
        Self {
            substrate,
            persistence,
        }
    }

    /// Open the substrate under `dir` for `admit`'s model and share it. The
    /// one-liner for a host with no records of its own to collect on the walk.
    pub fn open_in(dir: &Path, admit: Admit<'_>) -> Result<Self> {
        Self::open_in_reporting(dir, admit, &|_, _, _| {})
    }

    /// [`Self::open_in`], telling `progress` how the open goes — a store written
    /// before identities is rewritten whole before it returns.
    pub fn open_in_reporting(
        dir: &Path,
        admit: Admit<'_>,
        progress: OpenProgress<'_>,
    ) -> Result<Self> {
        let mut substrate = Substrate::new();
        let persistence = SubstratePersistence::open_adopting(
            &dir.join(SUBSTRATE_DIR),
            &mut substrate,
            admit,
            |_| {},
            progress,
        )?;
        Ok(Self::new(substrate, persistence))
    }

    /// Open the substrate under `dir` **read-only** and share it.
    ///
    /// For a tool that reads a workspace another process — the daemon — may be
    /// writing to at the same time. The store must already exist, and nothing
    /// under `dir/substrate/` is created, renamed, deleted, truncated, grown or
    /// written for as long as the pair lives. Everything in RAM behaves as it
    /// does on a writable substrate; only the durable side is absent — see
    /// [`SubstratePersistence::open_in_with_substrate_read_only`].
    pub fn open_in_read_only(dir: &Path) -> Result<Self> {
        let mut substrate = Substrate::new();
        let persistence =
            SubstratePersistence::open_in_with_substrate_read_only(dir, &mut substrate)?;
        Ok(Self::new(substrate, persistence))
    }

    /// Store a keyed [`CustomObjectPayload`] — durably and in RAM in one call.
    ///
    /// A later write with the same key supersedes this one (the greater stamp
    /// wins): the header carries the key's hash as its `stream_id`, and the
    /// in-RAM [`Substrate`] holds only the winner, under the identity its
    /// record carries, so compaction re-emits it. This is the write half of the
    /// durable host state (e.g. npcd's command-table open flag);
    /// [`Self::custom_object`] reads it back. A blob's tombstone
    /// ([`BlobKey::tombstone_key`], ceased) retires the blob here at once.
    pub fn put_custom_object(&self, obj: CustomObjectPayload) -> Result<()> {
        let retires = retired_blob(&obj);
        let id = self.persistence.call(|p| {
            let id = p.mint();
            p.write_custom_object(&obj, id)?;
            p.commit()?;
            Ok::<_, PersistenceError>(id)
        })?;
        self.substrate
            .update(|s| s.apply_custom_object(obj.clone(), id));
        self.forget_retired_blobs(retires);
        Ok(())
    }

    /// Forget each of `keys` whose tombstone now outranks the blob held —
    /// after the tombstones are durable and applied.
    pub(super) fn forget_retired_blobs(&self, keys: impl IntoIterator<Item = BlobKey>) {
        let keys: Vec<BlobKey> = keys.into_iter().collect();
        if keys.is_empty() {
            return;
        }
        let s = self.substrate.read();
        self.persistence.call(|p| {
            for key in keys {
                p.forget_retired_blob(&s, key);
            }
        });
    }

    /// Retire the blob `sha256` names: write its tombstone
    /// ([`BlobKey::tombstone_key`], ceased), which replicates like any keyed
    /// one. It reads as absent at once, and its bytes go at the next
    /// compaction or maintenance pass over them (`docs/replicated_substrate.md`
    /// §10.3).
    pub fn retire_blob(&self, sha256: &[u8; 32]) -> Result<()> {
        self.put_custom_object(CustomObjectPayload::ceased(
            BlobKey::of(sha256).tombstone_key(),
        ))
    }

    /// Read the live [`CustomObjectPayload`] for `key`, or `None` if none was
    /// ever written, or its winning record is its tombstone
    /// ([`CustomObjectPayload::ceased`]).
    pub fn custom_object(&self, key: &str) -> Option<CustomObjectPayload> {
        self.substrate.read().custom_object(key).cloned()
    }

    /// Store `objs` — values and tombstones alike — under the lease epoch
    /// `epoch`, in one commit (`docs/replicated_substrate.md` §9.1): a world's
    /// authority writes every entity that changed in a beat at once, each
    /// stamped with its lease's epoch so no earlier holder's write outranks it.
    pub fn put_custom_objects_under(
        &self,
        objs: Vec<CustomObjectPayload>,
        epoch: u64,
    ) -> Result<()> {
        if objs.is_empty() {
            return Ok(());
        }
        let ids = self.persistence.call(|p| {
            let mut ids = Vec::with_capacity(objs.len());
            for obj in &objs {
                let id = p.mint_under(epoch);
                p.write_custom_object(obj, id)?;
                ids.push(id);
            }
            p.commit()?;
            Ok::<_, PersistenceError>(ids)
        })?;
        let retires: Vec<BlobKey> = objs.iter().filter_map(retired_blob).collect();
        self.substrate.update(|s| {
            for (obj, &id) in objs.iter().zip(&ids) {
                s.apply_custom_object(obj.clone(), id);
            }
        });
        self.forget_retired_blobs(retires);
        Ok(())
    }

    /// Every live custom object whose key starts with `prefix`, in key order,
    /// with the stamp of the record that holds it.
    pub fn custom_objects_under(&self, prefix: &str) -> Vec<(CustomObjectPayload, VersionStamp)> {
        self.substrate
            .read()
            .custom_objects_under(prefix)
            .into_iter()
            .map(|(obj, id)| (obj, id.stamp()))
            .collect()
    }

    /// How many live custom objects' keys start with `prefix`.
    pub fn custom_objects_count_under(&self, prefix: &str) -> usize {
        self.substrate.read().custom_objects_count_under(prefix)
    }

    /// The live custom object under `prefix` whose key sorts last.
    pub fn custom_object_last_under(&self, prefix: &str) -> Option<CustomObjectPayload> {
        self.substrate.read().custom_object_last_under(prefix)
    }

    /// A count that moves whenever a custom object changes: read a prefix
    /// again only when it has.
    pub fn custom_objects_version(&self) -> u64 {
        self.substrate.read().custom_objects_version()
    }

    /// Write `write` as `timeline`'s file events, then its tombstones, in
    /// one commit; the ids the events took (`docs/zend_vfs_events.md`).
    pub fn write_vfs(&self, timeline: u64, write: &VfsWrite) -> Result<Vec<VfsEventId>> {
        self.persistence.call(|p| {
            let ids = p.write_vfs(timeline, write)?;
            p.commit()?;
            Ok(ids)
        })
    }

    /// Every live file event of `timeline`, in `(repo, key)` order and within
    /// a key in stamp order.
    ///
    /// The owner commits what is staged and says where the events are; they
    /// are read off its thread. Maintenance may move one first, and then the
    /// owner reads them all.
    pub fn vfs_events(&self, timeline: u64) -> Result<Vec<VfsEventPayload>> {
        let locs = self.persistence.call(|p| p.vfs_event_locs(timeline))?;
        match self.persistence.sealed_reader().read_records(&locs) {
            Some(Ok(records)) => vfs_events_of(&records),
            _ => self.persistence.call(|p| p.vfs_events(timeline)),
        }
    }

    /// Every timeline holding file events or tombstones, in id order.
    pub fn vfs_timelines(&self) -> Vec<u64> {
        self.persistence.call(|p| {
            p.vfs_index()
                .timelines()
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        })
    }

    /// Declare `timeline` a file store, so every rewrite of the log carries its
    /// events though no conversation is registered on it
    /// ([`Substrate::is_file_store`]) — once: a store already declared writes
    /// nothing.
    pub fn declare_file_store(&self, timeline: TimelineId) -> Result<()> {
        if self.substrate.read().is_file_store(timeline) {
            return Ok(());
        }
        self.put_custom_object(CustomObjectPayload::new(
            file_store_key(timeline),
            serde_json::Value::Null,
        ))
    }

    /// Keep `bytes` as a blob, durably, unless one of their hash is already
    /// held; their SHA-256 either way (`docs/replicated_substrate.md` §10.3).
    pub fn put_blob(&self, bytes: Vec<u8>) -> Result<[u8; 32]> {
        self.persistence.call(|p| {
            let sha256 = p.write_blob(bytes)?;
            p.commit_if_pending()?;
            Ok(sha256)
        })
    }

    /// The bytes named `sha256`, or `None` when no blob of that hash is held.
    ///
    /// The owner commits what is staged and says where the blob is; the bytes,
    /// up to a 64 MiB pack, are read off its thread. Maintenance may move the
    /// record first, and then the owner reads it.
    pub fn blob(&self, sha256: &[u8; 32]) -> Result<Option<Vec<u8>>> {
        let Some(loc) = self.persistence.call(|p| p.blob_loc(sha256))? else {
            return Ok(None);
        };
        match self.persistence.sealed_reader().read_record_payload(&loc) {
            Some(Ok(payload)) => blob_bytes(&payload, sha256),
            _ => self.persistence.call(|p| p.blob(sha256)),
        }
    }

    /// Whether a blob of `sha256` is held, without reading it: a pack of
    /// 64 MiB is known to have arrived without being read
    /// (`docs/substrate_git.md` §4.1).
    pub fn has_blob(&self, sha256: &[u8; 32]) -> bool {
        self.persistence.call(|p| p.has_blob(sha256))
    }

    /// Every kept custom object, tombstones included, and the blobs they
    /// name, as `daemon`'s export taken now ([`kept_objects`]).
    pub fn export_kept_objects(&self, daemon: &str) -> anyhow::Result<String> {
        let held = self.substrate.read().held_custom_objects();
        kept_objects::export(held, |name| Ok(self.blob(name)?), daemon, now_ms())
    }

    /// Write the union of the exports `text` holds — one from each member —
    /// leaving every key already live as it is ([`kept_objects`]).
    pub fn import_kept_objects(&self, text: &str, daemon: &str) -> anyhow::Result<ImportReport> {
        kept_objects::import(
            kept_objects::union(text, daemon)?,
            |key| self.custom_object(key).is_some(),
            |bytes| Ok(self.put_blob(bytes)?),
            |obj| Ok(self.put_custom_object(obj)?),
        )
    }
}
