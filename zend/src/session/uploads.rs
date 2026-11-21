//! Stores, records and reads back the files a user uploads into a conversation,
//! and runs their ingest.
//!
//! An upload is stored in its conversation's uploads store and recorded as an
//! event, so it recovers with the conversation and sits in its history at the
//! turn it arrived. It is read into the per-file ingest layer, bounded to just
//! those files. An upload's conversation is retired once no conversation holds
//! the file, so deleted uploads do not accumulate live turns. Upload records are
//! rewritten as one step, so two uploads, or an upload and a delete, both land
//! whole whatever order they race in.

use super::*;

impl InferenceState {
    /// Ingest **only** the given workspace-relative files — in the uploads
    /// repository — into the projection's per-file ingest layer: the upload's
    /// read_file phase. Bounded to those files (no branch walk, no sweep), so
    /// it can't overload the model with the entire repo. Reports per-file
    /// progress into `progress`. Returns `(ingested, failed)`: whether anything
    /// was read, and whether at least one file's ingest tolerated-failed (e.g.
    /// out of KV VRAM) — so the upload can report a real failure.
    ///
    /// If the projection declares no per-file ingest layer (a pure conversational
    /// mind), there is nowhere to read uploads into — a no-op. A per-file layer
    /// suppressed by `--disable-layer` still accepts uploads (they're bounded and
    /// safe).
    pub(crate) fn ingest_uploaded_files(
        &self,
        conv_id: &str,
        rel_paths: &[String],
        progress: &Arc<LoadProgress>,
    ) -> anyhow::Result<(bool, bool)> {
        let Some(files_layer) = self
            .ingest_layers
            .iter()
            .find(|l| l.mode == IngestMode::Files)
        else {
            tracing::debug!(
                "upload ingest: projection has no per-file ingest layer — skipping read_file"
            );
            return Ok((false, false));
        };
        // The uploads are the conversation's: its files, as its events hold
        // them, are what the read reads.
        let files = self.saved_files(conv_id);
        let ctx = RefreshContext {
            tool_ctx: self.tool_host.context_for(ToolMode::Restricted, &files, ""),
            ..self.refresh_ctx()
        };
        let (ingested, n_failed) = crate::code_read::ingest_files(
            &ctx,
            rel_paths,
            progress,
            &files_layer.name,
            self.ingest_base(&files_layer.name)?,
            files_layer.calibrate.as_deref(),
        )?;
        if ingested {
            // An upload is in every conversation's scope once it is committed.
            self.retrieval_scope.refresh(&self.engine);
        }
        Ok((ingested, n_failed > 0))
    }

    /// `conv_id`'s file stores as its saved events hold them: a fresh set,
    /// restored, never saved — what a read of its uploads reads.
    pub(crate) fn saved_files(&self, conv_id: &str) -> Arc<RepoFiles> {
        let files = self.tool_host.conversation_files();
        let mirror = Mutex::new(Mirror::unread());
        conv_overlay::restore(&self.engine, timeline_for(conv_id), &files, &mirror);
        files
    }

    /// Run `change` over `conv_id`'s file stores and save what it changed as
    /// the conversation's file events (§10.2). A conversation open here is
    /// changed under its own lock, so an upload and a turn never interleave;
    /// one that is not is changed over its stores restored from those events,
    /// put on each repository's base first when it has never worked anywhere.
    pub(crate) async fn change_files<T>(
        &self,
        conv_id: &str,
        change: impl FnOnce(&RepoFiles) -> T,
    ) -> T {
        let open = self.conversations.get(conv_id);
        if let Some(conv) = open {
            let cs = conv.lock_owned().await;
            let out = change(&cs.files);
            conv_overlay::save(&self.engine, timeline_for(conv_id), &cs.files, &cs.mirror);
            return out;
        }
        self.change_saved(conv_id, change)
    }

    /// [`Self::change_files`] for a conversation not open here: its stores
    /// restored from its events, changed, and saved back.
    pub(crate) fn change_saved<T>(&self, conv_id: &str, change: impl FnOnce(&RepoFiles) -> T) -> T {
        let timeline = timeline_for(conv_id);
        let files = self.tool_host.conversation_files();
        let mirror = Mutex::new(Mirror::unread());
        {
            let engine = &self.engine;
            conv_branches::seed(engine, timeline, &self.base_branches);
            conv_overlay::restore(engine, timeline, &files, &mirror);
        }
        let out = change(&files);
        conv_overlay::save(&self.engine, timeline, &files, &mirror);
        out
    }

    /// Uploads kept before conversations held them, each put in its
    /// conversation's uploads store and its upload record pointed at it; the
    /// old folder removed once every file is held (§10.2, §15.1). A file that
    /// cannot be stored keeps the folder, so nothing is lost.
    pub(crate) fn take_in_legacy_uploads(&self) {
        let root = self.workspace.root();
        let legacy = match crate::legacy_uploads::read(root) {
            Ok(legacy) => legacy,
            Err(e) => {
                tracing::warn!("the old uploads could not be read, and are left as they are: {e}");
                return;
            }
        };
        if legacy.is_empty() {
            return;
        }
        let mut all_held = true;
        for (conv_id, files) in legacy {
            let wanted = files.len();
            let moved: Vec<(u64, UploadInfo)> = self.change_saved(&conv_id, |repos| {
                let Ok(store) = repos.repo(UPLOADS_REPO) else {
                    return Vec::new();
                };
                files
                    .into_iter()
                    .filter_map(|f| {
                        let size = f.bytes.len() as u64;
                        match crate::uploads::store(&store, &f.name, f.bytes) {
                            Ok(stored) => Some((f.id, upload_record(stored, size))),
                            Err(e) => {
                                tracing::warn!(name = %f.name, "an old upload was not taken in: {e}");
                                None
                            }
                        }
                    })
                    .collect()
            });
            all_held &= moved.len() == wanted;
            let timeline = timeline_for(&conv_id);
            let written = edit_upload_records(&self.engine, timeline, |records| {
                for (old, new) in &moved {
                    match records.iter_mut().find(|r| r.id == *old) {
                        Some(r) => {
                            r.id = new.id;
                            r.path = new.path.clone();
                            r.name = new.name.clone();
                        }
                        None => records.push(new.clone()),
                    }
                }
            });
            if !written {
                tracing::warn!(conv_id = %conv_id, "an old upload's record was not moved");
                all_held = false;
            }
        }
        if !all_held {
            tracing::warn!("some old uploads were not taken in; their folder is kept");
            return;
        }
        match crate::legacy_uploads::remove(root) {
            Ok(()) => tracing::info!("the old uploads are held by their conversations"),
            Err(e) => tracing::warn!("the old uploads folder could not be removed: {e}"),
        }
    }

    /// Tombstone the per-file `code_read` conversation of every uploaded file
    /// that no conversation holds any more.
    ///
    /// Uploads are endpoint-managed and on no branch, so the branch pass never
    /// retires them — that would delete uploaded content on the next pass. But
    /// a deleted upload must still retire its conversation; otherwise deleted
    /// uploads accumulate live turns forever, growing the redo log and the
    /// projection candidate set without bound.
    ///
    /// An upload is held while some conversation's upload record names its
    /// path (§10.2), or while the uploads folder — the lower layer every
    /// conversation's uploads store reads — still has it. Cheap and
    /// self-limiting: a metadata scan, no walk and no re-ingest.
    ///
    /// Runs at startup, to retire uploads deleted since the last one.
    pub(crate) fn reconcile_uploaded_files(&self) {
        let engine = &self.engine;
        let held: HashSet<String> = engine
            .conversations_with_metadata_key("uploads")
            .into_iter()
            .filter_map(|(_, json)| serde_json::from_str::<Vec<UploadInfo>>(&json).ok())
            .flatten()
            .map(|u| u.path)
            .collect();
        let mut tombstoned = 0usize;
        for (tl, path) in engine.conversations_with_metadata_key("path") {
            if !crate::code_read::is_upload_path(&path) {
                continue;
            }
            // `path` is workspace-relative with `/` separators; `Path::join`
            // accepts them on win32, and `exists()` is case-insensitive there.
            if held.contains(&path) || self.workspace.root().join(&path).exists() {
                continue;
            }
            match engine.tombstone_timeline(tl) {
                Ok(()) => {
                    tombstoned += 1;
                    tracing::info!(
                        target: "zend::session",
                        path = %path,
                        "reconcile_uploaded_files: tombstoned deleted upload's conversation",
                    );
                }
                Err(err) => tracing::warn!(
                    target: "zend::session",
                    path = %path,
                    "reconcile_uploaded_files: tombstone failed: {err:#}",
                ),
            }
        }
        if tombstoned > 0 {
            tracing::info!(
                target: "zend::session",
                count = tombstoned,
                "reconcile_uploaded_files: retired deleted uploads",
            );
        }
    }
}

/// Measured throughput of an upload, filled in once its pipeline finishes and
/// persisted alongside the [`UploadInfo`] (optional — absent on older events or
/// when the model wasn't loaded to ingest). Batch-level: every file dropped
/// together shares one measurement. Drives the inline "in Xs (X t/s, X MB/s)"
/// stat and the upload-time line in the file viewer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UploadStats {
    /// Total bytes received across the batch.
    pub bytes: u64,
    /// Wall-clock to receive the batch over the wire (phase 1).
    pub upload_ms: u64,
    /// Tokens prefilled reading the batch into the substrate (phase 2).
    pub ingest_tokens: u64,
    /// Wall-clock of the read_file ingest (phase 2).
    pub ingest_ms: u64,
}

/// Rewrite `timeline`'s upload records with `edit`, as one step against the
/// records it holds — two uploads, or an upload and a delete, each landing
/// whole whatever order they race in. Whether the records were written.
pub(super) fn edit_upload_records(
    engine: &ConversationEngine,
    timeline: TimelineId,
    mut edit: impl FnMut(&mut Vec<UploadInfo>),
) -> bool {
    engine
        .update_conversation_metadata(timeline, "uploads", |_, current| {
            let mut records: Vec<UploadInfo> = current
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or_default();
            edit(&mut records);
            serde_json::to_string(&records).ok()
        })
        .is_some()
}

/// One uploaded file recorded against a conversation — the substrate
/// event that survives resume and drives the inline history tile. Stored
/// (JSON-serialized, as an array under the `uploads` custom-metadata key)
/// in the conversation's Label record, so it recovers with the conversation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UploadInfo {
    /// The handle the GUI opens the file's content by: a hash of its path in
    /// the conversation's uploads store ([`crate::uploads::id_of`]).
    pub id: u64,
    pub name: String,
    /// Workspace-relative path of the file (e.g. `uploads/notes-001.txt`),
    /// with `/` separators: `uploads/` and its path in the uploads store.
    pub path: String,
    pub ext: String,
    pub kind: String,
    /// Display size (e.g. `18.4 KB`).
    pub size: String,
    /// Its length in bytes.
    #[serde(default)]
    pub bytes: u64,
    /// Display timestamp (e.g. `just now`).
    pub added: String,
    /// Number of turns present when the file was uploaded — positions the
    /// tile in the history stream (the tile renders after this many turns).
    #[serde(default)]
    pub turn_index: u32,
    /// Measured upload/ingest/summarize throughput, filled in once the
    /// pipeline finishes (`record_upload_stats`). `None` until then, or when
    /// the model wasn't loaded to run the ingest phases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats: Option<UploadStats>,
}

/// The record of a file just stored at `stored` in a conversation's uploads
/// store, `size` bytes long: positioned and measured later.
pub(super) fn upload_record(stored: String, size: u64) -> UploadInfo {
    UploadInfo {
        id: crate::uploads::id_of(&stored),
        path: format!("{UPLOADS_REPO}/{stored}"),
        ext: crate::conv_files::ext_badge(&stored),
        kind: crate::conv_files::kind_for(&stored).as_str().to_string(),
        size: crate::conv_files::fmt_bytes(size),
        bytes: size,
        added: "just now".to_string(),
        name: stored,
        turn_index: 0,
        stats: None,
    }
}

impl ZendSession {
    /// **Phase 1 of the upload pipeline** — each of `files` into `conv_id`'s
    /// uploads store under a safe, unused name, saved as the conversation's
    /// file events (§10.2): what was stored, as its upload record, or the file
    /// and why not. `None` while the model is not loaded: the uploads are the
    /// substrate's.
    pub async fn store_uploads(
        &self,
        conv_id: &str,
        files: Vec<(String, Vec<u8>)>,
    ) -> Option<Vec<Result<UploadInfo, (String, String)>>> {
        let state = self.loaded()?;
        Some(
            state
                .change_files(conv_id, |repos| {
                    let store = match repos.repo(UPLOADS_REPO) {
                        Ok(store) => store,
                        Err(e) => {
                            return files
                                .into_iter()
                                .map(|(name, _)| Err((name, e.to_string())))
                                .collect();
                        }
                    };
                    files
                        .into_iter()
                        .map(|(name, bytes)| {
                            let size = bytes.len() as u64;
                            let stored = crate::uploads::store(&store, &name, bytes)
                                .map_err(|e| (name.clone(), e.to_string()))?;
                            Ok(upload_record(stored, size))
                        })
                        .collect()
                })
                .await,
        )
    }

    /// The upload record `id` of `conv_id`.
    pub(super) fn upload_of(&self, conv_id: &str, id: u64) -> Option<UploadInfo> {
        self.conversation_uploads(conv_id)
            .into_iter()
            .find(|u| u.id == id)
    }

    /// `conv_id`'s file stores as its saved events hold them; `None` while
    /// the model is not loaded.
    pub fn saved_files(&self, conv_id: &str) -> Option<Arc<RepoFiles>> {
        Some(self.loaded()?.saved_files(conv_id))
    }

    /// Upload `id`'s bytes, from `conv_id`'s uploads store. The outer `None`
    /// while the model is not loaded; the inner when there is no such upload.
    pub fn upload_content(&self, conv_id: &str, id: u64) -> Option<Option<Vec<u8>>> {
        let state = self.loaded()?;
        let Some(upload) = self.upload_of(conv_id, id) else {
            return Some(None);
        };
        let files = state.saved_files(conv_id);
        let inner = upload.path.split_once('/').map_or("", |(_, p)| p);
        Some(
            files
                .repo(UPLOADS_REPO)
                .ok()
                .and_then(|s| s.read_bytes(inner).ok().flatten()),
        )
    }

    /// Delete upload `id` from `conv_id`: from its uploads store, saved, and
    /// from its upload records. Whether it was there; `None` while the model
    /// is not loaded.
    pub async fn delete_upload(&self, conv_id: &str, id: u64) -> Option<bool> {
        let state = self.loaded()?;
        let Some(upload) = self.upload_of(conv_id, id) else {
            return Some(false);
        };
        let inner = upload
            .path
            .split_once('/')
            .map_or("", |(_, p)| p)
            .to_string();
        state
            .change_files(conv_id, |repos| {
                if let Ok(store) = repos.repo(UPLOADS_REPO) {
                    store.delete(&inner);
                }
            })
            .await;
        if !edit_upload_records(&state.engine, timeline_for(conv_id), |records| {
            records.retain(|u| u.id != id)
        }) {
            tracing::warn!(conv_id = %conv_id, "forgetting an upload failed");
        }
        Some(true)
    }

    /// **Phase 2 of the upload pipeline** — read the just-uploaded files into
    /// the substrate via the normal `code_read` ingest, scoped to **only those
    /// files** (`rel_paths`, workspace-relative, in `conv_id`'s uploads).
    /// Bounded to the uploaded files' work — never a whole-workspace re-ingest
    /// — so it can't overload the model and is safe under `--skip-code-read`.
    /// Returns `Ok(true)` when something was ingested, `Ok(false)` for a no-op
    /// (non-code files, or the model not loaded). Per-scope progress streams
    /// into `progress`. Returns `(ingested, failed)` — see
    /// `ingest_uploaded_files`.
    pub fn read_file_phase(
        &self,
        conv_id: &str,
        rel_paths: &[String],
        progress: &Arc<crate::loading::LoadProgress>,
    ) -> anyhow::Result<(bool, bool)> {
        let Some(state) = self.loaded() else {
            return Ok((false, false));
        };
        state.ingest_uploaded_files(conv_id, rel_paths, progress)
    }

    /// Record a batch of just-uploaded files as an event in the substrate,
    /// so they recover with the conversation and can be replayed inline in
    /// its history. Each file is stamped with `turn_index` — the number of
    /// turns present at upload time — which positions it in the history
    /// stream. Persisted (last-writer-wins) in the conversation's Label
    /// record `custom` bag under the `uploads` key, a JSON array appended to
    /// on every upload. No-op when the model isn't loaded.
    pub fn record_uploads(&self, conv_id: &str, files: &[UploadInfo]) {
        if files.is_empty() {
            return;
        }
        let Some(state) = self.inference.load_full() else {
            return;
        };
        let timeline = timeline_for(conv_id);
        let engine = &state.engine;
        // Register the conv_id so an upload-only conversation (one started by
        // dropping a file on the home page, which may never submit a chat turn)
        // is a *listed* conversation: `known_conversations` — and thus the
        // sidebar + the client's server-authoritative sync — only surface
        // timelines that carry a conv_id. Normally the chat submit sets this;
        // without it here the conversation exists only client-side and gets
        // dropped on the next sync (vanishes from the list, and a send bounces
        // back to home). Idempotent for a conversation that already has one.
        if let Err(e) = engine.set_conversation_conv_id(timeline, conv_id) {
            tracing::warn!(conv_id = %conv_id, "record_uploads: set conv_id failed: {e}");
        }
        conv_branches::seed(engine, timeline, &state.base_branches);
        engine.touch_conversation(timeline);
        // Give it a provisional label from the file(s) if it has none yet, so it
        // shows a sensible name in the sidebar before any chat turn — the titler
        // refines it once the user actually talks. Never overwrite an existing
        // label (an upload into a conversation that already has turns).
        if engine
            .conversation_label_of(timeline)
            .unwrap_or_default()
            .is_empty()
        {
            let label = match files.first() {
                Some(f) if files.len() == 1 => f.name.clone(),
                _ => format!("{} files", files.len()),
            };
            if let Err(e) = engine.set_conversation_label(timeline, &label) {
                tracing::warn!(conv_id = %conv_id, "record_uploads: set label failed: {e}");
            }
        }
        // Position: the number of already-present turns. `record_uploads`
        // runs between turns (the user drops a file, then sends), so every
        // prior turn is sealed and visible in `recovered_history`.
        let turn_index = {
            let base = &state.base_conv;
            base.recovered_history(timeline, false).len() as u32
        };
        // Append the new files to the uploads array it holds.
        if !edit_upload_records(engine, timeline, |records| {
            for f in files {
                let mut f = f.clone();
                f.turn_index = turn_index;
                records.push(f);
            }
        }) {
            tracing::warn!(conv_id = %conv_id, "record_uploads write failed");
        }
    }

    /// Attach measured throughput to an already-recorded upload batch. `ids`
    /// are the conv-file-store ids from the batch's [`UploadInfo`]s; every
    /// matching entry gets `stats` (batch-level, so they share one
    /// measurement). A no-op when the model isn't loaded. Called once the
    /// upload pipeline finishes, after `record_uploads` positioned the batch.
    pub fn record_upload_stats(&self, conv_id: &str, ids: &[u64], stats: &UploadStats) {
        if ids.is_empty() {
            return;
        }
        let Some(state) = self.inference.load_full() else {
            return;
        };
        if !edit_upload_records(&state.engine, timeline_for(conv_id), |records| {
            for entry in records.iter_mut() {
                if ids.contains(&entry.id) {
                    entry.stats = Some(stats.clone());
                }
            }
        }) {
            tracing::warn!(conv_id = %conv_id, "record_upload_stats write failed");
        }
    }

    /// The upload events recorded against a conversation, in upload order —
    /// backs both the files pane and the inline history tiles on resume.
    /// Empty when the model isn't loaded or nothing was uploaded.
    pub fn conversation_uploads(&self, conv_id: &str) -> Vec<UploadInfo> {
        let Some(state) = self.inference.load_full() else {
            return Vec::new();
        };
        let timeline = timeline_for(conv_id);
        let engine = &state.engine;
        engine
            .conversation_metadata_value(timeline, "uploads")
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
}
