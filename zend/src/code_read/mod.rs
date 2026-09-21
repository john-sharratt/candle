//! `code_reading` layer ingestion.
//!
//! Each file becomes ONE hidden conversation, run exactly the way a live
//! dialogue turn runs: forked from the same system-prompt prelude, tools ON
//! (forced to `file_read` alone), thinking at `ThinkMode::Quick` — the lowest
//! level, not fully off. Off was tried first and measured breaking the
//! pattern: with no room to reason at all, the model sometimes skipped
//! `file_read` entirely and guessed a summary from the filename. The opening
//! instruction requires the model to read the whole file — one or more REAL
//! `file_read` calls, ranged if it's long — before answering; whatever it
//! says once it stops calling tools IS the file's summary. There is no
//! synthetic prefill and no separate summary-decode step: it is a normal
//! tool-using conversation that happens not to be shown to anyone (see
//! [`run_file_conversation`], [`opening_prompt`]).
//!
//! Refresh is per-file: content hashes ([`CodeReadState`]) decide which files
//! changed; deleted files' conversations are tombstoned, changed files are
//! re-ingested, and unchanged files are skipped via the substrate resume
//! cache (the per-file `content_sha256` tag).
//!
//! **Parallel ingest.** [`CODE_READ_PARALLELISM`] workers each own one file's
//! conversation and drive it to completion — a bounded pool of real
//! sequences, exactly like any other batch of concurrent conversations the
//! engine wave-batches together.

pub mod carve;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::projection::{
    OptionalState, SelectionState, TimelineId, FORCE_TOOL_SELECTOR, NO_THINK_SELECTOR,
    TOOLS_ENABLED_SELECTOR,
};
use candle_conversation::stencil::TriggerRegistry;
use candle_conversation::{ConversationEngine, Sequence, TurnOptions, TurnText};
use sha2::{Digest, Sha256};
use zend_tools::ToolContext;

use crate::ingest_report::Failures;
use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::repo_scan::{is_binary_sample, FileEntry, Language, RepoMap, MAX_FILE_BYTES};
use crate::tool_round;
use crate::tools::format_tool_responses;

/// Per-file content hash record consulted by the refresh path so a
/// burst of editor saves doesn't trigger a re-prefill of unchanged
/// files.  Keyed by workspace-relative path.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodeReadState {
    pub file_hashes: BTreeMap<String, String>,
}

impl CodeReadState {
    /// Whether `self` and the freshly-walked map name the same files
    /// with the same content hashes.  Drives the no-op short-circuit
    /// in [`refresh_code_reading`].
    pub fn equivalent_to(&self, other: &CodeReadState) -> bool {
        self.file_hashes == other.file_hashes
    }

    /// Workspace-relative paths whose content hash differs (added,
    /// removed, or rewritten).  Informational — the refresh itself
    /// is wholesale.
    pub fn changed_files(&self, other: &CodeReadState) -> Vec<String> {
        let mut out = Vec::new();
        for (p, h) in &other.file_hashes {
            match self.file_hashes.get(p) {
                Some(prev) if prev == h => {}
                _ => out.push(p.clone()),
            }
        }
        for p in self.file_hashes.keys() {
            if !other.file_hashes.contains_key(p) {
                out.push(p.clone());
            }
        }
        out
    }

    /// This state without the files `map`'s `--max-depth` bound froze. A frozen
    /// file stays in the substrate, but the bounded walk never carves it — so
    /// comparing the whole state against the walk would read it as removed.
    pub fn without_frozen(&self, map: &RepoMap) -> Self {
        Self {
            file_hashes: self
                .file_hashes
                .iter()
                .filter(|(path, _)| !map.is_frozen_file(path))
                .map(|(path, hash)| (path.clone(), hash.clone()))
                .collect(),
        }
    }
}

/// Rebuild the [`CodeReadState`] from what the substrate has ALREADY ingested,
/// joining each conversation's `path` and `content_sha256` metadata by timeline.
///
/// This is the durable record of the last (partial or complete) ingest. It is
/// what `session.rs` seeds the in-memory `IngestConv` registry from at boot —
/// unconditionally, whether or not any pass has run this process — and it is
/// what [`refresh_code_reading`]'s first (and every later) pass diffs the
/// freshly-walked files against, entirely off zend's load-to-`ready` critical
/// path. Empty ⇒ nothing ingested yet ⇒ the next pass reads every file as new.
pub fn code_read_state_from_substrate(engine: &Mutex<ConversationEngine>) -> CodeReadState {
    let eng = engine.lock().unwrap();
    let hashes: HashMap<TimelineId, String> = eng
        .conversations_with_metadata_key("content_sha256")
        .into_iter()
        .collect();
    let mut state = CodeReadState::default();
    for (tl, path) in eng.conversations_with_metadata_key("path") {
        if let Some(hash) = hashes.get(&tl) {
            state.file_hashes.insert(path, hash.clone());
        }
    }
    state
}

/// Maximum tolerated per-file summary decode failures in a single
/// ingestion pass before the whole refresh aborts.  A single
/// failure can happen for legitimate reasons (scheduler hiccup,
/// transient resource pressure); a cascade signals something
/// systemic.
pub const MAX_DECODE_FAILURES: usize = 16;

/// Key this pass reports completeness under (see [`crate::ingest_report`]).
pub const PASS_NAME: &str = "code_read";

/// The one tool a file's hidden conversation may call — forced into the
/// catalog via [`FORCE_TOOL_SELECTOR`] so the model reads coherently and
/// never wanders into an unrelated tool.
const FILE_READ_TOOL: &str = "file_read";

/// Decode budget per turn in a file's hidden conversation. `ThinkMode::Quick`
/// alone budgets a graceful close at 1024 thinking tokens (`stencil::think`) —
/// a backstop well above what this turn's actual thinking typically costs, but
/// the cap still has to clear it plus room for the `<tool_call>` or answer
/// that follows, or a turn that legitimately uses the think budget would be
/// cut off before ever reaching its content.
const FILE_TURN_MAX_TOKENS: usize = 1536;

/// Real `file_read` rounds a single file's conversation may run before being
/// forced to answer on whatever it has already seen. The tool caps a single
/// response at `MAX_READ_LINES` lines (`zend-tools`), so a genuinely large
/// file may take several calls to read in full; this bounds the pathological
/// case (or a model that keeps re-reading) rather than the ordinary one, which
/// typically answers well before the cap.
const MAX_FILE_READ_ROUNDS: usize = 24;

/// The opening instruction for a file's hidden conversation. Names the file
/// and the one tool available, and REQUIRES reading it before answering —
/// left as "how much is enough", a model would sometimes skip the tool
/// entirely and guess a summary from the filename alone (measured: a
/// one-line `CHANGELOG.md` "summary" with no `file_read` call at all). "This
/// file, and only this file" also heads off a model wandering into whatever
/// else the workspace prompt might make it curious about.
///
/// A model losing track of its own earlier rounds several turns into a long
/// file was once worked around here, with prose telling it no further
/// question was coming — a symptom this prompt cannot fix, because the
/// symptom wasn't the prompt: this conversation's construction wasn't
/// forking `base_conv` the way a live turn does, and its layer's projection
/// window was sized for the old pre-carved scope excerpts, not a real
/// `file_read` response several times that size (see `session.rs`'s
/// `ingest_bases`, `resolver.rs`'s `score_belief_groups` target exemption,
/// and `code_reading`'s `window` in `projection.yaml`). With those fixed,
/// the model correctly recalls its own earlier rounds without being told to.
fn opening_prompt(path: &str) -> String {
    format!(
        "Read the entire contents of `{path}` — and only this file — using \
         {FILE_READ_TOOL}, calling it as many times as needed to see all of it if \
         it's long. Once you've read the whole thing, summarize what it contains: \
         its purpose, its main structures or functions, and how it fits into the \
         codebase."
    )
}

/// Number of files ingested concurrently by the worker pool. Each worker owns
/// one file's hidden conversation end to end — mint it, run it to a final
/// answer, tag it, free it — so this is the concurrency knob: the number of
/// real conversations feeding the engine's shared multi-session batching at
/// once (their prefills and decodes coalesce into the same forwards every
/// other concurrent conversation's do).
///
/// **Invariant:** each worker holds exactly ONE conversation slot, so this
/// must stay under the model's sequence-slot capacity with headroom for the
/// non-ingest slots the engine also needs concurrently: the live dialogue
/// session, the async summariser's compression passes, etc.
pub const CODE_READ_PARALLELISM: usize = 12;

/// Worker count for the parallel ingest — [`CODE_READ_PARALLELISM`].
fn parallelism() -> usize {
    CODE_READ_PARALLELISM
}

/// Per-file content hash (path-qualified) — the conversation's
/// content-addressed cache key. A file move/rename or any content edit
/// changes it, so the resume cache and the change-detection both key on
/// it. Doubles as the [`CodeReadState`] change-detection digest (keyed by
/// path), so a single hash per file serves both the resume cache and
/// refresh. Path-qualified, so a move/rename re-ingests and the per-path
/// invalidation scan is exact.
pub(crate) fn file_content_hash(path: &str, bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(path.as_bytes());
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// One file queued for ingest: its repo-map entry and content hash. Bytes are
/// read once, here, only to pass the size/binary guards and compute the hash
/// — never shown to the model directly; the model reads the file for itself
/// via a real `file_read` call, so there is nothing else to carry forward.
type QueuedFile = (FileEntry, String);

/// Scan `map`'s files: size-guard, binary-sniff, and hash each one, recording
/// every hash into a fresh [`CodeReadState`]. Files that fail either guard are
/// skipped (never queued, never marked as covered).
fn scan_workspace(workspace: &Path, map: &RepoMap) -> (Vec<QueuedFile>, CodeReadState) {
    let mut per_file = Vec::with_capacity(map.files.len());
    let mut state = CodeReadState::default();
    for file in &map.files {
        let path = workspace.join(&file.path);
        // Size guard (defense-in-depth with `walk_workspace`): the explicit-path
        // ingest builds its own `FileEntry` list and bypasses the walk's size cap,
        // so re-enforce it here.
        match fs::metadata(&path) {
            Ok(m) if m.len() > MAX_FILE_BYTES => {
                tracing::debug!(
                    file = %file.path,
                    bytes = m.len(),
                    "code_read: skip oversize file (> MAX_FILE_BYTES)",
                );
                continue;
            }
            Ok(_) => {}
            Err(e) => {
                tracing::debug!(file = %file.path, "code_read: skip unreadable file: {e}");
                continue;
            }
        }
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::debug!(file = %file.path, "code_read: skip unreadable file: {e}");
                continue;
            }
        };
        // Content guard (defense-in-depth with `walk_workspace`'s sniff): a binary
        // blob that reached here via the explicit-path ingest — which builds its
        // own `FileEntry` list and bypasses the walk — is rejected before its
        // hash is ever recorded.
        if is_binary_sample(&bytes) {
            tracing::debug!(file = %file.path, "code_read: skip binary file (content sniff)");
            continue;
        }
        let fhash = file_content_hash(&file.path, &bytes);
        state.file_hashes.insert(file.path.clone(), fhash.clone());
        per_file.push((file.clone(), fhash));
    }
    (per_file, state)
}

/// Ingest ONLY `rel_paths` into the `code_reading` layer — the upload
/// pipeline's read_file phase.
///
/// Unlike [`refresh_code_reading`], this does **not**
/// walk or reconcile the whole workspace: it scans and ingests just these
/// files, dedupes against already-ingested identical content, and **never**
/// tombstones anything. That matters for two reasons: (1) a full-workspace
/// re-ingest triggered by one upload is a huge, GPU-overloading amount of work
/// (and with `--skip-code-read` the empty prior state makes the refresh treat
/// *every* file as new — the exact overload that killed the expert pipeline
/// thread); (2) a partial file set fed to the workspace refresh would make
/// `reconcile_deleted` tombstone the entire rest of the corpus. This path is
/// bounded to the uploaded files and safe under `--skip-code-read`.
///
/// Files whose extension isn't a recognised code language are skipped (there is
/// nothing to read). Returns the per-file content-hash state for the files that
/// were ingested (to merge into the running [`CodeReadState`]) plus the count of
/// files whose ingest tolerated-failed (e.g. out of KV VRAM), so the upload can
/// surface a real failure.
pub fn ingest_files(
    ctx: &RefreshContext<'_>,
    workspace: &Path,
    rel_paths: &[String],
    progress: &Arc<LoadProgress>,
    layer_name: &str,
    base: &Mutex<Sequence>,
) -> anyhow::Result<(CodeReadState, usize)> {
    // Build a minimal RepoMap for just these files — `scan_workspace` needs
    // only the path + language; the other `FileEntry` fields are unused.
    let mut map = RepoMap::default();
    for rel in rel_paths {
        let norm = rel.replace('\\', "/");
        let ext = norm.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
        let Some(language) = Language::from_extension(&ext) else {
            continue; // not a recognised code language — nothing to read
        };
        map.files.push(FileEntry {
            path: norm,
            line_count: 0,
            language,
            size_bytes: 0,
            module_hint: None,
        });
    }
    if map.files.is_empty() {
        return Ok((CodeReadState::default(), 0));
    }

    let layer = ctx
        .proj_builder
        .id_for_layer(layer_name)
        .ok_or_else(|| anyhow::anyhow!("projection schema missing '{layer_name}' layer"))?;
    // Append-only ingest: a hidden conversation targeting this layer is
    // scored/selected self-local (belief groups masked to the fork's own
    // timeline) so its answer is grounded in its own file, not derailed by
    // cross-file retrieval.
    ctx.engine.lock().unwrap().mark_layer_append_only(layer);

    let (per_file, state) = scan_workspace(workspace, &map);
    progress.set_step_progress(0, per_file.len() as u64);

    // Dedup against already-ingested content so re-uploading identical bytes is
    // a no-op — but NO `reconcile_deleted`: a partial file set must never
    // tombstone the rest of the corpus.
    let present_hashes = ctx
        .engine
        .lock()
        .unwrap()
        .conversation_metadata_values("content_sha256");

    let n_failed = run_file_pool(
        ctx,
        base,
        &per_file,
        &present_hashes,
        progress,
        parallelism(),
    )?;
    Ok((state, n_failed))
}

/// Ingest one file for the priming chain (`crate::priming_chain`) — the same
/// scan/resume-cache/tag machinery [`process_one_file`] always uses, but
/// called directly for one unit instead of through the worker pool, so the
/// caller can pin exactly which conversation it adopts before this file's
/// own reading starts. `None` when `rel_path` isn't a recognised code
/// language, or the size guard / binary sniff drops it — nothing to chain.
pub(crate) fn ingest_chain_file(
    ctx: &RefreshContext<'_>,
    workspace: &Path,
    rel_path: &str,
    predecessor: TimelineId,
    base: &Mutex<Sequence>,
) -> anyhow::Result<Option<TimelineId>> {
    let ext = rel_path
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    let Some(language) = Language::from_extension(&ext) else {
        return Ok(None);
    };
    let mut map = RepoMap::default();
    map.files.push(FileEntry {
        path: rel_path.to_string(),
        line_count: 0,
        language,
        size_bytes: 0,
        module_hint: None,
    });
    let (per_file, _state) = scan_workspace(workspace, &map);
    let Some((file, file_hash)) = per_file.into_iter().next() else {
        return Ok(None);
    };
    let present_hashes = ctx
        .engine
        .lock()
        .unwrap()
        .conversation_metadata_values("content_sha256");
    // This link's parent is the chain so far, not whatever the daemon-wide
    // chain end will be once it is finished being built.
    let link_ctx = RefreshContext {
        priming_chain_end: Some(predecessor),
        ..ctx.clone()
    };
    if !present_hashes.contains(&file_hash) {
        let failures = Failures::new();
        process_one_file(
            &link_ctx,
            base,
            &file,
            &file_hash,
            &present_hashes,
            &failures,
        )?;
        let report = failures.into_report(1);
        if report.is_incomplete() {
            anyhow::bail!(
                "priming chain: {rel_path} ingest failed: {}",
                report
                    .failures
                    .first()
                    .map(|f| f.error.as_str())
                    .unwrap_or("unknown")
            );
        }
    }
    let e = ctx.engine.lock().unwrap();
    let found = e
        .find_conversations_by_metadata("path", rel_path)
        .into_iter()
        .find(|tl| {
            e.conversation_metadata(*tl)
                .is_some_and(|m| m.contains_key("content_sha256"))
        });
    // Idempotent on the fresh-build path (already recorded inside
    // `process_one_file`). The path that NEEDS it here is a resume-cache hit,
    // whose conversation was built in a prior run: its turns are reused as
    // they are, but the chain it hangs off is re-asserted every boot, so a
    // reordered or newly-present anchor still lands correctly.
    if let Some(tl) = found {
        e.set_forked_from(tl, predecessor)
            .map_err(|err| anyhow::anyhow!("priming chain: {rel_path} parent: {err}"))?;
    }
    Ok(found)
}

/// Whether a workspace-relative `path` (with `/` separators) lives under the
/// daemon's top-level `uploads/` dir. Matched on the FIRST segment only, and
/// case-insensitively (the win32 FS is case-insensitive, so an existing
/// `Uploads/` dir still resolves to the daemon's uploads dir) — so a nested
/// `src/uploads/…` in a real project is NOT matched. Keeps `reconcile_deleted`
/// in step with [`crate::repo_scan::walk_workspace`]'s uploads exclusion:
/// uploads are endpoint-managed and deliberately absent from the walk, so they
/// must never be tombstoned merely for being absent from `present_paths`.
pub(crate) fn is_upload_path(path: &str) -> bool {
    path.split('/')
        .next()
        .unwrap_or("")
        .eq_ignore_ascii_case("uploads")
}

/// Tombstone every live `code_read` conversation whose `path` is no longer
/// present in `present_paths`. Covers files deleted while the daemon was
/// down (the startup ingest only visits files that still exist) and files
/// removed between fs-watcher refreshes. Still-present *changed* files are
/// handled by [`process_one_file`], which tombstones a path's stale
/// conversation before re-ingesting it.
///
/// A path past `map`'s `--max-depth` bound is FROZEN, not deleted: the walk
/// never looked there, so its absence from `present_paths` proves nothing.
fn reconcile_deleted(
    engine: &Mutex<ConversationEngine>,
    map: &RepoMap,
    present_paths: &HashSet<&str>,
) {
    let e = engine.lock().unwrap();
    for (tl, path) in e.conversations_with_metadata_key("path") {
        // Uploaded files live under the endpoint-managed `uploads/` dir, which
        // `walk_workspace` deliberately skips — so they're always absent from
        // `present_paths`. Never tombstone them here; that would delete
        // freshly-uploaded content on the next workspace refresh.
        if is_upload_path(&path) || map.is_frozen_file(&path) {
            continue;
        }
        if !present_paths.contains(path.as_str()) {
            if let Err(err) = e.tombstone_timeline(tl) {
                tracing::warn!(
                    target: "zend::code_read::ingest",
                    path = %path,
                    "tombstone of deleted file's conversation failed: {err:#}",
                );
            }
        }
    }
}

/// Retire every crashed-partial `code_read` conversation, up front.
///
/// The `code_read` twin of `repo_scan::retire_crashed_partials`, and the same
/// completion protocol: `path` is written at conversation creation and
/// `content_sha256` only once the file's ingest succeeds, so `path` without a
/// hash means "started, never committed". [`process_one_file`] already retires
/// one such partial per path, but only when that path comes back through the
/// pool — so with `--skip-layer code_reading`, an aborted pass, or the failure
/// cap tripped, the debris stays live and keeps competing in the provenance
/// gather with turns whose answer was never decoded.
///
/// **Uploads are exempt, for the reason [`reconcile_deleted`] exempts them.**
/// They live under the endpoint-managed `uploads/` dir that `walk_workspace`
/// skips, so they never come back through the pool to be re-ingested — and
/// tombstoning one here would delete freshly-uploaded content with nothing to
/// rebuild it from. `reconcile_deleted` guards against exactly this and the
/// guard has to travel with the second sweep.
///
/// Called ONCE per boot from the session's ingest pre-loop, for every layer not
/// named by `--disable-layer` — including a `--skip-layer` layer, which runs no
/// pass and so would otherwise never sweep. Never called from
/// [`refresh_code_reading`]: a refresh can overlap a live pool, and an in-flight
/// file is indistinguishable from a crashed one by metadata alone.
pub(crate) fn retire_crashed_partials(engine: &Mutex<ConversationEngine>) {
    let e = engine.lock().unwrap();
    let mut retired = 0usize;
    for (tl, path) in e.conversations_with_metadata_key("path") {
        if is_upload_path(&path) {
            continue;
        }
        let committed = e
            .conversation_metadata(tl)
            .is_some_and(|m| m.contains_key("content_sha256"));
        if committed {
            continue;
        }
        match e.tombstone_timeline(tl) {
            Ok(()) => retired += 1,
            Err(err) => tracing::warn!(
                target: "zend::code_read::ingest",
                path = %path,
                "tombstone of crashed-partial conversation failed: {err:#}",
            ),
        }
    }
    if retired > 0 {
        tracing::info!(
            target: "zend::code_read::ingest",
            retired,
            "retired crashed-partial code_read conversations (no content hash) \
             so their half-built chains leave the provenance gather",
        );
    }
}

/// Tombstone EVERY `code_reading` conversation, committed or not (uploads
/// exempt, for the reason [`retire_crashed_partials`] exempts them) —
/// `--wipe-layer code_reading`'s targeted counterpart to
/// [`retire_crashed_partials`], which only removes the never-committed half.
///
/// Called from the session's ingest pre-loop, before the registry is seeded
/// from the substrate (`code_read_state_from_substrate`) — so once this
/// returns, that seed is empty and the background ingest worker's first pass
/// reads every file as new, exactly as it would on a truly fresh install.
/// Unlike `--wipe-substrate`, every other layer's content survives untouched.
pub(crate) fn wipe_layer(engine: &Mutex<ConversationEngine>) {
    let e = engine.lock().unwrap();
    let mut wiped = 0usize;
    for (tl, path) in e.conversations_with_metadata_key("path") {
        if is_upload_path(&path) {
            continue;
        }
        match e.tombstone_timeline(tl) {
            Ok(()) => wiped += 1,
            Err(err) => tracing::warn!(
                target: "zend::code_read::ingest",
                path = %path,
                "--wipe-layer code_reading: tombstone failed: {err:#}",
            ),
        }
    }
    tracing::info!(
        target: "zend::code_read::ingest",
        wiped,
        "--wipe-layer code_reading: every code_reading conversation tombstoned \
         (uploads kept) — the next pass re-ingests the whole layer",
    );
}

/// Drive a bounded worker pool over `per_file`: each worker pulls the
/// next file from a shared cursor and runs [`process_one_file`]. Workers
/// share progress / decode-failure counters and an abort flag (first
/// error stops the rest). Returns once every file is processed, yielding
/// the number of files whose ingest was *tolerated-failed* (e.g. the GPU
/// ran out of KV VRAM mid-decode) — so the upload can surface a real
/// failure instead of a silent "done".
///
/// The sole caller of this pool: [`refresh_code_reading`] (the background
/// worker's whole-workspace pass) and [`ingest_files`] (the upload path's
/// bounded file set) both funnel through here, so both drive the same
/// `crate::ingest_backlog` counter and get this same logging.
fn run_file_pool(
    ctx: &RefreshContext<'_>,
    base: &Mutex<Sequence>,
    per_file: &[QueuedFile],
    present_hashes: &HashSet<String>,
    progress: &Arc<LoadProgress>,
    n_workers: usize,
) -> anyhow::Result<usize> {
    let total = per_file.len();
    tracing::info!(
        n_workers = n_workers,
        n_files = total,
        n_cached = present_hashes.len(),
        "code_read: per-file ingest across {n_workers} file workers; each file is \
         a real hidden conversation that reads the file via file_read and answers \
         with its own summary",
    );

    // The GUI's merged background-ingest bar counts only files that will
    // REALLY run — a resume-cache hit is not backlog. Same snapshot every
    // worker probes below, so registration and completion can't disagree.
    let backlog_pending = per_file
        .iter()
        .filter(|(_, h)| !present_hashes.contains(h))
        .count() as u64;
    crate::ingest_backlog::add_pending(backlog_pending);
    let backlog_done = AtomicUsize::new(0);

    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    // Per-file failures are RECORDED, never propagated: the file keeps its prior
    // generation live and the pass carries on, so a bad file (or a systemic VRAM
    // squeeze) degrades the map instead of killing the daemon. The cap still
    // stops a flood, as reported state rather than a fatal error. See
    // `crate::ingest_report`.
    let failures = Failures::new();
    progress.set_step_progress(0, total as u64);

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_workers);
        for _ in 0..n_workers.max(1) {
            handles.push(s.spawn(|| loop {
                // Stop pulling new files on first-error abort OR a shutdown cancel.
                // The current file finishes (the scheduler is still live), so no
                // half-ingested file; the loader thread then drains the engine.
                if failures.aborted() || candle_conversation::ingest_cancelled() {
                    return;
                }
                let idx = cursor.fetch_add(1, Ordering::Relaxed);
                if idx >= per_file.len() {
                    return;
                }
                let (file, fhash) = &per_file[idx];
                if let Err(e) = process_one_file(ctx, base, file, fhash, present_hashes, &failures)
                {
                    // An error escaping `process_one_file` is an unexpected one
                    // (its own failure mode records and returns Ok). Record it
                    // so it reaches the report instead of vanishing, and let the
                    // cap decide whether to stop the pass.
                    let n = failures.record(&file.path, format!("{e:#}"));
                    tracing::warn!(
                        target: "zend::code_read::ingest",
                        file = %file.path,
                        "file ingest failed (will retry next run): {e:#}",
                    );
                    if n > MAX_DECODE_FAILURES {
                        failures.set_abort();
                    }
                }
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                progress.set_step_progress(d as u64, total as u64);
                // A registered file is "done" on every exit from
                // `process_one_file` above — success, tolerated failure, or a
                // mid-file shutdown cancel — so the backlog can never wedge
                // non-empty on a file that will just be retried next pass.
                if !present_hashes.contains(fhash) {
                    backlog_done.fetch_add(1, Ordering::Relaxed);
                    crate::ingest_backlog::item_done(&file.path);
                }
            }));
        }
        for h in handles {
            h.join().expect("code_read worker panicked");
        }
    });

    // An abort or a shutdown cancel can leave files claimed but never run —
    // hand them back, or the GUI's backlog bar never reaches zero.
    crate::ingest_backlog::drop_pending(
        backlog_pending.saturating_sub(backlog_done.load(Ordering::Relaxed) as u64),
    );

    // Reconcile the final progress to exactly `total`. Workers store
    // `set_step_progress` from their own `done` snapshot without a max, so
    // under the pool the last stored value can settle a step short even though
    // every unit ran; pin it to 100% now that the pool has fully drained.
    progress.set_step_progress(total as u64, total as u64);
    let report = failures.into_report(total);
    let n_failed = report.n_failed;
    // Say "incomplete" when it is incomplete, and carry the CAUSE — the old
    // line said "complete" with the count as a field, so a partial pass read as
    // success and every diagnosis started by scrolling back through warnings.
    if report.is_incomplete() {
        tracing::error!(
            target: "zend::code_read::ingest",
            n_files = total,
            n_failed,
            aborted = report.aborted,
            first_failure = report.failures.first().map(|f| f.error.as_str()).unwrap_or("-"),
            first_failure_file = report.failures.first().map(|f| f.unit.as_str()).unwrap_or("-"),
            "code_read per-file ingest INCOMPLETE — affected files keep their prior \
             generation and retry next pass (GET /v1/repo_map)",
        );
    } else {
        tracing::info!(n_files = total, "code_read per-file ingest complete");
    }
    crate::ingest_report::publish(PASS_NAME, report);
    Ok(n_failed)
}

/// Ingest one file into a fresh per-file conversation: skip via the
/// resume-cache snapshot if its content hash is already present; otherwise
/// mint the conversation, run it as a real tool-using exchange
/// ([`run_file_conversation`]), tag it with its content hash + metadata, then
/// drop it (freeing the GPU slot; the sealed turns + tags persist in the
/// substrate).
fn process_one_file(
    ctx: &RefreshContext<'_>,
    base: &Mutex<Sequence>,
    file: &FileEntry,
    file_hash: &str,
    present_hashes: &HashSet<String>,
    failures: &Failures,
) -> anyhow::Result<()> {
    // Resume cache: this content hash was already in the (live, non-
    // tombstoned) substrate at ingest start — skip the read+decode.
    if present_hashes.contains(file_hash) {
        tracing::debug!(
            target: "zend::code_read::ingest",
            file = %file.path,
            "skip: file already in substrate (resume cache hit)",
        );
        return Ok(());
    }

    // Cache miss → new / changed / crashed-partial file. Reconcile the existing
    // conversations for this path WITHOUT invalidating good content up front — a
    // DEFERRED tombstone:
    //   * a PARTIAL (has `path` but no `content_sha256` — a crashed/failed prior
    //     attempt) carries nothing to lose, so tombstone it now; and
    //   * a GOOD generation (has `content_sha256`) is DEFERRED into `superseded`:
    //     it stays live as the file's fallback content, and its resume hash stays
    //     in the cache, so a failed re-ingest below (e.g. a VRAM OOM) leaves the
    //     prior generation intact instead of destroying it. Its tombstone
    //     ACTIVATES only after this ingest commits its own `content_sha256` (see
    //     the success path below) — an atomic swap, "stale-but-present" over
    //     "gone", mirroring the repo_map refresh's keep-old-until-new-ready.
    // The engine lock covers only these quick ops and is released before the
    // decode-heavy body below.
    let (mut conv, superseded) = {
        let e = ctx.engine.lock().unwrap();
        let mut superseded = Vec::new();
        for tl in e.find_conversations_by_metadata("path", &file.path) {
            let is_good = e
                .conversation_metadata(tl)
                .is_some_and(|m| m.contains_key("content_sha256"));
            if is_good {
                superseded.push(tl);
            } else if let Err(err) = e.tombstone_timeline(tl) {
                tracing::warn!(
                    target: "zend::code_read::ingest",
                    file = %file.path,
                    "tombstone of stale partial conversation failed: {err:#}",
                );
            }
        }
        // Forks off `base` — this layer's prefilled template, `base_conv`'s
        // exact counterpart for ingestion (see `InferenceState::ingest_bases`)
        // — so this conversation shares the SAME already-computed prefix a
        // live dialogue turn forks from, rather than this pool's workers each
        // independently re-running the schema's "eager section ingestion"
        // under concurrent load. That used to be this function's own
        // `new_conversation_with_projection` call, on the strength of a
        // comment claiming the two constructions were equivalent because they
        // shared the same prompt text and config — they were not: a live
        // conversation was measured losing track of its own turns after being
        // built this way, and stopped doing so once it forked `base_conv`
        // like every dialogue turn already did.
        drop(e);
        let conv = base
            .lock()
            .unwrap()
            .fork()
            .map_err(|err| anyhow::anyhow!("code_reading conv create: {err}"))?;
        // Record the priming chain as this conversation's parent BEFORE its
        // own reading starts, so the projection for every turn below already
        // carries the anchor documents (`Substrate::inherited_chain`). Nothing
        // is copied and the parent needs no residency of its own: an ancestor
        // sitting warm or cold is elevated by the ordinary projection
        // working-set path when it is selected.
        {
            let e = ctx.engine.lock().unwrap();
            if let Some(parent) = ctx.priming_chain_end {
                e.set_forked_from(conv.timeline_id(), parent)
                    .map_err(|err| anyhow::anyhow!("code_reading priming-chain parent: {err}"))?;
            }
            e.set_timeline_summarize(conv.timeline_id(), false);
        }
        // Now that the lineage is on record, take the recurrent memory of the
        // conversation this one continues — the slot was seeded from its own
        // (empty) timeline when the fork returned.
        //
        // OUTSIDE the engine lock: this waits on a scheduler round-trip, and
        // every other worker in the pool wants that lock for its own mint.
        if ctx.priming_chain_end.is_some() {
            if let Err(err) = conv.seed_recurrent_from_lineage() {
                tracing::warn!(
                    target: "zend::code_read::ingest",
                    file = %file.path,
                    "seeding recurrent memory from the priming chain failed: {err:#}",
                );
            }
        }
        (conv, superseded)
    };

    // Tag the `path` IMMEDIATELY — before the decode-heavy run below that can
    // fail (GPU OOM mid-decode, a decode error). A partial left by such a failure
    // then still carries its path, so it (a) shows in the substrate as the file it
    // covers rather than "(untitled)", and (b) is found by the path-invalidation
    // scan above on the next run, which tombstones it and retries the file. The
    // resume-cache key (`content_sha256`) is deliberately withheld until success
    // (below), so a partial is never mistaken for a completed ingest and skipped.
    {
        let mut early = BTreeMap::new();
        early.insert("kind".to_string(), "code_read".to_string());
        early.insert("path".to_string(), file.path.clone());
        if let Err(e) = conv.set_metadata_many(&early) {
            tracing::warn!(
                target: "zend::code_read::ingest",
                file = %file.path,
                "failed to tag path metadata at conversation creation: {e:#}",
            );
        }
    }

    let summary = match run_file_conversation(
        &mut conv,
        &file.path,
        &ctx.think_triggers,
        &ctx.tool_ctx,
    ) {
        Ok(text) => text,
        Err(e) => {
            // The deferred tombstone is the safety net here: the prior good
            // generation in `superseded` was NEVER tombstoned, so it stays
            // live as the file's content and its resume hash stays in the
            // cache — this failed attempt invalidates nothing. Drop only
            // THIS attempt's partial; the retry re-mints cleanly.
            {
                let e2 = ctx.engine.lock().unwrap();
                if let Err(err) = e2.tombstone_timeline(conv.timeline_id()) {
                    tracing::warn!(
                        target: "zend::code_read::ingest",
                        file = %file.path,
                        "tombstone of failed-attempt partial failed: {err:#}",
                    );
                }
            }
            // If a graceful shutdown latched the cancel flag, this `Err` is the
            // interruptible decode-wait unwinding (`wait_cancellable` →
            // `IngestCancelled`), not a genuine decode failure — the anyhow layer
            // has erased the variant, so the global flag is the source of truth.
            // Don't record it against the failure cap; the partial was just
            // tombstoned, so the file re-ingests next run.
            if candle_conversation::ingest_cancelled() {
                tracing::debug!(
                    target: "zend::code_read::ingest",
                    file = %file.path,
                    "shutdown cancelled decode mid-file — dropped partial; will re-ingest next run",
                );
                return Ok(());
            }
            let n = failures.record(&file.path, format!("{e:#}"));
            tracing::warn!(
                target: "zend::code_read::ingest",
                file = %file.path,
                superseded_kept = superseded.len(),
                "file ingest failed (will retry next run; prior generation kept live): {e:#}",
            );
            if n > MAX_DECODE_FAILURES {
                tracing::error!(
                    target: "zend::code_read::ingest",
                    n, cap = MAX_DECODE_FAILURES,
                    "code_read ingest stopping early: failure cap reached (last: {e:#})",
                );
                failures.set_abort();
            }
            return Ok(()); // conv drops → slot freed; superseded generation still live.
        }
    };
    tracing::debug!(
        target: "zend::code_read::ingest",
        file = %file.path,
        summary_chars = summary.len(),
        "file conversation answered",
    );

    // Tag the conversation: `content_sha256` is the resume-cache key,
    // `path` is the invalidation-scan key, the rest is diagnostic.
    let mut tags = BTreeMap::new();
    tags.insert("kind".to_string(), "code_read".to_string());
    tags.insert("path".to_string(), file.path.clone());
    tags.insert("content_sha256".to_string(), file_hash.to_string());
    tags.insert("lang".to_string(), format!("{:?}", file.language));
    let committed = match conv.set_metadata_many(&tags) {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(
                target: "zend::code_read::ingest",
                file = %file.path,
                "failed to tag conversation metadata (resume cache): {e:#}",
            );
            false
        }
    };

    // Deferred tombstone ACTIVATES — but ONLY once the new generation is truly
    // committed (its `content_sha256` landed above). If that tag write failed the
    // replacement isn't resume-cached, so treat it as not-yet-committed and KEEP
    // the prior generation live (exactly as the failure path does) rather than
    // swapping to an untagged replacement.
    if committed && !superseded.is_empty() {
        let e = ctx.engine.lock().unwrap();
        for tl in &superseded {
            if let Err(err) = e.tombstone_timeline(*tl) {
                tracing::warn!(
                    target: "zend::code_read::ingest",
                    file = %file.path,
                    "deferred tombstone of superseded generation failed: {err:#}",
                );
            }
        }
    }

    // The file's conversation is now complete: nothing attends it again until a
    // projection retrieves it. Flag it for full KV eviction so the persistence
    // pipeline offloads its turns to cold (NVMe) and frees BOTH the VRAM and RAM
    // copies — otherwise the sealed turns linger hot and accumulate across a
    // large multi-file ingest until the card fills. `FreeSequence` on drop only
    // releases the batch slot, not the sealed KV, so this proactive flag is what
    // actually reclaims the space; `elevate_to_hot` pulls the file back from cold
    // on demand if reselected.
    let flagged = ctx
        .engine
        .lock()
        .unwrap()
        .evict_ingest_timeline(conv.timeline_id());
    tracing::debug!(
        target: "zend::code_read::ingest",
        file = %file.path,
        turns = flagged,
        "flagged completed file conversation for full KV eviction to cold",
    );

    // `conv` drops here → FreeSequence releases the GPU slot; the sealed
    // turns and metadata remain in the substrate.
    Ok(())
}

/// Drive `conv` as a REAL, hidden tool-using conversation asking it to read and
/// describe `path` — exactly the machinery a live chat turn uses
/// (`submit_turn_with_options` → real decode → real tool dispatch → repeat, see
/// `zend::session::run_inference_stream`), just run synchronously on this
/// worker thread and never shown to a user. [`opening_prompt`] requires the
/// model to read the whole file — one or more real `file_read` calls, ranged
/// if the file is long — before it answers; whatever it says once it stops
/// calling tools IS the file's summary — there is no separate summary-decode
/// step to run afterward.
fn run_file_conversation(
    conv: &mut Sequence,
    path: &str,
    triggers: &Arc<TriggerRegistry>,
    tool_ctx: &Arc<ToolContext>,
) -> anyhow::Result<String> {
    let tags = vec!["code".to_string(), path.to_string()];
    let mut selection = SelectionState::new();
    selection.set_optional(TOOLS_ENABLED_SELECTOR, OptionalState::Present);
    selection.select(FORCE_TOOL_SELECTOR, FILE_READ_TOOL);
    // Thinking stays ON, at `ThinkMode::Quick` (see `triggers`'s doc on
    // `RefreshContext::think_triggers`) — explicit `Absent` rather than
    // leaving the selector unset, so this isn't quietly riding whatever the
    // schema's own default happens to be.
    selection.set_optional(NO_THINK_SELECTOR, OptionalState::Absent);

    let mut current_message: TurnText = TurnText::from(opening_prompt(path));
    let mut closing = false;
    for round in 0..=MAX_FILE_READ_ROUNDS {
        if candle_conversation::ingest_cancelled() {
            anyhow::bail!("shutdown cancelled mid-file");
        }
        if round == MAX_FILE_READ_ROUNDS {
            tracing::warn!(
                target: "zend::code_read::ingest",
                path,
                "hit the read-round cap — forcing an answer on what has been read so far",
            );
            closing = true;
        }
        // The closing turn is forced to answer in prose: ban the tool-call
        // opener so a model still trying to read more cannot open another
        // call, mirroring the live dialogue loop's own repeat-guard closing
        // round (`session::run_inference_stream`'s `closing` handling).
        let sampling = if closing {
            let mut s = conv.default_sampling();
            let open = s.tool_call_open_token_id;
            if open >= 0 {
                s.banned_tokens.push(open);
            }
            Some(s)
        } else {
            None
        };
        let options = TurnOptions {
            max_tokens: Some(FILE_TURN_MAX_TOKENS),
            sampling,
            selection: selection.clone(),
            triggers: Arc::clone(triggers),
            tags: tags.clone(),
            ..Default::default()
        };
        let handle = conv
            .submit_turn_with_options(current_message, options)
            .map_err(|e| anyhow::anyhow!("submit_turn: {e}"))?;
        let resp = handle
            .wait_cancellable()
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        let steps = tool_round::plan(&resp.text);
        let call_idx = resp.seal.as_ref().and_then(|s| s.turn_index);
        conv.finish_turn(handle, &resp)
            .map_err(|e| anyhow::anyhow!("finish_turn: {e}"))?;

        if steps.is_empty() || closing {
            return Ok(resp.text);
        }

        let results = tool_round::run(tool_ctx, steps);
        let response = format_tool_responses(&results);
        // A tool round that produces no response text must NOT spawn a
        // follow-up turn — see `session::run_inference_stream`'s identical
        // guard. The answer already decoded for this turn stands.
        if response.is_blank() {
            return Ok(resp.text);
        }
        // The round-trip is now certain — the tool returned real output and the
        // follow-up turn is guaranteed to be submitted below — so couple the
        // just-sealed call turn to it, by its OWN sealed index (never "the last
        // turn": the async summariser could append a summary turn in this
        // window).
        if let Some(idx) = call_idx {
            conv.couple_turn(idx)
                .map_err(|e| anyhow::anyhow!("couple_turn: {e}"))?;
        }
        current_message = response;
    }
    unreachable!("the closing round at MAX_FILE_READ_ROUNDS always returns")
}

/// Outcome of a [`refresh_code_reading`] call. `Replaced` carries only
/// the new content-hash `state` — per-file conversations are freed after
/// seal and persist in the substrate, so there's no sequence list to swap.
pub enum RefreshOutcome {
    NoOp,
    Replaced {
        /// The merged per-file content-hash record after the refresh.
        /// No live sequences: per-file conversations are freed after seal
        /// and live in the substrate, so the caller just swaps in `state`.
        state: CodeReadState,
    },
}

/// Sole entry point for `code_reading` ingestion — startup's first pass,
/// every later filesystem-event-triggered pass, and (once seeded) a totally
/// fresh install all call this the same way. `prior` comes from
/// [`code_read_state_from_substrate`], so an empty prior (nothing durable
/// yet) makes every file read as changed — exactly the first-ever-boot
/// behavior, with no separate "ingest" variant needed.
///
/// Re-scans `map`. Returns `NoOp` when no file hash changed. Otherwise
/// `reconcile_deleted` tombstones conversations for files now gone, and the
/// pool re-ingests over all files — unchanged files hit the resume-cache
/// snapshot and are skipped, while a changed file misses the snapshot,
/// tombstones its stale conversation, and re-ingests. So only changed/added
/// files actually re-run.
///
/// The layer's append-only mark is NOT taken here: the session's ingest
/// pre-loop marks every enabled, non-skipped layer from the same builder
/// before any pool can start, for every layer and every restart alike (see
/// `session.rs`) — a second marking site here would be exactly the
/// duplicate path this repo's engineering rules forbid, and the mark's
/// absence was once responsible for ingest-layer scores going un-normalized
/// by a ~13,000x factor.
///
/// The engine mutex is taken only for the quick create/tombstone ops inside
/// the pool (released across each decode), so chat consumers keep running.
pub fn refresh_code_reading(
    ctx: &RefreshContext<'_>,
    workspace: &Path,
    map: &RepoMap,
    prior: &CodeReadState,
    progress: &Arc<LoadProgress>,
    base: &Mutex<Sequence>,
) -> anyhow::Result<RefreshOutcome> {
    // Scan once — drives both the change comparison and the re-ingest.
    let (per_file, next) = scan_workspace(workspace, map);
    let prior = prior.without_frozen(map);
    if prior.equivalent_to(&next) {
        tracing::debug!("code_read refresh: no file hash changed, skipping refresh");
        return Ok(RefreshOutcome::NoOp);
    }

    let changed = prior.changed_files(&next);
    tracing::info!(
        n_changed = changed.len(),
        sample_changed = ?changed.iter().take(5).collect::<Vec<_>>(),
        "code_read refresh: reconciling + re-ingesting changed files",
    );

    let n_workers = parallelism();

    // Tombstone conversations for deleted files, then snapshot surviving
    // hashes; changed files miss the snapshot and are re-ingested (their
    // stale conversation is tombstoned in process_one_file).
    let present_paths: HashSet<&str> = per_file.iter().map(|(f, _)| f.path.as_str()).collect();
    reconcile_deleted(ctx.engine, map, &present_paths);
    let present_hashes = ctx
        .engine
        .lock()
        .unwrap()
        .conversation_metadata_values("content_sha256");

    run_file_pool(ctx, base, &per_file, &present_hashes, progress, n_workers)?;

    Ok(RefreshOutcome::Replaced { state: next })
}

/// Byte offset of the start of each line.  `offsets[i]` is the start
/// of line `i + 1` (1-indexed).  Final entry is the source length.
///
/// Kept for [`crate::repo_scan::anchor`], which slices a folder's anchor
/// excerpt the same way — no longer used inside this module (the model reads
/// files for itself now, rather than a pre-sliced excerpt being shown to it).
pub(crate) fn compute_line_offsets(bytes: &[u8]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(bytes.len() / 40 + 1);
    offsets.push(0);
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
            offsets.push(i + 1);
        }
    }
    if offsets.last().copied() != Some(bytes.len()) {
        offsets.push(bytes.len());
    }
    offsets
}

pub(crate) fn slice_lines(
    bytes: &[u8],
    offsets: &[usize],
    start_line: u32,
    end_line: u32,
) -> String {
    // 1-indexed inclusive.  Last entry of `offsets` is bytes.len().
    let lines_total = offsets.len().saturating_sub(1) as u32;
    if lines_total == 0 || start_line > lines_total {
        return String::new();
    }
    let start_idx = (start_line as usize - 1).min(offsets.len() - 1);
    let end_idx = (end_line as usize).min(offsets.len() - 1);
    let start_byte = offsets[start_idx];
    let end_byte = offsets[end_idx];
    String::from_utf8_lossy(&bytes[start_byte..end_byte]).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With `--max-depth 2`, a file past the bound (`src/deep/c.rs`) is frozen:
    /// it leaves the state the refresh compares, so a bounded walk that never
    /// visits it cannot read it as removed. Unbounded, the state is untouched.
    #[test]
    fn a_frozen_file_leaves_the_compared_state() {
        let state = |pairs: &[(&str, &str)]| CodeReadState {
            file_hashes: pairs
                .iter()
                .map(|(path, hash)| (path.to_string(), hash.to_string()))
                .collect(),
        };
        let prior = state(&[("a.rs", "1"), ("src/b.rs", "2"), ("src/deep/c.rs", "3")]);
        let bounded = RepoMap {
            max_depth: Some(2),
            ..RepoMap::default()
        };
        assert_eq!(
            prior.without_frozen(&bounded),
            state(&[("a.rs", "1"), ("src/b.rs", "2")])
        );
        assert_eq!(prior.without_frozen(&RepoMap::default()), prior);
    }

    #[test]
    fn is_upload_path_matches_top_level_uploads_only() {
        // Top-level uploads/ (any case, since the win32 FS is case-insensitive).
        assert!(is_upload_path("uploads"));
        assert!(is_upload_path("uploads/notes.py"));
        assert!(is_upload_path("Uploads/notes.py"));
        assert!(is_upload_path("UPLOADS/a.rs"));
        // NOT a nested source dir, nor a lookalike.
        assert!(!is_upload_path("src/uploads/real.rs"));
        assert!(!is_upload_path("uploadsx/a.py"));
        assert!(!is_upload_path("docs/uploads.md"));
        assert!(!is_upload_path("src/main.rs"));
    }

    #[test]
    fn file_content_hash_deterministic_path_and_content_sensitive() {
        let h = file_content_hash("src/a.rs", b"fn x() {}");
        // Deterministic.
        assert_eq!(h, file_content_hash("src/a.rs", b"fn x() {}"));
        // SHA-256 hex.
        assert_eq!(h.len(), 64);
        // Content edit → different hash.
        assert_ne!(h, file_content_hash("src/a.rs", b"fn y() {}"));
        // Path-qualified: same content at a different path → different hash
        // (so a move/rename re-ingests, and per-path invalidation is exact).
        assert_ne!(h, file_content_hash("src/b.rs", b"fn x() {}"));
    }

    #[test]
    fn slice_lines_returns_exact_line_range() {
        let src = b"alpha\nbeta\ngamma\ndelta\n";
        let offsets = compute_line_offsets(src);
        let s = slice_lines(src, &offsets, 2, 3);
        assert_eq!(s, "beta\ngamma\n");
    }

    #[test]
    fn slice_lines_handles_no_trailing_newline() {
        let src = b"alpha\nbeta\ngamma";
        let offsets = compute_line_offsets(src);
        let s = slice_lines(src, &offsets, 2, 3);
        assert_eq!(s, "beta\ngamma");
    }

    #[test]
    fn slice_lines_clips_at_eof() {
        let src = b"alpha\nbeta\n";
        let offsets = compute_line_offsets(src);
        let s = slice_lines(src, &offsets, 1, 100);
        assert_eq!(s, "alpha\nbeta\n");
    }

    #[test]
    fn opening_prompt_names_the_path_and_the_one_tool() {
        let p = opening_prompt("src/lib.rs");
        assert!(p.contains("src/lib.rs"));
        assert!(p.contains(FILE_READ_TOOL));
    }

    /// The measured failure this prompt exists to close: left as "read as much
    /// as you need", a model sometimes skipped `file_read` entirely and
    /// guessed a summary from the filename alone. The prompt must require
    /// reading the file, not merely permit it.
    #[test]
    fn opening_prompt_requires_reading_the_whole_file() {
        let p = opening_prompt("CHANGELOG.md");
        assert!(
            p.contains("entire") || p.contains("whole"),
            "must ask for the whole file, not an unspecified amount: {p:?}",
        );
        assert!(
            p.contains("only this file"),
            "must scope the read to this file alone: {p:?}",
        );
    }
}
