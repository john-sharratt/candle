//! `code_reading` layer ingestion.
//!
//! Each file in the [`crate::repo_scan::RepoMap`] becomes ONE
//! conversation on the `code_reading` layer.  The file is parsed into
//! scope-aware parts; each part contributes a prefilled `read_file`
//! request, a prefilled `<tool_call>` echo, and a prefilled
//! `<tool_response>` carrying the source with line numbers.  The
//! whole-file summary is not decoded inline — it is the root of the
//! async summary tree the summariser rolls up over these scope turns.
//!
//! Refresh is per-file: content hashes ([`CodeReadState`]) decide
//! which files changed; deleted files' conversations are tombstoned,
//! changed files are re-ingested, and unchanged files are skipped via
//! the substrate resume cache (the per-file `content_sha256` tag).
//!
//! **Parallel ingest.**  At the candle workspace's tens of thousands
//! of files, a single-session ingest would run for tens of hours.
//! Instead [`CODE_READ_PARALLELISM`] workers each process whole files
//! concurrently, minting a distinct per-file conversation per file.
//! Within a file, scopes are ingested SERIALLY as two-coupled-turn tool
//! round-trips ([`Sequence::ingest_scope_roundtrip`]) — the response
//! turn's summary must decode with the call turn in its projected
//! prefix, so the two halves can't batch independently. The parallelism
//! is therefore across FILES: each concurrent worker contributes one
//! sequence to the engine's shared multi-session batching (its scope
//! decodes coalesce with every other worker's into one forward), which
//! amortises the MoE expert-weight load. The resolver's
//! `active_timelines_for_group` iterator surfaces every file's timeline
//! to dialogue retrieval without code changes.

pub mod carve;
pub mod header;
pub mod parsers;
#[cfg(test)]
pub mod test_util;
pub mod types;

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::projection::{Builder, GroupId, LayerId, SystemPromptItem};
use candle_conversation::{ConversationEngine, SequenceConfig};
use sha2::{Digest, Sha256};

use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::repo_scan::{utility_config, FileEntry, Language, RepoMap};
use crate::turn_sink::{InsertTurnSink, SequenceTurnSink};

pub use types::Scope;

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
}

/// Max tokens for a scope's decoded two-sentence summary (the response turn).
const SCOPE_SUMMARY_MAX_TOKENS: usize = 100;

/// Emit one file's scopes into `sink` — each scope as a TOOL ROUND-TRIP of two
/// coupled turns:
///   Turn A (call):     user("Summarize `path` (lines a-b)…") → assistant(<tool_call>)
///   Turn B (response):  user(<tool_response> with the source) → assistant(DECODED summary)
///
/// Recording each scope as two coupled turns (rather than one baked four-segment
/// blob) is what keeps the inter-turn role seams as REGENERATED glue — a seam
/// baked into K/V goes stale when the scope is re-injected at a different
/// position mid-dialogue. The `read_file` tool round-trip teaches the model the
/// "use a tool, read the response, answer" pattern, and the decoded summary
/// anchors the scope for provenance. Shared by the production per-file ingest
/// ([`process_one_file`]) and the sink-driven reference/test path
/// ([`ingest_code_reading_into_sink`]).
fn emit_file_turns<S: InsertTurnSink>(
    sink: &mut S,
    path: &str,
    language: Language,
    scopes: &[Scope],
    bytes: &[u8],
    on_prefilled: crate::turn_sink::ScopeProgressFn,
) -> anyhow::Result<()> {
    let line_offsets = compute_line_offsets(bytes);
    // Gather-scope tags `["code", <path>]` scope every turn into a code-tagged
    // provenance gallery (and out of the untagged dialogue partition).
    let tags = vec!["code".to_string(), path.to_string()];
    // Render every scope's tool round-trip up front, in file order. The sink
    // ingests them — the production sink forks each onto its own timeline and
    // runs the two-turn round-trips CONCURRENTLY (co-batched on the wave engine),
    // then splices the sealed pairs back onto this file's timeline in order; the
    // model-less test sink runs them serially. Either way the inter-turn seams
    // stay regenerated live glue (see `Sequence::ingest_scope_roundtrip_indices`).
    let prepared: Vec<(String, String, String)> = scopes
        .iter()
        .map(|scope| {
            let body = slice_lines(bytes, &line_offsets, scope.start_line, scope.end_line);
            (
                header::render_part_user_prompt(path, scope),
                header::render_tool_call(path, scope),
                header::render_tool_response(path, scope, language, &body),
            )
        })
        .collect();
    sink.ingest_scopes(prepared, tags, SCOPE_SUMMARY_MAX_TOKENS, &on_prefilled)
}

/// Progress units one file contributes: one per carved scope (prefill). The
/// file summary is no longer decoded inline — it is the async summary tree's
/// root, tracked separately — so a file's ingest units are exactly its scopes.
/// Keeps the `total` fed to [`LoadProgress`] in step with the
/// [`emit_file_turns`] `on_prefill` callbacks.
fn file_progress_units(scopes: &[Scope]) -> usize {
    scopes.len()
}

/// Sink-driven reference for the per-file `code_reading` ingest — carves
/// every file in `map` and drives [`emit_file_turns`] into `sink`. Returns
/// the per-file content-hash record and the number of parts emitted.
///
/// Kept for the integration test harness in
/// `tests/code_read_integration.rs`, which exercises the carve + per-file
/// emission against the [`RecordingTurnSink`] without a live engine. The
/// production ingest is [`ingest_code_reading`] (parallel per-file pool);
/// both shape turns through the shared [`emit_file_turns`].
///
/// `dead_code`: the lint fires for the `bin` target because nothing inside
/// the crate calls it (integration tests link as a separate crate).
#[allow(dead_code)]
pub fn ingest_code_reading_into_sink<S: InsertTurnSink>(
    sink: &mut S,
    workspace: &Path,
    map: &RepoMap,
    progress: &LoadProgress,
) -> anyhow::Result<(usize, CodeReadState)> {
    let (per_file, state) = carve_workspace(workspace, map);
    // This test/reference path reports and RETURNS the part (scope) count, not
    // the summary-inclusive progress unit — its callers assert one prefill turn
    // per part. (The production ingest bar's per-part + summary accounting lives
    // in `process_one_file`/`file_progress_units`.)
    let total: usize = per_file.iter().map(|(_, s, _, _)| s.len()).sum();
    tracing::info!(
        n_files = per_file.len(),
        n_scopes = total,
        "code_read carve complete"
    );
    progress.set_step_progress(0, total as u64);

    // This reference path drives its own coarse per-file progress below, so the
    // per-scope callback is a no-op.
    let noop: crate::turn_sink::ScopeProgressFn = Arc::new(|_| {});
    let mut done = 0usize;
    for (file, scopes, bytes, _fhash) in &per_file {
        emit_file_turns(
            sink,
            &file.path,
            file.language,
            scopes,
            bytes,
            Arc::clone(&noop),
        )?;
        done += scopes.len();
        progress.set_step_progress(done as u64, total as u64);
    }

    tracing::info!(n_scopes_emitted = done, "code_read prefill complete");
    Ok((done, state))
}

/// Maximum tolerated per-file summary decode failures in a single
/// ingestion pass before the whole refresh aborts.  A single
/// failure can happen for legitimate reasons (scheduler hiccup,
/// transient resource pressure); a cascade signals something
/// systemic.
pub const MAX_DECODE_FAILURES: usize = 16;

/// Number of files ingested concurrently by the worker pool. Each worker owns
/// one file conversation and drives it through [`process_one_file`], ingesting
/// that file's scopes SERIALLY as two-coupled-turn tool round-trips
/// ([`Sequence::ingest_scope_roundtrip`]). So this IS the concurrency knob: it's
/// the number of conversations feeding the engine at once, which is the width
/// the engine's multi-session batching coalesces — every worker blocking on its
/// scope's summary decode contributes one sequence to the shared
/// `batch_decode_step` forward. (Decode width scales with this directly; PREFILL
/// width is separately capped by `Scheduler::MAX_PREFILL_WIDTH` + the AIMD admit
/// window, so raising this past ~24 widens decodes but not prefills.)
///
/// **Invariant:** each worker holds exactly ONE conversation slot (no scratch
/// slots — the old parallel scope-ingest pump is gone), so this must stay under
/// the model's sequence-slot capacity WITH headroom for the non-ingest slots the
/// engine also needs concurrently: the live dialogue session, the async
/// summariser's compression passes, etc.
///
/// The engine is expected to absorb this many concurrent conversations and
/// manage VRAM itself (evict the right cold KV, keep the working set hot), so
/// this is a concurrency target, not a VRAM-safety valve — raising it should let
/// the engine batch wider, not thrash.
///
/// Each file worker now parallelises its OWN scopes
/// ([`crate::turn_sink::SCOPE_PARALLELISM`]), so the concurrent conversation
/// count is `CODE_READ_PARALLELISM × SCOPE_PARALLELISM`. Keep the product near the
/// engine's sequence-slot budget: 12 files × 4 scopes = 48, matching the prior
/// file-only concurrency while adding within-file (large-file) parallelism.
pub const CODE_READ_PARALLELISM: usize = 12;

/// [`utility_config`] specialised for the `code_reading` layer: append-only
/// (no reprojection), inheriting the utility C5 compression level.
///
/// Both K and V are adaptively quantized at C5 (the engine-wide uniform-K pin
/// is off in this config).
fn code_read_config(config: SequenceConfig) -> SequenceConfig {
    utility_config(config)
}

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
fn file_content_hash(path: &str, bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(path.as_bytes());
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// Carve `map`'s files into `(file, scopes, bytes, content_hash)` tuples,
/// recording each file's hash into a fresh [`CodeReadState`]. Each file is
/// hashed exactly once. Files that carve to no scopes are skipped.
fn carve_workspace(
    workspace: &Path,
    map: &RepoMap,
) -> (Vec<(FileEntry, Vec<Scope>, Vec<u8>, String)>, CodeReadState) {
    let mut per_file = Vec::with_capacity(map.files.len());
    let mut state = CodeReadState::default();
    for file in &map.files {
        let path = workspace.join(&file.path);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::debug!(file = %file.path, "code_read: skip unreadable file: {e}");
                continue;
            }
        };
        let is_tsx = file.path.ends_with(".tsx");
        let scopes = carve::carve(&bytes, file.language, is_tsx);
        if !scopes.is_empty() {
            let fhash = file_content_hash(&file.path, &bytes);
            state.file_hashes.insert(file.path.clone(), fhash.clone());
            per_file.push((file.clone(), scopes, bytes, fhash));
        }
    }
    (per_file, state)
}

/// Ingest ONLY `rel_paths` into the `code_reading` layer — the upload
/// pipeline's read_file phase.
///
/// Unlike [`ingest_code_reading`] / [`refresh_code_reading`], this does **not**
/// walk or reconcile the whole workspace: it carves and prefills just these
/// files, dedupes against already-ingested identical content, and **never**
/// tombstones anything. That matters for two reasons: (1) a full-workspace
/// re-ingest triggered by one upload is a huge, GPU-overloading amount of work
/// (and with `--skip-code-read` the empty prior state makes the refresh treat
/// *every* file as new — the exact overload that killed the expert pipeline
/// thread); (2) a partial file set fed to the workspace refresh would make
/// `reconcile_deleted` tombstone the entire rest of the corpus. This path is
/// bounded to the uploaded files' scopes and safe under `--skip-code-read`.
///
/// Files whose extension isn't a recognised code language are skipped (there is
/// nothing to read). Returns the per-file content-hash state for the files that
/// were ingested (to merge into the running [`CodeReadState`]) plus the count of
/// files whose ingest tolerated-failed (e.g. out of KV VRAM), so the upload can
/// surface a real failure. Summarisation of the ingested scopes runs entirely in
/// the background summariser and is not awaited here.
#[allow(clippy::too_many_arguments)]
pub fn ingest_files(
    engine: &Mutex<ConversationEngine>,
    proj_builder: &Builder,
    workspace: &Path,
    rel_paths: &[String],
    config: SequenceConfig,
    progress: &Arc<LoadProgress>,
    layer_name: &str,
    group_name: &str,
) -> anyhow::Result<(CodeReadState, usize)> {
    // Build a minimal RepoMap for just these files — `carve_workspace` needs
    // only the path + language; the other `FileEntry` fields are unused by the
    // carve, so they're left at defaults.
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

    let layer = proj_builder
        .id_for_layer(layer_name)
        .ok_or_else(|| anyhow::anyhow!("projection schema missing '{layer_name}' layer"))?;
    let group = proj_builder
        .id_for_group(group_name)
        .ok_or_else(|| anyhow::anyhow!("projection schema missing '{group_name}' group"))?;
    // Append-only ingest: a scope-summary projection targeting this layer is
    // scored/selected self-local (belief groups masked to the fork's own timeline)
    // so the summary is grounded in its own scope, not derailed by cross-file
    // retrieval. The multi-timeline scan stays on for dialogue.
    engine.lock().unwrap().mark_layer_append_only(layer);
    let system_prompt = layer_system_prompt(proj_builder, layer_name, &config);
    let utility_cfg = code_read_config(config);

    let (per_file, state) = carve_workspace(workspace, &map);
    let total: usize = per_file
        .iter()
        .map(|(_, s, _, _)| file_progress_units(s))
        .sum();
    progress.set_step_progress(0, total as u64);

    // Dedup against already-ingested content so re-uploading identical bytes is
    // a no-op — but NO `reconcile_deleted`: a partial file set must never
    // tombstone the rest of the corpus.
    let present_hashes = engine
        .lock()
        .unwrap()
        .conversation_metadata_values("content_sha256");

    let n_failed = run_file_pool(
        engine,
        proj_builder,
        &system_prompt,
        &utility_cfg,
        layer,
        group,
        &per_file,
        &present_hashes,
        total,
        progress,
        parallelism(),
    )?;
    Ok((state, n_failed))
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
fn reconcile_deleted(engine: &Mutex<ConversationEngine>, present_paths: &HashSet<&str>) {
    let e = engine.lock().unwrap();
    for (tl, path) in e.conversations_with_metadata_key("path") {
        // Uploaded files live under the endpoint-managed `uploads/` dir, which
        // `walk_workspace` deliberately skips — so they're always absent from
        // `present_paths`. Never tombstone them here; that would delete
        // freshly-uploaded content on the next workspace refresh.
        if is_upload_path(&path) {
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

/// Top-level `code_reading` ingestion — **one conversation per file**.
///
/// Each file becomes its own `(code_reading, scopes)` conversation: one
/// prefill turn per carved part (read_file tool-call + response) so the
/// model reads the whole file. The whole-file summary is not decoded
/// inline — it is the async summary tree's root, rolled up later by the
/// summariser. The conversation is tagged with a content hash +
/// descriptive metadata, then freed — its sealed turns and metadata
/// persist in the substrate, so retrieval and the restart-resume cache
/// work off the substrate, not a live sequence.
///
/// Reconciliation runs first: conversations for files no longer on disk
/// are tombstoned, then a one-pass snapshot of present content hashes lets
/// each worker skip already-ingested files in O(1). A bounded pool of
/// [`parallelism`] workers pulls files from a shared cursor, so at most
/// that many conversations are live at once (VRAM bound) while the
/// scheduler wave-batches their prefills/decodes. The engine mutex is
/// taken only for the quick create/tombstone ops, never across a decode.
#[allow(clippy::too_many_arguments)]
pub fn ingest_code_reading(
    engine: &Mutex<ConversationEngine>,
    proj_builder: Builder,
    workspace: &Path,
    map: &RepoMap,
    config: SequenceConfig,
    progress: &Arc<LoadProgress>,
    layer_name: &str,
    group_name: &str,
) -> anyhow::Result<CodeReadState> {
    let layer = proj_builder
        .id_for_layer(layer_name)
        .ok_or_else(|| anyhow::anyhow!("projection schema missing '{layer_name}' layer"))?;
    let group = proj_builder
        .id_for_group(group_name)
        .ok_or_else(|| anyhow::anyhow!("projection schema missing '{group_name}' group"))?;
    let system_prompt = layer_system_prompt(&proj_builder, layer_name, &config);
    let utility_cfg = code_read_config(config);
    let n_workers = parallelism();

    let (per_file, state) = carve_workspace(workspace, map);
    let total: usize = per_file
        .iter()
        .map(|(_, s, _, _)| file_progress_units(s))
        .sum();

    // Retire conversations whose source file is gone, then snapshot the
    // surviving content hashes once for O(1) per-file resume-cache probes.
    let present_paths: HashSet<&str> = per_file
        .iter()
        .map(|(f, _, _, _)| f.path.as_str())
        .collect();
    reconcile_deleted(engine, &present_paths);
    let present_hashes = engine
        .lock()
        .unwrap()
        .conversation_metadata_values("content_sha256");

    tracing::info!(
        n_workers = n_workers,
        n_files = per_file.len(),
        n_scopes = total,
        n_cached = present_hashes.len(),
        "code_read: per-file ingest across {n_workers} file workers; each file forks \
         its scopes onto per-scope timelines, runs the two-turn round-trips in \
         parallel (co-batched on the wave), and splices the pairs back in order",
    );
    progress.set_step_progress(0, total as u64);

    run_file_pool(
        engine,
        &proj_builder,
        &system_prompt,
        &utility_cfg,
        layer,
        group,
        &per_file,
        &present_hashes,
        total,
        progress,
        n_workers,
    )?;

    Ok(state)
}

/// Drive a bounded worker pool over `per_file`: each worker pulls the
/// next file from a shared cursor and runs [`process_one_file`]. Workers
/// share progress / decode-failure counters and an abort flag (first
/// error stops the rest). Returns once every file is processed, yielding
/// the number of files whose ingest was *tolerated-failed* (e.g. the GPU
/// ran out of KV VRAM mid-prefill) — so the upload can surface a real
/// failure instead of a silent "done".
#[allow(clippy::too_many_arguments)]
fn run_file_pool(
    engine: &Mutex<ConversationEngine>,
    proj_builder: &Builder,
    system_prompt: &str,
    utility_cfg: &SequenceConfig,
    layer: LayerId,
    group: GroupId,
    per_file: &[(FileEntry, Vec<Scope>, Vec<u8>, String)],
    present_hashes: &HashSet<String>,
    total: usize,
    progress: &Arc<LoadProgress>,
    n_workers: usize,
) -> anyhow::Result<usize> {
    let cursor = AtomicUsize::new(0);
    // `Arc` so per-file `process_one_file` can hand a clone to the scheduler's
    // per-scope progress callback (which must be `'static`).
    let done = Arc::new(AtomicUsize::new(0));
    let decode_failures = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    let first_error: Mutex<Option<anyhow::Error>> = Mutex::new(None);

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_workers);
        for _ in 0..n_workers.max(1) {
            handles.push(s.spawn(|| loop {
                if abort.load(Ordering::Relaxed) {
                    return;
                }
                let idx = cursor.fetch_add(1, Ordering::Relaxed);
                if idx >= per_file.len() {
                    return;
                }
                let (file, scopes, bytes, fhash) = &per_file[idx];
                if let Err(e) = process_one_file(
                    engine,
                    proj_builder,
                    system_prompt,
                    utility_cfg,
                    layer,
                    group,
                    file,
                    scopes,
                    bytes,
                    fhash,
                    present_hashes,
                    total,
                    &done,
                    &decode_failures,
                    progress,
                ) {
                    let mut slot = first_error.lock().unwrap();
                    if slot.is_none() {
                        *slot = Some(e);
                    }
                    abort.store(true, Ordering::Relaxed);
                    return;
                }
            }));
        }
        for h in handles {
            h.join().expect("code_read worker panicked");
        }
    });

    if let Some(e) = first_error.into_inner().unwrap() {
        return Err(e);
    }
    // Reconcile the final progress to exactly `total`. Workers store
    // `set_step_progress` from their own `done` snapshot without a max, so
    // under the pool the last stored value can settle a step short even though
    // every unit ran; pin it to 100% now that the pool has fully drained.
    progress.set_step_progress(total as u64, total as u64);
    let n_failed = decode_failures.load(Ordering::Relaxed);
    tracing::info!(
        n_files = per_file.len(),
        n_decode_failures = n_failed,
        "code_read per-file ingest complete",
    );
    Ok(n_failed)
}

/// Ingest one file into a fresh per-file conversation: skip via the
/// resume-cache snapshot if its content hash is already present;
/// otherwise prefill each carved part (read_file tool-call + response),
/// tag the conversation with its content hash + metadata, then drop it
/// (freeing the GPU slot; the sealed turns + tags persist in the
/// substrate). The file summary is not decoded here — it is the async
/// summary tree's root, rolled up later by the background summariser.
#[allow(clippy::too_many_arguments)]
fn process_one_file(
    engine: &Mutex<ConversationEngine>,
    proj_builder: &Builder,
    system_prompt: &str,
    utility_cfg: &SequenceConfig,
    layer: LayerId,
    group: GroupId,
    file: &FileEntry,
    scopes: &[Scope],
    bytes: &[u8],
    file_hash: &str,
    present_hashes: &HashSet<String>,
    total: usize,
    done: &Arc<AtomicUsize>,
    decode_failures: &AtomicUsize,
    progress: &Arc<LoadProgress>,
) -> anyhow::Result<()> {
    // Resume cache: this content hash was already in the (live, non-
    // tombstoned) substrate at ingest start — skip the prefill+decode.
    if present_hashes.contains(file_hash) {
        let units = file_progress_units(scopes);
        let done_now = done.fetch_add(units, Ordering::Relaxed) + units;
        progress.set_step_progress(done_now as u64, total as u64);
        tracing::debug!(
            target: "zend::code_read::ingest",
            file = %file.path,
            n_scopes = scopes.len(),
            "skip: file already in substrate (resume cache hit)",
        );
        return Ok(());
    }

    // Cache miss → new / changed / crashed-partial file. Tombstone any
    // stale conversation for this path (a prior generation, or a partial
    // left by a crash before its tag was written), then mint the fresh
    // conversation. The engine lock covers only these quick ops and is
    // released before the decode-heavy body below.
    let mut conv = {
        let e = engine.lock().unwrap();
        for tl in e.find_conversations_by_metadata("path", &file.path) {
            if let Err(err) = e.tombstone_timeline(tl) {
                tracing::warn!(
                    target: "zend::code_read::ingest",
                    file = %file.path,
                    "tombstone of stale conversation failed: {err:#}",
                );
            }
        }
        let conv = e
            .new_conversation_with_projection(
                system_prompt,
                proj_builder.clone(),
                layer,
                group,
                utility_cfg.clone(),
            )
            .map_err(|err| anyhow::anyhow!("code_reading conv create: {err}"))?;
        // Each code scope now carries its OWN decoded two-sentence summary as its
        // closing assistant turn (see the redesign note above `emit_file_turns`),
        // so the AVL summariser must not also compress these turns into a separate
        // summary tree — disable summarisation for this per-file conversation.
        e.set_timeline_summarize(conv.timeline_id(), false);
        conv
    };

    // Shape the file's turns (per-part prefills + final summary decode),
    // bumping shared progress after each part so the bar moves during the
    // ingest (a single-file upload otherwise sits at 0% until the whole file
    // completes).
    // Count the progress units this file actually reported, so a tolerated
    // failure can reconcile the rest (the per-part callbacks only fire for
    // parts that completed).
    // Per-scope progress: the scheduler fires this as each scope lands, so the
    // bar climbs and the token count ticks up live (one unit per scope) rather
    // than jumping when the whole file's batch flushes. `fired` counts scopes
    // that actually reported, so a tolerated failure can reconcile the rest.
    let fired = Arc::new(AtomicUsize::new(0));
    let on_prefilled: crate::turn_sink::ScopeProgressFn = {
        let done = Arc::clone(done);
        let fired = Arc::clone(&fired);
        let progress = Arc::clone(progress);
        Arc::new(move |tokens: usize| {
            fired.fetch_add(1, Ordering::Relaxed);
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            progress.set_step_progress(d as u64, total as u64);
            progress.add_prefill_tokens(tokens as u64);
        })
    };
    let emit_result = {
        let mut sink = SequenceTurnSink::new(&mut conv);
        emit_file_turns(
            &mut sink,
            &file.path,
            file.language,
            scopes,
            bytes,
            on_prefilled,
        )
    };

    if let Err(e) = emit_result {
        // Reconcile this file's progress share: the callbacks above only fired
        // for parts that completed, so a tolerated failure below would leave
        // `done` short of `total` and the bar never reaches 100%. Advance
        // `done` by the units this file didn't get to report.
        let missing = file_progress_units(scopes).saturating_sub(fired.load(Ordering::Relaxed));
        if missing > 0 {
            let d = done.fetch_add(missing, Ordering::Relaxed) + missing;
            progress.set_step_progress(d as u64, total as u64);
        }
        // Prefill/decode failed. Do NOT tag the conversation: leaving it
        // untagged means the next run misses the resume cache and retries
        // this file (the stale partial is tombstoned by the path scan
        // above), rather than caching it as permanently summary-less.
        let n = decode_failures.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::warn!(
            target: "zend::code_read::ingest",
            file = %file.path,
            "file ingest failed (will retry next run): {e:#}",
        );
        if n > MAX_DECODE_FAILURES {
            return Err(anyhow::anyhow!(
                "code_read ingest: {n} file ingests failed \
                 (cap = {MAX_DECODE_FAILURES}); aborting",
            ));
        }
        return Ok(()); // conv drops → slot freed; no tag written.
    }

    // Tag the conversation: `content_sha256` is the resume-cache key,
    // `path` is the invalidation-scan key, the rest is diagnostic.
    let mut tags = std::collections::BTreeMap::new();
    tags.insert("kind".to_string(), "code_read".to_string());
    tags.insert("path".to_string(), file.path.clone());
    tags.insert("content_sha256".to_string(), file_hash.to_string());
    tags.insert("lang".to_string(), format!("{:?}", file.language));
    tags.insert("scopes".to_string(), scopes.len().to_string());
    tags.insert("size".to_string(), bytes.len().to_string());
    if let Err(e) = conv.set_metadata_many(&tags) {
        tracing::warn!(
            target: "zend::code_read::ingest",
            file = %file.path,
            "failed to tag conversation metadata (resume cache): {e:#}",
        );
    }

    // The file's conversation is now complete: every scope's tool round-trip is
    // spliced onto its timeline and sealed, and nothing attends this file again
    // until a projection retrieves it. Flag it for full KV eviction so the
    // persistence pipeline offloads its turns to cold (NVMe) and frees BOTH the
    // VRAM and RAM copies — otherwise the sealed turns linger hot and accumulate
    // across a large multi-file ingest until the card fills (the VRAM-exhaustion
    // grind). `FreeSequence` on drop only releases the batch slot, not the
    // sealed KV, so this proactive flag is what actually reclaims the space;
    // `elevate_to_hot` pulls the file back from cold on demand if reselected.
    let flagged = engine
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
    // turns and metadata remain in the substrate. Summarisation of these
    // scopes happens later in the background summariser.
    Ok(())
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

/// Selective refresh of the `code_reading` layer.
///
/// Re-carves `map`. Returns `NoOp` when no file hash changed. Otherwise it
/// runs the same reconcile + per-file pool as [`ingest_code_reading`]:
/// `reconcile_deleted` tombstones conversations for files now gone, and the
/// pool re-ingests over all files — unchanged files hit the resume-cache
/// snapshot and are skipped, while a changed file misses the snapshot,
/// tombstones its stale conversation, and re-ingests. So only changed/added
/// files actually re-prefill + re-decode.
///
/// The engine mutex is taken only for the quick create/tombstone ops inside
/// the pool (released across each decode), so chat consumers keep running.
pub fn refresh_code_reading(
    ctx: &RefreshContext<'_>,
    workspace: &Path,
    map: &RepoMap,
    prior: &CodeReadState,
    progress: &Arc<LoadProgress>,
    layer_name: &str,
    group_name: &str,
) -> anyhow::Result<RefreshOutcome> {
    // Carve once — drives both the change comparison and the re-ingest.
    let (per_file, next) = carve_workspace(workspace, map);
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

    let layer = ctx
        .proj_builder
        .id_for_layer(layer_name)
        .ok_or_else(|| anyhow::anyhow!("projection schema missing '{layer_name}' layer"))?;
    let group = ctx
        .proj_builder
        .id_for_group(group_name)
        .ok_or_else(|| anyhow::anyhow!("projection schema missing '{group_name}' group"))?;
    let system_prompt = layer_system_prompt(&ctx.proj_builder, layer_name, &ctx.config);
    let utility_cfg = code_read_config(ctx.config.clone());
    let n_workers = parallelism();
    let total: usize = per_file
        .iter()
        .map(|(_, s, _, _)| file_progress_units(s))
        .sum();
    progress.set_step_progress(0, total as u64);

    // Tombstone conversations for deleted files, then snapshot surviving
    // hashes; changed files miss the snapshot and are re-ingested (their
    // stale conversation is tombstoned in process_one_file).
    let present_paths: HashSet<&str> = per_file
        .iter()
        .map(|(f, _, _, _)| f.path.as_str())
        .collect();
    reconcile_deleted(ctx.engine, &present_paths);
    let present_hashes = ctx
        .engine
        .lock()
        .unwrap()
        .conversation_metadata_values("content_sha256");

    run_file_pool(
        ctx.engine,
        &ctx.proj_builder,
        &system_prompt,
        &utility_cfg,
        layer,
        group,
        &per_file,
        &present_hashes,
        total,
        progress,
        n_workers,
    )?;

    Ok(RefreshOutcome::Replaced { state: next })
}

fn layer_system_prompt(builder: &Builder, layer_name: &str, config: &SequenceConfig) -> String {
    debug_assert!(
        builder.schema().layers.iter().any(|l| l.name == layer_name),
        "projection schema missing '{layer_name}' layer"
    );
    // Every ingest conversation frames on the single shared system prompt (bare
    // top-level sections only — the `section_tree` framing, incl. the `persona`
    // selector, is materialised per turn by the projection from the schema +
    // selection, so the summarization framing is driven by `persona: summarize`
    // set on the ingest selection, not baked here). See `ingest_scope_roundtrip`.
    let mut body = String::new();
    for item in &builder.schema().system_prompt.items {
        if let SystemPromptItem::Section(s) = item {
            body.push_str(&s.content);
        }
    }
    config.dialect.format_system_prompt(&body)
}

/// Byte offset of the start of each line.  `offsets[i]` is the start
/// of line `i + 1` (1-indexed).  Final entry is the source length.
fn compute_line_offsets(bytes: &[u8]) -> Vec<usize> {
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

fn slice_lines(bytes: &[u8], offsets: &[usize], start_line: u32, end_line: u32) -> String {
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
}
