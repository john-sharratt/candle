//! `code_reading` layer ingestion.
//!
//! Each file in the [`crate::repo_scan::RepoMap`] becomes ONE
//! conversation on the `code_reading` layer.  The file is parsed into
//! scope-aware parts; each part contributes a prefilled `read_file`
//! request, a prefilled `<tool_call>` echo, and a prefilled
//! `<tool_response>` carrying the source with line numbers.  The
//! conversation closes with a single decoded whole-file summary
//! (≤200 words) the model produces live.
//!
//! Refresh is per-file: content hashes ([`CodeReadState`]) decide
//! which files changed; deleted files' conversations are tombstoned,
//! changed files are re-ingested, and unchanged files are skipped via
//! the substrate resume cache (the per-file `content_sha256` tag).
//!
//! **Parallel ingest.**  At the candle workspace's tens of thousands
//! of files, a single-session ingest would run for tens of hours.
//! Instead [`CODE_READ_PARALLELISM`] workers each process whole files
//! concurrently (prefill + summary decode as one unit), minting a
//! distinct per-file conversation per file.  The scheduler's
//! wave-batched grouped GEMM coalesces work across the concurrent
//! sessions, and the resolver's `active_timelines_for_group` iterator
//! surfaces all of them to dialogue retrieval without code changes.
//! Override the worker count with `ZEND_CODE_READ_PARALLELISM`.

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
use std::sync::Mutex;

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

/// Emit one file's turns into `sink` — the one-conversation-per-file
/// shape: a prefill turn per carved part (read_file tool-call + response)
/// so the model reads the whole file across the conversation, then a final
/// decoded turn summarising the entire file in ≤200 words. Shared by the
/// production per-file ingest ([`process_one_file`]) and the sink-driven
/// reference/test path ([`ingest_code_reading_into_sink`]).
fn emit_file_turns<S: InsertTurnSink>(
    sink: &mut S,
    path: &str,
    language: Language,
    scopes: &[Scope],
    bytes: &[u8],
) -> anyhow::Result<()> {
    let line_offsets = compute_line_offsets(bytes);
    for scope in scopes {
        let body = slice_lines(bytes, &line_offsets, scope.start_line, scope.end_line);
        let user = header::render_part_user_prompt(path, scope);
        let assistant = format!(
            "{}\n{}",
            header::render_tool_call(path, scope),
            header::render_tool_response(path, scope, language, &body),
        );
        sink.insert_prefill_turn(&user, &assistant)?;
    }
    let summary_prompt = header::render_file_summary_prompt(path);
    sink.decode_summary_turn(&summary_prompt, header::FILE_SUMMARY_MAX_TOKENS)?;
    Ok(())
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
    let total: usize = per_file.iter().map(|(_, s, _, _)| s.len()).sum();
    tracing::info!(
        n_files = per_file.len(),
        n_scopes = total,
        "code_read carve complete"
    );
    progress.set_step_progress(0, total as u64);

    let mut done = 0usize;
    for (file, scopes, bytes, _fhash) in &per_file {
        emit_file_turns(sink, &file.path, file.language, scopes, bytes)?;
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

/// Default number of concurrent worker timelines used by the
/// parallel ingest path.  Override with `ZEND_CODE_READ_PARALLELISM`.
/// The scheduler's wave-batched grouped GEMM coalesces work across
/// these concurrent sessions, so the effective decode rate scales
/// near-linearly until the model's expert-cache hot set saturates.
///
/// Default is 16 — matched to the scheduler's prefill admission cap
/// (`MAX_ACTIVE_PREFILLS = 16`). Ingest is prefill-bound, and the scheduler only
/// ever has 16 prefills in flight, so workers beyond 16 cannot add prefill
/// throughput — they just queue while still pinning their per-file conversation
/// KV in VRAM and running per-turn post-processing on their own thread (extra
/// CPU contention). Matching the two keeps the live working set and the
/// per-insert CPU load bounded while keeping the prefill pipe full. Raise it via
/// `ZEND_CODE_READ_PARALLELISM` to lean on the VRAM-pressure backpressure path:
/// the admission gate stops promoting new prefills + force-compacts under
/// pressure, and the per-arena VRAM budget (`CANDLE_KV_VRAM_RESERVE_MB`) fails
/// fast + compacts rather than letting the driver page KV to host memory.
pub const CODE_READ_PARALLELISM: usize = 16;

/// [`utility_config`] specialised for the `code_reading` layer: append-only
/// (no reprojection), inheriting the utility C5 compression level.
///
/// Both K and V are adaptively quantized at C5 (the engine-wide uniform-K pin
/// is off in this config).
fn code_read_config(config: SequenceConfig) -> SequenceConfig {
    utility_config(config)
}

/// Resolve the worker count for the parallel ingest.  Reads
/// `ZEND_CODE_READ_PARALLELISM` if set and parseable, otherwise
/// returns [`CODE_READ_PARALLELISM`].  Clamped to `[1, 256]`.
fn parallelism() -> usize {
    std::env::var("ZEND_CODE_READ_PARALLELISM")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map(|n| n.clamp(1, 256))
        .unwrap_or(CODE_READ_PARALLELISM)
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

/// Tombstone every live `code_read` conversation whose `path` is no longer
/// present in `present_paths`. Covers files deleted while the daemon was
/// down (the startup ingest only visits files that still exist) and files
/// removed between fs-watcher refreshes. Still-present *changed* files are
/// handled by [`process_one_file`], which tombstones a path's stale
/// conversation before re-ingesting it.
fn reconcile_deleted(engine: &Mutex<ConversationEngine>, present_paths: &HashSet<&str>) {
    let e = engine.lock().unwrap();
    for (tl, path) in e.conversations_with_metadata_key("path") {
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
/// model reads the whole file, then a final decoded turn that summarises
/// the entire file in ≤200 words. The conversation is tagged with a
/// content hash + descriptive metadata, then freed — its sealed turns and
/// metadata persist in the substrate, so retrieval and the restart-resume
/// cache work off the substrate, not a live sequence.
///
/// Reconciliation runs first: conversations for files no longer on disk
/// are tombstoned, then a one-pass snapshot of present content hashes lets
/// each worker skip already-ingested files in O(1). A bounded pool of
/// [`parallelism`] workers pulls files from a shared cursor, so at most
/// that many conversations are live at once (VRAM bound) while the
/// scheduler wave-batches their prefills/decodes. The engine mutex is
/// taken only for the quick create/tombstone ops, never across a decode.
pub fn ingest_code_reading(
    engine: &Mutex<ConversationEngine>,
    proj_builder: Builder,
    workspace: &Path,
    map: &RepoMap,
    config: SequenceConfig,
    progress: &LoadProgress,
) -> anyhow::Result<CodeReadState> {
    let layer = proj_builder
        .id_for_layer("code_reading")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'code_reading' layer"))?;
    let group = proj_builder
        .id_for_group("scopes")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'scopes' group"))?;
    let system_prompt = layer_system_prompt(&proj_builder, "code_reading", &config);
    let utility_cfg = code_read_config(config);
    let n_workers = parallelism();

    let (per_file, state) = carve_workspace(workspace, map);
    let total: usize = per_file.iter().map(|(_, s, _, _)| s.len()).sum();

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

    let user_override = std::env::var("ZEND_CODE_READ_PARALLELISM").is_ok();
    tracing::info!(
        n_workers = n_workers,
        n_files = per_file.len(),
        n_scopes = total,
        n_cached = present_hashes.len(),
        env_override = user_override,
        "code_read: per-file ingest across {n_workers} workers \
         (set ZEND_CODE_READ_PARALLELISM=N to override; lower it if you hit CUDA OOM)",
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
/// error stops the rest). Returns once every file is processed.
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
    progress: &LoadProgress,
    n_workers: usize,
) -> anyhow::Result<()> {
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
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
    tracing::info!(
        n_files = per_file.len(),
        n_decode_failures = decode_failures.load(Ordering::Relaxed),
        "code_read per-file ingest complete",
    );
    Ok(())
}

/// Ingest one file into a fresh per-file conversation: skip via the
/// resume-cache snapshot if its content hash is already present;
/// otherwise prefill each carved part (read_file tool-call + response),
/// decode a final ≤200-word whole-file summary, tag the conversation
/// with its content hash + metadata, then drop it (freeing the GPU slot;
/// the sealed turns + tags persist in the substrate).
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
    done: &AtomicUsize,
    decode_failures: &AtomicUsize,
    progress: &LoadProgress,
) -> anyhow::Result<()> {
    // Resume cache: this content hash was already in the (live, non-
    // tombstoned) substrate at ingest start — skip the prefill+decode.
    if present_hashes.contains(file_hash) {
        let done_now = done.fetch_add(scopes.len(), Ordering::Relaxed) + scopes.len();
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
        e.new_conversation_with_projection(
            system_prompt,
            proj_builder.clone(),
            layer,
            group,
            utility_cfg.clone(),
        )
        .map_err(|err| anyhow::anyhow!("code_reading conv create: {err}"))?
    };

    // Shape the file's turns (per-part prefills + final summary decode).
    let emit_result = {
        let mut sink = SequenceTurnSink::new(&mut conv);
        emit_file_turns(&mut sink, &file.path, file.language, scopes, bytes)
    };
    // Progress is per-file (the unit is now the file, not the scope).
    let done_now = done.fetch_add(scopes.len(), Ordering::Relaxed) + scopes.len();
    progress.set_step_progress(done_now as u64, total as u64);

    if let Err(e) = emit_result {
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
    // `conv` drops here → FreeSequence releases the GPU slot; the sealed
    // turns and metadata remain in the substrate.
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
    progress: &LoadProgress,
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
        .id_for_layer("code_reading")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'code_reading' layer"))?;
    let group = ctx
        .proj_builder
        .id_for_group("scopes")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'scopes' group"))?;
    let system_prompt = layer_system_prompt(&ctx.proj_builder, "code_reading", &ctx.config);
    let utility_cfg = code_read_config(ctx.config.clone());
    let n_workers = parallelism();
    let total: usize = per_file.iter().map(|(_, s, _, _)| s.len()).sum();
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
    let layer = builder
        .schema()
        .layers
        .iter()
        .find(|l| l.name == layer_name)
        .unwrap_or_else(|| panic!("projection schema missing '{layer_name}' layer"));

    let mut body = String::new();
    for item in &layer.system_prompt.items {
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
