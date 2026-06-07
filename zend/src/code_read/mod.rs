//! `code_reading` layer ingestion.
//!
//! For every file in the [`crate::repo_scan::RepoMap`], parses into
//! scope-aware chunks and emits a tool-call conversation onto the
//! `code_reading` layer — for each scope, a prefilled
//! `read_file` request, a prefilled `<tool_call>` echo, a prefilled
//! `<tool_response>` carrying the source with line numbers, and a
//! decoded one-sentence summary the model produces live.
//!
//! Refresh is atomic: per-file content hashes ([`CodeReadState`])
//! decide whether anything changed; if so, a fresh set of timelines
//! is prefilled and the old set is tombstoned in a single swap.
//!
//! **Parallel ingest.**  At the candle workspace's ~80k scopes and
//! ~2-3s per scope decode, a single-session ingest would run for
//! tens of hours.  Instead we mint [`CODE_READ_PARALLELISM`]
//! Sequences on the same `(code_reading, scopes)` projection target
//! — each a distinct timeline — and distribute files round-robin
//! across them.  The scheduler's wave-batched grouped GEMM coalesces
//! work across the concurrent sessions, and the resolver's
//! `active_timelines_for_group` iterator surfaces all of them to
//! dialogue retrieval without code changes.  Override the worker
//! count with `ZEND_CODE_READ_PARALLELISM`.

pub mod carve;
pub mod header;
pub mod parsers;
#[cfg(test)]
pub mod test_util;
pub mod types;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use candle_conversation::projection::{Builder, SystemPromptItem, TimelineId};
use candle_conversation::{ConversationEngine, Sequence, SequenceConfig};
use sha2::{Digest, Sha256};

use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::repo_scan::{utility_config, FileEntry, RepoMap};
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

/// Sink-driven core of the `code_reading` ingestion — walks every
/// file in `map`, carves it into scopes, and emits the four-turn
/// tool-call conversation per scope into `sink`.  Returns the
/// per-file content hashes for the refresh path and reports progress
/// as `(scopes_done, scopes_total)`.
///
/// Kept for the integration test harness in
/// `tests/code_read_integration.rs`, which exercises the carve +
/// per-scope emission against the [`RecordingTurnSink`] without
/// needing a live engine.  The production ingest paths use the
/// parallel multi-Sequence wrapper [`ingest_code_reading`] instead;
/// this is the single-sink reference implementation.
///
/// `dead_code`: the lint fires for the `bin` target because nothing
/// inside the crate calls it (integration tests link as a separate
/// crate and aren't visible to the lint).
#[allow(dead_code)]
pub fn ingest_code_reading_into_sink<S: InsertTurnSink>(
    sink: &mut S,
    workspace: &Path,
    map: &RepoMap,
    progress: &LoadProgress,
) -> anyhow::Result<(usize, CodeReadState)> {
    // First pass: carve everything so we know the scope total before
    // we report any progress.  This is bounded by MAX_FILE_BYTES per
    // file so the carve pass fits in memory.
    let mut per_file: Vec<(FileEntry, Vec<Scope>, Vec<u8>)> = Vec::with_capacity(map.files.len());
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
            state.file_hashes.insert(file.path.clone(), sha256_hex(&bytes));
            per_file.push((file.clone(), scopes, bytes));
        }
    }

    let total: usize = per_file.iter().map(|(_, s, _)| s.len()).sum();
    tracing::info!(
        n_files = per_file.len(),
        n_scopes = total,
        "code_read carve complete",
    );
    progress.set_step_progress(0, total as u64);

    // Per-scope decode failures don't abort the whole ingest — a
    // single bad scope shouldn't waste minutes of work.  We log
    // each failure and keep walking; only when the failure count
    // exceeds `MAX_DECODE_FAILURES` do we give up and propagate.
    // Note: a failed scope leaves a half-emitted exchange — turn
    // pair 1+2 (the prefilled tool_call) is on the trunk; turn
    // pair 3+4 (the tool_response + decoded summary) is not.
    // Retrieval still works because the layer's projection scores
    // each turn pair independently.
    let mut done = 0usize;
    let mut decode_failures = 0usize;
    for (file, scopes, bytes) in &per_file {
        tracing::debug!(
            target: "zend::code_read::ingest",
            file = %file.path,
            file_bytes = bytes.len(),
            n_scopes = scopes.len(),
            "entering file",
        );
        let line_offsets = compute_line_offsets(bytes);
        for scope in scopes {
            let scope_label = format!("{}:{}-{}", file.path, scope.start_line, scope.end_line);
            let user_q = header::render_user_prompt(&file.path, scope);
            let tool_call = header::render_tool_call(&file.path, scope);
            let body = slice_lines(bytes, &line_offsets, scope.start_line, scope.end_line);
            let tool_response =
                header::render_tool_response(&file.path, scope, file.language, &body);

            tracing::debug!(
                target: "zend::code_read::ingest",
                scope = %scope_label,
                kind = ?scope.kind,
                body_bytes = body.len(),
                tool_response_bytes = tool_response.len(),
                progress = format!("{}/{}", done + 1, total),
                "scope start",
            );

            // Turn 1 (user) + Turn 2 (assistant tool_call), both prefilled.
            let prefill_start = std::time::Instant::now();
            sink.insert_prefill_turn(&user_q, &tool_call)?;
            tracing::debug!(
                target: "zend::code_read::ingest",
                scope = %scope_label,
                ms = prefill_start.elapsed().as_millis() as u64,
                "tool_call prefill done",
            );

            // Turn 3 (user tool_response) + Turn 4 (assistant summary,
            // decoded live).
            let decode_start = std::time::Instant::now();
            tracing::debug!(
                target: "zend::code_read::ingest",
                scope = %scope_label,
                tool_response_bytes = tool_response.len(),
                "submitting summary decode",
            );
            match sink.decode_summary_turn(&tool_response, header::SUMMARY_MAX_TOKENS) {
                Ok(summary) => {
                    tracing::debug!(
                        target: "zend::code_read::ingest",
                        scope = %scope_label,
                        ms = decode_start.elapsed().as_millis() as u64,
                        summary_chars = summary.chars().count(),
                        summary_head = %summary
                            .lines()
                            .next()
                            .unwrap_or("")
                            .chars()
                            .take(80)
                            .collect::<String>(),
                        "summary decode done",
                    );
                }
                Err(e) => {
                    decode_failures += 1;
                    tracing::warn!(
                        target: "zend::code_read::ingest",
                        scope = %scope_label,
                        ms = decode_start.elapsed().as_millis() as u64,
                        "scope summary decode failed: {e:#}",
                    );
                    if decode_failures > MAX_DECODE_FAILURES {
                        return Err(anyhow::anyhow!(
                            "code_read ingest: {decode_failures} scope summary decodes failed \
                             (cap = {MAX_DECODE_FAILURES}); aborting refresh",
                        ));
                    }
                }
            }
            done += 1;
            progress.set_step_progress(done as u64, total as u64);
            if done % 50 == 0 {
                tracing::info!(
                    target: "zend::code_read::ingest",
                    done = done,
                    total = total,
                    decode_failures = decode_failures,
                    "code_read progress",
                );
            }
        }
    }

    tracing::info!(
        n_scopes_emitted = done,
        n_decode_failures = decode_failures,
        "code_read prefill complete",
    );
    Ok((done, state))
}

/// Maximum tolerated per-scope summary decode failures in a single
/// ingestion pass before the whole refresh aborts.  A single
/// failure can happen for legitimate reasons (scheduler hiccup,
/// transient resource pressure); a cascade signals something
/// systemic.
pub const MAX_DECODE_FAILURES: usize = 16;

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

/// Default number of concurrent worker timelines used by the
/// parallel ingest path.  Override with `ZEND_CODE_READ_PARALLELISM`.
/// The scheduler's wave-batched grouped GEMM coalesces work across
/// these concurrent sessions, so the effective decode rate scales
/// near-linearly until the model's expert-cache hot set saturates.
///
/// Default is 8 — empirically OK on the 16 GB 4090 mobile.  16+
/// workers reliably trips CUDA OOM during the larger tool_response
/// prefills (the in-flight K/V across 16 concurrent prefills can
/// peak at 8+ GB even with chunk-sealed quantisation, on top of the
/// ~12 GB the model itself occupies).
pub const CODE_READ_PARALLELISM: usize = 8;

/// Resolve the worker count for the parallel ingest.  Reads
/// `ZEND_CODE_READ_PARALLELISM` if set and parseable, otherwise
/// returns [`CODE_READ_PARALLELISM`].  Clamped to `[1, 64]` because
/// the scheduler tops out around 64 concurrent sessions on the
/// 4090 mobile baseline.
fn parallelism() -> usize {
    std::env::var("ZEND_CODE_READ_PARALLELISM")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map(|n| n.clamp(1, 64))
        .unwrap_or(CODE_READ_PARALLELISM)
}

/// Top-level `code_reading` ingestion — mints
/// [`CODE_READ_PARALLELISM`] Sequences on the `(code_reading,
/// scopes)` projection target and processes the workspace's files
/// across them in parallel.  Files are distributed round-robin by
/// index; each worker processes all scopes within each of its files
/// serially before moving to the next file (reading the file's bytes
/// once and keeping line-offset state local).
///
/// Returns the constructed Sequences (caller stores them under
/// `CodeReadConv.sequences`) and the merged per-file content-hash
/// record used by [`refresh_code_reading`].
pub fn ingest_code_reading(
    engine: &ConversationEngine,
    proj_builder: Builder,
    workspace: &Path,
    map: &RepoMap,
    config: SequenceConfig,
    progress: &LoadProgress,
) -> anyhow::Result<(Vec<Sequence>, CodeReadState)> {
    let layer = proj_builder
        .id_for_layer("code_reading")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'code_reading' layer"))?;
    let group = proj_builder
        .id_for_group("scopes")
        .ok_or_else(|| anyhow::anyhow!("projection schema missing 'scopes' group"))?;

    let system_prompt = layer_system_prompt(&proj_builder, "code_reading", &config);
    let utility_cfg = utility_config(config);
    let n_workers = parallelism();
    let user_override = std::env::var("ZEND_CODE_READ_PARALLELISM").is_ok();
    tracing::info!(
        n_workers = n_workers,
        n_files = map.files.len(),
        env_override = user_override,
        "code_read: minting {n_workers} concurrent timelines \
         (set ZEND_CODE_READ_PARALLELISM=N to override; lower it if you hit CUDA OOM)",
    );

    let mut sequences: Vec<Sequence> = Vec::with_capacity(n_workers);
    for _ in 0..n_workers {
        let seq = engine
            .new_conversation_with_projection(
                &system_prompt,
                proj_builder.clone(),
                layer,
                group,
                utility_cfg.clone(),
            )
            .map_err(|e| anyhow::anyhow!("code_reading conv create: {e}"))?;
        sequences.push(seq);
    }

    let state = run_parallel_ingest(&mut sequences, workspace, map, progress, n_workers)?;
    Ok((sequences, state))
}

/// Carve every file in `map`, distribute across `sequences`
/// round-robin by file index, then drive the per-scope tool-call
/// ingest in parallel using `std::thread::scope`.  Workers share
/// atomic counters for progress + decode-failure accounting and an
/// abort flag so the first worker error stops the rest.
fn run_parallel_ingest(
    sequences: &mut [Sequence],
    workspace: &Path,
    map: &RepoMap,
    progress: &LoadProgress,
    n_workers: usize,
) -> anyhow::Result<CodeReadState> {
    // Carve every file sequentially up-front so we have an accurate
    // `total` for the progress bar and a single source of truth for
    // the per-file content hashes.  Carving is CPU-bound and small
    // relative to ingest, so we don't bother parallelising it.
    let mut per_file: Vec<(FileEntry, Vec<Scope>, Vec<u8>)> = Vec::with_capacity(map.files.len());
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
            state
                .file_hashes
                .insert(file.path.clone(), sha256_hex(&bytes));
            per_file.push((file.clone(), scopes, bytes));
        }
    }
    let total: usize = per_file.iter().map(|(_, s, _)| s.len()).sum();
    tracing::info!(
        n_files = per_file.len(),
        n_scopes = total,
        n_workers = n_workers,
        "code_read carve complete; dispatching to workers",
    );
    progress.set_step_progress(0, total as u64);

    // Round-robin: worker `i` gets files at indices `i, i+N, i+2N, …`.
    // Each chunk is a slice of file references — bytes/scopes stay
    // owned by `per_file` (read-only borrow shared across threads).
    let mut chunks: Vec<Vec<usize>> = (0..n_workers).map(|_| Vec::new()).collect();
    for (idx, _) in per_file.iter().enumerate() {
        chunks[idx % n_workers].push(idx);
    }

    let done = AtomicUsize::new(0);
    let decode_failures = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    let first_error: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let per_file_ref = &per_file;
    let done_ref = &done;
    let failures_ref = &decode_failures;
    let abort_ref = &abort;
    let first_error_ref = &first_error;
    let progress_ref = progress;

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_workers);
        for (worker_idx, (sequence, chunk)) in
            sequences.iter_mut().zip(chunks.into_iter()).enumerate()
        {
            let chunk = chunk;
            handles.push(s.spawn(move || {
                let mut sink = SequenceTurnSink::new(sequence);
                for file_idx in chunk {
                    if abort_ref.load(Ordering::Relaxed) {
                        return;
                    }
                    let (file, scopes, bytes) = &per_file_ref[file_idx];
                    if let Err(e) = process_file(
                        &mut sink,
                        worker_idx,
                        file,
                        scopes,
                        bytes,
                        total,
                        done_ref,
                        failures_ref,
                        progress_ref,
                    ) {
                        // Park the first error and signal the rest
                        // of the scope to wind down.  Subsequent
                        // errors from other workers are dropped —
                        // the first cause is what matters.
                        let mut slot = first_error_ref.lock().unwrap();
                        if slot.is_none() {
                            *slot = Some(e);
                        }
                        abort_ref.store(true, Ordering::Relaxed);
                        return;
                    }
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
        n_scopes_emitted = done.load(Ordering::Relaxed),
        n_decode_failures = decode_failures.load(Ordering::Relaxed),
        "code_read parallel ingest complete",
    );

    Ok(state)
}

/// Process every scope of one file on the calling worker's sink.
/// Mirrors the inner loop of [`ingest_code_reading_into_sink`] but
/// reports progress / decode-failure accounting through atomic
/// counters so multiple workers can share them safely.
#[allow(clippy::too_many_arguments)]
fn process_file(
    sink: &mut SequenceTurnSink<'_>,
    worker_idx: usize,
    file: &FileEntry,
    scopes: &[Scope],
    bytes: &[u8],
    total: usize,
    done: &AtomicUsize,
    decode_failures: &AtomicUsize,
    progress: &LoadProgress,
) -> anyhow::Result<()> {
    tracing::debug!(
        target: "zend::code_read::ingest",
        worker = worker_idx,
        file = %file.path,
        file_bytes = bytes.len(),
        n_scopes = scopes.len(),
        "entering file",
    );
    let line_offsets = compute_line_offsets(bytes);
    for scope in scopes {
        let scope_label = format!(
            "[w{worker_idx}] {}:{}-{}",
            file.path, scope.start_line, scope.end_line
        );
        let user_q = header::render_user_prompt(&file.path, scope);
        let tool_call = header::render_tool_call(&file.path, scope);
        let body = slice_lines(bytes, &line_offsets, scope.start_line, scope.end_line);
        let tool_response = header::render_tool_response(&file.path, scope, file.language, &body);

        // Turn 1 (user) + Turn 2 (assistant tool_call) — both prefilled.
        let prefill_start = std::time::Instant::now();
        sink.insert_prefill_turn(&user_q, &tool_call)?;
        tracing::debug!(
            target: "zend::code_read::ingest",
            scope = %scope_label,
            ms = prefill_start.elapsed().as_millis() as u64,
            "tool_call prefill done",
        );

        // Turn 3 (user tool_response) + Turn 4 (assistant summary, decoded live).
        let decode_start = std::time::Instant::now();
        match sink.decode_summary_turn(&tool_response, header::SUMMARY_MAX_TOKENS) {
            Ok(summary) => {
                tracing::debug!(
                    target: "zend::code_read::ingest",
                    scope = %scope_label,
                    ms = decode_start.elapsed().as_millis() as u64,
                    summary_chars = summary.chars().count(),
                    summary_head = %summary
                        .lines()
                        .next()
                        .unwrap_or("")
                        .chars()
                        .take(80)
                        .collect::<String>(),
                    "summary decode done",
                );
            }
            Err(e) => {
                let n = decode_failures.fetch_add(1, Ordering::Relaxed) + 1;
                tracing::warn!(
                    target: "zend::code_read::ingest",
                    scope = %scope_label,
                    ms = decode_start.elapsed().as_millis() as u64,
                    "scope summary decode failed: {e:#}",
                );
                if n > MAX_DECODE_FAILURES {
                    return Err(anyhow::anyhow!(
                        "code_read ingest: {n} scope summary decodes failed \
                         (cap = {MAX_DECODE_FAILURES}); aborting refresh",
                    ));
                }
            }
        }

        let done_now = done.fetch_add(1, Ordering::Relaxed) + 1;
        progress.set_step_progress(done_now as u64, total as u64);
        if done_now.is_multiple_of(50) {
            tracing::info!(
                target: "zend::code_read::ingest",
                done = done_now,
                total = total,
                decode_failures = decode_failures.load(Ordering::Relaxed),
                "code_read progress",
            );
        }
    }
    Ok(())
}

/// Outcome of an atomic [`refresh_code_reading`] call.  Same shape
/// as [`crate::repo_scan::RefreshOutcome`].  `Replaced.sequences`
/// carries the freshly-minted set of parallel timelines; the caller
/// is expected to swap them in atomically and tombstone all of the
/// previous-generation timelines in one go.
pub enum RefreshOutcome {
    NoOp,
    Replaced {
        sequences: Vec<Sequence>,
        state: CodeReadState,
    },
}

/// Atomic refresh of the `code_reading` conversation.
///
/// Re-hashes the files in `map`.  Returns `NoOp` when nothing
/// changed.  Otherwise mints a fresh set of parallel timelines (one
/// per worker, see [`parallelism`]), runs the full parallel
/// tool-call ingestion (including per-scope summary decodes) across
/// them, then tombstones every prior-generation timeline.  The
/// active resolver continues to serve the prior timelines through
/// the entire prefill + decode window — stale better than missing.
/// At the tombstone instant retrieval flips atomically to the new
/// content.
///
/// Like [`crate::repo_scan::refresh_repo_map`], the engine mutex on
/// `ctx` is acquired only at the boundary points (mint, then
/// tombstone) so concurrent chat consumers aren't blocked for the
/// duration of the decode-heavy middle.
pub fn refresh_code_reading(
    ctx: &RefreshContext<'_>,
    workspace: &Path,
    map: &RepoMap,
    prior: &CodeReadState,
    old_timelines: &[TimelineId],
    progress: &LoadProgress,
) -> anyhow::Result<RefreshOutcome> {
    // Cheap dry-run hash pass — compute the would-be next state and
    // compare against `prior` before any model work.
    let mut next = CodeReadState::default();
    for file in &map.files {
        let path = workspace.join(&file.path);
        if let Ok(bytes) = fs::read(&path) {
            let is_tsx = file.path.ends_with(".tsx");
            let scopes = carve::carve(&bytes, file.language, is_tsx);
            if !scopes.is_empty() {
                next.file_hashes
                    .insert(file.path.clone(), sha256_hex(&bytes));
            }
        }
    }
    if prior.equivalent_to(&next) {
        tracing::debug!("code_read refresh: no file hash changed, skipping refresh");
        return Ok(RefreshOutcome::NoOp);
    }

    let changed = prior.changed_files(&next);
    tracing::info!(
        n_changed = changed.len(),
        sample_changed = ?changed.iter().take(5).collect::<Vec<_>>(),
        n_old_timelines = old_timelines.len(),
        "code_read refresh: minting new generation and re-ingesting all scopes",
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
    let utility_cfg = utility_config(ctx.config.clone());
    let n_workers = parallelism();

    let mut new_sequences: Vec<Sequence> = {
        let engine = ctx.engine.lock().unwrap();
        let mut v = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            v.push(
                engine
                    .new_conversation_with_projection(
                        &system_prompt,
                        ctx.proj_builder.clone(),
                        layer,
                        group,
                        utility_cfg.clone(),
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("code_reading refresh: new conv create: {e}")
                    })?,
            );
        }
        v
    };

    // Lock-free prefill + decode window — concurrent engine
    // consumers (chat commits, sidebar reads) keep running while we
    // prefill the tool-call exchange and decode the per-scope
    // summaries across all parallel timelines.
    let fresh_state = run_parallel_ingest(&mut new_sequences, workspace, map, progress, n_workers)?;

    {
        let engine = ctx.engine.lock().unwrap();
        for &tl in old_timelines {
            engine
                .tombstone_timeline(tl)
                .map_err(|e| anyhow::anyhow!("code_read refresh: tombstone old timeline: {e}"))?;
        }
    }

    Ok(RefreshOutcome::Replaced {
        sequences: new_sequences,
        state: fresh_state,
    })
}

fn layer_system_prompt(
    builder: &Builder,
    layer_name: &str,
    config: &SequenceConfig,
) -> String {
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
