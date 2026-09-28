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

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::chain_health::{chain_break, ChainBreak};
use candle_conversation::projection::{
    OptionalState, SelectionState, TimelineId, FORCE_TOOL_SELECTOR, NO_THINK_SELECTOR,
    TOOLS_ENABLED_SELECTOR,
};
use candle_conversation::stencil::TriggerRegistry;
use candle_conversation::{ConversationEngine, Sequence, TurnOptions, TurnText};
use sha2::{Digest, Sha256};
use zend_tools::tools::file::read::MAX_READ_LINES;
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
///
/// # A file counts as ingested only once its chain FINISHED
///
/// The metadata this joins (`path`, `content_sha256`) is written when the file's
/// hidden conversation is minted, not when it answers — so a read that died
/// part-way carries exactly the same keys as one that succeeded, and the file
/// then counts as done forever. That is not hypothetical: a README read spent
/// its whole decode budget deliberating over how many `file_read` ranges to
/// issue, was cut off mid-word, and emitted no tool call — so no coupling, no
/// second turn, and no summary. The substrate stored it faithfully (CRC-clean,
/// chunk counts consistent), the file counted as ingested, and its one turn
/// stayed in the corpus as a worked example of an assistant that deliberates
/// and produces nothing, which later conversations then imitated.
///
/// So each candidate timeline is asked whether its chain finished
/// ([`chain_break`]). An unfinished one is left out of the state, which makes
/// the file read as NEW to the next pass and rebuilds it — recovery with no
/// tombstone and nothing deleted.
///
/// **A complete chain wins over an incomplete one for the same path.** A path
/// may carry more than one timeline (a failed attempt and the rebuild that
/// followed it), and whichever the iteration happens to reach last must not
/// decide the answer.
pub fn code_read_state_from_substrate(engine: &Mutex<ConversationEngine>) -> CodeReadState {
    let scan = scan_ingest_chains(engine);
    let mut state = CodeReadState::default();
    for (path, hash) in scan.finished {
        state.file_hashes.insert(path, hash);
    }
    // Reported here and not in the scan, which the resume gate also calls: one
    // line per unfinished chain per boot. A handful is ordinary recovery, while a
    // corpus-wide sweep of them means the decode budget or the opening prompt is
    // wrong for this model and every pass will keep redoing the same work.
    for (path, why) in &scan.broken {
        if state.file_hashes.contains_key(path) {
            continue; // a later, complete chain for the same file covers it
        }
        tracing::info!(
            file = %path,
            ?why,
            "code_read: ingest chain unfinished — the file reads as new and will be rebuilt",
        );
    }
    state
}

/// What [`scan_ingest_chains`] found: the files whose ingest finished, and the
/// ones whose chain stopped part-way.
struct IngestChains {
    /// `(path, content hash)` per finished chain.
    finished: Vec<(String, String)>,
    broken: Vec<(String, ChainBreak)>,
}

/// Join every ingested file's `path` and `content_sha256` metadata by timeline
/// and sort them by whether the chain that wrote them finished
/// ([`chain_break`]).
///
/// **The one place both resume gates read**, which is the whole point of
/// factoring it: a pass asks two separate questions — "which files changed?"
/// (the [`CodeReadState`] diff) and "which content is already in the substrate?"
/// (the [`process_one_file`] resume cache) — and they are computed from
/// different queries. Fixing only the diff makes a pass announce the file as
/// changed and then skip it on the resume hit, which is worse than not fixing it
/// at all: it reports recovery that never happens.
fn scan_ingest_chains(engine: &Mutex<ConversationEngine>) -> IngestChains {
    let eng = engine.lock().unwrap();
    let hashes: HashMap<TimelineId, String> = eng
        .conversations_with_metadata_key("content_sha256")
        .into_iter()
        .collect();
    let conv = eng.conversation();
    let substrate = conv.read();
    let mut out = IngestChains {
        finished: Vec::new(),
        broken: Vec::new(),
    };
    for (tl, path) in eng.conversations_with_metadata_key("path") {
        let Some(hash) = hashes.get(&tl) else {
            continue;
        };
        match chain_break(&substrate, tl) {
            Some(why) => out.broken.push((path, why)),
            None => out.finished.push((path, hash.clone())),
        }
    }
    out
}

/// Content hashes whose ingest chain finished — the resume cache
/// [`process_one_file`] probes before spending a decode on a file.
///
/// Replaces a bare sweep of every `content_sha256` value, which counted a
/// half-written chain's hash as present and so skipped the very file the diff
/// had just marked for rebuild.
fn finished_content_hashes(engine: &Mutex<ConversationEngine>) -> HashSet<String> {
    scan_ingest_chains(engine)
        .finished
        .into_iter()
        .map(|(_, hash)| hash)
        .collect()
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
/// **It names the file's LENGTH, and that is not a nicety.** Naming only the
/// per-call cap leaves the model to guess how many ranges cover the file, and a
/// guess is a decision it can spend its whole decode budget failing to make: a
/// README read deliberated over whether the file was under the cap, whether a
/// range past EOF would error, and whether to risk it — until the token budget
/// cut it off mid-word, before any `file_read` call was emitted. It produced no
/// tool call, so no coupling and no second turn, and the chain stood in the
/// corpus as an assistant that deliberates and answers nothing. With the length
/// given, the arithmetic is settled before the model starts.
///
/// **It asks for the calls in PARALLEL, and that is a round-count decision.**
/// `file_read` serves at most [`MAX_READ_LINES`] lines per call, so a long file
/// needs several — and a round is one decode plus one tool dispatch, so reading
/// a 2,000-line file one call per turn costs ten decodes where a single turn
/// carrying ten calls costs one. The loop already supports it in full:
/// `tool_round::plan` returns every call an answer makes, `tool_round::run`
/// executes them in order, and `format_tool_responses` hands back one
/// `<tool_response>` block per result, so the whole file arrives in the next
/// turn's context together. Naming the cap in the prompt is what lets the model
/// choose the ranges itself rather than discovering the cut one call at a time
/// from the excerpt header.
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
fn opening_prompt(path: &str, lines: usize) -> String {
    let calls = lines.div_ceil(MAX_READ_LINES as usize).max(1);
    format!(
        "Read the entire contents of `{path}` — and only this file — using \
         {FILE_READ_TOOL}. The file is {lines} lines long and each call returns at \
         most {MAX_READ_LINES} lines, so it takes exactly {calls} \
         {FILE_READ_TOOL} call(s): issue all {calls} in the SAME reply, one per \
         consecutive {MAX_READ_LINES}-line range covering lines 1 to {lines}, \
         instead of one call per reply. \
         Every call you make in a reply is run together and all of their results \
         come back to you at once. Once you've read the whole thing, summarize \
         what it contains: its purpose, its main structures or functions, and how \
         it fits into the codebase."
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
///
/// **16, and the ceiling here is VRAM rather than throughput.** Each worker is
/// serialised on its own file's whole turn — a prefill that attends over the
/// inherited priming chain (~32k KV, ~9.4 s) and then a decode — so the pool
/// width, not the wave, is what bounds the file phase.
///
/// Measured on the 72 GB card over the same corpus:
///
///   12   0.58 files/min   stable for three hours
///   32   1.66 files/min   K/V ran 13.9 GB against a 2.5 GB budget, the weight
///                         zone slid to 30.6 GB against a 28.8 GB floor, and the
///                         daemon died after ~30 minutes — no panic, no poison,
///                         the log simply stops
///
/// A file conversation is not a directory conversation: it carries 876k KV per
/// decode forward against a directory's 25k, so a width that is comfortable for
/// `repo_map` exhausts the card here. 16 keeps most of the gain over 12 with
/// margin against the floor, which matters more than rate — a dead daemon
/// ingests nothing.
pub const CODE_READ_PARALLELISM: usize = 16;

/// Metadata key a `code_reading` conversation is identified by — the file's
/// workspace-relative path. The `repo_map` twin of it is `repo_scan::DIR_KEY`.
pub(crate) const PATH_KEY: &str = "path";

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

/// One file queued for ingest: its repo-map entry, content hash, and length in
/// lines. Bytes are read once, here, only to pass the size/binary guards and
/// compute the hash — never shown to the model directly; the model reads the
/// file for itself via a real `file_read` call.
///
/// The line count travels because [`opening_prompt`] needs it: the model has to
/// choose its `file_read` ranges in its FIRST reply, and a model told only the
/// per-call cap has to guess how many ranges cover the file. Counting the
/// newlines of bytes already in hand costs nothing and removes the guess.
struct QueuedFile {
    file: FileEntry,
    hash: String,
    lines: usize,
}

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
        // What `file_read` will report as the file's length: its lines, counting
        // a final line that carries no trailing newline.
        let lines = bytes.iter().filter(|&&b| b == b'\n').count()
            + usize::from(!bytes.is_empty() && !bytes.ends_with(b"\n"));
        per_file.push(QueuedFile {
            file: file.clone(),
            hash: fhash,
            lines,
        });
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
    let present_hashes = finished_content_hashes(ctx.engine);

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
    let Some(queued) = per_file.into_iter().next() else {
        return Ok(None);
    };
    let present_hashes = finished_content_hashes(ctx.engine);
    // This link's parent is the chain so far, not whatever the daemon-wide
    // chain end will be once it is finished being built.
    let link_ctx = RefreshContext {
        priming_chain_end: Some(predecessor),
        ..ctx.clone()
    };
    if !present_hashes.contains(&queued.hash) {
        let failures = Failures::new();
        process_one_file(&link_ctx, base, &queued, &present_hashes, &failures)?;
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
        .filter(|q| !present_hashes.contains(&q.hash))
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
                let q = &per_file[idx];
                if let Err(e) = process_one_file(ctx, base, q, present_hashes, &failures) {
                    // An error escaping `process_one_file` is an unexpected one
                    // (its own failure mode records and returns Ok). Record it
                    // so it reaches the report instead of vanishing, and let the
                    // cap decide whether to stop the pass.
                    let n = failures.record(&q.file.path, format!("{e:#}"));
                    tracing::warn!(
                        target: "zend::code_read::ingest",
                        file = %q.file.path,
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
                if !present_hashes.contains(&q.hash) {
                    backlog_done.fetch_add(1, Ordering::Relaxed);
                    crate::ingest_backlog::item_done(&q.file.path);
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
    queued: &QueuedFile,
    present_hashes: &HashSet<String>,
    failures: &Failures,
) -> anyhow::Result<()> {
    let QueuedFile {
        file,
        hash: file_hash,
        lines,
    } = queued;
    let file_hash = file_hash.as_str();
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
        *lines,
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
    lines: usize,
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

    let mut current_message: TurnText = TurnText::from(opening_prompt(path, lines));
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
    let present_paths: HashSet<&str> = per_file.iter().map(|q| q.file.path.as_str()).collect();
    reconcile_deleted(ctx.engine, map, &present_paths);
    let present_hashes = finished_content_hashes(ctx.engine);

    run_file_pool(ctx, base, &per_file, &present_hashes, progress, n_workers)?;

    Ok(RefreshOutcome::Replaced { state: next })
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
    fn opening_prompt_names_the_path_and_the_one_tool() {
        let p = opening_prompt("src/lib.rs", 120);
        assert!(p.contains("src/lib.rs"));
        assert!(p.contains(FILE_READ_TOOL));
    }

    /// **The prompt must state the file's length and the exact number of calls.**
    ///
    /// Told only the per-call cap, a model has to guess how many ranges cover the
    /// file — and that guess is a decision it can spend its entire decode budget
    /// failing to reach. A README read did exactly that: it weighed whether the
    /// file was under the cap and whether a range past EOF would error until the
    /// budget cut it off mid-word, with no `file_read` call emitted, no coupling,
    /// no second turn and no summary. The arithmetic belongs in the prompt.
    #[test]
    fn opening_prompt_states_the_length_and_the_call_count() {
        let cap = MAX_READ_LINES as usize;
        // Exactly one cap's worth: one call, not two.
        let one = opening_prompt("src/small.rs", cap);
        assert!(one.contains(&cap.to_string()), "{one:?}");
        assert!(
            one.contains(" 1 "),
            "one call for a {cap}-line file: {one:?}"
        );
        // One line over: two calls.
        let two = opening_prompt("src/mid.rs", cap + 1);
        assert!(
            two.contains(&(cap + 1).to_string()),
            "must name the file's own length: {two:?}",
        );
        assert!(two.contains(" 2 "), "two calls just past the cap: {two:?}");
        // An empty file still asks for one call rather than zero.
        let empty = opening_prompt("src/empty.rs", 0);
        assert!(empty.contains(" 1 "), "never zero calls: {empty:?}");
    }

    /// The measured failure this prompt exists to close: left as "read as much
    /// as you need", a model sometimes skipped `file_read` entirely and
    /// guessed a summary from the filename alone. The prompt must require
    /// reading the file, not merely permit it.
    #[test]
    fn opening_prompt_requires_reading_the_whole_file() {
        let p = opening_prompt("CHANGELOG.md", 40);
        assert!(
            p.contains("entire") || p.contains("whole"),
            "must ask for the whole file, not an unspecified amount: {p:?}",
        );
        assert!(
            p.contains("only this file"),
            "must scope the read to this file alone: {p:?}",
        );
    }

    /// **The prompt must ask for the calls in ONE reply, and must name the cap.**
    ///
    /// A round is a decode plus a tool dispatch, so a file read one call per reply
    /// costs a decode per {MAX_READ_LINES} lines — ten for a 2,000-line file where
    /// one reply carrying ten calls costs one. The loop has always run every call
    /// an answer makes (`tool_round::plan` → `tool_round::run`); it was the prompt
    /// that asked for them one at a time. Naming the cap is what lets the model
    /// pick the ranges up front instead of learning where the cut fell from each
    /// excerpt header in turn.
    #[test]
    fn opening_prompt_asks_for_parallel_reads_and_names_the_cap() {
        let p = opening_prompt("src/big.rs", 2000);
        assert!(
            p.contains(&MAX_READ_LINES.to_string()),
            "must name the per-call line cap so ranges can be chosen up front: {p:?}",
        );
        assert!(
            p.contains("SAME reply"),
            "must ask for several calls in one reply: {p:?}",
        );
        assert!(
            p.contains("instead of one call per reply"),
            "must say what it is asking INSTEAD of — one call per reply is the \
             behaviour this prompt exists to replace: {p:?}",
        );
    }
}
