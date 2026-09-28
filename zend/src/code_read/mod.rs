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
//! What to read comes from the branches (`crate::branch_ingest`): each file
//! is keyed by its path and blob id, a conversation carrying that key is the
//! file read, and one is ingested only when no conversation carries it. Its
//! `file_read` calls read the commit it was found on, so the conversation
//! holds exactly the bytes its key names.
//!
//! **Parallel ingest.** [`CODE_READ_PARALLELISM`] workers each own one file's
//! conversation and drive it to completion — a bounded pool of real
//! sequences, exactly like any other batch of concurrent conversations the
//! engine wave-batches together.

mod lines;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::chain_health::{chain_break, ChainBreak};
use candle_conversation::projection::{
    OptionalState, SelectionState, TimelineId, FORCE_TOOL_SELECTOR, NO_THINK_SELECTOR,
    TOOLS_ENABLED_SELECTOR,
};
use candle_conversation::stencil::TriggerRegistry;
use candle_conversation::{ConversationEngine, Sequence, TurnOptions, TurnText};
use zend_tools::ToolContext;
use zend_vfs::vfs::PAGE_LINES;
use zend_vfs::Oid;

use self::lines::line_count;
use crate::branch_ingest::filter::{language_of, MAX_FILE_BYTES};
use crate::branch_ingest::keys::{file_key, BLOB_KEY, CONTENT_KEY, LINES_KEY};
use crate::branch_ingest::plan::Committed;
use crate::ingest_report::Failures;
use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::repo_path::split;
use crate::repo_scan::{is_binary_sample, Language};
use crate::tool_round;
use crate::tools::format_tool_responses;
use crate::workspace::UPLOADS_REPO;

/// Metadata key holding a file conversation's workspace-relative path.
pub(crate) const PATH_KEY: &str = "path";

/// One file to read: its workspace-relative path, its blob, its language, and
/// the commit its bytes are read from — `None` for an upload, which is read
/// from the uploads folder as it stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileJob {
    pub path: String,
    pub blob: Oid,
    pub language: Language,
    pub at: Option<Oid>,
}

impl FileJob {
    /// The file's content key.
    pub fn key(&self) -> String {
        file_key(&self.path, &self.blob)
    }
}

/// Every committed `code_reading` conversation of a repository's branches
/// whose chain FINISHED: its content key and path — what the pass plans
/// against. Uploads are not among them: their endpoint keeps them, never the
/// pass.
///
/// # A file counts as read only once its chain finished
///
/// The content key is written once the file's conversation stops calling
/// tools, and that is not the same thing as answering. A README read spent its
/// whole decode budget deliberating over how many `file_read` calls to issue,
/// was cut off mid-word, and emitted no tool call — so no coupling, no second
/// turn and no summary, yet the turn came back `Ok` and the key was written.
/// The substrate stored it faithfully (CRC-clean, chunk counts consistent), the
/// file counted as read forever, and its one turn stayed in the corpus as a
/// worked example of an assistant that deliberates and produces nothing, which
/// later conversations then imitated.
///
/// So each keyed timeline is asked whether its chain finished
/// ([`chain_break`]). An unfinished one is left out of what the pass plans
/// against, so its key reads as held by nothing and the file is queued again;
/// [`process_one_file`] defers the unfinished generation and retires it only
/// once the rebuild commits — recovery with nothing lost in between.
///
/// **One predicate for every gate.** The plan (this), the resume snapshot
/// ([`finished_content_keys`]) and the fast path ([`chain_finished`]) each ask
/// "is this file already read?" from their own query. A gate left reading the
/// bare key would announce the file as queued and then skip it as present,
/// which reports recovery that never happens — or, for the fast path, hands the
/// unfinished chain to a live conversation as a file it has read.
pub fn committed(engine: &Mutex<ConversationEngine>) -> Vec<Committed> {
    let scan = scan_ingest_chains(engine);
    // One line per unfinished chain per pass. A handful is ordinary recovery; a
    // corpus-wide sweep of them means the decode budget or the opening prompt
    // is wrong for this model, and every pass will keep redoing the same work.
    for (path, why) in &scan.broken {
        if scan.finished.iter().any(|c| c.subject == *path) {
            continue; // a complete chain for the same file covers it
        }
        tracing::info!(
            target: "zend::code_read::ingest",
            file = %path,
            ?why,
            "ingest chain unfinished — the file is queued again and will be rebuilt",
        );
    }
    scan.finished
        .into_iter()
        .filter(|c| !is_upload_path(&c.subject))
        .collect()
}

/// What [`scan_ingest_chains`] found: the keyed conversations whose chain
/// finished, and the paths whose chain stopped part-way.
struct IngestChains {
    finished: Vec<Committed>,
    broken: Vec<(String, ChainBreak)>,
}

/// Every keyed `code_reading` conversation, uploads included, sorted by
/// whether the chain that wrote it finished ([`chain_break`]).
fn scan_ingest_chains(engine: &Mutex<ConversationEngine>) -> IngestChains {
    let eng = engine.lock().unwrap();
    let keys: HashMap<TimelineId, String> = eng
        .conversations_with_metadata_key(CONTENT_KEY)
        .into_iter()
        .collect();
    let conv = eng.conversation();
    let substrate = conv.read();
    let mut out = IngestChains {
        finished: Vec::new(),
        broken: Vec::new(),
    };
    for (timeline, path) in eng.conversations_with_metadata_key(PATH_KEY) {
        let Some(key) = keys.get(&timeline) else {
            continue;
        };
        match chain_break(&substrate, timeline) {
            Some(why) => out.broken.push((path, why)),
            None => out.finished.push(Committed {
                timeline,
                key: key.clone(),
                subject: path,
            }),
        }
    }
    out
}

/// Content keys whose ingest chain finished, uploads included — the snapshot
/// [`ingest_jobs`] probes before spending a conversation on a file.
///
/// Replaces a bare sweep of every content key, which counted an unfinished
/// chain's key as present and so skipped the very file the plan had just
/// queued for rebuild.
fn finished_content_keys(engine: &Mutex<ConversationEngine>) -> HashSet<String> {
    scan_ingest_chains(engine)
        .finished
        .into_iter()
        .map(|c| c.key)
        .collect()
}

/// Whether `timeline`'s ingest chain finished — the fast path's gate before it
/// hands a file conversation to a live one as already read (see [`committed`]).
pub(crate) fn chain_finished(engine: &ConversationEngine, timeline: TimelineId) -> bool {
    chain_break(&engine.conversation().read(), timeline).is_none()
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
/// forced to answer on whatever it has already seen. The tool returns one
/// [`PAGE_LINES`]-line page per call (`zend-vfs`), so a genuinely large
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
/// per-call cap leaves the model to guess how many calls cover the file, and a
/// guess is a decision it can spend its whole decode budget failing to make: a
/// README read deliberated over whether the file was under the cap, whether a
/// read past EOF would error, and whether to risk it — until the token budget
/// cut it off mid-word, before any `file_read` call was emitted. It produced no
/// tool call, so no coupling and no second turn, and the chain stood in the
/// corpus as an assistant that deliberates and answers nothing. With the length
/// given, the arithmetic is settled before the model starts.
///
/// **It asks for the calls in PARALLEL, and that is a round-count decision.**
/// `file_read` serves one [`PAGE_LINES`]-line page per call, so a long file
/// needs several — and a round is one decode plus one tool dispatch, so reading
/// a 2,000-line file one call per turn costs ten decodes where a single turn
/// carrying ten calls costs one. The loop already supports it in full:
/// `tool_round::plan` returns every call an answer makes, `tool_round::run`
/// executes them in order, and `format_tool_responses` hands back one
/// `<tool_response>` block per result, so the whole file arrives in the next
/// turn's context together. Naming the page size and the page count in the
/// prompt is what lets the model issue every call at once rather than
/// discovering the file's length from the first page's header.
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
///
/// `key` is workspace-relative; the prompt names the repository and the path
/// inside it separately, the two arguments the call it asks for takes.
fn opening_prompt(key: &str, lines: usize) -> String {
    let (repo, path) = split(key);
    let pages = lines.div_ceil(PAGE_LINES as usize).max(1);
    let last = pages - 1;
    format!(
        "Read the entire contents of `{path}` in the `{repo}` repository — and only \
         this file — using {FILE_READ_TOOL}. The file is {lines} lines long and each \
         call returns one {PAGE_LINES}-line page, so it takes exactly {pages} \
         {FILE_READ_TOOL} call(s): issue all {pages} in the SAME reply, one per page \
         from page 0 to page {last}, instead of one call per reply. \
         Every call you make in a reply is run together and all of their results \
         come back to you at once. Once you've read the whole thing, summarize what \
         it contains: its purpose, its main structures or functions, and how it fits \
         into the codebase."
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

/// Worker count for the parallel ingest — [`CODE_READ_PARALLELISM`].
fn parallelism() -> usize {
    CODE_READ_PARALLELISM
}

/// Ingest ONLY `rel_paths` — workspace-relative, in the uploads repository —
/// into the `code_reading` layer: the upload pipeline's read_file phase.
///
/// Bounded to these files: no branch is walked and nothing else is
/// tombstoned. Each file is keyed the way a branch's file is — its path and
/// blob id, the id computed from the bytes on disk — so identical bytes
/// uploaded again are not read again, and an upload that replaces a path
/// retires the conversation of what that path held before.
///
/// Files whose extension isn't a recognised code language, and files over
/// [`MAX_FILE_BYTES`], are skipped. Returns whether any file was read, and
/// how many files' ingest tolerated-failed (e.g. out of KV VRAM), so the
/// upload can surface a real failure.
pub fn ingest_files(
    ctx: &RefreshContext<'_>,
    rel_paths: &[String],
    progress: &Arc<LoadProgress>,
    layer_name: &str,
    base: &Mutex<Sequence>,
) -> anyhow::Result<(bool, usize)> {
    let uploads = ctx.tool_ctx.files.repo(UPLOADS_REPO)?;
    let mut jobs = Vec::new();
    for rel in rel_paths {
        let path = rel.replace('\\', "/");
        let (_, inner) = split(&path);
        let Some(language) = language_of(&path) else {
            continue; // not a recognised code language — nothing to read
        };
        match uploads.content_id(inner) {
            Ok(Some((blob, size))) if size <= MAX_FILE_BYTES => jobs.push(FileJob {
                path,
                blob,
                language,
                at: None,
            }),
            Ok(_) => {
                tracing::debug!(file = %path, "code_read: skip an upload missing or over the size cap");
            }
            Err(e) => tracing::debug!(file = %path, "code_read: skip an unreadable upload: {e}"),
        }
    }
    if jobs.is_empty() {
        return Ok((false, 0));
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

    let live: HashSet<String> = jobs.iter().map(FileJob::key).collect();
    let n_failed = ingest_jobs(ctx, &jobs, &live, progress, base, &Mutex::default())?;
    Ok((true, n_failed))
}

/// Whether a workspace-relative `path` (with `/` separators) lives in the
/// daemon's `uploads` repository ([`UPLOADS_REPO`]). Matched on the FIRST
/// segment only, and case-insensitively (the win32 FS is case-insensitive, so
/// an existing `Uploads/` dir still resolves to the daemon's uploads dir) — so
/// an `uploads/` folder inside a repository is NOT matched. Uploads are
/// endpoint-managed and on no branch, so the branch pass must never read their
/// absence from the branches as a deletion.
pub(crate) fn is_upload_path(path: &str) -> bool {
    path.split('/')
        .next()
        .unwrap_or("")
        .eq_ignore_ascii_case(UPLOADS_REPO)
}

/// Retire every crashed-partial `code_read` conversation, up front.
///
/// The `code_read` twin of `repo_scan::retire_crashed_partials`, and the same
/// completion protocol: `path` is written at conversation creation and the
/// content key only once the file's ingest succeeds, so `path` without a key
/// means "started, never committed". A conversation ingested before files
/// were keyed by content carries no key either, and goes the same way. An
/// attempt that fails retires its own conversation ([`process_one_file`]), so
/// what is left for here is what a crash left: the debris would otherwise
/// stay live and keep competing in the provenance gather.
///
/// **Uploads are not exempt.** An upload's conversation without a content
/// key is in no conversation's scope (`crate::retrieval_scope`) — it holds
/// nothing anything retrieves.
///
/// Called ONCE per boot from the session's ingest pre-loop, for every layer not
/// named by `--disable-layer`, before any pool or upload runs. Never called
/// from [`ingest_jobs`]: a pass can overlap a live pool, and an in-flight file
/// is indistinguishable from a crashed one by metadata alone.
pub(crate) fn retire_crashed_partials(engine: &Mutex<ConversationEngine>) {
    let e = engine.lock().unwrap();
    let mut retired = 0usize;
    for (tl, path) in e.conversations_with_metadata_key(PATH_KEY) {
        let committed = e
            .conversation_metadata(tl)
            .is_some_and(|m| m.contains_key(CONTENT_KEY));
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
            "retired code_read conversations with no content key (crashed partials, \
             or ingested before files were keyed by content) so they leave the \
             provenance gather",
        );
    }
}

/// Tombstone EVERY `code_reading` conversation, committed or not —
/// `--wipe-layer code_reading`'s targeted counterpart to
/// [`retire_crashed_partials`], which only removes the never-committed half.
///
/// **Uploads are exempt.** They live in the endpoint-managed uploads folder,
/// on no branch, so no pass reads them again: tombstoning one would delete
/// uploaded content with nothing to rebuild it from.
///
/// Called from the session's ingest pre-loop, before the background ingest
/// worker's first pass — which then finds nothing committed and reads every
/// file as new, exactly as it would on a truly fresh install. Unlike
/// `--wipe-substrate`, every other layer's content survives untouched.
pub(crate) fn wipe_layer(engine: &Mutex<ConversationEngine>) {
    let e = engine.lock().unwrap();
    let mut wiped = 0usize;
    for (tl, path) in e.conversations_with_metadata_key(PATH_KEY) {
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

/// Ingest `jobs`: a bounded worker pool, each worker pulling the next file
/// from a shared cursor and running [`process_one_file`]. Workers share
/// progress / decode-failure counters and an abort flag (first error stops
/// the rest). Returns once every file is processed, yielding the number of
/// files whose ingest was *tolerated-failed* (e.g. the GPU ran out of KV VRAM
/// mid-decode) — so the upload can surface a real failure instead of a silent
/// "done".
///
/// `live` is every file key the caller holds current: when a file commits,
/// each other conversation of its path whose key is not in it is tombstoned.
/// `binary` remembers the keys found to be binary, so a later pass does not
/// read them again.
///
/// The branch pass and the upload path ([`ingest_files`]) both funnel
/// through here, so both drive the same `crate::ingest_backlog` counter and
/// get this same logging.
pub fn ingest_jobs(
    ctx: &RefreshContext<'_>,
    jobs: &[FileJob],
    live: &HashSet<String>,
    progress: &Arc<LoadProgress>,
    base: &Mutex<Sequence>,
    binary: &Mutex<HashSet<String>>,
) -> anyhow::Result<usize> {
    let n_workers = parallelism();
    let total = jobs.len();
    let keys: Vec<String> = jobs.iter().map(FileJob::key).collect();
    // One snapshot of the committed keys: a file committed since the caller
    // planned is not read twice. Finished chains only (see `committed`), so a
    // file queued for rebuild is not skipped here as present.
    let present_keys = finished_content_keys(ctx.engine);
    tracing::info!(
        n_workers = n_workers,
        n_files = total,
        "code_read: per-file ingest across {n_workers} file workers; each file is \
         a real hidden conversation that reads the file via file_read and answers \
         with its own summary",
    );

    // The GUI's merged background-ingest bar counts only files that will
    // REALLY run. Same snapshot every worker probes below, so registration
    // and completion can't disagree.
    let backlog_pending = keys.iter().filter(|k| !present_keys.contains(*k)).count() as u64;
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
                if idx >= jobs.len() {
                    return;
                }
                let (file, key) = (&jobs[idx], &keys[idx]);
                let job = Job {
                    file,
                    key,
                    live,
                    binary,
                };
                if let Err(e) = process_one_file(ctx, base, &job, &present_keys, &failures) {
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
                if !present_keys.contains(key) {
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

/// One file for [`process_one_file`], with what the pool shares across files.
struct Job<'a> {
    file: &'a FileJob,
    /// The file's content key.
    key: &'a str,
    /// Every key the caller holds current.
    live: &'a HashSet<String>,
    /// Keys found to be binary, kept for the life of the process.
    binary: &'a Mutex<HashSet<String>>,
}

/// The tools a file's conversation runs against: file stores reading the
/// commit the file was found on, so its `file_read` returns the bytes its key
/// names — or, for an upload, a fresh set over the uploads folder. `None`
/// when the file's repository is not read through git.
fn file_tools(tools: &ToolContext, file: &FileJob) -> Option<ToolContext> {
    let files = match &file.at {
        Some(at) => tools.files.fresh_at(split(&file.path).0, at)?,
        None => tools.files.fresh(),
    };
    Some(tools.with_files(Arc::new(files)))
}

/// Ingest one file into a fresh per-file conversation: skip it when its key
/// is already committed or it is binary; otherwise mint the conversation, run
/// it as a real tool-using exchange ([`run_file_conversation`]), tag it with
/// its content key + metadata, then drop it (freeing the GPU slot; the sealed
/// turns + tags persist in the substrate).
fn process_one_file(
    ctx: &RefreshContext<'_>,
    base: &Mutex<Sequence>,
    job: &Job<'_>,
    present_keys: &HashSet<String>,
    failures: &Failures,
) -> anyhow::Result<()> {
    let file = job.file;
    if present_keys.contains(job.key) {
        tracing::debug!(
            target: "zend::code_read::ingest",
            file = %file.path,
            "skip: file already committed",
        );
        return Ok(());
    }
    if job.binary.lock().unwrap().contains(job.key) {
        return Ok(());
    }
    let Some(tools) = file_tools(&ctx.tool_ctx, file) else {
        failures.record(
            &file.path,
            "its repository is not read through git".to_string(),
        );
        return Ok(());
    };
    // Content guard: an allowlisted extension does NOT guarantee text. A
    // compiled fatbin / object dump committed as `*.txt` clears both the
    // extension gate and the size gate, and a hidden conversation told to
    // read it would read noise. Sniffed from the blob before any
    // conversation is minted, and remembered, so it is read once. The same
    // bytes give the line count a fast-path answer reports.
    let (repo, inner) = split(&file.path);
    let bytes = tools
        .files
        .repo(repo)
        .ok()
        .and_then(|store| store.read_bytes(inner).ok().flatten());
    let lines = match bytes {
        Some(bytes) if is_binary_sample(&bytes) => {
            tracing::debug!(
                target: "zend::code_read::ingest",
                file = %file.path,
                "skip: binary content behind a text extension",
            );
            job.binary.lock().unwrap().insert(job.key.to_string());
            return Ok(());
        }
        Some(bytes) => line_count(&bytes),
        None => {
            failures.record(
                &file.path,
                "the file could not be read at its commit".to_string(),
            );
            return Ok(());
        }
    };

    // Reconcile the existing conversations for this path WITHOUT invalidating
    // good content up front — a DEFERRED tombstone:
    //   * a committed generation whose key no branch holds any more is
    //     DEFERRED into `superseded`: it stays live as the file's fallback
    //     content, so a failed re-ingest below (e.g. a VRAM OOM) leaves it
    //     intact instead of destroying it. Its tombstone ACTIVATES only after
    //     this ingest commits its own key (see the success path below) — an
    //     atomic swap, "stale-but-present" over "gone";
    //   * a generation carrying THIS file's key is one whose chain never
    //     finished — a finished one would have been in `present_keys` and
    //     skipped above — so it is deferred the same way: the rebuild replaces
    //     it, and until the rebuild commits nothing is lost;
    //   * a committed generation whose key is still live is the file as
    //     another branch holds it, and is left alone;
    //   * one with no content key yet is another worker's, reading the file
    //     as another branch holds it, and is left alone too — an attempt
    //     that fails retires its own, and a crash's are retired at boot
    //     ([`retire_crashed_partials`]).
    // The engine lock covers only these quick ops and is released before the
    // decode-heavy body below.
    let (mut conv, superseded) = {
        let e = ctx.engine.lock().unwrap();
        let superseded: Vec<TimelineId> = e
            .find_conversations_by_metadata(PATH_KEY, &file.path)
            .into_iter()
            .filter(|&tl| {
                e.conversation_metadata(tl)
                    .and_then(|m| m.get(CONTENT_KEY).cloned())
                    .is_some_and(|key| key == job.key || !job.live.contains(&key))
            })
            .collect();
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
        ctx.engine
            .lock()
            .unwrap()
            .set_timeline_summarize(conv.timeline_id(), false);
        (conv, superseded)
    };

    // Tag the `path` IMMEDIATELY — before the decode-heavy run below that can
    // fail (GPU OOM mid-decode, a decode error). A partial left by a crash then
    // still carries its path, so it (a) shows in the substrate as the file it
    // covers rather than "(untitled)", and (b) is found by the next boot's
    // [`retire_crashed_partials`]. The content key is deliberately withheld
    // until success (below), so a partial is never mistaken for a completed
    // ingest and skipped.
    {
        let mut early = BTreeMap::new();
        early.insert("kind".to_string(), "code_read".to_string());
        early.insert(PATH_KEY.to_string(), file.path.clone());
        if let Err(e) = conv.set_metadata_many(&early) {
            tracing::warn!(
                target: "zend::code_read::ingest",
                file = %file.path,
                "failed to tag path metadata at conversation creation: {e:#}",
            );
        }
    }

    // Each file's conversation is a conversation like any other, with file
    // stores of its own — reading the commit the file was found on: nothing
    // another unit, or a live dialogue, changed is what it reads.
    let unit_ctx = Arc::new(tools);
    let summary = match run_file_conversation(
        &mut conv,
        &file.path,
        lines,
        &ctx.think_triggers,
        &unit_ctx,
    ) {
        Ok(text) => text,
        Err(e) => {
            // The deferred tombstone is the safety net here: the prior good
            // generation in `superseded` was NEVER tombstoned, so it stays
            // live as the file's content — this failed attempt invalidates
            // nothing. Drop only THIS attempt's partial; the retry re-mints
            // cleanly.
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

    // Tag the conversation: the content key commits it — what the pass plans
    // against, the fast path finds and retrieval scopes by — `path` and `blob`
    // are its parts, `lines` is what a fast-path answer reports, the rest is
    // diagnostic. One record, so the line count lands with the key.
    let mut tags = BTreeMap::new();
    tags.insert("kind".to_string(), "code_read".to_string());
    tags.insert(PATH_KEY.to_string(), file.path.clone());
    tags.insert(BLOB_KEY.to_string(), file.blob.to_string());
    tags.insert(LINES_KEY.to_string(), lines.to_string());
    tags.insert(CONTENT_KEY.to_string(), job.key.to_string());
    tags.insert("lang".to_string(), format!("{:?}", file.language));
    if let Err(err) = conv.set_metadata_many(&tags) {
        // Not committed: no pass, fast path or scope will find it. The prior
        // generation stays live, exactly as on the failure path, and this
        // attempt's conversation goes — nothing else retires it before the
        // next boot.
        tracing::warn!(
            target: "zend::code_read::ingest",
            file = %file.path,
            "failed to tag conversation metadata (content key): {err:#}",
        );
        if let Err(err) = ctx
            .engine
            .lock()
            .unwrap()
            .tombstone_timeline(conv.timeline_id())
        {
            tracing::warn!(
                target: "zend::code_read::ingest",
                file = %file.path,
                "tombstone of the untagged attempt failed: {err:#}",
            );
        }
        failures.record(
            &file.path,
            format!("the content key was not written: {err:#}"),
        );
        return Ok(());
    }
    ctx.retrieval.mark_stale();

    // Deferred tombstone ACTIVATES, now that the new generation is committed.
    if !superseded.is_empty() {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A file job's key is the branch walk's key for the same file — the one
    /// the pass plans with and the fast path looks up.
    #[test]
    fn a_file_jobs_key_is_its_path_and_blob() {
        let job = FileJob {
            path: "candle/src/lib.rs".into(),
            blob: Oid::parse("ce013625030ba8dba906f756967f9e9ca394464a").unwrap(),
            language: Language::Rust,
            at: None,
        };
        assert_eq!(
            job.key(),
            "candle/src/lib.rs@ce013625030ba8dba906f756967f9e9ca394464a"
        );
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
    fn opening_prompt_names_the_path_and_the_one_tool() {
        let p = opening_prompt("candle/src/lib.rs", 120);
        assert!(
            p.starts_with("Read the entire contents of `src/lib.rs` in the `candle` repository")
        );
        assert!(p.contains(FILE_READ_TOOL));
    }

    /// **The prompt must state the file's length and the exact number of calls.**
    ///
    /// Told only the per-call cap, a model has to guess how many calls cover the
    /// file — and that guess is a decision it can spend its entire decode budget
    /// failing to reach. A README read did exactly that: it weighed whether the
    /// file was under the cap and whether a read past EOF would error until the
    /// budget cut it off mid-word, with no `file_read` call emitted, no coupling,
    /// no second turn and no summary. The arithmetic belongs in the prompt.
    ///
    /// Pages are 200 lines and 0-based, so the boundaries are pinned as text.
    #[test]
    fn opening_prompt_states_the_length_and_the_call_count() {
        assert_eq!(
            PAGE_LINES, 200,
            "the expected text below is written for 200-line pages"
        );
        // Exactly one page's worth: one call, not two.
        let one = opening_prompt("candle/src/small.rs", 200);
        assert!(
            one.contains(
                "The file is 200 lines long and each call returns one 200-line page, \
                 so it takes exactly 1 file_read call(s): issue all 1 in the SAME \
                 reply, one per page from page 0 to page 0,"
            ),
            "{one:?}",
        );
        // One line over: two calls, pages 0 and 1.
        let two = opening_prompt("candle/src/mid.rs", 201);
        assert!(
            two.contains(
                "The file is 201 lines long and each call returns one 200-line page, \
                 so it takes exactly 2 file_read call(s): issue all 2 in the SAME \
                 reply, one per page from page 0 to page 1,"
            ),
            "{two:?}",
        );
        // An empty file still asks for one call rather than zero.
        let empty = opening_prompt("candle/src/empty.rs", 0);
        assert!(
            empty.contains("exactly 1 file_read call(s)") && empty.contains("page 0 to page 0,"),
            "never zero calls: {empty:?}",
        );
    }

    /// The measured failure this prompt exists to close: left as "read as much
    /// as you need", a model sometimes skipped `file_read` entirely and
    /// guessed a summary from the filename alone. The prompt must require
    /// reading the file, not merely permit it.
    #[test]
    fn opening_prompt_requires_reading_the_whole_file() {
        let p = opening_prompt("candle/CHANGELOG.md", 40);
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
    /// costs a decode per page — ten for a 2,000-line file where one reply
    /// carrying ten calls costs one. The loop has always run every call an answer
    /// makes (`tool_round::plan` → `tool_round::run`); it was the prompt that
    /// asked for them one at a time. Naming the page size and count is what lets
    /// the model issue every call up front instead of learning the file's length
    /// from the first page's header.
    #[test]
    fn opening_prompt_asks_for_parallel_reads_and_names_the_cap() {
        let p = opening_prompt("candle/src/big.rs", 2000);
        assert!(
            p.contains("one 200-line page, so it takes exactly 10 file_read call(s)"),
            "must name the page size and the call count so every call can be issued \
             up front: {p:?}",
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
