//! The bounded worker pool that seals each queued file into the knowledge
//! base, and the upload path that seals a set of uploaded files.
//!
//! Each worker takes the next file from a shared cursor, and each member walks
//! the queue in an order of its own, so two members planning the same tree meet
//! on few files. A file another member holds the lease for is left to that
//! member and is come back to in a later round. Per-file failures are recorded
//! rather than propagated, so a bad file degrades the map and the pass goes on,
//! while the failure cap stops a flood as reported state. Uploads are keyed by
//! their bytes, so identical bytes uploaded again are not read again.

use super::*;

/// One file to read: its workspace-relative path, its blob, its language, and
/// the commit its bytes are read from — `None` for an upload, which is read
/// from the uploads folder as it stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileJob {
    pub path: String,
    pub blob: Oid,
    pub language: Language,
    pub at: Option<Oid>,
    /// Every branch whose tip holds this version, recorded on the
    /// conversation ([`BRANCHES_KEY`]); empty for an upload, which is on none.
    pub branches: Vec<String>,
}

impl FileJob {
    /// The file's content key.
    pub fn key(&self) -> String {
        file_key(&self.path, &self.blob)
    }
}

/// Worker count for the parallel ingest — [`CODE_READ_PARALLELISM`].
pub(super) fn parallelism() -> usize {
    CODE_READ_PARALLELISM
}

/// Ingest ONLY `rel_paths` — workspace-relative, in the uploads repository of
/// the conversation whose stores `ctx`'s tools hold — into the `code_reading`
/// layer: the upload pipeline's read_file phase.
///
/// Bounded to these files: no branch is walked and nothing else is
/// tombstoned. Each file is keyed the way a branch's file is — its path and
/// blob id, the id computed from the bytes it holds — so identical bytes
/// uploaded again are not read again, and an upload that replaces a path
/// retires the conversation of what that path held before.
///
/// Files whose extension isn't a recognised code language, and files over
/// [`MAX_FILE_BYTES`], are skipped. Returns whether any file was read, and
/// how many files' ingest tolerated-failed (e.g. out of KV VRAM), so the
/// upload can surface a real failure. The calibration sweep calibrates them
/// after, as it does every file (`Session::calibrate_layers`).
pub fn ingest_files(
    ctx: &RefreshContext<'_>,
    rel_paths: &[String],
    progress: &Arc<LoadProgress>,
    layer_name: &str,
    base: &Sequence,
) -> anyhow::Result<(bool, usize)> {
    let uploads = ctx.tool_ctx.files.repo(UPLOADS_REPO)?;
    let mut jobs = Vec::new();
    for rel in rel_paths {
        let path = rel.replace('\\', "/");
        let (_, inner) = split(&path);
        let Some(language) = language_of(&path) else {
            continue; // not a recognised code language — nothing to read
        };
        // An upload is the conversation's, in its store's session layer: its
        // id is computed from its bytes, as git would.
        match uploads.read_bytes(inner) {
            Ok(Some(bytes)) if bytes.len() as u64 <= MAX_FILE_BYTES => jobs.push(FileJob {
                path,
                blob: ObjectFormat::Sha1.blob_id(&bytes),
                language,
                at: None,
                branches: Vec::new(),
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
    ctx.engine.mark_layer_append_only(layer);

    let live: HashSet<String> = jobs.iter().map(FileJob::key).collect();
    let n_failed = ingest_jobs(ctx, &jobs, &live, progress, base, &Mutex::default())?;
    Ok((true, n_failed))
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
    base: &Sequence,
    binary: &Mutex<HashSet<String>>,
) -> anyhow::Result<usize> {
    let n_workers = parallelism();
    let total = jobs.len();
    let keys: Vec<String> = jobs.iter().map(FileJob::key).collect();
    // The recorded files restored from their records first, together: each
    // one restored whole is sealed here, and the snapshot below skips it.
    if boot_restore::restore_recorded(ctx.engine, base, &keys) > 0 {
        ctx.retrieval.mark_stale();
    }
    // One snapshot of the keys sealed here: a file sealed since the caller
    // planned is not sealed twice.
    let present_keys = sealed_keys(ctx.engine);
    tracing::info!(
        n_workers = n_workers,
        n_files = total,
        "code_read: per-file ingest across {n_workers} file workers; each file is \
         cut and prefilled as the knowledge base's members",
    );

    // The GUI's merged background-ingest bar counts only files that will
    // REALLY run. Same snapshot every worker probes below, so registration
    // and completion can't disagree.
    let backlog_pending = keys.iter().filter(|k| !present_keys.contains(*k)).count() as u64;
    ingest_backlog::add_pending(Work::Reading, backlog_pending);
    let backlog_done = AtomicUsize::new(0);

    // **Each member walks the queue in an order of its own**, so two members
    // planning the same unread tree meet on few files: walked in the same
    // order, both pools reached the same files at the same moment, and every
    // file one of them held the other passed over.
    // The files a drawing group draws — the guides and journals — lead: they
    // place nothing until they are read.
    let order = ingest_lease::order(&keys, |i| ctx.retrieval.draws(&jobs[i].path));
    // What the first round left to another member's lease, read in the
    // second once the rest is done: a holder that died or restarted has let
    // its lease lapse by then, and one that read it has committed it. One a
    // live holder still keeps is left again and counted done here, as every
    // file left to a holder was before: the holder's commit lands it.
    let left_over: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    // The files whose attempt found no ground, resubmitted once the round that
    // refused them is done ([`GROUND_RESUBMITS`]).
    let resubmitted: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    let done = AtomicUsize::new(0);
    // Per-file failures are RECORDED, never propagated: the file keeps its prior
    // generation live and the pass carries on, so a bad file (or a systemic VRAM
    // squeeze) degrades the map instead of killing the daemon. The cap still
    // stops a flood, as reported state rather than a fatal error. See
    // `crate::ingest_report`.
    let failures = Failures::new();
    progress.set_step_progress(0, total as u64);

    let round = |indices: &[usize], last: bool, may_resubmit: bool| {
        let cursor = AtomicUsize::new(0);
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(n_workers);
            for _ in 0..n_workers.max(1) {
                handles.push(s.spawn(|| loop {
                    // Stop pulling new files on first-error abort OR a shutdown
                    // cancel. The current file finishes (the scheduler is still
                    // live), so no half-ingested file; the loader thread then
                    // drains the engine.
                    if failures.aborted() || crate::ingest_cancel::cancelled() {
                        return;
                    }
                    let at = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(&idx) = indices.get(at) else {
                        return;
                    };
                    let (file, key) = (&jobs[idx], &keys[idx]);
                    let left = AtomicBool::new(false);
                    let resubmit = AtomicBool::new(false);
                    let job = Job {
                        file,
                        key,
                        live,
                        binary,
                        left: &left,
                        resubmit: may_resubmit.then_some(&resubmit),
                    };
                    if let Err(e) = process_one_file(ctx, base, &job, &present_keys, &failures) {
                        // An error escaping `process_one_file` is an unexpected
                        // one (its own failure mode records and returns Ok).
                        // Record it so it reaches the report instead of
                        // vanishing, and let the cap decide whether to stop the
                        // pass.
                        let n = failures.record(&file.path, format!("{e:#}"));
                        tracing::warn!(
                            target: "zend::code_read::ingest",
                            file = %file.path,
                            "file ingest failed (will retry next run): {e:#}",
                        );
                        ingest_holdoff::cool_down(&file.path, &format!("{e:#}"));
                        if n > MAX_DECODE_FAILURES {
                            failures.set_abort();
                        }
                    }
                    // Left to its holder in the first round, the file is come
                    // back to, and is not done yet.
                    if !last && left.load(Ordering::Relaxed) {
                        left_over.lock().unwrap().push(idx);
                        continue;
                    }
                    // Refused for want of ground, it goes back through admission
                    // after this round, and is not done yet either.
                    if resubmit.load(Ordering::Relaxed) {
                        resubmitted.lock().unwrap().push(idx);
                        continue;
                    }
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    progress.set_step_progress(d as u64, total as u64);
                    // A registered file is "done" on every exit from
                    // `process_one_file` above — success, tolerated failure, or
                    // a mid-file shutdown cancel — so the backlog can never
                    // wedge non-empty on a file that will just be retried next
                    // pass.
                    if !present_keys.contains(key) {
                        backlog_done.fetch_add(1, Ordering::Relaxed);
                        ingest_backlog::item_done(Work::Reading, &file.path);
                    }
                }));
            }
            for h in handles {
                h.join().expect("code_read worker panicked");
            }
        });
    };
    round(&order, false, true);
    let again = std::mem::take(&mut *left_over.lock().unwrap());
    if !again.is_empty() {
        tracing::info!(
            target: "zend::code_read::ingest",
            files = again.len(),
            "coming back to the files left to another member's lease",
        );
    }
    round(&again, true, true);
    for remaining in (0..GROUND_RESUBMITS).rev() {
        let again = std::mem::take(&mut *resubmitted.lock().unwrap());
        if again.is_empty() {
            break;
        }
        tracing::info!(
            target: "zend::code_read::ingest",
            files = again.len(),
            rounds_left = remaining,
            "resubmitting the files whose attempt found no ground, now the round that \
             refused them has given its ground back",
        );
        round(&again, true, remaining > 0);
    }

    // An abort or a shutdown cancel can leave files claimed but never run —
    // hand them back, or the GUI's backlog bar never reaches zero.
    ingest_backlog::drop_pending(
        Work::Reading,
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
pub(super) struct Job<'a> {
    pub(super) file: &'a FileJob,
    /// The file's content key.
    pub(super) key: &'a str,
    /// Every key the caller holds current.
    pub(super) live: &'a HashSet<String>,
    /// Keys found to be binary, kept for the life of the process.
    pub(super) binary: &'a Mutex<HashSet<String>>,
    /// Set when the file is left to the member holding its lease, so the pass
    /// comes back to it once the rest is read.
    pub(super) left: &'a AtomicBool,
    /// Set when the file's attempt found no ground and was dropped, so the pass
    /// resubmits it after this round; `None` in the last round that may, where
    /// such an attempt is a failure like any other.
    pub(super) resubmit: Option<&'a AtomicBool>,
}

/// The tools a file's conversation runs against: file stores reading the
/// commit the file was found on, so its `file_read` returns the bytes its key
/// names — or, for an upload, the uploading conversation's own stores, which
/// hold it. `None` when the file's repository is not read through git.
pub(super) fn file_tools(tools: &ToolContext, file: &FileJob) -> Option<ToolContext> {
    let files = match &file.at {
        Some(at) => Arc::new(tools.files.fresh_at(split(&file.path).0, at)?),
        None => Arc::clone(&tools.files),
    };
    Some(tools.with_files(files, ""))
}
