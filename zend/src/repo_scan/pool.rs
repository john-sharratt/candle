//! The bounded worker pool that seals a pass's folders, recording any failure
//! without ending the pass.
//!
//! Each worker claims its next folder from a shared cursor. A folder another
//! member holds the lease for is left to that member and come back to once
//! the rest is done. A failed folder keeps its earlier listing placeable and
//! is queued again. A unit counts as done on every exit, so the backlog cannot
//! wedge on a unit that will be retried next pass, and an abort or a shutdown
//! hands claimed units back.

use super::*;

/// Drive a bounded worker pool over `jobs`, each worker pulling the next
/// folder from a shared cursor and running [`process_one_dir`].
pub(crate) fn run_dir_pool(
    ctx: &RefreshContext<'_>,
    base: &Sequence,
    jobs: &[UnitJob],
    live: &HashSet<String>,
    progress: &Arc<LoadProgress>,
) -> IngestReport {
    let engine = ctx.engine;
    let total = jobs.len();
    progress.set_step_progress(0, total as u64);
    // One snapshot of the folders sealed here: one sealed since this pass
    // planned is not sealed twice.
    let present_keys = sealed_folder_keys(engine);
    let backlog_pending = jobs
        .iter()
        .filter(|j| !present_keys.contains(&j.unit.key))
        .count() as u64;
    ingest_backlog::add_pending(Work::Reading, backlog_pending);
    let backlog_done = AtomicUsize::new(0);

    // Each member walks the queue in its own order, the folders a drawing
    // group draws first, and comes back once the rest is done to what it left
    // to another member's lease (`ingest_lease::order`).
    let keys: Vec<String> = jobs.iter().map(|j| j.unit.key.clone()).collect();
    let order = ingest_lease::order(&keys, |i| {
        ctx.retrieval.draws(&as_folder(&jobs[i].unit.dir))
    });
    let left_over: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    let done = AtomicUsize::new(0);
    let failures = Failures::new();

    let round = |indices: &[usize], last: bool| {
        let cursor = AtomicUsize::new(0);
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(FOLDER_PARALLELISM);
            for _ in 0..FOLDER_PARALLELISM {
                handles.push(s.spawn(|| loop {
                    if failures.aborted() || crate::ingest_cancel::cancelled() {
                        return;
                    }
                    let at = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(&idx) = indices.get(at) else {
                        return;
                    };
                    let job = &jobs[idx];
                    let mut left = false;
                    let tracked = !present_keys.contains(&job.unit.key);
                    let result = if !tracked {
                        Ok(())
                    } else {
                        match ingest_lease::share(&job.unit.key) {
                            Share::Theirs => {
                                tracing::info!(
                                    target: "zend::repo_scan",
                                    dir = %job.unit.dir,
                                    "left to the member holding its ingest lease",
                                );
                                left = true;
                                Ok(())
                            }
                            Share::Mine(hold) => {
                                if hold.recheck()
                                    && sealed_folder_keys(engine).contains(&job.unit.key)
                                {
                                    Ok(())
                                } else {
                                    process_one_dir(ctx, base, job, live, &failures)
                                }
                            }
                        }
                    };
                    if left && !last {
                        left_over.lock().unwrap().push(idx);
                        continue;
                    }
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    progress.set_step_progress(d as u64, total as u64);
                    if tracked {
                        backlog_done.fetch_add(1, Ordering::Relaxed);
                        ingest_backlog::item_done(Work::Reading, &job.unit.dir);
                    }
                    if let Err(e) = result {
                        let n = failures.record(&job.unit.dir, format!("{e:#}"));
                        tracing::warn!(
                            target: "zend::repo_scan",
                            dir = %job.unit.dir,
                            "folder seal failed (will retry next run): {e:#}",
                        );
                        ingest_holdoff::cool_down(&job.unit.dir, &format!("{e:#}"));
                        if n > MAX_DECODE_FAILURES {
                            failures.set_abort();
                        }
                    }
                }));
            }
            for h in handles {
                h.join().expect("repo_map worker panicked");
            }
        });
    };
    round(&order, false);
    let again = std::mem::take(&mut *left_over.lock().unwrap());
    if !again.is_empty() {
        tracing::info!(
            target: "zend::repo_scan",
            dirs = again.len(),
            "coming back to the folders left to another member's lease",
        );
    }
    round(&again, true);

    ingest_backlog::drop_pending(
        Work::Reading,
        backlog_pending.saturating_sub(backlog_done.load(Ordering::Relaxed) as u64),
    );
    progress.set_step_progress(total as u64, total as u64);
    let report = failures.into_report(total);
    if report.is_incomplete() {
        tracing::error!(
            target: "zend::repo_scan",
            n_dirs = total,
            n_failed = report.n_failed,
            aborted = report.aborted,
            first_failure = report.failures.first().map(|f| f.error.as_str()).unwrap_or("-"),
            first_failure_dir = report.failures.first().map(|f| f.unit.as_str()).unwrap_or("-"),
            "repo map folder seal INCOMPLETE — affected folders keep their earlier \
             listing and retry next pass",
        );
    } else {
        tracing::info!(n_dirs = total, "repo map folder seal complete");
    }
    report
}
