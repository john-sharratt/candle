//! The bounded worker pool that ingests a pass's folders, recording any failure
//! without ending the pass.
//!
//! Each worker claims its next folder from a shared cursor and only then waits
//! for room to open a conversation, so a worker that finds the queue empty holds
//! no slot. Committed folders are probed before the VRAM gate is queued for, since
//! a hit opens nothing. The width is decided once, from the card, before any
//! worker spawns, because the runtime gate cannot bound the opening burst; when
//! neither the memory report nor the governor can say anything, the width is a
//! conservative blind value and not the ceiling measured for the largest card. A
//! failed directory keeps its prior generation live and is queued again. A unit
//! counts as done on every exit, so the backlog cannot wedge on a unit that will
//! be retried next pass, and an abort or a shutdown hands claimed units back.

use super::*;

/// Drive a bounded worker pool over `jobs`: each worker pulls the next
/// directory from a shared cursor and runs [`process_one_dir`]. Workers share
/// the progress counter, a tolerated-failure counter, and an abort flag (the
/// first hard error stops the rest).
pub(super) fn run_dir_pool(
    ctx: &RefreshContext<'_>,
    base: &Sequence,
    plan: &IngestPlan,
    jobs: &[UnitJob],
    live: &HashSet<String>,
    progress: &Arc<LoadProgress>,
) -> IngestReport {
    let engine = ctx.engine;
    let total = jobs.len();
    progress.set_step_progress(0, total as u64);
    // Price the pool against the arenas it is about to add, not the ones it
    // inherits. Must precede the width sizing below, which reads the estimate.
    anchor_scan_kv_baseline();
    // One snapshot of the committed keys: a unit another pass committed since
    // this one planned is not ingested twice.
    let present_keys = engine.conversation_metadata_values(CONTENT_KEY);

    // The GUI's merged background-ingest bar (`crate::ingest_backlog`) counts
    // only units that will REALLY run. Counted from the same snapshot the
    // workers probe below, so registration and the per-unit completions can
    // never disagree.
    let backlog_pending = jobs
        .iter()
        .filter(|j| !present_keys.contains(&j.unit.key))
        .count() as u64;
    ingest_backlog::add_pending(Work::Reading, backlog_pending);
    let backlog_done = AtomicUsize::new(0);

    // Each member walks the queue in its own order, the folders a drawing
    // group draws first, and comes back once the rest is done to what it left
    // to another member's lease (`ingest_lease::order`); one a live holder
    // still keeps is left again and counted done, its holder's commit landing
    // it.
    let keys: Vec<String> = jobs.iter().map(|j| j.unit.key.clone()).collect();
    let order = ingest_lease::order(&keys, |i| {
        ctx.retrieval.draws(&as_folder(&jobs[i].unit.dir))
    });
    let left_over: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    let done = AtomicUsize::new(0);
    // Conversations this pool currently holds open — the quantity that actually
    // costs VRAM (each pins its K/V until its chain completes). Gates worker
    // claims via `reserve_scan_slot`.
    let live_convs = AtomicUsize::new(0);
    // Per-directory failures are RECORDED, never propagated: a directory that
    // fails keeps its prior generation live and the pass carries on, so one bad
    // folder (or a systemic VRAM squeeze) degrades the map instead of killing
    // the daemon. The cap still stops a flood — `failures.set_abort()` — but as
    // reported state, not as a fatal error. See `report`.
    let failures = Failures::new();

    let round = |indices: &[usize], last: bool| {
        let cursor = AtomicUsize::new(0);
        std::thread::scope(|s| {
            // Size the pool to the CARD before spawning, not to a constant.
            //
            // The runtime gate alone cannot bound the opening burst: it reads
            // the scheduler's published memory report, and at scan start that
            // report is either absent or predates the scan, so every worker
            // sees an empty pool and claims. Measured: the gate computed
            // `max_live=6` while 21 conversations were already open. Deciding
            // the width once, up front, removes the race entirely; the runtime
            // gate then handles drift as directories vary in size.
            // The fallback is the BLIND width, taken when neither the memory
            // report nor the governor can say anything — which the call site
            // above notes is normal at scan start. It must NOT be
            // `REPO_MAP_PARALLELISM`: that constant is a measured ceiling for a
            // 72 GB card, and defaulting to it would open 96 conversations on
            // any card with zero memory input. Over-subscription there is not a
            // slowdown — the wave's transient tier fails and the ingest aborts,
            // losing files (see the table on `MAX_SCOPE_LINES`). Start
            // conservative and let the runtime gate widen.
            let n_workers = max_live_conversations()
                .or_else(scan_width_from_governor)
                .unwrap_or(REPO_MAP_BLIND_PARALLELISM);
            tracing::info!(
                target: "zend::repo_scan",
                n_workers,
                ceiling = REPO_MAP_PARALLELISM,
                "repo map pool width sized to available KV",
            );
            let mut handles = Vec::with_capacity(n_workers);
            for _ in 0..n_workers.max(1) {
                handles.push(s.spawn(|| {
                    loop {
                        // Stop before claiming the next directory on a
                        // first-error abort OR a shutdown cancel. The in-flight
                        // conversation finishes (the scheduler is still live),
                        // so the engine can drain rather than the shutdown
                        // losing the un-drained tier tail.
                        if failures.aborted() || crate::ingest_cancel::cancelled() {
                            return;
                        }
                        // Claim the unit BEFORE reserving a slot: a worker that
                        // finds the queue empty must not be holding one, or the
                        // last workers to notice would each pin a slot against
                        // the conversations still finishing.
                        let at = cursor.fetch_add(1, Ordering::Relaxed);
                        let Some(&idx) = indices.get(at) else {
                            return;
                        };
                        let job = &jobs[idx];
                        let mut left = false;
                        // Probe the committed keys BEFORE queueing for VRAM. A
                        // hit opens no conversation, so waiting on the KV gate
                        // for it buys nothing.
                        let tracked = !present_keys.contains(&job.unit.key);
                        let result = if !tracked {
                            tracing::debug!(
                                target: "zend::repo_scan",
                                dir = %job.unit.dir,
                                "skip: directory already committed",
                            );
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
                                    let already = hold.recheck()
                                        && engine
                                            .has_live_intact_metadata(CONTENT_KEY, &job.unit.key);
                                    if already {
                                        Ok(())
                                    } else {
                                        // Bound open conversations by the card,
                                        // not by the thread count — see
                                        // `scan_width`. The guard holds the slot
                                        // for the whole ingest, so no other
                                        // worker can decide against this state,
                                        // and an unwind cannot leak it from the
                                        // process-global gauge. `hold` stays
                                        // across that reading and hands the
                                        // lease back after.
                                        let _slot = ScanSlot::reserve(&live_convs);
                                        process_one_dir(ctx, base, plan, job, live, &failures)
                                    }
                                }
                            }
                        };
                        // Left to its holder in the first round, the unit is
                        // come back to, and is not done yet.
                        if left && !last {
                            left_over.lock().unwrap().push(idx);
                            continue;
                        }
                        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                        progress.set_step_progress(d as u64, total as u64);
                        // A registered unit is "done" on every exit — success
                        // or a tolerated failure — so the backlog can never
                        // wedge non-empty waiting on a unit that will just be
                        // retried next pass.
                        if tracked {
                            backlog_done.fetch_add(1, Ordering::Relaxed);
                            ingest_backlog::item_done(Work::Reading, &job.unit.dir);
                        }
                        // An error escaping `process_one_dir` is an unexpected
                        // one (its own two failure modes record and return Ok).
                        // Record it the same way so it lands in the report
                        // rather than vanishing, and let the cap decide whether
                        // to stop.
                        if let Err(e) = result {
                            let n = failures.record(&job.unit.dir, format!("{e:#}"));
                            tracing::warn!(
                                target: "zend::repo_scan",
                                dir = %job.unit.dir,
                                "directory ingest failed (will retry next run): {e:#}",
                            );
                            ingest_holdoff::cool_down(&job.unit.dir, &format!("{e:#}"));
                            if n > MAX_DECODE_FAILURES {
                                failures.set_abort();
                            }
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

    // An abort or a shutdown cancel can leave units claimed but never run —
    // hand them back, or the GUI's backlog bar never reaches zero.
    ingest_backlog::drop_pending(
        Work::Reading,
        backlog_pending.saturating_sub(backlog_done.load(Ordering::Relaxed) as u64),
    );

    // Pin the bar to 100%: workers store from their own `done` snapshot without a
    // max, so the last stored value can settle a step short even though every
    // unit ran.
    progress.set_step_progress(total as u64, total as u64);
    let report = failures.into_report(total);
    // Say "incomplete" when it is incomplete. The old line said "complete" with
    // the failure count as a field, so a quarter-empty map read as success at a
    // glance; the first failure's cause is carried here too, so the summary
    // alone explains the pass without scrolling back through the warnings.
    if report.is_incomplete() {
        tracing::error!(
            target: "zend::repo_scan",
            n_dirs = total,
            n_failed = report.n_failed,
            aborted = report.aborted,
            first_failure = report.failures.first().map(|f| f.error.as_str()).unwrap_or("-"),
            first_failure_dir = report.failures.first().map(|f| f.unit.as_str()).unwrap_or("-"),
            "repo map per-directory ingest INCOMPLETE — affected directories keep \
             their prior generation and retry next pass (GET /v1/repo_map)",
        );
    } else {
        tracing::info!(n_dirs = total, "repo map per-directory ingest complete",);
    }
    report
}
