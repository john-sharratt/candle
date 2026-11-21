//! Starts and stops the session's engine: the load, the work handed off to
//! background tasks, and the drain that makes state durable.
//!
//! Shutdown is ordered so nothing outlives the engine it depends on. Ingest is
//! signalled to stop, the loader thread is joined and drains any engine it was
//! loading, and the ingest worker is stopped and awaited before the engine
//! drains. The drain runs exactly once, on the loader thread or here. A leave
//! stops each live turn where it stands and marks its conversation for the
//! member that takes it, and waits for those turns to settle. A conversation
//! no one uses is released after a period, and a turn holding one keeps it.

use super::*;

/// How often a pass is asked for while a layer changes with time rather than
/// with its branches: a day layer reads a day that closed
/// (`crate::layers::days`), and a layer that expires reads again the units
/// whose time is up (`crate::layers::expiry`). No branch move announces
/// either, and a unit stands at most this long past its time.
pub(super) const LAYER_CLOCK_EVERY: std::time::Duration = std::time::Duration::from_secs(30 * 60);

impl InferenceState {
    /// Release every resident conversation nothing has used for
    /// [`CONV_IDLE_TTL`] (`conv_sweep`), returning how many went. One a turn
    /// holds is in use now, so its clock restarts: a long turn is not idle the
    /// moment it finishes.
    ///
    /// The released states are dropped after the map's lock is let go — each
    /// frees its scheduler slot as it drops.
    /// Drop what this member keeps in memory of the conversation `key` names —
    /// a dialogue's `conv_id`, a passthrough transcript's key — once it no
    /// longer holds the conversation (`turn_claims`). Another member may write
    /// its next turn, which this member's cached sequence would not hold: the
    /// next turn served here resumes from the substrate, which does. A turn
    /// still holding the state keeps its own reference to it.
    pub(super) fn forget_conversation(&self, key: &str) {
        drop(self.conversations.remove(key));
        // Its next turn may be another member's, so its journal is read again
        // from the record when this member next holds it.
        self.journals.forget(key);
        self.passthrough.forget(key);
    }

    pub(super) fn sweep_idle_conversations(&self, now: Instant) -> usize {
        let released = self.conversations.sweep(now, CONV_IDLE_TTL);
        // A released conversation's slot goes with it, and so does the asker a
        // draft would have put its questions through.
        for (conv_id, _) in &released {
            self.journals.closed(conv_id);
        }
        released.len()
    }
}

impl ZendSession {
    /// Kick the async summariser so the just-ingested turns start folding into
    /// their summary trees promptly instead of waiting for the next periodic
    /// tick. **Fire-and-forget**: summarisation is a fully background task,
    /// invisible to the upload — the pipeline never waits on it, shows no
    /// progress for it, and records nothing about it. A no-op when the model
    /// isn't loaded (the periodic tick picks the work up once it is).
    pub fn kick_summariser(&self) {
        if let Some(state) = self.inference.load_full() {
            state.engine.trigger_summariser();
        }
    }

    /// The reply in flight for `conv_id`, if there is one.
    pub fn live_turn(&self, conv_id: &str) -> Option<Arc<LiveTurn>> {
        self.live_turns.get(conv_id)
    }

    /// Hand every live turn on as the member leaves its clusters
    /// (`docs/turn_handover.md` §4): no turn starts from now, each running
    /// turn stops where it stands and marks its conversation for the member
    /// that takes it, and this waits — at most [`LEAVE_TURNS_WAIT`] — until
    /// they have. A turn still running past that is cut at the shutdown and
    /// resumed by this member when it starts again.
    pub async fn leave(&self) {
        let Some(state) = self.inference.load_full() else {
            return;
        };
        let running = state.leaving.count();
        let left = state.leaving.leave(LEAVE_TURNS_WAIT).await;
        if left > 0 {
            tracing::warn!(
                running,
                left,
                "leaving: turns still running after {} s are cut at the shutdown",
                LEAVE_TURNS_WAIT.as_secs()
            );
        } else {
            tracing::info!(running, "leaving: every live turn stopped and handed on");
        }
    }

    /// Whether the member is leaving its clusters.
    pub fn is_leaving(&self) -> bool {
        self.inference
            .load_full()
            .is_some_and(|s| s.leaving.is_leaving())
    }

    /// Count a turn's settling as running until the guard drops, so a leave
    /// waits for what it writes (`crate::leaving`). `None` before the load.
    pub fn turn_running(&self) -> Option<Running> {
        self.inference.load_full().map(|s| s.leaving.running())
    }

    pub fn start_loading(self: &Arc<Self>) {
        let slot = Arc::clone(&self.inference);
        let loading_substrate = Arc::clone(&self.loading_substrate);
        let proj_builder = self.projection_builder.clone();
        let ready_tx = self.ready_tx.clone();
        let status_tx = self.status_tx.clone();
        let load_progress = Arc::clone(&self.load_progress);
        let workspace = self.config.workspace.clone();
        let secrets = Arc::clone(&self.config.secrets);
        let disabled_layers = self.config.disabled_layers.clone();
        let wiped_layers = self.config.wiped_layers.clone();
        let ingest_dirs = self.config.ingest_dirs.clone();
        let max_depth = self.config.max_depth;
        let ignore = self.config.ignore.clone();
        let agent = self.config.agent.clone();
        let compact_substrate = self.config.compact_substrate;
        let self_check = self.config.self_check;
        let read_only_substrate = self.config.read_only_substrate;
        let qsa_selection_budget = self.config.qsa_selection_budget;
        let qsa_strata = self.config.qsa_strata;
        let summarize = self.config.summarize;
        let tombstone_retention = self.config.tombstone_retention;
        let cluster = self.config.cluster.clone();
        // A ref one member or aggregator issued, any other opens
        // (`docs/private_mode.md` §3.1); installed before `/mcp` serves.
        if let Some(join) = &cluster {
            self.mcp
                .install_refs(Refs::new(join.derive_key(refs::PURPOSE)));
        }
        let ops: Arc<dyn OpsRunner> = Arc::clone(self.mcp.ops()) as _;
        // The memory its writer keeps current once the engine is up.
        let memory = Arc::clone(self.mcp.memory());
        // Whom each conversation's retrieval reads the memory as.
        let readers: Arc<dyn Readers> = Arc::new(MemoryReaders::new(
            Arc::clone(&memory),
            Arc::clone(self.mcp.ops()),
        ));
        // Handed to both the downloader and the engine builder, so the artifact
        // fetched and the model built are the same one.
        let model = self.config.model.clone();
        let engine_source = self.config.engine.clone();
        // Re-arm zend's ingest-cancel latch for this load: clear any cancel left
        // by a prior load/shutdown before this one's ingest starts polling it.
        // The startup ingest and background reconcile poll
        // `ingest_cancel::cancelled()`; `shutdown` latches it.
        crate::ingest_cancel::token().reset();
        // Handle to the ambient Tokio runtime (if any). The loader runs on a
        // plain OS thread and drops its temporary download runtime before the
        // model load, so the ingest worker's `tokio::spawn` would otherwise
        // panic for lack of a runtime context — which killed the loader thread
        // before `mark_ready()`, wedging the loading screen. We re-enter this
        // handle after the download phase so the worker can spawn.
        let rt_handle = Handle::try_current().ok();
        // The runner the ops tools reach the MCP implementation through, bound
        // to this session by a weak handle so no tool context keeps it alive. A
        // session loaded with no runtime around it serves no `/mcp`. The same
        // runtime runs the sweep that archives the `/mcp` workers and threads
        // left idle a day.
        if let Some(rt) = &rt_handle {
            self.mcp.ops().bind(Arc::downgrade(self), rt.clone());
            archiving::start_sweeping(self, rt);
            resuming::start_resuming(self, rt);
            tending::start_tending(self, rt);
            chat_sweep::start_sweeping(self, rt);
        }
        // Held by the spawned thread, which stores the ingest worker on the
        // session and hands it to the resumed turns.
        let session_for_loader: Arc<Self> = Arc::clone(self);
        // The same runtime, where the cluster's sockets live: what an agent's
        // fetch for a repository read from the substrate is made on.
        let host_runtime = rt_handle.clone();
        // OS thread, not `tokio::spawn` — `start_loading` may be
        // called from contexts without an ambient Tokio runtime
        // (integration tests, alternative binaries).  The bits that
        // need an async runtime (HF download) drive a local
        // current-thread runtime built inside this thread.
        let loader_handle = std::thread::Builder::new()
            .name("zend-loader".into())
            .spawn(move || {
                #[cfg(feature = "memory-census")]
                set_thread(Scope::Loader);
                load_progress.set_step(LoadStep::Model);
                status_tx.send("Checking for model…".into()).ok();

                // The download path is async (hf-hub, tokio::fs, reqwest);
                // build a tiny current-thread runtime so we can `block_on`
                // it without requiring the caller to be on Tokio.
                let download_runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        status_tx.send(format!("Runtime build failed: {e}")).ok();
                        fatal(
                            "zend",
                            &format!("local tokio runtime for download failed: {e:#}"),
                        );
                    }
                };
                // The device the pack is resolved for — its int8 mode and layer
                // narrowing are this card's — and then loaded on.
                let device = match Device::cuda_if_available(0) {
                    Ok(d) => d,
                    Err(e) => {
                        status_tx.send(format!("Device init failed: {e}")).ok();
                        fatal("zend", &format!("device init failed: {e:#}"));
                    }
                };
                let pack = match ensure_model_pack(&model, &device, &download_runtime, &status_tx) {
                    Ok(p) => p,
                    Err(e) => {
                        // A missing model is fatal — the daemon cannot serve
                        // anything without it. Fail hard rather than limping
                        // into a fake-ready state that errors on first submit.
                        status_tx.send(format!("Model pack failed: {e}")).ok();
                        fatal("zend", &format!("model pack failed: {e:#}"));
                    }
                };
                // Drop the runtime; the model load below is sync.
                drop(download_runtime);

                // Re-enter the main Tokio runtime for the rest of this thread so the
                // ingest worker's `tokio::spawn` has a runtime context. Must come
                // *after* the download runtime is dropped (no nested `block_on`).
                // The model load is synchronous, so holding the enter guard is safe.
                let _rt_guard = rt_handle.as_ref().map(|h| h.enter());

                status_tx.send("Loading model…".into()).ok();
                // Named from the choice rather than spelled out, so the line
                // cannot go on claiming a model zend has stopped running.
                tracing::info!(
                    "loading inference engine ({}) …",
                    model.clone().spec().model_filename,
                );
                let load_progress_for_blocking = Arc::clone(&load_progress);
                // `InferenceState::load` is fully synchronous (CUDA model
                // load + substrate recovery + ingestion).  Call directly
                // on this thread — no `spawn_blocking` needed.
                match InferenceState::load(
                    proj_builder,
                    model,
                    engine_source,
                    device,
                    pack,
                    workspace,
                    secrets,
                    disabled_layers,
                    wiped_layers,
                    ingest_dirs,
                    max_depth,
                    ignore,
                    host_runtime,
                    agent,
                    compact_substrate,
                    self_check,
                    read_only_substrate,
                    qsa_selection_budget,
                    qsa_strata,
                    summarize,
                    tombstone_retention,
                    cluster,
                    status_tx.clone(),
                    load_progress_for_blocking,
                    loading_substrate,
                    ops,
                    Arc::clone(&memory),
                ) {
                    Ok(Some(state)) => {
                        // Before the state is published: no turn is scoped
                        // without its reader.
                        state.retrieval_scope.set_readers(readers);
                        slot.store(Some(Arc::clone(&state)));
                        tracing::info!("inference engine ready");
                        // A stop asked since the load's last boundary: the engine
                        // is published, so `ZendSession::shutdown` drains it —
                        // skip the warm-ups, the worker and the resumed turns, and
                        // release whatever waits on the load.
                        if crate::ingest_cancel::cancelled() {
                            ready_tx.send(true).ok();
                            return;
                        }
                        // The last load step: the tool catalog's normalization hit
                        // levels, relearned from its corpus in batch on the GPU.
                        // Before `ready`, so no query is ever scored against cold
                        // levels — and no live turn competes with the warm-up.
                        load_progress.set_step(LoadStep::Normalizing);
                        let schema = state.refresh_builder.schema().clone();
                        state.engine.warm_collection_normalization(&schema);
                        if crate::ingest_cancel::cancelled() {
                            ready_tx.send(true).ok();
                            return;
                        }
                        load_progress.set_step(LoadStep::Preloading);
                        let t_preload = Instant::now();
                        let preload = state.engine.preload_galleries(&schema);
                        tracing::info!(
                            turns = preload.turns,
                            resident = preload.resident,
                            tokens = preload.tokens,
                            ms = t_preload.elapsed().as_millis() as u64,
                            "projection galleries preloaded"
                        );
                        status_tx.send(String::new()).ok();
                        // The memory's writer, when this zend serves the
                        // memory and may write: summaries, folds and closes,
                        // at `Low`.
                        // A session loaded with no runtime around it has
                        // nothing to run the writer on, and no `/mcp` to serve
                        // the memory through.
                        if let Some(rt) = rt_handle.as_ref() {
                            if memory.serves() && !read_only_substrate {
                                let writer = rt.spawn(memory_writer::run(
                                    Arc::downgrade(&state),
                                    Arc::clone(&memory),
                                    state.memory_stop.clone(),
                                ));
                                *state.memory_writer.lock().unwrap() = Some(writer);
                            }
                        }
                        // Substrate persistence runs in the engine's own
                        // thread (`PersistenceThread`) — 5 s tick + per-turn
                        // trigger from the scheduler — so no periodic-flush
                        // task is needed at the daemon level here.
                        //
                        // The schema-driven `LoadStep::Ingesting` step is
                        // advanced inside `InferenceState::load` itself — the
                        // ingest passes own it and report sub-step progress
                        // through the same `LoadProgress` handle.

                        // Uploads kept before conversations held them are taken
                        // in once. Uploads are endpoint-managed, but a deletion
                        // of an uploaded file still has to retire its substrate
                        // conversation: the cheap tombstone-if-unheld reconcile
                        // runs once at startup, for uploads deleted since the
                        // last one. A read-only substrate takes none of
                        // this upkeep, here or below: it is written by the daemon
                        // beside it.
                        if !read_only_substrate {
                            state.take_in_legacy_uploads();
                            state.reconcile_uploaded_files();
                        }

                        // The single background ingest worker. Its pass at
                        // startup reads every branch as the tracking refs hold
                        // it (`crate::branch_ingest`), and is followed by the
                        // ingest normalization warm-up; after that it runs when
                        // an origin watcher fetched a moved branch, or a tool
                        // round published one — and on a clock while a layer
                        // changes with time.
                        let wake = Arc::new(Notify::new());
                        if !read_only_substrate {
                            let pass_state = Arc::clone(&state);
                            let pass: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                                match pass_state.refresh_ingest_layers() {
                                    Ok(true) => tracing::info!("ingest pass: layers updated"),
                                    Ok(false) => {
                                        tracing::debug!("ingest pass: no layer changed")
                                    }
                                    Err(e) => tracing::warn!("ingest pass failed: {e:#}"),
                                }
                                // Folded whether or not the pass succeeded: a
                                // failed pass is read again only on the next
                                // branch move or layer clock, and what it did
                                // write is folded now rather than then.
                                pass_state.fold_chunk_exclusions_behind();
                            });
                            let warm_state = Arc::clone(&state);
                            let after_first_pass: Arc<dyn Fn() + Send + Sync> =
                                Arc::new(move || {
                                    // The first pass is now DONE, so warm the
                                    // per-file normalization hit levels HERE —
                                    // off the load path and, crucially, with no
                                    // concurrent ingest writer to starve (running
                                    // the heavy self-match scan during ingest
                                    // freezes the scheduler). Covers the
                                    // first-run ingest and every restart's
                                    // reconcile. It scores on the scheduler's GPU
                                    // arena a slice of files at a time. The tool
                                    // catalog's own levels are already warmed by
                                    // the load's `Normalizing` step, before ready.
                                    let warmer = warm_state.engine.ingest_warmer();
                                    let schema = warm_state.refresh_builder.schema().clone();
                                    warmer.run(&schema);
                                });
                            let worker = crate::ingest_worker::spawn(
                                Arc::clone(&wake),
                                pass,
                                after_first_pass,
                            );
                            // The startup handoff: the initial backlog — first-ever
                            // ingest, or whatever drifted while the daemon was down
                            // — starts NOW, in the background, while the lines
                            // below take the daemon to `ready` without waiting on it.
                            worker.wake();
                            *session_for_loader.ingest_worker.lock().unwrap() = Some(worker);
                            let _ = state.ingest_wake.set(Arc::clone(&wake));
                            // A record branch that moves in the log — imported
                            // from origin through an agent, or another
                            // member's push — wakes the worker
                            // (`docs/substrate_git.md` §5). Registering wakes it
                            // once too: the follower has run since the load, and
                            // a move in between is reported to no one.
                            if let Some(repositories) = state.repositories.as_ref() {
                                // The wake on registering does not preempt: the
                                // startup pass woken above is the pass it asks
                                // for, and stopping it would read no layer above
                                // the branches until a second one.
                                let on_change: Arc<dyn Fn() + Send + Sync> = {
                                    let wake = Arc::clone(&wake);
                                    let registered = AtomicBool::new(false);
                                    Arc::new(move || {
                                        if registered.swap(true, Ordering::SeqCst) {
                                            crate::ingest_worker::request(&wake);
                                        } else {
                                            wake.notify_one();
                                        }
                                    })
                                };
                                repositories.on_change(on_change);
                            }
                            // A day layer's unit is a day once it is over, and a
                            // unit of a layer that expires goes stale with time;
                            // no branch move announces either. While a layer
                            // passed this boot does either, ask for a pass now
                            // and then. A pass that finds nothing new to read
                            // costs its plans.
                            let on_a_clock = state.branch_layers.iter().any(|(l, _)| {
                                l.spec.as_ref().is_some_and(|s| {
                                    s.unit == crate::layers::UnitKind::Day || s.expiry.is_some()
                                })
                            });
                            if on_a_clock {
                                let wake = Arc::clone(&wake);
                                tokio::spawn(async move {
                                    let mut every = tokio::time::interval(LAYER_CLOCK_EVERY);
                                    every.tick().await;
                                    loop {
                                        every.tick().await;
                                        if crate::ingest_cancel::cancelled() {
                                            return;
                                        }
                                        // A wake, not a request: a day closing or a
                                        // unit expiring is no work for the layers
                                        // below, so a pass in flight is not stopped
                                        // for it — the next pass reads it.
                                        wake.notify_one();
                                    }
                                });
                            }
                        }
                        // The engine is up — only NOW mark ready and unblock
                        // submit-flow waiters. Skipped on the shutdown-during-ingest
                        // path (the `Ok(None)` arm below).
                        load_progress.mark_ready();
                        ready_tx.send(true).ok();

                        // Whatever the last run was in the middle of, finish it
                        // — after `ready`, so the loading screen never waits on
                        // a decode, and never against a read-only substrate,
                        // which is written by the daemon beside this one.
                        if !read_only_substrate {
                            resume_unfinished_turns(
                                Arc::clone(&session_for_loader),
                                Arc::clone(&state),
                            );
                            // And what another member handed on as it left.
                            resume_handed_turns(
                                Arc::clone(&session_for_loader),
                                Arc::clone(&state),
                            );
                        }
                    }
                    Ok(None) => {
                        // Shutdown arrived during the load — at any step's
                        // boundary, the startup ingest among them: `load` already
                        // drained the engine on this thread (scheduler join →
                        // persistence hot→warm→cold → writer flush). Do NOT mark
                        // ready or publish an engine — the loader thread just exits;
                        // `ZendSession::shutdown`, which is joining it, then returns
                        // and the process exits with everything durable.
                        //
                        // The waiters on the load are released all the same: a
                        // request parked on it would otherwise hold the server's
                        // drain open for good. Each wakes, finds no engine, and
                        // answers so.
                        ready_tx.send(true).ok();
                        tracing::info!(
                            "zend-loader: shutdown during the load — engine drained, loader exiting"
                        );
                    }
                    // A step that ended on the stop's latch — a conversation
                    // create or ingest refused `IngestCancelled` between two
                    // `load_stopped` checks — is the stop, not a failed load:
                    // exiting 1 here would skip the drain and read as a crash.
                    Err(e) if crate::ingest_cancel::cancelled() => {
                        ready_tx.send(true).ok();
                        tracing::info!("zend-loader: the load ended by the stop: {e:#}");
                    }
                    Err(e) => {
                        status_tx.send(format!("Load failed: {e}")).ok();
                        fatal("zend", &format!("inference engine failed to load: {e:#}"));
                    }
                }
            })
            .expect("spawn zend-loader thread");
        // Hand the loader thread's handle to the session so `shutdown` can join it
        // — that join is what guarantees an in-progress load has broken its ingest
        // and finished draining the engine before the process exits.
        *self.load_thread.lock().unwrap() = Some(loader_handle);
    }

    /// Graceful shutdown: stop any in-flight ingest, durably drain + commit the
    /// substrate (hot→warm→cold + off-thread writer flush), then stop the
    /// scheduler. Idempotent, and — crucially — correct DURING loading too: a
    /// Ctrl-C mid-ingest (when `inference` is still `None`) is not a no-op. Runs
    /// the blocking work on the blocking pool since it does `fsync` I/O + joins.
    pub async fn shutdown(&self) {
        // 1. Signal any in-flight ingest — the blocking `raw` ingest inside
        //    `InferenceState::load` AND the background ingest worker's current
        //    pass — to stop submitting. Both poll zend's ingest-cancel latch
        //    cooperatively (as do the deep shared ingest hot paths — per-file
        //    worker pool, cluster scan, raw docs), so no thread is torn down
        //    mid-turn; they break at a safe boundary. `start_loading` clears it
        //    before each load.
        crate::ingest_cancel::token().request();
        // 2. Join the loader thread. If a load is IN PROGRESS, it breaks its ingest
        //    and drains the engine on that thread (the `Ok(None)` path in
        //    `start_loading`), and the join returns only once that drain is done.
        //    If the load already completed (or never started), the thread has ended
        //    and this returns immediately.
        let loader = self.load_thread.lock().unwrap().take();
        if let Some(h) = loader {
            let _ = tokio::task::spawn_blocking(move || {
                if h.join().is_err() {
                    tracing::warn!("shutdown: loader thread panicked");
                }
            })
            .await;
        }
        //    Then the repositories' follower, so no record branch that moves
        //    wakes a worker being stopped, and no import writes past the drain;
        //    the round in flight finishes first. A load that never
        //    published its state dropped its repositories, which stops theirs.
        let followed = self.inference.load_full();
        if let Some(state) = followed {
            let _ = tokio::task::spawn_blocking(move || {
                if let Some(repositories) = state.repositories.as_ref() {
                    repositories.stop();
                }
            })
            .await;
        }
        //    Then the ingest worker. It holds the inference state for its whole
        //    life, so nothing outside this call may keep the engine alive past
        //    the drain below. `stop` sets its flag, wakes it, and AWAITS the
        //    task: if a pass is in flight, the pool breaks at its next unit
        //    boundary on the cancel raised in step 1 (`repo_scan::run_dir_pool`,
        //    `code_read::ingest_jobs`) and the warm-up scan stops between
        //    probes, so the await is bounded. It is an await and never an
        //    `abort()`: Tokio cannot cancel the `spawn_blocking` a pass runs on,
        //    so aborting would return here with a pool still minting
        //    conversations — and the drain below would race it.
        let worker = self.ingest_worker.lock().unwrap().take();
        if let Some(w) = worker {
            w.stop().await;
        }
        // 3. Re-read `inference` AFTER the join. If the load COMPLETED (ready path,
        //    or it finished racing our cancel), the engine is published here and
        //    has NOT been drained — drain it below. If the loader drained mid-ingest
        //    (step 2), `inference` is still `None` and there is nothing left to do.
        //    Exactly-once: the drain runs either on the loader thread OR here, never
        //    both.
        let state = self.inference.load_full();
        let Some(state) = state else {
            tracing::info!(
                "shutdown: loader drained mid-ingest (or model never loaded) — nothing more to persist"
            );
            return;
        };
        // Signal the titler task to stop draining and wake it if it's idle.
        // This must happen before the scheduler stops so the task skips any
        // queued jobs (rather than failing each one against a dead scheduler).
        state.shutting_down.store(true, Ordering::Relaxed);
        let _ = state.titler_tx.try_send(TitleJob::Shutdown);
        // The memory's writer stops on its token: a turn in flight — which may
        // sit behind every other band for minutes — is abandoned at once, and
        // what it began on the blocking pool is awaited, so its question's
        // slot, timeline and frame are all let go while the scheduler still
        // answers.
        // A write waiting on the store waits for the sync thread's push or
        // fetch; past `memory_writer::GRACE` the writer is left to the
        // process's end rather than holding the drain.
        state.memory_stop.request();
        let writer = state.memory_writer.lock().unwrap().take();
        if let Some(mut writer) = writer {
            match tokio::time::timeout(memory_writer::GRACE, &mut writer).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) => tracing::warn!("shutdown: memory writer panicked"),
                Err(_) => {
                    tracing::warn!("shutdown: memory writer still busy; left behind");
                    writer.abort();
                }
            }
        }
        let engine_state = Arc::clone(&state);
        let _ = tokio::task::spawn_blocking(move || {
            // Passthrough conversations hold scheduler slots; free them while
            // the scheduler still answers.
            engine_state.passthrough.clear();
            let engine = &engine_state.engine;
            match engine.commit_persistence() {
                Ok(()) => tracing::info!("shutdown: substrate committed"),
                Err(e) => tracing::error!("shutdown: commit failed: {e}"),
            }
            if let Err(e) = engine.shutdown() {
                tracing::error!("shutdown: scheduler stop failed: {e}");
            }
        })
        .await;

        // Await the titler task so any in-flight title turn unwinds (via
        // `abort_turn`, against the now-stopped scheduler) before the process
        // exits. The guard is released before the await — the take is all it
        // covers.
        let worker = state.titler_worker.lock().unwrap().take();
        if let Some(worker) = worker {
            if worker.await.is_err() {
                tracing::warn!("shutdown: titler task panicked");
            }
        }
    }

    /// Drop what this member keeps of `conv_id` in memory — its sequence, its
    /// files, its journal — so its next turn here is built from the substrate:
    /// a worker's turns may have been written by another member since this one
    /// last ran it.
    pub fn forget(&self, conv_id: &str) {
        if let Some(state) = self.inference.load_full() {
            state.forget_conversation(conv_id);
        }
    }
}
