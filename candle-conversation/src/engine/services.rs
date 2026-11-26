//! The engine's long-running services: substrate reload, guest jobs, the
//! persistence thread, and shutdown.
//!
//! Reload progress is written by the scheduler thread and polled by the daemon.
//! The reload runs on the scheduler because it needs the model's backings. Guest
//! work is queued for a co-resident model and served between two forward waves,
//! and the scheduler is woken so an idle daemon sees the job. Shutdown detaches
//! this substrate, stops the scheduler only when it is the last attachment, and
//! flushes the redo log before the process exits.

use super::*;
use crate::conversation::run_state;
use crate::scheduler::exported_state::SharedState;

/// Live progress of the startup substrate reload (redo-log replay), shared
/// between the scheduler thread (writer) and the daemon's load-state machine
/// (reader). As the substrate grows this replay stops being instantaneous, so
/// the GUI needs a real progress signal instead of a stalled loading bar.
#[derive(Debug, Default)]
pub struct SubstrateReloadStatus {
    /// Turns restored so far.
    pub(super) done: AtomicUsize,
    /// Total turns to restore — `0` until the redo-log decl list is known.
    pub(super) total: AtomicUsize,
    /// Set once the reload pass has finished (success *or* error). Readers must
    /// key completion off this, not `done == total` — corrupt turns are skipped
    /// so `done` may never reach `total`.
    pub(super) finished: AtomicBool,
}

impl SubstrateReloadStatus {
    /// Writer (scheduler thread): record turns-restored / total-to-restore.
    pub fn record(&self, done: usize, total: usize) {
        self.total.store(total, Ordering::Relaxed);
        self.done.store(done, Ordering::Relaxed);
    }

    /// Writer (scheduler thread): mark the reload pass complete. Always called,
    /// even on the no-op / error paths, so a reader never waits forever.
    pub fn finish(&self) {
        self.finished.store(true, Ordering::Release);
    }

    /// Reader (load-state machine): `(done, total, finished)` snapshot.
    pub fn snapshot(&self) -> (usize, usize, bool) {
        let finished = self.finished.load(Ordering::Acquire);
        (
            self.done.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed),
            finished,
        )
    }
}

/// Reloads one substrate on its engine's scheduler thread — see
/// [`ConversationEngine::reloader`].
#[derive(Clone)]
pub struct SubstrateReloader {
    pub(super) scheduler_tx: Sender<SchedulerRequest>,
    pub(super) conversation: Conversation,
}

impl SubstrateReloader {
    /// Ask for the reload; poll the returned status until `finished`.
    pub fn reload(&self) -> Arc<SubstrateReloadStatus> {
        let status = Arc::new(SubstrateReloadStatus::default());
        let request = SchedulerRequest::ReconstructSubstrate {
            conversation: self.conversation.clone(),
            status: Arc::clone(&status),
        };
        if self.scheduler_tx.send(request).is_err() {
            // No scheduler to run it: nothing to wait for.
            status.finish();
        }
        status
    }

    /// Reload and return once it has finished.
    pub fn reload_and_wait(&self) {
        let status = self.reload();
        while !status.snapshot().2 {
            std::thread::sleep(RELOAD_POLL);
        }
    }
}

/// How often a reload waited on is looked at.
pub(super) const RELOAD_POLL: Duration = Duration::from_millis(50);

impl ConversationEngine {
    /// Queue work for a co-resident model and get a handle on its answer.
    ///
    /// The scheduler picks it up between two of its waves, evicts what it needs
    /// to make room, serves the whole backlog for that guest, and hands the
    /// ground back. Normal inference is blocked for the length of that drain —
    /// see [`crate::guest`] for why that is the design rather than a cost.
    ///
    /// Refuses synchronously — before anything is queued and before anything is
    /// evicted — when the request itself is not servable. A caller can block on
    /// [`GuestReceipt::wait`] or poll `try_take`.
    pub fn submit_guest(
        &self,
        request: crate::guest::GuestRequest,
    ) -> Result<crate::guest::GuestReceipt, crate::guest::GuestError> {
        self.submit_guest_watched(request, crate::guest::GuestSink::none())
    }

    /// The same, reporting progress to `sink` while the job runs.
    ///
    /// The sink is called on the scheduler thread with normal inference blocked,
    /// so it must not block — see [`crate::guest::progress`].
    pub fn submit_guest_watched(
        &self,
        request: crate::guest::GuestRequest,
        sink: crate::guest::GuestSink,
    ) -> Result<crate::guest::GuestReceipt, crate::guest::GuestError> {
        let receipt = self.core.guests.queue.submit_watched(request, sink)?;
        // **Wake the scheduler.** It parks in `rx.recv()` when there is nothing
        // to do, and the guest queue is the one producer that does not arrive
        // through that channel — so on an idle daemon a submitted job sat in
        // the queue indefinitely, below a loop that was never going to come
        // back round to look at it.
        //
        // Sent *after* the job is queued, so the loop either sees the work on
        // its own next pass or is woken to find it. There is no ordering in
        // which both miss: the channel buffers, so a wake that arrives while
        // the loop is still running is waiting for it when it parks.
        let _ = self.core.scheduler_tx.send(SchedulerRequest::Wake);
        Ok(receipt)
    }

    /// Withdraw guest job `seq` while it is still queued; `false` once a drain
    /// has taken it ([`crate::guest::GuestQueue::cancel`]).
    pub fn cancel_guest(&self, seq: u64) -> bool {
        self.core.guests.queue.cancel(seq)
    }

    /// The guests this deployment configured, for a status view.
    pub fn configured_guests(&self) -> Vec<crate::guest::Guest> {
        self.core.guests.registry.configured()
    }

    /// The vision tower this engine's model reads images with, or `None` for a
    /// model that reads text alone (`docs/vision_models.md` §2.2).
    pub fn vision(&self) -> Option<&'static VisionTowerSource> {
        self.core.guests.registry.vision()
    }

    /// The largest square side this node draws ([`crate::guest::reach`]).
    pub fn guest_reach(&self) -> u32 {
        self.core.guests.queue.reach()
    }

    /// How many guest jobs are waiting.
    pub fn guest_backlog(&self) -> usize {
        self.core.guests.queue.depth()
    }

    /// Shared handle to the startup substrate-reload progress. The daemon's
    /// load-state machine polls [`SubstrateReloadStatus::snapshot`] to drive
    /// the GUI's "Loading substrate" step while the redo log replays.
    pub fn substrate_reload_status(&self) -> Arc<SubstrateReloadStatus> {
        Arc::clone(&self.substrate_reload_status)
    }

    /// Re-reconstruct the substrate on the scheduler thread (needs the model
    /// backings for KV residence) — call after a compaction rewrites the redo log
    /// so all offsets / KV pointers are rebuilt from the new log. Returns a fresh
    /// status handle; poll [`SubstrateReloadStatus::snapshot`] until `finished`.
    pub fn reload_substrate(&self) -> Arc<SubstrateReloadStatus> {
        self.reloader().reload()
    }

    /// A handle that reloads this substrate without holding the engine — what
    /// replication keeps, to reload after a rejoin or a turn that arrived
    /// ranked below one held (`docs/replicated_substrate.md` §6.6).
    pub fn reloader(&self) -> SubstrateReloader {
        SubstrateReloader {
            scheduler_tx: self.core.scheduler_tx.clone(),
            conversation: self.conversation.clone(),
        }
    }

    /// The segmented redo log's maintenance state — `(segment_count, last_op)`,
    /// where `last_op` is `(label, unix_secs)` — for the daemon status / GUI
    /// compaction indicator.
    pub fn substrate_maintenance_status(&self) -> (usize, Option<(String, u64)>, bool) {
        self.conversation.maintenance_status()
    }

    /// Get the shared tokenizer.
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.core.tokenizer
    }

    /// Static model properties captured at engine construction.
    pub fn model_core_properties(&self) -> ModelCoreProperties {
        self.core.model_core
    }

    /// Low-level helper used by benchmarks (e.g. RULER): create a fresh
    /// sequence, prefill the supplied token IDs, decode argmax until EOS or
    /// `max_decode_tokens`, and return the decoded text.
    ///
    /// All parallelism comes from many threads calling this concurrently;
    /// the scheduler batches their prefills and decodes together.
    pub fn infer_raw_tokens(
        &self,
        tokens: &[u32],
        max_decode_tokens: usize,
    ) -> crate::Result<String> {
        let RawDecode {
            tokens: collected,
            text: text_from_done,
            ..
        } = self.infer_raw(tokens, max_decode_tokens)?;
        if let Some(t) = text_from_done {
            return Ok(t);
        }
        // Fallback: decode whatever tokens we collected.
        self.core
            .tokenizer
            .decode(&collected, true)
            .map_err(|e| ConversationError::Channel(format!("decode failed: {e}")))
    }

    /// [`Self::infer_raw_tokens`]'s decode as token ids: `tokens` prefilled
    /// on a fresh slot — every layer, the recurrent ones included, run over
    /// them in order — and decoded argmax for up to `max_decode_tokens`.
    pub fn infer_raw_token_ids(
        &self,
        tokens: &[u32],
        max_decode_tokens: usize,
    ) -> crate::Result<RawDecode> {
        self.infer_raw(tokens, max_decode_tokens)
    }

    /// [`Self::infer_raw_token_ids`] with the recurrent state swapped: `prompt`
    /// prefilled on a fresh slot — its K/V and its recurrence both computed
    /// over it — then the slot's delta-rule layers replaced by the ones
    /// `recurrent_over` leaves when run from zero, then `tail` prefilled and
    /// decoded argmax for up to `max_decode_tokens`. Fresh K/V under another
    /// recurrent state: what it answers says whether the K/V alone carries
    /// what the prompt holds. With `recurrent_over` absent nothing is
    /// installed, and the slot runs the two raw turns on its own state.
    pub fn infer_raw_token_ids_swapped(
        &self,
        prompt: &[u32],
        recurrent_over: Option<Vec<u32>>,
        tail: &[u32],
        max_decode_tokens: usize,
    ) -> crate::Result<RawDecode> {
        let sequence_id = self.new_raw_sequence()?;
        let out = (|| {
            self.raw_turn(sequence_id, prompt, 0)?;
            let state = match recurrent_over {
                Some(tokens) => run_state(&self.core.scheduler_tx, None, tokens)?,
                None => None,
            };
            if let Some(state) = state {
                // The delta-rule layers alone. The model's other state — the
                // index over its keys, its per-position history — describes the
                // K/V this slot holds, which is the prompt's and stays.
                let layers_only = SharedState {
                    aux: Arc::from(Vec::new()),
                    ..SharedState::from(&state)
                };
                let (tx, rx) = flume::bounded(1);
                self.core
                    .scheduler_tx
                    .send(SchedulerRequest::InstallRecurrentState {
                        sequence_ids: vec![sequence_id],
                        state: layers_only,
                        response_tx: tx,
                    })
                    .map_err(|_| ConversationError::SchedulerGone)?;
                rx.recv().map_err(|_| ConversationError::SchedulerGone)??;
            }
            self.raw_turn(sequence_id, tail, max_decode_tokens)
        })();
        let _ = self
            .core
            .scheduler_tx
            .send(SchedulerRequest::FreeSequence { sequence_id });
        out
    }

    /// The decode of [`Self::infer_raw_tokens`]: the ids it produced, and the
    /// text the turn finished with when it finished.
    fn infer_raw(&self, tokens: &[u32], max_decode_tokens: usize) -> crate::Result<RawDecode> {
        let sequence_id = self.new_raw_sequence()?;
        let out = self.raw_turn(sequence_id, tokens, max_decode_tokens);
        // Always release the sequence slot — RULER spawns one sequence per
        // sample and never reuses it, so without this the KV pool leaks
        // until OOM.
        let _ = self
            .core
            .scheduler_tx
            .send(SchedulerRequest::FreeSequence { sequence_id });
        out
    }

    /// A fresh slot bound to nothing: no projection, no substrate write.
    fn new_raw_sequence(&self) -> crate::Result<SequenceId> {
        let (resp_tx, resp_rx) = flume::bounded(1);
        self.core
            .scheduler_tx
            .send(SchedulerRequest::NewSequence {
                conversation: self.conversation.clone(),
                // Raw RULER eval path: no projection, no substrate
                // write, so no target binding either.
                target: None,
                parent: None,
                response_tx: resp_tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        resp_rx
            .recv()
            .map_err(|_| ConversationError::SchedulerGone)?
    }

    /// One raw turn on `sequence_id`: `tokens` prefilled after what the slot
    /// holds, then up to `max_decode_tokens` decoded argmax — none with `0`,
    /// the prefill alone. The scheduler carves a view over the parent's full
    /// block range and finalizes it onto the parent on Done; the raw slot keeps
    /// that K/V through the seal, so a later turn continues from here.
    fn raw_turn(
        &self,
        sequence_id: SequenceId,
        tokens: &[u32],
        max_decode_tokens: usize,
    ) -> crate::Result<RawDecode> {
        let (event_tx, event_rx) = flume::unbounded();
        self.core
            .scheduler_tx
            .send(SchedulerRequest::SubmitTurn {
                // One turn, and it is the slot's tail.
                seal_group: None,
                sequence_id,
                projection_inputs: None,
                prefill_tokens: TokenBuffer::from(tokens.to_vec()),
                prefill_text: String::new(),
                user_text: String::new(),
                user_content_start: 0,
                user_content_end: 0,
                assistant_content_start: 0,
                no_think: false,
                tags: Vec::new(),
                projection_offsets: Vec::new(),
                prefill_assistant_text: String::new(),
                post_decode_tokens: TokenBuffer::new(),
                max_decode_tokens,
                sampling: SamplingConfig::argmax(),
                event_tx,
                reprojection: None,
                // A raw slot continues what it holds: the assembly appends onto
                // a slot that already holds blocks rather than resetting it, so
                // its K/V stays beside the recurrence that ran over them. A
                // fresh slot holds nothing and assembles as any raw turn does.
                disable_reprojection: true,
                // Raw eval/summarisation path: no tools, no constrained decode.
                triggers: Arc::new(TriggerRegistry::new()),
                turn_grammar: None,
                free_tool_calls_from_penalties: false,
                recorded_reply: None,
                recurrent_spans: Vec::new(),
                image_spans: Vec::new(),
            })
            .map_err(|_| ConversationError::SchedulerGone)?;

        // Drain events until Done / Error.
        let mut collected: Vec<u32> = Vec::with_capacity(max_decode_tokens);
        let mut text_from_done: Option<String> = None;
        let mut finish = None;
        let mut last_error: Option<ConversationError> = None;
        while let Ok(ev) = event_rx.recv() {
            match ev {
                TurnEvent::Token(t) => collected.push(t),
                TurnEvent::Done(resp) => {
                    text_from_done = Some(resp.text);
                    finish = Some(resp.stats.finish);
                    break;
                }
                TurnEvent::Error(e) => {
                    last_error = Some(e);
                    break;
                }
                _ => {}
            }
        }
        if let Some(e) = last_error {
            return Err(e);
        }
        Ok(RawDecode {
            tokens: collected,
            text: text_from_done,
            finish,
        })
    }

    /// Get a `TokenDecoder` for decoding token IDs into text.
    ///
    /// The decoder is cheap to clone and can be used across threads.
    pub fn token_decoder(&self) -> TokenDecoder {
        TokenDecoder::new(Arc::clone(&self.core.tokenizer))
    }

    /// Durably flush the substrate redo log — the group-commit point.
    /// Call after a turn completes so an in-flight turn survives a crash.
    pub fn commit_persistence(&self) -> crate::Result<()> {
        Ok(self.conversation.commit_persistence()?)
    }

    /// Like [`Self::commit_persistence`] but skipped when nothing is staged.
    /// Returns `Ok(true)` when an `fsync` actually happened. The daemon's
    /// periodic flush task uses this so an idle workspace doesn't issue
    /// pointless syscalls — and so writes the persistence thread produced
    /// asynchronously aren't left stranded between turns.
    pub fn commit_persistence_if_pending(&self) -> crate::Result<bool> {
        Ok(self.conversation.commit_persistence_if_pending()?)
    }

    /// Force a full redo-log compaction (operator opt-in via the startup
    /// flag). Rewrites the log to just the live record set, reclaiming the
    /// dead weight that accrues from superseded turns and tombstoned
    /// timelines. The persistence thread also compacts automatically when
    /// the dead-byte ratio crosses the threshold. `progress` reports coarse
    /// phase progress (0..=5) for the loading screen.
    pub fn compact_substrate(
        &self,
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    ) -> crate::Result<()> {
        Ok(self.conversation.compact_substrate(progress)?)
    }

    /// Detach this handle's substrate from the engine, and stop the engine
    /// with the last one.
    ///
    /// Safe to call multiple times (idempotent). Takes `&self` so it can be
    /// called from a `&'static ConversationEngine` reference (e.g. a TLS drop
    /// guard in tests), which is necessary because Rust statics are never
    /// dropped and we must ensure the CUDA scheduler thread exits before the
    /// CUDA driver's atexit handler fires.
    ///
    /// While another substrate stays attached, the scheduler frees this one's
    /// slots and forgets it, and then its persistence thread drains and its
    /// writer flushes; the scheduler and the model go on serving the others.
    /// The last attachment stops the scheduler and releases the GPU.
    ///
    /// Either way the substrate's writer lock is released last, so the
    /// workspace can be opened again while this handle is still held.
    pub fn shutdown(&self) -> crate::Result<()> {
        if self.detached.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        // The request goes out under the lock that changed the list, so a
        // detach is always queued ahead of the `Shutdown` the last attachment
        // sends — and the scheduler answers it before it stops.
        let detached = {
            let mut attached = self
                .core
                .attachments
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            attached.list.retain(|(id, _)| *id != self.attachment);
            if attached.list.is_empty() {
                // The core's hold on the backings goes with it; the scheduler
                // and the persistence thread release theirs as they join.
                attached.persist_with = None;
                // **Before the scheduler is asked to stop.** A caller blocked
                // in `GuestReceipt::wait` is blocked on the scheduler thread;
                // once that thread joins there is nobody left to answer, and
                // the caller waits for the life of the process with its HTTP
                // request still open. Closing first also refuses anything
                // submitted during the teardown, so no job lands in a queue
                // nothing will drain.
                self.core.guests.queue.close();
                let _ = self.core.scheduler_tx.send(SchedulerRequest::Shutdown);
                None
            } else {
                let (tx, rx) = flume::bounded(1);
                let sent = self
                    .core
                    .scheduler_tx
                    .send(SchedulerRequest::DetachSubstrate {
                        id: self.attachment,
                        conversation: self.conversation.clone(),
                        response_tx: tx,
                    });
                Some(sent.ok().map(|()| rx))
            }
        };
        if let Some(reply) = detached {
            if let Some(rx) = reply {
                let _ = rx.recv();
            }
            // Nothing on the scheduler refers to this substrate any more, so
            // the drain captures its last work.
            self.persist_thread.shutdown();
            self.conversation.flush_writer();
            self.conversation.release_writer()?;
            return Ok(());
        }
        let handle = self
            .core
            .scheduler_handle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(h) = handle {
            h.join()
                .map_err(|_| ConversationError::Channel("scheduler thread panicked".into()))?;
        }
        // Tear down the persistence thread after the scheduler has
        // joined — by then no more turn-seals will fire triggers, so
        // the thread's final drain pass captures the last work. The
        // call is idempotent, so a redundant invocation from `Drop`
        // after this is a no-op.
        self.persist_thread.shutdown();
        // Tear down the summariser thread last: it depends on the
        // scheduler for §6 probes (via `ChannelProbeRunner` over the
        // scheduler request channel).  Once the scheduler has joined,
        // any in-flight probe request hangs forever — shutdown
        // signals the loop to exit on its next select rather than
        // wait for a probe response that will never arrive.
        self.core.summariser_thread.shutdown();
        // Terminal durability step: every enqueuer has now stopped (the
        // scheduler's seal path, the persistence thread's final hot→warm→cold
        // drain, and the summariser), so drain the off-thread writer's queue to
        // the redo log, fsync, and join it. This MUST run here because the daemon
        // force-exits (`std::process::exit`), which skips the writer's `Drop` —
        // without this flush, every warm→cold KV / tokens / sig append still in
        // the writer's queue at exit would be silently lost.
        self.conversation.flush_writer();
        // The scheduler thread owned every sequence, so by its join each
        // `(layer, slot)` has handed its staging pair to the process-wide
        // recycler. Nothing of this engine will take them again; unpin them
        // rather than leave them held for the life of the process. The same goes
        // for the recurrent stores' seal readback buffer: the scheduler was the
        // only thread that sealed.
        release_recycled_wc();
        release_recurrent_host_buffers();
        // Everything durable has landed, so the workspace is handed on: a
        // successor may open it while this engine is still held somewhere.
        self.conversation.release_writer()?;
        Ok(())
    }
}
