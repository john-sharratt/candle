//! Starts the pipeline thread, which holds the device context and serves the
//! forward's routed layers.
//!
//! The thread binds the device's CUDA context to itself when it starts, and it
//! is the thread that builds the views of landed slots and issues the relocations
//! of a boundary move, which need that context. A guard marks the thread dead
//! and raises the abort word when it is dropped, including on unwind. Every
//! worker waiting on the thread then gives its expert up and the forward fails,
//! rather than running against state that can no longer be trusted.

use super::*;

impl PipelineState {
    /// Make the device's CUDA context current on the calling thread — the
    /// pipeline thread, at start.
    pub(super) fn bind_device_to_thread(&self) -> Result<()> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(());
        };
        cd.cuda_stream()
            .context()
            .bind_to_thread()
            .map_err(candle::Error::wrap)
    }
}

// Thread spawn

/// Marks the thread dead and raises the abort word when dropped — including on
/// unwind. Nothing the GPU waits on comes from this thread, but a dead pipeline
/// means the cache's state can no longer be trusted, so every waiting worker
/// gives its expert up and the forward fails (`fault`).
pub(super) struct DeadFlagGuard {
    pub(super) dead: Arc<AtomicBool>,
    pub(super) abort: Arc<AbortWord>,
}

impl Drop for DeadFlagGuard {
    fn drop(&mut self) {
        self.abort.raise();
        self.dead.store(true, Ordering::Release);
    }
}

/// Spawn the pipeline thread. Returns the sender for routed layers and the
/// rest of the message set.
pub(crate) fn spawn_pipeline_thread(
    mut state: PipelineState,
    dead_flag: Arc<AtomicBool>,
) -> mpsc::SyncSender<PipelineMessage> {
    let (tx, rx) =
        mpsc::sync_channel::<PipelineMessage>(super::super::dispatch::PIPELINE_CHANNEL_BOUND);

    std::thread::Builder::new()
        .name("expert-pipeline".into())
        .spawn(move || {
            let _dead_on_exit = DeadFlagGuard {
                dead: dead_flag,
                abort: state.abort.clone(),
            };
            // This thread builds the views of landed slots and issues the
            // boundary moves' relocations, which need the device's context
            // current on it.
            if let Err(e) = state.bind_device_to_thread() {
                tracing::error!(
                    target: "candle_transformers::expert_lre",
                    "expert pipeline: could not bind the CUDA context to its thread: {e}"
                );
                return;
            }
            while let Ok(msg) = rx.recv() {
                match msg {
                    PipelineMessage::Routed(routed) => {
                        // Hand-off latency: forward-thread `send` → pickup here.
                        state.profile.record("pipe_inbound", routed.submitted_at);
                        let wt = profile_now();
                        let row = routed.row;
                        if let Err(e) = state.process_routed(routed) {
                            tracing::error!(
                                target: "candle_transformers::expert_lre",
                                row,
                                "expert pipeline: routed layer could not be served, aborting: {e}"
                            );
                            return;
                        }
                        state.profile.record("pipe_worker_total", wt);
                    }
                    PipelineMessage::Settle { response_tx } => {
                        // Land whatever has completed, so a reader of the
                        // counters sees it.
                        let near = state.routed_served.load(Ordering::Acquire);
                        if let Err(e) = state
                            .collect_device_promotions(near)
                            .and_then(|()| state.land_device_promotions(false))
                        {
                            tracing::error!(
                                target: "candle_transformers::expert_lre",
                                "expert pipeline: a promotion could not land, aborting: {e}"
                            );
                            return;
                        }
                        let _ = response_tx.send(());
                    }
                    PipelineMessage::DropDemandPromotions { response_tx } => {
                        let near = state.routed_served.load(Ordering::Acquire);
                        if let Err(e) = state.drop_demand_promotions(near) {
                            tracing::error!(
                                target: "candle_transformers::expert_lre",
                                "expert pipeline: a faulted forward's promotions could not be \
                                 dropped, aborting: {e}"
                            );
                            return;
                        }
                        let _ = response_tx.send(());
                    }
                    PipelineMessage::SnapshotProfile { response_tx } => {
                        let snap = state.profile.snapshot();
                        state.profile.reset();
                        let _ = response_tx.send(snap);
                    }
                    // The give-back, reachable without a completed forward.
                    // Answering zero is a legitimate outcome (nothing spare, a
                    // wave still live, an invocation still unserved).
                    PipelineMessage::RenegotiateBoundary { ask, response_tx } => {
                        let conceded = match state.renegotiate_if_quiet(ask) {
                            Ok(bytes) => bytes,
                            Err(e) => {
                                tracing::warn!("requested boundary move failed: {e}");
                                0
                            }
                        };
                        let _ = response_tx.send(conceded);
                    }
                }
            }
        })
        .expect("failed to spawn expert-pipeline thread");

    tx
}
