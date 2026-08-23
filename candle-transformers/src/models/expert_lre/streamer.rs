//! Off-thread expert streamer — the byte-mover for whole-layer prefill
//! streaming.
//!
//! During a prefill-width wave, the next layer will route most of its experts,
//! so there is nothing to predict — the win is turning the layer's expert
//! traffic from on-demand stalls inside `submit` into copy-stream DMA
//! overlapped with the current layer's compute. Running those loads on the
//! pipeline thread was measured to REGRESS bulk throughput (cfg8 437→416):
//! the next layer's work request queued behind ~170 loads instead of
//! overlapping them. This thread exists so the reads happen OFF the pipeline
//! thread.
//!
//! ## Division of labour
//!
//! The **pipeline thread** keeps every piece of bookkeeping: it chooses the
//! target layer, evicts for capacity, allocates slots, installs the slot
//! VIEWS (`build_slot_view` — pure geometry + address, no bytes), and later
//! uninstalls any job the streamer reports failed. This thread moves **bytes
//! only**: pack reads into its own staging ring, warm-tier reads, and H2D
//! enqueues on its own CUDA stream. It never touches `ExpertCacheInner`.
//!
//! ## Protocol
//!
//! One [`StreamPlan`] per target layer, at most one in flight per layer
//! (enforced by the issuer). The plan carries a compute-stream event the
//! streamer's stream waits BEFORE any H2D — the target slots were just
//! evicted from layers whose GEMMs may still be reading them. When every
//! job's copy is enqueued, the streamer records a fence event and answers on
//! the plan's own channel with [`StreamDone`]: the fence plus the jobs that
//! failed (whose installed views the issuer must take back). The issuer joins
//! the plan when the target layer's work request arrives — blocking only if
//! the streamer has not yet finished ENQUEUEING, with the fence covering the
//! in-flight DMA from there (the same wait-at-need shape as the prefetch
//! fence ring, which is where the fence is recorded).

use super::pack::{ExpertPack, PackRead};
use super::pinned::{LayerGeometry, WarmPool};
use super::pipeline::{build_slot_from_record_on_stream, ColdStaging};
use super::types::PipelineStats;
use candle::CudaDevice;
use cudarc::driver::{CudaEvent, CudaStream};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

/// Staging-ring depth for the streamer's cold reads. Smaller than the demand
/// path's ring: the streamer's reads overlap compute rather than stalling it,
/// so ring depth buys concurrency, not latency — and each buffer is a pinned
/// record stride (~14 MB on the 284B target).
const STREAM_STAGING_BUFFERS: usize = 16;

/// One expert's byte-move: everything resolved by the issuer so the streamer
/// needs no residency or zone state.
pub(crate) struct StreamJob {
    pub expert_idx: usize,
    pub slot_idx: usize,
    /// Device base address of the (already installed) slot.
    pub slot_base: u64,
    /// `Some(warm_slot)` → pinned-host H2D; `None` → pack read then H2D.
    pub warm_slot: Option<usize>,
}

/// A batch of byte-moves for one target layer.
pub(crate) struct StreamPlan {
    pub target_layer: usize,
    pub jobs: Vec<StreamJob>,
    /// Compute-stream event the streamer's stream must wait before writing:
    /// the destination slots' previous tenants may still be under read.
    pub after: Option<CudaEvent>,
    /// Where the outcome goes — one channel per plan, owned by the issuer's
    /// pending-streams map.
    pub done_tx: mpsc::Sender<StreamDone>,
}

/// Outcome of a [`StreamPlan`].
pub(crate) struct StreamDone {
    /// Event recorded after the last enqueued copy — the plan's fence. `None`
    /// when nothing was enqueued (all jobs failed, or the event could not be
    /// recorded and the stream was synchronized instead).
    pub fence: Option<CudaEvent>,
    /// Jobs whose bytes never moved: the issuer must uninstall their slots
    /// and let the demand path reload them.
    pub failed: Vec<(usize, usize)>,
    /// Bytes enqueued on the copy stream — the issuer folds this into its
    /// per-pass DMA accounting.
    pub bytes: usize,
}

pub(crate) enum StreamCmd {
    Load(StreamPlan),
}

/// The streamer's owned world: read-only shares of the two source tiers plus
/// its own staging ring and CUDA stream.
pub(crate) struct StreamerCtx {
    pub pack: Arc<ExpertPack>,
    pub warm: Arc<WarmPool>,
    pub layer_geometries: Arc<Vec<LayerGeometry>>,
    pub cuda_dev: CudaDevice,
    pub stream: Arc<CudaStream>,
    pub stats: Arc<Mutex<PipelineStats>>,
}

/// Handle owned by the pipeline thread. Dropping it closes the command
/// channel (ending the worker loop) and joins the thread, so the streamer's
/// staging buffers and stream never outlive the engine.
pub(crate) struct StreamerHandle {
    tx: Option<mpsc::SyncSender<StreamCmd>>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl StreamerHandle {
    pub(crate) fn send(&self, plan: StreamPlan) -> bool {
        self.tx
            .as_ref()
            .is_some_and(|tx| tx.send(StreamCmd::Load(plan)).is_ok())
    }
}

impl Drop for StreamerHandle {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// Spawn the streamer thread. Returns `None` when its staging ring cannot be
/// allocated — streaming is an optimisation, so the engine simply runs
/// without it.
pub(crate) fn spawn_streamer_thread(ctx: StreamerCtx) -> Option<StreamerHandle> {
    let stride = ctx.pack.stride();
    let staging = match ColdStaging::new(stride, STREAM_STAGING_BUFFERS) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                target: "candle_transformers::expert_lre::streamer",
                "streamer staging ring unavailable ({e}); expert streaming disabled"
            );
            return None;
        }
    };
    // Depth 2: the issuer sends at most one plan per layer and joins the
    // previous target before issuing far ahead, so the channel never grows.
    let (tx, rx) = mpsc::sync_channel::<StreamCmd>(2);
    let join = std::thread::Builder::new()
        .name("expert-streamer".into())
        .spawn(move || worker_loop(ctx, staging, rx))
        .ok()?;
    Some(StreamerHandle {
        tx: Some(tx),
        join: Some(join),
    })
}

fn worker_loop(ctx: StreamerCtx, mut staging: ColdStaging, rx: mpsc::Receiver<StreamCmd>) {
    while let Ok(StreamCmd::Load(plan)) = rx.recv() {
        let done = run_plan(&ctx, &mut staging, &plan);
        // The issuer may already have drained/dropped its receiver (pass
        // boundary teardown) — a failed send is fine, the fence's copies are
        // still stream-ordered behind whatever waits the stream next.
        let _ = plan.done_tx.send(done);
    }
}

/// Move every job's bytes, tolerating per-job failure: streaming is advisory,
/// so one unreadable record costs one uninstall, never the plan.
fn run_plan(ctx: &StreamerCtx, staging: &mut ColdStaging, plan: &StreamPlan) -> StreamDone {
    let stride = ctx.pack.stride();
    let layout = ctx.pack.layout(plan.target_layer);
    let geom = &ctx.layer_geometries[plan.target_layer];
    let mut failed: Vec<(usize, usize)> = Vec::new();
    let mut enqueued = 0usize;

    if let Some(after) = &plan.after {
        if ctx.stream.wait(after).is_err() {
            // Without the ordering guarantee the writes could race the
            // previous tenants' reads — fail the whole plan.
            return StreamDone {
                fence: None,
                failed: plan
                    .jobs
                    .iter()
                    .map(|j| (j.expert_idx, j.slot_idx))
                    .collect(),
                bytes: 0,
            };
        }
    }

    // Cold jobs first, chunked through the staging ring with one concurrent
    // striped read per chunk (the same shape as the demand loader); their
    // H2Ds enqueue as each chunk lands. Warm jobs follow as pure pinned→VRAM
    // H2Ds.
    let cold: Vec<&StreamJob> = plan.jobs.iter().filter(|j| j.warm_slot.is_none()).collect();
    for chunk in cold.chunks(STREAM_STAGING_BUFFERS) {
        let idxs = match staging.acquire_many(chunk.len()) {
            Ok(v) => v,
            Err(_) => {
                failed.extend(chunk.iter().map(|j| (j.expert_idx, j.slot_idx)));
                continue;
            }
        };
        let read_ok = {
            match staging.buffers_mut_for(&idxs, stride) {
                Ok(bufs) => {
                    let reads: Vec<PackRead<'_>> = chunk
                        .iter()
                        .zip(bufs)
                        .map(|(j, dest)| PackRead {
                            layer: plan.target_layer,
                            expert: j.expert_idx,
                            dest,
                        })
                        .collect();
                    ctx.pack.read_many_unverified(reads).is_ok()
                }
                Err(_) => false,
            }
        };
        if !read_ok {
            failed.extend(chunk.iter().map(|j| (j.expert_idx, j.slot_idx)));
            continue;
        }
        for (j, &buf_idx) in chunk.iter().zip(&idxs) {
            // SAFETY: `slot_base` names a slot the zone handed the issuer and
            // holds installed until this plan resolves; overwriting it is the
            // point. The slot VIEW built here is dropped — the issuer already
            // installed its own identical view at plan time.
            let up = unsafe {
                build_slot_from_record_on_stream(
                    staging.buffer_ref(buf_idx, stride),
                    layout,
                    geom,
                    &ctx.cuda_dev,
                    &ctx.stream,
                    j.slot_base,
                    None,
                )
            };
            match up {
                Ok(_view) => {
                    enqueued += 1;
                    if let Ok(event) = ctx.stream.record_event(None) {
                        staging.publish(buf_idx, event);
                    }
                    if let Ok(mut s) = ctx.stats.lock() {
                        s.cold_loads += 1;
                    }
                }
                Err(_) => failed.push((j.expert_idx, j.slot_idx)),
            }
        }
    }

    for j in plan.jobs.iter().filter(|j| j.warm_slot.is_some()) {
        let warm_slot = j.warm_slot.expect("filtered Some");
        // SAFETY: as above; the warm tier is immutable, so the source slice
        // is stable for the copy's lifetime.
        let up = unsafe {
            build_slot_from_record_on_stream(
                ctx.warm.slot_ref(warm_slot, stride),
                layout,
                geom,
                &ctx.cuda_dev,
                &ctx.stream,
                j.slot_base,
                None,
            )
        };
        match up {
            Ok(_view) => {
                enqueued += 1;
                if let Ok(mut s) = ctx.stats.lock() {
                    s.warm_loads += 1;
                }
            }
            Err(_) => failed.push((j.expert_idx, j.slot_idx)),
        }
    }

    let fence = if enqueued > 0 {
        match ctx.stream.record_event(None) {
            Ok(e) => Some(e),
            Err(_) => {
                // No fence means no wait-at-need cover: drain inline rather
                // than let a hit compute on half-copied bytes.
                let _ = ctx.stream.synchronize();
                None
            }
        }
    } else {
        None
    };
    if let Ok(mut s) = ctx.stats.lock() {
        s.stream_loads += enqueued;
        s.dma_loads += enqueued;
    }
    StreamDone {
        fence,
        failed,
        bytes: enqueued * stride,
    }
}

/// Immutable-share marker for the warm pool: see `pinned.rs` for the `Sync`
/// justification (static post-startup; `Arc` makes `&mut` unreachable).
#[allow(dead_code)]
fn _assert_ctx_send()
where
    StreamerCtx: Send,
{
}
