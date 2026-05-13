//! Background float→quant conversion worker.
//!
//! Callers enqueue pre-resolved `WorkItem`s (full migration vectors with arena
//! keys and compression policy already resolved).  The worker thread processes
//! each item using its own private `PinnedStager` bound to the bg CUDA stream,
//! then signals any waiters registered via `join()`.
//!
//! `BackgroundQuantizer` is a cheap `Clone`: it is a newtype over
//! `Arc<BgQuantInner>`, so all clones share the same worker thread.

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::{CudaContext, CudaStream};
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::PinnedStager;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use super::backing::BackingInner;
#[cfg(feature = "cuda")]
use super::head_gids::HeadGids;
#[cfg(feature = "cuda")]
use super::CompressionPolicy;

#[cfg(feature = "cuda")]
type Waiter = Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub(crate) struct ChunkMigration {
    pub layer_idx: usize,
    pub batch_idx: usize,
    pub block_idx: usize,
    pub head_gids: HeadGids,
}

/// A single unit of float→quant work for one layer.
#[cfg(feature = "cuda")]
pub(crate) struct WorkItem {
    pub migrations: Vec<ChunkMigration>,
}

#[cfg(feature = "cuda")]
struct QuantizerState {
    pending: Vec<WorkItem>,
    waiters: Vec<Waiter>,
}

#[cfg(feature = "cuda")]
struct QuantizerShared {
    state: std::sync::Mutex<QuantizerState>,
    condvar: std::sync::Condvar,
}

/// Private inner state — owned by the Arc inside BackgroundQuantizer.
#[cfg(feature = "cuda")]
struct BgQuantInner {
    shared: Arc<QuantizerShared>,
    #[allow(dead_code)]
    bg_stream: Option<Arc<CudaStream>>,
    _thread: std::thread::JoinHandle<()>,
    #[allow(dead_code)]
    compression: Option<CompressionPolicy>,
    #[allow(dead_code)]
    backing: std::sync::Weak<BackingInner>,
}

/// Background quantization worker.
///
/// Cheap to clone: each clone shares the same underlying worker thread via `Arc`.
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct BackgroundQuantizer(Arc<BgQuantInner>);

#[cfg(feature = "cuda")]
impl std::fmt::Debug for BackgroundQuantizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pending = self
            .0
            .shared
            .state
            .lock()
            .map(|g| g.pending.len())
            .unwrap_or(0);
        f.debug_struct("BackgroundQuantizer")
            .field("pending", &pending)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
impl BackgroundQuantizer {
    /// Spawn the background worker thread.
    ///
    /// `device` is used to extract the CUDA context for `bind_to_thread` and
    /// the bg stream for the private `PinnedStager`.
    pub(crate) fn new(
        device: &candle::Device,
        compression: Option<CompressionPolicy>,
        backing_weak: std::sync::Weak<BackingInner>,
    ) -> candle::Result<Self> {
        let cuda_context = match device {
            candle::Device::Cuda(d) => Some(d.cuda_context().clone()),
            _ => None,
        };
        let bg_stream = match device {
            candle::Device::Cuda(d) => d.cuda_bg_stream(),
            _ => candle::bail!("BackgroundQuantizer requires a CUDA device"),
        };
        let device = device.clone();
        let compression_clone = compression.clone();
        let thread_backing = backing_weak.clone();
        let shared = Arc::new(QuantizerShared {
            state: std::sync::Mutex::new(QuantizerState {
                pending: Vec::new(),
                waiters: Vec::new(),
            }),
            condvar: std::sync::Condvar::new(),
        });
        let thread_shared = shared.clone();
        let thread_stream = bg_stream.clone();
        let thread = std::thread::Builder::new()
            .name("bg-quantizer".into())
            .spawn(move || {
                Self::run_loop(
                    thread_shared,
                    device,
                    cuda_context,
                    thread_stream,
                    compression_clone,
                    thread_backing,
                )
            })
            .map_err(|e| candle::Error::Msg(format!("bg-quantizer: spawn failed: {e}")))?;
        Ok(Self(Arc::new(BgQuantInner {
            shared,
            bg_stream: Some(bg_stream),
            _thread: thread,
            compression,
            backing: backing_weak,
        })))
    }

    /// Create a no-op quantizer for non-CUDA devices (CPU tests).
    /// The thread runs but discards all work items; join() returns immediately.
    pub fn noop() -> Self {
        let shared = Arc::new(QuantizerShared {
            state: std::sync::Mutex::new(QuantizerState {
                pending: Vec::new(),
                waiters: Vec::new(),
            }),
            condvar: std::sync::Condvar::new(),
        });
        let thread_shared = shared.clone();
        let thread = std::thread::Builder::new()
            .name("bg-quantizer-noop".into())
            .spawn(move || loop {
                let waiters: Vec<Waiter> = {
                    let mut guard = thread_shared.state.lock().unwrap();
                    loop {
                        if !guard.pending.is_empty() || !guard.waiters.is_empty() {
                            guard.pending.clear();
                            break guard.waiters.drain(..).collect();
                        }
                        guard = match thread_shared.condvar.wait(guard) {
                            Ok(g) => g,
                            Err(_) => return,
                        };
                    }
                };
                for waiter in waiters {
                    let (lock, cvar) = &*waiter;
                    *lock.lock().unwrap() = true;
                    cvar.notify_one();
                }
            })
            .expect("bg-quantizer-noop: spawn failed");
        Self(Arc::new(BgQuantInner {
            shared,
            bg_stream: None,
            _thread: thread,
            compression: None,
            backing: std::sync::Weak::new(),
        }))
    }

    /// Block until the background thread has finished all currently-pending work.
    ///
    /// Registers a waiter, wakes the thread, then sleeps until the thread
    /// signals it after draining the batch (including the case of no work).
    pub fn join(&self) {
        let waiter: Waiter = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        {
            let mut guard = self.0.shared.state.lock().unwrap();
            guard.waiters.push(waiter.clone());
            self.0.shared.condvar.notify_one();
        }
        let (lock, cvar) = &*waiter;
        let mut done = lock.lock().unwrap();
        while !*done {
            done = cvar.wait(done).unwrap();
        }
    }

    /// Enqueue a pre-resolved work item for background quantization.
    pub(crate) fn enqueue_work_item(&self, item: WorkItem) {
        let mut guard = self.0.shared.state.lock().unwrap();
        guard.pending.push(item);
        drop(guard);
        self.0.shared.condvar.notify_one();
    }

    /// Enqueue all items under a single lock acquisition.
    /// Prefer this over repeated `enqueue_work_item` calls when batching multiple layers.
    pub(crate) fn enqueue_work_items_batch(&self, items: Vec<WorkItem>) {
        if items.is_empty() {
            return;
        }
        let mut guard = self.0.shared.state.lock().unwrap();
        guard.pending.extend(items);
        drop(guard);
        self.0.shared.condvar.notify_one();
    }

    fn run_loop(
        shared: Arc<QuantizerShared>,
        device: candle::Device,
        cuda_context: Option<Arc<CudaContext>>,
        bg_stream: Arc<CudaStream>,
        compression: Option<CompressionPolicy>,
        backing_weak: std::sync::Weak<BackingInner>,
    ) {
        // Bind the CUDA context to this thread before any CUDA operations.
        // Without this, cuMemHostAlloc and kernel launches fail with
        // CUDA_ERROR_INVALID_CONTEXT because new OS threads start with no
        // active context.
        if let Some(ref ctx) = cuda_context {
            if let Err(e) = ctx.bind_to_thread() {
                eprintln!("bg-quantizer: failed to bind CUDA context: {e:?}");
                return;
            }
        }

        // Allocate our private stager bound to the bg stream after bind_to_thread.
        let stager = match &device {
            candle::Device::Cuda(d) => PinnedStager::with_stream(d, bg_stream.clone()),
            _ => PinnedStager::noop(),
        };

        loop {
            let (items, waiters): (Vec<WorkItem>, Vec<Waiter>) = {
                let mut guard = shared.state.lock().unwrap();
                loop {
                    if !guard.pending.is_empty() || !guard.waiters.is_empty() {
                        let items = guard.pending.drain(..).collect();
                        let waiters = guard.waiters.drain(..).collect();
                        break (items, waiters);
                    }
                    guard = match shared.condvar.wait(guard) {
                        Ok(g) => g,
                        Err(_) => return,
                    };
                }
            };

            let generation = stager.begin_generation();

            let arc_backing = match backing_weak.upgrade() {
                Some(b) => b,
                None => {
                    eprintln!("bg_quantizer: BackingInner dropped, skipping work");
                    continue;
                }
            };

            let migrations = items
                .into_iter()
                .flat_map(|item| item.migrations.into_iter())
                .collect::<Vec<_>>();
            if !migrations.is_empty() {
                let result = arc_backing.reconcile_batch_float_to_quant_v2(
                    &arc_backing,
                    migrations,
                    compression.as_ref(),
                    &generation,
                    &bg_stream,
                    true,
                );
                if let Err(e) = result {
                    eprintln!("bg_quantizer: reconcile failed: {e:?}");
                }
            }

            drop(generation);

            // Signal all callers blocked in `join()` — always, even when no items
            // were present, so join() never deadlocks.
            for waiter in waiters {
                let (lock, cvar) = &*waiter;
                let mut done = lock.lock().unwrap();
                *done = true;
                cvar.notify_one();
            }
        }
    }
}
