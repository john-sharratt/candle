//! Copy-engine promotions, issued from a thread of their own.
//!
//! A host-to-device copy on the copy stream runs asynchronously, but *issuing*
//! it is not always quick: on WDDM a submission from one thread can stall
//! behind the work another thread has queued. Recorded as a chain of graphs,
//! the forward thread queues a whole forward ahead of the GPU, and on
//! Qwen3.8-Flash-Next (RTX 3090, BF16 ×1 decode) a single promotion's
//! `cuMemcpyHtoDAsync` held the pipeline thread for 157–189 ms once per step —
//! the rest of that forward. The promotion ring it keeps stocked ran dry
//! behind it, and three quarters of the step's misses found no ring slot and
//! crossed the link a second time.
//!
//! So the pipeline thread never issues a copy itself. It hands each one to
//! this thread as a [`CopyJob`] and collects the ids of the copies that have
//! completed ([`Copier::completed`]). Everything else it does per routed layer
//! — the ring, the device's own promotions, the reclaim rule — is host memory
//! and mapped words, and keeps time whatever the driver is doing.

use candle::Result;
use cudarc::driver::sys;
use cudarc::driver::{CudaEvent, CudaStream};
use std::collections::VecDeque;
use std::sync::mpsc::{self, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// One promotion copy: `bytes` from the pinned host image at `src` into the
/// VRAM slot at `dst`. `id` names it in [`Copier::completed`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct CopyJob {
    pub(crate) id: u64,
    pub(crate) dst: u64,
    pub(crate) src: u64,
    pub(crate) bytes: usize,
}

enum Msg {
    Copy(CopyJob),
    /// Issue everything queued, wait for all of it, then answer.
    Flush(mpsc::SyncSender<()>),
}

/// What the copier has finished, waiting for the pipeline thread to collect.
#[derive(Default)]
struct Done {
    ids: Vec<u64>,
    /// The copier stopped on a driver error; nothing it holds will complete.
    failed: Option<String>,
}

/// The copier thread's handle, owned by the pipeline thread.
pub(crate) struct Copier {
    tx: Option<mpsc::Sender<Msg>>,
    done: Arc<Mutex<Done>>,
    thread: Option<JoinHandle<()>>,
}

impl Copier {
    /// Start the thread that issues copies on `stream`.
    pub(crate) fn spawn(stream: Arc<CudaStream>) -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        let done = Arc::new(Mutex::new(Done::default()));
        let thread_done = done.clone();
        let thread = std::thread::Builder::new()
            .name("expert-copier".into())
            .spawn(move || run(&stream, &rx, &thread_done))
            .map_err(|e| candle::Error::Msg(format!("expert copier: spawn failed: {e}")))?;
        Ok(Self {
            tx: Some(tx),
            done,
            thread: Some(thread),
        })
    }

    /// Queue a copy. The bytes at `src` must stay unwritten and the slot at
    /// `dst` unread until its id comes back from [`Self::completed`].
    pub(crate) fn submit(&self, job: CopyJob) -> Result<()> {
        self.sender()?
            .send(Msg::Copy(job))
            .map_err(|_| self.stopped())
    }

    /// The ids of the copies completed since the last call, in completion
    /// order — or the error the copier stopped on.
    pub(crate) fn completed(&self) -> Result<Vec<u64>> {
        let mut d = self
            .done
            .lock()
            .map_err(|_| candle::Error::Msg("expert copier: state poisoned".into()))?;
        if let Some(e) = &d.failed {
            candle::bail!("expert copier stopped: {e}");
        }
        Ok(std::mem::take(&mut d.ids))
    }

    /// Wait until every copy submitted so far has completed; their ids are
    /// then all in [`Self::completed`].
    pub(crate) fn flush(&self) -> Result<()> {
        let (resp_tx, resp_rx) = mpsc::sync_channel(1);
        self.sender()?
            .send(Msg::Flush(resp_tx))
            .map_err(|_| self.stopped())?;
        resp_rx.recv().map_err(|_| self.stopped())
    }

    fn sender(&self) -> Result<&mpsc::Sender<Msg>> {
        self.tx
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("expert copier: already shut down".into()))
    }

    fn stopped(&self) -> candle::Error {
        let why = self
            .done
            .lock()
            .ok()
            .and_then(|d| d.failed.clone())
            .unwrap_or_else(|| "thread gone".into());
        candle::Error::Msg(format!("expert copier stopped: {why}"))
    }
}

impl Drop for Copier {
    fn drop(&mut self) {
        // Closing the channel ends the thread once what it holds has landed.
        self.tx = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The copier thread: issue what arrives, and while copies are in flight with
/// nothing new to issue, wait on the oldest — a blocking wait, so it sleeps.
fn run(stream: &CudaStream, rx: &mpsc::Receiver<Msg>, done: &Mutex<Done>) {
    let fail = |e: String| {
        if let Ok(mut d) = done.lock() {
            d.failed = Some(e);
        }
    };
    if let Err(e) = stream.context().bind_to_thread() {
        fail(format!("could not bind the CUDA context: {e}"));
        return;
    }
    let mut inflight: VecDeque<(u64, CudaEvent)> = VecDeque::new();
    loop {
        let msg = if inflight.is_empty() {
            match rx.recv() {
                Ok(m) => Some(m),
                Err(_) => break,
            }
        } else {
            match rx.try_recv() {
                Ok(m) => Some(m),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => break,
            }
        };
        let step = match msg {
            Some(Msg::Copy(job)) => issue(stream, &job).map(|ev| inflight.push_back((job.id, ev))),
            Some(Msg::Flush(resp)) => stream
                .synchronize()
                .map_err(|e| format!("copy stream synchronize: {e}"))
                .map(|()| {
                    finish(done, inflight.drain(..).map(|(id, _)| id));
                    let _ = resp.send(());
                }),
            None => wait_oldest(stream, &mut inflight, done),
        };
        if let Err(e) = step {
            fail(e);
            break;
        }
    }
    // Nothing may still be writing a slot once the thread is gone — on a
    // failure too: copies issued before it are still landing, and the owner
    // joins this thread before it releases the pinned sources they read.
    let _ = stream.synchronize();
}

fn issue(stream: &CudaStream, job: &CopyJob) -> std::result::Result<CudaEvent, String> {
    // SAFETY: the submitter keeps `src` (a pinned slot image of at least
    // `bytes`) unwritten and the slot at `dst` unread until this id completes.
    unsafe {
        let src = std::slice::from_raw_parts(job.src as *const u8, job.bytes);
        cudarc::driver::result::memcpy_htod_async(job.dst, src, stream.cu_stream())
    }
    .map_err(|e| format!("promotion copy: {e}"))?;
    stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| format!("promotion event: {e}"))
}

/// Nothing new to issue: submit what was issued, then block on the oldest copy
/// and hand back it and every later one already complete.
fn wait_oldest(
    stream: &CudaStream,
    inflight: &mut VecDeque<(u64, CudaEvent)>,
    done: &Mutex<Done>,
) -> std::result::Result<(), String> {
    // SAFETY: a query on a stream this thread issues to — on WDDM it submits
    // the batched copies.
    unsafe {
        let _ = sys::cuStreamQuery(stream.cu_stream());
    }
    if let Some((_, ev)) = inflight.front() {
        ev.synchronize()
            .map_err(|e| format!("promotion wait: {e}"))?;
    }
    let mut landed = Vec::new();
    while inflight.front().is_some_and(|(_, ev)| ev.is_complete()) {
        if let Some((id, _)) = inflight.pop_front() {
            landed.push(id);
        }
    }
    finish(done, landed.into_iter());
    Ok(())
}

fn finish(done: &Mutex<Done>, ids: impl Iterator<Item = u64>) {
    if let Ok(mut d) = done.lock() {
        d.ids.extend(ids);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// Copies issued through the copier land their bytes, every id comes back
    /// once, and a flush returns only after all of them are done.
    #[test]
    fn copies_land_their_bytes_and_report_every_id() {
        let Ok(Device::Cuda(dev)) = Device::new_cuda(0) else {
            return;
        };
        let stream = dev.cuda_context().new_stream().unwrap();
        let copier = Copier::spawn(stream.clone()).unwrap();
        let host: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let mut pinned = unsafe { dev.cuda_context().alloc_pinned::<u8>(4096) }.unwrap();
        pinned.as_mut_slice().unwrap().copy_from_slice(&host);
        let src = pinned.as_ptr().unwrap() as u64;
        // On the copy stream, so the zeroing is ordered before the copies.
        let dst = stream.alloc_zeros::<u8>(4096).unwrap();
        let (base, _g) = cudarc::driver::DevicePtr::device_ptr(&dst, &stream);
        for (id, off) in [(7u64, 0usize), (9, 1024), (11, 2048), (13, 3072)] {
            copier
                .submit(CopyJob {
                    id,
                    dst: base + off as u64,
                    src: src + off as u64,
                    bytes: 1024,
                })
                .unwrap();
        }
        copier.flush().unwrap();
        let mut ids = copier.completed().unwrap();
        ids.sort_unstable();
        assert_eq!(ids, vec![7, 9, 11, 13]);
        assert!(copier.completed().unwrap().is_empty(), "each id once");
        drop(_g);
        let back = stream.memcpy_dtov(&dst).unwrap();
        assert_eq!(back, host);
    }
}
