//! The wave capture: one forward's launches recorded as a chain of graphs.
//!
//! A [`CaptureHub`] belongs to a device and is shared by every clone of its
//! handle. While a wave capture is open, the thread that opened it gets the
//! hub's capture stream from [`crate::cuda_backend::CudaDevice::cuda_stream`],
//! so every launch that thread issues through the device — candle's own ops,
//! the custom launchers, cuBLAS — is recorded instead of executed. Every other
//! thread keeps getting the compute stream, so the persistence thread and the
//! expert pipeline run exactly as before.
//!
//! **A segment ends wherever the host has to interact with the device.** The
//! driver refuses a synchronise, an allocation, a free or a host upload from
//! the capturing thread, and lets a readback through unordered against the
//! recorded work. So each of those, at the device method that issues it,
//! first [pauses](CaptureHub::pause) the wave: the open segment is ended,
//! folded into its executable and launched into the compute stream, the eager
//! call then runs in order behind it, and recording resumes on the next
//! launch site. The MoE host protocol ends a segment the same way
//! ([`CaptureHub::flush`]), because its stager polls for the bucketize the
//! segment holds. What the chain changes is when launches reach the driver —
//! batched per segment instead of one at a time — never their order.
//!
//! **Per-wave scalars and addresses are refreshed, not keyed.** Segment `i` of
//! every wave folds with `cuGraphExecUpdate` into the executable its ordinal
//! keeps for that node count (`slot`), which rewrites kernel parameters in
//! place when the recorded topology matches and is re-instantiated when it
//! does not. A wave whose shape, tier placement or launch scalars differ from
//! the last one therefore replays correctly with no key and no epoch to check.

use super::slot::{ExecSlot, Folded};
use super::staging::StagingRing;
use super::GraphError;
use crate::cuda_backend::{CudaDevice, WrapErr};
use crate::Result;
use cudarc::driver::{result, sys, CudaEvent, CudaStream};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::ThreadId;
use std::time::Instant;

/// What the wave captures on a device have done since it was created.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CaptureStats {
    /// Waves that ran as a chain.
    pub waves: u64,
    /// Segments launched — one graph launch each.
    pub segments: u64,
    /// Segments folded into an existing executable in place.
    pub updated: u64,
    /// Of those, the ones folded into an executable last shaped differently —
    /// grids or functions rewritten as well as arguments, the costly fold.
    pub reshaped: u64,
    /// Segments that needed a fresh instantiation.
    pub instantiated: u64,
    /// Host time, in µs, folding segments in place into an executable already of
    /// their shape — arguments only.
    pub in_place_us: u64,
    /// Host time, in µs, folding segments into an executable of another shape.
    pub reshaped_us: u64,
    /// Host time, in µs, instantiating and uploading segments.
    pub instantiated_us: u64,
    /// Graph nodes launched — kernels, memsets, device copies and event
    /// records alike.
    pub nodes: u64,
}

/// One device's wave-capture state, shared by every clone of its handle.
#[derive(Default)]
pub(crate) struct CaptureHub {
    /// Set while a wave capture is open on any thread: the fast path of every
    /// stream lookup reads only this.
    active: AtomicBool,
    /// [`thread_tag`] of the thread the open wave belongs to. A thread whose
    /// tag differs is not that thread, so it answers without the lock; an
    /// equal tag is confirmed against the exact [`ThreadId`] under it.
    owner_tag: AtomicU64,
    state: Mutex<HubState>,
}

/// The calling thread's id folded into a word: a deterministic function of
/// the id, so unequal tags are different threads. A fold rather than a keyed
/// hash because every launch on a recording device asks for it.
fn thread_tag() -> u64 {
    let mut fold = IdFold(0);
    std::thread::current().id().hash(&mut fold);
    fold.0
}

struct IdFold(u64);

impl Hasher for IdFold {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = self.0.rotate_left(32) ^ n;
    }
}

/// Released device memory, dropped by the caller once the hub's lock is
/// released — so an item whose own drop retires or pauses cannot deadlock.
type Graveyard = Vec<Box<dyn Send>>;

#[derive(Default)]
struct HubState {
    wave: Option<Wave>,
    /// The non-blocking stream launches are recorded on, made on first use.
    capture: Option<Arc<CudaStream>>,
    /// The executables of each segment ordinal, kept across waves.
    slots: Vec<ExecSlot>,
    /// Device memory released while recording, freed once the segment that
    /// may still read it has been launched — see [`CaptureHub::retire`].
    graveyard: Graveyard,
    /// The two staging rings recorded uploads go through, alternating by wave
    /// — see [`super::staging`].
    rings: Vec<StagingRing>,
    /// The compute stream the wave launches into, kept for the hand-offs.
    compute: Option<Arc<CudaStream>>,
    /// The two hand-off events — see [`HubState::hand_to_capture`].
    to_capture: Option<CudaEvent>,
    to_compute: Option<CudaEvent>,
    stats: CaptureStats,
}

impl HubState {
    /// Order the capture stream behind everything issued on the compute stream
    /// so far — a device-side wait, no host wait.
    ///
    /// A launcher that took its stream from [`CaptureHub::launch_stream`] while
    /// recording and launches after an eager section has begun holds the
    /// capture stream, which is no longer capturing: its launch runs at once.
    /// Without this it would run unordered against the segments still in
    /// flight on the compute stream; with it, it runs after them, exactly where
    /// it was issued.
    fn hand_to_capture(&self) -> Result<()> {
        if let (Some(capture), Some(compute), Some(ev)) =
            (&self.capture, &self.compute, &self.to_capture)
        {
            ev.record(compute).w()?;
            capture.wait(ev).w()?;
        }
        Ok(())
    }

    /// The other direction, before recording resumes: whatever such a launch
    /// put on the capture stream completes before anything issued on the
    /// compute stream after it.
    fn hand_to_compute(&self) -> Result<()> {
        if let (Some(capture), Some(compute), Some(ev)) =
            (&self.capture, &self.compute, &self.to_compute)
        {
            ev.record(capture).w()?;
            compute.wait(ev).w()?;
        }
        Ok(())
    }
}

struct Wave {
    owner: ThreadId,
    /// Which wave this is, so a [`Paused`] guard acts only on its own.
    serial: u64,
    /// A capture is recording on the capture stream.
    recording: bool,
    /// Open [`Paused`] guards, plus one while `held`; recording resumes when
    /// the count reaches zero.
    paused: usize,
    /// The wave has not reached the launches it records yet.
    held: bool,
    /// The next segment's ordinal.
    segment: usize,
    /// A resume that failed inside a guard's `drop`, reported at the next
    /// segment boundary.
    error: Option<String>,
}

/// One forward's wave capture on the thread that opened it — see
/// [`CudaDevice::begin_wave_capture`]. [`Self::finish`] launches the last
/// segment; dropping it unfinished discards what is still recording, which is
/// only right when the forward has already failed.
///
/// Not `Send`: the driver ends a thread-local capture only on the thread that
/// began it, so a capture finished or dropped anywhere else could not be
/// closed and would leave the device's wave open for good.
#[must_use = "a wave capture that is not finished discards its last segment"]
pub struct WaveCapture {
    pub(crate) device: CudaDevice,
    pub(crate) finished: bool,
    pub(crate) _thread_bound: PhantomData<*const ()>,
}

impl WaveCapture {
    /// Launch the last segment and stop recording.
    pub fn finish(mut self) -> Result<()> {
        self.finished = true;
        self.device
            .capture_hub()
            .finish_wave(&self.device.compute_stream())
    }
}

impl Drop for WaveCapture {
    fn drop(&mut self) {
        if !self.finished {
            self.device
                .capture_hub()
                .abandon_wave(&self.device.compute_stream());
        }
    }
}

/// Recording is suspended on this thread until this drops. See
/// [`CaptureHub::pause`]. Not `Send`, for the same reason as
/// [`WaveCapture`]: resuming begins a capture on the dropping thread.
#[must_use = "recording resumes when the guard drops"]
pub struct Paused<'h> {
    hub: &'h CaptureHub,
    /// The wave this guard paused; a guard outliving it touches no other.
    serial: u64,
    _thread_bound: PhantomData<*const ()>,
}

impl Drop for Paused<'_> {
    fn drop(&mut self) {
        let mut st = self.hub.lock();
        let Some(wave) = st.wave.as_mut().filter(|w| w.serial == self.serial) else {
            return;
        };
        wave.paused -= 1;
        if wave.paused > 0 || wave.recording {
            return;
        }
        let capture = st.capture.clone().expect("a wave has a capture stream");
        let resumed = st.hand_to_compute().and_then(|()| begin(&capture));
        let wave = st.wave.as_mut().expect("checked above");
        match resumed {
            Ok(()) => wave.recording = true,
            Err(e) => wave.error = Some(e.to_string()),
        }
    }
}

fn begin(capture: &Arc<CudaStream>) -> Result<()> {
    capture.context().bind_to_thread().w()?;
    // SAFETY: the hub's own stream, not recording — a wave records on it only
    // between `begin` and `end_segment`, under the hub's lock.
    unsafe {
        result::stream::begin_capture(
            capture.cu_stream(),
            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
    }
    .w()
}

impl CaptureHub {
    fn lock(&self) -> MutexGuard<'_, HubState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The stream a launch from this thread goes on: the capture stream while
    /// this thread is recording a wave, `compute` otherwise.
    pub(crate) fn launch_stream(&self, compute: &Arc<CudaStream>) -> Arc<CudaStream> {
        if !self.owned_here() {
            return compute.clone();
        }
        let st = self.lock();
        match (&st.wave, &st.capture) {
            (Some(w), Some(c)) if w.recording && w.owner == std::thread::current().id() => {
                c.clone()
            }
            _ => compute.clone(),
        }
    }

    /// Whether a wave may be open on this thread — `false` is exact, `true`
    /// is confirmed under the lock. Lets every other thread's stream lookup
    /// pass the hub without taking its lock.
    fn owned_here(&self) -> bool {
        self.active.load(Ordering::Acquire)
            && self.owner_tag.load(Ordering::Acquire) == thread_tag()
    }

    /// Whether this thread has a wave capture open, recording or paused.
    pub(crate) fn capturing_here(&self) -> bool {
        self.owned_here()
            && self
                .lock()
                .wave
                .as_ref()
                .is_some_and(|w| w.owner == std::thread::current().id())
    }

    /// Open a wave capture on this thread.
    pub(crate) fn begin_wave(&self, compute: &Arc<CudaStream>) -> Result<()> {
        let mut st = self.lock();
        if st.wave.is_some() {
            crate::bail!("a wave capture is already open on this device");
        }
        if st.capture.is_none() {
            st.capture = Some(compute.context().new_stream().w()?);
        }
        if st.to_capture.is_none() || st.to_compute.is_none() {
            let flags = Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING);
            st.to_capture = Some(compute.context().new_event(flags).w()?);
            st.to_compute = Some(compute.context().new_event(flags).w()?);
        }
        st.compute = Some(compute.clone());
        // Anything a stale handle left on the capture stream after the last
        // wave runs before this one's work.
        st.hand_to_compute()?;
        while st.rings.len() < 2 {
            let ring = StagingRing::new(compute)?;
            st.rings.push(ring);
        }
        let ring = (st.stats.waves % 2) as usize;
        st.rings[ring].reopen()?;
        // Opened held: the forward's setup runs eagerly until the model marks
        // where its recorded launches begin (`record`).
        st.stats.waves += 1;
        st.wave = Some(Wave {
            owner: std::thread::current().id(),
            serial: st.stats.waves,
            recording: false,
            paused: 1,
            held: true,
            segment: 0,
            error: None,
        });
        // The tag before the flag: a thread that sees the wave active reads
        // its owner's tag.
        self.owner_tag.store(thread_tag(), Ordering::Release);
        self.active.store(true, Ordering::Release);
        Ok(())
    }

    /// Start recording this thread's wave, which opens held. Does nothing when
    /// this thread has no wave open or is already recording it.
    pub(crate) fn record(&self) -> Result<()> {
        if !self.active.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut st = self.lock();
        let capture = st.capture.clone();
        let Some(wave) = st.wave.as_mut() else {
            return Ok(());
        };
        if wave.owner != std::thread::current().id() || !wave.held {
            return Ok(());
        }
        wave.held = false;
        wave.paused -= 1;
        if wave.paused == 0 {
            // Launches issued on the capture stream while the wave was held
            // run before anything recorded from here.
            st.hand_to_compute()?;
            begin(&capture.expect("a wave has a capture stream"))?;
            st.wave.as_mut().expect("checked above").recording = true;
        }
        Ok(())
    }

    /// End the recording segment, if any, and launch it into `compute`.
    ///
    /// Memory retired while it recorded stays in the graveyard; the caller
    /// takes it once this succeeds and drops it after releasing the lock.
    fn end_segment(st: &mut HubState, compute: &Arc<CudaStream>) -> Result<()> {
        let capture = st.capture.clone().expect("a wave has a capture stream");
        let wave = st.wave.as_mut().expect("called inside a wave");
        // A resume that failed left the wave eager, so there is no capture
        // to end — only the failure to report.
        if let Some(e) = wave.error.take() {
            crate::bail!("wave capture failed to resume: {e}");
        }
        if !wave.recording {
            return Ok(());
        }
        wave.recording = false;
        capture.context().bind_to_thread().w()?;
        // SAFETY: this wave began the capture on the hub's stream.
        let graph = unsafe { result::stream::end_capture(capture.cu_stream()) }.w()?;
        if graph.is_null() {
            return Err(crate::Error::wrap(GraphError::NoGraph));
        }
        let mut nodes = 0usize;
        // SAFETY: a null node array asks only for the count.
        let counted =
            unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut nodes).result() };
        if counted.is_err() || nodes == 0 {
            // SAFETY: the capture's own graph, destroyed once.
            let _ = unsafe { result::graph::destroy(graph) };
            return counted.w();
        }
        let segment = wave.segment;
        wave.segment += 1;
        if st.slots.len() <= segment {
            st.slots.resize_with(segment + 1, ExecSlot::default);
        }
        let on = super::ComputeStream::from_stream(compute.clone());
        let started = Instant::now();
        // SAFETY: the capture's own graph; `fold` takes ownership.
        let (exec, folded) = unsafe { st.slots[segment].fold(graph, &on, nodes)? };
        let fold_us = started.elapsed().as_micros() as u64;
        exec.launch(&on)?;
        st.stats.segments += 1;
        st.stats.nodes += nodes as u64;
        match folded {
            Folded::InPlace => {
                st.stats.updated += 1;
                st.stats.in_place_us += fold_us;
            }
            Folded::Reshaped => {
                st.stats.updated += 1;
                st.stats.reshaped += 1;
                st.stats.reshaped_us += fold_us;
            }
            Folded::Instantiated => {
                st.stats.instantiated += 1;
                st.stats.instantiated_us += fold_us;
            }
        }
        Ok(())
    }

    /// Suspend recording on this thread so an eager call can run in order.
    ///
    /// `None` when this thread has no wave open — the call runs as it always
    /// does. Otherwise the recording segment is launched into `compute` first,
    /// so everything recorded so far executes before the eager call, and
    /// recording resumes when the guard drops.
    pub(crate) fn pause(&self, compute: &Arc<CudaStream>) -> Result<Option<Paused<'_>>> {
        if !self.capturing_here() {
            return Ok(None);
        }
        let (paused, dead) = {
            let mut st = self.lock();
            let ended = Self::end_segment(&mut st, compute);
            // Everything released while the segment recorded can go once it
            // is launched: the frees queue on the compute stream behind it.
            let dead = match ended {
                Ok(()) => std::mem::take(&mut st.graveyard),
                Err(_) => Graveyard::new(),
            };
            let paused = ended.and_then(|()| st.hand_to_capture()).map(|()| {
                let wave = st.wave.as_mut().expect("capturing here");
                wave.paused += 1;
                wave.serial
            });
            (paused, dead)
        };
        drop(dead);
        Ok(Some(Paused {
            hub: self,
            serial: paused?,
            _thread_bound: PhantomData,
        }))
    }

    /// Hand everything this thread has issued to the driver: the recording
    /// segment, if any, is launched, then `compute` is queried so WDDM submits
    /// it. Recording resumes at the next launch.
    pub(crate) fn flush(&self, compute: &Arc<CudaStream>) -> Result<()> {
        let _paused = self.pause(compute)?;
        // SAFETY: a query on the compute stream, which is never recording. Its
        // only purpose is the WDDM submission it forces; `NOT_READY` is the
        // expected answer.
        match unsafe { sys::cuStreamQuery(compute.cu_stream()) } {
            sys::CUresult::CUDA_SUCCESS | sys::CUresult::CUDA_ERROR_NOT_READY => Ok(()),
            e => e.result().w(),
        }
    }

    /// Close this thread's wave: launch the last segment and stop recording.
    pub(crate) fn finish_wave(&self, compute: &Arc<CudaStream>) -> Result<()> {
        let (result, dead) = {
            let mut st = self.lock();
            match &st.wave {
                Some(w) if w.owner == std::thread::current().id() => {}
                _ => crate::bail!("no wave capture is open on this thread"),
            }
            let ended = Self::end_segment(&mut st, compute);
            let ended = ended.and_then(|()| st.hand_to_capture());
            st.wave = None;
            // Launched or abandoned, nothing recorded still waits to read these.
            let dead = std::mem::take(&mut st.graveyard);
            let fenced = Self::close_ring(&mut st, compute);
            self.active.store(false, Ordering::Release);
            (ended.and(fenced), dead)
        };
        drop(dead);
        result
    }

    /// Fence the wave's staging ring behind everything the wave issued.
    fn close_ring(st: &mut HubState, compute: &Arc<CudaStream>) -> Result<()> {
        // `waves` was counted when this wave opened, so it names this wave's
        // ring the same way `begin_wave` chose it.
        let ring = ((st.stats.waves + 1) % 2) as usize;
        st.rings[ring].close(compute)
    }

    /// Record an upload of `src` to the device address `dst` into this
    /// thread's recording segment, staged through the wave's ring. `false`
    /// when this thread is not recording or the ring is full — the caller
    /// then uploads eagerly.
    pub(crate) fn record_upload(&self, dst: u64, src: &[u8]) -> bool {
        if !self.owned_here() {
            return false;
        }
        let mut st = self.lock();
        let recording = matches!(
            &st.wave,
            Some(w) if w.recording && w.owner == std::thread::current().id()
        );
        if !recording {
            return false;
        }
        let capture = st.capture.clone().expect("a wave has a capture stream");
        let ring = ((st.stats.waves + 1) % 2) as usize;
        st.rings[ring].record(&capture, dst, src)
    }

    /// Release `item` — device memory a recorded launch may still read —
    /// once that launch has been issued. Returns it to the caller to drop at
    /// once when no wave is recording.
    ///
    /// Any thread's release waits while a segment records, not only the
    /// recording thread's: the last reference to memory a recorded launch
    /// reads may be dropped elsewhere, and a free issued then would queue on
    /// the compute stream ahead of the segment that still reads it.
    pub(crate) fn retire(&self, item: Box<dyn Send>) -> Option<Box<dyn Send>> {
        if !self.active.load(Ordering::Acquire) {
            return Some(item);
        }
        let mut st = self.lock();
        match &st.wave {
            Some(w) if w.recording => {
                st.graveyard.push(item);
                None
            }
            _ => Some(item),
        }
    }

    /// Close this thread's wave without launching what is recording — the
    /// forward it belonged to has already failed.
    pub(crate) fn abandon_wave(&self, compute: &Arc<CudaStream>) {
        let dead = {
            let mut st = self.lock();
            let Some(wave) = st.wave.take() else {
                return;
            };
            self.active.store(false, Ordering::Release);
            if wave.recording {
                if let Some(capture) = &st.capture {
                    // SAFETY: this wave began the capture on the hub's stream.
                    if let Ok(graph) = unsafe { result::stream::end_capture(capture.cu_stream()) } {
                        if !graph.is_null() {
                            // SAFETY: the capture's own graph, destroyed once.
                            let _ = unsafe { result::graph::destroy(graph) };
                        }
                    }
                }
            }
            // Segments launched before the failure may still read the ring.
            let _ = Self::close_ring(&mut st, compute);
            // Nothing recorded will run, so nothing still reads these.
            std::mem::take(&mut st.graveyard)
        };
        drop(dead);
    }

    pub(crate) fn stats(&self) -> CaptureStats {
        self.lock().stats
    }
}
