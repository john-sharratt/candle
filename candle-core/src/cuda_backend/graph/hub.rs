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

use super::capture;
use super::exec::{audit, divergence, NodeKey, Template};
use super::library::ExecLibrary;
use super::record_gate;
use super::regime::{Key, LearnEnd, Mode, Phase, Regime, Transition};
use super::slot::{Admit, Folded};
use super::staging::StagingRing;
use super::thread_wave;
use super::timing::SegmentTimer;
use super::GraphError;
use crate::cuda_backend::{set_kernel_breadcrumb, CudaDevice, WrapErr};
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
    /// Host time, in µs, segments spent recording — from the capture's start to
    /// its end, everything the forward thread did to issue the segment's
    /// launches. Against the device time of the segment before it, this says
    /// whether the host keeps ahead of the GPU at a segment boundary.
    pub recording_us: u64,
    /// Graph nodes launched — kernels, memsets, device copies and event
    /// records alike.
    pub nodes: u64,
    /// Segments instantiated for one launch and not kept — the library was not
    /// admitting their shape, and no executable held it.
    pub transient: u64,
    /// Segments run unrecorded: their predicted shape banned, not held once the
    /// learn is over, or on the eager half of a trial.
    pub eager_segments: u64,
    /// Executables the library holds now — a gauge, not a count since start.
    pub execs_held: u64,
    /// Nodes across those executables, now.
    pub nodes_held: u64,
    /// Device bytes those executables are charged, now: driver memory outside
    /// the reservation, kept within the library's fixed budget.
    pub bytes_held: u64,
    /// Where the library is: learning, on trial, or locked (`regime`).
    pub phase: Phase,
    /// Shapes the trials banned, and proved, so far.
    pub banned: u64,
    pub proven: u64,
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

/// Eager runs in a row after which an ordinal records once, to see what it
/// carries now.
///
/// An eager segment is never recorded, so its ordinal's prediction is never
/// refreshed: an ordinal that once carried a shape the library does not hold
/// would run eagerly for good, even once the wave's composition puts a held,
/// proven shape there — and a locked library would drift towards running
/// everything eagerly. Recorded once in this many, the ordinal learns its
/// shape again: a held one folds in place and it is back on graphs; anything
/// else runs once, unkept, at the cost of one instantiation.
pub(super) const RECHECK_EVERY: u32 = 32;

#[derive(Default)]
struct HubState {
    wave: Option<Wave>,
    /// The non-blocking stream launches are recorded on, made on first use.
    capture: Option<Arc<CudaStream>>,
    /// The executables kept across waves, by segment topology.
    library: ExecLibrary,
    /// What the library keeps, learned and trialled (`regime`).
    regime: Regime,
    /// Device time of the segments on trial.
    timer: SegmentTimer,
    /// The topology each segment ordinal held in the last wave that reached
    /// it — what a new topology is compared against when it is logged.
    last_topology: Vec<Vec<NodeKey>>,
    /// The `(topology, shape)` key each ordinal held in the last wave that
    /// recorded it — the prediction `start_segment` decides a segment's mode by.
    last_key: Vec<Option<Key>>,
    /// Segments each ordinal has run eagerly in a row on its prediction alone
    /// — see [`RECHECK_EVERY`].
    eager_runs: Vec<u32>,
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
    /// When the recording segment began, for [`CaptureStats::recording_us`].
    recording_since: Option<Instant>,
    /// The current segment runs unrecorded (`CaptureHub::start_segment`), as
    /// a run of this predicted key.
    eager: bool,
    eager_key: Option<Key>,
    /// The current segment is on trial: the event its start was recorded at.
    timing: Option<CudaEvent>,
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
        // Finished or abandoned, the wave no longer records on this thread.
        thread_wave::leave(&self.device);
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
        if wave.paused > 0 || wave.recording || wave.eager {
            return;
        }
        let resumed = st
            .hand_to_compute()
            .and_then(|()| CaptureHub::start_segment(&mut st));
        if let Err(e) = resumed {
            st.wave.as_mut().expect("checked above").error = Some(e.to_string());
        }
    }
}

/// Start recording on the hub's stream, counted into the record gate for as
/// long as the segment records. Every successful `begin` is paired with one
/// [`end_capture`].
fn begin(capture: &Arc<CudaStream>) -> Result<()> {
    // SAFETY: the hub's own created stream, not recording — a wave records on it
    // only between `begin` and `end_segment`, under the hub's lock.
    unsafe { capture::begin(capture) }
}

/// Act on every library phase that has ended (`regime`), at wave `wave`: a
/// learn that hands over to a trial begins it on what the library holds —
/// which is why every path that may end a learn calls this at once — and each
/// is logged with what the library then holds.
fn act_on_transitions(st: &mut HubState, wave: u64) {
    for t in st.regime.take_transitions() {
        let (banned_total, proven_total) = st.regime.counts();
        let budget_mib = st.library.budget() >> 20;
        match t {
            Transition::Learned {
                waves,
                admitted,
                ended,
                next,
            } => {
                let judging = st.regime.begin_trial(st.library.held_keys());
                let (execs, nodes, bytes) = st.library.held();
                tracing::info!(
                    target: "candle_core::graph",
                    wave,
                    waves,
                    admitted,
                    judging,
                    ended = match ended {
                        LearnEnd::Full => "budget full",
                        LearnEnd::Quiet => "no new shape",
                    },
                    next = st.regime.phase().name(),
                    execs_held = execs,
                    nodes_held = nodes,
                    held_mib = bytes >> 20,
                    budget_mib,
                    "graph library: learn ended — {admitted} shape(s) admitted in {waves} wave(s), {}",
                    match (next, judging) {
                        (Phase::Trial, n) if n > 0 =>
                            format!("trialling the {n} shape(s) held unproven, graphed against eager"),
                        _ => "nothing to trial: locked".to_string(),
                    }
                );
            }
            Transition::Trialled {
                waves,
                judged,
                proven,
                proven_ratio,
                rare,
                untimed,
                slower,
                next,
            } => {
                let (execs, nodes, bytes) = st.library.held();
                tracing::info!(
                    target: "candle_core::graph",
                    wave,
                    waves,
                    judged,
                    proven,
                    graphed_over_eager =
                        proven_ratio.map(|r| format!("{r:.2}")).unwrap_or_default(),
                    banned_rare = rare,
                    banned_untimed = untimed,
                    banned_slower = slower,
                    next = next.name(),
                    banned_total,
                    proven_total,
                    execs_held = execs,
                    nodes_held = nodes,
                    held_mib = bytes >> 20,
                    budget_mib,
                    "graph library: trial ended — of {judged} judged, {proven} proven, {} banned \
                     ({rare} rarely hit, {untimed} never timed, {slower} not faster), {}",
                    rare + untimed + slower,
                    match next {
                        Phase::Learn => "learning again into the ground the bans freed",
                        _ => "nothing new worth keeping: locked",
                    }
                );
            }
        }
    }
}

/// Nudge WDDM to submit what has been issued on `compute`.
fn query_submits(compute: &Arc<CudaStream>) -> Result<()> {
    // SAFETY: a query on the compute stream, which is never recording, made
    // while this thread captures nothing. Its only purpose is the WDDM
    // submission it forces; `NOT_READY` is the expected answer.
    match unsafe { sys::cuStreamQuery(compute.cu_stream()) } {
        sys::CUresult::CUDA_SUCCESS | sys::CUresult::CUDA_ERROR_NOT_READY => Ok(()),
        e => e.result().w(),
    }
}

/// End the capture [`begin`] started, and count the segment out of the record
/// gate whether or not the driver hands back a graph.
fn end_capture(capture: &Arc<CudaStream>) -> Result<sys::CUgraph> {
    // SAFETY: this wave began the capture on the hub's stream, on this thread.
    unsafe { capture::end(capture) }
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

    /// The segment this thread is recording into right now, as `(wave serial,
    /// segment ordinal)` — `None` when its launches reach the device as issued.
    pub(crate) fn recording_segment(&self) -> Option<(u64, usize)> {
        if !self.owned_here() {
            return None;
        }
        self.lock()
            .wave
            .as_ref()
            .filter(|w| w.recording && w.owner == std::thread::current().id())
            .map(|w| (w.serial, w.segment))
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

    /// Whether another thread is recording a segment now — one whose launches
    /// are not issued yet, so an eager write from this thread to memory they
    /// read lands before them, not after.
    pub(crate) fn recording_elsewhere(&self) -> bool {
        self.active.load(Ordering::Acquire)
            && matches!(
                &self.lock().wave,
                Some(w) if w.recording && w.owner != std::thread::current().id()
            )
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
        let now = st.stats.waves;
        if !st.library.budget_known() {
            // The same figure the reservation left outside its span for this.
            if let Ok(budget) = super::budget::graph_budget_bytes() {
                st.library.set_budget(budget);
            }
        }
        {
            // Segments on trial that have finished on the device are credited,
            // and a trial that has run its course drops what it banned.
            let HubState {
                timer,
                regime,
                library,
                ..
            } = &mut *st;
            timer.drain(|key, mode, took| regime.sample(key, mode, took));
            for key in regime.wave_begins(now) {
                library.ban(key);
            }
        }
        act_on_transitions(&mut st, now);
        st.wave = Some(Wave {
            owner: std::thread::current().id(),
            serial: st.stats.waves,
            recording: false,
            paused: 1,
            held: true,
            segment: 0,
            error: None,
            recording_since: None,
            eager: false,
            eager_key: None,
            timing: None,
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
            Self::start_segment(&mut st)?;
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
        if wave.eager {
            // A segment left unrecorded: its launches went to the device as
            // issued, so it ends with nothing to launch — only, on trial, the
            // end of its timing.
            wave.eager = false;
            wave.segment += 1;
            let key = wave.eager_key.take();
            if let Some(start) = wave.timing.take() {
                match key {
                    Some(k) => st.timer.end(compute, start, k, Mode::Eager)?,
                    None => st.timer.discard(start),
                }
            }
            return Ok(());
        }
        if !wave.recording {
            return Ok(());
        }
        wave.recording = false;
        if let Some(since) = wave.recording_since.take() {
            st.stats.recording_us += since.elapsed().as_micros() as u64;
        }
        let wave = st.wave.as_mut().expect("called inside a wave");
        if let Err(e) = capture.context().bind_to_thread().w() {
            // The capture cannot be ended from an unbound thread; it is lost,
            // and so is its count in the gate.
            record_gate::recording_ends();
            return Err(e);
        }
        let graph = end_capture(&capture)?;
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
        let serial = wave.serial;
        let on = super::ComputeStream::from_stream(compute.clone());
        let started = Instant::now();
        // The capture's own graph, destroyed on every path from here.
        let template = Template(graph);
        let audited = audit(template.0, nodes)?;
        // **A topology seen for the first time, against what this ordinal held
        // last wave.** Instantiating a new topology is the cost that remains
        // once segments are found by topology, so where it departs from its
        // predecessor at the same place in the wave names what made it new —
        // a kernel chosen by width, a launch count that follows the wave.
        let first_sight = !st.library.holds(audited.key);
        let departs = match st.last_topology.get(segment) {
            Some(prev) if first_sight => divergence(&audited.topology, prev),
            _ => None,
        };
        if st.last_topology.len() <= segment {
            st.last_topology.resize_with(segment + 1, Vec::new);
        }
        st.last_topology[segment].clone_from(&audited.topology);
        if st.last_key.len() <= segment {
            st.last_key.resize(segment + 1, None);
        }
        let (key, shape) = (audited.key, audited.shape);
        st.last_key[segment] = Some((key, shape));
        let topologies = st.library.len() + usize::from(first_sight);
        // Only the learn keeps anything new, and never a banned shape.
        let admit = if st.regime.admits() && !st.regime.banned((key, shape)) {
            Admit::Keep
        } else {
            Admit::Never
        };
        // A proven executable keeps the shape its trial proved: a one-off is
        // never folded over it.
        let regime = &st.regime;
        let protected = |s| regime.proven((key, s));
        let (transient, folded, refusal) = match st
            .library
            .slot(key)
            .fold(template, audited, &on, nodes, admit, protected)
        {
            Ok(done) => done,
            Err(e) => {
                st.library.drop_if_empty(key);
                // Every executable is device memory taken from the driver,
                // outside the reservation, so a full card refuses the
                // instantiation here — named, with what the library holds.
                let (execs_held, nodes_held, bytes_held) = st.library.held();
                tracing::warn!(
                    target: "candle_core::graph",
                    wave = serial,
                    segment,
                    nodes,
                    topologies,
                    execs_held,
                    nodes_held,
                    bytes_held,
                    "segment could not be folded or instantiated: {e}"
                );
                return Err(e);
            }
        };
        let fold_us = started.elapsed().as_micros() as u64;
        // Both are routine — a wave's shape changes every few waves — so they
        // are debug: at info they were four lines in five of a daemon's log.
        if let Some(d) = departs {
            tracing::debug!(
                target: "candle_core::graph",
                wave = serial,
                segment,
                nodes,
                topologies,
                instantiate_ms = fold_us / 1000,
                diverges = d.as_str(),
                "segment of a new topology, against this ordinal's last"
            );
        }
        if let Some(r) = refusal {
            tracing::debug!(
                target: "candle_core::graph",
                wave = serial,
                segment,
                nodes,
                kept = ?r.kept_nodes,
                instantiate_ms = fold_us / 1000,
                diverges = r.divergence.as_deref().unwrap_or("nowhere: same kernels in order"),
                "segment instantiated beside its topology's executables"
            );
        }
        // The recorded launches' own breadcrumbs were left at record time; this
        // one marks where they were actually issued.
        set_kernel_breadcrumb("graph segment launch", file!(), line!());
        match &transient {
            // Launched once and dropped: destroying an executable in flight is
            // legal, and the driver frees it when its launch completes.
            Some(exec) => exec.launch(&on)?,
            None => st
                .library
                .front(key)
                .expect("a kept fold leaves its executable in front")
                .launch(&on)?,
        }
        drop(transient);
        if let Some(start) = st.wave.as_mut().and_then(|w| w.timing.take()) {
            if st.regime.timed((key, shape)) {
                st.timer.end(compute, start, (key, shape), Mode::Graph)?;
            } else {
                st.timer.discard(start);
            }
        }
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
                // Launched above, so a refusal may destroy it at once.
                if st.library.admitted(key) {
                    st.regime.admitted(serial);
                } else {
                    st.regime.full(serial);
                    act_on_transitions(st, serial);
                }
            }
            Folded::Transient => {
                st.stats.transient += 1;
                st.stats.instantiated_us += fold_us;
                st.library.drop_if_empty(key);
            }
        }
        Ok(())
    }

    /// Begin the wave's next segment, recorded or as issued, by the key this
    /// ordinal held the last time it was recorded:
    ///
    /// - banned: as issued — its graph was not worth keeping;
    /// - not held, once the learn is over: as issued — recording it would buy
    ///   one launch of an executable nothing keeps;
    /// - on trial: graphed or eager by turns ([`Regime::mode_for`]), its start
    ///   recorded for the device time the trial compares;
    /// - otherwise recorded. An ordinal never recorded is recorded while the
    ///   library learns, and runs as issued after.
    ///
    /// An ordinal that has run as issued on its prediction alone
    /// [`RECHECK_EVERY`] times in a row is recorded once, so its prediction
    /// follows what the wave now puts there.
    ///
    /// The key is a prediction: a segment's real key is known only once it has
    /// been recorded. A miss either way is harmless — a recorded segment whose
    /// shape the library does not hold runs once ([`Folded::Transient`]), and
    /// an unrecorded one is simply eager.
    fn start_segment(st: &mut HubState) -> Result<()> {
        let wave = st.wave.as_ref().expect("called inside a wave");
        let ordinal = wave.segment;
        let predicted = st.last_key.get(ordinal).copied().flatten();
        // Eager on the prediction alone — banned, unheld, or never recorded —
        // as opposed to the eager half of a trial.
        let unkept = match predicted {
            None => !st.regime.admits(),
            Some(k) => st.regime.banned(k) || (!st.regime.admits() && !st.library.holds_shape(k)),
        };
        if st.eager_runs.len() <= ordinal {
            st.eager_runs.resize(ordinal + 1, 0);
        }
        let mode = if unkept {
            st.eager_runs[ordinal] += 1;
            if st.eager_runs[ordinal] > RECHECK_EVERY {
                st.eager_runs[ordinal] = 0;
                Mode::Graph
            } else {
                Mode::Eager
            }
        } else {
            st.eager_runs[ordinal] = 0;
            match predicted {
                Some(k) => st.regime.mode_for(k),
                None => Mode::Graph,
            }
        };
        let timing = match predicted {
            Some(k) if st.regime.timed(k) => {
                let compute = st.compute.clone().expect("a wave has a compute stream");
                Some(st.timer.start(&compute)?)
            }
            _ => None,
        };
        if mode == Mode::Eager {
            st.stats.eager_segments += 1;
            let wave = st.wave.as_mut().expect("checked above");
            wave.eager = true;
            wave.eager_key = predicted;
            wave.timing = timing;
            return Ok(());
        }
        st.wave.as_mut().expect("checked above").timing = timing;
        let capture = st.capture.clone().expect("a wave has a capture stream");
        begin(&capture)?;
        let wave = st.wave.as_mut().expect("checked above");
        wave.recording = true;
        wave.recording_since = Some(Instant::now());
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
            // Launched and submitted: the segment may hold MoE invocations whose
            // summary words the expert threads wait on, and on WDDM a launched
            // batch waits for a query before it reaches the GPU. Nothing is
            // capturing between the launch and the hand-off, so the query is
            // legal here.
            let ended = Self::end_segment(&mut st, compute).and_then(|()| query_submits(compute));
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
    ///
    /// **No hand-offs.** A flush runs no eager section: between launching the
    /// segment and beginning the next one this thread issues nothing, and no
    /// other thread is ever given the capture stream. So the two event pairs a
    /// [`Self::pause`] records — which exist to order a stray launch made
    /// inside an eager section — order nothing here, and they cost the device a
    /// four-hop chain to resolve before the next segment may start, at every
    /// boundary. The segment is launched and recording resumes directly. A wave
    /// held, or paused by an outer guard, is not recording: there is nothing to
    /// end, and resuming is the guard's business.
    ///
    /// Everything that touches the legacy stream — the memory the segment
    /// retired, freed behind it, and the query that forces WDDM to submit —
    /// runs between the launch and the resume: a legacy-stream call made while
    /// this thread captures is refused, and refusing it invalidates the
    /// capture.
    pub(crate) fn flush(&self, compute: &Arc<CudaStream>) -> Result<()> {
        if !self.capturing_here() {
            return query_submits(compute);
        }
        let (ended, dead) = {
            let mut st = self.lock();
            if !st.wave.as_ref().is_some_and(|w| w.recording || w.eager) {
                // Held, or paused by an outer guard: nothing records on this
                // thread and there is nothing to end; resuming is the guard's.
                drop(st);
                return query_submits(compute);
            }
            match Self::end_segment(&mut st, compute) {
                Ok(()) => (Ok(()), std::mem::take(&mut st.graveyard)),
                Err(e) => (Err(e), Graveyard::new()),
            }
        };
        drop(dead);
        ended?;
        query_submits(compute)?;
        let mut st = self.lock();
        let resumed = Self::start_segment(&mut st);
        if let Err(e) = &resumed {
            st.wave
                .as_mut()
                .expect("this thread's wave stays open across its flush")
                .error = Some(e.to_string());
        }
        resumed
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

    /// How many releases are waiting for the recording segment to launch.
    #[cfg(test)]
    pub(crate) fn retired_len(&self) -> usize {
        self.lock().graveyard.len()
    }

    /// Replace the executable library with an empty one of `budget` bytes,
    /// learning from scratch, and forget every ordinal's last key.
    #[cfg(test)]
    pub(crate) fn reset_library(&self, budget: u64) {
        let mut st = self.lock();
        st.library = ExecLibrary::with_budget(budget);
        st.regime = Regime::default();
        st.timer = SegmentTimer::default();
        st.last_key.clear();
        st.eager_runs.clear();
    }

    /// Close the wave of a forward that failed, **launching what is recording**.
    ///
    /// The forward has failed, but the work it recorded before the failure was
    /// issued, and parties outside the wave already wait on some of it: an MoE
    /// invocation tells the stager and the expert pipeline its summary ticket
    /// before its bucketize is recorded, and a segment may hold up to
    /// `MAX_UNFLUSHED_INVOCATIONS` of them (`expert_lre::flush_schedule`).
    /// Discarding the segment would leave those summary words unwritten — the
    /// stager waiting on them for good, the pipeline thread aborting, the
    /// retried forward's cold experts never published. Launching it is what an
    /// eager forward would have done up to the failure; everything after it is
    /// stream-ordered behind it. A capture that can no longer be ended has lost
    /// its launches either way, and the failure that broke it is reported on
    /// its own path.
    pub(crate) fn abandon_wave(&self, compute: &Arc<CudaStream>) {
        let dead = {
            let mut st = self.lock();
            if st.wave.is_none() {
                return;
            }
            let _ = Self::end_segment(&mut st, compute)
                .and_then(|()| st.hand_to_capture())
                .and_then(|()| query_submits(compute));
            st.wave = None;
            self.active.store(false, Ordering::Release);
            // Segments launched before and by this close may still read the ring.
            let _ = Self::close_ring(&mut st, compute);
            // Everything recorded is launched; frees now queue behind it.
            std::mem::take(&mut st.graveyard)
        };
        drop(dead);
    }

    pub(crate) fn stats(&self) -> CaptureStats {
        let st = self.lock();
        let (execs_held, nodes_held, bytes_held) = st.library.held();
        let (banned, proven) = st.regime.counts();
        CaptureStats {
            execs_held: execs_held as u64,
            nodes_held: nodes_held as u64,
            bytes_held,
            phase: st.regime.phase(),
            banned: banned as u64,
            proven: proven as u64,
            ..st.stats
        }
    }
}
