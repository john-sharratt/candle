//! The wave capture: one forward's launches recorded as a chain of graphs.
//!
//! A [`CaptureHub`] belongs to a device and is shared by every clone of its
//! handle. A device has one stream, and a wave records on it: while a segment
//! records, every launch the wave's thread issues through the device — candle's
//! own ops, the custom launchers, cuBLAS — is recorded instead of executed.
//! Every other thread issues under a lease
//! ([`crate::cuda_backend::StreamLease`]), which waits while a segment records,
//! and a segment about to record while another thread holds one runs as issued
//! instead — so the capture takes in the wave's launches and nothing else.
//!
//! **A segment ends wherever the host has to interact with the device.** The
//! driver refuses a synchronise, an allocation, a free or a host upload from
//! the capturing thread, and lets a readback through unordered against the
//! recorded work. So each of those, at the device method that issues it,
//! first [pauses](CaptureHub::pause) the wave: the open segment is ended,
//! folded into its executable and launched onto the stream, the eager call then
//! runs in order behind it, and recording resumes on the next launch site. The
//! MoE host protocol ends a segment the same way ([`CaptureHub::flush`]),
//! because its stager polls for the bucketize the segment holds. What the chain
//! changes is when launches reach the driver — batched per segment instead of
//! one at a time — never their order.
//!
//! **A segment the library does not keep runs as issued** (`start_segment`), on
//! the same stream: nothing orders it against the recorded segments around it
//! but issue order, which is all one stream needs.
//!
//! **Per-wave scalars and addresses are refreshed, not keyed.** Segment `i` of
//! every wave folds with `cuGraphExecUpdate` into the executable its ordinal
//! keeps for that node count (`slot`), which rewrites kernel parameters in
//! place when the recorded topology matches and is re-instantiated when it
//! does not. A wave whose shape, tier placement or launch scalars differ from
//! the last one therefore replays correctly with no key and no epoch to check.

use super::capture;
use super::cuts::CutCounter;
use super::exec::{audit, divergence, explaining, reshaped_at, Template};
use super::library::ExecLibrary;
use super::ordinals::{wave_kind, OrdinalHistory, WaveKind};
use super::record_gate;
use super::regime::{Key, LearnEnd, Mode, Phase, Regime, Transition};
use super::slot::{Admit, Folded};
use super::staging::StagingRing;
use super::thread_wave;
use super::timing::SegmentTimer;
use super::GraphError;
use crate::cuda_backend::{set_kernel_breadcrumb, CudaDevice, StreamLease, WrapErr};
use crate::Result;
use cudarc::driver::{result, sys, CudaEvent, CudaStream};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::panic::Location;
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
    /// Of those, the ones that were to be recorded but ran as issued because
    /// another thread held a lease on the stream.
    pub leased_eager: u64,
    /// Executables the library holds now — a gauge, not a count since start.
    pub execs_held: u64,
    /// Nodes across those executables, now.
    pub nodes_held: u64,
    /// Device bytes those executables are charged, now: driver memory outside
    /// the reservation, kept within the library's fixed budget.
    pub bytes_held: u64,
    /// Where the library is: learning, on trial, or locked (`regime`).
    pub phase: Phase,
    /// Topologies the trials banned as not faster, and proved, so far.
    pub banned: u64,
    pub proven: u64,
    /// Benches the trials have handed out, in all — topologies too rarely hit
    /// to judge, or never timed (`regime`).
    pub benched: u64,
    /// Times a lock has reopened to learn traffic it never saw.
    pub reopened: u64,
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
    /// The executables kept across waves, by segment topology.
    library: ExecLibrary,
    /// What the library keeps, learned and trialled (`regime`).
    regime: Regime,
    /// Device time of the segments on trial.
    timer: SegmentTimer,
    /// Where waves are cut into segments, by the site that asked (`cuts`).
    cuts: CutCounter,
    /// What each segment ordinal held the last time a wave of the same kind
    /// reached it — the predictions `start_segment` decides a segment's mode
    /// by (`ordinals`).
    ordinals: OrdinalHistory,
    /// Device memory released while recording, freed once the segment that
    /// may still read it has been launched — see [`CaptureHub::retire`].
    graveyard: Graveyard,
    /// The two staging rings recorded uploads go through, alternating by wave
    /// — see [`super::staging`].
    rings: Vec<StagingRing>,
    /// The device's stream, which the wave records on and launches into.
    compute: Option<Arc<CudaStream>>,
    stats: CaptureStats,
}

struct Wave {
    owner: ThreadId,
    /// Which wave this is, so a [`Paused`] guard acts only on its own.
    serial: u64,
    /// A capture is recording on the device's stream.
    recording: bool,
    /// Open [`Paused`] guards, plus one while `held`; recording resumes when
    /// the count reaches zero.
    paused: usize,
    /// The wave has not reached the launches it records yet.
    held: bool,
    /// The next segment's ordinal.
    segment: usize,
    /// Where it was opened, which its segments' predictions are kept by.
    kind: WaveKind,
    /// A resume that failed inside a guard's `drop`, reported at the next
    /// segment boundary.
    error: Option<String>,
    /// When the recording segment began, for [`CaptureStats::recording_us`].
    recording_since: Option<Instant>,
    /// The current segment runs unrecorded, its launches issued as they come
    /// (`CaptureHub::start_segment`), as a run of this predicted key.
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
            .finish_wave(self.device.stream_unleased())
    }
}

impl Drop for WaveCapture {
    fn drop(&mut self) {
        if !self.finished {
            self.device
                .capture_hub()
                .abandon_wave(self.device.stream_unleased());
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
        let resumed = CaptureHub::start_segment(&mut st);
        if let Err(e) = resumed {
            st.wave.as_mut().expect("checked above").error = Some(e.to_string());
        }
    }
}

/// Start recording on the device's stream, counted into the record gate for as
/// long as the segment records — or `false`, beginning nothing, while another
/// thread holds a lease on the stream. Every `true` is paired with one
/// [`end_capture`].
fn try_begin(stream: &Arc<CudaStream>) -> Result<bool> {
    // SAFETY: the device's created stream, not recording — a wave records on it
    // only between `try_begin` and `end_segment`, under the hub's lock.
    unsafe { capture::try_begin_on_stream(stream) }
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
                let judging = st.regime.begin_trial(st.library.held_topologies());
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
                starts,
                tallies,
            } => {
                st.stats.benched += (rare + untimed) as u64;
                let (execs, nodes, bytes) = st.library.held();
                // The few with the most hits, which are what a verdict turns
                // on: `hits/graphed/eager` each.
                let top: Vec<String> = tallies
                    .iter()
                    .rev()
                    .take(8)
                    .map(|(h, g, e)| format!("{h}/{g}/{e}"))
                    .collect();
                tracing::info!(
                    target: "candle_core::graph",
                    wave,
                    waves,
                    starts_candidate = starts.candidate,
                    starts_other = starts.other,
                    starts_unpredicted = starts.unpredicted,
                    hits_missed = starts.missed,
                    most_hit = top.join(" "),
                    judged,
                    proven,
                    graphed_over_eager =
                        proven_ratio.map(|r| format!("{r:.2}")).unwrap_or_default(),
                    benched_rare = rare,
                    benched_untimed = untimed,
                    banned_slower = slower,
                    next = next.name(),
                    banned_total,
                    proven_total,
                    execs_held = execs,
                    nodes_held = nodes,
                    held_mib = bytes >> 20,
                    budget_mib,
                    "graph library: trial ended — of {judged} judged, {proven} proven, {} benched \
                     ({rare} rarely hit, {untimed} never timed), {slower} banned as not faster, {}",
                    rare + untimed,
                    match next {
                        Phase::Learn => "learning again into the ground they freed",
                        _ => "nothing new worth keeping: locked",
                    }
                );
            }
            Transition::Reopened { waves, recurring } => {
                st.stats.reopened += 1;
                let (execs, nodes, bytes) = st.library.held();
                tracing::info!(
                    target: "candle_core::graph",
                    wave,
                    waves,
                    recurring,
                    banned_total,
                    proven_total,
                    execs_held = execs,
                    nodes_held = nodes,
                    held_mib = bytes >> 20,
                    budget_mib,
                    "graph library: lock reopened after {waves} wave(s) — {recurring} recurring \
                     shape(s) it does not hold, {} of {budget_mib} MiB held: learning again, \
                     keeping {proven_total} proven",
                    bytes >> 20
                );
            }
        }
    }
}

/// Nudge WDDM to submit what has been issued on `compute`.
///
/// Only between this thread's segments, or under a lease: a query on a stream
/// that is capturing is refused, and refusing it invalidates the capture.
fn query_submits(compute: &Arc<CudaStream>) -> Result<()> {
    // SAFETY: a query on the device's stream, made while nothing captures on
    // it. Its only purpose is the WDDM submission it forces; `NOT_READY` is the
    // expected answer.
    match unsafe { sys::cuStreamQuery(compute.cu_stream()) } {
        sys::CUresult::CUDA_SUCCESS | sys::CUresult::CUDA_ERROR_NOT_READY => Ok(()),
        e => e.result().w(),
    }
}

/// End the capture [`try_begin`] started; the segment stays counted in the
/// record gate until `counted` drops, once its graph is launched.
fn end_capture(
    stream: &Arc<CudaStream>,
    counted: &record_gate::StreamRecording,
) -> Result<sys::CUgraph> {
    // SAFETY: this wave began the capture on the device's stream, on this
    // thread.
    unsafe { capture::end_on_stream(stream, counted) }
}

impl CaptureHub {
    fn lock(&self) -> MutexGuard<'_, HubState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
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

    /// This thread's open wave's serial — `None` when it has no wave open,
    /// whether the wave is recording, running a segment eagerly, held or
    /// paused. A host protocol that ends segments at fixed points in the wave
    /// counts by this rather than by [`Self::recording_segment`], so an eager
    /// segment is cut where a recorded one would be
    /// (`expert_lre::flush_schedule`).
    pub(crate) fn wave_here(&self) -> Option<u64> {
        if !self.owned_here() {
            return None;
        }
        self.lock()
            .wave
            .as_ref()
            .filter(|w| w.owner == std::thread::current().id())
            .map(|w| w.serial)
    }

    /// Whether this thread's wave is in a segment that runs eagerly: its
    /// launches reach the device as issued, so an upload may go straight onto
    /// the stream behind them instead of ending the segment.
    pub(crate) fn eager_segment_here(&self) -> bool {
        self.owned_here()
            && self
                .lock()
                .wave
                .as_ref()
                .is_some_and(|w| w.eager && w.paused == 0 && w.owner == std::thread::current().id())
    }

    /// Nudge WDDM to submit what this thread has queued, ending nothing: a
    /// host protocol waiting on an eagerly issued launch (the MoE stager
    /// polling a bucketize) would otherwise wait for WDDM to submit it on its
    /// own. Nothing to do while this thread records: what it issues waits for
    /// the segment's launch, and what it issued before was submitted when the
    /// segment began — a query now would end its capture. Leased on any thread
    /// that is not recording.
    #[track_caller]
    pub(crate) fn submit(&self, compute: &Arc<CudaStream>) -> Result<()> {
        if self.recording_segment().is_some() {
            return Ok(());
        }
        let _lease = StreamLease::take(compute);
        query_submits(compute)
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

    /// Open a wave capture on this thread, of the kind `opened_at` names
    /// (`ordinals`).
    pub(crate) fn begin_wave(
        &self,
        compute: &Arc<CudaStream>,
        opened_at: &'static Location<'static>,
    ) -> Result<()> {
        let mut st = self.lock();
        if st.wave.is_some() {
            crate::bail!("a wave capture is already open on this device");
        }
        // Segments on trial that have finished on the device are credited —
        // first, because it can fail: a pair the driver will not time leaves
        // this call with nothing else changed, not with the wave counted and a
        // staging ring reopened for a wave that never opened.
        {
            let HubState { timer, regime, .. } = &mut *st;
            timer.drain(|key, mode, took| regime.sample(key, mode, took))?;
        }
        st.compute = Some(compute.clone());
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
            // A trial that has run its course drops what it did not prove.
            let HubState {
                regime, library, ..
            } = &mut *st;
            for key in regime.wave_begins(now) {
                library.drop_topology(key);
            }
        }
        act_on_transitions(&mut st, now);
        if let Some(report) = st.cuts.wave() {
            let waves = report.waves.max(1) as f64;
            let sites: Vec<String> = report
                .sites
                .iter()
                .map(|((file, line), n)| format!("{file}:{line} {:.1}/wave", *n as f64 / waves))
                .collect();
            tracing::debug!(
                target: "candle_core::graph::cuts",
                waves = report.waves,
                cuts = report.cuts,
                per_wave = format!("{:.1}", report.cuts as f64 / waves),
                "segment cuts by site: {}",
                sites.join("; ")
            );
        }
        st.wave = Some(Wave {
            owner: std::thread::current().id(),
            serial: st.stats.waves,
            recording: false,
            paused: 1,
            held: true,
            segment: 0,
            kind: wave_kind(opened_at),
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
            Self::start_segment(&mut st)?;
        }
        Ok(())
    }

    /// End the recording segment, if any, and launch it into `compute`.
    ///
    /// Memory retired while it recorded stays in the graveyard; the caller
    /// takes it once this succeeds and drops it after releasing the lock.
    fn end_segment(st: &mut HubState, compute: &Arc<CudaStream>) -> Result<()> {
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
            let timing = wave.timing.take();
            if let Some(start) = timing {
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
        // Counted in the gate until this function returns — past the launch
        // below — on every path, the failed ones included.
        let counted = record_gate::StreamRecording::ending();
        // The capture cannot be ended from an unbound thread; it is lost, and
        // `counted` takes its count out of the gate.
        compute.context().bind_to_thread().w()?;
        let graph = end_capture(compute, &counted)?;
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
        let (key, shape) = (audited.key, audited.shape);
        let holds_shape = st.library.holds_shape(key, shape);
        // The explanations name kernels through the driver; only for the log.
        let explain = explaining();
        let kind = wave.kind;
        let o = st.ordinals.at(kind, segment);
        let prev_topology = &o.topology[segment];
        let departs = if explain && first_sight && !prev_topology.is_empty() {
            divergence(&audited.topology, prev_topology)
        } else {
            None
        };
        if explain {
            o.topology[segment].clone_from(&audited.topology);
        }
        // **A shape the library does not hold, of the topology this ordinal
        // held last wave.** The same launches at other dimensions are a key of
        // their own, so the launch whose grid or variant moved names what keeps
        // a segment from recurring.
        st.regime.recorded(o.key[segment].map(|(k, _)| k), key);
        let reshaped = match o.key[segment] {
            Some((k, s)) if explain && k == key && s != shape && !holds_shape => {
                reshaped_at(&audited.dims, &o.dims[segment])
            }
            _ => None,
        };
        if explain {
            o.dims[segment].clone_from(&audited.dims);
        }
        o.recorded(segment, (key, shape));
        let topologies = st.library.len() + usize::from(first_sight);
        // Only the learn keeps anything new, and never a banned topology;
        // after it, a shape the library does not hold runs once (`slot`).
        let admit = if st.regime.admits() && !st.regime.banned(key) {
            Admit::Keep
        } else {
            Admit::Never
        };
        let proven = st.regime.proven(key);
        let (transient, folded, refusal) = match st
            .library
            .slot(key)
            .fold(template, audited, &on, nodes, admit, proven)
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
        if let Some(d) = reshaped {
            tracing::debug!(
                target: "candle_core::graph",
                wave = serial,
                segment,
                nodes,
                reshaped = d.as_str(),
                "segment of a new shape, against this ordinal's last"
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
            if st.regime.timed(key) {
                st.timer.end(compute, start, key, Mode::Graph)?;
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
                // Locked, a recorded segment the library holds nothing for is
                // traffic the lock never learned: enough of it, with room in
                // the budget, reopens the learn.
                st.regime.missed(key);
                let (_, _, held) = st.library.held();
                if st.regime.reopen_due(serial) && held * 2 < st.library.budget() {
                    st.regime.reopen(serial);
                    act_on_transitions(st, serial);
                }
            }
        }
        Ok(())
    }

    /// Begin the wave's next segment, recorded or as issued, by the key this
    /// ordinal held the last time a wave of this kind recorded it
    /// (`ordinals`):
    ///
    /// - banned: as issued — its graph was not worth keeping;
    /// - its shape not held, in the lock: as issued — recording it would buy
    ///   one launch of an executable nothing keeps (in the trial it records, so
    ///   its prediction follows the wave: [`Regime::runs_unkept`]);
    /// - on trial: graphed or eager by turns ([`Regime::mode_for`]), its start
    ///   recorded for the device time the trial compares;
    /// - otherwise recorded. An ordinal never recorded is recorded while the
    ///   library learns or trials, and runs as issued in the lock.
    ///
    /// A segment to be recorded while another thread holds a lease on the
    /// stream runs as issued instead: the capture would take in that thread's
    /// launches, and waiting for the lease could wait on a lock this thread
    /// holds. Such a run is credited to nothing ([`CaptureStats::leased_eager`]).
    ///
    /// An ordinal that has run as issued on its prediction alone
    /// [`RECHECK_EVERY`] times in a row is recorded once, so its prediction
    /// follows what the wave now puts there.
    ///
    /// The key is a prediction: a segment's real key is known only once it has
    /// been recorded. A miss either way is harmless — a recorded segment whose
    /// topology the library does not hold runs once ([`Folded::Transient`]),
    /// and an unrecorded one is simply eager.
    fn start_segment(st: &mut HubState) -> Result<()> {
        let wave = st.wave.as_ref().expect("called inside a wave");
        let ordinal = wave.segment;
        let o = st.ordinals.at(wave.kind, ordinal);
        // The regime judges topologies; the shape is the ordinal's own record.
        let recorded = o.predicted(ordinal);
        let predicted = recorded.map(|(topology, _)| topology);
        let stable = o.stable(ordinal);
        // Eager on the prediction alone (`Regime::runs_unkept`), as opposed to
        // the eager half of a trial. **Held means the predicted shape is**: a
        // held topology at a shape the library does not hold would be recorded
        // only to be instantiated for its one launch, which costs more than
        // issuing it — measured in the lock as 10–90 ms of instantiation every
        // two seconds.
        let held = recorded.is_some_and(|(t, s)| st.library.holds_shape(t, s));
        st.regime.start(predicted);
        let unkept = st.regime.runs_unkept(predicted, held);
        let runs = &mut o.eager_runs[ordinal];
        let mode = if unkept {
            *runs += 1;
            if *runs > RECHECK_EVERY {
                *runs = 0;
                Mode::Graph
            } else {
                Mode::Eager
            }
        } else {
            *runs = 0;
            match predicted {
                Some(k) => st.regime.mode_for(k),
                None => Mode::Graph,
            }
        };
        // An eager run is credited to its prediction, since it is never
        // recorded; only a prediction that has held is trusted with that
        // (`Ordinals::stable`). A graphed run is credited to what it recorded.
        let trusted = mode == Mode::Graph || stable;
        let timing = match predicted {
            Some(k) if trusted && st.regime.timed(k) => {
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
        let compute = st.compute.clone().expect("a wave has a compute stream");
        if !try_begin(&compute)? {
            if let Some(start) = timing {
                st.timer.discard(start);
            }
            st.stats.eager_segments += 1;
            st.stats.leased_eager += 1;
            let wave = st.wave.as_mut().expect("checked above");
            wave.eager = true;
            wave.eager_key = None;
            return Ok(());
        }
        let wave = st.wave.as_mut().expect("checked above");
        wave.timing = timing;
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
    #[track_caller]
    pub(crate) fn pause(&self, compute: &Arc<CudaStream>) -> Result<Option<Paused<'_>>> {
        if !self.capturing_here() {
            return Ok(None);
        }
        let caller = Location::caller();
        let (paused, dead) = {
            let mut st = self.lock();
            if st.wave.as_ref().is_some_and(|w| w.recording || w.eager) {
                st.cuts.cut(caller);
            }
            // Launched and submitted: the segment may hold MoE invocations whose
            // summary words the expert threads wait on, and on WDDM a launched
            // batch waits for a query before it reaches the GPU. Nothing is
            // capturing once the segment has ended, so the query is legal here.
            let ended = Self::end_segment(&mut st, compute).and_then(|()| query_submits(compute));
            // Everything released while the segment recorded can go once it
            // is launched: the frees queue on the stream behind it.
            let dead = match ended {
                Ok(()) => std::mem::take(&mut st.graveyard),
                Err(_) => Graveyard::new(),
            };
            let paused = ended.map(|()| {
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
    /// segment, if any, is launched, then the stream is queried so WDDM submits
    /// it. The next segment begins at once. A wave held, or paused by an outer
    /// guard, is not recording: there is nothing to end, and resuming is the
    /// guard's business.
    ///
    /// Everything that touches the stream outside the segment — the memory the
    /// segment retired, freed behind it, and the query that forces WDDM to
    /// submit — runs between the launch and the resume: a query or a free made
    /// on the stream while it captures is refused, and refusing it invalidates
    /// the capture. A thread with no wave open queries under a lease.
    #[track_caller]
    pub(crate) fn flush(&self, compute: &Arc<CudaStream>) -> Result<()> {
        if !self.capturing_here() {
            let _lease = StreamLease::take(compute);
            return query_submits(compute);
        }
        let caller = Location::caller();
        let (ended, dead) = {
            let mut st = self.lock();
            if !st.wave.as_ref().is_some_and(|w| w.recording || w.eager) {
                // Held, or paused by an outer guard: nothing records on this
                // thread and there is nothing to end; resuming is the guard's.
                drop(st);
                return query_submits(compute);
            }
            st.cuts.cut(caller);
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
        let stream = st.compute.clone().expect("a wave has a compute stream");
        let ring = ((st.stats.waves + 1) % 2) as usize;
        st.rings[ring].record(&stream, dst, src)
    }

    /// Release `item` — device memory a recorded launch may still read —
    /// once that launch has been issued. Returns it to the caller to drop at
    /// once, under a lease, when no segment is recording.
    ///
    /// Any thread's release waits while a segment records, not only the
    /// recording thread's: the last reference to memory a recorded launch
    /// reads may be dropped elsewhere, and a free issued on the stream then
    /// would be taken into the capture. A segment run as issued holds nothing
    /// back: its launches are on the stream ahead of the free.
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

    /// Replace the executable library with an empty one of `budget` bytes,
    /// learning from scratch, and forget every ordinal's last key.
    #[cfg(test)]
    pub(crate) fn reset_library(&self, budget: u64) {
        let mut st = self.lock();
        st.library = ExecLibrary::with_budget(budget);
        st.regime = Regime::default();
        st.timer = SegmentTimer::default();
        st.ordinals.clear();
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
            let _ = Self::end_segment(&mut st, compute).and_then(|()| query_submits(compute));
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
