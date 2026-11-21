//! The cloneable trigger that wakes the persistence thread, and the readers the
//! scheduler polls for its drain state.
//!
//! A trigger is fire-and-forget, so one wake pending is as good as several,
//! while a flush blocks until the pass that moves what it asked for has run.
//! The readers return the gauges the loop last stamped, so a scheduler reading
//! them every wave never takes the substrate's lock. A no-op trigger stands in
//! for the thread in tests that build a scheduler without one.

use super::*;

/// Clone-able trigger for the persistence thread. Held by anything that
/// wants to wake the loop early (e.g. the scheduler signalling "turn just
/// sealed"). Cheap to clone — wraps two `crossbeam::channel` senders: a
/// fire-and-forget wake and a blocking "drain hot→warm now" flush.
#[derive(Clone)]
pub struct PersistenceTrigger {
    pub(super) tx: Sender<()>,
    /// Carries a one-shot flush into the loop: the next pass runs the hot→warm
    /// migration and answers once the request is met (see [`FlushRequest`]).
    pub(super) flush_tx: Sender<FlushRequest>,
    /// Live drain gauges, stamped by the persistence loop each pass. The
    /// scheduler polls them to throttle admission off a leading signal.
    pub(super) backlog: BacklogGauge,
    /// Turns asked to be brought from cold into warm ([`Self::warm_turns`]).
    pub(super) warm_asks: WarmAsks,
}

/// What a fall in the backlog calls: the scheduler's wake.
pub type BacklogWaker = Box<dyn Fn() + Send + Sync>;

impl PersistenceTrigger {
    /// Wake the persistence thread now. No-op when the trigger queue is
    /// already full (one wake is as good as several).
    pub fn fire(&self) {
        let _ = self.tx.try_send(());
    }

    /// Ask the persistence thread to bring `turns` from cold into warm,
    /// without placing them hot (`docs/kv_quantize_in_the_gap.md` §5). Returns
    /// at once: the next pass reads each listed turn that holds only a cold
    /// copy from the log into the warm tier, within this substrate's share of
    /// the warm budget, and drops what does not fit. A turn already warm or
    /// hot is skipped, so sending a whole list again costs a lookup a turn.
    /// Nothing reports back; a turn's tier is read as usual
    /// (`Substrate::turn_tier_state`).
    pub fn warm_turns(&self, turns: Vec<TurnKey>) {
        if turns.is_empty() {
            return;
        }
        self.warm_asks.ask(turns);
        self.fire();
    }

    /// Block until the persistence thread completes a pass that drains the
    /// pending hot→warm migration, giving substrate offload **complete
    /// priority**. After this returns `true`, just-sealed turns hold a warm
    /// (RAM) copy and so become eligible for hot-tier eviction — the VRAM can
    /// actually move to the substrate. Used by the scheduler under VRAM
    /// pressure, before it evicts hot KV.
    ///
    /// Returns `false` if the request couldn't be queued (a flush is already
    /// in flight) or the pass didn't ack within `timeout` (the timeout is a
    /// safety cap against a wedged thread, not the expected path).
    pub fn flush_blocking(&self, timeout: Duration) -> bool {
        let (ack_tx, ack_rx) = channel::bounded::<()>(1);
        if self.flush_tx.try_send(FlushRequest::drain(ack_tx)).is_err() {
            return false;
        }
        ack_rx.recv_timeout(timeout).is_ok()
    }

    /// [`Self::flush_blocking`] answered as soon as `bytes` are installed warm,
    /// not once the whole backlog is: what a caller that needs that much
    /// evictable — relief, short by a known amount — waits for. Same `false`
    /// cases.
    pub fn flush_at_least(&self, bytes: u64, timeout: Duration) -> bool {
        let (ack_tx, ack_rx) = channel::bounded::<()>(1);
        if self
            .flush_tx
            .try_send(FlushRequest::at_least(ack_tx, bytes))
            .is_err()
        {
            return false;
        }
        ack_rx.recv_timeout(timeout).is_ok()
    }

    /// The persistence thread's most recent hot→warm drain backlog, in bytes.
    /// A *leading* pressure signal (drain deficit) the scheduler reads every
    /// wave to size the ingest admission window — it rises as seals outrun the
    /// drain and falls as the drain catches up, before the lagging VRAM-pressure
    /// trip ever fires.
    pub fn pending_warm_bytes(&self) -> u64 {
        self.backlog.warm.load(Ordering::Relaxed)
    }

    /// Have `wake` called whenever [`Self::pending_warm_bytes`] falls — the
    /// scheduler's way to block, rather than poll, while that backlog holds its
    /// admission shut. Set once per persistence thread; a later call is ignored.
    pub fn on_backlog_fall(&self, wake: BacklogWaker) {
        let _ = self.backlog.on_fall.set(wake);
    }

    /// Hot KV the drain most recently SKIPPED because it is an in-flight
    /// decode's pinned working set. The VRAM forecast counts these bytes as
    /// reversibly evictable, but nothing can reclaim them while the pin holds —
    /// and they peak exactly when the drain has stalled, so admission subtracts
    /// them from its ceiling rather than planning against them. See
    /// [`Substrate::warm_backlog_split`].
    pub fn pinned_undrainable_bytes(&self) -> u64 {
        self.backlog.pinned.load(Ordering::Relaxed)
    }

    /// Residences currently holding a warm (host RAM) copy, as of the last
    /// persistence pass — the population the warm purge draws from.
    pub fn warm_resident_count(&self) -> u64 {
        self.backlog.warm_residents.load(Ordering::Relaxed)
    }

    /// Bytes held by warm copies, as of the last persistence pass.
    pub fn warm_resident_bytes(&self) -> u64 {
        self.backlog.warm_resident_bytes.load(Ordering::Relaxed)
    }

    /// Host bytes of the per-turn payloads held whole outside the warm budget,
    /// as of the last persistence pass.
    pub fn held_payload_bytes(&self) -> u64 {
        self.backlog.held_payload_bytes.load(Ordering::Relaxed)
    }

    /// Test-only no-op trigger. Holds senders whose receivers are dropped
    /// immediately, so [`Self::fire`]/[`Self::flush_blocking`] silently fail.
    /// For tests that construct a `Scheduler` without spawning a real
    /// persistence thread.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn noop() -> Self {
        let (tx, _rx) = channel::bounded(1);
        let (flush_tx, _flush_rx) = channel::bounded(1);
        Self {
            tx,
            flush_tx,
            backlog: BacklogGauge::default(),
            warm_asks: WarmAsks::default(),
        }
    }
}

/// Handle to a running persistence thread. Triggers wake the loop early;
/// [`Self::shutdown`] (or `Drop`) signals the loop, drains a final pass,
/// and joins the thread.
///
/// The `JoinHandle` lives behind `Mutex<Option<…>>` purely because
/// `JoinHandle::join` consumes itself — that lets both [`Self::shutdown`]
/// (via `&self`) and `Drop` reach in and take it. The choice is
/// **internal** to this type; owners hold a plain `PersistenceThread`.
pub struct PersistenceThread {
    pub(super) handle: Mutex<Option<JoinHandle<()>>>,
    pub(super) trigger_tx: Sender<()>,
    pub(super) flush_tx: Sender<FlushRequest>,
    pub(super) shutdown_tx: Sender<()>,
    /// Shared with the loop (which stamps them) and every [`PersistenceTrigger`]
    /// handed out (which reads them) — the live drain gauges.
    pub(super) backlog: BacklogGauge,
    /// Shared with the loop (which takes them) and every trigger (which asks).
    pub(super) warm_asks: WarmAsks,
}

impl PersistenceThread {
    /// Spawn the thread. Takes the workspace [`Conversation`] handle and
    /// the per-layer GPU backings (used by the hot→warm migration and
    /// the warm→cold gather).
    ///
    /// The persistence thread owns a **dedicated CUDA copy stream**
    /// (forked off the device's default CUDA context) and a **pinned
    /// host scratch buffer** that grows on demand across passes. The
    /// hot→warm migration uses both: the gather kernel and the DtoH
    /// run on the copy stream so they overlap with main-stream
    /// compute, and the DtoH target is write-combined pinned memory
    /// (no driver-side pageable bounce).
    ///
    /// `warm` is this substrate's share of the engine's one warm budget: the
    /// warm purge keeps this substrate to its share of it ([`WarmShare`]).
    pub fn spawn(
        conversation: Conversation,
        backings: Arc<Vec<ChunkedKvBacking>>,
        device: Device,
        compression_policy: Option<CompressionPolicy>,
        warm: WarmShare,
    ) -> Self {
        let (trigger_tx, trigger_rx) = channel::bounded::<()>(1);
        let (flush_tx, flush_rx) = channel::bounded::<FlushRequest>(1);
        let (shutdown_tx, shutdown_rx) = channel::bounded::<()>(1);
        let backlog = BacklogGauge::default();
        let loop_backlog = backlog.clone();
        let warm_asks = WarmAsks::default();
        let loop_warm_asks = warm_asks.clone();

        // All persist-thread CUDA work runs on the device's one stream —
        // selection, convert, gather, DtoH — taken under a lease per group
        // (`run_pass`). Sharing it with decode serialises persistence behind
        // in-flight inference work but eliminates the cross-stream
        // coordination bugs that arose when persist read shared mutable state
        // (per-head table, head_gids) on a side stream without proper fences.
        assert!(
            matches!(device, Device::Cuda(_)),
            "substrate-persistence: requires a CUDA device"
        );

        let handle = std::thread::Builder::new()
            .name("substrate-persistence".into())
            .spawn(move || {
                #[cfg(feature = "memory-census")]
                set_thread(Scope::Persistence);
                run_loop(
                    conversation,
                    backings,
                    device,
                    compression_policy,
                    trigger_rx,
                    flush_rx,
                    shutdown_rx,
                    loop_backlog,
                    warm,
                    loop_warm_asks,
                );
            })
            .expect("failed to spawn substrate-persistence thread");

        Self {
            handle: Mutex::new(Some(handle)),
            trigger_tx,
            flush_tx,
            shutdown_tx,
            backlog,
            warm_asks,
        }
    }

    /// Wake the thread now. No-op if a trigger is already pending or the
    /// thread is mid-pass — the next pass will see the latest state.
    pub fn trigger(&self) {
        let _ = self.trigger_tx.try_send(());
    }

    /// A clone-able trigger handle. Hand these out to anything that
    /// wants to fire the thread without holding the (single-owner)
    /// `PersistenceThread` itself.
    pub fn trigger_handle(&self) -> PersistenceTrigger {
        PersistenceTrigger {
            tx: self.trigger_tx.clone(),
            flush_tx: self.flush_tx.clone(),
            backlog: self.backlog.clone(),
            warm_asks: self.warm_asks.clone(),
        }
    }

    /// Stop the thread: send the shutdown signal, wait for the loop to
    /// drain any pending work, and join. Idempotent — second call is a
    /// no-op (the join handle has already been taken).
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.try_send(());
        if let Some(h) = self.handle.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = h.join();
        }
    }
}

impl Drop for PersistenceThread {
    fn drop(&mut self) {
        // Idempotent — if `shutdown` already joined, `handle.lock()` returns
        // a `None` and this is a no-op.
        self.shutdown();
    }
}
