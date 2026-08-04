//! Off-thread substrate writer (`docs/kv_tier_migration.md`).
//!
//! Every redo-log append that would otherwise block a caller on the persistence
//! lock (which a segment compaction can hold across its whole relocation I/O) is
//! enqueued here and drained by a dedicated writer thread. Two producers feed it:
//!
//! - **Phase 1 — stream-decl mirrors** (`WriteJob::StreamDecl`), from the
//!   inference thread's seal path. The stream id is a pure function of the decl
//!   (`decl.stream_id()`), so the enqueuer applies the decl in-memory
//!   synchronously and hands only the durable append here — the inference thread
//!   never touches the persistence lock.
//! - **Phase 2 — warm→cold KV writes** (`WriteJob::KvCold`), from the persistence
//!   thread. The GPU gather stays there; only the redo-log append + the
//!   `install_cold` bookkeeping move here. Crucially `install_cold` (which sets
//!   `slot.cold` and drops `hot`) runs on the writer AFTER the bytes are appended,
//!   so `slot.cold` only becomes true once the data is durable — `purge_warm`
//!   never drops the last copy of an un-written turn (the RAM-unload invariant).
//!
//! A DUAL-CAP bounded queue (pending events AND pending bytes) applies BLOCKING
//! backpressure: `enqueue` blocks when either cap trips, so the writer can never
//! fall so far behind that a crash loses an unbounded tail. The single writer
//! drains FIFO. Crash safety is unchanged: the reconstruct walker is
//! order-independent and per-turn fault-isolated (see `reconstruct_from_log`), so
//! an async / out-of-order / lost-on-crash write only ever yields an ABSENT turn,
//! never corruption; the bounded queue caps that window.

use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;

use crossbeam::channel::{self, Receiver, Sender};

use super::record::RecordType;
use super::resume::{self, TurnChunkGrid};
use super::{Result, SubstratePersistence};
use crate::persistence::streams::StreamId;
use crate::substrate::{ResidenceIndex, Substrate};

/// Max pending write EVENTS before an enqueue blocks. Sized to absorb a full
/// segment-compaction's worth of seal decls (the writer stalls on the persistence
/// lock for the compaction's whole relocation) without back-pressuring producers
/// in the common case.
const MAX_PENDING_EVENTS: usize = 16_384;
/// Max pending write BYTES before an enqueue blocks — the second cap, so a burst
/// of large KV-cold payloads can't balloon host RAM even under the event cap.
const MAX_PENDING_BYTES: u64 = 256 * 1024 * 1024;
/// Upper bound on jobs drained before a group fsync, so a sustained burst still
/// yields the persistence lock to the persistence thread's other work regularly.
const MAX_BATCH: usize = 512;

/// A queued redo-log write.
pub(crate) enum WriteJob {
    /// Phase 1: a stream-decl (turn / section) metadata mirror. `stream_id` is the
    /// pure `decl.stream_id().0`, `payload` is `decl.encode()`; already applied
    /// in-memory by the enqueuer.
    StreamDecl { stream_id: u64, payload: Vec<u8> },
    /// Phase 1: a turn/section `Tokens` record. Disk-only — the in-memory token
    /// buffer is already set at record time, so this is purely the durable copy
    /// for reload. Enqueued from the seal path so it never blocks on the
    /// persistence lock (which a compaction holds across its whole relocation).
    Tokens {
        stream_id: StreamId,
        token_ids: Vec<u32>,
    },
    /// Phase 1: a turn's wide-Q provenance signature (`WideQSig` record). The
    /// in-RAM blob is mirrored synchronously by the enqueuer (the same-session
    /// provenance scan needs it now); this appends only the durable copy.
    WideQSigs {
        stream_id: StreamId,
        payload: Vec<u8>,
    },
    /// Phase 2: a warm→cold KV migration. The GPU gather already produced `grid`;
    /// the writer appends its chunks, folds their locations into the substrate
    /// index, marks the stream durable-through, then `install_cold`s (drops hot)
    /// — all AFTER the append, so `slot.cold` is only ever set on durable data.
    KvCold {
        residence: ResidenceIndex,
        stream_id: StreamId,
        grid: TurnChunkGrid,
    },
    /// Drain everything queued, fsync, ack, and stop the thread.
    Shutdown(Sender<()>),
}

impl WriteJob {
    fn byte_size(&self) -> u64 {
        match self {
            WriteJob::StreamDecl { payload, .. } => payload.len() as u64,
            WriteJob::Tokens { token_ids, .. } => (token_ids.len() * 4) as u64,
            WriteJob::WideQSigs { payload, .. } => payload.len() as u64,
            WriteJob::KvCold { grid, .. } => grid.bytes() as u64,
            WriteJob::Shutdown(_) => 0,
        }
    }

    /// Whether this job counts against the BYTE cap. Only the large KV-cold
    /// payloads do; the small metadata records (stream-decl / tokens / sigs) are
    /// gated by the EVENT cap alone, so a KV-cold backlog (e.g. the writer stalled
    /// behind a compaction holding the persistence lock) can never block the
    /// scheduler's latency-critical seal metadata behind the byte cap.
    fn is_bulk(&self) -> bool {
        matches!(self, WriteJob::KvCold { .. })
    }
}

/// Dual-cap backpressure accounting shared between enqueuers and the writer.
struct Backpressure {
    /// `(pending_events, pending_bytes)`; `below_cap` wakes blocked enqueuers when
    /// the writer drains back under both caps.
    state: Mutex<(usize, u64)>,
    below_cap: Condvar,
}

impl Backpressure {
    /// Reserve queue capacity for one job, blocking under backpressure. `bulk`
    /// (KV-cold only) additionally honors the byte cap; metadata jobs are gated by
    /// the event cap alone so they never block behind a KV-cold byte backlog.
    fn acquire(&self, bytes: u64, bulk: bool) {
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        // Block until there's room. The `g.0 > 0` guard guarantees a single
        // in-flight job always makes progress even if it alone exceeds the byte cap
        // (an oversized payload must still drain, not deadlock).
        while g.0 >= MAX_PENDING_EVENTS || (bulk && g.0 > 0 && g.1 + bytes > MAX_PENDING_BYTES) {
            g = self.below_cap.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        g.0 += 1;
        g.1 += bytes;
    }

    fn release(&self, bytes: u64) {
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        g.0 = g.0.saturating_sub(1);
        g.1 = g.1.saturating_sub(bytes);
        self.below_cap.notify_all();
    }
}

/// The off-thread substrate writer: a bounded queue + a drainer thread.
pub struct SubstrateWriter {
    tx: Sender<(WriteJob, u64)>,
    backpressure: Arc<Backpressure>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl SubstrateWriter {
    /// Spawn the writer thread over the shared substrate + persistence handles.
    /// It holds the raw Arcs (never a `Conversation`) so there is no Arc cycle —
    /// the last `Conversation` drop releases the writer, whose `Drop` drains + joins.
    pub fn spawn(
        substrate: Arc<RwLock<Substrate>>,
        persistence: Arc<Mutex<SubstratePersistence>>,
    ) -> Self {
        let (tx, rx) = channel::unbounded::<(WriteJob, u64)>();
        let backpressure = Arc::new(Backpressure {
            state: Mutex::new((0, 0)),
            below_cap: Condvar::new(),
        });
        let bp = backpressure.clone();
        let handle = std::thread::Builder::new()
            .name("substrate-writer".into())
            .spawn(move || writer_loop(rx, substrate, persistence, bp))
            .expect("spawn substrate-writer thread");
        Self {
            tx,
            backpressure,
            handle: Mutex::new(Some(handle)),
        }
    }

    /// Enqueue a durable write. Blocks (backpressure) if the queue is at either
    /// cap — the enqueuer keeps the in-memory substrate consistent while the disk
    /// catches up.
    pub(crate) fn enqueue(&self, job: WriteJob) {
        let bytes = job.byte_size();
        self.backpressure.acquire(bytes, job.is_bulk());
        // `send` only fails once the writer thread is gone (post-shutdown). Release
        // the reservation and drop the write: in-memory is still consistent, and on
        // next boot the turn is simply absent — which reconstruct tolerates.
        if self.tx.send((job, bytes)).is_err() {
            self.backpressure.release(bytes);
        }
    }

    /// Drain all pending writes, fsync, and stop the thread. Idempotent.
    pub fn shutdown(&self) {
        let handle = self.handle.lock().unwrap_or_else(|e| e.into_inner()).take();
        let Some(handle) = handle else { return };
        let (ack_tx, ack_rx) = channel::bounded::<()>(1);
        self.enqueue(WriteJob::Shutdown(ack_tx));
        let _ = ack_rx.recv(); // wait for drain + fsync before joining
        let _ = handle.join();
    }
}

impl Drop for SubstrateWriter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Drainer loop: process a job, coalesce a burst, group-fsync once, repeat.
fn writer_loop(
    rx: Receiver<(WriteJob, u64)>,
    substrate: Arc<RwLock<Substrate>>,
    persistence: Arc<Mutex<SubstratePersistence>>,
    bp: Arc<Backpressure>,
) {
    while let Ok((first, first_bytes)) = rx.recv() {
        let mut ack = process_one(first, &substrate, &persistence);
        bp.release(first_bytes);
        let mut n = 1;
        while ack.is_none() && n < MAX_BATCH {
            match rx.try_recv() {
                Ok((job, bytes)) => {
                    ack = process_one(job, &substrate, &persistence);
                    bp.release(bytes);
                    n += 1;
                }
                Err(_) => break,
            }
        }
        // Group-commit (fsync) the burst.
        commit(&persistence);
        if let Some(ack) = ack {
            let _ = ack.send(());
            return;
        }
    }
    // All senders dropped without an explicit Shutdown — flush the tail.
    commit(&persistence);
}

/// Apply one job. Returns the shutdown ack when the job is `Shutdown` (the caller
/// commits + acks + exits), else `None`.
fn process_one(
    job: WriteJob,
    substrate: &Arc<RwLock<Substrate>>,
    persistence: &Arc<Mutex<SubstratePersistence>>,
) -> Option<Sender<()>> {
    match job {
        WriteJob::Shutdown(ack) => return Some(ack),
        WriteJob::StreamDecl { stream_id, payload } => {
            let mut p = persistence.lock().unwrap_or_else(|e| e.into_inner());
            if let Err(e) = p.append_record(RecordType::StreamDecl, 0, stream_id, 0, 0, 0, &payload)
            {
                tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    "stream-decl append failed: {e}"
                );
            }
        }
        WriteJob::Tokens {
            stream_id,
            token_ids,
        } => {
            // Append under the persistence lock, then register the record's
            // location in the substrate index under the substrate lock — taken
            // NON-nested, persistence released first (same order as the KV-cold
            // path). Without the `apply_tokens_loc`, maintenance/compaction can't
            // see the tokens and reclaim them (see `persist_tokens_only`).
            let loc = {
                let mut p = persistence.lock().unwrap_or_else(|e| e.into_inner());
                resume::persist_tokens_only(&mut p, stream_id, &token_ids)
            };
            match loc {
                Ok(loc) => substrate
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .apply_tokens_loc(stream_id, loc),
                Err(e) => tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    stream_id = stream_id.0,
                    "tokens append failed: {e}"
                ),
            }
        }
        WriteJob::WideQSigs { stream_id, payload } => {
            let mut p = persistence.lock().unwrap_or_else(|e| e.into_inner());
            if let Err(e) = p.append_wide_q_sigs(stream_id, &payload) {
                tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    stream_id = stream_id.0,
                    "wide-Q sigs append failed: {e}"
                );
            }
        }
        WriteJob::KvCold {
            residence,
            stream_id,
            grid,
        } => {
            if let Err(e) = write_kv_cold(substrate, persistence, residence, stream_id, &grid) {
                tracing::warn!(
                    target: "candle_conversation::persistence::writer",
                    residence = residence.0,
                    "warm→cold write failed: {e}"
                );
            }
        }
    }
    None
}

/// Append a turn's gathered KV chunks, fold their locations into the substrate
/// index, mark the stream durable-through, then `install_cold` (drop hot). The
/// locks are taken NON-nested and in the same order as the original synchronous
/// path (`persist_turn_chunks_capture` + `commit_stream_through` + `install_cold`):
/// persistence for the append/durability mark, then substrate for the index +
/// hot-drop — so `slot.cold` is set only after the bytes are appended.
fn write_kv_cold(
    substrate: &Arc<RwLock<Substrate>>,
    persistence: &Arc<Mutex<SubstratePersistence>>,
    residence: ResidenceIndex,
    stream_id: StreamId,
    grid: &TurnChunkGrid,
) -> Result<()> {
    let (stored, locs) = {
        let mut p = persistence.lock().unwrap_or_else(|e| e.into_inner());
        let out = resume::persist_turn_chunks_capture(&mut p, stream_id, grid)?;
        if !out.0.is_empty() {
            // Mark durable through the last-written chunk BEFORE the hot-drop, so a
            // crash after the drop still knows the turn's chunks landed.
            let through = (out.0.iter().map(|s| s.chunks.len()).sum::<usize>().max(1) - 1) as u64;
            p.commit_stream(stream_id, through)?;
        }
        out
    };
    if stored.is_empty() {
        return Ok(());
    }
    let mut view = substrate.write().unwrap_or_else(|e| e.into_inner());
    for (flat, loc) in locs {
        view.apply_chunk_loc(stream_id, flat, loc);
    }
    view.install_cold(residence, stored);
    Ok(())
}

fn commit(persistence: &Arc<Mutex<SubstratePersistence>>) {
    if let Ok(mut p) = persistence.lock() {
        if let Err(e) = p.commit() {
            tracing::error!(
                target: "candle_conversation::persistence::writer",
                "substrate writer commit failed: {e}"
            );
        }
    }
}
