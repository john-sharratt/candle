//! Mutations sent to a substrate rather than applied under its lock by the
//! sender (`docs/substrate_mutation_queue.md` §2).
//!
//! A sender pushes a mutation and returns at once: the push is a lock-free
//! channel send, so a thread that must never stall — the scheduler — never
//! waits for the substrate lock to make one. The queue is drained, in order,
//! by whoever next takes the write lock ([`SubstrateLock`](super::SubstrateLock)
//! drains it on every write acquisition, before its own mutation), and by an
//! applier thread started with the first message, which takes the write lock
//! whenever the queue is non-empty — so a message is applied within one long
//! hold of being sent, and a synchronous write never overtakes one sent before
//! it.
//!
//! A message runs on whatever thread drains it. It must not free device
//! memory there; one that takes device chunks out of the substrate hands them
//! back to the thread that owns their release.

use std::sync::{Arc, OnceLock, Weak};
use std::thread;
use std::time::Duration;

use flume::{Receiver, RecvTimeoutError, Sender};

use super::{Substrate, SubstrateLock};

/// One mutation, applied to the substrate under its write lock.
pub type Mutation = Box<dyn FnOnce(&mut Substrate) + Send>;

/// How long the applier sleeps without a wake before it looks again — a wake
/// lost to a full signal slot is found at the next look.
const LOOK_EVERY: Duration = Duration::from_millis(250);

/// See the module documentation.
pub struct MutationQueue {
    tx: Sender<Mutation>,
    rx: Receiver<Mutation>,
    /// One pending wake for the applier; a send with one already pending is
    /// dropped, since the applier drains everything it finds.
    wake_tx: Sender<()>,
    wake_rx: Receiver<()>,
    applier: OnceLock<()>,
}

impl Default for MutationQueue {
    fn default() -> Self {
        let (tx, rx) = flume::unbounded();
        let (wake_tx, wake_rx) = flume::bounded(1);
        MutationQueue {
            tx,
            rx,
            wake_tx,
            wake_rx,
            applier: OnceLock::new(),
        }
    }
}

impl MutationQueue {
    /// Send `mutation` to `lock`'s substrate, starting its applier with the
    /// first message.
    pub fn send(&self, lock: &Arc<SubstrateLock>, mutation: Mutation) {
        // A send to a queue its own lock holds never fails: the receiver lives
        // as long as the sender does.
        let _ = self.tx.send(mutation);
        self.applier
            .get_or_init(|| spawn_applier(Arc::downgrade(lock), self.wake_rx.clone()));
        let _ = self.wake_tx.try_send(());
    }

    /// Apply every queued mutation to `substrate`, oldest first. Called with
    /// the write lock held.
    pub fn drain(&self, substrate: &mut Substrate) {
        while let Ok(mutation) = self.rx.try_recv() {
            mutation(substrate);
        }
    }

    /// Whether a mutation is waiting.
    pub fn is_pending(&self) -> bool {
        !self.rx.is_empty()
    }
}

/// The applier: wakes on a send (or every [`LOOK_EVERY`]), and takes the write
/// lock — which drains the queue — whenever something is waiting. Ends when the
/// substrate does.
fn spawn_applier(lock: Weak<SubstrateLock>, wake: Receiver<()>) {
    let spawned = thread::Builder::new()
        .name("substrate-applier".into())
        .spawn(move || loop {
            match wake.recv_timeout(LOOK_EVERY) {
                Ok(()) | Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }
            let Some(lock) = lock.upgrade() else {
                return;
            };
            if lock.has_pending() {
                drop(lock.write().unwrap_or_else(|e| e.into_inner()));
            }
        });
    if let Err(e) = spawned {
        tracing::error!(
            "the substrate applier could not start: {e} — sent mutations wait for the next writer"
        );
    }
}
