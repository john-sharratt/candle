//! This member's own events going out (§6.3, §6.4).
//!
//! The log is the source of truth: the outbox holds `oseq`s only, and an event
//! is read from the store when it is sent. Each event is unsent, queued on a
//! socket, in flight on it, or received once that socket's gateway receipts
//! it; a socket that drops sends everything not receipted back to unsent.
//!
//! **Which socket.** A request served off a socket flushes before its
//! response ends ([`Replicator::flush`]): every event still unsent goes on the
//! request's own socket, ahead of the response's last bytes, and an event
//! that already left on another socket is waited for until that gateway has
//! receipted it. Every other event is dealt round robin across the admitted
//! sockets, one whole event at a time, once it has waited [`GRACE`] for a
//! request to take it.

use std::pin::pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::time::{sleep, timeout_at};

use super::replicator::{Phase, Replicator};

/// How long a fresh event waits for the request that wrote it to take it
/// onto its own socket before the round robin sends it.
pub const GRACE: Duration = Duration::from_millis(100);

/// The most events read and sent in one pass.
const BATCH: usize = 1024;

/// How long a request's flush waits for events that left on other sockets to
/// be receipted.
pub const FLUSH_WAIT: Duration = Duration::from_secs(2);

impl Replicator {
    /// The tap: own record `oseq` is written and may be sent.
    pub(super) fn tapped(&self, oseq: u64) {
        {
            let mut s = self.state();
            s.outbox.written(oseq);
            s.fresh.insert(oseq, Instant::now());
            s.ack_dirty = true;
        }
        self.send_wake.notify_one();
    }

    /// The round-robin sender: on every tap, and every [`GRACE`].
    pub(super) async fn sender(self: Arc<Self>) {
        loop {
            tokio::select! {
                _ = self.send_wake.notified() => {}
                _ = sleep(GRACE) => {}
            }
            while self.send_unsent(None).await == BATCH {}
        }
    }

    /// Send unsent events: with `on = Some((c, through))`, every one through
    /// `through` on socket `c`; otherwise every one past its grace, round
    /// robin. Returns how many were taken.
    async fn send_unsent(self: &Arc<Self>, on: Option<(usize, u64)>) -> usize {
        let picks = {
            let mut s = self.state();
            if matches!(s.phase, Phase::Rejoining | Phase::Reloading) && on.is_none() {
                return 0;
            }
            let sockets = s.admitted();
            if sockets.is_empty() {
                return 0;
            }
            let now = Instant::now();
            let due: Vec<u64> = s
                .outbox
                .unsent()
                .filter(|o| match on {
                    Some((_, through)) => *o <= through,
                    None => s.fresh.get(o).is_none_or(|at| now >= *at + GRACE),
                })
                .take(BATCH)
                .collect();
            let mut picks = Vec::with_capacity(due.len());
            for oseq in due {
                let (c, handle) = match on {
                    Some((c, _)) => match s.sockets[c].clone() {
                        Some(h) => (c, h),
                        None => return 0,
                    },
                    None => {
                        s.rr = (s.rr + 1) % sockets.len();
                        sockets[s.rr].clone()
                    }
                };
                s.outbox.queue(oseq, c as u32);
                // Not fresh: owed again after a socket dropped or the process
                // restarted.
                if s.fresh.remove(&oseq).is_none() {
                    s.stats.resent += 1;
                }
                picks.push((oseq, c, handle));
            }
            picks
        };
        let taken = picks.len();
        if taken == 0 {
            return 0;
        }
        let store = self.store.clone();
        let node = self.node;
        let Ok(read) = tokio::task::spawn_blocking(move || {
            picks
                .into_iter()
                .map(|(oseq, c, h)| (oseq, c, h, store.read(node, oseq)))
                .collect::<Vec<_>>()
        })
        .await
        else {
            return 0;
        };
        for (oseq, c, handle, bytes) in read {
            let Some(bytes) = bytes else {
                // Maintenance dropped it as superseded: what superseded it is
                // sent instead, and a catch-up records this one as gone.
                self.state().outbox.forget(oseq);
                continue;
            };
            match handle.event(node, oseq, &bytes, None, false) {
                Ok(written) => {
                    let me = self.clone();
                    tokio::spawn(async move {
                        let written = written.await;
                        let mut s = me.state();
                        match written {
                            Ok(frame) => s.outbox.sent(oseq, c as u32, frame),
                            Err(_) => s.outbox.unqueue(oseq),
                        }
                    });
                }
                Err(_) => self.state().outbox.unqueue(oseq),
            }
        }
        taken
    }

    /// Before the response to a request served off socket `c` ends: every
    /// record the store had queued is written, every own event still unsent
    /// goes on `c`, and every one that left on another socket is receipted
    /// there — or [`FLUSH_WAIT`] passes (§6.4).
    pub async fn flush(self: &Arc<Self>, c: usize) {
        let store = self.store.clone();
        let Ok(through) = tokio::task::spawn_blocking(move || {
            store.wait_written(store.ticket());
            store.own_through()
        })
        .await
        else {
            return;
        };
        while self.send_unsent(Some((c, through))).await == BATCH {}
        let deadline = tokio::time::Instant::now() + FLUSH_WAIT;
        loop {
            // Registered before the look, so a receipt between the two still
            // wakes it.
            let receipted = self.receipts.notified();
            let mut receipted = pin!(receipted);
            receipted.as_mut().enable();
            if !self.state().outbox.owed_elsewhere(through, c as u32) {
                return;
            }
            if timeout_at(deadline, receipted).await.is_err() {
                return;
            }
        }
    }
}
