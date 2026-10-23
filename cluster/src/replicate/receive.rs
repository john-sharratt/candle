//! What arrives for this member (§6.5, §7.2): records, live or fetched, applied
//! in batches as they come — no reorder buffer, arrival order does not matter
//! — and the answers to the catch-ups it asked for. Records and catch-up
//! replies share one queue, so a `DONE` is handled only once every record
//! ahead of it is applied.

use std::mem::take;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc::UnboundedReceiver;

use super::catchup::Purpose;
use super::replicator::{Arrival, Phase, Replicator};
use super::store::Applied;
use crate::fragment::Whole;
use crate::frame::{Done, Fetch, Frame, Gone, Have, Headers};

/// The most records, and bytes, applied in one batch.
const BATCH_RECORDS: usize = 512;
const BATCH_BYTES: usize = 32 << 20;

impl Replicator {
    pub(super) async fn receiver(self: Arc<Self>, mut rx: UnboundedReceiver<Arrival>) {
        while let Some(first) = rx.recv().await {
            let mut queue = vec![first];
            let mut bytes = 0;
            while queue.len() < BATCH_RECORDS && bytes < BATCH_BYTES {
                match rx.try_recv() {
                    Ok(a) => {
                        if let Arrival::Event { whole, .. } = &a {
                            bytes += whole.bytes.len();
                        }
                        queue.push(a);
                    }
                    Err(_) => break,
                }
            }
            let mut records = Vec::new();
            // Each record's share of the arrivals budget, given back once the
            // batch holding it is applied — so the room it frees is room in
            // memory, and a waiting `event` reads its socket again.
            let mut rooms = Vec::new();
            for arrival in queue {
                match arrival {
                    Arrival::Event {
                        c,
                        whole,
                        seq,
                        room,
                    } => {
                        records.push((c, whole, seq));
                        rooms.push(room);
                    }
                    Arrival::Headers { c, headers } => {
                        self.apply_batch(take(&mut records)).await;
                        self.headers(c, headers).await;
                    }
                    Arrival::Done { c, done } => {
                        self.apply_batch(take(&mut records)).await;
                        self.done(c, done).await;
                    }
                }
            }
            self.apply_batch(records).await;
            drop(rooms);
        }
    }

    /// Apply records that arrived, crediting fetched ones to the catch-up on
    /// their socket, then say through which arrival everything is applied.
    async fn apply_batch(self: &Arc<Self>, records: Vec<(usize, Whole, u64)>) {
        let Some(upto) = records.iter().map(|(_, _, seq)| *seq).max() else {
            return;
        };
        let advance = |at: &mut u64| {
            let advanced = upto > *at;
            *at = (*at).max(upto);
            advanced
        };
        // In fault mode nothing is applied: what arrives is fetched again by
        // the catch-up that leaving starts (§7.7). It counts as handled, so a
        // request waiting on it is not held for it.
        if self.faulted() {
            self.applied_upto.send_if_modified(advance);
            return;
        }
        // Per record, in the batch's order: its socket and length when it was
        // fetched for a catch-up.
        let fetched: Vec<Option<(usize, u64)>> = records
            .iter()
            .map(|(c, w, _)| w.reply_to.map(|_| (*c, w.bytes.len() as u64)))
            .collect();
        let batch: Vec<(u64, u64, Vec<u8>)> = records
            .into_iter()
            .map(|(_, w, _)| (w.origin, w.oseq, w.bytes))
            .collect();
        let store = self.store.clone();
        let applied = tokio::task::spawn_blocking(move || store.apply(&batch)).await;
        // Arrivals are applied in order, so everything through this batch is
        // done — refused records included: they are fetched again, not waited
        // for.
        self.applied_upto.send_if_modified(advance);
        let Ok(Applied { taken, reload }) = applied else {
            return;
        };
        {
            let mut s = self.state();
            let applied = taken.iter().filter(|t| **t).count() as u64;
            s.stats.applied += applied;
            s.stats.refused += taken.len() as u64 - applied;
            s.ack_dirty |= applied > 0;
            // Only a record the store took counts as caught up on. A refused
            // one is fetched again, and counting each arrival made a catch-up
            // that refused everything read as one that kept receiving — past
            // its own total, and enough to hold a trial open (`switch::SYNCING`).
            let received = fetched
                .into_iter()
                .zip(&taken)
                .filter_map(|(f, took)| f.filter(|_| *took));
            for (c, len) in received {
                let now = Instant::now();
                let ids: Vec<u64> = s
                    .catchups
                    .iter()
                    .filter(|(_, r)| r.socket == c && r.outstanding() > 0)
                    .map(|(id, _)| *id)
                    .take(1)
                    .collect();
                for id in ids {
                    s.heard.insert(id, now);
                    if let Some(r) = s.catchups.get_mut(&id) {
                        r.record(len);
                    }
                }
            }
        }
        if reload {
            self.reload().await;
        }
    }

    /// A `HEADERS` of one of this member's catch-ups.
    ///
    /// A listed record of a type this store never takes is counted gone, not
    /// wanted: fetched, it is refused, and fetched again every try of every
    /// catch-up — a member joining `.5` on 2026-10-17 refused 164 local-only
    /// records (the model binding, the tokenizer, branch checkpoints) and caught
    /// up on nothing. An answerer lists such records gone itself; this holds
    /// whatever the answerer runs.
    async fn headers(self: &Arc<Self>, c: usize, mut headers: Headers) {
        let (travel, stay): (Vec<_>, Vec<_>) = headers
            .records
            .into_iter()
            .partition(|r| self.store.travels(r.record_type));
        headers.records = travel;
        headers.gone.extend(stay.into_iter().map(|r| Gone {
            oseq: r.oseq,
            deleted_entry: None,
        }));
        let store = self.store.clone();
        let Ok(applied) = tokio::task::spawn_blocking(move || store.applied()).await else {
            return;
        };
        let origin = headers.origin;
        let gone = {
            let mut guard = self.state();
            let s = &mut *guard;
            let request = headers.request;
            let Some(r) = s.catchups.get_mut(&request) else {
                return;
            };
            if r.socket != c {
                return;
            }
            let claimed = &s.claimed;
            let (gone, wanted) = r.headers(headers, |o, q| {
                applied.contains(o, q) || claimed.contains(&(o, q))
            });
            s.claimed.extend(wanted);
            s.heard.insert(request, Instant::now());
            gone
        };
        if !gone.is_empty() {
            let store = self.store.clone();
            let _ = tokio::task::spawn_blocking(move || store.gone(origin, &gone)).await;
        }
    }

    /// A `DONE`: the end of a catch-up's headers, or of one of its fetches.
    async fn done(self: &Arc<Self>, c: usize, done: Done) {
        let store = self.store.clone();
        let Ok(applied) = tokio::task::spawn_blocking(move || store.applied()).await else {
            return;
        };
        let (fetches, finished, handle) = {
            let mut s = self.state();
            let owner = if s.catchups.contains_key(&done.request) {
                done.request
            } else if let Some(owner) = s.fetches.remove(&done.request) {
                owner
            } else {
                return;
            };
            let mut next = s.next_request;
            let mut ids = || {
                next += 1;
                next
            };
            let Some(r) = s.catchups.get_mut(&owner) else {
                return;
            };
            let (fetches, settled): (Vec<Fetch>, Vec<(u64, u64)>) = if owner == done.request {
                (r.headers_done(&mut ids), Vec::new())
            } else {
                r.fetch_done(done.request, |o, q| applied.contains(o, q), &mut ids)
            };
            let finished = r
                .finished()
                .then_some((r.purpose, r.socket, r.asks_again(&applied)));
            s.next_request = next;
            for k in settled {
                s.claimed.remove(&k);
            }
            for f in &fetches {
                s.fetches.insert(f.request, owner);
            }
            s.heard.insert(owner, Instant::now());
            if finished.is_some() {
                s.drop_catchup(owner);
            }
            (fetches, finished, s.sockets[c].clone())
        };
        if let Some(handle) = &handle {
            for f in fetches {
                let _ = handle.control(Frame::Fetch(f));
            }
        }
        match finished {
            Some((Purpose::Admission, socket, true)) => {
                self.start_catchup(socket, Purpose::Admission, None);
            }
            Some((purpose, socket, _)) => self.finished(purpose, socket).await,
            None => {}
        }
    }

    /// A catch-up finished — or, for an admission, went unanswered past
    /// `tick::ADMISSION_QUIET_LIMIT`.
    pub(super) async fn finished(self: &Arc<Self>, purpose: Purpose, c: usize) {
        if purpose != Purpose::Admission {
            return;
        }
        let (offer, phase) = {
            let mut s = self.state();
            if s.sockets[c].is_none() {
                return;
            }
            s.ready[c] = true;
            s.served = true;
            let offer = !s.offered;
            s.offered = true;
            (offer.then(|| s.sockets[c].clone()).flatten(), s.phase)
        };
        if let Some(handle) = offer {
            // The full resync's offer, once per process start (§7.5).
            let store = self.store.clone();
            if let Ok(applied) = tokio::task::spawn_blocking(move || store.applied()).await {
                let request = self.state().next_request();
                let _ = handle.catchup(Frame::Have(Have {
                    request,
                    reply_to: self.node,
                    applied,
                }));
            }
        }
        if phase == Phase::Rejoining {
            self.begin_rejoin();
        }
        if let Some(m) = self.member() {
            m.capacity_changed();
        }
        self.send_wake.notify_one();
    }

    /// A turn arrived ranked below one this member holds: stop taking work,
    /// hand back the claims, rebuild from the log, and come back (§6.6).
    pub(super) async fn reload(self: &Arc<Self>) {
        let was = {
            let mut s = self.state();
            if s.phase == Phase::Reloading {
                return;
            }
            let was = s.phase;
            s.phase = Phase::Reloading;
            s.stats.reloads += 1;
            was
        };
        if let Some(m) = self.member() {
            m.capacity_changed();
            m.release_all(self.store.own_through());
        }
        let store = self.store.clone();
        let _ = tokio::task::spawn_blocking(move || store.reload()).await;
        self.state().phase = was;
        if let Some(m) = self.member() {
            m.capacity_changed();
        }
    }
}
