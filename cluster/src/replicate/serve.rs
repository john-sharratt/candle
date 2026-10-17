//! Answering other members (§7.2, §7.5): a `CATCHUP` with the headers of what
//! this member covers completely, a `FETCH` with the records themselves, and
//! a restarted member's `HAVE` with a catch-up addressed back to it. Every
//! answer goes out on the socket its question came in on — that gateway
//! reaches the requester — and on the catch-up lane, behind live traffic.

use std::sync::Arc;

use futures_util::future::join_all;

use super::answer::answer_ranges;
use super::catchup::Purpose;
use super::replicator::Replicator;
use crate::frame::{Catchup, Done, Fetch, Frame, Have, Headers};

impl Replicator {
    /// List what the requester lacks, a range at a time. One answer runs per
    /// requester (`answering.rs`): asking again
    /// cancels the answer before, and so does the requester leaving every
    /// gateway's member list. A cancelled answer stops before its next range
    /// and sends no `DONE` — the requester has already dropped the request.
    pub(super) fn answer_catchup(self: &Arc<Self>, c: usize, asked: Catchup) {
        let (handle, ticket) = {
            let mut s = self.state();
            let Some(handle) = s.sockets[c].clone() else {
                return;
            };
            s.stats.answered += 1;
            (handle, s.answering.begin(asked.reply_to, asked.request))
        };
        let me = self.clone();
        let store = self.store.clone();
        let node = self.node;
        tokio::task::spawn_blocking(move || {
            let mine = store.applied();
            for (origin, from, to) in answer_ranges(&mine, &asked.since) {
                if !ticket.live() {
                    me.state().answering.end(&ticket);
                    return;
                }
                let listing = store.list(origin, from, to);
                let frame = Frame::Headers(Headers {
                    request: asked.request,
                    reply_to: asked.reply_to,
                    answerer: node,
                    origin,
                    from,
                    to,
                    gone: listing.gone,
                    records: listing.records,
                });
                if handle.catchup(frame).is_err() {
                    me.state().answering.end(&ticket);
                    return;
                }
            }
            let _ = handle.catchup(Frame::Done(Done {
                request: asked.request,
                reply_to: asked.reply_to,
            }));
            me.state().answering.end(&ticket);
        });
    }

    pub(super) fn answer_fetch(self: &Arc<Self>, c: usize, f: Fetch) {
        if f.from_node != self.node {
            return;
        }
        let Some(handle) = self.state().sockets[c].clone() else {
            return;
        };
        self.state().stats.fetched_for_others += f.oseqs.len() as u64;
        let store = self.store.clone();
        tokio::spawn(async move {
            let sending = handle.clone();
            let Ok(written) = tokio::task::spawn_blocking(move || {
                // Read together: a store whose reads take a lock takes it once
                // for the whole fetch, or not at all (`Store::read_many`).
                let read = store.read_many(f.origin, &f.oseqs);
                f.oseqs
                    .iter()
                    .zip(read)
                    .filter_map(|(oseq, bytes)| {
                        sending
                            .event(f.origin, *oseq, &bytes?, Some(f.reply_to), true)
                            .ok()
                    })
                    .collect::<Vec<_>>()
            })
            .await
            else {
                return;
            };
            // `DONE` follows the last record onto the lane, never ahead of it.
            join_all(written).await;
            let _ = handle.catchup(Frame::Done(Done {
                request: f.request,
                reply_to: f.reply_to,
            }));
        });
    }

    /// A restarted member offered what it holds: catch up from it on every
    /// origin where it holds more.
    pub(super) fn offered(self: &Arc<Self>, c: usize, h: Have) {
        if h.reply_to == self.node {
            return;
        }
        let me = self.clone();
        tokio::spawn(async move {
            let store = me.store.clone();
            let Ok(mine) = tokio::task::spawn_blocking(move || store.applied()).await else {
                return;
            };
            if !h.applied.origins_beyond(&mine).is_empty() {
                me.start_catchup(c, Purpose::Offer(h.reply_to), Some(h.reply_to));
            }
        });
    }
}
