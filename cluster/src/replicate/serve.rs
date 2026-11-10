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
            // What the requester holds of this member's own origin says which
            // of its gaps that member lacks; one every member lacks is gone,
            // and listed gone in this very answer (`own_gaps.rs`).
            let held = store.applied();
            if let Some(own) = held.origin(node) {
                // Read before the state is taken: the store's answer can wait
                // on a write whose tap takes the state.
                let written = store.own_through();
                let gone = {
                    let mut s = me.state();
                    let others = s.other_members(node);
                    s.own_gaps
                        .saw(asked.reply_to, asked.since.origin(node), own, written);
                    s.own_gaps.gone(&others)
                };
                for (from, to) in &gone {
                    tracing::warn!(
                        node,
                        from,
                        to,
                        "an own record no member holds is gone: every other member lacks it"
                    );
                    store.gone(node, &(*from..=*to).collect::<Vec<_>>());
                }
            }
            let mine = store.applied();
            let (mut ranges, mut listed, mut gone) = (0usize, 0usize, 0usize);
            for (origin, from, to) in answer_ranges(&mine, &asked.since) {
                if !ticket.live() {
                    me.state().answering.end(&ticket);
                    return;
                }
                let listing = store.list(origin, from, to);
                ranges += 1;
                listed += listing.records.len();
                gone += listing.gone.len();
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
            tracing::info!(
                node = format_args!("{node:016x}"),
                requester = format_args!("{:016x}", asked.reply_to),
                request = asked.request,
                ranges,
                listed,
                gone,
                "answered a catch-up"
            );
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
        let node = self.node;
        tokio::spawn(async move {
            let sending = handle.clone();
            let asked = f.oseqs.len();
            let (origin, reply_to, request) = (f.origin, f.reply_to, f.request);
            let Ok((written, unread)) = tokio::task::spawn_blocking(move || {
                // Read together: a store whose reads take a lock takes it once
                // for the whole fetch, or not at all (`Store::read_many`).
                let read = store.read_many(f.origin, &f.oseqs);
                let mut unread = Vec::new();
                let written = f
                    .oseqs
                    .iter()
                    .zip(read)
                    .filter_map(|(oseq, bytes)| {
                        let Some(bytes) = bytes else {
                            unread.push(*oseq);
                            return None;
                        };
                        sending
                            .event(f.origin, *oseq, &bytes, Some(f.reply_to), true)
                            .ok()
                    })
                    .collect::<Vec<_>>();
                (written, unread)
            })
            .await
            else {
                return;
            };
            // A record asked for and not held is fetched again by the
            // requester, and again: the same gap, every time it looks.
            if !unread.is_empty() || written.len() < asked {
                tracing::warn!(
                    node = format_args!("{node:016x}"),
                    requester = format_args!("{reply_to:016x}"),
                    request,
                    origin = format_args!("{origin:016x}"),
                    asked,
                    sent = written.len(),
                    unread = unread.len(),
                    unread_oseqs = ?unread.first().zip(unread.last()),
                    "a fetch was answered short: records asked for were not sent"
                );
            }
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
