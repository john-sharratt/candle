//! What a gateway's status page counts of its traffic (§12.2 Traffic and
//! Failures): streams open, requests by dispatch class, aggregates and the
//! members that missed their deadline, `409`s answered `503`, and the HTTP
//! exchanges a dropped socket failed in the last hour.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::dispatch::Dispatch;

/// How long a failure is counted.
pub const FAILURES_KEPT_MS: u64 = 60 * 60 * 1000;

/// Where a request went.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Best,
    Sticky,
    Owner,
    Aggregate,
    Any,
    Ui,
    /// The app was dumb: the route's configured upstream (§5.7).
    Configured,
}

impl Class {
    pub fn of(dispatch: &Dispatch) -> Class {
        match dispatch {
            Dispatch::Best => Class::Best,
            Dispatch::Sticky(_) => Class::Sticky,
            Dispatch::Owner(_) => Class::Owner,
            Dispatch::Aggregate(_) => Class::Aggregate,
            Dispatch::Any => Class::Any,
            Dispatch::Ui => Class::Ui,
        }
    }

    const ALL: [Class; 7] = [
        Class::Best,
        Class::Sticky,
        Class::Owner,
        Class::Aggregate,
        Class::Any,
        Class::Ui,
        Class::Configured,
    ];

    fn name(self) -> &'static str {
        match self {
            Class::Best => "best",
            Class::Sticky => "sticky",
            Class::Owner => "owner",
            Class::Aggregate => "aggregate",
            Class::Any => "any",
            Class::Ui => "ui",
            Class::Configured => "configured",
        }
    }
}

/// How a dropped socket failed an exchange (§5.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Failure {
    /// No response head yet: answered `503`.
    BeforeHead,
    /// A body in progress, aborted rather than ended.
    MidBody,
    /// A tunnelled WebSocket, closed `1011`.
    Tunnel,
}

#[derive(Debug, Default)]
pub struct Traffic {
    open_streams: Arc<AtomicU64>,
    requests: [AtomicU64; 7],
    aggregates: AtomicU64,
    past_deadline: AtomicU64,
    holder_elsewhere: AtomicU64,
    failures: Mutex<VecDeque<(u64, Failure)>>,
}

/// One open stream; counted until dropped.
pub struct OpenStream(Arc<AtomicU64>);

impl Drop for OpenStream {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrafficView {
    pub open_streams: u64,
    /// Requests since the gateway started, by class.
    pub requests: Vec<(&'static str, u64)>,
    pub aggregates: u64,
    pub members_past_deadline: u64,
    /// Requests a member answered `409` for a key held through another
    /// cluster, answered `503` (§5.4).
    pub holder_elsewhere: u64,
    /// The last hour's failed exchanges.
    pub failed_before_head: u64,
    pub aborted_mid_body: u64,
    pub tunnels_closed: u64,
}

impl Traffic {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stream(&self) -> OpenStream {
        self.open_streams.fetch_add(1, Ordering::Relaxed);
        OpenStream(self.open_streams.clone())
    }

    /// Streams open now.
    pub fn open_streams(&self) -> u64 {
        self.open_streams.load(Ordering::Relaxed)
    }

    pub fn request(&self, class: Class) {
        let i = Class::ALL.iter().position(|c| *c == class).expect("listed");
        self.requests[i].fetch_add(1, Ordering::Relaxed);
    }

    /// An aggregate answered, with `late` members past their deadline.
    pub fn aggregate(&self, late: u64) {
        self.aggregates.fetch_add(1, Ordering::Relaxed);
        self.past_deadline.fetch_add(late, Ordering::Relaxed);
    }

    pub fn holder_elsewhere(&self) {
        self.holder_elsewhere.fetch_add(1, Ordering::Relaxed);
    }

    pub fn failed(&self, how: Failure, at_ms: u64) {
        let mut f = self.failures.lock().unwrap();
        f.push_back((at_ms, how));
        while f
            .front()
            .is_some_and(|(at, _)| at_ms.saturating_sub(*at) > FAILURES_KEPT_MS)
        {
            f.pop_front();
        }
    }

    pub fn view(&self, now_ms: u64) -> TrafficView {
        let failures = self.failures.lock().unwrap();
        let count = |how: Failure| {
            failures
                .iter()
                .filter(|(at, f)| *f == how && now_ms.saturating_sub(*at) <= FAILURES_KEPT_MS)
                .count() as u64
        };
        TrafficView {
            open_streams: self.open_streams.load(Ordering::Relaxed),
            requests: Class::ALL
                .iter()
                .zip(&self.requests)
                .map(|(c, n)| (c.name(), n.load(Ordering::Relaxed)))
                .collect(),
            aggregates: self.aggregates.load(Ordering::Relaxed),
            members_past_deadline: self.past_deadline.load(Ordering::Relaxed),
            holder_elsewhere: self.holder_elsewhere.load(Ordering::Relaxed),
            failed_before_head: count(Failure::BeforeHead),
            aborted_mid_body: count(Failure::MidBody),
            tunnels_closed: count(Failure::Tunnel),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_requests_and_failures_are_counted() {
        let t = Traffic::new();
        let a = t.stream();
        let b = t.stream();
        drop(a);
        t.request(Class::Sticky);
        t.request(Class::Sticky);
        t.request(Class::Configured);
        t.aggregate(1);
        t.failed(Failure::BeforeHead, 1_000);
        t.failed(Failure::MidBody, 2_000);
        let v = t.view(3_000);
        assert_eq!(v.open_streams, 1);
        assert_eq!(v.requests[1], ("sticky", 2));
        assert_eq!(v.requests[6], ("configured", 1));
        assert_eq!((v.aggregates, v.members_past_deadline), (1, 1));
        assert_eq!((v.failed_before_head, v.aborted_mid_body), (1, 1));
        // An hour on, they no longer count.
        let later = t.view(2_000 + FAILURES_KEPT_MS + 1);
        assert_eq!((later.failed_before_head, later.aborted_mid_body), (0, 0));
        drop(b);
        assert_eq!(t.view(0).open_streams, 0);
    }
}
