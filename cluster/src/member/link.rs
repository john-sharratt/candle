//! A member's link to the clusters for one app (§3.2, §5.8): joined while at
//! least one of its sockets is admitted and ready, cut off when none is. At
//! startup it is neither until every cluster has failed one attempt or it has
//! been dialling for 10 s — a laptop off the network then serves locally at
//! once. A terminal refusal (4410, 4426) stops it with the reason.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::close::Close;

/// How long a starting member waits for a cluster before serving locally.
pub const START_WAIT: Duration = Duration::from_secs(10);

/// One app's state across its sockets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Link {
    /// Dialling, not yet decided.
    Starting,
    /// At least one socket is admitted and ready.
    Joined,
    /// No socket is: served locally, writes wait as detached (§6.7). Also a
    /// member admitted again that has not yet found a quorum to place what it
    /// wrote cut off.
    CutOff,
    /// Admitted again after writing cut off: catching up and taking its
    /// generation (§6.7).
    Rejoining,
    /// Rebuilding its substrate from the log (§6.6).
    Reloading,
    /// A refusal stopped the app on every cluster until the binary is replaced.
    Stopped(Close),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Socket {
    Dialling,
    Failed,
    Ready,
}

#[derive(Debug)]
pub struct Links {
    started: Instant,
    sockets: HashMap<usize, Socket>,
    stopped: Option<Close>,
}

impl Links {
    /// `clusters` sockets for one app, dialling from `started`.
    pub fn new(clusters: usize, started: Instant) -> Self {
        Self {
            started,
            sockets: (0..clusters).map(|c| (c, Socket::Dialling)).collect(),
            stopped: None,
        }
    }

    /// Cluster `c`'s socket was admitted and is ready.
    pub fn ready(&mut self, c: usize) {
        self.sockets.insert(c, Socket::Ready);
    }

    /// Cluster `c`'s socket dropped, or an attempt failed.
    pub fn failed(&mut self, c: usize) {
        self.sockets.insert(c, Socket::Failed);
    }

    /// A refusal that stops the app everywhere (4410).
    pub fn stop(&mut self, close: Close) {
        self.stopped = Some(close);
    }

    pub fn state(&self, now: Instant) -> Link {
        if let Some(close) = &self.stopped {
            return Link::Stopped(close.clone());
        }
        if self.sockets.values().any(|s| *s == Socket::Ready) {
            return Link::Joined;
        }
        let every_one_failed = self.sockets.values().all(|s| *s == Socket::Failed);
        if every_one_failed || now.duration_since(self.started) >= START_WAIT {
            return Link::CutOff;
        }
        Link::Starting
    }

    /// The clusters whose socket is ready, for dealing forwarded requests.
    pub fn ready_sockets(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .sockets
            .iter()
            .filter(|(_, s)| **s == Socket::Ready)
            .map(|(c, _)| *c)
            .collect();
        v.sort_unstable();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::close::CloseCode;

    #[test]
    fn a_member_joins_on_its_first_ready_socket_and_is_cut_off_after_its_last() {
        let t0 = Instant::now();
        let mut l = Links::new(2, t0);
        assert_eq!(l.state(t0), Link::Starting);
        l.ready(1);
        assert_eq!(l.state(t0), Link::Joined);
        l.failed(1);
        l.failed(0);
        assert_eq!(l.state(t0), Link::CutOff);
        l.ready(0);
        assert_eq!(l.state(t0), Link::Joined);
        assert_eq!(l.ready_sockets(), vec![0]);
    }

    /// A daemon with no reachable cluster serves locally once every cluster has
    /// failed one attempt, or after 10 s.
    #[test]
    fn a_member_with_no_reachable_cluster_is_cut_off_quickly() {
        let t0 = Instant::now();
        let mut l = Links::new(2, t0);
        l.failed(0);
        assert_eq!(l.state(t0 + Duration::from_secs(1)), Link::Starting);
        l.failed(1);
        assert_eq!(l.state(t0 + Duration::from_secs(1)), Link::CutOff);

        let slow = Links::new(1, t0);
        assert_eq!(slow.state(t0 + Duration::from_secs(9)), Link::Starting);
        assert_eq!(slow.state(t0 + START_WAIT), Link::CutOff);
    }

    #[test]
    fn a_terminal_refusal_stops_the_app() {
        let t0 = Instant::now();
        let mut l = Links::new(1, t0);
        l.ready(0);
        let close = Close::new(CloseCode::Superseded, "rebuild");
        l.stop(close.clone());
        assert_eq!(l.state(t0), Link::Stopped(close));
    }
}
