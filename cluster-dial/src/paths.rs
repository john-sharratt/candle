//! Which of a cluster's paths to dial, and when to leave one
//! (`docs/replicated_substrate.md` §3.2, paths). A path is one address an
//! endpoint resolved to. Paths are ranked by endpoint, then in the order the
//! endpoint resolved; a fault cools a path for a while, doubling per
//! consecutive fault.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

use tokio::time::Instant;

/// The timings of a cluster's paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timing {
    /// How long one path has to connect: TCP, TLS and the WebSocket handshake.
    pub connect: Duration,
    /// A path's first cooldown, doubled per consecutive fault.
    pub cool_first: Duration,
    /// The longest a path cools.
    pub cool_max: Duration,
    /// A session lasting this long clears its path's faults.
    pub healthy_after: Duration,
    /// How often a socket on a lesser path probes a better one.
    pub return_every: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(10),
            cool_first: Duration::from_secs(30),
            cool_max: Duration::from_secs(30 * 60),
            healthy_after: Duration::from_secs(10 * 60),
            return_every: Duration::from_secs(60),
        }
    }
}

/// One path: the endpoint's index in its cluster, and an address it resolved
/// to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PathId {
    pub endpoint: usize,
    pub addr: SocketAddr,
}

#[derive(Debug, Default, Clone, Copy)]
struct Health {
    /// Consecutive faults.
    strikes: u32,
    cool_until: Option<Instant>,
}

/// The faults a cluster's paths have shown.
#[derive(Debug)]
pub struct Paths {
    timing: Timing,
    health: HashMap<PathId, Health>,
}

impl Paths {
    pub fn new(timing: Timing) -> Self {
        Self {
            timing,
            health: HashMap::new(),
        }
    }

    pub fn cooling(&self, p: PathId, now: Instant) -> bool {
        self.health
            .get(&p)
            .and_then(|h| h.cool_until)
            .is_some_and(|until| now < until)
    }

    /// `ranked`, best first, in the order to dial them: the paths not cooling,
    /// then those cooling. A cooling path is still dialled when nothing else
    /// answers.
    pub fn order(&self, ranked: &[PathId], now: Instant) -> Vec<PathId> {
        let (well, cooling): (Vec<PathId>, Vec<PathId>) =
            ranked.iter().partition(|p| !self.cooling(**p, now));
        well.into_iter().chain(cooling).collect()
    }

    /// `p` showed a fault: it cools, for longer the more faults it has shown in
    /// a row. Returns how long.
    pub fn strike(&mut self, p: PathId, now: Instant) -> Duration {
        let h = self.health.entry(p).or_default();
        h.strikes = h.strikes.saturating_add(1);
        let doublings = (h.strikes - 1).min(16);
        let cool = self
            .timing
            .cool_first
            .saturating_mul(1 << doublings)
            .min(self.timing.cool_max);
        h.cool_until = Some(now + cool);
        cool
    }

    /// A session on `p` lasted: its faults are forgotten.
    pub fn lasted(&mut self, p: PathId) {
        self.health.remove(&p);
    }

    /// The best path not cooling of an endpoint better than `on`'s — where a
    /// socket on `on` returns to. A path of `on`'s own endpoint is its equal,
    /// never better.
    pub fn better(&self, ranked: &[PathId], on: PathId, now: Instant) -> Option<PathId> {
        ranked
            .iter()
            .copied()
            .find(|p| p.endpoint < on.endpoint && !self.cooling(*p, now))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(endpoint: usize, last: u8) -> PathId {
        PathId {
            endpoint,
            addr: SocketAddr::from(([192, 168, 0, last], 9443)),
        }
    }

    fn timing() -> Timing {
        Timing {
            cool_first: Duration::from_secs(30),
            cool_max: Duration::from_secs(240),
            ..Timing::default()
        }
    }

    /// The user's layout: `gateway.tokera.local` resolving to .4 and .5, then
    /// the internet name.
    fn ranked() -> Vec<PathId> {
        vec![path(0, 4), path(0, 5), path(1, 132)]
    }

    #[test]
    fn well_paths_are_dialled_in_rank_and_cooling_ones_last() {
        let t0 = Instant::now();
        let mut p = Paths::new(timing());
        assert_eq!(p.order(&ranked(), t0), ranked());

        p.strike(path(0, 4), t0);
        assert_eq!(
            p.order(&ranked(), t0),
            [path(0, 5), path(1, 132), path(0, 4)]
        );
        // Every path cooling: still every one dialled, best first.
        p.strike(path(0, 5), t0);
        p.strike(path(1, 132), t0);
        assert_eq!(p.order(&ranked(), t0), ranked());
        // The cooldown passes.
        assert_eq!(p.order(&ranked(), t0 + Duration::from_secs(30)), ranked());
    }

    #[test]
    fn a_cooldown_doubles_per_fault_to_its_cap_and_a_lasting_session_clears_it() {
        let t0 = Instant::now();
        let mut p = Paths::new(timing());
        let a = path(0, 5);
        let cools: Vec<u64> = (0..6).map(|_| p.strike(a, t0).as_secs()).collect();
        assert_eq!(cools, [30, 60, 120, 240, 240, 240]);
        assert!(p.cooling(a, t0 + Duration::from_secs(239)));
        assert!(!p.cooling(a, t0 + Duration::from_secs(240)));

        p.lasted(a);
        assert!(!p.cooling(a, t0));
        assert_eq!(p.strike(a, t0).as_secs(), 30);

        // Thousands of faults neither overflow nor pass the cap.
        for _ in 0..10_000 {
            p.strike(a, t0);
        }
        assert_eq!(p.strike(a, t0).as_secs(), 240);
    }

    #[test]
    fn a_socket_returns_only_to_a_better_endpoint_that_is_not_cooling() {
        let t0 = Instant::now();
        let mut p = Paths::new(timing());
        // On the internet path: the LAN's first address is better.
        assert_eq!(p.better(&ranked(), path(1, 132), t0), Some(path(0, 4)));
        // .4 cooling: .5 is.
        p.strike(path(0, 4), t0);
        assert_eq!(p.better(&ranked(), path(1, 132), t0), Some(path(0, 5)));
        // Both cooling: nowhere to return to, until one has cooled.
        p.strike(path(0, 5), t0);
        assert_eq!(p.better(&ranked(), path(1, 132), t0), None);
        assert_eq!(
            p.better(&ranked(), path(1, 132), t0 + Duration::from_secs(30)),
            Some(path(0, 4))
        );
        // On .5, .4 is its equal: a socket never leaves a well path for one.
        assert_eq!(
            p.better(&ranked(), path(0, 5), t0 + Duration::from_secs(30)),
            None
        );
        assert_eq!(p.better(&ranked(), path(0, 4), t0), None);
    }
}
