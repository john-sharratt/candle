//! Which of a cluster's paths to dial, and when to leave one
//! (`docs/replicated_substrate.md` §3.2, paths). A path is one address an
//! endpoint resolved to, dialled from one of this machine's addresses. Paths
//! are ranked by endpoint, then in the order the endpoint resolved, then by
//! source; a fault cools a path for a while, doubling per consecutive fault.
//!
//! A fault also makes its path's source *suspect*: a dead link fails against
//! every destination, so once it has failed against one, its paths to the
//! others are dialled after every path from a source that is not suspect.
//! Only a success from that same source clears it.

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, SocketAddr};
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

/// One path: the endpoint's index in its cluster, an address it resolved to,
/// and the address it is dialled from — `None` when the system chooses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PathId {
    pub endpoint: usize,
    pub addr: SocketAddr,
    pub source: Option<IpAddr>,
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
    /// Sources whose latest outcome was a fault.
    suspect: HashSet<IpAddr>,
}

impl Paths {
    pub fn new(timing: Timing) -> Self {
        Self {
            timing,
            health: HashMap::new(),
            suspect: HashSet::new(),
        }
    }

    pub fn cooling(&self, p: PathId, now: Instant) -> bool {
        self.health
            .get(&p)
            .and_then(|h| h.cool_until)
            .is_some_and(|until| now < until)
    }

    /// Whether `p`'s source has failed and not succeeded since.
    pub fn suspect(&self, p: PathId) -> bool {
        p.source.is_some_and(|s| self.suspect.contains(&s))
    }

    /// `ranked`, best first, in the order to dial them: the paths not cooling
    /// from a source not suspect, then those not cooling from a suspect
    /// source, then those cooling. A cooling path is still dialled when
    /// nothing else answers.
    pub fn order(&self, ranked: &[PathId], now: Instant) -> Vec<PathId> {
        let tier = |p: &PathId| match (self.cooling(*p, now), self.suspect(*p)) {
            (false, false) => 0,
            (false, true) => 1,
            (true, _) => 2,
        };
        let mut order = ranked.to_vec();
        // Stable: within a tier, the rank holds.
        order.sort_by_key(tier);
        order
    }

    /// `p` showed a fault: it cools, for longer the more faults it has shown in
    /// a row, and its source turns suspect. Returns how long it cools, and
    /// whether the source has just turned suspect.
    pub fn strike(&mut self, p: PathId, now: Instant) -> (Duration, bool) {
        let h = self.health.entry(p).or_default();
        h.strikes = h.strikes.saturating_add(1);
        let doublings = (h.strikes - 1).min(16);
        let cool = self
            .timing
            .cool_first
            .saturating_mul(1 << doublings)
            .min(self.timing.cool_max);
        h.cool_until = Some(now + cool);
        let turned = p.source.is_some_and(|s| self.suspect.insert(s));
        (cool, turned)
    }

    /// A session opened on `p`: its source works. Returns whether that clears
    /// a suspicion.
    pub fn connected(&mut self, p: PathId) -> bool {
        p.source.is_some_and(|s| self.suspect.remove(&s))
    }

    /// A session on `p` lasted: its faults are forgotten.
    pub fn lasted(&mut self, p: PathId) {
        self.health.remove(&p);
    }

    /// The best path of an endpoint better than `on`'s that is not cooling —
    /// from a source not suspect when there is one — where a socket on `on`
    /// returns to. A path of `on`'s own endpoint is its equal, never better.
    pub fn better(&self, ranked: &[PathId], on: PathId, now: Instant) -> Option<PathId> {
        self.order(ranked, now)
            .into_iter()
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
            source: None,
        }
    }

    fn from(p: PathId, last: u8) -> PathId {
        PathId {
            source: Some(IpAddr::from([192, 168, 0, last])),
            ..p
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
        let cools: Vec<u64> = (0..6).map(|_| p.strike(a, t0).0.as_secs()).collect();
        assert_eq!(cools, [30, 60, 120, 240, 240, 240]);
        assert!(p.cooling(a, t0 + Duration::from_secs(239)));
        assert!(!p.cooling(a, t0 + Duration::from_secs(240)));

        p.lasted(a);
        assert!(!p.cooling(a, t0));
        assert_eq!(p.strike(a, t0).0.as_secs(), 30);

        // Thousands of faults neither overflow nor pass the cap.
        for _ in 0..10_000 {
            p.strike(a, t0);
        }
        assert_eq!(p.strike(a, t0).0.as_secs(), 240);
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

    /// BlackWorld's two links to one LAN: a fault from .4 against one
    /// destination puts .4's paths to every other destination behind .5's, and
    /// a success from .5 does not clear .4 — only one from .4 does.
    #[test]
    fn a_failing_source_is_suspect_everywhere_until_it_succeeds_itself() {
        let t0 = Instant::now();
        let mut p = Paths::new(timing());
        let (lan, internet) = (path(0, 6), path(1, 132));
        let ranked = vec![
            from(lan, 4),
            from(lan, 5),
            from(internet, 4),
            from(internet, 5),
        ];
        assert_eq!(p.order(&ranked, t0), ranked);

        let (_, turned) = p.strike(from(lan, 4), t0);
        assert!(turned, ".4 turns suspect at its first fault");
        let (_, again) = p.strike(from(lan, 4), t0);
        assert!(!again, "and is not reported again while suspect");
        assert_eq!(
            p.order(&ranked, t0),
            [
                from(lan, 5),
                from(internet, 5),
                // Not cooling, yet behind every path from .5.
                from(internet, 4),
                from(lan, 4),
            ]
        );

        assert!(!p.connected(from(internet, 5)), ".5 was never suspect");
        assert!(
            p.suspect(from(internet, 4)),
            "a success from .5 leaves .4 suspect"
        );

        assert!(p.connected(from(internet, 4)));
        assert!(!p.suspect(from(lan, 4)));
        // The path that faulted still cools; the source is no longer suspect.
        assert_eq!(
            p.order(&ranked, t0),
            [
                from(lan, 5),
                from(internet, 4),
                from(internet, 5),
                from(lan, 4),
            ]
        );
    }

    /// A path the system routes has no source to suspect.
    #[test]
    fn an_unbound_path_is_never_suspect() {
        let t0 = Instant::now();
        let mut p = Paths::new(timing());
        let (_, turned) = p.strike(path(0, 5), t0);
        assert!(!turned);
        assert!(!p.suspect(path(0, 4)));
        assert!(!p.connected(path(0, 5)));
    }

    /// Returning prefers a better endpoint's path from a source that is not
    /// suspect.
    #[test]
    fn a_return_avoids_a_suspect_source() {
        let t0 = Instant::now();
        let mut p = Paths::new(timing());
        let (lan, internet) = (path(0, 6), path(1, 132));
        let ranked = vec![from(lan, 4), from(lan, 5), from(internet, 5)];
        p.strike(
            from(
                PathId {
                    endpoint: 1,
                    ..path(1, 9)
                },
                4,
            ),
            t0,
        );
        assert_eq!(
            p.better(&ranked, from(internet, 5), t0),
            Some(from(lan, 5)),
            ".4 is suspect: the return goes to the LAN from .5"
        );
    }
}
