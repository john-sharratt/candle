//! Which of a cluster's paths to dial, and when to leave one
//! (`docs/replicated_substrate.md` §3.2, paths). A path is one address an
//! endpoint resolved to, dialled from one of this machine's addresses. Paths
//! are ranked by endpoint, then in the order the endpoint resolved, then by
//! source; a fault cools a path for a while, doubling per consecutive fault.
//!
//! A failure to *reach* — the bind or the TCP connect failing — also makes its
//! path's source *suspect*: a dead link fails against every destination, so
//! once it has failed against one, its paths to the others are dialled after
//! every path from a source that is not suspect. Only a success from that same
//! source clears it. A fault after the connect — TLS, the handshakes, a session
//! gone silent — is the far end's or the path's, never the source's: the
//! source reached.

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

/// Which part of a path failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// The bind or the TCP connect: the source did not reach.
    Reach,
    /// Anything after the connect.
    After,
}

/// What a fault did to its path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Struck {
    /// How long the path cools.
    pub cool: Duration,
    /// The path had no fault standing: this is its first in a row.
    pub first: bool,
    /// Its source has just turned suspect.
    pub suspect: bool,
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
    /// source, then those cooling, the soonest out of its cooldown first. A
    /// cooling path is still dialled when nothing else answers — and by the
    /// time the paths ahead of it have timed out, the one nearest the end of
    /// its cooldown is the likeliest to answer.
    pub fn order(&self, ranked: &[PathId], now: Instant) -> Vec<PathId> {
        let until = |p: &PathId| {
            self.health
                .get(p)
                .and_then(|h| h.cool_until)
                .filter(|until| now < *until)
        };
        let key = |p: &PathId| match (until(p), self.suspect(*p)) {
            (None, false) => (0, None),
            (None, true) => (1, None),
            (Some(until), _) => (2, Some(until)),
        };
        let mut order = ranked.to_vec();
        // Stable: within a tier, and between cooldowns ending together, the
        // rank holds.
        order.sort_by_key(key);
        order
    }

    /// `p` showed a fault: it cools, for longer the more faults it has shown in
    /// a row, and a failure to reach makes its source suspect.
    pub fn strike(&mut self, p: PathId, now: Instant, fault: Fault) -> Struck {
        let h = self.health.entry(p).or_default();
        let first = h.strikes == 0;
        h.strikes = h.strikes.saturating_add(1);
        let doublings = (h.strikes - 1).min(16);
        let cool = self
            .timing
            .cool_first
            .saturating_mul(1 << doublings)
            .min(self.timing.cool_max);
        h.cool_until = Some(now + cool);
        let suspect = fault == Fault::Reach && p.source.is_some_and(|s| self.suspect.insert(s));
        Struck {
            cool,
            first,
            suspect,
        }
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

        p.strike(path(0, 4), t0, Fault::Reach);
        assert_eq!(
            p.order(&ranked(), t0),
            [path(0, 5), path(1, 132), path(0, 4)]
        );
        // Every path cooling for as long: still every one dialled, best first.
        p.strike(path(0, 5), t0, Fault::After);
        p.strike(path(1, 132), t0, Fault::After);
        assert_eq!(p.order(&ranked(), t0), ranked());
        // The cooldown passes.
        assert_eq!(p.order(&ranked(), t0 + Duration::from_secs(30)), ranked());
    }

    /// AlienWorld at 00:51:23: every path cooling — the internet name for 4 s
    /// more, .4 for longer, .5 just struck for 120 s. Rank order dialled .4
    /// first, whose 10 s timeout ran past the internet name's cooldown, and then
    /// .5. The soonest out of its cooldown goes first.
    #[test]
    fn cooling_paths_are_dialled_soonest_out_first() {
        let t0 = Instant::now();
        let mut p = Paths::new(Timing {
            cool_first: Duration::from_secs(4),
            cool_max: Duration::from_secs(3600),
            ..Timing::default()
        });
        p.strike(path(1, 132), t0, Fault::After); // 4 s
        for _ in 0..4 {
            p.strike(path(0, 4), t0, Fault::Reach); // 32 s
        }
        for _ in 0..5 {
            p.strike(path(0, 5), t0, Fault::After); // 64 s
        }
        assert_eq!(
            p.order(&ranked(), t0),
            [path(1, 132), path(0, 4), path(0, 5)]
        );
    }

    #[test]
    fn a_cooldown_doubles_per_fault_to_its_cap_and_a_lasting_session_clears_it() {
        let t0 = Instant::now();
        let mut p = Paths::new(timing());
        let a = path(0, 5);
        let struck: Vec<(u64, bool)> = (0..6)
            .map(|_| {
                let s = p.strike(a, t0, Fault::After);
                (s.cool.as_secs(), s.first)
            })
            .collect();
        assert_eq!(
            struck,
            [
                (30, true),
                (60, false),
                (120, false),
                (240, false),
                (240, false),
                (240, false)
            ]
        );
        assert!(p.cooling(a, t0 + Duration::from_secs(239)));
        assert!(!p.cooling(a, t0 + Duration::from_secs(240)));

        p.lasted(a);
        assert!(!p.cooling(a, t0));
        let again = p.strike(a, t0, Fault::After);
        assert_eq!((again.cool.as_secs(), again.first), (30, true));

        // Thousands of faults neither overflow nor pass the cap.
        for _ in 0..10_000 {
            p.strike(a, t0, Fault::After);
        }
        assert_eq!(p.strike(a, t0, Fault::After).cool.as_secs(), 240);
    }

    #[test]
    fn a_socket_returns_only_to_a_better_endpoint_that_is_not_cooling() {
        let t0 = Instant::now();
        let mut p = Paths::new(timing());
        // On the internet path: the LAN's first address is better.
        assert_eq!(p.better(&ranked(), path(1, 132), t0), Some(path(0, 4)));
        // .4 cooling: .5 is.
        p.strike(path(0, 4), t0, Fault::Reach);
        assert_eq!(p.better(&ranked(), path(1, 132), t0), Some(path(0, 5)));
        // Both cooling: nowhere to return to, until one has cooled.
        p.strike(path(0, 5), t0, Fault::Reach);
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

        assert!(
            p.strike(from(lan, 4), t0, Fault::Reach).suspect,
            ".4 turns suspect at its first failure to reach"
        );
        assert!(
            !p.strike(from(lan, 4), t0, Fault::Reach).suspect,
            "and is not reported again while suspect"
        );
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
        assert!(!p.strike(path(0, 5), t0, Fault::Reach).suspect);
        assert!(!p.suspect(path(0, 4)));
        assert!(!p.connected(path(0, 5)));
    }

    /// AlienWorld's Wi-Fi, 192.168.0.237, carried a session to the internet
    /// name that went silent because the gateway's sending stalled — and was
    /// made suspect for it, putting its Hyper-V address, which reaches
    /// nothing, first. A fault after the connect cools the path and leaves the
    /// source alone: it reached.
    #[test]
    fn a_fault_after_the_connect_never_makes_its_source_suspect() {
        let t0 = Instant::now();
        let mut p = Paths::new(timing());
        let wifi = PathId {
            source: Some(IpAddr::from([192, 168, 0, 237])),
            ..path(1, 132)
        };
        let struck = p.strike(wifi, t0, Fault::After);
        assert!(!struck.suspect);
        assert!(p.cooling(wifi, t0));
        assert!(!p.suspect(wifi));
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
            Fault::Reach,
        );
        assert_eq!(
            p.better(&ranked, from(internet, 5), t0),
            Some(from(lan, 5)),
            ".4 is suspect: the return goes to the LAN from .5"
        );
    }
}
