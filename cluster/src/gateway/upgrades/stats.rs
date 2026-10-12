//! What a gateway gathers while it certifies a candidate (§3.8): per route,
//! each side's live requests — failures, latency — and the mirrored calls
//! compared, with the last few mismatches kept to show side by side. And the
//! two shares it holds to: the candidate's slice of live traffic, and the cap on
//! mirroring.

use std::collections::{BTreeMap, VecDeque};

use serde::Serialize;

use super::mirror::Compared;
use super::verdict::{Mirrored, RouteSamples, Side};
use crate::dispatch::Mirror;

/// The latencies kept per route and side: a recent window.
pub const LATENCIES_KEPT: usize = 4096;
/// The mismatching pairs kept, newest first.
pub const MISMATCHES_KEPT: usize = 5;
/// The most of a body a mismatch keeps to show.
pub const BODY_SHOWN: usize = 2048;
/// Mirroring is capped at this share of eligible requests, in percent.
pub const MIRROR_CAP_PCT: u64 = 20;

/// One live request's outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    pub route: String,
    pub mirror: Mirror,
    pub candidate: bool,
    /// A 5xx answer or a transport failure.
    pub failed: bool,
    pub latency_ms: u32,
}

/// A mirrored call whose answers differed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Mismatch {
    pub route: String,
    pub why: String,
    pub baseline: String,
    pub candidate: String,
}

/// Every `n`-th of a stream goes one way: `share` percent of picks, spread
/// evenly — the `k`-th pick is taken when `k · share / 100` steps up.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Share {
    seen: u64,
}

impl Share {
    pub fn take(&mut self, share_pct: u64) -> bool {
        self.seen += 1;
        self.seen * share_pct / 100 > (self.seen - 1) * share_pct / 100
    }
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    routes: BTreeMap<String, RouteSamples>,
    /// Mirror-eligible requests seen, and how many were mirrored.
    eligible: u64,
    mirrored: u64,
    pub mismatches: VecDeque<Mismatch>,
}

fn push_latency(side: &mut Side, ms: u32) {
    if side.latencies_ms.len() == LATENCIES_KEPT {
        side.latencies_ms.remove(0);
    }
    side.latencies_ms.push(ms);
}

fn shown(body: &[u8]) -> String {
    let text = String::from_utf8_lossy(&body[..body.len().min(BODY_SHOWN)]);
    text.into_owned()
}

impl Stats {
    fn route(&mut self, route: &str, mirror: &Mirror) -> &mut RouteSamples {
        self.routes
            .entry(route.to_string())
            .or_insert_with(|| RouteSamples {
                route: route.to_string(),
                mirror: mirror.clone(),
                candidate: Side::default(),
                baseline: Side::default(),
                mirrored: Mirrored::default(),
            })
    }

    pub fn record(&mut self, s: Sample) {
        let r = self.route(&s.route, &s.mirror);
        let side = if s.candidate {
            &mut r.candidate
        } else {
            &mut r.baseline
        };
        side.requests += 1;
        side.errors += s.failed as u64;
        push_latency(side, s.latency_ms);
    }

    /// A baseline request on a mirrored route: whether to mirror it, at most
    /// [`MIRROR_CAP_PCT`] of them.
    pub fn should_mirror(&mut self) -> bool {
        self.eligible += 1;
        if self.mirrored * 100 < self.eligible * MIRROR_CAP_PCT {
            self.mirrored += 1;
            return true;
        }
        false
    }

    /// A mirrored call compared.
    pub fn mirrored(
        &mut self,
        route: &str,
        mirror: &Mirror,
        compared: &Compared,
        baseline: &[u8],
        candidate: &[u8],
    ) {
        let r = self.route(route, mirror);
        r.mirrored.compared += 1;
        if let Compared::Differs(why) = compared {
            r.mirrored.mismatches += 1;
            if self.mismatches.len() == MISMATCHES_KEPT {
                self.mismatches.pop_back();
            }
            self.mismatches.push_front(Mismatch {
                route: route.to_string(),
                why: why.clone(),
                baseline: shown(baseline),
                candidate: shown(candidate),
            });
        }
    }

    pub fn samples(&self) -> Vec<RouteSamples> {
        self.routes.values().cloned().collect()
    }

    /// Mirror-eligible requests seen, and how many were mirrored.
    pub fn mirror_counts(&self) -> (u64, u64) {
        (self.eligible, self.mirrored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The candidate's share holds to its percentage, spread evenly.
    #[test]
    fn the_live_share_holds_to_its_percentage() {
        let mut s = Share::default();
        let taken: Vec<bool> = (0..20).map(|_| s.take(10)).collect();
        assert_eq!(taken.iter().filter(|t| **t).count(), 2);
        assert!(taken[9] && taken[19]);
        let mut all = Share::default();
        assert!((0..5).all(|_| all.take(100)));
        let mut none = Share::default();
        assert!((0..5).all(|_| !none.take(0)));
    }

    /// Mirroring stays within its cap of eligible requests.
    #[test]
    fn mirroring_is_capped_at_a_fifth() {
        let mut s = Stats::default();
        let mirrored = (0..1000).filter(|_| s.should_mirror()).count();
        assert_eq!(mirrored, 200);
        assert_eq!(s.mirror_counts(), (1000, 200));
    }

    /// Samples land on their route and side; mismatches keep the newest few
    /// pairs.
    #[test]
    fn samples_and_mismatches_are_kept_per_route() {
        let mut s = Stats::default();
        for candidate in [true, false, false] {
            s.record(Sample {
                route: "GET /v1/tools".into(),
                mirror: Mirror::Shape,
                candidate,
                failed: candidate,
                latency_ms: 7,
            });
        }
        let r = &s.samples()[0];
        assert_eq!((r.candidate.requests, r.candidate.errors), (1, 1));
        assert_eq!((r.baseline.requests, r.baseline.errors), (2, 0));
        assert_eq!(r.baseline.latencies_ms, [7, 7]);
        for i in 0..7 {
            s.mirrored(
                "GET /v1/tools",
                &Mirror::Shape,
                &Compared::Differs(format!("#{i}")),
                b"{}",
                b"[]",
            );
        }
        assert_eq!(s.mismatches.len(), MISMATCHES_KEPT);
        assert_eq!(s.mismatches[0].why, "#6");
        assert_eq!(s.samples()[0].mirrored.mismatches, 7);
    }
}
