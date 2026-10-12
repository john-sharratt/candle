//! Certification's verdict (§3.8): per route, the candidate against the
//! baseline on live traffic and on mirrored calls, judged once the route has
//! enough samples, and the build's verdict from every route's.
//!
//! A route passes when it meets all of these, and fails on any:
//!
//! - **errors** — the candidate's rate of 5xx and transport failures is not
//!   worse than the baseline's by a one-sided two-proportion test at 99%
//!   confidence, with a tolerance of 0.5 percentage points;
//! - **latency** — the candidate's p50 is within 1.5× of the baseline's and its
//!   p95 within 2×;
//! - **mirrors** — `Exact` routes mismatch on at most 0.1% of compared calls,
//!   `Shape` routes on none.
//!
//! Live traffic counts once both sides have [`MIN_LIVE`] requests; mirrors once
//! [`MIN_MIRRORED`] calls were compared. A route with neither is a gap: it
//! neither passes nor fails.

use serde::Serialize;

use crate::dispatch::Mirror;

pub const MIN_LIVE: u64 = 200;
pub const MIN_MIRRORED: u64 = 100;
/// The one-sided 99% point of the standard normal.
pub const Z_99: f64 = 2.326_347_874_040_841;
pub const ERROR_TOLERANCE: f64 = 0.005;
pub const P50_RATIO: f64 = 1.5;
pub const P95_RATIO: f64 = 2.0;
pub const EXACT_MISMATCH_MAX: f64 = 0.001;

/// One side's live traffic on a route.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Side {
    pub requests: u64,
    /// 5xx answers and transport failures.
    pub errors: u64,
    /// Latencies in milliseconds — a recent window of them.
    pub latencies_ms: Vec<u32>,
}

/// A route's mirrored calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mirrored {
    pub compared: u64,
    pub mismatches: u64,
}

/// What one route gathered.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteSamples {
    /// `METHOD /path/pattern`.
    pub route: String,
    pub mirror: Mirror,
    pub candidate: Side,
    pub baseline: Side,
    pub mirrored: Mirrored,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Pass,
    Fail,
    Gap,
}

/// One route's figures, as the page and `VERDICT` carry them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RouteFigures {
    pub route: String,
    pub live: u64,
    pub baseline_live: u64,
    pub mirrored: u64,
    pub error_rate: f64,
    pub baseline_error_rate: f64,
    pub p50_ratio: Option<f64>,
    pub p95_ratio: Option<f64>,
    pub mismatch_rate: Option<f64>,
    pub outcome: Outcome,
    /// What failed, or what it lacks for a judgement.
    pub why: String,
}

/// The build's verdict on this gateway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Overall {
    Pass,
    /// Passed, with routes that had too few samples.
    PassWithGaps,
    Fail,
    /// No route had enough samples to judge.
    NoOpinion,
}

/// Everything a verdict rests on.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Figures {
    pub overall: Overall,
    pub routes: Vec<RouteFigures>,
}

fn rate(errors: u64, n: u64) -> f64 {
    if n == 0 {
        0.0
    } else {
        errors as f64 / n as f64
    }
}

/// Whether the candidate's error rate is significantly worse than the
/// baseline's, beyond the tolerance.
pub fn errors_worse(candidate: &Side, baseline: &Side) -> bool {
    let (nc, nb) = (candidate.requests as f64, baseline.requests as f64);
    let (pc, pb) = (
        rate(candidate.errors, candidate.requests),
        rate(baseline.errors, baseline.requests),
    );
    let pooled = (candidate.errors + baseline.errors) as f64 / (nc + nb);
    let se = (pooled * (1.0 - pooled) * (1.0 / nc + 1.0 / nb)).sqrt();
    let excess = pc - pb - ERROR_TOLERANCE;
    if se == 0.0 {
        return excess > 0.0;
    }
    excess / se > Z_99
}

/// The `q` quantile of `values`, nearest rank.
pub fn quantile(values: &[u32], q: f64) -> Option<u32> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_unstable();
    let rank = ((q * v.len() as f64).ceil() as usize).clamp(1, v.len());
    Some(v[rank - 1])
}

fn ratio(c: Option<u32>, b: Option<u32>) -> Option<f64> {
    match (c, b) {
        (Some(c), Some(b)) => Some(c as f64 / (b.max(1)) as f64),
        _ => None,
    }
}

/// Judge one route.
pub fn judge(r: &RouteSamples) -> RouteFigures {
    let live = r.candidate.requests >= MIN_LIVE && r.baseline.requests >= MIN_LIVE;
    let mirrored = r.mirrored.compared >= MIN_MIRRORED && r.mirror != Mirror::None;
    let p50 = ratio(
        quantile(&r.candidate.latencies_ms, 0.5),
        quantile(&r.baseline.latencies_ms, 0.5),
    );
    let p95 = ratio(
        quantile(&r.candidate.latencies_ms, 0.95),
        quantile(&r.baseline.latencies_ms, 0.95),
    );
    let mismatch_rate =
        (r.mirrored.compared > 0).then(|| rate(r.mirrored.mismatches, r.mirrored.compared));
    let mut failed = Vec::new();
    if live {
        if errors_worse(&r.candidate, &r.baseline) {
            failed.push("its error rate is worse than the baseline's".to_string());
        }
        if p50.is_some_and(|x| x > P50_RATIO) {
            failed.push(format!(
                "its p50 is {:.2}× the baseline's",
                p50.unwrap_or_default()
            ));
        }
        if p95.is_some_and(|x| x > P95_RATIO) {
            failed.push(format!(
                "its p95 is {:.2}× the baseline's",
                p95.unwrap_or_default()
            ));
        }
    }
    if mirrored {
        let allowed = match r.mirror {
            Mirror::Exact(_) => EXACT_MISMATCH_MAX,
            _ => 0.0,
        };
        if mismatch_rate.is_some_and(|m| m > allowed) {
            failed.push(format!(
                "{} of {} mirrored calls mismatched",
                r.mirrored.mismatches, r.mirrored.compared
            ));
        }
    }
    let (outcome, why) = if !failed.is_empty() {
        (Outcome::Fail, failed.join("; "))
    } else if live || mirrored {
        (Outcome::Pass, String::new())
    } else {
        (
            Outcome::Gap,
            format!(
                "{} live and {} mirrored: needs {MIN_LIVE} live on each side or {MIN_MIRRORED} \
                 mirrored",
                r.candidate.requests.min(r.baseline.requests),
                r.mirrored.compared
            ),
        )
    };
    RouteFigures {
        route: r.route.clone(),
        live: r.candidate.requests,
        baseline_live: r.baseline.requests,
        mirrored: r.mirrored.compared,
        error_rate: rate(r.candidate.errors, r.candidate.requests),
        baseline_error_rate: rate(r.baseline.errors, r.baseline.requests),
        p50_ratio: p50,
        p95_ratio: p95,
        mismatch_rate,
        outcome,
        why,
    }
}

/// The build's verdict from every route's.
pub fn verdict(routes: &[RouteSamples]) -> Figures {
    let routes: Vec<RouteFigures> = routes.iter().map(judge).collect();
    let any = |o: Outcome| routes.iter().any(|r| r.outcome == o);
    let overall = if any(Outcome::Fail) {
        Overall::Fail
    } else if !any(Outcome::Pass) {
        Overall::NoOpinion
    } else if any(Outcome::Gap) {
        Overall::PassWithGaps
    } else {
        Overall::Pass
    };
    Figures { overall, routes }
}

#[cfg(test)]
mod tests {
    use std::slice::from_ref;

    use super::*;

    fn side(requests: u64, errors: u64, latency: u32) -> Side {
        Side {
            requests,
            errors,
            latencies_ms: vec![latency; requests.min(500) as usize],
        }
    }

    fn route(candidate: Side, baseline: Side) -> RouteSamples {
        RouteSamples {
            route: "GET /v1/tools".into(),
            mirror: Mirror::None,
            candidate,
            baseline,
            mirrored: Mirrored::default(),
        }
    }

    /// Equal rates pass; a candidate failing far more often fails at 99%;
    /// a small excess inside the tolerance passes.
    #[test]
    fn the_error_test_passes_equal_rates_and_fails_a_worse_candidate() {
        assert!(!errors_worse(&side(1000, 10, 1), &side(1000, 10, 1)));
        assert!(errors_worse(&side(1000, 60, 1), &side(1000, 10, 1)));
        assert!(!errors_worse(&side(1000, 13, 1), &side(1000, 10, 1)));
        // With no errors on either side there is nothing worse.
        assert!(!errors_worse(&side(300, 0, 1), &side(300, 0, 1)));
        // Every request failing against none failing.
        assert!(errors_worse(&side(200, 200, 1), &side(200, 0, 1)));
    }

    /// The latency ratios pass at their bounds and fail past them.
    #[test]
    fn the_latency_ratios_hold_at_their_bounds() {
        let at = judge(&route(side(300, 0, 15), side(300, 0, 10)));
        assert_eq!((at.outcome, at.p50_ratio), (Outcome::Pass, Some(1.5)));
        let past = judge(&route(side(300, 0, 16), side(300, 0, 10)));
        assert_eq!(past.outcome, Outcome::Fail);
        assert_eq!(past.why, "its p50 is 1.60× the baseline's");
        let mut tail = side(300, 0, 10);
        for l in tail.latencies_ms.iter_mut().take(30) {
            *l = 25;
        }
        let p95 = judge(&route(tail, side(300, 0, 10)));
        assert_eq!(p95.p95_ratio, Some(2.5));
        assert_eq!(p95.outcome, Outcome::Fail);
    }

    /// Under the sample minimum a route is a gap; mirrors alone can judge it.
    #[test]
    fn a_route_under_its_minimum_is_a_gap() {
        let thin = judge(&route(side(199, 0, 1), side(500, 0, 1)));
        assert_eq!(thin.outcome, Outcome::Gap);
        assert_eq!(
            thin.why,
            "199 live and 0 mirrored: needs 200 live on each side or 100 mirrored"
        );
        let mut mirrored = route(side(10, 0, 1), side(10, 0, 1));
        mirrored.mirror = Mirror::Exact(vec![]);
        mirrored.mirrored = Mirrored {
            compared: 1000,
            mismatches: 1,
        };
        assert_eq!(judge(&mirrored).outcome, Outcome::Pass);
        mirrored.mirrored.mismatches = 2;
        assert_eq!(judge(&mirrored).outcome, Outcome::Fail);
        mirrored.mirror = Mirror::Shape;
        mirrored.mirrored.mismatches = 1;
        assert_eq!(judge(&mirrored).outcome, Outcome::Fail);
    }

    #[test]
    fn the_build_fails_on_any_route_and_passes_with_gaps() {
        let good = route(side(300, 0, 10), side(300, 0, 10));
        let thin = route(side(5, 0, 10), side(300, 0, 10));
        let bad = route(side(300, 100, 10), side(300, 0, 10));
        assert_eq!(verdict(from_ref(&good)).overall, Overall::Pass);
        assert_eq!(
            verdict(&[good.clone(), thin.clone()]).overall,
            Overall::PassWithGaps
        );
        assert_eq!(verdict(&[good, thin.clone(), bad]).overall, Overall::Fail);
        assert_eq!(verdict(&[thin]).overall, Overall::NoOpinion);
        assert_eq!(verdict(&[]).overall, Overall::NoOpinion);
    }

    #[test]
    fn quantiles_are_nearest_rank() {
        let v: Vec<u32> = (1..=100).collect();
        assert_eq!(quantile(&v, 0.5), Some(50));
        assert_eq!(quantile(&v, 0.95), Some(95));
        assert_eq!(quantile(&[], 0.5), None);
    }
}
