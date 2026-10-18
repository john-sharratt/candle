//! A candidate (§3.8): a member admitted above its app's floor, not a local
//! build. It offers its build — every sibling binary reporting the same build,
//! and its UI — and collects each gateway's `VERDICT`. Once every gateway it is
//! joined to has sent one, or at once on a failure, it proposes
//! `certify/<app>/<build>` with every gateway's figures: `Fail` if any failed,
//! else `Pass`. Every member then reads the same verdict from its substrate.
//!
//! A proposal that loses its number to another verdict is made again under the
//! next, until the group holds a verdict on the build.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;

use super::served::{offered, HeldAs};
use super::Upgrader;
use crate::clock::now_ms;
use crate::consensus::MAX_DECISION;
use crate::frame::{BuildId, Verdict as VerdictFrame, VerdictResult};
use crate::upgrade::artifacts::siblings;
use crate::upgrade::decisions::{
    build_name, certify_key, certify_prefix, encode, next, Settled, Verdict,
};

/// How long a proposal waits for the group to hold a verdict before it is
/// made again.
pub const PROPOSE_RETRY: Duration = Duration::from_secs(5);

/// The proposer a certification's verdict names.
pub const BY_CERTIFICATION: &str = "certification";

#[derive(Debug, Clone)]
pub struct Candidacy {
    pub build: BuildId,
    /// Each socket's verdict, by cluster.
    pub verdicts: BTreeMap<usize, VerdictFrame>,
    /// What it proposed, and when.
    pub proposed: Option<(Verdict, Instant)>,
}

/// The failing routes in a gateway's figures, as one line.
fn failing_routes(figures: &[u8]) -> Vec<String> {
    let Ok(v) = serde_json::from_slice::<Value>(figures) else {
        return Vec::new();
    };
    v["routes"]
        .as_array()
        .map(|routes| {
            routes
                .iter()
                .filter(|r| r["outcome"] == "fail")
                .map(|r| {
                    format!(
                        "{}: {}",
                        r["route"].as_str().unwrap_or_default(),
                        r["why"].as_str().unwrap_or_default()
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A gateway's figures in a line: its overall verdict and how its routes
/// came out. The routes themselves stay on the gateway's own page; the
/// decision carries what it rests on in a form that fits a Paxos value.
fn summary(figures: &[u8]) -> String {
    let Ok(v) = serde_json::from_slice::<Value>(figures) else {
        return String::new();
    };
    let routes = v["routes"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    let count = |o: &str| routes.iter().filter(|r| r["outcome"] == o).count();
    format!(
        "{}: {} pass, {} gap, {} fail",
        v["overall"].as_str().unwrap_or("no_opinion"),
        count("pass"),
        count("gap"),
        count("fail"),
    )
}

/// `v` cut to fit one decision ([`MAX_DECISION`]): a decision that does not
/// fit is refused by every vote journal, so the group would never hold it and
/// the candidate would propose it again forever. The reason is shortened
/// first, at a character boundary, then gateways' summaries are dropped from
/// the end.
fn fit(mut v: Verdict) -> Verdict {
    const MORE: &str = "…";
    while encode(&v).len() > MAX_DECISION {
        if !v.reason.is_empty() {
            let over = encode(&v).len() - MAX_DECISION + MORE.len();
            let mut keep = v.reason.len().saturating_sub(over);
            while !v.reason.is_char_boundary(keep) {
                keep -= 1;
            }
            v.reason.truncate(keep);
            if !v.reason.is_empty() {
                v.reason.push_str(MORE);
                if encode(&v).len() <= MAX_DECISION {
                    break;
                }
                // The JSON escaping of what is left still overflows: drop it.
                v.reason.clear();
            }
        } else if v.figures.pop().is_none() {
            break;
        }
    }
    v
}

/// The verdict a candidate proposes from the gateways' `verdicts`, once it
/// has one from every socket in `sockets`, or a failure from any.
pub fn settle(
    verdicts: &BTreeMap<usize, VerdictFrame>,
    sockets: &[usize],
    at_ms: u64,
) -> Option<Verdict> {
    let failed: Vec<&VerdictFrame> = verdicts
        .values()
        .filter(|v| v.result == VerdictResult::Fail)
        .collect();
    let all_in = !sockets.is_empty() && sockets.iter().all(|c| verdicts.contains_key(c));
    if failed.is_empty() && !all_in {
        return None;
    }
    let figures: Vec<String> = verdicts.values().map(|v| summary(&v.figures)).collect();
    Some(fit(if failed.is_empty() {
        Verdict {
            result: Settled::Pass,
            by: BY_CERTIFICATION.into(),
            at_ms,
            reason: String::new(),
            figures,
        }
    } else {
        Verdict {
            result: Settled::Fail,
            by: BY_CERTIFICATION.into(),
            at_ms,
            reason: failed
                .iter()
                .flat_map(|v| failing_routes(&v.figures))
                .collect::<Vec<_>>()
                .join("; "),
            figures,
        }
    }))
}

impl Upgrader {
    /// A gateway's verdict on this member's build: a gateway only judges a
    /// candidate, so one that arrives before this member's own tick has made
    /// it a candidate makes it one.
    pub(super) fn on_verdict(&self, app: &str, c: usize, v: VerdictFrame) {
        let own = self.setup.info.build();
        if v.build != own {
            return;
        }
        self.state()
            .candidates
            .entry(app.to_string())
            .or_insert_with(|| Candidacy {
                build: own,
                verdicts: BTreeMap::new(),
                proposed: None,
            })
            .verdicts
            .insert(c, v);
    }

    pub(super) async fn candidate_tick(self: &Arc<Self>) {
        let own = self.setup.info.build();
        if own.is_local() {
            return;
        }
        let own_name = build_name(&own);
        for m in self.apps() {
            let app = m.app().to_string();
            let sockets = m.admitted_sockets();
            let above = sockets
                .iter()
                .any(|c| m.floor(*c).is_some_and(|f| own.count > f.count));
            let (known, settled) = {
                let s = self.state();
                (
                    s.candidates.contains_key(&app),
                    s.facts
                        .verdicts
                        .contains_key(&(app.clone(), own_name.clone())),
                )
            };
            if !known {
                if !above || settled {
                    continue;
                }
                self.state().candidates.insert(
                    app.clone(),
                    Candidacy {
                        build: own.clone(),
                        verdicts: BTreeMap::new(),
                        proposed: None,
                    },
                );
                self.offer_siblings().await;
                tracing::info!(app, build = own_name, "a candidate: offering this build");
            }
            if settled {
                continue;
            }
            let proposal = {
                let mut s = self.state();
                let Some(cand) = s.candidates.get_mut(&app) else {
                    continue;
                };
                let due = cand
                    .proposed
                    .as_ref()
                    .is_none_or(|(_, at)| at.elapsed() >= PROPOSE_RETRY);
                if !due {
                    continue;
                }
                let Some(v) = settle(&cand.verdicts, &sockets, now_ms()) else {
                    continue;
                };
                cand.proposed = Some((v.clone(), Instant::now()));
                v
            };
            let Some(r) = m.replicator().cloned() else {
                continue;
            };
            let prefix = certify_prefix(&app, &own);
            let i = next(&r.decisions(&prefix).await, &prefix);
            tracing::warn!(
                app,
                build = own_name,
                result = ?proposal.result,
                "proposing the certification's verdict"
            );
            r.propose(certify_key(&app, &own, i).as_bytes(), encode(&proposal));
        }
    }

    /// Every sibling binary of this build, and its UI, served as its offer.
    async fn offer_siblings(&self) {
        if self
            .state()
            .served
            .iter()
            .any(|i| i.from == HeldAs::Offered)
        {
            return;
        }
        let own = self.setup.info.build();
        let dir: PathBuf = self
            .setup
            .exe
            .parent()
            .map(PathBuf::from)
            .unwrap_or_default();
        let probe = self.probe.clone();
        let build = own.clone();
        let found = tokio::task::spawn_blocking(move || siblings(&dir, &build, &*probe))
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or_default();
        let mut s = self.state();
        s.served.extend(offered(&own, found));
        s.announce_due = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(result: VerdictResult, figures: &str) -> VerdictFrame {
        VerdictFrame {
            build: BuildId {
                count: 12,
                hash: "h12".into(),
                local: String::new(),
            },
            result,
            figures: figures.as_bytes().to_vec(),
        }
    }

    /// The verdict waits for every gateway joined to, but a failure from any
    /// is proposed at once, naming its failing routes.
    #[test]
    fn every_gateway_answers_and_a_failure_is_proposed_at_once() {
        let mut got = BTreeMap::new();
        got.insert(0, verdict(VerdictResult::Pass, "{}"));
        assert_eq!(settle(&got, &[0, 1], 7), None);
        got.insert(1, verdict(VerdictResult::NoOpinion, "{}"));
        let pass = settle(&got, &[0, 1], 7).unwrap();
        assert_eq!(
            (pass.result, pass.by.as_str(), pass.figures.len()),
            (Settled::Pass, "certification", 2)
        );
        let mut one = BTreeMap::new();
        one.insert(
            1,
            verdict(
                VerdictResult::Fail,
                r#"{"overall":"fail","routes":[{"route":"GET /v1/tools","outcome":"fail","why":"its p50 is 1.60× the baseline's"},{"route":"GET /v1/x","outcome":"pass","why":""}]}"#,
            ),
        );
        let fail = settle(&one, &[0, 1], 9).unwrap();
        assert_eq!(fail.result, Settled::Fail);
        assert_eq!(
            fail.reason,
            "GET /v1/tools: its p50 is 1.60× the baseline's"
        );
        assert_eq!(settle(&BTreeMap::new(), &[], 1), None);
    }

    /// A gateway's figures are carried as a line: its overall verdict and
    /// how many routes passed, had gaps and failed.
    #[test]
    fn a_gateway_s_figures_are_summed_up_in_a_line() {
        let mut got = BTreeMap::new();
        got.insert(
            0,
            verdict(
                VerdictResult::Pass,
                r#"{"overall":"pass_with_gaps","routes":[{"route":"a","outcome":"pass"},{"route":"b","outcome":"gap"},{"route":"c","outcome":"pass"}]}"#,
            ),
        );
        got.insert(1, verdict(VerdictResult::NoOpinion, "not json"));
        let pass = settle(&got, &[0, 1], 3).unwrap();
        assert_eq!(
            pass.figures,
            vec![
                "pass_with_gaps: 2 pass, 1 gap, 0 fail".to_string(),
                String::new()
            ]
        );
    }

    /// However many routes fail and however long their reasons, the decision
    /// fits a vote journal: the reason is cut short and marked so.
    #[test]
    fn a_verdict_always_fits_a_decision() {
        let routes: Vec<String> = (0..40)
            .map(|i| {
                format!(
                    r#"{{"route":"GET /v1/route/{i}","outcome":"fail","why":"its p95 is 9.99× the baseline's — “slow”"}}"#
                )
            })
            .collect();
        let figures = format!(r#"{{"overall":"fail","routes":[{}]}}"#, routes.join(","));
        let mut got = BTreeMap::new();
        for c in 0..3 {
            got.insert(c, verdict(VerdictResult::Fail, &figures));
        }
        let fail = settle(&got, &[0, 1, 2], 5).unwrap();
        assert!(
            encode(&fail).len() <= MAX_DECISION,
            "{}",
            encode(&fail).len()
        );
        assert!(fail.reason.starts_with("GET /v1/route/0: its p95"));
        assert!(fail.reason.ends_with('…'));
        assert_eq!(
            fail.figures.len(),
            3,
            "summaries fit once the reason is cut"
        );
        assert_eq!(fail.figures[0], "fail: 0 pass, 0 gap, 40 fail");
    }

    /// A verdict that already fits is left as it is.
    #[test]
    fn a_verdict_that_fits_is_unchanged() {
        let v = Verdict {
            result: Settled::Fail,
            by: BY_CERTIFICATION.into(),
            at_ms: 1,
            reason: "GET /v1/x: slow".into(),
            figures: vec!["fail: 0 pass, 0 gap, 1 fail".into()],
        };
        assert_eq!(fit(v.clone()), v);
    }
}
