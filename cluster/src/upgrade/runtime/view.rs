//! What a member reports of its upgrades in `COORD` (§12.1): built from what
//! the upgrades last read, so a report never waits on a disk or a lock it does
//! not own.

use std::path::Path;

use super::Upgrader;
use crate::coord::{CandidateLine, FetchLine, HoldLine, KeptLine, RolloutLine, UpgradeView};
use crate::frame::VerdictResult;
use crate::upgrade::decisions::build_name;
use crate::upgrade::kept::{builds_dir, list};
use crate::upgrade::sum::hex;

impl Upgrader {
    /// `app`'s upgrade, as `COORD` reports it.
    pub fn view(&self, app: &str) -> UpgradeView {
        let s = self.state();
        UpgradeView {
            kind: self.setup.info.kind.clone(),
            platform: self.setup.info.platform.clone(),
            log_format: self.setup.info.opens.log_format,
            census: s.census.clone(),
            model: self.setup.info.opens.model.clone(),
            holds: s
                .served
                .iter()
                .map(|i| HoldLine {
                    build: build_name(&i.build),
                    kind: i.artifact.kind.clone(),
                    platform: i.artifact.platform.clone(),
                    size: i.artifact.size,
                    sha256: hex(&i.artifact.sha256),
                    source: i.from.name().into(),
                    opens: i.info.as_ref().map(|b| b.opens.clone()),
                })
                .collect(),
            kept: s.kept.clone(),
            fetching: s.fetch.as_ref().map(|f| FetchLine {
                build: build_name(&f.build),
                kind: f.artifact.kind.clone(),
                bytes: f.offset,
                size: f.artifact.size,
                resumes: f.resumes,
            }),
            candidate: s.candidates.get(app).map(|c| CandidateLine {
                build: build_name(&c.build),
                verdicts: c
                    .verdicts
                    .values()
                    .map(|v| {
                        match v.result {
                            VerdictResult::Pass => "pass",
                            VerdictResult::Fail => "fail",
                            VerdictResult::NoOpinion => "no_opinion",
                        }
                        .to_string()
                    })
                    .collect(),
                proposed: c.proposed.as_ref().map(|(v, _)| v.result),
            }),
            step: s.step.clone(),
            stranded: s.stranded.clone(),
            failure: s.failure.clone(),
            refused: s.facts.refused.iter().cloned().collect(),
            verdicts: s.verdicts.get(app).cloned().unwrap_or_default(),
            rollout: s.facts.rollouts.get(app).map(|(n, r)| RolloutLine {
                n: *n,
                decision: r.clone(),
            }),
            paused: s.facts.paused.contains(app),
        }
    }
}

/// The kept builds' sizes and retirement, read from disk: what the status page
/// shows of each, and what `prune-builds` would free.
pub fn kept_lines(exe: &Path) -> Vec<KeptLine> {
    list(&builds_dir(exe))
        .into_iter()
        .map(|k| KeptLine {
            build: k.name,
            bytes: k.bytes,
            retired_ms: k.json.retired_ms,
        })
        .collect()
}
