//! The job API's bodies (§11.4).
//!
//! `POST /v1/jobs` takes a [`Submit`]:
//!
//! - **`run`** — `{ repo, branch, overlay, program, args, timeout_secs }`: run
//!   `program` on the checkout, on `branch` at the base the overlay names,
//!   with the overlay's changes laid down. `overlay` is the conversation's or
//!   operation's store [`Snapshot`], its base included.
//! - **`push`** — `{ repo, bundle, specs }`: take the bundle's objects — what
//!   the member built that origin lacks — and push `specs` to origin in one
//!   atomic push, each under its lease. The member builds commits and merges
//!   in its own repository; only the write to origin needs the credentials an
//!   agent holds.
//!
//! Every answer is a [`JobView`]: where the job stands, and once it has
//! ended, what it came to.

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use zend_vfs::{
    JobStatus, Lease, Oid, PushAction, PushOutcome, PushResult, PushSpec, PushTarget, RefName,
    RunOutcome, Snapshot,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "job", rename_all = "snake_case", deny_unknown_fields)]
pub enum Submit {
    Run(RunJob),
    Push(PushJob),
}

impl Submit {
    pub fn repo(&self) -> &str {
        match self {
            Submit::Run(r) => &r.repo,
            Submit::Push(p) => &p.repo,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunJob {
    pub repo: String,
    pub branch: String,
    pub overlay: Snapshot,
    pub program: String,
    pub args: Vec<String>,
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PushJob {
    pub repo: String,
    /// The objects origin lacks, as a git bundle in base64; `None` when it
    /// lacks none.
    pub bundle: Option<String>,
    pub specs: Vec<PushLine>,
}

/// One ref a push changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PushLine {
    /// The full ref on origin: `refs/heads/<b>` or `refs/tags/<t>`.
    pub target: String,
    /// The object it is set to; `None` deletes it.
    pub to: Option<String>,
    /// What origin must hold for the push to go ahead; `None`: nothing.
    pub expect: Option<String>,
}

/// A ref's push, as it came out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pushed {
    pub target: String,
    pub outcome: PushOutcome,
}

/// Where a job stands, and what it came to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobView {
    pub id: String,
    pub repo: String,
    #[serde(flatten)]
    pub status: JobStatus,
    /// Lines and bytes written to its log so far.
    pub lines: u64,
    pub bytes: u64,
    /// A run's outcome, once it has ended.
    pub outcome: Option<RunOutcome>,
    /// A push's refs, once it has been made.
    pub pushed: Vec<Pushed>,
}

/// Why a [`PushLine`] or [`Pushed`] does not name a push.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct BadPush(pub String);

fn target_of(name: &str) -> Result<PushTarget, BadPush> {
    RefName::parse(name)
        .ok()
        .and_then(|r| PushTarget::from_ref(&r))
        .ok_or_else(|| BadPush(format!("`{name}` is not a branch or a tag")))
}

fn oid(text: &str) -> Result<Oid, BadPush> {
    Oid::parse(text).map_err(|e| BadPush(format!("`{text}`: {e}")))
}

impl PushLine {
    pub fn of(spec: &PushSpec) -> PushLine {
        PushLine {
            target: spec.target.to_ref().to_string(),
            to: match &spec.action {
                PushAction::Update(o) => Some(o.to_string()),
                PushAction::Delete => None,
            },
            expect: match &spec.lease {
                Lease::Expect(o) => Some(o.to_string()),
                Lease::Absent => None,
            },
        }
    }

    pub fn spec(&self) -> Result<PushSpec, BadPush> {
        Ok(PushSpec {
            target: target_of(&self.target)?,
            action: match &self.to {
                Some(o) => PushAction::Update(oid(o)?),
                None => PushAction::Delete,
            },
            lease: match &self.expect {
                Some(o) => Lease::Expect(oid(o)?),
                None => Lease::Absent,
            },
        })
    }
}

impl Pushed {
    pub fn of(result: &PushResult) -> Pushed {
        Pushed {
            target: result.target.to_ref().to_string(),
            outcome: result.outcome.clone(),
        }
    }

    pub fn result(&self) -> Result<PushResult, BadPush> {
        Ok(PushResult {
            target: target_of(&self.target)?,
            outcome: self.outcome.clone(),
        })
    }
}

/// A bundle's bytes as the wire carries them.
pub fn encode_bundle(bytes: &[u8]) -> String {
    STANDARD.encode(bytes)
}

pub fn decode_bundle(text: &str) -> Result<Vec<u8>, BadPush> {
    STANDARD
        .decode(text)
        .map_err(|e| BadPush(format!("the bundle is not base64: {e}")))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;
    use zend_vfs::{BranchName, Rejection};

    use super::*;

    const A: &str = "1111111111111111111111111111111111111111";
    const B: &str = "2222222222222222222222222222222222222222";

    /// A run carries its overlay; a push its bundle and leases — exactly
    /// these bytes.
    #[test]
    fn a_submit_is_its_json() {
        let run = Submit::Run(RunJob {
            repo: "candle".into(),
            branch: "zen/x".into(),
            overlay: Snapshot::new(None, BTreeMap::new()),
            program: "cargo".into(),
            args: vec!["test".into()],
            timeout_secs: 600,
        });
        assert_eq!(
            serde_json::to_value(&run).unwrap(),
            json!({
                "job": "run", "repo": "candle", "branch": "zen/x",
                "overlay": {},
                "program": "cargo", "args": ["test"], "timeout_secs": 600
            })
        );
        let push = Submit::Push(PushJob {
            repo: "candle".into(),
            bundle: Some(encode_bundle(b"PACK")),
            specs: vec![PushLine::of(&PushSpec::branch(
                Oid::parse(B).unwrap(),
                BranchName::parse("main").unwrap(),
                Lease::Expect(Oid::parse(A).unwrap()),
            ))],
        });
        let wire = serde_json::to_value(&push).unwrap();
        assert_eq!(
            wire,
            json!({
                "job": "push", "repo": "candle", "bundle": "UEFDSw==",
                "specs": [{ "target": "refs/heads/main", "to": B, "expect": A }]
            })
        );
        assert_eq!(serde_json::from_value::<Submit>(wire).unwrap(), push);
    }

    /// A line names its spec back, deletes and absent leases included, and a
    /// ref that is neither a branch nor a tag is refused.
    #[test]
    fn push_lines_round_trip_and_bad_refs_are_refused() {
        let main = BranchName::parse("main").unwrap();
        for spec in [
            PushSpec::branch(Oid::parse(A).unwrap(), main.clone(), Lease::Absent),
            PushSpec::delete_branch(main, Oid::parse(B).unwrap()),
        ] {
            assert_eq!(PushLine::of(&spec).spec().unwrap(), spec);
        }
        let bad = PushLine {
            target: "refs/notes/x".into(),
            to: None,
            expect: Some(A.into()),
        };
        assert!(bad.spec().is_err());
        let rejected = Pushed {
            target: "refs/heads/main".into(),
            outcome: PushOutcome::Rejected(Rejection::Stale),
        };
        assert_eq!(Pushed::of(&rejected.result().unwrap()), rejected);
    }

    /// A view's status is flattened beside its counts.
    #[test]
    fn a_view_flattens_its_status() {
        let view = JobView {
            id: "AAAAAAAAAAA".into(),
            repo: "candle".into(),
            status: JobStatus::Exited { code: Some(0) },
            lines: 3,
            bytes: 40,
            outcome: None,
            pushed: Vec::new(),
        };
        let wire = serde_json::to_value(&view).unwrap();
        assert_eq!(
            wire,
            json!({
                "id": "AAAAAAAAAAA", "repo": "candle", "status": "exited", "code": 0,
                "lines": 3, "bytes": 40, "outcome": null, "pushed": []
            })
        );
        assert_eq!(serde_json::from_value::<JobView>(wire).unwrap(), view);
    }
}
