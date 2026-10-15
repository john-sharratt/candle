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
//! - **`fetch`** — `{ repo, branch, have }`: fetch origin, and answer with
//!   what it holds for `branch` — every branch when `branch` is absent — and a
//!   bundle of their objects the member's `have` does not reach. A member
//!   whose repository lives with an agent reads origin through it
//!   (`docs/replicated_substrate.md` §10.3).
//!
//! Every answer is a [`JobView`]: where the job stands, and once it has
//! ended, what it came to.

use std::collections::BTreeMap;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use zend_vfs::{
    BranchName, FetchBundle, JobStatus, Lease, Oid, PushAction, PushOutcome, PushResult, PushSpec,
    PushTarget, RefName, RunOutcome, Snapshot,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "job", rename_all = "snake_case", deny_unknown_fields)]
pub enum Submit {
    Run(RunJob),
    Push(PushJob),
    Fetch(FetchJob),
}

impl Submit {
    pub fn repo(&self) -> &str {
        match self {
            Submit::Run(r) => &r.repo,
            Submit::Push(p) => &p.repo,
            Submit::Fetch(f) => &f.repo,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchJob {
    pub repo: String,
    /// The branch asked for; `None`: every branch.
    pub branch: Option<String>,
    /// Commits the member holds: what the bundle leaves out.
    pub have: Vec<String>,
    /// At most this many commits the member lacks in one answer, oldest first;
    /// a longer history comes back [`partial`](Fetched::partial), in slices an
    /// answer can carry (`docs/substrate_git.md` §5.1). Unbounded when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_commits: Option<u32>,
}

/// What origin holds for a fetch, and the objects the member lacks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fetched {
    /// Each branch's commit on origin.
    pub tips: BTreeMap<String, String>,
    /// A git bundle in base64; `None` when the member lacks nothing.
    pub bundle: Option<String>,
    /// A slice of the history, not all of it: take it and ask again
    /// ([`zend_vfs::FetchBundle::partial`]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub partial: bool,
}

impl Fetched {
    pub fn of(f: &FetchBundle) -> Fetched {
        Fetched {
            tips: f
                .tips
                .iter()
                .map(|(b, o)| (b.as_str().to_string(), o.to_string()))
                .collect(),
            bundle: f.bundle.as_deref().map(encode_bundle),
            partial: f.partial,
        }
    }

    pub fn bundle(&self) -> Result<FetchBundle, BadWire> {
        let tips = self
            .tips
            .iter()
            .map(|(b, o)| {
                let branch = BranchName::parse(b).map_err(|e| BadWire(format!("`{b}`: {e}")))?;
                Ok((branch, oid(o)?))
            })
            .collect::<Result<_, BadWire>>()?;
        Ok(FetchBundle {
            tips,
            bundle: self.bundle.as_deref().map(decode_bundle).transpose()?,
            partial: self.partial,
        })
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
    /// A fetch's answer, on the answer to its submit only.
    pub fetched: Option<Fetched>,
}

/// What a job does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Run,
    Push,
    Fetch,
}

/// One job as `GET /v1/jobs` lists it: what it runs, and where it stands —
/// without a run's outcome, which only the member that sent it reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobLine {
    pub id: String,
    pub repo: String,
    pub kind: JobKind,
    /// The branch a run's store is on; `None` for a push or a fetch.
    pub branch: Option<String>,
    /// Its command line: a run's program and arguments, or the git command a
    /// push or a fetch makes.
    pub command: Vec<String>,
    #[serde(flatten)]
    pub status: JobStatus,
    /// Lines and bytes written to its log so far.
    pub lines: u64,
    pub bytes: u64,
    /// When it started and ended, in Unix milliseconds.
    pub started_ms: u64,
    pub ended_ms: Option<u64>,
    /// The files a run changed, once it has ended.
    pub changed: usize,
}

/// `GET /v1/jobs`: whether the agent takes jobs, its slots, and the jobs it
/// keeps, newest first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentJobs {
    pub enabled: bool,
    pub slots: u32,
    pub running: u32,
    pub jobs: Vec<JobLine>,
}

/// `GET /v1/jobs/:id/tail?from=<byte>`: a job's log from byte `from`, up to
/// [`TAIL_MAX`](crate::jobs::TAIL_MAX) of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogChunk {
    pub from: u64,
    /// The bytes read, in base64: a command's output need not be UTF-8, and a
    /// chunk can end inside a character.
    pub data: String,
    /// The log's length when it was read.
    pub size: u64,
    /// The job had ended before the log was read, so `size` is all of it.
    pub ended: bool,
}

impl LogChunk {
    pub fn of(from: u64, bytes: &[u8], size: u64, ended: bool) -> LogChunk {
        LogChunk {
            from,
            data: STANDARD.encode(bytes),
            size,
            ended,
        }
    }

    pub fn bytes(&self) -> Result<Vec<u8>, BadWire> {
        STANDARD
            .decode(&self.data)
            .map_err(|e| BadWire(format!("the log chunk is not base64: {e}")))
    }
}

/// `PUT /v1/agent`: switch the agent on or off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Switching {
    pub enabled: bool,
}

/// Why a [`PushLine`], [`Pushed`] or [`Fetched`] does not read as one.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct BadWire(pub String);

fn target_of(name: &str) -> Result<PushTarget, BadWire> {
    RefName::parse(name)
        .ok()
        .and_then(|r| PushTarget::from_ref(&r))
        .ok_or_else(|| BadWire(format!("`{name}` is not a branch or a tag")))
}

fn oid(text: &str) -> Result<Oid, BadWire> {
    Oid::parse(text).map_err(|e| BadWire(format!("`{text}`: {e}")))
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

    pub fn spec(&self) -> Result<PushSpec, BadWire> {
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

    pub fn result(&self) -> Result<PushResult, BadWire> {
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

pub fn decode_bundle(text: &str) -> Result<Vec<u8>, BadWire> {
    STANDARD
        .decode(text)
        .map_err(|e| BadWire(format!("the bundle is not base64: {e}")))
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
            fetched: None,
        };
        let wire = serde_json::to_value(&view).unwrap();
        assert_eq!(
            wire,
            json!({
                "id": "AAAAAAAAAAA", "repo": "candle", "status": "exited", "code": 0,
                "lines": 3, "bytes": 40, "outcome": null, "pushed": [], "fetched": null
            })
        );
        assert_eq!(serde_json::from_value::<JobView>(wire).unwrap(), view);
    }

    /// A fetch asks for a branch past what the member holds; its answer
    /// names each branch's commit and carries the bundle — exactly these
    /// bytes — and reads back as the bundle the member takes.
    #[test]
    fn a_fetch_and_its_answer_are_their_json() {
        let fetch = Submit::Fetch(FetchJob {
            repo: "mind".into(),
            branch: Some("main".into()),
            have: vec![A.into()],
            max_commits: None,
        });
        assert_eq!(
            serde_json::to_value(&fetch).unwrap(),
            json!({ "job": "fetch", "repo": "mind", "branch": "main", "have": [A] })
        );
        let answer = Fetched {
            tips: BTreeMap::from([("main".to_string(), B.to_string())]),
            bundle: Some(encode_bundle(b"PACK")),
            partial: false,
        };
        assert_eq!(
            serde_json::to_value(&answer).unwrap(),
            json!({ "tips": { "main": B }, "bundle": "UEFDSw==" })
        );
        let taken = answer.bundle().unwrap();
        assert_eq!(
            taken.tips,
            BTreeMap::from([(BranchName::parse("main").unwrap(), Oid::parse(B).unwrap())])
        );
        assert_eq!(taken.bundle.as_deref(), Some(&b"PACK"[..]));
        assert_eq!(Fetched::of(&taken), answer);
        let bad = Fetched {
            tips: BTreeMap::from([("main".to_string(), "zz".to_string())]),
            bundle: None,
            partial: false,
        };
        assert!(bad.bundle().is_err());
    }

    /// **A listed job is its JSON**: its kind, its command line and its times
    /// beside the flattened status.
    #[test]
    fn a_listed_job_is_its_json() {
        let jobs = AgentJobs {
            enabled: false,
            slots: 4,
            running: 1,
            jobs: vec![JobLine {
                id: "AAAAAAAAAAA".into(),
                repo: "candle".into(),
                kind: JobKind::Run,
                branch: Some("zen/x".into()),
                command: vec!["cargo".into(), "test".into()],
                status: JobStatus::Running,
                lines: 2,
                bytes: 17,
                started_ms: 1_700_000_000_000,
                ended_ms: None,
                changed: 0,
            }],
        };
        let wire = serde_json::to_value(&jobs).unwrap();
        assert_eq!(
            wire,
            json!({
                "enabled": false, "slots": 4, "running": 1,
                "jobs": [{
                    "id": "AAAAAAAAAAA", "repo": "candle", "kind": "run", "branch": "zen/x",
                    "command": ["cargo", "test"], "status": "running",
                    "lines": 2, "bytes": 17, "started_ms": 1_700_000_000_000u64,
                    "ended_ms": null, "changed": 0
                }]
            })
        );
        assert_eq!(serde_json::from_value::<AgentJobs>(wire).unwrap(), jobs);
    }

    /// A log chunk carries its bytes in base64 — exactly these — and reads
    /// them back, a split character and all.
    #[test]
    fn a_log_chunk_carries_its_bytes() {
        let bytes = [b'o', b'k', b'\n', 0xE2, 0x80];
        let chunk = LogChunk::of(9, &bytes, 14, false);
        assert_eq!(
            serde_json::to_value(&chunk).unwrap(),
            json!({ "from": 9, "data": "b2sK4oA=", "size": 14, "ended": false })
        );
        assert_eq!(chunk.bytes().unwrap(), bytes);
    }
}
