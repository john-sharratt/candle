//! `run_command` and the git writers on agents (`docs/replicated_substrate.md`
//! §11.5): the [`Executor`] of a daemon given `--cluster`.
//!
//! A command goes to an agent with the conversation's store as its overlay —
//! its branch, the base it pinned, its uncommitted changes. What the command
//! changed comes back as deltas, and this daemon records them in the
//! conversation's store exactly as its own sandbox would: those the store
//! takes are the conversation's changes, the rest are reported not kept. The
//! job's log is kept beside this daemon's own job logs, so `run_output` reads
//! its pages here. A push is made by an agent ([`AgentPush`]).
//!
//! A daemon without `--cluster` runs commands on its own checkouts
//! ([`Sandboxes`](zend_tools::sandboxes::Sandboxes)); this is the other
//! deployment, never a fallback: no agent is the tool error `no_agent`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use agent::client::{AgentError, Agents, Ran as AgentRan};
use agent::pusher::AgentPush;
use agent::wire::RunJob;
use tokio::runtime::Handle;
use zend_tools::executor::Executor;
use zend_tools::sandboxes::{page, LogPage, Ran, SandboxesError};
use zend_vfs::{
    ChangedFile, DiskWriteGrant, JobId, JobInfo, JobNotFound, JobRequest, JobStatus, Push,
    RunOutcome, Unrecorded, VfsStore,
};

pub struct AgentExecutor {
    agents: Agents,
    runtime: Handle,
    /// Where each job's log is kept once it has come back.
    jobs_dir: PathBuf,
    /// Every job run, by id, as `run_output` reads it.
    ran: Mutex<HashMap<JobId, JobInfo>>,
}

impl AgentExecutor {
    pub fn new(agents: Agents, runtime: Handle, jobs_dir: PathBuf) -> AgentExecutor {
        AgentExecutor {
            agents,
            runtime,
            jobs_dir,
            ran: Mutex::new(HashMap::new()),
        }
    }

    /// Drive `job` on the daemon's runtime, where the cluster's sockets live,
    /// from the tool call's own thread.
    fn wait_for(&self, lineage: &str, job: RunJob) -> Result<AgentRan, AgentError> {
        let agents = self.agents.clone();
        let lineage = lineage.to_string();
        let (tx, rx) = mpsc::channel();
        self.runtime.spawn(async move {
            let _ = tx.send(agents.run(&lineage, job).await);
        });
        rx.recv()
            .map_err(|_| AgentError::Transport("the job was dropped".into()))?
    }
}

fn failed(e: AgentError) -> SandboxesError {
    match e {
        AgentError::NoAgent(why) => SandboxesError::NoAgent(why),
        AgentError::Refused { why, .. } => SandboxesError::AgentRefused(why),
        AgentError::Transport(why) => SandboxesError::AgentFailed(why),
    }
}

/// Record what the agent's command changed in `files`: what the store takes,
/// and what it refuses with why — beside what the agent's own store refused.
fn record(files: &VfsStore, outcome: RunOutcome) -> RunOutcome {
    let mut changed = Vec::new();
    let mut unrecorded = outcome.unrecorded;
    for ChangedFile { path, delta } in outcome.changed {
        match files.apply(&path, std::slice::from_ref(&delta)) {
            Ok(()) => changed.push(ChangedFile { path, delta }),
            Err(e) => unrecorded.push(Unrecorded {
                path,
                delta,
                why: e.to_string(),
            }),
        }
    }
    RunOutcome {
        changed,
        unrecorded,
        ..outcome
    }
}

impl Executor for AgentExecutor {
    /// Which agent holds the repository is the gateway's to know: a job none
    /// can take is answered `no_agent`.
    fn check(&self, _repo: &str) -> Result<(), SandboxesError> {
        Ok(())
    }

    fn run(
        &self,
        _grant: DiskWriteGrant,
        repo: &str,
        lineage: &str,
        request: JobRequest,
    ) -> Result<Ran, SandboxesError> {
        let job = RunJob {
            repo: repo.to_string(),
            branch: request.branch.to_string(),
            overlay: request.files.snapshot(),
            program: request.command.program.clone(),
            args: request.command.args.clone(),
            timeout_secs: request.command.timeout.as_secs(),
            stage: None,
        };
        let ran = self.wait_for(lineage, job).map_err(failed)?;
        let id = JobId::parse(&ran.view.id).ok_or_else(|| {
            SandboxesError::AgentFailed(format!("`{}` is not a job id", ran.view.id))
        })?;
        std::fs::create_dir_all(&self.jobs_dir).map_err(SandboxesError::Log)?;
        let log = self.jobs_dir.join(format!("{id}.log"));
        std::fs::write(&log, &ran.log).map_err(SandboxesError::Log)?;
        self.ran.lock().unwrap_or_else(|e| e.into_inner()).insert(
            id.clone(),
            JobInfo {
                status: ran.view.status.clone(),
                lines: ran.log.iter().filter(|b| **b == b'\n').count() as u64,
                bytes: ran.log.len() as u64,
                log: log.clone(),
            },
        );
        let outcome = match (ran.view.status, ran.outcome) {
            (JobStatus::Refused { why }, _) => return Err(SandboxesError::AgentRefused(why)),
            (JobStatus::Failed { why }, _) => return Err(SandboxesError::AgentFailed(why)),
            (_, Some(outcome)) => outcome,
            (status, None) => {
                return Err(SandboxesError::AgentFailed(format!(
                    "the job ended {status:?} with no outcome"
                )))
            }
        };
        Ok(Ran {
            job: id,
            outcome: record(&request.files, outcome),
            log,
        })
    }

    fn output(
        &self,
        _repo: &str,
        id: &JobId,
        page: usize,
    ) -> Result<(JobInfo, LogPage), SandboxesError> {
        let info = self
            .ran
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            .cloned()
            .ok_or_else(|| SandboxesError::NotFound(JobNotFound(id.clone())))?;
        let read = page::read(&info.log, page).map_err(SandboxesError::Log)?;
        Ok((info, read))
    }

    fn pusher(&self, repo: &str) -> Arc<dyn Push> {
        AgentPush::shared(self.agents.clone(), self.runtime.clone(), repo)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;
    use std::time::Instant;

    use agent::checkouts::Checkouts;
    use agent::config::{AgentConfig, CheckoutConfig};
    use agent::jobs::Jobs;
    use cluster::dispatch::RouteTable;
    use cluster::frame::{BuildId, Capacity};
    use cluster::handshake::GroupKey;
    use cluster::member::{AppMember, Identity};
    use tokio::task::spawn_blocking;
    use zend_vfs::{
        BranchName, CommandPolicy, FileDelta, GitSource, Lease, Oid, Priority, PushOutcome,
        PushSpec, RemoteName, Repo, Rev, SandboxCommand,
    };

    use super::*;

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["-c", "user.name=t", "-c", "user.email=t@t"])
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}: {out:?}");
        String::from_utf8(out.stdout).unwrap()
    }

    fn shell() -> &'static str {
        if cfg!(windows) {
            "cmd"
        } else {
            "sh"
        }
    }

    fn copy(from: &str, to: &str) -> SandboxCommand {
        if cfg!(windows) {
            SandboxCommand::new("cmd").args(["/C".to_string(), format!("copy {from} {to}")])
        } else {
            SandboxCommand::new("sh").args(["-c".to_string(), format!("cp {from} {to}")])
        }
    }

    /// A member that never dials: cut off, its jobs go to its own agent.
    fn cut_off() -> Arc<AppMember> {
        AppMember::new(
            Identity {
                app: "zend".into(),
                node: 1,
                host: "m".into(),
                card: 1,
                build: BuildId {
                    count: 1,
                    hash: "h".into(),
                    local: String::new(),
                },
                model: None,
                repos: Vec::new(),
                leaseholder: true,
            },
            GroupKey::from_bytes([1; 32]),
            Vec::new(),
            RouteTable::new(Vec::new()).unwrap(),
            Box::new(Capacity::default),
            None,
            Instant::now(),
        )
    }

    /// **`run_command` on an agent records what the command changed in the
    /// conversation's store**, keeps the log for `run_output`, and a push
    /// through the executor is made from the agent's checkout.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_command_and_a_push_run_on_the_agent() {
        let root = tempfile::tempdir().unwrap();
        let origin = root.path().join("origin.git");
        std::fs::create_dir(&origin).unwrap();
        git(&origin, &["init", "-q", "--bare", "-b", "main"]);
        let seed = root.path().join("seed");
        std::fs::create_dir(&seed).unwrap();
        git(&seed, &["init", "-q", "-b", "main"]);
        std::fs::write(seed.join("in.txt"), "hello\n").unwrap();
        git(&seed, &["add", "in.txt"]);
        git(&seed, &["commit", "-q", "-m", "first"]);
        git(&seed, &["push", "-q", origin.to_str().unwrap(), "main"]);
        let (member, agent_dir) = (root.path().join("member"), root.path().join("agent"));
        for dir in [&member, &agent_dir] {
            git(
                root.path(),
                &[
                    "clone",
                    "-q",
                    origin.to_str().unwrap(),
                    dir.to_str().unwrap(),
                ],
            );
        }
        let cfg = AgentConfig {
            jobs_dir: root.path().join("agent-jobs"),
            slots: Some(1),
            stage_dir: None,
            checkouts: vec![CheckoutConfig {
                name: "app".into(),
                dir: agent_dir.clone(),
                push: true,
            }],
        };
        let jobs = Jobs::new(
            Checkouts::open(&cfg, &CommandPolicy::allowing([shell()])).unwrap(),
            1,
            Priority::Below,
        );
        let executor = Arc::new(AgentExecutor::new(
            Agents::new(cut_off(), Some(jobs)),
            Handle::current(),
            root.path().join("jobs"),
        ));
        let main = BranchName::parse("main").unwrap();
        let files = Arc::new(VfsStore::on_branch(
            GitSource::open(&member).unwrap(),
            Rev::Branch(main.clone()),
        ));
        files.write("in.txt", "changed\n".into()).unwrap();

        let (ex, store) = (Arc::clone(&executor), Arc::clone(&files));
        let ran = spawn_blocking(move || {
            ex.run(
                DiskWriteGrant::issue(),
                "app",
                "conv-1",
                JobRequest {
                    branch: BranchName::parse("main").unwrap(),
                    files: store,
                    command: copy("in.txt", "out.txt"),
                },
            )
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(ran.outcome.exit_code, Some(0));
        assert!(matches!(
            &ran.outcome.changed[0].delta.delta,
            FileDelta::Replace { content } if content == "changed\n"
        ));
        assert_eq!(files.read("out.txt").unwrap().as_deref(), Some("changed\n"));
        let (info, _) = executor.output("app", &ran.job, 0).unwrap();
        assert_eq!(info.status, JobStatus::Exited { code: Some(0) });
        assert_eq!(info.log, ran.log);

        std::fs::write(member.join("in.txt"), "two\n").unwrap();
        git(&member, &["commit", "-q", "-am", "second"]);
        let tip = Oid::parse(git(&member, &["rev-parse", "HEAD"]).trim()).unwrap();
        let was = Oid::parse(git(&member, &["rev-parse", "origin/main"]).trim()).unwrap();
        let pusher = executor.pusher("app");
        let spec = PushSpec::branch(tip.clone(), main, Lease::Expect(was));
        let pushed = spawn_blocking(move || {
            Repo::open(&member)
                .unwrap()
                .pushing_through(pusher)
                .push(&RemoteName::parse("origin").unwrap(), &[spec])
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(pushed[0].outcome, PushOutcome::FastForward);
        assert_eq!(git(&origin, &["rev-parse", "main"]).trim(), tip.to_string());
    }
}
