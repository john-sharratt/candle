//! A member's jobs, sent to an agent (§11.5).
//!
//! While any of the member's sockets is admitted, a job goes up a stream to
//! `group:agent` and the gateway picks the agent (§11.3). Cut off from every
//! cluster, the member reaches no agent but one in its own process —
//! `motherd --agent` — and sends the job straight to it. With neither, the
//! job fails with [`AgentError::NoAgent`]: a zend tool reports `no_agent` to
//! the model, and a publish waits and tries again. That is the deployment, not
//! a fallback: a member given `--cluster` never runs a job on its own
//! checkout.
//!
//! [`Agents::run`] waits for the job and fetches its log; [`Agents::push`]
//! carries a push's bundle and answers how each ref came out;
//! [`Agents::fetch`] brings back what origin holds and the objects the member
//! lacks.
//!
//! An admin watches the agents through the same streams (§11.7):
//! [`Agents::listing`] names every agent the gateway admitted, and
//! [`Agents::jobs_of`], [`Agents::tail`], [`Agents::cancel`] and
//! [`Agents::switch`] address one of them by its node.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::{header, Method, Request, StatusCode};
use cluster::gateway::{
    state_name, AgentRepo, AgentRow, AGENTS_PATH, LINEAGE_HEADER, NODE_HEADER, PUSH_HEADER,
    REPO_HEADER, TOOLCHAIN_HEADER,
};
use cluster::member::AppMember;
use cluster::version::this_build;
use http_body_util::BodyExt;
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use serde::de::DeserializeOwned;
use zend_vfs::{FetchBundle, JobStatus, PushResult, RemoteRefs, RunOutcome};

use crate::jobs::{JobError, Jobs};
use crate::member::{capacity, node_in, APP};
use crate::wire::{
    AgentJobs, FetchJob, JobView, ListJob, LogChunk, PushJob, RunJob, Submit, Switching,
};

/// How long each wait on a running job is held open.
const WAIT: Duration = Duration::from_secs(25);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AgentError {
    /// No agent can be reached, or none holds the job's repository and
    /// toolchain.
    #[error("no_agent: {0}")]
    NoAgent(String),
    /// The agent refused the job.
    #[error("the agent refused the job ({status}): {why}")]
    Refused { status: u16, why: String },
    #[error("the job's stream failed: {0}")]
    Transport(String),
}

/// Where a job ran: the agent the gateway chose, or this process's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum At {
    Agent(u64),
    InProcess,
}

/// A run that has ended.
#[derive(Debug, Clone, PartialEq)]
pub struct Ran {
    pub at: At,
    pub view: JobView,
    pub outcome: Option<RunOutcome>,
    pub log: Vec<u8>,
}

/// What a member's jobs go to.
#[derive(Clone)]
pub struct Agents {
    /// `None` for a daemon with no cluster: its agent is the one in process.
    member: Option<Arc<AppMember>>,
    /// The agent in this process, reached only while cut off.
    in_process: Option<Arc<Jobs>>,
}

/// One job's request: where it goes and what the gateway routes it by.
struct Call<'a> {
    method: Method,
    path: String,
    body: Vec<u8>,
    to: Option<u64>,
    lineage: Option<&'a str>,
    repo: Option<&'a str>,
    toolchain: Option<&'a str>,
    push: bool,
}

impl Agents {
    pub fn new(member: Arc<AppMember>, in_process: Option<Arc<Jobs>>) -> Agents {
        Agents {
            member: Some(member),
            in_process,
        }
    }

    /// The agent in this process alone: a daemon with no cluster, whose
    /// repositories are read and written through it as a member's are through
    /// the cluster's (`docs/substrate_git.md` §9).
    pub fn in_process(jobs: Arc<Jobs>) -> Agents {
        Agents {
            member: None,
            in_process: Some(jobs),
        }
    }

    async fn call(&self, call: Call<'_>) -> Result<(Option<u64>, Bytes), AgentError> {
        let Some(conn) = self.member.as_ref().and_then(|m| m.open_group(APP)) else {
            return Err(AgentError::NoAgent(
                "cut off from every cluster, and no agent runs in this process".into(),
            ));
        };
        let (mut sender, connection) = http1::handshake(TokioIo::new(conn))
            .await
            .map_err(|e| AgentError::Transport(e.to_string()))?;
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::debug!(error = %e, "a job stream ended");
            }
        });
        let mut req = Request::builder()
            .method(call.method)
            .uri(&call.path)
            .header(header::CONTENT_TYPE, "application/json");
        let headers = [
            (NODE_HEADER, call.to.map(|n| format!("{n:016x}"))),
            (LINEAGE_HEADER, call.lineage.map(str::to_string)),
            (REPO_HEADER, call.repo.map(str::to_string)),
            (TOOLCHAIN_HEADER, call.toolchain.map(str::to_string)),
            (PUSH_HEADER, call.push.then(|| "1".to_string())),
        ];
        for (name, value) in headers {
            if let Some(v) = value {
                req = req.header(name, v);
            }
        }
        let req = req
            .body(Body::from(call.body))
            .map_err(|e| AgentError::Transport(e.to_string()))?;
        let res = sender
            .send_request(req)
            .await
            .map_err(|e| AgentError::Transport(e.to_string()))?;
        let status = res.status();
        let agent = res
            .headers()
            .get(NODE_HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| u64::from_str_radix(v, 16).ok());
        let body = res
            .into_body()
            .collect()
            .await
            .map_err(|e| AgentError::Transport(e.to_string()))?
            .to_bytes();
        if status.is_success() {
            return Ok((agent, body));
        }
        let why = serde_json::from_slice::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| {
                let error = v.get("error")?.as_str()?.to_string();
                let message = v.get("message").and_then(|m| m.as_str());
                Some(message.map_or(error.clone(), |m| format!("{error}: {m}")))
            })
            .unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned());
        if status == StatusCode::SERVICE_UNAVAILABLE && why.starts_with("no_agent") {
            return Err(AgentError::NoAgent(why));
        }
        Err(AgentError::Refused {
            status: status.as_u16(),
            why,
        })
    }

    fn view_of(body: &Bytes) -> Result<JobView, AgentError> {
        serde_json::from_slice(body).map_err(|e| AgentError::Transport(e.to_string()))
    }

    /// Whether this member is cut off with an agent in its own process.
    fn local(&self) -> Option<&Arc<Jobs>> {
        self.in_process.as_ref().filter(|_| !self.admitted())
    }

    /// Run `job` for `lineage` and wait for it to end.
    pub async fn run(&self, lineage: &str, job: RunJob) -> Result<Ran, AgentError> {
        if let Some(jobs) = self.local() {
            return run_in_process(jobs, job).await;
        }
        let body = serde_json::to_vec(&Submit::Run(job.clone())).expect("a job serialises");
        let (agent, started) = self
            .call(Call {
                method: Method::POST,
                path: "/v1/jobs".into(),
                body,
                to: None,
                lineage: Some(lineage),
                repo: Some(&job.repo),
                toolchain: Some(&job.program),
                push: false,
            })
            .await?;
        let mut view = Self::view_of(&started)?;
        let agent = agent.ok_or_else(|| {
            AgentError::Transport("the gateway named no agent for the job".into())
        })?;
        while !view.status.is_finished() {
            let (_, body) = self
                .call(Call {
                    method: Method::GET,
                    path: format!("/v1/jobs/{}?wait={}", view.id, WAIT.as_secs()),
                    body: Vec::new(),
                    to: Some(agent),
                    lineage: None,
                    repo: None,
                    toolchain: None,
                    push: false,
                })
                .await?;
            view = Self::view_of(&body)?;
        }
        let log = self.log(At::Agent(agent), &view.id).await?;
        Ok(Ran {
            at: At::Agent(agent),
            outcome: view.outcome.clone(),
            view,
            log,
        })
    }

    /// Job `id`'s log, from wherever it ran.
    pub async fn log(&self, at: At, id: &str) -> Result<Vec<u8>, AgentError> {
        match at {
            At::InProcess => {
                let jobs = self
                    .in_process
                    .as_ref()
                    .ok_or_else(|| AgentError::NoAgent("no agent runs in this process".into()))?;
                let path = jobs.log(id).map_err(refused)?;
                tokio::fs::read(&path)
                    .await
                    .map_err(|e| AgentError::Transport(e.to_string()))
            }
            At::Agent(agent) => self
                .call(Call {
                    method: Method::GET,
                    path: format!("/v1/jobs/{id}/log"),
                    body: Vec::new(),
                    to: Some(agent),
                    lineage: None,
                    repo: None,
                    toolchain: None,
                    push: false,
                })
                .await
                .map(|(_, b)| b.to_vec()),
        }
    }

    /// Job `id` as it stands, from wherever it ran.
    pub async fn view(&self, at: At, id: &str) -> Result<JobView, AgentError> {
        match at {
            At::InProcess => self
                .in_process
                .as_ref()
                .ok_or_else(|| AgentError::NoAgent("no agent runs in this process".into()))?
                .view(id)
                .map_err(refused),
            At::Agent(agent) => {
                let (_, body) = self
                    .call(Call {
                        method: Method::GET,
                        path: format!("/v1/jobs/{id}"),
                        body: Vec::new(),
                        to: Some(agent),
                        lineage: None,
                        repo: None,
                        toolchain: None,
                        push: false,
                    })
                    .await?;
                Self::view_of(&body)
            }
        }
    }

    /// Whether any of the member's sockets is admitted, so the gateway's
    /// agents are in reach.
    pub fn admitted(&self) -> bool {
        self.member.as_ref().is_some_and(|m| m.admitted())
    }

    /// Where requests for agent `node` go: this process's own agent while it
    /// is cut off and `node` is that agent's, and through a gateway
    /// otherwise.
    pub fn at(&self, node: u64) -> At {
        match self.local() {
            Some(jobs) if node_in(jobs.checkouts().jobs_dir()).ok() == Some(node) => At::InProcess,
            _ => At::Agent(node),
        }
    }

    /// A request to agent `node` for `path`, answered as `T`.
    async fn ask<T: DeserializeOwned>(
        &self,
        method: Method,
        node: u64,
        path: String,
        body: Vec<u8>,
    ) -> Result<T, AgentError> {
        let (_, body) = self
            .call(Call {
                method,
                path,
                body,
                to: Some(node),
                lineage: None,
                repo: None,
                toolchain: None,
                push: false,
            })
            .await?;
        serde_json::from_slice(&body).map_err(|e| AgentError::Transport(e.to_string()))
    }

    fn own(&self) -> Result<&Arc<Jobs>, AgentError> {
        self.in_process
            .as_ref()
            .ok_or_else(|| AgentError::NoAgent("no agent runs in this process".into()))
    }

    /// Every agent this member can reach: those its gateway admitted, or,
    /// cut off from every cluster, the agent in its own process.
    pub async fn listing(&self) -> Result<Vec<AgentRow>, AgentError> {
        if let Some(jobs) = self.local() {
            return Ok(vec![self.own_row(jobs)]);
        }
        if !self.admitted() {
            return Ok(Vec::new());
        }
        let (_, body) = self
            .call(Call {
                method: Method::GET,
                path: AGENTS_PATH.into(),
                body: Vec::new(),
                to: None,
                lineage: None,
                repo: None,
                toolchain: None,
                push: false,
            })
            .await?;
        serde_json::from_slice(&body).map_err(|e| AgentError::Transport(e.to_string()))
    }

    /// The agent in this process, as a gateway would list it.
    fn own_row(&self, jobs: &Jobs) -> AgentRow {
        let cap = capacity(jobs, jobs.checkouts().toolchains());
        AgentRow {
            node: node_in(jobs.checkouts().jobs_dir())
                .map_or_else(|_| String::new(), |n| format!("{n:016x}")),
            host: self
                .member
                .as_ref()
                .map_or_else(String::new, |m| m.host().to_string()),
            build: this_build(),
            state: state_name(cap.state).to_string(),
            os: cap.os,
            cores: cap.cores,
            load_milli: cap.load_milli,
            mem_free_mib: cap.mem_free_mib,
            slots_free: cap.slots_free,
            jobs: cap.jobs,
            toolchains: cap.toolchains,
            low_priority: cap.low_priority,
            repos: jobs
                .checkouts()
                .ads()
                .into_iter()
                .map(|r| AgentRepo {
                    name: r.name,
                    origin: r.origin,
                    push: r.push,
                })
                .collect(),
            connected_ms: 0,
        }
    }

    /// What agent `at` is running and has run, newest first, up to `limit`.
    pub async fn jobs_of(&self, at: At, limit: usize) -> Result<AgentJobs, AgentError> {
        match at {
            At::InProcess => Ok(self.own()?.list(limit)),
            At::Agent(node) => {
                self.ask(
                    Method::GET,
                    node,
                    format!("/v1/jobs?limit={limit}"),
                    Vec::new(),
                )
                .await
            }
        }
    }

    /// Job `id`'s log on agent `at`, from byte `from`.
    pub async fn tail(&self, at: At, id: &str, from: u64) -> Result<LogChunk, AgentError> {
        match at {
            At::InProcess => self.own()?.tail(id, from).await.map_err(refused),
            At::Agent(node) => {
                self.ask(
                    Method::GET,
                    node,
                    format!("/v1/jobs/{id}/tail?from={from}"),
                    Vec::new(),
                )
                .await
            }
        }
    }

    /// Cancel job `id` on agent `at`, killing its process tree.
    pub async fn cancel(&self, at: At, id: &str) -> Result<JobView, AgentError> {
        match at {
            At::InProcess => self.own()?.cancel(id).map_err(refused),
            At::Agent(node) => {
                self.ask(Method::DELETE, node, format!("/v1/jobs/{id}"), Vec::new())
                    .await
            }
        }
    }

    /// Switch agent `at` on or off; answers how it now stands.
    pub async fn switch(&self, at: At, on: bool) -> Result<bool, AgentError> {
        match at {
            At::InProcess => {
                let jobs = self.own()?;
                jobs.set_enabled(on).map_err(refused)?;
                Ok(jobs.enabled())
            }
            At::Agent(node) => {
                let body =
                    serde_json::to_vec(&Switching { enabled: on }).expect("a switch serialises");
                let now: Switching = self
                    .ask(Method::PUT, node, "/v1/agent".into(), body)
                    .await?;
                Ok(now.enabled)
            }
        }
    }

    /// Push `job` from an agent holding its repository with push rights.
    pub async fn push(&self, job: PushJob) -> Result<Vec<PushResult>, AgentError> {
        let view = if let Some(jobs) = self.local() {
            jobs.submit(Submit::Push(job)).await.map_err(refused)?
        } else {
            let body = serde_json::to_vec(&Submit::Push(job.clone())).expect("a job serialises");
            let (_, body) = self
                .call(Call {
                    method: Method::POST,
                    path: "/v1/jobs".into(),
                    body,
                    to: None,
                    lineage: None,
                    repo: Some(&job.repo),
                    toolchain: None,
                    push: true,
                })
                .await?;
            Self::view_of(&body)?
        };
        if let JobStatus::Failed { why } = &view.status {
            return Err(AgentError::Refused {
                status: 500,
                why: why.clone(),
            });
        }
        view.pushed
            .iter()
            .map(|p| p.result().map_err(|e| AgentError::Transport(e.0)))
            .collect()
    }

    /// Fetch `job` from an agent holding its repository: what origin holds,
    /// and the objects this member lacks.
    pub async fn fetch(&self, job: FetchJob) -> Result<FetchBundle, AgentError> {
        let view = if let Some(jobs) = self.local() {
            jobs.submit(Submit::Fetch(job)).await.map_err(refused)?
        } else {
            let body = serde_json::to_vec(&Submit::Fetch(job.clone())).expect("a job serialises");
            let (_, body) = self
                .call(Call {
                    method: Method::POST,
                    path: "/v1/jobs".into(),
                    body,
                    to: None,
                    lineage: None,
                    repo: Some(&job.repo),
                    toolchain: None,
                    push: false,
                })
                .await?;
            Self::view_of(&body)?
        };
        if let JobStatus::Failed { why } = &view.status {
            return Err(AgentError::Refused {
                status: 500,
                why: why.clone(),
            });
        }
        view.fetched
            .ok_or_else(|| AgentError::Transport("the fetch's answer carried nothing".into()))?
            .bundle()
            .map_err(|e| AgentError::Transport(e.0))
    }

    /// What origin's refs point at now, read by an agent holding `job`'s
    /// repository, fetching nothing.
    pub async fn list(&self, job: ListJob) -> Result<RemoteRefs, AgentError> {
        let view = if let Some(jobs) = self.local() {
            jobs.submit(Submit::List(job)).await.map_err(refused)?
        } else {
            let body = serde_json::to_vec(&Submit::List(job.clone())).expect("a job serialises");
            let (_, body) = self
                .call(Call {
                    method: Method::POST,
                    path: "/v1/jobs".into(),
                    body,
                    to: None,
                    lineage: None,
                    repo: Some(&job.repo),
                    toolchain: None,
                    push: false,
                })
                .await?;
            Self::view_of(&body)?
        };
        if let JobStatus::Failed { why } = &view.status {
            return Err(AgentError::Refused {
                status: 500,
                why: why.clone(),
            });
        }
        view.listed
            .ok_or_else(|| AgentError::Transport("the listing's answer carried nothing".into()))?
            .refs()
            .map_err(|e| AgentError::Transport(e.0))
    }
}

fn refused(e: JobError) -> AgentError {
    match e {
        JobError::UnknownRepo(_) | JobError::NoPush(_) | JobError::Disabled | JobError::Offline => {
            AgentError::NoAgent(e.to_string())
        }
        other => AgentError::Refused {
            status: 400,
            why: other.to_string(),
        },
    }
}

async fn run_in_process(jobs: &Arc<Jobs>, job: RunJob) -> Result<Ran, AgentError> {
    let mut view = jobs.submit(Submit::Run(job)).await.map_err(refused)?;
    while !view.status.is_finished() {
        view = jobs.wait(&view.id, WAIT).await.map_err(refused)?;
    }
    let path = jobs.log(&view.id).map_err(refused)?;
    let log = tokio::fs::read(&path)
        .await
        .map_err(|e| AgentError::Transport(e.to_string()))?;
    Ok(Ran {
        at: At::InProcess,
        outcome: view.outcome.clone(),
        view,
        log,
    })
}
