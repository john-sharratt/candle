//! A member's jobs, sent to an agent (§11.5).
//!
//! While any of the member's sockets is admitted, a job goes up a stream to
//! `group:agent` and the gateway picks the agent (§11.3). Cut off from every
//! cluster, the member reaches no agent but one in its own process —
//! `tokerad --agent` — and sends the job straight to it. With neither, the
//! job fails with [`AgentError::NoAgent`]: a zend tool reports `no_agent` to
//! the model, and a publish waits and tries again. That is the deployment, not
//! a fallback: a member given `--cluster` never runs a job on its own
//! checkout.
//!
//! [`Agents::run`] waits for the job and fetches its log; [`Agents::push`]
//! carries a push's bundle and answers how each ref came out;
//! [`Agents::fetch`] brings back what origin holds and the objects the member
//! lacks.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::{header, Method, Request, StatusCode};
use cluster::gateway::{LINEAGE_HEADER, NODE_HEADER, PUSH_HEADER, REPO_HEADER, TOOLCHAIN_HEADER};
use cluster::member::AppMember;
use http_body_util::BodyExt;
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use zend_vfs::{FetchBundle, JobStatus, PushResult, RunOutcome};

use crate::jobs::{JobError, Jobs};
use crate::member::APP;
use crate::wire::{FetchJob, JobView, PushJob, RunJob, Submit};

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
    member: Arc<AppMember>,
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
        Agents { member, in_process }
    }

    async fn call(&self, call: Call<'_>) -> Result<(Option<u64>, Bytes), AgentError> {
        let Some(conn) = self.member.open_group(APP) else {
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
        self.in_process.as_ref().filter(|_| !self.member.admitted())
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
}

fn refused(e: JobError) -> AgentError {
    match e {
        JobError::UnknownRepo(_) | JobError::NoPush(_) => AgentError::NoAgent(e.to_string()),
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
