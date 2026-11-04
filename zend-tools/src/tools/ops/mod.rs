//! The operations tools a worker reaches (`docs/jobs.md` §8): the five job
//! tools — `job_list`, `job_status`, `job_read`, `job_grep`, `job_cancel` —
//! over its tasks, builds, deploys and maintenance jobs; `service_deploy` and
//! `service_status`; `cluster_status`, `cluster_logs`, `cluster_audit` for
//! reading the cluster; and `timeline_read` for reading a stored conversation
//! — another run's turns, calls, projections and prompt. Each is a typed request handed whole to the daemon's
//! [`OpsRunner`], which runs the implementation its MCP tools run, so a worker
//! and Claude get one answer from one code path. The answer is the text Claude
//! reads too.
//!
//! [`OpsRunner`]: crate::OpsRunner

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ToolContext, ToolError};

pub mod act;
pub mod bots;
pub mod cluster;
pub mod job;
pub mod service;
pub mod task;
pub mod timeline;

#[derive(Debug, Error)]
pub enum OpsError {
    #[error("this conversation cannot reach the cluster's builds and deploys: no runner is configured for it")]
    NotConfigured,
    #[error("{0}")]
    Refused(String),
}

impl ToolError for OpsError {
    fn code(&self) -> &'static str {
        match self {
            OpsError::NotConfigured => "not_configured",
            OpsError::Refused(_) => "ops_refused",
        }
    }
}

/// What a build or deploy tool answers: where things are, and what to call
/// next.
#[derive(Serialize)]
pub struct Report {
    pub answer: String,
}

/// The services keep runs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Service {
    Motherd,
    Web,
    Mcp,
}

/// Hand `req` to the daemon's runner as tool `tool`.
pub fn forward<R: Serialize>(
    ctx: &ToolContext,
    tool: &'static str,
    req: &R,
) -> Result<Report, OpsError> {
    let runner = ctx.ops_runner.as_deref().ok_or(OpsError::NotConfigured)?;
    let args = serde_json::to_value(req).map_err(|e| OpsError::Refused(e.to_string()))?;
    runner
        .run(ctx.lineage(), tool, args)
        .map(|answer| Report { answer })
        .map_err(OpsError::Refused)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::{json, Value};

    use super::*;
    use crate::OpsRunner;

    /// A runner that keeps what it was asked and answers it back.
    #[derive(Default)]
    struct Echo(Mutex<Vec<(String, String, Value)>>);

    impl OpsRunner for Echo {
        fn run(&self, caller: &str, tool: &str, args: Value) -> Result<String, String> {
            self.0
                .lock()
                .unwrap()
                .push((caller.to_string(), tool.to_string(), args.clone()));
            if tool == "job_cancel" {
                return Err("nothing to stop".into());
            }
            Ok(format!("{tool} {args}"))
        }
    }

    /// **A request goes to the runner whole, as the tool it is**, its absent
    /// fields left out rather than nulled, and a refusal comes back as one.
    #[test]
    fn a_request_reaches_the_runner_whole() {
        let echo = Arc::new(Echo::default());
        let ctx = ToolContext::new().with_ops_runner(Arc::clone(&echo) as Arc<dyn OpsRunner>);
        let req = job::ReadRequest {
            job: "build:76c3a0f4d39bd9db:Xe".into(),
            page: 2,
            wait_s: 120,
        };
        let report = forward(&ctx, "job_read", &req).unwrap();
        assert_eq!(
            report.answer,
            r#"job_read {"job":"build:76c3a0f4d39bd9db:Xe","page":2,"wait_s":120}"#
        );
        let refused = forward(
            &ctx,
            "job_cancel",
            &job::CancelRequest {
                job: "deploy:76c3a0f4d39bd9db:Xe".into(),
            },
        );
        assert!(matches!(refused, Err(OpsError::Refused(why)) if why == "nothing to stop"));
        let listed = forward(
            &ctx,
            "job_list",
            &job::ListRequest {
                kind: Some(job::Kind::Deploy),
                running: None,
                limit: None,
            },
        )
        .unwrap();
        assert_eq!(listed.answer, r#"job_list {"kind":"deploy"}"#);
        let asked = echo.0.lock().unwrap();
        assert_eq!(asked[0].1, "job_read");
        assert_eq!(asked[1].2, json!({"job": "deploy:76c3a0f4d39bd9db:Xe"}));
        assert!(matches!(
            forward(&ToolContext::new(), "service_status", &json!({})),
            Err(OpsError::NotConfigured)
        ));
    }
}
