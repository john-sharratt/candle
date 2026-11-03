//! The build and deploy tools a worker reaches (`docs/build_deploy_tools.md`
//! §6): `service_deploy`, `service_status`, `service_log`, `service_cancel`,
//! and `repo_job_status`, `repo_job_log`, `repo_job_list`, `repo_job_cancel`
//! for watching what a deploy builds. Each is a typed request handed whole to
//! the daemon's [`OpsRunner`], which runs the implementation its MCP tools
//! run, so a worker and Claude get one answer from one code path. The answer
//! is the text Claude reads too.
//!
//! [`OpsRunner`]: crate::OpsRunner

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ToolContext, ToolError};

pub mod job;
pub mod service;

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

/// How a log is read (`docs/build_deploy_tools.md` §3).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum LogMode {
    Summary,
    Tail,
    Lines,
    Grep,
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
            if tool == "service_cancel" {
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
        let req = job::LogRequest {
            job: "76c3a0f4d39bd9db:Xe".into(),
            mode: LogMode::Grep,
            pattern: Some("FAILED".into()),
            context: None,
            from: None,
            count: None,
            lines: None,
        };
        let report = forward(&ctx, "repo_job_log", &req).unwrap();
        assert_eq!(
            report.answer,
            r#"repo_job_log {"job":"76c3a0f4d39bd9db:Xe","mode":"grep","pattern":"FAILED"}"#
        );
        let refused = forward(
            &ctx,
            "service_cancel",
            &service::CancelRequest {
                deploy: "76c3a0f4d39bd9db:Xe".into(),
            },
        );
        assert!(matches!(refused, Err(OpsError::Refused(why)) if why == "nothing to stop"));
        let asked = echo.0.lock().unwrap();
        assert_eq!(asked[0].1, "repo_job_log");
        assert_eq!(asked[1].2, json!({"deploy": "76c3a0f4d39bd9db:Xe"}));
        assert!(matches!(
            forward(&ToolContext::new(), "service_status", &json!({})),
            Err(OpsError::NotConfigured)
        ));
    }
}
