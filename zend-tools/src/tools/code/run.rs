//! code_run tool.

use std::collections::HashMap;
use std::io::Write as _;
use std::time::Instant;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::CodeError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct RunRequest {
    #[validate(length(min = 1))]
    pub language: String,
    #[validate(length(min = 1))]
    pub code: String,
    pub stdin: Option<String>,
    #[validate(range(min = 1, max = 300))]
    pub timeout_sec: Option<u32>,
    pub env: Option<HashMap<String, String>>,
}

#[derive(Serialize)]
pub struct RunResponse {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub duration_ms: u64,
}

pub struct CodeRun;

impl Tool for CodeRun {
    const NAME: &'static str = "code_run";
    const DESCRIPTION: &'static str =
        "Execute code directly on the host system. Supports python, javascript (node), \
         bash, sh. WARNING: runs without sandbox in current implementation. \
         Returns stdout, stderr, exit code, and duration.";

    type Request = RunRequest;
    type Response = RunResponse;
    type Error = CodeError;

    fn run(_ctx: &ToolContext, req: RunRequest) -> Result<RunResponse, CodeError> {
        let timeout = req.timeout_sec.unwrap_or(30);

        let (mut cmd, temp_file) = match req.language.as_str() {
            "python" | "python3" => {
                let tmp =
                    std::env::temp_dir().join(format!("zend_code_{}.py", uuid::Uuid::new_v4()));
                std::fs::write(&tmp, &req.code)
                    .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?;
                let mut c = std::process::Command::new(if cfg!(windows) {
                    "python.exe"
                } else {
                    "python3"
                });
                c.arg(&tmp);
                (c, Some(tmp))
            }
            "javascript" | "js" | "node" => {
                let tmp =
                    std::env::temp_dir().join(format!("zend_code_{}.js", uuid::Uuid::new_v4()));
                std::fs::write(&tmp, &req.code)
                    .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?;
                let mut c =
                    std::process::Command::new(if cfg!(windows) { "node.exe" } else { "node" });
                c.arg(&tmp);
                (c, Some(tmp))
            }
            "bash" => {
                let mut c = if cfg!(windows) {
                    let mut c = std::process::Command::new("cmd");
                    c.args(["/C", "bash", "-c"]);
                    c
                } else {
                    let mut c = std::process::Command::new("bash");
                    c.arg("-c");
                    c
                };
                c.arg(&req.code);
                (c, None)
            }
            "sh" => {
                let mut c = if cfg!(windows) {
                    let mut c = std::process::Command::new("cmd");
                    c.args(["/C", "sh", "-c"]);
                    c
                } else {
                    let mut c = std::process::Command::new("sh");
                    c.arg("-c");
                    c
                };
                c.arg(&req.code);
                (c, None)
            }
            other => return Err(CodeError::InterpreterNotFound(other.to_string())),
        };

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        if req.stdin.is_some() {
            cmd.stdin(std::process::Stdio::piped());
        }

        if let Some(env) = &req.env {
            for (k, v) in env {
                cmd.env(k, v);
            }
        }

        let start = Instant::now();
        let mut child = cmd.spawn().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                CodeError::InterpreterNotFound(req.language.clone())
            } else {
                CodeError::ExecutionFailed(e.to_string())
            }
        })?;

        if let (Some(stdin_data), Some(mut child_stdin)) = (&req.stdin, child.stdin.take()) {
            let _ = child_stdin.write_all(stdin_data.as_bytes());
        }

        let output = loop {
            if start.elapsed().as_secs() >= timeout as u64 {
                let _ = child.kill();
                if let Some(f) = temp_file {
                    let _ = std::fs::remove_file(f);
                }
                return Err(CodeError::Timeout);
            }
            match child.try_wait() {
                Ok(Some(_)) => {
                    break child
                        .wait_with_output()
                        .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?
                }
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(50)),
                Err(e) => return Err(CodeError::ExecutionFailed(e.to_string())),
            }
        };

        if let Some(f) = temp_file {
            let _ = std::fs::remove_file(f);
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        Ok(RunResponse {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code().unwrap_or(-1),
            duration_ms,
        })
    }
}

pub const CODE_RUN: RegisteredTool = RegisteredTool::new::<CodeRun>();
