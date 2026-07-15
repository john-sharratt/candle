//! code_run tool — one-shot JavaScript execution on the embedded boa VM.

use std::collections::HashMap;
use std::time::Instant;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::engine::run_js;
use super::{is_javascript, CodeError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct RunRequest {
    /// Language to run. Only JavaScript is supported (aliases: javascript, js,
    /// node). The field is kept explicit so the model states intent.
    #[validate(length(min = 1))]
    pub language: String,
    /// JavaScript source to execute. `console.log`/`console.error` output is
    /// captured; the value of the final expression is returned in `result`.
    #[validate(length(min = 1))]
    pub code: String,
    /// Optional string exposed to the script as the global `stdin`. Omit for none.
    pub stdin: Option<String>,
    /// Advisory wall-clock hint (1–300s). The VM cannot interrupt a synchronous
    /// eval, so runaway scripts are bounded by loop/recursion limits instead.
    #[validate(range(min = 1, max = 300))]
    pub timeout_sec: Option<u32>,
    /// Optional key/value map exposed to the script as the global object `env`.
    /// Omit for none.
    pub env: Option<HashMap<String, String>>,
}

#[derive(Serialize)]
pub struct RunResponse {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub duration_ms: u64,
    /// The final expression's value, stringified. Absent when the script's last
    /// statement produced `undefined`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
}

pub struct CodeRun;

impl Tool for CodeRun {
    const NAME: &'static str = "code_run";
    const DESCRIPTION: &'static str =
        "Execute a JavaScript snippet in an embedded, sandboxed engine (no filesystem, network, \
         or process access). Runs in-process on a pure-Rust VM — no Node or external interpreter \
         required. Use for arithmetic/logic the model would get wrong, data transformation, \
         string processing, JSON manipulation, and quick algorithms. `console.log` output is \
         returned in stdout; the final expression's value in result. Returns stdout, stderr, \
         exit_code (0 on success, 1 if the script throws), duration, and result.";

    type Request = RunRequest;
    type Response = RunResponse;
    type Error = CodeError;

    fn run(_ctx: &ToolContext, req: RunRequest) -> Result<RunResponse, CodeError> {
        if !is_javascript(&req.language) {
            return Err(CodeError::InterpreterNotFound(req.language));
        }

        // Expose stdin / env to the script as globals, injected as a silent
        // prelude (JSON is valid JS literal syntax).
        let mut prelude = String::new();
        if let Some(stdin) = &req.stdin {
            let lit = serde_json::to_string(stdin)
                .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?;
            prelude.push_str(&format!("globalThis.stdin = {lit};\n"));
        }
        if let Some(env) = &req.env {
            let lit = serde_json::to_string(env)
                .map_err(|e| CodeError::ExecutionFailed(e.to_string()))?;
            prelude.push_str(&format!("globalThis.env = {lit};\n"));
        }

        let start = Instant::now();
        let outcome = run_js(&prelude, &req.code);
        let duration_ms = start.elapsed().as_millis() as u64;

        // A thrown JS error is a script fault, not a tool error: report it via
        // exit_code / stderr, the way a shell would surface a non-zero exit.
        let (exit_code, stderr) = match outcome.error {
            None => (0, outcome.stderr),
            Some(err) => {
                let mut s = outcome.stderr;
                if !s.is_empty() && !s.ends_with('\n') {
                    s.push('\n');
                }
                s.push_str(&err);
                (1, s)
            }
        };

        Ok(RunResponse {
            stdout: outcome.stdout,
            stderr,
            exit_code,
            duration_ms,
            result: outcome.result,
        })
    }
}

pub const CODE_RUN: RegisteredTool = RegisteredTool::new::<CodeRun>();
