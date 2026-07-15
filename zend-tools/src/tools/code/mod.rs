//! Code execution tools: `code_run`, `code_session_{open,exec,list,close}`.
//!
//! Runs **JavaScript** on the embedded pure-Rust [`boa_engine`] VM — no external
//! interpreter, no subprocess. The VM is sandboxed by construction (no
//! filesystem / network / process access) and bounds runaway scripts with loop
//! and recursion limits. See [`engine`].
//!
//! # `code_run` — one-shot execution
//!
//! Evaluate a snippet in a fresh VM and return its console output, final value,
//! and success flag. Right for short scripts with no state across calls.
//!
//! # `code_session_*` — persistent REPL
//!
//! A session accumulates the source of every successful `code_session_exec`
//! call. Each subsequent exec replays that history in a fresh VM (silently) to
//! rebuild variable / function state, then runs the new snippet. State
//! (`let`/`const`/`function` bindings) therefore persists across calls without
//! keeping a live VM around — which matters because `boa_engine::Context` is not
//! `Send` and the session registry is shared across threads. The cost is that
//! non-deterministic prior expressions (`Math.random()`, `Date.now()`)
//! re-evaluate on replay; pure state rebuilds exactly.
//!
//! Because replay concatenates each snippet's source into one script, a
//! top-level `let`/`const` name may be declared **once** per session — a later
//! snippet that re-declares the same name (`let x = …` after an earlier
//! `let x = …`) fails with a redeclaration `SyntaxError`, exactly as it would in
//! a single script. Re-assign (`x = …`) to update a binding across calls, or use
//! a fresh name; this is a deliberate consequence of the state-rebuild model, not
//! a bug.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `interpreter_not_found` | requested a language other than JavaScript |
//! | `execution_failed` | engine setup failed (should not occur) |
//! | `session_not_found` | session ID not in registry |
//!
//! A thrown JS exception or a hit VM limit is **not** an error envelope: the
//! call succeeds with `ok: false` and the message in `error`, mirroring how a
//! REPL reports a runtime fault.

use crate::ToolError;
use thiserror::Error;

pub mod engine;
pub mod run;
pub mod session_close;
pub mod session_exec;
pub mod session_list;
pub mod session_open;

pub use run::CODE_RUN;
pub use session_close::CODE_SESSION_CLOSE;
pub use session_exec::CODE_SESSION_EXEC;
pub use session_list::CODE_SESSION_LIST;
pub use session_open::CODE_SESSION_OPEN;

/// Canonical language check: the code tools run JavaScript only. Accepts the
/// common aliases the model might use.
pub fn is_javascript(language: &str) -> bool {
    matches!(
        language.trim().to_ascii_lowercase().as_str(),
        "javascript" | "js" | "node" | "nodejs" | "ecmascript"
    )
}

#[derive(Debug, Error)]
pub enum CodeError {
    #[error("interpreter not found: {0}")]
    InterpreterNotFound(String),
    #[error("execution failed: {0}")]
    ExecutionFailed(String),
    #[error("session not found: {0}")]
    SessionNotFound(String),
}

impl ToolError for CodeError {
    fn code(&self) -> &'static str {
        match self {
            CodeError::InterpreterNotFound(_) => "interpreter_not_found",
            CodeError::ExecutionFailed(_) => "execution_failed",
            CodeError::SessionNotFound(_) => "session_not_found",
        }
    }
}

pub fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}
