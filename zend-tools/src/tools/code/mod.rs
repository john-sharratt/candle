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
//! Replay evaluates each earlier snippet as its own script in one VM, in order,
//! so the snippets share one global scope: a top-level `let`/`const` name may be
//! declared **once** per session — a later snippet that re-declares the same name
//! (`let x = …` after an earlier `let x = …`) fails with a redeclaration
//! `SyntaxError`. Re-assign (`x = …`) to update a binding across calls, or use a
//! fresh name; this is a deliberate consequence of the state-rebuild model, not a
//! bug.
//!
//! Every exec replays the whole history, so a session holds at most
//! [`session_exec::MAX_SESSION_SNIPPETS`]; past that, open a new session.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `interpreter_not_found` | requested a language other than JavaScript |
//! | `execution_failed` | engine setup failed (should not occur) |
//! | `session_not_found` | session ID not in registry |
//! | `session_full` | the session already holds [`session_exec::MAX_SESSION_SNIPPETS`] snippets |
//! | `not_permitted` | the context lacks the `sandbox` capability; no VM was created |
//!
//! A thrown JS exception or a hit VM limit is **not** an error envelope: the
//! call succeeds with `ok: false` and the message in `error`, mirroring how a
//! REPL reports a runtime fault.

use crate::{NotPermitted, ToolError};
use thiserror::Error;

pub mod engine;
pub mod files;
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
    /// Worded as what can be done, not only what cannot: a model told just
    /// "not found" reported its untested code as passing "conceptually".
    #[error(
        "no {0} interpreter here — only JavaScript runs. To check logic written in \
         another language, port it to JavaScript and run that; otherwise say plainly \
         that the code has not been run"
    )]
    InterpreterNotFound(String),
    #[error("execution failed: {0}")]
    ExecutionFailed(String),
    #[error("session not found: {0}")]
    SessionNotFound(String),
    /// The session's history is at its cap; every exec replays all of it.
    #[error(
        "session {0} already holds {1} snippets, the most one session replays — open a \
         new session with code_session_open and define what you still need there"
    )]
    SessionFull(String, usize),
    #[error(transparent)]
    NotPermitted(#[from] NotPermitted),
}

impl ToolError for CodeError {
    fn code(&self) -> &'static str {
        match self {
            CodeError::InterpreterNotFound(_) => "interpreter_not_found",
            CodeError::ExecutionFailed(_) => "execution_failed",
            CodeError::SessionNotFound(_) => "session_not_found",
            CodeError::SessionFull(..) => "session_full",
            CodeError::NotPermitted(_) => NotPermitted::CODE,
        }
    }
}

pub fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}
