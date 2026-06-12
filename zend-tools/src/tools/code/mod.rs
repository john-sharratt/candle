//! Code execution tools: `code_run`, `code_session_{open,exec,list,close}`.
//!
//! Run code in a subprocess interpreter.  Two modes:
//!
//! # `code_run` — one-shot execution
//!
//! Spawn an interpreter, execute a snippet, return stdout/stderr/exit_code,
//! then terminate.  Right for short scripts where you don't need persistent
//! state across calls.
//!
//! # `code_session_*` — persistent REPL
//!
//! Open a long-lived interpreter process with shared namespace state.  Each
//! `code_session_exec` call sends code to the running REPL and reads back the
//! result.  State (variables, imports, function definitions) persists across
//! calls within the same session.
//!
//! # Supported languages
//!
//! - **Python** — uses the embedded REPL defined by [`PYTHON_REPL`]; requires
//!   `python3` on PATH
//! - **JavaScript/Node** — uses the embedded REPL defined by [`NODE_REPL`];
//!   requires `node` on PATH
//!
//! # Communication protocol
//!
//! The REPL scripts use a length-prefixed stdin protocol:
//! 1. Orchestrator writes `<byte_count>\n` then the code bytes
//! 2. REPL executes, captures stdout/stderr via redirect
//! 3. REPL writes a JSON result line then the sentinel `__ZEND_DONE__\n`
//!
//! This avoids PTY complexity and makes the output boundary unambiguous.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `interpreter_not_found` | `python3` or `node` not found on PATH |
//! | `timeout` | Execution exceeded `timeout_sec` |
//! | `execution_failed` | Interpreter process died unexpectedly |
//! | `session_not_found` | Session ID not in registry |

use crate::ToolError;
use thiserror::Error;

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

#[derive(Debug, Error)]
pub enum CodeError {
    #[error("interpreter not found: {0}")]
    InterpreterNotFound(String),
    #[error("execution timed out")]
    Timeout,
    #[error("execution failed: {0}")]
    ExecutionFailed(String),
    #[error("session not found: {0}")]
    SessionNotFound(String),
}

impl ToolError for CodeError {
    fn code(&self) -> &'static str {
        match self {
            CodeError::InterpreterNotFound(_) => "interpreter_not_found",
            CodeError::Timeout => "timeout",
            CodeError::ExecutionFailed(_) => "execution_failed",
            CodeError::SessionNotFound(_) => "session_not_found",
        }
    }
}

pub fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

pub const PYTHON_REPL: &str = r#"import sys, json, traceback, io, contextlib
SENTINEL = '__ZEND_DONE__'
_ns = {}
while True:
    line = sys.stdin.readline()
    if not line: break
    try: n = int(line.strip())
    except ValueError: continue
    code = sys.stdin.read(n)
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(code, '<cell>', 'exec'), _ns)
        r = {'ok': True, 'stdout': out.getvalue(), 'stderr': err.getvalue()}
    except SystemExit as e:
        r = {'ok': False, 'error': 'SystemExit:'+str(e.code), 'stdout': out.getvalue(), 'stderr': err.getvalue()}
    except Exception:
        r = {'ok': False, 'error': traceback.format_exc(), 'stdout': out.getvalue(), 'stderr': err.getvalue()}
    sys.stdout.write(json.dumps(r)+'\n'+SENTINEL+'\n')
    sys.stdout.flush()
"#;

pub const NODE_REPL: &str = r#"const vm = require('vm');
const readline = require('readline');
const ctx = vm.createContext({require, console, process, Buffer, Math, JSON, Date, Array, Object, String, Number, Boolean, Error, Promise, Map, Set, setTimeout, clearTimeout, setInterval, clearInterval});
const SENTINEL = '__ZEND_DONE__';
let buf = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', d => {
    buf += d;
    while (true) {
        const nl = buf.indexOf('\n');
        if (nl < 0) break;
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        const n = parseInt(line, 10);
        if (isNaN(n)) continue;
        if (buf.length < n) { buf = line + '\n' + buf; break; }
        const code = buf.slice(0, n);
        buf = buf.slice(n);
        const logs = [];
        const origLog = console.log, origErr = console.error;
        console.log = (...a) => logs.push(a.map(String).join(' '));
        console.error = (...a) => logs.push('[err] '+a.map(String).join(' '));
        let r;
        try {
            vm.runInContext(code, ctx);
            r = {ok: true, stdout: logs.join('\n'), stderr: ''};
        } catch(e) {
            r = {ok: false, error: e.stack, stdout: logs.join('\n'), stderr: ''};
        }
        console.log = origLog; console.error = origErr;
        process.stdout.write(JSON.stringify(r)+'\n'+SENTINEL+'\n');
    }
});
"#;
