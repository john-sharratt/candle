# code — code_run, code_session_{open,exec,list,close}

Run Python or Node.js code in a subprocess interpreter.

## Files

| File | Tool | Description |
|------|------|-------------|
| `run.rs` | `code_run` | One-shot execution; subprocess spawned and killed per call |
| `session_open.rs` | `code_session_open` | Open a persistent REPL subprocess |
| `session_exec.rs` | `code_session_exec` | Execute a snippet in the running REPL |
| `session_list.rs` | `code_session_list` | List open code sessions |
| `session_close.rs` | `code_session_close` | Kill the interpreter subprocess |
| `mod.rs` | — | `CodeError`; `PYTHON_REPL`; `NODE_REPL`; `now()` |

## One-shot vs persistent REPL

| Aspect | `code_run` | `code_session_*` |
|--------|-----------|-----------------|
| State across calls | None (fresh interpreter each time) | Shared namespace persists |
| Imports, variables | Lost after each call | Accumulate across execs |
| Startup cost | Per call | Once at open |
| Use when | Self-contained script | Iterative / stateful exploration |

## Communication protocol

The REPL processes (see `PYTHON_REPL` and `NODE_REPL` in `mod.rs`) use a
length-prefixed stdin protocol:

1. Orchestrator writes `<byte_count>\n` then the code bytes to the subprocess stdin
2. REPL executes the code, capturing stdout/stderr via redirect
3. REPL writes a JSON result line: `{"ok":true,"stdout":"...","stderr":"..."}` or
   `{"ok":false,"error":"...","stdout":"...","stderr":"..."}`
4. REPL writes the sentinel line `__ZEND_DONE__`

This protocol avoids PTY complexity and makes the output boundary unambiguous.
The sentinel is hard-coded in both REPL scripts and in the read loop in `session_exec.rs`.

## Supported languages

| Language | Requires on PATH |
|----------|-----------------|
| `python` | `python3` |
| `javascript` / `node` | `node` |

## Error codes

| Code | When |
|------|------|
| `interpreter_not_found` | `python3` or `node` not found on PATH |
| `timeout` | Execution exceeded `timeout_sec` |
| `execution_failed` | Interpreter process died unexpectedly |
| `session_not_found` | Session ID not in registry |
