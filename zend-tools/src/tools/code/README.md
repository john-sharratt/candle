# code — code_run, code_session_{open,exec,list,close}

Run **JavaScript** on the embedded, pure-Rust [`boa_engine`](https://github.com/boa-dev/boa)
VM. No external interpreter, no subprocess, no `node` on PATH. The VM is
sandboxed by construction — no filesystem, network, or process access — and
runaway scripts are bounded by loop / recursion limits.

## Files

| File | Tool | Description |
|------|------|-------------|
| `engine.rs` | — | `run_js()`: boa `Context` + `console` capture + runtime limits |
| `run.rs` | `code_run` | One-shot execution in a fresh VM |
| `session_open.rs` | `code_session_open` | Open a persistent JS session |
| `session_exec.rs` | `code_session_exec` | Execute a snippet with prior state in scope |
| `session_list.rs` | `code_session_list` | List open code sessions |
| `session_close.rs` | `code_session_close` | Discard a session's state |
| `mod.rs` | — | `CodeError`; `is_javascript()`; `now()` |

## One-shot vs persistent session

| Aspect | `code_run` | `code_session_*` |
|--------|-----------|-----------------|
| State across calls | None (fresh VM each call) | `let`/`const`/`function` persist |
| Startup cost | Per call | Per call (history replayed) |
| Use when | Self-contained snippet | Iterative / stateful exploration |

## How session state works

A session has no live VM (a `boa` `Context` is not `Send`, and the session
registry is shared across threads). Instead it stores the **accumulated source**
of every successful `code_session_exec`. Each new exec spins up a fresh VM,
replays that history *silently* to rebuild variable / function state, then runs
the new snippet with output captured. A snippet that throws is **not** added to
the history, so it can't poison future replays.

Trade-off: purely stateful code rebuilds exactly, but non-deterministic prior
expressions (`Math.random()`, `Date.now()`) re-evaluate on each replay.

## Output

`console.log` / `console.info` / `console.debug` → `stdout`; `console.warn` /
`console.error` → `stderr`. Non-string arguments are `JSON.stringify`'d. The
value of the final expression is returned in `result`. `code_run` also reports
an `exit_code` (0 on success, 1 if the script throws); `code_session_exec`
reports an `ok` flag with the thrown message in `error`.

`code_run` exposes the request's `stdin` as the global `stdin` (a string) and
`env` as the global `env` (an object).

## Error codes

| Code | When |
|------|------|
| `interpreter_not_found` | a language other than JavaScript was requested |
| `execution_failed` | engine setup failed (should not occur) |
| `session_not_found` | session ID not in registry |

A thrown JS exception or a hit VM limit is **not** one of these error codes — the
call succeeds and reports the fault via `ok: false` / `exit_code: 1` and the
`error` / `stderr` fields.
