# code — code_run, code_session_{open,exec,list,close}

Run **JavaScript** on the embedded, pure-Rust [`boa_engine`](https://github.com/boa-dev/boa)
VM. No external interpreter, no subprocess, no `node` on PATH. The VM is
sandboxed by construction — no network or process access, and its only
filesystem is the context's file store through a `vfs` global — and runaway
scripts are bounded by loop / recursion limits. `run_js()` also refuses to
create a VM unless the context holds the `sandbox` capability, so running code
needs that grant even if the dispatch-level check in the registry were bypassed.

Because nothing a script does can reach past the file store, the sandbox is
granted in the overlay (Comprehensive) tools mode as well as Mutable, unlike
`exec` (programs on the host), which is Mutable-only.

## The `vfs` global

| Call | Returns |
|------|---------|
| `vfs.read(path)` | the file's text, or `null` when there is no such file |
| `vfs.write(path, text)` | the byte count written |
| `vfs.list(prefix)` | up to 200 paths under `prefix` |
| `require('./path')` | a CommonJS module from the store (`module.exports` / `exports`), resolved like Node's relative requires with `.js` optional; Node built-ins and npm modules are not available |

A script sees what the `file_*` tools see: the session overlay in Comprehensive,
the workspace on disk in Mutable. `eval(vfs.read('scratch/lib.js'))` loads a file
just written with `write`, so it can be tested in the same call. Protected paths
(`secrets/`) throw. When a session replays its history, `vfs.write` writes nothing.

## Files

| File | Tool | Description |
|------|------|-------------|
| `engine.rs` | — | `run_js()`: boa `Context` + `console` capture + runtime limits |
| `files.rs` | — | the `vfs` global over the file store |
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
| `not_permitted` | the context lacks the `sandbox` capability; no VM was created |

A thrown JS exception or a hit VM limit is **not** one of these error codes — the
call succeeds and reports the fault via `ok: false` / `exit_code: 1` and the
`error` / `stderr` fields.
