# ssh — ssh_session_{open,exec,exec_async,poll,list,close}

Persistent SSH shell sessions backed by `ssh2` (libssh2 bindings).

## Files

| File | Tool | Description |
|------|------|-------------|
| `open.rs` | `ssh_session_open` | Connect, authenticate, open shell channel |
| `exec.rs` | `ssh_session_exec` | Run command synchronously; return full output |
| `exec_async.rs` | `ssh_session_exec_async` | Start command; return `process_id` immediately |
| `poll.rs` | `ssh_session_poll` | Read output chunks; optionally send signal |
| `list.rs` | `ssh_session_list` | Enumerate sessions and processes |
| `close.rs` | `ssh_session_close` | Graceful disconnect |
| `mod.rs` | — | `SshError` enum; `exec_simple` helper; `MAX_OUTPUT` constant |

## How sync exec works

`ssh_session_exec` uses a sentinel-and-nonce protocol to capture output without
a PTY:

```
<command>
echo "__CMD_DONE_<nonce>__:$?"
echo "__PWD_<nonce>__:$(pwd)"
```

The orchestrator reads stdout until both sentinel lines appear, extracts the exit
code and post-command cwd, and returns.  Per-command nonces are generated with
the OS RNG; they prevent crafted command output from injecting fake sentinels.

## Output limits

`MAX_OUTPUT = 32 KiB` per stream (stdout and stderr each).  Output beyond the
cap is dropped; `stdout_truncated` / `stderr_truncated` flags are set.

## Host key verification

Trust-on-first-use (TOFU): the first successful open records the host fingerprint.
Subsequent opens to the same host:port must match that fingerprint.  A mismatch
returns `host_key_mismatch` rather than connecting.

## Error codes

| Code | When |
|------|------|
| `connection_failed` | TCP connect or SSH handshake error |
| `auth_failed` | Wrong credentials or no auth method succeeded |
| `credential_not_found` | Named credential absent from store |
| `session_not_found` | Session/process ID not in registry |
| `session_dead` | Connection lost or idle-timed-out |
| `session_busy` | Concurrent synchronous exec attempted |
| `process_not_found` | `process_id` not in registry |
| `timeout` | Command exceeded `timeout_sec` |
| `session_limit_exceeded` | 5-session-per-user cap |
| `denied_by_user` | User rejected confirmation prompt |
| `concurrency_cap_exceeded` | 4 async commands already running |

## Confirmation policy

| Tool | Confirms |
|------|----------|
| `ssh_session_open` | Once (shows host + credential name) |
| `ssh_session_exec` | Every call (shows exact command) |
| `ssh_session_exec_async` | Every call (shows command + timeout) |
| `ssh_session_poll` | Only when `signal` is provided |
| `ssh_session_list` | Never |
| `ssh_session_close` | Never |
