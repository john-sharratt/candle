---
name: up
description: Build and start every service this machine runs (web + zend on the gateway, npcd on the npcd box) with the arguments recorded in docs/deployment.md, make sure cf-ddns is running, then health-check the LAN, the gateway, the public sites through Cloudflare, and DNS against the public IP.
argument-hint: "[web|zend|npcd] [overrides, e.g. \"zend --model Qwen35_0_8B_Q8\"]"
disable-model-invocation: true
---

# /up — bring this machine's services up

**Arguments:** `$ARGUMENTS`

`docs/deployment.md` is the description of the deployment — read it first. It names the
machines, what each runs, the launch lines to replay, and what healthy looks like. The
repository rules in `CLAUDE.md` apply: files are read and edited with the file tools only,
never mask an exit status, never commit without permission.

## 1. Which machine, and what to start

Find this machine's `192.168.0.x` address and look it up in `docs/deployment.md` →
*Machines*:

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like '192.168.0.*' } | Select-Object IPAddress, InterfaceAlias
git rev-parse --show-toplevel
```

A machine not listed runs nothing: say so and stop. Everything below runs from the repo
root (`git rev-parse --show-toplevel`).

- **No arguments** — every service the doc lists for this machine.
- **Service names** (`web`, `zend`, `npcd`) — only those.
- **Overrides** — anything else in the arguments changes a launch line for this run, e.g.
  `zend --model Qwen35_0_8B_Q8` adds a flag, "zend without --skip-layer" removes one,
  "npcd wipe" adds `--forget-conversations`. Apply
  them to that service's recorded line. Without overrides the recorded line is used exactly.

## 2. What is already running

```powershell
Get-CimInstance Win32_Process -Filter "Name='zend.exe' OR Name='npcd.exe' OR Name='web.exe'" |
  Select-Object ProcessId, Name, ExecutablePath, CommandLine, CreationDate | Format-List
```

A service that is already running is **left running** — `/up` does not restart anything.
Report it with its start time, and if its command line differs from the doc's row (or from
the overrides asked for), say so and point at `/down <service>` then `/up` to restart it.
Do not build a service that is running: the running executable is locked, so a build that
needs to relink it fails.

## 3. Build — cargo, one package per invocation

For each service to start (on `.5` also make sure nothing else is using the card for
zend — `nvidia-smi`):

```bash
cargo build --release -p <service> > <scratchpad>/up/build-<service>.log 2>&1; echo "EXIT=$?"
```

Run it in the background; zend's build can take minutes. A non-zero exit is a build
failure: read the log (`Read`), report the first error, and do not start that service —
never fall back to an older binary.

## 4. Resolve the launch line

Take the service's row from *Launch lines* and apply the overrides. Then:

- **zend / npcd `--skip-layer`** — if the recorded line has it, keep it (these layers were
  skipped on purpose). Pass it only to a binary whose `--help` lists it.
- **npcd `--mind`** — use the recorded directory. If the doc has no directory yet (or it no
  longer exists), find it: `Glob` for `projection.yaml` in the repo's parent directory
  (`../*/projection.yaml`, `../*/*/projection.yaml`) and inside the repo, excluding
  `target/` and anything under `zend/` or `npcd/src/`. A mind directory holds
  `projection.yaml` **beside** `personalities/` and `worlds/` (check with `Glob`); zend's
  working dirs (e.g. `.mind-test`) hold a `projection.yaml` without them and are not minds.
  One candidate → use it. Several, or none → ask the user with `AskUserQuestion`.
- **Addresses** — as recorded. Never `127.0.0.1` for a daemon (the gateway reaches it over
  the LAN), and never `0.0.0.0` for zend on the DMZ box (see *Topology* in the doc).
- **Wiping flags** (`--forget-conversations`, `--wipe-substrate`) — replayed only when the
  recorded line has them (the last run wiped) or this run's arguments ask for a wipe; never
  added otherwise. When one will be passed, say so in the conversation before starting.
- **Every flag must exist** — check each flag of the resolved line against
  `target\release\<service>.exe --help`. A flag the binary does not know makes it refuse to
  start, and the recorded line may come from a newer checkout: do not drop it silently — ask
  the user (`AskUserQuestion`) whether to drop it for this run or stop.

For web, first check the config resolves without binding anything:

```powershell
& "<repo>\target\release\web.exe" --config web/web.yaml --check; "EXIT=$LASTEXITCODE"
```

## 5. Start — the executable directly, detached

Never `cargo run`. Start the built executable with the repo root as its working directory,
hidden, output to a timestamped log pair:

```powershell
$repo = (git rev-parse --show-toplevel) -replace '/', '\'
$stamp = Get-Date -Format yyyyMMdd-HHmmss
New-Item -ItemType Directory -Force "$repo\target\services" | Out-Null
$p = Start-Process -FilePath "$repo\target\release\<service>.exe" -ArgumentList @(<args, one string each>) `
  -WorkingDirectory $repo -WindowStyle Hidden `
  -RedirectStandardOutput "$repo\target\services\<service>-$stamp.out.log" `
  -RedirectStandardError  "$repo\target\services\<service>-$stamp.err.log" -PassThru
Start-Sleep -Seconds 5
"pid=$($p.Id) exited=$($p.HasExited)"
```

(`-WindowStyle Hidden` still gives it a console of its own, which is what lets `/down`
deliver Ctrl-C to it later. Never `-NoNewWindow`.)

**If Windows blocks it** — `Start-Process` throws (blocked by policy / Application Control /
"Access is denied" / "contains a virus"), or the process has exited within those seconds with
empty logs — build again and run again:

1. `cargo build --release -p <service>` again, then start again.
2. Still blocked and cargo had nothing to rebuild → force a fresh link with
   `cargo clean --release -p <service>`, build, start.
3. Still blocked after that → stop and report exactly what Windows said.

A process that exits with output in its logs is not a block — it is a failure: read the
`.err.log`/`.out.log` (`Read`) and report the cause.

## 6. Wait until each service answers

Poll (one command, bounded — not a sleep per call):

| Service | Ready when | Give it |
|---|---|---|
| web | `curl -s -o NUL -w "%{http_code}" -H "Host: tokera.com" http://127.0.0.1/` → `200` | 30 s |
| zend | `curl -s -o NUL -w "%{http_code}" http://192.168.0.5:8081/v1/status` → `200` | 10 min (model load) |
| npcd | `curl -s -o NUL -w "%{http_code}" http://192.168.0.6:8081/v1/status` → `200` | 2 min |

```powershell
$deadline = (Get-Date).AddMinutes(<n>)
do { $code = curl.exe -s -o NUL --max-time 5 -w "%{http_code}" <url>; if ($code -eq '200') { break }
     if ($p.HasExited) { "EXITED code=$($p.ExitCode)"; break }; Start-Sleep -Seconds 5 } while ((Get-Date) -lt $deadline)
"ready=$code"
```

If the process exits while waiting, read its logs and report. For npcd, confirm from its log
that it resolved the intended mind: `projection schema: … (collections resolve under <dir>)`.

## 7. cf-ddns — on the machine that has it

If the scheduled task `\cf-ddns` exists here (`Get-ScheduledTask -TaskName cf-ddns`):

```powershell
(Get-ScheduledTask -TaskName cf-ddns).State
Get-Process -Name cf-ddns -ErrorAction SilentlyContinue | Select-Object Id, StartTime
```

Healthy is `Running` **and** a `cf-ddns` process. Otherwise `Start-ScheduledTask -TaskName cf-ddns`
and check again after ~15 s. Read the end of its log
(`C:\Users\johna\AppData\Local\cf-ddns\cf-ddns.log`, with `Read`) for error lines.

## 8. Health checks

Run every row of *Health checks* in `docs/deployment.md` — the LAN, the gateway's routing,
the public sites **from the internet** (through Cloudflare), and **DNS against the public
IP** — for the whole estate, not only what this run started (from `.6`, check the gateway's
sites too: they are how npcd is reached).

For each failure, work it with *Reading a failure* in the doc before reporting:

- DNS ≠ public IP → is cf-ddns running and logging errors? Its interval is 30 s: after
  fixing, re-check after a minute.
- Cloudflare `52x` while the gateway answers locally → DNS first, then whether this box is
  still `192.168.0.5` (the router's DMZ points at that address), then the firewall for port
  80.
- `503` through the gateway for one site → that site's daemon is down (it may be on the
  other machine — say which).

## 9. Record and report

If a service was started with a line that differs from its *Launch lines* row (overrides, a
newly found `--mind`), update the row with `Edit` — the line as launched, `Recorded` = today,
"from /up". Update any other fact in `docs/deployment.md` this run found to be wrong (an
address, a record list, a health result that changed).

Then one table:

| Service | Action | Command line | PID | Ready after | Log |
|---|---|---|---|---|---|

- **Action** — `started`, `already running (since …)`, `build failed`, `blocked → rebuilt → started`, …
- Then the health-check table (check → result → healthy?), and every failure with what it
  means.
- If `docs/deployment.md` changed, show the changed rows. It is **not committed** — propose a
  commit message and wait for the user's go-ahead.
