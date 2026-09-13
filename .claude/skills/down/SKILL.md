---
name: down
description: Stop this machine's inference daemon (zend on the gateway, npcd on the npcd box) gracefully, after recording its exact command line in docs/deployment.md so /up restarts it identically. The web gateway is left running unless asked for; cf-ddns is never stopped.
argument-hint: "[zend|npcd|web|all]"
disable-model-invocation: true
---

# /down — stop this machine's daemons

**Arguments:** `$ARGUMENTS`

`docs/deployment.md` is the description of the deployment — read it first. It names the
machines, what each runs, the launch lines, and how to tell things are healthy. The
repository rules in `CLAUDE.md` apply: files are read and edited with the file tools only,
never mask an exit status, never commit without permission.

## 1. Which machine, and what to stop

Find this machine's `192.168.0.x` address:

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like '192.168.0.*' } | Select-Object IPAddress, InterfaceAlias
```

Look it up in `docs/deployment.md` → *Machines*. A machine not listed runs nothing: say so
and stop.

What to stop comes from the arguments:

| Argument | Stops |
|---|---|
| *(none)* | this machine's inference daemons — **zend** on `.5`, **npcd** on `.6` |
| `zend` / `npcd` | that daemon only |
| `web` | the gateway only |
| `all` | everything this machine runs, gateway included |

**Never stop web unless the arguments name it** (`web` or `all`): it is the public front
door for every site, including the ones that do not depend on the daemon being stopped.
**Never stop cf-ddns**, whatever the arguments.

## 2. Record how each target was started — before touching it

```powershell
Get-CimInstance Win32_Process -Filter "Name='zend.exe' OR Name='npcd.exe' OR Name='web.exe'" |
  Select-Object ProcessId, Name, ExecutablePath, CommandLine, CreationDate | Format-List
```

For each target that is running, write its `ExecutablePath`, full `CommandLine` and
`CreationDate` into the conversation. Then compare the arguments with its row in
`docs/deployment.md` → *Launch lines*. If they differ in any way — a flag added, removed or
changed — update that row with `Edit`: the executable path relative to the repo root, the
arguments exactly as the process has them, and `Recorded` set to today's date with
"from the running process". This is what the next `/up` on this machine replays, so it must
be the process's line, not a tidied version of it.

A target that is not running: say so and skip it. Nothing to record.

## 3. Stop gracefully — Ctrl-C first

All three daemons handle Ctrl-C, and zend uses it to flush its substrate to disk (see
*Stopping* in the doc). A detached process has its own hidden console, so the signal is
delivered from a helper that attaches to that console — never from this shell, which would
detach its own. For each target, with `$targetPid` set to its `ProcessId`:

```powershell
$inner = @"
Add-Type -Namespace Con -Name Ctrl -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern bool FreeConsole();
[DllImport("kernel32.dll")] public static extern bool AttachConsole(uint pid);
[DllImport("kernel32.dll")] public static extern bool SetConsoleCtrlHandler(System.IntPtr h, bool add);
[DllImport("kernel32.dll")] public static extern bool GenerateConsoleCtrlEvent(uint ev, uint group);
'@
[void][Con.Ctrl]::FreeConsole()
if (-not [Con.Ctrl]::AttachConsole($targetPid)) { exit 2 }
[void][Con.Ctrl]::SetConsoleCtrlHandler([IntPtr]::Zero, `$true)
if (-not [Con.Ctrl]::GenerateConsoleCtrlEvent(0, 0)) { exit 3 }
exit 0
"@
$enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))
$s = Start-Process powershell.exe -ArgumentList '-NoProfile','-NonInteractive','-EncodedCommand',$enc -WindowStyle Hidden -Wait -PassThru
"sender exit=$($s.ExitCode)"
```

`sender exit=2` means the process has no console to attach to (it was started some other
way, e.g. from a shell that has since exited) — go straight to the forced stop below and
say so.

Then wait for it to exit:

| Service | Wait up to | Why |
|---|---|---|
| zend | 180 s | it demotes every hot turn to disk and fsyncs before exiting |
| npcd | 60 s | |
| web | 15 s | |

```powershell
$p = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
if ($p) { $exited = $p.WaitForExit(180000); "exited=$exited" } else { "exited=True" }
```

If zend is still running at the deadline, read the end of its newest
`target\services\zend-*.err.log` / `.out.log` (with `Read`): while it is still logging
drain or flush progress, keep waiting; if it has gone quiet, stop it forcibly.

**Forced stop, last resort:** `Stop-Process -Id <pid> -Force -Confirm:$false`. Say plainly
that it was forced — for zend that skipped the substrate flush, so the last turn may not be
on disk.

## 4. Confirm it is down

- The process is gone (`Get-CimInstance` again).
- Its port is free: `Get-NetTCPConnection -State Listen -LocalPort <port> -ErrorAction SilentlyContinue`
  returns nothing (8081 for zend/npcd, 80 for web).
- For zend: `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` is back near
  the idle floor (~0.5 GB).

## 5. Check what is still meant to be up

Run the rows of *Health checks* in `docs/deployment.md` that apply to what is still running
— on the gateway after `/down` alone, that is web, the public sites that do not depend on
zend (`tokera.com`, `battlecities.net`), and cf-ddns. The stopped daemon's own site answering
`503` through the gateway is expected, not a failure.

## 6. Report

One table:

| Service | Was running | Recorded line changed | Stop | Result |
|---|---|---|---|---|

- **Stop** — `Ctrl-C (Ns)` or `forced` (and why).
- Then the health results for what is still up, and anything unexpected.
- If `docs/deployment.md` changed, say so and show the changed row. It is **not committed**
  — propose a commit message and wait for the user's go-ahead.
