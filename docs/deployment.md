# Deployment

The live estate: which machine runs what, with which arguments, and how to tell it is
healthy. `/up` and `/down` (`.claude/skills/`) read this file and act on it, and update it
whenever what is actually running differs from what is written here. If this file and a
running process disagree, the process is the fact and this file is corrected.

Secrets are not in this file and never go in it: the gateway's sign-in config is
`web/secrets/auth.yaml` (gitignored, on the gateway only), zend's API keys and tokens are
`~/.zend/secrets.yaml` for the user running it (the Tavily key for `web_search` and a
GitHub token; `zend --secrets <path>` names another file), and the DNS updater's
Cloudflare token is `D:\prog\cf-ddns\.env`.

The `git_*` tools do **not** read that GitHub token. `git_fetch` and `git_push` run the
`git` program, which authenticates the way it always does for that remote — the
workspace's repositories use SSH remotes (`git@github.com:…`), so the key in `~/.ssh`
is what authorises a push, and nothing puts a credential on a command line or into a
URL. The token in the secrets file is there for a future GitHub API consumer; a
repository configured with an HTTPS remote and no credential helper would fail to push
rather than fall back to it, because the git layer sets `GIT_TERMINAL_PROMPT=0` and
never prompts.

zend's file sits outside every workspace, so no repository the `file_*` tools mount can
reach it — they follow symlinks and junctions and refuse any path that leads outside its
repository — and zend refuses to load it unless it is private to zend's user. Inside a
repository, a `secrets` path segment is load-bearing rather than tidy:
zend's `file_*` tools resolve a path straight to disk and never consult `.gitignore`, so
`VfsStore` refuses any path containing that segment. A secret kept anywhere else in a
repository is readable by the model.

## Topology

```
 Internet ──TLS──► Cloudflare (proxied names; terminates TLS)
                      │  HTTP :80 to the public IP
                      ▼
                router ── DMZ ──► 192.168.0.5:80   web  (the gateway, by Host header)
                                     │
                     ┌───────────────┴────────────────┐
                     ▼                                ▼
          192.168.0.5:8081  zend               192.168.0.6:8081  npcd
          (code.tokera.com)                    (bot.tokera.com)
```

- **Cloudflare** proxies every public name and connects to the origin on port 80. The
  router puts `192.168.0.5` in its DMZ, so port 80 on the public IP is the gateway.
- **web** is the only process reachable from outside. It owns sign-in and forwards to the
  two daemons over the LAN, so neither daemon may bind `127.0.0.1` — loopback takes its site
  down. zend shares the DMZ box with the gateway, so it binds the LAN address only, never
  `0.0.0.0`. npcd's box is not in the DMZ and binds `0.0.0.0`.
- **cf-ddns** keeps the Cloudflare A records at the public IP, so a router reboot that
  changes the address heals on its own within one 30 s interval.

## Machines

A machine is identified by its `192.168.0.x` address
(`Get-NetIPAddress -AddressFamily IPv4`). Other interfaces (e.g. `wg0` 192.168.137.1) are
not identities.

| Address | Hostname | Runs | Repo |
|---|---|---|---|
| `192.168.0.5` | BlackWorld | web, zend, cf-ddns | `D:\prog\candle` |
| `192.168.0.6` | — | npcd | the checkout `/up` is run from; its mind is `D:\prog\mind` |

## Services

None of these is a Windows service. Each is built with cargo and started by running its
executable directly from `target\release\` (never `cargo run`), detached, with the repo root
as its working directory. cf-ddns is the exception — see below.

| Service | Machine | Listens | Build | Stop by default in `/down` |
|---|---|---|---|---|
| web | .5 | `0.0.0.0:80` (from `web/web.yaml`); `0.0.0.0:9443` tcp + udp, HTTPS with Let's Encrypt certificates (from `web/secrets/tls.yaml` — `rdp-tunnel-server` holds 443 and 8443) | `cargo build --release -p web` | **no** — only `/down web` or `/down all` |
| zend | .5 | `192.168.0.5:8081`; `192.168.0.5:8444` tcp + udp, HTTPS self-signed at startup — the gateway's upstream | `cargo build --release -p zend` | yes |
| npcd | .6 | `0.0.0.0:8081` (`--host 0.0.0.0 --port 8081`) | `cargo build --release -p npcd` | yes |
| cf-ddns | .5 | — (outbound only) | scheduled task, see below | **never** |

### Launch lines

The exact arguments each service runs with. `/down` records the live process's command line
here before stopping it; `/up` replays it unless told otherwise, and records any change.
Paths are relative to the repo root.

| Machine | Service | Command line | Recorded |
|---|---|---|---|
| .5 | web | `target\release\web.exe --config web/web.yaml` | 2026-09-13, from the running process |
| .5 | zend | `target\release\zend.exe D:\prog --host 192.168.0.5 --port 8081 --tls-port 8444 --max-depth 2 -v` | 2026-10-11, first run on this build, under the prescribed Flash-Next `Q4_KO` — the workspace is `D:\prog`, whose `workspace.yaml` lists the repositories (`candle`, `battle-cities`); its substrate is `D:\prog\substrate`, the Flash-Next store it ran on before, admitted by the one-time model check (`docs/shared_model_host.md` §4.2) and adopted at that first open (`docs/replicated_substrate.md` §6.2). `repo_map` and `code_reading` ingest to two path components below each repository's root; what was ingested deeper is retained while a branch holds it; `--tls-port 8444` is the gateway's upstream (`self_signed: true` in `web/web.yaml`); `-v` (DEBUG) |
| .6 | npcd | `target\release\npcd.exe --host 0.0.0.0 --port 8081 --gateway 192.168.0.5 --mind D:/prog/mind --forget-conversations` | 2026-09-29, from /up — mind moved to `D:\prog\mind`, which also holds its `.substrate\` and `accounts\` (`--data` defaults to `--mind`). Rewritten 2026-10-10 for the host's flags (not yet run): `--bind` became `--host`/`--port`, `--content` is gone, and `--gateway 192.168.0.5` is new — without it every signed-in visitor is anonymous. Its store was written by the AntiLoop + StyleTune fine-tune; since 2026-10-11 the host prescribes Flash-Next, so that store is refused at open, before anything in it is rewritten, and npcd's first run on this build needs a new `--data` |

Notes on the arguments:

- **The workspace** (zend's positional argument) is a folder holding `workspace.yaml`, which
  lists the repositories — each a folder directly inside it (`docs/zend_workspace_execution.md`
  §3). A repository's own folder is not a workspace: zend refuses to start without the
  manifest, before the model loads.

- **`--disable-layer`** (zend) removes a layer from retrieval. zend reads no layer from the
  repositories' folders at boot — they belong to the sandbox's jobs — so there is no flag
  for "stop re-reading the disk"; a recorded line carrying `--skip-layer` must drop it, as
  zend no longer accepts it.
- **`--mind`** (npcd) names the mind directory: a directory holding `projection.yaml`
  beside its content libraries (`personalities/`, `worlds/`, `responses/`, `moods/`).
  npcd refuses to start on a directory without `projection.yaml`, and without `--mind` it
  runs the bundled placeholder with no content. Its startup log names the directory it
  resolved: `projection schema: … (collections resolve under <dir>)`.
- **The consoles are served from disk, always** (`docs/shared_model_host.md` §3.4): a
  saved file is served on the next request and an open page reloads itself on its next
  status poll. npcd's `--content` is gone with its compiled-in copy — a recorded npcd
  line carrying it must drop it — and the directory is `--ui` (default
  `web/content/npcd`; zend's, `zend/web`).
- **`--gateway <ip>`** (zend and npcd) names a gateway whose identity headers are
  believed; loopback and the bound address always are. npcd believed them from anyone
  before; it now clears them from any other peer, so an npcd behind the gateway on
  another machine needs `--gateway 192.168.0.5` to see who is signed in.
- **`--model`** (zend) is gone: every daemon runs the prescribed model (`host::MODEL`).
  A substrate written under another model refuses to open, naming both — start on a new
  `--working-dir`, or move the old substrate aside.
- **Wiping flags** — `--forget-conversations` (npcd) and `--wipe-substrate` (zend) destroy
  state. `/up` replays one only when the recorded line has it (the last run wiped) or the
  user asks for a wipe in that run, and says so before launching. It never adds one on its
  own. A self-heal relaunch never replays one (`ZendArgs::ONE_SHOT`,
  `NpcdArgs::ONE_SHOT`), and `tokerad.yaml` cannot hold one: run the single binary with
  the flag once.
- **The process's flags are the same in every binary** (`host::HostArgs`): `--host`,
  `--port`, `--tls-port`, `--gateway`, `--local-signin`, `-v`. `--local-signin` is
  refused beside `--gateway`.
- **Every flag must exist in the binary.** `/up` checks each recorded flag against `<exe> --help` and asks rather than
  dropping one silently, because clap refuses to start on an unknown flag.

### tokerad (not deployed)

`tokerad` runs zend and npcd in one process on one model
(`docs/shared_model_host.md`): `cargo build --release -p tokerad`, then
`target\release\tokerad.exe --config tokerad\tokerad.yaml` from the repo root. The
config holds the process's settings under `host:` and each daemon's under its name, in
the same shape as that daemon's own flags. It serves both on zend's ports; the gateway
then points `bot.tokera.com` at `https://192.168.0.5:8444` beside `code.tokera.com`,
and the host picks each UI by name. Its log is `tokerad.log` beside zend's substrate.

### cf-ddns

`D:\prog\cf-ddns` (its own repo). Runs as the scheduled task **`\cf-ddns`** — boot trigger,
S4U as `johna`, restart count 999 — whose action is a PowerShell supervisor loop that runs
`D:\prog\cf-ddns\target\release\cf-ddns.exe` and restarts it 10 s after any exit. Log:
`C:\Users\johna\AppData\Local\cf-ddns\cf-ddns.log` (rotated at 5 MB).

It manages the records named on its startup log line — as of 2026-09-13:
`tokera.com, www.tokera.com, remote.tokera.com`, every 30 s. The other names
(`code.`, `bot.`, `battlecities.net`) are not in that list.

Healthy means: task state `Running` **and** a `cf-ddns` process exists. The process's
`ExecutablePath` reads empty from an interactive session (it runs in another logon
session), so match it by name. `/up` starts the task if it is not running
(`Start-ScheduledTask -TaskName cf-ddns`); `/down` never stops it.

## Logs

`/up` redirects each service's output to `target\services\<service>-<yyyyMMdd-HHmmss>.{out,err}.log`
under the repo. `target\` is gitignored; the newest pair is the current run.

## Stopping

All three daemons handle Ctrl-C: web and npcd stop serving, and zend drains in-flight work
and then **flushes the substrate** (demotes every hot turn to disk and fsyncs the redo log)
before exiting. So a stop is Ctrl-C first, delivered to the detached process's console, and
a forced kill only if it does not exit in time. A forced kill of zend skips the flush.

## Self-healing

None of these daemons runs under a service manager or a restart-on-exit wrapper — see
"Services" above — with the single exception of cf-ddns's scheduled-task supervisor. That
was a real gap for zend: a device stuck in an out-of-memory retry storm (KV migration
failing forever) served nothing for 40+ minutes before anyone noticed, because nothing was
watching, and the old GPU-poison watchdog only exited cleanly "for a supervisor to relaunch"
— a supervisor that does not exist here.

Every daemon now relaunches itself instead — the watchdog is the host's (`host::self_heal`),
so zend, npcd and tokerad all carry it. `candle::gpu_poison` flags the CUDA context poisoned
on a sticky fault (illegal address, launch failure, device assert, ECC) **or** on an
out-of-memory streak that has run for 60 s straight with no successful device operation in
between (`candle_core::gpu_poison::OOM_STICKY_AFTER`) — a single large-request OOM is not
sticky on its own and is left to admission control. On either, the watchdog:

1. logs the root fault (plus the recent-kernel-launch breadcrumb);
2. drops a marker next to the process's log (`<binary>.log` beside the highest-ranked
   tenant's substrate — `substrate/zend.log` for zend) so the relaunch **appends** instead
   of the ordinary fresh-boot truncate — the whole point is to keep the evidence readable
   across the restart, in one file;
3. spawns an identical process — same executable, same argv (captured from its own launch,
   not reconstructed from parsed flags), same working directory;
4. exits with code 75 (distinct from clean shutdown 0 and Ctrl-C 130).

The new process's own bind retries for ~5 s if the port is still held by the one it is
replacing, and its log-file init logs an explicit `=== resumed after a self-heal restart ===`
line so the boundary is visible when reading the file. The substrate redo log is crash-safe,
so the abrupt exit loses nothing durable.

**Relaunching is capped at 5 consecutive fast poisonings** (one within 5 minutes of the
relaunched process's own start) — `host::self_heal`'s `relaunch_decision`, count
persisted in `.self_heal_attempts` beside the log. Past the cap it stays down and exits 76
instead of relaunching again: a genuinely broken card or driver poisons every fresh process
within moments, and an uncapped watchdog would crash-loop on that forever rather than
surfacing "a human needs to look at this machine." A poisoning after a healthy multi-minute
uptime resets the count — it's a fresh occurrence, not a continuation of the same fault.

A binary built before the host (the npcd running on `.6` until it is rebuilt) has no
watchdog of its own, so the same class of stuck device still needs a human there.

## Health checks

Run by `/up` after everything is up, and by `/down` for whatever is still meant to be up.
Expected results, as measured 2026-09-13:

| Check | How | Healthy |
|---|---|---|
| zend on the LAN | `curl http://192.168.0.5:8081/v1/status` | `"state":"ready"` — it answers `200` with `"state":"loading"` throughout startup; a host-built zend also carries `"tenants":{"zend":{"state":"ready",…}}` |
| npcd on the LAN | `curl http://192.168.0.6:8081/v1/status` | `200`; a host-built npcd answers with `"tenants":{"npcd":…}` |
| gateway config | `target\release\web.exe --config web/web.yaml --check` | exit 0 |
| gateway routing | `curl -H "Host: <name>" http://127.0.0.1/` for `tokera.com`, `code.tokera.com`, `bot.tokera.com` | `200` / `30x` / `401`; `503` = that upstream is down |
| from the internet | `curl https://<name>/` for `tokera.com`, `www.tokera.com`, `battlecities.net`, `code.tokera.com`, `bot.tokera.com` | not `5xx`, not `000` |
| DNS vs public IP | `curl https://api.ipify.org` vs `Resolve-DnsName remote.tokera.com -Server 1.1.1.1 -DnsOnly` | equal |
| cf-ddns | task `Running` + a `cf-ddns` process; no error lines at the end of its log | yes |

Reading a failure:

- **Proxied names cannot be compared to the public IP** — they resolve to Cloudflare's
  addresses (`104.21.*`, `172.67.*`). `remote.tokera.com` is DNS-only (grey cloud) and
  resolves to the origin, so it is the one record that shows whether the IP is right.
- **Cloudflare `521` / `522` / `523`** with the gateway healthy locally means Cloudflare
  cannot reach port 80 on the public IP: a stale A record (check DNS vs public IP, then
  cf-ddns), the router's DMZ no longer pointing at `192.168.0.5` (e.g. DHCP handed this box
  a new address), or the firewall.
- **`503` from the gateway** is web's own page for an upstream that is down — web backs off
  and recovers on its own once the daemon answers.
- **`www.tokera.com` answers `301`** (to `tokera.com`, per `web.yaml`) — confirmed
  2026-09-16 on a freshly built `web.exe`. A `200` there means the running gateway predates
  the redirect and wants a rebuild and restart.
- **`bot.tokera.com` `503`** while the gateway is healthy means npcd on `.6` is down — it is
  started by `/up` on that machine, not on `.5`.
