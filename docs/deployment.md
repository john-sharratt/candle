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
                router ── DMZ ──► 192.168.0.5:80   web  (the gateway, by Host header;
                                     │                   cluster `tokera.com` on :9443)
                     ┌───────────────┴────────────────┐
                     ▼                                ▼
          192.168.0.5:8081  motherd            192.168.0.6:8081  (stopped)
          zend + npcd members, joined          bot.tokera.com's configured
          to the gateway's cluster             upstream; rebuilt later as a member
          (code.tokera.com, bot.tokera.com)
```

- **Cloudflare** proxies every public name and connects to the origin on port 80. The
  router puts `192.168.0.5` in its DMZ, so port 80 on the public IP is the gateway.
- **web** is the only process reachable from outside. It owns sign-in and forwards to the
  daemons over the LAN, so no daemon may bind `127.0.0.1` — loopback takes its site
  down. motherd shares the DMZ box with the gateway, so it binds the LAN address only,
  never `0.0.0.0`. npcd's box is not in the DMZ and binds `0.0.0.0`.
- **The cluster.** motherd's zend and npcd tenants join the gateway's cluster
  (`docs/replicated_substrate.md`): the gateway routes each name to a joined, ready
  member of its app over the member's socket, and to the route's `upstream` only while
  the app has no member (dumb mode). While every admitted member is still loading, the
  gateway answers with its own reconnecting page, which the consoles retry.
  npcd on `.6` was stopped on 2026-10-12: `bot.tokera.com` is motherd's npcd alone until
  `.6` is rebuilt onto this framework and joins.
- **cf-ddns** keeps the Cloudflare A records at the public IP, so a router reboot that
  changes the address heals on its own within one 30 s interval.

## Machines

A machine is identified by its `192.168.0.x` address
(`Get-NetIPAddress -AddressFamily IPv4`). Other interfaces (e.g. `wg0` 192.168.137.1) are
not identities.

| Address | Hostname | Runs | Repo |
|---|---|---|---|
| `192.168.0.5` | BlackWorld | web, motherd, keep, cf-ddns | `D:\prog\candle` |
| `192.168.0.6` | — | nothing — its standalone npcd was stopped on 2026-10-12, to be rebuilt as a member | the checkout `/up` is run from; its mind is `D:\prog\mind` |
| `192.168.0.237` | AlienWorld (the laptop) | keep, and under it `tokera-agent`: the cluster's standby agent. No motherd since 2026-10-25 | `C:\Users\johna\prog\candle`; the agent's own checkouts under `C:\Users\johna\prog\agent` |

### AlienWorld's standby agent

The laptop runs `tokera-agent` alone, under keep (`C:\Users\johna\prog\keep-local.yaml`),
on `motherd/agent.laptop.yaml`. That config names 2 slots and the agent's own checkouts,
`candle`, `battle-cities`, `battle-mind` and `journals`, cloned under
`C:\Users\johna\prog\agent`, named as blackworld's agent names them. A job fetches and
resets its checkout, so they are never the working trees under `C:\Users\johna\prog`.
The gateway's `standby_agents: ["alienworld"]` (`web/web.yaml`) makes it a spare: it is
given a job only when no other agent can run it, or when the job names it
(`repo_job_start {node: "alienworld"}`; `docs/replicated_substrate.md` §11.3). keep
runs from the scheduled task `\keep`, as on blackworld, with its supervisor at
`C:\Users\johna\AppData\Local\keep\supervise.ps1`. `register-task.ps1` beside it
registers the task, and is run once from an elevated session. motherd is not in
keep's config here, because a keep restart starts every listed service and a hold does
not survive one.

## Services

None of these is a Windows service. Each is built with cargo and started by running its
executable directly from `target\release\` (never `cargo run`), detached, with the repo root
as its working directory. cf-ddns is the exception — see below.

| Service | Machine | Listens | Build | Stop by default in `/down` |
|---|---|---|---|---|
| web | .5 | `0.0.0.0:80` (from `web/web.yaml`); `0.0.0.0:9443` tcp + udp, HTTPS with Let's Encrypt certificates (from `web/secrets/tls.yaml` — `rdp-tunnel-server` holds 443 and 8443) | `cargo build --release -p web` | **no** — only `/down web` or `/down all` |
| motherd | .5 | `192.168.0.5:8081`; `192.168.0.5:8444` tcp + udp, HTTPS self-signed at startup — the zend route's upstream in dumb mode; outbound to the gateway's cluster on `:9443` | `cargo build --release -p motherd` | yes |
| keep | .5 | — (watches web on `:80` and motherd on `:8081`) | `cargo build --release -p keep` | **never** |
| npcd | .6 | `0.0.0.0:8081` (`--host 0.0.0.0 --port 8081`) | `cargo build --release -p npcd` | yes |
| cf-ddns | .5 | — (outbound only) | scheduled task, see below | **never** |

### Launch lines

The exact arguments each service runs with. `/down` records the live process's command line
here before stopping it; `/up` replays it unless told otherwise, and records any change.
Paths are relative to the repo root.

| Machine | Service | Command line | Recorded |
|---|---|---|---|
| .5 | web | `keep\stable\web.exe --config web/web.yaml` | 2026-10-14, keep restarts a proven trial from this path |
| .5 | motherd | `keep\stable\motherd.exe --config motherd\motherd.yaml` | 2026-10-14, keep restarts a proven trial from this path — every setting is in `motherd/motherd.yaml` |
| .5 | keep | `target\release\keep.exe keep/keep.yaml` | 2026-10-14. Supervisor `C:\Users\johna\AppData\Local\keep\supervise.ps1` (restart 10 s after exit), run by the scheduled task `\keep` at boot since 2026-10-15. keep is not swapped by a rollout |
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
- **The consoles are compiled into the binaries, and the gateway serves them from disk**
  (`docs/shared_model_host.md` §3.4). Through the gateway (code.tokera.com,
  bot.tokera.com) a saved page is live on the next refresh with nothing rebuilt; a daemon
  reached directly serves the pages it was built with. There is no `--ui` and no
  `--content`: a recorded line carrying either must drop it.
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
  `NpcdArgs::ONE_SHOT`), and `motherd.yaml` cannot hold one: run the single binary with
  the flag once.
- **The process's flags are the same in every binary** (`host::HostArgs`): `--host`,
  `--port`, `--tls-port`, `--gateway`, `--local-signin`, `-v`. `--local-signin` is
  refused beside `--gateway`.
- **Every flag must exist in the binary.** `/up` checks each recorded flag against `<exe> --help` and asks rather than
  dropping one silently, because clap refuses to start on an unknown flag.

### motherd

`motherd` runs zend and npcd in one process on one model
(`docs/shared_model_host.md`), and on `.5` it is what runs in zend's place. The config,
`motherd/motherd.yaml`, holds the process's settings under `host:` and each daemon's
under its name, in the same shape as that daemon's own flags:

- **`host:`** — `192.168.0.5`, ports `8081` and `8444` (zend's, unchanged), and the
  cluster: `clusters: ["https://gateway.tokera.com:9443"]`, verified as `cluster_name:
  tokera.com`, with `group_key: web/secrets/group.key` and `agent: motherd/agent.yaml`.
- **`zend:`** — `working_dir: ..`, so the workspace is `D:\prog` and the substrate
  `D:\prog\substrate`, the one zend ran on before (adopted as the group's first member,
  `docs/replicated_substrate.md` §15.1).
- **`npcd:`** — `mind_repo: battle-mind`, `data: ../npcd-data`. Its substrate and
  accounts are under `D:\prog\npcd-data`, and its mind is a mirror at
  `D:\prog\npcd-data\mind\battle-mind`, whose `origin` is `agent:battle-mind`: it is
  fetched and pushed through an agent, never from GitHub directly. A start with no
  mirror runs with no mind, fetches it once joined, and relaunches onto it.

**The agent** runs in process (`motherd/agent.yaml`): `D:\agent\jobs` for jobs, 4
slots, and its own checkouts, cloned over SSH — `D:\agent\candle`,
`D:\agent\battle-cities`, `D:\agent\battle-mind` — each allowed to push. They are
apart from the working trees zend reads in `D:\prog`.

Its log is `motherd.log` beside zend's substrate, `D:\prog\substrate\motherd.log`; `/up`'s
redirected output is in `target\services\` as for the other services. Each start rotates
the previous run's log to `motherd.log.1` (up to `.4`), so a deploy whose trial fails and
is rolled back leaves the trial's fault in `.1` and the run it replaced in `.2`. The
gateway writes `web.log` beside its config, `D:\prog\candle\web\web.log`, the same way.

**Rolling back to the standalone zend**: stop motherd (Ctrl-C), then start zend with
the line the launch table records for it. The gateway needs no change — with no member
of an app it routes to the configured upstream.

### The cluster

The gateway on `.5` is the cluster `tokera.com` (`web/web.yaml`'s `cluster:` block):

- **The group key** is `web/secrets/group.key`, written once by
  `target\release\web.exe --new-group-key web/secrets/group.key` (it refuses to
  overwrite one). The gateway and every member on every machine hold the same file;
  it is a secret, gitignored with the rest of `web/secrets`, and never printed.
- **Members join over the TLS entrance**, `https://gateway.tokera.com:9443`, checked
  against the gateway's `tokera.com` certificate (`cluster_name: tokera.com`).
  `gateway.tokera.com` is a DNS-only record (not proxied) for the public IP, with
  9443 tcp + udp forwarded to `.5`; `tokera.com` itself is behind Cloudflare's proxy,
  which does not carry :9443.
- **The zend and npcd routes** carry `cluster: zend` and `cluster: npcd` and keep their
  `upstream:`. The gateway serves a name from the app's ready members while it has any,
  and from the upstream while it has none.
- **The status page** is `/gateway` on the gateway, its data `/gateway/status.json`
  (both an admin's; the data also answers `Authorization: Bearer <key>` with the key
  `web --new-status-key web/secrets/status.key` writes, named as the `cluster:` block's
  `status_key_file`): each app's mode,
  members, lag and claims, the build floor, and the gateway's peers.

### keep

`keep` is a separate process and a failsafe only. The last known-good `web` and
`motherd` are the files `keep/stable/web.exe` and `keep/stable/motherd.exe`. On startup
`keep` runs those files with the arguments in `keep/keep.yaml`. A switch is one POST to
`127.0.0.1:8092` whose body is the path of a staged executable (`keep/stage/`). `keep`
stops the running process, moves the stable file aside (`<name>.exe.prev`), copies the
staged file onto the stable path and starts it there. Windows firewall allows a
program by the path it was started from, and a self-heal relaunch starts the path it
is running from, so the trial runs from `keep/stable/web.exe` or
`keep/stable/motherd.exe` from its first moment; the staged file is only copied and
stays free for the next deploy. Its stderr goes to `keep/state/<name>.trial.log`,
read until the line `INFO serving` or the 10 minute limit; those lines are logged. The
limit restarts at every `INFO syncing …` line, which `motherd` prints each time a
tenant's cluster catch-up receives more: a member that is behind is waited out while
its sync keeps arriving, and rolled back 10 minutes after the sync stalls. A
file, not a pipe, so the process can go on writing to it after the proof, and after
`keep` itself restarts, with nothing draining it. `web` prints the line once it is
serving; `motherd` once every tenant has *proven* the build (`TenantStatus::proven`):
zend once section calibration is behind it — the model loaded, the substrate opened,
real forwards run — not once it is ready. What follows calibration, priming above all,
is as long as there is to read: after the ingest keys change it re-reads every
repository's anchors, an hour that tests nothing the forwards before it did not, and a
trial that waited for it ran out of time and rolled a working build back. **The line makes the trial the service: the
set-aside file is deleted and the process goes on running — a promotion does not
restart it.** No such line within 10 minutes, or the process dying, stops the trial,
puts the set-aside file back and starts it. A process `keep` starts at startup or on
rollback has stdout and stderr detached. A set-aside file exists only while a trial
is unproven, so a `keep` that starts and finds one stops whatever runs and puts it
back. It does not certify, roll out, or store binary bytes, and
it does no canary math. It also joins the gateway as the app `keep`, so `/gateway` lists
it beside zend and npcd; that socket declares no routes and takes no rollout. Its state
is `keep/state/`.

Its supervisor is `C:\Users\johna\AppData\Local\keep\supervise.ps1`, the same loop shape
as cf-ddns: it starts `target\release\keep.exe keep/keep.yaml` and starts it again 10 s
after any exit. The scheduled task **`\keep`** runs that supervisor — boot trigger, S4U
as `johna`, restart count 999, no time limit, normal priority (cf-ddns's below-normal
priority class would be inherited by `motherd`). So a reboot brings `keep` back, and
`keep` brings back `web` and `motherd`, with nobody logged on. Registering the task
needs an elevated session. Log: `C:\Users\johna\AppData\Local\keep\keep-<start>.log`,
a new file for every start with the newest five kept, and the supervisor's own lines in
`supervise.err.log` beside it. `web` and `motherd` inherit keep's handles and hold them
for as long as they run, which shapes both halves of the loop. keep is started with
`Process.Start` — no window, no pipe — and its output redirected by `cmd /c "… >> log
2>&1"`: run as a native command, cmd's output was a pipe to the script that the
services held open after keep exited, and the loop never restarted it. And every start
writes a file of its own: a log reused across starts is still open in the services, and
the next start's `cmd` cannot open it — keep then exits 1 on every attempt. Its
processes run in the task's logon session, so their start time and path read empty from
an interactive session; match them by name. A rollout does not swap `keep`.

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
"Services" above — with the exception of cf-ddns's and keep's scheduled-task supervisors (and
`web` and `motherd`, which keep restarts). That
was a real gap for zend: a device stuck in an out-of-memory retry storm (KV migration
failing forever) served nothing for 40+ minutes before anyone noticed, because nothing was
watching, and the old GPU-poison watchdog only exited cleanly "for a supervisor to relaunch"
— a supervisor that does not exist here.

Every daemon still relaunches itself on a poisoned GPU — that watchdog is the host's
(`host::self_heal`), so zend, npcd and motherd all carry it. `keep` is the outer failsafe
for `web` and `motherd`: if the process dies, `keep` starts the stable file again.
`candle::gpu_poison` flags the CUDA context poisoned
on a sticky fault (illegal address, launch failure, device assert, ECC) **or** on an
out-of-memory streak that has run for 60 s straight with no successful device operation in
between (`candle_core::gpu_poison::OOM_STICKY_AFTER`) — a single large-request OOM is not
sticky on its own and is left to admission control. On either, the watchdog:

1. logs the root fault (plus the recent-kernel-launch breadcrumb);
2. drops a marker next to the process's log (`<binary>.log` beside the highest-ranked
   tenant's substrate — `substrate/zend.log` for zend) so the relaunch **appends** instead
   of the ordinary start's rotation — the fault and the recovery read as one file;
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
uptime resets the count — it's a fresh occurrence, not a continuation of the same fault —
and so does any start that is not the watchdog's own relaunch: a process started by hand,
by an install or by a clean restart begins at zero whatever the file holds.

A binary built before the host (the npcd running on `.6` until it is rebuilt) has no
watchdog of its own, so the same class of stuck device still needs a human there.

## Health checks

Run by `/up` after everything is up, and by `/down` for whatever is still meant to be up.
Expected results, as measured 2026-09-13:

| Check | How | Healthy |
|---|---|---|
| motherd on the LAN | `curl http://192.168.0.5:8081/v1/status` | `"tenants":{"zend":{"state":"ready",…},"npcd":{"state":"ready",…}}` and `"cluster":{…"apps":{"zend":{"link":"joined",…},"npcd":{"link":"joined",…}}}` — it answers `200` with `"loading"` throughout startup, and npcd's first start on a fresh substrate ingests its mind for a while after zend is ready |
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
- **`bot.tokera.com` `503`** while the gateway is healthy means no npcd is serving: motherd's
  npcd member is not joined at all (a member joined and loading shows its loading page from
  the second request on), and the configured upstream on `.6` is down.
- **A console says "no zend member answered"** (or npcd) while the status page shows the
  member joined: the gateway asked the member and it did not answer within 3 s. Its
  forwarded requests are queued behind something on its socket — `"in_flight"` and
  `"unsent"` in the member's own `/v1/status` on `:8081` say how much. A restart no
  longer resends the member's history (`docs/replicated_substrate.md` §7.5), so a large
  figure there is new events, such as a heavy ingest.
