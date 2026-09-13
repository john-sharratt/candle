# Deployment

The live estate: which machine runs what, with which arguments, and how to tell it is
healthy. `/up` and `/down` (`.claude/skills/`) read this file and act on it, and update it
whenever what is actually running differs from what is written here. If this file and a
running process disagree, the process is the fact and this file is corrected.

Secrets are not in this file and never go in it: the gateway's sign-in config is
`web/secrets/auth.yaml` (gitignored, on the gateway only) and the DNS updater's Cloudflare
token is `D:\prog\cf-ddns\.env`.

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
| `192.168.0.6` | — | npcd | the checkout `/up` is run from; its mind is `C:\Users\johna\prog\mind` |

## Services

None of these is a Windows service. Each is built with cargo and started by running its
executable directly from `target\release\` (never `cargo run`), detached, with the repo root
as its working directory. cf-ddns is the exception — see below.

| Service | Machine | Listens | Build | Stop by default in `/down` |
|---|---|---|---|---|
| web | .5 | `0.0.0.0:80` (from `web/web.yaml`) | `cargo build --release -p web` | **no** — only `/down web` or `/down all` |
| zend | .5 | `192.168.0.5:8081` | `cargo build --release -p zend` | yes |
| npcd | .6 | `0.0.0.0:8081` | `cargo build --release -p npcd` | yes |
| cf-ddns | .5 | — (outbound only) | scheduled task, see below | **never** |

### Launch lines

The exact arguments each service runs with. `/down` records the live process's command line
here before stopping it; `/up` replays it unless told otherwise, and records any change.
Paths are relative to the repo root.

| Machine | Service | Command line | Recorded |
|---|---|---|---|
| .5 | web | `target\release\web.exe --config web/web.yaml` | 2026-09-13, from the running process |
| .5 | zend | `target\release\zend.exe D:\prog\candle --host 192.168.0.5 --port 8081 --skip-layer repo_map --skip-layer code_reading` | 2026-09-13, from past production runs (not yet confirmed by a `/down`) |
| .6 | npcd | `target\release\npcd.exe --bind 0.0.0.0:8081 --content web/content/npcd --mind C:/Users/johna/prog/mind --forget-conversations` | 2026-09-13, from the user (not yet confirmed by a `/down`) |

Notes on the arguments:

- **`--skip-layer`** (zend) keeps a layer in service but stops re-reading it from disk at
  boot — "the corpus is built". It is not `--disable-layer`, which removes the layer from
  retrieval. npcd has no such flag; pass `--skip-layer` only to a binary whose `--help`
  lists it.
- **`--mind`** (npcd) names the mind directory: a directory holding `projection.yaml`
  beside its content libraries (`personalities/`, `worlds/`, `responses/`, `moods/`).
  npcd refuses to start on a directory without `projection.yaml`, and without `--mind` it
  runs the bundled placeholder with no content. Its startup log names the directory it
  resolved: `projection schema: … (collections resolve under <dir>)`.
- **`--content`** (npcd) serves the console from `web/content/npcd` on disk instead of the
  compiled-in copy, so a console edit is live on a refresh.
- **Wiping flags** — `--forget-conversations` (npcd) and `--wipe-substrate` (zend) destroy
  state. `/up` replays one only when the recorded line has it (the last run wiped) or the
  user asks for a wipe in that run, and says so before launching. It never adds one on its
  own.
- **Every flag must exist in the binary.** `--forget-conversations` is not in this repo's
  `npcd` at `14eacff5` (2026-09-13) — it is either newer than that on `.6` or has since been
  removed. `/up` checks each recorded flag against `<exe> --help` and asks rather than
  dropping one silently, because clap refuses to start on an unknown flag.

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

## Health checks

Run by `/up` after everything is up, and by `/down` for whatever is still meant to be up.
Expected results, as measured 2026-09-13:

| Check | How | Healthy |
|---|---|---|
| zend on the LAN | `curl http://192.168.0.5:8081/v1/status` | `200` |
| npcd on the LAN | `curl http://192.168.0.6:8081/v1/status` | `200` |
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
- **`www.tokera.com` answered `200`, not `301`**, on 2026-09-13 — `web.yaml` redirects it,
  but the running `web.exe` was started 2026-09-10. A rebuild and restart of web should turn
  it into a `301`.
