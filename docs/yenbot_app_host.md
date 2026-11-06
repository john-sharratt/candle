# Yen Bots: an app host for Yen

**Status: design, not built.** This document is authoritative for the yenbot
service (the site **Yen Bots**), its repository, its place behind the gateway
and the changes the gateway, zend, npcd and tokera.com need for it.

## 1. What it is

**Yen Bots**, at `yenbot.com.au`, is where Yen makes apps and shares them: a
habit tracker, a recipe book, a planner for the kids, a gallery, a reminder
that pings a phone, a tool that asks the model something or draws a picture.
People sign in with Google and get an account on the site; they see the apps
Yen has made as icons on one page and tap one to open it. Yen is the site's
admin: she decides which apps everyone sees and which stay private.

She does not write code. She works in claude.ai, which reaches our machines
through the Zen MCP connector, from wherever she is. She describes what she
wants, Claude builds it in the yenbot repository through Zen, and it is live
on her site seconds later. So the platform has to make the right thing the
easy thing for a model writing code for someone who cannot review it: every
app is built on one framework, from one design system and one library of
art, against a guide read for every new app, and checked — by eye as well as
by rule — before it goes live.

## 2. Principles, and what each one decides

| Principle | Decision |
|---|---|
| HTML changes go live quickly | Apps are files the binary serves from disk and reloads on change (§5.3), and the platform's own web files — framework, design system, landing page — are served from disk by the gateway, as zend's and npcd's consoles are (§5.2). An edit is one `file_edit_and_commit` from claude.ai, live within a second (§5.4). Only Rust changes rebuild the binary, through the existing `service_deploy` (§5.5). |
| Many apps, like tokera.com | One site, many apps under `/a/<app>/`, each a folder with a manifest (§5). |
| One common framework | Every app is built on the platform's frontend framework, design system and SDK, served from `/_/` (§8). |
| An advanced frontend framework that stays simple | Components, signals and a router, with fixed conventions and no build step (§8.2–§8.4). |
| One Rust binary that serves its pages | `yenbot.exe` embeds the platform: the landing page, the framework, the design system, fonts and icons, the API, the store, the scheduler and the workflow runtime (§4), so it serves the whole site on its own. Apps and the art library come from directories beside it. |
| A common API for every app — and for everything | `/api/v1/…`: identity, data, files, events, AI, pictures, notifications, jobs, settings, and every admin action. Nothing on the site, the admin page included, does anything the API does not (§6). |
| Claude and Zen use the API | A Zen task calls it with its HTTP tools and a yenbot API token held as a Zen credential; the API is documented in full in the repository (§6.3, §17.1). |
| A state store inside the binary | SQLite, compiled in (`rusqlite` with `bundled`), one file on disk, migrations embedded, backups kept locally (§7). |
| Scheduled jobs from the records | Jobs are rows in the store, scheduled by time, by repeat rule, or by a field in an app's own records (§9). |
| Workflows run from the jobs | A workflow is an async JavaScript function in the app's folder, run by an embedded engine, hot like the pages (§10). |
| Notifications | Web Push to phones and browsers, and an in-app inbox. No email (§11). |
| Google sign-in, like tokera.com | The gateway signs people in and forwards who they are; yenbot never sees a Google secret (§12.1). |
| Accounts in the database | A person's first sign-in registers an account in the store (§12.2). |
| Yen is the admin | Her Google address is named in `yenbot.yaml`; the admin page is hers (§12.3, §13.2). |
| Everyone gets her apps; some can be private | An app is public to every account unless Yen marks it private on the admin page (§12.4). |
| A landing page of apps | `/` lists the apps the person may open, as icons; signed out, it shows the site and a sign-in button (§13.1). |
| AI and pictures through Zen, from this site only | Browsers and workflows call yenbot; yenbot calls zend for text and Zen's image engine for pictures, with a service key nothing else holds. No budget (§14). |
| A library to draw on | Tile icons, illustrations, empty-state art, backgrounds and stickers drawn with Zen's image engine in one house style, catalogued and reused (§15). |
| Yen is not a coder | The framework, templates, the art library, guides, an extensive `CLAUDE.md` and the checks are all there from the first day (§8, §15–§17). |
| The app-building guide is read for every new app | Every app's manifest carries the guide's edition it was built to; the gate refuses an app built against an older one (§17.3). |
| No command line | Everything is done with the connector's file tools or a Zen task; the platform does on load what a tool would otherwise do — sizing art, renaming apps, migrating data (§15.2, §20.1). |
| The styling must be amazing, to her attention to detail | One design system with fixed tokens, hot like the apps; an app may override a few named tokens for itself; a living catalogue; and Yen's eye on the result (§8.5). |
| Claude can see what it built | Browser errors, workflow runs, check refusals and the platform's own lines are logs the connector reads with `log_list`, `log_read`, `log_grep`; anything else — stored data, jobs, usage — is the API, called from a Zen task; and Yen pastes a screenshot when she wants Claude to see a page (§18). |
| An edit is all or nothing | `file_edit_and_commit` returns live or refused: the site checks the change before `main` moves (§5.4). |
| Claude remembers between chats | The `yenbot` journal holds each app's purpose, Yen's decisions and her preferences (§17.6). |
| No approvals while iterating | Nothing in editing, publishing or a Zen task on the yenbot repository waits on an approval (§17.7). |
| Data stays sound as apps change | Collections declare their fields; writes are validated; an app's version can carry a migration (§7.5). |
| Real apps' queries | Counts, sums and groups, full-text search, batches, pages (§7.6). |
| English, Vietnamese-safe | English interfaces; type, input, search and sorting that handle Vietnamese correctly (§8.7). |
| Families share | Groups, and roles per collection and per record; children with their own Google accounts (§12.5). |
| Safe by default | A strict content policy, request forgery refused, uploads checked, markup escaped, hashed API tokens; loopback identity is a known hole, noted (§12.6). |
| Google services through the gateway | The gateway's Google client, already used for sign-in, gains a "connect Google" flow; yenbot asks the gateway for a person's access token (§19.3). |
| Usage is visible | Every app open is recorded; the API and the admin page sum it (§6, §20.3). |
| tokera.com links to it | An entry in the estate dropdown (§21). |

## 3. Where it runs

```text
browser ──TLS──► Cloudflare (yenbot.com.au, proxied) ──HTTP :80──► web (gateway, BlackWorld .5)
                                                                     │  signs in (Google),
                                                                     │  forwards X-Tokera-* identity
                                                                     ▼
                                                        127.0.0.1:8096  yenbot.exe  (keep service)
                                                          │  SQLite: D:\yenbot\data\yenbot.db
                                                          │  apps:   D:\yenbot\live\apps\
                                                          │  art:    D:\yenbot\live\library\
                                                          │  push:   ──► browser push services
                                                          ▼
                     gateway (loopback), service key ──► zend /v1/chat/completions  (code.tokera.com)
                                                     ├─► npcd /v1/image/generate|edit (bot.tokera.com)
                                                     └─► /gateway/events  (subscribed: repo/yenbot)
```

- **BlackWorld only**, beside the gateway, supervised by keep as the service
  `yenbot` (`keep/stable/yenbot.exe`). It binds **loopback**, so nothing on
  the LAN reaches it except through the gateway, and it believes the gateway's
  identity headers on loopback, as the other daemons do — a known hole,
  accepted for now (§12.6).
- **It is not a cluster member.** It holds its own store and has no replicas;
  BlackWorld down means Yen Bots down, its jobs held until it is back (§9.4),
  as with any keep service that is not clustered (`docs/mcp_app.md`).
- **The port** is 8096 (8081 motherd, 8092 keep, 8095 mcp are taken), set in
  `yenbot.yaml`.
- **DNS** already points `yenbot.com.au` and `www` at `tokera.com`, proxied
  (Cloudflare, checked 2026-11-06). Today they fall through to the default
  site and show Tokera.

## 4. The repository

A new repository, `john-sharratt/yenbot`, added to zend's workspace
(`D:\prog\workspace.yaml`) and to the agent's checkouts (`motherd/agent.yaml`,
`push: true`), so Zen reads, edits, builds, tests, commits and pushes it as
it does candle. Two working copies on BlackWorld, never the same one:

- `D:\prog\yenbot` — zend's working tree, where Claude's edits land.
- `D:\yenbot\live` — what the site serves, fast-forwarded from `origin/main`
  (§5.4). Nothing edits it by hand.

```text
yenbot/
  CLAUDE.md                   the repository's rules and map, read first (§17.2)
  Cargo.toml                  the one binary
  src/                        the platform, one concern per file
    main.rs  config.rs  serve/  api/  store/  accounts/  jobs/  workflows/
    push/  ai/  apps/  library/  check/
  web/                        the platform's web files: served from disk by the gateway,
                              and embedded into the binary at build time (§5.2)
    index.html  _admin/       the landing page and the admin page
    _/framework/              yen.js: components, signals, router, SDK (§8)
    _/ui/                     design system: tokens.css, yen.css, components/
    _/vendor/                 pinned third-party modules (Lit, signals)
    _/icons/  _/fonts/        the vector UI icon set and the typefaces
    sw.js                     the service worker: push and offline (§11)
  apps/                       one folder per app — served from disk, hot
    <app-id>/
      app.yaml                manifest (§5.1)
      index.html  app.js  style.css
      pages/  components/  stores/   the app's frontend (§8.3)
      workflows/*.js          the app's server-side functions (§10)
  library/                    the art library — served from disk, hot (§15)
    catalogue.yaml
    tiles/  illustrations/  empty/  backgrounds/  stickers/
  templates/                  starting points for a new app (§16)
  docs/                       the manual, the API's reference included (§17)
  yenbot.yaml                 the service's config
  tests/
```

`web/` is embedded with `include_dir!`, so the binary is whole on its own and
serves the site with no checkout beside it; the gateway serves the same
directory from the live checkout (§5.2), so a change to it is live as an
app's is. `apps/` and `library/` are read from the live directory at runtime.
Only a change under `src/` is a rebuild.

## 5. Apps

### 5.1 The manifest

```yaml
id: reminders                  # the folder name, the URL, the data namespace
name: Reminders
description: Things to remember, with a ping when they are due.
tile: library/tiles/bell-coral # its icon on the landing page, from the library (§15)
colour: coral                  # the tile's accent, a palette token
status: live                   # live | draft (draft: not on the landing page yet)
guide: 3                       # the app-building guide's edition it was built to (§17.3)
collections:                   # what it stores, and each one's fields (§7.2, §7.5)
  reminders:
    scope: private
    fields: { title: { type: text, required: true }, due_ms: { type: time } }
    index: [due_ms]
ai: true                       # it may call the AI and draw pictures (§14)
notify: true                   # it may send notifications (§11)
schedules:                     # jobs (§9)
  - every: "daily 07:30"       #   a repeat rule
    run: morning-summary       #   a workflow in workflows/
  - from: reminders            #   one job per record,
    at: due_ms                 #   at that field's time,
    before: 10m                #   less this,
    run: remind                #   running this workflow with the record
```

Whether the app is public or private is not in the manifest: it is Yen's
choice on the admin page, kept in the store (§12.4).

The platform refuses an app whose manifest does not parse, names a tile,
workflow or collection that is not there, or declares a collection twice — at
load, with the reason on the admin page, and the app stays on its last good
version.

### 5.2 Serving

Two places serve the platform's web files, for two jobs — the arrangement
zend's and npcd's consoles already use (`docs/shared_model_host.md` §3.4):

- **yenbot embeds `web/`** (`include_dir!`, with the build script rerun when a
  file under it changes), so the binary serves the whole site on its own.
- **The gateway serves the same directory from disk**: the yenbot site in
  `web/web.yaml` has `roots: ["D:/yenbot/live/web"]`, read per request
  (`Roots::disk`), and forwards to yenbot only `/a/`, `/api/`, `/v1/`,
  `/hooks/` and the reload stream. So the framework, the design system and
  the landing page change as an app does — a commit through the gate, live on
  the next load — and the binary catches up at its next release.

`/a/<id>/…` is yenbot's, from the app's folder in the live checkout. Every
file goes through the same responder as the consoles' (`web::asset::respond`):
a strong `ETag` over the content, `304` when it matches, gzip, and a short
`max-age` for HTML, JS and CSS.

### 5.3 Hot reload

A watcher on `apps/`, `library/` and `web/` reloads a changed manifest, its
schedules and its workflows, and drops its cached files, within a second. A
page open in
a browser reloads itself when its app changes: the framework keeps an
`EventSource` on `/api/v1/events/_reload` while the tab is visible. Yen sees a
change Claude made without touching her phone.

### 5.4 Publishing

Publishing is a commit on `main`, and iterating happens on the live site. The
fast path is the Zen connector's `file_edit_and_commit`
(`docs/mcp_file_tools.md` §3): Claude reads a file with `file_read`, sends its
edits, and the call answers **live** or **refused** — one call, one commit,
all or nothing:

1. The call commits the edits as Yen and pushes them to a candidate ref, not
   to `main`.
2. It publishes a check request on the gateway's bus, `gate/yenbot`
   (`docs/mcp_file_tools.md` §5.4). yenbot is the repository's gate: it
   fetches the candidate, runs the rule check (§17.4) on the apps the change
   touches, and answers ok or refused with the rule, file and line.
3. On ok, the call moves `main` to the candidate (compare-and-swap) and
   publishes `repo/yenbot`; yenbot fast-forwards `D:\yenbot\live`, swaps the
   apps in, and the open tab reloads itself (§5.3). The call answers live, with
   the commit. On refused, the candidate is deleted, `main` never moved, and
   the call answers the reason.

The whole loop is about two seconds. A larger change made by a Zen task lands
the same way, each of its commits through the same gate. yenbot also fetches
on every (re)connection to the bus and every 60 s, for a push made outside
Zen; such a push is checked on arrival and an app that fails keeps its last
good version, with the reason on the admin page. A history rewrite is
refused (fast-forward only) and logged.

**Drafts**: a new app with `status: draft` is not on the landing page yet; the
admins open it by its URL while it is being made, and Claude asks Yen before
setting it `live`. It runs on the live store like any app. **Undoing** is
failing forward: a new edit that puts back what was there, from the journal
(§17.6) or the chat.

### 5.5 Releasing the binary

A change under `src/` — the Rust: the API, the store, the scheduler, the
workflow runtime — is a release, through the build and deploy that already
release motherd, web and mcp (`zend/src/mcp/tools/service/deploy.rs`):

1. A Zen task makes the change, tests it and commits it through the gate.
2. `service_deploy {host: BlackWorld, service: yenbot}` builds it on
   BlackWorld's agent from its checkout of the yenbot repository
   (`cargo build --release -p yenbot`), stages `yenbot.exe` for keep, and has
   keep try it: keep runs the new binary and keeps it once it prints
   `INFO serving`, or restores the old one (`keep/src/trial.rs`). keep accepts
   only a build of a commit on `origin/main` (`--build-info` with no local
   part, `keep/src/remote.rs`).
3. `service_status` and the job tools (`job_status`, `job_read`) watch it.

`service_control` restarts, stops and starts it through keep, as it does the
others, for a site that is wedged rather than broken.

`service_deploy` names its repository and services in constants today
(`REPO = "candle"`, `SERVICES = [motherd, web, mcp]`, shared with
`service_control` and `service_status`); they become a table of service →
(repository, package), with yenbot's row (§22). The yenbot binary
answers `--build-info` from the cluster crate's version module and prints
`INFO serving` once listening, as mcp does.

## 6. The API

**Everything has an API.** The pages — the landing page, the admin page, every
app — are clients of it and do nothing it does not, so anything a person can
do on the site a Zen task can do through it (§6.3). Every route is under
`/api/v1`, answers JSON, and acts as the account the request carries: a
signed-in browser's, or an API token's (§6.3). The app is named by the path
or the `X-Yen-App` header the framework sets; an account without access to
the app gets 403, and an account not yet approved gets nothing but `/me`
(§12.2).

### 6.1 The apps' routes

| Route | What it does |
|---|---|
| `GET /me` | The account: name, email, picture, status, whether the admin. |
| `GET /apps` | The apps this account may open, for the landing page. |
| `GET /data/<app>/<coll>?where=…&order=…&limit=…&after=…` | Query a collection, a page at a time (§7.2, §7.6). |
| `GET /data/<app>/<coll>/_sum?group=…&count&sum=…` | Counts, sums, averages, minimum and maximum, grouped (§7.6). |
| `GET /data/<app>/<coll>/_search?q=…` | Full-text search over the collection's searchable fields (§7.6). |
| `POST /data/<app>/<coll>` | Create a document, validated against the collection's fields (§7.5); answers it with its id and version. |
| `GET/PATCH/DELETE /data/<app>/<coll>/<id>` | Read, update (merge, with `If-Match` on the version), delete. |
| `POST /data/<app>/_batch` | Several creates, updates and deletes in one transaction, all or nothing. |
| `PUT /data/<app>/<coll>/<id>/_share` | Who else may read or edit one record (§12.5). |
| `POST /errors` | What went wrong in a browser: the framework reports every uncaught error and failed call (§18.1). |
| `POST /feedback/<app>` | A note to Yen from the app's own menu, into her inbox. |
| `GET /events/<app>` | Server-sent events: every change to the app's collections, so a list another person changes updates live. |
| `POST /files/<app>` | Upload a file (image, PDF): stored, sized, thumbnailed; answers its id and URLs. |
| `GET /files/<app>/<id>[/thumb]` | Fetch it. |
| `POST /ai/chat` | A chat completion, streamed or whole, optionally constrained to a JSON schema (§14). |
| `POST /ai/image` | Draw a picture, or edit one the app holds; stored as a file of the app, answered with its URLs (§14). |
| `POST /workflows/<app>/<name>` | Run a workflow now, as the account, with arguments (§10). |
| `GET /jobs/<app>` | The app's jobs: when each runs next, its last runs and their outcome (§9). |
| `POST /push/subscribe`, `DELETE /push/subscribe` | Register or remove this device for notifications (§11). |
| `GET /inbox`, `POST /inbox/<id>/read` | The account's notifications (§11). |
| `GET/PUT /settings/<app>` | A small per-app, per-account settings object. |
| `GET /library?kind=…&tag=…` | The art library's catalogue (§15). |
| `GET /export/<app>`, `POST /import/<app>` | An app's data as JSON or CSV, out and back in (§20.1). |
| `POST /opens/<app>` | The framework records that this account opened the app (§20.3). |
| `GET/POST /google/<app>/connect`, `GET /google/<app>/status` | Connect this account's Google services for the app, and whether they are (§19.3). |

### 6.2 The admins' routes

Everything the admin page shows and does, under `/api/v1/admin`:

| Route | What it does |
|---|---|
| `GET /admin/accounts`, `POST /admin/accounts/<id>/approve\|refuse\|block\|delete` | Accounts and their state (§12.2). |
| `GET/POST/DELETE /admin/groups[/<name>[/members/<id>]]` | Groups and who is in them (§12.5). |
| `GET /admin/apps`, `PUT /admin/apps/<id>/visibility`, `PUT /admin/apps/<id>/shares`, `PUT /admin/apps/order` | Apps, public or private, who they are shared with, their order on the landing page (§12.4). |
| `GET /admin/apps/<id>/check` | The app's last check: ok, or each refusal with its rule, file and line (§17.4). |
| `GET /admin/data/<app>/<coll>?…` | Any collection, every account's records, read-only. |
| `GET /admin/jobs?app=…&state=…` | Jobs: next run, last runs and their outcomes (§9). |
| `POST /admin/jobs/<id>/run`, `POST /admin/jobs/<id>/hold\|release` | Run a job now; hold or release it. |
| `POST /admin/workflows/<app>/<name>` | Run a workflow now as the app, with arguments, answering its outcome and log (§10). |
| `GET /admin/usage?app=…&since=…&by=day\|account` | Opens, active accounts and AI calls, summed (§20.3). |
| `GET /admin/errors?app=…`, `GET /admin/feedback?app=…` | Browser errors and feedback notes. |
| `PUT/DELETE /admin/secrets/<app>/<name>`, `GET /admin/secrets/<app>` | An app's integration keys: set, remove, list names (§19.1). |
| `POST /admin/apps/<id>/archive\|restore`, `GET /admin/archive` | Deleted apps' data, kept thirty days (§20.1). |
| `GET /admin/backups`, `POST /admin/backups/<date>/restore` | The nightly backups; restore one app's data, or the whole store (§7.4). |
| `GET/POST/DELETE /admin/tokens` | API tokens (§6.3). |
| `PUT /admin/registration` | Approve or open (§12.2). |

### 6.3 API tokens: how a Zen task calls it

A Zen task reaches the site with the HTTP tools every coding worker of an
Admin already has — `http_session_open` and `http_request`
(`zend-tools/src/tools/http_session/`): any method, headers, a JSON body, a
response up to 1 MiB. A request through the gateway carries no identity of
its own (the gateway believes a bearer token only on `/mcp`,
`docs/claude_zend_gateway.md` §12), so the site issues its own:

- **An API token** is made on the admin page (`/api/v1/admin/tokens`) for an
  admin's account, named, with an expiry, and shown once. yenbot keeps only
  its hash.
- **It is held as a Zen credential**, `yenbot`, saved once with
  `credential_save`, so a task names it rather than ever seeing it: 
  `http_session_open {base_url: "https://yenbot.com.au", credential_name:
  "yenbot"}`, then `http_request {path: "/api/v1/admin/jobs", …}`. The
  gateway passes the `Authorization` header to yenbot unread; yenbot matches
  the token's hash and acts as its account.
- **A request with a token** is answered as that account, admin routes
  included, and logged with the token's name.

The docs (§17.1) carry the recipe whole: opening the session, every route
with a request and its answer, the errors, and what to check after a write.

### 6.4 The reference

`docs/api/` holds a page per group of routes — every route, its parameters,
its body, its answer and its errors, each with a complete example — and the
site serves the same description as `/api/v1/openapi.json`, generated from the
routes' own definitions, so the reference cannot drift from the code: a test
fails when a route has no page or its page names a field the route does not
take.

Outside `/api/v1`, two groups of routes serve something other than an app's
own pages: `/hooks/<app>/<name>` (inbound hooks, §19.2) and `/v1/logs`
(§18.2).

## 7. The store

### 7.1 SQLite, compiled in

`rusqlite` with the `bundled` feature: SQLite is built into the binary, the
database is one file (`D:\yenbot\data\yenbot.db`) in WAL mode, and nothing is
installed beside it. It is the most standard embedded database there is, every
tool reads it, and migrations are plain SQL: `store/migrations/NNNN_*.sql`,
embedded, applied in order at start against `PRAGMA user_version`. A
migration never edits an earlier one.

### 7.2 One document store for every app

Apps do not get tables; they get collections in one schema, so an app never
needs a migration:

```sql
CREATE TABLE docs (
  app        TEXT NOT NULL,          -- the app's id
  coll       TEXT NOT NULL,
  id         TEXT NOT NULL,          -- ULID
  owner      TEXT NOT NULL,          -- the account that created it
  scope      TEXT NOT NULL,          -- 'shared' | 'private'
  data       TEXT NOT NULL,          -- JSON
  version    INTEGER NOT NULL,
  created_ms INTEGER NOT NULL,
  updated_ms INTEGER NOT NULL,
  PRIMARY KEY (app, coll, id)
);
CREATE INDEX docs_by_owner ON docs (app, coll, owner, updated_ms);
```

A query's `where` is a small fixed language (`field op value`, joined with
`and`) compiled to `json_extract` comparisons with bound parameters — never
string-built SQL. A private collection filters on `owner` in the server, not
in the request. A field an app filters or schedules on is named in its
manifest's `index:` and gets an expression index.

### 7.3 Platform tables

`accounts` (§12.2), `app_access` (visibility and shares, §12.4), `files`
(blob metadata; bytes under `data\files\`), `settings`, `jobs` and `job_runs`
(§9), `push_keys`, `push_subscriptions` and `inbox` (§11), `ai_calls` (what
was asked or drawn, by which app and account, how long, how many tokens) and
`events` (the change feed's recent tail).

### 7.4 Backups

Every night at 03:00, a platform job (§9) writes `VACUUM INTO
data\backups\yenbot-YYYY-MM-DD.db` and copies the files directory beside it;
fourteen are kept, on BlackWorld. Restoring is an API call, like everything
else (§6.2):

- **One app**, the usual case — "I deleted the wrong recipes" —
  `POST /api/v1/admin/backups/<date>/restore {app, collections?}` replaces that
  app's records and files (all of them, or the collections named) with the
  backup's, in one transaction, after writing a backup of what it replaces;
  the rest of the site is untouched and keeps running.
- **The whole store**, for a corrupted database: the same route without `app`
  swaps the file in under the store's write lock and reopens it, again after
  backing up what it replaces.

`GET /api/v1/admin/backups` lists them with their dates and sizes.

### 7.5 Fields, validation and migrations

A collection declares its fields in the manifest, so the server can refuse a
bad write and a form can be built from the declaration:

```yaml
collections:
  recipes:
    scope: shared
    version: 2                       # the shape's version
    fields:
      title:    { type: text, required: true, max: 120 }
      serves:   { type: number, min: 1, default: 2 }
      photo:    { type: file }
      tags:     { type: list, of: text }
      cooked:   { type: time }
      notes:    { type: text, long: true }
    search: [title, notes, tags]     # what full-text search covers (§7.6)
    index:  [cooked]
```

Types: `text`, `number`, `bool`, `time`, `date`, `file`, `list`, `object`,
`ref` (another record's id, with its collection). A write that does not match
is refused with the field and the reason, which the framework's forms show
beside the field. Fields not declared are refused, so a typo cannot quietly
start a second field.

**Migrations.** When an app raises a collection's `version`, it ships
`workflows/migrate-<coll>-<n>.js`, a workflow that takes one record at the
old shape and returns it at the new. On load the platform runs it over every
record, in batches, in the background, recording progress; until it finishes
the app is served its last good version. A raised version without its
migration is refused at load.

### 7.6 Queries, sums, search and batches

- **Pages**: `limit` and `after` (an opaque cursor), so a long list loads as
  it scrolls.
- **Sums**: `count`, `sum`, `avg`, `min`, `max`, grouped by one or two
  fields, for trackers and dashboards — the `_sum` route and
  `collection.sum({group, …})`.
- **Search**: each collection's `search:` fields are kept in an SQLite FTS5
  index, updated in the write's own transaction, with diacritics folded so
  "pho" finds "phở" (§8.7) — the `_search` route and `collection.search(q)`.
- **Batches**: several writes in one transaction, all or nothing —
  `app.batch(b => …)`.
- **References**: a `ref` field is fetched with the record when the query asks
  (`expand: [author]`), one level deep.

## 8. The frontend framework

### 8.1 No build step

Apps are plain HTML, CSS and ES modules. There is no bundler, no npm, no
compile: what Claude writes is what the browser runs, the instant after the
push. The third-party modules the framework rests on are vendored into the
binary at pinned versions and served from `/_/vendor/`; an app loads nothing
from outside the site.

### 8.2 The stack

Three standard pieces and a thin layer that ties them together, `yen.js`:

- **Components: Lit.** Each piece of an interface is a custom element — a
  class with reactive properties and a `render()` written as an `html` tagged
  template. Lit is a few kilobytes, follows the web-components standard, needs
  no compiler, and composes without limit: a component is used like any HTML
  tag. Design tokens are CSS custom properties, which reach inside every
  component, so the design system styles them all.
- **State: signals.** A signal is a value that tells whatever reads it when it
  changes (`@preact/signals-core`). A component that reads a signal re-renders
  when it changes, and nothing else does. The SDK's collections are signals,
  so a list on the screen updates when the record changes — locally, or on
  another person's phone through the change feed.
- **Pages: a router.** An app's pages are routes (`/a/recipes/`,
  `/a/recipes/r/:id`), each a component, with transitions, back and deep
  links handled.
- **`yen.js`**: the base class `YenElement` (Lit plus signals plus the design
  system), `defineApp()` (routes, title, tile), and the SDK — `app.collection`,
  `app.files`, `app.ai`, `app.notify`, `app.workflows`, `app.library`,
  `app.me`.

### 8.3 How an app is laid out — the conventions that keep it simple

Complexity stays flat because every app has the same shape, however large it
grows:

```text
apps/recipes/
  index.html                 three lines: load yen.js, load app.js, <yen-app>
  app.js                     defineApp({ routes, title, tile })
  pages/                     one file per page: a component per route
  components/                one file per reusable piece, used by pages
  stores/                    one file per kind of data: its collection, its derived signals
  style.css                  the app's own layout, in tokens
  workflows/                 server-side functions (§10)
```

The rules the guides teach and the check enforces (§17.4): a page reads and
writes data only through a store; a component takes its data as properties
and raises events, never reaching into a store itself; one component per
file, named for its tag; styles in tokens. A new feature adds files; it never
turns one file into a tangle.

```js
// stores/recipes.js
import { app, computed } from "/_/yen.js";
export const recipes = app.collection("recipes");
export const favourites = computed(() => recipes.value.filter(r => r.favourite));

// pages/home.js
import { YenElement, html } from "/_/yen.js";
import { recipes } from "../stores/recipes.js";
export class RecipesHome extends YenElement {
  render() {
    return html`<yen-page title="Recipes">
      <yen-list .items=${recipes.value} empty="no-recipes">
        ${r => html`<recipe-card .recipe=${r}></recipe-card>`}
      </yen-list>
      <yen-fab icon="plus" href="new">Add a recipe</yen-fab>
    </yen-page>`;
  }
}
customElements.define("recipes-home", RecipesHome);
```

### 8.4 Extending it

An app's own components extend `YenElement` and sit in its `components/`. A
component two apps want moves into the platform's `ui/components/` and is
documented in the catalogue — the platform grows by promotion, never by apps
reaching into each other. A platform component's properties and events are
its contract: changing one is a platform release, and every app's check runs
against it.

### 8.5 The design system

The look is the platform's, not each app's. `/_/yen.css` carries:

- **Tokens** (`tokens.css`): one palette with light and dark values, a type
  scale, a spacing scale, radii, shadows, motion durations and easings. Apps
  use tokens, never literal colours or pixel sizes.
- **Type**: a self-hosted text face and a display face, embedded in the
  binary, with set line heights and measures.
- **Components** (`ui/components/`): page and app shell, button, icon button,
  floating action button, text field, text area, select, date and time picker,
  toggle, checkbox, card, list and list item, grid, tabs, sheet, dialog, toast,
  empty state, skeleton, avatar, tag, progress, image picker, photo grid,
  chart, calendar, an AI chat panel and a picture-drawing panel. Each has its
  states designed — hover, focus, pressed, disabled, loading, error — and is
  keyboard- and screen-reader-correct.
- **Icons**: the interface's own icons — arrows, plus, trash, bell — are one
  vector set, embedded as an SVG sprite, so they stay crisp and take the
  theme's colour. Pictures — tiles, illustrations, art — come from the library
  (§15).
- **Motion**: page and sheet transitions, list insertions, a press response;
  all honour `prefers-reduced-motion`.

`/_design` lists every token and every component in every state, light and
dark, at phone and desktop width. It is the reference Claude builds against
and the page Yen points at when something looks wrong.

**An app may override a few named tokens for itself**, in its manifest:

```yaml
theme:
  accent: "#d94f70"          # the app's colour: buttons, links, its tile
  accent-dark: "#f07a95"     # the same in dark mode
  surface: warm              # one of the palette's named surfaces
  radius: round              # soft | round | square
  type: display              # text | display — the face for headings
  density: roomy             # compact | normal | roomy
```

The platform writes these as custom properties on the app's root, so every
component takes them without the app touching CSS. Only these names exist; the
check refuses any other, and refuses an accent whose contrast against its
surface falls below WCAG AA in either mode, so an override can change an app's
character but never break its legibility. `/_design?app=<id>` shows the
catalogue in the app's theme.

### 8.6 What every app gets without asking

The shell (title, back to the landing page, the account's avatar), sign-in
handled, light and dark following the device, phone-first layout, installable
to the home screen, offline read of what was last loaded, loading and empty
and error states in every component, the change feed, uploads with
thumbnails, AI and pictures with streaming progress, notifications, forms
built from a collection's fields with their errors beside each field, a
"send Yen feedback" item in the app's menu, every error reported to the site
(§18.1), and a "something went wrong" panel that says what and offers a
retry.

**Offline writes**: a change made with no connection is kept in the browser's
storage, shown as pending, and sent when the connection is back, in order,
each with the version it was made against; one that conflicts with a newer
change is shown to the person to keep or drop, never silently overwritten.

### 8.7 Language and type

Interfaces are in English, and everything handles Vietnamese: the text and
display faces cover the full Vietnamese set, stacked diacritics included
(chosen and checked against a test page of Vietnamese text); text is
normalised to NFC on every write so the same word is always the same bytes;
search folds diacritics (§7.6); sorting is by `Intl.Collator`; dates, times
and numbers are formatted for `en-AU` in Yen's time zone.

## 9. Scheduled jobs

### 9.1 What a job is

A job is a row in the store: which app, which workflow, with what arguments,
when it next runs, and what happened when it last did.

```sql
CREATE TABLE jobs (
  id          TEXT PRIMARY KEY,      -- ULID
  app         TEXT NOT NULL,         -- '_platform' for the platform's own
  workflow    TEXT NOT NULL,
  args        TEXT NOT NULL,         -- JSON
  source      TEXT NOT NULL,         -- 'schedule:<n>' | 'record:<coll>/<id>' | 'write:<coll>/<id>' | 'workflow'
  rule        TEXT,                  -- a repeat rule, for a repeating job
  next_ms     INTEGER,               -- NULL: not due again
  attempts    INTEGER NOT NULL DEFAULT 0,
  state       TEXT NOT NULL,         -- 'waiting' | 'running' | 'held' | 'done' | 'failed'
  run_as      TEXT,                  -- the account it acts for, if any
  updated_ms  INTEGER NOT NULL
);
CREATE INDEX jobs_due ON jobs (next_ms) WHERE state = 'waiting';
CREATE TABLE job_runs (job TEXT, started_ms INTEGER, ended_ms INTEGER,
                       outcome TEXT, error TEXT, log TEXT);
```

### 9.2 Four ways jobs are made

- **Repeat rules** in the manifest's `schedules:` — `every: "daily 07:30"`,
  `"weekly mon 09:00"`, `"monthly 1st 08:00"`, `"hourly"`, or a cron line for
  anything else — in Yen's time zone (`yenbot.yaml`). One job per rule, its
  `next_ms` advanced after each run.
- **From records** — `from: <collection>, at: <field>[, before: <offset>]`:
  one job per record whose field holds a time, kept in step with the record.
  Creating, moving or deleting the record creates, moves or deletes its job,
  in the same transaction as the write; a record whose time has passed gets
  none. A reminder is a record with a `due_ms`; its ping is its job.
- **From workflows** — `ctx.schedule(at, workflow, args)` (§10.2): a
  workflow that has to come back later — tomorrow's follow-up, a retry in an
  hour — leaves a job for itself.
- **On a write** — `when: {collection: chores, does: update, where: {done:
  true}}, run: chore_done`: a record created, changed or deleted, matching the
  filter, makes a job for the workflow with the record before and after, in
  the same transaction as the write. "Tell me when the kids finish a chore" is
  this and a notification; the workflow runs moments later, off the request,
  so a write never waits on it.

### 9.3 Running

One scheduler loop wakes at the earliest `next_ms` (and on any change to
`jobs`), claims due jobs in a transaction (`state = 'running'`), and runs each
as a workflow (§10) on a bounded pool, a few at a time. A run that finishes
marks its job `done`, or advances a repeating job to its next time. A run that
fails is retried with backoff (1 min, 5 min, 30 min, then `failed`) and its
error kept in `job_runs`; a failed job shows on the admin page and in
`GET /jobs/<app>`.

### 9.4 When the machine was down

On start the scheduler finds jobs due while it was not running. A job due
within the last hour runs once; an older one runs once only if its schedule
says `catch_up: true`, and a repeating job then moves to its next time rather
than running every missed one. Delivery is at least once: a workflow that
sends something keys it on the job and its time (`ctx.once(key, fn)`), so a
run repeated after a crash does not send twice.

## 10. Workflows

### 10.1 Async functions, in the app, hot

A workflow is an exported async function in the app's `workflows/` folder,
in JavaScript — the language the rest of the app is in:

```js
// workflows/remind.js
export default async function remind(ctx, { record }) {
  await ctx.notify(record.owner, {
    title: record.title,
    body: `Due at ${ctx.time(record.due_ms)}`,
    open: `/a/reminders/r/${record.id}`,
  });
}
```

The binary runs it in an embedded JavaScript engine — QuickJS, through
`rquickjs` — compiled into the binary, so a workflow is a file like any
other: published by a push and live a second later, with no build. QuickJS
runs modern JavaScript, `async`/`await` included, in a few hundred kilobytes.

### 10.2 What a workflow can do

Only what `ctx` hands it — there is no filesystem, no process and no network
in the engine:

| `ctx.` | What it does |
|---|---|
| `data.collection(name)` | The app's collections, read and write, as the run's account or the app. |
| `ai.chat(…)`, `ai.image(…)` | Text and pictures, as the app (§14). |
| `notify(account, message)` | A notification (§11). |
| `schedule(at, workflow, args)` | A job for later. |
| `run(workflow, args)` | Another of the app's workflows, awaited. |
| `once(key, fn)` | Run `fn` only once for this key, across retries. |
| `fetch(url, {method, headers, body})` | An HTTP call to a host the manifest's `fetch:` lists — a weather service, a calendar (§19.1). |
| `secret(name)` | One of the app's integration keys (§19.1). |
| `time(ms)`, `now()`, `log(…)` | Time in Yen's zone; a line in the run's log. |

### 10.3 Limits

Each run has a deadline (60 s; the manifest may ask for up to 10 min, which a
workflow drawing pictures needs), a memory ceiling (64 MiB) and an
instruction-count interrupt, so a loop that never ends is stopped and recorded
rather than holding a worker. Runs are isolated: one engine context per run,
nothing kept between them except what a workflow writes to the store.

### 10.4 Run now

`POST /api/v1/workflows/<app>/<name>` and `app.workflows.run(name, args)` run
one at once, as the account — a "send me today's summary now" button, or a
heavy job (forty pictures for a storybook) kept off the phone.

## 11. Notifications

### 11.1 Web Push

The site is an installable web app: a manifest at `/_/manifest.webmanifest`
and a service worker at `/sw.js`. An account that allows notifications in an
app registers the device (`POST /push/subscribe`, done by
`app.notify.enable()`); yenbot sends with the Web Push protocol (VAPID keys
made at first start and kept in `push_keys`), so it reaches Android, desktop
browsers, and iPhone and iPad once the site is added to the home screen (iOS
16.4 and later) — the site walks a person through that once.

A tap opens the URL the notification names. A subscription the push service
reports gone is dropped. There is no email.

### 11.2 The inbox

Every notification is also written to the account's `inbox`, so nothing is
lost on a device that was off or never subscribed: a bell on the landing page
and in every app's shell shows what is unread.

### 11.3 Who can be notified

A workflow may notify an account that has access to its app. Each account can
mute an app's notifications from the shell; a muted app still writes to the
inbox.

## 12. Accounts and access

### 12.1 Sign-in is the gateway's

The gateway signs people in with Google and yenbot believes it. yenbot never
holds a Google secret or the session key; it reads `X-Tokera-User`, `-Email`,
`-Name` and `-Picture`, which the gateway strips from every incoming request
and sets only from a session it verified (`web/src/proxy.rs`,
`web/src/server.rs`), on requests from loopback (§12.6).

**The gateway needs one change.** Its session cookie is set on `.tokera.com`,
its Google callback is `https://tokera.com/auth/callback`, and its post-login
redirect is allowed only within that domain (`web/web.yaml`,
`web/src/auth/oidc.rs` `safe_next`). A sign-in on tokera.com therefore never
reaches `yenbot.com.au`. So sign-in is per cookie domain (`web/src/auth/`,
built): `auth.yaml` lists the further domains, each with its own callback —

```yaml
domains:
  - {cookie_domain: .yenbot.com.au, redirect_uri: https://yenbot.com.au/auth/callback}
```

— registered as a second redirect URI on the same Google client. A request
signs in on the domain whose cookie domain covers its host
(`Auth::domain_for`), else on `cookie_domain`: the login returns to that
domain's callback, which the signed pending state carries to the exchange; the
session cookie is set and cleared there; and `next` must stay on it. One
Google client, one session key, two cookies.

### 12.2 Accounts

A person's first signed-in request registers an account:

```sql
CREATE TABLE accounts (
  id          TEXT PRIMARY KEY,      -- the Google subject (X-Tokera-User)
  email       TEXT NOT NULL,
  name        TEXT NOT NULL,
  picture     TEXT,
  status      TEXT NOT NULL,         -- 'pending' | 'active' | 'blocked'
  created_ms  INTEGER NOT NULL,
  seen_ms     INTEGER NOT NULL
);
```

Name and picture are refreshed from the gateway's headers on each visit. How
a new account becomes active is `registration:` in `yenbot.yaml`:

- `approve` (the default): a new account is `pending`. It sees a page saying
  Yen has been told, and Yen gets a notification and a one-tap approve or
  refuse on the admin page.
- `open`: every account is `active` at once.

An account Yen blocks sees nothing and can call nothing. Every record an
account owns, every job it runs as and every notification it gets hangs on
its `id`.

### 12.3 The admin

`yenbot.yaml` names the admins by Google address:

```yaml
admins:
  - <Yen's Google address>
  - <John's Google address>
registration: approve
```

The addresses are written only in the private yenbot repository's config,
never in this public one.

The admins see drafts, the admin page and every app's jobs. Nobody else is
admin, and the list is changed only in the file. Yen is the one notified of a
new account; either admin may approve it.

### 12.4 Public and private apps

Every live app is **public**: every active account sees it on the landing page
and may open it. Yen can mark an app **private** on the admin page — then only
she sees it, and the accounts she shares it with by name. Visibility and
shares are rows in `app_access`, changed on the admin page and effective at
once; a new app arrives public once it leaves draft, and the guides tell
Claude to ask Yen before taking an app out of draft. Making an app private
hides it and its data from everyone else at once; nothing is deleted. An app
can also be shared with a group (§12.5).

### 12.5 Groups and sharing

Families share some things and not others, so access below the app has three
levels:

- **Groups**, kept by the admins on the admin page: `household`, `kids`,
  `grandparents`, any name, each a set of accounts. Children sign in with
  their own Google accounts (Family Link works) and are approved like anyone
  else, then put in their groups.
- **Collection roles**, in the manifest: who may read and who may write a
  shared collection — `roles: { read: [everyone], write: [household] }` — so
  the kids can see the chores list and only the parents edit it. `private`
  collections stay their owner's.
- **Record shares**: one record shared with named accounts or groups, to read
  or to edit (`PUT …/_share`, `record.share(…)`) — a single shopping list
  shared with Grandma, a recipe sent to a friend.

The server applies all three on every read, write, search, sum, export and
change-feed message; the framework hides what an account cannot do rather
than letting it fail.

### 12.6 Safe by default

Apps show other people's words and pictures, so the platform closes the
common holes for every app, without the app asking:

- **A strict Content-Security-Policy** on every page: scripts, styles, fonts
  and images from the site only, no inline script, no `eval` — which is also
  what keeps every app off outside CDNs.
- **Request forgery refused**: the API requires the `X-Yen-App` header the
  framework sends and the session cookie is `SameSite=Lax`, so another site
  cannot make a signed-in browser call it.
- **Markup escaped**: Lit escapes every value it renders; the guides forbid
  `unsafeHTML`, and the check refuses it.
- **Uploads checked**: by their bytes, not their name, against the types the
  collection's `file` field allows, with a size ceiling; images are re-encoded
  when thumbnailed, which drops anything hidden in them.
- **Secrets never reach a browser** (§19.1).
- **API tokens** (§6.3) are stored only as their hash, act as one account,
  expire, and are revoked on the admin page.

**A known hole, accepted for now.** yenbot believes `X-Tokera-*` on any
request from loopback, as the daemons behind the gateway do. A Zen worker's
`tcp_session_open` can reach `127.0.0.1`, so a task could claim any identity
there. Only John and Yen start tasks, so it stays open for now; the fix, when
it is closed, is a per-upstream key set by the gateway
(`docs/mcp_file_tools.md` §4.1). A worker's sessions close when its
conversation is archived (`docs/mcp_file_tools.md` §4.2), so none outlives the
task that opened it.

## 13. Pages

### 13.1 The landing page

`/`: "Yen Bots" and a line about the site, then the apps as tiles — the tile
art, the name, the app's accent colour — in the order Yen sets on the admin
page, and the inbox bell. A tap opens `/a/<id>/` with a transition from the
tile. Signed out: the same page with the tiles hidden and a single "Sign in
with Google". Pending: a page saying the account is waiting for Yen.

### 13.2 The admin page

`/_admin`, for the admins only:

- **Accounts**: pending ones first with approve and refuse; then everyone,
  with when they joined and were last seen, their groups, and block.
- **Groups**: create, rename, add and remove accounts (§12.5).
- **Apps**: each with its tile, status, public or private, who it is shared
  with, the order on the landing page, its last publish, its check result
  (§17.4), its storage, and export, import and delete
  (§20.1).
- **Jobs**: what runs next, what failed and why.
- **Secrets**: each app's integration keys, set and replaced, never shown
  again once saved (§19.1).
- **Errors and feedback**: what browsers reported, and notes from the apps'
  feedback items.
- **Registration**: approve or open.

## 14. AI and pictures through Zen

**Nothing outside yenbot calls Zen.** An app calls `POST /api/v1/ai/chat` or
`/api/v1/ai/image` and a workflow `ctx.ai.chat` or `ctx.ai.image`; yenbot
checks the account is active and the app declares `ai: true`, then calls Zen
itself and streams the answer back. A browser never holds a credential for
Zen.

- **Text** goes to zend's OpenAI-compatible endpoint with `model:
  "passthrough"` (`zend/src/api/chat.rs`, `passthrough.rs`): the app's
  messages as given, its own system prompt, streaming, a JSON schema when
  asked — none of Zen's coding projection.
- **Pictures** go to Zen's image engine — npcd's `POST /v1/image/generate`
  and `/v1/image/edit`, the engine behind the MCP's `image_generate` and
  `image_edit` (`npcd/src/mcp/image.rs`). The route streams progress and then
  the picture; yenbot relays the progress to the app and stores the picture as
  a file of the app, so it has URLs, a thumbnail and a place in the app's data.

**Both are reached through the gateway** on loopback — `Host:
code.tokera.com` for text, `Host: bot.tokera.com` for pictures — so the
cluster routes each to a ready member, not only BlackWorld's.

**The credential is a service key**, which is new
(`docs/mcp_file_tools.md` §4): yenbot's key names the identity
`service:yenbot`, the routes it may call (`/v1/chat/completions`,
`/v1/models`, `/v1/image/generate`, `/v1/image/edit`) and the bus topics it
may use; `host/roles.yaml` grants `service:yenbot` the user role. The key
lives in yenbot's secrets (`D:\yenbot\secrets\zen.key`), is read at start and
never logged, and the gateway accepts it only from its trusted backends.

**No budget.** An app chooses its prompt, its length, its dials and its
picture sizes, and nothing caps how much it asks for. Every call is recorded
in `ai_calls`, and the admin page shows what each app and account has used.
What keeps a stranger from using it freely is registration (§12.2): only an
active account reaches an app.

**Asking about the app's own data.** "Which of my recipes use chicken?" is
search first, then the model: `app.ai.ask(question, {from: collection})`
searches the collection's searchable fields (§7.6), puts the best matches in
the prompt with their ids, and answers with the ids it drew on, so the app
links to them. A semantic index (embeddings) is not in the first version.

**No looking at pictures.** Zen's model reads text only; an app cannot ask
it about a photo. A vision path is designed when an app needs one.

## 15. The art library

### 15.1 What it holds

A library of pictures every app draws from, so an app looks finished the day
it is made and every app looks like part of one site:

| Kind | What it is | Sizes |
|---|---|---|
| `tiles` | App icons for the landing page: one subject on a soft shape, in the palette's accents | 512, 256, 128 |
| `illustrations` | Scenes for headers and onboarding: cooking, school, garden, travel, money, health, family, celebration… | 1600×900 and 800×450 |
| `empty` | Empty-state art: "nothing here yet" for lists, photos, messages, search, inbox | 600×450 |
| `backgrounds` | Soft textures and gradients behind a page or a card | 1600×1600, tiling where marked |
| `stickers` | Small cut-out objects and badges: stars, hearts, ticks, trophies, food, animals | 256 with transparency |

The interface's own icons are not pictures: they stay the vector set in the
design system (§8.5).

### 15.2 How it is made

With Zen's image engine, through the MCP's `image_generate` and `image_edit`
— drawn by Claude while building, the same way Yen's apps are built. Every
picture is drawn to the **house style**, written once in
`docs/library.md`: the palette (the design tokens' colours by name), the
light, the rendering (a soft, rounded, gently lit illustration style, the
same for every kind), what never appears (text, logos, photographs of real
people), and a fixed preamble every prompt begins with. Pictures are drawn in
families — one prompt, varied seeds — and the best of each is kept.

A kept picture is committed once, at full size, as `library/<kind>/<id>.png`,
with a line in `library/catalogue.yaml`: its kind, tags, prompt and seed, so
any picture can be drawn again in kind. Pictures are binary, so a Zen task
commits them (§17.7) — the worker takes the link `image_generate` answered
with. The platform does the rest on load: it sizes each picture to its kind's
sizes as WebP (PNG where transparency needs it), reads its dominant colour,
and caches the results; a catalogue line naming a picture that is not there,
or a picture with no line, is refused by the gate.

### 15.3 How apps use it

`/_library` shows the whole library, searchable by kind, tag and colour, for
Claude and for Yen. An app names pictures by id —
`<yen-empty art="no-recipes">`, `tile: library/tiles/bell-coral`,
`app.library.find({ kind: "illustrations", tag: "cooking" })` — and the check
refuses a name not in the catalogue. A picture an app needs that is not in the
library is drawn to the house style and added to the library, not to the app,
so the next app has it too.

### 15.4 The first library

Drawn before the first app: sixty tiles across the palette's accents, thirty
illustrations across the subjects above, twenty empty states, twelve
backgrounds and fifty stickers — about 170 pictures, enough that the
templates (§16) and a first app need nothing drawn.

## 16. Templates

`templates/` holds working apps to start from, each already passing the
checks in §17.4 and using the library's art: a blank app; a list (add, edit,
tick off, reorder); a journal (dated entries, photos); a collection (cards
with images, search, tags); a planner (a week and a month view); a tracker
(log a number, see a chart); an assistant (an AI chat with a system prompt
the app sets); a picture maker (describe it, draw it, keep it); reminders
(records with a due time, a notification at it); a daily digest (a morning
workflow that gathers and notifies); a form that saves answers and a page
that sums them up. A new app starts as a copy of the closest template, under
its new id, with the template's id replaced and `guide:` set to the building
guide's current edition — made by Claude with the file tools for a small
template, by a Zen task for a large one.

## 17. Docs, CLAUDE.md, the checks and the connector's guide

### 17.1 The manual

The docs are the product as much as the code is, and they are complete from
the first release. They live in the repository's `docs/`, versioned with the
code they describe, and they derive from this design: each part of it becomes
the doc a builder reads, written for a model building for someone who cannot
read the code.

- **`docs/building-an-app.md`** — the app-building guide: read in full before
  every new app, and before any change that adds a page, a collection, a
  workflow or a schedule (§17.3). How to turn what Yen asks for into an app:
  which template, which pieces, which library art; the layout and its
  conventions and why; drafts and going live; what the gate checks. It starts
  with its edition number.
- **`docs/index.md`** — one line per doc saying when to read it, so a session
  reads only what the task needs.
- One doc per part: the manifest; the framework (components, signals, pages,
  stores, forms, offline writes); the SDK; data, fields and migrations;
  queries, sums and search; files; AI, pictures and asking about an app's
  data; jobs and schedules; workflows and `ctx`; integrations, secrets and
  hooks; notifications and the home-screen install; the design system and
  every component; language and type; the art library and its house style;
  templates; accounts, groups, sharing and private apps; what the platform
  keeps safe and what an app must never do; editing and publishing through the
  gate; bigger changes through Zen tasks and watching them (§17.7); seeing what
  happened — logs, errors, data, and Yen's screenshots; the journal (§17.6);
  exporting, importing and deleting; backups and restoring; what to do when it
  breaks.
- **`docs/api/`** — the API reference (§6.4): a page per group of routes, each
  route with its parameters, body, answer, errors and a complete example,
  tested against the routes themselves.
- **`docs/calling-the-api.md`** — how Claude reaches the API: claude.ai has no
  HTTP of its own, so it starts a Zen task on the yenbot repository whose brief
  says which calls to make; the task opens `http_session_open {base_url:
  "https://yenbot.com.au", credential_name: "yenbot"}` and makes them with
  `http_request`, and its result carries the answers. The doc gives the brief
  to copy for each common need — see an app's records, list the jobs and run
  one now, approve an account, read the usage, check a write landed — and what
  to tell Yen while it runs.

Every rule comes with its reason, every API and every `ctx` call with a
complete working example, every component with its states, and every common
request ("add a photo to each entry", "let only me see it", "remind me the
night before", "send me a summary every Sunday", "draw a picture for each
recipe") is mapped to the exact pieces that do it.

### 17.2 CLAUDE.md

The repository's `CLAUDE.md` is what every session reads first, and it is
extensive:

- **What Yen Bots is and who it is for** — Yen, how she works (claude.ai
  through Zen, often from her phone, never at a terminal), and that she is not
  a coder: explain in plain words, never show her code, ask before anything
  she would not expect. She shows Claude a page by pasting a screenshot and
  saying what it is.
- **The map** — the repository's layout (§4), where every kind of thing
  goes, and what is served from disk versus embedded.
- **The order of work** — read the app's journal entries (§17.6); read
  `docs/building-an-app.md` before every new app (§17.3); start from a copy of
  the closest template (§16); build in the conventions; edit through
  `file_read` and `file_edit_and_commit`, which answers live or refused; ask
  Yen to look; note what was decided in the journal. Anything bigger or
  structural — a new app from a large template, new art, renaming, the
  platform, anything that is not a handful of an app's own text files — goes to
  a Zen task, briefed fully and watched to its end (§17.7).
- **The rules, each with its reason** — tokens only, never literal colours or
  sizes; nothing from outside the site; a component never reaches into a
  store; one component per file; library art by id; never edit the live
  directory; never commit a secret; never set an app live or make one public
  without Yen; undo by failing forward; platform changes are releases, app
  changes are edits.
- **There is no command line.** The connector's tools and Zen tasks do
  everything; for a task, how to brief it (what to change, which docs it must
  read, how to know it is done), and building, testing and releasing the
  platform inside one.
- **Seeing what happened** — `log_list`, `log_read` and `log_grep` for the
  site's and each app's lines, browser errors included; the API from a Zen
  task for anything stored — records, jobs, usage, accounts
  (`docs/calling-the-api.md`, §18). Look before guessing.
- **When something breaks** — reading the app's log first, failing forward,
  restoring a backup, and when to stop and ask.

### 17.3 Read for every new app

A rule a model might skip is enforced where it can be:

- Every manifest carries `guide: <edition>`, the edition of
  `building-an-app.md` the app was built to, written when the app is started
  from its template.
- The gate refuses a new app without it, and any changed app whose `guide:`
  is older than the guide's current edition. When the guide changes its
  edition, an app changed afterwards must be brought up to it, which means
  reading it again.
- `CLAUDE.md` and the Zen MCP guide both open with the rule, and the
  building guide ends with a short checklist the session confirms to Yen
  before it publishes.

### 17.4 The check, which is the gate

The check is the platform's, run inside the binary: by the gate on every
candidate (§5.4), on load for anything that arrived without one, and shown on
the admin page. It refuses an app that: has a manifest error; names a
workflow, collection, tile or library picture that is not there; is built to
an older guide edition; uses a literal colour, font or pixel size where a
token exists; breaks the layout conventions (§8.3) — a page or component
outside its folder, a component reaching into a store, two components in one
file; has an image without `alt`; lacks an empty or error state on a list;
uses `unsafeHTML`; writes a field its collection does not declare; raises a
collection's version without its migration; loads anything from outside the
site; or has a workflow that does not parse or exports no function.

It runs in well under a second, so it can stand in front of every edit. How a
page looks is Yen's to judge: she pastes a screenshot into the chat and says
what it shows.

### 17.5 The connector's guide

`yen-apps` (`mcp/src/guides/yen-apps.md`), derived from this design and kept
short, because it is read at the start of every chat and her context is not
spent on it. It says what Yen Bots is, the tools the work uses
(`file_read`, `file_edit_and_commit`, the log tools, the journal tools,
`repo_task_start` and its watching — which is also how the API is called), the
order of work, and that
everything else is in the yenbot repository: `CLAUDE.md` first, then
`docs/index.md`, reading only the docs the task needs. Every Claude session
Yen opens, on any device, reads it first, as `guide_list` already asks.

### 17.6 The journal

Claude does not remember between chats; the `yenbot` journal does. It is a
journal of Zen's memory (`docs/memory_backbone.md`), readable and writable from
claude.ai and from Zen's workers alike:

- **Per app**: what it is for, what Yen decided and why, what she asked for
  that is not done yet.
- **Yen's taste**: what she likes and does not — "no gradients", "bigger
  type on the kids' apps" — which every app follows.

A session reads the journal's entries for the app before it changes anything,
and notes each decision as it is made (`journal_read`, `journal_grep`,
`journal_note`). The journal is also what failing forward reads when Yen asks
for something back as it was.

### 17.7 Bigger changes, and no approvals

Anything that is not an app's own files — the platform's Rust, its embedded
framework, the design system's components — goes to a Zen task
(`repo_task_start`) and, for the binary, a release (`service_deploy`). A task
runs for minutes; a claude.ai chat does not wake on its own. So the docs say
how to watch one: start it, tell Yen roughly how long it will take, check it
with `job_status` while she waits or when she comes back, read its result
and its commits, and confirm on the site that it landed.

**Nothing in this iteration waits on an approval**: `file_edit_and_commit`,
the read and log tools, the journal, a Zen task on the yenbot repository and
a release of the yenbot service all run on the connector's grant alone. A step
that would ask Yen to approve something in another window breaks the loop she
works in, so none does.

### 17.8 Yen's claude.ai Project

The docs above are what Claude reads once it is in the repository; this is
how it gets there. Yen works in one claude.ai Project, "Yen Bots", set up once
by John, with the Zen connector enabled and these instructions:

```text
You build and change apps on Yen Bots (yenbot.com.au) for Yen, through the Zen
connector. Before anything else in a chat, read the Zen guide `yen-apps` with
guide_read, then follow it: it tells you what to read in the yenbot repository
and in what order. Yen is not a coder: speak plainly, show her the result,
and ask before anything she would not expect.
```

The text lives in `docs/claude-ai-project.md`, so a change to it is
reviewed like any other; John copies it into the Project when it changes.

## 18. Seeing what happened

Claude cannot see the site from claude.ai unless the site shows it. Everything
that happens on Yen Bots is therefore written down where the connector can
read it.

### 18.1 What is recorded

- **The platform's own lines** — starts, publishes and swaps, refused
  versions and why, migrations, failures — in the same line shape as the
  cluster's members (`host/src/logs.rs` `LogLine`: time, level, target,
  message).
- **Per app**, as its own log source `yenbot/<app>`:
  - **browser errors** — the framework catches every uncaught error, rejected
    promise and failed API call and posts it to `/api/v1/errors` with the
    page, the account, the stack and the browser; repeats are counted rather
    than flooding;
  - **workflow runs** — each run's start, its `ctx.log` lines, its outcome
    and its error;
  - **check results** — a refused version, its rule and the file and line.

### 18.2 `/v1/logs`

yenbot serves `GET /v1/logs` with the cluster members' query
(`grep`, `level`, `since`, `limit`; `zend/src/mcp/tools/cluster_logs.rs`)
plus `source`, and `GET /v1/logs/sources`, which lists `yenbot` and every app
with how many lines each holds. Lines are kept in the store, thirty days.

### 18.3 Everything else is the API

What an app stored, its jobs and their runs, accounts, usage, check results:
each is an admin route (§6.2), and Claude reads it the way it does anything
else on the site — a Zen task with the HTTP tools (§6.3,
`docs/calling-the-api.md`). There is no connector tool per kind of thing the
site holds; the API is the one way in, for the pages, for Zen and for Claude.

### 18.4 In the connector

Logs are the one exception, because they are what Claude reads first and
most, often while a task is running, and they are not the site's alone: the
connector reads them with general tools (`docs/mcp_file_tools.md` §6).
`log_list` names every log source — the cluster's members and yenbot's apps
alike — `log_read` reads one's newest lines and `log_grep` searches one or
all. They answer in the chat, at once, with no task; they act as the person
whose connector it is, so yenbot sees an admin. What a page looks like comes
from Yen: a screenshot pasted into the chat.

## 19. Integrations

### 19.1 Secrets and outbound calls

An app that talks to another service — a weather feed, a calendar, a school
portal — keeps its keys in the store, per app, set on the admin page and never
shown again once saved. A workflow reads one with `ctx.secret(name)`; no
browser ever receives one. `ctx.fetch` may `GET`, `POST`, `PUT` and `DELETE`
to the hosts the manifest's `fetch:` lists, with headers and a body.

### 19.2 Inbound hooks

`/hooks/<app>/<name>` runs the app's `workflows/hooks/<name>.js` with the
request, for a service that calls in — a form elsewhere, a shortcut on her
phone. A hook is reachable without signing in, so each carries a token made
on the admin page, in its URL or as an HMAC signature with an app secret; an
unsigned call is refused and logged.

### 19.3 Google services, through the gateway

An app that reads a person's calendar, writes to their Drive or adds a task
needs that person's Google consent. The gateway already holds the site's
Google client and signs people in with it (`web/src/auth/oidc.rs`), asking only
`openid email profile` and keeping no Google token past the callback. It
grows a **connect** flow; yenbot holds no Google secret and no refresh token.

- **The app declares what it needs**: `google: [calendar.readonly,
  tasks]` in its manifest; the check refuses a scope outside the platform's
  allowed list.
- **Connecting**: the framework's `app.google.connect()` sends the person to
  the gateway's `/auth/google/connect?scopes=…&next=…`, which asks Google for
  those scopes on top of sign-in (`include_granted_scopes=true`,
  `access_type=offline`) and stores the refresh token it gets, encrypted with
  the gateway's session key, keyed by Google subject and scope set.
  `/api/v1/google/<app>/status` answers whether the person has granted what
  the app declares, so the app shows a "Connect Google" button only when it
  is needed.
- **Using it**: a workflow calls `ctx.google.fetch(url, init)` for an account;
  yenbot asks the gateway `POST /gateway/google/token {user, scopes}` with its
  service key — a route in the key's list — and gets a short-lived access
  token, which it uses at once and never stores. The gateway refreshes and
  answers `not_connected` when the person has not granted the scopes or has
  revoked them, which `ctx.google.fetch` raises for the app to handle.
- **Disconnecting**: the account's menu has "Disconnect Google", which revokes
  the token at Google and deletes it at the gateway.

The Google Cloud console needs the scopes on the consent screen; while the
client is unverified for sensitive scopes, the accounts that may grant them
are its listed test users, which is the family. John adds them.

## 20. Lifecycle and health

### 20.1 Apps and accounts over time

- **Deleting an app** is removing its folder. Its data, files and jobs are
  moved to an archive and kept thirty days, shown on the admin page with
  restore; then a platform job purges them.
- **Renaming an app** is moving its folder to the new id and adding
  `renamed_from: <old>` to its manifest, in one commit (a Zen task, for an app
  of any size). On load the platform moves the app's data, files, jobs and
  access to the new id, once, and serves the old URL as a redirect.
- **Export and import**: an app's data as JSON (every collection) or CSV (one
  collection), from the admin page and `/api/v1/export`; an import is
  validated against the fields like any write.
- **An account deleted** by an admin loses its private records and files;
  records it shared stay, shown as by a former member.
- **Storage**: the admin page shows each app's records and files; a platform
  job removes files no record names.

### 20.2 Health

yenbot answers `GET /v1/status` (ready, version, the store, the scheduler,
the bus subscription). keep restarts it when it exits. The gateway reports
each site upstream's health in its status, so `cluster_status` shows Yen Bots
beside the cluster, and a site that stays down is a line there rather than a
silence.

### 20.3 Usage

Which apps get used is what tells Yen what to build next, so it is recorded:

```sql
CREATE TABLE app_opens (
  app      TEXT NOT NULL,
  account  TEXT NOT NULL,
  day      TEXT NOT NULL,          -- YYYY-MM-DD, site time
  opens    INTEGER NOT NULL,
  PRIMARY KEY (app, account, day)
);
```

The framework posts `/api/v1/opens/<app>` once per page load; the row's count
goes up. `GET /api/v1/admin/usage` sums `app_opens` and `ai_calls` (§14) by
app, account and day, and the admin page shows the same: each app's opens and
active accounts over the last weeks, and what it asked of the model and the
image engine.

## 21. tokera.com

The estate dropdown lists the sites in `ESTATE` (`web/src/site/tokera/page.rs`)
and its copy in `web/content/common/lib/estate.js`, kept equal by
`the_estate_list_matches_the_shared_module`. Yen Bots is a fifth entry there,
with its tile under `web/content/common/brand/`, and in the footer.

## 22. What changes outside the new repository

| Where | Change |
|---|---|
| `web/web.yaml` | A `yenbot` site: `hosts: [yenbot.com.au]`, `api: [{prefix: /, upstream: "http://127.0.0.1:8096"}]`; `www.yenbot.com.au` as a redirect site. |
| `web/src/auth/` | Sign-in per cookie domain (§12.1); the Google connect flow, the encrypted refresh-token store and `/gateway/google/token` (§19.3). |
| `zend-tools`, `zend/src/mcp/archiving.rs` | A worker's tool sessions closed when its conversation is archived (`docs/mcp_file_tools.md` §4.2). |
| `web/web.yaml` roots | `D:/yenbot/live/web` as the yenbot site's disk root (§5.2). |
| `web/src/`, `web.yaml`, zend | Service keys, the gateway's bus, `repo/<name>` on every push, and the connector's file tools — `docs/mcp_file_tools.md`. |
| `host/roles.yaml` | `service:yenbot` as a user. |
| Google Cloud console | `https://yenbot.com.au/auth/callback` as a redirect URI on the existing client; the scopes apps may ask for on the consent screen, and the family as test users (§19.3). John does this. |
| Zen credentials | `yenbot`, an API token for Yen's account, saved with `credential_save` (§6.3). |
| `keep/keep.yaml` | The `yenbot` service: `stable: keep/stable/yenbot.exe`, `args: [--config, D:/yenbot/yenbot.yaml]`. |
| `motherd/agent.yaml`, `D:\prog\workspace.yaml` | The `yenbot` repository, for Zen and the agent, with `author: {name: Yen, email: <Yen's address>}`. |
| `service_deploy`, `service_control`, `service_status` | Learn `yenbot`: its repository and its package (§5.5). |
| Zen connector | Every workspace repository is offered to the existing connectors (`docs/mcp_file_tools.md` §3.4); `log_list`, `log_read`, `log_grep` in place of `cluster_logs` (`docs/mcp_file_tools.md` §6); yenbot as the yenbot repository's gate (`docs/mcp_file_tools.md` §3.2). |
| Zen's memory | The `yenbot` journal (§17.6). |
| Zen grants | No approval on any step of the iteration (§17.7). |
| `web/web.yaml` | yenbot as a log source and a site whose health the gateway reports (§20.2). |
| `mcp/src/guides/` | `yen-apps.md` (§17.5). |
| Yen's claude.ai | The "Yen Bots" Project and its instructions (§17.8). John sets it up. |
| `web/src/site/tokera/page.rs`, `estate.js` | The estate entry (§21). |
| `docs/deployment.md` | The service, its port, its paths and its launch line. |

## 23. Decisions

Taken 2026-11-06:

- **Yen's Google account** (the one in Zen's `persons/yen` record) is admin, and the author of every commit to
  the yenbot repository.
- **John is admin too.**
- **Registration is approved by an admin.** With no AI budget, open
  registration would let any Google account use the model and the image
  engine through a public app.
- **No AI budget, no email, backups local.**
- **Edits are commits, all or nothing**: `file_edit_and_commit` answers live
  or refused, the site gating each one over the gateway's bus before `main`
  moves (`docs/mcp_file_tools.md`).
- **Iterating happens on the live site**; drafts only keep a new app off the
  landing page.
- **Undo is failing forward**; there are no history tools.
- **Children sign in with their own Google accounts**, approved and grouped
  like anyone else.
- **English interfaces, Vietnamese-safe** throughout.
- **No vision** in the first version.
- **Claude sees the site's logs through general connector tools**
  (`log_list`, `log_read`, `log_grep`), **and everything else through the
  API**, called from a Zen task with the HTTP tools and an API token held as a
  Zen credential. There are no site-specific connector tools. It sees a page
  when Yen pastes a screenshot and says what it shows.
- **Everything has an API**, the admin page's actions included, documented in
  `docs/api/` and served as OpenAPI.
- **The binary is released with the existing `service_deploy`**; the
  platform's web files are served from disk by the gateway, so only Rust
  changes need it.
- **Apps may override a fixed set of named tokens**, contrast-checked.
- **Google services go through the gateway's existing client**, which keeps
  the refresh tokens; yenbot only ever holds a short-lived access token.
- **Loopback identity stays a known hole for now**; a worker's sessions close
  when its conversation is archived.
- **Memory between chats is the `yenbot` journal.**
- **No approvals** anywhere in the iteration.
- **The connector's guide is short** and points into the repository's
  `docs/`, which derive from this design.

## 24. Order of work

1. The gateway and zend: sign-in per domain, the yenbot site, its route and
   disk root, and `docs/mcp_file_tools.md` — service keys, sessions closed on
   archive, the bus, `repo/<name>`, gates, the connector's file and log tools.
2. The repository, `CLAUDE.md`, and the binary's skeleton: serving, identity,
   accounts, the landing page, the store and its migrations, backups, the
   watcher and the publisher, its logs and `/v1/logs`, health, keep and
   `INFO serving`.
3. The API — fields and validation, queries, sums, search, batches, groups and
   shares, every admin route, API tokens, OpenAPI and `docs/api/` with the
   test that keeps them equal — with tests against a real SQLite file.
4. The frontend framework: `yen.js`, the vendored modules, routing, stores,
   forms, offline writes, error reporting.
5. The design system, its type and its Vietnamese test page, its catalogue
   page and the components.
6. The gate: the platform's check answering candidates on the bus.
7. The art library: the house style, sizing on load, the first library.
8. Jobs and workflows: the scheduler, the record-driven schedules, QuickJS and
   `ctx`, migrations.
9. Notifications: the service worker, Web Push, the inbox.
10. AI and pictures through Zen, and asking about an app's data.
11. Integrations: secrets, outbound calls, inbound hooks, and Google through
    the gateway's connect flow.
12. The admin page, usage, export, import, deleting and archiving.
13. Templates, the repository's `docs/` derived from this design (with
    `docs/calling-the-api.md`), `CLAUDE.md`, the connector's `yen-apps` guide,
    the `yenbot` journal, the `yenbot` Zen credential, the claude.ai Project.
14. `service_deploy`, `service_control` and `service_status` for yenbot, no
    approvals on the iteration, the estate entry, `deployment.md`.
15. The test cases (§25), run in claude.ai through the "Yen Bots" Project,
    every one passing.
16. Yen's first app, built by her in claude.ai from the docs alone — the test
    of everything above.

## 25. Test cases

The platform is done when every case below passes, run as Yen would run it: a
fresh chat in the "Yen Bots" claude.ai Project (§17.8), the request typed in
her words, nothing else said. Claude has only the connector, the guides and
the repository's docs. A case passes when the outcome holds on the live site
and Claude got there by the path named, reading the guide named. A case that
passes another way — a guess that happened to work, a tool the guide does not
name, a question to Yen the docs should have answered — fails, and the guide
or the docs are fixed until it passes as written.

"Task" is a Zen task calling the API with `http_session_open` and
`http_request` and the `yenbot` credential (§6.3); "edit" is
`file_edit_and_commit` (§5.4).

| # | Yen says | Path | Guide | Passes when |
|---|---|---|---|---|
| 1 | "Make me a recipe app" | `building-an-app.md` read in full; the closest template copied; edits (or a task for a large template); the gate checks it; it is a draft | §17.3, §16 | The app opens for Yen at `/a/recipes/`, is not on anyone else's landing page, and its manifest's `guide:` is the current edition. |
| 2 | "Make it pink, with rounder buttons" | `theme:` overrides in the manifest, one edit | §8.5 | The app shows the new accent and radius in light and dark; an accent too pale to read is refused by the gate and Claude picks a readable one. |
| 3 | "It's blank" (a screenshot) | `log_grep` on `yenbot/<app>` for browser errors; the fix; an edit | §18 | Claude names the error from the log before changing anything, and the page renders after the edit. |
| 4 | "Remind me every Sunday night to plan meals" | `schedules:` in the manifest; a workflow; Web Push | §9.2, §11 | A push arrives on Yen's phone at the set time. |
| 5 | "Did the reminder go out?" | `log_read` of the workflow's runs, or a task: `GET /admin/jobs` | §18, §6.2 | Claude answers with the run's time and outcome. |
| 6 | "Run it now so I can see" | A task: `POST /admin/jobs/<id>/run` | §6.2, `calling-the-api.md` | The push arrives within the task's run, and Claude told Yen roughly how long it would take. |
| 7 | "Approve my cousin" | Yen's admin page, or a task: `POST /admin/accounts/<id>/approve` | §6.2 | The cousin's account is active and sees the public apps. |
| 8 | "Only the family should see the chores app" | A task: `PUT /admin/apps/<id>/visibility` and `/shares` with `household` | §12.4–12.5 | An account outside `household` gets neither the tile nor the app's data. |
| 9 | "The kids can see the chores, only we can change them" | Collection `roles:` in the manifest, one edit | §12.5 | A child's account reads the list and cannot change it; the edit controls are hidden, not failing. |
| 10 | "Draw a cute icon for it" | The art library through a Zen task, in the house style | §15 | The tile shows the new art, sized, in the library by id. |
| 11 | "Are my recipes actually saved?" | A task: `GET /admin/data/recipes/recipes` | §6.2 | Claude answers with the records' count and names, from the store. |
| 12 | "Add a photo to each recipe" | A `file` field, the collection's version raised with its migration | §7.5 | Old recipes still open; new ones take a photo with a thumbnail. |
| 13 | "How much is each app being used?" | A task: `GET /admin/usage` | §20.3 | Claude answers opens and active accounts per app for the period asked. |
| 14 | "Show my Google calendar in the planner" | `google:` in the manifest; "Connect Google" through the gateway; `ctx.google.fetch` | §19.3 | After Yen connects once, the planner lists her events; an account that has not connected sees the button, not an error. |
| 15 | "Undo that, I liked it before" | Failing forward: the journal's note of what she had, and a task reading the earlier version from git and writing it back | §17.6 | The page is as it was, through a new commit; no history was rewritten. |
| 16 | "Tell me when the kids finish a chore" | A `when:` job on the chores collection, a workflow, a notification | §9.2, §11 | Marking a chore done sends Yen a push within seconds. |
| 17 | "Make a homework helper chat for the kids" | `ai: true` and the chat panel | §14 | A child's account asks and gets a streamed answer; the call is in `ai_calls`. |
| 18 | "I deleted the wrong recipes" | A task: `GET /admin/backups`, then `POST /admin/backups/<date>/restore {app}` | §7.4 | The recipes are back; every other app's data is untouched; a backup of what was replaced exists. |
| 19 | "The site's not loading" | `cluster_status` for the site's health, `log_grep` on `yenbot`, then `service_control` to restart | §20.2, §5.5 | Claude says what the log showed and the site answers again. |
| 20 | "Make the date picker nicer everywhere" | The platform component under `web/_/ui/`, an edit, served from disk | §5.2 | Every app's date picker changes without a release; `/_design` shows it in every state. |
| 21 | "Can the platform make QR codes?" | A task on the Rust, then `service_deploy` for yenbot, watched with `job_status` | §5.5, §17.7 | The feature is live; Claude told Yen how long it would take and confirmed on the site that it landed. |
| 22 | "Put my apps in this order" | Yen's admin page, or a task: `PUT /admin/apps/order` | §6.2 | The landing page shows the order she gave. |
| 23 | "Put the recipes in a spreadsheet" | `/export/<app>` as CSV | §20.1 | The CSV opens in a spreadsheet with every recipe and Vietnamese text intact. |
| 24 | "Put it on my phone's home screen" | The install prompt every app gets | §11 | The app opens from the home screen, full-screen, and its pushes still arrive. |
