# Yenbot: an app host for Yen

**Status: design, not built.** This document is authoritative for the yenbot
service, its repository, its place behind the gateway and the changes the
gateway, zend and tokera.com need for it.

## 1. What it is

`yenbot.com.au` is where Yen makes and keeps her own apps: a habit tracker, a
recipe book, a planner for the kids, a gallery, a reminder that pings her
phone, a tool that asks the model something. She signs in with Google, sees
her apps as icons on one page, and taps one to open it.

She does not write code. She works in claude.ai, which reaches our machines
through the Zen MCP connector, from wherever she is. She describes what she
wants, Claude builds it in the yenbot repository through Zen, and it is live
on her site seconds later. So the platform has to make the right thing the
easy thing for a model writing code for someone who cannot review it: every
app is built on one framework, from one design system, documented end to end,
and checked before it goes live.

## 2. Principles, and what each one decides

| Principle | Decision |
|---|---|
| HTML changes go live quickly | Apps are files the binary serves from disk and reloads on change (§5.3). Publishing an app is a git push: no build, no restart. Only platform changes rebuild the binary. |
| Many apps, like tokera.com | One site, many apps under `/a/<app>/`, each a folder with a manifest (§5). |
| One common framework | Every app is built on the platform's frontend framework, design system and SDK, served from `/_/` (§8). |
| An advanced frontend framework that stays simple | Components, signals and a router, with fixed conventions and no build step (§8.2–§8.4). |
| One Rust binary that serves its pages | `yenbot.exe` embeds the platform: the landing page, the framework, the design system, fonts and icons, the API, the store, the scheduler and the workflow runtime (§4). Apps come from a directory beside it. |
| A common API for every app | `/api/v1/…`: identity, data, files, events, AI, notifications, jobs, settings (§6). |
| A state store inside the binary | SQLite, compiled in (`rusqlite` with `bundled`), one file on disk, migrations embedded (§7). |
| Scheduled jobs from the records | Jobs are rows in the store, scheduled by time, by repeat rule, or by a field in an app's own records (§9). |
| Workflows run from the jobs | A workflow is an async JavaScript function in the app's folder, run by an embedded engine, hot like the pages (§10). |
| Notifications | Web Push to her phone and browsers, and an in-app inbox (§11). |
| Google sign-in, like tokera.com | The gateway signs people in and forwards who they are; yenbot never sees a Google secret (§12). |
| A landing page of apps | `/` lists the apps the person may open, as icons; signed out, it shows the site and a sign-in button (§13). |
| AI through Zen, from this site only | Browsers and workflows call yenbot; yenbot calls zend with a service key nothing else holds (§14). |
| Yen is not a coder | The framework, templates, guides and checks are all there from the first day (§8, §15, §16). |
| The styling must be amazing, to her attention to detail | One design system with fixed tokens, a living catalogue of every component, and a review checklist every app passes (§8.5, §16.3). |
| tokera.com links to it | An entry in the estate dropdown (§17). |

## 3. Where it runs

```text
browser ──TLS──► Cloudflare (yenbot.com.au, proxied) ──HTTP :80──► web (gateway, BlackWorld .5)
                                                                     │  signs in (Google),
                                                                     │  forwards X-Tokera-* identity
                                                                     ▼
                                                        127.0.0.1:8096  yenbot.exe  (keep service)
                                                          │  SQLite: D:\yenbot\data\yenbot.db
                                                          │  apps:   D:\yenbot\live\apps\
                                                          │  push:   ──► browser push services
                                                          ▼
                                       gateway, Host code.tokera.com, service key ──► zend (cluster)
```

- **BlackWorld only**, beside the gateway, supervised by keep as the service
  `yenbot` (`keep/stable/yenbot.exe`). It binds **loopback**, so nothing on
  the LAN reaches it except through the gateway, and it believes the gateway's
  identity headers from loopback alone — the model `mcp` already follows
  (`mcp/mcp.yaml`, `mcp/src/believed.rs`).
- **It is not a cluster member.** It holds its own store and has no replicas;
  BlackWorld down means yenbot down, its jobs held until it is back (§9.4), as
  with any keep service that is not clustered (`docs/mcp_app.md`).
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
  Cargo.toml                  the one binary
  src/                        the platform, one concern per file
    main.rs  config.rs  serve/  api/  store/  jobs/  workflows/  push/  ai/  apps/  identity/
  platform/                   embedded into the binary at build time
    landing/                  the home page
    framework/                yen.js: components, signals, router, SDK (§8)
    ui/                       design system: tokens.css, yen.css, components/
    vendor/                   pinned third-party modules (Lit, signals), served from /_/
    sw.js                     the service worker: push and offline (§11)
    icons/  fonts/
  apps/                       one folder per app — served from disk, hot
    <app-id>/
      app.yaml                manifest (§5.1)
      index.html  icon.svg
      pages/  components/  stores/  style.css   the app's frontend (§8.3)
      workflows/*.js          the app's server-side functions (§10)
  templates/                  starting points for a new app (§15)
  guides/                     everything a writer of an app needs (§16)
  yenbot.yaml                 the service's config
  tests/
```

The platform is embedded with `include_dir!`, so the binary is whole on its
own; `apps/` is read from the live directory at runtime. A platform change is
a rebuild; an app change — pages, styles, workflows, schedules — never is.

## 5. Apps

### 5.1 The manifest

```yaml
id: reminders                  # the folder name, the URL, the data namespace
name: Reminders
description: Things to remember, with a ping when they are due.
icon: icon.svg                 # 512×512, on the icon grid in guides/
colour: "#E07A5F"              # the tile's accent, one of the palette's
status: live                   # live | draft (draft: editors only)
access: [yen@…, john@…]        # who may open it; absent: everyone signed in
collections:                   # what it stores (§7.2)
  reminders: { scope: private, index: [due_ms] }
  lists: shared
ai: true                       # it may call the AI (§14)
notify: true                   # it may send notifications (§11)
schedules:                     # jobs (§9)
  - every: "daily 07:30"       #   a repeat rule
    run: morning-summary       #   a workflow in workflows/
  - from: reminders            #   one job per record,
    at: due_ms                 #   at that field's time,
    before: 10m                #   less this,
    run: remind                #   running this workflow with the record
```

The platform refuses an app whose manifest does not parse, names an icon,
workflow or collection that is not there, or declares a collection twice — at
load, with the reason on the editor's page (§13), and the app stays on its
last good version.

### 5.2 Serving

`/a/<id>/…` serves the app's folder. Static files carry an `ETag` from their
content hash and `Cache-Control: no-cache`, so a changed file is fetched on
the next load and an unchanged one costs a 304. The platform's own files
under `/_/` are versioned by the build and cached for a year.

### 5.3 Hot reload

A watcher on `apps/` reloads a changed manifest, its schedules and its
workflows, and drops its cached files, within a second. A page open in a
browser reloads itself when its app changes: the framework keeps an
`EventSource` on `/api/v1/events/_reload` while the tab is visible. Yen sees a
change Claude made without touching her phone.

### 5.4 Publishing

Publishing is a push to `main`. yenbot fetches `origin/main` every 10 s in the
background, fast-forwards `D:\yenbot\live` when it moved, and the watcher does
the rest; the fetch runs off every request path. A push that rewrites history
is refused (fast-forward only) and logged.

**Drafts**: an app with `status: draft` is live for editors only, its
schedules held, so a new app can be tried on the real site before Yen shows
anyone. **Rolling back** is a revert commit; every version of every app is in
git.

**Platform releases** go the ordinary way: an agent build on BlackWorld,
staged to keep, tried, and kept when it prints `INFO serving` (`CLAUDE.md`,
"Releasing and Testing Locally"; keep's trial in `keep/src/main.rs`).
`service_deploy` learns the `yenbot` service and its repository for this
(§18).

## 6. The API

Every route is under `/api/v1`, answers JSON, and acts as the signed-in
person. The app is named by the path or the `X-Yen-App` header the framework
sets; a person without access to the app gets 403.

| Route | What it does |
|---|---|
| `GET /me` | Who is signed in: name, email, picture, whether an editor. |
| `GET /apps` | The apps this person may open, for the landing page. |
| `GET /data/<app>/<coll>?where=…&order=…&limit=…` | Query a collection (§7.2). |
| `POST /data/<app>/<coll>` | Create a document; answers it with its id and version. |
| `GET/PATCH/DELETE /data/<app>/<coll>/<id>` | Read, update (merge, with `If-Match` on the version), delete. |
| `GET /events/<app>` | Server-sent events: every change to the app's collections, so a list another person changes updates live. |
| `POST /files/<app>` | Upload a file (image, PDF): stored, sized, thumbnailed; answers its id and URLs. |
| `GET /files/<app>/<id>[/thumb]` | Fetch it. |
| `POST /ai/chat` | A chat completion, streamed or whole, optionally constrained to a JSON schema (§14). |
| `POST /workflows/<app>/<name>` | Run a workflow now, as the person, with arguments (§10). |
| `GET /jobs/<app>` | The app's jobs: when each runs next, its last runs and their outcome (§9). |
| `POST /push/subscribe`, `DELETE /push/subscribe` | Register or remove this device for notifications (§11). |
| `GET /inbox`, `POST /inbox/<id>/read` | The person's notifications (§11). |
| `GET/PUT /settings/<app>` | A small per-app, per-person settings object. |

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
  app        TEXT NOT NULL,
  coll       TEXT NOT NULL,
  id         TEXT NOT NULL,          -- ULID
  owner      TEXT NOT NULL,          -- the person who created it
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

`files` (blob metadata; bytes under `data\files\`), `settings`, `jobs` and
`job_runs` (§9), `push_keys`, `push_subscriptions` and `inbox` (§11),
`ai_calls` (what was asked, by whom, how long, how many tokens) and `events`
(the change feed's recent tail).

### 7.4 Backups

Every night at 03:00 `VACUUM INTO` writes `data\backups\yenbot-YYYY-MM-DD.db`,
seven kept, and the files directory is copied beside it — itself a platform
job (§9). Where the copy goes off the machine is open (§19).

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
  system), `defineApp()` (routes, title, icon), and the SDK — `app.collection`,
  `app.files`, `app.ai`, `app.notify`, `app.workflows`, `app.me`.

### 8.3 How an app is laid out — the conventions that keep it simple

Complexity stays flat because every app has the same shape, however large it
grows:

```text
apps/recipes/
  index.html                 three lines: load yen.js, load app.js, <yen-app>
  app.js                     defineApp({ routes, title, icon })
  pages/                     one file per page: a component per route
  components/                one file per reusable piece, used by pages
  stores/                    one file per kind of data: its collection, its derived signals
  style.css                  the app's own layout, in tokens
  workflows/                 server-side functions (§10)
```

The rules the guides teach and the check enforces (§16.3): a page reads and
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
      <yen-list .items=${recipes.value} empty="No recipes yet">
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
  chart, calendar, and an AI chat panel. Each has its states designed — hover,
  focus, pressed, disabled, loading, error — and is keyboard- and
  screen-reader-correct.
- **Icons**: one icon set, embedded as an SVG sprite.
- **Motion**: page and sheet transitions, list insertions, a press response;
  all honour `prefers-reduced-motion`.

`/_design` lists every token and every component in every state, light and
dark, at phone and desktop width. It is the reference Claude builds against
and the page Yen points at when something looks wrong.

### 8.6 What every app gets without asking

The shell (title, back to the landing page, the person's avatar), sign-in
handled, light and dark following the device, phone-first layout, installable
to the home screen, offline read of what was last loaded, loading and empty
and error states in every component, the change feed, uploads with
thumbnails, AI with streaming, notifications, and a "something went wrong"
panel that says what and offers a retry.

## 9. Scheduled jobs

### 9.1 What a job is

A job is a row in the store: which app, which workflow, with what arguments,
when it next runs, and what happened when it last did.

```sql
CREATE TABLE jobs (
  id          TEXT PRIMARY KEY,      -- ULID
  app         TEXT NOT NULL,
  workflow    TEXT NOT NULL,
  args        TEXT NOT NULL,         -- JSON
  source      TEXT NOT NULL,         -- 'schedule:<n>' | 'record:<coll>/<id>' | 'workflow'
  rule        TEXT,                  -- a repeat rule, for a repeating job
  next_ms     INTEGER,               -- NULL: not due again
  attempts    INTEGER NOT NULL DEFAULT 0,
  state       TEXT NOT NULL,         -- 'waiting' | 'running' | 'held' | 'done' | 'failed'
  run_as      TEXT,                  -- the person it acts for, if any
  updated_ms  INTEGER NOT NULL
);
CREATE INDEX jobs_due ON jobs (next_ms) WHERE state = 'waiting';
CREATE TABLE job_runs (job TEXT, started_ms INTEGER, ended_ms INTEGER,
                       outcome TEXT, error TEXT, log TEXT);
```

### 9.2 Three ways jobs are made

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

### 9.3 Running

One scheduler loop wakes at the earliest `next_ms` (and on any change to
`jobs`), claims due jobs in a transaction (`state = 'running'`), and runs each
as a workflow (§10) on a bounded pool, a few at a time. A run that finishes
marks its job `done`, or advances a repeating job to its next time. A run that
fails is retried with backoff (1 min, 5 min, 30 min, then `failed`) and its
error kept in `job_runs`; a failed job shows on the editor page and in
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
| `data.collection(name)` | The app's collections, read and write, as the run's person or the app. |
| `ai.chat(…)` | The AI, as the app (§14). |
| `notify(person, message)` | A notification (§11). |
| `schedule(at, workflow, args)` | A job for later. |
| `run(workflow, args)` | Another of the app's workflows, awaited. |
| `once(key, fn)` | Run `fn` only once for this key, across retries. |
| `fetch(url)` | An HTTP GET to a host the manifest's `fetch:` lists — a weather service, a feed. |
| `time(ms)`, `now()`, `log(…)` | Time in Yen's zone; a line in the run's log. |

### 10.3 Limits

Each run has a deadline (60 s, the manifest may ask for up to 10 min), a
memory ceiling (64 MiB) and an instruction-count interrupt, so a loop that
never ends is stopped and recorded rather than holding a worker. Runs are
isolated: one engine context per run, nothing kept between them except what a
workflow writes to the store.

### 10.4 Run now

`POST /api/v1/workflows/<app>/<name>` and `app.workflows.run(name, args)` run
one at once, as the person — a "send me today's summary now" button, or a
heavy job (asking the AI about forty photos) kept off the phone.

## 11. Notifications

### 11.1 Web Push

The platform is an installable web app: a manifest at `/_/manifest.webmanifest`
and a service worker at `/sw.js`. A person who allows notifications in an app
registers the device (`POST /push/subscribe`, done by `app.notify.enable()`);
yenbot sends with the Web Push protocol (VAPID keys made at first start and
kept in `push_keys`), so it reaches Android, desktop browsers, and iPhone and
iPad once the site is added to the home screen (iOS 16.4 and later) — the
guides walk Yen through that once.

A tap opens the URL the notification names. A subscription the push service
reports gone is dropped.

### 11.2 The inbox

Every notification is also written to the person's `inbox`, so nothing is
lost on a device that was off or never subscribed: a bell on the landing page
and in every app's shell shows what is unread.

### 11.3 Who can be notified

A workflow may notify a person who has access to its app. Each person can
mute an app's notifications from the shell; a muted app still writes to the
inbox.

## 12. Sign-in and access

**The gateway signs people in and yenbot believes it.** yenbot never holds a
Google secret or the session key; it reads `X-Tokera-User`, `-Email`, `-Name`
and `-Picture` on requests from loopback, which the gateway strips from every
incoming request and sets only from a session it verified
(`web/src/proxy.rs`, `web/src/server.rs`).

**The gateway needs one change.** Its session cookie is set on `.tokera.com`,
its Google callback is `https://tokera.com/auth/callback`, and its post-login
redirect is allowed only within that domain (`web/web.yaml`,
`web/src/auth/oidc.rs` `safe_next`). A sign-in on tokera.com therefore never
reaches `yenbot.com.au`. The change: sign-in per cookie domain — `auth.yaml`
lists each domain with its own callback (`https://yenbot.com.au/auth/callback`,
registered as a second redirect URI on the same Google client), the session
cookie is set for the domain the request arrived on, and `safe_next` accepts
the domain the sign-in started on. One Google client, one session key, two
cookies.

**Who may open what.** Signed out, the site shows its landing page and a
sign-in button, nothing else. Signed in, a person sees the apps whose `access`
lists them (every signed-in person, when an app lists none). `yenbot.yaml`
names the **editors** — Yen and John — who see drafts, jobs and the editor
page. A signed-in stranger sees no apps and costs nothing: the AI route
requires access to an app that declares `ai: true`.

## 13. The landing page

`/`: the site's name and a line about it, then the apps as tiles — icon,
name, the app's accent colour — in the order `yenbot.yaml` lists them, then
alphabetically, and the inbox bell. A tap opens `/a/<id>/` with a transition
from the tile. Signed out: the same page with the tiles hidden and a single
"Sign in with Google". Editors also see draft apps, marked, and an editor page
(`/_editor`) listing every app's load status, its last publish, its jobs and
their last runs, and any manifest or check failure.

## 14. AI through Zen

**Nothing outside yenbot calls Zen.** An app calls `POST /api/v1/ai/chat` and
a workflow `ctx.ai.chat`; yenbot checks the app may use AI, then calls
zend's OpenAI-compatible endpoint itself and streams the answer back.

**zend is reached through the gateway**, `http://127.0.0.1` with `Host:
code.tokera.com`, so the cluster routes it to any ready zend member, not only
BlackWorld's. zend runs the request with `model: "passthrough"`
(`zend/src/api/chat.rs`, `passthrough.rs`): the app's messages as given, its
own system prompt, streaming, a JSON schema when asked — none of Zen's coding
projection.

**The credential is a service key, which is new.** There is no
service-to-service credential today: zend authenticates by the gateway's
identity headers, and the only key-like gateway credential is the operator's
`ops_key`, which is admin on every route and is not for an app
(`web/web.yaml`). The change: the gateway's config gains `service_keys:`, each
a key file, the identity it stands for (`service:yenbot`) and the routes it
may call (`/v1/chat/completions`, `/v1/models`). A request bearing the key is
forwarded with that identity; zend's `host/roles.yaml` grants
`service:yenbot` the user role. The key lives in yenbot's secrets
(`D:\yenbot\secrets\zen.key`), is read at start and never logged, and the
gateway accepts it only from its trusted backends.

"Unrestricted" means the app chooses its prompt, its length and its dials;
the limits are yenbot's: a per-person daily token budget in `yenbot.yaml`
(workflows count against the app's), every call recorded in `ai_calls`, and a
clear message in the app when the budget is spent.

## 15. Templates

`templates/` holds working apps to start from, each already passing the
checks in §16.3: a blank app; a list (add, edit, tick off, reorder); a journal
(dated entries, photos); a collection (cards with images, search, tags); a
planner (a week and a month view); a tracker (log a number, see a chart); an
assistant (an AI chat with a system prompt the app sets); reminders (records
with a due time, a notification at it); a daily digest (a morning workflow
that gathers and notifies); a form that saves answers and a page that sums
them up. A new app is a copy of the closest template, renamed.

## 16. Guides

### 16.1 Where they are

The guides are the product as much as the code is, and they are complete from
the first release. Two layers:

- **In the repository**, `guides/`: the whole manual, versioned with the code
  it describes — start here; how an app is laid out; the manifest; the
  framework (components, signals, pages, stores) with the conventions and why;
  the SDK; data; files; AI; jobs and schedules; workflows and `ctx`;
  notifications and the home-screen install; the design system and every
  component; templates; publishing, drafts and rolling back; what to do when
  it breaks; the review checklist.
- **In Zen's MCP**, one guide, `yen-apps` (`mcp/src/guides/yen-apps.md`): what
  the site is, that the manual is in the yenbot repository's `guides/`, and
  the order of work — read the guides, start from a template, build, run the
  checks, publish as a draft, show Yen, then go live. Every Claude session
  Yen opens, on any device, reads it first, as `guide_list` already asks.

### 16.2 How they are written

For a model building for someone who cannot check its work: every rule with
its reason, every API and every `ctx` call with a complete working example,
every component with its states, and every common request ("add a photo to
each entry", "let John see it too", "remind me the night before", "send me a
summary every Sunday") mapped to the exact pieces that do it.

### 16.3 The checks

`yenbot check <app>` — a subcommand of the binary, and the same check the
editor page shows — refuses an app that: has a manifest error; names a
workflow, collection or icon that is not there; uses a literal colour, font or
pixel size where a token exists; breaks the layout conventions (§8.3) —
a page or component outside its folder, a component reaching into a store,
two components in one file; has an image without `alt`; lacks an empty or
error state on a list; has no icon or one off the icon grid; loads anything
from outside the site; has a workflow that does not parse or exports no
function; or fails to load in a headless render at phone width. Claude runs it
before every publish, and the guides say so. A publish whose app fails the
check stays on its last good version and says why on the editor page.

## 17. tokera.com

The estate dropdown lists the sites in `ESTATE` (`web/src/site/tokera/page.rs`)
and its copy in `web/content/common/lib/estate.js`, kept equal by
`the_estate_list_matches_the_shared_module`. Yenbot is a fifth entry there,
with its icon under `web/content/common/brand/`, and in the footer.

## 18. What changes outside the new repository

| Where | Change |
|---|---|
| `web/web.yaml` | A `yenbot` site: `hosts: [yenbot.com.au]`, `api: [{prefix: /, upstream: "http://127.0.0.1:8096"}]`; `www.yenbot.com.au` as a redirect site. |
| `web/src/auth/` | Sign-in per cookie domain (§12). |
| `web/src/` and `web.yaml` | `service_keys:` (§14). |
| `host/roles.yaml` | `service:yenbot` as a user; Yen's Google account as a person. |
| Google Cloud console | `https://yenbot.com.au/auth/callback` as a redirect URI on the existing client. John does this. |
| `keep/keep.yaml` | The `yenbot` service: `stable: keep/stable/yenbot.exe`, `args: [--config, D:/yenbot/yenbot.yaml]`. |
| `motherd/agent.yaml`, `D:\prog\workspace.yaml` | The `yenbot` repository, for Zen and the agent. |
| `service_deploy` | Learns `yenbot`: its repository and its package. |
| Zen grants | Yen's account may work on and deploy the yenbot repository; not candle. |
| `mcp/src/guides/` | `yen-apps.md` (§16.1). |
| `web/src/site/tokera/page.rs`, `estate.js` | The estate entry (§17). |
| `docs/deployment.md` | The service, its port, its paths and its launch line. |

## 19. Open questions

- **The name on the site** — "Yenbot", or something Yen chooses.
- **Who besides Yen and John signs in** — family, friends; and whether any app
  should ever be public.
- **Where backups leave the machine** — WhiteWorld, a cloud bucket, or Yen's
  Google Drive.
- **The AI budget** — tokens per person per day, and per app for workflows.
- **Pictures** — whether apps may call Zen's image model as well
  (`image_generate`); the same service key, one more route.
- **Email** — whether notifications should also go by email, for a person
  who never installs the site; it needs a sending account.

## 20. Order of work

1. The gateway: sign-in per domain, service keys, the yenbot site and route.
2. The repository and the binary's skeleton: serving, identity, the landing
   page, the store and its migrations, the watcher and the publisher, keep and
   `INFO serving`.
3. The API, with tests against a real SQLite file.
4. The frontend framework: `yen.js`, the vendored modules, routing, stores.
5. The design system, its catalogue page and the components.
6. Jobs and workflows: the scheduler, the record-driven schedules, QuickJS and
   `ctx`.
7. Notifications: the service worker, Web Push, the inbox.
8. AI through Zen.
9. Templates, the checks, the guides, the MCP guide.
10. Zen access for Yen, `service_deploy`, the estate entry, `deployment.md`.
11. Yen's first app, built by her in claude.ai from the guides alone — the
    test of everything above.
