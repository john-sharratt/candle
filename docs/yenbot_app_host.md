# Yenbot: an app host for Yen

**Status: design, not built.** This document is authoritative for the yenbot
service, its repository, its place behind the gateway and the changes the
gateway, zend and tokera.com need for it.

## 1. What it is

`yenbot.com.au` is where Yen makes and keeps her own apps: a habit tracker, a
recipe book, a planner for the kids, a gallery, a tool that asks the model
something. She signs in with Google, sees her apps as icons on one page, and
taps one to open it.

She does not write code. She works in claude.ai, which reaches our machines
through the Zen MCP connector. She describes what she wants, Claude builds it
in the yenbot repository through Zen, and it is live on her site a few seconds
later. So the platform has to make the right thing the easy thing for a model
writing code for someone who cannot review it: every app is built on one
framework, from one design system, documented end to end, and checked before
it goes live.

## 2. Principles, and what each one decides

| Principle | Decision |
|---|---|
| HTML changes go live quickly | Apps are files the binary serves from disk and reloads on change (§5.3). Publishing an app is a git push; no build, no restart. Only platform changes rebuild the binary. |
| Many apps, like tokera.com | One site, many apps under `/a/<app>/`, each a folder with a manifest (§5). |
| One common framework | Every app loads the same SDK, components and stylesheet from `/_/` (§8). An app is markup and a little script on top of it. |
| One Rust binary that serves its pages | `yenbot.exe` embeds the platform: the landing page, the SDK, the design system, the fonts and icons, the API, the store (§4). Apps come from a directory beside it. |
| A common API for every app | `/api/v1/…`: identity, data, files, events, AI, settings (§6). |
| A state store inside the binary | SQLite, compiled in (`rusqlite` with `bundled`), one file on disk, migrations embedded (§7). |
| Google sign-in, like tokera.com | The gateway signs people in and forwards who they are; yenbot never sees a Google secret (§9). |
| A landing page of apps | `/` lists the apps the person may open, as icons; signed out it shows the site and a sign-in button (§10). |
| AI through Zen, from this site only | The browser calls yenbot; yenbot calls zend with a service key the browser never holds (§11). |
| Yen is not a coder | The framework, templates, guides and checks are all there from the first day (§8, §12, §13). |
| Styling must be amazing, with attention to detail | One design system with fixed tokens, a living catalogue of every component, and a review checklist every app passes (§8.2, §13.3). |
| tokera.com links to it | An entry in the estate dropdown (§14). |

## 3. Where it runs

```text
browser ──TLS──► Cloudflare (yenbot.com.au, proxied) ──HTTP :80──► web (gateway, BlackWorld .5)
                                                                     │  signs in (Google),
                                                                     │  forwards X-Tokera-* identity
                                                                     ▼
                                                        127.0.0.1:8096  yenbot.exe  (keep service)
                                                          │  SQLite: D:\yenbot\data\yenbot.db
                                                          │  apps:   D:\yenbot\live\apps\
                                                          ▼
                                       gateway, Host code.tokera.com, service key ──► zend (cluster)
```

- **BlackWorld only**, beside the gateway, supervised by keep as the service
  `yenbot` (`keep/stable/yenbot.exe`). It binds **loopback**, so nothing on
  the LAN reaches it except through the gateway, and it believes the gateway's
  identity headers from loopback alone — the model `mcp` already follows
  (`mcp/mcp.yaml`, `mcp/src/believed.rs`).
- **It is not a cluster member.** It holds its own store and has no replicas;
  BlackWorld down means yenbot down, as with any keep service that is not
  clustered (`docs/mcp_app.md`).
- **The port** is 8096 (8081 motherd, 8092 keep, 8095 mcp are taken); it is
  set in `yenbot.yaml`, and the build checks it is free at start.
- **DNS** already points `yenbot.com.au` and `www` at `tokera.com`, proxied
  (Cloudflare, checked 2026-11-06). Today they fall through to the default
  site and show Tokera.

## 4. The repository

A new repository, `john-sharratt/yenbot`, added to zend's workspace
(`D:\prog\workspace.yaml`) and to the agent's checkouts (`motherd/agent.yaml`,
`push: true`), so Zen reads, edits, builds, commits and pushes it as it does
candle. Two working copies on BlackWorld, never the same one:

- `D:\prog\yenbot` — zend's working tree, where Claude's edits land.
- `D:\yenbot\live` — what the site serves, fast-forwarded from `origin/main`
  (§5.4). Nothing edits it by hand.

```text
yenbot/
  Cargo.toml                  the one binary
  src/                        the platform (one concern per file)
    main.rs  config.rs  serve/  api/  store/  ai/  apps/  identity/
  platform/                   embedded into the binary at build time
    landing/                  the home page
    sdk/yen.js                the SDK every app imports
    ui/                       design system: tokens.css, yen.css, components/*.js
    icons/  fonts/
  apps/                       one folder per app — served from disk, hot
    <app-id>/
      app.yaml                manifest (§5.1)
      index.html  app.js  style.css  icon.svg
  templates/                  starting points for a new app (§12)
  guides/                     everything a writer of an app needs (§13)
  yenbot.yaml                 the service's config
  tests/
```

The platform is embedded with `include_dir!`, so the binary is whole on its
own; `apps/` is read from the live directory at runtime. A platform change is
a rebuild; an app change never is.

## 5. Apps

### 5.1 The manifest

```yaml
id: recipes                    # the folder name, the URL, the data namespace
name: Recipe Book
description: Family recipes, with photos and a shopping list.
icon: icon.svg                 # 512×512, from the icon rules in guides/
colour: "#E07A5F"              # the tile's accent, one of the palette's
status: live                   # live | draft (draft: shown only to editors)
access:                        # who may open it; default: everyone signed in
  - yen@…                      #   Google addresses
  - john@…
collections:                   # what it stores (§7.2)
  recipes: shared              #   shared: every person with access sees it
  favourites: private          #   private: each person sees their own
ai: true                       # it may call /api/v1/ai
```

The platform refuses an app whose manifest does not parse, names an icon that
is not there, or declares a collection twice — at load, with the reason on the
editor's page (§10), and the app stays on its last good version.

### 5.2 Serving

`/a/<id>/…` serves the app's folder. The platform injects nothing into an
app's HTML: an app opts in by loading `/_/yen.js` and `/_/yen.css`, which
every template already does. Static files are served with an `ETag` from
their content hash and `Cache-Control: no-cache`, so a changed file is fetched
on the next load and an unchanged one costs a 304.

### 5.3 Hot reload

A watcher on `apps/` reloads a changed manifest and drops its cached files
within a second. A page open in a browser reloads itself when its app
changes: `yen.js` keeps an `EventSource` on `/api/v1/events/_reload` while the
tab is visible. Yen sees a change Claude made without touching her phone.

### 5.4 Publishing

Publishing is a push to `main`. yenbot fetches `origin/main` every 10 s in the
background, fast-forwards `D:\yenbot\live` when it moved, and the watcher does
the rest; the fetch runs off every request path. A push that rewrites history
is refused (fast-forward only) and logged.

Drafts: an app with `status: draft` is live for editors only, so a new app can
be tried on the real site before Yen shows anyone. Rolling back is a revert
commit; every version of every app is in git.

Platform releases go the ordinary way: an agent build on BlackWorld, staged to
keep, tried, and kept when it prints `INFO serving` (`CLAUDE.md`, "Releasing
and Testing Locally"; keep's trial in `keep/src/main.rs`). `service_deploy`
learns the `yenbot` service and its repository for this (§15).

## 6. The API

Every route is under `/api/v1`, answers JSON, and acts as the signed-in
person. The app is named by the path or the `X-Yen-App` header `yen.js` sets;
a person without access to the app gets 403.

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
| `POST /ai/chat` | A chat completion, streamed or whole, optionally constrained to a JSON schema (§11). |
| `GET/PUT /settings/<app>` | A small per-app, per-person settings object. |

`yen.js` wraps all of it, so an app never builds a URL:

```js
import { app } from "/_/yen.js";
const recipes = app.collection("recipes");
await recipes.add({ title: "Pho", serves: 4 });
recipes.watch(list => render(list));            // live
const reply = await app.ai.chat("Suggest a side dish for pho");
```

## 7. The store

### 7.1 SQLite, compiled in

`rusqlite` with the `bundled` feature: SQLite is built into the binary, the
database is one file (`D:\yenbot\data\yenbot.db`) in WAL mode, and nothing is
installed beside it. It is the most standard embedded database there is, every
tool reads it, and migrations are plain SQL: `store/migrations/NNNN_*.sql`,
embedded, applied in order at start against `PRAGMA user_version`. A migration
never edits an earlier one.

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
in the request. An app that grows a field it filters on often gets an
expression index from its manifest (`index: [recipes.title]`).

Platform tables beside it: `files` (blob metadata; bytes on disk under
`data\files\`), `settings`, `ai_calls` (what was asked, by whom, how long,
how many tokens), `events` (the change feed's recent tail).

### 7.3 Backups

Every night at 03:00 `VACUUM INTO` writes `data\backups\yenbot-YYYY-MM-DD.db`,
seven kept; the files directory is copied beside it. Where the copy goes off
the machine is open (§16).

## 8. The framework

### 8.1 No build step

Apps are plain HTML, CSS and ES modules. There is no bundler, no npm, no
compile: what Claude writes is what the browser runs, and what Yen sees an
instant after the push. The platform's own JavaScript is the same — modern
browsers run it as written.

### 8.2 The design system

The look is the platform's, not each app's. `/_/yen.css` carries:

- **Tokens** (`tokens.css`): one palette with light and dark values, a type
  scale, a spacing scale, radii, shadows, motion durations and easings. Apps
  use tokens, never literal colours or pixel sizes; the check in §13.3 flags a
  literal.
- **Type**: a self-hosted text face and a display face, embedded in the
  binary, with set line heights and measures.
- **Components**, as custom elements in `components/*.js`: app shell (header,
  back, title), button, icon button, text field, select, date picker, toggle,
  card, list and list item, grid, tabs, sheet, dialog, toast, empty state,
  skeleton, avatar, tag, progress, image picker, photo grid, chart, and an AI
  chat panel. Each has its states designed — hover, focus, pressed, disabled,
  loading, error — and is keyboard- and screen-reader-correct.
- **Icons**: one icon set, embedded as an SVG sprite.
- **Motion**: page and sheet transitions, list insertions, a press response;
  all honour `prefers-reduced-motion`.

`/_design` is a page on the site listing every token and every component in
every state, light and dark, at phone and desktop width. It is the
reference Claude builds against and the page Yen points at when something
looks wrong.

### 8.3 What every app gets without asking

The shell (title, back to the landing page, the person's avatar), sign-in
handled, light and dark following the device, phone-first layout, offline
read of what was last loaded, loading and empty and error states in every
component, the change feed, uploads with thumbnails, AI with streaming, and a
"something went wrong" panel that says what and offers a retry.

## 9. Sign-in and access

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
names the **editors** — Yen and John — who see drafts and the editor page. A
signed-in stranger sees no apps and costs nothing: the AI route requires
access to an app that declares `ai: true`.

## 10. The landing page

`/`: the site's name and a line about it, then the apps as tiles — icon,
name, the app's accent colour — in the order `yenbot.yaml` lists them, then
alphabetically. A tap opens `/a/<id>/` with a transition from the tile.
Signed out: the same page with the tiles hidden and a single "Sign in with
Google". Editors also see draft apps, marked, and an editor page (`/_editor`)
listing every app's load status, its last publish, and any manifest error.

## 11. AI through Zen

**The browser never calls Zen.** An app calls `POST /api/v1/ai/chat`; yenbot
checks the person may use AI in that app, then calls zend's
OpenAI-compatible endpoint itself and streams the answer back.

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
the limits are yenbot's: a per-person daily token budget in `yenbot.yaml`,
every call recorded in `ai_calls`, and a clear message in the app when the
budget is spent.

## 12. Templates

`templates/` holds working apps to start from, each already passing the
checks in §13.3: a blank app; a list (add, edit, tick off, reorder); a journal
(dated entries, photos); a collection (cards with images, search, tags); a
planner (a week and a month view); a tracker (log a number, see a chart); an
assistant (an AI chat with a system prompt the app sets); a form that saves
answers and a page that sums them up. A new app is a copy of the closest
template, renamed.

## 13. Guides

### 13.1 Where they are

The guides are the product as much as the code is. Two layers:

- **In the repository**, `guides/`: the whole manual, versioned with the code
  it describes — start here; how an app is laid out; the manifest; the
  SDK; the data API; files; AI; the design system and every component;
  templates; publishing and drafts; rolling back; what to do when it breaks;
  the review checklist.
- **In Zen's MCP**, one guide, `yen-apps` (`mcp/src/guides/yen-apps.md`): what
  the site is, that the manual is in the yenbot repository's `guides/`, and
  the order of work — read the guides, start from a template, build, run the
  checks, publish as a draft, show Yen, then go live. Every Claude session
  Yen opens reads it first, as `guide_list` already asks for.

### 13.2 How they are written

For a model building for someone who cannot check its work: every rule with
its reason, every API with a complete working example, every component with
its states, and every common request ("add a photo to each entry", "let John
see it too", "make it remind me") mapped to the exact pieces that do it.

### 13.3 The checks

`yenbot check <app>` — a subcommand of the binary, and the same check the
editor page shows — refuses an app that: has a manifest error; uses a literal
colour, font or pixel size where a token exists; has an image without `alt`;
lacks an empty or error state on a list; has no icon or an icon off the icon
grid; loads anything from outside the site; or fails to load in a headless
render at phone width. Claude runs it before every publish, and the guides say
so. A publish whose app fails the check stays on its last good version and
says why on the editor page.

## 14. tokera.com

The estate dropdown lists the sites in `ESTATE` (`web/src/site/tokera/page.rs`)
and its copy in `web/content/common/lib/estate.js`, kept equal by
`the_estate_list_matches_the_shared_module`. Yenbot is a fifth entry there,
with its icon under `web/content/common/brand/`, and in the footer.

## 15. What changes outside the new repository

| Where | Change |
|---|---|
| `web/web.yaml` | A `yenbot` site: `hosts: [yenbot.com.au]`, `api: [{prefix: /, upstream: "http://127.0.0.1:8096"}]`; `www.yenbot.com.au` as a redirect site. |
| `web/src/auth/` | Sign-in per cookie domain (§9). |
| `web/src/` and `web.yaml` | `service_keys:` (§11). |
| `host/roles.yaml` | `service:yenbot` as a user; Yen's Google account as a person. |
| Google Cloud console | `https://yenbot.com.au/auth/callback` as a redirect URI on the existing client. John does this. |
| `keep/keep.yaml` | The `yenbot` service: `stable: keep/stable/yenbot.exe`, `args: [--config, D:/yenbot/yenbot.yaml]`. |
| `motherd/agent.yaml`, `D:\prog\workspace.yaml` | The `yenbot` repository, for Zen and the agent. |
| `service_deploy` | Learns `yenbot`: its repository and its package. |
| Zen grants | Yen's account may work on and deploy the yenbot repository; not candle. |
| `mcp/src/guides/` | `yen-apps.md` (§13.1). |
| `web/src/site/tokera/page.rs`, `estate.js` | The estate entry (§14). |
| `docs/deployment.md` | The service, its port, its paths and its launch line. |

## 16. Open questions

- **The name on the site** — "Yenbot", or something Yen chooses.
- **Who besides Yen and John signs in** — family, friends; and whether any app
  should ever be public.
- **Where backups leave the machine** — WhiteWorld, a cloud bucket, or Yen's
  Google Drive.
- **The AI budget** — tokens per person per day.
- **Pictures** — whether apps may call Zen's image model as well
  (`image_generate`); the same service key, one more route.
- **Notifications** — a reminder app wants to reach her phone: web push, or
  email. Not in the first version.

## 17. Order of work

1. The gateway: sign-in per domain, service keys, the yenbot site and route.
2. The repository and the binary's skeleton: serving, identity, the landing
   page, the store and its migrations, the watcher and the publisher, keep and
   `INFO serving`.
3. The API and `yen.js`, with tests against a real SQLite file.
4. The design system, its catalogue page and the components.
5. AI through Zen.
6. Templates, the checks, the guides, the MCP guide.
7. Zen access for Yen, `service_deploy`, the estate entry, `deployment.md`.
8. Yen's first app, built by her in claude.ai from the guides alone — the
   test of everything above.
