# The `web` gateway

Status: **built.** One crate is the front door for the whole estate: it serves
tokera.com, hosts the two consoles' files, forwards their APIs to the machines
that run them, and owns sign-in for all three. `web/src/`, `web/web.yaml`,
`web/content/`.

---

## 1. What it is

Three ways to run one crate. A route's `upstream` picks between the first two;
a flag picks the third.

| | Embedded | Gateway | `--authoritative` |
|---|---|---|---|
| Who runs it | `npcd`, `zend` — as a library | the DMZ box — the `web` binary | the `web` binary, one flag |
| Content | compiled in (`include_dir!`), or `--content <dir>` | read from disk | read from disk |
| `/v1`, `/ws` | `upstream: local` → the `Router` this process supplied | `upstream: <url>` → forwarded | every upstream forced local → `web::mock` |
| For | one binary that serves its console *and* its API, testable alone | top-level domains land here; daemons live elsewhere | the whole estate on a laptop, no daemons |

Promoting a site from local to remote is one line of YAML. Nothing above the
config knows which is in force, so iterating on a daemon as a single binary and
deploying it behind the gateway are the same build.

## 2. The deployment

```
                         ┌──────────────────────────────┐
   tokera.com     ──────▶│                              │
   battlecities.net      │   web  (DMZ)                 │
                         │   • tokera.com, rendered      │
   bot.tokera.com ──────▶│   • console files for both    │──▶ 192.168.0.6:8081  npcd
   code.tokera.com ─────▶│   • sign-in for the estate    │──▶ 192.168.0.5:8081  zend
                         └──────────────────────────────┘
```

The daemons listen on private addresses. That is not incidental — it is the
trust boundary §5 depends on.

## 3. Routing

```
Host header  ->  site      exact host, then *.wildcard, then the default
path prefix  ->  upstream  longest prefix wins; `exact: true` beats any prefix
otherwise    ->  a file, from that site's roots in order
```

**`roots` is a list, so the framework has no owner.** `content/common` holds the
router, signals, DOM helpers, live-update discipline and the base stylesheet;
`content/npcd` holds that product's pages and palette. `/lib/dom.js` falls
through to common while `/pages/roster.js` does not — one URL tree assembled
from two directories, and a second product gets the framework by naming the
same second root rather than by either depending on the other.

**`exact: true`** exists for tokera.com, whose home page is generated while
`/site.css` is a file. A bare `/` prefix would swallow every asset on the site.

**Extensionless paths fall back; assets do not.** `/npc/42` with no matching
file serves `index.html`, so a hash router owns navigation and a deep link
survives a refresh. Anything with an extension 404s instead — a missing `.js`
served as HTML surfaces as `Unexpected token '<'` a long way from its cause.

**A down daemon is a page, not a stack trace.** Each upstream has its own
backoff — 250 ms doubling to 10 s — and requests inside the window fail fast
rather than queueing behind another connect timeout. A browser gets a
self-contained error page (no stylesheet, no script: the thing that would serve
them is what is down) carrying `<meta http-equiv="refresh">` set to the retry
delay, so recovery looks automatic. An API caller gets the same failure as the
ordinary `{error, detail, field}` object. One probe is released when the window
expires; nobody restarts anything. Reaching the upstream *at all* counts as
success — a 500 from a live daemon is the daemon's answer to pass through, not
a reason to take the route out of service.

**Upgrades are tunnelled.** On a `101` the gateway stops speaking HTTP and joins
the two upgraded connections, so `/ws/logs` and `/ws/events` behave identically
proxied and direct. `Connection` is hop-by-hop and stripped from every other
response, but it is restated on a `101` — a handshake without it is rejected by
the client as malformed, which is a bug that only shows up through the proxy.

## 4. tokera.com

The one site with no daemon behind it. `web` is its backend, because the whole
of it is reading markdown off a disk and wrapping it in a shell; a service to
do that would be a process to deploy and monitor in exchange for nothing.

Everything is **server-rendered**. Papers are long documents people deep-link
into, quote, print and read on bad connections, and a page that must execute a
bundle before it shows a sentence fails all four. The only script on the page is
the few lines that swap "Sign in" for your name.

| Path | What |
|---|---|
| `/` | Home — the engine, Zen Code, Battle Cities |
| `/blog`, `/blog/{slug}` | One markdown file per post in `content/tokera/blog/` |
| `/papers`, `/papers/{slug}` | Rendered from the working documents in `docs/` |

### Papers are not copied

`papers: "../docs"` in the site table names a directory documents are read from
**live**, so a published paper is never a stale duplicate of the real one and
publishing is not a step anyone has to remember.

`content/tokera/papers.yaml` is what makes that safe. Only documents it names
are reachable, so pointing at `docs/` publishes the papers rather than the whole
design directory, and `source` must be a plain file name — a manifest entry
cannot reach out of the directory. The manifest also carries title, authors and
blurb: metadata that has no business inside a document still being edited, and
which means revising a working document's heading cannot rename a published
paper.

`papers:` is deliberately **not** a content root. A root would make every
document in `docs/` fetchable as a raw file.

### Blog posts are their own index

A post is one file with YAML front matter (`title`, `date`, `summary`, `tags`,
`draft`). The index is the directory, so publishing is `git add` and nothing
else — there is no manifest to fall out of step with, and a post cannot be
written and then forgotten because someone missed a second edit. Ordering is by
the front matter's `date`, not mtime: a typo fix should not move a two-year-old
post to the top.

### Markdown

`pulldown-cmark` with tables, footnotes, strikethrough and task lists. Smart
punctuation is **off**: these documents contain code and file paths, where
turning `--` into an en dash silently corrupts the text.

Three transforms on the way past, which is why it is an event walk and not a
call to `push_html`:

- **Headings get ids** and a permalink anchor. A paper's headings are its
  addresses; the slug is a pure function of the heading text so a re-render
  cannot break a shared link.
- **Maths becomes MathML.** The parser hands over `InlineMath`/`DisplayMath`
  events, which is what makes it safe — a `$` inside a code span never reaches
  the converter, because it never becomes a maths event.
- **The leading `# Title` is lifted out**, since the page shell renders it.

**Why MathML and not KaTeX.** KaTeX means a JavaScript bundle and sixty-odd font
files, typesetting in every reader's browser, and a reflow after first paint.
MathML is laid out natively by every current browser, so converting once on the
server costs the reader nothing and the repo no build step — the same reason the
consoles have no bundler.

`latex2mathml` does not know `\mathcal`, and the theorem the site exists to
publish states its bound over `\mathcal{W}`. It is expanded to the Unicode
mathematical script letter first, which is not a workaround but what the command
means. A construct that still will not convert renders as its literal LaTeX
rather than failing the page — a reader can read `\alpha=2.0`. The site's tests
assert **zero** fallbacks across the published papers, so that safety net cannot
quietly become the normal case.

Rendering a 170 KB paper is not free, so results are cached keyed by the file's
mtime: the first reader after an edit pays and touching the file is all it takes
to invalidate.

## 5. Sign-in, and why the gateway owns it

One Google account reaches tokera.com, code.tokera.com and bot.tokera.com. The
mechanism is deliberately boring:

1. The gateway runs the OIDC authorization-code exchange with PKCE.
2. On success it sets a session cookie **on the parent domain** —
   `Domain=.tokera.com`.
3. The browser presents that cookie to every host under the domain by itself.

That is the whole of the single sign-on. No cross-origin handshake, no token in
a URL fragment, no second login page, and the subsites are not involved in it.

```
GET  /auth/login?next=…   → the provider
GET  /auth/callback       → sets the session cookie, returns to `next`
POST /auth/logout         → clears it
GET  /auth/me             → who the browser is, or that nobody is
```

These are mounted **ahead of site routing**, so a site that proxies `/` — zend,
until its console is split out — cannot swallow them, and every hostname can ask
who you are.

### The daemons never authenticate anyone

They read the identity the gateway states, on `X-Tokera-User` / `-Email` /
`-Name` / `-Picture`. The gateway **strips those headers off every inbound
request** before setting its own: anything a client sent under those names is a
forgery attempt by construction, since only the gateway may set them.

That is sound exactly as far as the daemons are unreachable except through the
gateway, which is why they listen on private addresses. A daemon that would
rather verify than trust can check `X-Tokera-Assertion` — the signed session
token itself — against the same key the gateway signs with.

This supersedes `npc_api_gui_design.md` §8.1's per-daemon OAuth: `npcd` does not
run its own dance, and its `/v1/me` reports the identity it was handed.

### Decisions worth recording

**Stateless sessions.** A server-side session table would have to be shared by
three machines. The token carries the claims and an HMAC over them, so
validating is arithmetic rather than a lookup, and signing everyone out is one
operation: change the key.

**No JWT header.** There is one algorithm, and a field naming it is the part of
JWT that has caused the most trouble — `alg: none` is not expressible here.

**No JWKS fetch, no RSA verification.** The `id_token` is read from the token
endpoint over a TLS connection this process opened to the configured host, using
a client secret only this process holds. OIDC Core §3.1.3.7 permits using it
without validating the signature, and the alternative is a JWKS cache and a
key-rotation story to re-derive a fact TLS already established.

**No server-side state between the two requests.** `state`, `nonce` and the PKCE
verifier live in a short-lived signed cookie, so a redirect that returns to a
restarted gateway still completes.

**`Secure` is derived, not configured.** From the redirect URI's scheme: an https
deployment gets secure cookies without anyone remembering to ask, and a local
http provider still works because the same rule says no. A `secure_cookies: bool`
would eventually be deployed as `false`.

**Both secrets are files.** `session_secret_file` and `client_secret_file`, never
inline values, so the site table can be read and committed without carrying a
secret. The process **refuses to start** if either is missing — a gateway that
comes up with broken sign-in and reports it per request is worse than one that
does not come up. Omit the whole `auth:` block to run without sign-in; `/auth/me`
then reports `configured: false`, which is distinct from being signed out, and
the UI shows no button rather than one that cannot work.

For local development:

```sh
head -c 48 /dev/urandom | base64 > /path/to/session.key
# client secret from the Google Cloud console, OAuth 2.0 Client ID
# and register the redirect_uri EXACTLY, or Google answers redirect_uri_mismatch
```

### `next` is confined to the estate

An unchecked `next` turns the login endpoint into an open redirect, which is how
a phishing link gets to wear the real domain. Only a same-origin path or a host
at or under the cookie domain is accepted — `https://tokera.com.evil.example/`
is refused.

## 6. Mocks, at two depths

Neither is ever selected automatically. `?mock=1` and `--authoritative` are both
things a person types; a mock must never quietly substitute for a backend that
failed.

| | Replaces | Exercises | Used by |
|---|---|---|---|
| `?mock=1` (`api.mock.js`) | the network | the GUI only | Playwright, offline UI work |
| `web --authoritative` (`web::mock`) | the daemon | routing, error pages, the ws tunnel, real sockets | console development against real HTTP |

The server-side mock lives in `web/src/mock/npcd/`, beside the files it serves,
for the same reason `api.mock.js` does: a console and its fixtures ship
together. `npcd` calls the same router today, because a mock daemon is all
`npcd` is until there is an engine to put behind it. When there is, `npcd` grows
its own `api.rs` against the same routes and the fixture stays where it is.

A site with no mock — zend — says so at startup and returns a `502` naming the
site, rather than serving an empty page that looks like a bug in the GUI.

## 7. Layout

```
web/
  web.yaml                  the site table — the whole deployment
  src/
    config.rs               sites, routes, backoff, auth config
    server.rs               host → site → (local | proxy | files); the Builder
    proxy.rs                streaming forward, ws tunnel, identity headers
    health.rs               per-upstream backoff and single-probe recovery
    errors.rs               one failure, rendered as a page or as JSON
    content.rs              layered roots, disk or embedded; path safety
    markdown/               render, maths, front matter, heading slugs, cache
    auth/                   session, OIDC exchange, cookies
    site/tokera/            home, blog, papers, page shell
    mock/npcd/              the console's fixture daemon
  content/
    common/                 framework + base.css, shared by every site
    tokera/                 site.css, papers.yaml, blog/*.md
    npcd/                   the NPC console
```

## 8. Open

- **zend is still a pure gateway.** It embeds and serves its own assets, so it
  has no `roots` here and no mock. Splitting its console the same way as npcd's
  is `roots: ["content/zend", "content/common"]` plus deleting the embedded copy.
- **battlecities.net lands on tokera.com** until it has a page of its own.
- **`cache_control` is `no-store`** while iterating. It wants a real max-age once
  assets are content-hashed.
