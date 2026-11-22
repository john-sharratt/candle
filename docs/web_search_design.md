# The `web_*` tools: `web_search` and `web_fetch`

**Status: built.** `zend-tools` registers two web tools, both declaring the
`network` grant (`registry.rs`):

| Tool | Source | Operation |
|---|---|---|
| `web_search` | `zend-tools/src/tools/web_search.rs` | Query the Tavily search API → ranked results (title, URL, snippet, score) |
| `web_fetch` | `zend-tools/src/tools/web_fetch.rs`, `zend-tools/src/public_fetch/` | Fetch one public URL locally → the page as clean text |

The model chooses between them by what it knows: a topic with no URL goes to
`web_search`; a known URL goes to `web_fetch`. Each tool is one `Tool`
implementation in the `zend-tools` three-type pattern (a `Request` that derives
`Deserialize`, `JsonSchema` and `Validate`; a `Serialize` `Response`; an error
enum implementing `ToolError`).

---

## `web_search`

A query goes to Tavily's search endpoint (`https://api.tavily.com/search`,
`search_depth: "basic"`, `include_answer: false`) through the context's shared
HTTP client (30-second timeout, reached only through `ToolContext::http`, which
refuses a context without the `network` grant). Tavily's `results` array is
mapped onto the tool's response, one entry per result.

### Where the key comes from

The Tavily key is `tavily_api_key` in the daemon's secrets document
(`~/.zend/secrets.yaml` by default, `Secrets::DEFAULT_RELATIVE_PATH`), read once
at startup by `Secrets::load` and reached through `ctx.secrets`. It is not read
from the process environment: the key would then have to be exported by
whatever launches the daemon, and every child process would inherit it. The
document rejects unknown fields, so a misspelt `tavily_key:` fails at load
rather than reading as an unset key. A blank value reads as unset.

With no key the call fails with `search_unavailable` and a message that names no
path (the model reads the message; where the daemon's secrets live is the
operator's business, and the daemon logs the file it read at startup).

### Arguments

| Argument | Type | |
|---|---|---|
| `query` | string | Required, non-empty. |
| `max_results` | integer 1–10 | Optional; default 5. |

### Response

```json
{
  "searched_on": "2026-09-03",
  "results": [
    {"title": "...", "url": "https://...", "snippet": "...", "score": 0.91}
  ]
}
```

`snippet` is Tavily's `content` field; `score` is its relevance score (0.0 when
absent). `searched_on` is today's date (UTC, `YYYY-MM-DD`), carried on every
search and serialised before the results so it is read first. The model has no
clock of its own and dates the world by its training: asked for the latest Rust
release it searched "latest stable Rust release 2025" a year late and reported a
release the results themselves showed had been superseded. The tool description
tells the model to judge "latest" against `searched_on`, to prefer the newest
result, and to put no year in a query the user did not give one.

### Errors

One code: `search_unavailable`, with the reason as detail — no key configured, a
transport failure, a non-success HTTP status from Tavily (`HTTP <status>`), or an
unreadable response body. A search is not replay-safe (the default `Replay::Unsafe`):
running it again spends another API call.

---

## `web_fetch`

Fetches one `http://` or `https://` URL from the machine the daemon runs on and
returns its main content as text. The fetch is local, so it must not become a way
into that machine's network: keep's API and every daemon behind the gateway
believe identity headers from loopback, and a fetch runs inside that network.
`zend-tools/src/public_fetch/` is the guard, and every address a fetch may
connect to is checked there.

### The guard (`public_fetch`)

| Module | |
|---|---|
| `address` | which addresses are public, and resolving a host to them |
| `client` | an HTTP client pinned to a checked address, its resolver and redirects guarded |
| `body` | a response's bytes, read only up to a cap |
| `error` | why a fetch did not happen |

- **The scheme must be `http` or `https`**, and a URL that names a local target
  literally (`localhost`, `127.0.0.1`, `::1`, or a private address literal) is
  refused before anything resolves.
- **Every address is public.** `is_private_ip` refuses loopback, RFC 1918
  private, link-local (including the `169.254.169.254` metadata address),
  broadcast, documentation, unspecified, `0.0.0.0/8`, carrier-grade NAT
  (`100.64.0.0/10`), IPv6 unique-local (`fc00::/7`) and link-local (`fe80::/10`),
  and an IPv4 address carried in IPv6 (`::ffff:127.0.0.1`) is judged as the IPv4
  address it is.
- **The host is resolved once, and every address it resolves to must be public**
  (not just the first, because the connection may use any of them). The client is
  pinned to the address checked, so a host cannot answer the check with a public
  address and the connection with a private one.
- **Every other name the client looks up goes through the same check.** The
  client's resolver is `PublicOnly`, so a redirect to a name is checked where it
  is connected to; this closes DNS rebinding.
- **Each redirect is checked again**, at most five of them. An address-literal
  redirect target is resolved by nothing, so the redirect policy is its only
  check.
- **A body is read only up to its cap, and abandoned past it** (`read_within`,
  `read_head`; used by the file transfers that share this module).

### Arguments

| Argument | Type | |
|---|---|---|
| `url` | string | Required, non-empty; must start with `http://` or `https://`. |
| `max_tokens` | integer 500–16000 | Optional; default 4000. Content is cut at `max_tokens × 4` bytes, on a character boundary. |

### Response

```json
{
  "url": "https://example.com/post",
  "final_url": "https://example.com/post/",
  "title": "...",
  "content": "...",
  "truncated": false,
  "images": 2
}
```

`final_url` is the URL after redirects. `truncated` is true when the content was
cut at the cap. `images` counts the pictures the page showed.

### How a page becomes text (`web_fetch/html_text.rs`)

A GET with a 30-second timeout and the `User-Agent: Mozilla/5.0 (compatible;
zend-tools/0.1)`; a non-success status is an error. The body is read as text and
reduced to what a reader takes from the page:

- Every block of the body is kept — paragraphs, list items, table rows, figure
  captions, preformatted text — each on lines of its own, so a fact in a list or a
  table reaches the reader as one in a paragraph does.
- What is written to change a number's meaning keeps it: a superscript is `^`, a
  subscript `_` (`10^-3`, not `10-3`), and math is its TeX `alttext` rather than
  the digits of its MathML nodes run together.
- Chrome — scripts, styles, navigation, footers, buttons — is left out. A
  `<header>` or `<form>` is kept, since an article's title is in its header and a
  whole page may sit in one form.
- **A picture is marked where it stood, and counted.** Its pixels are not text, so
  a table or chart published as an image never reaches the reader, and a reader
  told nothing of it reads "the page has no table" as "none was published". Each
  `<img>` is written as `[image: <alt>]` (or `[image]`) in its place, and
  `images` says how many there were. The tool description tells the model to say
  what it could not read, never that it does not exist.

### Errors

| Code | Meaning |
|---|---|
| `url_blocked` | The scheme is not HTTP(S), the URL names a local target, or the host (or a redirect target) resolves to a private or local address. |
| `fetch_failed` | Transport failure, an unparseable URL, or a failed name lookup. |
| `http_error` | The server answered with a non-success status (`HTTP <status>: <reason>`). |
| `too_large` | A body ran past its cap and was abandoned (used by the transfers that share `public_fetch`). |

`web_fetch` is replay-safe (`Replay::Safe`): it retrieves a page by GET and
nothing at the far end changes. It sends no credentials; for authenticated requests the model uses `http_session_*`, and to land a URL's
bytes in a repository file without passing through the conversation it uses
`file_download` (`docs/tool-system.md`).

---

## Replaced design

The first design for this family ran every web tool on the Gemini API, with
nothing fetched locally:

- `web_search` as one grounded `generateContent` call (`google_search` hosted
  tool plus a structured-output schema), taking its URLs from
  `groundingMetadata.groundingChunks`;
- `web_fetch` as a `url_context` call that returned Markdown for up to eight URLs
  at once, with an `unretrieved` list built from `urlContextMetadata`;
- `web_extract` (caller-supplied JSON Schema applied to each URL),
  `web_compare` (a comparative question across 2–8 URLs) and
  `web_deep_research` (a multi-search, cited report), all on the same shared
  `gemini` helper keyed by `GEMINI_API_KEY`.

Its argument was that Gemini 3.x combines hosted tools with structured output in
one round trip, and that with no local fetch there is no SSRF surface to guard.

It was replaced by the tools above: search through Tavily, with its key held in
the daemon's secrets document rather than the process environment, and a local
fetch behind the `public_fetch` guard, which returns the page's own text and
marks the pictures it cannot read. The repository's history records no further
reason for the change. The Gemini helper, `web_extract`, `web_compare` and
`web_deep_research` were never built.
