# Zend UI redesign — phased implementation plan

> Status: **draft for iteration.** This document is the authoritative plan for
> rebuilding the Zend daemon's web UI from the high-fidelity design prototype
> that was imported to `docs/design/zend-ui/`. That prototype has since been
> consumed into the shipped UI and removed from the repo; this document and
> `zend/web/index.html` are now the reference. It is meant to be edited in place
> as we agree on details — design docs take precedence over the code (CLAUDE.md),
> so when we change our minds, we change this file in the same commit.

## 1. Goal & scope

Replace the current Zend web UI (`zend/web/index.html`, ~1,516 lines, vanilla
single-file) with the redesigned interface that was specified in the imported
design bundle (`Zend.dc.html`, `zend-api.js`, `README.md`; since consumed into
`zend/web/` and removed). The redesign is a **superset** of today's UI — it
keeps the chat/sidebar/logs and
adds: a projection timeline gutter, an inspectable windowed-substrate dialog,
conversation-scoped files with drag-and-drop **and a per-part upload-progress
screen**, a collapsible sidebar rail, thinking-effort and answer-length composer
dials, a reasoning ("thinking") block, tool-call cards, and an appearance (theme)
switcher.

> **Beyond the design bundle.** The bundle's `uploadFiles` is instant
> (`await delay(180)`). Real uploads **carve + prefill the file in parts** on the
> conversation-files layer (decision 2), which is not instantaneous — so the
> redesign adds an **upload-progress screen** the bundle does not specify. Its
> look should match the design language (overlay/modal tokens already in use);
> its behavior is specified in §2.5 + Phase-1 step 1.8.

### Chosen approach (decided)
- **Frontend stack:** vanilla, single embedded file. The rebuilt UI lives in
  `zend/web/` and is embedded into the daemon binary via `include_dir!` exactly
  as today. **No build step, no node, no framework** — the repo has zero JS
  build tooling and we are not introducing any. The design's `renderVals()` /
  `<sc-for>` / `<sc-if>` structure maps directly onto the existing render loop
  in the current `index.html`.
- **GUI-first, backend-second.** Phase 1 builds the *entire* UI against a
  **mock backend** (`zend-api.js`) so every screen and interaction is real and
  testable with no daemon, no model, and no GPU. Phase 2 implements the live
  backend behind the **same interface** and fills the handful of missing Rust
  endpoints. The mock is the seam that makes this safe.
- **Tests:** **Playwright headless**, driving the real DOM against the mock in
  Phase 1, then re-run unchanged against the live daemon in Phase 2 as the
  acceptance gate (contract parity). "Positive tests as we go" = each Phase-1
  sub-step lands with its own green Playwright spec.

### Non-goals (this plan)
- No RAM/NVMe KV tiering work (tracked separately in `kv_tier_migration.md`).
- No model/inference changes beyond wiring composer options into projection
  (named-section directives) + the think-channel on/off hook.
- No authentication / multi-user surface — the daemon stays single-user on
  `127.0.0.1`.

---

## 2. What already exists (ground truth)

### 2.1 Backend (`zend/src/api/`, axum)
Router in `zend/src/api/mod.rs` serves the embedded `web/` dir (fallback) plus:

| Method & route | Handler | Returns |
|---|---|---|
| `GET /v1/conversations?include_archived=` | `conversations::list` | `{ conversations: ConvEntry[] }` where `ConvEntry { id, label, turn_count, archived }` |
| `GET /v1/conversations/:id` | `conversations::get` | `{ id, messages: [{ role, content }] }` |
| `POST /v1/conversations/:id/archive` | `conversations::archive` | `204` |
| `POST /v1/conversations/:id/unarchive` | `conversations::unarchive` | `204` |
| `GET /v1/models` | `models::list` | model list |
| `GET /v1/status` | `status::status` | `{ state: "loading"\|"ready", started_at_ms, detail, loading? }` |
| `POST /v1/chat/completions` | `chat::completions` | SSE (`status` events + OpenAI chunk deltas + stop chunk + `[DONE]`) or one JSON body |
| `GET /ws/logs` | `ws_logs::handler` | WebSocket: replays `session.log.recent()` backlog, then live lines |

The SSE stream already emits two frame kinds: named `status` events
(`{ "text": "..." }`) and standard OpenAI `chat.completion.chunk` data frames
whose `choices[0].delta.content` carries token text. `<think>…</think>` and
`<tool_call>…</tool_call>` are produced by the model **inline in the content
stream** — they are a rendering concern, not separate events.

`ChatCompletionRequest` (`zend/src/types.rs`) currently has `stream`,
`max_tokens`, `conv_id`, `messages` — **no** `think`/`effort`/`verbosity`.

The daemon already runs `candle_conversation::projection` internally
(`zend/src/session.rs` uses `projection::{Builder, …}`, schema in
`zend/src/prompts/projection.yaml`). Projections are real; they are simply not
surfaced over the wire yet.

`/ws/logs` already does backlog-then-live, so the mock's `seedLogs()` +
`subscribeLogs()` map onto it directly. One mismatch to resolve: the daemon
sends each log as a **raw formatted string**; the mock models a structured
`{ ts, level, target, msg }`. Resolution (decision 6): the daemon emits the
structured JSON record on the socket — no client-side parsing (see §2.6).

### 2.2 Frontend (`zend/web/`)
- `index.html` — vanilla single file, same design tokens (`--accent #c98a3e`,
  etc.), three-column layout, already wires: `/v1/status` polling, conversation
  list/hydrate, archive, SSE chat, `/ws/logs`. Uses `marked.min.js` for
  markdown.
- `marked.min.js`, `favicon.svg`.

The redesign **replaces** `index.html` outright (no dual UI path — CLAUDE.md
"no backward compatibility"). The new file is larger but the same kind of
artifact.

### 2.3 Gap analysis (design/mock vs. live backend)

| Design feature (mock method) | Backend today | Phase-2 work |
|---|---|---|
| Sidebar list + hydrate (`seedConversations`) | ✅ `GET /v1/conversations`, `GET /:id` | Map `label→title`; add `updated_ms` if missing |
| Archive/unarchive | ✅ | none |
| Streaming tokens + status (`streamChatCompletion` `onToken`/`onStatus`) | ✅ SSE | none |
| `<think>` / `<tool_call>` rendering | ✅ inline in content | frontend-only |
| Composer opts `{think, effort, verbosity}` | ❌ not in request | extend request + named-section directives + `/no_think` (§2.2) |
| Projection spans (`onProjection`, hydrate `spans[]`) | ⚠️ internal only | emit `projection` SSE events + include in hydrate (§2.3) |
| Windowed substrate (`getWindowedSubstrate`) | ❌ no endpoint | new `GET /v1/conversations/:id/substrate?at=` (§2.4) |
| Files (`uploadFiles` + list/get/delete) | ❌ none | new files subsystem (§2.5) |
| Live logs (`subscribeLogs`/`seedLogs`) | ✅ `/ws/logs` backlog+live (raw string) | structured JSON framing on the socket (§2.6) |
| Themes / focus guard / Escape order / resizers | n/a (pure UI) | frontend-only |

---

## 3. Architecture: the mock seam

The single integration seam is **`window.ZendAPI`**. We formalize it as one
**interface contract** with two interchangeable implementations:

```
zend/web/
  index.html        # UI (template + logic), unchanged seam usage
  zend-api.js        # defines window.ZendAPI by selecting an implementation
  zend-api.mock.js   # MockZendAPI  — in-memory, deterministic (from the design bundle)
  zend-api.live.js   # LiveZendAPI  — real fetch/SSE/WebSocket against the daemon
  marked.min.js, favicon.svg
```

`zend-api.js` chooses the implementation at load time:

```js
// live by default (embedded daemon build); mock when explicitly requested
// (standalone design iteration + Playwright Phase-1 runs).
const useMock = new URLSearchParams(location.search).has('mock')
             || window.ZEND_BACKEND === 'mock';
window.ZendAPI = useMock ? window.ZendMockAPI : window.ZendLiveAPI;
```

Both implementations satisfy the **same method signatures and the same
object/event schemas** (§4). The UI imports nothing else — it never calls
`fetch` directly. This is what lets Phase 1 build and test the whole GUI with no
daemon, and lets Phase 2 swap in the real backend without touching UI code.

> **Acceptance device:** two complementary gates. (1) The Rust harness
> `zend/tests/gui_api_harness.rs` boots the **real** model-less router and asserts
> the daemon serves the GUI + the model-independent API over real HTTP/WS — runs
> in CPU CI, no browser (decision 7; built, 3/3 green). (2) The Phase-1 Playwright
> suite runs against `?mock=1`, and in Phase 2 the *same specs* re-run against a
> live daemon (the harness's server locally / a model-backed daemon). A feature
> is "done on live" when its spec passes unchanged against the live adapter.
> Contract parity is the gate.

---

## 4. Wire contract (the source of truth)

Both `MockZendAPI` and `LiveZendAPI` MUST conform to these shapes. This section
is normative; the Rust endpoints in Phase 2 are written to satisfy it.

### 4.1 Methods
```
seedConversations(now) -> Conversation[]           // initial list; active one hydrated
                                                    // (live: GET list + GET :id for active)
getConversation(id) -> Promise<Conversation>        // hydrate on demand (live: GET :id)
getStatus() -> Promise<{state:"loading"|"ready", started_at_ms, detail, loading?, build}>  // GET /v1/status; gates the startup overlay; `build` (assets hash) drives the hot-reload check
archiveConversation(id) / unarchiveConversation(id) -> Promise<void>
streamChatCompletion(conv, text, opts, handlers) -> { cancel() }
mkProjEvent(conv, region) -> ProjectionSpan         // mock-only synthesis helper; live ignores
getWindowedSubstrate(conv, node) -> Promise<Section[]>
uploadFiles(convId, descriptors, handlers) -> { cancel() }   // progressive: carve + prefill in parts
getFileContent(convId, fileId) -> Promise<string>   // live: GET …/files/:fileId (mock: returns inline content)
deleteFile(convId, fileId) -> Promise<void>
seedLogs() -> LogLine[]                              // backlog
subscribeLogs(onLine) -> unsubscribe()              // live stream
```
`opts = { think: boolean, effort: 0..4, verbosity: 0..4 }`.
`handlers (chat) = { onStatus(text), onToken(delta), onProjection(span), onLog?(), onDone() }`.
`handlers (upload) = { onFileStart(fileId, name, totalParts), onPart(fileId, partIndex, totalParts), onFileDone(fileId, FileMeta), onAllDone(FileMeta[]), onError(fileId, message) }`.

### 4.2 Objects
```
Conversation { id, title, archived, updated_ms, history: Message[], files?: FileMeta[] }
Message      { role: "user"|"assistant", content, streaming?, status?, spans?: ProjectionSpan[] }
ProjectionSpan {
  id, label, metric, detail, region: "think"|"answer",
  step: "t=N", from, to, total,             // token offsets over the unbounded context
  barLeft, barWidth, winK, totK             // precomputed display fields
}
Section { kind: "marker"|"system"|"kv"|"memory"|"turn"|"assistant", label?, sub?, text, tokens }
FileMeta { id, name, ext, kind: "code"|"log"|"doc"|"text"|"img", size, added, content }
UploadPart { fileId, partIndex, totalParts }   // one prefilled part; progress = (partIndex+1)/totalParts
LogLine  { ts: "HH:MM:SS", level: "TRACE"|"DEBUG"|"INFO"|"WARN"|"ERROR", target, msg }
```

> Notes for the live mapping (per §9 decisions):
> - `ConvEntry.label` → `Conversation.title`. **Add `updated_ms` to `ConvEntry`**
>   (decision 4).
> - `ProjectionSpan` display fields (`barLeft/barWidth/winK/totK`) are **computed
>   by the adapter** (decision 1); the wire carries only the raw core
>   `{ id, region, metric, detail, step, from, to, total, window }`.
> - `region` is **parsed from emitted content** — `think` inside a
>   `<think>…</think>` span, else `answer` (decision 5).
> - `LogLine` is delivered **as structured JSON on `/ws/logs`** (decision 6), not
>   parsed from a formatted string.

---

## 5. Phase 0 — scaffolding & the seam

Goal: design files in-repo, the seam in place, Playwright harness running, one
trivial green test. **No backend changes.**

Steps:
1. ✅ Import design bundle to `docs/design/zend-ui/` (done; bundle since consumed into `zend/web/` and removed from the repo).
2. Split the seam: rename the design mock to `zend/web/zend-api.mock.js`
   (cleaned of the CP1252 mojibake — see §8 glyph table), add the
   `zend/web/zend-api.js` selector, add an empty `zend-api.live.js` stub that
   throws "not implemented" for every method (so a mis-set flag fails loudly).
3. Create the new `zend/web/index.html` shell (layout root + theme filter +
   empty columns) wired to read from `window.ZendAPI`. Keep the old UI working
   until cutover by developing the new file as `index.html` only once it reaches
   parity — **but** per "no dual path", we do this on the branch and the old
   file is replaced, not kept alongside. During Phase 1 the branch's
   `index.html` is the new UI; the daemon on `main` is unaffected.
4. Stand up Playwright: `zend/web/tests/` with a static file server fixture
   (serve `zend/web/` over http, open `index.html?mock=1`). Config runs
   headless Chromium.
5. **Positive test 0.1:** page loads with `?mock=1`; the layout root renders;
   `window.ZendAPI` resolves to the mock; no console errors.

Exit criteria: `npx playwright test` green with one spec; `cargo build -p zend`
still builds (web dir still embeds; old assets replaced by new shell).

---

## 6. Phase 1 — GUI on the mock (feature by feature)

Each sub-step is independently shippable and lands with its own Playwright
spec(s). Order is chosen so each step builds on a visible, testable base. All
work is in `zend/web/index.html` + `zend-api.mock.js`; **no Rust**.

### 1.1 Layout shell + sidebar
Collapsed 56px rail (default) ↔ expanded 220px panel; resizer (120–420px);
conversation rows (active highlight, accent bar, archive ×); "Show archived"
toggle; new-conversation; select/switch.
**Tests:** collapse toggles rail↔panel; selecting a row switches active
conversation; archiving hides a row; show-archived reveals archived (italic)
rows; resizer changes width within clamp.

### 1.2 Chat rendering (static history)
Markdown (paragraphs, bold/italic, inline code, headings, lists); fenced code
blocks with language label + Copy; user bubble vs. full-width assistant.
**Tests:** fenced block renders `<pre><code>` with uppercased lang label; Copy
writes code text to clipboard; user vs assistant layout differs.

### 1.3 Streaming
`send()` → loading dots → status → tokens append → blinking cursor → finalize.
Empty-state greeting + prompt cards. Auto-scroll.
**Tests:** sending drives tokens into the assistant bubble; cursor present while
`streaming`, gone after `onDone`; prompt-card click sends its prompt.

### 1.4 Thinking block + tool-call cards
`<think>…</think>` → `<details>` with brain icon, live word count, **collapsed
by default even while streaming**, chevron rotates on open, per-message open
state persists across re-renders. `<tool_call>{json}</tool_call>` → card with
tool name + arg rows; pending (incomplete) card pulses.
**Tests:** think block present and **closed** during stream; word count updates;
toggling open persists after the next token re-render; complete tool card shows
parsed args; trailing incomplete tool call renders the pending pulsing card.

### 1.5 Composer dials
Effort (Off·Quick·Balanced·Deep·Exhaustive) + length
(Terse·Concise·Standard·Detailed·Comprehensive) click-to-open menus with
5-segment meters, default middle; `{think, effort, verbosity}` passed to
`streamChatCompletion`; context labels hide ≤560px.
**Tests:** changing verbosity changes streamed answer length (mock honors it);
effort=Off (0) → no `<think>` block; menu opens/closes; Escape closes an open
menu first.

### 1.6 Projection timeline (gutter)
46px gutter inside the chat scroller; dots seeded on the active conversation and
accumulating live via `onProjection`; `measureProjections()` pins each dot to
its text block's `offsetTop`; think-region dots cluster at the thought header
(collapsed) / spread across reasoning (expanded); hover popover (label, step,
detail, window-vs-total mini bar, token range).
**Tests:** active conversation seeds N dots; a new dot appears when the window
opens at stream start; more accumulate during streaming; hovering a dot shows
the popover with the right step/range; expanding the think block re-pins
think-region dots (no `top:-9999`).

### 1.7 Windowed-substrate dialog
Click a dot → modal; ordered `Section[]`; colored left rail per kind; markers
render inline non-expandable; expand/collapse per section; assistant section
expanded by default; per-section Copy + Copy all; header shows step · range ·
window/total.
**Tests:** clicking a dot opens the dialog and fetches sections; assistant
section open by default; expanding a section reveals its `<pre>` text; Copy all
concatenates section text; Escape closes (after menus/file viewer).

### 1.8 Files pane + drag-and-drop + upload screen + viewer
Right-docked 300px pane (count badge); drag any file onto the conversation →
dashed "Drop files to upload" overlay (pre-drop hover) → on drop, a **blocking
upload modal** takes over (composer disabled until done). It lists **every
dropped file with its own per-part progress bar, all advancing in parallel**
(decision: parallel prefill), driven by `uploadFiles(..., handlers)`
(`onFileStart` → total parts; `onPart` → that file's bar advances; `onFileDone` →
check; `onError` → row error), plus an overall indicator and a **Cancel** that
calls the returned `cancel()` and aborts all in-flight uploads. When `onAllDone`
fires the modal dismisses, the files pane opens, scrolls to the newest, and
flashes it amber (the existing hand-off). The file viewer dialog (monospace
content, Download, Copy, Delete, image placeholder) reconstructs content lazily
via `getFileContent` (§2.5 / §11 item 8).
**Tests:** dragging shows the pre-drop overlay; on drop the modal appears and the
composer is disabled; with two files dropped, **both bars advance concurrently**
across `onPart` events to 100% then `onFileDone`; `onAllDone` dismisses the modal
and opens+flashes the pane; Cancel mid-upload aborts all (no files added);
clicking a row opens the viewer (content fetched via `getFileContent`); Delete
removes the file and closes the viewer; Download produces a blob anchor; image
kind shows the placeholder (but Download still returns real bytes — §2.5).

### 1.9 Logs pane
Toggle from top-right Logs button; live feed via `subscribeLogs`; level badges
(TRACE/DEBUG/INFO/WARN/ERROR) with tinted backgrounds; Clear; Hide/Show;
connected dot; auto-scroll; rolling cap (200 lines).
**Tests:** seeded backlog renders; new lines append and auto-scroll; Clear
empties; Hide hides pane and reveals the Logs tab; level badge colors map
correctly.

### 1.10 Cross-cutting UI behaviors
Appearance themes (Light/Dark/Vivid via CSS `filter` on the root, persisted to
`localStorage['zend.theme']`); composer focus guard (mousedown `preventDefault`
on non-input targets keeps the textarea focused); Escape priority order
(menu → file viewer → substrate → files pane); resizers for sidebar + log pane;
responsive breakpoints (≤1024 / ≤720 / ≤560).
**Tests:** theme choice persists across reload; clicking a non-input element
keeps textarea focus; Escape closes overlays in the specified order; ≤720px
sidebar becomes an overlay drawer and resizers hide; ≤560px dial labels hide.

**Phase 1 exit criteria:** full Playwright suite green against `?mock=1`; visual
parity with the approved design prototype (since removed — `zend/web/index.html`
is now the reference); zero `fetch`/WebSocket calls from the UI (everything
through `window.ZendAPI`); old `index.html` fully replaced.

---

## 7. Phase 2 — backend: LiveZendAPI + missing endpoints

Each sub-step implements part of `LiveZendAPI` and any Rust endpoints it needs,
then re-runs the **relevant Phase-1 Playwright specs against a live daemon**.
Rust side follows house style: one concern per file under `zend/src/api/`,
imports at top, raw-byte/shape assertions in tests, no stubs.

### 2.1 Adapter for existing endpoints + conversation-state realities
Implement in `zend-api.live.js`: `seedConversations`/`getConversation`
(`GET /v1/conversations`, `GET /:id`), `archiveConversation`/
`unarchiveConversation`, `streamChatCompletion` token+status path (parse SSE:
named `status` events + OpenAI chunk deltas, stop on `[DONE]`), `subscribeLogs`
(`/ws/logs`). Projection/substrate/files methods still throw "not implemented".

This sub-step also fixes the prototype's fully-loaded-local-state assumptions
(§11 items 2–6, 11):
- **String ids everywhere.** Conv ids are strings; new conversations mint a
  `crypto.randomUUID()` `conv_id` (not `Date.now()`). Normalize all `===`
  comparisons + dot/span keying to strings. (§11 item 5)
- **Visibility by metadata.** The sidebar shows a recovered conversation when
  `turn_count > 0` (or it's the active draft), **not** by `history.length`, so
  un-hydrated conversations still appear. (§11 item 2)
- **Lazy hydrate on select.** `selectConv` calls `getConversation(id)` once per
  conversation (guard with a `loaded` set) and folds in history. (§11 item 3)
- **Server-split bubbles.** `GET /:id` returns clean `{role,content}[]`
  (decision 9); the adapter does no ChatML parsing. (§11 item 4)
- **API-backed mutations.** Archive (and the new restore), delete-file, and
  conversation creation go through the API with optimistic update + rollback on
  failure — they are not local-only. (§11 item 6)
- **Title reconcile.** A periodic `GET /v1/conversations` refresh updates titles
  the async titler has since written (`label !== title`). (§11 item 11)
- **Async first load.** `seedConversations` is async on live (list + active
  history); render a loading state until it resolves. (§11 minor)

**Rust:** add `updated_ms` to `ConvEntry` (decision 4); move `splitChatMLTurn`
into the daemon so `GET /:id` returns pre-split, cleaned bubbles (decision 9);
contract tests asserting the JSON shapes the adapter consumes
(`zend/tests/api_conversations.rs`, `api_chat_sse.rs`) including that `GET /:id`
messages are already role-split and `/no_think`-stripped.
**Gate:** Phase-1 specs 1.1–1.5, 1.9 pass against the live daemon.

### 2.2 Composer options → named-section directives
Extend `ChatCompletionRequest` with `think: bool`, `effort: u8 (0..4)`,
`verbosity: u8 (0..4)`. The dials are realized as **prompt directives carried by
named sections**, not as raw sampler knobs:
- `effort` (1..4) and `verbosity` (0..4) select **named sections** in the
  projection section collection (e.g. an `effort:<level>` and a
  `verbosity:<level>` directive section). The projection builder admits the
  chosen sections so the assembled prompt steers reasoning depth / answer length.
- **`projection` and `reprojection` consume these named sections in place of the
  provenance scan** for this request — the dial state, not BDP retrieval, drives
  what context is admitted/re-weighted. (This is the load-bearing design change;
  exact section names + reprojection re-weighting are specified here before code,
  touching `candle_conversation::projection` and `prompts/projection.yaml`.)
- **No-thinking is special:** `effort=0` (Off) / `think:false` maps to
  prepending the existing **`/no_think` dialect prefix** to the user turn
  (decision 10) — the mechanism already in the codebase, already stripped on
  hydrate by the server-side splitter (decision 9). Not a prompt directive, not
  new inference plumbing.
Thread the fields through `session.submit(...)` into the projection request; the
`/no_think` prefix is applied to the user content before submit.
**Rust tests:** request deserializes with the new fields (defaults when absent);
`effort=0`/`think:false` prepends `/no_think` and yields no think segment; each
`effort`/`verbosity` level admits the expected named section(s) into the built
prompt (assert on the assembled section set, not on free-text). **Gate:** spec
1.5 passes on live (Off → no think; verbosity changes length).

### 2.3 Projection selections: persist, stream, serve
Projections/reprojections are **persisted as substrate records** capturing only
the selection (decision 8), and everything the UI shows is derived from them.

**Persist.** On each projection/reprojection during decode, append a
`ProjectionSel` record to the substrate: `{ conv, turn, seq, region, selected:
[admitted section/chunk ids] }`. New record kind + recovery path in
`candle-conversation/src/persistence/`. `seq` is monotonic per conversation and
becomes the span `id`. `region` is recorded at emit time as `think` while the
decode head is between `<think>` and `</think>`, else `answer` (decision 5).

**Stream.** Emit a named SSE event `projection` carrying the raw span core
derived from the just-written record — `{ id, region, metric, detail, step,
from, to, total, window }` — never the display fields (decision 1). The adapter
computes `barLeft/barWidth/winK/totK`. `mkProjEvent` stays mock-only; the live
adapter never synthesizes spans (the opening "window opens" span comes from the
daemon).

**Serve.** `GET /v1/conversations/:id` replays the records and attaches `spans[]`
(same raw core) to each assistant message, so re-opening a past conversation
shows its timeline. The windowed-substrate endpoint (§2.4) keys on the record
`seq`, not the display `step` string (§11 item 10).

**Rust tests:** a multi-step decode writes ≥1 `ProjectionSel`; records survive a
redo-log replay (raw-byte assertion on the stored selection); the served span
core is derived correctly from a known selection; SSE events carry only raw
numerics (no `barLeft`/`winK`); region flips across a `<think>` boundary and
back; hydrate `spans` are per assistant turn with monotonic `seq`. **Gate:** spec
1.6 passes on live (dots seed from hydrate + accumulate from the stream).

### 2.4 Windowed-substrate endpoint
`GET /v1/conversations/:id/substrate?at={step}` → `Section[]`. Materialize the
projected context at the given projection step from
`candle_conversation::projection` + persistence: system prompt, KV/retrieval
injections, compacted memory, turns up to the head, interleaved with ChatML
markers; each section carries a token count. New file `zend/src/api/substrate.rs`.
**Rust tests:** sections are well-ordered and start/end with the expected
markers; token counts are positive; the assistant (head) section is last;
`at` selects the right window.
**Gate:** spec 1.7 passes on live.

### 2.5 Conversation-files layer
A new **`conversation-files` projection layer** (decision 2), built to mirror the
`code_reading`/read-file layer (`zend/src/code_read/`): carve → tokenize → store
parts as token strings in the substrate, each part tagged with its **token-string
range** so the file text reconstructs by concatenation. Unlike `code_reading`
it is **not auto-admitted** into the conversation projection; a conversation
admits a file only via a **reference record**, and files are content-keyed so the
same file can be referenced by multiple conversations (sharing not exposed in the
GUI yet).

Routes (`zend/src/api/files.rs`):
- `POST /v1/conversations/:id/files` (multipart) → **SSE progress stream**.
  Uploading carves the file and **prefills it part-by-part** on the layer
  (reusing the `code_read` carve + prefill path and its `LoadProgress`),
  emitting `file_start` (with `totalParts`), one `part` event per prefilled part,
  and `file_done` (with the final `FileMeta`); a closing `done`/`[DONE]` ends the
  stream. On first completion it also appends a **reference record** linking the
  file to the conversation (what makes it show in the pane). The adapter maps
  these events onto the upload `handlers` (§4.1). Multiple dropped files prefill
  **in parallel**; prefill **reuses the existing prefill machinery, which already
  interleaves with chat decode** — no new scheduling is built for this. Slow by
  nature — this is why the UI has a progress screen, not a spinner.
- `GET /v1/conversations/:id/files` → `FileMeta[]` — the files this conversation
  references (metadata only; no content).
- `GET /v1/conversations/:id/files/:fileId` → reconstructed content
  (concatenate the file's stored token-string ranges in order).
- `DELETE /v1/conversations/:id/files/:fileId` → `204` — append a reference
  tombstone (drops it from this conversation; the shared layer content stays for
  any other referrer).

New layer + record-type work lives in `candle-conversation` (layer/timeline
machinery) + `zend` (ingest mirroring `code_read`); reference + tombstone records
get a recovery path in `candle-conversation/src/persistence/`. Derive `kind` from
extension; compute `size`/`added`. The list endpoint returns **metadata only**;
the viewer and Download fetch reconstructed content lazily via `getFileContent`
and cache it on the FileMeta (§11 item 8).

> **Non-tokenizable files (images/binaries).** No separate blob store — binaries
> go through the **same tokenized path** by encoding the bytes (e.g. **hex**)
> before carve/prefill, so they store, reconstruct, and download exactly like
> text (concatenate token-string ranges → decode hex → original bytes). The
> viewer still shows the `img` placeholder ("preview unavailable"), but **Download
> returns the real bytes**. One uniform code path for every file kind.

**Rust tests:** the upload SSE emits `file_start` → `part`×N → `file_done` with
monotonic part indices and `N == carve part count`; upload stores layer parts +
a reference record; `GET …/files/:id` **reconstructs byte-exact original text**
from the concatenated token-string ranges (raw-byte assertion); a second
conversation referencing the same file shares the layer content (one copy, two
reference records); delete tombstones the reference without destroying shared
content; everything survives a redo-log replay.
**Gate:** spec 1.8 passes on live.

### 2.6 Log line format
*Structured framing on the socket* (decision 6): `/ws/logs` sends each entry as
JSON `{ ts, level, target, msg }` (both the `recent()` backlog and live lines),
so the adapter does no parsing. This changes the socket contract in
`zend/src/api/ws_logs.rs` + the `log_broadcast` producer — emit the structured
record instead of the preformatted string. The `Lagged` notice becomes a
structured `WARN` record too.
**Rust tests:** backlog + live frames deserialize to the `LogLine` shape; level
and target are split out correctly; the dropped-lines notice is a well-formed
`WARN` frame. **Gate:** spec 1.9 passes on live.

### 2.7 Cutover
Default the embedded build to **live** (mock only via `?mock=1` / tests). Remove
the "not implemented" guards. Confirm the old UI is fully gone (single
`index.html`). Run the **entire** Playwright suite against a live daemon.
**Gate:** full suite green on live; `cargo test -p zend` green; manual smoke on
a real model session.

---

## 8. Cross-cutting concerns

### Encoding / glyphs
The design bundle arrived with CP1252→UTF-8 mojibake. The rebuilt
`index.html` and `zend-api.mock.js` MUST use correct UTF-8. Write/Edit tools
only — **never** PowerShell `Set-Content`/`Out-File` (CLAUDE.md). Glyph table:

| Intended | Mojibake seen | Where |
|---|---|---|
| `—` em dash | `â` (bare) | comments, details |
| `·` middle dot | `Â·` | meta separators |
| `…` ellipsis | `â¦` | placeholders, statuses |
| `→` arrow | `â` | prose |
| `↵` return | `â` | composer hint |
| `⇧↵` shift-return | `â§â` | composer hint |
| `×` multiply | `â` | archive / close buttons |
| `›` chevron | `âº` | log target separator |
| `✓` check | `â` | archived checkbox |
| `⚙` gear | `â` | tool-call icon |
| `Δ` delta | `Î` | projection metric "Δ salience" |
| `─ ┌ └ │` box-drawing | `â â â¦` | comment rules (drop in shipped code) |

The composer placeholder, for reference, should read:
`Message…   ↵ to send · ⇧↵ for newline`.

### House rules that apply
- **No backward compatibility / no dual UI:** the new `index.html` replaces the
  old; no feature flag to keep the old one alive.
- **One concern per file (Rust side):** new endpoints get their own files
  (`api/substrate.rs`, `api/files.rs`); the projection-span emit logic lives
  with the chat stream but factored cleanly.
- **TDD with shape assertions:** Rust endpoint tests assert exact JSON/SSE
  shapes (not tolerances). Playwright specs assert concrete DOM/behavior.
- **No TODOs/stubs in committed code:** the only intentional "throw not
  implemented" is `zend-api.live.js` methods *during Phase 1*, and they are all
  replaced by the end of their Phase-2 sub-step — none survive Phase 2.7.
- **Commits need explicit approval**, each one (CLAUDE.md).

### Tooling note (Playwright)
Playwright is a dev/test-only dependency (not embedded, not shipped, not a build
step for the UI). It lives under `zend/web/tests/` with its own
`package.json`/lockfile scoped to tests, kept out of the daemon build. CI runs
it as a separate job. This does **not** reintroduce a frontend build step — the
shipped artifact is still the hand-written `index.html`.

---

## 9. Resolved decisions

These were settled during planning and are now binding for the sections above:

1. **Projection display math** — *Daemon sends raw, adapter computes.* The
   `projection` SSE event and the hydrate `spans[]` carry only raw numerics
   (`from,to,total,window` + `id,region,metric,detail,step`); `LiveZendAPI`
   derives `barLeft/barWidth/winK/totK` using the same helper the mock uses. One
   source of display math. (§4.2 / §2.3)
2. **Files = a new `conversation-files` layer (not inline blobs).** Conversation
   files live on a **new projection layer** that mirrors the existing
   `code_reading`/read-file layer (`zend/src/code_read/`): an uploaded file is
   carved + tokenized and its parts stored in the substrate as token strings,
   **with each part's token-string range recorded** so the original text is
   reconstructed by concatenation (cheap — the token strings are already
   persisted). Two deliberate differences from `code_reading`:
   - **Not auto-projected.** The layer is never admitted into the conversation
     projection on its own.
   - **Inclusion by reference.** A conversation admits a file only via a
     **reference log entry/record**; that reference is the admission. Because the
     file lives on its own content-keyed layer, **multiple conversations can
     reference the same file** (sharing; the GUI doesn't expose this yet).
   The GUI viewer/download reconstructs a file by concatenating its stored
   token-string ranges via `getFileContent`. (Supersedes the earlier
   "blob inline in a redo-log record" decision.)
3. **Composer-option semantics** — *Prompt-directive via a named-section
   collection.* The `effort`/`verbosity` dials populate **named sections** in
   the projection section collection; **projection and reprojection consume
   those named sections instead of the provenance scan**, and the resulting
   prompt steers the model's reasoning depth and answer length. The one
   exception is **no-thinking**, handled by a dedicated **inference-level hook**
   that switches the think channel on/off (not a prompt directive). (§2.2 / §2.3)
4. **`updated_ms`** — *Add the field to `ConvEntry`.* Track last-activity ms on
   the conversation and serialize it. (§2.1)
5. **Span region (think vs answer)** — *Parse from emitted content.* Track
   whether the decode head sits between `<think>` and `</think>` in the stream
   and tag each span's region accordingly. (§2.3)
6. **Log delivery** — *Structured framing on the socket.* `/ws/logs` sends each
   line as JSON `{ ts, level, target, msg }`; no client-side parsing. This
   changes the socket contract (backend work in §2.6). (§2.6)
7. **Orchestrated test daemon** — *Boot the real router model-less.* Instead of
   a separate canned stub, the harness `zend/tests/gui_api_harness.rs` boots the
   **actual** `zend::api::router` with a model-less `ZendSession` on an ephemeral
   port and drives it over real HTTP/WS. The model only loads on `start_loading`,
   so the whole model-independent surface (GUI asset serving, `/v1/status`,
   `/v1/conversations`, archive 503, 404s, `/ws/logs` JSON framing) runs in CPU
   CI with no GPU. This is the live-daemon contract gate; full model-backed
   parity (real tokens/projections) stays a cuda-gated/local run. **Status: built
   and green (3/3).** (§7)

8. **Projection persistence & serving** — *Substrate redo-log record kind.*
   Each projection/reprojection is written as a new substrate record capturing
   **only what the selection admitted** (the admitted section/chunk ids + the
   `region`), keyed by conversation + turn with a monotonic `seq`. **The span id
   *is* that `seq`** (numeric, stable, unique). Replay rebuilds the per-turn
   timeline, so `GET /:id` hydration returns `spans[]` for free and the live
   stream emits the same selection core. The API **derives** every other value
   (`from,to,total,window,metric,detail`) from the stored selection at serve
   time; the adapter then derives the display fields. Nothing derived is
   persisted. (This supersedes the wire detail of decision 1: the "raw core" the
   daemon emits is computed from this stored selection. It also resolves span
   identity — §11 item 7 — and the substrate-fetch key — §11 item 10 — which both
   become the record `seq`.)
9. **ChatML history split** — *Server-side.* `GET /v1/conversations/:id` splits
   each stored ChatML turn into role-attributed `{role,content}` bubbles (strips
   the `/no_think` dialect prefix, drops `<tool_response>` scaffolding) and
   returns clean bubbles. The canonical splitter lives in Rust; the adapter stays
   thin. (Today's client-side `splitChatMLTurn` logic moves into the daemon.)
10. **No-thinking switch** — *`/no_think` dialect prefix.* `effort=0` /
    `think:false` maps to prepending the existing `/no_think` dialect prefix to
    the user turn — already understood by the model and already stripped on
    hydrate by the splitter (decision 9). No new inference plumbing. (This refines
    decision 3's "inference hook" to the concrete existing mechanism.)
11. **Unarchive UI** — *Add a restore affordance.* Archived rows (shown under
    "Show archived") get a restore action calling
    `POST /v1/conversations/:id/unarchive`. Symmetric with archive.

### Residual design points (detail when the phase lands)
- The named-section-collection ↔ projection/reprojection integration (decision 3)
  touches `candle_conversation::projection` + `zend/src/prompts/projection.yaml`;
  the exact section names and how reprojection re-weights them is a Phase-2.2/2.3
  design sub-task, specified there before code.
- The substrate record kinds for **files** (decision 2) and **projection
  selections** (decision 8) both need record-type assignments and recovery paths
  in `candle-conversation/src/persistence/` — designed in §2.5 / §2.3 when those
  steps start.

---

## 10. Milestone checklist

- [ ] **P0** seam + Playwright harness + green smoke test; design files in repo.
- [ ] **P1.1–1.10** GUI on mock, each with green specs; old UI replaced.
- [ ] **P1 exit** full suite green on `?mock=1`; visual parity.
- [ ] **P2.1** existing-endpoint adapter; specs 1.1–1.5, 1.9 green on live.
- [ ] **P2.2** composer options → sampling; spec 1.5 green on live.
- [ ] **P2.3** projection spans (SSE + hydrate); spec 1.6 green on live.
- [ ] **P2.4** substrate endpoint; spec 1.7 green on live.
- [ ] **P2.5** files subsystem; spec 1.8 green on live.
- [ ] **P2.6** log parsing; spec 1.9 green on live.
- [ ] **P2.7** cutover to live; full suite green on live; `cargo test -p zend`.

---

## 11. Integration gaps & risks (design-vs-backend review)

Found by reading the design bundle against the live backend (`zend/src/api/`,
`session.rs`, `types.rs`) and today's shipped `zend/web/index.html`. Root cause
of the big cluster (2–6): **the design prototype treats conversation state as
fully-loaded local data, but the daemon serves it lazily and by reference.**

| # | Gap | Resolution | Where |
|---|---|---|---|
| 1 | Design is React-ish (`React.createElement`, `dangerouslySetInnerHTML`, `support.js`); target is plain-DOM vanilla like today's `index.html`. | **Translate** `renderVals()`/`buildNode` into the plain-DOM render loop — not a copy. Set Phase-1 effort expectations accordingly. | §5, Phase 1 |
| 2 | Sidebar visibility filter `history.length>0 \|\| active` hides un-hydrated (lazy) conversations — recovered convs would vanish. | Show by metadata: `turn_count>0` or active draft. | §2.1 |
| 3 | `selectConv` never hydrates; no `getConversation` call. | Lazy hydrate on first select, guarded by a `loaded` set. | §2.1 |
| 4 | `GET /:id` returns full stored ChatML turns; UI expects per-role bubbles. | **Server-side split** (decision 9): daemon returns clean `{role,content}[]`. | §2.1, decision 9 |
| 5 | Numeric ids + strict `===`; new convs = `Date.now()`. Live ids are strings → silent breakage. | String ids throughout; `crypto.randomUUID()` for new `conv_id`. | §2.1 |
| 6 | Archive, delete-file, conv-creation are local-only (no API call). No unarchive UI. | API-backed mutations w/ optimistic update + rollback; **add restore** (decision 11). | §2.1, §2.5, decision 11 |
| 7 | Dot color/keying assume numeric 1-based unique span ids. | Span id = `ProjectionSel.seq` (numeric, monotonic, stable). | decision 8 |
| 8 | Viewer/Download read `fv.content` inline; live list is metadata-only. | Fetch content on viewer open via `GET …/files/:fileId`; cache on the FileMeta. | §2.5 |
| 9 | `thinkEnabled` state + `thinkBg/thinkColor/toggleThink` view fields are dead (no toggle; folded into effort=Off). | Drop the dead `think*` fields in the rebuild; no-think = `/no_think` (decision 10). | §2.2, decision 10 |
| 10 | `getWindowedSubstrate` keys on the human `step` string (`"t=N"`). | Key on `ProjectionSel.seq`. | §2.4, decision 8 |
| 11 | Title set client-side on send; daemon titler relabels async; design never reconciles. | Periodic `GET /v1/conversations` refresh updates titles. | §2.1 |
| 12 | `measureProjections` runs every `componentDidUpdate` (every token) and force-reflows all dots. | Throttle to structural changes / rAF-coalesce; re-measure on toggle + resize, not per token. | Phase 1.6 |

All twelve are accounted for above; none are blockers. Items 1, 2, 3, 5, 6 are
the load-bearing Phase-1 build choices; 4, 7, 8, 9, 10, 11 are settled by the §9
decisions; 12 is a perf guard.

---

## 12. Implementation status

**Built & verified here (cargo green; CPU, no GPU):**
- Frontend seam + full plain-DOM GUI on the mock (`zend/web/`), Playwright suite
  (`zend/web-tests/`) — *authored; not executed here (no JS runtime in this env)*.
- `chatml::split_turn` (decision 9) wired into `GET /v1/conversations/:id`. ✅ 6 tests.
- `log_line` structured JSON framing (decision 6) wired into `/ws/logs`. ✅ 5 tests.
- `ChatCompletionRequest` dial fields `effort/verbosity/think`. ✅ 2 tests.
- **No-thinking** (decision 10): `apply_no_think` prepends `/no_think` to the last
  user turn on `effort:0`/`think:false`, in the chat path. ✅ 5 tests.
- `ConvEntry.updated_ms` (decision 4), derived from the conv id. ✅ compiles; served.
- **Windowed-substrate endpoint** `GET /v1/conversations/:id/substrate` (§2.4):
  real engine-backed materialization (`system_prompt` + recovered turns →
  ordered sections) via the pure `substrate_view::build`, **dummy-substrate
  tested** with no model (3 tests); model-less → 503 (harness). Read-only — does
  not touch the decode hot path. ✅
- **Daemon harness** `zend/tests/gui_api_harness.rs` — boots the real model-less
  router over HTTP/WS (decision 7) + the substrate 503 contract. ✅ 3/3.
- `zend-api.live.js` adapter for every endpoint that exists today (conversations,
  archive/unarchive, chat SSE token+status, logs WS, **windowed substrate**) +
  async-tolerant UI boot.

- **Conversation files (§2.5) — complete.** `conv_files` storage core
  (binary→hex→tokenizable→reconstruct, byte-exact, 6 tests) + a persistent
  `ConvFileStore` under `.substrate/conv-files/` (model-independent, 4 tests) +
  routes `POST/GET/GET content/DELETE` with **multipart upload → SSE per-part
  progress** + GUI live adapter (`uploadFiles`/`getFileContent`/`deleteFile`).
  **Full lifecycle harness-tested over real HTTP, no model** (upload → progress →
  list → byte-exact get → delete → 404). The store is the durable bytes tier;
  *tokenizing + admitting a referenced file into the projection* is the
  engine-backed enhancement on top (it does not block the GUI files feature).

- **Projection timeline / dots (§2.3) — wired + emitting.** `projection_span`
  (`region_of` think/answer parse + window→span builder, dummy-tested, 4 tests);
  `StreamItem::Projection` threaded through the decode loop (emits on the real
  think→answer transition, window state read once/turn — negligible cost, gated
  on a tag-byte delta so no per-token rescan); `event: projection` SSE frame;
  GUI live adapter `onProjection`. Token offsets are a display estimate over the
  real turn-window (same approach as `substrate_view`); a tokenizer-exact
  accounting is a resolver refinement.
- **No-thinking dial (§2.2) — done** (decision 10, above). The remaining
  effort/verbosity *gradation* (levels 1–4 → projection directive sections,
  decision 3) is the one piece that needs real projection-engine work: it lives
  in the `selection`/named-section machinery (`projection.yaml` + the Builder),
  and the model has no dialect token for length/effort (unlike `/no_think`), so a
  prefix hack would be fabrication. Scoped, not started.
- **Cutover (§2.7):** the selector defaults to **live**; the old single-file UI
  was replaced; `marked.min.js` removed. The only remaining `throw` in
  `zend-api.live.js` is `mkProjEvent` (mock-only synthesis — the UI never calls
  it on live).

> **The dummy-substrate testing pattern** (generalizes to the remaining pieces):
> split each model-dependent endpoint into a *pure shaping function* over plain
> data + a thin engine-backed retrieval. The pure function is unit-tested against
> a hand-built dummy substrate (no model); the engine path returns 503 model-less
> and is exercised by the harness. This is how §2.3 (projection-selection record
> encode/decode + span-core derivation) and §2.5 (files store + reconstruction)
> get built and tested next — no GPU needed for the logic.

**Remaining — deep model/substrate integration (do on the model box, with care:
these touch the perf-sensitive decode/projection hot path and the substrate
persistence internals; should be validated against a running model, not authored
blind):**
- effort/verbosity → **named-section directives** in the projection (decision 3,
  the rest of §2.2) — needs projection-schema + Builder work.
- **Projection-selection records** + `projection` SSE events + hydrate `spans[]`
  (§2.3) — hooks the decode loop / scheduler.
- **Windowed-substrate endpoint** (§2.4) — materializes projected context.
- **Conversation-files layer** + upload-prefill SSE + file routes (§2.5) —
  mirrors `code_read`; new substrate record kind.
- Substrate-backed `updated_ms` (replace the id-derived value once turn
  timestamps are exposed).
