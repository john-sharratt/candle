# Performance patch: zend, npcd, the gateway

A set of small, low-risk changes across zend (server and GUI), npcd, the
cluster gateway, the MCP aggregator and the tokera.com site. Each removes
work the code does not need — a deep clone, a full scan, a re-render, a
re-parse, a wait — without changing an answer a caller relies on: where an
item changes what is observed — a header, a field added, a delay or
staleness accepted, a record layout — it says so, and §7 records the
decision. None is a redesign.

**The standing rule for every item: no new lock on a hot path.** Where an
item touches shared state it uses an immutable snapshot published through
`arc_swap` (already a dependency of zend, npcd and the workspace), an atomic,
an `Arc` in place of a deep copy, or work moved onto owned data outside the
lock. An item that touches shared state says which locks it takes and for
how long; one with no **Locks** line touches none. Besides atomics and
`arc_swap` snapshots, the patch adds two kinds of lock: the static-file
store's std `RwLock` (W8, and W10's posts and manifest on the same store),
read briefly per request and written on a miss, never across an await; and
the MCP aggregator's listing generation (W5), held only by background tasks.
Neither is on a beat's path.

Each item: **what costs** (with the code it is in), **the change**,
**locks**, **tests**, **risk**. Tests assert raw expected values, as the
repository requires. File references are as the code stood when this was
written; the validation pass (§6) re-checks each against the code. Paths here
name where a type or function lives; the code `use`-imports each and refers
to it by its short name.

§5 lists what was considered and held back; §6 the validation, and §7 the
decisions taken on the open questions and how the patch lands.

---

## 1. zend server

### Z1. Decode a round's text incrementally

**Costs.** Every decoded token re-decodes the whole round:
`session.rs:4786` and `:6527` (`state.decoder.decode(&tokens)`, a full
`tokenizer.decode` — `candle-conversation/src/handle.rs:371`). After it,
`text.rfind("</think>")` (`session.rs:4805`) rescans the answer every token
until `hold_from` is set, and `ThinkProgress::observe`
(`think_progress.rs:44,52`) runs `find(OPEN)` over the whole text until a
block opens — every token, in a turn that never opens one — and
`find(CLOSE)` over the whole block until it closes. Quadratic in the round,
on runtime threads. The same pattern is in
`candle-conversation/src/narrator/converter.rs:67-71` (two decodes per
token, and a slice that can panic off a char boundary) and
`npcd/src/engine/prose.rs:432`.

**Change.**
- `candle-conversation/src/text_stream.rs`: `TextStream<'t>` borrows the
  tokenizer (`TokenDecoder::stream(&self) -> TextStream<'_>`), holding
  tokenizers' `DecodeStream` (0.21.4, `skip_special_tokens = true` — the
  same flag as today; `DecodeStream`'s type names the five `*Wrapper`
  parameters), the round's ids (so it can fall back), and the text so far.
  `push(&mut self, id) -> usize` appends what the step releases and returns
  the offset where the new bytes start; `text(&self) -> &str` and
  `tokens(&self) -> usize` read it. **The text only ever grows.** On a step
  error (`InvalidPrefix`, or a decode error) the stream falls back to the
  full decode of its ids until it can resume: a full decode that
  extends the text appends the extension, one that does not appends nothing
  that step; once a full decode extends the text and does not end in
  U+FFFD, a fresh `DecodeStream` takes over from the next id (sound for a
  ByteLevel decoder, which needs no context across a whole character).
- The orphaned `candle-conversation/src/token_stream.rs` (never declared a
  module) is deleted; `TokenDecoder`'s doc example is rewritten on
  `stream()`.
- Both zend loops push through it, and so do `narrator/converter.rs` and
  `npcd/src/engine/prose.rs`. `emitted_len` and `hold_from` stay byte offsets
  into the text — and into `resp.text` at the flush
  (`session.rs:5029-5040`, `:6559`) — as today. The U+FFFD hold and the
  `is_char_boundary` checks stay; they cost O(new bytes).
- The answer's start is carried (`answer_at`); each push searches only
  `[from − (CLOSE.len() − 1)..]` of `as_bytes()` for `</think>`, a hit moving
  `answer_at` past it (the old `rfind`'s last-occurrence rule).
  `ThinkProgress` keeps a private `scanned` cursor and searches only from
  `scanned − (max(OPEN, CLOSE).len() − 1)`, in bytes; its
  `observe(&str, usize)` signature is unchanged. (A byte-window search, or
  `memchr` added as a dependency.)

**Locks.** None: all state is the turn future's. `&Tokenizer` is `Send` (the
tokenizer is `Sync`), borrowed from the `Arc<InferenceState>` the future
owns.

**Tests.**
- Unit, on a synthetic ByteLevel/ByteFallback tokenizer built with
  `TokenizerBuilder` as in tokenizers' `DecodeStream` docs: `é` as two byte
  tokens yields `""` then `"é"`, the text's raw bytes `[0xC3, 0xA9]`. A
  tokenizer built so that a step's prefix re-decodes differently (the decoder
  construction that makes `DecodeStream::step` return `InvalidPrefix`, named
  in the test) drives the fallback, and the text stays a prefix of the full
  decode and equals it at the end. If no real decoder can produce
  `InvalidPrefix` with `skip_special_tokens = true`, the fallback covers only
  decode errors, and the test drives it with an id outside the vocabulary.
- On every Qwen tokenizer in the HF hub cache, found as
  `candle-conversation/tests/literal_text_real_vocab.rs` finds them and
  skipped loudly when there is none (on the dev machines this includes
  `host::MODEL`'s Qwen3.8-Flash-Next, whose `<think>`/`</think>` are not
  special): the streamed text equals `decode(all, true)` byte for byte over
  a fixture holding `<think>`, `</think>`, `<tool_call>`, multi-byte text and
  an emoji the test first asserts is split across byte tokens.
- `ThinkProgress` given `"<think>a</thi"` (4) then `"<think>a</think>"` (5)
  answers `Running(4)` then `Done(5)`; given `"<thi"` (1) then `"<think>a"`
  (2), `None` then `Running(2)`. The existing five `ThinkProgress` tests
  unchanged.

**Risk.** Low–medium: on the streaming path. The equality test is the gate.

### Z2. One substrate pass for a round's `file_read` lookups

**Costs.** `working_set/corpus.rs:81,96` call
`find_conversations_by_metadata` — a full timeline scan
(`substrate.rs:7161`) — once per read call (`session.rs:5475`) and once per
seed.

**Change.** `Substrate::timelines_with_metadata_values(key, &BTreeSet<String>)
-> BTreeMap<String, Vec<TimelineId>>` (one pass, the same tombstone filter),
passed through resolver (`find_timelines_by_metadata_values`) and engine
(`find_conversations_by_metadata_values`). A value with no live timeline is
absent from the map. `corpus::readings_of_all` keys the round's calls first,
looks them up once, and returns the timelines in call order, as `flat_map`
over the calls did. `file_conversation` splits into `file_key_of` and
`file_conversation_among`, so `Seeding::apply` keys every file seed first.
(The per-call `committed()` git read stays. `guidance/reads.rs:63`,
`code_read/link.rs:94` and `repo_scan/link.rs:68` scan the same way; they are
outside this patch — §5.)

**Locks.** One snapshot read instead of N; already inside `spawn_blocking`.

**Tests.** Three timelines (live `a`, live `b`, tombstoned `a`) asked
`{a, b, c}` → exactly `{"a": [tl1], "b": [tl2]}`.

**Risk.** Low.

### Z3. Read one metadata key without cloning the bag

**Costs.** `resolver.rs:3783-3790` deep-copies a conversation's whole
metadata map — which "can hold a whole journal" — into a `BTreeMap`, though
the bag is an `imbl` map it could read in place, for callers that mostly
want one key.

**Change.** `ConversationEngine` gains `conversation_metadata_value(timeline,
key) -> Option<String>` (delegating to `resolver.rs:3795`),
`has_conversation_metadata_key(timeline, key) -> bool`, and
`conversation_metadata_with<T>(timeline, f: impl FnOnce(&POrdMap<String,
String>) -> T) -> Option<T>`, which reads the bag in place under the
snapshot.
- Single-key readers move to the first two — zend: `session.rs:4101`
  (identity), `:5603`, `:8194`; `retrieval_scope/index.rs:72`;
  `corpus.rs:122`; `code_read/mod.rs:508`, `:929`; `conv_inherit.rs:36`;
  `answer_steps.rs:44`; `ingest_snapshots.rs:125`; `branch_ingest/tie.rs:66`,
  `:122`; `guidance/record.rs:45`; `layers/pass.rs:353`;
  `layers/run.rs:150`, `:457`, `:811`; `repo_scan/mod.rs:673`, `:1096`;
  `resume.rs:78`. npcd: `engine/dreams.rs:581`; `mind_record.rs:193`;
  `mind.rs:1330`, `:2139`.
- Readers of several keys move to `conversation_metadata_with`:
  `session.rs:4365` (the dials compare), `:8027` (`turn_errors` takes the
  `POrdMap`), `:8035` (dials), `layers/run.rs:770`.
- Readers that copy the whole map out keep `conversation_metadata`:
  `restore.rs:99`, `self_check/mod.rs:78`, `layers/run.rs:85`, `:784`, npcd
  `layers.rs:182`.

**Locks.** A snapshot read (`ArcSwap::load_full`); no lock.

**Tests.** At the resolver (`replication_tests.rs`'s `open` fixture): the
value for a present key, `None` for an absent key and an unregistered
timeline; `conversation_metadata_with` sees exactly the keys set. The engine
methods are delegations.

**Risk.** Low.

### Z4. `job_status` wakes on a task's change

**Costs.** `mcp/tools/job/status.rs:123-138` checks jobs one after another,
then sleeps 2 s; a finished task is reported up to 2 s late. `read.rs:100`
the same.

**Change.** Subscribe to `call.session.mcp().watch()` before the first look
(a subscription starts with the current value seen, as `workers.rs:1896`
does), then wait on `timeout(LOOK_EVERY.min(left), changed())`; when
`changed()` errs (sender gone) sleep that long instead, so it cannot spin. A
wake rechecks **task** jobs only; builds, deploys and cluster jobs are
checked on their own 2 s cadence, together with `futures::future::join_all`,
their last standings kept between. Clocks are `tokio::time::Instant`. The
loop is a function over its look-tasks, look-remote and wake futures, so it
is tested with fakes. `job_read` on a task waits the same way; any other job
keeps the 2 s sleep. zend's dev-dependency `tokio` gains `test-util` (today
no zend test can pause time).

**Locks.** A per-call watch receiver; no lock across an await. Workers bump
on every beat, so each wake recomputes task standings — cheap after Z9,
which lands first.

**Tests.** With `#[tokio::test(start_paused = true)]`: a task that ends on a
bump answers at t = 0; three bumps within 1 s make one remote check; remote
checks over 5 s land at 0, 2 and 4 s; a dropped sender makes one look per
2 s.

**Risk.** Low.

### Z5. Conversation list and GET off the runtime threads

**Costs.** `api/conversations.rs:49-62` (`list`) and `:108-265` (`get`) do
all their work synchronously in an `async fn`: recovered history, part
parsing, JSON encoding, gzip — tens of ms for a long transcript, on the
threads `/mcp` shares. The list also goes out uncompressed, polled every 5 s.

**Change.** Each body becomes a plain `fn` run in `spawn_blocking` (cloned
`Arc<ZendSession>`, id, `HeaderMap`), the live-turn read kept before the
history read inside the closure as now; the list returns
`compressed::json`.

**Locks.** None new.

The list also does a `read_dir` per call (`session.rs:7609`), now off the
runtime too. It gains `Cache-Control: no-store` and `Vary: accept-encoding`
from `compressed::json`, which gzips at 860 bytes or more.

**Tests.** The list's body is built by a pure `list_response(rows, &Viewer,
headers) -> Response` (Z6's `Viewer`, so Z5 lands after Z6): literal rows of at least 860 bytes of JSON with
`Accept-Encoding: gzip` answer `Content-Encoding: gzip` and `Cache-Control:
no-store`, the body inflating to the exact expected JSON bytes; without the
header, those bytes plain. (A handler-level test cannot: a model-less
session's list is empty.) The existing builder tests unchanged.

**Risk.** Low.

### Z6. Resolve the caller once per list

**Costs.** `conversations.rs:59` runs `ownership::may_open` per row, which
re-identifies the caller and reads the role table each time.

**Change.** `ownership::Viewer::of(headers, roles) -> Nobody | Admin |
Person(key)` (`Nobody` when `identify` fails, `Admin` at `Role::Admin` or
above, else `Person(person.key())`) and `Viewer::may_open(&self, conv_id)`.
The free `may_open` and `guard` are written on it, so the rule exists once;
`list` resolves it once. A person below `User` keeps today's answer: their
own `mcp:` conversations only.

**Tests.** A person opens their own `mcp:` conversation and not another's;
Admin opens both; Nobody neither.

**Risk.** Low.

### Z7. A journal draft reads only what it has not covered

**Costs.** `session.rs:5864` rebuilds the whole `recovered_history` to use
the parts after `covered`; `origin.rs:101-106` finds each message's written
part by a linear search.

**Change.** The per-turn body of `Sequence::recovered_history` moves into
`fn recovered_from(read: &Substrate, timeline, ghost, from_turn, estimate,
response_open) -> Vec<RecoveredMessage>`, reading `turn_indices(timeline)`
from `from_turn` on; `Sequence::recovered_history_from(timeline, ghost,
from_turn)` calls it and `recovered_history` is `…_from(.., 0)`. The draft
passes `origin::turn_of(covered).unwrap_or(0)`, inclusive — a turn's reply
half may still be uncovered. With `Cadence::Now` the newest turn is the
maximum over the partial history; an empty one is `NothingNew`, as before.
`parts_after` builds a `HashMap<u32, u64>` of `written` once.

**Tests.** `recovered_from` on a substrate fixture holding three turns with
text-bearing layouts, from turn 1, returns exactly turn 1's and turn 2's
bubbles (a small helper sealing a turn's layout from user and assistant text
is new); the existing `parts_after` tests unchanged.

**Risk.** Low.

### Z8. The ingest unit probe stops at its first match

**Costs.** `repo_scan/mod.rs:908-911` builds a set of every content value in
the substrate to test one.

**Change.** `Substrate::has_live_intact_metadata(key, value) -> bool`, with
`ConversationEngine::has_live_intact_metadata` delegating through the
resolver's snapshot: `.any()` testing the value first, then the tombstone,
then `timeline_kv_intact` — the same membership as the set.
(`code_read/mod.rs:851`'s probe — a full ingest-chain scan per file — is a
separate cost, listed in §5.)

**Tests.** A matching tombstoned timeline → false; a matching live intact one
→ true; no match → false.

**Risk.** Low.

### Z9. Count a task's output lines without building it

**Costs.** `job/source/task.rs:76-77,155` builds a worker's whole output text
to count its lines, on every `job_status` round and every GUI poll.

**Change.** `write_output(w, now, &mut impl fmt::Write)`, with a `String`
sink for the text and a counting sink that counts both newlines and bytes,
giving `Standing.total` (lines) and G3's `Standing.size` (bytes) from the one
pass. Every segment
ends in `\n`, so the count equals `lines().count()`.

Call sites: `standing` (`:76-77`), `list` (`:155`); `read` (`:106`) keeps the
text. `turn_report::render` still allocates its own string.

**Tests.** A fixture worker (a literal like `mcp/tools/wait.rs:198-229`'s,
at a fixed `now`) gives an exact line count, computed against that fixture,
equal to `output(&w, now).lines().count()`.

**Risk.** Low.

### Z10. The retrieval index is a lock-free snapshot

**Costs.** `retrieval_scope.rs:105,272-281`: a stale index makes every turn
queue on `rebuilt.lock()` behind a rebuild of three full scans — every second
during an ingest pass.

**Change.** `index: ArcSwap<IngestIndex>`; a turn takes `rebuilt.try_lock()`
and, when a rebuild is running, uses the published index (staleness the
design already allows, ≤ `REBUILD_EVERY`), leaving `stale` set for the next
turn. `refresh` (after an upload) keeps its blocking lock and finishes before
the upload returns, so the next turn sees the file.

**Locks.** The index read is an `ArcSwap` load; a turn never waits on
`rebuilt` (`try_lock`); only a rebuilder or `refresh` holds it. The
short-held `shared`, `trees` and `hints` mutexes every scope computation
takes are unchanged. A turn that computed on the previous generation may
insert into `trees`/`shared` after the publish's `retain` (as a turn
arriving mid-rebuild can today); those entries are never hit again and age
out of the 16-entry `Kept`. A turn that starts while `refresh` runs gets the
previous index rather than waiting; only turns after the upload returns are
promised the file, as the design states.

**Tests.** In `retrieval_scope.rs`'s tests: with `rebuilt` held by the test
and the index marked stale, `current(t, |_| panic!())` returns generation 0
without waiting and `stale` stays set; after the guard drops the next
`current` rebuilds to generation 1. The existing cadence tests (`:925-960`)
unchanged.

**Risk.** Low.

### Z11. Streamed frames borrow their id and model

**Costs.** `api/chat_frames.rs:302-306` clones the reply's id and model for
every token frame.

**Change.** `ChatCompletionChunk<'a> { id: &'a str, model: &'a str, … }`
(serialize-only).

**Tests.** The existing serialisation test asserts the same bytes.

**Risk.** Low.

---

## 2. zend GUI (`zend/web/index.html`)

### G1. A message re-renders only when a job it shows changes

**Costs.** Every job poll bumps `S.jobGen` (`:1117`) and renders
(`:1131`); the message memo keys on it (`:1491`), so while any job runs the
whole transcript re-renders every 2 s.

**Change.**
- Each job carries `rev`, bumped by `setJob` only when something
  `jobBlock`/`waitWindows` draws changes: `kind`, `ended`, `line`, `open`,
  the log's length and last line, the tail element by element (a task's
  output is not append-only), the messages' identity. Creating a job bumps
  it. `followJob`, the poll and the click handler write through `setJob`.
  `S.jobGen` goes.
- While `nodeHtmlMemo` renders a message it collects the `[handle, rev]`
  pairs `jobBlock`/`waitWindows` draw (recorded after `followJob`; a draw
  outside a memo frame, such as the streaming patch's, records nothing). A
  hit requires each pair to hold and every handle to exist. Sub-agent
  messages are drawn with `nodeHtml` inside the parent's frame, and always
  with their thinking closed: `restoreToolOpen` opens what the reader
  opened, so no open state is baked into cached HTML.
- **A poll that changes a job redraws that job's blocks and wait windows in
  place** — `harvestOpen`, replace with `jobBlock(h)` / `waitWindow(h)`,
  `restoreToolOpen`, follow the bottom if at it — falling back to `render()`
  under the streaming patch's own guards (a selection or a press in the
  chat). This matters: a running build's, deploy's or task's `line` carries
  a seconds clock, so while one runs every poll changes it, and without the
  in-place redraw the transcript would still be rebuilt every 2 s.
- A poll that changes nothing calls `tickJobTimes()`, which sets each
  `span.job-time[data-began]` (the attribute is new) in place; `render()`
  also calls it, so cached HTML never shows a stale clock.

**Tests.** `visiblyChanged(prev, next)` (`job-poll.js`, UMD like
`partial-call.js`, run with `node --test`) on literal records: a new tail
array with the same lines → false; a changed middle tail line → true;
`read` alone → false. Playwright with `page.clock` (installed before
`goto`) and a frozen mock job (`__ZEND_MOCK_JOB_STILL__`, beside the mock's
other globals): across two ticks every `[data-msg]` node keeps its identity
while `.job-time` advances 2 s; with the normal mock job, an unrelated
bubble's node keeps its identity across a tick that changes the job. A
sub-agent think opened, re-rendered and then closed stays closed.

**Risk.** Low.

### G2. A streaming reply redraws only its live part

**Costs.** `renderParts` (`:1310-1343`) rebuilds every part every 80 ms:
`JSON.stringify` of each result, `toolCard`, the FNV hash in `lazyAttr`,
`renderMd` of each prose part.

**Change.** A part that can no longer change caches its HTML on itself: a
prose part that is not the last; a closed think; a closed call whose card
draws **no job block** (not in `STARTS`/`WAITS`, so no job revs to track).
The key covers what changes after close: a call's `result`, `running`,
`tokens`, `ok`/`error` and its position (`data-tool`, stable because parts
only append); a think's open flag and the length it is labelled with (the
history's `thinking`, else its counted tokens); plus `S.toolSchemaGen` and
`S.approvals`' identity. A `journal_note` is never cached. In live mode only
a streaming reply is drawn from parts (the history is drawn from content
until G7), so the gain is on the streaming reply.

**Tests.** The mock streams three closed calls and a live think, each
result an object whose `toJSON` counts its calls; over ten frames
(`page.clock`) the first result's count is exactly 1.

**Risk.** Low.

### G3. The job poll: one at a time, visible tabs only, fetch on change

**Costs.** Calls from `followJob`, the click handler and the timer can each
start a poll chain while one is in flight (`:1059`, `:4433`, `:1132`),
multiplying requests and appending log ranges twice; the poll runs in hidden
tabs; every tick fetches the view and the tail, and for an open sub-agent its
whole conversation.

**Change.**
- One entry, single-flight: a call while busy marks `again`, run as soon as
  the poll in flight settles; `busy` clears in a `finally`; the next timer is
  set only while visible. (Today two in-flight polls append the same log
  range twice.)
- `document.hidden` stops it at the top; `onPageHidden` clears the timer;
  `onPageShown` polls (covering the bfcache `pageshow` path too).
- `Standing` (`mcp/tools/job/mod.rs`) gains `size`, the bytes of output; each
  source fills it from the read it already makes (a task from Z9's counting
  sink, a build from the agent's `LogRead.size`, a cluster job from its
  output, a deploy from its two reads). `/v1/jobs/view` answers `size`; for a
  task it adds `turns` (`ZendSession::conversation_turn_count`, new: the
  length of `turn_indices(timeline)`, a snapshot read — the same count
  `GET /v1/conversations/{id}`'s `turns` answers). `GET /v1/conversations/{id}` answers `turns` too, and the GUI keeps
  the turns of the conversation it actually fetched, so a lagging member
  cannot leave it stale.
- `reads(job, view)` decides a poll's reads. A build's, deploy's or cluster
  job's tail is read when `total`, `size` or `ended` changed; a running
  task's tail every tick (its turn report carries a clock, and it is a local
  read). An open log reads from the line after the last held when `total`
  grew, `ended` changed, or it has not been read since opening. An open
  task's conversation is refetched when the view's `turns` differs from the
  fetched conversation's, when `ended` changed, or when it has not been read
  since opening — so the `read` flag is always set eventually. Keyed on
  `ended`, never on whether a turn is live: that is visible only on the
  member running it.

**Tests.** `reads` (`node --test`) on literal records: an unchanged build
view → `[]`; growth on an open build → `['tail', 'from']`; an ended view on
an open, unread build with `total` unchanged → `['tail', 'from']`; a
running task, unchanged → `['tail']`; an open task with `turns` 3→4 →
`['tail', 'conversation']`. A pure `view_body` in `jobs.rs` asserts the
exact JSON for a task and a build fixture. Playwright with `page.clock` and
`ZendAPI.jobView` wrapped to count: with no clicks, view calls at 0, 2 and
4 s; with a poll held in flight, three head clicks add exactly one view call
when it settles; a page made hidden (`document.hidden` defined,
`visibilitychange` dispatched) makes none over 6 s, and one at once when
shown.

**Risk.** Low.

### G4. Tool results are parsed once per message

**Costs.** `buildChat` (`:2623-2629`) trims and regex-parses every tool
result on every render, and the fresh strings make `sameOutputs` compare
character by character.

**Change.** `responsesOf(m)` (`tool-responses.js`, UMD): `m._resp = {
content, blocks }`, `blocks` null for a message that is not a tool response,
reused while `m.content` is the same string; `buildChat` pairs from it.

**Tests.** (`node --test`) Two calls on one message return the identical
array (`===`); a message whose content changed returns a new one; a non-tool
message answers `null` both times.

**Risk.** Low.

### G5. The conversation list merges through a map

**Costs.** `:4746-4759`: `find` per row and a `concat` per new row.

**Change.** `mergeConversations(convs, list, activeId)`
(`conversation-merge.js`, UMD): a `Map` by `String(id)` over the kept rows, each new row
added to it as it is collected (so an id the server lists twice is added
once, as `find` did), the new rows concatenated once.

**Tests.** (`node --test`) A list repeating a new id adds it once; a known
row takes a new title and rank; an unknown row is dropped unless it is the
active one or streaming — each the exact resulting array.

**Risk.** Low.

### G6. The mock backend loads only when asked for

**Costs.** `:374` loads `zend-api.mock.js` (59 KB) on every console boot.

**Change.** An inline guard that `document.write`s the script tag only when
`?mock` is present or `window.ZEND_BACKEND === 'mock'` — the two ways
`zend-api.js` turns it on — keeping execution order.

**Tests.** A boot of `/index.html` without `?mock` (requests collected with
`page.on('request')`) requests no `zend-api.mock.js`; the Playwright suite
(`?mock=1`) still runs on it.

**Risk.** Low.

### G7. Live history keeps its parts (a correction, found here)

**Costs.** `zend-api.live.js`'s `getConversation` mapping drops the server's
`parts`, so live history is always drawn from content, and a sub-agent's job
block — given no paired outputs — shows its tool results as raw user text,
its cards "no result", and its own job blocks never drawn.

**Change.** The live mapping keeps `parts`; the sub-agent block draws each
assistant message from its parts and skips the tool-result turns its parts
already carry. This is what makes G2 apply to history in live mode too.

**Tests.** A Playwright case on the mock with a sub-agent conversation whose
history carries parts: its cards show their results and no tool-result turn
is drawn as text; a unit test that the live mapping keeps `parts`.

**Risk.** Low.

---

## 3. npcd

### N1. The cast reconcile stops copying every world

**Costs.** Every world's beat runs `reconcile_cast` (`beat.rs:87`), which
deep-clones every world's records (`cast.rs:124-125`, `worlds.rs:194`) and
parses every lease (`npc_lease.rs:72-79`): W² full copies per beat.
`recompute_held` re-parses every lease each time.

**Change.**
- `authority/written.rs`: `Written { records, leases }` (with `Debug`,
  `Clone`, `PartialEq`, `Default`) and a pure `held_of(worlds, me)`.
  `Standing.written: Mutex<BTreeMap<String, Arc<Written>>>`. The caller builds
  the `Written` before the lock — `Written::of(records)`, or
  `Written::new(records, lent)` in `write_world`, which already holds the
  leases it lent. `written()` returns the `Arc`.
- The set of characters held here is `held: ArcSwap<BTreeSet<u64>>`.
  `set_written` and `recompute_held` compute it with `held_of` from the
  cached leases and store it **while the `written` mutex is held**, reading
  `node()` inside the lock too (`OnceLock::get`, lock-free) — so `runs()` can
  never disagree with `written`, and a write racing `join_cluster` cannot
  store a set computed without the node. `runs()` reads it lock-free.
  `held()` has no callers and goes; the `recompute_held()` calls that follow
  a `set_written` (`beat.rs:71,152,218,268`) go; `join_cluster` keeps its.
- **Each world's beat reconciles only its own cast**: `reconcile_cast(world)`.
  Nothing in the reconcile crosses worlds, and every hosted world beats, so
  this is exact, O(W) per cycle, with the same one-beat latency — no clock
  and no shared throttle.
- Callers: `beat.rs:70,97-101,136,148,216,265`; `cast.rs:90,125`;
  `perform.rs:249`; `report.rs:28`; `runtime.rs:5485` (a whole-world copy
  per character at boot, today); `tests/worlds.rs` (15 sites, comparing
  `.records` / `.leases`).

**Locks.** `written` is held for an `Arc` insert and the held-set store,
never a deep copy or a parse; no other lock is taken inside it. `runs()`
takes none.

**Tests.** `Written::of` on `{"npclease/7": {holder, expires_ms}, "sim/meta":
{}}` gives exactly `{7: NpcLease{…}}`; `held_of` over two worlds gives exactly
the leases naming `me` (the computation split out because a unit-test
`Standing` has no node); the beat tests unchanged but comparing `.records`.

**Risk.** Low.

### N2. Witnessing returns at once when nothing is new

**Costs.** `npc-map/src/witness.rs:168`: every actor walks the whole event
log (up to 4,608 events) every beat, under the world lock.

**Change.** Return empty when `world.log().last().is_none_or(|e| e.at <=
reader.looked)`, after the actor lookup — every push stamps `at` from the
advancing clock (`witness` re-stamps incoming events) and the log only drains
from the front, so it is ordered and this is exact. `percept()` is still
rendered per actor per beat; this removes only the log walk.

**Tests.** After `mark_seen`, `since` is empty; one `say` after it yields
exactly one event with that `at`.

**Risk.** Low.

### N3. The pulse scan returns early on a quiet beat

**Costs.** `beat.rs:189` filters the whole log every beat.

**Change.** `since` starts at `log.partition_point(|e| e.at <= after)` — the
log is ordered by `at` (N2), so this is exact on every beat, quiet or busy,
and O(log n).

**Tests.** The existing `a_pulse_carries_what_happened_since_the_last`
unchanged (it asserts `since(&log, 8) == ([], 8)`), plus `since(&log, 4)` on
events at `[3, 5, 5, 8]` is exactly the last three.

**Risk.** Low.

### N4. World state is serialised outside the world lock

**Costs.** `records_of` (`world_state/mod.rs:46`) builds the world's body and
lift JSON inside the world lock (`snapshot.rs:76-116`); `split.rs:24`
deep-copies the whole Sim JSON, inside the Sim closure; `store.rs:43` copies
each changed value again.

**Change.** `world_records(State)` — `records_of` takes `World::state` under
the lock (the clone it already makes) and builds the records outside.
`sim_value(&Sim)` serialises inside the lock as now (see §5);
`sim_records_of(Value)` splits outside, `split` taking its `Value` by value;
`sim_records` stays as their composition for the tests. `store::write` takes
`Changes` by value (one production caller, `beat.rs:146`; the test callers in
`store.rs`, `renamed.rs`, `discord/state.rs` updated).

**Tests.** The existing pinned-literal tests
(`a_walking_body_s_record_holds_the_rest_of_its_way`,
`the_world_s_state_is_one_record_per_entity`, the split and store tests)
unchanged in value.

**Risk.** Low.

### N5. A request takes the accounts lock for reading

**Costs.** `api.rs:989-993`: every authenticated request takes
`accounts.write()` to `upsert`, which writes only on a change.

**Change.** `Accounts::current(id, now) -> Option<Value>` — the public answer
when the merge changes nothing (one merge, one substrate read) — and
`accounts::account_of(&RwLock<Accounts>, id, now)`, the one place that takes
the read lock, binds the answer with `let` (never a read guard alive in a
scrutinee around `write().await`, edition 2021; tokio's `RwLock` is not
reentrant), and upserts under `write()` only on a change. `upsert` and
`current` share one `merged`. Used at `api.rs:989`, `api.rs:2119`,
`discord/link/routes.rs:33`.

**Locks.** Exclusive per request becomes shared; `upsert` re-merges under the
write lock, so no update is lost.

**Tests.** `current` `None` for a new identity, `Some` after `upsert`, `None`
after a name change, `Some` when an email arrives empty.

**Risk.** Low.

### N6. Recent ticks are filtered before they are copied

**Costs.** `tick.rs:1378` clones up to 512 records of prose before callers
filter them (`engine/mod.rs:1556,1605`, `pulse.rs:160`,
`guardian/probe.rs:137`), the stream poll every 500 ms.

**Change.** `recent_where(limit, keep)`: of the newest `limit`, those kept —
the same window as today — the predicate taking no lock; and
`latest_tick_of(npc)`, which clones nothing, for the stream's starting point
(`engine/mod.rs:1558`, which only wants the id). Callers: `pulse.rs:160`,
`engine/mod.rs:1607`, `guardian/probe.rs:141`. `recent` goes once nothing
calls it.

**Tests.** `Scheduler::new(8)`, six ticks alternating npc 2 then npc 1 (ticks
0–5): `recent_where(4, npc 1)` → ticks `[3, 5]`; `latest_tick_of(1)` =
`Some(5)`, `latest_tick_of(9)` = `None`.

**Risk.** Low.

### N7. Phone delivery stops counting twice

**Costs.** `environment.rs:413-421`: up to five thread scans per actor per
beat, `waiting_for` among them, recomputing a sum `unread_due` already has;
`phone.rs:276,283` lowercase every member on every call.

**Change.** `unread_due -> (Option<Due>, usize)`, the count the sum of its
per-thread counts on every return path (`unread_threads` keeps n > 0 only and
`mark_delivered` touches no `seen_by`, so it equals `waiting_for`);
`waiting_for` leaves the sweep. `same_name(a, b)` — trimmed, ASCII
case-insensitive without allocating, `to_lowercase` when either side is not
ASCII — used by `has` and `others`.

**Tests.** `unread_due` answers its exact `(Due, count)` tuple; `has` true
for "zen"/"Zen" and "élan"/"ÉLAN", false for "zen"/"Zenith".

**Risk.** Low.

### N8. Zen-task tending polls zend concurrently

**Costs.** `authority/zen_task.rs:171`: one `call_zend` at a time, each up to
30 s, so one slow zend stalls every task's news.

**Change.** The carried tasks are collected synchronously, then asked after
through `tend(asks, 4, call, apply)` — `buffer_unordered(4)`, generic over
the call. Each answer is applied in its own `with_sim`, after re-checking
`standing.writes(world)` and that the carrier is not suspended: a call takes
up to 30 s, and either may change meanwhile. News is posted in completion
order. (`send_up` opens a stream per request, so concurrent calls are
independent.)

**Tests.** `tend` with paused time and a fake call sleeping 30 s: eight asks
complete at exactly 30 s and 60 s, four at each, never more than four in
flight (an atomic high-water mark). The apply step on a fixture world with
one carried task posts exactly one line on the carrier's thread, and none
when the world has turned replica or the carrier is suspended.

**Risk.** Low.

### N9. Console pages poll only while visible

**Costs.** Nine network polls run in hidden tabs, and `app.js` a tenth with
its own pause: `pulse.js:540` (1.8 s, three calls),
`npc.js:300,403,472,655,1240`, `operations.js:354`, `profile.js:315`,
`life.js:269`, `app.js:720-727`.
`pulse.rs:226,252` runs `census()` twice.

**Change.** Each uses W12's `/common/lib/visible.js` `pollWhileVisible(ms,
fn)` — one helper for the estate — whose stop removes its listener and may be
called from inside `fn` (`life.js` stops itself). `npc.js:183,1238` and
`life.js:135,274` call the stop instead of `clearInterval`; `app.js`'s own
visibility code goes. The npcd shell's `index.html` preloads
`/common/lib/visible.js` with the rest of its module graph
(`console::tests::the_preloads_are_exactly_the_shells_module_graph`). The census
route takes one census, filtered once, for both its rows and its names. Lands
after N12, which changes `world_ms` there.

**Tests.** A test walking the embedded console (`tenant::CONSOLE`) asserts
the files containing `setInterval(` are exactly `pages/landing.js` (an
animation) and `lib/api.mock.js`.

**Risk.** Low.

### N10. The rooms' fault list is built once per pass

**Costs.** `engine/rooms.rs:141-143` rebuilds the building-wide fault list
inside the per-room loop.

**Change.** Build it once before the loop as `Arc<[Report]>`;
`Building::hear_of` and `Watch.standing` take the `Arc`, so `watch()` stops
cloning it per call (`stir/mod.rs:190,398,474`, `binding.rs:95`, the test
constructors). A fault reaches rooms ordered before its own one pass
(500 ms) later; rooms speak minutes apart.

**Tests.** The board tests in `engine/rooms/tests.rs` unchanged —
`the_status_board_writes_up_the_fault_that_really_stands` already carries a
fault in one room onto another's board, its text a pinned literal.

**Risk.** Low.

### N11. Bindings are read as a range

**Costs.** `world/binding.rs:160` filters every world's bindings per call;
`mind_of` allocates per lookup; both per delta per beat.

**Change.** `in_world` and `release_world` read
`by_body.range(Bound{world, ""}..)` while the world equals. `advance_for`
builds one body → mind map per moment (`environment::Minds`, from `in_world`)
and hands it to `push_minds` and `deliver_messages`; `push` keeps its
signature, building the map itself for its callers (`perceive` and the
tests). No public signature changes. A bind mid-moment is seen at the next
moment.

**Tests.** Bindings in worlds `a` and `ab`: `in_world("a")` has exactly
`a`'s; `release_world("a")` frees exactly them and leaves `ab`'s bound.

**Risk.** Low.

### N12. World clocks from a lock-free snapshot

**Costs.** `runtime.rs:4453` → `tenant.rs:933`: `block_in_place` plus a
blocking read of the worlds registry, per character per beat; a reload
blocks every beat. `Authored::world_ms` takes the registry's read lock per
call (eleven routes, the census once per row), and `Runtime::world_ms_async`
hops to the blocking pool only to take it.

**Change.** `world_clocks.rs`: `WorldClocks`, an `ArcSwap<BTreeMap<world,
Option<Map>>>` of each world's raw `time` block, built in `Authored::new`
and stored by `reload_authored` while its write guard is held (bound with
`let`; stored on success and on failure, from the registry as it stands).
`Clock::of_time(Option<&Map>, now)` parses it on every read, `of_world`
delegating — so a missing field still means "now" on every call; a world with
no `time` block (`started_now`) and an absent world (0) stay distinct.
`Authored::world_ms` becomes a sync lookup used by every route
(`api.rs:1451`, `ask_api.rs:156`, `engine/mod.rs:1017,1145`,
`mission_api.rs:458,493`, `pulse.rs:232,744,810,863,1009`); the runtime's
resolver reads the same snapshot, so `block_in_place`, `blocking_read` and
`world_ms_async` go. `world_time_for` has no callers and goes. During a
reload a reader sees the clocks before it rather than waiting. (The persona
source, `tenant.rs:958,970`, has the same blocking read per tick and is
listed in §5.)

**Locks.** None on any read path.

**Tests.** At `T0 + 1000`: a world with `time {world_ms: 0, at_ms: T0, scale:
60}` reads `60_000`; one with no `time` block reads `T0 + 1000`; one with
`time {world_ms: 500}` reads `500` at any `now`; an absent world reads `0`.
`world_ms` called on a multi-thread runtime worker answers without
panicking.

**Risk.** Low.

### N13. The tower's clock is its own record (promoted)

**Costs.** Every beat with anyone in a world that has a tower, `tower.advance`
(`engine/watch.rs:78`, `sim/upkeep.rs:181-218`) rewrites the clock's `carry`,
`tap_carry` and `quiet_ms`, and a closing contact's `arrives_in_ms`. The tower
is not in `SIM_MAPS` (`snapshot.rs:36-57`), so it lives in `sim/meta` beside
`built`, findings, the roster and `part_tools`, and the whole meta record is
diffed, committed and replicated twice a second. Nothing else in meta changes
per beat; the stockpile's energy changes each time a whole unit drains (every
3 s at the base draw, about 1.2 s with shields up). Only `battle-cities` has a
tower.

**Change.**
- `SIM_FACETS = ["tower/clock", "tower/contact"]` in `snapshot.rs`.
  `split(value, maps, facets)` takes each facet out whole, `null` included,
  only when its parent is present, so a world with no tower emits no tower
  records. `join(records, maps, facets)` sets a facet that has a record into
  its parent only when the parent is an object — never creating it, so a null
  tower never becomes `{clock: …}` — and leaves one with no record as meta
  holds it. `sim_records_of` (N4) passes `SIM_FACETS`; `apply.rs`'s `sim_of`
  joins with them; `split.rs`'s own tests pass `&[]`. Lands after N4.
- **The rest of the tower stays in `sim/meta`, and no build reads this layout
  as "no tower" — so no lease floor is needed.** Both facets are
  `#[serde(default)]` on `Tower`, so the running build decodes the new
  records with the tower whole (its clock at default, no contact) and ignores
  the two new records. A build that writes meta after reading them writes
  the tower back with its own clock and tombstones the two facets, and `join`
  reads that too. A mixed or rolled-back cluster loses at most the tower's
  clock and a contact that is closing. (A lease floor was considered and
  cannot work: each member takes leases itself, so an old binary never runs a
  new check. Moving the whole tower out of meta would need two releases — a
  reader first, then the writer — and is not taken.)
- `sim/tower/clock` changes every beat; `sim/tower/contact` every beat while a
  contact closes; `sim/meta` for tower reasons only when stock, posture,
  shields, siege, position or the queue does. A commit still happens every
  beat, with less in it.

**Locks.** None new.

**Tests.** `split` of a facet whose parent is present, null or absent; a
`null` facet round-trips; `join` never turns a null parent into an object.
The battle-cities seed splits into exactly `sim/meta` (its tower with no
`clock` or `contact`), `sim/tower/clock = {"carry":0,"tap_carry":0,
"quiet_ms":0,"sightings":0}` and `sim/tower/contact = null`, plus its map
records, and `sim_of` gives the seed back; the vault has no `sim/tower/`
records; a meta whose tower still holds its `clock`, with no facet records,
decodes with that clock. With a body in the foundry and no contact, the diff
across one `watch_at(+500 ms)` puts exactly `sim/tower/clock = {"carry":10000,
"tap_carry":0,"quiet_ms":500,"sightings":0}` — so no other meta field churns
on a beat.

**Risk.** Low.

### N14. A replica skips re-applying an unchanged world (promoted, with a dirty flag)

**Costs.** `beat.rs:212-219`: a replica re-reads and re-applies its whole
world whenever the substrate-wide `custom_objects_version` moves — which any
custom-object write anywhere does: every world's beat, accounts, bot state.
`apply` (`apply.rs:144-158`) decodes the Sim and replays it, under the world
lock — about one full decode per world per beat on every replica.

**Change.**
- **The flag lives in `Hosted`'s own methods, not at their callers** — a list
  of call sites would miss one (Discord tasks landing after an apply, a
  replica's task acts, handset issue, the mission loop's loads). `Hosted`
  carries `local: AtomicBool`, "changed here since its records were last
  taken in", starting set. Every method that can change what the records
  hold sets it: `with`, `with_sim`, `with_both`, `with_dms`, `with_servers`,
  `with_mirror`, `tick`, `stir`, `watch_tower`. What the records do not hold
  does not: the attention sweep, the event log a pulse is witnessed into
  (`Hosted::witness`), a body's decision (`begin_decision`, `end_decision`,
  `restore_decision`), and reads — including a new `read_both`, which
  `push`, `push_one` and `mission_compass` use (they use `with_both` today
  only to read). `apply` clears the flag **before** it takes the records in,
  through accessors that do not set it, so a mutation racing the apply leaves
  the world local. A world its authority beats is always local, so a world
  moved to Replica applies on its first follow beat.
- `follow_world`, after `store::read`, skips `apply` and `set_written` (which
  after N1 also recomputes the held set) when the world is not local and the
  records equal `standing.written(world).records` (N1's `Arc`, compared
  without a clone); it still sets `followed`. Every input to apply is then
  what the last apply had, and apply is idempotent and touches neither
  attention nor the scheduler. The skip saves the decode and the apply; the
  `store::read` still copies the world.
- `Missions::adopt_workflows` carries `round_items` as it carries the
  routines. (Today every apply on a replica wipes the round, so `table_summons`
  for a character running on a replica misses it — a live defect, fixed here
  because N14's skip would otherwise make it intermittent.)

**Locks.** The world lock is not taken on a follow beat that skips; the flag
is an atomic.

**Tests.**
- A replica restored from records, then an unrelated custom-object write:
  the follow beat takes nothing in (a take-in counter unchanged) and
  `followed` is at the new version.
- A body placed locally through `with`: the next follow beat takes the
  records in and the body stands where its record says.
- A DM put with `with_dms` on a replica is gone after the next follow beat.
- A witnessed pulse and a `push` leave the world not local.
- After an authority beat the world is local; `hand_over` to Replica, then
  one follow beat, takes the records in.
- The round set on a replica survives an apply.

**Risk.** Low, given the flag lives in `Hosted`'s mutating methods.

### N15. Percepts are composed only for bodies perceived here (promoted)

**Costs.** `Attention::sweep` (`npc-map/src/delta.rs:142-181`) composes a
full percept, and walks `since`, for every actor every beat under the world
lock; `push` (`environment.rs:287-292`) then discards the deltas of bodies
with no mind here or whose character runs on another member — about half the
cast on a two-member cluster.

**Change.** `Attention::sweep_for(world, keep)`: a kept body goes through
`take` as today; a body not kept gets `shown.remove(id)` and
`world.mark_seen(id)` — its cursor still advances, as `environment.rs:269-273`
requires, so a character lent here later gets no backlog — and nothing is
composed. `sweep` is `sweep_for` keeping every body; `Hosted::sweep_for(keep)`
replaces `Hosted::sweep`. `push` builds the body → character map of the
bodies perceived here (N11's map, filtered by `reach.here`) before the world
lock, sweeps with it and delivers from it; the per-delta `mind_of` and
`here` checks go. `wake_character` calls `Hosted::forget` for the body it
wakes (nothing in production called it), so a character lent back is
grounded afresh even if its fingerprint outlived its context. The first delta
after a body starts being perceived here re-grounds it with a full percept;
`shown` is read nowhere but `Attention`. With N1, a world's beat reconciles
its own cast after its own sweep, on the same thread, so a wake cannot race
the keep set.

**Locks.** The world lock is held for less time on the beat; nothing new is
taken inside it.

**Tests.**
- Bodies m1 and m2, only m1 kept: after a third body speaks, `sweep_for`
  returns only m1's delta and m2's `since` is empty afterwards; with both
  kept, m2's next delta carries a percept.
- A character lent back here (`wake_character`) receives its percept on its
  first delta.
- `a_body_with_no_mind_still_has_its_cursor_advanced` changes to assert the
  extra's `peek(..).events` is empty (its percept is now composed afresh when
  asked).
- `cargo test -p npcd --lib authority::` and `cargo test -p npcd --test
  worlds` pass unchanged — in particular the four lending and authority
  tests in `tests/worlds.rs`, `npc_lease.rs`'s
  `characters_are_lent_by_the_plan`, and `environment.rs`'s
  `a_character_running_elsewhere_is_told_its_messages_through_the_reach`.

**Risk.** Low–medium; it lands in its own commit.

---

## 4. Gateway, cluster, MCP aggregator, tokera.com

### W1. Route tables are shared, not cloned

**Costs.** `cluster/src/gateway/hub.rs:1849` deep-clones a member's route
table (about 100 routes and a matchit tree) per request, under the hub's
global mutex; `registry.rs:463` and `pick.rs:212` allocate match params to
test a match; `hub.member()` clones whole member views at `aggregate.rs:232`,
`route.rs:202`, `merge_ws.rs:146`, `page.rs:227`, `jobs.rs:85`.

**Change.** `MemberView.routes: Option<Arc<RouteTable>>`; `Registry::routes`
wraps the decoded table in an `Arc`; `Registry::{table, table_for}` return
`Option<&Arc<RouteTable>>` and `Hub::{table, table_for}` return
`Option<Arc<RouteTable>>`. `RouteTable::answers(method, path) -> bool` (the
same method rule and matcher; matchit keeps its params inline, so no
allocation) replaces `lookup().is_some()` in `Registry::table_for` and
`Registry::declaring`. Hub methods `has_member`, `member_host`,
`member_labels(app, &[u64])` replace the member clones at `route.rs:202`,
`page.rs:227`, `jobs.rs:85`, `merge_ws.rs:146`, `aggregate.rs:232` (one call
after the fan-out, zipped by index). `Hub::member` stays for the tests that
read sessions and states through it. `cluster/tests/hub_member.rs:189`
compares through `as_deref()`.

**Locks.** The hub mutex is held for a scan and an `Arc` bump instead of a
deep copy; an aggregate takes it once instead of once per member.

**Tests.** `answers` agrees with `lookup().is_some()` over `zend_like()` for
every route and for unknown paths and methods; two `table_for` calls are
`Arc::ptr_eq` and deref-equal the table given; `member_labels` answers exact
tuples, `("", 0)` for a node not admitted (the `replace_tests` fixture,
lifted to a shared test helper).

**Risk.** Low.

### W2. Routing is decided under one lock

**Costs.** `server.rs:721` (`admitted`), `route.rs:196` (`table_for`) and
`none_ready` (`route.rs:184,187`) lock two to four times per request.

**Change.** `Hub::routing(app, method, path) -> Unadmitted |
StandingBy(Option<u64>) | Table(Arc<RouteTable>)` under one lock;
`server.rs` calls `route::route` unconditionally and `route` answers `Dumb`
for `Unadmitted`. The `Pick::Configured` branch after `place().await`
(`route.rs:311`) reads fresh state through `Hub::standing(app) ->
Unadmitted | StandingBy(Option<u64>)` — one lock in place of `admitted` +
`standing_by`, never a table, so it cannot re-place. `Hub::admitted` and
`Hub::standing_by` stay for their other callers.

**Locks.** Two (ready) or four (miss) acquisitions become one, plus one on
the `Configured` path.

**Tests.** Exact variants for: an unadmitted app; a loading-only app
(`StandingBy(Some(n))`, lowest session); a ready member with no table yet
(`StandingBy(None)`); a ready app (`Table`, `Arc::ptr_eq` to the member's).

**Risk.** Low. Lands after W1.

### W3. Aggregation borrows its answers

**Costs.** `web/src/cluster/aggregate.rs:241` clones each member's body to
`fuse` them.

**Change.** `fuse(&[&Value])`; the bodies are moved into the envelope after.

**Tests.** `fuse`'s cases unchanged in value.

**Risk.** Low.

### W4. The proxy moves its headers

**Costs.** `proxy.rs:219` clones the request's headers, never read again.

**Change.** `std::mem::take(&mut parts.headers)`.

**Tests.** The proxy tests unchanged.

**Risk.** Low.

### W5. An MCP answer never waits for a listing

**Costs.** `mcp/src/app.rs:491-492` lists every backend before recording the
call, with the answer in hand: seconds after a restart, up to 60 s with a
backend down, on every call while listings keep failing.

**Change.** `context::after` splits into `stamps(name, begins, failed,
delivered)` and `records(name, failed, from_backend, read_only)`, with
`may_record(name, failed)` the part of `records` known before a listing. The
call keeps or gives up its claim at once and returns its answer. A call that
could record (`may_record`: not failed, not a memory or chat tool) hands its
caller, session id, session and episode to a spawned task, which — when this
process has not yet listed successfully (`listed: AtomicBool`, set on any
backend's successful listing, so a listing that marks no tool read-only does
not respawn forever) — lists in its turn (`turn: tokio::sync::Mutex<()>`),
counted by a `generation: AtomicU64` read before waiting: a task that waited
while another listed does not list again, and decides with what that listing
found, as a call does today. Only then does it read `route(name)`,
`reads_only(name)` and `route(GET)` (today read before the listing, so a fresh
process recorded nothing) and record as today. `list` takes
`(&Caller, &OnBehalf<'_>)`, the task rebuilding its `OnBehalf` from an owned
id and cloned session as `record_episode` does. The listing state and the
record task live in `mcp/src/listing.rs`; `app.rs` calls it.

**Locks.** `turn` is held across the listing's awaits, by background
tasks only; no request takes it. A tokio mutex does not poison, so a failed
or panicked listing releases it.

**Tests.** In `listing.rs`, against a fake gateway on a local port whose
`tools/list` waits on a `Notify`: the first call returns while the listing is
held; once released, a read-only tool's call records no `episode` and a write
tool's records exactly one; three calls during one held listing cause one
listing. (`mcp` is a binary crate; the fixture is new and in-module.)

**Risk.** Low. `mcp` is outside zend's `latency` gate, so this test is the
item's only timing check.

### W6. Merging by id is linear

**Costs.** `aggregate.rs:179-183` searches the merged array per item.

**Change.** An id → index map per array, seeded from the first body keeping
each id's first occurrence and inserted on push; null ids skipped. Keys are
the canonical JSON of a string or number id; an id of any other type is
matched by the linear search as now.

**Tests.** The exact merged output for overlapping bodies, a duplicate id
within one body, an item without an id.

**Risk.** Low.

### W7. Merged log lines are tagged by splicing

**Costs.** `cluster/merge_ws.rs:43-55` parses and re-serialises each line.

**Change.** A per-member suffix (`,"node":…,"host":…,"member":…}`, built once
in `read_member` with serde) spliced before the last `}` of a line that
parses as JSON (`IgnoredAny`, no allocation) and starts with `{`; an empty
object takes it without the leading comma; anything else wrapped as today.
The member's own bytes are kept; a line already carrying one of the three
keys has it twice, and `JSON.parse` keeps the last, as the browser sees today.

**Tests.** Exact outputs for an object, `{}`, `{ }`, trailing whitespace
after the brace, a brace-delimited line that is not JSON (`{oops}` →
wrapped), and a non-object line; the existing two tests unchanged.

**Risk.** Low.

### W8. Static files are prepared once per version

**Costs.** `content.rs:137-155` reads each file per request (embedded ones
copied too); `asset.rs:149,158` hash and gzip it every time — the 327 KB
console about 6 ms per load.

**Change.**
- `asset/prepared.rs`: `Prepared { mime, tag, tag_gz: Option<String>,
  plain: Bytes, gz: Option<Bytes> }`, built by `Prepared::typed(mime,
  bytes)`, with `Prepared::of(name, bytes)` its composition with
  `mime_guess`.
  `respond(&Prepared, Cache, &HeaderMap)` picks the representation, the tag
  and the `Cache-Control` from `directive(mime, cache)` per answer — the
  policy is the caller's (the gateway's config, the host's default), never
  the cache's. Every `respond` call moves to it (`server.rs:811`,
  `host/src/ui.rs:89`, the `asset.rs` tests).
- `content/` (from `content.rs`): `Roots { sources, stamped:
  Arc<Stamped<Prepared>> }`, `Stamped<V>` generic over the value kept (W10
  reuses it)
  (with `Default` and a manual `Debug`), so every clone — the gateway's, the
  tokera router's, the host's — shares one cache.
  `Roots::resolve(rel)` walks the roots in order every request (canonicalise,
  `starts_with`, `metadata`) and yields `Disk { real, (mtime, len) }` or
  `Embedded(&'static File)`, so a file added to an earlier root still
  shadows. `Roots::prepared(rel)` looks the result up by `(real path or the
  File's address, mime)` — never by the bytes' address, which the linker may
  share between byte-identical files. A disk entry is kept only when a stat
  after the read agrees with the one before it.
- Kept: embedded files whole as `Bytes::from_static`, outside the cap; disk
  text up to 8 MiB whole; other disk files `(mime, tag)` only, so a `304`
  reads nothing and a `200` reads the bytes and reuses the tag. Kept bytes are
  capped at 64 MiB; an insert past the cap clears the map (deleted files'
  entries live at most until then).

**Locks.** std `RwLock`, read for a lookup, written on a miss, never held
across an await; a racing miss prepares twice, harmlessly. A miss hashes and
gzips on the runtime thread, as every request does today.

**Tests.** The gzip's fixed header bytes and `gunzip(gz) == plain` (not raw
deflate bytes, which depend on the backend); the plain body and both tags as
literals; two `prepared()` calls are `Arc::ptr_eq`; a later mtime
(`filetime`, already a dev-dependency) serves new bytes and a new tag; an
embedded file is prepared once; a file added to an earlier root shadows the
cached one; the `asset.rs` tests unchanged in value through `Prepared::of`.

**Risk.** Low.

### W9. tokera.com pages answer 304

**Costs.** Pages (`site/tokera/papers.rs:209`, `blog.rs:160,226`,
`home.rs:343`) go out with no `ETag` or `Cache-Control`
(`server.rs:683`); the home page, nav bar and footer are formatted per
request from constants.

**Change.** `asset::respond_typed(mime, Bytes, Cache, &HeaderMap)` —
`respond` over `Prepared::typed`. `site::for_site` and `tokera::router` take
the server's `config::Cache` (`State.policy`, beside the markdown `cache`);
each route's closure extracts `HeaderMap`, and the handlers take
`(state, &HeaderMap)`. Pages answer `text/html; charset=utf-8` (the type
`Html` sends today) through `respond_typed`: tag `"<sha16>"`
(`"<sha16>-gz"` when gzipped there, which the compress layer then passes
through untouched), `Vary: accept-encoding`, `public, max-age=<short_secs>`,
304 on a match. No page reads cookies or identity — sign-in is filled in by
`/lib/auth.js` — so `public` is right. A dynamic page is still rendered and
hashed to answer a 304. 404 and 500 unchanged; robots, sitemap and feed
unchanged. The home page (with its gz and tags) and the footer are
`LazyLock` statics; `nav_bar` returns one static per `Nav`.

**Tests.** `/blog` without `Accept-Encoding` answers 200, `public,
max-age=60`, and the tag of the received bytes; a repeat with it answers 304
with an empty body; with `Accept-Encoding: gzip` the tag ends `-gz` and the
body inflates to the plain page; a missing page 404 with no tag.

**Risk.** Low.

### W10. Blog posts and the papers manifest are parsed once per change

**Costs.** `blog.rs:43-64` reads and parses every post per request;
`papers.rs:87-103` parses the manifest per request.

**Change.** `content::Stamped<V>` is generic — the store under W8's file
cache, keyed and validated the same way, by `Roots::resolve`. The tokera
`State` keeps `posts: Stamped<Option<Arc<Post>>>` (`None` for a draft or
broken front matter, so a broken file is not re-parsed per request) and a
one-slot `Arc<Manifest>`. `listing`, `published` and `manifest` return
`Arc`s; `discovery.rs` (`:86`, `:98`, `:135`) and the `pane_index` callers
read them by reference. The directory is still listed per request (a
`read_dir` and a stat per post — far cheaper than reading and parsing).

**Tests.** Two listings return the same `Arc` posts; an edited post's alone
changes (`tempfile` + `filetime`, as `markdown/mod.rs` does).

**Risk.** Low.

### W11. The papers redirect is cacheable

**Change.** `papers.rs:155`'s 307 carries `public, max-age=<short>` (from
W9's `State.policy`); its comment, which says 302, is corrected to "307
(temporary), never 301".

**Tests.** The `/papers` 307 carries `public, max-age=60`.

**Risk.** Low. Lands with W9.

### W12. Console pages poll only while visible

**Change.** One helper for the whole estate,
`web/content/common/lib/visible.js` `pollWhileVisible(ms, fn) -> stop`:
skips while hidden, polls once on becoming visible, `stop` clears the timer
and removes the listener. Used by `perf.js:834` (its `stop` calls it; its
pause and its age clock at `:826` stay its own), `substrate.js:483`,
`zend/web/substrate.html:246`, `project.html:282`, and by N9's npcd sites
(npcd serves `content/common`). `agents.html` already gates on
`document.hidden` and is left; `npcd/app.js` moves to the helper under N9.
The zend console's job poll (`index.html`) is a single-flight timeout chain,
not an interval, and keeps its own visibility gate (G3).

**Tests.** The `setInterval(` scan as two more tests beside N9's — host over
`ui::COMMON` (`content/common`) and zend over `tenant::WEB` (`zend/web`):
`setInterval(` appears only in `visible.js` and the listed non-polling files
(`perf.js`'s age clock, `agents.html`, `index.html`'s retry tick, the mocks).

**Risk.** Low.

### W13. A smaller favicon

**Change.** `web/content/common/brand/zen.png`, `zend/web/favicon.png` and
`zend/web/favicon.ico` (itself a PNG), byte-identical at 109,363 bytes,
replaced together by a lossless 256×256 recompress — the size the MCP
`serverInfo` icon declares (`mcp-server/src/protocol.rs:186-190`). The site's
JSON-LD logo is `tokera/favicon.png`, a different file, untouched. The
re-encode is done through Zen's `image_edit` (a 256×256 resize of the current
mark, PNG); John places the returned file at the three paths, since no file
tool writes a binary and the repository forbids a shell write into the tree.

**Tests.** `the_switcher_icons_are_the_marks_their_sites_serve`
(`page.rs:781`) also pins `zend/web/favicon.ico` to `brand/zen.png`; a new
test reads `brand/zen.png`'s IHDR and asserts width and height are exactly
256 (bytes 16–23).

**Risk.** Low.

---

## 5. Considered and held back

- **Cloning the Sim to serialise it outside the lock** (npcd): `Sim::clone`
  copies fields serde skips; needs measuring first.
- **gzip on zlib-rs**: features unify into zend-vfs's git pack code.
- **The hub registry as one snapshot**: placement mutates it per request.
- **Keeping message objects across a re-hydrate**, **counting history without
  building it**, **evicting the GUI's lazy bodies**: correctness traps for
  little gain.
- **`code_read`'s per-file probe** (`code_read/mod.rs:851`, a full
  ingest-chain scan per file): a separate cost from Z8, worth its own change.
- **npcd's persona source** (`tenant.rs:958,970`): a blocking read of the
  worlds and personalities registries per tick — the same trap as N12, a
  outside this patch: it reads a second registry (personalities) that N12's
  snapshot does not hold.
- **The other per-call metadata scans** (`guidance/reads.rs:63`,
  `code_read/link.rs:94`, `repo_scan/link.rs:68`): the same shape as Z2,
  outside this patch: each needs its callers restructured to batch, which Z2
  does only for `file_read`.

## 6. Validation

Each item went through three passes against the code: a design pass, an
adversarial pass that tried to break it (contention, lock order, staleness,
behaviour changes, wrong premises), and an implementation pass that worked out
every call site, signature, lifetime and test fixture as it would be written.
Their findings are folded into the items above; none was found unworkable.
N13–N15, promoted from §5 on 2026-11-09, went through the same checks plus a
final cross-item pass over the whole document. What the passes changed most:

- **N1** reconciles each world at its own beat instead of throttling across
  worlds, and computes the held set under the `written` lock.
- **Z1** defines its fallback, and covers the same quadratic decode in the
  narrator and in npcd's prose preview.
- **Z3** names every single-key reader (about twenty more than first listed).
- **W5** keeps the record-or-not decision exact: a waiting task decides with
  what the one listing found, never with an empty read-only set.
- **W8** keys on the canonical path and the mime, keeps the cache policy out
  of the cache, and caps its memory.
- **G1** redraws a changed job in place: a running job's line carries a
  seconds clock, so a memo alone would still rebuild the transcript every 2 s.
- **G7** was found here: live history drops its parts, which left a
  sub-agent's tool results drawn as text.
- **N13** splits only the tower's clock and contact out of meta, which every
  build already reads, in place of a lease floor no old binary would obey.
- **N14** sets its dirty flag inside `Hosted`'s mutating methods, and fixes
  the replica round being wiped by every apply.
- **The landing order** was corrected: G3 needs Z9's sink and Z5's `get`, and
  Z5 needs Z6's `Viewer`.

**Found in live code, outside this patch:** `zen_task_act` run on a replica
sets the task's handle on the local copy only — it never reaches the
authority and is wiped at the next apply, so the authority never polls that
task. It needs its own fix (the act sent to the authority, as other acts of
the world are).

**Test harness additions.** zend's dev-dependency `tokio` gains `test-util`;
the Playwright suite (`zend/web-tests`, `@playwright/test` ^1.45) starts
using `page.clock`, installed before `goto`; the zend mock gains a frozen job
(`__ZEND_MOCK_JOB_STILL__`, G1), a streamed reply of three closed calls and a
live think whose results count their `toJSON` calls (G2), and a sub-agent
conversation whose history carries parts (G7); pure GUI modules
(`job-poll.js`, `tool-responses.js`, `conversation-merge.js`) follow
`partial-call.js` with `node --test`; `mcp` gains an in-module fake-gateway
fixture (W5); `candle-conversation/tests/common/real_vocab.rs` takes
`literal_text_real_vocab.rs`'s tokenizer finder, shared by it and Z1; a
substrate helper sealing a turn's layout from user and assistant text (Z7);
`cluster`'s `replace_tests` fixture lifted to a shared test helper (W1); and
the `setInterval(` scan as three tests — npcd over `tenant::CONSOLE` (N9),
host over `ui::COMMON` and zend over `tenant::WEB` (W12).

## 7. Decisions and landing

**Decided (2026-11-09):**
- **Z1 is in**, in a commit of its own, landed last, so any regression in the
  streamed text points at one change. **Its gate is a landing requirement:**
  `host/tests/text_stream_model_vocab.rs` holds `#[ignore]`d
  `the_streamed_text_is_the_full_decode_for_the_prescribed_model`, which
  fetches `MODEL.spec()`'s tokenizer through `hf-hub` (a cache hit on every
  dev machine) and **fails, never skips**, when it cannot. BlackWorld runs it
  through its agent before the Z1 commit, with motherd stopped:
  `cargo test --release -p host --test text_stream_model_vocab -- --ignored
  --nocapture`, and the passing line goes into the Z1 commit message. The
  loud-skip test in `candle-conversation` stays for breadth across every Qwen
  tokenizer in the cache.
- **W9's pages use the gateway's short cache age** (60 s today), with an
  ETag and 304.
- **N10's one-pass delay and Z10's briefly stale index are accepted.**
- **N8's completion-order news is accepted.**
- **N13, N14 and N15 are promoted** from §5: the tower's clock split out of
  meta, the replica skip behind a dirty flag, and percepts only for bodies
  perceived here. (A lease floor was chosen for N13 first; the final check
  showed it cannot bind an old binary, and a narrower split that every build
  already reads made it unnecessary.)
- **W13's favicon** is re-encoded through Zen's `image_edit` (a lossless
  256×256 resize of the current mark, PNG), which John places at the three
  paths.
- **No before-and-after measurement**: the tests and the trial carry it.

**Who and how.** BlackWorld implements and tests the patch in batches, each
its own commit, in this order:

1. npcd beat: N1, N2, N3, N12, N4, N11, N7, N10, then W12's helper and N9
   (after N12), then N5, N6, N8.
2. npcd promoted: N14, then N13, then N15 (its own commit).
3. Gateway and aggregator: W1, W2 (after W1), W3, W4, W6, W7, W5.
4. Static and site: W8, W10, W9, W11 (after W9), W12's remaining sites
   (the consoles' polls), W13.
5. zend server: Z9, then Z4 (after Z9), Z2, Z3, Z6, then Z5 (after Z6), Z7,
   Z8, Z10, Z11.
6. GUI: G1, G3 (after Z9 and Z5: its `size` is Z9's sink, its `turns` is
   added to Z5's `get`), G2, G4, G5, G6, G7.
7. Z1, alone.

Dependencies within: N1 before N14 (the `Arc<Written>` compare); N4 before
N13 (`split` by value, `sim_records_of`); N11 before N15 (the body → mind
map); N12 before N9; W12's helper before N9; W1 before W2; W8 before W9 and
W10 (`Prepared`, `respond_typed`, `Stamped<V>`); W9 before W11; Z9 before Z4
and G3; Z6 before Z5; Z5 before G3; G1 before G3 (`setJob`, `rev`); G2
before G7. W12's scan tests land in batch 4, once its last sites are
converted.

**Release.** The batches land on one branch and go out together as a single
trial and release once every batch is done and green. Merging to main, the
trial and the rollout to WhiteWorld are John's calls.
