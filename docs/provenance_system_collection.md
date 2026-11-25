# Provenance in the system prompt: the `files` collection

**Status:** design, 2026-11-26. Supersedes the reading-turn placement of
`docs/zend_working_set.md` §4.9 (faces) and the priming-chain lineage of
`docs/zend_workspace_execution.md` §6 / `docs/zend_branch_ingest.md` §6 for a
dialogue's context. Where this document and those disagree, this one holds.

## 1. The change in one paragraph

Everything provenance places today as sealed *turns* — code readings
(`scopes`), guides (`guidance`), journal notes (`journals`) and the priming
chain the dialogue forks from — moves into the **system prompt**, as members
of one dynamic collection, `files`, rendered in Qwen-Agent's knowledge format.
It is built per projection the way the `tools` collection is, and selected
the way readings are selected now: a large pool chosen by provenance
momentum, reduced to what fits by the QSA index. The seed stays a chain
computed with the recurrent state and snapshotted, now as the collection's
last members; the dynamic members are placed before it, and when a
conversation opens the system prompt's close is forwarded from the seed's
snapshot over what the opening placed, so its end is a real forward and the
conversation's recurrent state starts there. A reading stops
being a conversation: ingest prefills each file's chunks as collection
members and asks the file's calibration questions; nothing is decoded about
a file but those answers. The dialogue's turns are the dialogue's own and
nothing else.

## 2. Why: the model cannot read files presented as turns

Measured 2026-11-26 on BlackWorld (build 5602) against the 128-question
dependency battery (`docs/unbounded_agents.md` §9.12):

- **The paper's gap.** Of the expected items in the text of an answer file
  the memory placed, C1 names 69.9%; from the same kind of text BM25 hands
  over as a files block, 97.6% (§9.12 table, `table_9_12_in_view.csv`).
- **Whole files, not fragments, are missed.** In C1's seventh run seven
  questions (t13, t25, a15, a13, a26, t07, t40) had every expected item in an
  injected answer file and named none. The answer files were placed whole
  (a15 5/5 turns, t07 4/4, a26 2/2 and 6/6). The answers decline, listing an
  inventory of "what I read" that leaves the answer file out.
- **The model reads the slot as one confused dialogue.** Reproduced on
  conversations `mcp:685da41c19f7f5ff:probe-kv/c1-{t07,a15,a26}-a`. t07's
  thinking: *"User: file=journals/… Assistant: file_read … then I wrote a
  response about docs-write.md?? That's odd"*, after which it stops. a26:
  "tiers.rs, the only file I read" with pass.rs and warm_copy.rs placed
  beside it. a15: content of `ingest_lease.rs` credited to `member.rs`.
- **It is the presentation, not the K/V or the recurrence.** Decode diag run
  `1795634912233` on a26, forced to list every file read: *sparse*, *dense*
  and *fresh* (every piece re-prefilled, the recurrence run over all of it)
  each list one file, the same journal note. Re-sealing does not help, so
  neither K/V portability nor the recurrence decides it here.
- **The same files as plain text are read.** Probe arms of the battery
  (`zend/examples/dependency_battery/probe.rs`, passthrough route, C1's own
  selection shared per question), on those seven questions:

  | Arm | Where the selected files sit | In-view items named |
  |---|---|---|
  | C1 run 7 | reading turns (production) | 0 |
  | S1 | end of the system prompt, `=== path ===` blocks | 15 / 16 |
  | Q1 | end of the system prompt, Qwen-Agent knowledge format | **16 / 16** |
  | M1, U1, S2 | own system block / user message / seed + 96K budget | ≈ all |

  (`docs/results/placement_probe_2026-11-26/answers.jsonl`,
  `qwen_format.jsonl`.) The three questions whose answer files the selection
  did not place stay near 0 in every arm: those are selection misses, a
  different problem.

Qwen's own agent framework places retrieved files exactly here: appended to
the system message under `# Knowledge Base`, each file `## The content from
<source>:` in a fence (`qwen_agent/agents/assistant.py`,
`KNOWLEDGE_TEMPLATE_EN`, `KNOWLEDGE_SNIPPET_EN`). Every other system surveyed
(Aider, Anthropic's and OpenAI's long-context guidance) gives each file a
plain header and puts the question after the files; none renders a file as a
tool call.

## 3. The prompt

```
<|im_start|>system
persona · dials · role · history_stance · grounding
journal_intro · journal members
tools_overview · tool summary · <tools> tool members </tools>
files_open                      # "\n# Knowledge Base\n\n"
dynamic members                 # provenance's selection, sealed, attended only
seed members                    # the seed chain, sealed with the recurrence
close row  <|im_end|>\n         # forwarded once, at the opening (§6.2)
<|im_start|>user …              # the dialogue's own turns only
```

The prompt checkpoint holds the system block with the collection **empty**
(`files_open` and its members are outside the checkpoint, as tool members
are). Everything the collection holds arrives per projection.

- `files` is **the end of the system prompt**: the last thing before
  `system_close` and so before the first user turn, after the tool catalog.
  That is where Qwen-Agent appends knowledge and where the probe that named
  16 of 16 in-view items put it. Nothing in the system block follows it.
- The notation is **Q1's, byte for byte** (`probe::qwen_knowledge`, pinned by
  its tests): `files_open` is `\n# Knowledge Base\n\n` — after the section
  before it ends its line, that is the blank line Q1 had between the persona
  and the heading — a template gated `depends_on: files` like `tools_open`;
  and each file placed is one snippet

  ````
  ## The content from candle/zend/src/ingest_lease.rs:

  ```
  …the file's placed text…
  ```
  ````

  the snippets joined by a blank line. The source is the workspace path,
  repository first; there is no line range in the heading, because Q1 had
  none.
- A file whose chunks are not all placed keeps **one heading**: its placed
  chunks follow in line order inside the one fence, a contiguous run joined
  as the file's own text and a gap marked by Qwen-Agent's own separator for
  snippets of one document, `\n\n...\n\n`. A file placed whole renders
  exactly as Q1 rendered it.
- So a file's rendering is built from sealed pieces: its **heading piece**
  (`## The content from <path>:\n\n```\n`), its **chunk members**, the
  **gap piece** (`\n\n...\n\n`) where placed chunks are not contiguous, and
  its **closing piece** (`\n```\n`, a blank line after it when another file
  follows, as Q1 joined its snippets). The heading, gap and closing pieces
  are small sections of the collection, sealed as members are (§5). Every
  file, the last included, seals its own closing fence, so the collection
  ends `\n```\n` and the close row is `<|im_end|>\n` (§6.2).
- The **dynamic members** come first and the **seed** last, nearest the
  question, in its chain's fixed order (§6).
- **Dynamic order is strongest first, and stable once placed (decided
  2026-11-26).** The opening places its files in fill order, best-ranked
  first, the order every probe arm used. From then on a placed file keeps
  its place relative to the others: a reprojection never reorders what it
  keeps. A file it adds goes after the kept ones, at the end of the dynamic
  run before the seed, and a file it drops leaves the others in their order.
  Reordering placed content during a decode has been seen to unsettle it;
  rank decides what is placed, not where a kept file moves to.

## 4. Selection: the pool and the fill, unchanged, now feeding a collection

Readings are not chosen as tools are (one belief step over every member),
and must not become so: their reduction is what lets an unbounded corpus fit.
The two stages of `docs/hit_driven_projection.md` §7 and
`docs/zend_working_set.md` §4 keep their meaning and their code:

1. **Pool — provenance proposes.** The working set's momentum over the
   gallery scan (`working_set/state.rs` `choose`; `resolver/belief_groups.rs`,
   `cases: file`, `vote_top_k 8`, `kind_z`) picks whole files, the card's
   `PoolSize` worth (2M tokens on 72 GB, 500K on 24 GB, 256K on 16 GB) with
   their index pages only.
2. **Fill — the index decides.** `dialogue_fill::plan` / `fill` places chunks
   by score under the kind caps, the score lifted by QSA index hits
   (`base·(1+gain·h)`), the shadow scan scoring pooled-but-unplaced chunks
   (`scheduler/pooled_rounds.rs`).

What changes is the destination: the fill's placed chunks become **members
of `files`** (`ProjectionSegment::Sealed(SealedKind::Section)` inside the
system block) instead of sealed assistant turns after it.

**A file is keyed as a file (decided 2026-11-26).** Every structure the two
stages use names a file by a reading's timeline and turns today: the working
set (`working_set/state.rs:89,133,223-239`), the fill's pieces
(`WorkingPiece.exchanges: Vec<Vec<TurnKey>>`, a reading's floor its request
and answer, `dialogue_fill.rs:36-40,110-118`), the pool's pages and hits
(`PoolPages`, `pooled_rounds.rs:21,26-64`), belief and normalization
(`FileScan.timeline`, `ChildKey::timeline`, `belief_groups.rs:328-341`), and
calibration's votes (`CALIBRATES_KEY` on the unit's timeline). They are
re-keyed to **`FileKey`** (the workspace path and the content's blob) and
**`ChunkKey`** (a `FileKey` and the chunk's part span):

- the working set's members and momentum are `FileKey`s;
- a fill piece is a file, its units its chunks; a file has no floor (there
  is no request or answer turn), and its first chunk is placed with any
  other;
- pool pages and hits are per `ChunkKey`;
- the gallery holds a member's signatures under its `ChunkKey`, and a
  calibration's votes go to its `FileKey`;
- a superseded member is retired by its `FileKey`'s blob changing, which
  `code_read/process.rs:128-137` does by `PATH_KEY`/`CONTENT_KEY` today.

Members are **shared submitted sections**, not schema members: a schema
collection's members join the birth prefix, cannot be offloaded, and must
fit the KV reservation whole (`sequence_construction.rs:484-498`), which for
a corpus is impossible. They are sealed and placed as the per-conversation
journal's members are (`conversation/submitted_sections.rs`), shared across
conversations (`Sharing`, `shared_section_prefix`, `:455-459`).

The QSA's stratified selection ranks the prompt in every window, and the
prompt is the leading run of section and glue pieces (`slot_state.rs:274-283`,
`qwen4exp/indexer/cache.rs:89-97`). The prompt therefore ends at
`files_open`: members are selected by the index like any other piece, never
pinned (`apply_build.rs:526-532`), and counted where the trained window's
shares are checked (`apply_build.rs:533-557`).

- `scopes`, `guidance` and `journals` stop being groups of the
  `code_reading` layer, and `structure` of `repo_map`; all four become
  **kinds of the one collection**, each with its share of the room as now
  (`working_set.shares`: readings, guides, journals, folders). The `draws:
  from: scopes, paths: …` rules carry over as the kind's path filter.
- **A folder is its listing (decided 2026-11-26).** A `repo_map` folder unit
  stops being a conversation ending in a decoded summary
  (`projection.yaml:726-795`). It is its listing, rendered as a snippet in
  Q1's notation (`## The content from candle/zend/src/:` and the listing in
  the fence), prefilled and calibrated as a file is, with no summary decode.
  The seed's root listings are folders of this kind. The listing is a plain
  path list: one entry a line, repository-relative, a directory ending in
  `/`, a file followed by its size in bytes.
- The schema allows `selection: { kind: working_set, share: … }` on a
  collection; `yaml/parse_groups.rs:277` refuses it today and is lifted.
- The fill's room is computed as now (`slot_window − reserve − system −
  decode_pos`, `sink.rs:1013-1017`), with the `files` members counted out of
  it rather than out of the turn room. The system prompt's fixed part is
  measured without them.
- Reprojection re-runs the pool and the fill at every boundary exactly as it
  does now; the system block's membership changes with them.

## 5. Sealing a member

A member's K/V is sealed **behind the system prompt before the collection**
plus the **earlier chunks of its own file**. That prefix is the base
conversation's projected prompt, as a code reading's is today
(`code_read/process.rs:150-160`) — at most five tools and none of a
conversation's journal — and not `prefix_before_collection`
(`projection/owned_prefix.rs:31`), which holds every member of the
collections before it: the whole tool catalog and one conversation's
journal entries.

- chunk *k* of a file seals against that prompt, then `files_open`, then
  the file's heading piece, then chunks 1…*k*−1, and
  starts the recurrent state from the state after chunk *k*−1
  (`conversation/prefix_state.rs:194` `continue_state`, as
  `submitted_sections.rs:422` uses it). A chunk is thereby sealed in the
  very text Q1 showed: under its file's heading, as the continuation of its
  file. That recurrent state exists only to seal the chunk's K/V: it is the
  member's, never a slot's, and no conversation's recurrent state is run
  over a dynamic member (§8);
- its content address is `{prefix_hash over the system prefix and the file's
  earlier chunks, section_hash}`, members excluded from every other
  section's `prefix_hash` as tool members are (`sequence_sections.rs:246-251`);
- keys and index rows are stored un-rotated and rotated at load, so placing
  chunk *k* without chunk *k*−1, or in another order, re-seats it for free
  (`hit_driven_projection.md` §4).

Nothing after the collection is sealed: `system_close` cannot seal against
the whole catalog as `tools_close` does (`sequence_construction.rs:432-436`),
the catalog being the corpus. It is forwarded live with the collection's
last closing fence, as the close row (§6.2).

## 6. The seed chain and the close row: the recurrent state

The 36 DeltaNet layers of Flash-Next carry state the attention layers' K/V
does not: what has been read, and in what order. A dynamic member is
attended but never run through them (§8). The recurrent state a
conversation runs on must still be one a forward actually produced, over a
system prompt that ends where the dialogue begins. The seed chain provides
it, and the close row ties it to what each projection placed.

### 6.1 The seed chain is computed with the recurrence and snapshotted

The seed's files — per repository the root listing, README, ARCHITECTURE,
AGENTS, CLAUDE (`branch_ingest/prime.rs`) — are members of `files`, but they
are not sealed as dynamic members are. They are built as a **chain**, as the
priming chain is built today:

- each seed file is prefilled behind the system prompt with the collection
  empty, then `files_open`, then every seed file before it, in the chain's
  fixed order — its K/V sealed as members are (§5), and the **recurrent
  state run over it** and carried to the next;
- the recurrent state is **snapshotted at the end of each seed file**, as
  each chain link's is today (`ingest_snapshots.rs`, a link keeps its tail;
  `scheduler/recurrent.rs` `restore_recurrent_state`). A seed file that
  changes re-prefills the chain from the last unchanged link, as a link
  whose key holds is reused today (`code_read/link.rs`);
- the snapshot at the **seed's end** is the state of a system prompt that
  has read the seed, closing fence included.
- `system_close` leaves the birth prefix and the prompt checkpoint. It is a
  template with no `depends_on` today, so it sits in `fixed_prefix`
  (`sequence_construction.rs:449-451`) and the checkpoint has read
  `<|im_end|>` (`prompt_branch`, `:733-743`). The checkpoint ends at the
  last section before `files_open`, so the seed chain starts from a prompt
  still open, and `<|im_end|>` is forwarded only as the close row, for every
  layer, with or without a `files` collection.
- a changed seed file re-prefills the chain **from that file on**, the
  later links included, their snapshots taken again. That replaces today's
  rule: a later link is re-pointed, not re-read, and keeps K/V and a
  snapshot computed against the old predecessor
  (`code_read/link.rs:24-27,40-46`; `branch_ingest/prime.rs:31-34,279-281`).
  A recurrent state is only correct if the forward that made it read what
  precedes it.

The seed is the one part of the collection the recurrence has read, and it
sits last, nearest the question.

Two rules fix it (decided 2026-11-26):

- **One chain, behind the default branch.** The system prompt before the
  collection is a section tree a turn's dials select, with a checkpoint per
  branch (`install_branch_states`). The seed chain and its snapshots are
  computed once, behind the default branch, and read under every branch as
  sealed sections are. A turn on another branch reads a seed-end state that
  carries the default branch's dial text; the dials' sections themselves are
  read under the turn's branch.
- **Pinned, with no cap.** The close row's state assumes the whole seed is in
  the slot, so the seed is always placed whole and never gives way to the
  selection; the dynamic members get the room that remains. This replaces
  the working-set seeds, which give way to what the question ranks
  (`working_set/state.rs:85-89,218-239`), and the 64K
  `INHERITED_CHAIN_TOKEN_CAP` (`substrate/registry.rs:269`). The seed is
  bounded only by the slot: a chain that leaves less than the slot's
  reserve and the dialogue's minimum room refuses to build, with the seed's
  size and the slot's in the error.

### 6.2 The close row: forwarded once, at the conversation's opening

**Decided 2026-11-26:** the close row is forwarded only when a conversation
opens. At that point the dialogue is empty, so the close row is the first
thing the opening prefills, at the slot's tail. Its recurrent state starts
from the seed-end snapshot. Its attention reads the knowledge base the
opening placed, and the conversation's first user message is prefilled
straight after it, in the same prefill, through `drive_prefill_and_capture`
(`projection_assembler/prefill.rs:188-276`), the existing path that forwards
at the tail and captures K/V and index pages.

The close row is then **one sealed piece of the conversation**, its own and
no other's. At every later rebuild it is injected between the collection
and the dialogue like any sealed piece. A running conversation never
forwards it again:

- its recurrent state is its own, live, moved across the rebuild as now
  (`move_recurrent`, `reprojection.rs:403-434`), and never receives a close
  row a second time — the re-feed's failure cannot occur;
- the close row's K/V keeps the opening's placement and does not see later
  placements. What a later placement adds reaches the conversation through
  attention, by the dialogue's tokens forwarded after it;
- a rebuild needs no forward in the middle of the slot, no second recurrent
  store and no new step in the assembler. A changed member costs that
  member only, as a changed reading's piece costs only that piece today
  (`apply_build.rs:85-91,143-146`).

The close row rides the opening's prefill, so conversations opening
together forward their close rows in one batched prefill, never one launch
a conversation (hot-path invariant 5).

The request is still read twice (decided 2026-11-26). Its first reading
(`scheduler/first_reading.rs`) prefills the question against the opening's
projection with the seeds held out, and seals nothing. It forwards its own close row from the state at `files_open`
on the same discarded prefill. The second reading is the real opening: the
close row from the seed-end snapshot, then the question, sealed.

The close row is **`<|im_end|>\n` in every case**. Every file seals its own
closing fence (§3), so the close row does not depend on which file ends the
collection. That matters because a running conversation keeps its opening's
close row while the files before it change. An empty collection omits
`files_open` and ends on the same close row. The one byte Q1 did not have
is the newline between the last fence and `<|im_end|>`, which is ChatML's
own shape. The close row is checked at load to be at most 8 tokens.

The rest of this section is the analysis behind the decision, kept as the
record of the alternative.

### 6.2a Forwarding the close row at every projection (not chosen)

At every projection that changes the slot (a reprojection that keeps the
selection does not rebuild, and the close row it holds stands), after the
fill has chosen the dynamic members and the slot is rebuilt with them placed
**before the seed**, the collection's close
— the last seed file's closing fence and `system_close`, `\n```<|im_end|>\n`
— is **forwarded**, not injected:

- its **recurrent state starts from the seed-end snapshot** (§6.1), the state
  of a system prompt that has read the seed;
- its **attention reads everything placed**: the dynamic members and the
  seed, at their positions in this slot. Its K/V is computed in the context
  this projection built, and its rows reach the recurrent layers above each
  attention layer having attended to the dynamic content;
- it rides a wave as a row of **at most 8 tokens**, the recurrent layers'
  short span (`delta_net::timeline::TIMELINE_ROWS`), the most one wave
  writes for a row — the decode-wave row the re-feed used
  (`docs/archived/reprojection_refeed.md` §2 item 1), here at the system
  block's end rather than the turn's tail. The close is 4–5 tokens of
  Flash-Next's tokenizer; a close longer than the row refuses to load.

The forward writes a position in the middle of the slot, below the
dialogue's turns. The writer only writes at the tail
(`truncate_sequence_to_tokens` clamps to the sealed boundary below
`writer_start`, `candle-nn/src/kv_cache/chunked/sequence_ops/tail.rs:288-303`),
and the hybrid stacks refuse a forward into the middle of a slot outright
(`qwen4exp/wave/wave_sweep.rs:159-164`, `qwen35/forward/layer_sweep.rs:159-165`). So the rebuild
is done in two steps: the parent is rebuilt to the collection's end, the
close row is forwarded as its tail, and the dialogue's turns and the turn's
captured tail are then injected after it (`inject_sealed_at_tail`) and the
view carved over all of it, as a reprojection carves it now
(`docs/archived/reprojection_refeed.md` §2.1).

**What the close row's recurrent state is used for depends on the
conversation:**

- **A new conversation** opens on it. The state after the close row — the
  seed-end snapshot advanced by a close computed against the opening's
  placement — is the recurrent state its first user message is prefilled
  on. That replaces forking from the seed and `seed_recurrent_from_lineage`.
- **A conversation already running** keeps its own live recurrent state, as
  a reprojection keeps it now (`move_recurrent`, `reprojection.rs:403-434`):
  it is already bootstrapped and has read its own turns. The close row's
  K/V is still recomputed against the new placement, which is what the
  dialogue's later positions attend. The close row's recurrent output is
  discarded: it was computed on a scratch copy of the seed-end snapshot, and
  never touches the turn's own state.

This is what the re-feed got wrong and this does not. The re-feed wrote the
turn's last tokens into the live recurrent state a second time at every
boundary, and the reasoning looped (`docs/archived/reprojection_refeed.md`,
*Why it went*). The close row is forwarded on a scratch copy of the
seed-end snapshot, and a running conversation's state never receives it. A
new conversation receives it once, as the end of its system prompt.

### 6.2b As built: the seed and the close are part of the prompt

The schema's branch checkpoint already is what §6.1–§6.2 describe: the
recurrent state a new conversation opens on, computed by a forward over the
birth prefix — the system prompt's fixed sections in order, collection
members and `depends_on` sections left out (`sequence_construction.rs:449-451`,
`prompt_branch`) — one per branch of the dials' section trees, sealed and
persisted with the schema's sections (`install_branch_states`). So the
build puts the seed and the close *into the prompt*:

```
… tools_close
files_open          fixed section — in the birth prefix
files               collection, no schema members — the dynamic members,
                    placed by the fill, attended only
seed                section tree, option present | absent (the `seed` dial)
system_close        fixed section — the close
```

- The **seed** is one section: the seed files in Q1's notation, rendered at
  startup from the repositories (`branch_ingest::prime`'s links) and
  installed into the schema before any conversation is built, as the tool
  catalog is installed (`tools/catalog.rs`). It is sealed behind
  `files_open` and read by the checkpoint's recurrence: §6.1's chain, as
  one section.
- The **close** is `system_close`, sealed behind the seed and in the
  checkpoint: a new conversation's recurrent state is the checkpoint's, the
  state of a prompt that has read the seed and its close. A running
  conversation never receives it again; nothing is forwarded at a
  projection, and no conversation's live state is touched (§6.2's property).
- The **dynamic members** sit between `files_open` and the seed, outside
  the birth prefix as tool members are: placed at projection time, attended
  only, never in any recurrent state (§8).
- The **`seed` dial** selects the tree's branch, so seed-on and seed-off are
  two checkpoints, each exact — finer than §6.1's one chain behind the
  default branch.
- A prompt without a `files` collection (npcd, the reserved layers) is
  unchanged: its checkpoint already ends on `system_close`.

What this gives up from §6.2: the close's attention reads the seed but not
the dynamic members, because it is sealed in the prompt rather than
forwarded over each opening's placement. The close is two tokens; the
recurrence reads placed content through every later token's attention, the
dialogue's own, as the user put it — it is in the session's context, it
holds no neighbour.

The seed is refreshed when the schema is built — at startup, and whenever
the builders are rebuilt. A seed file changed while the daemon runs reaches
new conversations once the schema is rebuilt; a conversation already open
keeps the state it opened on and reads the new seed by attention.

### 6.3 What goes

- `forked_from = "seed"`, `set_forked_from_seed` and the dialogue's
  `seed_recurrent_from_lineage` (`stream_turn.rs:199-258`): a dialogue no
  longer forks from the seed. Its recurrent start is the close row's.
- The seed as a lineage of turns: `INHERITED_CHAIN_TOKEN_CAP`, the lineage
  base score in the fill (`sink.rs:1048-1063`), the lineage drop below
  `ProjectionLevel::Full`. Forks of a *dialogue* keep their lineage.
- The `seed` request dial gates the seed members. With the seed off, the
  close row starts from the state at `files_open`, the end of the empty
  collection.

## 7. Ingest: prefill and calibration, nothing decoded about the file

A file's ingest is:

1. **Cut** into chunks as `file_read` pages it now: whole parts, up to 200
   lines a page (`PAGE_LINES`, `zend-tools/src/sandboxes/page.rs:20`;
   `tools/file/read.rs:126`).
2. **Prefill** each chunk as a `files` member (§5) — no reading
   conversation, no `file=<path>` face, no `file_read` call, no tool
   response wrapper, no summary decode, no summary tree (`code_reading`
   layer's `summary:` block goes).
3. **Calibrate**, in the two steps of today (`zend/src/calibrate/run.rs`):
   - the **draft** writes the file's questions with the file in front of
     it — now its members in the collection, in Q1's layout, where it was
     the reading's turns (`calibrate/sections.rs`, one message, one part per
     prompt);
   - each question is then **asked again, alternating between two
     knowledge bases** (decided 2026-11-26). A file's questions take them in
     turn, by their order in the draft: the first, third, fifth… behind the
     **empty** knowledge base, the second, fourth, sixth… behind the
     **target**:
     - *empty* — `files_open` emitted with no members, the knowledge base's
       heading and nothing under it, on the seed's absent branch; the
       recurrent state that branch's checkpoint. The frame a query has before it has
       found the file: what provenance must match to pull the file in.
     - *target* — the collection holding **only the target file's members**,
       in Q1's layout, attended as a dialogue's dynamic members are; the
       recurrent state the empty frame's, since no production query's
       recurrence has read a dynamic file (§12 D11). The frame a query has
       once the file is in front of it.

     Either way the close row and the question follow, thinking on, and the
     question and thinking are sealed as the calibration
     (`Substrate::CALIBRATES_KEY`, now voting for the `FileKey`), then
     distilled provenance-only. Alternating rather than asking every
     question twice keeps a file's signature bytes where they are. Neither
     frame holds the seed or another file, so nothing bleeds in and a seed
     change recalibrates nothing. This replaces asking alone with nothing
     projected (`calibrate/run.rs:10-23`;
     `docs/provenance_scoped_signatures.md` §2.5, revised with this change):
     both knowledge bases are real frames, so the question's K/V and the
     thinking's queries are computed where a live query's are.

   The calibrations and the members' own signatures are what the gallery
   scans (`belief_groups.rs`). The draft and the calibrations are the only
   things decoded.

The seed's files are ingested as the chain of §6.1 instead — prefilled in
order with the recurrent state carried and snapshotted at each file's end —
and are calibrated as every other file is.

Faces (`substrate/faces.rs`, `zend/src/faces.rs`, the reserved `Face` layer)
are removed: there is no reading turn to face.

A gallery entry is keyed by member section rather than by reading turn;
`cases: file` groups a file's members as it grouped a reading's turns.

### 7.1 Calibration is load-bearing

Provenance ranks a file by the agreement of the live decode's queries with
the file's stored signatures (BDP, `sign(Q_PCA^T K)`; `docs/attention_provenance.md`).
The live queries come from an assistant turn reasoning in a dialogue. A
member's **content** signatures now come from system-block content, where
they once came from tool-response turns, and the two can point differently.
Content-only ranking may weaken.

A **calibration** is a question about the file asked as a live turn opens,
behind one of two knowledge bases (§7 step 3): a user turn, its thinking
after it, behind a system prompt that ends as every dialogue's does. Its
signatures sit where a dialogue's queries do — the same role, the same
frame — and they carry the ranking where the content's fall short.

The two frames cover the two queries provenance answers. Behind the
**empty** knowledge base the thinking reasons toward a file it does not
have, as a query does before the file is placed: the discovery that pulls
it in. Behind the **target** it reasons with the file in front of it, as a
query does once it is placed, and leans toward the file's own content. A
file's questions alternate between them, so each file carries both kinds
in the bytes one kind had. The ranking split below measures what each kind
contributes. The
numbers already lean on them: the scoring lab's formula took MRR 0.235 →
0.663 largely through how calibration votes are weighed
(`docs/results/provenance_scoring_lab_2026-11-16.md`), and the needle
ranking (49% first of 7,943 files) was measured with question-and-thinking
probes, the shape a calibration has (`docs/unbounded_agents.md` §9.12).

Three consequences:

- **Coverage is correctness.** A file without its calibrations is all but
  invisible to provenance. The memory is settled only when every ingested
  file is both prefilled and calibrated; a calibration that fails is
  retried, and a file still without one is reported by the settled check
  (`zend::settled`, `waiting_on`) rather than counted as read.
- **Calibration's known failures are failures of the memory.** Calibrations
  flooding the votes and outranking the content they describe, and drafts
  written blind to their file, are held to the per-file competition already
  tuned (`calibration_weight`, `vote_top_k`, `kind_z`,
  `projection.yaml` `scopes` policy) and checked on the first memory.
- **The split is measured.** On the depth-1 memory (§9 step 5) the test
  questions are ranked by the scoring lab (`provenance_gates lab`) before
  their answers are read: content signatures alone; empty-frame
  calibrations alone; target-frame calibrations alone; and all of them
  together. Content-only collapsing while calibration carries the rank
  confirms this section; everything collapsing stops the rollout there. The
  split between the two calibration kinds says whether the alternation
  should lean one way.

### 7.2 What the system prompt tells the model

Two sections describe the old arrangement and change with it
(`zend/src/prompts/projection.yaml`):

- `history_stance` says most earlier turns are the model's reading and that
  the conversation "descends from the workspace's listings and documents".
  After this change the turns are the conversation and nothing else, and
  the files are in the knowledge base below. It says that: earlier turns are
  the model's own work with the user, built on rather than repeated; the
  files it has read are under *Knowledge Base* at the end of this prompt;
  the user's latest message is what to do now.
- `grounding` and `grounding_no_tools` say "answer from what is actually in
  this conversation, the reading done before this session began included".
  They say instead: from this conversation and the files in the knowledge
  base.

Their wording is set with the user and pinned by the schema's tests, as the
persona's is.

## 8. The recurrent layers — what remains a risk

The seed has been read by the recurrence (§6.1). The dynamic members are
not run through any conversation's recurrence: each is sealed with its own
file's state (§5) and only attended in the slot. A new conversation's
recurrent state learns of the dynamic files its opening placed through the
close row's few positions (§6.2), and every conversation's through the
tokens forwarded after a placement, whose attention reads the members — the
user's message, the reply, tool results. A running conversation's state
never receives a close row.

The probes do not settle whether that is enough: on the passthrough route
every file was prefilled, so the recurrence had run over all of them. The
diag result (§2) says the recurrence was not what failed the turn
presentation; it does not say attention alone suffices for the knowledge
presentation. It is measured on the first memory the new build ingests (§9,
step 5): sealed dynamic members against the same members re-prefilled. If
attention alone falls short, the rollout stops at step 5 and the fallback is
decided with the user from the measurement. Recurrent states do not compose
(`conversation/prefix_state.rs:10-11`), so no captured state is folded in,
and the decode diag's `capture`/`merge` stays a diagnostic.

## 9. Rollout

The change is built whole and tried on BlackWorld against a fresh, small
memory, with WhiteWorld's zend held out so nothing of the old memory
replicates back into it. The wipes below are approved by the user for this
rollout (2026-11-26).

1. **Take WhiteWorld's zend out.** Stop the zend tenant on WhiteWorld and
   keep npcd running there. Wipe WhiteWorld's zend substrate. A member with
   no zend substrate holds nothing to offer, so the old readings cannot
   replicate back into the new memory, and BlackWorld serves zend alone.
2. **Implement the whole design** (§3–§7) in one change, with its tests
   written alongside:
   - the `files` collection at the end of the system prompt, in Q1's
     notation. The pieces are sealed separately, so the test asserts
     **token ids**, not only bytes: the assembled pieces' tokens against the
     tokenization of `probe::qwen_knowledge`'s text for whole files and of
     the `...`-joined text for a file with a gap, a seam that tokenizes
     differently cut where it tokenizes alike;
   - the pool and the fill feeding the collection, and the `working_set`
     selection allowed on a collection;
   - ingest as chunk prefill plus calibration (§7);
   - the seed chain built with the recurrence and snapshotted (§6.1), placed
     last in the collection;
   - the close row forwarded once at the opening from the seed-end snapshot,
     the conversation's first message prefilled straight after it, and the
     close row a sealed piece of the conversation from then on (§6.2);
   - `system_close` out of the birth prefix and the prompt checkpoint
     (§6.1);
   - faces, reading summaries, the seed lineage and the code_reading groups'
     turn placement removed.
3. **Wipe BlackWorld's zend substrate** and start the new build on it with
   the ingest's **max depth set to 1** (`max_depth: 1`, which admits only each
   repository's own root files, `branch_ingest/filter.rs:90`), so the memory
   holds the seed documents and the other root files: small enough to
   ingest, calibrate and normalize in one sitting.
4. **Run the ingest to settled.** Every depth-1 file prefilled, its
   calibration questions asked, and normalization run, until the member
   reports its memory settled (`/v1/status`, `tenants.zend.settled`) — no
   file uncalibrated (§7.1).
5. **Test** with questions answered by the depth-1 documents and by the
   seed documents, on the production chat path:
   - the ranking split of §7.1: content alone, calibration alone, both;
   - the production placement, which Q1 did not test: Q1's system prompt
     was the persona alone and its user message told the model to answer
     from the files above (`probe.rs:96-99`). Here the files sit after the
     full system prompt and tool catalog, and the question carries no such
     instruction;
   - every in-view item named, as Q1 named 16 of 16 — the measure of
     whether the presentation is fixed in production and not only on the
     passthrough route;
   - the decode diag's forced inventory lists the placed files;
   - *sealed* members against *fresh* (§8): attention alone reading the
     members as well as a re-prefill does. If it does not, §8's fold is
     built before anything else;
   - the close row: a new conversation's recurrent state at its first user
     token equals the seed-end snapshot advanced by the close row, and a
     running conversation's state is the same before and after a
     reprojection (both asserted by GPU tests at one thread); no reasoning
     loop over a long thinking block, the failure that withdrew the re-feed.

   This is the test of whether the fix holds. The battery's memory arms
   follow once it does.

## 10. Documents this change revises

Design documents are authoritative, so the implementing change updates
these in the same commit:

| Document | What changes |
|---|---|
| `docs/zend_working_set.md` §4.1, §4.9 | members keyed by `FileKey`; faces removed |
| `docs/hit_driven_projection.md` §7, §8 | the fill's pieces are files of chunks placed in the system block; no reading floor |
| `docs/zend_workspace_execution.md` §6, `docs/zend_branch_ingest.md` §6 | the seed is a chain of collection members, re-prefilled from a changed file on |
| `docs/deltanet_state_persistence.md` §4.7d | `system_close` forwarded as the close row, out of the checkpoint |
| `docs/provenance_scoped_signatures.md` §2.5 | calibration asked behind its target file |
| `docs/guidance_layer.md` | guides and journal notes are kinds of the collection |

## 11. Open questions

- **Partial files.** The probes placed whole files (cut only at the 24K
  budget). Whether a file shown as chunks with `...` gaps reads as well as a
  whole file, and whether the fill should place a chunk's predecessor with
  it so a definition split at a boundary is not shown in halves, is measured
  in step 5.
- **Calibration context** — decided 2026-11-26: a draft sees the target
  file's members only, and its questions alternate between the empty and
  the target knowledge base (§7 step 3). The seed files included: a seed
  file is calibrated behind itself alone, not behind the chain, so
  `OWN_FILE` (`calibrate/sections.rs`), which exists because a link's draft
  saw the chain, goes. The draft's reading check
  (`calibrate/draft_sight.rs`, `reading_held`) and its prompts ("It is above,
  already read", `projection.yaml:850-874`) are rewritten for a file in the
  knowledge base.
- **Journal notes and guides.** They move with readings; whether the
  per-conversation `journal` collection should render in the same format is
  left as it is.

## 12. Implementation decisions

Made while planning the build (2026-11-26), each consistent with the
sections above. The build may adjust a decision its tests disprove, and
records the change here.

| # | Decision |
|---|---|
| D1 | `FileKey(u64)` is the first 8 bytes of `content_hash` over the content key `v{N}:{path}@{blob}` (a folder: `@{listing hash}`); the path and blob live on the file's record. `ChunkKey { file: FileKey, lines: LineSpan { first, last } }`, 1-based and inclusive, the spans `file_read` pages report. |
| D2 | The pieces (`candle-conversation/src/files/render.rs`): `files_open` `"\n# Knowledge Base\n\n"`; heading `"## The content from {path}:\n\n```\n"`; a chunk's text, every line ending its `\n`; gap `"...\n\n"`; `closing_join` `"```\n\n"` when another file follows, `closing_last` `"```\n"` for the last. Nothing that follows a chunk opens with a newline: Qwen's pre-tokenizer joins a line's punctuation and the newlines after it into one pre-token, so a piece opening on `\n` would merge with the chunk's last `\n` across the seam. The test `tests/knowledge_pieces_real_vocab.rs` holds every cut on all eight cached Qwen vocabularies, Flash-Next's included. Two bytes differ from Q1's text as the probe wrote it: a file whose text ends in a newline has no blank line before its closing fence, and a gap has one newline before its dots where Qwen-Agent's separator has two. The probe now renders through the same code. |
| D3 | A chunk is whole lines, up to `PAGE_LINES`, its last line given a `\n` when the file's has none. Every cut is checked against the tokenizer — the tokens of the joined text equal the two parts' tokens joined — and a cut whose seam tokenizes differently moves forward a line at a time until it does not, the chunk running to the end of the file when no cut holds (`files::cut_lines`). |
| D4 | A member is sealed behind the default branch's birth prefix up to and including `files_open` (`Sequence::birth_prefix`'s sections before the `files` collection: no collection member, no `depends_on` section, so no tool and no journal; never the seed or the close, which come after). Its K/V prefix is the prefix its recurrent start was computed over. |
| D5 | A file is a chain: heading → chunk 1 … chunk N, sealed behind D4's prefix, each continuing the one before (its recurrent state run on from the predecessor's end, so a chunk's K/V is computed as the continuation of its file). Gap and the two closing pieces are shared singletons sealed behind D4's prefix. The seed is not members: it is one section of the prompt (§6.2b). Its files are also ordinary files, so each is calibrated behind itself. |
| D6 | `files_open` is a fixed section of the dialogue's prompt and always emits: an empty knowledge base is the heading alone. Every conversation opens on the branch checkpoint (§6.2b). A prompt without a `files` collection is unchanged. |
| D7 | (Superseded by §6.2b.) No close row is forwarded at an opening: the close is `system_close`, sealed behind the seed and read by the checkpoint. |
| D8 | (Superseded by §6.2b.) The fill needs no floor for the close; a file has no floor (§4). |
| D9 | The seed is placed exactly when the turn's `seed` dial selects the tree's present branch; the conversation's recurrent state is that branch's checkpoint, so the two always agree. |
| D10 | A question's first reading is read against its projection as every reading is, the seed placed when its dial says so: the seed is background every conversation keeps, part of the prompt, and is no longer held out of the first reading. |
| D11 | A calibration conversation is a conversation of the calibration layer on the same prompt, on the seed's absent branch. Its empty frame is `files_open` with nothing placed; its target frame places only the target file's members (`FilesPin::Only`), attended as a dialogue's dynamic members are. Both open on the absent branch's checkpoint, as a production query opens on its own: the target frame's recurrence has not read the file, as no production query's has. |
| D12 | A calibration records its frame (`calibration_frame`: `empty` or `target`); `CALIBRATES_KEY` holds the `FileKey`. The scoring lab takes `--calibrations none|empty|target|all` for §7.1's split. |
| D13 | `calibrate:` moves onto the `files` collection, its profiles unchanged and its guidance prompts per kind; `OWN_FILE` goes. |
| D14 | A folder's listing is its immediate children in git tree order, each repository-relative, a directory ending in `/`, a file followed by a space and its size in bytes; its heading's source is the folder's path (`candle/zend/src/`). It is chunked as a file is. |
| D15 | Member section ids are per process, from the owned partition under one owner for the collection, never persisted: records name content addresses and streams. Members register at boot from the file records, and on a replica's arrival, in one batched restore. |
| D16 | The seed refuses to install when the system prompt with it leaves the dialogue less than its share of the room: `slot_window − reserve − system prompt < dialogue share × (slot_window − reserve)`, the error naming the seed's tokens and the slot's. |
| D17 | Analysis layers of `unit: file` fork from a file's reading, which no longer exists; no shipped schema declares one. `UnitKind::File` is removed; `unit: day` stays. |
| D18 | A sweep releases member streams no live file record and not the seed reference, once a superseded record's tombstone activates. |

The build is staged on the branch `provenance-system-collection`, each
stage compiling with its tests passing, and merged to `main` whole:
1 the notation · 2 file records and the member seal · 3 the close row ·
4 the collection in the projection (schema, scan, working set and fill) ·
5 ingest as prefill · 6 the seed chain · 7 calibration · 8 prompt text,
removals and documents.
