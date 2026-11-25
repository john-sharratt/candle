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
last members; the dynamic members are placed before it, and at every
projection the collection's close is forwarded live from the seed's
snapshot, so the system prompt's end is a real forward over what was placed
and a new conversation starts its recurrent state there. A reading stops
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
close row  …```<|im_end|>\n     # forwarded live at every projection (§6.2)
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
  its **closing piece** (`\n```\n\n`). The heading, gap and closing pieces
  are small sections of the collection, sealed as members are (§5). The one
  exception is the collection's last file, the seed's last: its closing
  fence is not sealed but forwarded in the close row (§6.2), so the bytes end
  `\n```<|im_end|>\n` exactly as Q1's did.
- The **dynamic members** come first, in **fill order, strongest first** —
  the order the probes used, not declaration order as tools have — and the
  **seed** last, nearest the question, in its chain's fixed order (§6).

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

- `scopes`, `guidance` and `journals` stop being groups of the
  `code_reading` layer and become **kinds of the one collection**, each with
  its share of the room as now (`working_set.shares`: readings, guides,
  journals). The `draws: from: scopes, paths: …` rules carry over as the
  kind's path filter.
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
— the same prefix a code reading is sealed behind today
(`code_read/process.rs:153-160`), so nothing about a member's portability
changes — plus the **earlier chunks of its own file**:

- chunk *k* of a file seals against `prefix_before_collection`, then
  `files_open`, then the file's heading piece, then chunks 1…*k*−1, and
  starts the recurrent state from the state after chunk *k*−1
  (`submitted_sections.rs` `continue_state`). A chunk is thereby sealed in
  the very text Q1 showed: under its file's heading, as the continuation of
  its file;
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
  has read the seed. The last seed file's closing fence is held out of the
  chain (§3); it is forwarded in the close row.

The seed is the one part of the collection the recurrence has read, and it
sits last, nearest the question.

### 6.2 The close row: forwarded live at every projection

At every projection, after the fill has chosen the dynamic members and the
slot is rebuilt with them placed **before the seed**, the collection's close
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
dialogue's turns, and the writer only writes at the tail
(`truncate_sequence_to_tokens` refuses below `writer_start`). So the rebuild
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

1. **Cut** into chunks as now (200 lines, `MAX_READ_LINES`).
2. **Prefill** each chunk as a `files` member (§5) — no reading
   conversation, no `file=<path>` face, no `file_read` call, no tool
   response wrapper, no summary decode, no summary tree (`code_reading`
   layer's `summary:` block goes).
3. **Calibrate.** The file's calibration questions are asked against a
   context of its own members in the knowledge format, and their thinking
   and answers are sealed as today; they and the members' own signatures are
   what the gallery scans (`belief_groups.rs`). The calibration conversation
   is the only thing decoded.

The seed's files are ingested as the chain of §6.1 instead — prefilled in
order with the recurrent state carried and snapshotted at each file's end —
and are calibrated as every other file is.

Faces (`substrate/faces.rs`, `zend/src/faces.rs`, the reserved `Face` layer)
are removed: there is no reading turn to face.

A gallery entry is keyed by member section rather than by reading turn;
`cases: file` groups a file's members as it grouped a reading's turns.

## 8. The recurrent layers — what remains a risk

The seed has been read by the recurrence (§6.1), and the close row reaches
the recurrent layers having attended to the dynamic members (§6.2). The
dynamic members themselves are not run through the recurrence: they are
sealed with their own file's state (§5) and only attended in the slot. The
recurrent layers therefore learn of a dynamic file only through the close
row's few positions and the tokens forwarded after it.

The probes do not settle whether that is enough: on the passthrough route
every file was prefilled, so the recurrence had run over all of them. The
diag result (§2) says the recurrence was not what failed the turn
presentation; it does not say attention alone suffices for the knowledge
presentation. It is measured on the first memory the new build ingests (§9,
step 5): sealed dynamic members against the same members re-prefilled. If
attention alone falls short, the members' recurrent rows are captured at
seal time (the `capture`/`merge` machinery of the decode diag) and folded in
where the collection sits.

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
     notation, its pieces asserted byte for byte against
     `probe::qwen_knowledge` for whole files and against the
     `...`-joined text for a file with a gap;
   - the pool and the fill feeding the collection, and the `working_set`
     selection allowed on a collection;
   - ingest as chunk prefill plus calibration (§7);
   - the seed chain built with the recurrence and snapshotted (§6.1), placed
     last in the collection;
   - the close row forwarded at every projection from the seed-end
     snapshot, the two-step rebuild around it, a new conversation opening
     on its recurrent state and a running one keeping its own (§6.2);
   - faces, reading summaries, the seed lineage and the code_reading groups'
     turn placement removed.
3. **Wipe BlackWorld's zend substrate** and start the new build on it with
   the ingest's **max depth set to 1**, so the memory holds the workspace
   roots, their seed documents and the files one level down: small enough
   to ingest, calibrate and normalize in one sitting.
4. **Run the ingest to settled.** Every depth-1 file prefilled, its
   calibration questions asked, and normalization run, until the member
   reports its memory settled (`/v1/status`, `tenants.zend.settled`).
5. **Test** with questions answered by the depth-1 documents and by the
   seed documents, on the production chat path:
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

## 10. Open questions

- **Partial files.** The probes placed whole files (cut only at the 24K
  budget). Whether a file shown as chunks with `...` gaps reads as well as a
  whole file, and whether the fill should place a chunk's predecessor with
  it so a definition split at a boundary is not shown in halves, is measured
  in step 5.
- **Calibration context.** Whether a calibration question sees only its own
  file's members or a selection like a dialogue's.
- **The seed chain under the dials.** The system prompt before the
  collection is a section tree whose branch a turn's dials select, and a
  prompt checkpoint exists per branch (`install_branch_states`). Whether the
  seed chain and its snapshots are computed per branch, or once behind the
  default branch and read under every branch as sealed sections are, is
  decided in the implementation and recorded here.
- **Journal notes and guides.** They move with readings; whether the
  per-conversation `journal` collection should render in the same format is
  left as it is.
