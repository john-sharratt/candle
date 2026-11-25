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
momentum, reduced to what fits by the QSA index. A reading stops being a
conversation: ingest prefills each file's chunks as collection members and
asks the file's calibration questions; nothing is decoded about a file but
those answers. The dialogue's turns are the dialogue's own and nothing else.

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
files members                   # pinned seed first, then the selection
<|im_end|>
<|im_start|>user …              # the dialogue's own turns only
```

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
  its **closing piece** (`\n```\n\n`, the last file's without the trailing
  blank line). The heading, gap and closing pieces are small sections of the
  collection, sealed as members are (§5).
- Files emit in **fill order, strongest first**, the order the probes used,
  not in declaration order as tools do. The pinned seed files (§6) come
  first.

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

`system_close` cannot seal against the whole catalog, as `tools_close` does
(`sequence_construction.rs:432-436`): the catalog is the corpus. It seals
against the prefix without members. It is a three-token marker; the cost is
nil.

## 6. The seed becomes pinned members

The priming chain stops being a fork lineage. Its files — per repository the
root listing, README, ARCHITECTURE, AGENTS, CLAUDE (`branch_ingest/prime.rs`)
— are ordinary members of `files`, **pinned** (the collection's `mandatory`
members, as a mandatory tool is) and placed first, under a pinned share of
the room.

- `forked_from = "seed"`, `set_forked_from_seed` and the dialogue's
  `seed_recurrent_from_lineage` are removed (`stream_turn.rs:199-258`). A
  dialogue opens on the system prompt alone; its recurrent state starts from
  the prompt checkpoint.
- The inherited-chain handling for the seed goes with it:
  `INHERITED_CHAIN_TOKEN_CAP`, the lineage base score in the fill
  (`sink.rs:1048-1063`), the lineage drop below `ProjectionLevel::Full`.
  Forks of a *dialogue* keep their lineage; only the seed stops being one.
- The `seed` request dial gates the pinned seed members, as it gates the
  working-set seed files today (`turn_dials.rs:130-146`).

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

Faces (`substrate/faces.rs`, `zend/src/faces.rs`, the reserved `Face` layer)
are removed: there is no reading turn to face.

A gallery entry is keyed by member section rather than by reading turn;
`cases: file` groups a file's members as it grouped a reading's turns.

## 8. The recurrent layers — the risk this design carries

Collection members are never folded into a slot's recurrent state: the
prompt checkpoint skips them (`sequence_construction.rs:733-736`) and a
reprojection moves the turn's state on unchanged (`reprojection.rs:403-434`).
The 36 DeltaNet layers will therefore not have run over a placed file; only
the attention layers read it.

The probes do not settle this: on the passthrough route every file was
prefilled, so the recurrence *had* run over them. The diag result (§2) says
the recurrence was not what failed the turn presentation; it does not say
attention alone suffices for the knowledge presentation. It is measured on
the first memory the new build ingests (§9, step 5). If attention alone falls
short, the members' recurrent rows are captured at seal time (the
`capture`/`merge` machinery of the decode diag) and folded into the slot's
state where the collection sits.

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
   - the seed as pinned members (§6);
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
     built before anything else.

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
- **Journal notes and guides.** They move with readings; whether the
  per-conversation `journal` collection should render in the same format is
  left as it is.
