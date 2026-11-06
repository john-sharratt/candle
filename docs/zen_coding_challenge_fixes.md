# Zen coding challenges: what slowed them, and the next fixes

**Status:** reviewed with the user (2026-11-06), and built. Each cause
gives its evidence, the decision, and where the change lands. All are
decided, except whether the stacked-layer machinery stays (Cleanup).

**Built** (2026-11-06): every cause. Cause 4's read format shipped first, in
build 4471 (`5dddd4896`). Where the build departs from the decision, the
departure is recorded here:

- **Cause 1** records each round's queue wait, prefill tokens and time, and
  generated tokens and decode time (`zend/src/round_timing.rs`, metadata
  `round.<turn>`). `timeline_read`'s `calls` view shows them beside the
  round's calls and totals them under its header. Reasoning and call tokens
  are recorded together, as the engine counts them.
- **Cause 3** is held on the answer, not on a journal field. A brief's
  numbered or bulleted lines are its steps (`zend/src/mcp/steps.rs`). An
  answer to two or more steps ends with a `Steps:` section, and one that
  leaves a step out is sent back once, in the citation check's send-back. The
  coder and reviewer role documents ask for the section. That a `done` step
  cites a call is asked, not verified.
- **Cause 4**'s line ranges are zend's `file_edit` (`part`, `from`, `to`,
  `through`). npcd's bench act keeps the whole-unit form under its turn
  grammar's path budget (`docs/npcd_anchored_edits.md` §6).
- **Cause 5** names each file's chain by the SHA-256 of its JSON, not by its
  record's `oseq`. `run_command`'s `AgentExecutor` holds a store snapshot,
  not records, so a content digest is the one identity both senders share,
  and the agent can check it. The agent keeps the chains it was sent in
  memory, up to 256 MiB, least recently used out first
  (`agent/src/chains.rs`). A `409 missing` resend carries exactly the chains
  named, with no size budget beyond the store's own 10 MiB cap. An agent
  restart loses the cache, and the next run asks again.

**Source:** six worker tasks run on 2026-11-06 after
`docs/zen_coding_session_fixes.md` was built:

- `single-segment-handles`
- `npcd-reads-with-catalog`
- `verify-handles`
- `edit-shows-changes`
- `sandbox-priority-test`
- `map-npcd-parse-callers`

The evidence is their call listings (`timeline_read`, view `calls`), their
journal entries, and BlackWorld's `cluster_logs` for the same window.

## What the tasks did

| Task | Wall time | In tools | In the model | Calls | Failed | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| single-segment-handles | 39m | 15m28s | 23m33s | 71 | 2 | merged (`cfc7339ce`) |
| npcd-reads-with-catalog | 2h54m | 19m | 2h35m | 210 | 4 | merged after repair (`11de3cab0`, `dacb2b769`) |
| verify-handles | 5m22s | — | — | `git_grep` only | 0 | reported "verified" without running anything |
| edit-shows-changes | 55m45s | 3m24s | 52m21s | 89 | 9 | merged (`3dc39ea3c`); skipped the reviewer it was asked for |
| sandbox-priority-test | — | — | — | — | 1 | merged via `zen/priority-merge` |
| map-npcd-parse-callers | — | — | — | — | 1 | answered |

Three things stand out.

1. **The model's time is nearly all of the time.** It took 89% of the
   npcd task and 94% of edit-shows-changes.
2. **Journal notes are the most frequent call.** They were 63 of the npcd
   task's 210 calls and 14 of single-segment-handles' 71.
3. **Workers end without doing what they were asked.** verify-handles never
   tested anything. edit-shows-changes skipped the reviewer and the wider
   tests. Both wrote a closing entry that read as complete.

## Summary

| # | Cause | Decision | State |
| --- | --- | --- | --- |
| 1 | Model time is 89–94% of a task, and we cannot yet say where it goes | Record each round's queue wait, prefill and decode, and decide from that | agreed |
| 2 | Forced `journal_note` is 20–30% of all calls | No change: the notes are the journal, and the record future training learns from | decided: keep |
| 3 | A worker ends without the steps it was asked for | The closing entry accounts for each step asked for | agreed, to try |
| 4 | A whole-part replace drops more than was meant | Each part a fenced block with a `+N` gutter; `file_edit` by part and relative range; `file_read` by `part`, `lines` or `list`; fix the carver's doc attachment | built |
| 5 | A build ran without the worker's uncommitted edits, and passed | A job names the overlay it expects; the agent answers with what it lacks; the caller resubmits it from the substrate | decided |
| 6 | `branch: main` reads a local cache of origin | A branch revision reads origin's record, never the local cache | decided: fix |
| 7 | A nudge reads as the user speaking | A nudge arrives with its sender's name | agreed |
| 8 | Some refusals do not say what to do next | Empty `file_grep` pattern, `file_edit` `not_found`, and a stale anchor each answer with the way on | agreed |
| 9 | `run_command` is not replayed after a restart | Replay the read-only programs (`cargo test`/`check`/`clippy`, `fmt --check`) | agreed |

Order of work: 1 first, then 3, 4 and 5, which cost the most correctness. 6 to
9 are small and independent.

## Cause 1 — the model's time is not yet accounted for

**Evidence.**

- The npcd task ran 128 rounds in 2h35m of model time: 73 s a round.
- BlackWorld's log over the same window shows, per round:
  - prefill of 0.7–5.6 s;
  - 16–68 decoded tokens per second;
  - many turns of 57–889 tokens.
- That decode rate and turn length come to well under 73 s, so the rest is
  spent somewhere the call listing does not show. It could be queueing behind
  other sessions, prefill after reprojection, or long reasoning.
- The reprojection log shows each submit doing two reprojections
  (`whence="submit"` then `whence="reproject"`), at 60–90 ms each. That is small
  against 73 s, so it is not the answer, but it should be explained as well.

**Decision.** Measure first: no fix in this cause is chosen before the numbers
exist.

- Every round of a task records:
  - its time queued for the GPU;
  - its prefill tokens and milliseconds;
  - its reasoning tokens and its call tokens;
  - its decode milliseconds.
- The `calls` view shows this beside each round's calls, and the task's
  summary row totals it.
- Then the run is decided against those totals:
  - if queueing dominates, the scheduler's admission for task sessions;
  - if prefill dominates, the reprojection's tail;
  - if reasoning dominates, the prompt and Cause 2.

**Where.** The round boundary in `zend/src/session.rs`, which already sees the
turn's `prefill_ms` and token counts, and the `calls` view in the timeline
tools.

## Cause 2 — the forced note fires every round

**Evidence.**

- `journal_note` was 63 of the npcd task's 210 calls (30%), and 14 of
  single-segment-handles' 71 (20%).
- Each note is a decoded call, and a forced call after it
  (`docs/zen_coding_session_fixes.md` Cause 4).
- `NOTE_AFTER_REASONING` (`zend/src/session.rs:623`, 200 tokens) is measured
  per round. So a worker that reasons 200 tokens before each edit writes a note
  before each edit.

**Decision: no change.** The notes are what the journal is built from, and
the record that future training and learning draw on. A note before each edit
is the intended density, not overhead. Their cost is counted in Cause 1's
measurement like any other round's, and is not cut separately.

## Cause 3 — a worker ends without the steps it was asked for

**Evidence.**

- **verify-handles** was asked to verify a change. It ran `git_grep` and wrote
  "no CUDA build in this session". It ran no test. The reviewer role now names
  `run_command` (`cfc7339ce`), which fixes this case but not the class.
- **edit-shows-changes** was asked to make the change, run its tests and have a
  reviewer check it. It ran lib tests only, started no reviewer, and its closing
  entry read as complete.
- In both, the starter (Claude) only found out by reading the call listing.

**Decision.** The task's closing entry is held to account for each step it was
asked for.

- When a task starts, its brief's steps are taken as a list:
  - the numbered or bulleted items of `repo_task_start`'s `task`;
  - or, if it has none, the whole brief as one step.
- The closing `journal_note` is held to a `steps` field: one line per step,
  `done`, `not done` or `could not`, each with its evidence (a call, a commit, a
  job handle) or its reason.
- A step marked `done` must cite a call this task made. A citation of nothing is
  refused.
- `repo_task_start`'s answer and the task's result show the list, so the
  starter sees `not done` without reading turns.

**Where.**

- `zend/src/mcp/workers.rs`: the brief's steps, kept with the worker.
- The closing note's held fields: `journal/style.rs`, and the stencil for
  `journal_note`.
- `docs/claude_zend_gateway.md` §6.

## Cause 4 — a whole-part replace drops more than was meant

**Evidence: why it replaces more than it intended.** The npcd task's call
listing (`T:session-fixes-trials/npcd-reads-with-catalog`, view `calls`) shows
the same mistake five times. Each time the worker meant to change a piece of a
part, and replaced the whole part with only that piece.

| Turn | Meant to change | The part also held | Lost |
| --- | --- | --- | --- |
| #29 | the `stencil` import | three other `use` lines | `Serialize`, `Map`, `Value`, `escape_control_in_strings` (found at #107, after a failed build) |
| #41 | one call inside `parse` | the rest of the part | the part became one line; redone with `old` at #42 |
| #53 | `read_answer`'s doc comment | `read_answer` itself | the function body (noticed at #56, restored from memory at #57) |
| #65 | `compose::read`'s doc comment | `compose::read` itself | the function body (#66) |
| #30–#32 | `function_blocks_to_lines` | its tail, which it rewrote | the tail duplicated, then `Some(out)` became `out` (#115) |

The worker named the cause itself at #68: *"a part may hold doc+fn; use `old`
for in-fn changes, never whole-part replace unless rewriting the whole thing."*

**What the research says** (Aider's edit-format benchmarks and leaderboard,
JetBrains' Diff-XYZ, the hashline write-up and its independent replication,
Cursor's, Morph's and Serena's reports; sources at the end of this cause):

- **Exact-substring replace is the best-measured format for Qwen-family
  models.** It is the only one with useful accuracy below 32B (Diff-XYZ). Qwen3
  and DeepSeek make 83–98% well-formed edits with it on Aider. Our `old` is that
  format, scoped to one part.
- **Locating the target is not where models fail.** In the replication's 134
  replace edits the fuzzy fallback never fired. The failures were invented old
  text and line numbers pasted into old text. Anchors (hashline) beat patch
  formats, but are within ±11 points of replace.
- **Every whole-unit format guards its extent; ours does not.** Apply models
  require explicit `// ... existing code ...` markers. Symbol tools re-send the
  full body by contract. A whole-file or whole-function rewrite invites elision,
  which was GPT-4 Turbo's top failure. Our `replace` with no `old` silently takes
  a fragment for the whole part.
- **The format has to look like the training data.** Models count line numbers
  badly, and copy numbers shown beside lines into their edits. A custom anchor
  shape is out of distribution unless it reads unambiguously.

**What the read blocks show.** Reading the pages the worker read (turn #2,
`act.rs` and `element_close.rs`) and the carver behind them (`doc-units`), the
fault is structural, in three places.

1. **The Rust carver does not attach doc comments.** The design says *"an
   item's leading doc comments and attributes are part of it"*
   (`docs/npcd_anchored_edits.md` §3). On the page they are not:

   ```
   Jt3n| /// One act a character took.
   1Uax| struct Act
   #[derive(Clone, Debug, PartialEq, Serialize)]
   pub struct Act {
   ```

   - The attribute attached and the doc did not.
   - `start_with_leading_comments`
     (`doc-units/src/carve/parsers/tree_sitter_util.rs:181`) attaches a sibling
     only when it ends on the line directly above the item.
   - tree-sitter-rust 0.24's doc comment (`///`) appears to end one line later,
     because it holds its newline. A plain `//` does not, which is why
     `struct_scope_starts_at_struct_keyword` passes.
   - `extracts_doc_decorated_fn` asserts only the label, not the start line, so
     the gap was never seen.
   - An orphaned doc then joins the unclaimed lines above it into one
     paragraph: `vaeb| impl Act {` followed by the doc of the impl's first
     method.
   - So #53 and #65 were the worker trusting the stated rule. It took `fn
     read_answer`'s part to hold its doc, and replaced the function with only a
     new doc. The orphaned old doc (`SFKu`) stayed, beside the new one.
2. **A paragraph's anchor sits on its first line.** A named item reads as a
   header line, then its lines. A paragraph (an import block, an orphaned doc,
   a closing brace) reads `5Jr7| use candle_conversation::…;` with its other
   lines bare beneath it. That is the shape of a per-line anchor (hashline's
   `LINE:HASH|content`, or a numbered listing). Read that way, `5Jr7` is one
   `use` line, which is what #29 replaced. The same misreading turned a part
   into one line at #41.
3. **A part's end is invisible.** It ends where the next anchor begins, and
   nothing says how long it is.

`replace` with no `old` adds a fourth fault. It is the same operation as the
safe `old` form, distinguished only by a missing field. So "change this piece"
and "this is the whole part now" are one keystroke apart, and the second is
taken silently.

**Decision: every part is its own fenced block, its lines numbered from `+1`,
and an edit names the part and the relative range.** Worked through with the
user (2026-11-06), from four alternatives:

| Option | Why not chosen |
| --- | --- |
| a header line per part, with its range | the extent shows, but a line of content shaped like a header still reads as one; it needs a heredoc-style closer |
| absolute line numbers | numbers shift after every edit, so a rebase through the VFS splices is needed; four digits a line in a large file |
| the anchor on every line (`KehT+19`) | about 6 tokens a line, 4× the gutter, for a locality no observed failure needed |
| **fenced part, `+N` gutter, edit by part and range** | — |

### The read

1. **One fenced block per part.** The fence names the file, the part, its
   absolute range and its scope. The part's lines follow, each behind its
   number from `+1`, then the fence closes.

   ````
   ```rust file=candle/npcd/src/engine/act.rs part=1Uax lines=46–52 scope="struct Act"
    +1 /// One act a character took.
    +2 #[derive(Clone, Debug, PartialEq, Serialize)]
    +3 pub struct Act {
    +4     pub tool: &'static str,
    +5     /// The call's arguments, minus the `tool` key.
    +6     pub args: Map<String, Value>,
    +7 }
   ```
   ````

   - **The extent is structural.** A part ends where its fence closes.
   - **The gutter protects the fence.** A content line holding ```` ``` ```` reads
     ` +3 //! ```text`. A fence only counts at the start of a line, so content
     can never close a block early. A line shaped like an old header
     (`Qm7x| …`, 11 such lines in this repository, all in format docs) sits
     behind its number like any other.
   - **`+N` says this is the part's own numbering.** It cannot be read as a
     compiler's line. A part is at most 80 lines (`MAX_UNIT_LINES`), so `N` is
     at most two digits, about 1.5 tokens a line. The model's tokenizer is
     measured for this before the cost is quoted as fact.
   - **Each block names its file**, so a part shown on its own (a `part` read,
     an edit's answer, a working-set placement) still says what it is.
   - A paragraph's `scope` is its kind (`use`, `doc`, `paragraph`, `}`). A
     container's opening line is its scope (`impl Act`).
2. **Fix the carver.** A doc comment attaches to its item in every language.
   This now affects scopes and page breaks, not edits, but a part still means
   what its scope says.
   - For each language, a test: a doc or comment immediately above an item, with
     and without an attribute between them, starts the item's part. Assert the
     part's start line, not only its label.
   - `start_with_leading_comments`
     (`doc-units/src/carve/parsers/tree_sitter_util.rs:181`) reads "immediately
     above" from the sibling's last line of text, not its end position.

### `file_read`: one of three keys

After `repo` and `path`, a read names exactly one of:

| Key | Value | For | Returns |
| --- | --- | --- | --- |
| `part` | an anchor, `"KehT"` | a part named by a hit, a list, an edit's answer | that part's block |
| `lines` | absolute, `"84"` or `"120-180"` | where a compiler or a stack trace points; "lines 120 to 180" | the whole parts covering them. The response notes `line 84 is part=KehT +30` |
| `list` | a page, `1` | "what is in this file", "cat it" | one line per part: `part=KehT lines=55–101 impl Act > fn summary` |

- **`list` entries.** A code part shows its scope. A paragraph (prose, YAML, an
  import block) shows its first line, cut at about 80 characters. Every entry
  carries its range.
- **A short file reads whole.** A `list` of a file that fits one page (200
  lines) returns its parts' blocks, so "cat it" stays one call.
- **A long list pages.** Its closing line names the next page.
- **`page` is removed.** Reading on is `lines`, and a response's closing line
  names the next range.
- **Absolute and relative never meet.** `file_read`'s `lines` is absolute
  with no `+`. `file_edit`'s numbers are relative and start with `+`.

### `file_edit`: the part and the relative range

```json
{"repo": "candle", "path": "npcd/src/engine/act.rs", "part": "KehT", "do": "replace", "from": "+19", "to": "+20", "text": "…"}
```

- **`from` and `to` are forced to `+N` by the grammar.** They are strings with
  the pattern `\+[1-9][0-9]?`; the stencil compiles a string `pattern` into
  the value's rule (`candle-conversation/src/stencil/tool_call.rs:540`). An
  absolute number cannot be decoded into an edit.
- A range past the part's last line is refused, naming the part's length. The
  grammar does not bind it per part, for the reason `at` is free text: the turn's
  decode budget (`docs/npcd_anchored_edits.md` §6).
- **Operations:**
  - `replace` takes `from`…`to` and `text`;
  - `insert_before` and `insert_after` take `from` and `text`;
  - `delete` takes `from`…`to`;
  - `move` stays part-level, naming `dest` and `where`.
  - `through` names the last part of a run of parts, as `to` does today.
- **Staleness is unchanged.** The anchor is a hash of the part's text, so a part
  that changed since it was read is refused.
- **No arithmetic.** Numbers count from the part's own first line, so an edit
  in another part never shifts them, and several edits from one read stay
  valid.
- **The answer returns the parts it wrote as blocks**, the same shape as a read,
  so the next edit needs no read. "Changed lines:" stays.
- **A `text` copied with its gutter is refused.** If every line of `text` starts
  with ` +N `, the refusal says to leave the numbers out.

### The stencil: exactly one of a group of keys

`merge_variants` (`tool_call.rs:283`) merges a `oneOf` into the union of its
arms' fields. That would admit `part` with `lines`, or neither, and only the tool
would refuse it. The grammar gains a group of alternative keys: after `repo` and
`path`, it offers one of `part`, `lines`, `list`, then that key's value shape,
then the close. In `field_order` this is one more kind of owed field, a group
satisfied by any one member, whose other members are then closed.

### Cold reads

| Starting from | Call | Then |
| --- | --- | --- |
| an error or a stack trace at line 84 | `file_read {lines: "84"}` | the part holding it, with `line 84 is part=KehT +30` |
| a name or a string | `file_grep` | its hits say `part=KehT +30`; `file_read {part}` reads that part |
| only the file | `file_read {list: 1}` | the outline; `file_read {part}` for what is wanted |
| a `code_reading` page in context | none | the corpus shows the same blocks. If the part's text is unchanged its anchor is still valid, so the edit can be made straight from it; if not, it is refused as stale |

### Ingest

- `code_reading` prefills `lines` reads in windows of whole parts, up to 200
  lines each, so the corpus is exactly what a live read returns.
- `KEY_VERSION` moves from 3 to 4 (`zend/src/branch_ingest/keys.rs:87`):
  - every file is read again once;
  - the old reading stays readable until its replacement commits;
  - a member still on 3 leaves a peer's 4 alone.
- No layer builds on `code_reading` (see Cleanup), so nothing else is re-read.
  `repo_map` folders whose README excerpt now reads differently re-read
  themselves, since a folder's key covers what its turns show.

### What is impacted

**Tools: definitions, behaviour and answers.**

| Tool | Change | Where |
| --- | --- | --- |
| `file_read` | `part` / `lines` / `list`; blocks; `page` removed | `zend-tools/src/tools/file/read.rs`, `render.rs`; `zend/src/prompts/tools/file_read.yaml` (description, questions, every example) |
| `file_edit` | `part` + `from`/`to` (`+N`); `through`; blocks in the answer; refusal of a copied gutter | `zend-tools/src/tools/file/edit.rs`, `edit_changes.rs`; `doc-units/src/edit.rs`; `file_edit.yaml` |
| `file_grep` | hits name `part=… +N` instead of part and page | `zend-tools/src/tools/file/grep.rs`; `file_grep.yaml` |
| `file_search` | its results point at a read; named as `part` | `zend-tools/src/tools/file/search.rs`; `file_search.yaml` |
| `file_list` | its answer says how to read a file (`list`) | `file_list.yaml`; `repo_scan/render.rs`, whose prefilled chain calls it with `page` |
| `write`, `file_present`, `file_image`, `git_show` | definitions that mention `file_read`'s pages | their `.yaml` |
| `git_show` (`what: file`) | a file at a revision, as raw text | whether it renders as parts too is decided in the build |
| zend's MCP endpoint | carries the same `file_read` / `file_edit` | `mcp/src/guides/*.md` and the tool guidance they quote |

**zend, around the tools.**

- `zend/src/code_read/` (`mod.rs`, `lines.rs`): the ingest's prefilled calls.
- `zend/src/repo_scan/` (`render.rs`, `converse.rs`, `dir_unit.rs`): a folder's
  prefilled `file_list` / `file_read` calls, with `page`.
- `zend/src/working_set/coverage.rs`: a chain is complete when its pages cover
  `1..=pages`. It becomes the reads covering every part.
- `zend/src/working_set/corpus.rs`: what the working set places from a reading.
- Inline read placement (`docs/inline_read_injection.md`,
  `candle-conversation/src/placed_span.rs`): a served read's tokens are found in
  the reading by occurrence. A `part` read's block is a run inside the ingested
  window, so it should place. The build checks that it does.
- `zend/src/repeat_guard.rs`: which `file_read` / `file_edit` calls count as
  repeats, by their arguments.
- `zend/src/tool_summary.rs`, `zend/src/tool_round.rs`, `zend/src/tools.rs`,
  `zend/src/tool_def.rs`, `zend/src/tool_availability.rs`,
  `zend/src/tool_guidance.rs`: what they say about a read or an edit.
- `zend/src/self_check/` and `zend/src/calibrate/run.rs`: fixtures holding
  `page=` fences.
- `zend/src/mcp/tools/timeline_read/` and `turn_report.rs`: how a call is shown.
- `zend/web/index.html`: a `file_read`'s detail is `page N` from its arguments,
  and its fence is read once the response lands.
- `zend/src/prompts/tool_constraints.yaml`, `projection.yaml`.

**npcd, which shares `doc-units`.**

- `npcd/src/engine/work.rs`, `npcd/src/sim/bench.rs` (and `bench/tests.rs`):
  the bench's read and edit acts.
- `npcd/src/engine/tools.rs`: the `file_edit` act's catalog entry, its
  `BRINGS`, and the stencil params.
- `npcd/src/engine/compose.rs`, `npcd/src/engine/runtime.rs`: the mending
  sitting. Its `mend` edits bind `at` to the piece's anchors, which become
  `part`, with `from`/`to`.
- npcd's own read header (`… (14 paragraphs):`) becomes the block shape.

**Shared.**

- `doc-units`: `render.rs` (the block), `edit.rs` (ranges), `carve/` (the
  attachment).
- `candle-conversation/src/stencil/`: the alternative-key group (`tool_call.rs`,
  `field_order.rs`), and the `pattern` on `from`/`to`.
- `candle-conversation/tests/tool_cases/` (`MANIFEST.json`, `prompts.json`,
  `test_config.json`): calls written with `page`.

**Tests**, each asserting exact text:

- `zend-tools/tests/file.rs`, `file_overlay.rs`, `file_repos.rs`,
  `file_search_grep.rs`;
- `npcd/tests/vault.rs`;
- the render, carve and edit tests in `doc-units`.

**Docs.**

- `docs/npcd_anchored_edits.md` (§3, §5, §6, §7, §7a, §8) is rewritten to this
  design.
- These change to match:
  - `docs/tool-system.md`
  - `docs/zend_working_set.md`
  - `docs/zend_branch_ingest.md`
  - `docs/inline_read_injection.md`
  - `docs/mcp_vfs_branch_file_reads.md`
  - `zend-tools/src/tools/file/README.md`
  - `ARCHITECTURE.md`

**Where the losses go.**

- #29 and #41: the model names the lines it replaces.
- #53 and #65: the carver fix, and the range naming exactly what is replaced.
- #30–#32: shown by "Changed lines:".

**Sources.** aider.chat/docs/unified-diffs.html; aider.chat/docs/leaderboards;
arxiv.org/html/2510.12487v2 (Diff-XYZ); stencil.so/blog/the-harness-problem;
dev.to/nwyin/hashline-vs-replace-does-the-edit-format-matter-15n2;
cursor.com/blog/instant-apply; morphllm.com/benchmarks/fast-apply;
developers.openai.com/cookbook/examples/gpt4-1_prompting_guide.

## Cause 5 — a build ran without the worker's edits, and passed

This is about build jobs (`repo_job_start`), not sub-agent tasks
(`repo_task_start`).

**Evidence.** From the npcd task's call listing:

1. At #95 the worker made a branch from `origin/main`, carrying its 10 modified
   files as uncommitted changes.
2. At #96 it started `cargo test -p npcd --lib engine` on that branch.
3. The agent ran the branch as origin held it, which was `origin/main`'s tree
   without the worker's edits.
4. At #99 the worker recorded *"npcd lib engine tests passed: 1399 passed"*.
   That was a false pass.
5. Only my nudge (#101) caught it. The first build that held the edits (#105)
   failed on the imports dropped at #29.

`docs/claude_zend_gateway.md` says a worker builds with `run_command`, which
lays its uncommitted work down, and that `repo_job_start` is Claude's. But the
worker has `repo_job_start` and used it. Its answer gave no sign that the
worker's edits were missing.

**What was checked.**

- **The changes are in the substrate.** Every delta is a `VfsEvent` record
  (`RecordType` 25; a dropped chain is a `VfsTombstone`, 26), keyed by the
  conversation's timeline. They are saved after every tool round
  (`conv_overlay::save` → `store_events::save`). They replicate: `is_local_only`
  leaves them out. Any member can rebuild a conversation's store from them
  (`conv_overlay::restore`).
- **The agent holds no substrate.** `agent/` depends on `cluster` and
  `zend-vfs` only (`agent/src/member.rs`: "no model and no substrate").
- **`run_command` already ships the changes.** On the cluster it runs through
  `AgentExecutor` (`zend/src/agent_exec.rs`), which sends
  `RunJob { overlay: files.snapshot(), … }`. The agent builds a store on the
  branch, restores the overlay (`agent/src/jobs.rs`), materialises it and runs.
- **`repo_job_start` sends an empty overlay** (`Snapshot::default()`,
  `zend/src/mcp/tools/repo_job/start.rs:117`). That empty overlay is the whole
  bug.
- **The protocol named is zend's worker hand-off, not npcd's.** npcd's NPC
  lease (`npc_lease.rs`, `cast.rs`) carries no records: the next holder waits
  until it has applied every origin through the `through` the hand-back names.
  The ask / reject-with-missing / resubmit handshake is zend's worker hand-off:
  1. `workers::hand_on` posts a signed `Handing` to `/v1/mcp/worker/continue`.
  2. The receiver, lacking records, answers `409 {error: "missing", missing:
     [[s, e], …]}` (`hand_records::missing`, at most 256 ranges).
  3. The sender gathers them (`hand_records::gather`, within 30 MiB, or the
     conversation's own records alone). It posts once more with them in the
     body, and the receiver applies them (`apply_carried`) before taking.

**Decision.** A job started from a conversation runs on the conversation's
changes, by the hand-off's handshake.

1. The job request names the overlay it expects: for each repository, the base
   commit and the identity of each file's chain (the SHA-256 of its JSON).
2. The agent holds overlays it has been sent, by those identities, as it holds
   its warm checkout. If it holds every delta named, it runs. Otherwise it
   answers `409 missing` with the deltas it lacks.
3. The caller resubmits to that agent with those chains in the body, from the
   snapshot it named them from. The agent checks each against its name,
   restores the store, materialises it and runs.
4. A job's answer names the commit and how many uncommitted files were laid onto
   it. A pass is always a pass of what was asked.
5. A caller with no conversation (Claude through the gateway) names no overlay,
   and the job runs the branch as origin holds it.
6. `run_command`'s `AgentExecutor` moves onto the same handshake. Then one path
   carries a conversation's changes to an agent, and a long session's unchanged
   deltas are sent once, not on every run.

**Open.** Sub-agent tasks: a task a worker starts sees only its starter's
findings, through its prompt. Since the starter's changes are records already,
the new task's store could start from the starter's (its `VfsEvent`s copied
under the new timeline). A reviewer or a test-writer would then read the change
as it stands, without a commit first.

**Where.**

- `zend/src/mcp/tools/repo_job/start.rs` and `zend/src/agent_exec.rs`, the two
  senders, both through `Agents::submit` (`agent/src/client.rs`), which sends
  the names and resends what is missing.
- `agent/src/overlay.rs`: the overlay by name, and the `409` body.
- `agent/src/chains.rs`: the agent's chains by digest.
- `agent/src/jobs.rs` and `agent/src/routes.rs`: resolving a run's overlay, and
  the 409.
- `docs/jobs.md`; `docs/replicated_substrate.md` §11.4.

## Cause 6 — `branch: main` reads a local cache of origin

**Evidence.**

- single-segment-handles called `git_show` with `{"kind": "branch", "name":
  "main"}`, and it was refused `unknown_revision refs/heads/main`.
- The worker spent two rounds finding `origin/main`.

**What was checked.** Origin is meant to be the only record, and the design
says so: `docs/zend_git.md` §7.8, *"A repository's local refs are a cache of
origin's."*

- A conversation's commit is published to origin before its local branch moves
  (`Committing::land` → `publish_commit`), so nothing a conversation writes is
  local only.
- But reads "read the local refs".
- `RevArg::resolve` (`zend-tools/src/tools/git/mod.rs:332`) turns `kind: branch`
  into `Rev::Branch`, which is `refs/heads/<name>`.
- A local branch exists only once a write has pulled it (`pull_branch` makes it
  "when missing"). So `main`, which this checkout never wrote to, had no local
  ref, while `refs/remotes/origin/main` had it all along.
- Where a local branch does exist, it can lag origin's.
- So the tools do expose a "local", against the design.

**Decision.** A branch revision reads origin's record, never the local cache.

- `kind: branch` resolves through the record, as `Repo::record_branches` does:
  `refs/remotes/origin/<name>`, or with no `origin` remote the local branch.
- `git_refs` lists branches the same way.
- `remote_branch` stays for a remote other than origin.
- An unknown branch is refused, and the refusal lists the record's branches.

**Where.**

- `RevArg::resolve` and `rev_or_head` in `zend-tools/src/tools/git/mod.rs`.
- `git_refs`'s `branches` in `zend-tools/src/tools/git/`.
- `docs/zend_git.md` §7.8: reads resolve branches through the record.

## Cause 7 — a nudge reads as the user speaking

**Evidence.**

- A `repo_task_nudge` from Claude reached the npcd worker as a message with no
  sender.
- The worker's journal records it as "Note: CORRECTION from the user".
- The journal then credits the user with a decision they never made.

**Decision.** A nudge or a reply arrives with its sender's name: the
caller's display name, or "the agent that started this task". The worker's
prompt gives it in that form.

**Where.** `workers::nudge` and the reply path in `zend/src/mcp/workers.rs`,
where the queued text is delivered.

## Cause 8 — refusals without a way on

**Evidence.**

- `file_grep` was refused `invalid_arguments` "pattern: length" twice, in two
  tasks. Both patterns were empty, because the workers wanted every file under
  a prefix.
- `file_edit` answered `not_found` three times when its `old` text was not in
  the part.
- `file_edit` once answered stale because an anchor came from a read made before
  an earlier edit in the same round.

**Decision.** Each of these answers with what to do next.

- **Empty `file_grep` pattern:** "the pattern is empty; to list the files use
  `file_list`".
- **`file_edit` `not_found`:** the answer carries the part's current text, so
  the retry has it.
- **Stale anchor:** the answer carries the part's current anchor and text.

**Where.**

- The empty pattern: `zend/src/tool_guidance.rs`, per tool and code.
- The other two: the `file_edit` error paths in
  `zend-tools/src/tools/file/edit.rs`.

## Cause 9 — `run_command` is not replayed after a restart

**Evidence.**

- sandbox-priority-test's `cargo test` call was answered
  `not_replayed_after_restart`.
- `run_command` is `Replay::Unsafe`, because a command can do anything.
- The worker asked again, losing one round.

**Decision.** `run_command` declares `Replay::Safe` for these programs, whose
only effect is the build directory the sandbox throws away:

- `cargo test`, `cargo check`, `cargo clippy`, `cargo build`
- `cargo fmt --check`, `git status`, `git diff`, `git log`

Anything else stays `Unsafe`.

**Where.** `run_command`'s `replay` in `zend-tools`, with a test per program and
one for a refused program.

## Cleanup — references to layers that no longer exist

**Evidence.** The live layers are `repo_map`, `code_reading` and `dialogue`
(`zend/src/prompts/projection.yaml`). None declares an `ingest:` block, so no
layer builds on `code_reading`. The code still speaks of `static_analysis`,
and that misled this study into costing a re-read of layers above
`code_reading` for Cause 4. The references:

- `zend/src/layers/spec.rs` module doc. Its example declares
  `dependency_analysis` with `after: static_analysis`.
- Test fixtures that use `static_analysis` as a layer name:
  - `zend/src/layers/spec.rs` (`spec()`, `check_tools` tests);
  - `zend/src/layers/pass.rs`;
  - `zend/src/layers/keys.rs`.
- `zend/src/layers/keys.rs` module doc, on "a layer above `code_reading`".
- `docs/coding_assistant.md` (two schema examples) and
  `docs/provenance_adaptive_projection.md` (a list of layers that "cannot
  score").

**Decision.** Each goes.

- The fixtures take a neutral name that names no real or past layer, and assert
  exactly what they did.
- The module docs describe the code as it stands, without an example of a layer
  that is not declared.
- The two docs drop the layer, or say it was removed.
- `docs/archived/` is left as it is.

**Not in this cleanup.** Whether the stacked-layer machinery in
`zend/src/layers/` (`ingest:` specs, `after:`, edges, expiry, `unit_key`) stays
for layers to come, or goes as unused, is the user's call. It is not decided
here.

## Out of scope

- The gateway page's 688 KB status poll, mostly its 500-entry log.
- The MCP connector's tool schemas being cached per Claude Code session.

Neither affects how well Zen codes. Both are worth their own change.
