# Anchored document edits in npcd and zend

How a character changes one part of a file without writing the rest of it
again: every unit of the file (a paragraph of prose, a function or type of
code) carries a short content anchor, `file_read` shows it, and `file_edit`
names the unit by its anchor and says what to do there.

Status: implemented (`doc-units`, `zend-tools/src/tools/file/`, zend's tool
definitions, npcd's bench acts and `compose`). Replaces `file_edit`'s
`old_str`/`new_str` form and the whole-piece mending sitting of `compose`;
`file_write` and `compose` for a first draft are unchanged. It covers npcd's
characters and zend's coding conversations alike: zend's file tools are
`zend-tools`', npcd's are its own bench acts, and both carve, anchor and edit
through `doc-units`.

## 1. Why

Every fix and review step of the story pipeline (`docs/npcd_workflows.md`) runs
through `compose`, which decodes the whole piece again: 600–900 words, about
800–1,300 tokens, up to `compose::COMPOSE_TOKENS` (2,400). A reviewer mending
one sentence pays for all of them.

Decode is the scarce side on the machines npcd runs on. On WhiteWorld (RTX 3090,
PCIe 3.0, Qwen3.8-Flash-Next) with five Makers decoding, a decode-only step is
~500 ms and the cast decodes ~28 tok/s between them — roughly 6–10 tok/s each
(measured 2026-10-26 with `--features profile`). A whole-piece mend therefore
costs a Maker 1.5–3 minutes of decode; an edit of the paragraph it changes costs
10–20 s.

`file_edit` exists for this, but its anchor is the text being replaced
(`old_str`), which the model has to copy exactly and which must appear once.
Every token of that copy is decoded, a slip is refused and retried, and a short
repeated line ("No.") needs widening with its neighbours until it is unique.

## 2. Alternatives considered

| anchor | survives edits elsewhere | stale read caught | decode cost | why not |
|---|---|---|---|---|
| whole document (`file_write`, `compose`) | — | — | the whole piece | §1 |
| `old_str` (today's `file_edit`) | yes | yes, by matching | copies the old text | decoded copying; refusals on slips and duplicates |
| unified diff | no | partly | context lines | token-heavy, fragile for models |
| line number | **no** — shifts on insert/delete | only with an engine version check | ~2 tokens | several edits from one read drift |
| line number + hash (hashline, `7:aZ3q`) | no | yes | ~2 + forced | still shifts |
| stable IDs (CRDT, block IDs) | yes | needs a per-paragraph check | ~2 tokens | an ID map to store and carry across outside edits |
| **paragraph content hash** | **yes** | **yes, by construction** | **~1–2 decoded + forced** | — |

A content hash needs no stored state: it is computed from the text at read
time, so the engine's own rewrites (`gates::tidy`) and merges from the record
need no reconciliation. A paragraph another hand changed has a different hash,
so an anchor taken before the change no longer exists and the edit is refused
rather than applied to text the character never saw.

A per-version seed was considered and rejected: it changes every anchor on every
edit, so nothing survives the first edit and the character must re-read before
the second.

## 3. The unit

A unit is what an author means by "that part": a paragraph of prose, a function
of code. Units partition the file's non-blank lines: in order, never
overlapping, never starting or ending on a blank line, and together holding
every line that has anything on it. Blank lines between units belong to none —
they are the separators an edit keeps. The carving, the anchors, the rendering
and the edits are one crate, `doc-units`, which `zend-tools`, `zend` and `npcd`
all use.

- **`.md` documents: the paragraph** — a run of non-blank lines between blank
  lines. A heading, a list, a block quote or a scene break is its own unit. A
  fenced code block is one unit however many blank lines it holds, and YAML
  front matter is one unit. Blank lines carry no anchor; the edit act restores
  the separating blank line on insert and delete.
- **`.yaml` documents: the line.** Character sheets, world files and
  `missions.yaml` are line-structured, and a paragraph of YAML is not a unit
  anybody edits.
- **Code: the semantic scope**, carved by the per-language tree-sitter carver
  zend used to chunk `code_reading` before that layer moved to real file reads
  (removed in `46e247083`; restored by §9). Its rules carry over:
  - **Tier 1, tree-sitter**: Rust, Python, TypeScript/TSX, JavaScript, Go, C,
    C++ (CUDA carves as C++), Java, Ruby, PHP, Bash, HTML, CSS. Each language
    maps node kinds to a unit: a function or method, a type definition (struct,
    enum, trait, class, interface, type alias), a run of constants, or another
    top-level item (`macro_rules!`, a module-level statement). An item's
    leading doc comments and attributes are part of it. Containers are not
    units — an `impl` or `mod` is the path its members are named under
    (`impl KvCache > fn seal_chunk`), so a large impl never becomes one unit.
  - **The innermost item owns its lines.** Parsers emit nested items — a
    method inside a class, a function inside a function — and an outer item
    keeps only the lines none of its members claims: a class is its header and
    fields, each method a unit of its own.
  - **Tier 2, keys**: TOML and JSON split at top-level keys.
  - **Unclaimed lines are paragraphs**: `use` lines, module attributes, stray
    statements, a container's opening and closing lines, and the whole of a
    file no parser fires on are split at blank lines into units of their own,
    so coverage is total. A text file with no carver is paragraphs throughout.
  - **File header**: a leading doc or licence comment of two lines or more is
    its own first unit.
  - **Size cap**: an item over `MAX_UNIT_LINES` (80) or `MAX_UNIT_CHARS`
    (4,000) is split by its parser where one of its body's statements begins
    (labelled `[part N]`), and anything still over a cap — a paragraph too —
    at line boundaries. Long lines are left as they are: an edit unit is the
    file's own text, never a re-wrapped copy of it.
  - **No merge pass.** The old carver merged small scopes forward to at least 50
    lines, which sized a prefill turn for ingest. An edit unit is the item
    itself: a three-line helper is its own anchor.

  Semantic units make collisions rare in code: a function or type is almost
  always unique text, so most of the per-line duplicates that dominate source
  (`return Ok(());`, blank lines) never get an anchor of their own. A
  container's bare closing brace is the exception — it is a unit, and two of
  them take seeds 0 and 1 — which costs nothing, since a closing brace is not
  what anybody edits. A 1,000-line source file is 30–60 units, not 1,000
  lines.

## 4. The anchor

```
anchor(p) = base64url( H( seed ‖ normalise(text) ) )[0..4]
```

- **`H`** is SHA-256. Only its first 24 bits are used: four base64url
  characters (`A–Z a–z 0–9 - _`), so no JSON escaping.
- **`normalise`** strips trailing whitespace from each line and normalises line
  endings, so formatting noise does not change an anchor.
- **`seed`** makes every anchor unique within the document. The engine computes
  the document's anchors at once, in document order, each at `seed = 0`; a
  paragraph whose anchor an earlier paragraph already holds is hashed again at
  `seed = 1`, then 2, until its anchor is free.
  - **Duplicates are collisions.** Two identical paragraphs collide at seed 0,
    so the second takes seed 1: repeated text needs no mechanism of its own.
    Short repeated lines are the common case in prose ("No.", a scene-break
    line), and they are exactly what a reviewer edits.
  - **Every anchor is four characters**, whatever collided, so the grammar and
    the decode cost are uniform.
  - **Document order decides who keeps seed 0**, so the same text always reads
    with the same anchors. A unique paragraph's anchor depends on its own text
    alone. A bumped anchor depends on the earlier paragraph it collided with: if
    that one is removed, the bumped paragraph's anchor changes, and an edit
    holding the old one is refused (§6), never applied to the wrong copy.
  - Correctness never depends on the hash length; the length only bounds how
    often a seed is bumped and how likely a stale anchor is to name a new
    paragraph by chance.

How often a seed is bumped, and how likely a stale anchor is to match new text:

| anchor | space | collision, 60 paragraphs or code units | collision, 300 YAML lines | stale anchor matching new text |
|---|---|---|---|---|
| 3 chars | 262 K | 0.7% | 16% | 1 in 262 K |
| **4 chars** | **16.7 M** | **0.01%** | **0.3%** | **1 in 16.7 M** |

## 5. Reading

`file_read` labels each unit with its anchor. A unit named by its own first
line — a paragraph, a YAML line — shows that line after its anchor and the rest
beneath it (npcd's header shown):

```
layers/life/bram/2834-11 The Body.md (14 paragraphs):
Qm7x| The bay was warm when you walked in, the tank fluid already cycling,
and the light was the light of a room nobody had left in years.
b4_K| Then the order came down on the shared frequency…
9fTa| You said fourteen.
```

A code unit shows its anchor and scope path, then its lines exactly as they
are, indentation and all (zend's fence shown):

````
```rust file=candle/candle-nn/src/kv_cache/chunked/backing.rs page=1 pages=3 lines=620
Hk2a| file header
//! Chunked KV backing.
Zp0_| use std::sync::Arc;
use std::collections::BTreeMap;
r7Lm| impl KvCache > fn seal_chunk
    pub fn seal_chunk(&mut self, gid: ChunkGid) -> Result<()> {
        …
    }
```
end of candle/candle-nn/src/kv_cache/chunked/backing.rs page 1 of 3 — next: page=2
````

- **Paging packs whole units**: a page holds whole units, up to 200 lines of
  them (`doc_units::render::PAGE_LINES`), so no unit is ever cut across two
  pages. `page` counts from 1; a page past the end reads the last. A page's
  line range depends on the file, so zend's `file_read` also takes `line` — an
  error's or a stack trace's — and returns the page holding it.
- `file_grep` names each hit's unit and the page holding it (§7a).
- The label path is not hashed: the anchor is the unit's text alone, so moving a
  function into another `impl` does not change its anchor, and renaming it does.
- **Nothing about a read is remembered.** Anchors are computed from the text
  every time, so the staleness check (§6) needs no record of what was shown,
  and a restart forgets nothing it depends on.

## 6. Editing

`file_edit` takes an anchor and an operation:

```json
{"path": "layers/life/bram-the-gardener/2834-11 The Body.md", "at": "9fTa", "do": "replace", "text": "You said fourteen, and meant it."}
{"path": "…", "at": "9fTa", "do": "replace", "old": "fourteen", "text": "fifteen"}
{"path": "…", "at": "9fTa", "do": "insert_after",  "text": "The tank sighed."}
{"path": "…", "at": "Qm7x", "do": "insert_before", "text": "…"}
{"path": "…", "at": "b4_K", "do": "delete"}
{"path": "…", "at": "b4_K", "to": "9fTa", "do": "replace", "text": "…"}
{"path": "…", "at": "Qm7x", "do": "move", "dest": "cX0e", "where": "after"}
```

- **`replace`** without `old` replaces the whole unit with `text`. With `old`,
  it replaces that substring of the unit, which must appear exactly once **within
  the unit** — so `old` can be just the words being changed, not their
  neighbours.
- **`insert_before` / `insert_after`** add `text` as a new unit beside the
  anchored one.
- **`delete`** removes the unit; `text` is not given.
- **`to`** (optional, on `replace` without `old` and on `delete`) extends the
  span from `at` through `to`, inclusive, in document order: three adjacent
  paragraphs or helpers are one call. A `to` before `at` is refused.
- **`move`** takes the unit at `at` (through `to`, when given) and puts it
  `before` or `after` the unit at `dest`, without decoding its text again. A
  `dest` inside the moved span is refused.
- One operation per call. Several calls from one read stay valid, because an
  edit changes only its own unit's anchor (and the anchor of any later unit
  whose seed was bumped by it — a copy of the same text).
- **Code is carved again after every edit.** An edit can change where units
  begin and end: an inserted function is its own unit, and replacing a unit with
  two functions yields two. The reply (below) reports the units the edit's span
  carves into, with their anchors. An edit that leaves the file unparseable
  still applies — tree-sitter recovers what it can and the rest is carved as
  paragraphs — and the reply says so, so the author knows its next edit should
  mend it.
- **Separators are kept.** An insert is set apart by a blank line where the
  file sets its units apart there (always, in prose); a delete takes the blank
  line after it, so no double gap is left. Line endings and a missing final
  newline are the file's own and stay so.
- **In prose, an edit writes only the part it names.** A replace or an insert
  whose `text` holds a unit that already stands in the document — outside the
  span it replaces, whitespace aside — is refused, naming that unit's anchor:
  *"that text holds Qm7x, which already stands in it as it is; an edit writes
  only the part it names — give that part's new words alone, or, to write the
  whole of it again, replace from its first part `to` its last"*. A Maker
  replaced one paragraph of a story with the whole story, heading and all, on
  edit after edit: each pasted every other paragraph back in, until each
  sentence stood eight times and the story ran to 5,474 words against 1,300.
  Replacing a paragraph with itself and more is allowed (the copy is inside the
  span), and so is rewriting the whole document from its first unit `to` its
  last. Code is left alone: two scopes of one text are its author's to want.

### The grammar

The edit is decoded under the tool grammar like every act.

- **`do`** is an enum of the five operations and **`where`** of `before` and
  `after`. In npcd's turn grammar each operation **brings the fields it takes**
  (`engine::tools::BRINGS`, the stencil's per-value `shapes`): a replace or an
  insert its `text`, a move its `dest` and `where`, a delete nothing. Left as
  optional siblings of `do`, those fields doubled the decodes the act admits
  once each, and with `file_edit` at every bench a crowded room went to
  4 × 10⁷ decodes over a turn against the 2 × 10⁶ budget
  (`estimated_paths`, `MAX_TURN_PATHS`); brought by their operation, they are
  summed instead. zend's schema carries the same enums.
- **`path` and `at` are free text in a character's turn grammar.** Binding `at`
  to the anchors shown — each `path` bringing its document's anchors — puts an
  enum of tens of anchors per document into every turn and multiplies the turn's
  decodes past the same budget. An anchor the character mistypes or remembers
  from an earlier version is refused instead (below), and its read is right
  above it to copy four characters from. A mending sitting, whose grammar is its
  own and covers one document, binds them (§7).
- **`old`** is free text checked by the act: the stencil has no substring
  constraint, so an `old` that does not occur once in the unit is refused.
- A document with no units (empty, or whitespace only) offers nothing to anchor
  to; its first text is written with `file_write`.

### Staleness

An anchor is its unit's text, hashed, so an edit is applied only to a unit
whose text is what the anchor was taken from: if the document changed under it
— another edit, the engine's tidy pass, a merge from the record — the anchor no
longer names any unit, and the act refuses: *"Qm7x is not a part of it as it
stands now — it has changed since you read it, or was never one of its anchors;
read it again and edit by the anchors it shows."* No edit is ever applied to text
the character did not see. (A different text matching a stale anchor by chance
is 1 in 16.7 M.)

Because the check is the document itself, nothing is remembered between a read
and an edit: an edit after a restart is checked exactly as one before it.

**Re-sending an edit is safe except an insert.** A replace or delete re-sent
after it landed names an anchor the document no longer has and is refused; a
move re-sent puts its unit where it already stands. An insert leaves its anchor
as it was, so zend's resumed turns do not re-issue one (`Replay::Unsafe`).

### The reply

Every edit answers with what was done, then the units it wrote, whole, under
their new anchors, between the unit before and the unit after, each in one line:

```
Changed 9fTa.
b4_K| Then the order came down on the shared frequency…
R2kp| You said fifteen.
cX0e| The tower lord's reply came back flat…
```

A delete shows the units now either side of where it was. The next edit can
name the new anchors without reading the document again.

## 7. Mending sittings

The mend steps (`fix`, and `review` when it sends a document back) stop
re-composing the piece. The mending sitting is put the faults and the piece
under its paragraphs' anchors, and offered two calls: **`mend`**, whose `edits`
(one to `compose::MOST_EDITS`, 12) each name a paragraph as `at` — bound in the
sitting's grammar to the piece's own anchors, and `to` likewise — and say what
to `do` (`replace`, `insert_before`, `insert_after`, `delete`, with `text` and
`old` as in §6); and **`draft`**, the whole piece written again, for a sitting
that has to change most of it. The engine makes the edits one by one
(`compose::mended`) and writes the result into the working copy through
`file_write`, as a draft is; an edit naming a paragraph an earlier one already
changed finds its anchor gone and is refused, and the answer says which. A
mend none of whose edits could be made is refused whole.

`compose` keeps writing first drafts whole, and a draft short of its floor is
carried on as before; only the first sitting may mend.

## 7a. Other places that cite lines

- **`file_grep`** labels each hit with the anchor and scope path of the unit it
  falls in (`part`) and the `file_read` page that holds it (`page`), so a hit
  can be read whole, or edited, without counting pages.
- **Tool output that is not a read** — compiler errors, test failures, diffs,
  blame — keeps its own line numbers. It describes the world, not the editing
  surface; an author goes from it to a read — `file_read`'s `line` opens the
  page holding the line — and edits from the read.
- **zend's MCP endpoint** carries the same `file_read`/`file_edit` as every other
  caller: one form, anchors. Its tool description teaches the convention.

## 8. Expected cost

| change | today | anchored |
|---|---|---|
| one word in one sentence | ~800–1,300 decoded (`compose`), or the `old_str` copy plus context | ~15–25 (`at` + `old` + `text`) |
| one paragraph rewritten | ~800–1,300 | ~60–110 (the paragraph) |
| a reread of a 60-paragraph story | 60 line numbers | + ~240 prefill tokens for the anchors |
| one line in a 40-line function | the `old_str` copy, widened until unique | ~15–25 (`at` + scoped `old` + `text`) |
| a reread of a 1,000-line source file | 1,000 line numbers (~3,000 tokens) | 30–60 anchors and paths (~400–800 tokens) |

Prefill is the cheap side (hundreds of tok/s on the same machine), so the read
cost is negligible beside the decode it saves.

Where `at` is bound (a mending sitting), the anchor's forced remainder is
decided without logits (`batched_sampler`'s single-allowed-token path), but
whether it is fed in one step or one decode step per token is measured, not
assumed (§10). Where it is free (a turn), an anchor is its 2–4 tokens decoded.
Either way it is a handful of tokens against the hundreds a quoted `old_str`
cost.

## 9. Migration

There is no backward compatibility, by the repository's standing rule: the
anchored form **replaces** the line-numbered one, with no dual path and no flag
to keep the old one alive.

1. **The carver is restored as the `doc-units` crate.** `zend/src/code_read/`
   `carve.rs`, `types.rs` and `parsers/` came back from `46e247083^` into
   `doc-units/src/carve/`, with their parser tests — a crate of its own because
   npcd, which does not depend on `zend-tools`, edits with it too. The merge
   pass, the fixed-window and header carvers and the long-line rewrite are not
   restored, and the size caps are the edit-unit caps of §3. The tree-sitter
   grammar dependencies moved from `zend/Cargo.toml`, where nothing called them,
   to `doc-units`; the single `tree-sitter` 0.26 pin `yamlpatch` needs stays
   workspace-wide. `zend-vfs`'s `old_text` replace engine is gone.
2. **`file_read` and `file_edit` change in place**, in `zend-tools`, zend's tool
   definitions (`zend/src/prompts/tools/`) and npcd's bench acts. Line numbers
   and `old_str`/`old_text` are removed from the reads and the act; every
   caller, prompt and example is moved to anchors in the same change.
3. **zend's `code_reading` units are re-read.** Their turns were read with line
   numbers and cite them, so they describe a read format that no longer exists.
   `KEY_VERSION` 3 (`zend/src/branch_ingest/keys.rs`) re-keys every unit, so the
   next ingest pass reads each one again with anchored pages and tombstones the
   old reading once the new one commits. A unit now records the pages it read
   (`pages`, counted from the content) where it recorded a line count, which is
   what the fast path checks a chain read in full. The re-key reaches every unit
   keyed by `KEY_VERSION` — `repo_map`'s folders and the layers above
   `code_reading` too — so the whole corpus is re-read.
4. **The layer is also wiped at release**, so no line-numbered turn stands while
   its replacement is read: the first boot of the new build on each member runs
   with `--wipe-layer code_reading` (`DaemonConfig::wiped_layers`), once.
5. **npcd needs no migration of its own.** Anchors are computed from the text
   at read time and nothing about them is persisted (§5), so documents, drafts
   in operation stores and the record carry over unchanged. A mission open
   across the upgrade reads its document again before it edits, which the
   staleness rule (§6) requires anyway.

## 10. Tests

Unit tests beside the code, asserting raw values:

- **Anchors**: fixed expected strings for known paragraphs; trailing-whitespace
  and line-ending normalisation; two identical paragraphs taking seeds 0 and 1;
  a constructed collision between different texts resolved by a seed bump, the
  anchor still four characters.
- **Units**: the restored parser tests, asserting each language's expected
  scope paths; coverage (every line in exactly one unit) for every language;
  no merge — two adjacent three-line functions are two units; a 200-line
  function splits on child boundaries within the caps; a fenced code block in
  `.md` with blank lines inside is one unit; front matter is one unit.
- **Edits**: each operation on a small document, the restored blank lines, the
  reply's new anchors; a scoped `old` that occurs twice in the unit is refused,
  once is applied; inserting a function after a code unit returns it as its own
  unit; an edit that breaks the parse applies and reports the fallback.
- **Staleness**: an anchor read, then the unit changed by another hand → refused;
  a unit elsewhere changed → the edit still applies.
- **Duplicates**: deleting the first of two identical paragraphs leaves the
  second's old anchor refused, never applied to the wrong copy.
- **Grammar**: each `do` value brings exactly the fields it takes; a crowded
  room stays inside the turn's path budget; zend's catalog, `file_edit`
  included, compiles against the real tokenizer; a mending sitting's `at` and
  `to` are the piece's own anchors.
- **Ranges and moves**: `to` before `at` refused; a move into its own span
  refused; a moved unit keeps its anchor.
- **Pasting a part in again**: a prose replace or insert whose text holds a unit
  standing elsewhere is refused with the exact message above; a replace with
  itself and more, a whole rewrite first `to` last, and code are not.
- **Grep**: a hit names its unit's anchor and page, and that anchor then edits
  the unit with no read in between.
- **Mending**: a `mend` is read from its call and made edit by edit; an edit on
  a paragraph an earlier one changed is refused.

Measurement, before and after landing:

- **Baseline first**, from the build before the change: decoded tokens and
  refusals per `fix` and `review` step on WhiteWorld, and zend's `file_edit`
  refusal rate.
- **After landing**, the same figures, edit refusals by cause, operation
  turnaround through the pipeline, and decode steps per emitted anchor (§8). A
  mending sitting logs `mended a piece by its anchors` with how many edits were
  made and refused, beside `composed a draft`.
