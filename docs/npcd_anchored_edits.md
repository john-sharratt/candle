# Anchored document edits in npcd and zend

How a character changes one part of a file without writing the rest of it
again: every unit of the file (a paragraph of prose, a function or type of
code) carries a short content anchor, `file_read` shows it, and `file_edit`
names the unit by its anchor and says what to do there.

Status: design. Replaces `file_edit`'s `old_str`/`new_str` form and the
whole-piece mending sitting of `compose`; `file_write` and `compose` for a first
draft are unchanged. `file_read` and `file_edit` live in `zend-tools`
(`tools/file/`), so the change covers npcd's characters and zend's coding
conversations alike.

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
of code. Units cover the file completely and in order, so every line belongs to
exactly one.

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
    top-level item (`macro_rules!`, a module-level statement). Containers are
    not units — an `impl`, `mod` or `class` is the path its members are named
    under (`impl KvCache > fn seal_chunk`), so a large impl never becomes one
    unit.
  - **Tier 2, headers**: TOML and JSON split at top-level keys.
  - **Tier 3, fixed windows** for a language with no carver, or a file its
    parser rejects.
  - **Gap fill**: lines no scope claims (`use` lines, module attributes, stray
    statements) become units of their own, so coverage is total.
  - **File header**: a leading doc or licence comment of two lines or more is
    its own first unit.
  - **Size cap**: a scope over `MAX_UNIT_LINES` (80) or `MAX_UNIT_CHARS` (4,000)
    splits on its children's boundaries first (match arms, nested blocks,
    labelled `[part N]`), and at line boundaries only when it has none. Lines
    over 160 characters are broken at safe points outside string literals
    before carving, as before.
  - **No merge pass.** The old carver merged small scopes forward to at least 50
    lines, which sized a prefill turn for ingest. An edit unit is the item
    itself: a three-line helper is its own anchor.

  Semantic units make collisions rare in code: a function or type is almost
  always unique text, so the per-line duplicates that dominate source (`}`,
  blank lines, `return Ok(());`) never get an anchor of their own and never
  cascade seed bumps through the file. A 1,000-line source file is 30–60
  units, not 1,000 lines, which bounds both the read cost and the grammar's
  `at` enum (§6).

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

`file_read` labels each unit with its anchor:

```
(paragraphs 1–14 of 14)
Qm7x| The bay was warm when you walked in, the tank fluid already cycling…
b4_K| Then the order came down on the shared frequency…
9fTa| You said fourteen.
```

A code unit is labelled with its anchor and its scope path, then shown whole:

```
(units 1–41 of 41)
Hk2a| file header
//! Chunked KV backing.
Zp0_| use
use std::sync::Arc;
r7Lm| impl KvCache > fn seal_chunk
    pub fn seal_chunk(&mut self, gid: ChunkGid) -> Result<()> {
        …
    }
```

- Paging is by unit: at most 200 a read; `start_at` takes an anchor and the
  header says how many there are in all.
- The label path is not hashed: the anchor is the unit's text alone, so moving a
  function into another `impl` does not change its anchor, and renaming it does.
- The engine records, per character and document, the anchors and their text as
  last shown to that character (§7). Nothing is persisted: it lives with the
  character's working set for the length of the mission.

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
  still applies — the file falls back to fixed windows until it parses again —
  and the reply says so, so the author knows its next edit should mend it.

### The grammar

The edit is decoded under the tool grammar like every act
(`engine::tools`):

- **`path`** is an enum of the documents the character holds anchors for, and
  each path **brings its own `at`** (the stencil's per-value `shapes`, as
  `invoke`'s `url` brings its `body`): after a path, `at`, `to` and `dest` are
  enums of that document's shown anchors only. Paths with equal anchor sets
  share one sub-tree.
- **`at`** is therefore an anchor the character was last shown for that
  document (§5). An anchor that does not exist cannot be emitted. Once the
  model has emitted enough of it to pick one out — usually its first token —
  the rest is forced.
- **`do`** is an enum of the five operations; **`text`** is free prose.
- **`old`** is free text checked by the act: the stencil has no substring
  constraint, so an `old` that does not occur once in the unit is refused, as
  `file_edit` refuses today.
- A document with no units (empty, or whitespace only) offers no anchors and so
  no `file_edit`; its first text is written with `file_write`.

### Staleness

An edit is applied only if the anchored unit's text is what the character was
last shown for that anchor (§5). If the document changed under it — another
edit, the engine's tidy pass, a merge from the record — the anchor either no
longer exists or names different text, and the act refuses: *"That paragraph
has changed since you read it; read it again."* No edit is ever applied to text
the character did not see.

The shown anchors live in memory, so a restart forgets them; the first edit of a
document after one is refused the same way — *read it again* — never as "no such
paragraph".

### The reply

Every edit answers with the units it touched and their new anchors, and the
unit before and after each for context:

```
Changed 9fTa → R2kp.
b4_K| Then the order came down on the shared frequency…
R2kp| You said fifteen.
cX0e| The tower lord's reply came back flat…
```

The engine adds those anchors to what the character has been shown, so its next
edit can name them without reading the document again.

## 7. Mending sittings

The mend steps (`fix`, and `review` when it sends a document back) stop
re-composing the piece. The mending sitting is put the faults and the piece as
`file_read` shows it — anchors and all — and answers with a list of edits in the
§6 form, under the same grammar, which the engine applies one by one through
`file_edit`. A sitting that would rewrite most of the piece says so and is
composed whole, as today.

`compose` keeps writing first drafts and genuine rewrites whole.

## 7a. Other places that cite lines

- **`file_grep`** labels each hit with the anchor and scope path of the unit it
  falls in, and records those anchors as shown, so a hit can be edited without
  a read.
- **Tool output that is not a read** — compiler errors, test failures, diffs,
  blame — keeps its own line numbers. It describes the world, not the editing
  surface; an author goes from it to a read, and edits from the read.
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

The anchor's forced remainder is decided without logits (`batched_sampler`'s
single-allowed-token path), but whether it is fed in one step or one decode
step per token is measured, not assumed (§10): at worst an anchor costs 2–4
decode steps, not 1–2.

## 9. Migration

There is no backward compatibility, by the repository's standing rule: the
anchored form **replaces** the line-numbered one, with no dual path and no flag
to keep the old one alive.

1. **The carver is restored into `zend-tools`.** `zend/src/code_read/carve.rs`,
   `types.rs` and `parsers/` come back from `46e247083^` as
   `zend-tools/src/tools/file/units/`, beside the file tools that use them,
   with their parser tests. The merge pass (`refine`'s forward merge) is not
   restored, and the size caps become the edit-unit caps of §3. The tree-sitter
   grammar dependencies move from `zend/Cargo.toml`, where nothing calls them
   now, to `zend-tools/Cargo.toml`; the single `tree-sitter` 0.26 pin `yamlpatch`
   needs stays workspace-wide. The Markdown paragraph rule and the YAML line
   rule are new code in the same module.
2. **`file_read` and `file_edit` change in place.** Line numbers and `old_str`
   anchors are removed from the reads and the act; every caller, prompt and
   example in the tool catalogue is moved to anchors in the same change.
3. **zend's `code_reading` layer is wiped and re-ingested.** Its turns were read
   with line numbers and cite them, so they describe a read format that no
   longer exists. It is tombstoned whole on the first boot of the new build —
   `--wipe-layer code_reading` (`DaemonConfig::wiped_layers`), on every member,
   once — and the background ingest rebuilds it from the repositories with
   anchored reads. Every other layer is untouched.
4. **npcd needs no migration of its own.** Anchors are computed from the text
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
- **Grammar**: the edit act's `at` enum is exactly the shown anchors; an empty
  set removes the act (`tools::stencil_params`' empty-set rule).

- **Ranges and moves**: `to` before `at` refused; a move into its own span
  refused; a moved unit keeps its anchor.
- **Grep**: a hit is labelled with its unit's anchor, and that anchor is then
  editable without a read.

Measurement, before and after landing:

- **Baseline first**, from the build before the change: decoded tokens and
  refusals per `fix` and `review` step on WhiteWorld, and zend's `file_edit`
  refusal rate.
- **After landing**, the same figures, edit refusals by cause, operation
  turnaround through the pipeline, and decode steps per emitted anchor (§8).
