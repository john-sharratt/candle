# Anchored document edits in npcd

How a character changes one part of a document without writing the rest of it
again: every paragraph carries a short content anchor, `file_read` shows it, and
`file_edit` names the paragraph by its anchor and says what to do there.

Status: design. Replaces `file_edit`'s `old_str`/`new_str` form and the
whole-piece mending sitting of `compose`; `file_write` and `compose` for a first
draft are unchanged.

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

- **`.md` documents: the paragraph** — a run of non-blank lines between blank
  lines. A heading, a list, a block quote or a scene break is its own unit.
  Blank lines carry no anchor; the edit act restores the separating blank line
  on insert and delete.
- **`.yaml` documents: the line.** Character sheets, world files and
  `missions.yaml` are line-structured, and a paragraph of YAML is not a unit
  anybody edits.

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

| anchor | space | collision, 60 paragraphs | collision, 300 lines | stale anchor matching new text |
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

- Paging is by unit: at most 200 a read; `start_at` takes an anchor and the
  header says how many there are in all.
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
```

- **`replace`** without `old` replaces the whole unit with `text`. With `old`,
  it replaces that substring of the unit, which must appear exactly once **within
  the unit** — so `old` can be just the words being changed, not their
  neighbours.
- **`insert_before` / `insert_after`** add `text` as a new unit beside the
  anchored one.
- **`delete`** removes the unit; `text` is not given.
- One operation per call. Several calls from one read stay valid, because an
  edit changes only its own unit's anchor (and the anchor of any later unit
  whose seed was bumped by it — a copy of the same text).

### The grammar

The edit is decoded under the tool grammar like every act
(`engine::tools`):

- **`at`** is an enum of the anchors the character was last shown for that
  document (§5). An anchor that does not exist cannot be emitted. Once the
  model has emitted enough of it to pick one out — usually its first token —
  the rest is forced.
- **`do`** is an enum of the four operations; **`text`** is free prose; **`old`**
  is bound to the substrings of the anchored unit where the stencil can express
  that, and otherwise checked by the act (an `old` that does not occur once in
  the unit is refused, as `file_edit` refuses today).

### Staleness

An edit is applied only if the anchored unit's text is what the character was
last shown for that anchor (§5). If the document changed under it — another
edit, the engine's tidy pass, a merge from the record — the anchor either no
longer exists or names different text, and the act refuses: *"That paragraph
has changed since you read it; read it again."* No edit is ever applied to text
the character did not see.

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

## 8. Expected cost

| change | today | anchored |
|---|---|---|
| one word in one sentence | ~800–1,300 decoded (`compose`), or the `old_str` copy plus context | ~15–25 (`at` + `old` + `text`) |
| one paragraph rewritten | ~800–1,300 | ~60–110 (the paragraph) |
| a reread of a 60-paragraph story | 60 line numbers | + ~240 prefill tokens for the anchors |

Prefill is the cheap side (hundreds of tok/s on the same machine), so the read
cost is negligible beside the decode it saves.

## 9. Tests

Unit tests beside the code, asserting raw values:

- **Anchors**: fixed expected strings for known paragraphs; trailing-whitespace
  and line-ending normalisation; two identical paragraphs taking seeds 0 and 1;
  a constructed collision between different texts resolved by a seed bump, the
  anchor still four characters.
- **Edits**: each operation on a small document, the restored blank lines, the
  reply's new anchors; a scoped `old` that occurs twice in the unit is refused,
  once is applied.
- **Staleness**: an anchor read, then the unit changed by another hand → refused;
  a unit elsewhere changed → the edit still applies.
- **Duplicates**: deleting the first of two identical paragraphs leaves the
  second's old anchor refused, never applied to the wrong copy.
- **Grammar**: the edit act's `at` enum is exactly the shown anchors; an empty
  set removes the act (`tools::stencil_params`' empty-set rule).

Live measurement on WhiteWorld after landing: decoded tokens per `fix` and
`review` step, edit refusals by cause, and operation turnaround through the
pipeline, against the same figures for whole-piece mending.
