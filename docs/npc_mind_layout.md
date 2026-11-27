# The mind's layout: six regions on one native window

**Status:** design. Written 2026-11-27. Builds on `docs/placed_section_layout.md`
(reserved ranges, holes, birth order) for npcd's characters — Zen and the
Drones on the moon, the cast of Battle Cities.

## 1. What a mind reads today

A mind's prompt is battle-mind's `projection.yaml`: one system prompt of
folder-backed collections, then the turns of the layers ranked below its
target. Every mind turn targets `interaction` (a conversation targets `talk`),
which sees `world`, `eras`, `stories`, `dreams` and `guides` and nothing else.
Zen's mind on WhiteWorld, 2026-11-27, read:

| Part | Tokens | What |
|---|---:|---|
| Persona | ≈ 5,400 | anchor 1,938, place 2,275, acting 477, stance 224, voice 161, identity 125, world 107, processing 77 |
| Tools | ≈ 4,860 | the overview 428 and every offered act's line (`budget: {min: 16, max: 16}`) |
| Guides (`methods`) | ≈ 860 | the two `guides` documents its opening scored highest (`top_k: 2`), of 32 (≈ 10K tokens) |
| Knowledge base (`canon`) | ≈ 2,050 | three `world` documents (`top_k: 6`, gated) |
| Journal | 14 | `journal_none`: the newest five entries when there are any |
| Turns | the rest | `recent: 16`, `historical_top_k: 8`, re-chosen by score at every boundary |

Everything is laid out by concatenation, so a region that grows moves every
region after it, and the dialogue's older turns come back by score — an
exchange the mind left behind can return under the reply that has moved past it.

## 2. The layout

Qwen3.6-35B-A3B reads 262,144 positions natively (`docs/performance.md`, the
trained-window table); no region needs YaRN. Six regions, each a reserved
range whose unfilled positions are a hole (`placed_section_layout.md` §3.1):

```
| frame | tools | guides | knowledge base | persona | journal | system_close | turns →
0       12K     44K      108K             124K      172K               ≈ 262K
```

| Region | Order | Range | Fills | Holds |
|---|---|---:|---|---|
| **Tools** | relevance | 12,288 | from the top, offered acts by score while they fit | the acts this turn offers; every act's line the room's own acts need |
| **Guides** | relevance | 32,768 | from the top, by score while they fit | the `guides` documents the personality is drawn |
| **Knowledge base** | relevance | 65,536 | from the top, strongest nearest the persona | Battle Cities: the story corpus (`world`, `eras`, `stories`). Zen: the journals repository |
| **Persona** | fixed | 16,384 | from the bottom; the hole after it is room to grow | voice, processing, anchor, identity, world, place, mission, unassigned, channels, catch-up, stance, acting |
| **Journal** | birth | 49,152 | from the bottom, as one chain; full, the oldest entry leaves | this conversation's own journal: every kept entry and what is left open |
| **Turns** | birth | the rest, ≈ 87K; window 73,728 | the newest whole exchanges; nothing shed in a request | `interaction` for the mind, `talk` for a conversation |

The ranges sum to 175,104 positions before the turns. The frame and the
templates before them take about 300, so the turns region is about 86,700
positions; its window — the past exchanges a request opens with — is 73,728,
and the remainder, about 13K, is what one request grows into before it
reaches past 262,144 and takes a rung (`placed_section_layout.md` §3.7).

### 2.1 Why this order

- **Shared before personal.** A sealed member's K/V is addressed by the prompt
  before it (`persistence/content_hash.rs`). The tools, the guides and the
  knowledge base stand behind the frame alone, which every character of a
  world shares, so each member is sealed once per world and every mind places
  the same K/V. Behind a persona they would be sealed once per character.
- **Personal nearest the question.** The persona, the journal and the turns
  are what make the turn this character's; the end of a long context is what
  is read best, and the gap the shared ranges leave no longer stands between
  the persona and the question (`placed_section_layout.md` §5, risk 1).
- **The journal last in the system prompt.** It is the mind's working state:
  what it was asked, what it found, what is still open.

### 2.2 The knowledge base and the journal

The two hold different things and are never filtered against each other.

- The **journal** is this conversation's own: the mind's journal for the mind
  (`docs/journal.md`), a conversation's for a conversation. It is placed by
  birth and holds every entry that fits its range, not the newest five.
- The **knowledge base** is everything else the character may know, by
  relevance. For Zen it is the whole journals repository — every
  conversation's journal, the people, the topics, the threads — so Zen knows
  what was said in every conversation without carrying their turns. The
  current conversation's entries may be placed there too; that costs a few
  positions and nothing else.
- For a Battle Cities character it is the story system: `world`, `eras` and
  `stories`, filtered by the world's tags as the layers are today.

### 2.3 What leaves the turns

The `world`, `eras`, `stories` and `guides` layers stop being turns: their
documents are members of the knowledge base and guides ranges. `dreams` stays
a layer — a dream is the character's own and is recalled as a turn, labelled.
`perception`, `action`, `memory`, `environment` and `relationships` are not
visible to `interaction` today and are unchanged.

## 3. Mechanics

| Region | Built from | Change |
|---|---|---|
| Ranges, holes, fill by tokens | `ReservedRange`, `emit_in_range`, `range_fill::fit_by_tokens` | `range:` on each collection in `projection.yaml` |
| Persona as one range | the persona's collections and sections | a `span` item: the items between its open and close take one reserved range, the hole after them |
| Tools by tokens | the `tools` collection's belief selection | `range: {tokens: 12288}`; `budget.max` no longer binds (`collection.rs`) |
| Guides in the system prompt | the `guides` layer's documents | a `guides` collection, one member per document, scored by provenance against the turn's opening (`question_pin`) as the tools are |
| Knowledge base | zend's `files` collection: the files set, the pool, the fill (`files/set.rs`, `files/fill.rs`) | npcd's corpus as a files set: the story layers' documents for a world that names them, the journals repository for Zen |
| Journal by birth | the `journal` collection, `RangeFill::Bottom`'s window (`emit.rs`) | every kept entry submitted as a chained section, not the newest five |
| Turns by birth | `dialogue_fill::window_from` | a `birth` selection for a conversation group: the current exchange and the newest past exchanges whole while they fit the layer's window |

Qwen3.6 has no QSA index (`quantized_qwen36_moe.rs`), so nothing here is
decided per decode step: the relevance ranges are filled from provenance at
boundaries, and a member entering mid-request enters in front
(`placed_section_layout.md` §3.8).

## 4. Card memory

Positions in a hole cost nothing; a filled position costs its K/V. Per world,
the shared ranges (tools, guides, knowledge base) are sealed once — about
110K positions at most, held where the tier puts them, placed hot. Per mind,
the persona, journal and turns fill as the character lives: about 5K at
first, up to 154K once its journal and window are full. The cast a card can
hold is bounded by the KV regions (`docs/expert_cache_design.md`,
`region_pool`); a card that cannot hold every mind's full journal and window
parks the minds that are not turning, as it does today.

## 5. Build

1. `span` and the persona range; tools by tokens; the journal's range and
   every entry; the turns' `birth` selection; the `guides` collection.
2. The knowledge base: npcd's files set over the story layers and the
   journals repository.
3. The substrate: every member sealed under the new prompt and addresses, so
   npcd's substrate on WhiteWorld is wiped and its corpus re-ingested once
   the build is green.

Each building block is unit-tested against exact values: the positions of
every region for a persona of a given size, the hole after it, the tools a
12K range holds, the exchanges a window holds, the journal's window dropping
its oldest entry.
