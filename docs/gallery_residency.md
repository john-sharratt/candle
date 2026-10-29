# Gallery residency: shared turns and conversation turns

The provenance gallery (`candle-conversation/src/provenance/gallery_arena/`)
keeps each turn's folded signatures resident in VRAM so a belief scan reads
them in place. This document says **which turns may be resident**, and when
each kind goes. How a page is laid out, uploaded and scanned is in
`docs/archived/paged_gallery_arena.md`; the keep trade (when an idle turn is
worth less than the weight ground it holds) is in `gallery_arena/eviction.rs`.

## 1. Two kinds of resident turn

A turn's signatures are worth holding only while a scan can read them, and
the schema already says who can read what:

- **Shared turns** — read across conversations. Every turn of a layer whose
  `gather_scope` is `shared` (zend's `code_reading`, `repo_map`, the review
  layers; npcd's `world`), and every turn a section collection gathers (zend's
  `tools` calibration turns; npcd's `response` and `mood` votes, which are
  drawn from every character's turns). Any conversation's next projection may
  scan them, so they are preloaded at start and stay until the keep trade
  gives them up.
- **Conversation turns** — read only by their own conversation. A belief scan
  masked to its target's own timeline (the target's own group, or every group
  of an append-only target) reads nothing else. zend's `dialogue` layer is the
  case that matters: its scans are masked to the chat being answered, so one
  chat's turns are never read by another. Such turns are resident only while
  their conversation is **loaded** — while a scheduler slot targets its
  timeline — and go when the last such slot is freed.

A layer is declared one or the other with `gather_scope:` in the schema
(`shared` is the default). zend's `dialogue` and every npcd layer but `world`
declare `conversation`.

## 2. The holder is set by the reader

The schema decides the preload; the **scan** decides what a resident turn is,
because the same turn can be read both ways — an npcd character's exchange is
its own conversation's history and also a vote in the `response` collection
every character scans. Each resident turn carries a holder:

| A read by | Holder it gives |
|---|---|
| a scan (or a normalization warm-up) masked to one timeline `T`, of a `conversation`-scoped layer | `Conversation(T)` |
| the same, of a `shared` layer | `Shared` — every conversation will read it next |
| any group scan across conversations, a collection scan, a preload | `Shared` |

A shared layer's turn read by its own conversation alone (a `code_reading`
unit scoring itself while it is written) is still held shared: dropping it
when the unit's slot frees would only make the next dialogue scan upload it
again.

Holders only widen: a `Conversation(T)` turn that a shared reader touches
becomes `Shared`, and one two different conversations read becomes `Shared`.
A turn re-uploaded under a new fingerprint (a re-seal) keeps the wider of its
old and new holder. Nothing narrows a holder; a turn that stops being read
shared is given up by the keep trade like any idle shared turn.

A cached scan index is keyed by its segments, holder included, so an index
built for one reader is never reused to stand in for another's.

## 3. What goes, and when

Between forwards, beside the keep trade (`Scheduler::sweep_gallery`), the
gallery gives up every unpinned resident turn that **no scan can read**:

1. **A conversation turn whose conversation is not loaded** — its holder is
   `Conversation(T)` and no live slot targets `T`. Checked when a slot has been
   freed since the last sweep.
2. **A turn of a tombstoned timeline, or a tombstoned turn** — every scan skips
   them, so their pages are dead weight. Checked when a substrate's
   tombstone count (`Substrate::tombstone_epoch`) has moved, which covers
   tombstones written here and tombstones replicated from another member.
3. **A turn no attached substrate declares** — its substrate was detached.
   Only when every substrate could be read: a substrate whose lock a writer
   holds is skipped, and its turns are not judged absent.

A pinned turn (an in-flight scan reads its pages) is never dropped; the next
sweep takes it. A dropped turn costs nothing but its next upload, so a
conversation reloaded later pays one upload of its own turns on its first
scan.

Before this, a tombstoned turn's pages stayed until the keep trade found it
idle (hours, with no weight pressure never), and every dialogue turn of every
conversation the substrate held was preloaded at start: on BlackWorld
(2026-10-29) 60,285 zend dialogue turns, 1.07 GiB, for the handful of chats
actually open.

## 4. Only the weighted fold groups are resident

A signature carries three fold groups (layer regions of the model, 8 words a
token each), and every scan weighs them by its policy: `code_reading`'s and the
review layers' content by `layer_weights: [0, 1, 1]`, their calibration
questions by `calibration_layer_weights: [0, 0, 1]`, `repo_map` by
`[0.1, 1, 0.1]`, the tool catalog and dialogue by all three. A group weighed
zero adds exactly nothing — the needle gate and the tally are both linear in
the weights (`needle_tally_segments`) — so its words are neither uploaded nor
scanned.

- **A resident turn holds a group set** — the groups its readers weigh, packed
  in a page of `|set| × 32 × 8` words, group-major as before. A read that needs
  a group the turn lacks widens it (re-upload with the union), as holders
  widen; a read that needs fewer is served from the wider page.
- **A segment declares the groups it weighs** (`PagedSegment::groups`, from the
  weights its scan applies), and a launch scans the union of its segments'
  groups only: its probe is narrowed to them and its tally weighs them.
- **The index addresses a (page, group), not a page.** `page_ptr[page × G + g]`
  is the address of launch-group `g`'s words in that page — a descriptor table
  (CLAUDE.md invariant 2b) — so the kernels read any packing. A group a turn
  does not hold, scanned only because another segment of the launch weighs it,
  points at one resident zero page; the segment's tally weighs it zero.
- **Raw votes carry their groups** (`TokenVotes::groups`): the chunk-exclusion
  learn scans the groups its competitions weigh, so learning never widens a
  turn the live scans keep narrow, and never cuts a chunk for an anomaly in a
  group no scan counts.

Measured before this, on BlackWorld (2026-10-29): 5.5 GiB of gallery pages, of
which `code_reading` content held 4.3 GiB with a third of every page weighed
zero, and the calibration questions 1.09 GiB with two thirds weighed zero.

## 5. Seeing it

`/v1/memory` → `state.gallery.rows` sums resident turns by status (live,
tombstoned, absent), holder (`shared`, `conversation`) and layer/group. After
a sweep, tombstoned and absent rows hold only pinned turns, and conversation
rows only the loaded conversations' turns.
