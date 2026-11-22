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
  `gather_scope` is `shared` (zend's `code_reading` and `repo_map`; npcd's
  `world`), and every turn a section collection gathers (zend's
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

A turn an in-flight scan reads is never dropped; the next sweep takes it. The
scan that builds an index pins each turn; every scan that reuses one leases the
index as a whole, two atomics and no lock, and anything about to evict, move or
replace a turn counts the change first and then looks for a lease on an index
naming it (`gallery_arena/lease.rs`). A dropped turn costs nothing but its next upload, so a
conversation reloaded later pays one upload of its own turns on its first
scan.

Before this, a tombstoned turn's pages stayed until the keep trade found it
idle (hours, with no weight pressure never), and every dialogue turn of every
conversation the substrate held was preloaded at start: on BlackWorld
(2026-10-29) 60,285 zend dialogue turns, 1.07 GiB, for the handful of chats
actually open.

## 4. Only the weighted fold groups are resident

A signature carries three fold groups (layer regions of the model, 8 words a
token each), and every scan weighs them by its policy: `code_reading`'s
`scopes` content by `layer_weights: [1, 1, 1]` and its calibration questions by
`calibration_layer_weights: [1, 1, 1]`, its `guidance` and `journals` groups by
`[0, 1, 1]` and `[0, 0, 1]`, `repo_map` by `[0.1, 1, 0.1]`, the tool catalog
and dialogue by all three. A group weighed
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
  (AGENTS.md invariant 2b) — so the kernels read any packing. A group a turn
  does not hold, scanned only because another segment of the launch weighs it,
  points at one resident zero page; the segment's tally weighs it zero.
- **Readers weighing different groups share an index.** The index cache is
  keyed on the segments without their groups; a cached index serves a reader
  when it scans every group each segment weighs, and is rebuilt over the
  union when it does not. Its rows for a group a segment does not weigh read
  as no vote, so an index wider than its reader gives exactly that reader's
  scan. Keyed on the groups, a sealed turn read both by its conversation's
  dialogue (every group) and by a narrower layer was two indices and two
  launches on every scan.
- **Raw votes cover every group** (`TokenVotes`), a group a segment does not
  weigh voting for nothing in it. The chunk-exclusion learn scans the groups
  its competitions weigh, so learning never widens a turn the live scans keep
  narrow, and never cuts a chunk for an anomaly in a group no scan counts.

Measured on BlackWorld (2026-10-29): before, 5.49 GiB of gallery pages, of
which `code_reading` content held 4.29 GiB with a third of every page weighed
zero, and the calibration questions 1.11 GiB with two thirds weighed zero;
after, 3.37 GiB (`code_reading` 2.88, calibration 0.42). The kernel's share of
a belief scan fell from 2.25 to 2.12 ms.

Within the groups a turn holds, the dedup overlay keeps only the tokens its
readers ask for — every one, or the first of each code within the turn — and
a turn several read holds the union of their keeps, as it holds the union of
their groups (`docs/gallery_dedup.md`).

## 5. The scan index addresses pages, not tokens

A scan reads its gallery in place through an index — a descriptor table the
kernels walk (AGENTS.md invariant 2b) — built once per segment set and cached
while every turn it names keeps the pages it was built against. The index used
to carry two arrays **per scanned token**: `pos_map[j] = (page << 5) | slot`
and `case[j]`, 8 bytes a token. Calibrated, the `code_reading` scan covers 36
million tokens across 173,833 turns, so its index was ~315 MB; past the cache's
256 MiB it was never cached, and every reprojection rebuilt it — ~550 ms of a
36M-step walk and an upload — for 3.9 s of projection in a 5 s slot (BlackWorld,
2026-11-16).

Every token of a page belongs to one turn, so its case is the page's; and the
tokens a segment scans of a page are a subset of its 32 slots — all of them for
a whole page, a run at a window's edge, a scattered few under a narrower keep.
So the index carries one **entry** per (page, segment) instead:

| Array | Per | Holds |
|---|---|---|
| `page_ptr[page × G + g]` | page × group | group `g`'s strip in the page (as before) |
| `entry_page[e]` | entry | the page the entry scans |
| `entry_slots[e]` | entry | the page's slots it scans, one bit each |
| `entry_case[e]` | entry | the global case its votes go to |
| `seg_slot[s]` | segment | its first slot (`entries × 32`), prefix |
| `seg_tok[s]` | segment | its scanned tokens, prefix — the `n` its vote's mean and spread are over |
| `seg_case[s]` | segment | its first case, prefix (as before) |

A kernel's token space is `entries × 32` slots: slot `j` is slot `j & 31` of
entry `j >> 5`, scanned when its bit is set. A slot not scanned stages zero
words and the case sentinel, which the tensor kernels' dense-rank already skips
(`RANK_INVALID`), and the scalar kernel skips. A gap in a case's run gives that
case a second rank in the chunk, which is harmless: ranks flush to the global
per-case accumulators by case id (`atomicMax` / `atomicAdd`), so they merge.
The finalize reads `seg_tok`, the tokens actually scanned, not the slot range.

- **Size**: 12 bytes an entry plus `page_ptr` — ~41 MB for the 36M-token
  scan, against ~315 MB. An index is ~0.5% of the gallery pages it names.
- **Build**: still a walk of the scanned tokens, to test each against its
  segment's keep — but it now runs only when an index is first built or goes
  stale, not on every reprojection.
- **Cost**: the slots a partial page leaves unscanned are still staged — about
  half a page per turn window, ~8% more slots than tokens at today's ~207
  tokens a window. The kernel was never the scan's cost; the rebuild was.

**The cache evicts by idleness first, by size only as a backstop.** An index
goes when it is stale (a turn it names is evicted, moved or re-uploaded) or when
no scan has used it for a minute (`INDEX_IDLE_MS`, `gallery_arena/mod.rs`). Past
a forty-eighth of the card (~1.5 GB on the 72 GB card) the least recently used
go too. With indices a few bytes a page, that holds dozens of even the largest
scan's, so a live reader's index is evicted only when that many segment sets
are in use at once. The backstop is needed because the count of distinct
indices is bounded by how fast readers make new segment sets, not by the
gallery: a gate that left each probe's own turn out built 110 indices in two
minutes, kept them past the scans that used them, and ran the card out of
memory (BlackWorld, 2026-11-16).

## 6. Seeing it

`/v1/memory` → `state.gallery.rows` sums resident turns by status (live,
tombstoned, absent), holder (`shared`, `conversation`) and layer/group. After
a sweep, tombstoned and absent rows hold only pinned turns, and conversation
rows only the loaded conversations' turns.

## 7. One scan thread, and scans from elsewhere

The arena has exactly one scan thread, the scheduler's wave thread. A re-seal
frees a turn's old pages on the grounds that no other scan can hold a pin on
them, uploads and page moves are ordered on the device's stream, and a cached
index's arrays are read by the next scan that reuses it — none of which holds
with a second thread scanning on a stream of its own.

A belief scan, though, is mostly host work: each collection's gallery and each
turn group's files are assembled from the substrate, and what the launches
return is normalized and folded into the backgrounds. The seal scan — every
sealed turn's, the once-per-turn learning point — held the wave thread ~575 ms
a turn on BlackWorld (2026-11-16), all but ~50 ms of it that host work. So a
scan that runs elsewhere reaches the arena through `BeliefArena`, the five
launches a belief scan makes (`scan_weighted`, `scan_segment_weighted`,
`scan_keyed`, `scan_votes`, `scan_stats`): the arena implements it in place, and
`scheduler::RemoteArena` implements it by sending each launch to the wave
thread as an owned `HeldScan` (`SchedulerRequest::ArenaScan`) and waiting for
its answer. A held launch is the segments' turns as shared signature blobs — a
reference count each, not a copy — so the wave thread runs exactly the launch
the scan built. The launch's share of the arena split (index, kernel,
readback) comes back with its answer and is counted on the scanning thread, so
the wave thread's own scans are not credited with it.

The seal scan runs this way on the sealing conversation's thread
(`Sequence::last_turn_belief_scores`). A live reprojection scans on the wave
thread itself, with the arena in place.

The untagged collections — the journal, a conversation's memory — draw from
dialogue turns, which the substrate indexes as it indexes the tagged turns
(`substrate::tag_index`), so their gallery walks only those: walking every turn
to find them was ~440 ms of each assembly, paid by the seal scan and by the
first reprojection after every seal.
