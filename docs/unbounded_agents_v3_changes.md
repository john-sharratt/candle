# Unbounded agents v3: what changed since v2

**Status:** working record for v3, 2026-11-28. v2 (`docs/unbounded_agents.md`,
published 2026-11-25) stays as published. This document lists every change to
the system since then that v3 must describe: the evidence for each change, the
v2 section it revises, and what the final battery must measure before v3 can
report it. Design documents remain authoritative for the mechanisms. This
record points at them rather than repeating them.

## 1. In brief

v2's §9.12 found that the memory **placed** the right files but the model did
not **read** them. C1 named 69.9% of the expected items in the text of an
answer file it had placed. From the same text handed over as a files block,
BM25's arm named 97.6%. v2 reported C1 at 37.5% mean recall on the 128
questions.

Since then, five changes have been built:

1. **The knowledge base is part of the system prompt.** Provenance places
   files there, in the format the model's own agent framework uses (§2.1).
2. **Every item of the prompt has a fixed position**, including reserved
   ranges with holes. What follows a range never moves (§2.2).
3. **Each file is placed as sections.** Selection, placement and calibration
   all work per section (§2.3).
4. **The range is re-ordered at every decode step.** Order follows the
   index's attention, and a strongly read section is locked in place (§2.4,
   §2.5).
5. **Each section is sealed behind context that says where it belongs.**
   That context is its repository's root listing and its own folder's
   listing. Each collection also has a record of its own kind that is never
   shown, so no section is sealed as the only one of its kind (§2.6).

Two correctness fixes came with these changes: holes are pieces of the layout
(§2.7), and compaction catches up an outstanding gap before it relocates
anything (§2.8).

The two 24-question batteries put the change in scale:

- **Before (reading turns):** the pilot battery scored 46.2% mean recall on
  production on 2026-11-18. It rose to 62.2% by 2026-11-20 as retrieval
  changed (v2 §9.12).
- **After (system prompt):** the depth-1 battery, over the repositories' root
  files, first measured 74.3%. Production C1 now reaches **84.7% mean
  recall** (95% interval 72.9–95.1%, two runs).

The system prompt itself was the largest single step (§2.1, §3). The two
batteries ask different questions at different ingest depths, and neither is
the paper's. v3's numbers must come from the 128-question battery, rerun on a
full-depth ingest (§5).

## 2. The changes

### 2.1 The knowledge base moves into the system prompt

**What.** v2 placed code readings as sealed *turns*: the model's own earlier
`file_read` calls and results. They now go into one dynamic collection,
`files`, at the end of the system prompt. Each file is rendered as Qwen-Agent
renders retrieved knowledge: `# Knowledge Base`, then `## The content from
<path>:` and the file in a fence. A reading is no longer a conversation.
Ingest prefills each file's sections as members of the collection and asks
the file's calibration questions; nothing else about a file is decoded. The
seed (the repositories' root listings and anchor documents) is the
collection's mandatory set of files.

**Why (measured 2026-11-26, build 5602, BlackWorld).** In C1's seventh run,
seven questions had every expected item inside an answer file the memory had
placed, whole. The model named none of them. Its thinking read the slot as
one confused dialogue ("then I wrote a response about docs-write.md?? That's
odd"). Re-prefilling every piece fresh did not change that, so the cause was
how the files were presented, not the K/V or the recurrence.

The same selection on the same seven questions, presented four ways:

| Presentation | In-view items named |
|---|---|
| Reading turns (production, v2) | 0 / 16 |
| End of the system prompt, `=== path ===` blocks | 15 / 16 |
| End of the system prompt, Qwen-Agent knowledge format | 16 / 16 |

Data: `docs/results/placement_probe_2026-11-26/`. Design:
`docs/provenance_system_collection.md`. Built in 6608781bb–109cf0a06.

**Revises:** v2 §3 (architecture), §6.5 (what reprojection places and where),
§9.12 (the P1 baseline and the "in view" table, both defined over reading
turns).

### 2.2 Placed-section layout: every item has a fixed position

**What.** Every item of the system prompt takes a fixed slot. A section is
padded to its slot, and a collection reserves a range whose unused part is a
**hole**. A hole is a run of positions that holds no K/V and is skipped by
RoPE. In the K/V it is recorded as the lead of the chunk after it, so the
position frontier and the token count are different numbers. The index places
its pages past holes exactly as the K/V does.

A member is sealed at the top of its range, behind the hole above it, so its
address names the hole too. The collections are:

- `tools`: 8K positions.
- `journal`: 8K positions, filled from the bottom.
- `files`: 112K positions. The seed in `file_read`'s format must take at most
  half of it.

Whatever is placed or evicted, everything after a range stands where it
stood. The decode therefore never re-reads a moved prompt.

Design: `docs/placed_section_layout.md` §3.1–§3.1c, §4.1. Built in
aa250f2d6, 5de1dbccb, 5b5e5eec5, 916d9ba10, ad01a5517 and cc0a74a66.

**Revises:** v2 §7 (the slot's layout and the three tiers), §5 (sections and
the quantization policy they are sealed under).

### 2.3 A file is sections, sealed as a chain

**What.** A file is cut where `file_read` pages it. Each section carries its
own heading, its range read, and the cut marker. Section *k* is sealed after
sections 1…*k*−1, its recurrence continued from where section *k*−1 ended,
so its K/V is the continuation of its file.

Selection and placement work per section: a long file contributes only the
sections that are read. Calibration also works per section, with questions
asked under each part's heading. The ingest key moved to version 8.

Design: `docs/placed_section_layout.md` §3.5b–§3.5d. Built in fe9848f63,
2ae9fc332, 2530414da and fcd447c01.

**Revises:** v2 §6.2–§6.6 (the unit the signature, the gallery and the
selection work on).

### 2.4 The range is ordered by the index, every step

**What.** The files collection is placed by its scope:

- **First step.** The question's first reading is the range's first step, so
  the opening already sits in the order the decode will read it in.
- **Every decode step after that:**
  - The step rates the sections it holds by their hit rate in the QSA index,
    as a 0–1000 band.
  - It evicts a section that the selection has stopped reading.
  - It admits candidates in front.
  - It reorders the range by momentum.

The decision is made per step, on the host, from what the device publishes.
No kernel changed.

Design: `docs/placed_section_layout.md` §3.2, §3.4, §4.4. Built in
51f25d7ac, 6f8130677, 0cf633e80, 6868add8f, 776290b60 and 8451d6c25.

**Revises:** v2 §6.5 (reprojection becomes a per-step re-layout of one range,
not a re-selection of turns).

### 2.5 Strong attention locks a section's position; normalized once

**What.** A section the index reads strongly keeps its **position**, not
merely its rank. The range may fragment around it while it is locked, and it
defragments when the lock is released. A gap that a departing section leaves
can be filled by a newly hot section, which then locks in turn. A locked
section keeps the relative RoPE distance to what attends it.

To be tuned once rather than per kind, the lock reads the range's attention
**normalized to 1**: each placed section's share of that step's attention.
Admission into the lock and release from it work as follows:

| Rule | Value | Meaning |
|---|---|---|
| Nucleus to lock | `P_LOCK` 0.5 | The smallest set of sections holding half the attention locks |
| Nucleus to stay | `P_UNLOCK` 0.7 | A locked section stays while it is within the 0.7 nucleus |
| Share kept | `KEEP_OF_LOCKED` 0.5 | ...and while it holds at least half the share it locked at |
| Reorder margin | `REORDER_SHARE` 0.05 | Unlocked sections swap only past a 5% share difference |
| Eviction | `EVICT_OF_EVEN` 0.25 | A placed section under a quarter of an even share (`1/n`) counts a step toward leaving; a candidate enters only at a rate that would hold the weakest locked share |
| Engagement | — | Nothing locks unless the mean placed rate is at least the step's chance rate |

Code: `candle-conversation/src/files/scope/nucleus.rs` and `range_scope.rs`
(`reorder_locked`). Built in 5aa6b4941, a1ee94d4b, 8555b5473 and b38b1272a.

**Normalization of small kinds.** A file's score is read against the spread
of its own extension's scores (`kind_z`). A kind with fewer than eight
samples has no usable spread of its own, so it is read against every kind
pooled. Built in 4cde1fc5e.

**Revises:** v2 §6.4 (normalization), §6.5.

### 2.6 A section is sealed behind context that says where it belongs

**The failure.** Two files of the same shape, such as two `Cargo.toml`s, were
each sealed behind the prompt alone, as if no other existed. On BlackWorld a
question about candle's `serde` version was answered with yenbot's whenever
yenbot's `Cargo.toml` sat nearer the question, and with candle's when
candle's did. In eight orderings logged for k02, candle's `Cargo.toml#1`
ranked around 9th–11th in the range, with yenbot's always above it. A fresh
prefill of the same prompt (same tokens, same recurrent state, same C4
quantization) found candle's wherever it sat. So the error lies in what a
separately sealed section lacks, not in quantization or the recurrence.

Interpretability work on how models bind entities finds the bindings carried
by an ordering, *the first, the second*. A separately computed section has no
way to take part in that ordering (Feng & Steinhardt, ICLR 2024; Dai et al.,
EMNLP 2024).

**What.**

- **Listings.** Each file is sealed behind its repository's root listing,
  then the listing of the folder it is in. Each section is computed as part
  of that repository and that folder: its bearings. The listings' K/V is the
  copy the knowledge base already holds, reused as a seed chain reuses its
  sections, so the lead costs no prefill.
- **A record per collection.** Every collection of the system prompt declares
  a `lead:` in `projection.yaml`: a record of its own kind that is never
  placed. These are a signature for a tool that does not exist, a journal
  entry 0, and a file in a repository that is not in the workspace. The
  record is sealed once and every member is sealed behind it. It is never
  selected, offered or emitted, so no tool can be called that is not there.
  npcd's schema declares the same records for its twelve filled collections.

Both kinds of lead are part of every member's address, so the substrate is
re-sealed when they change.

Design: `docs/placed_section_layout.md` §3.5e, §3.5f. Built in 10712fbe8 and
0ba161ecc.

**Revises:** v2 §9.12 (a new failure analysis), §11.4 (attention as
retrieval: what a retrieved section needs to be read as retrieved).

### 2.7 A hole is a piece of the layout

**The failure (found 2026-11-28).** With positions locked, the range
fragments. A departing section leaves a gap, and a newly hot section may
later fill it. The rebuild planner aligned holes as *forwarded* pieces:
pieces kept only while everything before them is unchanged, and cut at where
they change. A hole holds no chunk of its own; in the K/V it is the lead of
the chunk after it. So when a rebuild removed a hole, it spliced an empty
range in front of that chunk and the old gap stayed in the K/V.

From that point the slot's K/V stood 2,046 positions past its index. The QSA
select refused the query ("needs 4226 blocks but the index cache holds
3715"), which failed a whole wave of calibration decodes.

**What.** A hole holds no K/V and is computed against nothing, so it is now
named by its size alone. The planner aligns it as it aligns a sealed piece:
a rebuild keeps, puts in, removes, moves or resizes it where it stands. Once
the edits stand, the slot's leads are set from the placed records
(`projection_assembler::apply::lay_leads`, on
`set_sequence_position_leads`). Each block-holding piece's first chunk takes
the holes since the block before it, and the holes after the last block are
left pending at the tail. The K/V's holes can no longer disagree with the
layout. A forwarded piece after a hole still reads it, through its chained
identity.

Tests: `candle-conversation/src/scheduler/tests/splice_rebuild.rs`. They
cover a gap the next layout fills, a gap that opens, a hole moved, and a hole
resized. Each rebuild must be the same slot, at the same position, as a fresh
build of its result.

Built in 14a820d2a, 45e621a5d (an interim cut) and d37aed913 (the structural
fix).

**Revises:** v2 §7 (the slot as a layout of pieces, holes included).

### 2.8 Smaller changes

- **Compaction and the gap quantize.** A compaction pass catches up an
  outstanding gap quantize before it relocates anything. It refuses while
  one is outstanding, and deferred blocks hold no band (1e0291adf,
  dc4481dbf; `docs/kv_quantize_in_the_gap.md`).
- **Ingest by name.** Build files with no extension (Makefile, Dockerfile,
  Justfile, Procfile, Rakefile, Gemfile) are read, under the language their
  name implies (4cde1fc5e).
- **Collections fill by tokens.** A ranged collection is filled by tokens,
  not by a count (e1a95f5de). The tools range takes signatures by score until
  the next one does not fit.
- **Sealed-fresh diagnostic.** `POST /v1/substrate/zend/sealed-fresh` decodes
  one question twice over the same tokens: once over sealed K/V, once
  prefilled fresh. It reports where the two diverge (20e8a4bc4 onward). Its
  transplant modes do not yet handle holes.

## 3. Evidence so far

### 3.1 Before the move: the knowledge base as reading turns

v2's own measurements of the memory, before §2.1:

| Measurement | Questions | Mean recall | Source |
|---|---:|---:|---|
| Pilot battery on production, 2026-11-18 | 24 | 46.2% | `docs/results/dependency_battery_gap_analysis_2026-11-18.md` |
| Pilot as retrieval changed, by 2026-11-20 | 24 | 57.6% → 62.2% | v2 §9.12, `docs/results/dependency_battery_2026-11-20_b.md` |
| Pilot, reprojecting in thinking against held, 2026-11-23 | 24 | 57.3% against 38.8% | v2 §9.12, `docs/results/dependency_battery_2026-11-23_l_m.md` |
| The 128-question battery, C1 | 128 | 37.5% | v2 §9.12, `docs/results/battery_128/` |

The 18 November note found where the pilot lost its points. Of 54 answer
files, 52% were placed, 22% could never be placed (over the 48K file cap),
and 26% were not drawn. The 128-question battery then showed the model
failing to read even the files it was given: 69.9% of in-view items named,
against 97.6% from text (§2.1).

### 3.2 After the move: the depth-1 battery

The two batteries are not the same questions. The pilot is 24 hand-picked
questions over the whole fork. The depth-1 battery is 24 questions over the
root files. The step from 46.2% to the 70s is the move's scale, not a
controlled contrast: its controlled measure is the 0/16 → 16/16 probe of
§2.1, and in v3 it must be the 128-question battery rerun (§5).

`zend/examples/dependency_battery/battery_depth1.yaml` holds 24 questions
(k01–k24) over what an ingest at `max_depth: 1` holds: the root files of
candle, battle-cities, yenbot and journals. The arm is C1, the production
dials. Each figure is mean recall, measured on BlackWorld on 2026-11-28.

| Configuration | Mean recall |
|---|---:|
| *(before the move: the pilot battery, reading turns, 2026-11-18)* | *46.2%* |
| Order by momentum, no lock | 74.3% |
| Order-lock bar 50 | 74.3% |
| Order-lock bar 150 | 70.1% |
| Positional lock, fixed bar | 70.1% |
| Lift over the step's selection | 69.1% |
| Nucleus of shares, first tuning | 76.4% |
| Nucleus of shares, second tuning | 75.3% |
| + build files ingested, small kinds pooled | 75.3% |
| **+ listings and collection records, holes as pieces** | **84.7%** |

The last row has its full data in
`docs/results/battery_depth1/2026-11-28_leads/` (`summary.md` and every
figure's CSV): 48 answers, two runs (86.8% and 82.6%), 95% bootstrap
interval 72.9–95.1%, 18 of 24 questions at full marks. During that run the
memory held 1.26× compression on average and 1.61× at peak, with 16 sessions
in flight.

**Caveats v3 must keep.**

- The earlier rows' answers were not kept, and their run counts were not
  recorded with them; only the last row is reproducible from data.
- Before 330446e88, k14 expected a stale toolchain value. k11's file (a
  Makefile) was not ingested before 4cde1fc5e.
- The last row changes three things at once: a fresh substrate re-sealed
  under the leads, the listings and collection records, and the hole fix.
  It is not an ablation of any one of them.
- Depth 1 is the easy end: the root files only, about 22 files. It tests
  placement and reading over look-alike root files, not multi-hop retrieval
  through 2.2M lines.

## 4. What is still open

- **Look-alike files.** k02 (candle's `serde` version) was right in one run
  of two. The other answered 1.0.228, which is yenbot's version. k15
  (battle-cities' Rust edition) was wrong in both runs: it answered 2021, but
  battle-cities' `Cargo.toml` declares 2024. The leads halved this failure
  but did not remove it.
- **Partial answers.** k03 (33%) and k06, k16 and k17 (50%) are partial in
  both runs. They have not yet been examined.
- **A device fault during calibration.** Twice (05:34 and 05:46 UTC on
  2026-11-28), a speculative-draft decode step over calibration sequences
  ended in `CUDA_ERROR_ILLEGAL_ADDRESS`: "pinned staging: fence record
  failed", then "decode draft failed". The first followed the hole mismatch;
  the second had no mismatch before it. The battery ran clean on the hole
  fix, but calibration decodes have not run on it since. If the fault
  recurs, it gets the `tensor-assert` harness (`AGENTS.md`).

## 5. Before the final battery

1. **Full-depth ingest, settled.** BlackWorld re-ingests at `max_depth: 5`
   (from 2026-11-28; 896 folders queued). WhiteWorld re-ingests npcd on the
   same build. The battery starts only once the member reports its memory
   settled: every file ingested, calibrated and normalized, as v2's P1 run
   required.
2. **The analysis must learn placed sections.** `dependency_battery analyse`
   still looks for answer files as reading turns. On the depth-1 run it
   reports "answer file placed" and "in an answer file's text" as 0% while
   recall is 84.7%. Before the final run, the placement and in-view tables
   must read the placed sections from the projection points (which already
   record the knowledge base's files, 7bd8a8956).
3. **Arms and baselines, redefined where the change made them so.**
   - *Seed on/off:* the seed is now the collection's mandatory files.
   - *P1:* it took the files of the reading turns C1 placed. It becomes the
     files of the sections C1 placed.
   - *R1 and T1:* rerun on the same build, so every contrast is measured
     against one system.
   - *The 128 questions:* unchanged, still kept out of the memory by
     `zend.ignore`.
4. **One build, recorded.** As in v2 §9.12, the build every answer ran at,
   and confirmation that no retrieval code changed during the run.

## 6. The v2 sections v3 revises

| v2 section | Change |
|---|---|
| §1.2 Contributions | The knowledge base as placed sections; positional lock; sealed context (leads) |
| §3 Architecture | The system prompt's layout: fixed slots, ranges, the `files` collection |
| §5 Two-phase quantization | Sections sealed by collection level; boundary vs member policy |
| §6.2–§6.6 Provenance | The section as the unit; per-step re-layout; nucleus lock; `kind_z` pooling |
| §7 Three-tier context | Holes in position space; a hole as a piece; leads laid from the layout |
| §9.12 Dependency analysis | New results; the failure analysis of reading placed turns (§2.1) and of look-alike sections (§2.6); P1 redefined |
| §11.4 Attention was always retrieval | What a retrieved section needs to be read as retrieved: its bearings |

## 7. Commits

Every commit since v2 that touches the retrieval path, oldest last, is listed
by `git log --since=2026-11-25 --no-merges -- candle-conversation candle-nn
candle-transformers zend`. The ones this record cites are, in order:

- **2026-11-26:** 6608781bb–109cf0a06 (the knowledge base in the system
  prompt).
- **2026-11-27 to 28:** aa250f2d6–cc0a74a66 (the placed-section layout and
  holes).
- **2026-11-28, ordering and the lock:** 51f25d7ac, 0cf633e80, 776290b60,
  5aa6b4941, a1ee94d4b, 8555b5473, b38b1272a.
- **2026-11-28, ingest and normalization:** 4cde1fc5e.
- **2026-11-28, compaction:** 1e0291adf, dc4481dbf.
- **2026-11-28, leads:** 10712fbe8, 0ba161ecc.
- **2026-11-28, holes:** 14a820d2a, d37aed913.
