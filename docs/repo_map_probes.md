# Folder Probes — a generated retrieval surface for `repo_map`

**Status:** built and running. Code in `zend/src/repo_scan/probe/` (pure) and
`zend/src/repo_scan/probe_pass.rs` (engine-side). Measured results in §7.

---

## 1. The problem

A folder is not retrievable because of what it *is*. It is retrievable because
of the questions it is the answer to.

The `repo_map` layer stores, per directory, a decoded two-sentence summary of
what that folder is for. That summary describes the folder to a reader who has
**already found it**. Nothing about it is shaped like the query that should have
found it, and a BDP scan comparing `sign(Q_decode)` against stored `K` is asked
to bridge that genre gap on every lookup — a question and a declarative summary
differ in register, length and syntax.

The consequence is visible in the schema. `zend/src/prompts/projection.yaml`
carries, for this one layer, `layer_weights: [0.1, 1.0, 0.1]`, `beta: 0.65`,
`k: 3` justified as "root + 2 content", a `score_threshold` zero-cut to stop
TopK's tie-break seeding arbitrary leaves, and per-level normalisation floors.
That is a great deal of tuning holding up a weak surface.

The fix is the one that took tool selection to 100% top-3 / 97% top-1: stop
matching a query against a **definition** and match it against **examples of
queries that should land here**. For tools those examples are per-tool question
exemplars, distilled provenance-only. For folders they are **probes**.

## 2. Two artifacts per directory

| artifact | job | conversation | lifetime |
|---|---|---|---|
| **summary** | the payload injected when the folder is retrieved | `kind=repo_map`, `dir=<path>` | durable |
| **probes** | question-shaped turns whose only job is to be hit by a scan | `kind=repo_map_probe`, `probe_dir=<path>` | durable, evicted to cold |

Both carry the directory as metadata, and **the tag is the join**: a scan hitting
a probe resolves to the same folder as one hitting its summary. There is no
second index to maintain and no embedding-space mismatch — the index lives in the
same KV substrate as the content and is matched by the same attention.

The two conversations are labelled identically (the bare directory) so a
projection tile names the folder either way; `kind` is what distinguishes them.

## 3. The four registers

Real queries arrive in distinct shapes, and a generator asked simply for
"questions" returns twelve of one. The budget is spent explicitly — 6 per
register, 24 per directory:

| register | shape | rarity gate | dedup scope |
|---|---|---|---|
| `locational` | where is X / which file handles Y | required | within folder |
| `mechanistic` | how does Z work / what happens when | required | within folder |
| `conceptual` | what is a `ChunkGid` / what does BDP mean | required | within folder |
| `systemic` | subject-bearing, vocabulary-free | **exempt** | **global** |

### Why the fourth register exists

A specificity-only design indexes each folder for people who already know its
vocabulary — and those people do not need a repo map. The people who do cannot
name anything yet, and their questions are the ones the layer must answer. The
opening query of every quality battery (`tools/quality_harness.sh`) is *"Give me
a tour of the codebase — main crates, key files, and how everything connects"*,
which contains no distinctive term at all.

The rule that keeps this register from becoming noise is not rarity but
**subject-bearing but vocabulary-free**. *"What does this module do?"* carries no
subject and matches every folder equally — it is not a probe, it is a constant.
*"How does the assistant decide which parts of a huge codebase to show the
model?"* carries the subject in plain language and discriminates.

## 4. Specificity is an input, not a hope

Asked for good questions, a model writes *"What does this module do?"* twelve
times. So the prompt is handed the terms that characterise the folder, computed
against the whole workspace:

1. **Declared symbols** (`probe/symbols.rs`) — a line scan for `pub fn` /
   `struct` / `#define` / `def` / `func` … per language. Deliberately
   recall-oriented: precision is the rarity gate's job.
2. **Directory frequency** (`probe/idf.rs`) — document = directory. Ranked by
   `tf × ln(N/df) × structure`, capped at `RARITY_MAX_DF_PCT` (2% of
   directories, floor 3).

Three extraction rules earned their place by failing first (§7).

## 5. Admission: two gates and one oracle

**Every register** — no self-address (the folder's path with a separator, or a
filename with its extension), no deixis ("this module"), question-shaped,
bounded length. A **bare stem** is deliberately allowed: `dir_unit` is also the
name of a real type, and *"what is a `DirUnit`"* is a perfect conceptual probe.

**Registers 1–3** — must carry a term the frequency index calls distinctive.

**Register 4** — exempt by construction. Demanding a rare term there would
recreate the other three in worse prose.

**Duplicates** — a distinctive term is *decisive*: two questions naming different
rare terms are different questions whatever framing they share. Word overlap
alone collapses a register's six questions to one, because they naturally share
their framing ("where does X get initialised", "where does Y…").

**Cross-folder collision** (`filters::systemic_collisions`) — registers 1–3
cannot collide much; register 4 collides constantly, because `kv_cache/` and
`kv_cache/chunked/` independently produce the same vocabulary-free question.

### The oracle

Scoring a probe that is *in* the corpus measures almost nothing: its own
signature is resident, so it self-matches. The quality measure needs a query the
corpus has never seen, and that is free — each register generates 12 candidates
and keeps 6, so the next admissible candidates are **held out**: never ingested,
written to `.zend/probe_holdout.json`, kept only as test queries.

`zend/examples/probe_eval.rs` drives `POST /v1/substrate/project` (which captures
the query's live decode-Q and writes nothing) and reports hit@1 / hit@3 / MRR,
per register, with the folders that most often steal first place.

This is the discipline the tool work taught: a battery scored 3/6 when the truth
was 0/6 because its questions were answerable without retrieving anything. An
oracle must demand something the model cannot supply unaided.

## 6. Execution

Three conversations per directory, never more than two live at once:

1. **Folder chain** (existing) — list, read the anchor, decode the summary.
2. **Generation** — a throwaway conversation framed on `GENERATE_BRANCH`
   (`persona: question_writer`), prefilled with a compact evidence block: the
   folder's leaf name, its summary, its file **basenames**, and its distinctive
   terms. Decoded once, parsed, tombstoned. It gets its own conversation because
   its answer is a list of 48 questions that would otherwise seal into the
   folder's timeline as the densest question-shaped thing it owns — and because
   `Sequence::fork` does not inherit the parent's turns.
3. **Probes** — one conversation framed on `ANSWER_BRANCH`
   (`persona: assistant`, `thinking_effort: quick`), seeded with a single
   prefilled turn carrying the folder's summary, then one decoded turn per probe.

**The answering conversation cannot be the folder's own.** The folder is framed
as the summariser: `thinking_effort: off`, `response_length: terse`. A probe
answered there produces two clipped sentences and **no `<think>` block** — and a
provenance scan runs *during* decode, so mid-reasoning the Q hitting the corpus
is reasoning-shaped, not question-shaped. A corpus of terse answers is indexed in
the wrong shape for most of the moments it is actually scanned.

Probes are **interleaved across registers** so the context accumulated over 24
consecutive decoded turns falls evenly, rather than entirely on whichever
register came last — which would be `systemic`, the one this design most depends
on.

### Invalidation

`DirUnit::content_hash` now covers the walked listing, the anchor text, **and the
set of names the folder's files declare**. Hashing the symbol *set* rather than
file bodies keeps the trigger at API churn (rare) rather than editing
(constant): re-ingesting a directory costs a summary plus 25 decodes.

### Failure policy

The probe layer is never fatal. A directory whose generation or decode fails
keeps its summary and its place in the map — a missing index entry costs
retrieval quality for one folder, where a propagated error costs the folder
itself. The pass runs **before** the hash tag commits, so a crash re-ingests the
directory whole rather than marking it complete with a partial probe set.

## 7. What failed first

Every rule below exists because its absence produced a plausible-looking result.

| defect | symptom | fix |
|---|---|---|
| **Parser required list markers** | 7 of 9 directories produced `candidates=0`. The model writes bare question lines under a heading, no numbering. Indistinguishable from a generation failure in the counters. | Key on the trailing `?`, not on a bullet. Plus a raw-text warning whenever a generation parses to nothing. |
| **Test names as seeds** | `repo_scan/` probes came out as *"What constitutes an 'empty anchor' in the context of `an_empty_anchor_file_yields_nothing`?"* — a test function is maximally distinctive (df=1, tops every ranking) and completely useless. | Suppress `#[cfg(test)]` blocks and `#[test]`-attributed declarations. |
| **Alphabetical tie-break** | All 28 seeds for one folder began with `a` (`add_collection`, `adopt_turn`, `aggregate`, `all_section_ids`, `AnchorConfig`…). Most terms tie on score, so the tie-break decides everything. | Break ties on a stable FNV hash of the term. |
| **Language primitives as vocabulary** | Top terms for `candle-kernels/src/simple/` were `usize`, `double2`, `__float2bfloat16`, `half2`, `decltype`. They leak through weak paths (one `#define`, one backticked doc word) so they land in few directories and read as distinctive. | Explicit noise list + a structure weighting (compound identifiers outrank bare lowercase words). |
| **Duplicate rule too aggressive** | Three questions about three *different* symbols collapsed to one, because they shared "where does X get initialised". Would have gutted every folder's locational register. | A differing distinctive term makes two questions distinct regardless of shared framing. |
| **Path leaked into the prompt** | The evidence block showed full paths, handing the generator exactly the strings rule 1 forbids. | Leaf folder name and file basenames only. |
| **Decorated probe label** | Probe conversations labelled `"<dir> probes"` would not match the directory in the harness — reporting a formatting artifact as 0% retrieval. | Label both conversations with the bare directory; `kind` distinguishes them. |
| **Register briefs without examples** | Told a locational question "must name a distinctive term", the model wrote *"Where can I find the unit tests for the allocation strategies?"* — well-formed, on-topic, carrying no distinctive term, so the rarity gate discarded it. | One worked example per register, built from the folder's own terms, each from a different seed. |
| **Systemic register drifted generic** | *"How does the system ensure consistency between cached metadata and live sources?"* — fits every folder ever written. | The brief now demands the subject in plain words and states the disqualifying test: "if your question would still make sense pasted under a different folder, it is wrong." |

## 8. Measured results

### Retrieval — the headline

Scored with `probe_eval` against a live daemon, 19 directories probed out of 353,
372 probes ingested, 186 held-out queries:

| population | hit@1 | hit@3 | n |
|---|--:|--:|--:|
| **held-out** (never ingested — the quality measure) | **6.7%** | **11.7%** | 60 |
| **resident** (self-match — necessary condition only) | **64.4%** | **73.3%** | 45 |

**What this says.** The mechanism is wired correctly: a probe that is in the
corpus retrieves its own folder 64% of the time, against ~0.3% for a random pick
among 353 — so generation, admission, ingest, the tag join, the projection and
the scan all work end to end.

**What it does not yet say.** Generalisation is weak. A *different* question
about the same folder retrieves it only ~1 time in 10, so the corpus is matching
close to signature identity rather than to what a question is about. The
hypothesis — that query-shaped exemplars beat a declarative summary — is not
demonstrated by these numbers.

**The confound, which is large and must be removed before re-reading them.** Only
19 of 353 directories (5%) carry probes. A probed folder holds 24 extra
question-shaped turns; an unprobed one holds a summary. That is a mass imbalance
the layer's scoring was never calibrated for, and the miss table shows it
directly — first place is taken over and over by whichever *probed* folder is
nearby:

```
folders most often taking first place from the right answer:
    12x  candle-conversation/src/provenance/gallery_arena/
     8x  .
     7x  candle-core/src/vram/
     5x  candle-conversation/tests/
```

Every one of those is a probed folder. With a uniformly probed corpus the mass is
even again and this term disappears. **The next experiment is therefore a subtree
in which every directory is probed** — not a larger partial corpus.

Two further caveats on the numbers: the `structure` group selects `k: 3`, so
hit@3 is really "was it selected at all"; and the group's tuning
(`beta: 0.65`, `layer_weights: [0.1, 1.0, 0.1]`, the normalisation floors) was
derived for the summary-only surface and has not been revisited.

### Generation reliability

Roughly one directory in four yields nothing, because the reasoning block never
closes. The block is budgeted for rather than competed with
(`GENERATION_THINK_CAP + GENERATION_ANSWER_BUDGET`) and stripped before parsing;
the cap then bounds the degenerate case rather than letting one folder consume
unbounded decode. Observed working: `think_ran_away=true` at 22,569 characters on
`candle-core/src/quantized/`.

### What the pipeline produces when it works

Per-directory yields from the live runs (Qwen3.6-35B-A3B, 16 GB RTX 4090 Mobile,
index over all 353 directories, rarity gate df ≤ 7):

| directory | seeds | candidates | admitted | held out | rejected |
|---|--:|--:|--:|--:|--:|
| `candle-conversation/src/persistence/` | 28 | 48 | **24** | 12 | 0 |
| `.` (workspace root) | 28 | 47 | **24** | 12 | 0 |
| `candle-book/` | 16 | 48 | **24** | 12 | 1 |
| `candle-book/src/` | 13 | 43 | **24** | 11 | 2 |
| `candle-conversation/src/` | 28 | 39 | 15 | 6 | 0 |

Admission is doing real work rather than passing everything: rejections are
dominated by `no_distinctive_term` (a question in a specific register carrying no
rare term) and `self_address`.

### Term extraction, whole workspace

2,402 files walked, 2,086 carrying symbols, 41,123 symbols, 353 directories,
walk + index in **0.7 s**. Distinctive terms are domain vocabulary —
`RotaryEmbedding`, `apply_rotary_emb_qkv`, `WideQSig`, `gather_wide_sigs`,
`quantize_warp_reduce_max`, `Y_TYPE_F32`, `Q5_K`, `ChunkGid`. 49 of 353
directories (14%) have no distinctive term and can carry systemic probes only.

### Blocker 1 — the generator will not stop reasoning

The model opens every generation with a `<think>` block and does not close it
inside any budget tried. Measured blocks: 2,183 / 5,342 / 6,363 / 6,735 / 12,752
/ 15,021 / 15,140 characters. When the block outruns the budget the decode is cut
mid-thought and the directory yields nothing.

Four suppression attempts, none sufficient (§7 records why each failed):
`thinking_effort: off`; the schema's `no_think` toggle; `/no_think` prepended to
the system-prompt body; `/no_think` on the user turn. Splitting generation into
one turn per register — a twelve-question ask instead of forty-eight — still drew
12,752-character blocks, so the cause is not task size.

**The remaining fix is the half of the mechanism not yet applied**: the soft
switch also requires an empty `<think></think>` prefilled into the *assistant*
turn, which the dialect does at projection and these statically-prompted
conversations never run. A prefix-then-decode submission is what this needs.

### Blocker 2 — throughput

At pool width 8: `fwd avg=663–1000 ms`, roughly **12 tok/s aggregate**, against
124 tok/s documented for this pool at width 8.7. The cause is visible in the log
— the elastic weight/KV boundary ping-pongs, conceding ~200 expert slots and
reclaiming them every second (710 concessions in one sample), each concession
invalidating the GPU-native dispatch tables. Host RAM sits at 1.5 GiB free of
31.5 GiB with GPU utilisation at 65%, so the ingest is outrunning the warm→cold
drain.

At that rate a directory costs ~90 minutes of wall clock and the full 353 would
take ~50 hours. Width was already lowered from 25 → 12 → 8 (via
`SCAN_CONV_KV_MIN`); the remaining levers are the conversation churn the probe
layer introduces (three conversations per directory where there was one) and the
host RAM partition.

## 9. Open items

- **BDP dedup.** Near-duplicate detection is lexical. The signatures themselves
  are the right metric — two probes that are BDP-identical are redundant *as
  probes* however differently they read — but that needs the prefill to have
  happened, so it belongs in a second pass.
- **Distillation.** Calibration exemplars end at `DistillMode::ProvenanceOnly`
  (signatures kept, text dropped). Probes are currently evicted to cold with
  their text intact, which is useful while the layer is being tuned and wasteful
  once it is settled.
- **Systemic collision resolution.** Collisions are detected; the winner is
  currently the first directory in walk order. The retrieval harness is what
  should arbitrate — the probe belongs to whichever folder it actually retrieves.
- **Extending to `code_reading`.** The same machinery applies per file scope; the
  registers and gates carry over unchanged.
