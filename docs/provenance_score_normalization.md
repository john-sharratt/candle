# Provenance Score Normalization

**Status:** built (`candle-conversation/src/normalization/`). Supersedes the
per-node absolute threshold tuning for provenance selection. Since 2026-11-11 a
score is read against the candidate's **background** (§3.1); the hit-level EWMA
it replaced is recorded in §3.1a.

This document captures what we learned diagnosing repo_map retrieval and specifies
the **score-normalization module** that fixes it. It is authoritative for that
module: if the code disagrees, fix the code (or fix this doc in the same change).

---

## 1. The problem

Provenance selection scores each candidate (a tool section, a repo_map folder, a
conversation turn) against a probe (the live decode Q signature) with the
late-fusion scorer (`score_provenance_late_fusion`): per query token, the
best-agreeing gallery token per case casts a `z × margin` vote, tallied per case.
Selection (`belief_step`) then compares these **raw** tallies across candidates and
picks the top-`k` above `min_score`.

The raw tallies are **not comparable across candidates**, because every candidate
sits on its own absolute scale. Measured over the substrate (see §7):

- The workspace-root cluster (`.`) has a running-mean score of **458**; a specific
  dir like `candle-core/src` sits near **10–130**. Root is not large — it is
  *generic*: its listing names every crate, so its signature agrees with many
  query tokens regardless of relevance. It is a **stopword cluster**.
- `perf-investigation/` is a **numeric attractor**: its benchmark-number listing
  scores ~310 on a `sqrt of N` query. Provenance is Q-similarity, not semantics.
- For the **identical** query "what is in candle-core?", the candle-core cluster's
  raw rank swings across **#1, #2, #3, #4, #10** on different turns, while the
  generic clusters (root, `docs/`, `docs/archived/`) reliably own the top three.

Because the budget picks the top-`k` raw scores, generic loud clusters win the
slots and the specific target is crowded out. No absolute `min_score` fixes this:
a threshold that admits candle-core at rank #10 (score ~19) admits everything.

This is a **normalization** problem — we must compare each candidate against its
own baseline, not against a shared absolute scale.

---

## 2. What we rejected (and why)

Measured with the `provenance_normalize` (§84) harness over six probe types
(tour/time/sqrt + three candle-core turns) against the repo_map gallery:

| candidate normalizer | seal-time? | result | why it fails |
|---|---|---|---|
| **self-match ceiling** (score a cluster's own sig against the gallery, take its entry) | yes | scores collapse to ~0; root not discounted | the perfect self-match is dominated by token count and a degenerate `z` spike — it scales with *size*, not genericness, so a big specific cluster is penalized harder than a loud generic one |
| **gallery column promiscuity** (other clusters as pseudo-probes) | yes | all ~0 | winner-take-all: a cluster used as a probe perfectly self-matches and grabs all its own votes, leaving cross entries empty |
| **leave-one-out promiscuity** (score each cluster vs the gallery minus itself) | yes | ≈ raw | a cluster's runner-up affinity to *other clusters* does not predict its loudness against *natural-language queries* |
| **cross-probe z-score** `(x − mean)/std` | no (running) | inconsistent; amplifies noise | dividing by a near-zero std blows up obscure clusters that spike once |

**The structural lesson:** nothing computable from the gallery alone works. A
cluster's loudness is a property of how its signature responds to the
**query distribution** (natural-language decode signatures), which is simply not
present in gallery-vs-gallery comparisons. The normalizer must be **learned from
query traffic**.

---

## 3. What works: per-scope background normalization

### 3.1 Against the background, as tail surprise

A candidate's **background** is the distribution of the (reference-scaled, §4.2)
scores it reaches on probes that are *not about it*. A probe's score is read as
how far into that background's tail it sits:

```
surprise(c | x) = −ln P(background(c) ≥ x)          (add-one smoothed)
normalized(c | x) = 1000 × surprise(c | x) / ln(1 / full_hit_tail)
```

Under the null — a probe not about `c` — every candidate's surprise follows the
same distribution, so a loud candidate (a hub that scores on everything: the
workspace root, a crate's `lib.rs`) wins no more often by chance than a quiet
one, while a genuine hit sits in its own far tail. It is the empirical-tail
formula the normalization gate measured best (`provenance_gates normalization`,
§7.1).

- **The background is counts.** Scores fall into log-spaced bins (8 per nat of
  `ln(1 + x)`, a 13% step); a score's tail is the count above its bin plus the
  share of its own bin above it. Counts commute, so a background — and every
  score read against it — is the same whatever order its evidence arrived in.
- **A probe's known answer is left out of its background.** A warm-up exemplar
  is about its own tool, an ingest self-probe about its own exchange, a sealed
  tool turn about the tool its tags name (`Observe::answer` — so a tool called
  on many turns does not carry its own hits as background): those scores are
  the hit, not what a hit is measured against. A dialogue seal names no answer
  and teaches every candidate — which is what mutes a hub: its loud scores on
  everything become its own background.
- **A warm-up probe is scored without its answer in the gallery.** The scan is
  late fusion — each probe token votes for the case of its best-matching
  gallery token — and a warm-up probe is a copy of something the gallery
  holds: a tool's exemplar is one of that tool's windows, an ingest self-probe
  one of its exchange's turns. Scored against the whole gallery it matches
  itself on every token, its answer takes every vote, and every other
  candidate's sample is 0. Eight zeros are a background, so a live score then
  reads past the whole of it: measured on the tool catalog on 2026-11-11,
  "hi, how are you doing today?" put five tools near 3000 and every probe put
  a dozen tools past 2600 — a flat, high band nothing was told apart in. So
  `fold_warm_probes` scores each tool's exemplars against the gallery minus
  that whole tool (its sibling exemplars win every vote as surely as its own
  window), and `warm_timeline` each self-probe against its file minus its
  exchange (`FileScan::without_exchange`). Each sample is then what the
  other candidates score on a probe that is not about them, which is what a
  background is. One scan per member or exchange probed, the same probe work
  as one batched scan. Pinned bit for bit against the public scorer by
  `turn_belief_scan::the_collection_warm_up_scores_each_exemplar_without_its_own_member`
  and `the_ingest_warm_up_scores_each_probe_without_its_own_exchange`.
- **Past everything the background holds** a score keeps rising one nat per nat
  of `ln(1 + x)` from the top bin's edge, so two scores beyond the whole
  background still rank by size.
- **A candidate too new for a background** (fewer than `min_background`, 8,
  samples) is read against its scope's pooled background — every sample any of
  its siblings was taught — and a scope with too little pooled either, against
  the probe's own scores across the competition. A competition is therefore
  normalized in one call (`NormalizationCache::normalize`), the pooled documents
  of a group included.
- A score of no evidence (zero) reads zero.

### 3.1a The hit level it replaced

From 2026-08 to 2026-11 a score was divided by the candidate's **hit level**, an
asymmetric EWMA (`α_up 0.30`, `α_dn 0.02`, prior 400) meant to settle at the score
it reaches when it is the answer, so a real hit landed near 1000. Every fold
observed every candidate, so each probe *not* about a candidate pulled its level
2% toward that probe's score: a candidate stays at its hit level only if it is
the answer about once in fifty folds. The tool warm-up folded each tool's eight
exemplars in catalog order, so an early tool sat through ~1,240 more folds
(`0.98^1240 ≈ e^-25`) and ended at its background, scoring a real hit near
10,000, while the last tools kept their hit level and were muted. Measured on
2026-11-11: every out-of-band tool score belonged to an early name
(`journal_check` 10,513, `file_grep` 8,138) and the tools that failed to qualify
were all late ones (`signature_verify` 60, `ssh_open` 9, `web_fetch` 1.0) — a
tool's score followed its place in the alphabet. code_reading's floored path
divided by the traffic **peak** instead (`max(peak, size floor)`, Concept A.4),
which the gate measured barely better than raw: R@1 0.134, one file taking 15.3%
of every win.

### 3.2 Semantics of the band

`full_hit_tail` is 0.01, so:

- **1000** — a full hit: beyond 99 in 100 of the candidate's background.
- **> 1000** — past that: about 1500 is beyond 1 in 1000, and a score past the
  whole background rides higher still.
- **~500** — beyond 9 in 10; **~150** — the background's median. The tool gate's
  `min_score 800` / `evict_score 600` are a 1-in-40 and a 1-in-16 score.

### 3.3 Per-scope = per score-competition node (verified against `project.rs`)

Normalization is computed **per scope where scores are actually compared**, not
over the whole substrate. Reading the selection path, every score comparison is
**within a single group or collection**:

- Member selection (`apply_selection` / the belief path) picks members *within*
  one turn group or section collection, comparing scores among that node's
  members only.
- The token trim (`trim_to_budget_low_score_first`) runs on **one group's**
  selected set against **that group's** budget.
- The **layer is not a score competition.** The layer/group budget is a
  `flexbox_distribute` over `budget` priority + natural consumption — it hands out
  *token space*, never compares member scores across groups. So there is **no
  layer-scope normalization**; scores from different groups are never put on the
  same axis.

The score-competition scopes are therefore:

- **Turn group** → children are its turns (repo_map folder turns, dialogue turns).
- **Section collection** → children are its sections (tools).
- **Sub-window** (future) → children are the regions within one turn (files inside
  a repo_map listing — self-referencing sub-windows).

The nested "coarse → fine" selection — *pick the dir, then the file inside it* — is
**turn group → sub-window**, both score-competition scopes. A member is
re-normalized at each scope it competes in: a loud region inside an irrelevant
cluster is filtered when the cluster loses at the group scope, before its
sub-windows ever compete. (Layer naming in the earlier sketch mapped to the group
level; the code's "layer" is the token-budget tier, orthogonal to scoring.)

Each node keeps its own per-child hit levels, floor, and cold-start prior over
**its direct children only**, so a rarely-touched dir warms up against sibling
dirs and a rarely-touched file against sibling files. Because each scope
normalizes into the same ~0–1000 band, selection thresholds become **uniform
across scopes** (the "800 for tools vs 200 for repo_map" problem dissolves).
Per-node thresholds remain *available* (varied now, a good uniform default later).

### 3.3a A group of one-exchange documents is one competition

The scorer votes `z × margin`, and the margin is the leading case's agreement
over the runner-up's — so a competition needs two cases. A file scan of a file
with one exchange has none to beat: its margin is its raw agreement, about half
the signature bits for any pair at all. A group whose every timeline is one
document (npcd's world, eras and stories layers — one conversation per document)
scored each document as itself against nothing; normalized against its own
level, every document landed in the same 0.7–0.9 band against every probe, and
the same generic entries (a derelict ship, a tower's construction) came top for
every character whatever it was doing.

So `belief_files::competitions` pools a scan's one-exchange files into one
gallery, a case each, and a file of several exchanges stays its own competition
(zend's code reads). The pool is one score-competition scope —
`ScopeKey::Pool { group }`, children keyed `ChildKey::Timeline` — normalized in
one call and folded by a seal all at once. Its backgrounds are **learned from
traffic only**: a document probed by its own signature agrees with
itself on every bit and leaves the next document far behind, which is the
self-match ceiling §2 rejects (measured live: every hit at 1–2% of the band), so
the ingest warm-up self-matches only files of several exchanges. Measured on the
live mind after the change: top hits 1,000–2,500 with lock-ons above (a reviewer
locked on its own draft at 13,365), documents with no evidence at exactly 0, and
no document winning across unrelated activities. Pinned by
`turn_belief_scan::one_exchange_documents_compete_as_one_pool`.

### 3.4 Subdividing a scope by phase — tried, measured, REMOVED

> **Status: not in the engine.** The phase lens was built, A/B'd, and found to be
> a regression; it shipped `None`-by-default and was then deleted rather than
> left as an off-by-default second scoring path. `ScanPolicy::phase_lens`,
> `PhaseSpans`, `MIN_PHASE_OBSERVATIONS` and the `CollectionPhase` fusion are all
> gone from the code. What survives is `turn_layout::phase_span_of`, which the
> **offline** analysis path (`substrate_inspect --probe-phase` /
> `--gallery-phase`) uses to re-run the comparison on any corpus.
>
> This section is kept because the measurement is the useful part, and because
> anyone reading §20.3's 65.6-vs-21.5 will otherwise rebuild it.

A turn is `user question → <think>…</think> → response`, and those regions do not
carry the same retrieval signal. Measured for tool routing
(`tool_selection_provenance_results.md` §20.3): routing on the `user` phase scores
**65.6**, on `user+think` **63.4**, on `think+resp` **21.5** — the task
description is what maps onto a tool definition, while by the time the model is
emitting the `<tool_call>` JSON it has already committed. Scored as one window,
which is what the whole-turn scan does, a live question is matched against an
exemplar that is ~95% think block and call, so the region carrying the link
contributes a few percent of the evidence and the rest is surface the gate has to
suppress.

**That reasoning did not survive contact with the corpus.** Fusing the `user`
lens with the whole-window scan moved Tool-1 74.5% → **73.6%** and exact-1
70.0% → 68.9%; its only real gain was total misses 6.3% → **5.7%**. Reading ~6% of
each exemplar finds evidence where the whole window found none and dilutes the
ranking where the whole window was already right — which is the shape of those
numbers exactly. §20.3 does not transfer: it measured a CCA-rotated call→def
routing, not this BDP self-retrieval.

The design below is what was built, retained as the record of the attempt.

So a collection also normalizes **per phase**:
`CollectionPhase { group, name, phase }`. Three rules make the subdivision safe:

1. **Fuse, don't filter.** Every lens produces scores on the same 0–1000 band and
   the section takes the **max** across them. A lens can add evidence; none can
   veto another. The whole-turn scan is therefore a floor, and the phase lens can
   only raise a section the question actually matches.
2. **A cold sub-scope falls back to its parent, it does not divide by noise.**
   Subdividing splits the traffic, so a phase scope is cold for far longer than
   the undivided one — and dividing by a learning-starved level does not blur the
   ranking, it *inverts* it (§A.4 measured normalization dropping code Top-1
   57.1% → 47.5% on cold levels). `normalize_with_fallback` therefore splits a
   scope's children per child: those with ≥ `MIN_PHASE_OBSERVATIONS` (8, matching
   the warm-up's probes per member) normalize on the phase scope, the rest on the
   undivided collection, in one call.
3. **Each lens is learned on the band it is read on.** The phase lens is probed
   with the **pinned question window** (Concept F) and the undivided scope with
   the whole probe — live *and* in the warm-up, which pairs each exemplar's whole
   signature with its own user span. A phase level learned from a whole-turn probe
   would sit on a different band from the one a live query lands on, and the
   max-fusion would then be decided by that mismatch rather than by the evidence.

**Why it was removed rather than left off.** An off-by-default lens is a second
scoring path that nothing exercises: it cannot be trusted without re-measuring,
it has to be kept compiling and correct against every change to the path that IS
used, and its cost is paid in reading — the gallery walk built spans for every
exemplar on every reprojection whether or not a node had named a phase. The
repo's standing rule is that a path is either the path or it is not landed. The
full table is in `tool_selection_provenance_results.md` §26.

Re-run the A/B offline before rebuilding any of it:

```
substrate_inspect belief-eval --tag tool --normalize \
    --probe-phase user --gallery-phase whole,user
```

`--probe-phase user` is load-bearing. With a whole-turn probe the exemplar carries
its own `<tool_call>` JSON, which contains the tool's name, so the scan matches on
the answer and the whole-window baseline reads 97.4% — a number no live query can
reach, because live the call has not been emitted yet.

The offline harness reads all three spans straight from the decls, so the
comparison stays available per corpus without any of it living in the engine.

**No new persistence.** The spans come from `TurnDecl.segments`, already written at
seal, read through `turn_layout::phase_span_of`. Signatures are captured one per
real token, 1:1 with that grid, so a segment's K/V span indexes a signature window
directly. On the GPU the lens is a `start`/`end` on the existing `PagedWindow`:
residency is shared with the whole-turn scan (same pages, no extra upload) and the
scan index is keyed on `(sid, fingerprint, len, start, end, case)`, so the two
scans cache separately and cannot serve each other's regions.

---

## 4. Module design

**Separation of duties:** normalization data is **runtime-derived, in-memory, and
NOT persisted.** It is not part of the substrate/redo-log — it is a cache
*derived* from the substrate. On load it is **rebuilt from the substrate's
existing turns** (not started cold), then evolved as new turns seal. Not persisting
it is fine because it is reconstructible from the substrate at any time.

Location: `candle-conversation/src/normalization/` (one concern per file):

| file | role |
|---|---|
| `mod.rs` | public API: `NormalizationCache`, `ScopeKey`, `ChildKey`, `NormConfig` |
| `background.rs` | `Background` — one child's log-binned counts; `add`, tail `surprise` |
| `scope.rs` | `ScopeState` — per-child backgrounds and their pool; `normalize` + `observe` |
| `cache.rs` | `NormalizationCache` — scope map keyed by `ScopeKey`, each scope an atomically swapped immutable state |
| `tests.rs` | unit tests (§6) |

### 4.1 Types

Both keys are **structured enums** (not formatted strings), so the scan allocates
nothing per candidate and the cache can reason about them (§4.3 eviction):

- `ScopeKey` — a **score-competition** scope. **No `Layer` variant** — the layer is
  token distribution, not a score competition (§3.3):
  - `TurnGroup { group: u64, timeline: u64 }` — a turn group's gallery on a
    specific timeline. A re-scan mints a new timeline → a new scope.
  - `Collection { group: u64, name: String }` — a section collection (tool
    catalog); its gallery is stable, so no timeline.
  - `CollectionPhase { group: u64, name: String, phase: Phase }` — the same
    competition scored over only the `user` / `thinking` / `response` region of
    each exemplar (§3.4). Its parent is the `Collection` scope of the same
    `(group, name)`, which is what a child it has not yet learned falls back to.
  - `SubWindow { turn: u64 }` — a sub-window within one turn (future).
- `ChildKey` — a candidate within a scope: `Turn(u64)` (turn index — allocation-
  free, stable within a gallery version) or `Named(String)` (tool / section name,
  how `belief_gallery` already keys collection members via `slot_of(tag)`).
- `Background { zeros: u32, bins: Vec<(u16, u32)>, total: u32 }` — one child's
  samples: those of no evidence, and the rest as `(bin, count)` ascending.
- `ScopeState { children: HashMap<ChildKey, Background>, pooled: Background,
  observed: HashSet<u64> }` — one scope's state: each child's background, all of
  them pooled (§3.1, a new child), and the evidence ids already folded (§4.2).
- `NormalizationCache { cfg: NormConfig, scopes: HashMap<ScopeKey, ArcSwap<ScopeState>> }`.
- `NormConfig { scale 1000, full_hit_tail 0.01, min_background 8, probe_t_ref 256 }`.

Turn groups key the child by index rather than by path *tag*: tag-keying (so
learning survives a re-scan) would need an `all_streams()` scan per turn per
reprojection — prohibitive on the hot path. Instead a re-scan makes a new
`(group, timeline)` scope, which correctly resets learning for the regenerated
clusters and rebuilds it over subsequent queries, its children read against the
pool and then the probe until they have backgrounds of their own (§3.1).

### 4.2 Lifecycle and API

The cache is owned alongside the live projection state (per conversation session /
per substrate handle — TBD in the plan, see §8). Two operations, both driven from
the scan (`score_belief_groups` / `score_belief_collections`):

1. `normalize(scope, &[(child, raw)], probe_tokens) -> Vec<(child, normalized)>` —
   read path. Each child's reference-scaled score read against its background
   (§3.1), a child without one against the pool or the probe. Pure read; does not
   mutate. Runs on **every reprojection**, so selection always sees normalized
   scores.
2. `observe(scope, source, &[(child, raw)], answer, probe_tokens)` — write path.
   Each raw, reference-scaled, joins its child's background and the pool — every
   child's but `answer`'s, when the probe is known to be about one (a warm-up
   exemplar, an ingest self-probe; `None` for a live seal). **Idempotent per
   observation:** `source` identifies the evidence (the turn this scoring pass came
   from) and a scope folds each source exactly once, so re-observing it is a no-op
   — one hash lookup rejects it before any child is touched.

   That idempotency is what lets the backgrounds be rebuilt on **every load, from
   empty**, against a substrate already on disk, while the same scopes keep
   learning from live traffic afterwards. Deduplicating on the *score* instead
   would be wrong — a promiscuous child scores about the same on everything, so it
   would be recorded once and never learn that it is loud across all traffic,
   which is exactly what its background exists to show.

   Runs **once per turn, at seal** — hooked into
   `Conversation::last_turn_belief_scores`
   ([conversation.rs](../candle-conversation/src/conversation.rs)), which already
   re-scores the just-sealed turn's **whole-turn sig** against every gallery. This
   is exactly the probe §85 calibrated against.

**Why observe is per-turn, not per-reprojection.** A turn fires many reprojections,
each with a *different* sliding probe (query-head + trailing window). Observing on
each would weight a turn in every background by its length, and teach each
background its own turn's partial windows. Observing once at seal makes each turn
one sample.

**Probe scale.** `observe` uses the whole-turn sig; `normalize` uses the sliding
reprojection probe. Both are restated over the 256-token reference
(`NormConfig::probe_t_ref`) before they meet a background, so a score is a rate
whatever the window.

Selection consumes the **normalized** scores: `belief_step` runs on the 0–1000
values, so belief accumulation, `min_score`, `evict_score`, and the early-decode
window all operate on the normalized scale.

### 4.3 Substrate change / bounded growth

Because a turn group's scope is keyed by `(group, timeline)`, a **re-scan** simply
produces a *new* scope — the regenerated clusters start with no background
(correct: their content changed) and rebuild over subsequent queries.

**Every `(group, timeline)` scope is retained independently.** An earlier version
evicted a group's *other* timelines on each observe, to hold one scope per active
group. That assumed one active timeline per group. It is true of a re-scanned
single cluster and **catastrophic for `code_reading`**, which has many
simultaneously-active timelines — one per ingested file — each needing its own
learned backgrounds so a cross-file query can put them on the common band and
compare them fairly. The eviction wiped every file but the last, leaving the cache
empty for all the others, and a promiscuous low-entropy file won every query.

Stale scopes from a re-scan are therefore left in place: dead once their timeline
is inactive, bounded by the re-scan count, and one small `ScopeState` each. A
background is a few hundred bytes — its occupied bins — whatever its sample
count.

*Within* one gallery version the membership is fixed (a re-scan makes a new scope,
not a mutated one), so there is no per-child add/drop reconciliation to do. A
background keeps every sample it was taught, so when the raw scale shifts (the
gallery grows and the scorer's `z` term moves) the old samples weigh with the new
until enough traffic has arrived; it rebuilds from the current substrate on every
load. Collections (stable galleries) never re-mint, so their scopes are not
evicted.

### 4.4 Warm-on-load

The cache is **not** left cold on restart. On the first belief scan after load,
`ensure_normalization_warm` replays the substrate's existing **sealed dialogue
turns** (empty-tagged; gallery turns are tagged) through the same turn-group scan
with `observe = true`, so the backgrounds are rebuilt to what live traffic would
have produced. The tool collection's exemplars and each ingested file's own turns
are folded too (`warm_collection_normalization`, `warm_ingest_normalization`),
each with its own member or exchange as the probe's answer, and scored against
the gallery without it (§3.1). Details that matter:

- **Exactly once, no half-warm reads.** Guarded by `std::sync::Once`, which blocks
  any concurrent first callers until the replay completes — so a second session
  can't score against a partially-warmed cache.
- **Order-free.** A background is counts, so the warmed state does not depend on
  the order turns are folded in.
- **Bounded cost.** Only the most recent `WARM_REPLAY_MAX_TURNS` (512) turns are
  replayed — 512 samples read a background's tail to its 1-in-100 — and the
  one-time cost stays bounded on a huge substrate.
- The still-decoding current turn is excluded (no sealed signature yet). The replay
  reuses the live scan, so it also computes normalized scores it discards — reusing
  the scan beats duplicating the sub-window flattening for a marginal saving.

Remaining follow-up: the replay still runs on the *first reprojection* (hot path).
At 512 turns that is negligible; moving it to an explicit pre-serving startup step
would remove even that first-scan blip.

---

## 5. Integration points

The cache lives on the `Conversation` substrate handle (`Arc<Mutex<…>>`, shared
across clones, not persisted). `score_beliefs` carries an `observe: bool` that
splits read from write:

- **Read** (`observe = false`, every reprojection): `resolver.rs::score_belief_groups`
  computes the raw fresh scores, calls `cache.normalize(scope, raw)`, and feeds the
  **normalized** scores to `set_turn` / the challenger candidates. Runs from the
  scheduler reproject loop (`mod.rs`, passes `false`).
- **Write** (`observe = true`, once per turn): `conversation.rs::last_turn_belief_scores`
  is the seal-time whole-turn scan; it passes `observe = true`, so after computing
  the same raw scores the group scan folds them into the backgrounds. The only
  live writer — reprojections never learn.

**Status.** Wired for turn groups (repo_map, code_reading, dialogue, the pooled
one-exchange documents) and section collections (tools). Sub-windows are not.
`layer_weights` stay upstream — they shape the raw vote; normalization reads it.

---

## 6. Unit tests

Build alongside the module (TDD), asserting against exact expected values where
possible:

- `Background` (`background.rs`): log-spaced bins against hand-computed bin
  indices; the tail read off the counts (99 samples of 1.0: the top of their bin is
  `ln 100`, 1.0 itself `0.776`); the rise past the highest bin; zeros are no
  evidence; the same evidence in any order is the same background.
- `normalize` (`tests.rs`): a score beyond 99 in 100 is exactly 1000; **the warm-up
  order does not decide a score** (three tools folded in either order land at the
  same 1074.5 — the §3.1a regression); a loud child is muted by its own loudness
  and the specific one wins its probe (43.2 against 587.9); a known answer is not
  its own background; a new child reads against the pool; an untaught scope reads
  the probe against itself.
- Idempotency per source; scope isolation; sibling-timeline scopes retained.
- The band is a rate: invariant to probe length on the read path and on the write
  path; an equal raw total from a shorter probe scores as the larger rate.

---

## 7. Calibration

### 7.1 The background tail (`provenance_gates normalization`, 2026-11-11)

37,284 calibration questions over 1,740 code_reading files' own turns, scored on
BlackWorld's gallery; production's cache taught with every question as one probe,
its own file the answer. R@k: the question's own file in the top k; "top win":
the share of every file's wins on other files' questions taken by the biggest
winner.

| formula | R@1 | R@10 | MRR | win Gini | top win |
|---|---|---|---|---|---|
| raw | 0.107 | 0.351 | 0.188 | 0.942 | 54.5% |
| over peak (code_reading before, §3.1a) | 0.134 | 0.326 | 0.196 | 0.808 | 15.3% |
| over hit level (§3.1a) | 0.225 | 0.383 | 0.280 | 0.471 | 1.3% |
| **production — background tail** | **0.239** | **0.400** | **0.295** | **0.125** | **0.1%** |
| empirical tail, exact sort | 0.242 | 0.400 | 0.297 | 0.127 | 0.1% |
| z-score | 0.232 | 0.395 | 0.289 | 0.258 | 0.1% |

The binned background reads within 0.003 of the exact empirical tail. The two
"before" rows are the same gate on 2026-11-11 before the change (37,308 questions).

### 7.2 The hit level (`provenance_calibrate` §85, historical)

What the hit-level normalizer (§3.1a) measured when it was built. Replaying all
56 dialogue turns through the repo_map group scope with hit-level normalization,
cold-start prior, per-scope floor:

Normalized-score distribution (0–1000 scale):

| percentile | 10% | 25% | 50% | 75% | 90% | 95% | 99% |
|---|---|---|---|---|---|---|---|
| top-1 | 236 | 363 | 670 | 1485 | 2327 | 3280 | 6823 |
| top-2 | 139 | 224 | 466 | 785 | 1408 | 1747 | 2295 |
| all (noise) | 0 | 0 | 0 | 70 | 253 | 433 | 1090 |

- Noise floor (95th of `all`): **433**. Candle-core's weakest true hit: **495**.
- **`min_score` ≈ 450–500, `evict` ≈ 350** on the normalized scale cleanly
  separates hits from noise.
- Result: candle-core goes from raw #1 on 1/6 queries to **normalized #1 on 4/6
  and top-3 on 6/6** — always selectable at `top_k: 3` — while root and
  perf-investigation are discounted. The scores are legible: the clearest query
  locks on at 1933; marginal queries land ~500.

Tuning constants used: `α_up 0.30`, `α_dn 0.02`, `HIT_PRIOR 400`, `FLOOR_PCTL 0.10`
(floor is a *low* percentile of hit levels — a hard minimum, not a discriminator).

---

## 8. Open questions / follow-ups

- **Ownership**: where the `NormalizationCache` lives (per `Conversation`, per
  substrate handle, or a scheduler-level singleton) and how concurrent sessions
  share or isolate scopes. `observe` runs from the seal path and `normalize` from
  the reprojection path, so both need a handle to the same cache.
- **Live threshold confirmation**: re-measure the normalized distribution on the
  live reprojection probe to confirm `min_score` / `evict` on the background band
  (§3.2).
- **Sub-window scope**: normalize file-within-listing selection (self-referencing
  sub-windows) once the coarser scopes are in — the fine half of the group →
  sub-window nesting (§3.3).

---

## 9. Experiments (reference binaries)

- `zend/examples/provenance_layers.rs` — §83, layer-group weight sweep (produced
  the `layer_weights` tuning).
- `zend/examples/provenance_normalize.rs` — §84, normalizer bake-off (rejected the
  gallery-intrinsic normalizers).
- `zend/examples/provenance_calibrate.rs` — §85, per-scope running-mean/hit-level
  model over the whole substrate (produced §7).
