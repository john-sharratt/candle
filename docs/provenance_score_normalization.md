# Provenance Score Normalization

**Status:** design (module not yet built). Supersedes the per-node absolute
threshold tuning for provenance selection.

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

## 3. What works: per-scope hit-level normalization

### 3.1 Hit level, not mean

Two learned-from-traffic normalizers were validated:

- **Running mean** of a candidate's raw scores across queries (its promiscuity).
  Dividing by it discounts generic clusters correctly, but puts the output on a
  ~0–10000 scale (a specific cluster's mean is tiny, so a hit is 10–40× it).
- **Hit level** — the score a candidate reaches *when it is genuinely the answer*,
  tracked as an **asymmetric EWMA** (rises fast toward a strong match, decays
  slowly). Dividing by it puts a real hit at **≈ 1000 by construction**, with a
  sustained decode lock-on riding above it. This is the chosen normalizer: the
  0–1000 scale is legible for diagnostics and thresholds.

```
normalized(c | probe) = 1000 × raw(c | probe) / max(hit_level(c), floor(scope))
```

- `hit_level(c)` update, per probe, per candidate:
  `hit += (raw > hit ? α_up : α_dn) × (raw − hit)`, with `α_up ≈ 0.30`,
  `α_dn ≈ 0.02`, cold-start `HIT_PRIOR ≈ 400`.
- `floor(scope)` = a low percentile of the scope's current hit levels, hard-floored
  at the prior — kills "divide by ~0" explosions for cold/quiet candidates.

### 3.2 Semantics of the 0–1000 scale

- **~1000** — a full hit: the probe matches the candidate as strongly as the
  candidate matches when it is the answer.
- **> 1000** — lock-on: a sustained strong match during decode (the clearest
  query for a candidate locks on hardest).
- **< ~450** — below the noise floor; not a real hit.

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

Proposed location: `candle-conversation/src/normalization/` (one concern per
file):

| file | role |
|---|---|
| `mod.rs` | public API: `NormalizationCache`, `ScopeKey`, `normalize`/`observe` |
| `hit_level.rs` | `HitLevel` (asymmetric-EWMA state + update), `Running` mean for diagnostics |
| `scope.rs` | `ScopeState` — per-child hit-level map, floor, prior; `normalize_scores` + `observe` |
| `cache.rs` | `NormalizationCache` — scope map keyed by `ScopeKey`; evicts a group's stale-timeline scopes on `observe` |
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
- `HitLevel { level: f32, count: u32, peak: f32 }` — the EWMA state for one child,
  plus the highest raw score it has ever been observed at (the Concept A.4
  traffic-peak denominator) and how many distinct observations shaped it (what a
  subdivided scope consults before trusting it, §3.4).
- `ScopeState { children: HashMap<ChildKey, HitLevel>, observed: HashSet<u64> }` —
  one scope's state, plus the evidence ids already folded into it (§4.2).
- `NormalizationCache { cfg: NormConfig, scopes: HashMap<ScopeKey, ScopeState> }`.

Turn groups key the child by index rather than by path *tag*: tag-keying (so
learning survives a re-scan) would need an `all_streams()` scan per turn per
reprojection — prohibitive on the hot path. Instead a re-scan makes a new
`(group, timeline)` scope, which correctly resets learning for the regenerated
clusters and rebuilds it via the cold-start prior over subsequent queries. (Future
refinement: a cached path-tag→index map to preserve learning across re-scans, if
the post-re-scan cold-start proves noticeable.)

### 4.2 Lifecycle and API

The cache is owned alongside the live projection state (per conversation session /
per substrate handle — TBD in the plan, see §8). Two operations, both driven from
the scan (`score_belief_groups` / `score_belief_collections`):

1. `normalize(scope, &[(child, raw)]) -> Vec<(child, normalized)>` — read path.
   For each child, `1000 × raw / max(hit_level, floor)`; a child absent from the
   scope (never seen) uses the cold-start prior. Pure read; does not mutate. Runs
   on **every reprojection**, so selection always sees normalized scores.
2. `observe(scope, source, &[(child, raw)])` — write path. Fold each raw into the
   child's hit-level EWMA (creating it at the prior if new). **Idempotent per
   observation:** `source` identifies the evidence (the turn this scoring pass came
   from) and a scope folds each source exactly once, so re-observing it is a no-op
   — one hash lookup rejects it before any child is touched.

   That idempotency is what lets the levels be rebuilt on **every load, from
   empty**, against a substrate already on disk, while the same scopes keep
   learning from live traffic afterwards. Without it the two paths fight: a replay
   drags every level toward whatever it re-feeds, so the levels a load reproduces
   differ from the ones originally learned and the ranking they drive drifts on
   nothing but a restart. Deduplicating on the *score* instead would be wrong — a
   promiscuous child scores about the same on everything, so it would be recorded
   once and never learn that it is loud across all traffic, which is exactly what
   the hit level exists to discount.

   Runs **once per turn, at seal** — hooked into
   `Conversation::last_turn_belief_scores`
   ([conversation.rs](../candle-conversation/src/conversation.rs)), which already
   re-scores the just-sealed turn's **whole-turn sig** against every gallery. This
   is exactly the probe §85 calibrated against.

**Why observe is per-turn, not per-reprojection.** A turn fires many reprojections,
each with a *different* sliding probe (query-head + trailing window). Observing on
each would (a) update a child's level N times for one turn — over-decaying idle
children in proportion to turn length — and (b) diverge from the §85 model, which
saw one whole-turn probe per turn. Observing once at seal fixes both.

**Probe-scale caveat.** `observe` uses the whole-turn sig; `normalize` uses the
sliding reprojection probe. For turns shorter than `max_probe_tokens` these are the
same tokens; for longer turns the reprojection probe drops the middle, so its raw
magnitude differs by a roughly constant factor. The **ratio** normalizer preserves
ranking regardless; only the absolute 0–1000 band shifts by that factor. So the
§7 thresholds are the validated *mechanism + ballpark* — `min_score` / `evict` get
a final confirmation on the live reprojection scale. (If the shift proves
material, observe with a reprojection-style probe at seal — query-head + trailing
window — instead of the whole-turn sig.)

Selection consumes the **normalized** scores: `belief_step` runs on the 0–1000
values, so belief accumulation, `min_score`, `evict_score`, and the early-decode
window all operate on the normalized scale.

### 4.3 Substrate change / bounded growth

Because a turn group's scope is keyed by `(group, timeline)`, a **re-scan** simply
produces a *new* scope — the regenerated clusters start fresh at the cold-start
prior (correct: their content changed) and rebuild over subsequent queries.

**Every `(group, timeline)` scope is retained independently.** An earlier version
evicted a group's *other* timelines on each observe, to hold one scope per active
group. That assumed one active timeline per group. It is true of a re-scanned
single cluster and **catastrophic for `code_reading`**, which has many
simultaneously-active timelines — one per ingested file — each needing its own
learned levels so a cross-file query can rescale them onto the common band and
compare them fairly. The eviction wiped every file but the last, leaving the cache
empty for all the others, so normalization degenerated to a flat `scale/prior`
multiple of the raw score and a promiscuous low-entropy file won every query.

Stale scopes from a re-scan are therefore left in place: dead once their timeline
is inactive, bounded by the re-scan count, and one small `ScopeState` each.

*Within* one gallery version the membership is fixed (a re-scan makes a new scope,
not a mutated one), so there is no per-child add/drop reconciliation to do. When
the raw scale shifts (the gallery grows and the scorer's `z` term moves), the hit
level — an EWMA *of raw scores* — rides the shift, so `raw / hit_level` is
self-stabilizing; a change that *lowers* a child's magnitude re-settles slowly
(α_dn), the deliberate cost of not letting rarely-queried children collapse.
Collections (stable galleries) never re-mint, so their scopes are not evicted.

### 4.4 Warm-on-load

The cache is **not** left cold on restart. On the first belief scan after load,
`ensure_normalization_warm` replays the substrate's existing **sealed dialogue
turns** (empty-tagged; gallery turns are tagged) through the same turn-group scan
with `observe = true`, so the hit levels are rebuilt to what live traffic would
have produced. Details that matter:

- **Exactly once, no half-warm reads.** Guarded by `std::sync::Once`, which blocks
  any concurrent first callers until the replay completes — so a second session
  can't score against a partially-warmed cache.
- **Deterministic order.** `all_streams()` is a `HashMap` walk (unordered), so the
  turns are sorted by `(timeline, turn index)` before replay; the warmed levels
  don't depend on iteration order across restarts.
- **Bounded cost.** Only the most recent `WARM_REPLAY_MAX_TURNS` (512) turns are
  replayed — the asymmetric EWMA converges in a few dozen steps, so older turns
  barely move the levels, and the one-time cost stays bounded on a huge substrate.
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
  the same raw scores the group scan folds them into the hit levels. The only
  writer — reprojections never learn.

**Status.** Wired for **turn groups** (repo_map). **Collections (tools) and
sub-windows are not yet wired** — tools work well on their raw thresholds, and
normalizing them needs their own threshold re-derivation on the 0–1000 scale
(a tools-scope §85 pass); doing it blind would risk a working path. `layer_weights`
(already shipped) stay upstream — they shape the raw vote; normalization rescales.
Thresholds in `projection.yaml` for the wired group are re-interpreted on the
0–1000 scale (§7) and get a final live confirmation (§4.2 probe-scale caveat).

---

## 6. Unit tests

Build alongside the module (TDD), asserting against exact expected values where
possible:

- `HitLevel`: asymmetric EWMA math — rise-fast/decay-slow against hand-computed
  sequences; cold-start prior; monotonic settling to a constant input.
- `ScopeState::normalize`: a hit at the hit level → ~1000; a 2× hit → ~2000
  (lock-on); a noise-floor score → < threshold; a never-seen child uses the prior.
- Floor: a quiet child cannot amplify a partial match past the floor; floor tracks
  the scope percentile.
- Re-scan eviction: observing a new `(group, timeline)` scope drops that group's
  stale-timeline scope while leaving other groups untouched.
- Per-scope isolation: two scopes with different scales do not cross-contaminate;
  the same child key in two scopes is independent.
- Determinism: same probe stream → same normalized output (no `Date::now`/RNG).
- Regression guard: a synthetic gallery reproducing the root-vs-candle-core shape
  (one loud generic child, one quiet specific child) selects the specific child on
  its query after warm-up.

---

## 7. Calibration (measured, `provenance_calibrate` §85)

Replaying all 56 dialogue turns through the repo_map group scope with hit-level
normalization, cold-start prior, per-scope floor:

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
- **Tools + dialogue scope calibration**: §7 calibrated the repo_map turn-group
  scope; extend the model to the tools collection and a dialogue turn group and
  confirm the same parameters (or per-scope priors) hold.
- **Live threshold confirmation**: re-measure the normalized distribution on the
  live **reprojection** probe (not the whole-turn sig) to pin `min_score` / `evict`
  after the probe-scale shift noted in §4.2.
- **Gallery-change stress test**: inject/remove clusters mid-replay to measure the
  re-settling lag empirically and lock the α values.
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
