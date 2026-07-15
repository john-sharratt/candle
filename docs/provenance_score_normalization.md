# Provenance Score Normalization

**Status:** design (module not yet built). Supersedes the per-node absolute
threshold tuning for provenance selection.

This document captures what we learned diagnosing repo_map retrieval and specifies
the **score-normalization module** that fixes it. It is authoritative for that
module: if the code disagrees, fix the code (or fix this doc in the same change).

---

## 1. The problem

Provenance selection scores each candidate (a tool section, a repo_map cluster, a
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

### 3.3 Per-scope, at every budget node

Normalization is computed **per budget scope**, not over the whole substrate,
because the budget only ever competes *within* a scope. Selection is a tree of
budgets:

- **Layer** node → children are turn groups / top-level turns. Normalizing here
  answers "which conversation/dir" (coarse) and discounts a container that is
  generically loud as a whole.
- **Turn group / section collection** node → children are members (clusters /
  tools / files). Normalizing here answers "which member within" (fine).
- **…recursively** into sub-windows within a turn (the "which file inside this
  listing" case — the self-referencing sub-windows).

Each node keeps its own per-child hit levels, floor, and cold-start prior over
**its direct children only**, so a rarely-touched dir warms up against sibling
dirs and a rarely-touched file against sibling files. A member runs the gauntlet:
it survives the layer cut on its group's layer-normalized score, then the group
cut on its own group-normalized score (effectively multiplicative — a loud member
in an irrelevant container is filtered coarsely before it competes finely).

Because each scope normalizes into the same ~0–1000 band, selection thresholds
become **uniform across scopes** (the "800 for tools vs 200 for repo_map" problem
dissolves), and any layer-level token trim comparing members across groups sees a
common scale for free. Per-node thresholds remain *available* (varied now, a good
uniform default later).

---

## 4. Module design

**Separation of duties:** normalization data is **runtime-derived, in-memory, and
NOT persisted.** It is not part of the substrate/redo-log — it is a cache computed
from query traffic that changes continuously at runtime. At process load it starts
empty and builds up as projections run.

Proposed location: `candle-conversation/src/normalization/` (one concern per
file):

| file | role |
|---|---|
| `mod.rs` | public API: `NormalizationCache`, `ScopeKey`, `normalize`/`observe` |
| `hit_level.rs` | `HitLevel` (asymmetric-EWMA state + update), `Running` mean for diagnostics |
| `scope.rs` | `ScopeState` — per-child hit-level map, floor, prior; `normalize_scores` + `observe` |
| `cache.rs` | `NormalizationCache` — scope map keyed by `ScopeKey`, substrate-generation reconciliation |
| `tests.rs` | unit tests (§6) |

### 4.1 Types

- `ScopeKey` — identifies a budget scope: `Layer(LayerId)`, `Collection(GroupId,
  CollectionId)`, `TurnGroup(GroupId)`, and (later) `SubWindow(TurnIndex)`. Derived
  from the schema node currently being scored.
- `HitLevel { level: f32, count: u32 }` — the EWMA state for one child.
- `ScopeState { children: HashMap<ChildKey, HitLevel>, generation: u64 }` — one
  budget scope's normalization state. `ChildKey` is the member identity within the
  scope (tool name / `TurnIndex` / path tag).
- `NormalizationCache { scopes: HashMap<ScopeKey, ScopeState> }`.

### 4.2 Lifecycle and API

The cache is owned alongside the live projection state (per conversation session /
per substrate handle — TBD in the plan, see §8). Two operations, both driven from
the scan (`score_belief_groups` / `score_belief_collections`):

1. `normalize(scope, &[(child, raw)]) -> Vec<(child, normalized)>` — read path.
   For each child, `1000 × raw / max(hit_level, floor)`; a child absent from the
   scope (never seen) uses the cold-start prior. Pure read; does not mutate.
2. `observe(scope, &[(child, raw)])` — write path. Fold each raw into the child's
   hit-level EWMA (creating it at the prior if new). Called once per reprojection
   after normalization (causal: normalize against pre-update levels, then update).

Selection consumes the **normalized** scores: `belief_step` runs on the 0–1000
values, so belief accumulation, `min_score`, `evict_score`, and the early-decode
window all operate on the normalized scale.

### 4.3 Cache invalidation / substrate change

The cache tracks the substrate generation each `ScopeState` was reconciled at. On
a substrate change (a turn sealed, a repo re-scan, tiering) the scope is
**reconciled, not wiped**:

- **New child** (a candidate that now exists) → inserted at the cold-start prior;
  warms up over its next hits.
- **Removed child** → dropped; the floor recomputes from survivors.
- **Surviving child** → keeps its accumulated hit level. The raw scale may shift
  when the gallery changes (the scorer's `z` term normalizes over all gallery
  tokens), but because the hit level is an EWMA *of raw scores* it rides the shift
  — the ratio `raw / hit_level` is self-stabilizing. A change that *lowers* a
  child's magnitude re-settles slowly (α_dn), which is the deliberate cost of not
  letting rarely-queried children collapse between hits.

> **Decision to confirm:** preserve accumulated hit levels across reconciliation
> (recommended — they self-adapt and preserve learning) vs. reset a child on
> signature change. Recommendation: preserve; add a lazy re-seed of a child only
> when a *near-duplicate* is detected to be stealing its matches, if measurements
> show under-selection-after-shrink.

At process restart the cache is empty and rebuilds from traffic; this is
acceptable because it is diagnostic/selection-shaping state, not ground truth.

---

## 5. Integration points

- `resolver.rs::score_belief_groups` / `score_belief_collections`: after computing
  raw per-member scores for a scope, call `cache.normalize(scope, …)` and hand the
  normalized scores to the selection path; then `cache.observe(scope, …)`.
- `scheduler`: the reproject loop already re-scores every cadence; that is the
  natural `observe` cadence. Submit-time scoring participates too.
- Thresholds in `policy.rs` / `projection.yaml` are re-interpreted on the 0–1000
  scale (§7 calibration). `layer_weights` (already shipped) stay upstream of
  normalization — they shape the raw vote; normalization then rescales per scope.

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
- Reconciliation: new child seeded at prior; removed child dropped; surviving child
  keeps its level and re-settles under a scale shift (synthetic raw ×k).
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
  share or isolate scopes.
- **Layer scope + tools scope**: §7 calibrated the repo_map group scope; extend the
  model to the tools collection and the layer scope and confirm the same
  parameters (or per-scope priors) hold.
- **Gallery-change stress test**: inject/remove clusters mid-replay to measure the
  re-settling lag empirically and lock the α values.
- **Sub-window scope**: normalize file-within-listing selection (self-referencing
  sub-windows) once the coarser scopes are in.

---

## 9. Experiments (reference binaries)

- `zend/examples/provenance_layers.rs` — §83, layer-group weight sweep (produced
  the `layer_weights` tuning).
- `zend/examples/provenance_normalize.rs` — §84, normalizer bake-off (rejected the
  gallery-intrinsic normalizers).
- `zend/examples/provenance_calibrate.rs` — §85, per-scope running-mean/hit-level
  model over the whole substrate (produced §7).
