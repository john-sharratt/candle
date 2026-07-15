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

- **Turn group** → children are its turns (repo_map dir clusters, dialogue turns).
- **Section collection** → children are its sections (tools).
- **Sub-window** (future) → children are the regions within one turn (files inside
  a repo_map listing — the self-referencing sub-windows).

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
  - `SubWindow { turn: u64 }` — a sub-window within one turn (future).
- `ChildKey` — a candidate within a scope: `Turn(u64)` (turn index — allocation-
  free, stable within a gallery version) or `Named(String)` (tool / section name,
  how `belief_gallery` already keys collection members via `slot_of(tag)`).
- `HitLevel { level: f32, count: u32 }` — the EWMA state for one child.
- `ScopeState { children: HashMap<ChildKey, HitLevel> }` — one scope's state.
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
2. `observe(scope, &[(child, raw)])` — write path. Fold each raw into the child's
   hit-level EWMA (creating it at the prior if new). Runs **once per turn, at
   seal** — hooked into `Conversation::last_turn_belief_scores`
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
prior (correct: their content changed) and rebuild over subsequent queries. The
old scope must not linger, or the cache would leak one `ScopeState` per historical
re-scan. So `NormalizationCache::observe` **evicts a group's stale-timeline
scopes**: after observing `TurnGroup { group, timeline }`, it drops any
`TurnGroup { group, timeline: other }`. The cache therefore holds **one scope per
active turn group** at steady state. This runs once per turn (at seal) over a
handful of scopes — cheap.

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
