# Provenance-Adaptive Projection — Scoring Coverage, Attention Budgets, Turn Locality, Anchors, Momentum, Probe Composition, Consensus Fusion

**Status:** design — approved direction, pre-implementation. Revised 2026-08-02
with the experiment-battery evidence (§2.4); v1's Concepts A–E stand refined,
Concepts F and G are new and measurement-driven.
**Branch:** `provenance-projection`.
**Prereq reading:** `docs/tool_selection_provenance_results.md` (§23+ shipped design),
`docs/provenance_score_normalization.md`. Companion harness:
`candle-conversation/examples/selection_experiments.rs` (§11.4).

**Hard invariant — generalization.** Every mechanism in this design is
layer-agnostic; anything that makes one layer behave differently from another is
a **configuration item** (policy, group, or layer level), and every such setting
must be meaningful when applied to a completely different layer. Differences
between axes — including staged rollouts like the tools-axis fusion gate (§9.2)
— are YAML *values*, never code paths keyed on layer, group, or collection
identity. A knob that could only ever make sense for one layer is a design
smell: restate it in layer-agnostic terms (this is how "bring in the code file's
first turn" became `anchor {member: first}` — any timeline's first exchange —
and how the dialogue floor is nothing but `min_percent` + `selection: Sequence`).
Every knob defaults to today's behavior so absent config means unchanged
semantics.

---

## 1. Motivation — evidence from a live conversation

A live daemon conversation (`c0294996`, "Codebase Tour Overview", 2026-08-02, 340 GB
substrate with the full repo ingested: repo_map 4 convs / 142 K tokens, code_reading
1 873 convs / 64 M tokens) produced these failures, all visible in its persisted
projection events:

| Turn | Outcome | Projection evidence |
|---|---|---|
| "Give me a tour of the codebase" | Confabulated "TLS/SQL session test suite" | `structure` turns WERE selected — but they were ~27-token cluster stubs (two of ~460 tokens late in decode), drowned by the tools catalog + random scopes the model narrated as the codebase. (Earlier analysis said "zero structure turns"; the exported events corrected this — the GUI just couldn't render them, their conversation being token-less.) |
| "tell me what the ModelBuilder struct does" | "no codebase context has been provided" | Freshly selected scopes were `candle-nn/src/ops.rs` **lines 1275–1276 (a two-line `#[cfg]` fragment)** and `zend/src/session.rs` 1285–1380 (a *call site* of `.builder()`). The definition in `candle-conversation/src/models/builder.rs` never surfaced. |
| "what questions did i ask" | Dropped the first question | 12 irrelevant scope turns interleaved with 7 dialogue turns diluted self-history. |
| Scope turns generally | Model believes the *user pasted* the code | Scope turns render as raw calibration dialogue ("Summarize `glm4/main.rs` …" + `<tool_response>`), so the model attributes them to the user. |

Five mechanical causes, each addressed by one design concept below (the first
three identified from the recorded events, the last two measured by the
experiment battery, §2.4):

1. **Selection relevance at chance for content turns.** The scopes group competes
   under a near-zero gate (`default_policy: high_recall_scope`, min 40 on a 0–1000
   band) and the collection path isn't normalized at all — scores are not comparable
   where it matters, and weak/no gates admit junk. → **Concept A.**
2. **Static budgets ignore what the scan is asking for.** repo_map is capped at
   `max_percent: 5` regardless of a tour-shaped probe screaming for structure;
   the scopes group holds `top_k k:4` regardless of whether the probe cares about
   code at all. → **Concept B.**
3. **Hits land on fragments, not units of meaning.** A probe that hits one scope of
   `builder.rs` selects that scope alone — without the neighboring scopes, and
   without the file header that carries the imports and module doc. → **Concepts C
   and D.**
4. **The probe contaminates itself.** The trailing probe window evicts the user's
   question after ~256 decode tokens, after which the probe is the model narrating
   whatever junk is already projected — a feedback loop that re-retrieves the junk
   (measured: the top junk slot's score grows 415 → 3 663 across one turn's decode
   while the true target sinks). → **Concept F.**
5. **Additive late fusion lets one fold group outvote the rest.** The dominating
   junk slot's entire score is an identity-group spike with **zero** content-group
   agreement; summing group votes lets it bury balanced true matches. →
   **Concept G.**

Concept E (momentum) is a forward-looking addition to the carry machinery, not a fix
for an observed failure — and §2.4's contamination finding constrains it.

---

## 2. Current state (verified against the tree)

This section is the ground truth the design builds on. File:line refs are to the
`provenance-projection` branch head.

### 2.1 Scoring pipeline

- **Index side:** every sealed turn stores a per-token folded `sign(Q)` signature
  (`WideQSig`, fold `[46,1,1]` layer-groups × 4 heads = 1536 bits,
  `provenance/wide_sig.rs:75-111`), captured at seal
  (`scheduler/mod.rs:6346`, blob record `RecordType::WideQSig = 17`).
- **Query side:** every reprojection gathers a live probe — optional query-head
  chunks + the trailing ≤ 256 decode tokens of the current view
  (`scheduler/mod.rs:6923-6945`). At turn open the tail **is the user's question**;
  mid-decode the tail is the model's reasoning. Both probe kinds the user asked for
  already exist; nothing new is needed on the capture side.
- **Scorer:** `score_provenance_late_fusion_weighted` — per-token per-fold-group
  `z × margin` vote, needle gate keeps top 25 % of query tokens
  (`provenance/scan.rs:81-183`). Group votes are **summed** (additive late
  fusion) — the vulnerability Concept G addresses.
- **Section collections** (tools): `score_belief_collections`
  (`projection/resolver.rs:467`) → `set_section`. **Raw votes, no normalization.**
  Thresholds (`committed_tool_scope` min 800 / evict 600) are on the raw scale.
- **Turn groups** (repo_map clusters, code scopes): `score_belief_groups`
  (`resolver.rs:820`) → per-exchange sub-windows → **hit-level normalization**
  (`normalization/`, EWMA α↑ 0.30 / α↓ 0.02, 0–1000 band, scope =
  `TurnGroup{group, timeline}`, child = exchange head-turn) → `set_turn`
  (`resolver.rs:1115`). `observe` (learning) fires once per turn at seal;
  reprojections are read-only.
- **Selection:** both axes converge on `belief_step` (RelLeak β + hysteresis
  min/evict + `GroupBudget{min,max}`, `provenance/gather.rs:80`), with cross-turn
  carry `PriorBelief` (decayed ×0.5 at turn boundaries,
  `scheduler/mod.rs:683,2651`), a turn-boundary challenger
  (`project.rs:441`), and an early-decode carry floor ("lock-on",
  `policy.rs:172`, `selection.rs:119`).
- **Layers are not scored.** Layers are token-distribution only: two-level flexbox
  (`project.rs:1385-1454`, `reconcile.rs:131`) splits `layer.window` by static
  `budget {priority, min_percent, max_percent}`.

### 2.2 Linked turns (already shipped)

Exchange coupling (`TurnCoupling` record, `summary_tree/exchange.rs`): a tool-call
turn and its response turn score as **one case** and are selected/carried together
(`resolver.rs:942-979, 1108-1118`). This is the existing "directly linked turns drag
each other in". Concepts C and D extend the same idea outward: C to *neighboring*
exchanges, D to the *file-head* exchange.

### 2.3 What the YAML can express today

`layer.budget {priority, min_percent, max_percent}`; group `selection`
(`always_visible | top_k | single | named | conversation`); `policy` blocks
(preset, β, min/evict, early-window, budget min/max, tags, layer_weights);
group `default {tag}` (empty-selection fallback, used by repo_map's root cluster).
Nothing expresses: normalization applicability, adaptive budgets, locality, anchors,
momentum, probe composition, or fusion mode. All seven get YAML surface in §10.

### 2.4 Measured: the experiment battery + selection inertia (2026-08-02)

Two measured foundations the concepts build on. Both are reproducible offline
against the checked-in fixture (`tests/selection_replay_data/`), no model, no
substrate load. The full experiment narrative with per-round numbers is
tracked as `docs/tool_selection_provenance_results.md` **§25**; this table is
the design-facing summary.

**Selection inertia.** The first fixture replay (`tests/selection_replay.rs`,
310 recorded projection points) measured that only **~7 %** of the daemon's
recorded memory-tier winners re-rank near the top of the candidate pool under
instantaneous raw scoring. The recorded selected sets are therefore mostly
**retention** — belief accumulation + hysteresis keeping earlier picks alive —
rather than fresh per-probe signal. Consequences: junk persistence is as much a
carry/eviction-tuning problem as a scoring problem (Concept E and the β/evict
band must be tuned against this measured inertia); and replaying the *recorded*
selections offline requires the full normalize + belief chain (the §11 harness
against the substrate snapshot), while the fixture-level golden
(`baseline_golden.json`) pins the raw-signal layer the TDD targets score with.

**The experiment battery** (`examples/selection_experiments.rs`, experiments
E1–E7 and R2a–R2d over the captured probes and galleries):

| # | Finding | Consequence |
|---|---|---|
| F1 | The dominating junk is a **single-group spike**: `session.rs#35` scores 2 004 = L0–45 **0.0** + L46 1 160 + L47 861 — zero content-group agreement. Per-slot **min across groups** sends it #1 → #227 and lifts builder.rs #10 → #2 (balanced 23/58/58). | Concept G |
| F2 | The **question-time probe is weak but right; the decode tail is strong but wrong**: at t = 0 structure already ranks #2 (score 8.7); by t = 448 the junk tops at 3 663 and structure sits #21 — the tail echoes the projected junk. The trailing window fully evicts the question after 256 decode tokens. | Concept F |
| F3 | **Question probe × min-consensus solves both red cases**: tour → structure cluster rank 1; ModelBuilder → `builder.rs#0` (the file-header exchange) rank 1, over every junk slot the daemon selected. | Concepts F + G, delivery order |
| F4 | **Sum-mass is inverted; concentration discriminates.** Raw code-slot mass: recall probe 5 591 > code probe 3 470 (wrong direction). Normalized sum barely flips (203 vs 190). Top-1 share: 0.579 (code) vs 0.368 (recall); entropy 1.85 vs 2.55 — strong, correct contrast. | Concept B |
| F5 | **Hit-level normalization fixes promiscuity but amplifies quiet fragments.** Score/level flips the tour ranking to structure #1 (from #23) — but the normalized ModelBuilder winner becomes `ops.rs#47`, the same two-line fragment the daemon actually selected live: near-zero-level children explode when divided by their own level. Floor sweep: tour wants floor ≤ 2, ModelBuilder wants ~10 — no single global floor. | Concept A (level prior) |
| F6 | **Group weights are axis- and direction-specific.** repo_map's tour signal lives in **L0–45** (structure best #2 under `[1,0,0]`) and dies under `[0,1,1]` (score exactly 0). The §83 id-group weighting is the wrong direction for content retrieval. | Concept G, per-axis config |
| F7 | **The gallery is healthy; failure shortens probes.** builder.rs self-match 8/8 top-1 against the junk pool — the miss is probe- and fusion-side, not storage. The failed ModelBuilder turn has only 88 sig tokens (a bad projection produces a short refusal answer produces a weak probe). | Concepts F, C/D sequencing |
| F8 | **Every non-additive fusion destroys the tools axis** (190-probe LOO over the fixture's 95 tools): additive 83.7/84.2 % Top-1, `consensus_min` 34.2 %, `consensus_geo` 48.9 %, `content_gated` 26.3 % — tool identity *is* an id-group spike with no content agreement, by fold design. Per-axis fusion values are mandatory, and the tools axis keeps `additive` (fixture-level answer to the §9.2 gate; the ported `belief-eval` full-corpus run is the final confirmation). | Concept G |
| F9 | **`consensus_min` is recall-fragile on large pools; `content_gated` is the content-axis operator.** On the full 406-slot pool, min zeroes every structure/builder slot (a slot must lead all three groups to survive — vanishingly rare as candidates grow), while `content_gated` (id-group votes count only when the content group agrees) preserves additive magnitude and recall and still kills pure id-spikes. | Concept G |
| F10 | **The full pipeline — `content_gated` → hit-level normalize → per-slot max over Q/D windows — ranks both targets #1** (tour → a structure cluster; ModelBuilder → a builder.rs exchange), holds rank 1 across the whole decode trajectory (the contamination takeover of F2 is gone), and satisfies **all three ideal mass orderings at every swept `k`/`ρ`** — recall-vs-code collapses 8.9× (was inverted). Per-window unit-max scaling *hurts* (re-inflates the weaker window's junk): Concept A's level normalization alone is the correct window equalizer, exactly as F.1 claims. Generalizes: structure ranks 1–5 on 28 of 30 captured dialogue turns (was 3–230). | Concepts F + G + A composed; B constants |
| F11 | **Offline Q-window reconstruction pitfall:** recorded event `start_token`s count *view* tokens and exceed the sealed sig length on long turns, so deriving the user-prefix length from them degenerates to a 1-token window. The fixture approximates the Q-window as the turn's head 64 sig tokens; production must read the persisted boundary, never derive it. | Concept F, §11 harness |
| F12 | **The question boundary is ALREADY persisted.** `TurnDecl.segments` carries `User { kv: KvSpan }` — the user half's exact span in the turn's real-KV grid — and `gather_wide_sigs`' contract is one sig per real token, 1:1 with that grid. The Q-window is `sigs[user_span.range()]`; no new record or field is needed, and `export-replay` now emits `user_spans` so offline replay reconstructs the exact window. | Concept F (F.1) |
| F13 | **Momentum is rejected by measurement.** Simulating the per-event selection sequence under the full pipeline with a velocity term: μ > 0 never raises target-top1 (tour 16/18 unchanged), *lowers* it on the ModelBuilder turn (4/7 → 3/7 — velocity locks in an early junk riser), and amplifies top-1 churn on a no-target turn (1 → 4 distinct winners). The rising-interest-lost pattern does not exist once F + G + A stabilize the target; no momentum plumbing is built. | Concept E (closed) |
| F14 | **The root cluster never wins organically** — across all 30 dialogue turns its within-structure rank is 2–22 (median ~10); topic-specific clusters win, which is correct for specific questions. `default {tag "."}` is therefore the *load-bearing* mechanism for root presence, not a backstop, and the tour composition is floor + `k = 2` organic picks. | §13, repo_map config |
| F15 | **The short-probe promotions are tool-shaped questions** ("what time is it?", 24–28 sig tokens): no code slot is right for them, so the code-axis top is arbitrary — and their absolute offline scores overlap genuine code hits, so within-axis gating alone cannot close it. The production discriminators are cross-axis (the tools collection wins the mass for these probes — the guards prove the signal) and the real 0–1000 band (levels learned from strong self-matches separate one-off spurious matches). Verification is a Phase-1 harness acceptance criterion, not a new mechanism. | Concept B, §11 harness |
| F16 | **The multi-segment `belief-*` port is built and confirms everything at full-corpus scale** (snapshot, 745 tagged tool turns / 93 tools — the corpus doubled since the 372-turn baseline): additive reproduces the baseline at **Top-1 97.3 % / Top-5 100 % / MRR 0.985**; `content_gated` collapses to **32.9 %** with **66.7 % of probes scoring 0 for their own tool** (tool identity has literally no content-group agreement — the per-axis fusion split is proven, not provisional); **normalization holds ranking exactly** (Top-1 97.3 %) while improving selection at the same nominal gate (exact-1 1.2 % → 35.3 %, mean set 2.95 → 2.04). The normalized `belief-sweep` derived Concept A's threshold table: true-tool scores sit at p25 ≈ 949 / p50 ≈ 1394 on the band; at budget 3, `min ≈ 60–80` holds the 99.2 % recall ceiling at ~50 % exact-1, `min ≈ 949` trades to 97.2 % recall at 94.8 % exact-1 / 0.06 FP; the budget-3 recall ceiling is 99.2 % at this corpus size (budget 5 → 100 %). | Concepts A + G, §12 Phase 1 |
| F18 | **The implementation round (R5 + acceptance, 2026-08-03) locked the shipping chain and overturned three v2 details.** (a) `ContentGated`'s law is the **grouped sum** (per-group needle-gated tallies, gate on the content group, `Σ_g w_g·t_g`); the full-additive-gated-by-one-hot variant collapsed the target to the pool bottom and is rejected. (b) Gated-fusion axes normalize **traffic-relative**: the A.4 floored path divides by the child's observed-traffic **peak** (floored by size), and `warm_ingest` self-match warming is skipped for non-additive-fusion groups (config-keyed) — self-levels would erase the quiet-child standout the design requires. (c) Concept B's mass keys on the **ungated** additive sum with `k = 1, ρ = 2` — the gate deliberately removes the concentrated spike mass must see, and the normalized band compresses it. Acceptance through the production chain: tour → structure **#1**; ModelBuilder → builder.rs **#3 = inside the top_k 4 selection budget + anchor** (vs absent entirely live; the two slots above are same-repo test fixtures sharing the probe's vocabulary — strict rank-1 relaxed to the selection-level criterion); recall-vs-code mass contrast 0.72× on the ungated formula. | Concepts A.4 + B + G as shipped |
| F17 | **Code self-match at corpus scale is regime-dependent, and normalization inverts on cold levels.** Bounded 600-turn LOO (single-turn-file probes excluded — and in these runs excluded from the gallery too, leaving ≈160 multi-turn file slots as the effective pool): raw Top-1 **57.1 %** (chance ≈ 0.6 %) with a 20 % zero-self floor — cross-scope same-file retrieval is genuinely hard, which independently validates C + D (neighbors must be dragged in, not expected to score). `--normalize` **drops** to 47.5 %: with each file observed once or twice the causal pass divides by learning-starved levels — quiet-slot amplification (F5) at corpus scale. Not an A refutation (tools, observation-dense, held exactly; production warm-replays levels) but the proof that **A.4's level prior is load-bearing for cold scopes** (newly ingested files). Warm-level code verification = the §11 harness (Phase 1). | Concepts A (A.4), C + D, §12 Phase 1 |

---

## 3. Concept A — universal scoring on one normalized scale

**Goal:** every belief-driven axis — section collections *and* turn groups — scores
on the same 0–1000 hit-level-normalized band, so (a) gates mean the same thing
everywhere, (b) Concept B can compare attention *across* layers and collections.

### A.1 Normalize the collection path

`ScopeKey::Collection{group, name}` already exists in the normalization keyspace
(`normalization/mod.rs:38-46`) but is never invoked. Change
`score_belief_collections` to mirror the turn-group path:

- scope = `ScopeKey::collection(collection_name)`, child = `ChildKey::Named(section
  name)`;
- `normalize` on every reprojection, `observe` only at seal (same
  read/learn split as turn groups, `resolver.rs:1100-1107`);
- warm replay (`ensure_normalization_warm`, `resolver.rs:648`) extends to collection
  scopes so a daemon restart does not reset tool scoring to priors.

**Threshold migration.** `committed_tool_scope` min 800 / evict 600 are raw-scale;
under normalization a *sustained strong* match sits near 1000 by construction —
**measured (F16): true-tool normalized scores land at p25 ≈ 949 / p50 ≈ 1394.**
The migration is now derived, not pending: normalization holds full-corpus
ranking exactly (Top-1 97.3 % raw and normalized, 745 turns) while improving
selection at any given gate (exact-1 1.2 % → 35.3 % at the same nominal
`min_score`), and the normalized `belief-sweep` table gives the operating
points — budget 3 with `min ≈ 60–80` keeps the 99.2 % recall ceiling at ~50 %
exact-1; `min ≈ 949` trades to 97.2 % recall at 94.8 % exact-1 / 0.06 mean FP.
One corpus-growth note: the budget-3 recall ceiling is 99.2 % at 745 turns
(budget 5 recovers 100 %) — the Phase-3 commit picks the recall/precision
point and possibly `budget.max 4–5` from this table. The raw scorer path dies
with the migration: no dual thresholds, no per-collection opt-out.

### A.2 Right dimensions — the normalization contract

Normalization divides each child's raw vote by that child's own learned strong-match
level (per-child hit-level), then scales to 0–1000. The dimension choices, stated
explicitly (today's behavior, kept, now universal):

| Dimension | Choice | Why |
|---|---|---|
| Learning key (child) | per exchange (turn groups) / per section (collections) | a promiscuous child (changelog file, verbose tool) self-mutes; a quiet child's rare hit stands out |
| Scope | per `(group × timeline)` / per collection | levels learned where the competition happens |
| Cross-scope comparability | shared 0–1000 output band | enables cross-layer mass comparison (Concept B) |
| Not normalized per-probe | — | needle gate + z already handle probe-side variance |

### A.3 Scoring diagnostics (the "layers that are not scoring" audit)

A populated layer that never scores is silent today. Add per-reprojection
diagnostics (behind the existing `SelectionDiagnostics` sink):

- WARN-once per (layer, group) when the gallery is empty despite the group having
  candidate turns (extends the existing empty-collection warning,
  `resolver.rs:488-522`);
- WARN-once when a scope has never been `observe`d (normalization cold) after its
  first N reprojections;
- the projection event gains nothing — `SelectedTurn.score` / `SelectedSection.score`
  already record the normalized values.

Layers with `conv_count = 0` (static_analysis … bug_analysis) cannot score and are
out of scope here; they start scoring the moment their ingest phases populate them,
through exactly this path.

### A.4 Level prior — the quiet-slot guard (measured requirement, F5)

Dividing by a near-zero learned level turns a fragment's chance match into a
top rank: the normalized ModelBuilder winner in the battery was the very
two-line `ops.rs` fragment the daemon selected live — the recorded junk is
largely *this mechanism*, not raw-score error. The floor under the level is
therefore not a numerical epsilon but a first-class prior, and the measured
optima conflict (tour ≤ 2 vs ModelBuilder ~10) shows a single global floor
cannot work. The prior is **size-aware** — a child's opportunity to learn a
level scales with its token count, so small children get proportionally higher
floors:

```
level_floor(child) = floor_base × clamp(T_ref / tokens(child), 1, floor_cap)
level_eff(child)   = max(level_learned(child), level_floor(child))
```

- `T_ref` = the reference window (`reproject_max_probe_tokens`, 256);
- a two-line fragment (`tokens ≈ 30`) gets ≈ 8.5 × `floor_base`; a full scope or
  cluster window (`tokens ≥ 256`) gets exactly `floor_base` — muting fragment
  noise without touching well-sampled children;
- `floor_base` and `floor_cap` are derived by harness sweep (§11), per preset,
  before the threshold migration in A.1 is committed. **Measured (rounds 3 +
  X7): with Concept G's content gate upstream, the fixture ranking is
  insensitive to the floor across `base ∈ [0.5, 5] × cap ∈ [1, 16]` (the gate
  suppresses fragment spikes — `ops.rs` sits ~#74 at every floor) — but the
  prior is NOT optional polish: the corpus-scale code eval (F17) showed
  normalization *inverting* (57.1 % → 47.5 % Top-1) when levels are cold
  (files observed once or twice), which is the exact regime every newly
  ingested file passes through. The floor is what keeps a cold scope's
  scoring no worse than raw until its level accrues; shipped values
  `floor_base 2.0 / floor_cap 16`;**
- **shipped semantics (F18): a child with a positive size floor normalizes
  against its observed-traffic PEAK, floored by the size floor** — not the
  EWMA level, whose prior seed and slow decay block the quiet-child standout
  the contract requires. A promiscuous child's peak is high (it hits on
  everything), so its cross-hits mute; a quiet child's rare genuine hit lands
  near its own peak and stands out. Correspondingly, `warm_ingest`'s
  self-match warming **skips non-additive-fusion groups** (config-keyed):
  self-levels would stamp every file's peak at its self-match magnitude and
  erase the contrast — the content gate already solves the promiscuous
  domination that warm-up existed to fix;
- this also resolves §13's "structure stubs" question at the scoring level: a
  ~27-token stub cannot win by quiet-slot amplification, only by genuine
  agreement.

---

## 4. Concept B — provenance-adaptive budgets

**Goal:** the scan redistributes token budget and member budget toward where the
attention mass is. A tour-shaped probe lifts repo_map above its static 5 %; a
code-shaped probe grows the scopes budget; a tools-shaped probe grows the tools
`k` — each within YAML-declared bounds.

### B.1 Attention mass (concentration-weighted — measured requirement, F4)

A plain sum over scores is **measured invalid** as a budget signal: a diffuse
history probe generates *more* raw code-slot mass (5 591) than a genuine code
probe (3 470), because generic probe tokens light up promiscuous slots
everywhere. What separates the two is **concentration**: the code probe puts
0.579 of its mass on its top slot, the history probe 0.368. Mass is the sum
*scaled by* the top-share concentration:

```
sum(g)  = Σ_i  s_i                       (the UNGATED additive group sum — see below)
conc(g) = Σ_{top-k} s_i / max(sum(g), ε)     (k = 1)
mass(g) = sum(g) × conc(g)^ρ                 (ρ = 2)
```

**Shipped basis (F18):** mass keys on the **ungated raw additive sum** of the
same per-group tallies the fusion consumes (zero extra scan cost) — NOT the
normalized or gated scores: the content gate deliberately removes the
concentrated spike mass must see (measured: gated mass inverted the contrast),
and traffic-peak normalization compresses it. `k = 1, ρ = 2` is the measured
operating point (code 3 470 × 0.579² = 1 163 vs recall 5 591 × 0.368² = 757 —
a 0.65–0.72× contrast, direction correct with margin).

Comparable across nodes and near zero when the mass is smeared across the
gallery (a diffuse probe self-mutes). `k` and `ρ` are policy-level settings
(inherited from `default_policy`, overridable per group/collection policy like
every other policy knob). (F10's earlier finding — normalized-band mass with
any `k`/`ρ` — held under the round-3 mean-level approximation but NOT under
the shipped traffic-peak normalizer, which is why the ungated-raw basis above
is what ships; F18.) The acceptance is the mass red test
(`recall_probe_code_mass_collapses_relative_to_code_probe`). Computed inside
`score_beliefs` and carried on `ProjectionScores` (new field
`group_mass: HashMap<GroupKey, f32>`), so it rides the existing
`read_for_scored` plumbing into `project()` — selection stays a pure function of
its inputs.

### B.2 Token-budget modulation (flexbox)

Flexbox priorities become mass-modulated. For each layer ℓ with adaptive budgeting
enabled:

```
effective_priority(ℓ) = priority(ℓ) × (1 + gain × Σ_g mass(g) / 1000)
```

clamped by the layer's declared `min_percent` / `max_percent` — the YAML bounds stay
the outer authority; adaptivity only moves allocations *between* those rails. Same
formula one level down for groups inside a layer. `FlexItem` construction
(`project.rs:1424-1454`) is the only integration point; `flexbox_distribute` itself
is untouched.

### B.3 Member-budget modulation (belief budget / top-k)

`GroupBudget.max` (and the TopK→budget mapping in `belief_config`,
`schema.rs:918-933`) gains a mass-driven extension:

```
effective_max = clamp(base_max + floor(mass / per_extra), base_max, absolute_max)
```

`per_extra` = mass required per additional member; `absolute_max` from YAML. The
selector (`SectionSelector::apply_selection`) already takes budget as an input —
no selector change.

### B.4 What adaptivity is *not*

- Not a gate: a layer whose mass is zero keeps its `min_percent` floor and its
  `default` fallback; adaptivity can only shrink toward the floor, never below.
- Not stateful: mass is recomputed from the current probe each reprojection.
  Persistence of interest across reprojections is the belief carry's job (and
  Concept E's).
- Not cross-target: any group whose configured `selection` is `Sequence`
  (recency-driven, not belief-driven) has no mass and does not participate; its
  layer's `min_percent` floor stands like every floor. The rule keys on the
  configured selection kind, never on which layer it is — the zend dialogue
  floor (`Sequence`, priority 100, min 50 %) is an instance, not a special case.

---

## 5. Concept C — turn locality

**Goal:** a hit on exchange *e* of a timeline pulls its neighbors *e±1* (and, under
massive attention, further) into contention, because code meaning is spatially
local — the scope above/below the hit is usually the context the hit needs. The
mechanism is group-agnostic: it operates on timeline adjacency, which every turn
group has; a `locality` block on a dialogue-history or repo_map group means
exactly the same thing (adjacent exchanges ride along). Code scopes are the
motivating instance, not a condition.

**Sequencing note (measured, F3/F7):** locality amplifies *selected* seeds — it
cannot create them. Under today's fusion the best builder.rs exchange ranks #8
with a per-exchange median of 0, so there is no seed to amplify; under F + G the
seed ranks #1–2. C therefore lands after Concepts F, G, and A (§12).

### C.1 Mechanism

Operates on **exchange** granularity (post-coupling — locality extends the shipped
pair-coupling outward), inside `score_belief_groups` after normalization, before
`set_turn` / `belief_step`:

```
for each exchange e with score s ≥ locality.seed_threshold:
    radius = base_radius + min(extra_radius_max,
                               floor((s − seed_threshold) / extend_per))
    for d in 1..=radius:
        for n in {e−d, e+d} within the same timeline:
            score(n) = max(score(n), s × decay^d)
```

- `max`, not sum: two adjacent hits don't double-count into a runaway; a neighbor
  that also scored on its own merits keeps its own (higher) score.
- Radius grows with attention: `s = 700` with `seed_threshold 600, extend_per 200`
  → radius 1; `s = 1000` → radius 2 — "massive attention could bring in more."
- Boosted neighbors enter `belief_step` as ordinary fresh scores: they compete for
  budget, are carried by `PriorBelief`, decay normally. No special selection state.
- Origin: neighbors that were selected *only* because of the boost are stamped with
  a new `SelectionOrigin::Locality` so projection events show why they're present.

### C.2 Interaction with budgets

Locality raises candidate scores, which raises `mass(g)` (Concept B) — a strong
code hit therefore *also* argues for more code budget. That is intended: locality
says "this neighborhood matters", mass converts it into room. The clamps in B.2/B.3
bound the feedback.

---

## 6. Concept D — file-head anchor

**Goal:** whenever any exchange of a code file's conversation is selected, the
file's *first* exchange rides along — it is the `ChunkKind::FileHeader` scope
(module doc / license / file-overview comment, `zend/src/code_read/types.rs:22-28`,
always the first scope, never merged forward), followed by the import scopes. The
model reading `builder.rs:250-310` without the imports and module doc is reading
context-free code.

**Validation (measured, F3):** under question-anchored consensus scoring the
*organic* top-1 for the ModelBuilder probe is `builder.rs#0` — the file-header
exchange itself. The header carries the file's identity signal; the anchor rides
a real regularity, not a heuristic hope.

### D.1 Mechanism

Post-selection injection in the phase-1 group loop, structurally parallel to the
existing `default` fallback (`project.rs:1237-1244`):

- for each timeline T with ≥ 1 selected exchange in an anchored group, if T's first
  exchange is not already selected, inject it with score = the max score among T's
  selected exchanges (it travels *with* the hit, not above it), origin
  `SelectionOrigin::Anchor`;
- the anchor participates in token accounting normally, but
  `trim_to_budget_low_score_first` treats it as paired to its timeline's best
  exchange: the anchor is only trimmed when every other exchange of its timeline
  has been trimmed first (trim key ordering, no new trim pass);
- belief carry: the anchor is stamped into `PriorBelief` like any selected turn, so
  it persists while its file stays live and decays out with it.

### D.2 Non-goals

- Not a summary substitute: the anchor is the real header exchange, full fidelity.
- Not unconditional: a timeline with zero selected exchanges gets no anchor
  (that is `default {tag}`'s job, which stays as-is for repo_map).

---

## 7. Concept E — score momentum (measured and REJECTED)

**The idea:** interest *decays* on three clocks — RelLeak β per reprojection,
×0.5 carry decay per turn boundary, hysteresis eviction — and nothing rewards
*sustained growth*, so a per-slot velocity term
(`v ← γ·v + max(0, Δscore)`, `effective_seed = score + μ·v`) was proposed to
lock rising interest in faster.

**The measurement (F13):** simulated over every recorded reprojection sequence
of the tour conversation under the full F + G + A pipeline, at
μ ∈ {0, 0.5, 1.0}, γ = 0.5:

| turn | μ = 0 target-top1 | μ > 0 target-top1 | top-1 churn μ = 0 → μ > 0 |
|---|--:|--:|--:|
| tour (18 events) | 16/18 | 16/18 (no change) | 2 → 2 |
| ModelBuilder (7 events) | 4/7 | **3/7 (worse)** | 2 → 2 |
| recall (14 events, no code target) | — | — | **1 → 4 (worse)** |

Momentum never helps and twice hurts: the velocity term locks in early junk
risers and amplifies churn on diffuse turns. The pattern it exists to fix —
rising interest lost to noise — **does not occur** once the pipeline
stabilizes the target (the F + G + A trajectory holds rank 1 across the whole
decode, F10). The pre-pipeline instability that motivated the idea (E6's junk
takeover) was the contamination loop, and Concept F removes it at the source.

**Stance:** no momentum plumbing is built; no YAML surface ships. The
mechanism and its measurement are recorded here so it is not re-proposed.
Revisit only if the §11 production-faithful harness ever shows target
instability across reprojections that the pipeline does not already absorb —
that instability, not intuition, would set the gain.

---

## 8. Concept F — question-anchored probe composition

**Goal:** the user's question keeps voting for the whole turn. Today the probe is
one trailing window; after ~256 decode tokens the question has scrolled out and
the probe is purely the model's own narration — which describes whatever the
projection currently contains, junk included. That feedback loop is measured
(F2): the question-time probe ranks the true target #2 with almost no magnitude,
then the tail probe buries it under an ever-growing echo of the junk.

### F.1 Mechanism

The probe becomes **two windows, scanned separately**:

- **Q-window** — the current turn's user-authored prefix tokens (the question),
  captured once at turn open, **pinned unchanged for the whole turn**, capped at
  `reproject_max_probe_tokens`;
- **D-window** — the trailing decode tail, exactly today's probe.

Each window runs the full scan (fold-group scoring, needle gate, Concept G
fusion, Concept A normalization) independently, producing two per-slot score
vectors on the normalized band. They fuse per slot by **max**:

```
score(slot) = max(score_Q(slot), score_D(slot))
```

- `max`, not a weighted sum: the battery measured that mid-range blends can rank
  *worse* than either pure window (each window's junk is different; summing
  admits both). Max keeps whichever window found the needle and ignores the
  other's noise for that slot.
- Per-window normalization (Concept A) makes the two windows comparable despite
  the Q-window's small raw magnitude (a short question generates few votes —
  the failed ModelBuilder turn's whole probe was 88 tokens, F7). **Measured
  (F10): level normalization alone is the correct equalizer — an additional
  per-window unit-max scaling step was tested and *degrades* both targets
  (#1 → #2), because rescaling a weak window re-inflates its junk to parity.
  No per-window scaling step exists in this design.**
- The Q-window boundary needs **no new persistence — it is already stored**
  (F12): `TurnDecl.segments` carries the user half's exact `KvSpan` in the
  turn's real-KV grid, and `gather_wide_sigs` emits one sig per real token,
  1:1 with that grid — so `Q-window = sigs[user_span.range()]`, live and
  offline alike. What is *forbidden* is deriving the boundary from event
  `start_token`s (view-token counts that exceed the sig length on long turns,
  F11). `export-replay` emits `user_spans` per turn so the replay harness
  reconstructs the exact window; turns with multiple user segments
  concatenate their spans, capped at `reproject_max_probe_tokens`.
- The Q-window is static per turn, so its scan result is computed once per turn
  per gallery and cached across that turn's reprojections — the steady-state
  per-reprojection cost is one scan (the D-window), same as today.
- Downstream consumers see one fused vector: `belief_step`, mass (Concept B),
  momentum (Concept E, which *requires* the fused signal), locality seeds
  (Concept C).
- Diagnostics: `SelectionDiagnostics` records which window won each selected
  slot, so contamination (D-window-only selections that the Q-window scores at
  zero) is visible per projection event.

### F.2 What the Q-window is not

- Not a recency re-weighting of one window: the question must survive *eviction*,
  not merely out-vote 250 newer tokens through decay weights — and the needle
  gate is deliberately position-blind.
- Not the belief carry: `PriorBelief` carries *outcomes* (selected slots) across
  reprojections; the Q-window carries the *query itself*. Lock-on keeps an early
  selection alive; F keeps the early *evidence* alive, so a mid-turn challenger
  faces the question, not just the echo.
- Not multi-turn: the Q-window resets at each turn boundary (the prior turns'
  influence remains the belief carry's job).

---

## 9. Concept G — cross-group consensus fusion

**Goal:** identity-group votes only count for a slot the gate (content) group
agrees with — a pure identity spike scores zero. The shipped scorer sums the
three fold groups' `z × margin` votes, so a single
group can produce the whole score: the dominating ModelBuilder junk
(`session.rs#35`, a `.builder()` *call site*) scores L0–45 = **0.0**,
L46 = 1 160, L47 = 861 — a pure identity-layer spike with zero content
agreement, out-voting the true definition whose signal is balanced (23/58/58)
across all three groups (F1).

### G.1 Mechanism

The scan keeps per-group per-slot tallies instead of summing at tally time (the
per-token loop in `score_provenance_late_fusion` already computes per-group
votes; only `needle_gate_tally` collapses them). The needle gate applies
per group over that group's own token-vote magnitudes (as measured — each group
finds its own needle). Fusion across the group vectors is a per-policy mode:

```
additive:        score = single-pass Σ (cross-group needle gate — today, default)
content_gated:   score = Σ_g w_g·t_g  if t_gate > 0, else 0
                 (t_g = group g's own needle-gated tally; gate_group from config)
consensus_min:   score = min_g t_g
consensus_geo:   score = (Π_g t_g)^(1/n_groups)
```

Each per-group tally equals a one-hot-weighted additive scan, so any backend
implementing the additive scan — including the GPU gallery arena — serves the
non-additive modes with `n_groups` scans. (An alternative `content_gated` law —
the full additive score gated by a one-hot gate scan — was implemented,
measured, and REJECTED: the true target collapsed to the pool bottom, F18.)

- **`content_gated` is the measured content-axis operator** (F9/F10): identity
  groups amplify only what the gate group (the content fold, L0–45) agrees
  with at all — "identity confirms, content decides". It kills pure id-spikes
  (session.rs L0–45 = 0 → score 0) while preserving additive magnitude and,
  critically, *recall*: `consensus_min` — the first-round winner (F1) — zeroes
  every true target on the full 406-slot pool, because leading all three
  groups simultaneously becomes vanishingly rare as the candidate count grows
  (F9). The gate group index is config (`fusion: {mode: content_gated,
  gate_group: 0}`), not a hardcoded fold — the invariant holds.
- `group_weights` (already in the API) compose: a weighted group's votes scale
  before fusion, and a zero-weighted group drops out of min/gating (so
  `[1,0,0]` degenerates to single-group scoring, not constant zero).
- A threshold-count form ("2-of-3 groups above τ") was measured and rejected:
  the session.rs spike clears any per-group τ in *two* groups; only requiring
  the *gate* group's agreement kills it.

### G.2 Per-axis scope — and the tools-axis gate

Fusion mode is **one policy key** (`fusion:`), identical in meaning on every
axis; the scan code never branches on which layer, group, or collection it is
scoring. The split below is the first set of YAML *values*, chosen because the
axes are measured to need different directions (F6):

- **Content axes** (code scopes, repo_map clusters): `content_gated`. Content
  identity lives in L0–45; the id-groups contribute confirmation, not the
  signal. repo_map additionally re-derives its `layer_weights` — the §83
  `[0,1,1]` inheritance from tools is measured to zero out structure retrieval.
- **Tools axis**: stays `additive` — measured and now **confirmed at
  full-corpus scale** (F16). Tool identity deliberately lives in L46/L47 (the
  fold design, results doc §23): the fixture LOO showed every non-additive
  operator collapsing tool Top-1 (83.7 % → 13.7–48.9 %), and the ported
  `belief-eval` on the 745-turn snapshot corpus settles it — additive 97.3 %
  Top-1 vs `content_gated` 32.9 %, with **66.7 % of tool probes scoring
  exactly 0 for their own tool** under the gate: two-thirds of tool turns
  have *no* content-group agreement with their own gallery. The per-policy
  mode keeps both axes correct simultaneously.
- The fixture's tool guards (datetime / calculator top-1) run both modes in the
  unit suite from day one — a cheap regression tripwire in both directions.

---

## 10. YAML configuration surface

All seven concepts configure through `projection.yaml`. New blocks, with the zend
values proposed for the first iteration:

```yaml
# Layer level — Concept B (token budget)
- name: repo_map
  budget:
    priority: 5
    max_percent: 5          # static ceiling today…
    adaptive:               # …becomes the *floor-to-ceiling rail* under adaptivity
      gain: 2.0             # priority multiplier per 1000 mass
      max_percent: 25       # adaptive ceiling (overrides static ceiling when mass demands)

# Group level — Concepts B (member budget), C (locality), D (anchor)
  groups:
    - id: scopes
      selection: { kind: top_k, k: 4 }
      budget_adaptive:
        per_extra: 800       # +1 member per 800 mass above the base k
        absolute_max: 8
      locality:
        seed_threshold: 600  # normalized score to start dragging neighbors
        decay: 0.5           # per-step falloff
        base_radius: 1
        extend_per: 200      # +1 radius per this much score above seed_threshold
        extra_radius_max: 2
      anchor:
        member: first        # inject the timeline's first exchange when any is selected

# Policy level — Concepts A (normalization + level prior), F (probe
# composition), G (fusion). No momentum block: Concept E is measured-rejected
# (F13) and ships no surface.
default_policy:
  preset: high_recall_scope
  normalized: true           # all belief scoring on the 0–1000 band (the only mode
                             # after migration; the key exists so presets can carry
                             # re-derived thresholds explicitly)
  level_prior:
    floor_base: 2.0          # per-preset; swept before the threshold migration
    floor_cap: 10.0          # max size-scaling multiple for tiny children
  fusion:                    # additive | content_gated | consensus_min | consensus_geo
    mode: content_gated      # (per axis: content axes gated, tools additive — F8/F9)
    gate_group: 0            # which fold group gates (the content fold for this model)
  probe:
    question_pin: true       # Concept F: pinned Q-window (the persisted user span,
                             # F12) + decode tail, max-fused
  mass:                      # Concept B — mass definition; a policy knob like any
    concentration_top_k: 1   # other (inherited from default_policy, overridable
    concentration_rho: 2.0   # per group/collection policy). Shipped values (F18).
```

Parsing lands in `yaml.rs` beside the existing `parse_selection`/`parse_default`;
validation (rails: `adaptive.max_percent ≥ max_percent ≥ min_percent`,
`absolute_max ≥ k`, `decay ∈ (0,1]`, `gain ≥ 0`, `floor_cap ≥ 1`,
`concentration_rho ≥ 0`, fusion ∈ the three modes) in `validate_policy`'s
sibling. Every knob has a schema-level default equal to today's behavior
(`fusion: additive`, `question_pin: false`, `level_prior` off, static mass), so
existing YAML files parse and behave unchanged until they opt in.

---

## 11. The replay harness — iterating against the captured conversation

The tour conversation is a complete, labeled, adversarial test case, and everything
needed to replay it is already persisted per turn: `WideQSig` blobs (1:1 with
tokens), `ProjectionEvents` (reprojection cadence, per-member scores, selected
sets, origins), `TurnCoupling`, stream decls with tags. `belief-replay` in
`substrate_inspect` already proves the production-faithful replay pattern for the
tools axis (§80.3); this harness generalizes it to the full projection.

### 11.1 `substrate_inspect selection-replay` — BUILT (2026-08-03)

Shipped as a `belief-*`-family subcommand (merged multi-segment substrate,
CPU, model-free):

```
substrate_inspect --log <snapshot> selection-replay <conversation> \
    [--code-tag code] [--structure-tag repo_map] [--gate-group 0] \
    [--top N] [--limit N]
```

Per dialogue turn of the target conversation it runs the **production
content-axis chain** against the real substrate galleries — `content_gated`
grouped-scan fusion (G) → hit-level normalization warmed by self-match with
the A.4 size floors (A) → per-slot max of the head (question, F11) and tail
scans (F) — and prints the top-N selected members per axis plus the
code-vs-`repo_map` attention-mass contrast (B, on the ungated raw sum,
`k=1, ρ=2`). `--limit` caps each gallery (the full snapshot code gallery is
O(100k) turns). This is the tuning loop; the always-on `selection_replay.rs`
acceptance tests are its CI-pinned fixture counterpart.

The original design sketch (kept for reference):

```
selection-replay --conversation <label|timeline> \
                 --expect <expectations.json> \
                 [--yaml <projection.yaml>] [--compare-recorded]
```

Per dialogue turn of the target conversation, in order:

1. reconstruct each recorded reprojection's probe from the turn's stored sigs
   (window bounds from the recorded event cadence — exactly `belief-replay`'s
   slicing), split into Q-window + D-window per Concept F;
2. run the real `score_beliefs` → fusion (G) → mass → locality → anchor →
   `belief_step` → flexbox pipeline against the substrate galleries under the
   candidate YAML;
3. score the outcome against the expectations file;
4. `--compare-recorded` additionally diffs against the historical
   `ProjectionEvents` (what the old config actually selected) — the
   before/after view for every iteration.

Determinism: the harness is pure CPU scoring over stored bytes — no model, no
sampling — so every run is bit-identical and CI-able. Normalization state is
rebuilt by corpus-order warm replay (the shipped `ensure_normalization_warm` path)
so learned levels are reproducible.

### 11.2 Expectations fixture for the tour conversation

`candle-conversation/tests/fixtures/selection_replay_tour.json` (checked in):

```json
{
  "conversation": "Codebase Tour Overview",
  "turns": [
    { "probe": "tour",       "expect": { "layer_hits": ["repo_map"],
        "must_select": [{"group":"structure","tag":"."}],
        "mass_order": ["repo_map", "code_reading"] } },
    { "probe": "datetime",   "expect": { "collection_top1": "datetime" } },
    { "probe": "sqrt",       "expect": { "collection_top1": "calculator" } },
    { "probe": "ModelBuilder", "expect": {
        "must_select_file": "candle-conversation/src/models/builder.rs",
        "anchor_present": true,
        "mass_order": ["code_reading", "repo_map"] } },
    { "probe": "recall",     "expect": { "max_scope_turns": 4 } }
  ]
}
```

Metrics reported per turn and aggregate: hit@budget, MRR of the expected member,
per-layer mass and awarded budget, contamination count (selected members matching
no expectation), anchor/locality origin counts, Q-vs-D win counts (Concept F
diagnostics). The ModelBuilder turn is the headline number: today it selects a
two-line `ops.rs` fragment; the target is `builder.rs` scopes + file-head within
budget.

### 11.3 Unit tests (per CLAUDE.md, built alongside each mechanism)

- **A:** collection normalization — same probe, promiscuous vs quiet section,
  normalized ordering flips vs raw; `observe` at seal only; warm replay
  reproduces levels byte-for-byte; level prior — fragment-sized child gets the
  scaled floor, full-window child gets `floor_base`, learned level above floor
  wins.
- **B:** mass computation (gate boundary, cap, concentration factor — a smeared
  score line masses below a concentrated one of equal sum); flexbox modulation
  respects rails (mass 0 → static shares; mass ∞ → clamped at adaptive
  ceiling); member budget extension exact at `per_extra` boundaries.
- **C:** locality propagation table-driven over synthetic score lines (radius
  growth, decay powers, max-not-sum, timeline boundary, exchange granularity);
  origin stamping.
- **D:** anchor injection (fires on any-selected, not on none-selected; score
  inheritance; trim ordering keeps anchor until its timeline empties; no
  double-inject when the head is organically selected).
- **F:** Q-window pinning (identical Q scores across a turn's reprojections),
  per-slot max fusion (a slot strong in either window survives; mid-blend
  regression case from the battery encoded as a fixture), Q-window reset at
  turn boundary.
- **G:** consensus modes over synthetic per-group tallies (single-group spike
  dies under min, balanced signal survives; geo-mean ordering; zero-weighted
  group drops out of the min); the fixture tool guards (datetime / calculator)
  run under **both** additive and consensus_min.
- Existing suites are regression gates: `belief-eval` (tools Top-1 97.8 %),
  `selection.rs` recency tests (dialogue untouched), projection integration
  tests, and the selection-replay golden
  (`baseline_every_recorded_projection_point_replays` — regenerate + review on
  every intentional scoring change).

### 11.4 The experiment battery (idea iteration)

`candle-conversation/examples/selection_experiments.rs` — the offline battery
that produced §2.4 (E1–E7, R2a–R2d): fixture-loaded once, pure CPU, deterministic.
It is the *exploration* tool (mechanism candidates measured in minutes against
the real probes/galleries); the selection-replay subcommand is the *verification*
tool (production-faithful pipeline). New mechanism ideas go through the battery
first; only measured winners graduate into the design and the production path.

---

## 12. Delivery order

Each phase lands complete (code + YAML + unit tests + replay run) before the next.
Re-sequenced 2026-08-02: F + G move ahead of everything else because they are
measured to solve both ranking red cases (F3) and every later concept consumes
their fused scores.

1. **Harness first:** `selection-replay` + the tour expectations fixture, run
   against today's config — pins the baseline numbers the design claims to fix.
   Includes porting the `belief-*` suite to the multi-segment loader
   (`export-replay` already walks all segments): measured 2026-08-02, `belief-eval`
   on the segmented store inspects only the active segment and finds **0 tagged
   turns** — the Top-1 97.8 % tools baseline is not currently re-verifiable
   offline, and it must be before Concept A's threshold migration leans on it
   **and before Concept G's tools-axis gate can be evaluated** (§9.2).
2. **G + F — content-gated fusion + question-anchored probing** on the content
   axes (`fusion: {mode: content_gated}` on scopes and repo_map policies; tools
   keeps `additive` per the F8 measurement — a YAML-values difference only, the
   same scan code runs on every axis). Scorer- and probe-side only — no
   selection-machinery change; the Q-window reads the already-persisted user
   `KvSpan` from the turn layout (F12). Acceptance: the two ranking red tests
   (`modelbuilder_probe_ranks_builder_rs_over_observed_junk`,
   `tour_probe_ranks_structure_over_scopes`) go green through the production
   scan path; the tool guards stay green in both fusion modes.
3. **A — normalization everywhere** + level prior (A.4) + threshold
   re-derivation (sweep) + scoring diagnostics. Replay: contamination drops
   without quiet-fragment promotion (the `ops.rs#47` regression case from F5 is
   the guard); tools suite must hold.
4. **B — adaptive budgets** (concentration mass → flexbox + member budgets).
   Acceptance: the mass red test
   (`recall_probe_code_mass_collapses_relative_to_code_probe`) goes green; the
   tour turn's repo_map budget grows; the ModelBuilder turn's code budget grows.
5. **C + D — locality + anchor** (one phase: both are "hits select neighborhoods,
   not fragments"; both now amplify real seeds per F3). Replay: ModelBuilder turn
   selects `builder.rs` scopes with file-head present.

Concept E ships nothing (measured-rejected, §7/F13); there is no phase 6.

---

## 13. Open questions — status after the closure rounds (2026-08-02)

**Closed by measurement or static analysis:**

- **Momentum** — rejected (F13, §7). The pattern it would fix does not occur
  under the pipeline; μ > 0 is neutral-to-harmful on every measured sequence.
  Nothing ships.
- **Root cluster / structure composition** — answered (F14): the root never
  wins organically (within-structure rank 2–22 across all 30 turns; topical
  clusters win, correctly). `default {tag "."}` is the load-bearing root
  mechanism; the tour composition is the default floor + `k = 2` organic
  picks. Concept B's adaptive rail sizes the layer; presence is the floor's
  job.
- **Q-window boundary** — answered better than designed (F12): already
  persisted as the turn layout's user `KvSpan`, 1:1 with the sig grid. No new
  record; `export-replay` emits `user_spans` for exact offline windows.
- **Short-probe residual** — labeled and reframed (F15): the promoted turns
  are tool-shaped questions where no code slot is correct; discrimination is
  cross-axis (tools mass + the real 0–1000 band), not a new probe-length
  mechanism. Folded into the Phase-1 harness acceptance below.

**Closed by the multi-segment port + snapshot battery (F16):**

- **Full-corpus tools confirmation** — done. The port is built (the six
  `belief-*` commands load the merged all-segment substrate; `belief-eval`
  gained `--scorer gated` and `--limit`, `belief-sweep` gained
  `--normalize`). Additive holds the baseline (97.3 % Top-1 / 100 % Top-5 at
  745 turns), `content_gated` collapses tools (32.9 %, 66.7 % zero-self), and
  normalization holds ranking while improving selection — the per-axis fusion
  split and the Concept A migration are both confirmed at scale.
- **Threshold derivation** — done: the normalized `belief-sweep` table (A.1)
  gives the Phase-3 operating points; only the recall/precision *choice*
  (and possibly `budget.max` 4–5 for the grown corpus) remains, a policy
  pick, not a measurement.

- **Code-corpus self-match at scale** — measured (F17): raw 57.1 % Top-1 on
  the hard cross-scope regime (strong signal at ~0.3 % chance; the 95–99 %
  figure belongs to the complete-file-conversation regime), normalization
  inverting on cold levels. The residue this leaves is not a question but a
  requirement already in the design: A.4's floor carries cold scopes, and the
  warm-level verification runs in the §11 harness.

**Remaining:**

- **Constants:** level-prior `floor_base`/`floor_cap` per preset (A.4 — now
  measured load-bearing for cold scopes, F17; production-corpus sweep),
  concentration `k`/`ρ` first values (B.1 — measured low-stakes), repo_map
  `layer_weights` (F6 invalidates the `[0,1,1]` inheritance).
- **Production-faithful confirmation:** the §11 selection-replay harness
  against the snapshot (full `ensure_normalization_warm` + belief chain — the
  venue where F15's cross-axis discrimination and F17's warm-level code
  verification are asserted), then the live `provenance_query_eval` suite
  (100 labeled queries) after the daemon restarts as the end-to-end check.
