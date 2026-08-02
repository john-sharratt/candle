# selection_replay_data — captured baseline for the provenance-adaptive-projection harness

> **Local-only data.** `export/` (657 files, ~46 MiB of sig blobs + recorded
> events) and the captured conversation JSONs are **gitignored** (see
> `.gitignore` here) — too many files, too large for the repo. Checked in:
> this README, `ideal_projections.json` (the hand-authored test spec), and
> `baseline_golden.json` (the pinned characterization digest). When the local
> data is absent, `tests/selection_replay.rs` **skips** with a message; the
> `export/` command below regenerates it from the pre-repair snapshot.

Captured **2026-08-02** from the live zend daemon (port 80) against the full-repo
substrate (repo_map 4 convs / 142K tokens, code_reading 1,873 convs / 64M tokens),
**before** the startup integrity repair first ran — i.e. while the damaged
(token-less) code_read/repo_map conversations were still live in the galleries.
This is the corpus state that produced the selection failures motivating
`docs/provenance_adaptive_projection.md` §1.

## Files

| File | Conversation | Why it matters |
|---|---|---|
| `tour_overview.json` | "Codebase Tour Overview" (7 turns) | The primary fixture: tour confabulation, datetime/calculator tool turns, the ModelBuilder retrieval miss, the self-recall turn. |
| `tour_overview_b.json` | second overview run (2 turns) | Band comparison. |
| `tour_crates_a/b/c.json` | "Codebase Tour Main Crates" (3 runs) | Repeat-run variance of the same probe shape. |
| `what_time_hostab.json` | "What time is it" (3 turns) | Tool-selection control (datetime). |
| `substrate_overview.json` | — | Layer inventory + counts at capture time (which layers were populated, budgets, selection rules). |

Each conversation JSON is the `/v1/conversations/{id}` body: role-split
messages, and per assistant message `spans` = the full recorded
`ProjectionEvent` sequence (reprojection cadence, per-member normalized
scores, selected/skipped flags, `SelectionOrigin`, materialized order), plus
`turn_content` bodies for resolvable turns.

These are the **recorded** side of `selection-replay --compare-recorded`
(design doc §11.1): what the shipped selector actually chose, with scores. The
**re-run** side (galleries + probe sigs) comes from the substrate itself.

## The pre-repair substrate snapshot

The startup integrity repair tombstones the damaged conversations, and
background maintenance then compacts their records away — so the live
`.substrate` loses the exact pre-repair gallery state shortly after the first
post-repair boot. A raw snapshot was taken before that restart:

```
D:\prog\substrate_pre_repair_20260802   (sealed segments, backup/ excluded)
```

Loading that snapshot read-only (e.g. `substrate_inspect --log <dir>`, or the
replay harness pointed at it) reproduces the failing selections byte-exactly:
same galleries (damaged conversations still live), same sigs, same recorded
events. The post-repair live substrate is the *healthy* baseline; the snapshot
is the *as-observed* one.

## `export/` — the sig fixture (binary probes + galleries)

Produced by `substrate_inspect export-replay` against the snapshot:

```
substrate_inspect --log D:\prog\substrate_pre_repair_20260802 export-replay \
    --timeline <each dialogue timeline> \
    --tag models/builder.rs --tag tool --tag repo_map \
    --out <dir>
```

then curated (tool gallery trimmed to 2 exemplar turns per tool; everything
else kept whole). Contents:

- `manifest.json` — every exported turn's `(timeline, index, tags, block_span,
  tombstoned, sig file)`, grouped into `dialogue_turns` (30 turns across the 6
  conversations, with recorded-event counts), `selected_candidates` (every
  turn any recorded projection selected — the observed junk), and `targets`
  (`models/builder.rs` 42 turns, `repo_map` 177 cluster turns, `tool` 190
  turns across 95 tools).
- `sigs/{timeline}_{index}.wqs` — raw `encode_wide_sigs` blobs
  (`decode_wide_sigs` reads them), 624 files.
- `events/{timeline}_{index}.events.json` — each dialogue turn's recorded
  `ProjectionEvent` sequence, verbatim.

`tests/selection_replay.rs` consumes this fixture. Always-on tests guard the
export chain and the already-correct tool selection (datetime / calculator
top-1 from the real probes). The `#[ignore]`d tests are the RED TDD targets
for `docs/provenance_adaptive_projection.md`, with today's measured failures:

| Target | Today |
|---|---|
| builder.rs outranks observed junk (ModelBuilder probe) | `session.rs` call-site 2004 vs `builder.rs` 351 |
| structure outranks scopes (tour probe) | `test_config.json` scope 2653 tops the ranking |
| history probe's code mass collapses vs code probe | inverse today: 6122 vs 3576 |

## `baseline_golden.json` — the current state, pinned

`baseline_every_recorded_projection_point_replays` (always on, one fixture
load) replays all 310 recorded projection points through the raw scorer and
compares an exact per-point top-3 digest against this golden. Any scoring
change surfaces as a precise per-point diff; intentional changes regenerate
with `SELECTION_REPLAY_REGEN=1 cargo test --test selection_replay baseline`
and the diff is reviewed like source. Deterministic across debug/release.

**Measured at capture:** only ~7% of the daemon's recorded winners re-rank
near the top under instantaneous raw scoring — the recorded selections are
dominated by belief accumulation, normalization, and hysteresis retention
(selection inertia), not fresh signal. The recorded events are therefore the
as-observed reference; the golden pins the raw-signal layer the TDD targets
score with; and the inertia gap itself is part of what the design doc's
concepts address.
