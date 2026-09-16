# Open items

State as of `3d9ed922` (on `main` and `qwen38-moe`, pushed), plus the fixes recorded under
"Fixed" below, committed in `81e487b5`. Nothing here is speculative: every claim
has the command or file:line that produced it, and the one claim that is code-derived rather
than measured says so and names what would falsify it.

---

## Fixed (committed in `81e487b5`)

### 1. `persistence_integration` — the carved calibration seal, not the scheduler

`cargo test -p zend --features cuda --tests` failed
`persistence::turns_persist_and_recover_across_a_simulated_restart` on the dev-profile
`debug_assert_eq!` in `perform_seal_and_write` (`scheduler/mod.rs`, "persisted token_ids must
align 1:1 with the K/V chunk grid", left 25 / right 32), surfacing as the 600 s timeout.

**The earlier diagnosis here was wrong.** Neither documented scheduler site is this class:
`mod.rs:1205-1236` (`forwarded_generated`) is an index-vs-K/V bug and `mod.rs:7110-7116`
(`begin_unit`) is about index page widths. The failure is in the **stuffed calibration
prefill** zend runs at load, before either test turn is submitted:

- `stuffed_grid` pads every case to a block boundary (`pad_to_block`), and the seal captures
  the whole block range — so the persisted K/V holds the padding.
- `Conversation::submit_prefilled_turn_group` pinned `grid.tokens[region.token_range()]`, the
  real tokens only, under a comment claiming that aligned 1:1 with the K/V. It did not: a
  15-token question is a 25-token case, padded 7, sealed as one 32-token block.
- The substrate's own contract (`substrate.rs`, `TurnPart::token_count`) is
  `token_count == token_ids.len()`, which cross-process replay rebuilds the grid from. The
  assert was right; the carved content was wrong. It was introduced by `d94c20f5` against an
  assert from `e10ad58f`.

**Fix:** `CarvedRegion::sealed_range()` (every token the blocks hold) is what the turn pins,
and `CarvedRegion::layout()` tiles that range by folding the padding into the closing
`ImEnd` glue — so no phase span touches padding and **no exemplar's K/V or `sign(Q)` window
changes**. Tests: `a_regions_turn_pins_every_token_its_blocks_hold` (raw expected ids,
`validate_tiling`), and `phase_spans_land_on_real_tokens_and_never_on_padding` now built
through `CarvedRegion::layout`.

**Verified end-to-end:** with the full tool catalog it passed in 241.5 s, daemon stopped,
and the dev-profile assert no longer fires on any seal — of that, 170.9 s was the model load
and 24.9 s tool-section prefill. The test is now `#[ignore]`d (it boots the production model)
and runs in a **tool-free mind** workspace — the production schema as its `projection.yaml`
plus an empty `tools/` (`n_tools=0`, calibration 0 ms). Re-run that way it passed in 221.0 s:
173.6 s model load, 23.3 s in the "Prefilling tool sections" step (it still seals the two
tool-summary sections — the empty catalog saves ~1.6 s there), 16.8 s workspace ingest. The
model load is the floor; the tool-free workspace keeps the test off the catalog, not fast. Run
with `cargo test -p zend --features cuda --test persistence_integration -- --ignored --nocapture`.

### 2. `web` — 5 `site::tokera` failures: stale tests, not the post walk

`cargo test -p web` now **171 / 171**. **The earlier diagnosis was wrong on both counts.**
Nothing enumerates `feed.xml` as a post — the walk (`content.rs:80` `is_file`, `blog.rs:46`
`.md` only) cannot, and the feed is generated per request, never a file. And the content did
not arrive with `38f86265` (it touches only `web/content/npcd/index.html`); it arrived with
`aae66ee8`.

The cause is `6dab06f8` (2026-08-30), which deliberately added an RSS `<link …
href="/blog/feed.xml">` to every page's `<head>` (`page.rs:240`) and per-page titles
(`page.rs:186-190`), without updating tests from 2026-08-26:
- three tests scraped every `href="/blog/` on the index, so the head's feed link read as the
  first post. They now share `listed_slugs`, which reads the entry headings
  (`<h2><a href="/blog/`, `blog.rs:132`); the fourth scrape (`no_post_leaks_…`) was passing
  only because the RSS body contains no escaped tags, and uses it too.
- two title tests asserted the reverted design. `the_head_carries_a_title_and_description`
  now expects `H · Tokera`; `the_tab_title_does_not_follow_the_page` became
  `the_tab_title_is_the_heading_then_the_brand_except_at_home`.

### 3. Thinking-off prefill regression on Qwen3.5 / 3.8 (found while tracing item 1)

`193f35a5` made suppression structural on block-suppressing families by prefilling
`assistant_lead = closed_think + assistant_prefill`. **Merge `aae66ee8` silently dropped it**
(`git log -m -G assistant_lead`: no ordinary commit ever removed it), leaving
`format!("{assistant_head}{assistant_prefill}")` — so a thinking-off turn on Qwen3.5/3.8
prefilled no closed block, while `assistant_content_start` (non-empty-prefill arm) still
measured past one. Restored through `conversation.rs::assistant_lead`, with
`assistant_lead_tests` pinning both dialect families.

### 4. Smaller

- **`for_display` ownership.** The reporting bound moved from `ToolBelief` to the readout it
  bounds: `ProjectTile::SCORE_CAP` + `ProjectTile::cap_score` (`zend/src/api/substrate.rs`,
  with a unit test). `ToolBelief::update`'s doc keeps why the accumulator is never clamped.
- **Stale test comments** in `belief.rs` citing the removed `ToolBelief::CAP` and a deleted
  saturation test.
- **Mojibake** (`Â§` for `§`, a CP1252 round-trip): `scheduler/mod.rs` ×2 and
  `projection/tests.rs` ×2, all from `80b5541b` (2026-05-31).

### 5. Default-run test times

Every test in a plain `cargo test` now finishes under ~20 s except `tools_integration`'s
scenarios (below). None of the slow ones loaded a model; they were **unoptimised host code**.

- **`candle-conversation` and `tokenizers` join the workspace's per-package
  `opt-level = 2` list** (both `dev` and `test` profiles, beside candle-core/nn/transformers):
  `selection_replay`'s golden baseline 107.7 s → 7.2 s, each `token_bias_real_vocab` test
  ~26 s → ~4.5 s, and `prefill_ab`'s 47.7 s of test time now fits in a 13 s step, build
  included.
- **`network_diag::ping_icmp_structure`** 22.9 s → the whole suite in 0.8 s: the tool's
  defaults are 4 echoes × 5 s timeout; the test only checks the response's shape and now asks
  for one echo, 1 s.
- **`tools_integration` runs on `Qwen35_0_8B_Q8`**, a new preset (the gate's own pinned
  0.8B, same lineage/dialect/tool-call style as production), selected through a new
  `DaemonConfig::model: ModelChoice` — `MeasuredVram` (the existing ladder, the default) or
  `Preset(Box<Model>)` — with a matching `zend --model <PRESET>` flag and
  `Model::{PRESETS, from_override_key}`. `download::ensure_model` and the session build from
  the one resolved model. Scenarios run thinking-off (`api::chat::dial_selection`, now `pub`),
  one at a time, on a persistent `target/tmp/tools_integration_ws` whose calibration is paid
  once, force-compacted past 2 GiB. **9/10 pass in the default suite at ~20–30 s each**;
  `hash_compute` (the 0.8B answered a decimal integer) runs on the production model + live
  repo and is `#[ignore]`d. The 0.8B's boot floor is ~20 s (model load ~9–13 s, mostly
  tokenizer parse/verify; tool-section prefill ~4–10 s), so the 20 s target is not met there.
- **`zend/README.md`** documents `--model`, and its stale `--disable-summariser` row (no such
  flag) is gone.

### 6. `docs/performance.md` — second width sweep

All 12 `test_parallel_batched_forwarding*` gates re-run 2026-09-13, one process per model,
12/12 pass. §3.6 *Width* and §3.7 now report each cell as the better of the two sweeps (†
marks the new one; ‡ a width only the new ladder reaches); depth tables are unchanged. The
new rows are in `performance_rtx_pro_5000_72gb_rows_2026-09-13.tsv` (191 rows); the
authored TSV is untouched. §4 gains two entries — Qwen3.6-35B's extreme-width validation
failure **did not reproduce**, and **C10 compression moved between the builds** (lower on
most models, higher on Qwen3.8-27B; cause not established).

---

## Open — found while speeding up the tests

### 8. Every zend boot re-seals the whole tool catalog into the redo log — RESOLVED

Section streams are content-addressed (`section_stream_id`), so a section prefilled again
supersedes the last boot's records — dead records only compaction reclaims; the
`tools_integration` workspace grew ~140 MB a boot. The cause was the restore triage refusing
every section on the hybrid lineage (the persisted grid checked against transformer depth
instead of the KV backing count), fixed in `01b5c559`: a refused restore falls back to a
prefill, and the prefill's seal is the rewrite. `tools_integration::
a_second_boot_restores_every_prompt_section` boots a `ZendSession` twice on the suite's
workspace — the tool catalog included, which only a zend boot installs — and asserts the
second boot prefills no prompt section (passes, 23 s). The `zend/src/session.rs` comment that
called section cold-load disabled now describes the triage.

### 9. Tool-section prefill slows across boots within one process — does not reproduce

Measured 2026-09-13: same 93 sections, same workspace, 4.4 s on a process's first
`ZendSession` boot and 9.9 s on its second. That was with every section refused by the
restore triage and prefilled (item 8). Re-measured 2026-09-15 by the two-boot test above
run with `--nocapture`: the "Prefilling tool sections" step takes **207 ms on the first boot
and 197 ms on the second**, because both now restore. Of the per-process candidates, the
first is refuted by the same log: boot 2 prints "a reservation already exists — this model's
weights load through the CUDA pool", but `close_load` then retracts the previous model's
weight zone and the dense block locks inside the span at 250 MiB exactly as on boot 1. A
second boot that genuinely prefills (a changed catalog) has not been timed, so whether
per-process state slows a real prefill is not established; nothing in this path pays it.

---

## Open — needs a decision

### 10. One KV threshold row per model, or one per card

`bb363015` re-derived two `*_KV_FACTORS` rows on the RTX 3090 (sm_86), where the
old pairs failed C10 validation (9/10, three runs each): Qwen3.5-9B 1.07/1.85 →
0.85/1.45 and Qwen3.8-27B 0.8/2.05 → 0.7/1.4. The 2026-09-15 sweeps on the RTX
PRO 5000 (sm_120) validated every C row at the new pairs, so one constant holds
on both cards — at a price on the one the old pairs already passed on: C10
5.87× → 5.13× on the 9B and 5.56× → 4.81× on the 27B, about 13 % of the top
rung's compression. The rows' own notes record why the edge moves with the card
(the int8 numeric path differs per arch, and the 27B's checkpoint is already
chosen per card). Keeping one row per model buys cross-card safety; a row per
card buys back the compression. Nothing is wrong either way — it is a choice.

---

## Open — found in the 2026-09-15 sweeps

### 11. Qwen3.6-35B-A3B C10×16 decodes 10 % slower than `81e487b5`

The one row of the fleet sweep (`docs/performance.md` §4, *The decode-slot refresh
prefill regression*) outside noise: C10 ×16 decode 769.5 t/s on `81e487b5` (runs
769.5 / 768.3) against 690.9 on `23623c6b` (687.5 / 690.9), both builds on the
synchronised harness. The same model's C10 ×8, ×32 and ×64 rows are flat or faster
(×64 1,182.9 → 1,201.6), and so is every other decode row in the fleet, so it is
specific to this width rather than a decode-path cost. Not yet attributed. Candidates
to measure: the speculative verify wave's per-sequence slot-state resync (one upload
per sequence per layer per step since the refresh became a mark + sync) at a width
where acceptance is low, and the expert-residency ladder's behaviour at ×16.

## Open — tool scoring (deferred)

### 5. The "0–1000 normalized band" is violated by ~1000×

Measured over the two newest segments (`substrate_inspect projections --jsonl`, 7,149 events,
478,175 values): 473,044 exactly 0; 3,142 in (0, 1000]; 13 above 500k, max 1,459,468.

**Corrections to the earlier analysis:**
- **The floored path is not in play for tools.** The tools collection normalizes through
  `cache.normalize` with no floors (`resolver.rs:945`), i.e. the unfloored denominator
  `max(level or hit_prior, floor)`. The `peak()`/`child_floor` path applies only to the
  per-file groups (`resolver.rs:1855`). The ~1e6 arithmetic (1,170 × 42.7 × 1000 / d) needs
  `d = 50` — which is `floor_min`, i.e. a collapsed level.
- **Why levels collapse (code-derived, not yet measured).** `HitLevel::observe` folds zeros
  into the level (`hit_level.rs:73-84`, `alpha_dn = 0.02`), and every observe passes the
  whole member slice. `warm_collection_normalization` warms members in `BTreeMap` (name)
  order, 8 probes each (`WARM_COLLECTION_PROBES_PER_MEMBER`) — so an early member such as
  `datetime` absorbs ~600 zero-folds from later members' exemplars (0.98^624 ≈ 3e-6). Levels
  then depend on a tool's **name**, and early ones divide by `floor_min`. Falsified if a dump
  of per-member levels after warm-up shows them independent of name order.
- **The table mixes builds and kinds.** It pools collection section beliefs with turn scores
  (many zeros are turns), and spans the clamped `fedd96e2` build and unclamped ones.
- **Lever 3 is not wired.** `normalize_with_fallback` has no production caller — it and
  `ScopeKey::CollectionPhase` survive the phase-lens removal only in tests.
- **The warm-up races the first query.** It runs on a thread spawned just before
  `mark_ready` (`session.rs:4672`), so an early query scores against cold levels (prior 400).
- **The design doc is stale.** `docs/provenance_score_normalization.md` still says "module not
  yet built", "collections (tools) … not yet wired" (§5) and "hard-floored at the prior"
  (§3.1) against `floor_min = 50`.

Candidate levers (each needs a scored sweep): a ceiling on `to_ref`; raising `floor_min`
toward the prior §3.1 names; not folding zeros into non-scoring members' levels; wiring the
`min_observations` gate.

### 6. The opening projection has no tool catalog (first-ever turn)

A turn that should call a tool answered *"there's no time-lookup tool available"*; `proj #0`
selected nothing and `proj #1` selected `datetime` at 5000.000.

**Corrections to the earlier analysis:**
- **The opening projection scores nothing at all.** Submit reads through the unscored
  `read_for` (`mod.rs:3643`; `resolver.rs:532-563`: "every score lookup returns zero"). There
  is no prefill-Q scoring path — refutation #3's premise, and the comments at
  `mod.rs:3644-3648` / `3662-3663`, describe code that no longer exists.
- **`5000.000` is the `fedd96e2` accumulator clamp**, removed by `3d9ed922` for exactly this
  symptom. The captured events predate the fix: **re-capture on the current build first.**
- **The first reprojection fires at ~token 1** (`prefill.rs:2942-2954`), so `proj #1` at
  4.46 s wall-clock may still be `start_token ≈ 1`. That number decides it: if ≈ 1, only the
  first sampled token (from prefill logits, against an empty `<tools>`) is exposed; if large,
  the early reprojection was lost.
- **`ProbeCtx::probe` scores the same tool-tagged gallery as reprojection**
  (`conversation.rs:4579`), not a different one; it works because it has a prefilled query.
- **`projection.yaml:387-402` is false**: `question_pin` is read only in `score_beliefs`,
  never at submit, so it does not rescue the opening projection; the same comment still lists
  `tools_overview` under `depends_on: tools`. The header (`:58-61`) still says min112/evict60.

Options unchanged in shape: (a) accept; (b) score the opening projection, which needs the
query's Q before its prefix exists; (c) sample the first token after the early reprojection
rather than from prefill logits — the smallest change, reusing the one scoring path.

### 7. repo_map retrieves ingest scaffolding for NL queries

"what time is it?" selects `marian-mt/python/` (385.28), `py_src/candle/testing/` (360.58),
`paged-glue/` (357.49): every high hit is an ingest turn shaped *question + `<tool_call>`*,
and `score_belief_groups` scores the whole stored signature, user half included. Levers:
exclude the prompt half of the ingest turn from the gallery; or fix item 5. **Do not retry
`content_gated` fusion** — introduced `660a1b66`, reverted `cbf2574c`; `session.rs` asserts
`FusionMode::Additive`.

---

## Standing notes

- **One stale docs pointer remains, deliberately.** `zend/src/prompts/tools/file_read.yaml:4`
  reads `docs/kv_tier_migration.md` — a calibration exemplar's user phrasing, not a reference;
  rewriting it forces a re-calibration for no benefit.
- **`real_cuda` tests need the card alone.** `vram::tests::real_cuda` gates on a present CUDA
  device and asserts on global headroom, so a co-resident daemon inverts it. A filter that
  matches nothing exits 0 with `0 passed; N filtered out`.
- **Three zend tests act on the real `.substrate`.** `startup_stall_watchdog` deletes it;
  `reproject_control` / `reproject_wave` copy the whole segment set into `target/`. All are
  `#[ignore]` — **never pass `--ignored` to the whole zend suite.**
- **Model gates run one cargo process per model**, daemon stopped (`deepseek4.rs:621`).

---

## Verification

Final run, on the working tree with every change above (2026-09-13, daemon stopped, one
`cargo` process at a time):

| check | result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --workspace --tests --examples -- -D warnings` | exit 0 |
| `cargo test -p candle-conversation --lib` | 1,447 passed, 4 ignored |
| `cargo test -p zend --features cuda --lib` | 616 passed |
| `cargo test -p zend-tools --test network_diag` | 9 passed, 0.8 s |
| `cargo test -p zend --features cuda --test tools_integration` | 9 passed, 1 ignored, 216.7 s |
| `persistence_integration` (`--ignored`, tool-free workspace) | 1 passed, 221.0 s |
| 12 × `test_parallel_batched_forwarding*` (release, one process each) | **12/12** |

The broader sweep ran earlier the same day, before the §5 test-speed changes: `candle-conversation
--features hub --tests`, `npcd --lib`, `zend-tools`, `web`, `candle-kernels`, CUDA `candle-nn` and
`candle-transformers`, `matmul_tests` (16/16) all exit 0, and `persistence_integration --ignored`
passed in 241.5 s with the full tool catalog.

Not re-run since `3d9ed922`: `substrate_inspect validate`.
