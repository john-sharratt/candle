# The §9.12 dependency battery — production, 2026-11-20 (f): a reading's minimum and maximum

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run on BlackWorld at `fad0a7ffd`, against (e) — one change, the fill
(`docs/hit_driven_projection.md` §7.2 steps 4–5, `projection/fill.rs`):

- the readings the pool ranks highest by momentum each seat their best round and floor first,
  up to half of `readings`, and what they seat does not leave at that boundary;
- no reading takes more than two fifths of `readings` (≈ 21K);
- a held round leaves by the model's selection reads first, then its score; a floor leaves with
  its piece's last round only when the entry outscores it; a displaced round does not re-enter
  at the same boundary.

Boundaries as in (e): the cadence alone, counted from the last boundary, none inside a tool call,
none at the opening's end. WhiteWorld was not serving. The chats were kept open for review.

## Results

| Category | Mean recall | Full marks | 11-20 (e) | 11-20 (d) | 11-20 (b) |
|---|---:|---:|---:|---:|---:|
| Direct | 25.0% | 1 of 8 | 38.5%, 1 | 31.3%, 1 | 47.9%, 2 |
| Transitive | 59.4% | 2 of 8 | 32.3%, 1 | 51.0%, 1 | 51.0%, 1 |
| Architectural | 68.8% | 4 of 8 | 68.8%, 4 | 79.2%, 5 | 87.5%, 6 |
| **All 24** | **51.0%** | **7** | 46.5%, 6 | 53.8%, 7 | 62.2%, 9 |

| Id | Recall | (e) | Answer file in the slot | What the answer said |
|---|---:|---:|---|---|
| d1 | 0/2 | 0/2 | `recorded_reply.rs` #1, 3/3 at every projection | "never saw a definition of `recorded_reply`" |
| d2 | 0/2 | 1/2 | `span_geometry.rs` #1, 3/3 from t=131; 17–30% of the hits on 7% of the tokens | "never actually read the file that defines it" |
| d3 | 1/2 | 1/2 | `elevate.rs` 3–4/8, then 8/8 from t=321 | `elevate.rs`; no caller |
| d4 | 0/6 | 1/6 | none of the six placed; `scan.rs` pooled #15–26 | declined |
| d5 | 0/4 | 1/4 | `quantized/cuda.rs` climbs to #1 by t=360; `dispatch.rs` 5/9 only at t=360 | declined |
| d6 | 1/2 | 2/2 | `expert_lre/cache.rs` #1, 7 of 11 (at the maximum) throughout | "I have not read `cache.rs`" |
| d7 | 3/3 | 2/3 | `layer_loader.rs` #1 3/3, `cache.rs` 7/11 | all three, muddled |
| d8 | 0/2 | 0/2 | `wide_sig.rs` #1, 7/7 at every projection | "none of them mention `FoldParams::derive`" |
| t1 | 2/4 | 1/4 | | `FoldParams`, `heads_per_group` |
| t2 | 2/4 | 0/4 | `replay.rs` 5/5 and `recorded_reply.rs` 3/3 throughout | `replayed_step` quoted, given to `replay.rs` |
| t3 | 3/4 | 3/4 | | no `blocks.cuh` |
| t4 | 4/4 | 0/4 | | the whole chain |
| t5 | 1/3 | 1/3 | `expert_lre/cache.rs` #1, 7/11 throughout | declined; `PINNED_LAYERS` from a journal |
| t6 | 4/4 | 1/4 | | both config fields, `boundary_rule`, `ReprojectionPolicy` |
| t7 | 0/3 | 0/3 | | declined |
| t8 | 2/3 | 3/3 | `elevate.rs` 8/8 from t=132 | `elevate_to_hot` put in `substrate.rs` |
| a1–a8 | 3,2,2,0,2,1,1,2 | 3,2,2,0,2,1,1,2 | | a4 read "keys" as record keys; a6 never names `FoldParams` |

## What the answers show

**51.0% with 7 full marks, against (e)'s 46.5% with 6.** Transitive rose (32.3% → 59.4%: t2,
t4, t6), direct fell (38.5% → 25.0%), the architectural set is unchanged. Within the runs'
spread; the per-question change is the evidence.

**The fill now keeps what provenance found.** Every answer file the pool ranked first is placed
and stays placed for the whole reply: `recorded_reply.rs` (d1), `wide_sig.rs` (d8),
`span_geometry.rs` (d2), `replay.rs` (t2), `cache.rs` at its maximum (d6, t5), `elevate.rs`
(t8). In (e) four of these were pushed out by one large reading within two projections.

**The direct failures are now the model's, not the fill's.** In d1, d2, d6 and d8 the answer
file leads the slot from the first or second projection to the end, and the answer says it was
never read. Each reasoning opens the same way, in its first ~64 tokens — before the first
projection, on the slot the question was prefilled against, which holds the seeds: "I read
`candle/README.md`, `candle/docs/zend_working_set.md`, `candle/AGENTS.md` … None of these
files mention `FoldParams::derive`" (d8). The verdict is written before the file arrives, and
the reasoning keeps it after: d8's reasoning later sees "the `FoldParams` I saw was in
`wide_sig.rs`" and still answers that nothing mentions `derive`. d2's selection read
`span_geometry.rs` at 2.5–4× its share of the slot and the answer still declined.

**That is what (b) did not have.** With the question read twice, the files placed for it were in
the slot before the reply's first token, and the reasoning's opening inventory included them.
Here the first projection is at decoded token 63–69, and the re-feed of the last 8 tokens does
not revisit an inventory written 40 tokens earlier.

**d6 may have lost its answer to the maximum:** `cache.rs` held 7 of its 11 rounds, and in (e),
with all 11, the answer named both files. Whether `set_eviction_costs`'s round was among the four
left out is not in the record.
