# The §9.12 dependency battery — production, 2026-11-20 (e): the cadence alone

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run on BlackWorld at `136638274`, against (d):

- no early boundaries — the reasoning's boundaries at 9, 16 and 32 decoded tokens are gone;
- the cadence counts tokens since the last boundary, restarted at every boundary, instead of
  firing on multiples of the decoded count;
- no boundary inside a tool call, its `<tool_call>` tag included.

Still no boundary at the opening's end, so a reply's first projection is the cadence's, 64
tokens in, or a line feed past the content gate. Every reply measured here took its first at
decoded token 53–69 and then one about every 64 tokens. That is four reprojections fewer per
reply than (d) in its first 64 tokens.

WhiteWorld was not serving. The chats were kept open for review; 16 was the most one person
could keep, raised to 64 (`4e14c0329`) to finish the run.

## Results

| Category | Mean recall | Full marks | 11-20 (d) | 11-20 (c) | 11-20 (b) |
|---|---:|---:|---:|---:|---:|
| Direct | 38.5% | 1 of 8 | 31.3%, 1 | 27.1%, 1 | 47.9%, 2 |
| Transitive | 32.3% | 1 of 8 | 51.0%, 1 | 43.8%, 2 | 51.0%, 1 |
| Architectural | 68.8% | 4 of 8 | 79.2%, 5 | 87.5%, 6 | 87.5%, 6 |
| **All 24** | **46.5%** | **6** | 53.8%, 7 | 52.8%, 9 | 62.2%, 9 |

| Id | Recall | (d) | Answer | Why it failed (projection records, reasoning) |
|---|---:|---:|---|---|
| d1 | 0/2 | 1/2 | `recorded_reply` "in `cuda_tests.rs`" | **Misread.** `recorded_reply.rs` placed and #1 of 74 pieces from t=66 (fresh 13,670), and the reasoning found it ("Yes! … `fn recorded_reply` defined in `recorded_reply.rs`"), then gave its content to the 44k-token `cuda_tests.rs` reading placed beside it. No such function exists there. `conversation.rs` pooled at #5–6, never placed. |
| d2 | 1/2 | 0/2 | `region_pool.rs` named in passing | **Not placed.** `region_pool.rs` pooled #4–10, unplaced in every projection; `span_geometry.rs` never seen. |
| d3 | 1/2 | 2/2 | `elevate.rs`, "the persistence thread" | **Not placed.** `persistence/elevate.rs` pooled #2 by momentum at every projection, never placed: the reading budget went to `scheduler/mod.rs` (16 of 75 turns, 52,781 tokens). |
| d4 | 1/6 | 0/6 | `resolver.rs` named | **Not pooled.** None of the six files appears in any projection. |
| d5 | 1/4 | 0/4 | `moe_bucketize.cu` | **Late.** `quantized/cuda.rs` and `moe_bucketize.cu` climb from #14–48 to #2–4 by t=311–377 and are placed there, after the reasoning had decided. |
| d6 | 2/2 | 1/2 | `cache.rs`; `pipeline.rs` in passing | Placed #1 from t=63. |
| d7 | 2/3 | 0/3 | `= 2`, `cache.rs` | **Misread.** `expert_lre/cache.rs` #1 and `qwen35/layer_loader.rs` #2, both placed from t=53. The reasoning: "It contains `PINNED_LAYERS` … It does NOT contain `PINNED_LAYERS`", "I must not confuse `PINNED_LAYERS` with `PINNED_LAYERS`". |
| d8 | 0/2 | 1/2 | declined, guides and READMEs only | **Placed then evicted.** `wide_sig.rs` #1 placed at t=66; still #1 by momentum at t=130 and t=194 and unplaced, behind `scheduler/mod.rs` (16 of 104 turns). The reasoning loops on "It had `FoldParams`?". |
| t1 | 1/4 | 1/4 | `FoldParams`, from `scheduler/mod.rs` | **Not placed.** `wide_sig.rs` pooled #2–4 from t=128, unplaced; `scan.rs`, `window_score.rs` never seen. |
| t2 | 0/4 | 2/4 | declined | **Misread.** `recorded_reply.rs` #1 placed from t=66 and `session/replay.rs` #4 placed at t=66, evicted by t=129; the reasoning: "The conversation file I read was `recorded_reply.rs`?" and declined. |
| t3 | 3/4 | 3/4 | `CHUNK_SIZE`, `types.rs`, `arena_table` | `kv_cache/arena_table.rs` placed from t=321; `blocks.cuh` never seen. |
| t4 | 0/4 | 4/4 | declined | **Placed then evicted; no AGENTS.md.** `moe_bucketize.cu` and `expert_lre/dispatch.rs` #1 placed at t=66, out by t=194. The reasoning opened "The `dispatch.rs` file I read contained relevant content", then concluded the chain is in AGENTS.md, which this conversation had not been given. (d) answered it from AGENTS.md. |
| t5 | 1/3 | 2/3 | `PINNED_LAYERS = 2`, from a journal | **Misread.** `expert_lre/cache.rs` #1 placed from t=68; the reasoning: "What I have NOT read: any file in `expert_lre/` — not `cache.rs`". |
| t6 | 1/4 | 1/4 | `ReprojectionPolicy` | `scheduler/mod.rs` placed #1; `config.rs`, `boundaries.rs` never placed. |
| t7 | 0/3 | 0/3 | declined | **Not pooled.** The assembler's reading pooled at #19–29, fresh 1,003; the glue cache never seen. |
| t8 | 3/3 | 2/3 | `elevate_to_hot`, `persistence/elevate.rs`, the scheduler | |
| a1 | 3/3 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` | |
| a2 | 2/2 | 2/2 | `weight_floor`, `region_pool::tier_fits` | |
| a3 | 2/3 | 3/3 | recurrent, `reserve_glue_island` | Correct; "gap‑fill" written with a non-breaking hyphen, which the rule does not match. |
| a4 | 0/2 | 1/2 | content-addressed reading keys | Read "keys" as reading keys, not attention keys. |
| a5 | 2/2 | 2/2 | invariant 3, `moe_bucketize` | |
| a6 | 1/2 | 1/2 | refused; "fold parameters" | Correct; never names `FoldParams`. |
| a7 | 1/3 | 1/3 | declined; "think" incidental | |
| a8 | 2/2 | 2/2 | `substrate/substrate.log`, cold | |

## What the answers show

**46.5% with 6 full marks, against (d)'s 53.8% with 7.** Direct rose (31.3% → 38.5%);
transitive fell (51.0% → 32.3%) on t2 and t4; the architectural set fell (79.2% → 68.8%) on a4
and a3, one of them a hyphen. Run to run, single questions move a full mark either way. This is
the low edge of the 50–62% these runs have spread over, with four reprojections fewer per reply.

**Provenance finds the answer file.** In d1, d6, d7, d8, t2, t4 and t5 the answer file is the #1
piece by momentum at the first projection, 53–69 tokens in, and placed. In d3, t1 and d2 it is
#2–4. Only d4, t7, t3's `blocks.cuh` and t6's config fields are never pooled.

**Three ways it is lost after that:**

1. **The fill gives the reading budget to one file.** On average 2.9 readings are placed per
   projection, and the largest takes 70–87% of the reading budget: `scheduler/mod.rs` (16 of
   75–104 turns, ~52k tokens) in 61 of 170 projections, `expert_lre/cache.rs` in 36, the 44k-token
   `cuda_tests.rs` in 18. Its turns each outscore the best turn of the #1 or #2 piece, so a 7- or
   8-turn file ranked above it — `wide_sig.rs` in d8 and t1, `elevate.rs` in d3 — gets no turns
   at all (d3, d8, t1).
2. **A placed file is evicted at the next projection.** `wide_sig.rs` (d8), `replay.rs` (t2),
   `moe_bucketize.cu` and `dispatch.rs` (t4) are placed at t≈66 with zero hits and out within
   one or two cadences; their hits never accumulate before the file goes.
3. **What is in attention is misread.** In d1, d7 and t5 the file is placed and leads from the
   first projection, and the reasoning misattributes it — `recorded_reply.rs`'s function given to
   `cuda_tests.rs`; "`PINNED_LAYERS` … not `PINNED_LAYERS`"; "not `cache.rs`" with `cache.rs`
   leading the slot. The content is read; its identity is not. That is the failure of sealed
   K/V read under a context other than the one it was sealed in (`docs/reprojection_refeed.md`
   §1, the 11-10 `provenance_depth.rs` measurement): the readings carry the conversation they
   were read in.

**The reasoning still opens on the seeds.** Every reply's first sentences list what it has read
from the slot the opening was prefilled against — READMEs, guides, AGENTS.md, a journal note —
before the first projection at 53–69 tokens.
