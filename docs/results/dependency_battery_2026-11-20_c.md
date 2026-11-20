# The §9.12 dependency battery — production, 2026-11-20 (c): the re-feed

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run on BlackWorld at `af4135616`: the request's two readings of (b) removed and the re-feed of
`docs/reprojection_refeed.md` in their place —

- every projection that rebuilds a turn's slot leaves the turn's last 5–8 tokens (back to an
  index block, never past a closed page) out of the rebuild and forwards them again over it,
  the recurrent state folded on top;
- the gap leaves the chunks holding a writer's last 8 tokens native;
- the shadow scan scores the pool as one context behind each kept query, at the rung covering
  it (×8 to 2,097,152).

WhiteWorld was not serving.

## How it was run

As on every production run: one fresh connector chat per question, one at a time, deleted once
answered, every question — in the battery file's exact wording — prefixed *"Answer from your memory
of the repository only — do not call any tool, do not read or search any file. If you do not
remember, say so."* No answer called a tool. Scored by the battery's rule (`score.rs`:
case-insensitive substrings, so a name mentioned in a decline counts). a4's first chat timed out
at the connector before it answered and was asked again in a fresh chat.

## Results

| Category | Mean recall | Full marks | 11-20 (b) | 11-20 (a) | 11-19 (d) | 11-19 (c) |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 27.1% | 1 of 8 | 47.9%, 2 | 38.5%, 0 | 31.3%, 1 | 41.7%, 2 |
| Transitive | 43.8% | 2 of 8 | 51.0%, 1 | 50.0%, 3 | 41.7%, 2 | 52.1%, 2 |
| Architectural | 87.5% | 6 of 8 | 87.5%, 6 | 83.3%, 5 | 77.1%, 5 | 85.4%, 5 |
| **All 24** | **52.8%** | **9** | 62.2%, 9 | 57.3%, 8 | 50.0%, 8 | 59.7%, 9 |

| Id | Recall | (b) | What the answer named, and what it said it had read |
|---|---:|---:|---|
| d1 | 0/2 | 0/2 | declined; had read journal YAML and `scheduler/mod.rs` |
| d2 | 2/2 | 2/2 | `span_geometry.rs` from `region_pool.rs`'s import; `try_place` |
| d3 | 0/2 | 2/2 | declined; had read a journal note pointing at `substrate.rs` |
| d4 | 2/6 | 1/6 | declined; `scan.rs` (gallery arena) and `resolver.rs` named in passing |
| d5 | 0/4 | 0/4 | declined; `moe_bucketize` only from AGENTS.md |
| d6 | 1/2 | 1/2 | read `cache.rs`, the definition; no caller |
| d7 | 1/3 | 2/3 | `= 2`, but in `wave_plan.rs`; no loader |
| d8 | 0/2 | 1/2 | declined; had read a guide and Battle Cities files |
| t1 | 0/4 | 1/4 | declined; `fold_provenance`, `wide_sig.rs`, no group constant |
| t2 | 2/4 | 2/4 | read `recorded_reply.rs`: `recorded_turn`, `replayed_step`; not `replay.rs` |
| t3 | 3/4 | 3/4 | `CHUNK_SIZE`, `types.rs`, `arena_table`; not `blocks.cuh` |
| t4 | 4/4 | 4/4 | read `dispatch.rs`: bucketize → mapped ring → stager → cold spin |
| t5 | 0/3 | 2/3 | declined; had read one blog journal note |
| t6 | 1/4 | 1/4 | `ReprojectionPolicy`, `BoundaryRule`; not the config fields |
| t7 | 0/3 | 0/3 | declined |
| t8 | 3/3 | 2/3 | `elevate_to_hot` in `persistence/elevate.rs`, the scheduler |
| a1 | 3/3 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | 2/2 | `weight_floor`, `region_pool::tier_fits` |
| a3 | 2/3 | 2/3 | declined; "recurrent", "gap-fill" incidental |
| a4 | 2/2 | 2/2 | un-rotated K bytes, RoPE at read |
| a5 | 2/2 | 2/2 | invariant 3, `moe_bucketize` |
| a6 | 2/2 | 2/2 | `FoldParams`, a mismatched fold not comparable |
| a7 | 1/3 | 1/3 | declined; "thinking" incidental |
| a8 | 2/2 | 2/2 | `substrate/substrate.log`, cold |

## What the answers show

**52.8% with 9 full marks, against 62.2% for the request read twice.** The architectural set is
unchanged (87.5%, 6 full); the loss is in direct (47.9% → 27.1%) and transitive (51.0% → 43.8%),
the questions whose answer is a file placed for them. d3, d8, t1 and t5 lost what (b) had; t8
gained.

**The declines say what was in context: journals and guides, not the files asked about.** d1, d3,
d5, d8, t1, t5 and t7 each name a journal note, a guide or a README as what they had read. That
is the failure (b) fixed at the request: its first reading scored the question against the
slot before the opening was ranked. A request's opening boundary fires just after `<think>`, so
its re-feed is the token or two after that cut (`docs/reprojection_refeed.md` §2 item 5) — the
question itself is never read again over the files placed for it, and the opening is ranked
without the question's own scan.

The re-feed is measured here only as a replacement for the two readings. Whether it helps a
mid-decode reprojection — the `recorded_reply.rs` case of §1 — is not separable from this run.
