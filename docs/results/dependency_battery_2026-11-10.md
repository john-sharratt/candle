# The §9.12 dependency battery — production, 2026-11-10

The codebase dependency battery of `docs/unbounded_agents.md` §9.12, run once:
24 questions about this repository (`zend/examples/dependency_battery/battery.yaml`),
eight each **direct** (where a thing is defined and what calls it), **transitive**
(a chain of three or more hops) and **architectural** (an assumption several
components share with no single call path), each with an `expect` list of the
files, identifiers or values a correct answer must name, checked against the
source when the battery was written.

## How it was run

On the production cluster (build `4794-d5b267051`), through the connector's
private chats (`zen_chat_send`): one fresh chat per question, closed — and so
deleted — as soon as it was answered, so no question could read another's. The
model is the daemon's, Qwen3.8-Flash-Next, with the production projection over
the production substrate: the `code_reading` layer's files and the workspace's
documents, retrieved by provenance as the chat runs.

Every question was prefixed: *"Answer from your memory of the repository only —
do not call any tool, do not read or search any file. If you do not remember,
say so."* The model complied on all 24: no answer issued a tool call. Recall is
scored by the battery's own rule (`score.rs`: an item is named when any of its
`|`-separated spellings appears in the answer, case-insensitively).

## Results

| Category | Questions | Mean recall | Full marks | Abstained outright |
|---|---:|---:|---:|---:|
| Direct | 8 | 33.3% | 1 | 3 |
| Transitive | 8 | 54.2% | 2 | 1 |
| Architectural | 8 | 85.4% | 5 | 1 |
| **All** | **24** | **57.6%** | **8** | **5** |

| Id | Recall | What the answer named, and missed |
|---|---:|---|
| d1 | 1/2 | `recorded_reply.rs` (from `lib.rs`'s module list); declined to name a caller |
| d2 | 0/2 | `region_pool` and `tier_fits`'s signature, from the docs; no file |
| d3 | 1/2 | `elevate.rs`; caller described as "the persistence thread", not the scheduler |
| d4 | 1/6 | abstained; `resolver.rs` matched only as part of a directory listing |
| d5 | 2/4 | `cuda.rs`, `moe_bucketize.cu`; named an out-of-date caller (`quantized_qwen3_moe.rs`) |
| d6 | 2/2 | `cache.rs`, `pipeline.rs` — both offered as inference from the design doc |
| d7 | 0/3 | abstained ("I don't know it, and I'm not going to guess at it") |
| d8 | 0/2 | abstained |
| t1 | 2/4 | `FOLD_GROUPS`, `heads_per_group`; not `FoldParams` or `window_score` |
| t2 | 1/4 | `recorded_turn`; declined the extraction function |
| t3 | 3/4 | `CHUNK_SIZE`, `types.rs`, the arena tables; CUDA side placed in `arena_table.cuh`, not `blocks.cuh` |
| t4 | 4/4 | bucketize → mapped pinned ring → stager-published word → cold wait |
| t5 | 1/3 | `PINNED_LAYERS`; abstained on the victim search and startup |
| t6 | 2/4 | named `REPROJECT_EVERY_N_TOKENS` / `…_TRIGGER_TEXTS`, then retracted them as unverified |
| t7 | 0/3 | abstained |
| t8 | 3/3 | scheduler → `elevate_to_hot` → `persistence/elevate.rs` |
| a1 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | `weight_floor`, `tier_fits` |
| a3 | 2/3 | recurrence and gap-fill, offered as a guess; not `reserve_glue_island` |
| a4 | 2/2 | keys stored un-rotated, RoPE applied at read |
| a5 | 2/2 | invariant 3 (no GPU→CPU readback), `moe_bucketize` |
| a6 | 1/2 | abstained; "mismatched geometry" matched |
| a7 | 2/3 | `turn_layout`, thinking; not the re-prefill |
| a8 | 2/2 | `substrate/substrate.log`, the cold tier |

## Reading

- **What the memory holds is the documentation and the module structure, more
  than the call graph.** Every architectural question's answer is in
  `CLAUDE.md` or a design document, and the model reproduced five of eight in
  full; the direct questions ask for call sites, which documents rarely name,
  and it reproduced one. The answers say where they came from — "from
  `ARCHITECTURE.md` §3", "from the design doc's §0.7.4" — and the files they
  cite are documents and module headers far more often than source bodies.
- **It abstains instead of inventing.** Five answers declined outright, several
  more named what they knew and declined the rest, and one (t6) withdrew
  identifiers it had just named because it had not read the file that would
  confirm them. The errors found on reading are near-misses (the CUDA constant
  placed in `arena_table.cuh` beside `blocks.cuh`; one out-of-date caller), not
  fabricated names.
- **The one-shot ablation is not run.** It needs reprojection switched off for
  a production turn, which the API does not offer; the harness that can switch
  it (`ZendSession::ask`, `--arm one-shot`) does not reach the code memory
  (below), so its result would not be the ablation it names. The comparison the
  battery was designed around is therefore still owed.

## The read-only harness, and why its numbers are not the battery's

`zend/examples/dependency_battery` runs the same questions through
`ZendSession::ask` on a read-only copy of the production substrate, with tools
off and a per-arm switch for reprojection. Run on the card alone the same day,
all three of its arms (full, one-shot, code layer out of service) scored 0% on
direct and transitive questions and the same 10.4% on architectural — the last a
false match. Its answers were "I don't have any files or code loaded in this
conversation", and recording each answer's final projection showed why: **no
turn and no collection member was projected at all** in any of its 24 answers,
across 4–18 projection points each. A fresh conversation forked from the base
conversation with the chat's projection does not draw on the code-reading
layer the way a production chat does, so the harness measured the model
without its memory — a no-memory control, in effect, against which production's
57.6% is the memory's contribution. Making the harness reach the memory as the
chat does is what the one-shot arm needs before it can run.
