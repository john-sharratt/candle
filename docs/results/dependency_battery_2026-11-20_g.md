# The §9.12 dependency battery — production, 2026-11-20 (g): the re-decode

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run on BlackWorld at `efb042864`, against (f) — one change, the re-decode
(`docs/reprojection_refeed.md` §2.6, `scheduler/refeed.rs`):

- at every projection that rebuilds a decoding turn's slot, the turn keeps the first of its last
  5–8 tokens and drops the rest, the pending token included, and decodes on from there over the
  rebuilt slot, instead of forwarding them again unchanged;
- the recurrent state is not rewound: the DeltaNet layers keep the dropped tokens folded;
- the sampler forgets the dropped tokens' penalty counts; its RNG goes on, so each re-decoded
  token is a fresh draw;
- the stream holds back the newest 8 tokens, so nothing dropped was ever sent.

Fill and boundaries as in (f). WhiteWorld was not serving. The chats were kept open for review
(the per-person limit raised to 128, `efb042864`).

## Results

| Category | Mean recall | Full marks | 11-20 (f) | 11-20 (e) | 11-20 (b) |
|---|---:|---:|---:|---:|---:|
| Direct | 22.9% | 0 of 8 | 25.0%, 1 | 38.5%, 1 | 47.9%, 2 |
| Transitive | 50.0% | 3 of 8 | 59.4%, 2 | 32.3%, 1 | 51.0%, 1 |
| Architectural | 64.6% | 4 of 8 | 68.8%, 4 | 68.8%, 4 | 87.5%, 6 |
| **All 24** | **45.8%** | **7** | 51.0%, 7 | 46.5%, 6 | 62.2%, 9 |

| Id | Recall | (f) | What the answer said |
|---|---:|---:|---|
| d1 | 0/2 | 0/2 | declined; `recorded_reply.rs` placed 3/3 from t=69; the reasoning loops on eleven tool names |
| d2 | 1/2 | 0/2 | `region_pool.rs`; no definition site |
| d3 | 1/2 | 1/2 | `elevate.rs`; caller given as `transfer.rs` |
| d4 | 0/6 | 0/6 | declined |
| d5 | 0/4 | 0/4 | declined |
| d6 | 1/2 | 1/2 | declined; `cache.rs` only as `expert_cache.rs` |
| d7 | 1/3 | 3/3 | declined; "two" matched in "the two files I read" |
| d8 | 0/2 | 0/2 | `derive`'s signature and its tests quoted, given to `research.md` |
| t1 | 1/4 | 2/4 | `FoldParams::derive` quoted, then `fold_provenance_checked` repeated a few hundred times |
| t2 | 2/4 | 2/4 | `replay.rs`, `recorded_turn`, `replayed_step` |
| t3 | 1/4 | 3/4 | `CHUNK_SIZE` only |
| t4 | 4/4 | 4/4 | the whole chain |
| t5 | 0/3 | 1/3 | declined, `cache.rs` read |
| t6 | 0/4 | 4/4 | declined |
| t7 | 3/3 | 0/3 | declined, naming `glue_islands`, its hash key and `GLUE_ISLAND_RETAIN_GENERATIONS` |
| t8 | 3/3 | 2/3 | the scheduler, `elevate_to_hot`, `elevate.rs` |
| a1–a8 | 3,2,1,0,2,1,1,2 | 3,2,2,0,2,1,1,2 | a3 and a6 lost a name each |

## What the answers show

**45.8% with 7 full marks, against (f)'s 51.0% with 7.** Within the runs' spread on the score;
the text is not.

**The re-decode joins two continuations at every cut.** The attention layers and the text step
back to the anchor; the 36 recurrent layers keep the dropped tokens, so the token sampled after
the anchor follows a context that already holds what was dropped. The seams show in the
answers: 8 of 24 hold an unpaired backtick, against none in (f) — a name cut off where the
dropped text resumed ("elevate_`elevate.rs`", "the `candle-kernels`arena_table.cuh`",
"`CHUNK_SIZE = 32, but the mirror is") — and words are missing at a join ("and iting mechanism
you're asking about", "a reason why a reason why"). Two answers fall into a repetition loop,
none in (f): t1's `fold_provenance_checked` a few hundred times, d1's reasoning on the journal
tool names. The sampler's penalty counts forgot the dropped tokens; the recurrent state did not.
(Run (h), without the re-decode, looped on t1's name list too
(`dependency_battery_2026-11-20_h.md`): t1's loop is not the re-decode's alone; the broken
names are — 8 answers here, 1 in (h).)

**The answer files are placed as in (f), and the direct declines are the same.** d1's
`recorded_reply.rs` leads the slot from t=69 and the reasoning still says it never saw the
function; d8 reads `wide_sig.rs` — `derive`'s signature, its tests — and files it under a guide's
name.
