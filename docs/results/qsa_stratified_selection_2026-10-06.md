# Stratified QSA selection — kernel and live recall, 2026-10-06

Machine: RTX PRO 5000 Blackwell 72 GB (sm_120). Model: Qwen3.8-Flash-Next
(Q4_KO), ratio 4, budget 2048. Design: `docs/qsa_stratified_selection.md`.

## Selection kernel

`cargo test --release --features cuda -p candle-transformers --test qsa_topk_bench -- --ignored --nocapture`

The values are ms per launch. Each run has a 2,000-block prompt, and the windowed
modes use 128K-position windows with an 8K recent span. "HEAD" is the
single-pass bitonic kernel before this change, which only ever ran WHOLE.

| rows | blocks | HEAD | whole | windows, candidate | windows, forced |
|---:|---:|---:|---:|---:|---:|
| 16 | 16,384 | 0.186 | 0.052 | 0.054 | 0.051 |
| 16 | 32,768 | 0.217 | 0.061 | 0.062 | 0.060 |
| 16 | 73,728 | 0.292 | 0.072 | 0.108 | 0.105 |
| 16 | 262,144 | 0.550 | 0.099 | 0.264 | 0.264 |
| 1024 | 16,384 | 0.718 | 0.166 | 0.168 | 0.158 |
| 1024 | 32,768 | 0.860 | 0.279 | 0.280 | 0.274 |
| 1024 | 73,728 | 1.103 | 0.476 | 0.668 | 0.645 |
| 1024 | 262,144 | 1.975 | 1.316 | 4.929 | 4.810 |
| 8192 | 16,384 | 4.955 | 1.382 | 1.406 | 1.336 |
| 8192 | 32,768 | 5.923 | 2.011 | 2.035 | 1.990 |
| 8192 | 73,728 | 7.354 | 3.508 | 4.927 | 4.685 |

ncu at 73,728 blocks (locked clocks):

| shape | HEAD | after |
|---|---|---|
| decode, 16 rows, whole | 366 µs, one pass on 16 blocks | 33 µs segment (16×11 blocks) + 49 µs merge |
| prefill, 1024 rows, whole | 1.10 ms | 477 µs, 77% achieved occupancy, 50% memory throughput |
| decode, 16 rows, windows + candidate | — | 43 µs segment + merge |

The tuning sequence measured:

1. **Strata added to the bitonic kernel.** Windowed decode cost 0.76–0.85 ms,
   and WHOLE regressed by 20%.
2. **Split launches.** Decode WHOLE fell to 0.32 ms. ncu showed bitonic trims
   dominating both passes, at 179 µs and 154 µs.
3. **Radix-select trims.** Decode WHOLE fell to 0.072 ms and prefill to 0.476 ms.
4. **Windows side by side, one slice each.** Windowed decode at 1M tokens fell
   from 0.725 to 0.264 ms.

A uniformity race fixed along the way was also present at HEAD. The trim
decision was read back from a shared counter that a faster thread could already
have advanced, which could split a block across barriers. It surfaced as an
intermittent `CUDA_ERROR_ILLEGAL_ADDRESS`. The fix reads the count from
`__syncthreads_count`.

## Live recall

zend on the production workspace. The prompt was 294–394K tokens: a 250K
working set plus the priming chain. Probe conversations were deleted after each
run.

**Short conversation, 5 turns, about 297K tokens.** The turns:

1. A CHUNK_SIZE question.
2. Quote the previous message.
3. Recall the constant and the caveat.
4. A Tokyo-time tool round.
5. List every question.

| | Candidate | Forced | WHOLE control |
|---|---|---|---|
| quote previous message | exact | exact | exact |
| recall constant and caveat | exact | exact | exact |
| tool round | correct | correct answer, confused post-tool reasoning | — |
| list my questions | exact, 5 of 5, placed read requests excluded | listed placed read requests as questions | — |

**Long conversation, 11 turns, tool-heavy, 383–394K tokens.** The probes:

- Turn 7: a planted codename from turn 1.
- Turn 9: the content of a middle turn.
- Turns 10 and 11: ordering around the tool round.
- Turn 8: "quote my second message".

Both Candidate and WHOLE answered turns 7, 9, 10 and 11 correctly. Both failed
turn 8 by quoting a placed indexer request. That is the separate structural
defect: placed read requests sit in the USER role.

**Replaying the original failing probe (qa-11) on WHOLE: passes.** It quotes the
previous message exactly. The failure recorded on 2026-10-05 no longer
reproduces on this build without strata. Tonight's `history_stance` prompt
rewrite targets exactly that failure mode (placed requests read as the user's
turns), but its effect has not been isolated.

## Decode cost, live

These are single-sequence decode forwards at a KV depth above 280K, taken from
the scheduler's wave lines.

| mode | mean forward | steps |
|---|---:|---:|
| WHOLE | 63.3 ms | 4,535 |
| Candidate | 72.5–77.7 ms | 3,832 / 791 |
| Forced | 122.0 ms | 1,332 |

The cost comes from the attention reading about 3× (Candidate) or 7× (Forced)
the cells, not from the selection kernel, which got faster.

## Gates on the final code (WHOLE, the default)

Full sweep: all 12 `test_parallel_batched_forwarding*` gates pass, with every
row valid. Engine probes:

| model | story | worst sustained efficiency | weight uptake |
|---|---|---:|---:|
| Qwen3-30B-A3B | 20/20 | 98% | 0% |
| Flash-Next | 8/8 | 94% | 51% |

In both, the weight zone is at its limit.

Flash-Next gate after the review fixes: C10 7.13× at 100%, BF16 ×16 at
4,140 t/s prefill and 777 t/s decode.

Test suites by crate. Each crate's CUDA build ran at one thread.

| crate | passed | failed |
|---|---:|---:|
| candle-core | 1,007 | 0 |
| candle-nn | 767 | 0 |
| candle-transformers | 1,161 | 0 |
| candle-conversation | 2,398 | 0 |
| zend | 609 | 0 |
| npcd | 2,231 | 0 |
| zend-tools, web, npc-map, target-prune | 1,149 | 0 |

The source tree was unchanged by the test runs.
