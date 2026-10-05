# Stratified QSA selection

Qwen3.8-Flash-Next selects, per query, the `top_k + ratio − 1 = 2051` cells
(513 blocks of 4) whose indexer score `Σ_h ReLU(⟨q_h, k̄_b⟩)` is highest, plus
the query's own partial block. Nothing else is guaranteed a seat: there is no
sliding window and no sink.

## The problem

The budget is a fixed count, so the share of a deep context it can reach
shrinks with depth: 513 of ~73,700 blocks at 294K tokens. Every candidate
competes with every other. At zend's production size — a 250K-token working set
of ~150 near-identical placed file reads, then the dialogue — the dialogue's own
short turns are outscored and drop out of the selection entirely. The model
then cannot see what it was asked one turn earlier.

This was measured, not inferred. At ~294K the dialogue could not quote its own
previous message. Two controls restored it: the dense control (a budget past
the context, every cell read) and the same conversation at a 68K context. The
K/V, positions and index rows are therefore sound. The fault is competition.

## The design

The candidate blocks are cut into **windows** of `window_blocks` blocks, walking
forward from block 0. Each window ranks its own **pool** and spends the **whole
budget** on it. A window's pool is:

- its own blocks;
- the system prompt's blocks (`IndexCache::prompt_blocks`), wherever they sit;
- when the recent mode is **Candidate**, the `recent_blocks` blocks nearest the
  query.

When the recent mode is **Forced**, the recent span is left out of every pool and
attended whole.

The attended set is the union of the windows' choices, plus the forced span,
plus the query's tail. A block chosen by several windows (a prompt block, or a
recent candidate) is attended once, at its widest cut.

Each window asks the checkpoint the question it was trained on — the best 2051
cells among ~128K positions — instead of the best 2051 among 294K. The
prompt is ranked in every window, so it competes everywhere. In Candidate mode
the recent span gets as many chances as there are windows, and is still chosen
only on score. In Forced mode it is always attended.

**One window over every candidate with no recent span (`Strata::WHOLE`, the
default) is exactly the checkpoint's selection.** This is not an approximation:
the oracle, the kernel and the gates all take that path unchanged.

Cost: `windows × 513` blocks attended instead of 513. At 294K with 128K windows
that is 3 × 513, plus 2,048 forced blocks in Forced mode. That is about 6.2K
cells in Candidate mode and 14.4K cells in Forced mode, against a dense read of
294K.

## Where it lives

| Piece | File |
|---|---|
| `Strata`, `Recent`, `StrataTokens`, the oracle `selection_entries`, bounds `max_entries_for` / `max_gathered_for` | `candle-transformers/src/models/qwen4exp/qsa_select.rs` |
| The kernel (single pass, plus the split segment/merge pair) | `candle-kernels/src/simple/qsa_topk.cu` |
| `SelectionTable`, `selection_stride`, `IndexCache::{set_prompt_end, prompt_blocks}` | `candle-transformers/src/models/qwen4exp/indexer.rs` |
| `Qwen4ExpBatched::{set_selection_strata, set_selection_prompt}` | `candle-transformers/src/models/qwen4exp/wave.rs` |
| `ManagedBatchedModel::set_selection_prompt` (no-op for every other model) | `candle-transformers/src/models/batched_inference.rs` |
| `ModelBuilder::qsa_strata` | `candle-conversation/src/models/builder.rs` |
| The prompt span is declared after each projection walk, from the leading run of sections and glue (`prompt_end_of`) | `candle-conversation/src/scheduler/projection_assembler.rs` |
| `--qsa-window`, `--qsa-recent`, `--qsa-recent-mode` | `zend/src/main.rs` |

The strata is stated in positions (`StrataTokens`) and turned into blocks at
the selecting layers' compression ratio, which must be a single ratio. It is
refused at load when the windows would gather more entries than the kernel's
union buffer holds (`MAX_ENTRIES = 16384`) at the deepest position the RoPE
schedule reaches.

That check runs at `reach / ratio` plus one more window. A sequence assembled
from sealed pages has a short block at every page boundary, so its candidate
count can exceed `reach / ratio`. The extra window covers any number of
boundaries narrower than a window; a sequence past even that is refused, by
name, by the wave that reaches it. The check is re-run when the budget changes,
and skipped for a budget at or past the reach, since nothing selects there.

## The kernel

Keys are `bits(score) << 32 | (0xFFFFFFFF − block)`. Scores are non-negative,
so comparing keys as unsigned integers orders them exactly by
`(score desc, block asc)`, which is the reference order.

**Streaming with a radix-select trim.** Candidates stream through a shared
buffer of 1024 keys against a running threshold. When the buffer could not
absorb another chunk, an MSB radix select (byte histograms over the keys, held
in registers) finds the top `keep` and compacts them to the front, and the
smallest of them becomes the threshold. Nothing is sorted until the row's
chosen blocks are sorted once, by block, for output.

That last-ranked key is also where the budget's partial cut lands.

The trim decision is made from a per-thread count advanced by
`__syncthreads_count`. A count read back from shared memory is not uniform: a
thread that runs ahead into the next chunk changes it before a slow thread has
read it, and a split decision leaves the block across two barriers.

**Split launches.** One block per row fills the device for a prefill tile and
leaves it idle for decode. A launch with fewer than about one row per SM
(`qsa_topk_split_parts`) does two things:

- **Segment kernel:** runs each window in its own blocks, and cuts large windows
  into `≈ √(window / keep)` slices. Each slice writes its top `keep` keys to a
  fixed scratch (`SPLIT_KEYS`).
- **Merge kernel:** ranks each window's slice survivors and writes the row.

This is exact: a key in a window's top `keep` is in the top `keep` of its own
slice.

**The union.** Window entries are sorted by `(block, cells)`. Each run's last
entry (its widest cut) is kept by a block-wide scan. The forced span and the
tail sit above every pool, so they are appended in order unsorted.

## Measured

RTX PRO 5000 Blackwell, ratio 4, budget 2048, a 2,000-block prompt. Each figure
is ms per launch (`tests/qsa_topk_bench.rs`), followed by its HEAD value where
one exists.

| rows | blocks (tokens) | HEAD whole | whole | 128K windows, candidate | 128K windows, forced |
|---:|---:|---:|---:|---:|---:|
| 16 | 73,728 (295K) | 0.292 | 0.072 | 0.108 | 0.105 |
| 16 | 262,144 (1M) | 0.550 | 0.099 | 0.264 | 0.264 |
| 1024 | 73,728 | 1.103 | 0.476 | 0.668 | 0.645 |
| 1024 | 262,144 | 1.975 | 1.316 | 4.929 | 4.810 |
| 8192 | 73,728 | 7.354 | 3.508 | 4.927 | 4.685 |

Under ncu at 294K, WHOLE decode (16 rows) is 33 µs segment plus 49 µs merge,
against 366 µs for HEAD's single pass. WHOLE prefill (1024 rows) is 477 µs at
77% achieved occupancy, against 1.10 ms.
