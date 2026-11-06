# Inline read injection: a served page placed from its reading

**Status: withdrawn (2026-11-06), measured.** The placed span was built and
run (`52b6e7d4d`, `f4b1923fe`) and removed after the measurement below: a
page placed with its edges prefilled is read but judged missing, and the
tail that would fix the judgment is most of a page, so a single page gains
nothing. Reads prefill for real (`docs/zend_working_set.md` §4.5); a read that
leaves the context is forgotten and read again, not placed. What follows is
the design as it was built and what it measured, kept so the experiment is
not repeated blind.

## 1. The idea

A `file_read` of a page a `code_reading` conversation already holds prefills
it again: about 2,200 tokens for page 1 of `zend/src/tool_guidance.rs`. The
page's K/V already exists, sealed in the reading, and keys are stored
un-rotated, so its chunks can be placed at any position. The recurrent state
cannot be placed: Flash-Next is 36 Gated DeltaNet layers and 12 attention
layers, and a DeltaNet layer keeps no per-token K/V, only the state its tokens
advanced.

So the turn answering the call prefilled its edges and placed its middle:

```text
<tool_response>  [page head]  [page middle — placed]  [page tail]  </tool_response>
 prefilled        prefilled    the reading's chunks    prefilled    prefilled
```

The recurrence ran over the opening, the page's first and last chunks and the
close, at their own positions; every attention layer read the middle.

## 2. How it was built

- **Resolution**, at submit: the run's interior found in the source
  conversation's turns (exactly once), cut on the source turn's grid — each
  index page's start, every 32 tokens into a page, each page's end — from the
  `edge`-th grid point past its start to the `edge`-th before its end.
- **Placement**: the prefill stopped at the span's start. As the next wave
  group formed, the index tail closed into a page, the source's hot K/V
  windowed to the span (`window_sealed_tokens`) went on with
  `inject_sealed_at_tail`, the source's index pages were sliced to the span on
  4-token row boundaries and pushed, and the prefill carried on from the
  span's end. Placing as a group *completed* failed: the index pages are placed
  in the index tenant, which a forward still in flight refuses ("kv arena
  creation deferred: wave in flight").
- **Seal and reprojection** needed nothing: both take the turn by its blocks
  and its pages by width.

## 3. What it measured

`candle-incontext-probe/small-guide`, restored onto
`candle-incontext-probe/small-guide-restored` (`POST
/v1/conversations/{id}/restore`), target turn 1 with the real page 1 inline;
the probe asks for two constants and to say VISIBLE or NOT VISIBLE first. BW,
build `4453-f4b1923fe`, 2026-11-06:

| Run | Prefill | Answer |
|---|---|---|
| Page prefilled inline, sparse, seeds 1–3 | 830–1,056 ms | VISIBLE 3/3, both constants exact |
| Placed, edge 2 (57 head + 38 tail tokens), sparse, seeds 1–3 | 181–187 ms | NOT VISIBLE 3/3, "truncated and garbled" |
| Placed, edge 2, dense, seeds 1–2 | 481–567 ms | NOT VISIBLE 2/2 |
| Page prefilled inline, dense, seed 1 | 1,303 ms | VISIBLE |

2,112 tokens were placed, at prefill position 57, with one sliced page. Dense
attention fails alike, so the index pages are not the cause.

**The model describes the recurrence's path, not the K/V.** Its transcript of
what it read:

```text
//! The prompt carries full definitions for only70|     /// **A refused call learns the signature.**
```

"…for only" is where the prefilled head ends and "70| /// **A refused call…"
is where the prefilled tail begins — head straight into tail, the path the
state took. Attention did read the middle: about 20% of the decode's top
attention picks fell in the placed range (32% with the page prefilled), and
the model quoted anchor lines from inside it (`UFVs| const INVALID_ARGUMENTS`,
`o_Lm| const SUGGESTIONS`). It did not take the code lines that carry their
values, and judged the page truncated.

## 4. Why it does not pay

This is the split `docs/recurrent_reply_calibration.md` §1 measured and the
literature names: attention K/V carries literal recall, the recurrent state
carries how the model responds — its judgment of what it has
([What Attention Recalls and Recurrence Controls in Hybrid Language Models](https://arxiv.org/html/2609.04434)).
The recurrent state cannot be rebuilt from K/V: a DeltaNet layer's input is the
residual stream at that layer, which needs every layer below it, MoE included,
and a layer's keys are a low-rank projection of it. It is prefilled, stored, or
approximated.

The published approximation is a tail: reuse the attention K/V and replay only
the last 5–10% of the prefix through the recurrence
([Tail-Replay](https://arxiv.org/html/2608.30310), 96–99.9% of full quality at
9–14× TTFT on 32K prefixes), which works because the state is dominated by its
recent tail. On this architecture the median DeltaNet head's 10⁻³ horizon is
about 220 tokens and the slowest are far longer
([DASC](https://arxiv.org/html/2608.30386), Qwen3-Next). A tail long enough to
put the seam out of the state's window is hundreds to ~2,000 tokens — most of
a 2,200-token page. A page is the unit a read returns, so per page there is
nothing left to save; only a multi-page round long against its tail would gain,
and that needs spans across pages and turns for a saving reads do not need:
one page prefills in about a second.

Exact composition stores state instead (`S = T·S_prefix + S_seg`,
[HYPIC](https://arxiv.org/html/2607.01299),
[LinearKV](https://arxiv.org/html/2608.11231)) — about two state-sized tensors
per DeltaNet layer per segment, ~115 MB per page here, the cost
`docs/recurrent_reply_calibration.md` was withdrawn for.

## 5. What replaced it

Reads run for real and are dialogue turns like any other. A read turn past the
dialogue's window is projected only when provenance selects it, whole; the
summariser that would compress it is off (`zend/src/prompts/projection.yaml`,
the dormant `summary:` block, for the false memories it wrote). A forgotten
read is read again when it is needed.

The recurrence still carries a decayed trace of a forgotten read — the mirror
of §3, a state that remembers reading what attention no longer holds — so the
model is told how to treat it: the system prompt's `placed_content` rule says
that something remembered and no longer here has faded, and to fetch it again
rather than guess. What a forgotten read leaves is that rule and the turns
around it; nothing stands in its place.
