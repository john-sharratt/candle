# Inline read injection: a served page placed from its reading

**Status: built for the diagnostic decode; the live serve is not built.** This
document is authoritative for the placed span, its resolution and the
scheduler step that places it.

## 1. What it is for

A `file_read` of a page a `code_reading` conversation already holds prefills
the page again: about 2,900 tokens for a page of `zend/src/tool_guidance.rs`.
The page's K/V already exists, sealed in the reading, and keys are stored
un-rotated (`SealedChunk`, `types.rs`), so the same chunks can be placed at
any position. What cannot be placed is the recurrent state: Flash-Next is 36
Gated DeltaNet layers and 12 attention layers, and a DeltaNet layer keeps no
K/V for a span, only the state its tokens advanced
(`docs/recurrent_reply_calibration.md` §1).

So the turn that answers the call prefills its edges and places its middle:

```text
<tool_response>  [page head]  [page middle — placed]  [page tail]  </tool_response>
 prefilled        prefilled    the reading's chunks    prefilled    prefilled
```

The recurrence runs over the response's opening, the page's first and last
chunks and the response's close, each at its own position. The middle is
placed: every attention layer reads it, and the tail's DeltaNet layers above
the first attention layer take what those layers read of it through the
residual stream. The state passes over the middle unchanged.

## 2. The placed span

A turn's options name a run of its user half and the conversation that holds
the same text (`TurnOptions::placed`, a [`PlacedText`]): the run's bytes, the
source timeline, and how many chunks of each edge to prefill.

Resolution (`placed_span::place`), at submit:

1. The run's ids are the half's encoding over its bytes (`turn_text::token_span`:
   a tool round's body is a literal piece of its own, so its ends are token
   boundaries).
2. The run's interior — the run less [`SLACK`] ids at each end, where a
   multi-call round's `[i/n file_read]` label can merge with the body's first
   token — is found in the source's turns. It must occur exactly once across
   all of them.
3. The placed range is cut on the source turn's **grid**: every index page's
   start, each `CHUNK_SIZE` (32) tokens into a page, and each page's end. A
   page is pooled in blocks of the model's QSA ratio (4 on Flash-Next), which
   divides 32, so a range on the grid slices whole blocks of every page it
   crosses. The range starts at the `edge`-th grid point past the interior's
   start and ends at the `edge`-th before its end — `edge = 2` prefills one
   partial chunk and one whole chunk at each edge.
4. The result is a [`PlacedSpan`]: the prefill positions the placed tokens
   occupy, the source turn, and their positions in the source turn's grid.

A run that is not found, is found twice, or is too short to leave a range
between its edges refuses the submit.

## 3. Placing it

The prefill is already cut into chunks, one per wave group; a chunk that
would reach into a placed span stops at its start (`placed_span::snap_advance`).
When the group that ends there completes (`complete_wave_group`), every layer
has run it, and the scheduler places the span before the next group forms:

1. **The index tail closes** (`close_positional_page`): the head's QSA rows
   become a page, so the placed pages land on an empty tail.
2. **The K/V goes in** (`inject_sealed_at_tail`): the source turn's hot K/V,
   windowed to the placed range (`window_sealed_tokens`, zero-copy). The
   writer boundary moves past it and the sequence's offset advances by its
   width, so the tail prefills at the positions after it.
3. **The index pages go in** (`push_positional_state`): each source page the
   range crosses, sliced to it (`slice_positional_page`, the model's).
4. The prefill's cursor moves past the span; the next group prefills the tail.

The source turn is lifted hot and pinned with the projection's working set at
admission (`turn_keys_for_elevate`). A source with no hot K/V when the span is
reached fails the turn.

Nothing writes near the placed chunks: they sit below the writer boundary, as
every placed turn's do, and the seal's compression passes chunks already in a
quantized format through unchanged (`quantize_sealed_in_place`).

## 4. What follows from it

- **Seal and reprojection** take the turn from `turn_start_parent_blocks` and
  its pages by width (`seal_positional_tail_span`), so both carry the placed
  chunks and their pages with the rest of the turn. The page boundary at each
  edge of the span is what lets the width walk cover the turn exactly.
- **The recurrent conv** keeps the last three raw inputs of the head, so the
  tail's first rows convolve against them rather than the middle's last. The
  middle's raw inputs are not stored; the seam is measured, not corrected.
- **The cost** is one extra prefill group for this turn. The middle costs no
  forward.

## 5. Tests

- Resolution: the run found in one turn and refused when absent or twice; the
  range on the grid of ragged pages; a run too short for its edges refused.
- The snap: a chunk stops at a span's start.
- Page slicing (`qwen4exp::paged_index::slice_page`): a sliced page's rows are
  the source's rows over the range, against raw expected values, and a range
  off the block grid or into the open block is refused.

The behavioural test is the diagnostic decode (`DiagRun::placed`) on the
restored `candle-incontext-probe/small-guide` conversation: the page inline,
fully prefilled, against the same page placed with its edges prefilled.
