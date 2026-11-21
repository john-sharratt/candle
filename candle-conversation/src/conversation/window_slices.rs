//! Cuts a sealed turn's key-value chunks to a token range, and caps oversized
//! probes to the part that is a query.
//!
//! A window shares the physical chunks at its edges, carrying an offset and a
//! shorter count, and the kernel honours that offset. A range the layer cannot
//! satisfy is a fault that stops the seal, since clamping it would persist a turn
//! whose key-value state is silently short of its tokens. A whole-turn probe keeps
//! its head and most recent tail, because a tool result can run to thousands of
//! tokens and its tail is not a query.

use super::*;

/// Slice a per-layer sealing down to the chunk range `[from..to)`.
///
/// Each layer's `SealedSequence` is replaced with one whose `chunks`
/// covers just the requested block window, with the token count
/// recomputed from those chunks.  Used by the substrate seal path
/// (and by parallel section ingestion) to extract a unit's *delta*
/// bytes from a whole-sequence snapshot.
pub(crate) fn slice_per_layer_sealed(
    full: &[SealedSequence],
    from: usize,
    to: usize,
) -> crate::Result<Vec<SealedSequence>> {
    full.iter()
        .enumerate()
        .map(|(layer, seq)| {
            // **A range this layer cannot satisfy is a fault, and the seal
            // stops.** There is no correct smaller answer: the caller has
            // already resolved `[from, to)` against the slot's block count and
            // is about to write it as one indivisible turn, so a layer that
            // cannot produce it means the two disagree about what the slot
            // holds. Sealing the remainder would persist a turn whose K/V is
            // silently short of its tokens, which reads back as a coherent turn
            // and attends over the wrong span.
            //
            // It used to be `get(from..to.min(len)).unwrap_or(&[])`: `to` was
            // clamped and `from` never was, so a `from` past the layer's end
            // inverted the range and the fallback turned that into an empty
            // seal. Measured: a `repo_map` folder's FIRST turn sealed blocks
            // 77..80 against per-layer lists holding only the view's three new
            // chunks, so every layer produced nothing. The turn persisted with
            // its tokens and `chunks=0(3blk×0L)`, and nothing said so — the
            // substrate validator cannot see it either, since it checks
            // `layers × chunks_per_layer` only on turns that HAVE chunks.
            // `from > to` is checked too, and not as a formality: an INVERTED
            // range is the exact shape the old clamp produced, and both bounds
            // tested against `len` alone would let `5..3` through to
            // `chunks[5..3]` — a panic, from the one function that was made
            // fallible so a bad range could be reported instead. With
            // `from <= to <= len`, `from <= len` follows.
            if from > to || to > seq.chunks.len() {
                return Err(crate::ConversationError::Model(candle::Error::Msg(
                    format!(
                        "seal range {from}..{to} is outside layer {layer}'s {} sealed chunk(s) — \
                     the slot's block count and this layer's K/V disagree, and sealing any \
                     part of it would persist a turn whose K/V is short of its tokens",
                        seq.chunks.len(),
                    ),
                )));
            }
            let chunks: Vec<_> = seq.chunks[from..to].to_vec();
            let token_count = chunks.iter().map(|c| c.token_count as usize).sum();
            Ok(SealedSequence {
                chunks,
                token_count,
                chunk_size: seq.chunk_size,
                location: seq.location,
            })
        })
        .collect()
}

/// Window a turn's per-layer sealed K/V down to the token range
/// `[start_tok, end_tok)` as a zero-copy view over the *same* physical
/// chunks.
///
/// A turn is sealed once as a single contiguous unit, with the chat
/// template's role markers baked into the K/V grid:
/// `[user_start][no_think][user_msg][user_end][assistant_start][closed_think][prefill][response]`
///
/// `no_think` and `closed_think` are the two halves of
/// `Dialect::thinking_suppression` and are mutually exclusive — a family uses
/// the user-turn switch or the prefilled closed block, never both, and both are
/// absent on a thinking turn. `prefill` is a caller-supplied assistant seed
/// (e.g. `<tool_call>`), usually absent. `assistant_content_start` sits after
/// `closed_think` and before `prefill`: the block is scaffolding stripped from
/// the stored text, the seed is content that stays in it.
/// The compressor injects *content-only* halves of a turn — the user
/// message body or the assistant response body — without those markers,
/// so each half is derived here by windowing the sealed grid to the
/// content's token span.
///
/// Each returned `SealedSequence` references the same `ChunkGid`s the
/// turn already owns, windowing the partially-covered boundary chunks by
/// `offset` / `token_count` so no K/V bytes are copied.  The kernel
/// honours `offset` (it packs into the high 16 bits of `ChunkMeta`), so
/// a window can be injected at any tail position and read correctly.
///
/// Walking each layer's chunks with a running token start `acc` and the
/// chunk's own `c = token_count`, the chunk's slot span is `[acc, acc + c)`.
/// Intersecting it with `[start_tok, end_tok)`:
/// - empty intersection → the chunk is skipped.
/// - the intersection equals the whole chunk → the chunk is cloned as-is.
/// - otherwise the chunk straddles a window edge → a clone is taken with
///   `offset += (overlap_start - acc)` and `token_count = overlap_len`.
///   The clone shares the same `ChunkGid`s as the source chunk.
///
/// The result's `token_count == end_tok - start_tok` (clamped to the
/// sequence's own length).  `start_tok >= end_tok` yields an empty window.
pub(crate) fn window_sealed_tokens(
    sealed: &[SealedSequence],
    start_tok: usize,
    end_tok: usize,
) -> Vec<SealedSequence> {
    let mut layers: Vec<SealedSequence> = Vec::with_capacity(sealed.len());

    for seq in sealed {
        let total = seq.token_count;
        let win_start = start_tok.min(total);
        let win_end = end_tok.min(total).max(win_start);
        let mut chunks: Vec<SealedChunk> = Vec::new();

        let mut acc = 0usize;
        for chunk in &seq.chunks {
            let c = chunk.token_count as usize;
            let chunk_start = acc;
            let chunk_end = acc + c;
            acc = chunk_end;

            // Intersect the chunk's slot span with the window.
            let overlap_start = chunk_start.max(win_start);
            let overlap_end = chunk_end.min(win_end);
            if overlap_start >= overlap_end {
                // No overlap — chunk lies wholly outside the window.
                continue;
            }
            let overlap_len = (overlap_end - overlap_start) as u16;
            if overlap_start == chunk_start && overlap_len as usize == c {
                // Whole chunk inside the window — clone as-is.
                chunks.push(chunk.clone());
            } else {
                // Partial chunk: window it, sharing the same physical
                // chunk via the cloned `ChunkGid`s.
                let mut w = chunk.clone();
                w.offset = chunk.offset + (overlap_start - chunk_start) as u16;
                w.token_count = overlap_len;
                chunks.push(w);
            }
        }

        layers.push(SealedSequence {
            chunks,
            token_count: win_end - win_start,
            chunk_size: seq.chunk_size,
            location: seq.location,
        });
    }

    layers
}

/// Pure block-range windowing for a bounded rolling-window ingest (design
/// `docs/unified_wave_inference_engine.md` §4.7): given the system-prompt end
/// block `sys_end`, the ascending per-turn start blocks `turn_starts` (one per
/// sealed turn), the current `total` block count, and `window_turns`, return the
/// parent block ranges the next prefill should attend — the system prompt plus
/// the most recent `window_turns` sealed turns.
///
/// `window_turns == 0` (unbounded) or fewer sealed turns than the window returns
/// the whole parent `[(0, total)]` — a no-op, byte-for-byte the unwindowed
/// behaviour. Split out from [`Conversation::windowed_ingest_ranges`] so it is
/// unit-testable without a live substrate.
pub(crate) fn windowed_ingest_ranges_impl(
    sys_end: usize,
    turn_starts: &[usize],
    total: usize,
    window_turns: usize,
) -> Vec<(usize, usize)> {
    let whole = || {
        if total == 0 {
            Vec::new()
        } else {
            vec![(0, total)]
        }
    };
    if window_turns == 0 || turn_starts.len() <= window_turns {
        return whole();
    }
    let keep_from = turn_starts[turn_starts.len() - window_turns];
    // If the window already reaches back into (or to) the system prompt, the two
    // pieces are contiguous — borrow the whole parent.
    if keep_from <= sys_end {
        return whole();
    }
    let mut ranges = Vec::with_capacity(2);
    if sys_end > 0 {
        ranges.push((0, sys_end));
    }
    ranges.push((keep_from, total));
    ranges
}

/// Cap a whole-turn probe to the same head + trailing window the live
/// reproject scan uses: the turn's first `HEAD` signatures (the user query —
/// the strongest intent signal in the gallery domain) followed by its most
/// recent `max_tail`.
///
/// A turn is not bounded by anything the engine controls — a tool result goes
/// into it verbatim — so an uncapped probe is both a cost and a correctness
/// problem: the scan is O(probe × gallery), and the tail of a large tool
/// payload is not a query, so scoring it dilutes the turn's actual intent
/// across whatever the payload happens to mention.
pub(crate) fn cap_probe_window(probe: Vec<WideQSig>, max_tail: usize) -> Vec<WideQSig> {
    /// Head signatures kept regardless of how far the tail window has slid —
    /// mirrors the scheduler's `QUERY_HEAD_CHUNKS` (2 chunks of 32).
    const HEAD: usize = 64;
    let max_tail = max_tail.max(1);
    if probe.len() <= HEAD + max_tail {
        return probe;
    }
    let tail_lo = probe.len() - max_tail;
    let mut out = Vec::with_capacity(HEAD + max_tail);
    out.extend_from_slice(&probe[..HEAD]);
    out.extend_from_slice(&probe[tail_lo..]);
    out
}
