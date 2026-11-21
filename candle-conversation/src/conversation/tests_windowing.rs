//! Tests how sealed turns are windowed for injection, so that a span's key-value
//! state and its index pages describe the same tokens.
//!
//! A window shares the chunks it cuts rather than copying them, and a span that
//! begins and ends inside one chunk emits that chunk twice with disjoint windows.
//! A layer slice that cannot be satisfied is an error, never an empty seal. The
//! K/V half and the index-page half are checked against each other, since nothing
//! downstream compares them and a mismatch is silent below the identity threshold.

use super::*;

#[cfg(test)]
pub(super) mod windowed_ingest_tests {
    use super::windowed_ingest_ranges_impl as w;

    #[test]
    fn window_bounds_to_system_prompt_plus_last_n_turns() {
        // system prompt occupies [0,2); five sealed turns start at 2,5,9,12,16;
        // current total is 20 blocks.
        let sys = 2;
        let starts = [2usize, 5, 9, 12, 16];
        // Unbounded (0) or window >= turn count → whole parent (no-op).
        assert_eq!(w(sys, &starts, 20, 0), vec![(0, 20)]);
        assert_eq!(w(sys, &starts, 20, 5), vec![(0, 20)]);
        assert_eq!(w(sys, &starts, 20, 9), vec![(0, 20)]);
        // Keep last 2 turns → system [0,2) + [starts[3]=12, 20).
        assert_eq!(w(sys, &starts, 20, 2), vec![(0, 2), (12, 20)]);
        // Keep last 1 → [0,2) + [16, 20).
        assert_eq!(w(sys, &starts, 20, 1), vec![(0, 2), (16, 20)]);
        // No system prompt → single tail range.
        assert_eq!(w(0, &starts, 20, 2), vec![(12, 20)]);
        // Empty parent → no ranges.
        assert_eq!(w(0, &[], 0, 3), Vec::<(usize, usize)>::new());
        // Window reaches into the system-prompt region (keep_from <= sys_end) →
        // contiguous → whole parent.
        assert_eq!(w(13, &starts, 20, 2), vec![(0, 20)]);
    }
}

/// The seal-range guard on [`slice_per_layer_sealed`]. It exists because a range
/// the layers cannot satisfy used to be clamped into an EMPTY seal, persisting a
/// turn with its tokens and no K/V; so every rejection path has to return the
/// error, never panic and never silently narrow.
/// A `SealedChunk` carries RAII per-head `ChunkGid`s, so these cases use
/// EMPTY layers: the guard reads `chunks.len()` and nothing else, which is
/// exactly what both rejection paths turn on. The multi-chunk success path is
/// what every production seal exercises.
#[cfg(test)]
pub(super) mod slice_per_layer_sealed_tests {
    use super::slice_per_layer_sealed;
    use candle_nn::kv_cache::{ArenaLocation, SealedSequence};

    fn empty_layer() -> SealedSequence {
        SealedSequence {
            chunks: Vec::new(),
            token_count: 0,
            chunk_size: 32,
            location: ArenaLocation::Cpu,
        }
    }

    /// An empty range over an empty layer is legitimate — a section pin that
    /// sealed nothing — and must slice cleanly rather than be rejected.
    #[test]
    fn an_empty_range_is_not_an_error() {
        let out = slice_per_layer_sealed(&[empty_layer()], 0, 0).expect("empty range");
        assert_eq!(out.len(), 1);
        assert!(out[0].chunks.is_empty());
        assert_eq!(out[0].token_count, 0);
    }

    /// The `chunks=0(3blk×0L)` shape that started this: a range the layer cannot
    /// satisfy must report, not narrow.
    #[test]
    fn a_range_past_a_layers_end_is_an_error() {
        let err = slice_per_layer_sealed(&[empty_layer()], 77, 80)
            .expect_err("an empty layer cannot satisfy 77..80");
        let msg = err.to_string();
        assert!(msg.contains("seal range 77..80"), "{msg}");
        assert!(
            msg.contains("layer 0"),
            "names the layer that disagrees: {msg}"
        );
    }

    /// **An inverted range is an error, not a panic.** Both bounds tested against
    /// `len` alone would admit `5..3` and then panic in the slice — from the one
    /// function made fallible so a bad range could be reported instead.
    #[test]
    fn an_inverted_range_is_an_error_not_a_panic() {
        let err = slice_per_layer_sealed(&[empty_layer()], 5, 3).expect_err("5..3 is inverted");
        assert!(err.to_string().contains("seal range 5..3"), "{err}");
    }

    /// Zero layers yields zero sequences and NO error — which is exactly why this
    /// guard could not have caught the missing-K/V seal on its own: with no
    /// layers the closure never runs, so the caller must treat an empty result as
    /// a failed seal rather than trusting `Ok`.
    #[test]
    fn no_layers_yields_no_sequences_and_no_error() {
        let out = slice_per_layer_sealed(&[], 77, 80).expect("no layers to check");
        assert!(out.is_empty());
    }
}

#[cfg(test)]
pub(super) mod window_sealed_tokens_tests {
    use super::cap_probe_window;
    use super::window_sealed_tokens;
    use crate::provenance::WideQSig;
    use candle_nn::kv_cache::{ArenaLocation, SealedChunk, SealedSequence};

    /// Build a one-layer sealed turn with the given per-chunk token
    /// counts.  Each chunk gets a distinct detached gid (1000 + idx) so
    /// the boundary-sharing assertion can compare raw ids.
    fn one_layer(counts: &[u16]) -> Vec<SealedSequence> {
        let chunks: Vec<SealedChunk> = counts
            .iter()
            .enumerate()
            .map(|(i, &c)| SealedChunk::for_test(1000 + i as i64, c))
            .collect();
        let token_count = counts.iter().map(|&c| c as usize).sum();
        vec![SealedSequence {
            chunks,
            token_count,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }]
    }

    #[test]
    fn window_spanning_both_boundaries_windows_first_and_last_chunk() {
        // Four chunks [32, 32, 32, 20] = 116 tokens. Window [20, 90) starts
        // mid-chunk-0 (offset 20) and ends mid-chunk-2 (offset 90 -> the
        // chunk spanning [64,96) is cut at local index 26).
        let sealed = one_layer(&[32, 32, 32, 20]);
        let first_gid = sealed[0].chunks[0].gids.as_slice()[0].raw();
        let last_gid = sealed[0].chunks[2].gids.as_slice()[0].raw();

        let win = window_sealed_tokens(&sealed, 20, 90);

        // Chunk-0 partial (20..32 -> 12 tokens), chunk-1 whole (32),
        // chunk-2 partial (64..90 -> 26 tokens). Chunk-3 wholly outside.
        assert_eq!(win[0].chunks.len(), 3);

        // First chunk: windowed at the window start — offset advanced to 20,
        // token_count = 32 - 20 = 12.
        let first = &win[0].chunks[0];
        assert_eq!(first.offset, 20);
        assert_eq!(first.token_count, 12);

        // Middle chunk: wholly inside — cloned as-is.
        assert_eq!(win[0].chunks[1].offset, 0);
        assert_eq!(win[0].chunks[1].token_count, 32);

        // Last chunk: windowed at the window end — offset unchanged (its
        // span begins inside the window), token_count = 90 - 64 = 26.
        let last = &win[0].chunks[2];
        assert_eq!(last.offset, 0);
        assert_eq!(last.token_count, 26);

        // Sequence-level token count equals the window width, and the
        // summed chunk token counts reconstruct it.
        assert_eq!(win[0].token_count, 70);
        let sum: usize = win[0].chunks.iter().map(|c| c.token_count as usize).sum();
        assert_eq!(sum, 90 - 20);

        // The boundary chunks are physically SHARED: the windowed clones
        // carry the same raw gid as their source chunk.
        assert_eq!(first.gids.as_slice()[0].raw(), first_gid);
        assert_eq!(last.gids.as_slice()[0].raw(), last_gid);
    }

    #[test]
    fn empty_window_yields_no_chunks() {
        let sealed = one_layer(&[32, 10]);
        // start >= end -> empty window.
        let win = window_sealed_tokens(&sealed, 20, 20);
        assert_eq!(win[0].chunks.len(), 0);
        assert_eq!(win[0].token_count, 0);
    }

    #[test]
    fn window_beyond_total_clamps_to_sequence_length() {
        let sealed = one_layer(&[32, 10]);
        let win = window_sealed_tokens(&sealed, 0, 1000);
        assert_eq!(win[0].chunks.len(), 2);
        assert_eq!(win[0].token_count, 42);
        assert_eq!(win[0].chunks[0].token_count, 32);
        assert_eq!(win[0].chunks[1].token_count, 10);
    }

    /// **The reasoning hole, as the projection cuts it.** A turn is windowed to
    /// `[0, span.offset) ++ [span.end(), total)` and the two halves are
    /// concatenated per layer — so a span that begins and ends INSIDE one
    /// physical chunk emits that chunk twice, as two disjoint windows sharing
    /// one refcounted gid.
    ///
    /// The gid identity is the assertion that matters: it is what makes the
    /// hole free. Copying the chunk would pass every token-count check here and
    /// silently double a turn's arena footprint on every projection.
    #[test]
    fn windowing_out_a_span_inside_one_chunk_emits_that_chunk_twice() {
        // Three chunks [32, 32, 20] = 84. Reasoning at [40, 50) sits wholly
        // inside chunk-1 (which spans [32, 64)).
        let sealed = one_layer(&[32, 32, 20]);
        let mid_gid = sealed[0].chunks[1].gids.as_slice()[0].raw();
        let head = window_sealed_tokens(&sealed, 0, 40);
        let tail = window_sealed_tokens(&sealed, 50, 84);

        // Head: chunk-0 whole, chunk-1 cut at local 8.
        assert_eq!(head[0].token_count, 40);
        assert_eq!(head[0].chunks.len(), 2);
        assert_eq!(head[0].chunks[1].offset, 0);
        assert_eq!(head[0].chunks[1].token_count, 8);
        // Tail: chunk-1 from local 18, then chunk-2 whole.
        assert_eq!(tail[0].token_count, 34);
        assert_eq!(tail[0].chunks.len(), 2);
        assert_eq!(tail[0].chunks[0].offset, 18);
        assert_eq!(tail[0].chunks[0].token_count, 14);
        assert_eq!(tail[0].chunks[1].token_count, 20);

        // The same physical chunk, twice, with disjoint windows — shared, not
        // copied.
        assert_eq!(head[0].chunks[1].gids.as_slice()[0].raw(), mid_gid);
        assert_eq!(tail[0].chunks[0].gids.as_slice()[0].raw(), mid_gid);

        // The concatenation the projection injects covers the turn minus the
        // reasoning, exactly.
        let width = head[0].token_count + tail[0].token_count;
        assert_eq!(width, 84 - 10);
    }

    /// A span abutting the end of the turn leaves an empty tail, and a span
    /// abutting the start an empty head — both must contribute zero chunks
    /// rather than a zero-length window, which would put a chunk with no valid
    /// tokens into the slot's block list.
    #[test]
    fn a_span_at_either_edge_leaves_no_empty_chunk() {
        let sealed = one_layer(&[32, 32]);
        // Reasoning runs to the end.
        let tail = window_sealed_tokens(&sealed, 64, 64);
        assert_eq!(tail[0].token_count, 0);
        assert!(tail[0].chunks.is_empty());
        // Reasoning starts at the very beginning.
        let head = window_sealed_tokens(&sealed, 0, 0);
        assert_eq!(head[0].token_count, 0);
        assert!(head[0].chunks.is_empty());
    }

    /// **The two halves must describe the same tokens, checked against each
    /// other rather than each against an expectation.**
    ///
    /// A windowed turn injects K/V from [`window_sealed_tokens`] and rows from
    /// [`index_pages::without_span`]. Nothing downstream compares them — a
    /// mismatch is silent below the QSA identity threshold and a refused select
    /// above it — so the comparison belongs here, over the real functions on
    /// both sides.
    ///
    /// This is the composition that failed on the daemon: the K/V half was right
    /// and the page half spanned an extra 323 tokens, and no unit test looked at
    /// the two together.
    #[test]
    fn a_windowed_turns_kv_and_its_retained_pages_span_the_same_tokens() {
        use crate::index_pages;

        // A turn of 84 tokens across chunks [32, 32, 20], sealed as three pages
        // — the shape the cuts produce: [prefill][reasoning][answer].
        let sealed = one_layer(&[32, 32, 20]);
        let total = 84usize;
        let cases: [(usize, usize, &[usize]); 3] = [
            // (span start, span end, page widths)
            (40, 50, &[40, 10, 34]), // reasoning in the middle
            (0, 12, &[12, 72]),      // reasoning abutting the start
            (60, 84, &[60, 24]),     // reasoning running to the end
        ];

        for (start, end, widths) in cases {
            let pages: Vec<(usize, Vec<u8>)> = widths.iter().map(|w| (*w, vec![0u8; 4])).collect();
            let blob = index_pages::encode(&pages);

            // The K/V half: everything either side of the span.
            let head = window_sealed_tokens(&sealed, 0, start);
            let tail = window_sealed_tokens(&sealed, end, total);
            let kv_tokens = head[0].token_count + tail[0].token_count;

            // The index half, from the same span.
            let kept = index_pages::without_span(&blob, start..end)
                .unwrap_or_else(|| panic!("pages must align for span {start}..{end}"));
            let page_tokens: usize = index_pages::decode(&kept)
                .expect("kept pages decode")
                .iter()
                .map(|(w, _)| *w)
                .sum();

            assert_eq!(
                kv_tokens, page_tokens,
                "span {start}..{end}: the windowed K/V spans {kv_tokens} token(s) but its \
                 retained pages span {page_tokens}"
            );
            assert_eq!(
                kv_tokens,
                total - (end - start),
                "and both equal the turn minus its span"
            );
        }
    }

    /// When the pages do NOT line up with the span, both halves must stay whole
    /// together — the refusal in `without_span` is what the caller keys off, and
    /// a turn injected whole is consistent where a half-windowed one is not.
    #[test]
    fn a_misaligned_span_leaves_both_halves_whole() {
        use crate::index_pages;

        let sealed = one_layer(&[32, 32, 20]);
        let total = 84usize;
        // Pages [40, 44]: the span 40..50 ends inside the second page.
        let pages: Vec<(usize, Vec<u8>)> =
            [40usize, 44].iter().map(|w| (*w, vec![0u8; 4])).collect();
        let blob = index_pages::encode(&pages);

        assert!(
            index_pages::without_span(&blob, 40..50).is_none(),
            "a span ending inside a page must be refused, not guessed at"
        );

        // The caller's response is to inject the turn whole: full K/V, full pages.
        let whole = window_sealed_tokens(&sealed, 0, total);
        let page_tokens: usize = index_pages::decode(&blob)
            .unwrap()
            .iter()
            .map(|(w, _)| *w)
            .sum();
        assert_eq!(whole[0].token_count, total);
        assert_eq!(page_tokens, total, "whole means whole on both sides");
    }

    #[test]
    fn chunk_aligned_window_does_not_window_chunks() {
        // Aligned window [32, 64): keeps only the second of three chunks.
        let sealed = one_layer(&[32, 32, 32]);
        let win = window_sealed_tokens(&sealed, 32, 64);
        assert_eq!(win[0].chunks.len(), 1);
        assert_eq!(win[0].chunks[0].token_count, 32);
        assert_eq!(win[0].chunks[0].offset, 0);
        assert_eq!(win[0].token_count, 32);
    }

    #[test]
    fn windowing_the_user_and_assistant_content_halves() {
        // A turn laid out as [markers | user_msg | markers | response]
        // across chunks [32, 32, 20]. Content bounds: user [4, 50),
        // assistant content starts at 56. The two halves carry only the
        // content span, never the leading/trailing template markers.
        let sealed = one_layer(&[32, 32, 20]);
        let user = window_sealed_tokens(&sealed, 4, 50);
        let asst = window_sealed_tokens(&sealed, 56, 84);

        assert_eq!(user[0].token_count, 46);
        // user-half: chunk-0 windowed (4..32 -> 28), chunk-1 windowed (32..50 -> 18).
        assert_eq!(user[0].chunks.len(), 2);
        assert_eq!(user[0].chunks[0].offset, 4);
        assert_eq!(user[0].chunks[0].token_count, 28);
        assert_eq!(user[0].chunks[1].offset, 0);
        assert_eq!(user[0].chunks[1].token_count, 18);

        assert_eq!(asst[0].token_count, 28);
        // assistant-half: chunk-1 windowed (56..64 -> 8), chunk-2 whole (20).
        assert_eq!(asst[0].chunks.len(), 2);
        assert_eq!(asst[0].chunks[0].offset, 24);
        assert_eq!(asst[0].chunks[0].token_count, 8);
        assert_eq!(asst[0].chunks[1].token_count, 20);
    }

    /// A probe within the window is untouched — the common case is a normal turn.
    #[test]
    fn cap_probe_window_leaves_a_normal_turn_whole() {
        let probe = vec![WideQSig::default(); 200];
        assert_eq!(cap_probe_window(probe.clone(), 256).len(), 200);
        // Exactly at the boundary (HEAD 64 + tail 256) is still whole.
        let probe = vec![WideQSig::default(); 320];
        assert_eq!(cap_probe_window(probe, 256).len(), 320);
    }

    /// The failure case: a turn carrying a large tool payload. The probe becomes
    /// head + tail rather than the whole 5.7k-token turn.
    #[test]
    fn cap_probe_window_bounds_an_oversized_turn() {
        let mut probe: Vec<WideQSig> = (0..5702)
            .map(|i| WideQSig {
                n_heads: i as u16,
                words: Vec::new(),
            })
            .collect();
        probe.truncate(5702);
        let capped = cap_probe_window(probe, 256);
        assert_eq!(capped.len(), 64 + 256);
        // The head is the turn's opening — the query — not the payload's tail.
        assert_eq!(capped[0].n_heads, 0);
        assert_eq!(capped[63].n_heads, 63);
        // …followed by the most recent window, ending at the turn's last token.
        assert_eq!(capped[64].n_heads, (5702 - 256) as u16);
        assert_eq!(capped.last().unwrap().n_heads, 5701);
    }

    /// A zero cap must not panic or produce an empty probe.
    #[test]
    fn cap_probe_window_tolerates_a_zero_cap() {
        let probe = vec![WideQSig::default(); 1000];
        assert_eq!(cap_probe_window(probe, 0).len(), 65);
    }
}
