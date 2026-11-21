//! Tests for the sliding-window eviction boundary, at the exact query positions
//! where a front chunk leaves the window.
//!
//! Each case fixes the query position at which a chunk's last token exits the
//! window, and checks that eviction happens there and not one token earlier.
//! The writer and everything past the writer boundary survive any query
//! position. A ring simulation decodes two thousand tokens and checks that the
//! resident span stays within the window plus one partial chunk, and that no key
//! still inside the window is ever dropped.

use super::*;

#[cfg(test)]
pub(super) mod front_evict_tests {
    use super::front_evict_count;

    use crate::CHUNK_SIZE;

    // Full 32-token chunks (the sliding-window ring's sealed chunks) plus a
    // partial writer tail — the exact shape the DeepSeek FP8 window holds.
    pub(in super::super) fn full_chunks(n: usize) -> Vec<u32> {
        vec![CHUNK_SIZE as u32; n]
    }

    #[test]
    fn window_zero_never_evicts() {
        // window_size 0 means "no ring bound" — must never drop a chunk.
        let usages = full_chunks(10);
        assert_eq!(front_evict_count(&usages, 10, 0, 0, 10_000), 0);
    }

    #[test]
    fn nothing_evicted_before_window_fills() {
        // 4 full chunks = 128 tokens, window 128, query at pos 127 (the 128th
        // token). Lowest in-window pos = 127 - 128 + 1 = 0, so chunk 0 (pos
        // [0,32)) is still partly in window → evict nothing.
        let usages = full_chunks(4);
        // writer_start = 3: chunks 0..3 sealed, chunk 3 is the writer tail.
        assert_eq!(front_evict_count(&usages, 3, 0, 128, 127), 0);
    }

    #[test]
    fn evicts_exactly_when_front_chunk_fully_exits() {
        // window 128; chunk 0 spans abs [0,32). Its last token (pos 31) leaves
        // the window when the lowest in-window pos exceeds 31, i.e.
        // abs_pos - 128 + 1 > 31 → abs_pos > 158 → abs_pos >= 159.
        let usages = full_chunks(8);
        let writer = 7;
        // At 158: lowest in-window = 158-128+1 = 31 ≤ 31 → chunk 0 still touches.
        assert_eq!(front_evict_count(&usages, writer, 0, 128, 158), 0);
        // At 159: lowest in-window = 32 > 31 → chunk 0 fully out, evict exactly 1.
        assert_eq!(front_evict_count(&usages, writer, 0, 128, 159), 1);
        // At 191: lowest in-window = 64; chunks 0 (end 32) and 1 (end 64) both
        // ≤ 64 → evict 2. Chunk 2 (end 96) > 64 stays.
        assert_eq!(front_evict_count(&usages, writer, 0, 128, 191), 2);
    }

    #[test]
    fn base_pos_shifts_the_absolute_frame() {
        // After earlier eviction (base_pos = 64), chunk 0 now spans abs
        // [64,96). Query at 223, window 128: lowest in-window = 96. Chunk 0
        // (end 96) ≤ 96 → evictable; chunk 1 (abs [96,128), end 128) > 96 stays.
        let usages = full_chunks(6);
        assert_eq!(front_evict_count(&usages, 5, 64, 128, 223), 1);
    }

    #[test]
    fn never_evicts_the_writer_or_beyond() {
        // Even with a huge query position, the writer chunk (and anything at or
        // past writer_start) is never dropped — the ring must keep a live tail.
        let usages = full_chunks(4);
        // writer_start = 1 → only chunk 0 is ever eligible.
        assert_eq!(front_evict_count(&usages, 1, 0, 32, 1_000_000), 1);
        // writer_start = 0 → nothing is eligible.
        assert_eq!(front_evict_count(&usages, 0, 0, 32, 1_000_000), 0);
    }

    #[test]
    fn resident_token_count_stays_bounded_past_window() {
        // Faithful ring simulation: decode 2000 tokens one at a time, sealing a
        // fresh 32-token writer chunk at each boundary and evicting fronts after
        // each step. The resident token span must stay bounded ABOVE by
        // `window + one partial chunk` (the ring is O(window), not O(N)), and
        // never drop BELOW the in-window token count (no attended key is
        // evicted). This is the property `window_bytes_flat_beyond_window_size`
        // asserts of the accounting model — here proven of the real eviction.
        let window = 128usize;
        let mut resident: Vec<u32> = Vec::new(); // usages of resident chunks
        let mut base = 0u32; // tokens evicted off the front
        for tok in 0..2000usize {
            if tok % CHUNK_SIZE == 0 {
                resident.push(0); // new writer chunk at each 32-token boundary
            }
            *resident.last_mut().unwrap() += 1; // the just-decoded token
            let live_before: usize = resident.iter().map(|&u| u as usize).sum();
            let pos = base as usize + live_before - 1; // absolute pos of this token == tok
            debug_assert_eq!(pos, tok);
            let writer = resident.len() - 1; // only sealed fronts are evictable
            let drained = front_evict_count(&resident, writer, base, window, pos);
            base += resident[..drained].iter().sum::<u32>();
            resident.drain(..drained);
            let live: usize = resident.iter().map(|&u| u as usize).sum();
            assert!(
                live <= window + CHUNK_SIZE,
                "ring unbounded: resident {live} > window+chunk at tok {tok}",
            );
            assert!(
                live >= window.min(tok + 1),
                "over-evicted an in-window key: resident {live} < {} at tok {tok}",
                window.min(tok + 1),
            );
        }
    }
}
