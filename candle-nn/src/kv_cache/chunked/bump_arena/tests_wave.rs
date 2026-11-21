//! Tests for range disjointness, alignment, and layer-phase span reuse.
//!
//! Ranges from one generation never overlap, and alignment only moves the
//! cursor forward, on the absolute address, even when the span's base is
//! unaligned. The two layer phases share their ground, and the next layer
//! reuses its phase's span rather than accumulating into it. Re-entering a
//! phase that is still open is refused, and a peak survives the reset that
//! clears the cursor, since the span is sized from its peak.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::aligned_start;

    /// Ranges from one generation never overlap — the property that removes
    /// the need for any slot table or disjointness bookkeeping (§3.6).
    #[test]
    fn ranges_within_a_generation_are_disjoint() {
        let (base, cap) = (0x1000u64, 4096usize);
        let mut cursor = 0usize;
        let mut alloc = |len: usize, align: usize| -> (u64, usize) {
            let start = aligned_start(base, cursor, align);
            assert!(start + len <= cap);
            cursor = start + len;
            (base + start as u64, len)
        };
        let a = alloc(100, 16);
        let b = alloc(200, 16);
        let c = alloc(1, 256);
        assert!(a.0 + a.1 as u64 <= b.0, "a and b overlap");
        assert!(b.0 + b.1 as u64 <= c.0, "b and c overlap");
        assert_eq!(c.0 % 256, 0, "alignment must be honoured");
    }

    /// Alignment rounds the cursor up, never down — a range must start at or
    /// after the previous one's end.
    #[test]
    fn alignment_only_moves_the_cursor_forward() {
        for align in [1usize, 4, 16, 256, 4096] {
            for cursor in 0usize..64 {
                let start = aligned_start(0, cursor, align);
                assert!(start >= cursor, "align {align} moved cursor {cursor} back");
                assert_eq!(start % align, 0);
                assert!(start - cursor < align, "over-aligned");
            }
        }
    }

    /// Alignment holds on the **absolute address**, whatever the span's base —
    /// the dense-Qwen3 fault shape: a base ≡ 8 (mod 16) made every carve whose
    /// *offset* was 256-aligned land at an address the tensor-core kernels'
    /// 16-byte `cp.async` reads fault on.
    #[test]
    fn carves_align_absolutely_even_on_an_unaligned_base() {
        for base in [8u64, 88, 0x1008, 0xb1d4c5758] {
            for cursor in [0usize, 1, 100, 4096] {
                for align in [16usize, 256] {
                    let start = aligned_start(base, cursor, align);
                    let abs = base + start as u64;
                    assert_eq!(abs % align as u64, 0, "base {base:#x} cursor {cursor}");
                    assert!(start >= cursor, "moved backward");
                    assert!(abs - (base + cursor as u64) < align as u64, "over-aligned");
                }
            }
        }
        // An aligned base costs nothing extra: identical to offset-only rounding.
        assert_eq!(aligned_start(0x1000, 100, 256), 256);
        assert_eq!(aligned_start(0x1000, 0, 256), 0);
    }
}

#[cfg(all(test, feature = "cuda"))]
pub(super) mod wave_tests {
    use super::{begin_wave, wave_domain_stats, LayerPhase};

    use candle::Result;

    /// The wave domain is process-global and `cargo test` runs tests in
    /// parallel, so these take turns. It is the *crate-wide* lock rather than a
    /// local one: the KV selection tests build backings on the same device, and
    /// a lock that only excludes this module would not exclude them.
    use super::super::super::gpu_test_lock::gpu_serial as serial;

    use super::super::tests_window::wave_tests::stream;

    /// Inside a wave, ranges come from the half and do not overlap.
    #[test]
    fn ranges_within_a_wave_are_disjoint() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let guard = begin_wave(&s, LayerPhase::Attention)?;
        let a = guard.alloc(1000, 256)?;
        let b = guard.alloc(2000, 256)?;
        assert!(a.ptr + a.len as u64 <= b.ptr, "ranges overlap");
        assert_eq!(b.ptr % 256, 0);
        // Nothing to assert about life after `drop(guard)`: `a` and `b` borrow
        // it, so a program that used them afterwards would not compile. That is
        // the property this used to check at run time.
        Ok(())
    }

    /// **The two layer phases share their ground** ([`phase_offsets`]): a layer's
    /// FFN carves from where its attention carved, once the attention guard has
    /// dropped. Safe because a range borrows its guard — the attention output the
    /// residual add reads cannot outlive the attention phase — and because
    /// neither phase opens while the other is live
    /// ([`a_layer_phase_refuses_while_the_other_is_live`]).
    #[test]
    fn the_layer_phases_share_their_ground() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let g1 = begin_wave(&s, LayerPhase::Attention)?;
        let attn = g1.alloc(4096, 256)?.ptr;
        drop(g1);

        let g2 = begin_wave(&s, LayerPhase::Ffn)?;
        let ffn = g2.alloc(4096, 256)?.ptr;
        drop(g2);

        assert_eq!(
            attn, ffn,
            "the FFN did not carve where the attention phase did"
        );
        Ok(())
    }

    /// The next layer reuses its phase's span rather than accumulating — the
    /// whole point of resetting on guard drop. Without it a 48-layer wave would
    /// hold 48 layers of transients instead of one layer's.
    #[test]
    fn the_next_layer_reuses_the_same_phase_span() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let mut first = None;
        for _layer in 0..3 {
            let g = begin_wave(&s, LayerPhase::Attention)?;
            let ptr = g.alloc(4096, 256)?.ptr;
            drop(g);
            match first {
                None => first = Some(ptr),
                Some(f) => assert_eq!(ptr, f, "a later layer moved off the span"),
            }
        }
        Ok(())
    }

    /// Re-entering a phase while it is still open is refused (principle 7): the
    /// inner guard's drop would reset the span the outer one is filling. The
    /// forward span may stand beside a layer phase — it has ground of its own.
    #[test]
    fn re_entering_a_live_phase_is_refused() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let forward = begin_wave(&s, LayerPhase::Forward)?;
        let attn = begin_wave(&s, LayerPhase::Attention)?;
        assert!(
            begin_wave(&s, LayerPhase::Attention).is_err(),
            "re-entering the live attention span must be refused"
        );
        drop(attn);
        let ffn = begin_wave(&s, LayerPhase::Ffn)?;
        assert!(
            begin_wave(&s, LayerPhase::Ffn).is_err(),
            "re-entering the live FFN span must be refused"
        );
        assert!(
            begin_wave(&s, LayerPhase::Forward).is_err(),
            "re-entering the live forward span must be refused"
        );
        drop(ffn);
        drop(forward);
        Ok(())
    }

    /// Peaks are what the span is sized from, so they must survive the reset
    /// that clears the cursor.
    #[test]
    fn peak_outlives_the_reset() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let guard = begin_wave(&s, LayerPhase::Attention)?;
        guard.alloc(8192, 256)?;
        drop(guard);
        let stats = wave_domain_stats(0).expect("domain exists");
        let peak = stats[0].1.max(stats[1].1);
        assert!(peak >= 8192, "peak {peak} lost across the reset");
        assert_eq!(stats[0].0.min(stats[1].0), 0, "a cursor failed to rewind");
        Ok(())
    }
}
