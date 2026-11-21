//! Tests that a cache survives its record byte for byte, and resumes appending as
//! if it had never stopped.
//!
//! Seals land at every ragged width, so the round trip covers all four
//! remainders, and equality is checked on bytes because the record holds copies.
//! A resumed cache must append the same rows and select the same blocks as one
//! that never stopped. That fails if the open block is dropped.

use super::*;

// Pinned ranges are lists of ranges, and a list of one is the common case.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
pub(super) mod tests {
    use super::*;

    use crate::models::qwen4exp::key_q8::{blocks, BLOCK};

    use half::f16;

    use super::super::tests_support::tests::{cuda, Rig};

    /// **The cache survives the record byte-for-byte, at every ragged width.**
    ///
    /// `T mod ratio` is uniformly distributed over `0..ratio` because a turn
    /// ends where its text ends, so all four remainders are exercised. Byte
    /// equality, not a tolerance: these are copies.
    #[test]
    fn a_cache_round_trips_through_the_record_at_every_ragged_width() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device, 128)?;
        for extra in 0..rig.ratio {
            let tokens = 10 * rig.ratio + extra;
            let cache = rig.build(tokens)?;
            assert_eq!(
                cache.seal_shape(),
                (tokens / rig.ratio, extra),
                "the built cache is not at the width the test intends"
            );

            let back = rig.round_trip(&cache)?;
            assert_eq!(
                back.seal_shape(),
                cache.seal_shape(),
                "the restored cache stands at a different shape ({extra} carried)"
            );
            assert_eq!(
                back.len(rig.ratio),
                tokens,
                "the restored cache reports {} tokens against {tokens} — it would \
                 index every later token against the wrong block",
                back.len(rig.ratio)
            );
            assert_eq!(
                back.live_rows_host()?,
                cache.live_rows_host()?,
                "the completed rows changed"
            );
            assert_eq!(
                back.open_rows_host()?,
                cache.open_rows_host()?,
                "the open block changed"
            );
        }
        Ok(())
    }

    /// **A resumed cache keeps decoding as though it never stopped.**
    ///
    /// The property the round trip alone cannot establish. Two caches reach the
    /// same 43 tokens — one continuously, one by resuming from a record sealed
    /// at a ragged boundary — and are then appended the same further tokens.
    /// Their rows must be identical.
    ///
    /// This is what fails if the open block is dropped: the restored cache pools
    /// its next row over the wrong tokens, every row after it is shifted, and
    /// both caches remain internally consistent while disagreeing about the
    /// sequence. Nothing downstream reports it.
    #[test]
    fn a_resumed_cache_appends_the_same_rows_as_one_that_never_stopped() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device, 128)?;
        // 43 is not a multiple of 4, so the seal lands mid-block — the case a
        // boundary-aligned fixture would miss entirely.
        let sealed_at = 43usize;
        let mut live = rig.build(sealed_at)?;
        let mut resumed = rig.round_trip(&live)?;

        for (start, rows) in [
            (sealed_at, 9usize),
            (sealed_at + 9, 17),
            (sealed_at + 26, 4),
        ] {
            rig.append(&mut live, start, rows)?;
            rig.append(&mut resumed, start, rows)?;
            assert_eq!(
                resumed.seal_shape(),
                live.seal_shape(),
                "after appending {rows} at {start} the two caches are at different shapes"
            );
            assert_eq!(
                resumed.live_rows_host()?,
                live.live_rows_host()?,
                "the resumed cache's rows diverged after appending {rows} at {start}"
            );
        }
        Ok(())
    }

    /// **And it SELECTS the same rows.**
    ///
    /// The end of the chain: identical keys are worth nothing if the selection
    /// built from them differs, and the selection is what the attention kernel
    /// actually consumes. Exact equality per row, for the same reason
    /// `device_selection_matches_the_cpu_oracle` requires it.
    #[test]
    fn a_resumed_cache_selects_exactly_what_the_live_one_selects() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device, 128)?;
        let sealed_at = 43usize;
        let mut live = rig.build(sealed_at)?;
        let mut resumed = rig.round_trip(&live)?;
        rig.append(&mut live, sealed_at, 21)?;
        rig.append(&mut resumed, sealed_at, 21)?;

        // Query positions spanning the seal boundary, so rows that see only
        // pre-seal blocks, only post-seal ones, and both are all represented.
        let qpos: Vec<usize> = (40..64).collect();
        assert_eq!(
            rig.select(&mut resumed, &qpos)?,
            rig.select(&mut live, &qpos)?,
            "the resumed cache selects different blocks than the live one — the \
             conversation would attend to different history after a restart"
        );
        Ok(())
    }

    /// **A stored row is the pooled, normed, UN-rotated block key.** The
    /// append kernel pools `ratio` projected rows and applies the indexer's
    /// `k_norm`, and nothing more: the rotation happens in the scorer, at
    /// whatever position the row is read at. Checked against the host's own
    /// arithmetic, at a nonzero `tail_base` so a stored row that picked up a
    /// rotation would show it.
    #[test]
    fn stored_rows_are_the_pooled_normed_unrotated_key() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let rig = Rig::new(device.clone(), 64)?;
        let (ratio, d) = (rig.ratio, rig.cfg.head_dim);
        let mut cache = IndexCache::new(d, &device)?;
        cache.skip_to(1000)?;
        rig.append(&mut cache, 0, 7)?;
        rig.append(&mut cache, 7, 13)?;
        let stored = cache.live_rows_host()?;
        assert_eq!(stored.n_rows(), 20 / ratio);
        let values = stored.dequantize();
        let nb = blocks(d);

        let keys = rig.keys.to_vec2::<f32>()?;
        let k_norm = rig.w.k_norm.to_vec1::<f32>()?;
        for b in 0..stored.n_rows() {
            let mean: Vec<f64> = (0..d)
                .map(|c| {
                    (0..ratio)
                        .map(|r| keys[b * ratio + r][c] as f64)
                        .sum::<f64>()
                        / ratio as f64
                })
                .collect();
            let ms = mean.iter().map(|v| v * v).sum::<f64>() / d as f64;
            let scale = 1.0 / (ms + rig.eps).sqrt();
            let want: Vec<f32> = (0..d)
                .map(|c| (mean[c] * scale * k_norm[c] as f64) as f32)
                .collect();
            // The kernel's key, stored as Q8_0, against the host's key stored by
            // the host definition: the kernel runs under fast math and pools in
            // registers, so a code may land the other side of a tie and a scale
            // one step off — and no further.
            let host = Q8Keys::quantize(&want, d);
            let (codes, scales) = stored.rows(b, 1);
            for (q, (&got, &exp)) in scales.iter().zip(&host.scales).enumerate() {
                assert!(
                    got.abs_diff(exp) <= 1,
                    "block {b} scale {q}: stored {got:#06x} against the host's {exp:#06x}"
                );
            }
            for c in 0..d {
                let (got, exp) = (codes[c] as i8, host.codes[c] as i8);
                assert!(
                    (got as i32 - exp as i32).abs() <= 1,
                    "block {b} channel {c}: stored code {got} against the host's {exp}"
                );
                // And the value it holds is the key, to half a step.
                let step = f16::from_bits(scales[c / BLOCK]).to_f32() as f64;
                let held = values[b * d + c] as f64;
                assert!(
                    (held - want[c] as f64).abs()
                        <= 0.5 * step + 2e-3 * want[c].abs() as f64 + 1e-5,
                    "block {b} channel {c}: stored {held} against the pooled normed key {} at \
                     step {step}",
                    want[c]
                );
            }
            assert_eq!(scales.len(), nb);
        }
        Ok(())
    }
}
