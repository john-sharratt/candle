//! Tests for sealing: signature runs with a reasoning span removed, seal ranges,
//! and the GPU signature assembly matching its CPU fold bit for bit.
//!
//! A signature run is checked by identity rather than by length, so a filter that
//! drops the wrong tokens cannot pass. A seal range clamps to the shorter of the
//! slot and the snapshot, and it is `None` when empty rather than a zero-width
//! range. An ephemeral seal skips the state export by design, so only a requested
//! snapshot can count as missing. The assembled GPU signatures are compared with
//! the CPU fold under the model's derived fold.

use super::*;

// •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
// Scheduler unit tests
// •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
//
// The `DummyModel` is a minimal `ManagedBatchedModel` that returns dummy CPU
// tensors so no GPU is needed. `BatchedInferenceSession` is constructed on
// `Device::Cpu` with tiny arena dimensions so these tests run without CUDA.

#[cfg(test)]
pub(super) mod tests {
    // Expected token totals are written as the sum of the per-item terms they
    // came from, including the zero ones, so the arithmetic under test is
    // visible in the assertion.
    #![allow(clippy::identity_op)]

    use super::*;

    /// A signature run with the reasoning span cut out of it.
    ///
    /// The run is 1:1 with the turn's grid, so the assertion is identity-based:
    /// each sig is tagged with its own grid index, and the survivors must be
    /// exactly the non-reasoning indices, in order. A length check alone would
    /// pass on a filter that removed the wrong tokens.
    #[test]
    fn sigs_without_span_drops_exactly_the_reasoning_tokens() {
        let run = |n: usize| -> Vec<WideQSig> {
            (0..n)
                .map(|i| WideQSig {
                    n_heads: 1,
                    words: vec![i as u64],
                })
                .collect()
        };
        let ids = |v: &[WideQSig]| -> Vec<u64> { v.iter().map(|s| s.words[0]).collect() };

        // Reasoning at grid [3, 6) of a 10-token turn.
        assert_eq!(
            ids(&sigs_without_span(run(10), 3..6)),
            vec![0, 1, 2, 6, 7, 8, 9]
        );
        // Abutting the start, and running to the end.
        assert_eq!(ids(&sigs_without_span(run(5), 0..2)), vec![2, 3, 4]);
        assert_eq!(ids(&sigs_without_span(run(5), 2..5)), vec![0, 1]);
        // An empty span is a no-op, not an off-by-one.
        assert_eq!(ids(&sigs_without_span(run(4), 2..2)), vec![0, 1, 2, 3]);
    }

    /// A span past the end of the run leaves it untouched.
    ///
    /// Not hypothetical: the last generated token is never forwarded, so a turn
    /// whose final token is `</think>` has a layout span one wider than the
    /// signature run the seal gathered. Clamping rather than truncating is what
    /// keeps that from silently eating the answer.
    #[test]
    fn sigs_without_span_clamps_a_span_past_the_run() {
        let run: Vec<WideQSig> = (0..3)
            .map(|i| WideQSig {
                n_heads: 1,
                words: vec![i as u64],
            })
            .collect();
        assert_eq!(sigs_without_span(run.clone(), 5..9).len(), 3);
        // Overlapping the end: only the in-range part goes.
        assert_eq!(sigs_without_span(run, 2..9).len(), 2);
    }

    /// A seal is the tail of its slot: it runs from its start to the slot's
    /// last block.
    #[test]
    fn a_seal_takes_everything_from_its_start() {
        assert_eq!(resolve_seal_range(4, 10, 10), Some((4, 10)));
        assert_eq!(resolve_seal_range(0, 10, 10), Some((0, 10)));
    }

    /// The snapshot is taken a moment before the range resolves, so it can be
    /// the shorter of the two. The end clamps to it — indexing
    /// `snapshot.chunks[from..to]` past its end would panic the seal.
    #[test]
    fn the_range_clamps_to_the_shorter_of_slot_and_snapshot() {
        // Snapshot shorter than the slot.
        assert_eq!(resolve_seal_range(0, 10, 6), Some((0, 6)));
        // Slot shorter than the snapshot.
        assert_eq!(resolve_seal_range(0, 6, 10), Some((0, 6)));
    }

    /// An empty range is `None`, never `Some((n, n))` — the caller warns and
    /// skips on it, and a zero-width range that reached the gather would seal a
    /// turn with no K/V and an empty gallery window.
    #[test]
    fn an_empty_range_is_none_not_a_zero_width_range() {
        // Start at or past the end.
        assert_eq!(resolve_seal_range(10, 10, 10), None);
        assert_eq!(resolve_seal_range(11, 10, 10), None);
        // An empty slot has nothing to seal.
        assert_eq!(resolve_seal_range(0, 0, 0), None);
    }

    #[test]
    fn carve_ms_redistributes_and_bounds() {
        // Carves in priority order, draining each bucket before the next.
        let (mut a, mut b) = (100u64, 50u64);
        let taken = carve_ms(120, &mut [&mut a, &mut b]);
        assert_eq!((taken, a, b), (120, 0, 30)); // 100 from a, 20 from b

        // Bounded by what the buckets hold — never carves more than available, so a
        // segment can't push the stack past the window.
        let (mut a, mut b) = (10u64, 5u64);
        let taken = carve_ms(1000, &mut [&mut a, &mut b]);
        assert_eq!((taken, a, b), (15, 0, 0));

        // Sum preservation: carved segment + remaining buckets == original total,
        // for any request. This is the invariant the phase decomposition relies on.
        for req in [0u64, 7, 33, 80, 200] {
            let orig = 80u64;
            let mut only = orig;
            let seg = carve_ms(req, &mut [&mut only]);
            assert_eq!(seg + only, orig, "req={req}");
        }
    }

    /// **An ephemeral seal skips the export, and that is not a missing hook.**
    /// Judged as one, every calibration seal of a recurrent model reported a
    /// wiring gap — 93 tool sections' worth at each fresh boot of Qwen3.6.
    #[test]
    fn only_a_requested_snapshot_can_be_missing() {
        assert!(missing_recurrent_hook(false, true), "asked for, and absent");
        assert!(!missing_recurrent_hook(true, true), "skipped by design");
        assert!(
            !missing_recurrent_hook(false, false),
            "a model with no such state"
        );
        assert!(!missing_recurrent_hook(true, false));
    }

    /// The GPU provenance path's on-CPU tail (`assemble_folded_prov_sigs`) must be
    /// bit-identical to the CPU path (`fold_provenance(WideQSig::from_band(..))`).
    /// This drives both from the same synthetic signs — the kernel just produces
    /// the packed `u32`s this test hand-packs — so it validates the whole host
    /// mapping (palette→word, warp indexing, layout) without a GPU.
    ///
    /// The two sides are folded differently on purpose: the assembled side takes
    /// the **derived** fold the scheduler now carries, the reference side the
    /// **locked** constants. At Qwen3-30B's geometry the derivation is an
    /// identity, so equality here is simultaneously the assembly-mapping check
    /// and the tier-1 half of the no-regression gate on the outgoing model.
    #[test]
    fn gpu_assembled_prov_sigs_match_cpu_fold() {
        use crate::provenance::{fold_provenance, WideQSig};
        const N_LAYERS: usize = 48;
        const N_KV_HEAD: usize = 4; // == PROV_HEADS_PER_LAYER
        const N_PALETTE: usize = 4;
        const SUB: usize = 32; // sub_head_dim
        const HEAD_DIM: usize = N_PALETTE * SUB; // 128
        let chunk = candle_nn::CHUNK_SIZE;
        let n_real = 5usize; // real tokens in the single block

        // Deterministic sign for (layer, head, token, global-dim): true = bit set.
        let sgn = |l: usize, h: usize, t: usize, d: usize| -> bool {
            let x = ((((l * 131 + h) * 131 + t) * 131 + d) as u64).wrapping_mul(2654435761);
            (x >> 13) & 1 == 0
        };

        // CPU reference: build the f32 band, from_band, fold — per token.
        let mut cpu: Vec<WideQSig> = Vec::new();
        for t in 0..n_real {
            let mut band = Vec::with_capacity(N_LAYERS * N_KV_HEAD * HEAD_DIM);
            for l in 0..N_LAYERS {
                for h in 0..N_KV_HEAD {
                    for d in 0..HEAD_DIM {
                        band.push(if sgn(l, h, t, d) { 1.0f32 } else { -1.0f32 });
                    }
                }
            }
            cpu.push(fold_provenance(&WideQSig::from_band(&band, HEAD_DIM)));
        }

        // Hand-pack the kernel's warp-major u64 output for the same signs.
        let n_blocks = 1usize;
        let n_warps = N_LAYERS * n_blocks * N_KV_HEAD * N_PALETTE;
        let mut packed = vec![0u64; n_warps * chunk];
        for l in 0..N_LAYERS {
            for h in 0..N_KV_HEAD {
                for p in 0..N_PALETTE {
                    let warp = ((l * n_blocks) * N_KV_HEAD + h) * N_PALETTE + p;
                    for t in 0..n_real {
                        let mut bits = 0u64;
                        for d in 0..SUB {
                            if sgn(l, h, t, p * SUB + d) {
                                bits |= 1u64 << d;
                            }
                        }
                        packed[warp * chunk + t] = bits;
                    }
                }
            }
        }
        let ps = ProvSignPacked {
            packed,
            block_indices: vec![0],
            n_layers: N_LAYERS,
            n_kv_head: N_KV_HEAD,
            n_palette: N_PALETTE,
            sub_head_dim: SUB,
        };
        // Block 0: real tokens at physical slots 0..n_real.
        let layout = vec![(0u16, n_real as u16, 0usize)];
        let gpu = assemble_folded_prov_sigs(
            &ps,
            &layout,
            HEAD_DIM,
            FoldParams::derive(N_KV_HEAD, N_LAYERS, HEAD_DIM),
        );

        assert_eq!(gpu.len(), cpu.len(), "token count");
        for (t, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(g.n_heads, c.n_heads, "token {t}: n_heads");
            assert_eq!(
                g.words, c.words,
                "token {t}: assembled words != from_band+fold"
            );
        }
    }

    #[test]
    fn pending_prefill_backlog_sums_remaining_tokens() {
        // queued (offset 0), partially-advanced, and fully-consumed units.
        let items = [
            (512, 0),   // queued: all 512 pending
            (300, 120), // in-flight: 180 remaining
            (64, 64),   // finished but not yet drained: 0 remaining
            (0, 0),     // empty unit
        ];
        assert_eq!(sum_pending_prefill_tokens(items), 512 + 180 + 0 + 0);
        // Empty backlog.
        assert_eq!(sum_pending_prefill_tokens(std::iter::empty()), 0);
        // Defensive: consumed > total (should never happen) saturates to 0, not
        // underflow.
        assert_eq!(sum_pending_prefill_tokens([(10, 25)]), 0);
    }
}
