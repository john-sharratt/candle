//! A sealed residence's native chunks quantized by a gap
//! (`docs/kv_quantize_in_the_gap.md` §5): the copy handed back holds exactly
//! the chunks today's seal makes of the same K/V — every band's format, outer
//! scale, palette map and payload, a record each — with its windows intact,
//! and the hot copy it was made from is left as it was.

use super::super::compression_policy::CompressionPolicy;
use super::super::types::SealedSequence;
use super::cycle::GapCycle;
use super::launch_tests::{
    backing, cuda, sealed_bands, setup_tokens, todays_seal, write_tokens, ARENA_CAPACITY, LAYERS,
};
use crate::kv_cache::QuantFormat;

/// Two full blocks and a partial one: a turn stored with its writer chunk.
const TOKENS: usize = 80;

#[test]
fn a_residences_native_chunks_come_back_as_todays_seal() {
    for policy in [
        CompressionPolicy::new(5),
        CompressionPolicy::new(5).with_override_k_quant(Some(QuantFormat::Q8_0)),
    ] {
        let Some((device, _cuda, _gpu)) = cuda() else {
            return;
        };
        let (reference, srcs) = setup_tokens(&device, &policy, TOKENS);
        let want = todays_seal(&device, &reference, &srcs, &policy);

        let mut layers = vec![backing(&device, &policy)];
        for l in 1..LAYERS {
            let next = layers[0].new_layer(l, 4, ARENA_CAPACITY);
            layers.push(next);
        }
        let hot: Vec<SealedSequence> = layers
            .iter()
            .enumerate()
            .map(|(l, b)| {
                let slot = write_tokens(b, &device, (l as u32 + 1) * 1000, TOKENS);
                let sealed = b.record_turn(slot).unwrap();
                b.free_sequence(slot).unwrap();
                sealed
            })
            .collect();
        assert!(hot.iter().all(|s| s.chunks.iter().all(|c| c.is_native())));

        let mut cycle = GapCycle::new(&layers, 1024).unwrap();
        let done = cycle
            .quantize_sealed(&layers, &[policy; LAYERS], &hot)
            .unwrap()
            .expect("native chunks to quantize");
        assert_eq!((done.installed, done.deferred), (3 * LAYERS, 0), "{done:?}");
        assert!(!cycle.is_outstanding());
        for (layer, (got, was)) in done.seqs.iter().zip(&hot).enumerate() {
            assert_eq!(got.token_count, was.token_count, "layer {layer}");
            for (blk, (c, w)) in got.chunks.iter().zip(&was.chunks).enumerate() {
                assert!(!c.is_native(), "layer {layer} block {blk}");
                assert!(c.meta.is_some(), "layer {layer} block {blk}: a record");
                assert_eq!((c.offset, c.token_count), (w.offset, w.token_count));
                assert!(w.is_native(), "the hot copy read is left as it was");
            }
        }
        let refs: Vec<&SealedSequence> = done.seqs.iter().collect();
        let got = sealed_bands(&layers, &refs);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "band {i}");
        }

        // Quantized, it has nothing left for a gap.
        assert!(cycle
            .quantize_sealed(&layers, &[policy; LAYERS], &done.seqs)
            .unwrap()
            .is_none());
    }
}
