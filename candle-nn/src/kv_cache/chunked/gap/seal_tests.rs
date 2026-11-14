//! The seal gap on the device (`docs/kv_quantize_in_the_gap.md` §5): a turn
//! sealed by [`GapCycle::seal`] — its writer chunk and its partial chunk with
//! the rest — comes out exactly as today's seal makes the same K/V, every block
//! in place and none left native; and a seal leaves the deferred blocks of
//! other sequences standing for the gaps after it.

use candle::Device;

use super::super::compression_policy::CompressionPolicy;
use super::super::ChunkedKvBacking;
use super::cycle::{GapCycle, GapSeq};
use super::demand::GapKind;
use super::install_tests::bands_of;
use super::launch_tests::{
    backing, cuda, setup_tokens, todays_seal, write_tokens, ARENA_CAPACITY, LAYERS,
};
use crate::kv_cache::QuantFormat;

/// Two full blocks and a partial writer chunk of 16 tokens.
const TOKENS: usize = 80;

/// `LAYERS` live layers over one set of pools, each holding one sequence of
/// `tokens`: its batch index.
fn live_layers(
    device: &Device,
    policy: &CompressionPolicy,
    tokens: usize,
    base: u32,
    layers: &mut Vec<ChunkedKvBacking>,
) -> usize {
    if layers.is_empty() {
        layers.push(backing(device, policy));
        for l in 1..LAYERS {
            let next = layers[0].new_layer(l, 4, ARENA_CAPACITY);
            layers.push(next);
        }
    }
    let slots: Vec<usize> = layers
        .iter()
        .enumerate()
        .map(|(l, b)| write_tokens(b, device, (l as u32 + 1) * base, tokens))
        .collect();
    assert!(
        slots.iter().all(|&s| s == slots[0]),
        "one batch index on every layer"
    );
    slots[0]
}

fn every_layer(slot: usize, policy: &CompressionPolicy) -> GapSeq {
    GapSeq {
        seq: slot,
        layer_policy: vec![*policy; LAYERS],
    }
}

#[test]
fn a_sealed_turn_is_todays_seal_writer_and_partial_chunks_included() {
    for policy in [
        CompressionPolicy::new(5),
        CompressionPolicy::new(5).with_override_k_quant(Some(QuantFormat::Q8_0)),
    ] {
        let Some((device, _cuda, _gpu)) = cuda() else {
            return;
        };
        let (reference, srcs) = setup_tokens(&device, &policy, TOKENS);
        let want = todays_seal(&device, &reference, &srcs, &policy);
        let mut layers = Vec::new();
        let slot = live_layers(&device, &policy, TOKENS, 1000, &mut layers);
        let mut cycle = GapCycle::new(&layers, 1024).unwrap();
        let sealed = cycle
            .seal(&layers, &every_layer(slot, &policy), 0..3)
            .unwrap();
        assert_eq!(sealed.installed, 3 * LAYERS, "{sealed:?}");
        assert_eq!((sealed.stale, sealed.deferred), (0, 0));
        assert!(!cycle.is_outstanding(), "a seal leaves no gap outstanding");
        for b in &layers {
            let state = b.state.read().unwrap();
            let seq = state.sequences[slot].as_ref().unwrap();
            for blk in 0..3 {
                let c = &seq.chunks_slice()[blk];
                assert!(!c.takes_writes(), "layer {} block {blk}", b.layer_idx);
                assert!(c.meta.is_some(), "layer {} block {blk}", b.layer_idx);
            }
            assert_eq!(
                seq.chunks_slice()[2].usage,
                16,
                "the partial keeps its window"
            );
        }
        let got = bands_of(&layers, slot, 0..3);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "band {i}");
        }
    }
}

#[test]
fn a_seal_catches_up_the_outstanding_gap_and_leaves_no_block_native() {
    let policy = CompressionPolicy::new(5);
    let Some((device, _cuda, _gpu)) = cuda() else {
        return;
    };
    let mut layers = Vec::new();
    let slot = live_layers(&device, &policy, TOKENS, 1000, &mut layers);
    let other = live_layers(&device, &policy, 96, 3000, &mut layers);
    let mut cycle = GapCycle::new(&layers, 1024).unwrap();
    // A gap over both sequences' complete blocks, left outstanding.
    let both = [every_layer(slot, &policy), every_layer(other, &policy)];
    let q = cycle.quantize(&layers, &both, GapKind::Decode).unwrap();
    assert_eq!(q.jobs, (2 + 3) * LAYERS);
    assert!(cycle.is_outstanding());
    // The seal waits it out and installs it, then quantizes the partial.
    let sealed = cycle
        .seal(&layers, &every_layer(slot, &policy), 0..3)
        .unwrap();
    assert_eq!(sealed.installed, (2 + 3 + 1) * LAYERS, "{sealed:?}");
    assert_eq!(sealed.deferred, 0);
    // Only the partial was left to the seal. Its first lease is sized by the
    // shares of the blocks before it, so a gap may defer it once and install
    // it at its exact demand the next.
    assert_eq!(sealed.jobs, LAYERS * sealed.gaps, "{sealed:?}");
    assert!(sealed.gaps <= 2, "{sealed:?}");
    for b in &layers {
        let state = b.state.read().unwrap();
        for (s, n) in [(slot, 3), (other, 3)] {
            let seq = state.sequences[s].as_ref().unwrap();
            for blk in 0..n {
                assert!(
                    !seq.chunks_slice()[blk].takes_writes(),
                    "layer {} sequence {s} block {blk}",
                    b.layer_idx
                );
            }
        }
    }
}
