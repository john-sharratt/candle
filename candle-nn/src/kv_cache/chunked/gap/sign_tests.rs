//! Provenance signatures through the gap (`docs/kv_quantize_in_the_gap.md`
//! §3.3): Q lives inside the R16 K band, so the gap sign-packs every job's Q
//! before its convert replaces the band, and the catch-up hands each chunk its
//! bits. Read after the gap, a converted block's signs must be exactly the bits
//! the sign-pack read from its Q before it.

use candle::{DType, Device};

use super::super::compression_policy::CompressionPolicy;
use super::super::types::ProvSource;
use super::super::ChunkedKvBacking;
use super::codes::CodeTables;
use super::install::catch_up;
use super::install_tests::take_jobs;
use super::launch::{GapDevice, GapGroup};
use super::launch_tests::{
    backing, cuda, lease_for, stack, write_turn, ARENA_CAPACITY, HEAD_DIM, LAYERS, N_KV_HEAD,
};
use crate::kv_cache::arena_table::N_PALETTE;
use crate::kv_cache::{KvFormat, QuantFormat, CHUNK_SIZE};

/// A layer whose native K is R16 — F16 with the Q capture beside it — as a
/// quantizing model's is.
fn r16_backing(device: &Device, policy: &CompressionPolicy) -> ChunkedKvBacking {
    let b = ChunkedKvBacking::new_with_format(
        4,
        N_KV_HEAD,
        HEAD_DIM,
        KvFormat::Quantized(QuantFormat::R16),
        KvFormat::Float(DType::F16),
        device,
        ARENA_CAPACITY,
    )
    .expect("backing");
    b.keep_reserves(Some(policy)).expect("reserves");
    b
}

/// Each block's signs on `layer`, as the provenance gather reads them now:
/// the sign-pack over a block's Q, or the bits it carries.
fn signs_now(layer: &ChunkedKvBacking, slot: usize) -> Vec<(usize, bool, Vec<u64>)> {
    let block_words = N_KV_HEAD * N_PALETTE * CHUNK_SIZE;
    layer
        .provenance_sources(slot, None)
        .unwrap()
        .into_iter()
        .map(|(block, source)| match source {
            ProvSource::Q(ptrs) => {
                let bits = layer
                    .run_prov_sign_pack(&ptrs, HEAD_DIM / N_PALETTE)
                    .unwrap();
                assert_eq!(bits.len(), block_words);
                (block, false, bits)
            }
            ProvSource::Signed(bits) => (block, true, bits.to_vec()),
        })
        .collect()
}

#[test]
fn a_converted_block_carries_the_signs_its_q_had() {
    let policy = CompressionPolicy::new(5);
    let Some((device, cuda, _gpu)) = cuda() else {
        return;
    };
    let mut layers = vec![r16_backing(&device, &policy)];
    for l in 1..LAYERS {
        let next = layers[0].new_layer(l, 4, ARENA_CAPACITY);
        layers.push(next);
    }
    let slots: Vec<usize> = layers
        .iter()
        .enumerate()
        .map(|(l, b)| write_turn(b, &device, (l as u32 + 1) * 1000))
        .collect();
    let slot = slots[0];
    assert!(slots.iter().all(|&s| s == slot));

    let before: Vec<_> = layers.iter().map(|b| signs_now(b, slot)).collect();
    for layer in &before {
        assert_eq!(layer.len(), 3, "every block reads its Q");
        assert!(layer.iter().all(|(_, signed, _)| !signed));
    }

    let jobs = take_jobs(&layers, slot);
    let n = jobs.len();
    let refs: Vec<&ChunkedKvBacking> = layers.iter().collect();
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..n,
    }];
    let mut lease = lease_for(&layers[0], &policy, |_| n * N_KV_HEAD * 2 * N_PALETTE);
    let codes = CodeTables::new(layers[0].elems_per_chunk()).unwrap();
    let mut gap = GapDevice::new(&cuda, N_KV_HEAD, HEAD_DIM, n).unwrap();
    gap.launch(&refs, &jobs, &groups, &stack(&lease), 1)
        .unwrap();
    gap.wait().unwrap();
    gap.order_compute_after().unwrap();
    let finished = gap.finished().expect("complete");
    let done = catch_up(&finished, jobs, &groups, &refs, &mut lease, &codes).unwrap();
    assert_eq!((done.installed, done.stale), (n, 0));

    for (l, layer) in layers.iter().enumerate() {
        let after = signs_now(layer, slot);
        assert_eq!(after.len(), 3, "layer {l}: every block still has its signs");
        for ((block, signed, bits), (want_block, _, want)) in after.iter().zip(&before[l]) {
            assert_eq!(block, want_block);
            assert!(
                signed,
                "layer {l} block {block}: converted, so from its bits"
            );
            assert_eq!(bits, want, "layer {l} block {block}: the bits its Q had");
        }
    }
}

#[test]
fn a_block_with_no_q_carries_no_signs() {
    // F16 natives have no Q band: the gap captures nothing, and the gather
    // finds nothing for the converted blocks.
    let policy = CompressionPolicy::new(5);
    let Some((device, cuda, _gpu)) = cuda() else {
        return;
    };
    let layers = vec![backing(&device, &policy)];
    let slot = write_turn(&layers[0], &device, 1000);
    assert!(signs_now(&layers[0], slot).is_empty());
    let jobs = take_jobs(&layers, slot);
    let n = jobs.len();
    let refs: Vec<&ChunkedKvBacking> = layers.iter().collect();
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..n,
    }];
    let mut lease = lease_for(&layers[0], &policy, |_| n * N_KV_HEAD * 2 * N_PALETTE);
    let codes = CodeTables::new(layers[0].elems_per_chunk()).unwrap();
    let mut gap = GapDevice::new(&cuda, N_KV_HEAD, HEAD_DIM, n).unwrap();
    gap.launch(&refs, &jobs, &groups, &stack(&lease), 1)
        .unwrap();
    gap.wait().unwrap();
    gap.order_compute_after().unwrap();
    let finished = gap.finished().expect("complete");
    let done = catch_up(&finished, jobs, &groups, &refs, &mut lease, &codes).unwrap();
    assert_eq!(done.installed, n);
    assert!(signs_now(&layers[0], slot).is_empty());
}
