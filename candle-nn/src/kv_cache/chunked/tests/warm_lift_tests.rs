//! The warm lift (`docs/warm_lift.md`): every layer and item lands byte for
//! byte through the pinned ring, however many slot batches it takes, and the
//! lift returns without waiting on the device's stream.
#![cfg(feature = "cuda")]

use std::sync::Arc;
use std::time::Instant;

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::quantized::pinned_staging::PinnedBuf;
use candle::{DType, Device, Tensor};

use super::stream_gate::StreamGate;
use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
use crate::kv_cache::chunked::{
    lift_warm_to_gpu, payload_bytes_for_tag, ChunkedKvBacking, SealedSequence, WarmLiftRing,
    GID_STRIDE,
};
use crate::kv_cache::ArenaLocation;

const N_KV_HEAD: usize = 4;
const HEAD_DIM: usize = 64;
/// One full chunk and one partial.
const N_TOKENS: usize = 40;

fn stream_of(device: &Device) -> Arc<CudaStream> {
    let Device::Cuda(cuda) = device else {
        unreachable!("a CUDA test");
    };
    (*cuda.cuda_stream()).clone()
}

/// A hot sequence of `N_TOKENS` holding a pattern particular to `seed`.
fn seeded(backing: &ChunkedKvBacking, device: &Device, seed: usize) -> SealedSequence {
    let slot = backing.alloc_sequence().unwrap();
    backing.ensure_for_offset(slot, 0, N_TOKENS).unwrap();
    let n = N_KV_HEAD * N_TOKENS * HEAD_DIM;
    let data: Vec<f32> = (0..n)
        .map(|i| ((seed * 7919 + i) % 4093) as f32 * 0.25)
        .collect();
    let k = Tensor::from_vec(data, (1, N_KV_HEAD, N_TOKENS, HEAD_DIM), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let v = (&k + 1.0).unwrap();
    backing.write_contiguous(slot, 0, &k, &v).unwrap();
    backing.set_len(slot, N_TOKENS);
    backing.record_turn(slot).unwrap()
}

/// A warm sequence's stored payloads, band by band in chunk order — each
/// band's own length, read from its warm arena slot.
fn warm_bytes(backing: &ChunkedKvBacking, seq: &SealedSequence) -> Vec<Vec<u8>> {
    let elems = backing.inner.elems_per_chunk();
    backing
        .inner
        .storage
        .read(|s| {
            seq.chunks
                .iter()
                .flat_map(|chunk| chunk.bands())
                .map(|(gid, tag)| {
                    let raw = gid.raw() as usize;
                    let len = payload_bytes_for_tag(tag, elems).unwrap();
                    let mut buf = vec![0u8; len];
                    s.arenas()
                        .get(&(raw / GID_STRIDE))
                        .unwrap()
                        .copy_slot_bytes(raw % GID_STRIDE, &mut buf)
                        .unwrap();
                    buf
                })
                .collect()
        })
        .unwrap()
}

/// A lifted sequence's bytes, read in the warm tier: moved back down and
/// read there, which reads any format the same way.
fn lifted_bytes(
    backing: &ChunkedKvBacking,
    device: &Device,
    stream: &Arc<CudaStream>,
    lifted: &SealedSequence,
) -> Vec<Vec<u8>> {
    let mut pinned: Option<PinnedBuf> = None;
    let down = backing
        .migrate_sealed_to_cpu_batch_async(device, stream, &mut pinned, &[lifted])
        .unwrap();
    warm_bytes(backing, &down[0])
}

/// **Every layer and item lands byte for byte through a ring of 4 KiB slots**
/// — some hundreds of batches, each slot refilled many times over, so a slot
/// written before its last scatter ran, or a band cut wrong at a slot edge,
/// shows up as a byte that differs.
#[test]
fn every_layer_and_item_lands_through_a_small_ring() {
    let _gpu = gpu_serial();
    let device = Device::new_cuda(0).unwrap();
    let stream = stream_of(&device);
    let backings: Vec<ChunkedKvBacking> = (0..3)
        .map(|_| ChunkedKvBacking::new(4, N_KV_HEAD, HEAD_DIM, DType::BF16, &device, 256).unwrap())
        .collect();
    let mut pinned: Option<PinnedBuf> = None;
    // layer → item: the warm copy of a seeded hot sequence.
    let mut warm: Vec<Vec<SealedSequence>> = Vec::new();
    for (l, backing) in backings.iter().enumerate() {
        let hot: Vec<SealedSequence> = (0..2)
            .map(|i| seeded(backing, &device, l * 10 + i))
            .collect();
        let refs: Vec<&SealedSequence> = hot.iter().collect();
        let down = backing
            .migrate_sealed_to_cpu_batch_async(&device, &stream, &mut pinned, &refs)
            .unwrap();
        assert!(down.iter().all(|s| s.location == ArenaLocation::Cpu));
        warm.push(down);
    }
    let layers: Vec<Vec<&SealedSequence>> = warm.iter().map(|w| w.iter().collect()).collect();
    let mut ring = WarmLiftRing::with_slot_bytes(4096).unwrap();
    let lift = lift_warm_to_gpu(&backings, &mut ring, &device, &layers).unwrap();
    // 3 layers x 2 items x 64 bands of 1 KiB, four to a 4 KiB slot.
    assert_eq!(lift.split.batches, 96, "{:?}", lift.split);
    let lifted = lift.layers;

    assert_eq!(lifted.len(), 3);
    for (l, backing) in backings.iter().enumerate() {
        assert_eq!(lifted[l].len(), 2);
        for (i, seq) in lifted[l].iter().enumerate() {
            assert_eq!(seq.location, ArenaLocation::Gpu);
            assert_eq!(seq.token_count, N_TOKENS);
            let source = warm_bytes(backing, &warm[l][i]);
            assert!(source.iter().any(|band| band.iter().any(|&b| b != 0)));
            assert_eq!(
                lifted_bytes(backing, &device, &stream, seq),
                source,
                "layer {l} item {i}"
            );
        }
    }
}

/// **A lift that fits the ring returns without waiting on the stream**: with
/// the device's stream gated shut, the call comes back long before the gate
/// would open on its own, and once the gate opens the bytes are the source's.
#[test]
fn a_lift_returns_while_the_stream_is_held() {
    let _gpu = gpu_serial();
    let device = Device::new_cuda(0).unwrap();
    let stream = stream_of(&device);
    let backing = ChunkedKvBacking::new(4, N_KV_HEAD, HEAD_DIM, DType::BF16, &device, 256).unwrap();
    let hot = seeded(&backing, &device, 3);
    let mut pinned: Option<PinnedBuf> = None;
    let warm = backing
        .migrate_sealed_to_cpu_batch_async(&device, &stream, &mut pinned, &[&hot])
        .unwrap();
    let mut ring = WarmLiftRing::new().unwrap();
    device.synchronize().unwrap();

    let gate = StreamGate::close_on(&stream);
    let t = Instant::now();
    let lifted = lift_warm_to_gpu(
        std::slice::from_ref(&backing),
        &mut ring,
        &device,
        &[vec![&warm[0]]],
    )
    .unwrap()
    .layers;
    let took = t.elapsed();
    gate.open();
    assert!(
        took < StreamGate::TIMEOUT / 4,
        "the lift waited {took:?} on a stream held shut — it must only enqueue"
    );
    device.synchronize().unwrap();
    assert_eq!(
        lifted_bytes(&backing, &device, &stream, &lifted[0][0]),
        warm_bytes(&backing, &warm[0])
    );
}

/// Nothing to lift is nothing done.
#[test]
fn an_empty_lift_returns_empty_layers() {
    let _gpu = gpu_serial();
    let device = Device::new_cuda(0).unwrap();
    let backing = ChunkedKvBacking::new(4, N_KV_HEAD, HEAD_DIM, DType::BF16, &device, 256).unwrap();
    let mut ring = WarmLiftRing::new().unwrap();
    let lifted = lift_warm_to_gpu(
        std::slice::from_ref(&backing),
        &mut ring,
        &device,
        &[Vec::new()],
    )
    .unwrap()
    .layers;
    assert_eq!(lifted.len(), 1);
    assert!(lifted[0].is_empty());
}
