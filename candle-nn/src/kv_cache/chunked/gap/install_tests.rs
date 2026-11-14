//! The catch-up on the device (`docs/kv_quantize_in_the_gap.md` §3.5, §7):
//! live sequences quantized by a gap and put in place by its catch-up must hold
//! exactly the chunks today's seal builds from the same K/V — every band's
//! format, outer scale, palette map and payload — in place at their indices,
//! with their records, past the writer boundary; the native bands back in their
//! pool; the lease empty; a block that changed under its gap left alone; and a
//! deferred block back for the next gap, which then installs it.

use std::ops::Range;

use candle::quantized::cuda::ggml_dtype_to_arena_fmt_code;
use candle::{CudaDevice, DType, Device};

use super::super::arena::ArenaKey;
use super::super::compression_policy::CompressionPolicy;
use super::super::gid_pool::ChunkGid;
use super::super::gpu_test_lock::GpuGuard;
use super::super::head_gids::ChunkBands;
use super::super::size_class::{payload_bytes, SizeClass};
use super::super::ChunkedKvBacking;
use super::codes::CodeTables;
use super::install::{catch_up, CatchUp};
use super::launch::{GapDevice, GapGroup, GapJob};
use super::launch_tests::{
    backing, cuda, lease_for, setup, stack, todays_seal, write_turn, Band, ARENA_CAPACITY,
    HEAD_DIM, LAYERS, N_KV_HEAD,
};
use super::lease::Lease;
use super::lease_alloc_tests::leased;
use super::table::gap_take;
use crate::kv_cache::arena_table::{ArenaLocation, N_PALETTE};
use crate::kv_cache::{KvFormat, QuantFormat};

/// Live sequences ready for a gap, and what today's seal makes of the same K/V.
pub(super) struct Live {
    /// Layers over one set of pools, as a model's are; one live sequence each.
    pub layers: Vec<ChunkedKvBacking>,
    pub slot: usize,
    /// Today's seal of identical K/V on separate backings, band by band in
    /// table order.
    pub want: Vec<Band>,
    _reference: Vec<ChunkedKvBacking>,
    pub cuda: CudaDevice,
    pub device: Device,
    _gpu: GpuGuard,
}

pub(super) fn live(policy: &CompressionPolicy) -> Option<Live> {
    let (device, cuda, gpu) = cuda()?;
    // The reference: the same pattern sealed and quantized by today's path.
    let (reference, srcs) = setup(&device, policy);
    let want = todays_seal(&device, &reference, &srcs, policy);
    // The live side: one pool set, one block table per layer.
    let mut layers = vec![backing(&device, policy)];
    for l in 1..LAYERS {
        let next = layers[0].new_layer(l, 4, ARENA_CAPACITY);
        layers.push(next);
    }
    let slots: Vec<usize> = layers
        .iter()
        .enumerate()
        .map(|(l, b)| write_turn(b, &device, (l as u32 + 1) * 1000))
        .collect();
    assert!(
        slots.iter().all(|&s| s == slots[0]),
        "one batch index on every layer"
    );
    Some(Live {
        layers,
        slot: slots[0],
        want,
        _reference: reference,
        cuda,
        device,
        _gpu: gpu,
    })
}

/// The gap's jobs for `slot`: what `gap_take` hands over, layer-major.
pub(super) fn take_jobs(layers: &[ChunkedKvBacking], slot: usize) -> Vec<GapJob> {
    let range = gap_take(layers, slot, usize::MAX);
    assert!(!range.is_empty(), "complete blocks to take");
    let mut jobs = Vec::new();
    for (layer, b) in layers.iter().enumerate() {
        let state = b.state.read().unwrap();
        let seq = state.sequences[slot].as_ref().unwrap();
        for block in range.clone() {
            let w = &seq.chunks_slice()[block];
            jobs.push(GapJob {
                layer,
                seq: slot,
                block,
                bands: ChunkBands::from_window(w),
                valid_range: (i32::from(w.offset) << 8) | w.usage as i32,
            });
        }
    }
    jobs
}

/// One gap over `jobs` on `lease`, waited for, and its catch-up.
fn gap_and_catch_up(
    l: &Live,
    policy: &CompressionPolicy,
    jobs: Vec<GapJob>,
    lease: &mut Lease<ChunkGid>,
    between: impl FnOnce(),
) -> CatchUp {
    let refs: Vec<&ChunkedKvBacking> = l.layers.iter().collect();
    let groups = [GapGroup {
        policy,
        jobs: 0..jobs.len(),
    }];
    let codes = CodeTables::new(l.layers[0].elems_per_chunk()).unwrap();
    let mut gap = GapDevice::new(&l.cuda, N_KV_HEAD, HEAD_DIM, jobs.len()).unwrap();
    gap.launch(&refs, &jobs, &groups, &stack(lease), 1).unwrap();
    between();
    gap.wait().unwrap();
    gap.order_compute_after().unwrap();
    let finished = gap.finished().expect("complete");
    catch_up(&finished, jobs, &groups, &refs, lease, &codes).unwrap()
}

/// The bands of `slot`'s blocks `blocks` on every layer, in table order, as
/// they now stand in the live sequences.
pub(super) fn live_bands(l: &Live, blocks: Range<usize>) -> Vec<Band> {
    let elems = l.layers[0].elems_per_chunk();
    let mut out = Vec::new();
    for b in &l.layers {
        let state = b.state.read().unwrap();
        let seq = state.sequences[l.slot].as_ref().unwrap();
        for blk in blocks.clone() {
            let c = &seq.chunks_slice()[blk];
            for h in 0..N_KV_HEAD {
                for p in 0..N_PALETTE {
                    for side in 0..2 {
                        let i = h * N_PALETTE + p;
                        let (gid, tag, pal, scale) = if side == 0 {
                            (c.gids.k_gid_pal(h, p), c.k_fmt[i], &c.k_pal, &c.k_scale)
                        } else {
                            (c.gids.v_gid_pal(h, p), c.v_fmt[i], &c.v_pal, &c.v_scale)
                        };
                        let format = KvFormat::from_tag(tag).expect("a stored format");
                        let KvFormat::Quantized(q) = format else {
                            panic!("layer {} block {blk}: a float band", b.layer_idx)
                        };
                        let mut bytes = vec![0u8; payload_bytes(format, elems)];
                        b.read_chunk_into_bytes(gid.arena_idx(), gid.chunk_idx(), &mut bytes)
                            .unwrap();
                        let pb = HEAD_DIM / 4;
                        out.push(Band {
                            fmt: ggml_dtype_to_arena_fmt_code(q.to_ggml_dtype()).unwrap(),
                            scale: scale.get(i).copied().unwrap_or(1.0).to_bits(),
                            pal: pal[h * pb..(h + 1) * pb].to_vec(),
                            bytes,
                        });
                    }
                }
            }
        }
    }
    out
}

/// Every layer's chunk at `blk`: quantized, with a record, sealed below the
/// writer boundary.
pub(super) fn assert_installed(l: &Live, blk: usize) {
    for b in &l.layers {
        let state = b.state.read().unwrap();
        let seq = state.sequences[l.slot].as_ref().unwrap();
        let c = &seq.chunks_slice()[blk];
        assert!(
            !c.takes_writes(),
            "layer {} block {blk} is quantized",
            b.layer_idx
        );
        assert!(
            c.meta.is_some(),
            "layer {} block {blk} has its record",
            b.layer_idx
        );
        assert!(
            seq.writer_start_idx() > blk,
            "block {blk} is below the writer boundary"
        );
    }
}

fn native_key(l: &Live) -> ArenaKey {
    l.layers[0]
        .inner
        .arena_key_for(KvFormat::Float(DType::F16), ArenaLocation::Gpu)
        .unwrap()
}

#[test]
fn a_caught_up_gap_leaves_todays_chunks_in_place() {
    for policy in [CompressionPolicy::new(5), {
        let mut p = CompressionPolicy::new(5);
        p.override_k_quant = Some(QuantFormat::Q8_0);
        p
    }] {
        let Some(l) = live(&policy) else {
            return;
        };
        let jobs = take_jobs(&l.layers, l.slot);
        let n = jobs.len();
        let mut lease = lease_for(&l.layers[0], &policy, |_| n * N_KV_HEAD * 2 * N_PALETTE);
        let done = gap_and_catch_up(&l, &policy, jobs, &mut lease, || {});
        assert_eq!((done.installed, done.stale), (n, 0));
        assert!(done.retry.is_empty() && !done.errored);
        assert!(lease.is_empty(), "the catch-up empties the lease");
        for blk in 0..3 {
            assert_installed(&l, blk);
        }
        let got = live_bands(&l, 0..3);
        assert_eq!(got.len(), l.want.len());
        for (i, (g, w)) in got.iter().zip(&l.want).enumerate() {
            assert_eq!(g, w, "band {i}");
        }
    }
}

/// The native bands go back to their pool once installed, and every leased
/// slot either holds a block or is back in its own.
#[test]
fn installed_blocks_free_their_natives_and_the_rest_of_the_lease() {
    let policy = CompressionPolicy::new(5);
    let Some(l) = live(&policy) else {
        return;
    };
    let jobs = take_jobs(&l.layers, l.slot);
    let bands = jobs.len() * N_KV_HEAD * 2 * N_PALETTE;
    let native_free = l.layers[0].inner.pool.free_slots(native_key(&l));
    let mut lease = lease_for(&l.layers[0], &policy, |_| bands);
    let held = lease.held();
    let keys: Vec<ArenaKey> = (0..SizeClass::COUNT)
        .map(|c| ArenaKey::new(SizeClass::from_index(c).unwrap(), ArenaLocation::Gpu))
        .collect();
    let free_leased: Vec<usize> = keys
        .iter()
        .map(|&k| l.layers[0].inner.pool.free_slots(k))
        .collect();
    let done = gap_and_catch_up(&l, &policy, jobs, &mut lease, || {});
    assert_eq!(
        l.layers[0].inner.pool.free_slots(native_key(&l)),
        native_free + bands,
        "every native band is back"
    );
    // The native bands' class is a size class like any other: its pool gets
    // the natives back as well as its share of the lease's surplus.
    let native_class = native_key(&l).class.index();
    for c in 0..SizeClass::COUNT {
        let natives = if c == native_class { bands } else { 0 };
        assert_eq!(
            l.layers[0].inner.pool.free_slots(keys[c]),
            free_leased[c] + held[c] - done.claims[c] + natives,
            "class {c}: the lease's surplus is back, the claims hold blocks"
        );
    }
}

/// A block truncated away while its gap runs is left alone: its row is
/// discarded, its slots go back, and every other block installs.
#[test]
fn a_block_truncated_under_its_gap_is_discarded() {
    let policy = CompressionPolicy::new(5);
    let Some(l) = live(&policy) else {
        return;
    };
    let jobs = take_jobs(&l.layers, l.slot);
    let n = jobs.len();
    let mut lease = lease_for(&l.layers[0], &policy, |_| n * N_KV_HEAD * 2 * N_PALETTE);
    let done = gap_and_catch_up(&l, &policy, jobs, &mut lease, || {
        l.layers[1].truncate_sequence_to_blocks(l.slot, 1).unwrap();
    });
    assert_eq!((done.installed, done.stale), (n - 2, 2));
    assert!(lease.is_empty());
    let state = l.layers[1].state.read().unwrap();
    let seq = state.sequences[l.slot].as_ref().unwrap();
    assert_eq!(seq.block_count(), 1);
    assert!(
        !seq.chunks_slice()[0].takes_writes(),
        "the block it kept installed"
    );
}

/// A lease short of the table defers blocks; the catch-up hands them back with
/// their exact demand, leaves them native, and a second gap leading with them
/// on exactly that demand installs every one as today's seal does.
#[test]
fn deferred_blocks_come_back_and_install_in_the_next_gap() {
    let policy = CompressionPolicy::new(5);
    let Some(l) = live(&policy) else {
        return;
    };
    let jobs = take_jobs(&l.layers, l.slot);
    let n = jobs.len();
    let mut lease = lease_for(&l.layers[0], &policy, |_| 40);
    let done = gap_and_catch_up(&l, &policy, jobs, &mut lease, || {});
    assert!(!done.retry.is_empty() && done.installed > 0);
    assert_eq!(done.installed + done.retry.len(), n);
    assert_eq!(
        done.retried_demand.iter().sum::<usize>(),
        done.retry.len() * N_KV_HEAD * 2 * N_PALETTE
    );
    for (_, job) in &done.retry {
        let state = l.layers[job.layer].state.read().unwrap();
        let seq = state.sequences[l.slot].as_ref().unwrap();
        assert!(
            seq.chunks_slice()[job.block].takes_writes(),
            "a deferred block stays native"
        );
    }

    // The next gap: the retried blocks lead, leased at their exact demand.
    let retry: Vec<GapJob> = done.retry.into_iter().map(|(_, j)| j).collect();
    let (mut lease, short) = leased(&l.layers[0], &done.retried_demand).unwrap();
    assert_eq!(short, [0; SizeClass::COUNT]);
    let again = gap_and_catch_up(&l, &policy, retry, &mut lease, || {});
    assert!(again.retry.is_empty());
    assert_eq!(again.installed + done.installed, n);
    for blk in 0..3 {
        assert_installed(&l, blk);
    }
    assert_eq!(live_bands(&l, 0..3), l.want);
}

/// A catch-up reads the summary of the table it was launched with, and
/// empties the lease even when it refuses.
#[test]
fn a_catch_up_refuses_a_table_other_than_its_gaps() {
    let policy = CompressionPolicy::new(5);
    let Some(l) = live(&policy) else {
        return;
    };
    let mut jobs = take_jobs(&l.layers, l.slot);
    let n = jobs.len();
    let mut lease = lease_for(&l.layers[0], &policy, |_| n * N_KV_HEAD * 2 * N_PALETTE);
    let refs: Vec<&ChunkedKvBacking> = l.layers.iter().collect();
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..n,
    }];
    let codes = CodeTables::new(l.layers[0].elems_per_chunk()).unwrap();
    let mut gap = GapDevice::new(&l.cuda, N_KV_HEAD, HEAD_DIM, n).unwrap();
    gap.launch(&refs, &jobs, &groups, &stack(&lease), 1)
        .unwrap();
    gap.wait().unwrap();
    let finished = gap.finished().expect("complete");
    jobs.pop();
    assert!(catch_up(&finished, jobs, &groups, &refs, &mut lease, &codes).is_err());
    assert!(
        lease.is_empty(),
        "nothing stays leased past a refused catch-up"
    );
    for blk in 0..3 {
        let state = l.layers[0].state.read().unwrap();
        let seq = state.sequences[l.slot].as_ref().unwrap();
        assert!(
            seq.chunks_slice()[blk].takes_writes(),
            "nothing was installed"
        );
    }
}
