//! The gap on the device against today's seal (`docs/kv_quantize_in_the_gap.md`
//! §7): the same native chunks, quantized by `quantize_layers_deferred` and its
//! convert, and by one gap launch — selection, claim, convert — must come out
//! with the same format, outer scale, palette map and payload bytes in every
//! band. Only the slots differ: the gap's come from its lease.

use candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async;
use candle::quantized::cuda::{ggml_dtype_to_arena_fmt_code, identity_pal_map};
use candle::quantized::pinned_staging::PinnedBuf;
use candle::{DType, Device, Tensor};
use half::f16;

use super::super::compress::{band_ptr, policy_selection, quantize_layers_deferred_within};
use super::super::compression_policy::CompressionPolicy;
use super::super::gid_pool::ChunkGid;
use super::super::gpu_test_lock::{gpu_serial, GpuGuard};
use super::super::head_gids::ChunkBands;
use super::super::selection_batches::SELECTION_BUDGET_BYTES;
use super::super::size_class::payload_bytes;
use super::super::ChunkedKvBacking;
use super::launch::{GapDevice, GapGroup, GapJob};
use super::lease::{Lease, StackImage};
use super::summary::HeadRow;
use crate::kv_cache::arena_table::{ArenaLocation, N_PALETTE};
use crate::kv_cache::{convert_deferred_descs, KvFormat, SealedSequence};

const N_KV_HEAD: usize = 2;
const HEAD_DIM: usize = 128;
const ARENA_CAPACITY: usize = 256;
const LAYERS: usize = 3;
/// Three full chunks a layer.
const TOKENS: usize = 96;

fn cuda() -> Option<(Device, candle::CudaDevice, GpuGuard)> {
    match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => {
            let Device::Cuda(c) = &d else { unreachable!() };
            let c = c.clone();
            Some((d, c, gpu_serial()))
        }
        _ => None,
    }
}

fn backing(device: &Device, policy: &CompressionPolicy) -> ChunkedKvBacking {
    let b = ChunkedKvBacking::new_with_format(
        4,
        N_KV_HEAD,
        HEAD_DIM,
        KvFormat::Float(DType::F16),
        KvFormat::Float(DType::F16),
        device,
        ARENA_CAPACITY,
    )
    .expect("backing");
    b.keep_reserves(Some(policy)).expect("reserves");
    b
}

/// `TOKENS` of an F16 pattern sealed as one turn: native F16 bands on the GPU.
/// The pattern varies by layer and has structure across channels, so the
/// selection picks more than one format.
fn seed(backing: &ChunkedKvBacking, device: &Device, base: u32) -> SealedSequence {
    let slot = backing.alloc_sequence().unwrap();
    backing.ensure_for_offset(slot, 0, TOKENS).unwrap();
    let total = N_KV_HEAD * TOKENS * HEAD_DIM;
    let data: Vec<f16> = (0..total)
        .map(|i| {
            let d = i % HEAD_DIM;
            let t = (i / HEAD_DIM) % TOKENS;
            let wide = if d.is_multiple_of(7) { 8.0 } else { 1.0 };
            let x = ((base as usize + i) as f32 * 0.37).sin() * wide + (t as f32) * 0.01;
            f16::from_f32(x * (1.0 + (d / 32) as f32))
        })
        .collect();
    let k = Tensor::from_vec(data, (1, N_KV_HEAD, TOKENS, HEAD_DIM), &Device::Cpu)
        .unwrap()
        .to_device(device)
        .unwrap();
    let v = (&k * 0.5).unwrap();
    backing.write_contiguous(slot, 0, &k, &v).unwrap();
    backing.set_len(slot, TOKENS);
    backing.record_turn(slot).unwrap()
}

/// What one band came out as: record format code, outer scale bits, its
/// head's packed palette map, and its converted payload.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Band {
    fmt: u8,
    scale: u32,
    pal: Vec<u8>,
    bytes: Vec<u8>,
}

/// Today's seal of every layer's chunks, band by band in table order: job
/// (layer, chunk), head, palette, K before V.
fn todays_seal(
    device: &Device,
    backings: &[ChunkedKvBacking],
    srcs: &[SealedSequence],
    policy: &CompressionPolicy,
) -> Vec<Band> {
    let refs: Vec<&ChunkedKvBacking> = backings.iter().collect();
    let per_layer: Vec<Vec<&SealedSequence>> = srcs.iter().map(|s| vec![s]).collect();
    let mut descs = Vec::new();
    let mut pinned: Option<PinnedBuf> = None;
    let Device::Cuda(c) = device else {
        unreachable!()
    };
    let sealed = quantize_layers_deferred_within(
        &refs,
        &per_layer,
        policy,
        device,
        &c.cuda_stream(),
        &mut pinned,
        &mut descs,
        &mut 0,
        &mut 0,
        SELECTION_BUDGET_BYTES,
    )
    .expect("today's quantize");
    convert_deferred_descs(&backings[0], &descs, N_KV_HEAD, device).expect("today's convert");
    device.synchronize().unwrap();

    let elems = backings[0].elems_per_chunk();
    let ident = identity_pal_map(HEAD_DIM);
    let pal_bytes = HEAD_DIM / 4;
    let mut out = Vec::new();
    for (layer, seqs) in sealed.iter().enumerate() {
        for chunk in &seqs[0].chunks {
            for h in 0..N_KV_HEAD {
                for p in 0..N_PALETTE {
                    for side in 0..2 {
                        let (gid, tag, pal, scale) = if side == 0 {
                            (
                                chunk.gids.k_gid_pal(h, p),
                                chunk.k_fmt[h * N_PALETTE + p],
                                &chunk.k_pal,
                                &chunk.k_scale,
                            )
                        } else {
                            (
                                chunk.gids.v_gid_pal(h, p),
                                chunk.v_fmt[h * N_PALETTE + p],
                                &chunk.v_pal,
                                &chunk.v_scale,
                            )
                        };
                        let format = KvFormat::from_tag(tag).expect("a stored format");
                        let KvFormat::Quantized(q) = format else {
                            panic!("today's seal left a float band")
                        };
                        let mut bytes = vec![0u8; payload_bytes(format, elems)];
                        backings[layer]
                            .read_chunk_into_bytes(gid.arena_idx(), gid.chunk_idx(), &mut bytes)
                            .expect("read a sealed band");
                        out.push(Band {
                            fmt: ggml_dtype_to_arena_fmt_code(q.to_ggml_dtype()).unwrap(),
                            scale: scale
                                .get(h * N_PALETTE + p)
                                .copied()
                                .unwrap_or(1.0)
                                .to_bits(),
                            pal: if pal.is_empty() {
                                ident[..pal_bytes].to_vec()
                            } else {
                                pal[h * pal_bytes..(h + 1) * pal_bytes].to_vec()
                            },
                            bytes,
                        });
                    }
                }
            }
        }
    }
    out
}

/// The gap's jobs: every layer's chunks, layer-major, as the table orders them.
fn jobs(srcs: &[SealedSequence]) -> Vec<GapJob> {
    let mut jobs = Vec::new();
    for (layer, s) in srcs.iter().enumerate() {
        for c in &s.chunks {
            let len = usize::from(c.token_count).clamp(1, 32) as i32;
            jobs.push(GapJob {
                layer,
                bands: ChunkBands::from_sealed(c),
                valid_range: ((c.offset as i32) << 8) | len,
            });
        }
    }
    jobs
}

/// A lease of `per_class(class)` slots for every class the policy's candidates
/// and overrides reach, claimed from `backing`'s pools.
fn lease_for(
    backing: &ChunkedKvBacking,
    policy: &CompressionPolicy,
    per_class: impl Fn(usize) -> usize,
) -> Lease<ChunkGid> {
    let sel = policy_selection(policy).unwrap();
    let mut formats: Vec<_> = sel
        .k
        .iter()
        .chain(&sel.v)
        .filter_map(|f| f.to_quant_format())
        .collect();
    formats.extend(policy.override_k_quant);
    formats.extend(policy.override_v_quant);
    let mut lease = Lease::default();
    let mut done = Vec::new();
    for q in formats {
        let key = backing
            .inner
            .arena_key_for(KvFormat::Quantized(q), ArenaLocation::Gpu)
            .unwrap();
        let class = key.class.index();
        if done.contains(&class) {
            continue;
        }
        done.push(class);
        let slots: Vec<ChunkGid> = (0..per_class(class))
            .map(|_| backing.alloc_chunk_for_key(key).expect("lease a slot"))
            .collect();
        lease.add(class, slots);
    }
    lease
}

fn stack(backing: &ChunkedKvBacking, lease: &Lease<ChunkGid>) -> StackImage {
    backing
        .inner
        .storage
        .read(|s| lease.stack_image(|g| band_ptr(s, g, "lease")))
        .unwrap()
        .unwrap()
}

/// Launch one gap and wait for it; every head's summary row and every job's
/// deferred flag.
fn run_gap(
    cuda: &candle::CudaDevice,
    device: &Device,
    backings: &[ChunkedKvBacking],
    jobs: &[GapJob],
    groups: &[GapGroup<'_>],
    image: &StackImage,
) -> (Vec<HeadRow>, Vec<bool>, bool) {
    let refs: Vec<&ChunkedKvBacking> = backings.iter().collect();
    let mut gap = GapDevice::new(cuda, N_KV_HEAD, HEAD_DIM, jobs.len().max(4)).unwrap();
    gap.launch(&refs, jobs, groups, image, 7).unwrap();
    device.synchronize().unwrap();
    assert!(
        gap.poll(),
        "the completion word is in once the stream drains"
    );
    let s = gap.summary();
    // SAFETY: the gap has completed.
    unsafe {
        let rows = (0..jobs.len() * N_KV_HEAD)
            .map(|h| s.head_row(h).unwrap())
            .collect();
        let deferred = (0..jobs.len()).map(|j| s.deferred(j)).collect();
        (rows, deferred, s.errored())
    }
}

/// A row's bands in table order, the payload read from each band's slot.
fn gap_bands(cuda: &candle::CudaDevice, rows: &[HeadRow], elems: usize) -> Vec<Band> {
    let stream = cuda.cuda_stream();
    let mut out = Vec::new();
    for row in rows {
        for p in 0..N_PALETTE {
            for side in 0..2 {
                let (fmt, scale, pal, ptr) = if side == 0 {
                    (row.k_fmt[p], row.k_scale[p], &row.k_pal, row.k_ptr[p])
                } else {
                    (row.v_fmt[p], row.v_scale[p], &row.v_pal, row.v_ptr[p])
                };
                let format = arena_code_format(fmt);
                let mut bytes = vec![0u8; payload_bytes(format, elems)];
                // SAFETY: a leased slot of at least this format's payload,
                // written by a completed gap; waited for just below.
                unsafe { memcpy_dtoh_async(&mut bytes, ptr, stream.cu_stream()).unwrap() };
                stream.synchronize().unwrap();
                out.push(Band {
                    fmt,
                    scale: scale.to_bits(),
                    pal: pal.clone(),
                    bytes,
                });
            }
        }
    }
    out
}

/// The quantized format a record format code names.
fn arena_code_format(code: u8) -> KvFormat {
    crate::kv_cache::all_kv_formats()
        .find(|f| match f {
            KvFormat::Quantized(q) => {
                ggml_dtype_to_arena_fmt_code(q.to_ggml_dtype()).ok() == Some(code)
            }
            KvFormat::Float(_) => false,
        })
        .unwrap_or_else(|| panic!("no quantized format has record code {code}"))
}

fn setup(
    device: &Device,
    policy: &CompressionPolicy,
) -> (Vec<ChunkedKvBacking>, Vec<SealedSequence>) {
    let backings: Vec<_> = (0..LAYERS).map(|_| backing(device, policy)).collect();
    let srcs = backings
        .iter()
        .enumerate()
        .map(|(l, b)| seed(b, device, (l as u32 + 1) * 1000))
        .collect();
    (backings, srcs)
}

fn assert_matches_todays_seal(policy: &CompressionPolicy, split: bool) {
    let Some((device, cuda, _gpu)) = cuda() else {
        return;
    };
    let (backings, srcs) = setup(&device, policy);
    let want = todays_seal(&device, &backings, &srcs, policy);
    let jobs = jobs(&srcs);
    let bands = jobs.len() * N_KV_HEAD * 2 * N_PALETTE;
    let lease = lease_for(&backings[0], policy, |_| bands);
    let image = stack(&backings[0], &lease);
    let groups = if split {
        // Two groups of one policy: the second continues the first's counts.
        vec![
            GapGroup { policy, jobs: 0..4 },
            GapGroup {
                policy,
                jobs: 4..jobs.len(),
            },
        ]
    } else {
        vec![GapGroup {
            policy,
            jobs: 0..jobs.len(),
        }]
    };
    let (rows, deferred, errored) = run_gap(&cuda, &device, &backings, &jobs, &groups, &image);
    assert!(!errored, "every code is covered");
    assert!(deferred.iter().all(|d| !d), "an ample lease defers nothing");
    let got = gap_bands(&cuda, &rows, backings[0].elems_per_chunk());
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "band {i} (job {}, head {})", i / 16, (i / 8) % 2);
    }
    let formats: std::collections::BTreeSet<u8> = want.iter().map(|b| b.fmt).collect();
    assert!(
        formats.len() > 1,
        "the selection chose more than one format: {formats:?}"
    );
    // Every band took a distinct slot.
    let mut slots: Vec<u64> = rows
        .iter()
        .flat_map(|r| r.k_ptr.iter().chain(&r.v_ptr).copied())
        .collect();
    slots.sort_unstable();
    slots.dedup();
    assert_eq!(slots.len(), bands);
}

#[test]
fn the_gap_seals_the_same_bands_as_today() {
    assert_matches_todays_seal(&CompressionPolicy::new(5), false);
}

#[test]
fn a_gap_of_two_groups_seals_the_same_bands_as_one() {
    assert_matches_todays_seal(&CompressionPolicy::new(5), true);
}

#[test]
fn the_gap_seals_the_same_bands_as_today_under_a_k_override() {
    let mut policy = CompressionPolicy::new(5);
    policy.override_k_quant = Some(crate::kv_cache::QuantFormat::Q8_0);
    assert_matches_todays_seal(&policy, false);
}

#[test]
fn the_gap_seals_the_same_bands_as_today_at_the_deepest_level() {
    assert_matches_todays_seal(&CompressionPolicy::new(10), false);
}

/// A lease short of the gap's demand defers exactly the jobs whose bands fall
/// past it, in table order; their bands go to the sink, every other job's
/// bands are today's, and each band's lease index is its place in its class.
#[test]
fn a_short_lease_defers_the_tail_and_seals_the_rest() {
    let Some((device, cuda, _gpu)) = cuda() else {
        return;
    };
    let policy = CompressionPolicy::new(5);
    let (backings, srcs) = setup(&device, &policy);
    let want = todays_seal(&device, &backings, &srcs, &policy);
    let jobs = jobs(&srcs);
    const SHORT: usize = 40;
    let lease = lease_for(&backings[0], &policy, |_| SHORT);
    let image = stack(&backings[0], &lease);
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..jobs.len(),
    }];
    let (rows, deferred, errored) = run_gap(&cuda, &device, &backings, &jobs, &groups, &image);
    assert!(!errored);

    // The expected claim: each band's index is the count of earlier bands of
    // its class, and a job defers when any of its bands is past the lease.
    let elems = backings[0].elems_per_chunk();
    let class_of = |fmt: u8| {
        backings[0]
            .inner
            .arena_key_for(arena_code_format(fmt), ArenaLocation::Gpu)
            .unwrap()
            .class
            .index()
    };
    let mut taken = [0usize; 32];
    let per_job = N_KV_HEAD * 2 * N_PALETTE;
    let mut want_deferred = Vec::new();
    let mut want_idx = Vec::new();
    for job in want.chunks(per_job) {
        let mut d = false;
        for band in job {
            let c = class_of(band.fmt);
            want_idx.push(taken[c] as u32);
            d |= taken[c] >= SHORT;
            taken[c] += 1;
        }
        want_deferred.push(d);
    }
    assert_eq!(deferred, want_deferred);
    assert!(deferred.iter().any(|&d| d) && deferred.iter().any(|&d| !d));
    let got_idx: Vec<u32> = rows
        .iter()
        .flat_map(|r| r.lease_idx.iter().copied())
        .collect();
    assert_eq!(got_idx, want_idx);

    let got = gap_bands(&cuda, &rows, elems);
    for (j, d) in deferred.iter().enumerate() {
        let span = j * per_job..(j + 1) * per_job;
        let heads = &rows[j * N_KV_HEAD..(j + 1) * N_KV_HEAD];
        let sinks = heads
            .iter()
            .flat_map(|r| r.k_ptr.iter().chain(&r.v_ptr))
            .collect::<std::collections::BTreeSet<_>>();
        if *d {
            assert_eq!(sinks.len(), 1, "job {j}'s bands all go to the sink");
            for (g, w) in got[span.clone()].iter().zip(&want[span]) {
                assert_eq!((g.fmt, g.scale, &g.pal), (w.fmt, w.scale, &w.pal));
            }
        } else {
            assert_eq!(got[span.clone()], want[span], "job {j}");
        }
    }
}

/// A second gap is refused while the first is outstanding, and accepted once
/// its completion word is seen.
#[test]
fn one_gap_is_outstanding_at_a_time() {
    let Some((device, cuda, _gpu)) = cuda() else {
        return;
    };
    let policy = CompressionPolicy::new(5);
    let (backings, srcs) = setup(&device, &policy);
    let jobs = jobs(&srcs);
    let bands = jobs.len() * N_KV_HEAD * 2 * N_PALETTE;
    let lease = lease_for(&backings[0], &policy, |_| bands);
    let image = stack(&backings[0], &lease);
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..jobs.len(),
    }];
    let refs: Vec<&ChunkedKvBacking> = backings.iter().collect();
    let mut gap = GapDevice::new(&cuda, N_KV_HEAD, HEAD_DIM, jobs.len()).unwrap();
    gap.launch(&refs, &jobs, &groups, &image, 1).unwrap();
    // Not yet polled complete: refused whether or not the device has finished.
    assert!(gap.launch(&refs, &jobs, &groups, &image, 2).is_err());
    device.synchronize().unwrap();
    assert!(gap.poll());
    gap.launch(&refs, &jobs, &groups, &image, 2).unwrap();
    device.synchronize().unwrap();
    assert!(gap.poll());
}
