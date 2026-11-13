//! The gap on the device against today's seal (`docs/kv_quantize_in_the_gap.md`
//! §7): the same native chunks, quantized by `quantize_layers_deferred` and its
//! convert, and by one gap launch — selection, claim, convert — must come out
//! with the same format, outer scale, palette map and payload bytes in every
//! band. Only the slots differ: the gap's come from its lease.

use std::collections::BTreeSet;
use std::sync::Arc;

use candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async;
use candle::quantized::cuda::{ggml_dtype_to_arena_fmt_code, identity_pal_map};
use candle::quantized::pinned_staging::PinnedBuf;
use candle::{CudaDevice, DType, Device, Tensor};
use half::f16;

use super::super::arena::ArenaKey;
use super::super::compress::{policy_selection, quantize_layers_deferred_within};
use super::super::compression_policy::CompressionPolicy;
use super::super::gid_pool::ChunkGid;
use super::super::gpu_test_lock::{gpu_serial, GpuGuard};
use super::super::head_gids::ChunkBands;
use super::super::selection_batches::SELECTION_BUDGET_BYTES;
use super::super::size_class::{class_for_format, payload_bytes, SizeClass};
use super::super::ChunkedKvBacking;
use super::codes::CodeTables;
use super::demand::{count_bands, gap_demand, PerClass};
use super::launch::{GapDevice, GapGroup, GapJob};
use super::lease::{Lease, StackImage};
use super::lease_alloc_tests::leased;
use super::summary::{completion_word, HeadRow};
use crate::kv_cache::arena_table::{ArenaLocation, N_PALETTE};
use crate::kv_cache::{
    all_kv_formats, convert_deferred_descs, KvFormat, QuantFormat, SealedSequence,
};

const N_KV_HEAD: usize = 2;
const HEAD_DIM: usize = 128;
const ARENA_CAPACITY: usize = 256;
const LAYERS: usize = 3;
/// Three full chunks a layer.
const TOKENS: usize = 96;

pub(super) fn cuda() -> Option<(Device, CudaDevice, GpuGuard)> {
    match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => {
            let Device::Cuda(c) = &d else { unreachable!() };
            let c = c.clone();
            Some((d, c, gpu_serial()))
        }
        _ => None,
    }
}

pub(super) fn backing(device: &Device, policy: &CompressionPolicy) -> ChunkedKvBacking {
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

/// The classes the policy's candidates and overrides reach.
pub(super) fn reachable(backing: &ChunkedKvBacking, policy: &CompressionPolicy) -> PerClass<bool> {
    let sel = policy_selection(policy).unwrap();
    let mut formats: Vec<_> = sel
        .k
        .iter()
        .chain(&sel.v)
        .filter_map(|f| f.to_quant_format())
        .collect();
    formats.extend(policy.override_k_quant);
    formats.extend(policy.override_v_quant);
    let mut reach = [false; SizeClass::COUNT];
    for q in formats {
        let key = backing
            .inner
            .arena_key_for(KvFormat::Quantized(q), ArenaLocation::Gpu)
            .unwrap();
        reach[key.class.index()] = true;
    }
    reach
}

/// A lease of `per_class(class)` slots for every class the policy reaches,
/// allocated from `backing`'s pools as a gap's lease is.
fn lease_for(
    backing: &ChunkedKvBacking,
    policy: &CompressionPolicy,
    per_class: impl Fn(usize) -> usize,
) -> Lease<ChunkGid> {
    let reach = reachable(backing, policy);
    let demand: PerClass<usize> = std::array::from_fn(|c| if reach[c] { per_class(c) } else { 0 });
    let (lease, short) = leased(backing, &demand).expect("lease");
    assert_eq!(
        short,
        [0; SizeClass::COUNT],
        "the test card covers the lease"
    );
    lease
}

fn stack(lease: &Lease<ChunkGid>) -> StackImage {
    lease.stack_image().unwrap()
}

/// Launch one gap and wait for it; every head's summary row and every job's
/// deferred flag.
fn run_gap(
    cuda: &CudaDevice,
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
fn gap_bands(cuda: &CudaDevice, rows: &[HeadRow], elems: usize) -> Vec<Band> {
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
    all_kv_formats()
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
    let image = stack(&lease);
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
    let formats: BTreeSet<u8> = want.iter().map(|b| b.fmt).collect();
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
    policy.override_k_quant = Some(QuantFormat::Q8_0);
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
    let image = stack(&lease);
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
            .collect::<BTreeSet<_>>();
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
    let image = stack(&lease);
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

/// Each band of `row` with the class its record format claimed from, in lease
/// index order: palette-major, K before V.
fn row_bands(row: &HeadRow, codes: &CodeTables) -> [(usize, u32); 8] {
    std::array::from_fn(|b| {
        let (p, side) = (b / 2, b % 2);
        let fmt = if side == 0 {
            row.k_fmt[p]
        } else {
            row.v_fmt[p]
        };
        let class = codes.class_of_fmt(fmt).expect("a claimed format");
        (class, row.lease_idx[b])
    })
}

/// One block's slots out of the lease, as the catch-up takes them: per class,
/// the indices its rows name, in band order.
fn take_block(lease: &mut Lease<ChunkGid>, rows: &[HeadRow], codes: &CodeTables) -> Vec<ChunkGid> {
    let mut by_class = vec![Vec::new(); SizeClass::COUNT];
    for row in rows {
        for (class, idx) in row_bands(row, codes) {
            by_class[class].push(idx);
        }
    }
    let mut out = Vec::new();
    for (c, idx) in by_class.iter().enumerate() {
        out.extend(lease.take(c, idx).expect("the summary matches its lease"));
    }
    out
}

/// One gap's whole cycle against the pools (§3, §3.5): a lease short of the
/// table defers some blocks; the catch-up takes the installed blocks' slots by
/// the summary's indices and returns every other slot to its pool; the next
/// gap leases the deferred blocks' exact demand from their rows, leads with
/// them, defers nothing, and seals them as today's seal does.
#[test]
fn a_gap_cycle_returns_every_unused_slot_and_a_retry_places_every_deferred_block() {
    let Some((device, cuda, _gpu)) = cuda() else {
        return;
    };
    let policy = CompressionPolicy::new(5);
    let (backings, srcs) = setup(&device, &policy);
    let want = todays_seal(&device, &backings, &srcs, &policy);
    let jobs = jobs(&srcs);
    let codes = CodeTables::new(backings[0].elems_per_chunk()).unwrap();
    let per_job = N_KV_HEAD * 2 * N_PALETTE;

    // ── Gap 1, on a short lease ──
    let mut lease = lease_for(&backings[0], &policy, |_| 40);
    let held = lease.held();
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..jobs.len(),
    }];
    let (rows, deferred, errored) =
        run_gap(&cuda, &device, &backings, &jobs, &groups, &stack(&lease));
    assert!(!errored);
    assert!(deferred.iter().any(|&d| d) && deferred.iter().any(|&d| !d));

    // The summary's demand: every band of the table, deferred or not.
    let demand = count_bands(&rows, |f| codes.class_of_fmt(f)).unwrap();
    assert_eq!(demand.iter().sum::<usize>(), jobs.len() * per_job);

    // ── Catch-up: the installed blocks take their slots, the rest go back ──
    let keys: Vec<ArenaKey> = (0..SizeClass::COUNT)
        .map(|c| ArenaKey::new(SizeClass::from_index(c).unwrap(), ArenaLocation::Gpu))
        .collect();
    let free = |c: usize| backings[0].inner.pool.free_slots(keys[c]);
    let free_leased: Vec<usize> = (0..SizeClass::COUNT).map(free).collect();
    // Block by block, as the catch-up installs them.
    let mut taken_by_class = [0usize; SizeClass::COUNT];
    let mut installed: Vec<ChunkGid> = Vec::new();
    for (j, _) in deferred.iter().enumerate().filter(|(_, &d)| !d) {
        installed.extend(take_block(
            &mut lease,
            &rows[j * N_KV_HEAD..(j + 1) * N_KV_HEAD],
            &codes,
        ));
        for row in &rows[j * N_KV_HEAD..(j + 1) * N_KV_HEAD] {
            for (class, _) in row_bands(row, &codes) {
                taken_by_class[class] += 1;
            }
        }
    }
    drop(lease.release());
    assert!(lease.is_empty());
    for c in 0..SizeClass::COUNT {
        assert_eq!(
            free(c),
            free_leased[c] + held[c] - taken_by_class[c],
            "class {c}: every slot not installed went back"
        );
    }

    // ── Gap 2: the deferred blocks alone, leased at their exact demand ──
    let retry: Vec<usize> = (0..jobs.len()).filter(|&j| deferred[j]).collect();
    let retried = count_bands(
        retry
            .iter()
            .flat_map(|&j| &rows[j * N_KV_HEAD..(j + 1) * N_KV_HEAD]),
        |f| codes.class_of_fmt(f),
    )
    .unwrap();
    let reach = reachable(&backings[0], &policy);
    let exact = gap_demand(&retried, 0, None, &reach);
    assert_eq!(exact, retried);
    let (lease2, short) = leased(&backings[0], &exact).unwrap();
    assert_eq!(short, [0; SizeClass::COUNT]);
    let retry_jobs: Vec<GapJob> = retry.iter().map(|&j| jobs[j].clone()).collect();
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..retry_jobs.len(),
    }];
    let (rows2, deferred2, errored2) = run_gap(
        &cuda,
        &device,
        &backings,
        &retry_jobs,
        &groups,
        &stack(&lease2),
    );
    assert!(!errored2);
    assert!(
        deferred2.iter().all(|&d| !d),
        "the exact demand places every retried block"
    );
    let got = gap_bands(&cuda, &rows2, backings[0].elems_per_chunk());
    for (k, &j) in retry.iter().enumerate() {
        assert_eq!(
            got[k * per_job..(k + 1) * per_job],
            want[j * per_job..(j + 1) * per_job],
            "retried job {j}"
        );
    }
    // The retry used its whole lease and nothing more.
    let used: usize = rows2.iter().map(|r| row_bands(r, &codes).len()).sum();
    assert_eq!(used, exact.iter().sum::<usize>());
    drop(installed);
}

/// Retried blocks keep their exact lease only if they claim first. The claim
/// counts every group of the gap in table order, so the retried blocks lead in
/// groups of their own, ahead of every group of new blocks — whatever the new
/// blocks' policy. Leased exactly the retried demand, that order places every
/// retried block and defers every new one; the reverse order lets the new
/// blocks take the retried blocks' slots.
#[test]
fn retried_blocks_in_their_own_leading_groups_keep_their_exact_lease() {
    let Some((device, cuda, _gpu)) = cuda() else {
        return;
    };
    let c5 = CompressionPolicy::new(5);
    let c10 = CompressionPolicy::new(10);
    let (backings, srcs) = setup(&device, &c5);
    let jobs = jobs(&srcs);
    let codes = CodeTables::new(backings[0].elems_per_chunk()).unwrap();

    // Gap 1 on a short lease: some blocks defer.
    let lease = lease_for(&backings[0], &c5, |_| 40);
    let all = [GapGroup {
        policy: &c5,
        jobs: 0..jobs.len(),
    }];
    let (rows, deferred, _) = run_gap(&cuda, &device, &backings, &jobs, &all, &stack(&lease));
    drop(lease);
    let retry: Vec<usize> = (0..jobs.len()).filter(|&j| deferred[j]).collect();
    let fresh: Vec<usize> = (0..jobs.len()).filter(|&j| !deferred[j]).collect();
    assert!(!retry.is_empty() && !fresh.is_empty());
    let retried = count_bands(
        retry
            .iter()
            .flat_map(|&j| &rows[j * N_KV_HEAD..(j + 1) * N_KV_HEAD]),
        |f| codes.class_of_fmt(f),
    )
    .unwrap();

    let table = |first: &[usize], then: &[usize]| -> Vec<GapJob> {
        first.iter().chain(then).map(|&j| jobs[j].clone()).collect()
    };
    let (nr, nf) = (retry.len(), fresh.len());

    // Retried first, under C5; the new blocks after them, under C10.
    let (lease, short) = leased(&backings[0], &retried).unwrap();
    assert_eq!(short, [0; SizeClass::COUNT]);
    let ordered = table(&retry, &fresh);
    let groups = [
        GapGroup {
            policy: &c5,
            jobs: 0..nr,
        },
        GapGroup {
            policy: &c10,
            jobs: nr..nr + nf,
        },
    ];
    let (_, d, errored) = run_gap(&cuda, &device, &backings, &ordered, &groups, &stack(&lease));
    assert!(!errored);
    assert!(d[..nr].iter().all(|&x| !x), "every retried block placed");
    assert!(
        d[nr..].iter().all(|&x| x),
        "the exact lease leaves nothing new"
    );
    drop(lease);

    // The new blocks first: they claim the retried blocks' slots.
    let (lease, _) = leased(&backings[0], &retried).unwrap();
    let reversed = table(&fresh, &retry);
    let groups = [
        GapGroup {
            policy: &c10,
            jobs: 0..nf,
        },
        GapGroup {
            policy: &c5,
            jobs: nf..nf + nr,
        },
    ];
    let (_, d, _) = run_gap(
        &cuda,
        &device,
        &backings,
        &reversed,
        &groups,
        &stack(&lease),
    );
    assert!(
        d[nf..].iter().any(|&x| x),
        "behind the new blocks a retried block is deferred again"
    );
}

/// A gap whose lease covers its table, through the catch-up: every block is
/// installed, every index the summary names is a distinct slot of the lease,
/// the summary's demand is the table's band count, and only the surplus goes
/// back — with no override, and with each side under a uniform override,
/// whose bands all claim from the override's class.
#[test]
fn an_ample_gap_installs_every_block_and_returns_only_the_surplus() {
    let Some((device, cuda, _gpu)) = cuda() else {
        return;
    };
    for (k, v) in [
        (None, None),
        (Some(QuantFormat::Q8_0), None),
        (None, Some(QuantFormat::Q4_0)),
    ] {
        let mut policy = CompressionPolicy::new(5);
        policy.override_k_quant = k;
        policy.override_v_quant = v;
        let (backings, srcs) = setup(&device, &policy);
        let jobs = jobs(&srcs);
        let codes = CodeTables::new(backings[0].elems_per_chunk()).unwrap();
        let bands = jobs.len() * N_KV_HEAD * 2 * N_PALETTE;
        let mut lease = lease_for(&backings[0], &policy, |_| bands);
        let held = lease.held();
        let groups = [GapGroup {
            policy: &policy,
            jobs: 0..jobs.len(),
        }];
        let (rows, deferred, errored) =
            run_gap(&cuda, &device, &backings, &jobs, &groups, &stack(&lease));
        assert!(!errored && deferred.iter().all(|&d| !d), "{k:?} {v:?}");

        let demand = count_bands(&rows, |f| codes.class_of_fmt(f)).unwrap();
        assert_eq!(demand.iter().sum::<usize>(), bands, "{k:?} {v:?}");
        let mut by_class = vec![Vec::new(); SizeClass::COUNT];
        for row in &rows {
            for (class, idx) in row_bands(row, &codes) {
                by_class[class].push(idx);
            }
        }
        for (c, idx) in by_class.iter().enumerate() {
            assert_eq!(idx.len(), demand[c], "{k:?} {v:?} class {c}");
            let mut sorted = idx.clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                (0..idx.len() as u32).collect::<Vec<_>>(),
                "{k:?} {v:?} class {c}: the first slots of the run, each once"
            );
        }
        // Under an override, every band of that side claims its format's class.
        let side_class = |q: QuantFormat| {
            class_for_format(KvFormat::Quantized(q), backings[0].elems_per_chunk())
                .unwrap()
                .index()
        };
        for row in &rows {
            if let Some(q) = k {
                for &f in &row.k_fmt {
                    assert_eq!(codes.class_of_fmt(f), Some(side_class(q)));
                }
            }
            if let Some(q) = v {
                for &f in &row.v_fmt {
                    assert_eq!(codes.class_of_fmt(f), Some(side_class(q)));
                }
            }
        }

        // Block by block, as the catch-up installs them.
        let mut installed = Vec::new();
        for j in 0..jobs.len() {
            installed.extend(take_block(
                &mut lease,
                &rows[j * N_KV_HEAD..(j + 1) * N_KV_HEAD],
                &codes,
            ));
        }
        let back = lease.release();
        for c in 0..SizeClass::COUNT {
            assert_eq!(
                back[c].len(),
                held[c] - demand[c],
                "{k:?} {v:?} class {c}: only the surplus goes back"
            );
        }
        assert_eq!(installed.len(), bands);
    }
}

/// One gap's fixture: the backings, the jobs, an ample lease and its stack.
/// Fields drop in order: the lease's slots go back to their pools while the
/// backings still stand, and the card is released last.
struct Fixture {
    /// Held for the fixture's life: the stack names these slots.
    _lease: Lease<ChunkGid>,
    image: StackImage,
    jobs: Vec<GapJob>,
    backings: Vec<ChunkedKvBacking>,
    cuda: CudaDevice,
    device: Device,
    _gpu: GpuGuard,
}

fn fixture(policy: &CompressionPolicy) -> Option<Fixture> {
    let (device, cuda, gpu) = cuda()?;
    let (backings, srcs) = setup(&device, policy);
    let jobs = jobs(&srcs);
    let bands = jobs.len() * N_KV_HEAD * 2 * N_PALETTE;
    let lease = lease_for(&backings[0], policy, |_| bands);
    let image = stack(&lease);
    Some(Fixture {
        _lease: lease,
        image,
        jobs,
        backings,
        cuda,
        device,
        _gpu: gpu,
    })
}

/// A job whose token window is empty or runs past its chunk is refused before
/// anything is launched, exactly as today's convert refuses it, and the gap is
/// left free for the next launch.
#[test]
fn a_window_outside_its_chunk_is_refused_before_any_launch() {
    let policy = CompressionPolicy::new(5);
    let Some(f) = fixture(&policy) else {
        return;
    };
    let refs: Vec<&ChunkedKvBacking> = f.backings.iter().collect();
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..f.jobs.len(),
    }];
    let mut gap = GapDevice::new(&f.cuda, N_KV_HEAD, HEAD_DIM, f.jobs.len()).unwrap();
    for bad in [0, (8 << 8) | 25, (32 << 8) | 1, 33] {
        let mut jobs = f.jobs.clone();
        jobs[4].valid_range = bad;
        let err = gap
            .launch(&refs, &jobs, &groups, &f.image, 1)
            .expect_err("a bad window");
        assert!(format!("{err}").contains("token window"), "{bad:#x}: {err}");
        assert!(gap.poll(), "a refused gap leaves nothing outstanding");
    }
    gap.launch(&refs, &f.jobs, &groups, &f.image, 1).unwrap();
    f.device.synchronize().unwrap();
    assert!(gap.poll());
}

/// Refuse `jobs` with `gap`, then show the refusal left nothing outstanding:
/// the fixture's own table launches straight after.
fn assert_refused(f: &Fixture, policy: &CompressionPolicy, jobs: &[GapJob], what: &str) {
    let refs: Vec<&ChunkedKvBacking> = f.backings.iter().collect();
    let mut gap = GapDevice::new(&f.cuda, N_KV_HEAD, HEAD_DIM, f.jobs.len()).unwrap();
    let groups = [GapGroup {
        policy,
        jobs: 0..jobs.len(),
    }];
    assert!(
        gap.launch(&refs, jobs, &groups, &f.image, 1).is_err(),
        "{what}"
    );
    assert!(gap.poll(), "{what}: nothing outstanding");
    let groups = [GapGroup {
        policy,
        jobs: 0..f.jobs.len(),
    }];
    gap.launch(&refs, &f.jobs, &groups, &f.image, 2).unwrap();
    f.device.synchronize().unwrap();
    assert!(gap.poll());
}

#[test]
fn a_band_that_is_not_a_native_float_is_refused() {
    let policy = CompressionPolicy::new(5);
    let Some(f) = fixture(&policy) else {
        return;
    };
    let q8 = KvFormat::Quantized(QuantFormat::Q8_0).to_tag();
    let mut jobs = f.jobs.clone();
    jobs[2].bands.v_fmt = Arc::new(vec![q8; N_KV_HEAD * N_PALETTE]);
    assert_refused(&f, &policy, &jobs, "a quantized source band");
}

#[test]
fn a_band_with_no_recorded_tag_is_refused() {
    let policy = CompressionPolicy::new(5);
    let Some(f) = fixture(&policy) else {
        return;
    };
    let mut jobs = f.jobs.clone();
    // One head short: the second head's bands have no tag.
    jobs[0].bands.k_fmt = Arc::new(jobs[0].bands.k_fmt[..N_PALETTE].to_vec());
    assert_refused(&f, &policy, &jobs, "an unrecorded band");
}

#[test]
fn a_job_on_a_layer_past_the_backings_is_refused() {
    let policy = CompressionPolicy::new(5);
    let Some(f) = fixture(&policy) else {
        return;
    };
    let mut jobs = f.jobs.clone();
    let last = jobs.len() - 1;
    jobs[last].layer = LAYERS;
    assert_refused(&f, &policy, &jobs, "layer past the backings");
}

#[test]
fn an_empty_gap_launches_nothing_and_leaves_the_device_free() {
    let policy = CompressionPolicy::new(5);
    let Some(f) = fixture(&policy) else {
        return;
    };
    let refs: Vec<&ChunkedKvBacking> = f.backings.iter().collect();
    let mut gap = GapDevice::new(&f.cuda, N_KV_HEAD, HEAD_DIM, f.jobs.len()).unwrap();
    gap.launch(&refs, &[], &[], &f.image, 1).unwrap();
    assert!(gap.poll(), "an empty gap is not outstanding");
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..f.jobs.len(),
    }];
    gap.launch(&refs, &f.jobs, &groups, &f.image, 2).unwrap();
    f.device.synchronize().unwrap();
    assert!(gap.poll());
}

#[test]
fn a_gap_wider_than_its_device_is_refused() {
    let policy = CompressionPolicy::new(5);
    let Some(f) = fixture(&policy) else {
        return;
    };
    let refs: Vec<&ChunkedKvBacking> = f.backings.iter().collect();
    let mut gap = GapDevice::new(&f.cuda, N_KV_HEAD, HEAD_DIM, f.jobs.len() - 1).unwrap();
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..f.jobs.len(),
    }];
    assert!(gap.launch(&refs, &f.jobs, &groups, &f.image, 1).is_err());
    assert!(gap.poll());
}

#[test]
fn a_layer_of_another_geometry_is_refused() {
    let policy = CompressionPolicy::new(5);
    let Some(f) = fixture(&policy) else {
        return;
    };
    let refs: Vec<&ChunkedKvBacking> = f.backings.iter().collect();
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..f.jobs.len(),
    }];
    for (nkh, hd) in [(N_KV_HEAD + 1, HEAD_DIM), (N_KV_HEAD, HEAD_DIM * 2)] {
        let mut gap = GapDevice::new(&f.cuda, nkh, hd, f.jobs.len()).unwrap();
        assert!(
            gap.launch(&refs, &f.jobs, &groups, &f.image, 1).is_err(),
            "{nkh} heads at {hd}"
        );
    }
}

#[test]
fn the_device_refuses_a_shape_the_claim_and_convert_do_not_take() {
    let Some((_device, cuda, _gpu)) = cuda() else {
        return;
    };
    for (nkh, hd, jobs, what) in [
        (0, 128, 8, "no heads"),
        (65, 128, 8, "more heads than a tile"),
        (4, 96, 8, "a head_dim with no record"),
        // The claim takes 64 and 512; the convert has no record for them.
        (4, 64, 8, "a head_dim the convert has no record for"),
        (4, 512, 8, "a head_dim the convert has no record for"),
        (4, 1024, 8, "a head_dim past the widest record"),
        (4, 128, 0, "a table of no jobs"),
        (4, 128, 65_536, "more jobs than one convert launch"),
    ] {
        assert!(GapDevice::new(&cuda, nkh, hd, jobs).is_err(), "{what}");
    }
    for (nkh, hd) in [(1, 128), (64, 128), (8, 256)] {
        GapDevice::new(&cuda, nkh, hd, 4).unwrap_or_else(|e| panic!("{nkh} at {hd}: {e}"));
    }
}

/// The claim is a prefix count, not a race: the same table against the same
/// lease takes the same slots, gap after gap.
#[test]
fn the_same_table_claims_the_same_slots_every_time() {
    let policy = CompressionPolicy::new(10);
    let Some(f) = fixture(&policy) else {
        return;
    };
    let groups = [
        GapGroup {
            policy: &policy,
            jobs: 0..1,
        },
        GapGroup {
            policy: &policy,
            jobs: 1..f.jobs.len(),
        },
    ];
    let first = run_gap(&f.cuda, &f.device, &f.backings, &f.jobs, &groups, &f.image);
    for _ in 0..3 {
        let again = run_gap(&f.cuda, &f.device, &f.backings, &f.jobs, &groups, &f.image);
        assert_eq!(again, first);
    }
}

/// A gap over a single job of a single layer.
#[test]
fn a_gap_of_one_job_seals_it_as_today() {
    let policy = CompressionPolicy::new(5);
    let Some(f) = fixture(&policy) else {
        return;
    };
    let one = &f.jobs[..1];
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..1,
    }];
    let (rows, deferred, errored) =
        run_gap(&f.cuda, &f.device, &f.backings, one, &groups, &f.image);
    assert!(!errored);
    assert_eq!(deferred, vec![false]);
    assert_eq!(rows.len(), N_KV_HEAD);
    // Its bands are the only claimants, so each index is below its band count.
    for r in &rows {
        assert!(r
            .lease_idx
            .iter()
            .all(|&i| (i as usize) < 2 * N_KV_HEAD * N_PALETTE));
    }
}

/// A gap over jobs naming no layer it was given is refused, not a panic.
#[test]
fn a_gap_over_no_layers_is_refused() {
    let policy = CompressionPolicy::new(5);
    let Some(f) = fixture(&policy) else {
        return;
    };
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..f.jobs.len(),
    }];
    let mut gap = GapDevice::new(&f.cuda, N_KV_HEAD, HEAD_DIM, f.jobs.len()).unwrap();
    assert!(gap.launch(&[], &f.jobs, &groups, &f.image, 1).is_err());
    assert!(gap.poll());
}

/// Dropping the gap's device state with a gap still in flight waits it out
/// before the summary is freed: the stream drains clean afterwards.
#[test]
fn dropping_the_gap_mid_flight_waits_for_it() {
    let policy = CompressionPolicy::new(5);
    let Some(f) = fixture(&policy) else {
        return;
    };
    let refs: Vec<&ChunkedKvBacking> = f.backings.iter().collect();
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..f.jobs.len(),
    }];
    for seq in 0..4 {
        let mut gap = GapDevice::new(&f.cuda, N_KV_HEAD, HEAD_DIM, f.jobs.len()).unwrap();
        gap.launch(&refs, &f.jobs, &groups, &f.image, seq).unwrap();
        drop(gap);
    }
    f.device.synchronize().unwrap();
}

/// Completion words repeat once `seq` wraps `u32`: the gap after the wrap
/// stores the same word as the gap before it, and is not seen complete until
/// it has run, because arming clears the word.
#[test]
fn a_gap_after_the_seq_wrap_is_not_seen_complete_early() {
    let policy = CompressionPolicy::new(5);
    let Some(f) = fixture(&policy) else {
        return;
    };
    let (before, after) = (u32::MAX as u64, 1u64 << 32);
    assert_eq!(completion_word(before), completion_word(after));
    let refs: Vec<&ChunkedKvBacking> = f.backings.iter().collect();
    let groups = [GapGroup {
        policy: &policy,
        jobs: 0..f.jobs.len(),
    }];
    let mut gap = GapDevice::new(&f.cuda, N_KV_HEAD, HEAD_DIM, f.jobs.len()).unwrap();
    gap.launch(&refs, &f.jobs, &groups, &f.image, before)
        .unwrap();
    f.device.synchronize().unwrap();
    assert!(gap.poll());
    assert!(gap.summary().complete(completion_word(after)));
    gap.summary().arm();
    assert!(
        !gap.summary().complete(completion_word(after)),
        "an armed summary reads as no gap complete"
    );
    gap.launch(&refs, &f.jobs, &groups, &f.image, after)
        .unwrap();
    f.device.synchronize().unwrap();
    assert!(gap.poll());
}
