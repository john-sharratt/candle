//! Device-launch tests for `moe_bucketize`, and the equivalence of the
//! GPU-native grouped matmul.
//!
//! Each bucketize output is checked bit for bit against the CPU sort, over
//! decode, skewed prefill and sentinel-strewn routings. The device-table grouped
//! matmul must produce the same bits as the host-orchestrated one, which closes
//! the equivalence for the GPU-native expert forward. The bench replays the
//! launch in a CUDA graph and reports device time per launch, and every
//! configuration passes its correctness gate before it is timed.

use super::*;

/// The live-layer buffers of a production bucketize: the live table, the
/// routing summary and the started words in mapped host memory, as
/// `expert_lre` allocates them, a promotion ring offering nothing, and the
/// device-side snapshot, remote list, counters and promotion slots. Every
/// expert's entries name VRAM, so the launch is the all-resident decode step:
/// no remote expert, no ring word read. Freed on drop.
pub(super) struct BenchLive {
    pub(super) table_h: *mut c_void,
    pub(super) summary_h: *mut c_void,
    pub(super) started_h: *mut c_void,
    pub(super) _ring: TestRing,
    pub(super) _device: Vec<CudaSlice<u64>>,
    pub(super) live: BucketizeLive,
}

impl BenchLive {
    pub(super) fn new(dev: &CudaDevice, n_experts: usize) -> Result<Self> {
        let table = test_live_table(&vec![CLS_VRAM; n_experts], &[]);
        let (table_h, table_d) = mapped_host(table.len() * 8);
        // SAFETY: the mapped allocation just made, `table.len()` u64s long.
        unsafe { std::ptr::copy_nonoverlapping(table.as_ptr(), table_h as *mut u64, table.len()) };
        let (summary_h, summary_d) = mapped_host((n_experts + 1) * 4);
        let (started_h, started_d) = mapped_host(8);
        let ring = TestRing::new(16, 1, n_experts);
        let stream = dev.cuda_stream();
        let device: Vec<CudaSlice<u64>> = vec![
            dev.memcpy_stod(&vec![0u64; 3 * n_experts])?,
            dev.memcpy_stod(&vec![0u64; 2 * n_experts])?,
            dev.memcpy_stod(&[0u64; 2])?,
            dev.memcpy_stod(&vec![0u64; n_experts])?,
        ];
        let ptr = |i: usize| device[i].device_ptr(&stream).0;
        let live = BucketizeLive {
            gate_row: table_d,
            table_plane: n_experts as i64,
            snap: ptr(0),
            pinned: TEST_PINNED,
            summary: summary_d,
            summary_seq: 5,
            remote: ptr(1),
            counters: ptr(2),
            row: 0,
            promo: Some(ring.ring()),
            remote_dst: ptr(3),
            started_rows: started_d,
            ticket: 9,
            owner: OwnerCheck::default(),
            ahead: None,
            cpu: None,
        };
        Ok(Self {
            table_h,
            summary_h,
            started_h,
            _ring: ring,
            _device: device,
            live,
        })
    }

    /// The routing summary as the host reads it, sequence word last.
    pub(super) fn summary(&self, n_experts: usize) -> Vec<u32> {
        (0..=n_experts)
            // SAFETY: inside the mapped summary, written by the device and drained.
            .map(|e| unsafe { std::ptr::read_volatile((self.summary_h as *const u32).add(e)) })
            .collect()
    }
}

impl Drop for BenchLive {
    fn drop(&mut self) {
        // SAFETY: allocated by `mapped_host`, each freed once.
        unsafe {
            cuMemFreeHost(self.table_h);
            cuMemFreeHost(self.summary_h);
            cuMemFreeHost(self.started_h);
        }
    }
}

/// A workspace's eleven device addresses in `run_moe_bucketize`'s argument
/// order (`tok_ids` … `scan`), read once: a graph capture has no place for the
/// sync event reading a slice's address records.
pub(super) fn bucketize_ws_ptrs(dev: &CudaDevice, ws: &MoeBucketizeWorkspace) -> [u64; 11] {
    let s = dev.cuda_stream();
    let ptrs = [
        ws.tok_ids.device_ptr(&s).0,
        ws.weight_ids.device_ptr(&s).0,
        ws.tile_expert.device_ptr(&s).0,
        ws.tile_b_start.device_ptr(&s).0,
        ws.tile_b_cnt.device_ptr(&s).0,
        ws.perm.device_ptr(&s).0,
        ws.rw_ids.device_ptr(&s).0,
        ws.token_starts.device_ptr(&s).0,
        ws.header.device_ptr(&s).0,
        ws.inv.device_ptr(&s).0,
        ws.scan.device_ptr(&s).0,
    ];
    ptrs
}

/// One `run_moe_bucketize` on `stream`, over a workspace's addresses
/// ([`bucketize_ws_ptrs`]) already sized for the launch — what the bench's
/// graph replays.
pub(super) fn launch_bucketize_raw(
    ids: u64,
    n_tokens: usize,
    k: usize,
    n_experts: usize,
    tile_w: usize,
    ws: &[u64; 11],
    live: Option<&BucketizeLive>,
    decode: &DecodeRows,
    stream: CUstream,
) -> Result<()> {
    let p = |b: u64| b as *mut c_void;
    let (lo, hi) = decode.ranges();
    let promo = live.and_then(|l| l.promo);
    let ahead = live.and_then(|l| l.ahead);
    let owner = live.map_or(OwnerCheck::default(), |l| l.owner);
    // SAFETY: every address is a live allocation sized for the launch, on this
    // device; the stream is the caller's.
    let status = unsafe {
        run_moe_bucketize(
            p(ids),
            n_tokens as i32,
            k as i32,
            n_experts as i32,
            tile_w as i32,
            p(ws[0]),
            p(ws[1]),
            p(ws[2]),
            p(ws[3]),
            p(ws[4]),
            p(ws[5]),
            p(ws[6]),
            p(ws[7]),
            p(ws[8]),
            p(ws[9]),
            p(ws[10]),
            p(live.map_or(0, |l| l.gate_row)),
            live.map_or(0, |l| l.table_plane),
            p(live.map_or(0, |l| l.snap)),
            live.map_or(0, |l| l.pinned[0].0),
            live.map_or(0, |l| l.pinned[0].1),
            live.map_or(0, |l| l.pinned[1].0),
            live.map_or(0, |l| l.pinned[1].1),
            lo.as_ptr(),
            hi.as_ptr(),
            lo.len() as i32,
            p(live.map_or(0, |l| l.summary)),
            live.map_or(0, |l| l.summary_seq),
            p(live.map_or(0, |l| l.remote)),
            p(live.map_or(0, |l| l.counters)),
            p(promo.map_or(0, |r| r.slots)),
            p(promo.map_or(0, |r| r.log)),
            p(promo.map_or(0, |r| r.head)),
            p(promo.map_or(0, |r| r.tail)),
            promo.map_or(0, |r| r.cap),
            p(promo.map_or(0, |r| r.marks)),
            p(promo.map_or(0, |r| r.reserve)),
            p(promo.map_or(0, |r| r.sweep)),
            p(promo.map_or(0, |r| r.victims)),
            p(promo.map_or(0, |r| r.retarget)),
            p(owner.owners),
            owner.zone_end,
            owner.slot_bytes,
            owner.slots,
            live.map_or(0, |l| l.row),
            p(live.map_or(0, |l| l.remote_dst)),
            p(live.map_or(0, |l| l.started_rows)),
            live.map_or(0, |l| l.ticket),
            p(promo.map_or(0, |r| r.window)),
            p(promo.map_or(0, |r| r.depth)),
            p(promo.map_or(0, |r| r.ahead_n)),
            p(promo.map_or(0, |r| r.ahead_list)),
            p(promo.map_or(0, |r| r.ahead_src)),
            promo.map_or(0, |r| r.ahead_cap),
            ahead.map_or(0, |a| a.rows),
            p(ahead.map_or(0, |a| a.row_layout)),
            p(ahead.map_or(0, |a| a.items)),
            p(ahead.map_or(0, |a| a.done)),
            ahead.map_or(0, |a| i32::from(a.predict)),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            stream as *mut c_void,
        )
    };
    if status != BUCKETIZE_LAUNCHED {
        crate::bail!("bench bucketize: launcher status {status}");
    }
    Ok(())
}

/// Micro-benchmark harness for `moe_bucketize`. For each size regime the
/// models hit — Qwen3 (k=8, 128 experts), Qwen3.8-Flash-Next (k=10, 512
/// experts: a draft step's one row, a verify's five, a 16-row wave, prefill) —
/// it FIRST asserts every output table is bit-identical to the CPU reference,
/// then measures DEVICE µs per launch: `per_graph` launches captured into a
/// CUDA graph and replayed between events, so no host launch cost is in the
/// figure. The `live` rows run the production path — live table, summary and
/// started words in mapped host memory, a promotion ring present, every expert
/// resident. `--ignored` (needs a GPU); run with `--nocapture`.
#[test]
#[ignore]
pub(super) fn bench_moe_bucketize() -> Result<()> {
    use super::super::graph_bench::time_graph;
    let dev = CudaDevice::new(0)?;
    let per_graph = 200usize;
    println!(
        "\n=== moe_bucketize micro-bench (tile_w={BENCH_TILE_W}, device µs, graph replay) ==="
    );
    println!(
        "{:<24} {:>8} {:>7} {:>12} {:>14}",
        "config", "a_ub", "n_exp", "us/call", "M-assign/s"
    );
    for &(n_tokens, k, n_experts, is_live, label) in BUCKETIZE_BENCH_CONFIGS {
        let case = BenchCase::new(&dev, n_tokens, k, n_experts, is_live, label)?;
        let items = vec![(); per_graph];
        let us = time_graph(&dev, &items, |(), cs| case.launch(cs))?;
        let a_ub = n_tokens * k;
        let massign = (a_ub as f64) / (us * 1e-6) / 1e6;
        println!("{label:<24} {a_ub:>8} {n_experts:>7} {us:>12.2} {massign:>14.1}");
    }
    Ok(())
}

/// The bench's configurations once each — exactly one launch per
/// configuration (its correctness gate), in [`BUCKETIZE_BENCH_CONFIGS`] order:
/// the target for an `ncu` pass, where launch `i` is configuration `i`.
/// `--ignored` (needs a GPU).
#[test]
#[ignore]
pub(super) fn profile_moe_bucketize_once() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    for &(n_tokens, k, n_experts, is_live, label) in BUCKETIZE_BENCH_CONFIGS {
        BenchCase::new(&dev, n_tokens, k, n_experts, is_live, label)?;
    }
    Ok(())
}

/// The bench's tile width.
pub(super) const BENCH_TILE_W: usize = 32;

/// `(n_tokens, k, n_experts, live, label)` for the bucketize bench.
pub(super) const BUCKETIZE_BENCH_CONFIGS: &[(usize, usize, usize, bool, &str)] = &[
    (1, 8, 128, false, "decode-1"),
    (1, 10, 512, true, "draft-1-e512-live"),
    (5, 10, 512, false, "verify-5-e512"),
    (5, 10, 512, true, "verify-5-e512-live"),
    (16, 10, 512, false, "decode-16-e512"),
    (16, 10, 512, true, "decode-16-e512-live"),
    (64, 8, 128, false, "decode-64"),
    (512, 8, 128, false, "prefill-512"),
    (2048, 8, 128, false, "prefill-2048"),
    (4096, 8, 128, false, "prefill-4096"),
    (8192, 8, 128, false, "prefill-8192"),
    (4096, 8, 256, false, "prefill-4096-e256"),
    // Qwen3.8-Flash-Next prefill: §0.4 rule 2 — the width the bound was
    // raised for AND the width it was raised from, so a regression for the
    // existing callers cannot hide.
    (713, 10, 512, false, "prefill-713-e512"),
    (713, 10, 512, true, "prefill-713-e512-live"),
    (2048, 10, 512, false, "prefill-2048-e512"),
    (4096, 10, 512, false, "prefill-4096-e512"),
    (8192, 10, 512, false, "prefill-8192-e512"),
    (4096, 10, 256, false, "prefill-4096-e256-k10"),
];

/// One bench configuration, set up and gated: deterministic dense routing,
/// a workspace, the live-layer buffers for a `live` row, and ONE launch whose
/// every table is asserted bit-exact against the CPU reference (and, live, the
/// summary too) before anything is timed.
pub(super) struct BenchCase {
    // The routing and the workspace the addresses below point into.
    pub(super) _t: crate::Tensor,
    pub(super) _ws: MoeBucketizeWorkspace,
    pub(super) bench_live: Option<BenchLive>,
    pub(super) decode: DecodeRows,
    pub(super) ids_ptr: u64,
    pub(super) ws_ptrs: [u64; 11],
    pub(super) n_tokens: usize,
    pub(super) k: usize,
    pub(super) n_experts: usize,
}

impl BenchCase {
    pub(super) fn new(
        dev: &CudaDevice,
        n_tokens: usize,
        k: usize,
        n_experts: usize,
        is_live: bool,
        label: &str,
    ) -> Result<Self> {
        let device = crate::Device::Cuda(dev.clone());
        let a_ub = n_tokens * k;
        let ids: Vec<u32> = (0..a_ub)
            .map(|i| ((i as u64).wrapping_mul(2654435761) % n_experts as u64) as u32)
            .collect();
        let t = crate::Tensor::from_vec(ids.clone(), (n_tokens, k), &device)?;
        let mut ws = MoeBucketizeWorkspace::new(dev, n_tokens, k)?;
        let bench_live = if is_live {
            Some(BenchLive::new(dev, n_experts)?)
        } else {
            None
        };
        let live = bench_live.as_ref().map(|b| b.live);
        let decode = DecodeRows::prefix(n_tokens);
        moe_bucketize(&t, n_experts, BENCH_TILE_W, &mut ws, live.as_ref(), &decode)?;
        assert_bucketize_ws(dev, &ws, &ids, n_tokens, k, n_experts, BENCH_TILE_W, label)?;
        if let Some(b) = &bench_live {
            let mut want = bucketize_ref(&ids, n_tokens, k, n_experts, BENCH_TILE_W).summary;
            want.push(5);
            assert_eq!(b.summary(n_experts), want, "{label}: summary");
        }
        let ids_ptr = {
            let stream = dev.cuda_stream();
            let (s, _) = t.storage_and_layout();
            let ptr = match &*s {
                crate::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream).0,
                _ => unreachable!(),
            };
            ptr
        };
        let ws_ptrs = bucketize_ws_ptrs(dev, &ws);
        Ok(Self {
            _t: t,
            _ws: ws,
            bench_live,
            decode,
            ids_ptr,
            ws_ptrs,
            n_tokens,
            k,
            n_experts,
        })
    }

    /// One launch on `stream`, with no host-side address reads.
    pub(super) fn launch(&self, stream: CUstream) -> Result<()> {
        launch_bucketize_raw(
            self.ids_ptr,
            self.n_tokens,
            self.k,
            self.n_experts,
            BENCH_TILE_W,
            &self.ws_ptrs,
            self.bench_live.as_ref().map(|b| &b.live),
            &self.decode,
            stream,
        )
    }
}

/// The GPU-native dispatch gate: `grouped_qmatmul_dev_q8a128` (full resident
/// pointer table + `moe_bucketize` device tile tables, upper-bound launch) must
/// produce BIT-IDENTICAL output rows to the host-orchestrated `grouped_qmatmul`
/// (active-compacted pointer array + host-built tile tables) on the same
/// operand and weights. Same tile decomposition, same per-tile pointer, same
/// kernel ⇒ same bits; this pins it. With the bucketize outputs proven
/// bit-identical to the CPU sort separately, and the gather/silu/scatter
/// kernels shared verbatim between paths, this closes the equivalence chain
/// for the whole GPU-native expert forward.
#[test]
pub(super) fn cuda_grouped_qmatmul_dev_matches_host_tables() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let n_experts = 32usize; // smaller expert pool keeps the fixture fast
    let nrows = 256usize;
    let ncols = 1024usize;
    let wdtype = GgmlDType::Q6_K; // production base quant
    let ko_dtype = GgmlDType::Q6_KO; // its KO twin (int8 grouped path)

    // Random per-expert weights, quantized then KO-repacked (the model path).
    let mut rng = StdRng::seed_from_u64(0x9e4a_11ce);
    let shape = crate::Shape::from((nrows, ncols));
    let mut full_ptrs: Vec<u64> = Vec::with_capacity(n_experts);
    let mut storages = Vec::with_capacity(n_experts);
    for _ in 0..n_experts {
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut q = QCudaStorage::zeros(&dev, ncols * nrows, wdtype)?;
        q.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&w)?,
            dev.clone(),
        ))?;
        let ko = q.repack_ko(&shape, ko_dtype)?;
        full_ptrs.push(ko.data_ptr());
        storages.push(ko);
    }
    let full_table = dev.memcpy_stod(&full_ptrs)?;

    for &(n_tokens, k, label) in &[
        (1usize, 8usize, "decode-1tok"),
        (7, 8, "mixed-small"),
        (64, 8, "prefill-64"),
        (200, 8, "prefill-200-multitile"),
    ] {
        let a_ub = n_tokens * k;
        // Random routing (dense; every slot valid — the sentinel path is pinned
        // by the bucketize tests and skipped rows never reach the GEMM).
        let ids: Vec<u32> = (0..a_ub)
            .map(|_| rng.random_range(0..n_experts as u32))
            .collect();
        let reference = bucketize_ref(&ids, n_tokens, k, n_experts, 32);
        let total_valid = reference.header[1] as usize;

        // GPU tables.
        let t = crate::Tensor::from_vec(ids.clone(), (n_tokens, k), &device)?;
        let mut ws = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
        moe_bucketize(
            &t,
            n_experts,
            32,
            &mut ws,
            None,
            &DecodeRows::prefix(n_tokens),
        )?;

        // One shared stacked activation covering the full launch bound.
        let act: Vec<f32> = (0..a_ub * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let op = quantize_acts_q8a128_test(&dev, &act, a_ub, ncols)?;

        // Host path: active-compacted pointers + per-active-expert offsets,
        // exactly as `forward_with_indices` builds them.
        let mut counts = vec![0i32; n_experts];
        for &e in &ids {
            counts[e as usize] += 1;
        }
        let mut host_ptrs: Vec<u64> = Vec::new();
        let mut host_offsets: Vec<i32> = vec![0];
        for e in 0..n_experts {
            if counts[e] > 0 {
                host_ptrs.push(full_ptrs[e]);
                host_offsets.push(host_offsets.last().unwrap() + counts[e]);
            }
        }
        let host_out = grouped_qmatmul(
            DynamicTensor::Int8(&op),
            &host_ptrs,
            ko_dtype,
            nrows,
            &host_offsets,
            &dev,
            Backing::Owned,
        )?;

        // Device path: full table, raw ids, upper-bound launch.
        let dev_out = grouped_qmatmul_dev_q8a128(
            &op,
            &full_table,
            0,
            n_experts,
            ko_dtype,
            nrows,
            &ws.tile_expert,
            &ws.tile_b_start,
            &ws.tile_b_cnt,
            a_ub,
            2,
            None,
            &dev,
        )?;

        let host_v = read_f32_tensor(&dev, &host_out)?;
        let dev_v = read_f32_tensor(&dev, &dev_out)?;
        let n_cmp = total_valid * nrows;
        assert_eq!(
            host_v[..n_cmp]
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<u32>>(),
            dev_v[..n_cmp]
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<u32>>(),
            "{label}: dev-table GEMM must be bit-identical to host-table GEMM"
        );
    }
    Ok(())
}

#[test]
pub(super) fn cuda_moe_bucketize_matches_cpu_reference() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());

    // Decode: 1 token × k=8 distinct experts (the hot interactive case).
    assert_bucketize_case(
        &device,
        vec![7, 3, 100, 42, 0, 127, 55, 9],
        1,
        8,
        128,
        32,
        "decode-1tok",
    )?;

    // Duplicate experts within a token (allowed by the contract even though
    // top-k emits distinct ids) and multiple tokens sharing experts.
    assert_bucketize_case(
        &device,
        vec![3, 3, 5, 5, 5, 0, 1, 2, 3, 5, 0, 0, 0, 1, 2, 7],
        2,
        8,
        8,
        32,
        "dup-experts",
    )?;

    // Router empty-slot sentinels (id == n_experts) and beyond must be skipped.
    assert_bucketize_case(
        &device,
        vec![1, 16, 2, 99, 3, 16, 16, 4],
        1,
        8,
        16,
        32,
        "sentinels",
    )?;

    // Everything routes to ONE expert → a deep bucket crossing many tile
    // boundaries (multi-tile single expert, non-divisible tail).
    assert_bucketize_case(
        &device,
        vec![5u32; 100 * 8],
        100,
        8,
        128,
        32,
        "one-expert-multitile",
    )?;

    // All slots sentinel (pathological: nothing valid anywhere).
    assert_bucketize_case(&device, vec![128u32; 4 * 8], 4, 8, 128, 32, "all-sentinel")?;

    // k = 1 and tile_w = 16 shape edges.
    assert_bucketize_case(&device, vec![2, 0, 2, 1, 2], 5, 1, 4, 16, "k1-tile16")?;

    // 512 experts — MORE experts than the kernel has threads, so every
    // per-expert phase must stride rather than assume one thread per expert.
    // The upper half is what a one-thread-per-expert form would have silently
    // dropped, so these cases put weight there: an expert at the very top, and
    // the sentinel at exactly `n_experts`.
    assert_bucketize_case(
        &device,
        vec![511, 256, 0, 511, 300, 511, 1, 480, 511, 257],
        1,
        10,
        512,
        32,
        "e512-top-half",
    )?;
    assert_bucketize_case(
        &device,
        vec![511u32; 64 * 10],
        64,
        10,
        512,
        32,
        "e512-one-expert",
    )?;
    assert_bucketize_case(
        &device,
        vec![512, 300, 512, 511, 0, 999, 512, 256, 7, 512],
        1,
        10,
        512,
        32,
        "e512-sentinels",
    )?;

    // The narrow kernel's edge: 256 assignments to one expert at one token per
    // tile — every lane of its packed scan at its largest total (256 rows, 256
    // tiles) — and one assignment either side of the bound, each kernel's.
    assert_bucketize_case(
        &device,
        vec![511u32; 256],
        128,
        2,
        512,
        1,
        "narrow-one-expert-256x1",
    )?;
    assert_bucketize_case(
        &device,
        (0..256u32).map(|i| (i * 7) % 3).collect(),
        256,
        1,
        3,
        1,
        "narrow-256-k1-three-experts",
    )?;
    assert_bucketize_case(
        &device,
        (0..257u32).map(|i| (i * 7) % 3).collect(),
        257,
        1,
        3,
        1,
        "general-257-k1-three-experts",
    )?;

    // A skewed bulk prefill: a third of every assignment to one expert, so a
    // warp's lanes share an expert more often than not and one bucket spans
    // every tile of the walk; the rest strided over the table, a sentinel in
    // every 97th slot.
    assert_bucketize_case(
        &device,
        (0..8192u32 * 10)
            .map(|i| match (i % 3, i % 97) {
                (_, 0) => 512,
                (0, _) => 5,
                _ => (i.wrapping_mul(2654435761) >> 7) % 512,
            })
            .collect(),
        8192,
        10,
        512,
        64,
        "e512-skewed-8192",
    )?;

    // Seeded fuzz across prefill-like shapes, with a sprinkle of sentinels.
    let mut rng = StdRng::seed_from_u64(0xb0cc_e71e);
    for &(n_tokens, k, n_experts, tile_w) in &[
        (4usize, 8usize, 128usize, 32usize),
        (64, 8, 128, 32),
        (333, 8, 128, 32),
        (1024, 8, 128, 32),
        (2048, 8, 128, 32),
        (7, 3, 16, 32),
        (129, 8, 128, 16),
        (17, 8, 128, 1),
        // Qwen3.8-Flash-Next's own width and top-k, across decode and prefill
        // shapes; 257 sits one past the old bound.
        (16, 10, 512, 32),
        (713, 10, 512, 32),
        (2048, 10, 512, 32),
        (64, 10, 257, 32),
        (33, 10, 512, 16),
        // Either side of the narrow kernel's bound, and a verify's width.
        (5, 10, 512, 32),
        (25, 10, 512, 32),
        (32, 8, 128, 32),
        (8, 32, 512, 32),
        (257, 1, 512, 1),
        (26, 10, 512, 32),
        // Bulk-prefill widths at the wide tile modes, where the scatter walks
        // the list in many block-wide tiles: an exact multiple of the tile, a
        // partial last tile, and the 8,192-row chunk.
        (4096, 10, 512, 128),
        (4097, 10, 512, 64),
        (8192, 10, 512, 64),
    ] {
        let ids: Vec<u32> = (0..n_tokens * k)
            .map(|_| {
                if rng.random_ratio(1, 50) {
                    n_experts as u32 // router sentinel
                } else {
                    rng.random_range(0..n_experts as u32)
                }
            })
            .collect();
        assert_bucketize_case(
            &device,
            ids,
            n_tokens,
            k,
            n_experts,
            tile_w,
            &format!("fuzz-{n_tokens}x{k}-e{n_experts}-w{tile_w}"),
        )?;
    }

    // Determinism: identical input twice through the SAME workspace must give
    // byte-identical buffers (no atomics, no ordering hazards).
    let n_tokens = 512usize;
    let k = 8usize;
    let ids: Vec<u32> = (0..n_tokens * k)
        .map(|_| rng.random_range(0..128))
        .collect();
    let t = crate::Tensor::from_vec(ids, (n_tokens, k), &device)?;
    let mut ws = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
    let decode = DecodeRows::prefix(n_tokens);
    moe_bucketize(&t, 128, 32, &mut ws, None, &decode)?;
    let first = (
        dev.memcpy_dtov(&ws.tok_ids.slice(..n_tokens * k))?,
        dev.memcpy_dtov(&ws.perm.slice(..n_tokens * k))?,
        dev.memcpy_dtov(&ws.tile_b_cnt.slice(..n_tokens * k))?,
    );
    moe_bucketize(&t, 128, 32, &mut ws, None, &decode)?;
    let second = (
        dev.memcpy_dtov(&ws.tok_ids.slice(..n_tokens * k))?,
        dev.memcpy_dtov(&ws.perm.slice(..n_tokens * k))?,
        dev.memcpy_dtov(&ws.tile_b_cnt.slice(..n_tokens * k))?,
    );
    assert_eq!(first, second, "repeat run must be byte-identical");

    Ok(())
}
