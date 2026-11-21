//! The live expert launch, where workers compute remote experts that are not yet
//! in VRAM.
//!
//! A third of the experts are in VRAM, a third pinned, and a third cold. The cold
//! experts are published from another thread once the stream reaches the launch,
//! as the stager publishes them, so the workers' wait is exercised rather than
//! skipped. Every launch must match the all-VRAM output bit for bit at each token
//! tile width. A short promotion ring splits the remote experts between the slot
//! path and the scratch path. The gate-launch test checks that read-ahead items
//! land byte for byte in their slots and are counted once.

use super::*;

/// A live expert table for the worker tests: `n_experts` KO experts with a VRAM
/// copy each and a pinned host copy each (`cuMemAllocHost`, one block), and a
/// mapped `[3][n_experts]` table whose entries the test sets by class. The test
/// drives one projection, so the up and down planes hold the gate's address —
/// what bucketize classifies on, never read by the launch.
pub(super) struct LiveFixture {
    pub(super) vram_ptrs: Vec<u64>,
    pub(super) host: *mut c_void,
    pub(super) host_stride: usize,
    pub(super) host_bytes: usize,
    pub(super) table_h: *mut u64,
    pub(super) table_d: u64,
    pub(super) abort_h: *mut u32,
    pub(super) abort_d: u64,
    /// The launch's fault word (`MoeLive::fault`), in the abort word's
    /// allocation, 8 bytes in.
    pub(super) fault_d: u64,
    /// One 8-row KO chunk, in bytes.
    pub(super) chunk_bytes: usize,
    /// One expert's projection, in bytes.
    pub(super) expert_bytes: usize,
    pub(super) _storages: Vec<QCudaStorage>,
}

impl LiveFixture {
    pub(super) fn new(
        dev: &CudaDevice,
        n_experts: usize,
        nrows: usize,
        ncols: usize,
        seed: u64,
    ) -> Result<Self> {
        use cudarc::driver::sys;
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let shape = crate::Shape::from((nrows, ncols));
        let mut storages = Vec::with_capacity(n_experts);
        for _ in 0..n_experts {
            let w: Vec<f32> = (0..nrows * ncols)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let mut q = QCudaStorage::zeros(dev, ncols * nrows, GgmlDType::Q6_K)?;
            q.quantize(&CudaStorage::wrap_cuda_slice(
                dev.memcpy_stod(&w)?,
                dev.clone(),
            ))?;
            storages.push(q.repack_ko(&shape, GgmlDType::Q6_KO)?);
        }
        let expert_bytes = storages[0].storage_size_in_bytes();
        let host_stride = expert_bytes.div_ceil(256) * 256;
        let host_bytes = n_experts * host_stride;
        let mut host: *mut c_void = std::ptr::null_mut();
        assert_eq!(
            unsafe { sys::cuMemAllocHost_v2(&mut host, host_bytes) },
            sys::CUresult::CUDA_SUCCESS,
            "pinned expert copies"
        );
        let vram_ptrs: Vec<u64> = storages.iter().map(|s| s.data_ptr()).collect();
        for (e, &p) in vram_ptrs.iter().enumerate() {
            let dst = host as u64 + (e * host_stride) as u64;
            assert_eq!(
                copy_to_host(dev, dst as *mut c_void, p, expert_bytes),
                sys::CUresult::CUDA_SUCCESS
            );
        }
        let mapped = |bytes: usize| -> (*mut c_void, u64) {
            let mut raw: *mut c_void = std::ptr::null_mut();
            let mut d: sys::CUdeviceptr = 0;
            unsafe {
                assert_eq!(
                    sys::cuMemHostAlloc(&mut raw, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP),
                    sys::CUresult::CUDA_SUCCESS
                );
                assert_eq!(
                    sys::cuMemHostGetDevicePointer_v2(&mut d, raw, 0),
                    sys::CUresult::CUDA_SUCCESS
                );
                std::ptr::write_bytes(raw as *mut u8, 0, bytes);
            }
            (raw, d)
        };
        let (table_h, table_d) = mapped(3 * n_experts * 8);
        // The abort word, the fault word 8 bytes in, and the reached word 16 in.
        let (abort_h, abort_d) = mapped(24);
        let chunk_bytes = expert_bytes / ((nrows / 8) * (ncols / 128));
        Ok(Self {
            vram_ptrs,
            host,
            host_stride,
            host_bytes,
            table_h: table_h as *mut u64,
            table_d,
            abort_h: abort_h as *mut u32,
            abort_d,
            fault_d: abort_d + 8,
            chunk_bytes,
            expert_bytes,
            _storages: storages,
        })
    }

    pub(super) fn host_ptr(&self, e: usize) -> u64 {
        self.host as u64 + (e * self.host_stride) as u64
    }

    /// The fault word as the launches left it — 0 when none gave an expert up.
    pub(super) fn fault(&self) -> u64 {
        unsafe { std::ptr::read_volatile((self.abort_h as *const u8).add(8) as *const u64) }
    }

    /// Point expert `e`'s entries at `class`'s copy: VRAM, pinned, or 0 (cold).
    /// Up and down first, gate last, as the host publishes.
    pub(super) fn set(&self, e: usize, class: u8) {
        let v = match class {
            CLS_VRAM => self.vram_ptrs[e],
            CLS_PINNED => self.host_ptr(e),
            _ => 0,
        };
        // SAFETY: the fixture's mapped table, `[3][n_experts]`, lives as long as it.
        unsafe { publish_entry(self.table_h, self.vram_ptrs.len(), e, v) }
    }

    /// Queue a store of 1 to the reached word on `stream`: it lands when the
    /// stream reaches this point, which the publisher waits for.
    pub(super) fn mark_reached(&self, stream: &CudaStream) {
        use cudarc::driver::sys;
        let reached_d = self.abort_d + 16;
        // SAFETY: the reached word is in the fixture's mapped allocation.
        let r = unsafe { sys::cuMemsetD32Async(reached_d, 1, 1, stream.cu_stream()) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "mark the launch reached");
    }

    /// A publisher for the cold experts in `class`: their pinned copies, written
    /// from another thread `delay` after the stream reaches the launch
    /// ([`Self::mark_reached`]), as the stager does.
    ///
    /// **Not from the launching thread.** The stager makes no driver call, and a
    /// live launch depends on that: its workers spin until the host publishes,
    /// and a driver call on the publishing thread can wait behind another
    /// thread's call that is itself waiting for the device to go idle — a module
    /// load is one. Published from the launching thread after its next launch,
    /// the parallel suite held that launch for the whole spin limit, the workers
    /// gave the cold experts up, and the output was missing their rows.
    ///
    /// **Timed from the stream, not from the call.** With other tests' kernels on
    /// the card, the launch can start after a delay counted from the host's
    /// enqueue has already run out; the cold experts are then published before
    /// any worker waits, and the launch never exercises the wait it is testing.
    /// The publisher polls the reached word — a plain read of mapped memory,
    /// still no driver call — and counts the delay from there.
    pub(super) fn publish_cold_later(
        &self,
        class: &[u8],
        delay: std::time::Duration,
    ) -> std::thread::JoinHandle<()> {
        let n = self.vram_ptrs.len();
        let cold: Vec<(usize, u64)> = class
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == CLS_COLD)
            .map(|(e, _)| (e, self.host_ptr(e)))
            .collect();
        let table = self.table_h as usize;
        let reached = self.abort_h as usize + 16;
        std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            // SAFETY: the caller joins this thread before the fixture drops.
            while unsafe { std::ptr::read_volatile(reached as *const u32) } == 0 {
                assert!(
                    t0.elapsed() < std::time::Duration::from_secs(60),
                    "the stream never reached the live launch"
                );
                std::thread::yield_now();
            }
            std::thread::sleep(delay);
            for (e, v) in cold {
                // SAFETY: as above.
                unsafe { publish_entry(table as *mut u64, n, e, v) }
            }
        })
    }
}

/// Point expert `e`'s three entries in a mapped `[3][n]` table at `v`: up and
/// down first, gate last, as the host publishes. Plain stores to mapped memory —
/// no driver call.
///
/// # Safety
/// `table` must address a live, mapped table of `3 * n` entries, `e < n`.
pub(super) unsafe fn publish_entry(table: *mut u64, n: usize, e: usize, v: u64) {
    std::ptr::write_volatile(table.add(n + e), v);
    std::ptr::write_volatile(table.add(2 * n + e), v);
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    std::ptr::write_volatile(table.add(e), v);
}

impl Drop for LiveFixture {
    fn drop(&mut self) {
        use cudarc::driver::sys;
        unsafe {
            sys::cuMemFreeHost(self.host);
            sys::cuMemFreeHost(self.table_h as *mut c_void);
            sys::cuMemFreeHost(self.abort_h as *mut c_void);
        }
    }
}

/// **The live launch: workers compute the remote experts, bit for bit.**
///
/// A third of the experts are in VRAM, a third pinned, a third cold. The
/// pinned and cold ones are computed by the launch's worker blocks — each copies
/// a row tile's slice into the expert's promotion slot and runs the unmodified
/// impl over that row tile there (or, without a slot, into its VRAM scratch as
/// a 32-row matrix) — and the cold ones are released only by host
/// stores made 50 ms after the launch, with work already queued behind it. The
/// output must equal the all-VRAM launch's bit for bit, and the worker counters
/// must count every remote item once.
#[test]
pub(super) fn cuda_live_launch_workers_compute_remote_experts_bit_identically() -> Result<()> {
    live_launch_bit_identical(2, Slotted::All)
}

/// The same at the prefill tile modes: 64-wide tiles (`n_sub` 4)…
#[test]
pub(super) fn cuda_live_launch_at_bm64_computes_remote_experts_bit_identically() -> Result<()> {
    live_launch_bit_identical(4, Slotted::All)
}

/// …and 128-wide (`n_sub` 8), where a remote expert's workers run the wide kernel's
/// sub-tile sweep over the projection they copied.
#[test]
pub(super) fn cuda_live_launch_at_bm128_computes_remote_experts_bit_identically() -> Result<()> {
    live_launch_bit_identical(8, Slotted::All)
}

/// **A ring that runs short mid-list**: the first half of the remote experts
/// get a promotion slot and are computed from it, the rest from the workers'
/// scratch — both paths in one launch, still bit for bit.
#[test]
pub(super) fn cuda_live_launch_with_a_short_ring_computes_both_paths_bit_identically() -> Result<()>
{
    live_launch_bit_identical(2, Slotted::Half)
}

/// The same at the 128-wide tile mode.
#[test]
pub(super) fn cuda_live_launch_at_bm128_with_a_short_ring_is_bit_identical() -> Result<()> {
    live_launch_bit_identical(8, Slotted::Half)
}

/// The token-tile modes the host tile builder and every device-table launch share,
/// at each band's edges: affine KO formats run mode-4 from 32 rows/expert, MXFP4 mode-4
/// below 64 and mode-8 from there, a table shared with a non-KO format the decode width.
#[test]
pub(super) fn grouped_int8_tile_mode_bands() {
    let rows = [0, 31, 32, 63, 64, 127, 128, 300];
    assert_eq!(
        rows.map(|r| grouped_int8_n_sub(r, &[GgmlDType::Q4_KO])),
        [2, 2, 4, 4, 4, 4, 4, 4],
    );
    assert_eq!(
        rows.map(|r| grouped_int8_n_sub(r, &[GgmlDType::Q8_KO, GgmlDType::Q6_KO])),
        [2, 2, 4, 4, 4, 4, 4, 4],
    );
    assert_eq!(
        rows.map(|r| grouped_int8_n_sub(r, &[GgmlDType::MXFP4_KO])),
        [2, 2, 4, 4, 8, 8, 8, 8],
    );
    assert_eq!(
        rows.map(|r| grouped_int8_n_sub(r, &[GgmlDType::Q4_KO, GgmlDType::MXFP4_KO])),
        [2, 2, 4, 4, 8, 8, 8, 8],
    );
    assert_eq!(grouped_int8_n_sub(300, &[GgmlDType::Q4_K]), 2);
    assert_eq!(
        grouped_int8_n_sub(300, &[GgmlDType::Q4_KO, GgmlDType::Q4_K]),
        2
    );
    assert_eq!(grouped_int8_n_sub(300, &[]), 2);
}

/// One live launch at token-tile mode `n_sub` (tiles `16·n_sub` wide) against the
/// all-VRAM launch at the same mode.
///
/// **One at a time.** A live launch holds the card's SMs while its workers wait
/// for the host, which is the condition this measures: the cold wait must span
/// the publisher's delay. A second live launch on the card — never a state
/// production reaches, with one wave engine per device — keeps the first's
/// blocks off the SMs for its own wait, so the first starts after its experts
/// are already published and the wait it is testing never happens.
pub(super) fn live_launch_bit_identical(n_sub: usize, slotted: Slotted) -> Result<()> {
    use cudarc::driver::sys;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::sync::Mutex;

    static ONE_LIVE_LAUNCH: Mutex<()> = Mutex::new(());
    let _one = ONE_LIVE_LAUNCH.lock().unwrap_or_else(|p| p.into_inner());

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (n_experts, nrows, ncols) = (32usize, 256usize, 1024usize);
    let fx = LiveFixture::new(&dev, n_experts, nrows, ncols, 0x5e1f_0001)?;
    let full_table = dev.memcpy_stod(&fx.vram_ptrs)?;
    let mut rng = StdRng::seed_from_u64(0x5e1f_0002);

    // ~1.25 tiles' worth of rows an expert: most take two token tiles
    // (prefill-like), some one.
    let tile_w = 16 * n_sub;
    let (n_tokens, k) = (80 * n_sub, 8usize);
    let a_ub = n_tokens * k;
    let ids: Vec<u32> = (0..a_ub)
        .map(|_| rng.random_range(0..n_experts as u32))
        .collect();
    let class: Vec<u8> = (0..n_experts)
        .map(|e| [CLS_PINNED, CLS_COLD, CLS_VRAM][e % 3])
        .collect();
    let reference = bucketize_ref_live(&ids, n_tokens, k, n_experts, tile_w, &class, n_tokens);
    let total_valid = reference.header[1] as usize;
    let n_remote = reference.header[3] as u64;
    assert!(
        reference.remote.iter().any(|r| r[3] == 1) && reference.remote.iter().any(|r| r[2] > 1),
        "the fixture must route a cold expert and a multi-tile remote expert"
    );

    let t = crate::Tensor::from_vec(ids.clone(), (n_tokens, k), &device)?;
    let act: Vec<f32> = (0..a_ub * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let op = quantize_acts_q8a128_test(&dev, &act, a_ub, ncols)?;
    let stream = dev.cuda_stream();

    // Reference: every expert in VRAM, no workers.
    let mut ws_ref = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
    moe_bucketize(
        &t,
        n_experts,
        tile_w,
        &mut ws_ref,
        None,
        &DecodeRows::prefix(n_tokens),
    )?;
    let want = grouped_qmatmul_dev_q8a128(
        &op,
        &full_table,
        0,
        n_experts,
        GgmlDType::Q6_KO,
        nrows,
        &ws_ref.tile_expert,
        &ws_ref.tile_b_start,
        &ws_ref.tile_b_cnt,
        a_ub,
        n_sub,
        None,
        &dev,
    )?;
    let want = read_f32_tensor(&dev, &want)?;

    // The live launch.
    for (e, &c) in class.iter().enumerate() {
        fx.set(e, c);
    }
    // The launch's weight table: bucketize's snapshot, `[3][n_experts]`.
    let snap = dev.memcpy_stod(&vec![0u64; 3 * n_experts])?;
    let summary = dev.memcpy_stod(&vec![0u32; n_experts + 1])?;
    let remote = dev.memcpy_stod(&vec![0i32; n_experts * 4])?;
    let counters = dev.memcpy_stod(&[0i32; 3])?;
    let stall = dev.memcpy_stod(&[0u64; STALL_WORDS])?;
    // More workers than the projection has row tiles (256 rows = 8): the workers
    // take two of the grid's rows, ahead of the tile rows.
    const W: usize = 16;
    let slot_bytes = (ncols / 128) * 4 * fx.chunk_bytes;
    let scratch = unsafe { dev.alloc::<u8>(W * slot_bytes)? };
    let mut ws = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
    let tail = crate::Tensor::zeros((256, 256), crate::DType::F32, &device)?;
    drop((&tail + 0.0)?); // the successor's kernel, loaded before the launch
    let (sp, _g1) = summary.device_ptr(&stream);
    let (rp, _g2) = remote.device_ptr(&stream);
    let (cp, _g3) = counters.device_ptr(&stream);
    let (stp, _g4) = stall.device_ptr(&stream);
    let (scp, _g5) = scratch.device_ptr(&stream);
    let (hp, _g6) = ws.header.device_ptr(&stream);
    let (np, _g7) = snap.device_ptr(&stream);
    // Promotion slots for the first `n_slotted` remote experts: the workers copy
    // each slice there and compute from it, so each must end up holding its
    // expert whole. The rest are copied into scratch.
    let n_slotted = match slotted {
        Slotted::All => n_remote,
        Slotted::Half => n_remote / 2,
    };
    assert!(
        n_slotted > 0,
        "the fixture must slot at least one remote expert"
    );
    let promo_slots = dev.memcpy_stod(&vec![0u8; n_remote as usize * fx.expert_bytes])?;
    let remote_dst = dev.memcpy_stod(&vec![0u64; n_experts])?;
    let (pp, _g8) = promo_slots.device_ptr(&stream);
    let (dp, _g9) = remote_dst.device_ptr(&stream);
    let ring = TestRing::new(n_experts, 4, n_experts);
    ring.push(
        &(0..n_slotted)
            .map(|i| pp + i * fx.expert_bytes as u64)
            .collect::<Vec<u64>>(),
    );
    let pinned = [
        (fx.host as u64, fx.host as u64 + fx.host_bytes as u64),
        (0, 0),
    ];
    let blive = BucketizeLive {
        gate_row: fx.table_d,
        table_plane: n_experts as i64,
        snap: np,
        pinned,
        summary: sp,
        summary_seq: 1,
        remote: rp,
        counters: cp,
        row: 3,
        promo: Some(ring.ring()),
        remote_dst: dp,
        started_rows: 0,
        ticket: 0,
        owner: OwnerCheck::default(),
        ahead: None,
        cpu: None,
    };
    drop((_g1, _g2, _g3, _g4, _g5, _g6, _g7, _g8, _g9));
    moe_bucketize(
        &t,
        n_experts,
        tile_w,
        &mut ws,
        Some(&blive),
        &DecodeRows::prefix(n_tokens),
    )?;
    let table = &snap; // the gate plane: expert_base 0
    let (cold_wait_h, cold_wait_d) = mapped_host(16);
    let live = MoeLive {
        abort: fx.abort_d,
        fault: fx.fault_d,
        live_row: fx.table_d,
        remote_dst: dp,
        dst_offset: 0,
        remote: rp,
        header: hp,
        counter: cp,
        scratch: scp,
        slot_bytes: slot_bytes as u64,
        stall: stp,
        ahead: 0,
        ahead_done: 0,
        spin_limit_ns: 10_000_000_000,
        cpu_done: 0,
        cpu_seq: 0,
        cpu_rows: 0,
        cpu_dst_row: 0,
        cpu_alive: 0,
        cpu_tight_ns: 0,
        cpu_wait: 0,
        cold_wait: cold_wait_d,
        workers: W as i32,
        row: 3,
    };
    // The cold experts published late from the stager's side, counted from the
    // stream reaching the launch, and work queued behind the launch meanwhile.
    let delay = std::time::Duration::from_millis(50);
    fx.mark_reached(&stream);
    let publisher = fx.publish_cold_later(&class, delay);
    let got = grouped_qmatmul_dev_q8a128(
        &op,
        table,
        0,
        n_experts,
        GgmlDType::Q6_KO,
        nrows,
        &ws.tile_expert,
        &ws.tile_b_start,
        &ws.tile_b_cnt,
        a_ub,
        n_sub,
        Some(&live),
        &dev,
    )?;
    let tail = (&tail + 1.0)?;
    unsafe {
        let _ = sys::cuStreamQuery(stream.cu_stream());
    }
    if let Err(panic) = publisher.join() {
        std::panic::resume_unwind(panic);
    }

    let got = read_f32_tensor(&dev, &got)?;
    let n_cmp = total_valid * nrows;
    let bad: Vec<usize> = (0..n_cmp)
        .filter(|&i| got[i].to_bits() != want[i].to_bits())
        .collect();
    if let Some(&first) = bad.first() {
        // Which experts' rows differ, by class: rows are grouped by ascending
        // expert id.
        let mut counts = vec![0usize; n_experts];
        for &e in &ids {
            counts[e as usize] += 1;
        }
        let expert_of = |row: usize| {
            let mut end = 0;
            (0..n_experts)
                .find(|&e| {
                    end += counts[e];
                    row < end
                })
                .unwrap_or(n_experts)
        };
        let mut by_expert = std::collections::BTreeMap::<usize, usize>::new();
        for &i in &bad {
            *by_expert.entry(expert_of(i / nrows)).or_default() += 1;
        }
        let worst = bad
            .iter()
            .map(|&i| (got[i] - want[i]).abs())
            .fold(0f32, f32::max);
        let named: Vec<String> = by_expert
            .iter()
            .map(|(&e, &n)| {
                format!(
                    "{e} ({}): {n}",
                    ["vram", "pinned", "cold"][class[e] as usize]
                )
            })
            .collect();
        panic!(
            "the live launch must equal the all-VRAM launch: {} of {n_cmp} values differ, first at \
             row {} col {}, max |Δ| {worst}; by expert: {}",
            bad.len(),
            first / nrows,
            first % nrows,
            named.join(", ")
        );
    }
    assert_eq!(
        tail.flatten_all()?.get(0)?.to_scalar::<f32>()?,
        1.0,
        "the queued work ran"
    );
    assert_eq!(fx.fault(), 0, "every cold expert arrived: no fault");
    let c = dev.memcpy_dtov(&stall)?;
    let items = n_remote * (nrows as u64 / 32);
    assert_eq!(c[1], items, "every remote item once");
    assert_eq!(c[2], items * slot_bytes as u64, "bytes copied");
    assert_eq!(c[4], 1, "launches");
    assert!(
        c[0] >= delay.as_nanos() as u64 / 2,
        "cold wait {} ns for a {delay:?} delay",
        c[0]
    );
    // One sample per cold expert — its first row tile's wait. The experts whose
    // first tiles were in flight when the publish came waited about the delay;
    // the ones claimed after it waited nothing, so the sum is at least one
    // delay and at most one per worker.
    let n_cold = reference.remote.iter().filter(|r| r[3] == 1).count() as u64;
    let (cold_ns, cold_n) = unsafe {
        (
            std::ptr::read_volatile(cold_wait_h as *const u64),
            std::ptr::read_volatile((cold_wait_h as *const u64).add(1)),
        )
    };
    assert_eq!(cold_n, n_cold, "one cold-wait sample per cold expert");
    let d = delay.as_nanos() as u64;
    assert!(
        cold_ns >= d / 2 && cold_ns <= W as u64 * 2 * d,
        "cold waits {cold_ns} ns over {n_cold} experts for a {delay:?} delay"
    );
    unsafe { sys::cuMemFreeHost(cold_wait_h) };
    assert_eq!(
        dev.memcpy_dtov(&counters)?[0] as u64,
        items + W as u64,
        "counter: every item, then one past per worker"
    );

    // The first `n_slotted` remote experts in walk order — cold first, then
    // pinned (`promotion_walk`) — took a promotion slot, and the workers left
    // each projection there byte for byte.
    assert_eq!(
        ring.head() as u64,
        n_slotted,
        "one slot per slotted remote expert"
    );
    let promoted = dev.memcpy_dtov(&promo_slots)?;
    let n_pinned = reference.remote.iter().filter(|r| r[3] == 0).count();
    let walk = |i: usize| &reference.remote[(i + n_pinned) % reference.remote.len()];
    for i in 0..n_slotted as usize {
        let e = walk(i)[0] as usize;
        assert_eq!(
            ring.log(i),
            (1 << 32) | (3 << 16) | e as u64,
            "log entry {i}"
        );
        let mut want = vec![0u8; fx.expert_bytes];
        assert_eq!(
            copy_to_host(
                &dev,
                want.as_mut_ptr() as *mut c_void,
                fx.vram_ptrs[e],
                fx.expert_bytes
            ),
            sys::CUresult::CUDA_SUCCESS
        );
        assert!(
            promoted[i * fx.expert_bytes..(i + 1) * fx.expert_bytes] == want[..],
            "promotion slot {i} does not hold expert {e}"
        );
    }
    Ok(())
}

/// **The gate launch's workers read ahead after their own items.** Every expert
/// is in VRAM, so the launch has no items of its own and its workers go
/// straight to two read-ahead items: one image of 100,000 bytes and one of
/// 4,144 (fewer 16-byte units than pieces have, so its last pieces are empty).
/// Each lands in its slot byte for byte; each item's entries hold their values
/// and the first its owner tag — the second has none to write; each piece is
/// counted once; the profile counters see two items and their bytes; and the
/// GEMM's own output is still the all-VRAM launch's, bit for bit.
#[test]
pub(super) fn cuda_live_gate_launch_reads_ahead_and_publishes() -> Result<()> {
    use candle_kernels::simple::moe_bucketize::{AHEAD_CHUNKS, AHEAD_ITEM_WORDS, AHEAD_MAX};
    use cudarc::driver::sys;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (n_experts, nrows, ncols) = (8usize, 256usize, 1024usize);
    let fx = LiveFixture::new(&dev, n_experts, nrows, ncols, 0xa4ea_d001)?;
    for e in 0..n_experts {
        fx.set(e, CLS_VRAM);
    }
    let full_table = dev.memcpy_stod(&fx.vram_ptrs)?;
    let mut rng = StdRng::seed_from_u64(0xa4ea_d002);
    let n_sub = 2;
    let tile_w = 16 * n_sub;
    let (n_tokens, k) = (16usize, 2usize);
    let a_ub = n_tokens * k;
    let ids: Vec<u32> = (0..a_ub)
        .map(|_| rng.random_range(0..n_experts as u32))
        .collect();
    let t = crate::Tensor::from_vec(ids, (n_tokens, k), &device)?;
    let act: Vec<f32> = (0..a_ub * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let op = quantize_acts_q8a128_test(&dev, &act, a_ub, ncols)?;
    let stream = dev.cuda_stream();

    let mut ws_ref = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
    moe_bucketize(
        &t,
        n_experts,
        tile_w,
        &mut ws_ref,
        None,
        &DecodeRows::prefix(n_tokens),
    )?;
    let want = grouped_qmatmul_dev_q8a128(
        &op,
        &full_table,
        0,
        n_experts,
        GgmlDType::Q6_KO,
        nrows,
        &ws_ref.tile_expert,
        &ws_ref.tile_b_start,
        &ws_ref.tile_b_cnt,
        a_ub,
        n_sub,
        None,
        &dev,
    )?;
    let want = read_f32_tensor(&dev, &want)?;

    // The images, in pinned host memory: a byte pattern no two offsets share
    // within either image.
    let sizes = [100_000usize, 4_144];
    let total: usize = sizes.iter().sum();
    let mut src: *mut c_void = std::ptr::null_mut();
    assert_eq!(
        unsafe { sys::cuMemAllocHost_v2(&mut src, total) },
        sys::CUresult::CUDA_SUCCESS
    );
    let pattern: Vec<u8> = (0..total).map(|i| (i * 7 + i / 251) as u8).collect();
    unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), src as *mut u8, total) };
    let slots = [
        dev.memcpy_stod(&vec![0u8; sizes[0]])?,
        dev.memcpy_stod(&vec![0u8; sizes[1]])?,
    ];
    // The entries and the owner tag, mapped as the live table and tags are.
    let (entries_h, entries_d) = mapped_host(6 * 8);
    let (owner_h, owner_d) = mapped_host(4);
    let mut words = vec![0u64; 1 + AHEAD_MAX * AHEAD_ITEM_WORDS];
    words[0] = 2;
    let mut at = 0u64;
    for (i, &bytes) in sizes.iter().enumerate() {
        let item = &mut words[1 + i * AHEAD_ITEM_WORDS..1 + (i + 1) * AHEAD_ITEM_WORDS];
        let vals = [
            0x1111 * (3 * i as u64 + 1),
            0x1111 * (3 * i as u64 + 2),
            0x1111 * (3 * i as u64 + 3),
        ];
        item.copy_from_slice(&[
            src as u64 + at,
            slots[i].device_ptr(&stream).0,
            bytes as u64,
            entries_d + (24 * i) as u64,
            entries_d + (24 * i + 8) as u64,
            entries_d + (24 * i + 16) as u64,
            vals[0],
            vals[1],
            vals[2],
            if i == 0 { owner_d } else { 0 },
            0xabcd,
        ]);
        at += bytes as u64;
    }
    let items = dev.memcpy_stod(&words)?;
    let done = dev.memcpy_stod(&vec![0u32; AHEAD_MAX])?;

    // The live bucketize: no remote experts, so the launch's items are all
    // read-ahead pieces.
    let snap = dev.memcpy_stod(&vec![0u64; 3 * n_experts])?;
    let summary = dev.memcpy_stod(&vec![0u32; n_experts + 1])?;
    let remote = dev.memcpy_stod(&vec![0i32; n_experts * 4])?;
    let counters = dev.memcpy_stod(&[0i32; 3])?;
    let stall = dev.memcpy_stod(&[0u64; STALL_WORDS])?;
    const W: usize = 8;
    let slot_bytes = (ncols / 128) * 4 * fx.chunk_bytes;
    let scratch = unsafe { dev.alloc::<u8>(W * slot_bytes)? };
    let mut ws = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
    let blive = BucketizeLive {
        gate_row: fx.table_d,
        table_plane: n_experts as i64,
        snap: snap.device_ptr(&stream).0,
        pinned: [
            (fx.host as u64, fx.host as u64 + fx.host_bytes as u64),
            (0, 0),
        ],
        summary: summary.device_ptr(&stream).0,
        summary_seq: 1,
        remote: remote.device_ptr(&stream).0,
        counters: counters.device_ptr(&stream).0,
        row: 0,
        promo: None,
        remote_dst: 0,
        started_rows: 0,
        ticket: 0,
        owner: OwnerCheck::default(),
        ahead: None,
        cpu: None,
    };
    moe_bucketize(
        &t,
        n_experts,
        tile_w,
        &mut ws,
        Some(&blive),
        &DecodeRows::prefix(n_tokens),
    )?;
    let live = MoeLive {
        abort: fx.abort_d,
        fault: fx.fault_d,
        live_row: fx.table_d,
        remote_dst: 0,
        dst_offset: 0,
        remote: remote.device_ptr(&stream).0,
        header: ws.header.device_ptr(&stream).0,
        counter: counters.device_ptr(&stream).0,
        scratch: scratch.device_ptr(&stream).0,
        slot_bytes: slot_bytes as u64,
        stall: stall.device_ptr(&stream).0,
        ahead: items.device_ptr(&stream).0,
        ahead_done: done.device_ptr(&stream).0,
        spin_limit_ns: 10_000_000_000,
        cpu_done: 0,
        cpu_seq: 0,
        cpu_rows: 0,
        cpu_dst_row: 0,
        cpu_alive: 0,
        cpu_tight_ns: 0,
        cpu_wait: 0,
        cold_wait: 0,
        workers: W as i32,
        row: 0,
    };
    let got = grouped_qmatmul_dev_q8a128(
        &op,
        &snap,
        0,
        n_experts,
        GgmlDType::Q6_KO,
        nrows,
        &ws.tile_expert,
        &ws.tile_b_start,
        &ws.tile_b_cnt,
        a_ub,
        n_sub,
        Some(&live),
        &dev,
    )?;
    let got = read_f32_tensor(&dev, &got)?;
    let n_cmp = a_ub * nrows;
    assert_eq!(
        got[..n_cmp]
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<u32>>(),
        want[..n_cmp]
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<u32>>(),
        "reading ahead must not disturb the launch's own output"
    );
    let mut at = 0usize;
    for (i, &bytes) in sizes.iter().enumerate() {
        assert!(
            dev.memcpy_dtov(&slots[i])? == pattern[at..at + bytes],
            "slot {i} does not hold its image"
        );
        at += bytes;
    }
    let entries: Vec<u64> = (0..6)
        .map(|i| unsafe { std::ptr::read_volatile((entries_h as *const u64).add(i)) })
        .collect();
    assert_eq!(
        entries,
        vec![0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666]
    );
    assert_eq!(
        unsafe { std::ptr::read_volatile(owner_h as *const u32) },
        0xabcd
    );
    assert_eq!(
        dev.memcpy_dtov(&done.slice(..3))?,
        vec![AHEAD_CHUNKS as u32, AHEAD_CHUNKS as u32, 0]
    );
    let c = dev.memcpy_dtov(&stall)?;
    assert_eq!(
        (c[1], c[5], c[6]),
        (0, 2, total as u64),
        "no own items; two read ahead"
    );
    assert_eq!(
        dev.memcpy_dtov(&counters)?[0] as usize,
        2 * AHEAD_CHUNKS + W,
        "counter: every piece, then one past per worker"
    );
    unsafe {
        sys::cuMemFreeHost(src);
        sys::cuMemFreeHost(entries_h);
        sys::cuMemFreeHost(owner_h);
    }
    Ok(())
}
