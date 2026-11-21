//! The CPU expert class: which misses the host pool takes, how they are
//! published, and the round trip of their results.
//!
//! The split chooses the cheapest misses for the pool under a cost model, and
//! bucketize publishes them without changing the row layout, so the grouped
//! GEMM's rows are the same with and without the CPU class. The round-trip tests
//! export the CPU rows into a mapped ring and ring the doorbell, then check that
//! each launch pulls its own invocation's results. A dead pool must fault
//! promptly rather than wait out the spin limit. The doorbell counter is keyed
//! by ticket, so a launch that did not count all its blocks out cannot stop the
//! next launch from ringing.

use super::*;

/// The CPU split's reference (`moe_bucketize_cpu.cuh`): which experts the host
/// pool takes, given each expert's routed count, class, and whether a
/// promotion of it is in flight — at most `s.max_experts` of them.
pub(super) fn cpu_split_ref(
    counts: &[u32],
    class: &[u8],
    marked: &[bool],
    s: &CpuSplit,
) -> Vec<bool> {
    let max_experts = s.max_experts as usize;
    let table = |t: &[f32], n: u32| -> f32 {
        if n as usize <= CPU_MAX_N {
            t[n as usize]
        } else {
            t[CPU_MAX_N] * n as f32 / CPU_MAX_N as f32
        }
    };
    let mut cpu = vec![false; counts.len()];
    if s.cap_rows == 0 {
        return cpu;
    }
    let max_n = s.max_n.min(CPU_MAX_N as u32);
    let misses: Vec<usize> = (0..counts.len())
        .filter(|&e| counts[e] > 0 && class[e] != CLS_VRAM)
        .collect();
    let base: f32 = misses
        .iter()
        .map(|&e| s.copy_ns + table(&s.gpu_n_ns, counts[e]))
        .sum();
    let mut order: Vec<usize> = (0..counts.len())
        .filter(|&e| {
            (class[e] == CLS_PINNED || (class[e] == CLS_COLD && !marked[e]))
                && counts[e] > 0
                && counts[e] <= max_n
        })
        .collect();
    order.sort_by_key(|&e| (counts[e], class[e] != CLS_COLD, e));
    let (mut best, mut best_k) = (base, 0usize);
    let (mut comp, mut gpu_left, mut rows) = (0f32, base, 0u32);
    let mut first = base;
    for (j, &e) in order.iter().enumerate() {
        if rows + counts[e] > s.cap_rows || j >= max_experts {
            break;
        }
        rows += counts[e];
        comp += table(&s.cpu_n_ns, counts[e]);
        gpu_left -= s.copy_ns + table(&s.gpu_n_ns, counts[e]);
        let kk = j + 1;
        let read = if kk < misses.len() {
            s.read_shared_ns
        } else {
            s.read_alone_ns
        };
        let mem = kk as f32 * read;
        let t_cpu = s.handoff_ns + mem.max(comp) + rows as f32 * s.row_ns;
        let tot = t_cpu.max(gpu_left.max(0.0));
        if kk == 1 {
            first = tot;
        }
        if tot < best {
            best = tot;
            best_k = kk;
        }
    }
    let mut take = if base - best >= s.min_gain_ns {
        best_k
    } else {
        0
    };
    if take == 0
        && !order.is_empty()
        && max_experts > 0
        && s.explore_ns > 0.0
        && first - base <= s.explore_ns
        && counts[order[0]] <= s.cap_rows
    {
        take = 1;
    }
    for &e in &order[..take] {
        cpu[e] = true;
    }
    cpu
}

/// A split whose costs make the CPU worth it for a few cheap experts: a copy
/// is 1,000 ns, the pool's read 100 and its compute 50 a token, the cut 3.
/// Integer costs, so every sum is exact in f32 whatever its order.
pub(super) fn test_cpu_split(min_gain_ns: f32) -> CpuSplit {
    let mut s = CpuSplit::OFF;
    s.cap_rows = 1_000;
    s.max_n = 3;
    s.copy_ns = 1_000.0;
    s.read_shared_ns = 100.0;
    s.read_alone_ns = 100.0;
    s.handoff_ns = 5.0;
    s.row_ns = 1.0;
    s.min_gain_ns = min_gain_ns;
    s.max_experts = u32::MAX;
    for n in 1..=CPU_MAX_N {
        s.cpu_n_ns[n] = 50.0 * n as f32;
        s.gpu_n_ns[n] = 10.0 * n as f32;
    }
    s
}

/// THE CPU CLASS. Over a live table, bucketize gives the host pool the misses
/// its split picks — pinned, or cold with no promotion of it in flight — the
/// cheapest first, ascending routed count — and for each:
/// - sets the summary's CPU bit, emits no tile, keeps it off the remote list;
/// - keeps its rows where they were: the row layout (gather lists, `perm`,
///   `token_starts`) is byte-identical to the launch without the CPU class;
/// - publishes it: the header's CPU rows and experts, the work slot's header
///   and entry (ring start, count, the snapshot's three addresses), and the
///   ring-row → grouped-row map.
///
/// Run through the narrow kernel (16 assignments) and the general one (400),
/// against the split's reference, with a gain bar no split clears, with that
/// bar on an exploring invocation, with the light cold expert's promotion
/// marked in flight, and with the expert cap at two, then at none.
#[test]
pub(super) fn cuda_moe_bucketize_cpu_class_publishes_cheap_misses() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let (k, n_experts, tile_w) = (2usize, 16usize, 4usize);
    // 0..4 VRAM, 4..12 pinned, 12..16 cold.
    let class: Vec<u8> = (0..n_experts)
        .map(|e| match e {
            0..=3 => CLS_VRAM,
            4..=11 => CLS_PINNED,
            _ => CLS_COLD,
        })
        .collect();
    let table = test_live_table(&class, &[]);
    // Row 3's e15 marked in flight: a promotion of it may yet publish a VRAM
    // address the pool could not read.
    let ring = TestRing::new(4, 4, n_experts);
    unsafe { std::ptr::write_volatile(ring.mark_ptr(3, 15), 9) };
    for n_tokens in [8usize, 200] {
        // Token t routes VRAM expert t % 4, and a second expert: the pinned
        // 4 + t % 8 for the first nine tokens — counts of 1 and 2, under the
        // cut — the cold 15 for the tenth, once, and then the cold 12 + t % 3,
        // heavily.
        let mut ids = Vec::with_capacity(n_tokens * k);
        for t in 0..n_tokens {
            let b = match t {
                0..=8 => 4 + t % 8,
                9 => 15,
                _ => 12 + t % 3,
            };
            ids.push((t % 4) as u32);
            ids.push(b as u32);
        }
        let mut counts = vec![0u32; n_experts];
        for &e in &ids {
            counts[e as usize] += 1;
        }
        // A split that takes something; a bar none clears; the same bar on
        // an exploring invocation, which takes the single cheapest expert;
        // and the first split with e15 marked.
        for (min_gain, explore, marked, cap) in [
            (0.0f32, 0.0f32, false, u32::MAX),
            (1e9, 0.0, false, u32::MAX),
            (1e9, 1e9, false, u32::MAX),
            (0.0, 0.0, true, u32::MAX),
            (0.0, 0.0, false, 2),
            (1e9, 1e9, false, 0),
        ] {
            let mut split = test_cpu_split(min_gain);
            split.explore_ns = explore;
            split.max_experts = cap;
            let marks: Vec<bool> = (0..n_experts).map(|e| marked && e == 15).collect();
            let want = cpu_split_ref(&counts, &class, &marks, &split);
            if cap == 2 {
                let uncapped = cpu_split_ref(&counts, &class, &marks, &test_cpu_split(min_gain));
                assert!(
                    uncapped.iter().filter(|&&c| c).count() > 2,
                    "{n_tokens} tokens: the cap binds — the uncapped split takes more"
                );
                assert_eq!(want.iter().filter(|&&c| c).count(), 2);
            }
            if cap == 0 {
                assert!(want.iter().all(|&c| !c), "a cap of none takes nothing");
            }
            if n_tokens == 200 && min_gain == 0.0 {
                assert_eq!(
                    want[15], !marked,
                    "the light cold expert is the pool's unless marked"
                );
            }
            let device = crate::Device::Cuda(dev.clone());
            let t = crate::Tensor::from_vec(ids.clone(), (n_tokens, k), &device)?;
            let table_dev = dev.memcpy_stod(&table)?;
            let snap_dev = dev.memcpy_stod(&vec![0u64; 3 * n_experts])?;
            let summary_dev = dev.memcpy_stod(&vec![0u32; n_experts + 1])?;
            let remote_dev = dev.memcpy_stod(&vec![-1i32; n_experts * 4])?;
            let counters_dev = dev.memcpy_stod(&[0i32; 3])?;
            let work_dev = dev.memcpy_stod(&vec![0u8; CPU_WORK_SLOT_BYTES])?;
            let dst_dev = dev.memcpy_stod(&vec![u32::MAX; n_tokens * k])?;
            // With the ring: its slots for the remote experts, of which it
            // holds none to give.
            let remote_dst_dev = dev.memcpy_stod(&vec![0u64; n_experts])?;
            let stream = dev.cuda_stream();
            let run = |ws: &mut MoeBucketizeWorkspace, with_cpu: bool| -> Result<()> {
                let (gp, _g0) = table_dev.device_ptr(&stream);
                let (sp, _g1) = summary_dev.device_ptr(&stream);
                let (rp, _g2) = remote_dev.device_ptr(&stream);
                let (cp, _g3) = counters_dev.device_ptr(&stream);
                let (np, _g4) = snap_dev.device_ptr(&stream);
                let (wp, _g5) = work_dev.device_ptr(&stream);
                let (dp, _g6) = dst_dev.device_ptr(&stream);
                let live = BucketizeLive {
                    gate_row: gp,
                    table_plane: n_experts as i64,
                    snap: np,
                    pinned: TEST_PINNED,
                    summary: sp,
                    summary_seq: 41,
                    remote: rp,
                    counters: cp,
                    row: 3,
                    promo: marked.then(|| ring.ring()),
                    remote_dst: if marked {
                        remote_dst_dev.device_ptr(&stream).0
                    } else {
                        0
                    },
                    started_rows: 0,
                    ticket: 0,
                    owner: OwnerCheck::default(),
                    ahead: None,
                    cpu: with_cpu.then_some(BucketizeCpu {
                        split,
                        work: wp,
                        dst_row: dp,
                        carry: 0,
                        tok_first: 0,
                        in_tok: 0,
                        input_map: 0,
                    }),
                };
                moe_bucketize(&t, n_experts, tile_w, ws, Some(&live), &DecodeRows::none())
            };
            let a_ub = n_tokens * k;
            let rows_of = |ws: &MoeBucketizeWorkspace| -> Result<Vec<Vec<u32>>> {
                Ok(vec![
                    dev.memcpy_dtov(&ws.tok_ids.slice(..a_ub))?,
                    dev.memcpy_dtov(&ws.weight_ids.slice(..a_ub))?,
                    dev.memcpy_dtov(&ws.perm.slice(..a_ub))?,
                    dev.memcpy_dtov(&ws.rw_ids.slice(..a_ub))?,
                    dev.memcpy_dtov(&ws.token_starts.slice(..n_tokens + 1))?
                        .iter()
                        .map(|&v| v as u32)
                        .collect(),
                ])
            };
            let mut ws_off = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
            run(&mut ws_off, false)?;
            let rows_off = rows_of(&ws_off)?;
            let mut ws = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
            run(&mut ws, true)?;
            let tag = format!(
                "n_tokens {n_tokens}, min_gain {min_gain}, explore {explore}, marked {marked}, \
                 cap {cap}"
            );

            // The row layout does not move.
            assert_eq!(rows_of(&ws)?, rows_off, "{tag}: row layout");

            // The summary's CPU bit is the reference's choice.
            let summary = dev.memcpy_dtov(&summary_dev)?;
            for e in 0..n_experts {
                assert_eq!(
                    summary[e] >> SUMMARY_CPU_BIT & 1 == 1,
                    want[e],
                    "{tag}: CPU bit of expert {e}"
                );
                assert_eq!(
                    summary[e] & SUMMARY_COUNT_MASK,
                    counts[e],
                    "{tag}: count {e}"
                );
            }
            let n_cpu = want.iter().filter(|&&c| c).count();
            if cap == 0 {
                assert_eq!(
                    n_cpu, 0,
                    "{tag}: a cap of none takes nothing, exploring or not"
                );
            } else if explore > 0.0 {
                assert_eq!(n_cpu, 1, "{tag}: exploring takes one expert");
            } else if min_gain > 0.0 {
                assert_eq!(n_cpu, 0, "{tag}: no split clears the bar");
            } else {
                assert!(n_cpu > 0, "{tag}: the split takes some expert");
            }

            // No tile and no remote entry for a CPU expert.
            let hd = dev.memcpy_dtov(&ws.header.slice(..HEADER_WORDS))?;
            let tiles = dev.memcpy_dtov(&ws.tile_expert.slice(..a_ub))?;
            let want_tiles: i32 = (0..n_experts)
                .filter(|&e| !want[e])
                .map(|e| counts[e].div_ceil(tile_w as u32) as i32)
                .sum();
            assert_eq!(hd[2], want_tiles, "{tag}: tiles");
            for &e in &tiles[..hd[2] as usize] {
                assert!(!want[e as usize], "{tag}: a tile of CPU expert {e}");
            }
            let remote = dev.memcpy_dtov(&remote_dev)?;
            for r in 0..hd[3] as usize {
                assert!(
                    !want[remote[4 * r] as usize],
                    "{tag}: CPU expert on the remote list"
                );
            }

            // The publication.
            let rows_cpu: u32 = (0..n_experts).filter(|&e| want[e]).map(|e| counts[e]).sum();
            assert_eq!(
                (hd[5], hd[6]),
                (rows_cpu as i32, n_cpu as i32),
                "{tag}: header"
            );
            let work = dev.memcpy_dtov(&work_dev)?;
            let word = |at: usize| u32::from_le_bytes(work[at..at + 4].try_into().unwrap());
            let dword = |at: usize| u64::from_le_bytes(work[at..at + 8].try_into().unwrap());
            if n_cpu > 0 {
                assert_eq!(
                    [word(0), word(4), word(8), word(12)],
                    [n_cpu as u32, rows_cpu, 3, 41],
                    "{tag}: work header"
                );
            }
            let dst = dev.memcpy_dtov(&dst_dev)?;
            let (mut j, mut ring) = (0usize, 0u32);
            for e in 0..n_experts {
                if !want[e] {
                    continue;
                }
                let at = 16 + 40 * j;
                assert_eq!(
                    [word(at), word(at + 4), word(at + 8)],
                    [e as u32, ring, counts[e]],
                    "{tag}: work entry {j}"
                );
                for p in 0..3 {
                    assert_eq!(
                        dword(at + 16 + 8 * p),
                        table[p * n_experts + e],
                        "{tag}: entry {j} address {p}"
                    );
                }
                let offset: u32 = counts[..e].iter().sum();
                for q in 0..counts[e] {
                    assert_eq!(
                        dst[(ring + q) as usize],
                        offset + q,
                        "{tag}: ring row {}",
                        ring + q
                    );
                }
                ring += counts[e];
                j += 1;
            }
        }
    }
    Ok(())
}

/// **The CPU class's device plumbing, end to end, in raw bytes.** Bucketize
/// gives the pinned experts to the CPU; the export kernel copies their rows of
/// the operand into the mapped row ring and rings the doorbell; a host thread
/// standing in for the pool writes a sentinel result row per ring row 20 ms
/// later, then `done`; one live launch — the down launch's place — runs with
/// the CPU item. Every CPU row of its output must hold its sentinel exactly,
/// every other row the all-VRAM launch's bytes, and no fault be claimed.
#[test]
pub(super) fn cuda_cpu_class_exports_rows_and_pulls_results() -> Result<()> {
    cpu_class_round_trip(true, 2, 24, 16)
}

/// The same at the prefill widths: the 64- and 128-wide tile modes, over a
/// prompt-sized launch whose experts mix counts above and below the cut.
#[test]
pub(super) fn cuda_cpu_class_round_trip_at_prefill_widths() -> Result<()> {
    cpu_class_round_trip(true, 4, 160, 16)?;
    cpu_class_round_trip(true, 8, 300, 16)
}

/// Many CPU experts in one layer: 64 experts, 32 of them pinned, so the split
/// hands the pool far more than a warp's or a super-block's worth.
#[test]
pub(super) fn cuda_cpu_class_round_trip_with_many_cpu_experts() -> Result<()> {
    cpu_class_round_trip(true, 2, 24, 64)?;
    cpu_class_round_trip(true, 8, 300, 64)
}

/// A pool that died: the liveness word reads 0 and `done` never lands. The CPU
/// item claims the fault word with the CPU expert code at once — the launch
/// completes, it does not wait out the spin limit.
#[test]
pub(super) fn cuda_cpu_class_faults_promptly_on_a_dead_pool() -> Result<()> {
    cpu_class_round_trip(false, 2, 24, 16)
}

/// **A count another launch left behind cannot keep this one from ringing.**
/// The export's last block rings the doorbell, found by a counter every launch
/// shares; a launch that did not count all its blocks out — discarded
/// mid-layer, or of another grid — left it part-way, and the next launch's
/// count then reached the grid early or never, so nothing rang and the down
/// launch waited out its deadline with `done` level with the doorbell
/// (WhiteWorld, e119fd7d7). Keyed by ticket, a stale count is replaced, and
/// every launch rings its own.
#[test]
pub(super) fn cuda_cpu_export_rings_whatever_count_another_launch_left() -> Result<()> {
    use candle_kernels::simple::moe_cpu_export::{run_moe_cpu_export, EXPORT_WARPS};

    let dev = CudaDevice::new(0)?;
    let stream = dev.cuda_stream();
    let (rows, ncols) = (16usize, 1024usize);
    let tiles = ncols / 128;
    let act: Vec<f32> = (0..rows * ncols)
        .map(|i| ((i % 97) as f32 - 48.0) / 50.0)
        .collect();
    let op = quantize_acts_q8a128_test(&dev, &act, rows, ncols)?;
    let in_tok = dev.memcpy_stod(&(0..rows as u32).collect::<Vec<_>>())?;
    let mut header = vec![0i32; 16];
    header[7] = rows as i32;
    let header = dev.memcpy_stod(&header)?;
    let (_ring_h, ring_d) = mapped_host((rows * tiles).div_ceil(8) * 1152);
    let (word_h, word_d) = mapped_host(64);
    // Several blocks, so the count matters: one block would ring alone.
    let grid = 3i32;
    assert!((rows * tiles).div_ceil(EXPORT_WARPS) >= grid as usize);
    let launch = |blocks: &CudaSlice<u64>, seq: u64| {
        let (hp, _g1) = header.device_ptr(&stream);
        let (ip, _g2) = in_tok.device_ptr(&stream);
        let (bp, _g3) = blocks.device_ptr(&stream);
        let status = op.with_device_ptr(&dev, |src| unsafe {
            Ok(run_moe_cpu_export(
                src as *const c_void,
                tiles as i32,
                ip as *const c_void,
                hp as *const c_void,
                ring_d as *mut c_void,
                rows as u32,
                word_d as *mut c_void,
                seq,
                bp as *mut c_void,
                grid,
                stream.cu_stream() as *mut c_void,
            ))
        })?;
        assert_eq!(status, 0, "export launched");
        stream.synchronize().map_err(crate::Error::wrap)?;
        Ok::<_, crate::Error>(unsafe { std::ptr::read_volatile(word_h as *const u64) })
    };
    // Another ticket's launch, two of its three blocks counted out.
    let stale = dev.memcpy_stod(&[(7u64 << 32) | 2])?;
    assert_eq!(
        launch(&stale, 8)?,
        8,
        "the stale count did not complete or block this launch"
    );
    // And back to back from there, as layers run.
    assert_eq!(launch(&stale, 9)?, 9);
    assert_eq!(launch(&stale, 10)?, 10);
    // A count left at this very ticket's grid by a launch that rang already
    // is the one case the key cannot tell apart, and it cannot arise: a ticket
    // is launched once.
    Ok(())
}

/// **Layers back to back, as a forward runs them.** Three invocations —
/// bucketize, export, live launch each — enqueued on the stream with no host
/// synchronisation between them, the stand-in pool answering each doorbell in
/// turn with different results at the same mapped addresses. Every launch must
/// pull its own invocation's results: the host rewrites the result ring in
/// place, and a pull served from a line an earlier launch cached would return
/// that launch's rows instead.
#[test]
pub(super) fn cuda_cpu_class_back_to_back_layers_pull_their_own_results() -> Result<()> {
    use candle_kernels::simple::moe_cpu_export::{
        run_moe_cpu_export, EXPORT_MAX_BLOCKS, EXPORT_WARPS,
    };
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const ROUNDS: u64 = 3;
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (n_experts, nrows, ncols) = (16usize, 256usize, 1024usize);
    let fx = LiveFixture::new(&dev, n_experts, nrows, ncols, 0x0c90_0101)?;
    let mut rng = StdRng::seed_from_u64(0x0c90_0102);
    let (n_sub, n_tokens, k) = (2usize, 1usize, 10usize);
    let tile_w = 16 * n_sub;
    let a_ub = n_tokens * k;
    // One decode token routed to ten distinct experts, five of them pinned.
    let mut ids: Vec<u32> = (0..n_experts as u32).collect();
    for i in (1..ids.len()).rev() {
        ids.swap(i, rng.random_range(0..=i));
    }
    ids.truncate(k);
    let class: Vec<u8> = (0..n_experts)
        .map(|e| if e % 2 == 0 { CLS_PINNED } else { CLS_VRAM })
        .collect();
    for (e, &c) in class.iter().enumerate() {
        fx.set(e, c);
    }
    let t = crate::Tensor::from_vec(ids.clone(), (n_tokens, k), &device)?;
    let act: Vec<f32> = (0..a_ub * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let op = quantize_acts_q8a128_test(&dev, &act, a_ub, ncols)?;
    let stream = dev.cuda_stream();
    let snap = dev.memcpy_stod(&vec![0u64; 3 * n_experts])?;
    let summary = dev.memcpy_stod(&vec![0u32; n_experts + 1])?;
    let remote = dev.memcpy_stod(&vec![0i32; n_experts * 4])?;
    let counters = dev.memcpy_stod(&[0i32; 3])?;
    let dst_row = dev.memcpy_stod(&vec![u32::MAX; a_ub])?;
    let tok_first = dev.memcpy_stod(&vec![0u32; n_tokens])?;
    let in_tok = dev.memcpy_stod(&vec![u32::MAX; a_ub])?;
    let (inputs_h, inputs_d) = mapped_host(4 * (1 + a_ub));
    let blocks_done = dev.memcpy_stod(&[0u64])?;
    const W: usize = 8;
    let slot_bytes = (ncols / 128) * 4 * fx.chunk_bytes;
    let scratch = unsafe { dev.alloc::<u8>(W * slot_bytes)? };
    let tiles = ncols / 128;
    let (_work_h, work_d) = mapped_host(CPU_WORK_SLOT_BYTES);
    let (_ring_h, ring_d) = mapped_host((a_ub * tiles).div_ceil(8) * 1152);
    let (res_h, res_d) = mapped_host(a_ub * nrows * 4);
    let (word_h, word_d) = mapped_host(64);
    let (doorbell_d, done_d, alive_d) = (word_d, word_d + 16, word_d + 32);
    unsafe { std::ptr::write_volatile((word_h as *mut u8).add(32) as *mut u32, 1) };
    let mut split = CpuSplit::OFF;
    split.cap_rows = a_ub as u32;
    split.max_n = CPU_MAX_N as u32;
    split.copy_ns = 1e6;
    split.read_shared_ns = 1.0;
    split.read_alone_ns = 1.0;
    split.max_experts = u32::MAX;
    for n in 1..=CPU_MAX_N {
        split.cpu_n_ns[n] = n as f32;
    }
    let base: u64 = 0xb2b0;
    let sentinel =
        |round: u64, j: usize, c: usize| (j * 1000 + c) as f32 + 0.5 + round as f32 * 0.25;

    // The stand-in pool: each doorbell in turn, its round's rows, then `done`.
    let pool = {
        let (word_h, res_h) = (word_h as usize, res_h as usize);
        std::thread::spawn(move || {
            let doorbell = word_h as *const u64;
            for round in 0..ROUNDS {
                let seq = base + round;
                let t0 = Instant::now();
                while unsafe { std::ptr::read_volatile(doorbell) } != seq {
                    assert!(t0.elapsed().as_secs() < 5, "doorbell {round} never rang");
                    std::hint::spin_loop();
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
                let res = res_h as *mut f32;
                for j in 0..a_ub {
                    for c in 0..nrows {
                        unsafe { *res.add(j * nrows + c) = sentinel(round, j, c) };
                    }
                }
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                unsafe { std::ptr::write_volatile((word_h as *mut u8).add(16) as *mut u64, seq) };
            }
        })
    };

    // Every round enqueued before anything is read back.
    let mut outs = Vec::new();
    let mut headers = Vec::new();
    for round in 0..ROUNDS {
        let seq = base + round;
        let mut ws = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
        {
            let (sp, _g1) = summary.device_ptr(&stream);
            let (rp, _g2) = remote.device_ptr(&stream);
            let (cp, _g3) = counters.device_ptr(&stream);
            let (np, _g4) = snap.device_ptr(&stream);
            let (dp, _g5) = dst_row.device_ptr(&stream);
            let blive = BucketizeLive {
                gate_row: fx.table_d,
                table_plane: n_experts as i64,
                snap: np,
                pinned: [
                    (fx.host as u64, fx.host as u64 + fx.host_bytes as u64),
                    (0, 0),
                ],
                summary: sp,
                summary_seq: 9,
                remote: rp,
                counters: cp,
                row: 2,
                promo: None,
                remote_dst: 0,
                started_rows: 0,
                ticket: 0,
                owner: OwnerCheck::default(),
                ahead: None,
                cpu: Some(BucketizeCpu {
                    split,
                    work: work_d,
                    dst_row: dp,
                    carry: 0,
                    tok_first: tok_first.device_ptr(&stream).0,
                    in_tok: in_tok.device_ptr(&stream).0,
                    input_map: inputs_d,
                }),
            };
            drop((_g1, _g2, _g3, _g4, _g5));
            moe_bucketize(
                &t,
                n_experts,
                tile_w,
                &mut ws,
                Some(&blive),
                &DecodeRows::prefix(1),
            )?;
        }
        {
            let (hp, _g1) = ws.header.device_ptr(&stream);
            let (ip, _g2) = in_tok.device_ptr(&stream);
            let (bp, _g3) = blocks_done.device_ptr(&stream);
            let status = op.with_device_ptr(&dev, |src| unsafe {
                Ok(run_moe_cpu_export(
                    src as *const c_void,
                    tiles as i32,
                    ip as *const c_void,
                    hp as *const c_void,
                    ring_d as *mut c_void,
                    a_ub as u32,
                    doorbell_d as *mut c_void,
                    seq,
                    bp as *mut c_void,
                    (a_ub * tiles).div_ceil(EXPORT_WARPS).min(EXPORT_MAX_BLOCKS) as i32,
                    stream.cu_stream() as *mut c_void,
                ))
            })?;
            assert_eq!(status, 0, "export launched");
        }
        let (rp, _g1) = remote.device_ptr(&stream);
        let (cp, _g2) = counters.device_ptr(&stream);
        let (hp, _g3) = ws.header.device_ptr(&stream);
        let (scp, _g4) = scratch.device_ptr(&stream);
        let (dp, _g5) = dst_row.device_ptr(&stream);
        let live = MoeLive {
            abort: fx.abort_d,
            fault: fx.fault_d,
            live_row: fx.table_d,
            remote: rp,
            remote_dst: 0,
            header: hp,
            counter: cp,
            scratch: scp,
            slot_bytes: slot_bytes as u64,
            dst_offset: 0,
            stall: 0,
            ahead: 0,
            ahead_done: 0,
            spin_limit_ns: 10_000_000_000,
            cpu_done: done_d,
            cpu_seq: seq,
            cpu_rows: res_d,
            cpu_dst_row: dp,
            cpu_alive: alive_d,
            cpu_tight_ns: 50_000,
            cpu_wait: 0,
            cold_wait: 0,
            workers: W as i32,
            row: 2,
        };
        drop((_g1, _g2, _g3, _g4, _g5));
        outs.push(grouped_qmatmul_dev_q8a128(
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
        )?);
        headers.push(ws);
    }
    pool.join().expect("the pool stand-in");
    assert_eq!(fx.fault(), 0, "no fault");
    // Every round routed the same, so one map serves all three.
    let dst = dev.memcpy_dtov(&dst_row)?;
    for (round, (out, ws)) in outs.iter().zip(&headers).enumerate() {
        let got = read_f32_tensor(&dev, out)?;
        let hd = dev.memcpy_dtov(&ws.header.slice(..HEADER_WORDS))?;
        let n_cpu_rows = hd[5] as usize;
        assert!(n_cpu_rows > 1, "round {round}: the split gave the CPU rows");
        // One decode token: however many CPU experts it routes, one input row.
        assert_eq!(hd[7], 1, "round {round}: the token crosses once");
        let map = unsafe { std::slice::from_raw_parts(inputs_h as *const u32, 1 + n_cpu_rows) };
        assert_eq!(map[0], 1);
        assert!(
            map[1..].iter().all(|&r| r == 0),
            "every ring row reads input row 0"
        );
        for (j, &r) in dst[..n_cpu_rows].iter().enumerate() {
            let row = &got[r as usize * nrows..(r as usize + 1) * nrows];
            for (c, &v) in row.iter().enumerate() {
                assert_eq!(
                    v,
                    sentinel(round as u64, j, c),
                    "round {round}: CPU row {r} (ring {j}) col {c} — another round's result"
                );
            }
        }
    }
    Ok(())
}

pub(super) fn cpu_class_round_trip(
    pool_alive: bool,
    n_sub: usize,
    n_tokens: usize,
    n_experts: usize,
) -> Result<()> {
    use candle_kernels::quantized::{MOE_FAULT_CPU_EXPERT, MOE_FAULT_EXPERT_SHIFT};
    use candle_kernels::simple::moe_cpu_export::{
        run_moe_cpu_export, EXPORT_MAX_BLOCKS, EXPORT_WARPS,
    };
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (nrows, ncols) = (256usize, 1024usize);
    let fx = LiveFixture::new(&dev, n_experts, nrows, ncols, 0x0c90_0001)?;
    let full_table = dev.memcpy_stod(&fx.vram_ptrs)?;
    let mut rng = StdRng::seed_from_u64(0x0c90_0002);
    let tile_w = 16 * n_sub;
    let k = 2usize;
    let a_ub = n_tokens * k;
    let ids: Vec<u32> = (0..n_tokens)
        .flat_map(|_| {
            let a = rng.random_range(0..n_experts as u32);
            let b = (a + 1 + rng.random_range(0..n_experts as u32 - 1)) % n_experts as u32;
            [a, b]
        })
        .collect();
    // Even experts pinned, odd in VRAM.
    let class: Vec<u8> = (0..n_experts)
        .map(|e| if e % 2 == 0 { CLS_PINNED } else { CLS_VRAM })
        .collect();
    let t = crate::Tensor::from_vec(ids.clone(), (n_tokens, k), &device)?;
    let act: Vec<f32> = (0..a_ub * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let op = quantize_acts_q8a128_test(&dev, &act, a_ub, ncols)?;
    let stream = dev.cuda_stream();

    // Reference: every expert in VRAM.
    let mut ws_ref = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
    moe_bucketize(
        &t,
        n_experts,
        tile_w,
        &mut ws_ref,
        None,
        &DecodeRows::none(),
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

    for (e, &c) in class.iter().enumerate() {
        fx.set(e, c);
    }
    let snap = dev.memcpy_stod(&vec![0u64; 3 * n_experts])?;
    let summary = dev.memcpy_stod(&vec![0u32; n_experts + 1])?;
    let remote = dev.memcpy_stod(&vec![0i32; n_experts * 4])?;
    let counters = dev.memcpy_stod(&[0i32; 3])?;
    let dst_row = dev.memcpy_stod(&vec![u32::MAX; a_ub])?;
    let tok_first = dev.memcpy_stod(&vec![0u32; n_tokens])?;
    let in_tok = dev.memcpy_stod(&vec![u32::MAX; a_ub])?;
    let (inputs_h, inputs_d) = mapped_host(4 * (1 + a_ub));
    let blocks_done = dev.memcpy_stod(&[0u64])?;
    const W: usize = 8;
    let slot_bytes = (ncols / 128) * 4 * fx.chunk_bytes;
    let scratch = unsafe { dev.alloc::<u8>(W * slot_bytes)? };
    let tiles = ncols / 128;
    let ring_bytes = (a_ub * tiles).div_ceil(8) * 1152;
    let (work_h, work_d) = mapped_host(CPU_WORK_SLOT_BYTES);
    let (ring_h, ring_d) = mapped_host(ring_bytes);
    let (res_h, res_d) = mapped_host(a_ub * nrows * 4);
    // doorbell u64 @0, done u64 @16, alive u32 @32, the device's wait u64 @40
    let (word_h, word_d) = mapped_host(64);
    let (doorbell_d, done_d, alive_d, wait_d) = (word_d, word_d + 16, word_d + 32, word_d + 40);
    unsafe {
        std::ptr::write_volatile(
            (word_h as *mut u8).add(32) as *mut u32,
            u32::from(pool_alive),
        );
    }
    let mut split = CpuSplit::OFF;
    split.cap_rows = a_ub as u32;
    split.max_n = CPU_MAX_N as u32;
    split.copy_ns = 1e6;
    split.read_shared_ns = 1.0;
    split.read_alone_ns = 1.0;
    split.max_experts = u32::MAX;
    for n in 1..=CPU_MAX_N {
        split.cpu_n_ns[n] = n as f32;
    }
    let mut ws = MoeBucketizeWorkspace::new(&dev, n_tokens, k)?;
    let seq: u64 = 0x5eed;
    let sentinel = |j: usize, c: usize| (j * 1000 + c) as f32 + 0.5;
    {
        let (sp, _g1) = summary.device_ptr(&stream);
        let (rp, _g2) = remote.device_ptr(&stream);
        let (cp, _g3) = counters.device_ptr(&stream);
        let (np, _g4) = snap.device_ptr(&stream);
        let (dp, _g5) = dst_row.device_ptr(&stream);
        let blive = BucketizeLive {
            gate_row: fx.table_d,
            table_plane: n_experts as i64,
            snap: np,
            pinned: [
                (fx.host as u64, fx.host as u64 + fx.host_bytes as u64),
                (0, 0),
            ],
            summary: sp,
            summary_seq: 9,
            remote: rp,
            counters: cp,
            row: 2,
            promo: None,
            remote_dst: 0,
            started_rows: 0,
            ticket: 0,
            owner: OwnerCheck::default(),
            ahead: None,
            cpu: Some(BucketizeCpu {
                split,
                work: work_d,
                dst_row: dp,
                carry: 0,
                tok_first: tok_first.device_ptr(&stream).0,
                in_tok: in_tok.device_ptr(&stream).0,
                input_map: inputs_d,
            }),
        };
        drop((_g1, _g2, _g3, _g4, _g5));
        moe_bucketize(
            &t,
            n_experts,
            tile_w,
            &mut ws,
            Some(&blive),
            &DecodeRows::none(),
        )?;
    }
    // The export: each distinct token's row of the input into the ring, once.
    {
        let (hp, _g1) = ws.header.device_ptr(&stream);
        let (ip, _g2) = in_tok.device_ptr(&stream);
        let (bp, _g3) = blocks_done.device_ptr(&stream);
        let status = op.with_device_ptr(&dev, |src| unsafe {
            Ok(run_moe_cpu_export(
                src as *const c_void,
                tiles as i32,
                ip as *const c_void,
                hp as *const c_void,
                ring_d as *mut c_void,
                a_ub as u32,
                doorbell_d as *mut c_void,
                seq,
                bp as *mut c_void,
                (a_ub * tiles).div_ceil(EXPORT_WARPS).min(EXPORT_MAX_BLOCKS) as i32,
                stream.cu_stream() as *mut c_void,
            ))
        })?;
        assert_eq!(status, 0, "export launched");
    }
    // The pool's stand-in: once the doorbell rings, sentinel rows, then `done`.
    // A decode-sized launch answers late enough that its CPU item is surely
    // waiting by then, even beside other tests on the device, so the wait it
    // records is checked below.
    let host_rows = a_ub;
    let answer_after = std::time::Duration::from_millis(if n_tokens <= 24 { 200 } else { 20 });
    let pool = {
        let (word_h, res_h) = (word_h as usize, res_h as usize);
        std::thread::spawn(move || {
            let doorbell = word_h as *const u64;
            let t0 = Instant::now();
            while unsafe { std::ptr::read_volatile(doorbell) } != seq {
                assert!(t0.elapsed().as_secs() < 5, "the doorbell never rang");
                std::hint::spin_loop();
            }
            if !pool_alive {
                return;
            }
            std::thread::sleep(answer_after);
            let res = res_h as *mut f32;
            for j in 0..host_rows {
                for c in 0..nrows {
                    unsafe { *res.add(j * nrows + c) = sentinel(j, c) };
                }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            unsafe { std::ptr::write_volatile((word_h as *mut u8).add(16) as *mut u64, seq) };
        })
    };
    let got = {
        let (rp, _g1) = remote.device_ptr(&stream);
        let (cp, _g2) = counters.device_ptr(&stream);
        let (hp, _g3) = ws.header.device_ptr(&stream);
        let (scp, _g4) = scratch.device_ptr(&stream);
        let (dp, _g5) = dst_row.device_ptr(&stream);
        let live = MoeLive {
            abort: fx.abort_d,
            fault: fx.fault_d,
            live_row: fx.table_d,
            remote: rp,
            remote_dst: 0,
            header: hp,
            counter: cp,
            scratch: scp,
            slot_bytes: slot_bytes as u64,
            dst_offset: 0,
            stall: 0,
            ahead: 0,
            ahead_done: 0,
            spin_limit_ns: 10_000_000_000,
            cpu_done: done_d,
            cpu_seq: seq,
            cpu_rows: res_d,
            cpu_dst_row: dp,
            cpu_alive: alive_d,
            cpu_tight_ns: 50_000,
            cpu_wait: wait_d,
            cold_wait: 0,
            workers: W as i32,
            row: 2,
        };
        drop((_g1, _g2, _g3, _g4, _g5));
        let t0 = Instant::now();
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
        assert!(
            t0.elapsed().as_secs() < 2,
            "the launch must not wait out the spin limit"
        );
        got
    };
    pool.join().expect("the pool stand-in");

    let hd = dev.memcpy_dtov(&ws.header.slice(..HEADER_WORDS))?;
    let n_cpu_rows = hd[5] as usize;
    assert!(n_cpu_rows > 0, "the split gave the CPU some rows");
    // The pool reads the same publication from the mapped work slot.
    let work = unsafe { std::slice::from_raw_parts(work_h as *const u32, 4) };
    assert_eq!(
        [work[0], work[1], work[2], work[3]],
        [hd[6] as u32, hd[5] as u32, 2, 9],
        "the work slot's header: experts, rows, row, sequence"
    );
    let dst = dev.memcpy_dtov(&dst_row)?;
    assert_eq!(
        unsafe { std::ptr::read_volatile(word_h as *const u64) },
        seq,
        "the doorbell rang"
    );
    // The ring holds the CPU rows' operand bytes, tile for tile.
    let src_bytes = op.with_device_ptr(&dev, |p| {
        let mut v = vec![0u8; op.byte_len()];
        let r = copy_to_host(&dev, v.as_mut_ptr() as *mut c_void, p, v.len());
        assert_eq!(
            r,
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "operand to host"
        );
        Ok(v)
    })?;
    // The inputs: the distinct tokens of the CPU ring rows, in order of each
    // token's first ring row, and each ring row's input row — raw, against
    // the same walk over `tok_ids` on the host.
    let tok_ids = dev.memcpy_dtov(&ws.tok_ids.slice(..a_ub))?;
    let mut want_tok: Vec<u32> = Vec::new();
    let mut want_map: Vec<u32> = Vec::new();
    for &g in &dst[..n_cpu_rows] {
        let t = tok_ids[g as usize];
        let r = match want_tok.iter().position(|&x| x == t) {
            Some(r) => r,
            None => {
                want_tok.push(t);
                want_tok.len() - 1
            }
        };
        want_map.push(r as u32);
    }
    assert_eq!(hd[7] as usize, want_tok.len(), "input rows");
    assert!(want_tok.len() <= n_cpu_rows);
    let map = unsafe { std::slice::from_raw_parts(inputs_h as *const u32, 1 + n_cpu_rows) };
    assert_eq!(map[0] as usize, want_tok.len());
    assert_eq!(&map[1..], &want_map[..], "ring row → input row");
    assert_eq!(
        dev.memcpy_dtov(&in_tok.slice(..want_tok.len()))?,
        want_tok,
        "input row → token"
    );
    // The ring holds each input row's token's operand bytes, tile for tile.
    let ring = unsafe { std::slice::from_raw_parts(ring_h as *const u8, ring_bytes) };
    let qs_off = |ft: usize| (ft >> 3) * 1152 + (ft & 7) * 128;
    let ds_off = |ft: usize| (ft >> 3) * 1152 + 1024 + (ft & 7) * 16;
    for (r, &tok) in want_tok.iter().enumerate() {
        for tt in 0..tiles {
            let (ft, fs) = (r * tiles + tt, tok as usize * tiles + tt);
            assert_eq!(
                &ring[qs_off(ft)..qs_off(ft) + 128],
                &src_bytes[qs_off(fs)..qs_off(fs) + 128],
                "input row {r} tile {tt} codes"
            );
            assert_eq!(
                &ring[ds_off(ft)..ds_off(ft) + 4],
                &src_bytes[ds_off(fs)..ds_off(fs) + 4],
                "input row {r} tile {tt} scale"
            );
        }
    }
    let waited = unsafe { std::ptr::read_volatile((word_h as *const u8).add(40) as *const u64) };
    if !pool_alive {
        let f = fx.fault();
        assert_ne!(f, 0, "a dead pool faults the wait");
        assert_eq!(
            f >> MOE_FAULT_EXPERT_SHIFT & 0xffff,
            MOE_FAULT_CPU_EXPERT,
            "the CPU code"
        );
        assert_eq!(
            waited, 0,
            "a wait that ended without results records no lateness"
        );
        return Ok(());
    }
    assert_eq!(fx.fault(), 0, "no fault");
    // A decode-sized launch reaches its CPU item long before the stand-in's
    // 200 ms answer; a prompt-sized one may still be on its own items at 20
    // ms, and waits less or not at all.
    assert!(
        waited < 2_000_000_000,
        "the device's lateness on the pool: {waited} ns"
    );
    if n_tokens <= 24 {
        assert!(
            waited > 50_000_000,
            "the device's lateness on the pool: {waited} ns"
        );
    }
    let cpu_rows: std::collections::HashMap<usize, usize> =
        (0..n_cpu_rows).map(|j| (dst[j] as usize, j)).collect();
    let total_valid = hd[1] as usize;
    for r in 0..total_valid {
        let row = &got[r * nrows..(r + 1) * nrows];
        match cpu_rows.get(&r) {
            Some(&j) => {
                for (c, &v) in row.iter().enumerate() {
                    assert_eq!(v, sentinel(j, c), "CPU row {r} (ring {j}) col {c}");
                }
            }
            None => assert_eq!(
                row.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                want[r * nrows..(r + 1) * nrows]
                    .iter()
                    .map(|f| f.to_bits())
                    .collect::<Vec<_>>(),
                "GPU row {r}"
            ),
        }
    }
    Ok(())
}

/// How many of a live launch's remote experts the test ring holds a slot for.
#[derive(Clone, Copy)]
pub(super) enum Slotted {
    All,
    Half,
}
