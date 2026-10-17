//! The CPU expert kernel's optimisation bench (`docs/cpu_expert_compute.md`
//! §5.10): one or several Flash-Next-geometry Q3_KO experts through the
//! production layer schedule on the pool, at the split's token-count buckets
//! and several pool sizes. It reports time per layer, weight bytes per second
//! against the measured DRAM rate, and int8 MAC/s — the `cpu_n_ns` table.

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::ptr::null_mut;
    use std::time::{Duration, Instant};

    use candle::quantized::ko_quant::quantize_ko;
    use candle::quantized::GgmlDType;
    use candle::Device;
    use cudarc::driver::sys;

    use crate::models::expert_lre::cpu_pool::kernel::act::ActTiles;
    use crate::models::expert_lre::cpu_pool::kernel::expert::Expert;
    use crate::models::expert_lre::cpu_pool::kernel::ko::Ko;
    use crate::models::expert_lre::cpu_pool::kernel::simd::Isa;
    use crate::models::expert_lre::cpu_pool::layer::{
        run_layer, run_layer_on, ExpertWork, LayerWork,
    };
    use crate::models::expert_lre::cpu_pool::pin::pin_current_thread;
    use crate::models::expert_lre::cpu_pool::pool::{Pool, PoolWaits};
    use crate::models::expert_lre::cpu_pool::topology::{
        performance_cores, physical_cores, pool_cores,
    };

    const HIDDEN: usize = 2560;
    const FF: usize = 640;
    /// The split's count buckets (`SplitParams::cpu_n_ns`).
    const BUCKETS: [usize; 10] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn slot(seed: &mut u64) -> (Vec<u8>, (usize, usize, usize)) {
        let mut w = |n: usize| -> Vec<f32> { (0..n).map(|_| lcg(seed) * 0.05).collect() };
        let (g, u, d) = (
            quantize_ko(&w(FF * HIDDEN), FF, HIDDEN, GgmlDType::Q3_KO),
            quantize_ko(&w(FF * HIDDEN), FF, HIDDEN, GgmlDType::Q3_KO),
            quantize_ko(&w(HIDDEN * FF), HIDDEN, FF, GgmlDType::Q3_KO),
        );
        let offsets = (0, g.len(), g.len() + u.len());
        ([g, u, d].concat(), offsets)
    }

    /// Best-of-`reps` wall time of one `run_layer`.
    fn time_layer(pool: &Pool, isa: Isa, work: &LayerWork, reps: usize) -> Duration {
        let mut best = Duration::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let out = run_layer(pool, isa, work);
            best = best.min(t.elapsed());
            std::hint::black_box(out);
        }
        best
    }

    /// `cargo test --release -p candle-transformers --features cuda --lib cpu_expert_kernel_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement, ~1 min; read its output"]
    fn cpu_expert_kernel_bench() {
        let isa = Isa::detect();
        let cores = physical_cores().unwrap();
        println!("isa {isa:?}; {} physical cores", cores.len());
        for c in &cores {
            println!(
                "  core {:>2}: class {} llc {} numa {} sets {:?}",
                c.core, c.efficiency_class, c.llc, c.numa, c.cpu_sets
            );
        }
        let pool_set = pool_cores(&cores, 1);
        // The caller stands in for the forward thread, on the reserved core.
        pin_current_thread(performance_cores(&cores)[0].primary()).unwrap();
        // 64 distinct experts, so successive runs read cold weights from DRAM
        // the way a layer does, not from the 24 MB L3.
        let mut seed = 7;
        let slots: Vec<_> = (0..64).map(|_| slot(&mut seed)).collect();
        let expert_bytes = slots[0].0.len();
        let x: Vec<f32> = (0..32 * HIDDEN).map(|_| lcg(&mut seed)).collect();
        let act = ActTiles::quantize(&x, 32, HIDDEN);
        let mut next = 0usize;
        // As the service runs a layer: a result row per assignment.
        let mut experts = |k: usize, n: usize| -> Vec<ExpertWork> {
            (0..k)
                .map(|j| {
                    let (bytes, off) = &slots[next % slots.len()];
                    next += 1;
                    ExpertWork {
                        expert: Expert::from_slot(bytes, *off, HIDDEN, FF, Ko::Q3),
                        tokens: (0..n).collect(),
                        weights: vec![1.0; n],
                        outputs: (j * n..(j + 1) * n).collect(),
                    }
                })
                .collect()
        };

        // The caller is the forward thread, on the reserved core; the workers
        // are the pool's cores, taken in pool order.
        for workers in [3usize, 5, pool_set.len()] {
            let sets: Vec<u32> = pool_set.iter().take(workers).map(|c| c.primary()).collect();
            // Every core counted fast: the bench sweeps the pool's size itself.
            let pool =
                Pool::new(&sets, sets.len(), PoolWaits::new(Duration::from_millis(5))).unwrap();
            // Fresh threads are still starting and migrating to their cores for
            // the first few milliseconds; a pool that lives for the process
            // never pays that, so the bench does not either.
            for _ in 0..200 {
                pool.run(&|_| {});
            }
            println!(
                "\n{} participants (caller + {} workers):",
                pool.participants(),
                sets.len()
            );
            println!("  experts × tokens   µs/layer   µs/expert   GB/s weights   GMAC/s");
            for &(k, n) in BUCKETS
                .iter()
                .map(|n| (1usize, *n))
                .chain([(5, 1), (10, 1), (5, 4)])
                .collect::<Vec<_>>()
                .iter()
            {
                let work = LayerWork {
                    act: &act,
                    experts: experts(k, n),
                    rows: k * n,
                };
                let t = time_layer(&pool, isa, &work, 9).as_secs_f64();
                let bytes = (k * expert_bytes) as f64;
                let macs = (k * n * 3 * HIDDEN * FF) as f64;
                println!(
                    "  {k:>3} × {n:<3}        {:>8.1}   {:>9.1}   {:>12.1}   {:>6.1}",
                    t * 1e6,
                    t * 1e6 / k as f64,
                    bytes / t / 1e9,
                    macs / t / 1e9
                );
            }
            // Every repetition of the single-expert, single-token layer — the
            // common decode case — not just the best.
            let work = LayerWork {
                act: &act,
                experts: experts(1, 1),
                rows: 1,
            };
            let reps: Vec<String> = (0..20)
                .map(|_| {
                    let t = Instant::now();
                    std::hint::black_box(run_layer(&pool, isa, &work));
                    format!("{:.0}", t.elapsed().as_secs_f64() * 1e6)
                })
                .collect();
            println!("  1 × 1 every rep (µs): {}", reps.join(" "));
        }
    }

    /// Where a decode layer's time goes on the pool: `k` one-token experts —
    /// decode's shape — at pool sizes from one participant up, each timed
    /// back to back (workers awake) and after a 2 ms gap (workers parked past
    /// a 200 µs spin, as between decode layers). Prints the median and best
    /// µs a layer, and the weight bytes per second the median moves.
    ///
    /// `cargo test --release -p candle-transformers --features cuda --lib decode_layer_latency -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement, ~1 min; read its output"]
    fn decode_layer_latency() {
        let isa = Isa::detect();
        let cores = physical_cores().unwrap();
        let pool_set = pool_cores(&cores, 1);
        pin_current_thread(performance_cores(&cores)[0].primary()).unwrap();
        let mut seed = 11;
        let slots: Vec<_> = (0..64).map(|_| slot(&mut seed)).collect();
        let expert_bytes = slots[0].0.len();
        let x: Vec<f32> = (0..HIDDEN).map(|_| lcg(&mut seed)).collect();
        let act = ActTiles::quantize(&x, 1, HIDDEN);
        let mut next = 0usize;
        let mut layer = |k: usize| -> LayerWork {
            let experts = (0..k)
                .map(|j| {
                    let (bytes, off) = &slots[next % slots.len()];
                    next += 1;
                    ExpertWork {
                        expert: Expert::from_slot(bytes, *off, HIDDEN, FF, Ko::Q3),
                        tokens: vec![0],
                        weights: vec![1.0],
                        outputs: vec![j],
                    }
                })
                .collect();
            LayerWork {
                act: &act,
                experts,
                rows: k,
            }
        };
        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        let sets: Vec<u32> = pool_set.iter().skip(1).map(|c| c.primary()).collect();
        let pool = Pool::new(
            &sets,
            sets.len(),
            PoolWaits::new(Duration::from_micros(200)),
        )
        .unwrap();
        for _ in 0..200 {
            pool.run(&|_| {});
        }
        println!(
            "experts  participants  awake median/best µs   parked median/best µs   GB/s awake"
        );
        for k in [1usize, 2, 3, 4, 6] {
            for parts in [1usize, 2, 4, 6, 8, pool.participants()] {
                let mut awake = Vec::new();
                for _ in 0..30 {
                    let w = layer(k);
                    let t = Instant::now();
                    std::hint::black_box(run_layer_on(&pool, isa, &w, parts));
                    awake.push(t.elapsed().as_secs_f64() * 1e6);
                }
                let mut parked = Vec::new();
                for _ in 0..30 {
                    std::thread::sleep(Duration::from_millis(2));
                    let w = layer(k);
                    let t = Instant::now();
                    std::hint::black_box(run_layer_on(&pool, isa, &w, parts));
                    parked.push(t.elapsed().as_secs_f64() * 1e6);
                }
                let (am, ab) = (
                    median(awake.clone()),
                    awake.iter().cloned().fold(f64::MAX, f64::min),
                );
                let (pm, pb) = (
                    median(parked.clone()),
                    parked.iter().cloned().fold(f64::MAX, f64::min),
                );
                println!(
                    "{k:>7}  {parts:>12}  {am:>9.0} / {ab:<9.0}  {pm:>9.0} / {pb:<9.0}  {:>8.1}",
                    (k * expert_bytes) as f64 / am / 1e3
                );
            }
        }
        // The same experts again and again — their 2k MB inside the 24 MB L3 —
        // against the DRAM-cold figures above: how much of a layer's time is
        // the weights' trip from DRAM.
        println!("\nL3-hot (the same experts each run), workers awake:");
        for k in [1usize, 3, 6] {
            for parts in [4usize, 6, 8] {
                let w = layer(k);
                let mut hot = Vec::new();
                for _ in 0..40 {
                    let t = Instant::now();
                    std::hint::black_box(run_layer_on(&pool, isa, &w, parts));
                    hot.push(t.elapsed().as_secs_f64() * 1e6);
                }
                let hm = median(hot.clone());
                println!(
                    "{k:>7}  {parts:>12}  {hm:>9.0} µs median  {:>8.1} GB/s",
                    (k * expert_bytes) as f64 / hm / 1e3
                );
            }
        }

        // The same 64 experts in page-locked memory from `cuMemAllocHost`, as
        // the warm tier and the pad hold them, against the pageable figures
        // above.
        let _context = Device::new_cuda(0).unwrap();
        let total = slots.iter().map(|(b, _)| b.len()).sum::<usize>();
        let mut raw: *mut c_void = null_mut();
        // SAFETY: freed below, after the last read.
        let r = unsafe { sys::cuMemAllocHost_v2(&mut raw, total) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS);
        // SAFETY: `total` bytes just allocated, filled from the slots.
        let pinned = unsafe { std::slice::from_raw_parts_mut(raw as *mut u8, total) };
        let mut at = 0;
        let mut pinned_slots = Vec::new();
        for (bytes, off) in &slots {
            pinned[at..at + bytes.len()].copy_from_slice(bytes);
            pinned_slots.push((at, bytes.len(), *off));
            at += bytes.len();
        }
        println!("\npinned (cuMemAllocHost), DRAM-cold, workers awake:");
        for k in [1usize, 3, 6] {
            for parts in [4usize, 6, 8] {
                let mut times = Vec::new();
                for _ in 0..30 {
                    let experts = (0..k)
                        .map(|j| {
                            let (start, len, off) = pinned_slots[next % pinned_slots.len()];
                            next += 1;
                            ExpertWork {
                                expert: Expert::from_slot(
                                    &pinned[start..start + len],
                                    off,
                                    HIDDEN,
                                    FF,
                                    Ko::Q3,
                                ),
                                tokens: vec![0],
                                weights: vec![1.0],
                                outputs: vec![j],
                            }
                        })
                        .collect();
                    let w = LayerWork {
                        act: &act,
                        experts,
                        rows: k,
                    };
                    let t = Instant::now();
                    std::hint::black_box(run_layer_on(&pool, isa, &w, parts));
                    times.push(t.elapsed().as_secs_f64() * 1e6);
                }
                let m = median(times);
                println!(
                    "{k:>7}  {parts:>12}  {m:>9.0} µs median  {:>8.1} GB/s",
                    (k * expert_bytes) as f64 / m / 1e3
                );
            }
        }
        // SAFETY: allocated above; no reader remains.
        unsafe { sys::cuMemFreeHost(raw) };
    }
}
