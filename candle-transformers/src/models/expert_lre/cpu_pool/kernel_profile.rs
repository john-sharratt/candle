//! The CPU expert kernel's tuning harness: Flash-Next-geometry Q3_KO experts
//! under the profiles that bound a layer's time, in ~20 s.
//!
//! | Profile | Bound it measures |
//! |---|---|
//! | `roof` | the DRAM read rate of one core and of the pool's performance cores |
//! | `hot` | the matvec on one core with its weights in L2 — the compute ceiling, per ISA and token count |
//! | `cold` | the same on DRAM-cold weights — whether one core is compute- or memory-bound |
//! | `expert` | one whole expert on one thread, DRAM-cold — what SwiGLU and the down input's quantize add |
//! | `layer` | decode and prefill shapes on the production pool, DRAM-cold, against the roof |
//!
//! The pool is built as `attach` builds it: the caller pinned to the first
//! pool core, the workers on the rest, the performance cores counted fast —
//! and held warm ahead of every run, as the service holds it through
//! inference.
//!
//! `cargo test --release -p candle-transformers --features cuda --lib cpu_kernel_profile -- --ignored --nocapture`

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use std::arch::asm;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use candle::quantized::ko_quant::quantize_ko;
    use candle::quantized::GgmlDType;

    use crate::models::expert_lre::cpu_pool::backoff::COOL_DOWN;
    use crate::models::expert_lre::cpu_pool::bandwidth::{read_rate, touched_buffer};
    use crate::models::expert_lre::cpu_pool::kernel::act::ActTiles;
    use crate::models::expert_lre::cpu_pool::kernel::expert::{forward, Expert};
    use crate::models::expert_lre::cpu_pool::kernel::ko::{Ko, KoMatrix, Permuted};
    use crate::models::expert_lre::cpu_pool::kernel::simd::{matvec, Isa};
    use crate::models::expert_lre::cpu_pool::layer::{run_layer_on, ExpertWork, LayerWork};
    use crate::models::expert_lre::cpu_pool::pin::pin_current_thread;
    use crate::models::expert_lre::cpu_pool::pool::{Pool, PoolWaits};
    use crate::models::expert_lre::cpu_pool::stats::{stats, ClassStats};
    use crate::models::expert_lre::cpu_pool::topology::{physical_cores, pool_cores};

    const HIDDEN: usize = 2560;
    const FF: usize = 640;
    /// Distinct experts the cold profiles rotate through: 96 MB, four times
    /// the laptop's 24 MB L3.
    const CORPUS: usize = 48;
    /// The layer profiles: `(experts, tokens per expert)`. Decode hands the
    /// pool one to five one-token experts; a prompt's layer more tokens.
    const LAYERS: [(usize, usize); 8] = [
        (1, 1),
        (2, 1),
        (3, 1),
        (5, 1),
        (1, 4),
        (5, 2),
        (5, 4),
        (2, 16),
    ];

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    type Slot = (Vec<u8>, (usize, usize, usize));

    /// One quantized expert image; the corpus is copies of a few, since the
    /// values do not change the time — only that each copy is its own memory.
    fn slot(seed: &mut u64) -> Slot {
        let mut w = |n: usize| -> Vec<f32> { (0..n).map(|_| lcg(seed) * 0.05).collect() };
        let (g, u, d) = (
            quantize_ko(&w(FF * HIDDEN), FF, HIDDEN, GgmlDType::Q3_KO),
            quantize_ko(&w(FF * HIDDEN), FF, HIDDEN, GgmlDType::Q3_KO),
            quantize_ko(&w(HIDDEN * FF), HIDDEN, FF, GgmlDType::Q3_KO),
        );
        let offsets = (0, g.len(), g.len() + u.len());
        ([g, u, d].concat(), offsets)
    }

    /// Both classes' pool counters, summed: a layer here may fall in either.
    fn pool_stats() -> ClassStats {
        let [s, l] = stats();
        ClassStats {
            runs: s.runs + l.runs,
            participants: s.participants + l.participants,
            join_ns: s.join_ns + l.join_ns,
            work_ns: s.work_ns + l.work_ns,
            in_job_ns: s.in_job_ns + l.in_job_ns,
            ..ClassStats::default()
        }
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    }

    /// This core's clock: a chain of dependent 64-bit multiplies retires one
    /// per three cycles. Written as assembly, since the compiler would fold a
    /// chain — and not as adds, which this core folds at rename.
    #[cfg(target_arch = "x86_64")]
    fn core_ghz() -> f64 {
        const TURNS: u64 = 10_000_000;
        const IMUL_LATENCY: f64 = 3.0;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let mut v = black_box(3u64);
            let t = Instant::now();
            for _ in 0..TURNS {
                // SAFETY: register arithmetic only.
                unsafe {
                    asm!(
                        "imul {v}, {v}", "imul {v}, {v}", "imul {v}, {v}", "imul {v}, {v}",
                        "imul {v}, {v}", "imul {v}, {v}", "imul {v}, {v}", "imul {v}, {v}",
                        v = inout(reg) v,
                        options(nomem, nostack),
                    );
                }
            }
            best = best.min(t.elapsed().as_secs_f64());
            black_box(v);
        }
        (8 * TURNS) as f64 * IMUL_LATENCY / best / 1e9
    }

    fn isas() -> Vec<Isa> {
        [Isa::Avx2, Isa::AvxVnni, Isa::Avx512Vnni]
            .into_iter()
            .filter(|i| i.available())
            .collect()
    }

    #[test]
    #[ignore = "a measurement, ~20 s; read its output"]
    fn cpu_kernel_profile() {
        let cores = pool_cores(&physical_cores().unwrap(), 1);
        let (first, rest) = cores.split_first().unwrap();
        let fast = cores
            .iter()
            .take_while(|c| c.efficiency_class == first.efficiency_class)
            .count();
        pin_current_thread(first.primary()).unwrap();
        let isa = Isa::detect();
        println!(
            "isa {isa:?}; pool {} cores, {fast} performance",
            cores.len()
        );

        // ── roof ──
        let buf = touched_buffer(256 << 20);
        let window = Duration::from_millis(150);
        let one = read_rate(buf.clone(), &[first.primary()], window, 2).unwrap();
        let fast_sets: Vec<u32> = cores.iter().take(fast).map(|c| c.primary()).collect();
        let all = read_rate(buf, &fast_sets, window, 2).unwrap();
        println!(
            "roof: DRAM 1 core {:.1} GB/s, {fast} performance cores {:.1} GB/s",
            one.gb_per_s(),
            all.gb_per_s()
        );

        let mut seed = 7;
        let base: Vec<Slot> = (0..2).map(|_| slot(&mut seed)).collect();
        let corpus: Vec<Slot> = (0..CORPUS).map(|i| base[i % 2].clone()).collect();
        let expert_bytes = corpus[0].0.len();
        let x: Vec<f32> = (0..16 * HIDDEN).map(|_| lcg(&mut seed)).collect();
        let act16 = ActTiles::quantize(&x, 16, HIDDEN);
        let expert = |i: usize| -> Expert {
            let (b, off) = &corpus[i % CORPUS];
            Expert::from_slot(b, *off, HIDDEN, FF, Ko::Q3)
        };

        // ── hot: 64 rows × 2560, in L2 — cycles per chunk (8 rows × 128 K)
        // at the clock measured just before, so throttling cancels ──
        let ghz = core_ghz();
        println!(
            "\nhot (L2) matvec, one core at {ghz:.2} GHz, cycles per chunk (GB/s of weights):"
        );
        println!("  format isa        tokens 1          2          4          8");
        let mut seed4 = 3;
        let q4: Vec<f32> = (0..64 * HIDDEN).map(|_| lcg(&mut seed4) * 0.05).collect();
        let q4 = quantize_ko(&q4, 64, HIDDEN, GgmlDType::Q4_KO);
        for (ko, bytes) in [(Ko::Q3, &corpus[0].0), (Ko::Q4, &q4)] {
            let hot = KoMatrix::new(bytes, 64, HIDDEN, ko);
            let chunks = (HIDDEN / 128) * 8;
            let hot_bytes = chunks * ko.chunk_bytes();
            for isa in isas() {
                let mut line = format!("  {:<6} {:<10}", format!("{ko:?}"), format!("{isa:?}"));
                for tokens in [1usize, 2, 4, 8] {
                    let act = ActTiles::quantize(&x[..tokens * HIDDEN], tokens, HIDDEN);
                    let pa = Permuted::new(&act, ko);
                    let mut out = vec![0f32; tokens * 64];
                    let mut best = f64::MAX;
                    for _ in 0..400 {
                        let t = Instant::now();
                        matvec(isa, &hot, &act, &pa, 0..64, &mut out);
                        best = best.min(t.elapsed().as_secs_f64());
                        black_box(&out);
                    }
                    line += &format!(
                        "{:>6.0} ({:>4.1})",
                        best * ghz * 1e9 / chunks as f64,
                        hot_bytes as f64 / best / 1e9
                    );
                }
                println!("{line}");
            }
        }

        // ── cold: a whole gate projection per call, rotating the corpus ──
        println!("\ncold matvec, one core, gate 640×2560, GB/s of weights (median):");
        let gate_bytes = FF * HIDDEN / 128 / 8 * Ko::Q3.chunk_bytes();
        for tokens in [1usize, 4] {
            let act = ActTiles::quantize(&x[..tokens * HIDDEN], tokens, HIDDEN);
            let pa = Permuted::new(&act, Ko::Q3);
            let mut out = vec![0f32; tokens * FF];
            let mut times = Vec::new();
            for i in 0..3 * CORPUS {
                let e = expert(i);
                let t = Instant::now();
                matvec(isa, &e.gate, &act, &pa, 0..FF, &mut out);
                times.push(t.elapsed().as_secs_f64());
                black_box(&out);
            }
            println!(
                "  tokens {tokens}: {:.1} GB/s",
                gate_bytes as f64 / median(times) / 1e9
            );
        }

        // ── expert: forward on one thread, DRAM-cold ──
        println!("\none expert, one core, DRAM-cold (median):");
        for tokens in [1usize, 4] {
            let act = ActTiles::quantize(&x[..tokens * HIDDEN], tokens, HIDDEN);
            let mut out = vec![0f32; tokens * HIDDEN];
            let mut times = Vec::new();
            for i in 0..2 * CORPUS {
                let e = expert(i);
                let t = Instant::now();
                forward(isa, &e, &act, &mut out);
                times.push(t.elapsed().as_secs_f64());
                black_box(&out);
            }
            let m = median(times);
            println!(
                "  tokens {tokens}: {:.0} µs, {:.1} GB/s",
                m * 1e6,
                expert_bytes as f64 / m / 1e9
            );
        }

        // ── serial: the steps one participant runs while an expert's other
        // tasks wait — its input's permute, and its down input's quantize and
        // permute ──
        println!("\nserial steps per expert, one core, µs (median of 200):");
        println!("  tokens   permute gate input   quantize + permute down input");
        let h: Vec<f32> = (0..16 * FF).map(|_| lcg(&mut seed)).collect();
        for tokens in [1usize, 4, 16] {
            let act = ActTiles::quantize(&x[..tokens * HIDDEN], tokens, HIDDEN);
            let (mut perm, mut down) = (Vec::new(), Vec::new());
            for _ in 0..200 {
                let t = Instant::now();
                black_box(Permuted::new(&act, Ko::Q3));
                perm.push(t.elapsed().as_secs_f64());
                let t = Instant::now();
                let ha = ActTiles::quantize(&h[..tokens * FF], tokens, FF);
                black_box(Permuted::new(&ha, Ko::Q3));
                down.push(t.elapsed().as_secs_f64());
            }
            println!(
                "  {tokens:>6}   {:>18.2}   {:>29.2}",
                median(perm) * 1e6,
                median(down) * 1e6
            );
        }

        // ── layer: the production pool ──
        let workers: Vec<u32> = rest.iter().map(|c| c.primary()).collect();
        let pool = Pool::new(
            &workers,
            fast - 1,
            PoolWaits::new(Duration::from_micros(200)),
        )
        .unwrap();
        // Held warm ahead of every run, as the service holds it through
        // inference (`CpuService::wake`).
        let warmth = pool.warmth();
        warmth.warm(COOL_DOWN);
        for _ in 0..200 {
            pool.run(&|_| {});
        }
        println!(
            "\nlayer on the pool, DRAM-cold, workers awake (median of 40):\n  \
             experts×tokens  participants      µs   GB/s   % of roof   join µs   busy %"
        );
        let mut next = 0usize;
        for (k, n) in LAYERS {
            for parts in [4usize, fast] {
                let before = pool_stats();
                let mut times = Vec::new();
                for _ in 0..40 {
                    let experts: Vec<ExpertWork> = (0..k)
                        .map(|j| {
                            let e = expert(next);
                            next += 1;
                            ExpertWork {
                                expert: e,
                                tokens: (0..n).collect(),
                                weights: vec![1.0; n],
                                outputs: (j * n..(j + 1) * n).collect(),
                            }
                        })
                        .collect();
                    let work = LayerWork {
                        act: &act16,
                        experts,
                        rows: k * n,
                    };
                    warmth.warm(COOL_DOWN);
                    let t = Instant::now();
                    black_box(run_layer_on(&pool, isa, &work, parts));
                    times.push(t.elapsed().as_secs_f64());
                }
                let m = median(times);
                let gbs = (k * expert_bytes) as f64 / m / 1e9;
                // The pool's own split over these runs: the last
                // participant's entry, and the share of time in the job that
                // was spent computing rather than waiting on the others.
                let [_, join, work, in_job] = pool_stats().since(&before).per_run();
                println!(
                    "  {k:>3} × {n:<3}       {parts:>12}  {:>7.0}  {gbs:>5.1}  {:>9.0}%  {join:>8.1}  {:>7.0}",
                    m * 1e6,
                    100.0 * gbs / all.gb_per_s(),
                    100.0 * work / in_job.max(1e-9)
                );
            }
        }

        // ── layer, L3-hot: the same experts every run — what the schedule
        // costs once DRAM is out of the picture ──
        println!(
            "\nlayer on the pool, L3-hot, workers awake (median of 40):\n  \
             experts×tokens  participants      µs   GB/s"
        );
        for (k, n) in [(1usize, 1usize), (3, 1), (5, 1), (5, 4)] {
            let mut times = Vec::new();
            for _ in 0..40 {
                let experts: Vec<ExpertWork> = (0..k)
                    .map(|j| ExpertWork {
                        expert: expert(j),
                        tokens: (0..n).collect(),
                        weights: vec![1.0; n],
                        outputs: (j * n..(j + 1) * n).collect(),
                    })
                    .collect();
                let work = LayerWork {
                    act: &act16,
                    experts,
                    rows: k * n,
                };
                warmth.warm(COOL_DOWN);
                let t = Instant::now();
                black_box(run_layer_on(&pool, isa, &work, fast));
                times.push(t.elapsed().as_secs_f64());
            }
            let m = median(times);
            println!(
                "  {k:>3} × {n:<3}       {fast:>12}  {:>7.0}  {:>5.1}",
                m * 1e6,
                (k * expert_bytes) as f64 / m / 1e9
            );
        }
    }
}
