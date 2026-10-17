//! The pool's waits under decode's own pattern, on the production cores — the
//! harness the hot-spin counts (`backoff`) are tuned on before the engine.
//!
//! A "device" thread rings a doorbell word the way the export kernel does: a
//! wave of 48 layers ~1.5 ms apart, waves back to back, then an idle spell
//! past the cool-down, with the forward's `warm` ahead of each wave as the
//! engine calls it. A "service" thread on the caller's core waits for each
//! doorbell, runs a job on the pool's performance-core participants — a few
//! tens of microseconds of work each, unevenly, like a decode layer's tasks —
//! and waits for its participants as `run_on` does.
//!
//! Per setting it reports the latencies the waits decide — doorbell to the
//! service starting, the job's start to its last participant joining, the
//! last participant finishing to the caller returning — as p50/p99, and the
//! cores the process kept busy (its CPU time over wall time), which is what
//! the package's clock pays for.

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::models::expert_lre::cpu_pool::backoff::{
        Backoff, COOL_DOWN, INTER_LAYER_SPINS, INTRA_LAYER_SPINS, LOW_POWER_CYCLES,
    };
    use crate::models::expert_lre::cpu_pool::pin::{pin_current_thread, raise_priority};
    use crate::models::expert_lre::cpu_pool::pool::{Pool, PoolWaits};
    use crate::models::expert_lre::cpu_pool::topology::{
        performance_cores, physical_cores, pool_cores,
    };

    const LAYERS: usize = 48;
    const LAYER_GAP: Duration = Duration::from_micros(1_500);
    const WAVES: usize = 6;
    /// Work per participant: 40 µs, plus 8 µs per index — the uneven tail a
    /// layer's last tasks leave.
    const WORK_US: u64 = 40;
    const WORK_STEP_US: u64 = 8;

    fn busy(d: Duration) {
        let t = Instant::now();
        while t.elapsed() < d {
            std::hint::spin_loop();
        }
    }

    /// The process's CPU time so far, user and kernel.
    fn cpu_time() -> Duration {
        use windows_sys::Win32::Foundation::FILETIME;
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
        let zero = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let (mut c, mut e, mut k, mut u) = (zero, zero, zero, zero);
        // SAFETY: the current process's pseudo-handle and four out-params.
        unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
        let ft = |f: FILETIME| ((f.dwHighDateTime as u64) << 32 | f.dwLowDateTime as u64) * 100;
        Duration::from_nanos(ft(k) + ft(u))
    }

    fn pct(v: &mut [f64], q: f64) -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        v[((v.len() - 1) as f64 * q) as usize]
    }

    struct Sample {
        start_us: Vec<f64>,
        join_us: Vec<f64>,
        done_us: Vec<f64>,
        cores_busy: f64,
        first_after_idle_us: f64,
    }

    /// One setting through `WAVES` waves, then an idle spell and one more
    /// wave; returns its latencies and cost.
    fn run(waits: PoolWaits, doorbell_hot: u32) -> Sample {
        let cores = pool_cores(&physical_cores().unwrap(), 1);
        let fast = cores
            .iter()
            .take_while(|c| c.efficiency_class == cores[0].efficiency_class)
            .count();
        let caller = cores[0].primary();
        let workers: Vec<u32> = cores[1..].iter().map(|c| c.primary()).collect();
        let pool = Pool::new(&workers, fast - 1, waits).unwrap();
        let warmth = pool.warmth();
        let doorbell = Arc::new(AtomicU64::new(0));
        // When each layer was rung, ns after `origin`.
        let origin = Instant::now();
        let rung = Arc::new(
            (0..(WAVES + 1) * LAYERS)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>(),
        );
        let total = (WAVES + 1) * LAYERS;

        // The forward thread's core: the performance core the pool leaves out.
        let reserved = performance_cores(&physical_cores().unwrap())[0].primary();
        let device = {
            let (doorbell, rung, warmth) = (doorbell.clone(), rung.clone(), warmth.clone());
            thread::spawn(move || {
                pin_current_thread(reserved).unwrap();
                let mut n = 0usize;
                for wave in 0..=WAVES {
                    if wave == WAVES {
                        // Idle past the cool-down: the next wave finds the pool
                        // parked, as a new request does.
                        thread::sleep(COOL_DOWN + Duration::from_millis(150));
                    }
                    for _ in 0..LAYERS {
                        // The forward's `wake`, run ahead of the device.
                        warmth.warm(COOL_DOWN);
                        busy(Duration::from_micros(150));
                        rung[n].store(origin.elapsed().as_nanos() as u64, Ordering::Release);
                        n += 1;
                        doorbell.store(n as u64, Ordering::Release);
                        busy(LAYER_GAP);
                    }
                    // Between waves: the step's tail, sampling and the next
                    // step's first launches.
                    busy(Duration::from_millis(3));
                }
            })
        };

        let (mut start_us, mut join_us, mut done_us) = (Vec::new(), Vec::new(), Vec::new());
        let mut first_after_idle_us = 0.0;
        let cpu0 = cpu_time();
        let wall0 = Instant::now();
        thread::scope(|s| {
            s.spawn(|| {
                pin_current_thread(caller).unwrap();
                let _ = raise_priority();
                let mut last = 0u64;
                let mut backoff = Backoff::new(doorbell_hot);
                while (last as usize) < total {
                    let now = doorbell.load(Ordering::Acquire);
                    if now == last {
                        if warmth.is_warm() {
                            // SAFETY: a live atomic.
                            unsafe { backoff.wait_u64(doorbell.as_ptr(), last, LOW_POWER_CYCLES) };
                        } else {
                            thread::park_timeout(Duration::from_micros(500));
                        }
                        continue;
                    }
                    backoff.reset();
                    last = now;
                    let at = origin.elapsed().as_nanos() as u64;
                    let rang = rung[(now - 1) as usize].load(Ordering::Acquire);
                    let lat = (at.saturating_sub(rang)) as f64 / 1e3;
                    // The first wave warms the threads and pages up: not counted.
                    let counted = now as usize > LAYERS;
                    if now as usize == WAVES * LAYERS + 1 {
                        first_after_idle_us = lat;
                    } else if counted {
                        start_us.push(lat);
                    }
                    let t0 = Instant::now();
                    let joined = AtomicU64::new(0);
                    let finished = AtomicU64::new(0);
                    pool.run_on(fast, &|i| {
                        joined.fetch_max(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        busy(Duration::from_micros(WORK_US + WORK_STEP_US * i as u64));
                        finished.fetch_max(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    });
                    let back = t0.elapsed().as_nanos() as u64;
                    if counted {
                        join_us.push(joined.load(Ordering::Relaxed) as f64 / 1e3);
                        done_us.push(
                            back.saturating_sub(finished.load(Ordering::Relaxed)) as f64 / 1e3,
                        );
                    }
                }
            });
        });
        device.join().unwrap();
        let wall = wall0.elapsed().as_secs_f64();
        Sample {
            start_us,
            join_us,
            done_us,
            cores_busy: (cpu_time() - cpu0).as_secs_f64() / wall,
            first_after_idle_us,
        }
    }

    /// `cargo test --release -p candle-transformers --features cuda --lib wait_patterns -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement, ~1 min; read its output"]
    fn wait_patterns() {
        let spin = Duration::from_micros(200);
        let settings: Vec<(&str, PoolWaits, u32)> = vec![
            ("production", PoolWaits::new(spin), INTER_LAYER_SPINS),
            (
                "worker hot 64",
                PoolWaits {
                    worker_hot: 64,
                    ..PoolWaits::new(spin)
                },
                INTER_LAYER_SPINS,
            ),
            (
                "worker hot 4096",
                PoolWaits {
                    worker_hot: 4096,
                    ..PoolWaits::new(spin)
                },
                INTER_LAYER_SPINS,
            ),
            (
                "worker spins always",
                PoolWaits {
                    worker_hot: u32::MAX,
                    ..PoolWaits::new(spin)
                },
                INTER_LAYER_SPINS,
            ),
            ("production again", PoolWaits::new(spin), INTER_LAYER_SPINS),
            (
                "caller hot 0",
                PoolWaits {
                    caller_hot: 0,
                    ..PoolWaits::new(spin)
                },
                INTER_LAYER_SPINS,
            ),
            (
                "caller hot 256",
                PoolWaits {
                    caller_hot: INTRA_LAYER_SPINS,
                    ..PoolWaits::new(spin)
                },
                INTER_LAYER_SPINS,
            ),
            (
                "caller hot 4096",
                PoolWaits {
                    caller_hot: 4096,
                    ..PoolWaits::new(spin)
                },
                INTER_LAYER_SPINS,
            ),
            ("doorbell hot 256", PoolWaits::new(spin), INTRA_LAYER_SPINS),
            ("doorbell hot 4096", PoolWaits::new(spin), 4096),
        ];
        println!(
            "{:<20} {:>22} {:>22} {:>22} {:>11} {:>10}",
            "setting",
            "start p50/p90/p99",
            "join p50/p90/p99",
            "done p50/p90/p99",
            "after idle",
            "cores busy"
        );
        let three =
            |v: &mut Vec<f64>| format!("{:.1}/{:.1}/{:.1}", pct(v, 0.5), pct(v, 0.9), pct(v, 0.99));
        for (name, waits, doorbell_hot) in settings {
            let mut s = run(waits, doorbell_hot);
            println!(
                "{name:<20} {:>22} {:>22} {:>22} {:>9.1}µs {:>10.2}",
                three(&mut s.start_us),
                three(&mut s.join_us),
                three(&mut s.done_us),
                s.first_after_idle_us,
                s.cores_busy,
            );
        }
    }
}
