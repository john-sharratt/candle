//! The DRAM read rate a set of pinned workers sustains — `B_d` in the split
//! (`docs/cpu_expert_compute.md` §3.2, §5.3).
//!
//! The figure the split needs is what the *pool's own layout* reads, not a
//! datasheet peak: one worker per chosen core, pinned, each streaming its own
//! slice of a buffer far larger than the last-level cache. Calibration runs
//! this at load; the `#[ignore]`d sweep below runs it across layouts to choose
//! the pool's cores.

use std::hint::black_box;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use super::pin::pin_current_thread;

/// Words a worker reads between looks at the stop flag: 1 MiB.
const CHUNK_WORDS: usize = 1 << 17;

/// The aggregate read rate of a set of workers over one window.
#[derive(Debug, Clone, Copy)]
pub struct ReadRate {
    /// Σ over workers of each worker's own bytes ÷ its own seconds.
    pub bytes_per_s: f64,
}

impl ReadRate {
    pub fn gb_per_s(&self) -> f64 {
        self.bytes_per_s / 1e9
    }
}

/// Runs one worker per CPU set in `cpu_sets` for `window`, each streaming its
/// own slice of `buf` round and round, and returns the best of `passes`.
///
/// Each worker times itself and counts its own bytes, and the rate is the sum
/// of the workers' rates. A wall clock taken by a coordinating thread would be
/// wrong exactly when it matters: with a worker on every logical processor the
/// coordinator is descheduled and starts its clock late. A fixed window also
/// keeps a slow core from setting the time for the fast ones, as equal slices
/// timed to the last finisher would. The buffer must be far larger than the
/// last-level cache, or this measures the cache.
pub fn read_rate(
    buf: Arc<Vec<u64>>,
    cpu_sets: &[u32],
    window: Duration,
    passes: usize,
) -> io::Result<ReadRate> {
    assert!(!cpu_sets.is_empty() && passes > 0);
    let workers = cpu_sets.len();
    let per = (buf.len() / workers) / CHUNK_WORDS * CHUNK_WORDS;
    assert!(per >= CHUNK_WORDS, "buffer too small for {workers} workers");
    let mut best: Option<ReadRate> = None;
    for _ in 0..passes {
        let start = Arc::new(Barrier::new(workers + 1));
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::with_capacity(workers);
        for (w, &cpu_set) in cpu_sets.iter().enumerate() {
            let (buf, start, stop) = (buf.clone(), start.clone(), stop.clone());
            handles.push(thread::spawn(move || -> io::Result<(u64, f64, u64)> {
                let pinned = pin_current_thread(cpu_set);
                start.wait();
                pinned?;
                let slice = &buf[w * per..(w + 1) * per];
                let t0 = Instant::now();
                let (mut bytes, mut fold, mut at) = (0u64, 0u64, 0usize);
                while !stop.load(Ordering::Relaxed) {
                    fold ^= stream(&slice[at..at + CHUNK_WORDS]);
                    bytes += (CHUNK_WORDS * 8) as u64;
                    at = (at + CHUNK_WORDS) % per;
                }
                Ok((bytes, t0.elapsed().as_secs_f64(), fold))
            }));
        }
        start.wait();
        thread::sleep(window);
        stop.store(true, Ordering::Relaxed);
        let mut bytes_per_s = 0.0;
        for h in handles {
            let (bytes, secs, fold) = h.join().expect("bandwidth worker panicked")?;
            black_box(fold);
            bytes_per_s += bytes as f64 / secs;
        }
        let rate = ReadRate { bytes_per_s };
        if best.is_none_or(|b| rate.bytes_per_s > b.bytes_per_s) {
            best = Some(rate);
        }
    }
    Ok(best.expect("passes > 0"))
}

/// A buffer of `bytes` with every page touched, so the measurement reads DRAM
/// rather than faulting pages in.
pub fn touched_buffer(bytes: usize) -> Arc<Vec<u64>> {
    let words = bytes / 8;
    Arc::new((0..words as u64).collect())
}

/// Reads every word of `s` and folds them so the loads cannot be elided.
fn stream(s: &[u64]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was just detected.
        return unsafe { stream_avx2(s) };
    }
    s.chunks(8)
        .fold(0u64, |acc, c| c.iter().fold(acc, |a, &w| a ^ w))
}

/// Four independent 256-bit xor chains, so the loop is bound by loads, not by
/// one dependency chain.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn stream_avx2(s: &[u64]) -> u64 {
    use std::arch::x86_64::{
        __m256i, _mm256_extract_epi64, _mm256_loadu_si256, _mm256_setzero_si256, _mm256_xor_si256,
    };
    let p = s.as_ptr().cast::<__m256i>();
    let vecs = s.len() / 4;
    let (mut a, mut b, mut c, mut d) = (
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
    );
    let mut i = 0;
    while i + 4 <= vecs {
        a = _mm256_xor_si256(a, _mm256_loadu_si256(p.add(i)));
        b = _mm256_xor_si256(b, _mm256_loadu_si256(p.add(i + 1)));
        c = _mm256_xor_si256(c, _mm256_loadu_si256(p.add(i + 2)));
        d = _mm256_xor_si256(d, _mm256_loadu_si256(p.add(i + 3)));
        i += 4;
    }
    let x = _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d));
    let mut acc = (_mm256_extract_epi64::<0>(x)
        ^ _mm256_extract_epi64::<1>(x)
        ^ _mm256_extract_epi64::<2>(x)
        ^ _mm256_extract_epi64::<3>(x)) as u64;
    for &w in &s[i * 4..] {
        acc ^= w;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::expert_lre::cpu_pool::topology::{performance_cores, physical_cores};

    /// The AVX2 fold and the scalar fold agree, including a ragged tail.
    #[test]
    fn stream_folds_every_word() {
        let v: Vec<u64> = (0..1003u64)
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        let expect = v.iter().fold(0u64, |a, &w| a ^ w);
        assert_eq!(stream(&v), expect);
    }

    /// Measurement 1 (`docs/cpu_expert_compute.md` §8): DRAM read bandwidth
    /// with the pool's layout — 1..N performance cores one worker each, then
    /// every physical core, then every logical processor. 2 GiB buffer, 300 ms
    /// windows, best of five. Run alone, with the GPU idle and then under load:
    /// `cargo test --release -p candle-transformers dram_read_sweep -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dram_read_sweep() {
        let cores = physical_cores().unwrap();
        let perf = performance_cores(&cores);
        println!(
            "{} physical cores ({} performance); classes {:?}",
            cores.len(),
            perf.len(),
            cores.iter().map(|c| c.efficiency_class).collect::<Vec<_>>()
        );
        let buf = touched_buffer(2 << 30);
        let mut layouts: Vec<(String, Vec<u32>)> = (1..=perf.len())
            .map(|n| {
                (
                    format!("{n} P"),
                    perf[..n].iter().map(|c| c.primary()).collect(),
                )
            })
            .collect();
        layouts.push((
            "all physical".into(),
            cores.iter().map(|c| c.primary()).collect(),
        ));
        layouts.push((
            "all logical".into(),
            cores.iter().flat_map(|c| c.cpu_sets.clone()).collect(),
        ));
        for (name, sets) in layouts {
            let r = read_rate(buf.clone(), &sets, Duration::from_millis(300), 5).unwrap();
            println!(
                "{name:>14}: {:6.1} GB/s ({} workers)",
                r.gb_per_s(),
                sets.len()
            );
        }
    }
}
