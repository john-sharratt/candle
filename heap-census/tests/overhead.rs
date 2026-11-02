//! **The census costs almost nothing**: sixteen threads allocating and freeing
//! a mix of sizes through [`Counting`] run within [`BUDGET`] of the same work
//! through [`System`] alone.
//!
//! A timing gate, so it is ignored by default and run on its own, in release,
//! on a quiet machine: `cargo test --release -p heap-census --test overhead
//! -- --ignored --nocapture`.
//!
//! Both allocators are called directly rather than installed, so one binary
//! times both on the same machine in the same minute; the counting is the same
//! code either way.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::ptr::null_mut;
use std::sync::Barrier;
use std::thread;
use std::time::{Duration, Instant};

use heap_census::Counting;

const THREADS: usize = 16;
const PAIRS: usize = 400_000;
/// A small object, a page and a replicated chunk record's size.
const SIZES: [usize; 3] = [32, 4_096, 30_000];
/// The most the counted run may take, as a multiple of the plain one.
const BUDGET: f64 = 1.05;
/// Runs of each, alternated so neither gets the warmer machine.
const RUNS: usize = 5;

/// Every thread allocates and frees `PAIRS` blocks through `a`, keeping eight
/// live at a time so frees are not always of the block just made.
fn run<A: GlobalAlloc + Sync>(a: &A) -> Duration {
    let start = Barrier::new(THREADS + 1);
    let t = thread::scope(|s| {
        for th in 0..THREADS {
            let start = &start;
            s.spawn(move || {
                let mut live: [(*mut u8, Layout); 8] = [(null_mut(), Layout::new::<u8>()); 8];
                start.wait();
                for i in 0..PAIRS {
                    let size = SIZES[(i + th) % SIZES.len()];
                    let layout = Layout::from_size_align(size, 8).unwrap();
                    // SAFETY: each block is freed once, with the layout it was
                    // allocated with.
                    unsafe {
                        let p = a.alloc(layout);
                        black_box(p);
                        let slot = &mut live[i % 8];
                        if !slot.0.is_null() {
                            a.dealloc(slot.0, slot.1);
                        }
                        *slot = (p, layout);
                    }
                }
                for (p, l) in live {
                    if !p.is_null() {
                        // SAFETY: as above.
                        unsafe { a.dealloc(p, l) };
                    }
                }
            });
        }
        start.wait();
        Instant::now()
    });
    t.elapsed()
}

#[test]
#[ignore = "a timing gate: run on its own, in release, on a quiet machine"]
fn the_census_costs_almost_nothing() {
    // Warm both, so the first timed run is not the one that grew the heap.
    run(&System);
    run(&Counting);
    let (mut plain, mut counted) = (Vec::new(), Vec::new());
    for _ in 0..RUNS {
        plain.push(run(&System));
        counted.push(run(&Counting));
    }
    plain.sort();
    counted.sort();
    let (p, c) = (plain[RUNS / 2], counted[RUNS / 2]);
    let ops = (THREADS * PAIRS * 2) as f64;
    let ratio = c.as_secs_f64() / p.as_secs_f64();
    eprintln!(
        "System {:.1} ns/op, Counting {:.1} ns/op, ratio {ratio:.3} (median of {RUNS})",
        p.as_nanos() as f64 / ops,
        c.as_nanos() as f64 / ops,
    );
    assert!(
        ratio <= BUDGET,
        "the census costs {ratio:.3}× the plain allocator"
    );
}
