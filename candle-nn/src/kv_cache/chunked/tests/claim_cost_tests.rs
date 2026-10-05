//! What claiming a prefill's chunks costs, at a wide model's geometry.
//!
//! A prefill wave claims every chunk it will write before anything computes
//! (`wave_admit`), once per layer per sequence. At Llama-2's 32 KV heads a chunk is
//! 256 band slots, so the claim is the one host cost that grows with both width
//! and depth. These are measurements, not assertions: they print what a claim and
//! a release cost so a change to the allocator is judged against a number.
#![cfg(feature = "cuda")]

use std::time::Instant;

use candle::{DType, Device};

use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
use crate::kv_cache::ChunkedKvBacking;

const N_KV_HEAD: usize = 32;
const HEAD_DIM: usize = 128;
const SEQUENCES: usize = 48;
/// A wave of ~218 prompt tokens is seven 32-token chunks.
const CHUNKS_PER_SEQUENCE: usize = 7;

#[test]
#[ignore = "a measurement, not an assertion"]
fn claiming_a_chunk_costs() {
    let _gpu = gpu_serial();
    let dev = Device::new_cuda(0).unwrap();
    let backing =
        ChunkedKvBacking::new(SEQUENCES, N_KV_HEAD, HEAD_DIM, DType::BF16, &dev, 512).unwrap();
    let n = SEQUENCES * CHUNKS_PER_SEQUENCE;

    // Cold: the first claims create the arenas they land in.
    let t = Instant::now();
    let cold: Vec<_> = (0..n)
        .map(|_| backing.alloc_block_chunks(0, 0).unwrap())
        .collect();
    let cold_us = t.elapsed().as_micros() as f64 / n as f64;
    let t = Instant::now();
    drop(cold);
    let drop_us = t.elapsed().as_micros() as f64 / n as f64;

    println!("claim per chunk ({N_KV_HEAD} heads): cold {cold_us:.1} us; release {drop_us:.1} us");

    // The call a prefill's admit makes: one `ensure_for_batch_entries` per sequence,
    // asking for the wave's tokens plus one.
    for round in 1..=3 {
        let seqs: Vec<usize> = (0..SEQUENCES)
            .map(|_| backing.alloc_sequence().unwrap())
            .collect();
        let t = Instant::now();
        for &s in &seqs {
            backing
                .ensure_for_batch_entries(&[(s, 0)], CHUNKS_PER_SEQUENCE * 32 - 10 + 1)
                .unwrap();
        }
        let per_chunk = t.elapsed().as_micros() as f64 / n as f64;
        let t = Instant::now();
        for &s in &seqs {
            backing.free_sequence(s).unwrap();
        }
        let free_us = t.elapsed().as_micros() as f64 / n as f64;
        println!("  admit round {round}: {per_chunk:.1} us per chunk; free {free_us:.1} us");
    }

    // Warm: the same claim, again and again, over arenas that already exist.
    for round in 1..=5 {
        let t = Instant::now();
        let warm: Vec<_> = (0..n)
            .map(|_| backing.alloc_block_chunks(0, 0).unwrap())
            .collect();
        let warm_us = t.elapsed().as_micros() as f64 / n as f64;
        let t = Instant::now();
        drop(warm);
        let warm_drop_us = t.elapsed().as_micros() as f64 / n as f64;
        println!("  round {round}: claim {warm_us:.1} us per chunk, release {warm_drop_us:.1} us");
    }
}
