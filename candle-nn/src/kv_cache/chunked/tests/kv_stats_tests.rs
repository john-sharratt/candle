//! KV-cache chunk statistics generator.
//!
//! Reads the binary dump produced by `test_dump_kv_cache_data`, computes a
//! comprehensive set of per-chunk statistics for both K and V caches, and
//! writes the results to:
//!
//!   data/kv_cache_stats.csv   — human-readable, one row per layer/chunk/K|V
//!   data/kv_cache_stats.bin   — binary, for fast loading in future unit tests
//!
//! Run with:
//!   cargo test --release --lib --package candle-nn \
//!     kv_stats_tests::test_generate_kv_stats -- --nocapture
//!
//! Binary stats format (little-endian):
//!   magic        : [u8; 8]  = b"KVSTATS\0"
//!   version      : u32      = 2
//!   num_layers   : u32
//!   n_kv_head    : u32
//!   chunk_size   : u32
//!   head_dim     : u32
//!   num_tokens   : u32
//!   tokens       : [u32; num_tokens]
//!   Per layer (num_layers total):
//!     num_chunks : u32
//!     Per chunk:
//!       block_idx        : u32
//!       token_start      : u32   (first sequence position in this chunk)
//!       k_stats, v_stats : ChunkStats (12 f32 each, see struct below)
//!
//! ChunkStats (12 × f32):
//!   amax          — max |x|
//!   rms           — sqrt(mean(x²))
//!   mean_abs      — mean(|x|)
//!   std           — population std-dev
//!   kurtosis      — excess kurtosis (0 for Gaussian, +ve for heavy tails)
//!   sparsity_pct  — % of values with |x| < 0.01 * amax
//!   q8_cos_mean   — mean cosine-distance Q8_0 round-trip across 32-elem sub-blocks
//!   q8_cos_max    — max  cosine-distance Q8_0
//!   q4_cos_mean   — mean cosine-distance Q4_0
//!   q4_cos_max    — max  cosine-distance Q4_0
//!   q3_cos_mean   — mean cosine-distance Q3_0
//!   q2_cos_mean   — mean cosine-distance Q2_0

use super::dump_reader::load_dump;
use std::io::Write;

const DUMP_REL_PATH: &str = "src/kv_cache/chunked/tests/data/qwen3-kv-data.bin";
const QWEN3_DUMP_REL_PATH: &str = "src/kv_cache/chunked/tests/data/qwen3-kv-data.bin";
const LLAMA_DUMP_REL_PATH: &str = "src/kv_cache/chunked/tests/data/llama-kv-data.bin";
const STATS_BIN_PATH: &str = "src/kv_cache/chunked/tests/data/kv_cache_stats.bin";
const STATS_CSV_PATH: &str = "src/kv_cache/chunked/tests/data/kv_cache_stats.csv";

/// Sub-block size matching the CUDA selection kernel constant.
const BLOCK: usize = 32;

// ---------------------------------------------------------------------------
// Statistics struct
// ---------------------------------------------------------------------------

/// Per-chunk statistics for one component (K or V).
#[derive(Debug, Clone, Copy)]
struct ChunkStats {
    amax: f32,
    rms: f32,
    mean_abs: f32,
    std: f32,
    kurtosis: f32,
    sparsity_pct: f32,
    q8_cos_mean: f32,
    q8_cos_max: f32,
    q4_cos_mean: f32,
    q4_cos_max: f32,
    q3_cos_mean: f32,
    q2_cos_mean: f32,
}

impl ChunkStats {
    fn as_f32_array(&self) -> [f32; 12] {
        [
            self.amax,
            self.rms,
            self.mean_abs,
            self.std,
            self.kurtosis,
            self.sparsity_pct,
            self.q8_cos_mean,
            self.q8_cos_max,
            self.q4_cos_mean,
            self.q4_cos_max,
            self.q3_cos_mean,
            self.q2_cos_mean,
        ]
    }

    fn csv_headers() -> &'static str {
        "token_start,amax,rms,mean_abs,std,kurtosis,sparsity_pct,\
         q8_cos_mean,q8_cos_max,q4_cos_mean,q4_cos_max,q3_cos_mean,q2_cos_mean"
    }

    fn to_csv_with_token_start(&self, token_start: usize) -> String {
        format!("{},{}", token_start, self.to_csv(),)
    }

    fn to_csv(&self) -> String {
        format!(
            "{:.6},{:.6},{:.6},{:.6},{:.4},{:.2},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8}",
            self.amax,
            self.rms,
            self.mean_abs,
            self.std,
            self.kurtosis,
            self.sparsity_pct,
            self.q8_cos_mean,
            self.q8_cos_max,
            self.q4_cos_mean,
            self.q4_cos_max,
            self.q3_cos_mean,
            self.q2_cos_mean,
        )
    }
}

// ---------------------------------------------------------------------------
// Quantisation helpers (same as kv_selection_tests)
// ---------------------------------------------------------------------------

fn round_trip_q8_0(b: &[f32; BLOCK]) -> [f32; BLOCK] {
    let amax = b.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return [0.0; BLOCK];
    }
    let s = amax / 127.0;
    let mut o = [0.0f32; BLOCK];
    for (i, &x) in b.iter().enumerate() {
        o[i] = ((x / s).round().clamp(-128.0, 127.0) as i8) as f32 * s;
    }
    o
}

fn round_trip_q4_0(b: &[f32; BLOCK]) -> [f32; BLOCK] {
    let amax = b.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return [0.0; BLOCK];
    }
    let s = amax / 7.0;
    let mut o = [0.0f32; BLOCK];
    for (i, &x) in b.iter().enumerate() {
        o[i] = ((x / s).round().clamp(-8.0, 7.0) as i8) as f32 * s;
    }
    o
}

fn round_trip_q3_0(b: &[f32; BLOCK]) -> [f32; BLOCK] {
    let amax = b.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return [0.0; BLOCK];
    }
    let s = amax / 3.0;
    let mut o = [0.0f32; BLOCK];
    for (i, &x) in b.iter().enumerate() {
        o[i] = ((x / s).round().clamp(-4.0, 3.0) as i8) as f32 * s;
    }
    o
}

fn round_trip_q2_0(b: &[f32; BLOCK]) -> [f32; BLOCK] {
    let amax = b.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return [0.0; BLOCK];
    }
    let s = amax;
    let mut o = [0.0f32; BLOCK];
    for (i, &x) in b.iter().enumerate() {
        o[i] = ((x / s).round().clamp(-2.0, 1.0) as i8) as f32 * s;
    }
    o
}

fn cosine_distance(a: &[f32; BLOCK], b: &[f32; BLOCK]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    (1.0 - dot / denom).max(0.0)
}

// ---------------------------------------------------------------------------
// Core stats computation
// ---------------------------------------------------------------------------

fn compute_stats(data: &[f32]) -> ChunkStats {
    let n = data.len() as f64;
    if n == 0.0 {
        return ChunkStats {
            amax: 0.0,
            rms: 0.0,
            mean_abs: 0.0,
            std: 0.0,
            kurtosis: 0.0,
            sparsity_pct: 100.0,
            q8_cos_mean: 0.0,
            q8_cos_max: 0.0,
            q4_cos_mean: 0.0,
            q4_cos_max: 0.0,
            q3_cos_mean: 0.0,
            q2_cos_mean: 0.0,
        };
    }

    // Scalar stats
    let amax = data.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    let sum_sq: f64 = data.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let sum: f64 = data.iter().map(|&x| x as f64).sum();
    let sum_abs: f64 = data.iter().map(|&x| (x as f64).abs()).sum();
    let rms = (sum_sq / n).sqrt() as f32;
    let mean_abs = (sum_abs / n) as f32;
    let mean = sum / n;
    let variance: f64 = data
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = variance.sqrt() as f32;

    let kurtosis = if variance > 1e-30 {
        let m4: f64 = data
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d * d * d
            })
            .sum::<f64>()
            / n;
        (m4 / (variance * variance) - 3.0) as f32 // excess kurtosis
    } else {
        0.0
    };

    let near_zero_thresh = amax * 0.01;
    let near_zero = data
        .iter()
        .filter(|&&x| x.abs() <= near_zero_thresh)
        .count();
    let sparsity_pct = near_zero as f32 / data.len() as f32 * 100.0;

    // Per sub-block quantisation distances
    let num_blocks = data.len() / BLOCK;
    let mut q8_sum = 0.0f32;
    let mut q8_max = 0.0f32;
    let mut q4_sum = 0.0f32;
    let mut q4_max = 0.0f32;
    let mut q3_sum = 0.0f32;
    let mut q2_sum = 0.0f32;

    for b in 0..num_blocks {
        let blk: [f32; BLOCK] = data[b * BLOCK..(b + 1) * BLOCK].try_into().unwrap();

        let d8 = cosine_distance(&blk, &round_trip_q8_0(&blk));
        q8_sum += d8;
        q8_max = q8_max.max(d8);

        let d4 = cosine_distance(&blk, &round_trip_q4_0(&blk));
        q4_sum += d4;
        q4_max = q4_max.max(d4);

        let d3 = cosine_distance(&blk, &round_trip_q3_0(&blk));
        q3_sum += d3;

        let d2 = cosine_distance(&blk, &round_trip_q2_0(&blk));
        q2_sum += d2;
    }

    let cnt = num_blocks.max(1) as f32;
    ChunkStats {
        amax,
        rms,
        mean_abs,
        std,
        kurtosis,
        sparsity_pct,
        q8_cos_mean: q8_sum / cnt,
        q8_cos_max: q8_max,
        q4_cos_mean: q4_sum / cnt,
        q4_cos_max: q4_max,
        q3_cos_mean: q3_sum / cnt,
        q2_cos_mean: q2_sum / cnt,
    }
}

// ---------------------------------------------------------------------------
// Test: generate stats files
// ---------------------------------------------------------------------------

/// Helper: resolve a crate-relative path.
fn data_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

// Measurement/tooling, not a regression guard: writes CSV + binary stat files
// and prints tables (run manually with `--nocapture`). ~10s of GPU + I/O, so
// keep it out of the default `cargo test` cycle — run with
// `--ignored`/by name when analysing KV distributions.
#[test]
#[ignore = "kv-stats generator: heavy measurement tool, run manually with --nocapture"]
fn test_generate_kv_stats() {
    let dump_path = data_path(DUMP_REL_PATH);
    if !dump_path.exists() {
        println!(
            "kv_stats_tests: dump file absent, skipping.\n\
             Generate with: cargo test --release --features cuda -p candle-transformers \
             quantized_llama::tests::test_dump_kv_cache_data -- --ignored --nocapture"
        );
        return;
    }

    let (header, chunks) = load_dump(&dump_path).expect("failed to parse dump file");
    println!(
        "Loaded dump v2: {} layers, {} kv-heads, chunk_size={}, head_dim={}, {} tokens",
        header.num_layers,
        header.n_kv_head,
        header.chunk_size,
        header.head_dim,
        header.tokens.len()
    );
    println!("Total chunks: {}", chunks.len());

    // -----------------------------------------------------------------------
    // Compute stats for every chunk
    // -----------------------------------------------------------------------
    struct Row {
        layer_idx: usize,
        block_idx: usize,
        token_start: usize,
        k: ChunkStats,
        v: ChunkStats,
    }

    let mut rows: Vec<Row> = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let k = compute_stats(&chunk.k);
        let v = compute_stats(&chunk.v);
        rows.push(Row {
            layer_idx: chunk.layer_idx,
            block_idx: chunk.block_idx,
            token_start: chunk.token_start,
            k,
            v,
        });
    }

    // -----------------------------------------------------------------------
    // Write CSV
    // -----------------------------------------------------------------------
    let csv_path = data_path(STATS_CSV_PATH);
    {
        let mut f = std::fs::File::create(&csv_path).expect("cannot create kv_cache_stats.csv");
        writeln!(f, "layer,block_idx,component,{}", ChunkStats::csv_headers()).unwrap();
        for row in &rows {
            writeln!(
                f,
                "{},{},K,{}",
                row.layer_idx,
                row.block_idx,
                row.k.to_csv_with_token_start(row.token_start)
            )
            .unwrap();
            writeln!(
                f,
                "{},{},V,{}",
                row.layer_idx,
                row.block_idx,
                row.v.to_csv_with_token_start(row.token_start)
            )
            .unwrap();
        }
    }
    println!("Written CSV  -> {:?}", csv_path);

    // -----------------------------------------------------------------------
    // Write binary stats file
    // -----------------------------------------------------------------------
    let bin_path = data_path(STATS_BIN_PATH);
    {
        let mut f = std::fs::File::create(&bin_path).expect("cannot create kv_cache_stats.bin");

        // Header
        f.write_all(b"KVSTATS\0").unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap(); // version
        f.write_all(&(header.num_layers as u32).to_le_bytes())
            .unwrap();
        f.write_all(&(header.n_kv_head as u32).to_le_bytes())
            .unwrap();
        f.write_all(&(header.chunk_size as u32).to_le_bytes())
            .unwrap();
        f.write_all(&(header.head_dim as u32).to_le_bytes())
            .unwrap();
        f.write_all(&(header.tokens.len() as u32).to_le_bytes())
            .unwrap();
        for &t in &header.tokens {
            f.write_all(&t.to_le_bytes()).unwrap();
        }

        // Group rows by layer, preserving order
        for layer_idx in 0..header.num_layers {
            let layer_rows: Vec<&Row> = rows.iter().filter(|r| r.layer_idx == layer_idx).collect();
            f.write_all(&(layer_rows.len() as u32).to_le_bytes())
                .unwrap();
            for row in &layer_rows {
                f.write_all(&(row.block_idx as u32).to_le_bytes()).unwrap();
                f.write_all(&(row.token_start as u32).to_le_bytes())
                    .unwrap();
                for &s in &row.k.as_f32_array() {
                    f.write_all(&s.to_le_bytes()).unwrap();
                }
                for &s in &row.v.as_f32_array() {
                    f.write_all(&s.to_le_bytes()).unwrap();
                }
            }
        }
    }
    let bin_size = std::fs::metadata(&bin_path).unwrap().len();
    println!("Written binary -> {:?}  ({} bytes)", bin_path, bin_size);

    // -----------------------------------------------------------------------
    // Print summary table
    // -----------------------------------------------------------------------
    println!("\n{:-<97}", "");
    println!(
        "{:>6}  {:>5}  {:>6}  {:>7}  {:>8}  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
        "Layer",
        "Block",
        "Comp",
        "TokStart",
        "amax",
        "rms",
        "std",
        "kurt",
        "q8_mean",
        "q4_mean",
        "q2_mean"
    );
    println!("{:-<97}", "");

    for row in &rows {
        for (label, s) in [("K", &row.k), ("V", &row.v)] {
            println!(
                "{:>6}  {:>5}  {:>6}  {:>7}  {:>8.4}  {:>8.4}  {:>8.4}  {:>10.3}  {:>10.6}  {:>10.6}  {:>10.6}",
                row.layer_idx, row.block_idx, label, row.token_start,
                s.amax, s.rms, s.std,
                s.kurtosis,
                s.q8_cos_mean, s.q4_cos_mean, s.q2_cos_mean,
            );
        }
    }
    println!("{:-<97}", "");

    // Aggregate summary
    let all_k: Vec<&ChunkStats> = rows.iter().map(|r| &r.k).collect();
    let all_v: Vec<&ChunkStats> = rows.iter().map(|r| &r.v).collect();

    let mean_of = |vals: &[&ChunkStats], f: fn(&ChunkStats) -> f32| -> f32 {
        vals.iter().map(|s| f(s)).sum::<f32>() / vals.len() as f32
    };
    let max_of = |vals: &[&ChunkStats], f: fn(&ChunkStats) -> f32| -> f32 {
        vals.iter().map(|s| f(s)).fold(0.0f32, f32::max)
    };

    println!("\n=== Global averages ===");
    println!(
        "  K: amax={:.4}  rms={:.4}  kurt={:.3}  q8={:.8}  q4={:.8}  q3={:.8}  q2={:.8}",
        mean_of(&all_k, |s| s.amax),
        mean_of(&all_k, |s| s.rms),
        mean_of(&all_k, |s| s.kurtosis),
        mean_of(&all_k, |s| s.q8_cos_mean),
        mean_of(&all_k, |s| s.q4_cos_mean),
        mean_of(&all_k, |s| s.q3_cos_mean),
        mean_of(&all_k, |s| s.q2_cos_mean),
    );
    println!(
        "  V: amax={:.4}  rms={:.4}  kurt={:.3}  q8={:.8}  q4={:.8}  q3={:.8}  q2={:.8}",
        mean_of(&all_v, |s| s.amax),
        mean_of(&all_v, |s| s.rms),
        mean_of(&all_v, |s| s.kurtosis),
        mean_of(&all_v, |s| s.q8_cos_mean),
        mean_of(&all_v, |s| s.q4_cos_mean),
        mean_of(&all_v, |s| s.q3_cos_mean),
        mean_of(&all_v, |s| s.q2_cos_mean),
    );
    println!(
        "  K max q4_cos_max={:.8}  K max q2_cos_mean={:.8}",
        max_of(&all_k, |s| s.q4_cos_max),
        max_of(&all_k, |s| s.q2_cos_mean),
    );
}

// ---------------------------------------------------------------------------
// Test: per-(chunk, head, channel) amax distribution across the head dimension
//
// For each (layer, head): one amax value per (chunk, channel) — taking the
// absolute maximum over the 32 tokens in the chunk for a fixed channel.
// Pool all those amaxes for a head, then bucket them into a histogram.
// Shows whether channel magnitudes are uniform across the head dim
// (flat distribution) or whether a few channels dominate (long tail).
//
// Run:
//   cargo test --release --lib --package candle-nn \
//     kv_stats_tests::test_kv_amax_channel_distribution -- --nocapture
// ---------------------------------------------------------------------------

/// Sort a copy and return percentiles at the requested points (0..=100).
fn percentiles(values: &[f32], points: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return vec![0.0; points.len()];
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    points
        .iter()
        .map(|&p| {
            let idx = ((p / 100.0) * (sorted.len() - 1) as f32).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        })
        .collect()
}

/// Print an ASCII bar histogram. Buckets are linear over [0, p99] of values
/// (anything past p99 is collapsed into the last bucket), so a long tail
/// doesn't squash the body of the distribution.
fn print_amax_histogram(label: &str, values: &[f32], num_buckets: usize, bar_width: usize) {
    if values.is_empty() {
        println!("{} (empty)", label);
        return;
    }
    let pcts = percentiles(values, &[50.0, 90.0, 99.0, 100.0]);
    let cap = pcts[2].max(1e-6); // p99 — use as bucket range cap
    let mut buckets = vec![0usize; num_buckets];
    let mut overflow = 0usize;
    for &v in values {
        if v > cap {
            overflow += 1;
            buckets[num_buckets - 1] += 1;
        } else {
            let b = ((v / cap) * num_buckets as f32) as usize;
            buckets[b.min(num_buckets - 1)] += 1;
        }
    }
    let max_count = *buckets.iter().max().unwrap_or(&1);
    let max_count = max_count.max(1);

    println!(
        "{}  n={}  p50={:.4}  p90={:.4}  p99={:.4}  max={:.4}  overflow_past_p99={}",
        label,
        values.len(),
        pcts[0],
        pcts[1],
        pcts[2],
        pcts[3],
        overflow
    );
    let bucket_w = cap / num_buckets as f32;
    for (i, &c) in buckets.iter().enumerate() {
        let blo = bucket_w * i as f32;
        let bhi = bucket_w * (i + 1) as f32;
        let bar_len = c * bar_width / max_count;
        let bar: String = std::iter::repeat('█').take(bar_len).collect();
        println!("  [{:>6.4}, {:>6.4}): {:>7}  {}", blo, bhi, c, bar);
    }
}

// Measurement/tooling, not a regression guard: prints per-(layer, head)
// channel-amax percentiles + histograms (run manually with `--nocapture`).
// ~5s, kept out of the default `cargo test` cycle.
#[test]
#[ignore = "kv-amax distribution: measurement tool, run manually with --nocapture"]
fn test_kv_amax_channel_distribution() {
    let dump_path = data_path(DUMP_REL_PATH);
    if !dump_path.exists() {
        println!(
            "kv_amax_channel_distribution: dump file absent, skipping.\n\
             Generate with: cargo test --release --features cuda -p candle-transformers \
             quantized_llama::tests::test_dump_kv_cache_data -- --ignored --nocapture"
        );
        return;
    }

    let (header, chunks) = load_dump(&dump_path).expect("failed to parse dump file");
    let n_kv_head = header.n_kv_head;
    let chunk_size = header.chunk_size;
    let head_dim = header.head_dim;
    let num_layers = header.num_layers;

    println!(
        "Loaded dump: {} layers, {} kv-heads, chunk_size={}, head_dim={}, {} chunks total",
        num_layers,
        n_kv_head,
        chunk_size,
        head_dim,
        chunks.len()
    );

    // amaxes[layer][head] = Vec<f32> of channel-amaxes (one per chunk * channel)
    // Length per (layer, head) = num_chunks_in_layer * head_dim
    let mut k_amaxes: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); n_kv_head]; num_layers];
    let mut v_amaxes: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); n_kv_head]; num_layers];

    for chunk in &chunks {
        let stride_head = chunk_size * head_dim;
        for h in 0..n_kv_head {
            let k_base = h * stride_head;
            let v_base = h * stride_head;
            for c in 0..head_dim {
                let mut k_amax = 0.0f32;
                let mut v_amax = 0.0f32;
                for t in 0..chunk_size {
                    let idx = t * head_dim + c;
                    k_amax = k_amax.max(chunk.k[k_base + idx].abs());
                    v_amax = v_amax.max(chunk.v[v_base + idx].abs());
                }
                k_amaxes[chunk.layer_idx][h].push(k_amax);
                v_amaxes[chunk.layer_idx][h].push(v_amax);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Compact percentile table — one line per (layer, head, K|V)
    // -----------------------------------------------------------------------
    println!("\n=== Per-(layer, head) channel-amax percentiles ===");
    println!(
        "{:>6} {:>5} {:>4}  {:>6}  {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "layer", "head", "K|V", "n", "p10", "p25", "p50", "p75", "p90", "p99"
    );
    println!("{:-<88}", "");
    for layer in 0..num_layers {
        for h in 0..n_kv_head {
            for (label, vals) in [("K", &k_amaxes[layer][h]), ("V", &v_amaxes[layer][h])] {
                let p = percentiles(vals, &[10.0, 25.0, 50.0, 75.0, 90.0, 99.0]);
                println!(
                    "{:>6} {:>5} {:>4}  {:>6}  {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
                    layer,
                    h,
                    label,
                    vals.len(),
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    p[4],
                    p[5]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Per-layer aggregate histograms (pooled across heads)
    // -----------------------------------------------------------------------
    println!("\n=== Per-layer pooled histograms (across heads) ===");
    for layer in 0..num_layers {
        let k_pool: Vec<f32> = k_amaxes[layer].iter().flatten().copied().collect();
        let v_pool: Vec<f32> = v_amaxes[layer].iter().flatten().copied().collect();
        println!("\n--- layer {} ---", layer);
        print_amax_histogram(&format!("K  layer={}", layer), &k_pool, 20, 50);
        print_amax_histogram(&format!("V  layer={}", layer), &v_pool, 20, 50);
    }

    // -----------------------------------------------------------------------
    // Global aggregate (all layers, all heads, all chunks, all channels)
    // -----------------------------------------------------------------------
    println!("\n=== Global histograms (all layers + heads pooled) ===");
    let k_all: Vec<f32> = k_amaxes.iter().flatten().flatten().copied().collect();
    let v_all: Vec<f32> = v_amaxes.iter().flatten().flatten().copied().collect();
    print_amax_histogram("K  global", &k_all, 30, 60);
    print_amax_histogram("V  global", &v_all, 30, 60);
}

// ---------------------------------------------------------------------------
// Test: Q0_V table calibration
//
// Produces three tables in dependency order, each consuming output from the
// previous stage:
//
//   1. Scale table       — distribution of outer_scale (= 1/sub_group_amax)
//                          across all (layer, head, sub-group) tuples.
//   2. Amax curve table  — 4 distinct curves describing how block_amax is
//                          distributed across the 32 blocks of a sub-group,
//                          in outer-normalised space [0, 1]. K-means over
//                          sorted block_amax vectors. INT8 [0, 127].
//   3. Within-block      — 32 templates describing relative element
//      shape table         magnitudes within a block (after dividing by the
//                          block's amax), in outer-normalised + amax-
//                          normalised space [0, 1]. K-means over per-block
//                          profile vectors. INT8 [0, 127].
//
// Pipeline: sign → magnitude → shape. Each block's signs are stripped via
// best-matching sign template, then per-block amax is captured (feeds the
// curve table), then per-element profile (after dividing by block amax)
// feeds the shape table.
//
// Run:
//   cargo test --release --lib --package candle-nn \
//     kv_stats_tests::test_q0_v_calibrate_tables -- --nocapture
// ---------------------------------------------------------------------------

// Q0_V sign codebook (16 × u32 = 4 bits/block). Empirically calibrated by
// `test_q0_v_calibrate_tables` against real K/V data:
//   - First swap:  4-bit interleaved patterns (0xAAAA…, 0x5555…, 0xCCCC…,
//                  0x3333…) → byte-stripe + edge-flip (idx 8..11)
//   - Second swap: nibble-stripe patterns (0x0F0F…, 0xF0F0…) → 24/8 edge-run
//                  splits (idx 12..13). These replace the worst-fitting
//                  templates (27% miss rate) with structural complements of
//                  the 8-lane edge runs already at idx 3 and 5.
#[allow(dead_code)]
const Q0V_SIGN_TABLE: [u32; 16] = [
    0x00000000, 0xFFFFFFFF, 0x0000000F, 0x000000FF, 0xF0000000, 0xFF000000, 0x0000FFFF, 0xFFFF0000,
    0xFF00FF00, 0x00FF00FF, 0x80FFFFFF, 0x7FFFFFFF, // byte-stripe + edge-flip
    0xFFFFFF00, 0x00FFFFFF, 0x00FFFF00, 0xFF0000FF, // 24/8 edge-runs (was nibble-stripe)
];

/// Print a histogram of a 1D distribution with ASCII bars.
#[allow(dead_code)]
fn print_distribution(label: &str, values: &[f32], num_buckets: usize) {
    if values.is_empty() {
        println!("{} (empty)", label);
        return;
    }
    let pcts = percentiles(values, &[1.0, 50.0, 90.0, 99.0]);
    let lo = 0.0f32;
    let hi = pcts[3].max(1e-6);
    let mut buckets = vec![0usize; num_buckets];
    for &v in values {
        let b = if hi > lo {
            (((v - lo) / (hi - lo)) * num_buckets as f32) as usize
        } else {
            0
        };
        buckets[b.min(num_buckets - 1)] += 1;
    }
    let max_count = *buckets.iter().max().unwrap_or(&1).max(&1);
    println!(
        "{}  n={}  p1={:.4}  p50={:.4}  p90={:.4}  p99={:.4}",
        label,
        values.len(),
        pcts[0],
        pcts[1],
        pcts[2],
        pcts[3]
    );
    let bw = (hi - lo) / num_buckets as f32;
    for (i, &c) in buckets.iter().enumerate() {
        let blo = lo + bw * i as f32;
        let bhi = lo + bw * (i + 1) as f32;
        let bar_len = (c * 50) / max_count;
        let bar: String = std::iter::repeat('█').take(bar_len).collect();
        println!("  [{:>7.4}, {:>7.4}): {:>7}  {}", blo, bhi, c, bar);
    }
}

// ---------------------------------------------------------------------------
// (Removed) test_q0_v_calibrate_tables — the legacy votes-based curve
// picker. Superseded by `test_q0_v_iterative_curve_selection` which uses
// the production Q0_V kernel as a greedy set-cover oracle on the actual
// data. Scale/centroid/peak emit will be re-derived from the new curves
// once the family pool is finalised.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Kernel-driven calibration filter — drops blocks the production format
// selector would hand to Q0 or Q0_X (the cheaper formats sitting ahead of
// Q0_V in the BPE-ascending walk). What's left are exactly the blocks that
// Q0_V is asked to handle at runtime.
//
// Returns (k_blocks, k_head_amax, v_blocks, v_head_amax) of the survivors,
// or None when CUDA isn't available (falls back to no-filter calibration).
//
// Implementation notes:
//   - One kernel run per dump source (Qwen3 / Llama have different n_kv_head)
//   - K is uploaded as R16 (K f16 + Q f16 / block) so the kernel computes
//     real per-block q-relevance and the IQR-standardised kthresh; V uploaded
//     as F32. Both are dim-major transposed first.
//   - Candidate list: `[Q0, Q0_X, Q8_0]` for both K and V. Q8_0 is the
//     high-quality escape — blocks that fall through to Q8_0 are the
//     Q0_V-territory survivors.
//   - The selector's per-model factors aren't applied here (we want a
//     model-agnostic filter — calibration produces ONE shared codebook).
//     We use the geomean point at C9 by feeding the kernel the C9 raw
//     thresholds. This is the operating regime where Q0_V's marginal
//     contribution actually matters.
// ---------------------------------------------------------------------------
#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn kernel_drop_cheap_format_blocks(
    chunks: &[super::dump_reader::ChunkData],
    source_ranges: &[(usize, usize, usize)],
    head_dim: usize,
) -> Option<(Vec<[f32; 32]>, Vec<f32>, Vec<[f32; 32]>, Vec<f32>)> {
    use crate::kv_cache::chunked::sampled_selection::{
        PRODUCTION_K_QREL_HIGH_THRESHOLDS, PRODUCTION_K_QREL_LOW_THRESHOLDS,
        PRODUCTION_V_QREL_HIGH_THRESHOLDS, PRODUCTION_V_QREL_LOW_THRESHOLDS,
    };
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::{cuda::select_kv_format_paged_batched_raw, GgmlDType};
    use candle::Device;

    let dev = match Device::new_cuda(0) {
        Ok(Device::Cuda(d)) => d,
        _ => return None,
    };
    let stream = dev.cuda_stream();

    const SELECT_BLOCK: usize = 32;
    const N_PALETTE: usize = 4;
    let blocks_per_head = head_dim;
    let single_head_bytes = (blocks_per_head * 128) as i64;
    // The kernels address each palette band through its own gid with a
    // per-band chunk stride; the monolithic per-head upload is presented as
    // N_PALETTE contiguous band slots (gid chunk_idx = palette index).
    let band_chunk_stride = single_head_bytes / N_PALETTE as i64;

    fn dim_major_block(
        chunk: &[f32],
        b: usize,
        n_kv_head: usize,
        blocks_per_chunk: usize,
    ) -> [f32; SELECT_BLOCK] {
        let blocks_per_head = blocks_per_chunk / n_kv_head;
        let head_dim = blocks_per_head;
        let sub_head_dim = head_dim / N_PALETTE;
        let h = b / blocks_per_head;
        let in_head = b % blocks_per_head;
        let p = in_head / sub_head_dim;
        let c = in_head % sub_head_dim;
        let elems_per_band = SELECT_BLOCK * sub_head_dim;
        let base = ((h * N_PALETTE) + p) * elems_per_band;
        std::array::from_fn(|t| chunk[base + t * sub_head_dim + c])
    }
    let to_dim_major = |chunk: &[f32], n_kv_head: usize, blocks_per_chunk: usize| -> Vec<f32> {
        let mut out = Vec::with_capacity(blocks_per_chunk * SELECT_BLOCK);
        for b in 0..blocks_per_chunk {
            out.extend_from_slice(&dim_major_block(chunk, b, n_kv_head, blocks_per_chunk));
        }
        out
    };
    let pack_r16 = |k_data: &[f32], q_data: &[f32]| -> Vec<u8> {
        debug_assert_eq!(k_data.len(), q_data.len());
        let n = k_data.len() / SELECT_BLOCK;
        let mut buf = vec![0u8; n * 128];
        for b in 0..n {
            let off = b * 128;
            for i in 0..SELECT_BLOCK {
                let kf = half::f16::from_f32(k_data[b * SELECT_BLOCK + i]);
                let qf = half::f16::from_f32(q_data[b * SELECT_BLOCK + i]);
                buf[off + i * 2..off + i * 2 + 2].copy_from_slice(&kf.to_le_bytes());
                buf[off + 64 + i * 2..off + 64 + i * 2 + 2].copy_from_slice(&qf.to_le_bytes());
            }
        }
        buf
    };

    // Kernel candidate list — keep cheap formats + high-quality escape.
    let candidates = vec![GgmlDType::Q0, GgmlDType::Q0_X, GgmlDType::Q8_0];
    // Filter operating point — C9 raw production thresholds (no model
    // factors, since the calibration produces ONE codebook used by both
    // models). Kernel will still apply per-block kthresh interpolation
    // from the Q-relevance signal.
    let filter_level: usize = 9;
    let k_hi = PRODUCTION_K_QREL_HIGH_THRESHOLDS[filter_level];
    let k_lo = PRODUCTION_K_QREL_LOW_THRESHOLDS[filter_level];
    let v_hi = PRODUCTION_V_QREL_HIGH_THRESHOLDS[filter_level];
    let v_lo = PRODUCTION_V_QREL_LOW_THRESHOLDS[filter_level];

    // Kernel emits SELECT_FMT_* tags (per `SampleFormat::try_from_cuda_tag`):
    //   Q0 = 33,  Q0_X = 30,  Q8_0 = 7
    let drop_tag_q0: i32 = 33;
    let drop_tag_q0x: i32 = 30;

    let mut k_blocks_out: Vec<[f32; 32]> = Vec::new();
    let mut k_head_amax_out: Vec<f32> = Vec::new();
    let mut v_blocks_out: Vec<[f32; 32]> = Vec::new();
    let mut v_head_amax_out: Vec<f32> = Vec::new();

    let mut total_k_seen = 0usize;
    let mut total_k_dropped_q0 = 0usize;
    let mut total_k_dropped_q0x = 0usize;
    let mut total_v_seen = 0usize;
    let mut total_v_dropped_q0 = 0usize;
    let mut total_v_dropped_q0x = 0usize;

    for &(start, end, src_n_kv_head) in source_ranges {
        let source_chunks = &chunks[start..end];
        if source_chunks.is_empty() {
            continue;
        }
        let n_chunks = source_chunks.len();
        let blocks_per_chunk = src_n_kv_head * head_dim;

        // Upload all chunks as dim-major; K packed as R16 with Q.
        let zero_q: Vec<f32> = vec![0.0; blocks_per_chunk * SELECT_BLOCK];
        struct Cg {
            k_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<u8>,
            v_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<f32>,
        }
        let chunk_gpus: Vec<Cg> = source_chunks
            .iter()
            .map(|c| {
                let k_dm = to_dim_major(&c.k, src_n_kv_head, blocks_per_chunk);
                let v_dm = to_dim_major(&c.v, src_n_kv_head, blocks_per_chunk);
                let q_dm = match &c.q {
                    Some(q) => to_dim_major(q, src_n_kv_head, blocks_per_chunk),
                    None => zero_q.clone(),
                };
                let k_r16 = pack_r16(&k_dm, &q_dm);
                Cg {
                    k_gpu: dev.memcpy_stod(&k_r16).expect("upload k r16"),
                    v_gpu: dev.memcpy_stod(&v_dm).expect("upload v"),
                }
            })
            .collect();

        // per_head_table: 36 i64 per (chunk, head) row = four palette
        // sub-entries of 9. Every band of this fixture lives in the same
        // buffer at the same stride, so all four sub-entries are identical —
        // but they must all be *populated*: the kernel reads the sub-entry for
        // the band it is resolving, not palette 0.
        // metadata = K=R16 (3<<16), V=F32.
        let outer_one_bits = 1.0_f32.to_bits() as i64;
        let metadata_kr16_vf32: i64 = (3i64 << 16) | (0i64 << 8);
        let mut per_head_table: Vec<i64> = Vec::with_capacity(n_chunks * src_n_kv_head * 36);
        for cg in &chunk_gpus {
            let (kp, _g1) = cg.k_gpu.device_ptr(&stream);
            let (vp, _g2) = cg.v_gpu.device_ptr(&stream);
            for h in 0..src_n_kv_head {
                let head_off = (h as i64) * single_head_bytes;
                let sub = [
                    kp as i64,
                    vp as i64,
                    head_off,
                    head_off,
                    band_chunk_stride,
                    band_chunk_stride,
                    metadata_kr16_vf32,
                    outer_one_bits,
                    outer_one_bits,
                ];
                for _ in 0..N_PALETTE {
                    per_head_table.extend_from_slice(&sub);
                }
            }
        }
        let per_head_table_gpu = dev.memcpy_stod(&per_head_table).expect("phtab upload");

        const TEST_ARENA_CHUNKS: i64 = 8192;
        let mut head_gids: Vec<i64> = Vec::with_capacity(n_chunks * src_n_kv_head * N_PALETTE * 2);
        for ci in 0..n_chunks {
            for _h in 0..src_n_kv_head {
                for p in 0..N_PALETTE as i64 {
                    head_gids.push(ci as i64 * TEST_ARENA_CHUNKS + p); // K band p
                    head_gids.push(ci as i64 * TEST_ARENA_CHUNKS + p); // V band p
                }
            }
        }

        let (k_tags_gpu, v_tags_gpu) = select_kv_format_paged_batched_raw(
            &per_head_table_gpu,
            &head_gids,
            &candidates,
            &candidates,
            k_hi,
            k_lo,
            v_hi,
            v_lo,
            blocks_per_head,
            src_n_kv_head,
            TEST_ARENA_CHUNKS as usize,
            &dev,
        )
        .ok()?;
        let k_tags: Vec<i32> = dev.memcpy_dtov(&k_tags_gpu).ok()?;
        let v_tags: Vec<i32> = dev.memcpy_dtov(&v_tags_gpu).ok()?;

        // Kernel block-tag layout: [(chunk_idx, head_idx, palette, channel)]
        // flat with `chunk_idx * src_n_kv_head * head_dim + h * head_dim + p * sub_head_dim + c`.
        // For each calibration block, look up the kernel tag and decide
        // keep/drop. Calibration's accumulation order matches this exactly:
        // chunk-major, then head, then channel-within-head.
        let sub_head_dim = head_dim / N_PALETTE;
        for (ci, chunk) in source_chunks.iter().enumerate() {
            let stride_head = SELECT_BLOCK * head_dim;
            for h in 0..src_n_kv_head {
                for side in 0..2 {
                    let (data, base) = if side == 0 {
                        (&chunk.k, h * stride_head)
                    } else {
                        (&chunk.v, h * stride_head)
                    };
                    // Per-head head_amax (must match calibration's
                    // sg_amax > 0 skip semantics — sub_groups with zero
                    // sg_amax are dropped from the calibration set).
                    let mut head_amax = 0.0f32;
                    for t in 0..SELECT_BLOCK {
                        for c in 0..head_dim {
                            let v = data[base + t * head_dim + c].abs();
                            if v > head_amax {
                                head_amax = v;
                            }
                        }
                    }
                    if head_amax <= 0.0 {
                        continue;
                    }
                    for sg in 0..(head_dim / SELECT_BLOCK) {
                        let chan_start = sg * SELECT_BLOCK;
                        let mut sg_amax = 0.0f32;
                        for c in chan_start..chan_start + SELECT_BLOCK {
                            for t in 0..SELECT_BLOCK {
                                let v = data[base + t * head_dim + c].abs();
                                if v > sg_amax {
                                    sg_amax = v;
                                }
                            }
                        }
                        if sg_amax <= 0.0 {
                            continue;
                        }
                        for c in chan_start..chan_start + SELECT_BLOCK {
                            let mut blk = [0.0f32; SELECT_BLOCK];
                            for t in 0..SELECT_BLOCK {
                                blk[t] = data[base + t * head_dim + c];
                            }
                            // Kernel block index = h * head_dim + p * sub_head_dim + c_in_sub
                            let p = c / sub_head_dim;
                            let c_in_sub = c % sub_head_dim;
                            let kernel_b = h * head_dim + p * sub_head_dim + c_in_sub;
                            let chunk_offset = ci * src_n_kv_head * head_dim;
                            let tag = if side == 0 {
                                k_tags[chunk_offset + kernel_b]
                            } else {
                                v_tags[chunk_offset + kernel_b]
                            };
                            if side == 0 {
                                total_k_seen += 1;
                                if tag == drop_tag_q0 {
                                    total_k_dropped_q0 += 1;
                                } else if tag == drop_tag_q0x {
                                    total_k_dropped_q0x += 1;
                                } else {
                                    k_blocks_out.push(blk);
                                    k_head_amax_out.push(head_amax);
                                }
                            } else {
                                total_v_seen += 1;
                                if tag == drop_tag_q0 {
                                    total_v_dropped_q0 += 1;
                                } else if tag == drop_tag_q0x {
                                    total_v_dropped_q0x += 1;
                                } else {
                                    v_blocks_out.push(blk);
                                    v_head_amax_out.push(head_amax);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("\n========================================================================");
    println!("  Kernel-driven calibration filter (candidates = [Q0, Q0_X, Q8_0] @ C9)");
    println!("========================================================================");
    let pct = |n: usize, d: usize| -> f64 { 100.0 * n as f64 / d.max(1) as f64 };
    println!(
        "  K: {} blocks → kept {} ({:.2}%)  |  dropped Q0={} ({:.2}%)  Q0_X={} ({:.2}%)",
        total_k_seen,
        k_blocks_out.len(),
        pct(k_blocks_out.len(), total_k_seen),
        total_k_dropped_q0,
        pct(total_k_dropped_q0, total_k_seen),
        total_k_dropped_q0x,
        pct(total_k_dropped_q0x, total_k_seen),
    );
    println!(
        "  V: {} blocks → kept {} ({:.2}%)  |  dropped Q0={} ({:.2}%)  Q0_X={} ({:.2}%)",
        total_v_seen,
        v_blocks_out.len(),
        pct(v_blocks_out.len(), total_v_seen),
        total_v_dropped_q0,
        pct(total_v_dropped_q0, total_v_seen),
        total_v_dropped_q0x,
        pct(total_v_dropped_q0x, total_v_seen),
    );

    Some((k_blocks_out, k_head_amax_out, v_blocks_out, v_head_amax_out))
}

#[cfg(not(feature = "cuda"))]
fn kernel_drop_cheap_format_blocks(
    _chunks: &[super::dump_reader::ChunkData],
    _source_ranges: &[(usize, usize, usize)],
    _head_dim: usize,
) -> Option<(Vec<[f32; 32]>, Vec<f32>, Vec<[f32; 32]>, Vec<f32>)> {
    None
}

// ---------------------------------------------------------------------------
// Helper: load all per-side blocks + head_amax from the dumps. Same data
// pipeline as `test_q0_v_calibrate_tables` (Phase A walk over Qwen3 + Llama
// dumps, building the per-(chunk, head, side, sub-group) block stream),
// trimmed to just the (blocks, head_amax) outputs used by downstream tests.
// Returns None if no dump files are present.
// ---------------------------------------------------------------------------
struct LoadedKvBlocks {
    k_blocks: Vec<[f32; 32]>,
    k_blocks_head_amax: Vec<f32>,
    v_blocks: Vec<[f32; 32]>,
    v_blocks_head_amax: Vec<f32>,
}

fn load_kv_blocks_for_q0_v_tests() -> Option<LoadedKvBlocks> {
    let qwen3_path = data_path(QWEN3_DUMP_REL_PATH);
    let llama_path = data_path(LLAMA_DUMP_REL_PATH);
    if !qwen3_path.exists() && !llama_path.exists() {
        return None;
    }
    let (primary_path, _primary_label) = if qwen3_path.exists() {
        (qwen3_path.clone(), "qwen3-kv-data.bin")
    } else {
        (llama_path.clone(), "llama-kv-data.bin")
    };
    let (header, mut chunks) = load_dump(&primary_path)?;
    let mut source_ranges: Vec<(usize, usize, usize)> = vec![(0, chunks.len(), header.n_kv_head)];
    if qwen3_path.exists() && llama_path.exists() {
        if let Some((other_header, other_chunks)) = load_dump(&llama_path) {
            if other_header.head_dim == header.head_dim
                && other_header.chunk_size == header.chunk_size
            {
                let llama_start = chunks.len();
                chunks.extend(other_chunks);
                let llama_end = chunks.len();
                source_ranges.push((llama_start, llama_end, other_header.n_kv_head));
            }
        }
    }
    let n_kv_head = header.n_kv_head;
    let chunk_size = header.chunk_size;
    let head_dim = header.head_dim;
    assert_eq!(chunk_size, 32);
    assert!(head_dim % 32 == 0);
    let sub_groups_per_head = head_dim / 32;

    let est = chunks.len() * n_kv_head * head_dim;
    let mut k_blocks: Vec<[f32; 32]> = Vec::with_capacity(est);
    let mut v_blocks: Vec<[f32; 32]> = Vec::with_capacity(est);
    let mut k_blocks_head_amax: Vec<f32> = Vec::with_capacity(est);
    let mut v_blocks_head_amax: Vec<f32> = Vec::with_capacity(est);

    for &(start, end, src_n_kv_head) in &source_ranges {
        for chunk in &chunks[start..end] {
            let stride_head = chunk_size * head_dim;
            for h in 0..src_n_kv_head {
                for side in 0..2 {
                    let (data, base) = if side == 0 {
                        (&chunk.k, h * stride_head)
                    } else {
                        (&chunk.v, h * stride_head)
                    };
                    let mut head_amax = 0.0f32;
                    for t in 0..chunk_size {
                        for c in 0..head_dim {
                            let v = data[base + t * head_dim + c].abs();
                            if v > head_amax {
                                head_amax = v;
                            }
                        }
                    }
                    if head_amax <= 0.0 {
                        continue;
                    }
                    for sg in 0..sub_groups_per_head {
                        let chan_start = sg * 32;
                        let mut sg_amax = 0.0f32;
                        for c in chan_start..chan_start + 32 {
                            for t in 0..chunk_size {
                                let v = data[base + t * head_dim + c].abs();
                                if v > sg_amax {
                                    sg_amax = v;
                                }
                            }
                        }
                        if sg_amax <= 0.0 {
                            continue;
                        }
                        for (bi, c) in (chan_start..chan_start + 32).enumerate() {
                            let _ = bi;
                            let mut blk = [0.0f32; 32];
                            for t in 0..chunk_size {
                                blk[t] = data[base + t * head_dim + c];
                            }
                            if side == 0 {
                                k_blocks.push(blk);
                                k_blocks_head_amax.push(head_amax);
                            } else {
                                v_blocks.push(blk);
                                v_blocks_head_amax.push(head_amax);
                            }
                        }
                    }
                }
            }
        }
    }
    Some(LoadedKvBlocks {
        k_blocks,
        k_blocks_head_amax,
        v_blocks,
        v_blocks_head_amax,
    })
}

// ---------------------------------------------------------------------------
// Test: round-trip every K and V block through the *real* CUDA Q0_V
// quantize/dequantize kernels (not the Rust mirror) and report the same
// pass-rate metrics as the calibration modelling pass. Direct apples-to-
// apples comparison against `test_q0_v_calibrate_tables`'s modelled numbers.
//
// Why this exists: the calibration's modelling pass uses the Rust encoder
// and computes pass_metric in normalised space. The format-selection kernel
// in production sees Q0_V at 0–0.7% in the empirical distribution despite
// the modelled 14–84% pass rate. To isolate "is the quant/dequant pair
// itself broken" from "is the wiring around it broken", this test exercises
// the production CUDA encoder + decoder under the IS_K compile-time flag
// (the same code paths the format selector uses) and reports the same
// metrics.
//
// If the CUDA round-trip pass rate matches the modelled rate, the
// quantize/dequantize kernels are correct and the issue must be in the
// selection wiring or the slot-quota logic. If the CUDA pass rate is much
// lower than modelled, the kernels themselves have a bug (encoder mismatch,
// table-load issue, etc.) that needs fixing first.
//
// Run with:
//   cargo test --release --features cuda,dont_check --lib --package candle-nn \
//     kv_stats_tests::test_q0_v_kernel_roundtrip_pass_rates -- --nocapture
// ---------------------------------------------------------------------------
#[cfg(feature = "cuda")]
#[test]
fn test_q0_v_kernel_roundtrip_pass_rates() {
    use crate::kv_cache::arena_table::N_PALETTE;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Device;
    use rayon::prelude::*;
    use std::ffi::{c_int, c_void};

    extern "C" {
        fn run_roundtrip_q0_v(
            src: *const c_void,
            recon: *mut c_void,
            num_blocks: c_int,
            outer: f32,
            is_k: c_int,
        );
    }

    let loaded = match load_kv_blocks_for_q0_v_tests() {
        Some(b) => b,
        None => {
            println!("test_q0_v_kernel_roundtrip_pass_rates: dump files absent, skipping.");
            return;
        }
    };

    let dev = match Device::new_cuda(0) {
        Ok(Device::Cuda(d)) => d,
        Ok(_) => {
            println!("CUDA device not selected, skipping.");
            return;
        }
        Err(e) => {
            println!("CUDA device unavailable: {e}, skipping.");
            return;
        }
    };
    let stream = dev.cuda_stream();

    fn erf_approx(x: f32) -> f32 {
        let a1 = 0.254829592_f32;
        let a2 = -0.284496736_f32;
        let a3 = 1.421413741_f32;
        let a4 = -1.453152027_f32;
        let a5 = 1.061405429_f32;
        let p = 0.3275911_f32;
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        sign * y
    }
    fn phi(x: f32) -> f32 {
        0.5 * (1.0 + erf_approx(x / std::f32::consts::SQRT_2))
    }

    let run_side = |side_label: &str, blocks: &[[f32; 32]], head_amax: &[f32]| {
        println!("\n========================================================================");
        println!(
            "  Q0_V CUDA-kernel round-trip: side = {} ({} blocks)",
            side_label.to_uppercase(),
            blocks.len()
        );
        println!("========================================================================");

        // Build a flat f32 buffer of N*32 elements, normalising each block
        // by 1/head_amax so values lie in [-1, +1] (the range Q0_V's codebook
        // is calibrated for). Drop blocks with head_amax ≤ 0.
        let n_total = blocks.len();
        let mut flat: Vec<f32> = Vec::with_capacity(n_total * 32);
        let mut keep_idx: Vec<usize> = Vec::with_capacity(n_total);
        for (i, (blk, &ha)) in blocks.iter().zip(head_amax.iter()).enumerate() {
            if ha <= 0.0 {
                continue;
            }
            let inv = 1.0 / ha;
            for k in 0..32 {
                flat.push(blk[k] * inv);
            }
            keep_idx.push(i);
        }
        let n = keep_idx.len();
        if n == 0 {
            println!("  (no eligible blocks)");
            return;
        }

        // Upload + alloc + launch.
        let t0 = std::time::Instant::now();
        let src_gpu = dev.memcpy_stod(&flat).expect("upload src");
        let recon_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<f32> =
            stream.alloc_zeros(flat.len()).expect("alloc recon");

        let (src_ptr, _g1) = src_gpu.device_ptr(&stream);
        let (recon_ptr, _g2) = recon_gpu.device_ptr(&stream);
        let is_k: c_int = if side_label == "k" { 1 } else { 0 };
        unsafe {
            run_roundtrip_q0_v(
                src_ptr as *const c_void,
                recon_ptr as *mut c_void,
                n as c_int,
                1.0,
                is_k,
            );
        }
        stream.synchronize().expect("sync");
        let recon: Vec<f32> = dev.memcpy_dtov(&recon_gpu).expect("download recon");
        println!(
            "  CUDA round-trip {} blocks in {:.2}s",
            n,
            t0.elapsed().as_secs_f64()
        );

        // Compute the side's pass_metric per block (in normalised space,
        // where head_amax has cancelled out of both sides of the metric).
        let metrics: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|b| {
                let off = b * 32;
                if side_label == "k" {
                    let mut errs: [f32; 32] =
                        std::array::from_fn(|i| (flat[off + i] - recon[off + i]).abs());
                    errs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    (errs[0] + errs[1] + errs[2] + errs[3]) * 0.25
                } else {
                    let mse: f32 = (0..32)
                        .map(|i| {
                            let d = flat[off + i] - recon[off + i];
                            d * d
                        })
                        .sum::<f32>()
                        / 32.0;
                    mse.sqrt()
                }
            })
            .collect();

        let mut sorted = metrics.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct = |q: f64| sorted[((q * (n - 1) as f64) as usize).min(n - 1)];
        let mean_metric: f32 = sorted.iter().sum::<f32>() / n as f32;

        let metric_label = if side_label == "k" {
            "mean_top4(|err|) / head_amax"
        } else {
            "sqrt(mean(err²)) / head_amax   (RMSE form)"
        };
        println!("  Metric: {}", metric_label);
        println!(
            "  mean = {:.6}, p50 = {:.6}, p75 = {:.6}, p90 = {:.6}, p95 = {:.6}, p99 = {:.6}, max = {:.6}",
            mean_metric,
            pct(0.50),
            pct(0.75),
            pct(0.90),
            pct(0.95),
            pct(0.99),
            sorted[n - 1]
        );

        // Pass rates with kernel's interpolation formula, both factor sets.
        use crate::kv_cache::chunked::sampled_selection::{
            LLAMA_KV_FACTORS, PRODUCTION_K_QREL_HIGH_THRESHOLDS, PRODUCTION_K_QREL_LOW_THRESHOLDS,
            PRODUCTION_V_QREL_HIGH_THRESHOLDS, PRODUCTION_V_QREL_LOW_THRESHOLDS,
            QWEN3_MOE_KV_FACTORS,
        };
        let (thr_hi_base, thr_lo_base): (&[f32; 11], &[f32; 11]) = if side_label == "k" {
            (
                &PRODUCTION_K_QREL_HIGH_THRESHOLDS,
                &PRODUCTION_K_QREL_LOW_THRESHOLDS,
            )
        } else {
            (
                &PRODUCTION_V_QREL_HIGH_THRESHOLDS,
                &PRODUCTION_V_QREL_LOW_THRESHOLDS,
            )
        };
        let (f_hi_l, f_lo_l) = if side_label == "k" {
            (LLAMA_KV_FACTORS.k_hi, LLAMA_KV_FACTORS.k_low)
        } else {
            (LLAMA_KV_FACTORS.v_hi, LLAMA_KV_FACTORS.v_low)
        };
        let (f_hi_q, f_lo_q) = if side_label == "k" {
            (QWEN3_MOE_KV_FACTORS.k_hi, QWEN3_MOE_KV_FACTORS.k_low)
        } else {
            (QWEN3_MOE_KV_FACTORS.v_hi, QWEN3_MOE_KV_FACTORS.v_low)
        };

        let model_pass = |hi: f32, lo: f32| -> f64 {
            if side_label == "k" {
                metrics
                    .par_iter()
                    .map(|&m| {
                        if m <= hi {
                            1.0_f64
                        } else if m >= lo {
                            0.0_f64
                        } else {
                            let base = (hi * lo).sqrt();
                            let z_star = (base / m).ln();
                            phi(z_star) as f64
                        }
                    })
                    .sum::<f64>()
                    / n as f64
            } else {
                metrics
                    .par_iter()
                    .map(|&m| {
                        if m <= hi {
                            1.0_f64
                        } else if m >= lo {
                            0.0_f64
                        } else {
                            ((lo - m) / (lo - hi)) as f64
                        }
                    })
                    .sum::<f64>()
                    / n as f64
            }
        };

        // Calibration-equivalent pass rate: deterministic threshold =
        // sqrt(HIGH × LOW) with NO model factors. Matches what the
        // iterative-curve-selection test (`test_q0_v_iterative_curve_selection`)
        // uses as its `pass_threshold`. Gives an upper bound on what
        // production can achieve before model-specific scaling and per-
        // block kthresh interpolation tighten things.
        let calib_pass = |hi: f32, lo: f32| -> f64 {
            let thr = (hi * lo).sqrt();
            metrics.par_iter().filter(|&&m| m <= thr).count() as f64 / n as f64
        };

        println!(
            "\n  CUDA-kernel pass rate ({}-side):",
            side_label.to_uppercase()
        );
        println!(
            "      Calibration: deterministic threshold = sqrt(HIGH × LOW), no factors  (matches `test_q0_v_iterative_curve_selection`)"
        );
        println!(
            "      Llama factors: hi×{:.3}, lo×{:.3}     Qwen3 factors: hi×{:.3}, lo×{:.3}",
            f_hi_l, f_lo_l, f_hi_q, f_lo_q
        );
        println!("    C   |  Calib (det.) |  Llama (interp)  |  Qwen3 (interp)  | min      max");
        println!("    --- |  -----------  |  --------------  |  --------------  | ------   ------");
        for c in 0..11 {
            let hi_raw = thr_hi_base[c];
            let lo_raw = thr_lo_base[c];
            let p_calib = calib_pass(hi_raw, lo_raw);
            let hi_l = thr_hi_base[c] * f_hi_l;
            let lo_l = thr_lo_base[c] * f_lo_l;
            let hi_q = thr_hi_base[c] * f_hi_q;
            let lo_q = thr_lo_base[c] * f_lo_q;
            let p_l = model_pass(hi_l, lo_l);
            let p_q = model_pass(hi_q, lo_q);
            let p_min = p_l.min(p_q);
            let p_max = p_l.max(p_q);
            println!(
                "    C{:>2} |    {:>6.2}%    |     {:>6.2}%       |     {:>6.2}%       | {:>5.2}%   {:>5.2}%",
                c, 100.0 * p_calib, 100.0 * p_l, 100.0 * p_q,
                100.0 * p_min, 100.0 * p_max
            );
        }
    };

    run_side("k", &loaded.k_blocks, &loaded.k_blocks_head_amax);
    run_side("v", &loaded.v_blocks, &loaded.v_blocks_head_amax);

    // ============================================================
    // Selection-kernel comparison: candidate list = [Q0_V, Q8_1]
    // ============================================================
    //
    // Run the production format-selection kernel on the same dump data,
    // limiting the candidate list to just Q0_V and Q8_1. This isolates
    // exactly what the selector picks for each block when its only
    // alternatives are Q0_V (0.5 bpe) and Q8_1 (8.5 bpe). Direct apples-
    // to-apples against the round-trip pass rates above:
    //
    //   - If the selector picks Q0_V at ~15–84% (matching the modelled
    //     pass rates), the codec is correctly recognised and the wider 0%
    //     usage in production is a *competition* effect (other 0.5-bpe
    //     formats winning) rather than a Q0_V bug.
    //   - If the selector picks Q0_V at ~0%, the bug is in the
    //     selector kernel's Q0_V evaluation path, NOT in the encoder/
    //     decoder pair.
    //
    // The kernel processes whole chunks (32 tokens × n_kv_head heads ×
    // head_dim channels per chunk), so we feed the dump's raw chunks
    // directly rather than the 32-element blocks we extracted earlier.
    use crate::kv_cache::chunked::sampled_selection::{
        LLAMA_KV_FACTORS, PRODUCTION_K_QREL_HIGH_THRESHOLDS, PRODUCTION_K_QREL_LOW_THRESHOLDS,
        PRODUCTION_V_QREL_HIGH_THRESHOLDS, PRODUCTION_V_QREL_LOW_THRESHOLDS, QWEN3_MOE_KV_FACTORS,
    };
    use candle::quantized::cuda::select_kv_format_paged_batched_raw;
    use candle::quantized::GgmlDType;

    println!("\n========================================================================");
    println!("  Selection kernel head-to-head: candidates = [Q0_V, Q8_1]");
    println!("========================================================================");

    // Run the selector for each (dump, factor-set) pair separately so that
    // Llama factors are evaluated against Llama data and Qwen3 factors
    // against Qwen3 data — matching the apples-to-apples contract of the
    // production `test_candidate_list_compression_curve` report. Loading
    // each dump per pairing keeps the per_head_table pointers and R16
    // packing localised to one dataset; the cost is two upload+selector
    // passes instead of one, but each takes <2s on the GPU.
    use crate::kv_cache::chunked::sampled_selection::params::production_adaptive_candidates;
    use crate::kv_cache::chunked::sampled_selection::KvErrorThresholdFactors;
    use crate::kv_cache::KvFormat;
    let qwen3_path = data_path(QWEN3_DUMP_REL_PATH);
    let llama_path = data_path(LLAMA_DUMP_REL_PATH);
    let datasets: [(&str, std::path::PathBuf, &KvErrorThresholdFactors); 2] = [
        ("Qwen3", qwen3_path, &QWEN3_MOE_KV_FACTORS),
        ("Llama", llama_path, &LLAMA_KV_FACTORS),
    ];
    for (this_model_name, this_dump_path, this_factors) in &datasets {
        if !this_dump_path.exists() {
            println!(
                "\n  [skip] {} dump absent at {:?}",
                this_model_name, this_dump_path
            );
            continue;
        }
        println!(
            "\n  ── Dataset: {} (dump = {:?}) ──",
            this_model_name,
            this_dump_path.file_name().unwrap_or_default()
        );
        let (header, chunks) = super::dump_reader::load_dump(this_dump_path).expect("load dump");
        let n_kv_head = header.n_kv_head;
        let chunk_size = header.chunk_size;
        let head_dim = header.head_dim;
        assert_eq!(chunk_size, 32);

        // Each chunk's f32 layout: [n_kv_head, chunk_size, head_dim].
        // For the selector kernel we treat each dump chunk as its own arena.
        // blocks_per_head = head_dim → one 32-element block per channel of one head.
        // Each (arena, head) pair gets its own per_head_table row pointing at
        // that head's slice of the chunk; multi-head dumps use byte_offset to
        // pick the head within the chunk.
        let blocks_per_head = head_dim;
        // K is uploaded as R16 (K f16 + Q f16, 128 bytes/block). V is uploaded as
        // F32 (32 × 4 = 128 bytes/block). Both arenas have the same per-block byte
        // size for 32-element blocks, so per-head byte offsets / strides are
        // identical between K and V.
        let single_head_bytes = (head_dim * 128) as i64; // head_dim blocks × 128 bytes/block
                                                         // The kernels address each palette band through its own gid with a
                                                         // per-band chunk stride; the monolithic per-head upload is presented
                                                         // as 4 contiguous band slots (gid chunk_idx = palette index).
        let band_chunk_stride = single_head_bytes / 4;

        // Layout note: the dump stores chunks token-major (`[head, token, dim]`),
        // but the selection kernel sees each block as 32 tokens of one channel
        // within a (head, palette) sub-band. Reading the dump verbatim gives the
        // kernel the *transposed* view (32 channels of one token), which makes
        // every block look near-constant and biases the picker toward Q0. We
        // transpose to dim-major before uploading using the same helper that
        // `test_candidate_list_compression_curve` validates. The result for each
        // chunk is `blocks_per_chunk = n_kv_head * head_dim` consecutive
        // 32-element blocks in dim-major iteration order: head, palette, channel.
        //
        // K-side additionally packs Q into the back half of each R16 block so the
        // kernel can compute real per-block q_relevance (and the IQR-standardised
        // exp(-z) per-block kthresh) instead of falling back to the lenient
        // geometric-mean threshold. v4 dumps include Q vectors; pre-v4 dumps
        // (chunk.q == None) fall back to zero-Q which collapses to geometric-mean
        // thresholds — same regime the F32-K path was previously hitting.
        let blocks_per_chunk = n_kv_head * head_dim;
        fn dim_major_block_from_token_major(
            chunk: &[f32],
            b: usize,
            n_kv_head: usize,
            blocks_per_chunk: usize,
        ) -> [f32; 32] {
            const SELECT_BLOCK: usize = 32;
            const N_PALETTE: usize = 4;
            debug_assert!(n_kv_head > 0);
            let blocks_per_head = blocks_per_chunk / n_kv_head; // = head_dim
            let head_dim = blocks_per_head;
            let sub_head_dim = head_dim / N_PALETTE;
            let h = b / blocks_per_head;
            let in_head = b % blocks_per_head;
            let p = in_head / sub_head_dim;
            let c = in_head % sub_head_dim;
            let elems_per_band = SELECT_BLOCK * sub_head_dim;
            let base = ((h * N_PALETTE) + p) * elems_per_band;
            let mut blk = [0f32; SELECT_BLOCK];
            for t in 0..SELECT_BLOCK {
                blk[t] = chunk[base + t * sub_head_dim + c];
            }
            blk
        }
        let to_dim_major = |chunk: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(blocks_per_chunk * 32);
            for b in 0..blocks_per_chunk {
                out.extend_from_slice(&dim_major_block_from_token_major(
                    chunk,
                    b,
                    n_kv_head,
                    blocks_per_chunk,
                ));
            }
            out
        };

        // R16 block layout (128 bytes / 32-element block):
        //   d[0..32]: K values as F16 (offset  0, 64 bytes)
        //   q[0..32]: Q values as F16 (offset 64, 64 bytes)
        let pack_r16_blocks = |k_data: &[f32], q_data: &[f32]| -> Vec<u8> {
            debug_assert_eq!(k_data.len(), q_data.len());
            debug_assert!(k_data.len() % 32 == 0);
            let n_blocks = k_data.len() / 32;
            let mut buf = vec![0u8; n_blocks * 128];
            for b in 0..n_blocks {
                let block_start = b * 128;
                for i in 0..32 {
                    let kf = half::f16::from_f32(k_data[b * 32 + i]);
                    let qf = half::f16::from_f32(q_data[b * 32 + i]);
                    buf[block_start + i * 2..block_start + i * 2 + 2]
                        .copy_from_slice(&kf.to_le_bytes());
                    buf[block_start + 64 + i * 2..block_start + 64 + i * 2 + 2]
                        .copy_from_slice(&qf.to_le_bytes());
                }
            }
            buf
        };

        let mut q_present_chunks: usize = 0;
        let mut q_missing_chunks: usize = 0;

        struct ChunkGpu {
            k_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<u8>,
            v_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<f32>,
        }
        let upload_start = std::time::Instant::now();
        let zero_q: Vec<f32> = vec![0.0; blocks_per_chunk * 32];
        let chunk_gpus: Vec<ChunkGpu> = chunks
            .iter()
            .map(|c| {
                let k_dim_major = to_dim_major(&c.k);
                let v_dim_major = to_dim_major(&c.v);
                let q_dim_major = match &c.q {
                    Some(q) => {
                        q_present_chunks += 1;
                        to_dim_major(q)
                    }
                    None => {
                        q_missing_chunks += 1;
                        zero_q.clone()
                    }
                };
                let k_r16_bytes = pack_r16_blocks(&k_dim_major, &q_dim_major);
                ChunkGpu {
                    k_gpu: dev.memcpy_stod(&k_r16_bytes).expect("upload k r16"),
                    v_gpu: dev.memcpy_stod(&v_dim_major).expect("upload v"),
                }
            })
            .collect();
        println!(
            "  Uploaded {} chunks (dim-major: {} kv-heads × {} head_dim × 32 tokens) in {:.2}s",
            chunk_gpus.len(),
            n_kv_head,
            head_dim,
            upload_start.elapsed().as_secs_f64()
        );
        println!(
        "  K = R16 (K[32] f16 + Q[32] f16 / block), V = F32. Q-capture: {} present, {} missing.",
        q_present_chunks, q_missing_chunks
    );

        // per_head_table layout: 36 i64 per (arena, head) row =
        // 4 palettes × 9 i64 per palette sub-entry. We populate palette[0] with
        // real data and zero the other palettes (only palette[0] is read by the
        // 2-candidate test path).
        //   palette sub-entry (9 i64): [k_ptr, v_ptr, k_byte_offset, v_byte_offset,
        //                               k_chunk_byte_stride, v_chunk_byte_stride,
        //                               metadata, k_outer_scale_bits, v_outer_scale_bits]
        //   metadata = (k_format_tag << 16) | (v_format_tag << 8) | location
        //   K = R16 (ArenaFormatTag::R16 = 3), V = F32 (= 0), location = GPU (= 0)
        //   → metadata = (3 << 16) | (0 << 8) | 0 = 0x30000
        let outer_one_bits = 1.0_f32.to_bits() as i64;
        let metadata_kr16_vf32: i64 = (3i64 << 16) | (0i64 << 8) | 0i64;
        let mut per_head_table_host: Vec<i64> =
            Vec::with_capacity(chunk_gpus.len() * n_kv_head * 36);
        for cg in &chunk_gpus {
            let (kp, _) = cg.k_gpu.device_ptr(&stream);
            let (vp, _) = cg.v_gpu.device_ptr(&stream);
            for h in 0..n_kv_head {
                let head_off = (h as i64) * single_head_bytes;
                // All four palette sub-entries point at this head's slice:
                // the kernel resolves each band through its own sub-entry.
                let sub = [
                    kp as i64,
                    vp as i64,
                    head_off,
                    head_off,
                    band_chunk_stride,
                    band_chunk_stride,
                    metadata_kr16_vf32,
                    outer_one_bits,
                    outer_one_bits,
                ];
                for _ in 0..N_PALETTE {
                    per_head_table_host.extend_from_slice(&sub);
                }
            }
        }
        let per_head_table_gpu = dev
            .memcpy_stod(&per_head_table_host)
            .expect("per_head upload");

        // head_gids: 8 entries per (chunk, head) — K/V per palette band. We use
        // arena_idx = chunk_idx, chunk_idx = palette, ARENA_CHUNKS = 8192.
        const TEST_ARENA_CHUNKS: i64 = 8192;
        let mut head_gids: Vec<i64> = Vec::with_capacity(chunks.len() * n_kv_head * 8);
        for ci in 0..chunks.len() {
            for _h in 0..n_kv_head {
                for p in 0..4i64 {
                    head_gids.push(ci as i64 * TEST_ARENA_CHUNKS + p); // K band p
                    head_gids.push(ci as i64 * TEST_ARENA_CHUNKS + p); // V band p
                }
            }
        }

        // ----- Pass 1: 2-candidate ladder [Q0_V, Q8_1] -----
        // Q0_V (most aggressive, 0.5 bpe) then Q8_1 (high quality, 8.5 bpe) as
        // the fallback. The selector tries Q0_V first; if it can't claim a
        // slot, it falls back to Q8_1.
        let candidates_two = vec![GgmlDType::Q0_V, GgmlDType::Q8_1];
        let levels_to_run: [usize; 4] = [5, 7, 9, 10];

        println!("\n    ── Pass 1: candidate ladder = [Q0_V, Q8_1] (clean head-to-head) ──");
        println!("      Model      C    side     Q0_V%     Q8_1%   total_blocks");
        println!("      --------   --   ----    ------    ------   ------------");

        {
            let model_name = *this_model_name;
            let factors = *this_factors;
            for &level in &levels_to_run {
                let k_hi_eff = PRODUCTION_K_QREL_HIGH_THRESHOLDS[level] * factors.k_hi;
                let k_lo_eff = PRODUCTION_K_QREL_LOW_THRESHOLDS[level] * factors.k_low;
                let v_hi_eff = PRODUCTION_V_QREL_HIGH_THRESHOLDS[level] * factors.v_hi;
                let v_lo_eff = PRODUCTION_V_QREL_LOW_THRESHOLDS[level] * factors.v_low;

                let (k_tags_gpu, v_tags_gpu) = select_kv_format_paged_batched_raw(
                    &per_head_table_gpu,
                    &head_gids,
                    &candidates_two,
                    &candidates_two,
                    k_hi_eff,
                    k_lo_eff,
                    v_hi_eff,
                    v_lo_eff,
                    blocks_per_head,
                    n_kv_head,
                    TEST_ARENA_CHUNKS as usize,
                    &dev,
                )
                .expect("selector launch");
                let k_tags: Vec<i32> = dev.memcpy_dtov(&k_tags_gpu).expect("dl k");
                let v_tags: Vec<i32> = dev.memcpy_dtov(&v_tags_gpu).expect("dl v");

                // Kernel emits SELECT_FMT_* codes (Q0_V = 28, Q8_1 = 8).
                let q0v_tag: i32 = 28;
                let q81_tag: i32 = 8;
                let count_pct = |tags: &[i32]| -> (f64, f64, usize) {
                    let total = tags.len();
                    let q0v = tags.iter().filter(|&&t| t == q0v_tag).count();
                    let q81 = tags.iter().filter(|&&t| t == q81_tag).count();
                    (
                        100.0 * q0v as f64 / total.max(1) as f64,
                        100.0 * q81 as f64 / total.max(1) as f64,
                        total,
                    )
                };
                let (k_q0v_pct, k_q81_pct, k_total) = count_pct(&k_tags);
                let (v_q0v_pct, v_q81_pct, v_total) = count_pct(&v_tags);
                println!(
                    "    {:<8}  C{:>2}    K     {:>5.2}%   {:>5.2}%   {:>10}",
                    model_name, level, k_q0v_pct, k_q81_pct, k_total
                );
                println!(
                    "    {:<8}  C{:>2}    V     {:>5.2}%   {:>5.2}%   {:>10}",
                    model_name, level, v_q0v_pct, v_q81_pct, v_total
                );
            }
        }

        // ----- Pass 2: full production ladder -----
        // Run the production candidate list (16 K formats / 14 V formats) at
        // each level for both factor sets and tabulate the format share. This
        // is the apples-to-apples comparison against the empirical distribution
        // table from `test_candidate_list_compression_curve`. If Q0_V's share
        // here is comparable to that table (≈ 0%), the wiring matches; if
        // it's higher here, the discrepancy is coming from elsewhere in the
        // production path (palette-4 reduction, head-tag promotion, etc.).
        println!("\n    ── Pass 2: candidate ladder = full production list ──");

        {
            let model_name = *this_model_name;
            let factors = *this_factors;
            for &level in &levels_to_run {
                let k_hi_eff = PRODUCTION_K_QREL_HIGH_THRESHOLDS[level] * factors.k_hi;
                let k_lo_eff = PRODUCTION_K_QREL_LOW_THRESHOLDS[level] * factors.k_low;
                let v_hi_eff = PRODUCTION_V_QREL_HIGH_THRESHOLDS[level] * factors.v_hi;
                let v_lo_eff = PRODUCTION_V_QREL_LOW_THRESHOLDS[level] * factors.v_low;

                let (k_kv_cands, v_kv_cands) = production_adaptive_candidates(level as u8);
                let k_cands: Vec<GgmlDType> = k_kv_cands
                    .iter()
                    .filter_map(|f| match f {
                        KvFormat::Quantized(qf) => Some(qf.to_ggml_dtype()),
                        _ => None,
                    })
                    .collect();
                let v_cands: Vec<GgmlDType> = v_kv_cands
                    .iter()
                    .filter_map(|f| match f {
                        KvFormat::Quantized(qf) => Some(qf.to_ggml_dtype()),
                        _ => None,
                    })
                    .collect();

                let (k_tags_gpu, v_tags_gpu) = select_kv_format_paged_batched_raw(
                    &per_head_table_gpu,
                    &head_gids,
                    &k_cands,
                    &v_cands,
                    k_hi_eff,
                    k_lo_eff,
                    v_hi_eff,
                    v_lo_eff,
                    blocks_per_head,
                    n_kv_head,
                    TEST_ARENA_CHUNKS as usize,
                    &dev,
                )
                .expect("selector launch");
                let k_tags: Vec<i32> = dev.memcpy_dtov(&k_tags_gpu).expect("dl k");
                let v_tags: Vec<i32> = dev.memcpy_dtov(&v_tags_gpu).expect("dl v");

                // Per-tag count summary, using the same SELECT_FMT_* mapping the
                // kernel emits. Names match `SampleFormat::try_from_cuda_tag`.
                let tag_name = |t: i32| -> &'static str {
                    match t {
                        1 => "F16",
                        2 => "BF16",
                        7 => "Q8_0",
                        8 => "Q8_1",
                        10 => "Q8KS",
                        12 => "Q5_0",
                        13 => "Q5_1",
                        15 => "Q4_0",
                        16 => "Q4_1",
                        18 => "Q4KS",
                        19 => "Q3_0",
                        20 => "Q3_1",
                        22 => "Q2_0",
                        23 => "Q2_1",
                        25 => "Q2_S",
                        26 => "Q2_A",
                        27 => "Q1_S",
                        28 => "Q0_V",
                        29 => "Q1_A",
                        30 => "Q0_X",
                        31 => "Q0_M2",
                        32 => "Q0_M4",
                        33 => "Q0",
                        99 => "FALLBACK",
                        _ => "OTHER",
                    }
                };
                let format_share = |tags: &[i32]| -> Vec<(&'static str, f64)> {
                    let total = tags.len() as f64;
                    let mut counts: std::collections::BTreeMap<&'static str, usize> =
                        std::collections::BTreeMap::new();
                    for &t in tags {
                        *counts.entry(tag_name(t)).or_insert(0) += 1;
                    }
                    let mut out: Vec<(&'static str, f64)> = counts
                        .into_iter()
                        .map(|(name, c)| (name, 100.0 * c as f64 / total))
                        .collect();
                    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    out
                };

                let k_share = format_share(&k_tags);
                let v_share = format_share(&v_tags);
                let render = |share: &[(&'static str, f64)]| -> String {
                    share
                        .iter()
                        .filter(|(_, p)| *p >= 0.05)
                        .map(|(n, p)| format!("{}={:.1}%", n, p))
                        .collect::<Vec<_>>()
                        .join("  ")
                };
                println!(
                    "    {:<8}  C{:>2}    K     [{}]",
                    model_name,
                    level,
                    render(&k_share)
                );
                println!(
                    "    {:<8}  C{:>2}    V     [{}]",
                    model_name,
                    level,
                    render(&v_share)
                );
            }
        }
    } // end for (this_model_name, this_dump_path, this_factors) in &datasets
}

// ---------------------------------------------------------------------------
// Iterative curve selection — greedy set-cover with the Q0_V kernel as oracle
// ---------------------------------------------------------------------------
//
// At each iteration we test every remaining (freq, shape) candidate by
// instantiating it at all 16 phases the production codebook expects, packing
// those 16 phases into the 256-curve runtime table (replicated 16× across
// buckets so the kernel's hierarchical Stage A + B + peak-bin path runs
// unchanged), and asking the kernel: how many blocks does this codebook
// claim under the production pass_metric? Whichever family wins the most
// blocks gets selected; those blocks are then removed from the working set
// for the next iteration. Repeat until 16 families are picked.
//
// This optimises curve coverage on the actual data Q0_V will see, accounting
// for redundancy: the second-best family in raw L2 may overlap heavily with
// the first; greedy selection picks each new family on the *residual* the
// previous picks didn't already cover.
//
// Output: 16 (freq, shape) families per side. Run as:
//   cargo test --release --features cuda,dont_check --lib --package candle-nn \
//     kv_stats_tests::test_q0_v_iterative_curve_selection -- --nocapture
//
// ---------------------------------------------------------------------------
#[cfg(feature = "cuda")]
#[test]
fn test_q0_v_iterative_curve_selection() {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Device;
    use rayon::prelude::*;
    use std::ffi::{c_int, c_void};
    use std::time::Instant;

    // Local FFI declaration — the symbol is linked via candle-kernels' static
    // archive, but the kv_cache crate doesn't depend on candle-kernels
    // directly so we declare the C ABI here.
    extern "C" {
        fn run_roundtrip_q0_v_runtime(
            src: *const c_void,
            recon: *mut c_void,
            num_blocks: c_int,
            outer: f32,
            curve_table_flat: *const c_void,
            scale_table_bits: *const c_void,
            centroid_table_bits_flat: *const c_void,
            peak_curve_indices: *const c_void,
            peak_bin_offsets: *const c_void,
        );
    }

    let qwen3_path = data_path(QWEN3_DUMP_REL_PATH);
    let llama_path = data_path(LLAMA_DUMP_REL_PATH);
    if !qwen3_path.exists() && !llama_path.exists() {
        println!("test_q0_v_iterative_curve_selection: dump files absent, skipping.");
        return;
    }

    let dev = match Device::new_cuda(0) {
        Ok(Device::Cuda(d)) => d,
        _ => {
            println!("CUDA not available, skipping.");
            return;
        }
    };
    let stream = dev.cuda_stream();

    // ============================================================
    // Helpers
    // ============================================================

    // Quantise a [-1, +1] curve to i8 [-127, +127].
    let quantise_i8 = |c: &[f32; 32]| -> [i8; 32] {
        let mut out = [0i8; 32];
        for i in 0..32 {
            let q = (c[i].clamp(-1.0, 1.0) * 127.0).round() as i32;
            out[i] = q.clamp(-127, 127) as i8;
        }
        out
    };

    // Build a 256-entry curve table by replicating one (16-phase) family.
    // Layout matches production: [bucket_b * 16 + phase_p] for b,p ∈ 0..16.
    // We replicate the same 16 phases across all 16 buckets so Stage A's
    // 16-rep scan picks any bucket (they all tie) and Stage B searches the
    // 16 phases of that bucket. Net effect: kernel scores 16 unique curves,
    // exactly what we want for "how well does this family fit".
    // 8 buckets × 32 lane-shift phases = 256 entries (still fills the 8-bit
    // curve_idx). Each phase p of family F is a hard cyclic shift of F's
    // canonical phase-0 curve: phases[p][lane] = phase0[(lane + p) mod 32].
    // The kernel's Stage A walks `curve_table[b * 16]` for b ∈ 0..16, which
    // hits phase 0 and phase 16 of each of the 8 buckets — 16 reps spanning
    // 8 families × 2 phase-strides. Stage B then searches 16 phases (phases
    // 0..15 of the chosen bucket); the remaining 16 phases (16..31) get
    // covered by Stage C peak-bin refinement when the target's peak doesn't
    // align with the Stage B half.
    // 7-bit curve_idx → 128-slot table. Side picks how phases divide the
    // 128 slots:
    //   K: 8 buckets × 16 phases (stride-2 lane shifts)
    //   V: 16 buckets × 8 phases (stride-4 lane shifts) — coarser phase
    //      coverage but Stage A picks among more bucket reps; the V stats
    //      are looser in residual quality so phase resolution matters less
    //      than R2 candidate diversity.
    // The kernel hierarchical search (Stage A: 8 reps at b*16; Stage B: 16
    // phases of chosen bucket) sees the same 128-slot table either way.
    let build_replicated_curve_table = |phases: &[[i8; 32]]| -> Vec<i8> {
        let n_phases = phases.len();
        debug_assert!(n_phases > 0 && 128 % n_phases == 0);
        let n_buckets = 128 / n_phases;
        let mut buf = Vec::with_capacity(128 * 32);
        for _b in 0..n_buckets {
            for p in 0..n_phases {
                buf.extend_from_slice(&phases[p]);
            }
        }
        buf
    };

    // Compute peak_curve_indices (256) + peak_bin_offsets (33) for a given
    // 256-curve table. Same logic as the calibration's peak-bin emit.
    let build_peak_tables = |curves_i8: &[i8]| -> (Vec<u8>, Vec<u16>) {
        debug_assert_eq!(curves_i8.len(), 128 * 32);
        let mut peak: [u8; 128] = [0; 128];
        for slot in 0..128 {
            let mut best_v = -1.0f32;
            let mut best_l = 0u8;
            for lane in 0..32 {
                let v = curves_i8[slot * 32 + lane] as i32;
                let a = v.unsigned_abs() as f32;
                if a > best_v {
                    best_v = a;
                    best_l = lane as u8;
                }
            }
            peak[slot] = best_l;
        }
        let mut order: Vec<u8> = (0u16..128).map(|x| x as u8).collect();
        order.sort_by(|&a, &b| peak[a as usize].cmp(&peak[b as usize]).then(a.cmp(&b)));
        let mut hist = [0u32; 32];
        for &p in peak.iter() {
            hist[p as usize] += 1;
        }
        let mut offsets = [0u16; 33];
        let mut acc = 0u16;
        for i in 0..32 {
            offsets[i] = acc;
            acc += hist[i] as u16;
        }
        offsets[32] = acc;
        (order, offsets.to_vec())
    };

    // Side-specific pass_metric (matches `compute_pass_metric` in the kernel).
    enum SideMetric {
        K, // mean_top4(|err|) / head_amax
        V, // sqrt(mean(err²)) / head_amax
    }
    let pass_metric = |side: &SideMetric, orig: &[f32; 32], recon: &[f32; 32]| -> f32 {
        match side {
            SideMetric::K => {
                let mut errs: [f32; 32] = std::array::from_fn(|i| (orig[i] - recon[i]).abs());
                errs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                (errs[0] + errs[1] + errs[2] + errs[3]) * 0.25
            }
            SideMetric::V => {
                let mse: f32 = (0..32)
                    .map(|i| {
                        let d = orig[i] - recon[i];
                        d * d
                    })
                    .sum::<f32>()
                    / 32.0;
                mse.sqrt()
            }
        }
    };

    // ============================================================
    // Shared scale + centroid tables (one set per side)
    // ============================================================
    //
    // Iterative curve selection runs the kernel encoder which needs
    // scale_table_bits and centroid_table_bits to be present. We use the
    // existing production K/V tables (from k_quants::q0_v_tables) — they're
    // the calibrated codebook, identical to what's compiled into the kernel.
    // After all 16 curves are picked, scale/centroid would normally be
    // recomputed on the new block→curve assignments; that re-derivation is
    // out of scope for this test (which just outputs the picked curves).
    use candle::quantized::k_quants::q0_v_tables as q0v_tables;

    // ============================================================
    // Candidate pool — same parametric grid as the existing calibration
    // ============================================================
    const N_FREQS: usize = 21;
    const N_PHASES: usize = 16;
    let frequencies: Vec<f32> = {
        let mut f = Vec::with_capacity(N_FREQS);
        f.push(0.0);
        for i in 0..12 {
            f.push(0.0625 + (0.5 - 0.0625) * (i as f32) / 11.0);
        }
        for i in 1..9 {
            let t = i as f32 / 8.0;
            f.push(0.5 * (8.0_f32 / 0.5).powf(t));
        }
        f
    };
    let _phases: Vec<f32> = (0..N_PHASES)
        .map(|i| 2.0 * std::f32::consts::PI * (i as f32) / (N_PHASES as f32))
        .collect();
    let shapes = [ShapeChar::Sine, ShapeChar::Triangle, ShapeChar::Sharp];
    let n_families = N_FREQS * shapes.len();
    let mut candidate_families: Vec<(f32, ShapeChar)> = Vec::with_capacity(n_families);
    for &freq in &frequencies {
        for &shape in &shapes {
            candidate_families.push((freq, shape));
        }
    }

    // Pre-build all 32-phase i8 curve tables for every family (cached).
    // Phase 0 is `generate_curve(freq, 0.0, shape)`. Phases 1..31 are pure
    // cyclic lane shifts of phase 0:
    //     phases32[p][lane] = phase0[(lane + p) mod 32]
    // This gives full lane-resolution phase coverage (one phase per lane)
    // while costing only one curve generation per family — phases are
    // generated by index rotation, not by parametric re-evaluation.
    // Per-family phase tables. Side-specific phase resolution: K gets 16
    // phases (stride-2 cyclic shifts), V gets 8 (stride-4). Both fill the
    // 128-slot kernel curve table via build_replicated_curve_table.
    let make_family_phase_tables = |n_phases: usize| -> Vec<Vec<[i8; 32]>> {
        debug_assert!(n_phases > 0 && 32 % n_phases == 0);
        let stride = 32 / n_phases;
        candidate_families
            .iter()
            .map(|&(freq, shape)| {
                let phase0 = quantise_i8(&generate_curve(freq, 0.0, shape));
                let mut out: Vec<[i8; 32]> = Vec::with_capacity(n_phases);
                for p in 0..n_phases {
                    let mut v = [0i8; 32];
                    for lane in 0..32 {
                        v[lane] = phase0[(lane + stride * p) % 32];
                    }
                    out.push(v);
                }
                out
            })
            .collect()
    };
    let make_family_runtime_tables =
        |phase_tables: &[Vec<[i8; 32]>]| -> Vec<(Vec<i8>, Vec<u8>, Vec<u16>)> {
            phase_tables
                .par_iter()
                .map(|phases| {
                    let curves = build_replicated_curve_table(phases.as_slice());
                    let (peak_idx, peak_off) = build_peak_tables(&curves);
                    (curves, peak_idx, peak_off)
                })
                .collect()
        };
    let family_phase_tables_k = make_family_phase_tables(16);
    let _family_phase_tables_v = make_family_phase_tables(8);
    let family_runtime_tables_k = make_family_runtime_tables(&family_phase_tables_k);
    let _family_runtime_tables_v = make_family_runtime_tables(&_family_phase_tables_v);

    // ============================================================
    // Per-side iterative selection
    // ============================================================
    let run_side = |side_label: &str,
                    side: SideMetric,
                    blocks: &[[f32; 32]],
                    head_amax: &[f32],
                    scale_bits: &[u16; 32],
                    centroid_bits: &[[u16; 16]; 32],
                    pass_threshold: f32,
                    n_phases: usize,
                    r2_max_picks: usize,
                    family_runtime_tables: &[(Vec<i8>, Vec<u8>, Vec<u16>)],
                    dump_n_samples: usize,
                    kmeans_rho_min: f32,
                    kmeans_smooth_taps: usize,
                    kmeans_n_strata: usize,
                    n_new_curves: usize,
                    sign_canonicalize: bool,
                    kmeans_skip_lane0: bool,
                    strata_by_second_peak: bool| {
        println!("\n========================================================================");
        println!(
            "  Iterative curve selection — side = {} ({} blocks, {} families × 32 lane-shift phases, picking 8)",
            side_label.to_uppercase(),
            blocks.len(),
            n_families
        );
        println!(
            "  Pass metric ≤ {:.5} ({} threshold @ filter operating point)",
            pass_threshold,
            match side {
                SideMetric::K => "K",
                SideMetric::V => "V",
            }
        );
        println!("========================================================================");

        // ────────────────────────────────────────────────────────────────
        // Step 1: head-amax-normalise + collect per-block (actual_scale,
        //         actual_centroid). These mirror what the kernel computes
        //         internally (compute_target_and_indices).
        // ────────────────────────────────────────────────────────────────
        let n_total = blocks.len();
        let mut head_norm_blocks: Vec<[f32; 32]> = Vec::with_capacity(n_total);
        let mut block_scale: Vec<f32> = Vec::with_capacity(n_total);
        let mut block_centroid: Vec<f32> = Vec::with_capacity(n_total);
        let mut active_idx: Vec<usize> = Vec::with_capacity(n_total);
        for (i, (blk, &ha)) in blocks.iter().zip(head_amax.iter()).enumerate() {
            if ha <= 0.0 {
                continue;
            }
            let inv = 1.0 / ha;
            let hn: [f32; 32] = std::array::from_fn(|j| blk[j] * inv);
            let mean: f32 = hn.iter().sum::<f32>() / 32.0;
            let scale: f32 = hn.iter().map(|x| (x - mean).abs()).fold(0.0f32, f32::max);
            head_norm_blocks.push(hn);
            block_scale.push(scale);
            block_centroid.push(mean);
            active_idx.push(i);
        }
        let n_normalised = active_idx.len();

        // ────────────────────────────────────────────────────────────────
        // Step 2: calibrate a 32-entry scale codebook by quantile-binning
        //         the actual_scale distribution (per-bin median).
        // ────────────────────────────────────────────────────────────────
        let mut scales_sorted = block_scale.clone();
        scales_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cal_scale_table: [f32; 32] = std::array::from_fn(|k| {
            let lo = (k * n_normalised) / 32;
            let hi = ((k + 1) * n_normalised) / 32;
            let mid = ((lo + hi) / 2).min(n_normalised.saturating_sub(1));
            scales_sorted[mid]
        });

        // ────────────────────────────────────────────────────────────────
        // Step 3: per-block scale_idx (nearest entry).
        // ────────────────────────────────────────────────────────────────
        let pick_scale_idx = |s: f32| -> usize {
            let mut best = 0usize;
            let mut best_err = f32::INFINITY;
            for (i, &v) in cal_scale_table.iter().enumerate() {
                let e = (s - v).abs();
                if e < best_err {
                    best_err = e;
                    best = i;
                }
            }
            best
        };

        // ────────────────────────────────────────────────────────────────
        // Step 4: bucket blocks by scale_idx, calibrate [32][8] centroid
        //         table by quantile-binning centroids within each bucket.
        // ────────────────────────────────────────────────────────────────
        let mut centroid_buckets: Vec<Vec<f32>> = vec![Vec::new(); 32];
        let mut block_scale_idx: Vec<u8> = Vec::with_capacity(n_normalised);
        for (s, c) in block_scale.iter().zip(block_centroid.iter()) {
            let idx = pick_scale_idx(*s);
            block_scale_idx.push(idx as u8);
            centroid_buckets[idx].push(*c);
        }
        let mut cal_centroid_table: [[f32; 16]; 32] = [[0.0f32; 16]; 32];
        for (b, cs) in centroid_buckets.iter_mut().enumerate() {
            if cs.is_empty() {
                continue;
            }
            cs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = cs.len();
            for k in 0..16 {
                let lo = (k * n) / 16;
                let hi = ((k + 1) * n) / 16;
                let mid = ((lo + hi) / 2).min(n - 1);
                cal_centroid_table[b][k] = cs[mid];
            }
        }

        // ────────────────────────────────────────────────────────────────
        // Step 5: keep inputs in head-amax-relative space (the kernel still
        //         expects xi in [-1,+1] head-amax units). Just flatten the
        //         head-amax-normalised blocks into the upload buffer.
        // ────────────────────────────────────────────────────────────────
        let mut normalised: Vec<f32> = Vec::with_capacity(n_normalised * 32);
        for hn in head_norm_blocks.iter() {
            normalised.extend_from_slice(hn);
        }
        println!(
            "  Calibrated scale/centroid tables from {} head-amax-normalised blocks.",
            n_normalised
        );

        // Print the calibrated scale + centroid tables so the operator can
        // inspect them (these were derived from THIS dataset).
        print!("  cal_scale_table[32] = [");
        for (i, &s) in cal_scale_table.iter().enumerate() {
            if i % 8 == 0 {
                print!("\n    ");
            }
            print!("{:>9.5}", s);
            if i < 31 {
                print!(",");
            }
        }
        println!("\n  ]");
        println!("  cal_centroid_table[32][16]:");
        for (b, row) in cal_centroid_table.iter().enumerate() {
            print!(
                "    s={:>2}  scale_q={:>8.5}  centroids=[",
                b, cal_scale_table[b]
            );
            for (k, &c) in row.iter().enumerate() {
                print!("{:>+8.5}", c);
                if k < 15 {
                    print!(",");
                }
            }
            println!("]");
        }

        // Convert calibrated f32 tables to f16 bits for the kernel.
        let f32_to_f16_bits = |v: f32| -> u16 {
            // Use half::f16 if available, else round-trip via __float2half-equivalent.
            // We can build f16 bits manually to avoid an external dep.
            let bits = v.to_bits();
            let sign = ((bits >> 31) & 0x1) as u16;
            let mut exp = ((bits >> 23) & 0xff) as i32;
            let mut mant = (bits & 0x7fffff) as u32;
            if exp == 0xff {
                // Inf / NaN
                return (sign << 15) | 0x7c00 | (if mant != 0 { 0x200 } else { 0 });
            }
            // Rebias.
            exp = exp - 127 + 15;
            if exp >= 31 {
                return (sign << 15) | 0x7c00; // overflow → inf
            }
            if exp <= 0 {
                if exp < -10 {
                    return sign << 15;
                }
                // Subnormal.
                mant |= 0x800000;
                let shift = (1 - exp) as u32 + 13;
                let m_sub = (mant >> shift) as u16;
                return (sign << 15) | m_sub;
            }
            let mant_h = (mant >> 13) as u16;
            (sign << 15) | ((exp as u16) << 10) | mant_h
        };

        let cal_scale_bits: [u16; 32] = std::array::from_fn(|i| {
            // Kernel formula: s_units = scale_baked * 127. We want
            // s_units == cal_scale_table[i], so scale_baked = s_units / 127.
            f32_to_f16_bits(cal_scale_table[i] / 127.0)
        });
        let mut cal_centroid_bits: [u16; 32 * 16] = [0u16; 32 * 16];
        for (s, row) in cal_centroid_table.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                cal_centroid_bits[s * 16 + c] = f32_to_f16_bits(v);
            }
        }

        // Upload our calibrated tables. The kernel's curve-search step will
        // now use scale/centroid bins fitted to THIS dataset — no double
        // quantisation, no production-table mismatch.
        let _ = scale_bits;
        let _ = centroid_bits;
        let scale_bits_gpu = dev
            .memcpy_stod(cal_scale_bits.as_slice())
            .expect("scale upload");
        let centroid_bits_gpu = dev
            .memcpy_stod(cal_centroid_bits.as_slice())
            .expect("centroid upload");

        // Working set: indices into `active_idx` of blocks still up for grabs.
        // Initially all blocks are active.
        let mut working_set: Vec<usize> = (0..n_normalised).collect();

        // Track which families are still candidates.
        let mut family_used = vec![false; n_families];

        // Recon buffer (assigned from memcpy_dtov each kernel call; the
        // initial empty Vec is replaced on first use).
        #[allow(unused_assignments)]
        let mut recon_host: Vec<f32> = Vec::new();
        let recon_gpu = stream
            .alloc_zeros::<f32>(n_normalised * 32)
            .expect("recon alloc");
        let normalised_gpu = dev.memcpy_stod(&normalised).expect("normalised upload");

        let mut selected_families: Vec<(usize, f32, ShapeChar, usize)> = Vec::with_capacity(8);

        for iter in 0..8 {
            let iter_start = Instant::now();

            // For every still-unused family, run the runtime kernel and
            // count blocks in the WORKING SET that pass the threshold.
            // Parallelise the per-family kernel launches: launch one per
            // family sequentially, but the per-family Rust pass_metric
            // computation runs on rayon afterwards. (The kernel itself is
            // serialised on the single CUDA stream — typical 1M-block
            // round-trip is < 100 ms so 50 sequential launches per
            // iteration is ~5 s.)
            let mut best_family: Option<usize> = None;
            let mut best_wins: usize = 0;
            let mut all_results: Vec<(usize, usize)> = Vec::with_capacity(n_families);

            for (fi, (curves, peak_idx, peak_off)) in family_runtime_tables.iter().enumerate() {
                if family_used[fi] {
                    continue;
                }

                // Upload this family's tables.
                let curves_gpu = dev.memcpy_stod(curves).expect("curves upload");
                let peak_idx_gpu = dev.memcpy_stod(peak_idx).expect("peak_idx upload");
                let peak_off_gpu = dev.memcpy_stod(peak_off).expect("peak_off upload");

                // Round-trip the entire normalised set (we'll filter to
                // working_set in the score loop below).
                let (src_p, _g1) = normalised_gpu.device_ptr(&stream);
                let (recon_p, _g2) = recon_gpu.device_ptr(&stream);
                let (curves_p, _g3) = curves_gpu.device_ptr(&stream);
                let (scale_p, _g4) = scale_bits_gpu.device_ptr(&stream);
                let (cent_p, _g5) = centroid_bits_gpu.device_ptr(&stream);
                let (pi_p, _g6) = peak_idx_gpu.device_ptr(&stream);
                let (po_p, _g7) = peak_off_gpu.device_ptr(&stream);
                unsafe {
                    run_roundtrip_q0_v_runtime(
                        src_p as *const c_void,
                        recon_p as *mut c_void,
                        n_normalised as i32,
                        1.0,
                        curves_p as *const c_void,
                        scale_p as *const c_void,
                        cent_p as *const c_void,
                        pi_p as *const c_void,
                        po_p as *const c_void,
                    );
                }
                stream.synchronize().expect("sync");
                recon_host = dev.memcpy_dtov(&recon_gpu).expect("recon download");

                // Score: pass_metric on per-block-normalised data, compared
                // directly against the threshold (no head_amax conversion).
                let wins: usize = working_set
                    .par_iter()
                    .filter(|&&w| {
                        let off = w * 32;
                        let orig: [f32; 32] = std::array::from_fn(|j| normalised[off + j]);
                        let recon: [f32; 32] = std::array::from_fn(|j| recon_host[off + j]);
                        pass_metric(&side, &orig, &recon) <= pass_threshold
                    })
                    .count();

                // Diagnostic: on iter 0, dump the residual distribution for
                // the very first family scored so we can see what threshold
                // makes sense in the pre-normalised regime.
                if iter == 0 && all_results.is_empty() {
                    let mut residuals: Vec<f32> = working_set
                        .par_iter()
                        .map(|&w| {
                            let off = w * 32;
                            let orig: [f32; 32] = std::array::from_fn(|j| normalised[off + j]);
                            let recon: [f32; 32] = std::array::from_fn(|j| recon_host[off + j]);
                            pass_metric(&side, &orig, &recon)
                        })
                        .collect();
                    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = residuals.len();
                    let pct = |q: f32| residuals[((q * (n - 1) as f32) as usize).min(n - 1)];
                    println!(
                        "    [diag iter 0 family[{:>3}]] residual quantiles: \
                         p05={:.4} p25={:.4} p50={:.4} p75={:.4} p90={:.4} p99={:.4}  threshold={:.5}",
                        fi,
                        pct(0.05),
                        pct(0.25),
                        pct(0.50),
                        pct(0.75),
                        pct(0.90),
                        pct(0.99),
                        pass_threshold,
                    );
                }

                all_results.push((fi, wins));
                if wins > best_wins {
                    best_wins = wins;
                    best_family = Some(fi);
                }
            }

            // Diagnostic: top-10 + best-of-each-shape so Square/Pulse are
            // visible even when they don't break into the leaderboard.
            let mut sorted = all_results.clone();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            let pct_of = |w: usize| 100.0 * (w as f64) / n_normalised as f64;
            print!("    top-10: ");
            for (rank, (fi, wins)) in sorted.iter().take(10).enumerate() {
                let (freq, shape) = candidate_families[*fi];
                if rank > 0 {
                    print!(", ");
                }
                print!(
                    "[{:>3}]{}@{:.2}={:>5.2}%",
                    fi,
                    shape.label(),
                    freq,
                    pct_of(*wins)
                );
            }
            println!();
            // Best of each shape.
            print!("    best-by-shape: ");
            let all_shapes = [ShapeChar::Sine, ShapeChar::Triangle, ShapeChar::Sharp];
            let mut first = true;
            for &sh in &all_shapes {
                if let Some((fi, wins)) = sorted
                    .iter()
                    .find(|(fi, _)| candidate_families[*fi].1 == sh)
                {
                    let (freq, _) = candidate_families[*fi];
                    if !first {
                        print!(", ");
                    }
                    first = false;
                    print!("{}@{:.2}={:>5.2}%", sh.label(), freq, pct_of(*wins));
                }
            }
            println!();

            let chosen_fi = match best_family {
                Some(fi) => fi,
                None => {
                    println!(
                        "  iter {}: no family found a passing block — stopping",
                        iter
                    );
                    break;
                }
            };
            let (chosen_freq, chosen_shape) = candidate_families[chosen_fi];
            let cov_pct = 100.0 * best_wins as f64 / n_normalised as f64;

            // Abort BEFORE committing the pick: if the best family covered
            // <1%, don't add it to selected_families and don't drop its
            // would-be-won blocks from working_set. R2 then sees the full
            // residual (including those blocks), which keeps R1 and R2
            // candidate sets fairly comparable.
            if cov_pct < 1.0 {
                println!(
                    "\n  ✗ ABORT — iter {} winning family[{}] f={:.2} {} covered only {:.3}% (< 1.0% floor) — not added.",
                    iter, chosen_fi, chosen_freq, chosen_shape.label(), cov_pct
                );
                println!(
                    "    Dumping 16 random samples of the {} remaining normalised chunks ({}-side).",
                    working_set.len(),
                    side_label.to_uppercase()
                );
                println!("    Plot: blocks are already per-block normalised (block / block_amax).");
                println!("    Top→bottom = +1.0→−1.0, lanes 0..31 left→right.\n");

                // Deterministic LCG seeded from side label + iter so re-runs
                // are reproducible.
                let mut seed: u64 = 0x9E37_79B9_7F4A_7C15
                    ^ (iter as u64).wrapping_mul(0x100_0000_01B3)
                    ^ side_label
                        .bytes()
                        .fold(0u64, |a, b| a.wrapping_mul(131).wrapping_add(b as u64));
                let mut next_u64 = || {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    seed
                };

                // Lag-1 autocorrelation of a 32-lane block — used to filter
                // near-white-noise candidates out of the SAMPLE display
                // (the noisy blocks stay in the working set; only the dump
                // is filtered). White noise has ρ₁ ≈ 0; smooth curves and
                // sharp/oscillating shapes both have |ρ₁| ≫ 0. Conservative
                // threshold (0.15) — we only filter blocks with essentially
                // no lag-1 structure, not anything we might want to see.
                let lag1_autocorr = |b: &[f32; 32]| -> f32 {
                    let mean: f32 = b.iter().sum::<f32>() / 32.0;
                    let mut num = 0.0f32;
                    let mut den = 0.0f32;
                    for i in 0..32 {
                        let d = b[i] - mean;
                        den += d * d;
                        if i + 1 < 32 {
                            num += d * (b[i + 1] - mean);
                        }
                    }
                    if den > 0.0 {
                        num / den
                    } else {
                        0.0
                    }
                };
                const NOISE_RHO_FLOOR: f32 = 0.15;

                // Compute xi_norm for a working-set block (curve-fit residual
                // — what the kernel implicitly tries to fit a curve to).
                let xi_norm_for = |bidx: usize| -> [f32; 32] {
                    let off = bidx * 32;
                    let scale_idx = block_scale_idx[bidx] as usize;
                    let scale_q = cal_scale_table[scale_idx];
                    let actual_c = block_centroid[bidx];
                    let row = &cal_centroid_table[scale_idx];
                    let mut best_c = 0usize;
                    let mut best_e = f32::INFINITY;
                    for (kk, &cv) in row.iter().enumerate() {
                        let e = (actual_c - cv).abs();
                        if e < best_e {
                            best_e = e;
                            best_c = kk;
                        }
                    }
                    let centroid_q = row[best_c];
                    let inv_s = if scale_q > 0.0 { 1.0 / scale_q } else { 0.0 };
                    std::array::from_fn(|j| (normalised[off + j] - centroid_q) * inv_s)
                };

                let n_samples = dump_n_samples.min(working_set.len());
                let mut sample_block_indices: Vec<usize> = Vec::with_capacity(n_samples);
                let mut retries: usize = 0;
                let max_retries = 16 * n_samples; // cap so we exit even if pool is mostly noise
                let mut filtered_out: usize = 0;
                while sample_block_indices.len() < n_samples && retries < max_retries {
                    let pick = (next_u64() as usize) % working_set.len();
                    let bidx = working_set[pick];
                    if sample_block_indices.contains(&bidx) {
                        retries += 1;
                        continue;
                    }
                    let xi = xi_norm_for(bidx);
                    let rho = lag1_autocorr(&xi);
                    if rho.abs() < NOISE_RHO_FLOOR {
                        filtered_out += 1;
                        retries += 1;
                        continue;
                    }
                    sample_block_indices.push(bidx);
                }
                if sample_block_indices.len() < n_samples {
                    // Couldn't find enough structured blocks within retry
                    // budget — top up with whatever passes the dedup check
                    // so the operator still gets samples (annotated with
                    // their ρ₁ so they can see we relaxed the filter).
                    while sample_block_indices.len() < n_samples {
                        let pick = (next_u64() as usize) % working_set.len();
                        let bidx = working_set[pick];
                        if !sample_block_indices.contains(&bidx) {
                            sample_block_indices.push(bidx);
                        }
                    }
                }
                println!(
                    "    Noise filter: |ρ₁| ≥ {:.2}.  Rejected {} candidate(s) before locking 16 samples.",
                    NOISE_RHO_FLOOR, filtered_out
                );

                let plot_height: usize = 9;
                let plot_block = |curve: &[f32; 32]| -> Vec<String> {
                    let mut rows = vec![String::new(); plot_height];
                    for lane in 0..32 {
                        let v = curve[lane].clamp(-1.0, 1.0);
                        let r_f = (1.0 - v) * 0.5 * (plot_height - 1) as f32;
                        let r = (r_f.round() as isize).clamp(0, plot_height as isize - 1) as usize;
                        for (ri, row) in rows.iter_mut().enumerate() {
                            if ri == r {
                                row.push('●');
                            } else if ri == plot_height / 2 {
                                row.push('─');
                            } else {
                                row.push(' ');
                            }
                        }
                    }
                    rows
                };

                println!("    (Each sample shown as the curve-fit residual:");
                println!("       xi_norm = (block/head_amax − centroid_q) / scale_q,");
                println!("     using the calibrated tables, then cyclically rotated");
                println!("     so the |xi_norm|-peak lane lands at position 0 (peak-");
                println!("     leading canonical form). Kernel input is still head-");
                println!("     amax-relative — this normalisation is for visualisation.)");
                println!();

                for (k, &w) in sample_block_indices.iter().enumerate() {
                    // Head-amax-normalised block (kernel input domain).
                    let off = w * 32;
                    let block_head: [f32; 32] = std::array::from_fn(|j| normalised[off + j]);

                    // Look up this block's calibrated (scale_idx, centroid_idx)
                    // and produce the curve-fit residual the kernel implicitly
                    // works on (xi_norm in approximately [-1, +1] with mean ~0).
                    let scale_idx = block_scale_idx[w] as usize;
                    let scale_q = cal_scale_table[scale_idx];
                    let actual_c = block_centroid[w];
                    let row = &cal_centroid_table[scale_idx];
                    let mut best_c = 0usize;
                    let mut best_e = f32::INFINITY;
                    for (kk, &cv) in row.iter().enumerate() {
                        let e = (actual_c - cv).abs();
                        if e < best_e {
                            best_e = e;
                            best_c = kk;
                        }
                    }
                    let centroid_q = row[best_c];
                    let inv_s = if scale_q > 0.0 { 1.0 / scale_q } else { 0.0 };
                    let xi_norm_raw: [f32; 32] =
                        std::array::from_fn(|j| (block_head[j] - centroid_q) * inv_s);

                    // Peak-center: cyclically rotate so the lane with max
                    // |xi_norm| lands at position 0. This aligns shapes
                    // across samples (any phase becomes the "peak-leading"
                    // canonical form), making it easier to spot recurring
                    // shapes when scanning the dump.
                    let mut peak_lane = 0usize;
                    let mut peak_abs = 0.0f32;
                    for (j, &v) in xi_norm_raw.iter().enumerate() {
                        let a = v.abs();
                        if a > peak_abs {
                            peak_abs = a;
                            peak_lane = j;
                        }
                    }
                    let xi_norm: [f32; 32] =
                        std::array::from_fn(|j| xi_norm_raw[(j + peak_lane) % 32]);
                    let xi_norm_i8 = quantise_i8(&xi_norm);

                    let rho1 = lag1_autocorr(&xi_norm);
                    println!(
                        "    [sample {:>2}] block_idx={}  scale_idx={:>2} (scale_q={:.5})  centroid_idx={} (centroid_q={:+.5})  peak_lane={:>2}  ρ₁={:+.3}",
                        k, w, scale_idx, scale_q, best_c, centroid_q, peak_lane, rho1
                    );
                    let plot = plot_block(&xi_norm);
                    for line in plot.iter() {
                        println!("      {}", line);
                    }

                    print!("      xi_norm i8  (peak-centered): [");
                    for (lane, v) in xi_norm_i8.iter().enumerate() {
                        print!("{:>4}", v);
                        if lane < 31 {
                            print!(",");
                        }
                    }
                    println!(" ]");

                    print!("      xi_norm f32 (peak-centered): [");
                    for (lane, v) in xi_norm.iter().enumerate() {
                        print!("{:>+6.3}", v);
                        if lane < 31 {
                            print!(",");
                        }
                    }
                    println!(" ]");
                    println!();
                }

                println!(
                    "    {}-side iter {} stopped: only {:.3}% covered (< 1.0% floor); {} samples dumped above.",
                    side_label.to_uppercase(),
                    iter,
                    cov_pct,
                    n_samples
                );

                // ============================================================
                // Polynomial fit across the remaining (unserved) blocks.
                // For each block, compute xi_norm (the curve-fit residual
                // already discussed), then least-squares-fit T_0..T_D
                // Chebyshev coefficients on the 32 lanes mapped to [-1, +1].
                // Histogram each coefficient across all remaining blocks to
                // reveal the parameter distribution of what's left.
                // ============================================================
                const POLY_DEGREE: usize = 6;
                const POLY_NCOEF: usize = POLY_DEGREE + 1;

                // Basis: T_n(x_lane) where x_lane = 2k/31 - 1 ∈ [-1, +1].
                let x_lane: [f32; 32] = std::array::from_fn(|k| 2.0 * (k as f32) / 31.0 - 1.0);
                let mut basis: [[f32; POLY_NCOEF]; 32] = [[0.0; POLY_NCOEF]; 32];
                for k in 0..32 {
                    let x = x_lane[k];
                    basis[k][0] = 1.0;
                    if POLY_NCOEF > 1 {
                        basis[k][1] = x;
                    }
                    for n in 2..POLY_NCOEF {
                        basis[k][n] = 2.0 * x * basis[k][n - 1] - basis[k][n - 2];
                    }
                }

                // Build B^T B (POLY_NCOEF × POLY_NCOEF), invert via
                // Gauss-Jordan (small matrix). P = (B^T B)^-1 B^T gives the
                // pseudoinverse mapping [32] → [POLY_NCOEF].
                let mut btb = [[0.0f32; POLY_NCOEF]; POLY_NCOEF];
                for i in 0..POLY_NCOEF {
                    for j in 0..POLY_NCOEF {
                        let mut s = 0.0f32;
                        for k in 0..32 {
                            s += basis[k][i] * basis[k][j];
                        }
                        btb[i][j] = s;
                    }
                }
                // Gauss-Jordan invert (augmented [btb | I]).
                let mut aug = [[0.0f32; 2 * POLY_NCOEF]; POLY_NCOEF];
                for i in 0..POLY_NCOEF {
                    for j in 0..POLY_NCOEF {
                        aug[i][j] = btb[i][j];
                    }
                    aug[i][POLY_NCOEF + i] = 1.0;
                }
                for i in 0..POLY_NCOEF {
                    // Pivot.
                    let mut pivot = i;
                    let mut pmax = aug[i][i].abs();
                    for r in (i + 1)..POLY_NCOEF {
                        if aug[r][i].abs() > pmax {
                            pmax = aug[r][i].abs();
                            pivot = r;
                        }
                    }
                    if pivot != i {
                        aug.swap(i, pivot);
                    }
                    let div = aug[i][i];
                    for j in 0..(2 * POLY_NCOEF) {
                        aug[i][j] /= div;
                    }
                    for r in 0..POLY_NCOEF {
                        if r == i {
                            continue;
                        }
                        let factor = aug[r][i];
                        for j in 0..(2 * POLY_NCOEF) {
                            aug[r][j] -= factor * aug[i][j];
                        }
                    }
                }
                let mut btb_inv = [[0.0f32; POLY_NCOEF]; POLY_NCOEF];
                for i in 0..POLY_NCOEF {
                    for j in 0..POLY_NCOEF {
                        btb_inv[i][j] = aug[i][POLY_NCOEF + j];
                    }
                }
                // P[i][k] = sum_j btb_inv[i][j] * basis[k][j]
                let mut p_mat = [[0.0f32; 32]; POLY_NCOEF];
                for i in 0..POLY_NCOEF {
                    for k in 0..32 {
                        let mut s = 0.0f32;
                        for j in 0..POLY_NCOEF {
                            s += btb_inv[i][j] * basis[k][j];
                        }
                        p_mat[i][k] = s;
                    }
                }

                // Per-block fit: coefs = P · xi_norm.
                let coefs_per_block: Vec<[f32; POLY_NCOEF]> = working_set
                    .par_iter()
                    .map(|&w| {
                        let off = w * 32;
                        let scale_idx = block_scale_idx[w] as usize;
                        let scale_q = cal_scale_table[scale_idx];
                        let actual_c = block_centroid[w];
                        let row = &cal_centroid_table[scale_idx];
                        let mut best_c = 0usize;
                        let mut best_e = f32::INFINITY;
                        for (kk, &cv) in row.iter().enumerate() {
                            let e = (actual_c - cv).abs();
                            if e < best_e {
                                best_e = e;
                                best_c = kk;
                            }
                        }
                        let centroid_q = row[best_c];
                        let inv_s = if scale_q > 0.0 { 1.0 / scale_q } else { 0.0 };
                        let xi_norm: [f32; 32] =
                            std::array::from_fn(|j| (normalised[off + j] - centroid_q) * inv_s);
                        let mut coefs = [0.0f32; POLY_NCOEF];
                        for i in 0..POLY_NCOEF {
                            let mut s = 0.0f32;
                            for k in 0..32 {
                                s += p_mat[i][k] * xi_norm[k];
                            }
                            coefs[i] = s;
                        }
                        coefs
                    })
                    .collect();

                println!();
                println!(
                    "  ── Polynomial fit across {} remaining blocks ──",
                    working_set.len()
                );
                println!(
                    "  Basis: Chebyshev T_0..T_{} on x ∈ [-1, +1] (lane k → x = 2k/31 − 1).",
                    POLY_DEGREE
                );
                println!("  Histograms (50 bins, 1st–99th percentile range):");
                println!();

                // Histogram each coefficient.
                const N_BINS: usize = 50;
                const BAR_MAX_W: usize = 50;
                for n in 0..POLY_NCOEF {
                    let mut vals: Vec<f32> = coefs_per_block.iter().map(|c| c[n]).collect();
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let nv = vals.len();
                    let pq = |q: f32| vals[((q * (nv - 1) as f32) as usize).min(nv - 1)];
                    let lo = pq(0.01);
                    let hi = pq(0.99);
                    let median = pq(0.50);
                    let mean: f32 = vals.iter().copied().sum::<f32>() / (nv as f32);
                    let span = hi - lo;
                    if span <= 0.0 {
                        println!("    T_{}  (range collapsed; all values ≈ {:.4})", n, lo);
                        continue;
                    }
                    let mut bins = vec![0usize; N_BINS];
                    for &v in &vals {
                        let idx = (((v - lo) / span) * N_BINS as f32) as i32;
                        let idx = idx.clamp(0, N_BINS as i32 - 1) as usize;
                        bins[idx] += 1;
                    }
                    let max_bin = bins.iter().max().copied().unwrap_or(1);
                    println!(
                        "    T_{}  p01={:+.3}  median={:+.3}  mean={:+.3}  p99={:+.3}",
                        n, lo, median, mean, hi
                    );
                    for (b_idx, &c) in bins.iter().enumerate() {
                        let bin_lo = lo + (b_idx as f32) * span / N_BINS as f32;
                        let bar_w = ((c as f32 / max_bin as f32) * BAR_MAX_W as f32) as usize;
                        let bar = "█".repeat(bar_w);
                        // Mark the zero crossing visibly.
                        let mark = if bin_lo <= 0.0 && (bin_lo + span / N_BINS as f32) > 0.0 {
                            "│0"
                        } else {
                            "  "
                        };
                        println!("      {:>+7.3} {}: {} {}", bin_lo, mark, bar, c);
                    }
                    println!();
                }
                // ============================================================
                // Round 2: re-derive 64 curves from the *current* residual
                //          via k-means before each pick. After every pick
                //          we drop the won blocks and rerun k-means on the
                //          shrunken residual — this lets the next pick
                //          target shapes that genuinely remain unfit
                //          rather than recycling clusters that overlapped
                //          earlier picks. No 1% abort: we always emit
                //          r2_max_picks curves so the final table is
                //          complete (e.g. V where coverage is naturally
                //          low).
                // ============================================================
                let _n_new_curves_const: usize = 64; // historical const, replaced
                #[allow(non_snake_case)]
                let N_NEW_CURVES: usize = n_new_curves;
                const KMEANS_ITERS: usize = 25;

                println!();
                println!(
                    "  ── Round 2: re-running k-means + match per pick (4 picks, no abort) ──"
                );

                let mut r2_picks: Vec<(usize, usize, [i8; 32])> = Vec::new();
                let mut r2_set: Vec<usize> = working_set.clone();
                let r2_n_total = working_set.len();
                let r2_aborted = false; // retained for downstream rendering

                // Cumulative candidate pool. Each R2 iter runs k-means on
                // the current residual, but instead of replacing the
                // candidate pool we APPEND new centroids (and their mirrors,
                // if enabled). This way curves picked in early iters AND
                // their unpicked siblings/mirrors get re-tested against
                // every later residual — a curve that didn't win when
                // first generated may turn out to be the best fit for the
                // new residual after some blocks are gone.
                let mut cum_centroids: Vec<[f32; 32]> = Vec::new();
                let mut cum_used: Vec<bool> = Vec::new();

                // Build a peak-rotated-xi_norm row for one block (used by
                // each iter's k-means). Captures everything that doesn't
                // change across iters (calibration tables, normalised data).
                let xi_norm_rot_for = |w: usize| -> [f32; 32] {
                    let off = w * 32;
                    let scale_idx = block_scale_idx[w] as usize;
                    let scale_q = cal_scale_table[scale_idx];
                    let actual_c = block_centroid[w];
                    let row = &cal_centroid_table[scale_idx];
                    let mut bc = 0usize;
                    let mut be = f32::INFINITY;
                    for (kk, &cv) in row.iter().enumerate() {
                        let e = (actual_c - cv).abs();
                        if e < be {
                            be = e;
                            bc = kk;
                        }
                    }
                    let centroid_q = row[bc];
                    let inv_s = if scale_q > 0.0 { 1.0 / scale_q } else { 0.0 };
                    let raw: [f32; 32] =
                        std::array::from_fn(|j| (normalised[off + j] - centroid_q) * inv_s);
                    let mut peak = 0usize;
                    let mut pmax = 0.0f32;
                    for (j, &v) in raw.iter().enumerate() {
                        let a = v.abs();
                        if a > pmax {
                            pmax = a;
                            peak = j;
                        }
                    }
                    std::array::from_fn(|j| raw[(j + peak) % 32])
                };

                for r2_iter in 0..r2_max_picks {
                    let r2_start = std::time::Instant::now();
                    if r2_set.is_empty() {
                        println!(
                            "    R2 iter {}: residual is empty — stopping early.",
                            r2_iter
                        );
                        break;
                    }
                    println!();
                    println!(
                        "    ── R2 iter {} (residual = {} blocks) ──",
                        r2_iter,
                        r2_set.len()
                    );

                    // Compute peak-rotated xi_norm + scale_idx for CURRENT residual.
                    let xi_full: Vec<([f32; 32], u8)> = r2_set
                        .par_iter()
                        .map(|&w| (xi_norm_rot_for(w), block_scale_idx[w]))
                        .collect();

                    // Structuredness filter (kmeans_rho_min > 0): drop blocks
                    // with |lag-1 autocorrelation| below the threshold from
                    // the k-means input. The dropped blocks remain in r2_set
                    // so the kernel's win count still measures coverage of
                    // the full population.
                    let xi_filtered: Vec<([f32; 32], u8)> = if kmeans_rho_min > 0.0 {
                        xi_full
                            .par_iter()
                            .filter(|(x, _)| {
                                let mean: f32 = x.iter().sum::<f32>() / 32.0;
                                let mut num = 0.0f32;
                                let mut den = 0.0f32;
                                for i in 0..32 {
                                    let d = x[i] - mean;
                                    den += d * d;
                                    if i + 1 < 32 {
                                        num += d * (x[i + 1] - mean);
                                    }
                                }
                                let rho = if den > 0.0 { num / den } else { 0.0 };
                                rho.abs() >= kmeans_rho_min
                            })
                            .copied()
                            .collect()
                    } else {
                        xi_full.clone()
                    };
                    let n_full = xi_full.len();
                    let n_kept = xi_filtered.len();
                    if kmeans_rho_min > 0.0 {
                        println!(
                            "    structuredness filter |ρ₁| ≥ {:.2}:  kept {}/{} ({:.1}%) for k-means",
                            kmeans_rho_min,
                            n_kept,
                            n_full,
                            100.0 * (n_kept as f64) / (n_full as f64)
                        );
                    }
                    if n_kept == 0 {
                        println!(
                            "    R2 iter {}: no structured blocks left after filter — stopping.",
                            r2_iter
                        );
                        break;
                    }

                    // Cyclic moving-average pre-smooth (kmeans_smooth_taps > 1).
                    let xi_filtered: Vec<([f32; 32], u8)> = if kmeans_smooth_taps > 1 {
                        let t = kmeans_smooth_taps;
                        let half = (t / 2) as i32;
                        let inv_t = 1.0 / (t as f32);
                        xi_filtered
                            .par_iter()
                            .map(|(x, s)| {
                                let mut out = [0.0f32; 32];
                                for k in 0..32 {
                                    let mut sm = 0.0f32;
                                    for d in -half..=half {
                                        let idx = ((k as i32 + d).rem_euclid(32)) as usize;
                                        sm += x[idx];
                                    }
                                    out[k] = sm * inv_t;
                                }
                                (out, *s)
                            })
                            .collect()
                    } else {
                        xi_filtered
                    };
                    if kmeans_smooth_taps > 1 {
                        println!(
                            "    pre-smooth: cyclic {}-tap moving average on k-means input",
                            kmeans_smooth_taps
                        );
                    }

                    // Build strata. kmeans_n_strata > 1: split blocks by
                    // scale_idx quantile so we run k-means separately within
                    // each magnitude regime. Per the V token-structure
                    // observation: outlier-rich blocks (high amax) cluster
                    // differently from sink-heavy / normal-only blocks (low
                    // amax). Stratifying lets each regime's k-means converge
                    // on its native shapes rather than averaging across
                    // regimes. Centroids from all strata are concatenated
                    // into the candidate pool.

                    // Optionally replace the per-block "scale_idx" feature
                    // with a "second-peak position" feature for stratification.
                    // After peak rotation, the dominant peak is at lane 0 by
                    // construction; the *next* most informative thing is where
                    // the SECOND peak (or deepest sink) lands. Stratifying by
                    // this gives each second-peak-position its own k-means
                    // cluster pool — directly targets the user's V token
                    // structure observation (outliers/sinks distributed
                    // across the rest of the chunk).
                    let xi_filtered: Vec<([f32; 32], u8)> = if strata_by_second_peak {
                        xi_filtered
                            .par_iter()
                            .map(|(x, _)| {
                                let mut best_l = 1u8;
                                let mut best_v = -1.0f32;
                                for lane in 1..32 {
                                    let a = x[lane].abs();
                                    if a > best_v {
                                        best_v = a;
                                        best_l = lane as u8;
                                    }
                                }
                                (*x, best_l)
                            })
                            .collect()
                    } else {
                        xi_filtered
                    };

                    // `sign_canonicalize` means "emit mirror curves at output":
                    // we let k-means cluster the data as-is (no input flipping),
                    // but for every centroid we ALSO append its negation to the
                    // candidate pool. The kernel scoring then has both polarities
                    // available and picks whichever fits each block better. This
                    // doubles the effective candidate count without doubling
                    // k-means work.
                    if sign_canonicalize {
                        println!("    sign mirror: each k-means centroid will be emitted alongside its negation (doubling the candidate pool)");
                    }

                    let n_strata = kmeans_n_strata.max(1);
                    let strata_assignments: Vec<usize> = if n_strata == 1 {
                        vec![0; n_kept]
                    } else {
                        // Sort by scale_idx, split into N equal-count buckets.
                        let mut idx: Vec<usize> = (0..n_kept).collect();
                        idx.sort_by_key(|&i| xi_filtered[i].1);
                        let mut a = vec![0usize; n_kept];
                        for (rank, &orig_i) in idx.iter().enumerate() {
                            let bucket = (rank * n_strata) / n_kept;
                            a[orig_i] = bucket.min(n_strata - 1);
                        }
                        a
                    };
                    if n_strata > 1 {
                        let mut counts = vec![0usize; n_strata];
                        for &b in &strata_assignments {
                            counts[b] += 1;
                        }
                        print!("    strata by scale_idx ({} buckets, k=", n_strata);
                        let per_stratum = N_NEW_CURVES / n_strata;
                        print!("{} per stratum):", per_stratum);
                        for (i, c) in counts.iter().enumerate() {
                            // Show range of scale_idx in each bucket.
                            let mut lo = u8::MAX;
                            let mut hi = 0u8;
                            for (j, b) in strata_assignments.iter().enumerate() {
                                if *b == i {
                                    lo = lo.min(xi_filtered[j].1);
                                    hi = hi.max(xi_filtered[j].1);
                                }
                            }
                            print!("  [{}: {}..{}, n={}]", i, lo, hi, c);
                        }
                        println!();
                    }

                    // K-means init: random distinct samples (seed advances
                    // per iter so we don't re-pick identical centroids).
                    let mut k_seed: u64 = 0xDEAD_BEEF_CAFE_BABE
                        ^ (r2_iter as u64).wrapping_mul(0x100_0000_01B3)
                        ^ side_label
                            .bytes()
                            .fold(0u64, |a, b| a.wrapping_mul(131).wrapping_add(b as u64));
                    let mut k_next = || {
                        k_seed = k_seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        k_seed
                    };
                    // Run k-means independently per stratum. Stratum s gets
                    // `k_per` clusters from blocks where strata_assignments[i]==s.
                    let k_per_stratum = N_NEW_CURVES / n_strata;
                    let mut centroids: Vec<[f32; 32]> = Vec::with_capacity(N_NEW_CURVES);

                    for stratum in 0..n_strata {
                        // Gather this stratum's xi vectors.
                        let stratum_xi: Vec<[f32; 32]> = xi_filtered
                            .iter()
                            .zip(strata_assignments.iter())
                            .filter_map(|((x, _), &b)| if b == stratum { Some(*x) } else { None })
                            .collect();
                        if stratum_xi.is_empty() {
                            // Pad with zeros if a stratum is empty.
                            for _ in 0..k_per_stratum {
                                centroids.push([0.0f32; 32]);
                            }
                            continue;
                        }
                        // Random distinct-sample init (k-means++ tested but
                        // marginally worse on V; the random init benefits
                        // from clusters being naturally well-separated by
                        // the scale_idx stratification).
                        let n_init = k_per_stratum.min(stratum_xi.len());
                        let mut s_centroids: Vec<[f32; 32]> = Vec::with_capacity(k_per_stratum);
                        let mut taken: Vec<usize> = Vec::with_capacity(k_per_stratum);
                        while s_centroids.len() < n_init {
                            let p = (k_next() as usize) % stratum_xi.len();
                            if !taken.contains(&p) {
                                taken.push(p);
                                s_centroids.push(stratum_xi[p]);
                            }
                        }
                        while s_centroids.len() < k_per_stratum {
                            s_centroids.push([0.0f32; 32]);
                        }

                        let lane_start = if kmeans_skip_lane0 { 1 } else { 0 };
                        for kmi in 0..KMEANS_ITERS {
                            let assignments: Vec<u8> = stratum_xi
                                .par_iter()
                                .map(|x| {
                                    let mut best: u8 = 0;
                                    let mut best_d = f32::INFINITY;
                                    for (ci, c) in s_centroids.iter().enumerate() {
                                        let mut d = 0.0f32;
                                        // Distance over lanes [lane_start..32].
                                        // Skipping lane 0 focuses clustering
                                        // on the tail shape (lane 0 is always
                                        // the dominant peak after rotation).
                                        for k in lane_start..32 {
                                            let dd = x[k] - c[k];
                                            d += dd * dd;
                                        }
                                        if d < best_d {
                                            best_d = d;
                                            best = ci as u8;
                                        }
                                    }
                                    best
                                })
                                .collect();
                            let mut sums = vec![[0.0f32; 32]; k_per_stratum];
                            let mut counts = vec![0usize; k_per_stratum];
                            for (i, &a) in assignments.iter().enumerate() {
                                let ai = a as usize;
                                for k in 0..32 {
                                    sums[ai][k] += stratum_xi[i][k];
                                }
                                counts[ai] += 1;
                            }
                            let mut moved = 0.0f32;
                            let mut empty_clusters = 0usize;
                            for ci in 0..k_per_stratum {
                                if counts[ci] > 0 {
                                    let inv = 1.0 / (counts[ci] as f32);
                                    for k in 0..32 {
                                        let nc = sums[ci][k] * inv;
                                        let dd = nc - s_centroids[ci][k];
                                        moved += dd * dd;
                                        s_centroids[ci][k] = nc;
                                    }
                                } else {
                                    empty_clusters += 1;
                                }
                            }
                            if kmi == 0 || kmi == KMEANS_ITERS - 1 {
                                println!(
                                    "      kmeans stratum {} iter {:>2}: total-shift={:.4}  empty-clusters={}",
                                    stratum,
                                    kmi,
                                    moved.sqrt(),
                                    empty_clusters
                                );
                            }
                        }
                        centroids.extend(s_centroids);
                    }
                    // Sign-mirror emit step: append the negation of every
                    // new centroid so the kernel can match either polarity.
                    if sign_canonicalize {
                        let positives = centroids.clone();
                        for c in positives.iter() {
                            let mut neg = [0.0f32; 32];
                            for k in 0..32 {
                                neg[k] = -c[k];
                            }
                            centroids.push(neg);
                        }
                    }
                    let n_total_candidates = if sign_canonicalize {
                        N_NEW_CURVES * 2
                    } else {
                        N_NEW_CURVES
                    };
                    while centroids.len() < n_total_candidates {
                        centroids.push([0.0f32; 32]);
                    }
                    centroids.truncate(n_total_candidates);

                    // Append THIS iter's new centroids to the cumulative
                    // pool. Each cumulative slot stays alive across all
                    // subsequent iters (until picked) — gives previous-iter
                    // curves and their mirrors a chance to win on later
                    // residuals.
                    let n_new_this_iter = centroids.len();
                    cum_centroids.extend(centroids.iter().copied());
                    cum_used.extend(std::iter::repeat(false).take(n_new_this_iter));
                    println!(
                        "    candidate pool: {} new added → {} total ({} unused)",
                        n_new_this_iter,
                        cum_centroids.len(),
                        cum_used.iter().filter(|u| !**u).count()
                    );

                    // Quantize the FULL cumulative pool to i8.
                    let derived_curves_i8: Vec<[i8; 32]> = cum_centroids
                        .iter()
                        .map(|c| {
                            let amax = c.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                            let inv = if amax > 0.0 { 1.0 / amax } else { 0.0 };
                            let mut out = [0i8; 32];
                            for k in 0..32 {
                                let q = ((c[k] * inv).clamp(-1.0, 1.0) * 127.0).round() as i32;
                                out[k] = q.clamp(-127, 127) as i8;
                            }
                            out
                        })
                        .collect();

                    // Side-specific phase resolution for derived curves
                    // (matches the parametric family phase count).
                    let stride_local = 32 / n_phases;
                    let derived_phase_tables: Vec<Vec<[i8; 32]>> = derived_curves_i8
                        .iter()
                        .map(|phase0| {
                            let mut out: Vec<[i8; 32]> = Vec::with_capacity(n_phases);
                            for p in 0..n_phases {
                                let mut v = [0i8; 32];
                                for lane in 0..32 {
                                    v[lane] = phase0[(lane + stride_local * p) % 32];
                                }
                                out.push(v);
                            }
                            out
                        })
                        .collect();
                    let derived_runtime_tables: Vec<(Vec<i8>, Vec<u8>, Vec<u16>)> =
                        derived_phase_tables
                            .par_iter()
                            .map(|phases| {
                                let curves = build_replicated_curve_table(phases.as_slice());
                                let (peak_idx, peak_off) = build_peak_tables(&curves);
                                (curves, peak_idx, peak_off)
                            })
                            .collect();

                    // Score each unused candidate against the current residual.
                    let mut best_curve: Option<usize> = None;
                    let mut best_wins: usize = 0;
                    let mut n_scored: usize = 0;
                    for (ci, (curves, peak_idx, peak_off)) in
                        derived_runtime_tables.iter().enumerate()
                    {
                        if cum_used[ci] {
                            continue;
                        }
                        n_scored += 1;
                        let curves_gpu = dev.memcpy_stod(curves).expect("curves upload");
                        let peak_idx_gpu = dev.memcpy_stod(peak_idx).expect("peak_idx upload");
                        let peak_off_gpu = dev.memcpy_stod(peak_off).expect("peak_off upload");
                        let (src_p, _g1) = normalised_gpu.device_ptr(&stream);
                        let (recon_p, _g2) = recon_gpu.device_ptr(&stream);
                        let (curves_p, _g3) = curves_gpu.device_ptr(&stream);
                        let (scale_p, _g4) = scale_bits_gpu.device_ptr(&stream);
                        let (cent_p, _g5) = centroid_bits_gpu.device_ptr(&stream);
                        let (pi_p, _g6) = peak_idx_gpu.device_ptr(&stream);
                        let (po_p, _g7) = peak_off_gpu.device_ptr(&stream);
                        unsafe {
                            run_roundtrip_q0_v_runtime(
                                src_p as *const c_void,
                                recon_p as *mut c_void,
                                n_normalised as i32,
                                1.0,
                                curves_p as *const c_void,
                                scale_p as *const c_void,
                                cent_p as *const c_void,
                                pi_p as *const c_void,
                                po_p as *const c_void,
                            );
                        }
                        stream.synchronize().expect("sync");
                        recon_host = dev.memcpy_dtov(&recon_gpu).expect("recon download");
                        let wins: usize = r2_set
                            .par_iter()
                            .filter(|&&w| {
                                let off = w * 32;
                                let orig: [f32; 32] = std::array::from_fn(|j| normalised[off + j]);
                                let recon: [f32; 32] = std::array::from_fn(|j| recon_host[off + j]);
                                pass_metric(&side, &orig, &recon) <= pass_threshold
                            })
                            .count();
                        if wins > best_wins {
                            best_wins = wins;
                            best_curve = Some(ci);
                        }
                    }
                    let _ = n_scored;

                    // Always commit a pick (no <1% abort). If no curve
                    // managed any wins (degenerate), still record the best
                    // centroid so the final table has 4 curves — it'll
                    // simply rank low in the combined top-4.
                    let chosen = best_curve.unwrap_or(0);
                    let chosen_curves = &derived_runtime_tables[chosen].0;
                    {
                        let curves_gpu = dev.memcpy_stod(chosen_curves).expect("curves upload");
                        let peak_idx_gpu = dev
                            .memcpy_stod(&derived_runtime_tables[chosen].1)
                            .expect("peak_idx upload");
                        let peak_off_gpu = dev
                            .memcpy_stod(&derived_runtime_tables[chosen].2)
                            .expect("peak_off upload");
                        let (src_p, _g1) = normalised_gpu.device_ptr(&stream);
                        let (recon_p, _g2) = recon_gpu.device_ptr(&stream);
                        let (curves_p, _g3) = curves_gpu.device_ptr(&stream);
                        let (scale_p, _g4) = scale_bits_gpu.device_ptr(&stream);
                        let (cent_p, _g5) = centroid_bits_gpu.device_ptr(&stream);
                        let (pi_p, _g6) = peak_idx_gpu.device_ptr(&stream);
                        let (po_p, _g7) = peak_off_gpu.device_ptr(&stream);
                        unsafe {
                            run_roundtrip_q0_v_runtime(
                                src_p as *const c_void,
                                recon_p as *mut c_void,
                                n_normalised as i32,
                                1.0,
                                curves_p as *const c_void,
                                scale_p as *const c_void,
                                cent_p as *const c_void,
                                pi_p as *const c_void,
                                po_p as *const c_void,
                            );
                        }
                        stream.synchronize().expect("sync");
                        recon_host = dev.memcpy_dtov(&recon_gpu).expect("recon download");
                    }
                    let new_set: Vec<usize> = r2_set
                        .par_iter()
                        .copied()
                        .filter(|&w| {
                            let off = w * 32;
                            let orig: [f32; 32] = std::array::from_fn(|j| normalised[off + j]);
                            let recon: [f32; 32] = std::array::from_fn(|j| recon_host[off + j]);
                            pass_metric(&side, &orig, &recon) > pass_threshold
                        })
                        .collect();
                    let removed = r2_set.len() - new_set.len();
                    r2_set = new_set;
                    cum_used[chosen] = true;
                    r2_picks.push((r2_iter, removed, derived_curves_i8[chosen]));

                    let cov_pct_d = 100.0 * best_wins as f64 / n_normalised as f64;
                    println!(
                        "      → chose curve[{:>3}]  wins={:>7} ({:.3}% of orig)  removed={:>6} ({:.3}% of r2-set)  remaining={:>9}  [{:.2}s]",
                        chosen,
                        best_wins,
                        cov_pct_d,
                        removed,
                        100.0 * removed as f64 / r2_n_total as f64,
                        r2_set.len(),
                        r2_start.elapsed().as_secs_f64()
                    );
                }

                // ============================================================
                // FINAL: combine R1 (parametric) + R2 (data-derived) picks,
                //        sort by wins, keep the top 4. R1 picks tend to win
                //        the high-coverage early slots; R2 picks fill in
                //        uniquely-shaped residuals R1's parametric pool
                //        can't represent.
                // ============================================================
                #[derive(Clone)]
                struct FinalPick {
                    source: String,
                    wins: usize,
                    curve_i8: [i8; 32],
                }
                let mut combined: Vec<FinalPick> = Vec::new();
                for (it, freq, shape, wins) in &selected_families {
                    let f = generate_curve(*freq, 0.0, *shape);
                    combined.push(FinalPick {
                        source: format!("R1[iter {}] f={:.2} {}", it, freq, shape.label()),
                        wins: *wins,
                        curve_i8: quantise_i8(&f),
                    });
                }
                for (it, wins, curve) in &r2_picks {
                    combined.push(FinalPick {
                        source: format!("R2[iter {}]", it),
                        wins: *wins,
                        curve_i8: *curve,
                    });
                }
                combined.sort_by(|a, b| b.wins.cmp(&a.wins));
                combined.truncate(r2_max_picks);

                println!();
                println!("  ╔═══════════════════════════════════════════════════════════════╗");
                println!(
                    "  ║  FINAL TOP {} ({}-side) — best of R1 (parametric) + R2 (data)  ║",
                    r2_max_picks,
                    side_label.to_uppercase()
                );
                println!("  ╚═══════════════════════════════════════════════════════════════╝");
                let total_final: usize = combined.iter().map(|p| p.wins).sum();
                println!(
                    "  R1 picks (parametric, kept): {}  |  R2 picks (data-derived, kept): {}  |  combined-pool: {}  →  top {}",
                    selected_families.len(),
                    r2_picks.len(),
                    selected_families.len() + r2_picks.len(),
                    r2_max_picks
                );
                println!(
                    "  Final coverage (top {} wins, summed): {} blocks ({:.2}% of {} active)",
                    r2_max_picks,
                    total_final,
                    100.0 * total_final as f64 / n_normalised as f64,
                    n_normalised
                );
                println!();

                for (rank, p) in combined.iter().enumerate() {
                    println!(
                        "    [{}]  {:<32}  wins={:>7}  ({:>5.2}% of orig)",
                        rank,
                        p.source,
                        p.wins,
                        100.0 * (p.wins as f64) / n_normalised as f64
                    );
                    let curve_f32: [f32; 32] =
                        std::array::from_fn(|i| p.curve_i8[i] as f32 / 127.0);
                    let plot = plot_block(&curve_f32);
                    for line in plot.iter() {
                        println!("      {}", line);
                    }
                    print!("      i8: [");
                    for (l, v) in p.curve_i8.iter().enumerate() {
                        print!("{:>4}", v);
                        if l < 31 {
                            print!(",");
                        }
                    }
                    println!(" ]");
                    println!();
                }
                let _ = r2_aborted; // diagnostic only; the abort doesn't suppress combined output

                // Stop iterating this side (no more useful curves) but
                // return cleanly so the caller can move on to the next side.
                break;
            }

            // ── Non-abort path: pick is above 1% floor; commit it.
            family_used[chosen_fi] = true;
            // Re-run roundtrip with the chosen family on the working set
            // (the previous recon_host belongs to whichever family was
            // scored last in the loop above — re-run for the chosen one).
            let chosen_curves = &family_runtime_tables[chosen_fi].0;
            {
                let curves_gpu = dev.memcpy_stod(chosen_curves).expect("curves upload");
                let peak_idx_gpu = dev
                    .memcpy_stod(&family_runtime_tables[chosen_fi].1)
                    .expect("peak_idx upload");
                let peak_off_gpu = dev
                    .memcpy_stod(&family_runtime_tables[chosen_fi].2)
                    .expect("peak_off upload");
                let (src_p, _g1) = normalised_gpu.device_ptr(&stream);
                let (recon_p, _g2) = recon_gpu.device_ptr(&stream);
                let (curves_p, _g3) = curves_gpu.device_ptr(&stream);
                let (scale_p, _g4) = scale_bits_gpu.device_ptr(&stream);
                let (cent_p, _g5) = centroid_bits_gpu.device_ptr(&stream);
                let (pi_p, _g6) = peak_idx_gpu.device_ptr(&stream);
                let (po_p, _g7) = peak_off_gpu.device_ptr(&stream);
                unsafe {
                    run_roundtrip_q0_v_runtime(
                        src_p as *const c_void,
                        recon_p as *mut c_void,
                        n_normalised as i32,
                        1.0,
                        curves_p as *const c_void,
                        scale_p as *const c_void,
                        cent_p as *const c_void,
                        pi_p as *const c_void,
                        po_p as *const c_void,
                    );
                }
                stream.synchronize().expect("sync");
                recon_host = dev.memcpy_dtov(&recon_gpu).expect("recon download");
            }
            let new_working_set: Vec<usize> = working_set
                .par_iter()
                .copied()
                .filter(|&w| {
                    let off = w * 32;
                    let orig: [f32; 32] = std::array::from_fn(|j| normalised[off + j]);
                    let recon: [f32; 32] = std::array::from_fn(|j| recon_host[off + j]);
                    pass_metric(&side, &orig, &recon) > pass_threshold
                })
                .collect();
            let removed = working_set.len() - new_working_set.len();
            working_set = new_working_set;
            selected_families.push((iter, chosen_freq, chosen_shape, removed));
            println!(
                "  iter {:>2}: family[{:>3}] freq={:>5.2}  shape={:<8}  wins={:>8}  ({:.2}%)  remaining={:>9}  [{:.2}s]",
                iter,
                chosen_fi,
                chosen_freq,
                chosen_shape.label(),
                removed,
                cov_pct,
                working_set.len(),
                iter_start.elapsed().as_secs_f64()
            );
        }

        // Final summary.
        println!(
            "\n  Selected 8 families ({}-side):",
            side_label.to_uppercase()
        );
        println!("    iter  freq    shape     wins     %coverage");
        println!("    ----  ------  --------  -------  ---------");
        let total_won: usize = selected_families.iter().map(|x| x.3).sum();
        for (iter, freq, shape, wins) in &selected_families {
            println!(
                "    {:>4}  {:>6.3}  {:<8}  {:>7}  {:>6.2}%",
                iter,
                freq,
                shape.label(),
                wins,
                100.0 * (*wins as f64) / n_normalised as f64
            );
        }
        println!(
            "    ──────────────────────────────────  TOTAL  {:>7}  {:>6.2}%",
            total_won,
            100.0 * total_won as f64 / n_normalised as f64
        );

        // ============================================================
        // Render the 16 selected families as a 4×4 ASCII grid + dump
        // their i8 constant tables (phase 0 representative for each).
        // ============================================================
        // Rendering convention: each cell is 32 lanes wide × 9 rows tall.
        // y-axis runs from +127 (top) to -127 (bottom), x-axis runs lane
        // 0..31 left→right. The plotted symbol is the curve at phase 0
        // (the canonical representative). Phases 1..15 are cyclic shifts
        // of the same shape, so phase=0 is enough to see the family.
        //
        // The constant tables below the grids reproduce the phase-0
        // representative as i8[32] — paste-able straight into a
        // `q0_v_curve_table_<side>[256][32]` slot at index `bucket*16`.

        // Build phase-0 i8 curves for the 8 selected families (in selection
        // order — iter 0 is best-coverage, iter 7 is worst).
        let mut selected_curves_i8: Vec<[i8; 32]> = Vec::with_capacity(8);
        let mut selected_curves_f32: Vec<[f32; 32]> = Vec::with_capacity(8);
        for (_, freq, shape, _) in &selected_families {
            let f = generate_curve(*freq, 0.0, *shape);
            selected_curves_f32.push(f);
            selected_curves_i8.push(quantise_i8(&f));
        }

        // 4×4 ASCII grid. Each curve plotted as 9 rows × 33 chars (32
        // lane symbols + 1 spacer) with a header row showing the
        // (freq, shape, coverage) and a small y-axis tick on the left.
        let plot_height: usize = 9;
        let lane_w: usize = 1;
        let _ = lane_w;
        let plot_curve = |curve: &[i8; 32]| -> Vec<String> {
            // Map i8 [-127, +127] to row 0..plot_height-1 (top → bottom).
            let mut rows = vec![String::new(); plot_height];
            for lane in 0..32 {
                let v = curve[lane] as f32 / 127.0; // [-1, +1]
                                                    // Row 0 = +1.0, row plot_height-1 = -1.0.
                let r_f = (1.0 - v) * 0.5 * (plot_height - 1) as f32;
                let r = (r_f.round() as isize).clamp(0, plot_height as isize - 1) as usize;
                for (ri, row) in rows.iter_mut().enumerate() {
                    if ri == r {
                        row.push('●');
                    } else if ri == plot_height / 2 {
                        row.push('─');
                    } else {
                        row.push(' ');
                    }
                }
            }
            rows
        };

        println!(
            "\n  ── {} selected families, 4×2 grid (top→bottom = +127→−127, left→right = lane 0→31) ──",
            selected_families.len()
        );
        let n_picked = selected_families.len();
        let n_rows = n_picked.div_ceil(4);
        for grid_row in 0..n_rows {
            // For each grid row, gather up to 4 plotted curves and print
            // column-major. If fewer than 4 in this row, render blank cells.
            let plots: Vec<Option<Vec<String>>> = (0..4)
                .map(|c| {
                    let idx = grid_row * 4 + c;
                    if idx < n_picked {
                        Some(plot_curve(&selected_curves_i8[idx]))
                    } else {
                        None
                    }
                })
                .collect();
            // Header row above each grid row.
            let mut header = String::from("    ");
            for c in 0..4 {
                let idx = grid_row * 4 + c;
                if idx < n_picked {
                    let (_, freq, shape, wins) = selected_families[idx];
                    header.push_str(&format!(
                        "[{:>2}] f={:>5.2} {:<8} {:>5.2}%   ",
                        idx,
                        freq,
                        shape.label(),
                        100.0 * wins as f64 / n_normalised as f64,
                    ));
                } else {
                    header.push_str(&format!("{:<35}", ""));
                }
            }
            println!("{}", header);
            for row in 0..plot_height {
                let mut line = String::from("    ");
                for c in 0..4 {
                    match &plots[c] {
                        Some(p) => line.push_str(&p[row]),
                        None => line.push_str(&" ".repeat(32)),
                    }
                    line.push_str("   ");
                }
                println!("{}", line);
            }
            println!();
        }

        // Constant tables — paste-able C array fragments. Each entry is a
        // 32-i8 phase-0 canonical curve. The 32 deployment phases per
        // family are pure cyclic lane shifts of phase 0, not stored.
        println!(
            "  ── i8 constant tables (phase=0 canonical per family, {}-side) ──",
            side_label.to_uppercase()
        );
        println!(
            "  /* paste-able snippet: {} phase-0 canonicals, indexed by selection rank */",
            n_picked
        );
        println!(
            "  static const int8_t q0_v_selected_phase0_{}[{}][32] = {{",
            side_label, n_picked
        );
        for (idx, curve_i8) in selected_curves_i8.iter().enumerate() {
            let (_, freq, shape, wins) = selected_families[idx];
            print!(
                "    /* family {:>2}  f={:>5.2} {:<8} cov={:>5.2}% */ {{ ",
                idx,
                freq,
                shape.label(),
                100.0 * wins as f64 / n_normalised as f64
            );
            for (lane, v) in curve_i8.iter().enumerate() {
                print!("{:>4}", v);
                if lane < 31 {
                    print!(",");
                }
            }
            println!(" }},");
        }
        println!("  }};");
    };

    // Use the same blocks the calibration test loads.
    let loaded = match load_kv_blocks_for_q0_v_tests() {
        Some(b) => b,
        None => {
            println!("dump files absent, skipping.");
            return;
        }
    };

    // Operating-point thresholds — C9 raw production geomeans (kernel-side
    // pass-metric units; matches the kernel-driven calibration filter).
    use crate::kv_cache::chunked::sampled_selection::{
        PRODUCTION_K_QREL_HIGH_THRESHOLDS, PRODUCTION_K_QREL_LOW_THRESHOLDS,
        PRODUCTION_V_QREL_HIGH_THRESHOLDS, PRODUCTION_V_QREL_LOW_THRESHOLDS,
    };
    const FILTER_LEVEL: usize = 9;
    let k_thr = (PRODUCTION_K_QREL_HIGH_THRESHOLDS[FILTER_LEVEL]
        * PRODUCTION_K_QREL_LOW_THRESHOLDS[FILTER_LEVEL])
        .sqrt();
    let v_thr = (PRODUCTION_V_QREL_HIGH_THRESHOLDS[FILTER_LEVEL]
        * PRODUCTION_V_QREL_LOW_THRESHOLDS[FILTER_LEVEL])
        .sqrt();

    run_side(
        "k",
        SideMetric::K,
        &loaded.k_blocks,
        &loaded.k_blocks_head_amax,
        &q0v_tables::SCALE_TABLE_BITS_K,
        &q0v_tables::CENTROID_TABLE_BITS_K,
        k_thr,
        16, // K: 16 phases (stride-2 lane shifts)
        4,  // K: 4 final picks
        &family_runtime_tables_k,
        16,    // K: 16 dump samples (well-clustered residual)
        0.0,   // K: no k-means structuredness filter
        0,     // K: no k-means pre-smoothing
        1,     // K: single stratum (no stratification)
        64,    // K: 64 candidate curves
        false, // K: no sign canonicalization
        false, // K: don't skip lane 0
        false, // K: stratify by scale_idx (default)
    );
    run_side(
        "v",
        SideMetric::V,
        &loaded.v_blocks,
        &loaded.v_blocks_head_amax,
        &q0v_tables::SCALE_TABLE_BITS_V,
        &q0v_tables::CENTROID_TABLE_BITS_V,
        v_thr,
        16,                       // V: 16 phases (stride-2 lane shifts) — same as K
        4,                        // V: 4 final picks
        &family_runtime_tables_k, // both sides share the 16-phase tables now
        64,                       // V: 64 dump samples — wider view of noise-dominated residual
        0.20,                     // V: drop |ρ₁|<0.20 from k-means input (filter random noise)
        0,                        // V: no pre-smoothing (3-tap dampened the discriminative spike)
        8,                        // V: 8 scale_idx strata
        64,                       // V: 64 base k-means candidates → 8 per stratum
        true,  // V: emit mirror curves (centroid + negation) — doubles candidate pool
        false, // V: don't skip lane 0
        false, // V: stratify by scale_idx
    );
}

// ---------------------------------------------------------------------------
// Test: print stats from saved file (fast, no-model, loads bin/csv)
// ---------------------------------------------------------------------------

/// Load the binary stats file and print per-layer Q8/Q4 cosine distance trends.
#[test]
fn test_print_stats_by_layer() {
    let bin_path = data_path(STATS_BIN_PATH);
    if !bin_path.exists() {
        println!("kv_stats_tests: stats file absent, run test_generate_kv_stats first.");
        return;
    }

    use super::dump_reader::{read_f32_le, read_u32_le};
    let bytes = std::fs::read(&bin_path).expect("cannot read stats file");
    let mut pos: usize;

    if &bytes[0..8] != b"KVSTATS\0" {
        panic!("bad magic in stats file");
    }
    pos = 8;
    let version = read_u32_le(&bytes, &mut pos).unwrap();
    assert!(
        version == 1 || version == 2,
        "unsupported stats version {}",
        version
    );
    let num_layers = read_u32_le(&bytes, &mut pos).unwrap() as usize;
    let _n_kv_head = read_u32_le(&bytes, &mut pos).unwrap();
    let chunk_size = read_u32_le(&bytes, &mut pos).unwrap();
    let _head_dim = read_u32_le(&bytes, &mut pos).unwrap();
    let num_tokens = read_u32_le(&bytes, &mut pos).unwrap() as usize;
    let mut tokens = Vec::with_capacity(num_tokens);
    for _ in 0..num_tokens {
        tokens.push(read_u32_le(&bytes, &mut pos).unwrap());
    }

    println!("\n=== KV stats from binary file ===");
    println!(
        "Layers: {}  Tokens: {:?}",
        num_layers,
        &tokens[..tokens.len().min(16)]
    );
    println!(
        "\n{:>6}  {:>5}  {:>7}  {:>10}  {:>10}  {:>10}  {:>10}",
        "Layer", "Block", "TokSt", "K_q8", "K_q4", "V_q8", "V_q4"
    );

    for layer_idx in 0..num_layers {
        let num_chunks = read_u32_le(&bytes, &mut pos).unwrap() as usize;
        for _ in 0..num_chunks {
            let block_idx = read_u32_le(&bytes, &mut pos).unwrap() as usize;
            let token_start = if version >= 2 {
                read_u32_le(&bytes, &mut pos).unwrap() as usize
            } else {
                block_idx * chunk_size as usize
            };
            let k_arr: Vec<f32> = (0..12)
                .map(|_| read_f32_le(&bytes, &mut pos).unwrap())
                .collect();
            let v_arr: Vec<f32> = (0..12)
                .map(|_| read_f32_le(&bytes, &mut pos).unwrap())
                .collect();
            // k_arr[6] = q8_cos_mean, k_arr[8] = q4_cos_mean
            println!(
                "{:>6}  {:>5}  {:>7}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}",
                layer_idx, block_idx, token_start, k_arr[6], k_arr[8], v_arr[6], v_arr[8],
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Sign-fit residual helpers — used by the per-block filter that drops blocks
// whose sign pattern doesn't fit any of the 16 byte-polarity templates well.
// Filtered data feeds the curve-codebook calibration in
// `test_q0_v_calibrate_tables`.
// ---------------------------------------------------------------------------

/// Fast precomputed-block residual. `abs_block[i] = |block[i]|`,
/// `sign_mask` has bit i set iff `block[i] < 0`. The residual against any
/// template `t` is then: sum of |block[i]| for bits set in (sign_mask ^ t).
/// Avoids re-deciding the per-lane sign across 16 template scans.
#[inline]
#[allow(dead_code)]
fn fast_sign_residual(abs_block: &[f32; 32], sign_mask: u32, template: u32) -> f32 {
    let mut mismatch = sign_mask ^ template;
    let mut s = 0.0f32;
    while mismatch != 0 {
        let i = mismatch.trailing_zeros() as usize;
        s += abs_block[i];
        mismatch &= mismatch - 1;
    }
    s
}

/// Precompute (abs_block, sign_mask, total_abs) for a block.
#[inline]
#[allow(dead_code)]
fn precompute_block(block: &[f32; 32]) -> ([f32; 32], u32, f32) {
    let mut abs_block = [0.0f32; 32];
    let mut sign_mask = 0u32;
    let mut total = 0.0f32;
    for i in 0..32 {
        let a = block[i].abs();
        abs_block[i] = a;
        total += a;
        if block[i] < 0.0 {
            sign_mask |= 1 << i;
        }
    }
    (abs_block, sign_mask, total)
}

// ---------------------------------------------------------------------------
// Parametric curve helpers — used by the curve-codebook calibration that
// replaces the (sign, shape, shift) factorisation with a single 8-bit index
// into a parametric (frequency, phase, shape-character) family.
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ShapeChar {
    Sine,
    Triangle,
    Sharp,
}

impl ShapeChar {
    fn label(self) -> &'static str {
        match self {
            ShapeChar::Sine => "sine",
            ShapeChar::Triangle => "triangle",
            ShapeChar::Sharp => "sharp",
        }
    }
}

/// Generate a length-32 normalised curve (output ∈ [-1, +1]) for a given
/// (frequency, phase, shape_character) tuple. Frequency is cycles per
/// 32-lane block; phase is radian offset. Inversion is reachable by
/// phase += π so no separate invert flag is needed.
fn generate_curve(freq: f32, phase: f32, shape: ShapeChar) -> [f32; 32] {
    let mut out = [0.0f32; 32];
    let two_pi = 2.0 * std::f32::consts::PI;
    for lane in 0..32 {
        let theta = two_pi * freq * (lane as f32) / 32.0 + phase;
        out[lane] = match shape {
            ShapeChar::Sine => theta.cos(),
            ShapeChar::Triangle => {
                let t = theta / two_pi;
                let f = t - t.floor();
                4.0 * (f - 0.5).abs() - 1.0
            }
            ShapeChar::Sharp => {
                let c = theta.cos();
                c.signum() * c.abs().powi(3)
            }
        };
    }
    out
}

/// Normalise a block to zero mean + unit max-deviation so it can be compared
/// against curves living in [-1, +1]. Returns (normalised, mean, max_dev).
/// For nearly-DC blocks (max_dev ≈ 0), returns the zero curve and lets the
/// DC entries in the codebook absorb them naturally.
#[allow(dead_code)]
fn normalise_block(block: &[f32; 32]) -> ([f32; 32], f32, f32) {
    let mean: f32 = block.iter().sum::<f32>() / 32.0;
    let max_dev = block
        .iter()
        .map(|x| (x - mean).abs())
        .fold(0.0f32, f32::max);
    if max_dev < 1e-8 {
        return ([0.0; 32], mean, 0.0);
    }
    let inv = 1.0 / max_dev;
    let mut out = [0.0f32; 32];
    for i in 0..32 {
        out[i] = (block[i] - mean) * inv;
    }
    (out, mean, max_dev)
}

#[inline]
#[allow(dead_code)]
fn l2_sq_curve(a: &[f32; 32], b: &[f32; 32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..32 {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}
