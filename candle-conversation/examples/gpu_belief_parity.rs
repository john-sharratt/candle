//! GPU belief-scan parity + speed, on the REAL substrate.
//!
//! Reloads the segment set (metadata-only, no KV), rebuilds the true per-file
//! galleries from the captured `wide_q_sigs`, and compares the belief scan run
//! two ways over the SAME data:
//!   • CPU  — `score_slots_weighted` PER FILE (today's production per-file z), and
//!   • GPU  — one segmented `BatchedGpuGallery::scan` (per-file z in one launch).
//! They must produce the same exchange SELECTION and scores that agree to within
//! fast-math ULP (~1e-3 relative — the final z*margin is fast-math f32 on GPU vs
//! IEEE f32 on CPU; the integer agreement reductions are exact) — that is the
//! wiring's correctness gate — and the GPU should be far faster. An example (not a
//! test) so it never adds to `cargo test` time.
//!
//! Run:
//!   HARNESS_SUBSTRATE_DIR=C:/Users/johna/prog/candle \
//!     cargo run -p candle-conversation --release --example gpu_belief_parity

use std::time::Instant;

use candle::Device;
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::provenance::{
    decode_wide_sigs, score_slots_weighted, BatchedGpuGallery, SegmentInput, WideQSig,
};
use candle_conversation::substrate::Substrate;

fn topk(scores: &[f32], k: usize) -> std::collections::HashSet<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx.truncate(k);
    idx.into_iter().collect()
}

fn main() {
    let dir = std::env::var("HARNESS_SUBSTRATE_DIR").unwrap_or_else(|_| ".".to_string());
    let dir = std::path::PathBuf::from(dir);

    let t = Instant::now();
    let mut sub = Substrate::new();
    SubstratePersistence::open_in_with_substrate(&dir, &mut sub).expect("reload substrate");
    println!("reloaded substrate in {:.1}s", t.elapsed().as_secs_f64());

    // True per-file galleries: each turn's sig window grouped by its timeline.
    let mut by_tl: std::collections::HashMap<u64, Vec<Vec<WideQSig>>> =
        std::collections::HashMap::new();
    for (_sid, e) in sub.all_streams() {
        let Some(StreamDecl::Turn(d)) = e.decl.as_ref() else {
            continue;
        };
        if let Some(sig) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) {
            if !sig.is_empty() {
                by_tl.entry(d.timeline_id).or_default().push(sig);
            }
        }
    }
    let mut files: Vec<(u64, Vec<Vec<WideQSig>>)> = by_tl.into_iter().collect();
    files.sort_by_key(|(tl, _)| *tl);
    if files.len() < 2 {
        println!("too few files ({}) — nothing to compare", files.len());
        return;
    }

    // Query = the largest single turn window (capped at the real 256 probe
    // budget), removed from its file so it isn't self-matched.
    let (mut qf, mut qt, mut qlen) = (0usize, 0usize, 0usize);
    for (fi, (_, turns)) in files.iter().enumerate() {
        for (ti, w) in turns.iter().enumerate() {
            if w.len() > qlen {
                qlen = w.len();
                qf = fi;
                qt = ti;
            }
        }
    }
    let mut query = files[qf].1.remove(qt);
    query.truncate(256);
    files.retain(|(_, turns)| !turns.is_empty());

    let n_exchanges: usize = files.iter().map(|(_, t)| t.len()).sum();
    let n_tokens: usize = files
        .iter()
        .flat_map(|(_, t)| t.iter())
        .map(|w| w.len())
        .sum();
    println!(
        "{} files, {} exchanges, {} gallery tokens, query {} tokens",
        files.len(),
        n_exchanges,
        n_tokens,
        query.len()
    );

    // ── CPU: score_slots_weighted PER FILE (today's per-file z) ─────────────
    let t = Instant::now();
    let mut cpu: Vec<f32> = Vec::with_capacity(n_exchanges);
    for (_, turns) in &files {
        let wref: Vec<&[WideQSig]> = turns.iter().map(|w| w.as_slice()).collect();
        let wslot: Vec<usize> = (0..turns.len()).collect();
        cpu.extend(score_slots_weighted(
            &query,
            &wref,
            &wslot,
            turns.len(),
            &[],
        ));
    }
    let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;

    // ── GPU: one segmented scan (per-file z in one launch) ──────────────────
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            println!("no CUDA device ({e}) — CPU-only, skipping GPU comparison");
            return;
        }
    };
    let segments: Vec<SegmentInput> = files
        .iter()
        .map(|(_, turns)| SegmentInput {
            windows: turns.iter().map(|w| w.as_slice()).collect(),
            window_case: (0..turns.len()).collect(),
            n_cases: turns.len(),
        })
        .collect();

    let t = Instant::now();
    let gallery = BatchedGpuGallery::from_segments(&segments).expect("build gallery");
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Warm + time the scan (the per-reprojection cost once the gallery is resident).
    let gpu = gallery
        .scan_weighted(&device, &[query.as_slice()], &[])
        .expect("gpu scan");
    let gpu = &gpu[0];
    let t = Instant::now();
    let reps = 5;
    for _ in 0..reps {
        std::hint::black_box(
            gallery
                .scan_weighted(&device, &[query.as_slice()], &[])
                .unwrap(),
        );
    }
    let gpu_ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    // ── Compare ─────────────────────────────────────────────────────────────
    assert_eq!(gpu.len(), cpu.len(), "case count mismatch");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (g, c) in gpu.iter().zip(&cpu) {
        let d = (g - c).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / (1.0 + g.abs().max(c.abs())));
    }
    let mut worst_k_overlap = 1.0f32;
    for k in [4usize, 8, 16, 32] {
        let k = k.min(n_exchanges);
        let inter = topk(&cpu, k).intersection(&topk(gpu, k)).count();
        let ov = inter as f32 / k as f32;
        worst_k_overlap = worst_k_overlap.min(ov);
        println!("  top-{k:<2}: selection overlap CPU vs GPU = {ov:.3}");
    }
    println!(
        "max abs diff = {max_abs:.4}, max rel diff = {max_rel:.6}\n\
         CPU per-file scan: {cpu_ms:.1} ms   |   GPU scan: {gpu_ms:.2} ms  (build {build_ms:.1} ms)  ⇒ {:.1}x",
        cpu_ms / gpu_ms.max(1e-6)
    );

    // The gate: same exchange SELECTION, scores within fast-math tolerance.
    assert!(
        worst_k_overlap >= 0.999,
        "GPU-wired belief scan must select the SAME exchanges as the CPU per-file scan \
         (worst top-K overlap {worst_k_overlap:.3})"
    );
    assert!(
        max_rel <= 1e-3,
        "GPU vs CPU per-exchange scores exceed tolerance (max rel {max_rel:.6})"
    );
    println!(
        "\nPARITY OK — GPU-wired belief scan matches CPU per-file, {:.1}x faster.",
        cpu_ms / gpu_ms.max(1e-6)
    );
}
