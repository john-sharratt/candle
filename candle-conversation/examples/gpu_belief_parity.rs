//! Paged belief-scan parity + speed, on the REAL substrate.
//!
//! Reloads the segment set (metadata-only, no KV), rebuilds the true per-file
//! galleries from the captured `wide_q_sigs`, and compares the belief scan run
//! two ways over the SAME data:
//!   • CPU  — `score_slots_weighted` PER FILE (today's production per-file z), and
//!   • GPU  — the PAGED scan over the resident [`GalleryArena`] (per-file z in one
//!     launch, records resident in VRAM, only a tiny index uploaded per scan).
//! They must produce the same exchange SELECTION and scores that agree to within
//! fast-math ULP (~1e-3 relative — the final z*margin is fast-math f32 on GPU vs
//! IEEE f32 on CPU; the integer agreement reductions are exact) — that is the
//! correctness gate. Timings separate the COLD first scan (which makes every turn
//! resident) from the STEADY-STATE scan (all resident — the real per-reproject
//! cost, where the ~108 MB upload is gone).
//!
//! Run:
//!   HARNESS_SUBSTRATE_DIR=C:/Users/johna/prog/candle \
//!     cargo run -p candle-conversation --release --example gpu_belief_parity

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use candle::Device;
use candle_conversation::persistence::content_hash::turn_stream_id;
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::provenance::gallery_arena::{PagedSegment, PagedWindow};
use candle_conversation::provenance::{
    decode_wide_sigs, score_slots_weighted, GalleryArena, WideQSig,
};
use candle_conversation::substrate::Substrate;

fn topk(scores: &[f32], k: usize) -> std::collections::HashSet<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx.truncate(k);
    idx.into_iter().collect()
}

/// A stable per-turn residency fingerprint (content sample).
fn fp_of(sigs: &[WideQSig]) -> u64 {
    let mut h = DefaultHasher::new();
    sigs.len().hash(&mut h);
    if let Some(w) = sigs.first().and_then(|s| s.words.first()) {
        w.hash(&mut h);
    }
    if let Some(w) = sigs.last().and_then(|s| s.words.last()) {
        w.hash(&mut h);
    }
    h.finish()
}

fn main() {
    let dir = std::env::var("HARNESS_SUBSTRATE_DIR").unwrap_or_else(|_| ".".to_string());
    let dir = std::path::PathBuf::from(dir);

    let t = Instant::now();
    let mut sub = Substrate::new();
    SubstratePersistence::open_in_with_substrate(&dir, &mut sub).expect("reload substrate");
    println!("reloaded substrate in {:.1}s", t.elapsed().as_secs_f64());

    // True per-file galleries: each turn's (turn_index, sig window) by timeline.
    let mut by_tl: std::collections::HashMap<u64, Vec<(u32, Vec<WideQSig>)>> =
        std::collections::HashMap::new();
    for (_sid, e) in sub.all_streams() {
        let Some(StreamDecl::Turn(d)) = e.decl.as_ref() else {
            continue;
        };
        if let Some(sig) = e.wide_q_sigs.as_ref().and_then(|b| decode_wide_sigs(b)) {
            if !sig.is_empty() {
                by_tl
                    .entry(d.timeline_id)
                    .or_default()
                    .push((d.turn_index, sig));
            }
        }
    }
    let mut files: Vec<(u64, Vec<(u32, Vec<WideQSig>)>)> = by_tl.into_iter().collect();
    files.sort_by_key(|(tl, _)| *tl);
    if files.len() < 2 {
        println!("too few files ({}) — nothing to compare", files.len());
        return;
    }

    // Query = the largest single turn window (capped at the 256 probe budget),
    // removed from its file so it isn't self-matched.
    let (mut qf, mut qt, mut qlen) = (0usize, 0usize, 0usize);
    for (fi, (_, turns)) in files.iter().enumerate() {
        for (ti, (_, w)) in turns.iter().enumerate() {
            if w.len() > qlen {
                qlen = w.len();
                qf = fi;
                qt = ti;
            }
        }
    }
    let mut query = files[qf].1.remove(qt).1;
    query.truncate(256);
    files.retain(|(_, turns)| !turns.is_empty());

    let n_exchanges: usize = files.iter().map(|(_, t)| t.len()).sum();
    let n_tokens: usize = files
        .iter()
        .flat_map(|(_, t)| t.iter())
        .map(|(_, w)| w.len())
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
        let wref: Vec<&[WideQSig]> = turns.iter().map(|(_, w)| w.as_slice()).collect();
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

    // ── GPU: paged scan over the resident gallery arena ─────────────────────
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            println!("no CUDA device ({e}) — CPU-only, skipping GPU comparison");
            return;
        }
    };
    let arena = GalleryArena::new(&device, 24, 3).expect("gallery arena");

    // Build the paged segments (each turn a whole window; case = its file slot).
    let segments: Vec<PagedSegment> = files
        .iter()
        .map(|(tl, turns)| PagedSegment {
            windows: turns
                .iter()
                .enumerate()
                .map(|(slot, (turn_idx, sigs))| PagedWindow {
                    sid: turn_stream_id(*tl, *turn_idx),
                    fingerprint: fp_of(sigs),
                    turn: sigs.as_slice(),
                    start: 0,
                    end: sigs.len(),
                    case: slot,
                })
                .collect(),
            n_cases: turns.len(),
        })
        .collect();

    // COLD: first scan makes every turn resident (transpose + delta H2D) and scans.
    let t = Instant::now();
    let cold = arena
        .scan_weighted(&segments, &[query.as_slice()], &[])
        .expect("paged scan (cold)");
    let cold_ms = t.elapsed().as_secs_f64() * 1000.0;
    let gpu = &cold[0];
    println!(
        "arena resident: {} turns, {:.1} MB VRAM",
        arena.resident_turns(),
        arena.resident_bytes() as f64 / (1024.0 * 1024.0)
    );

    // STEADY-STATE: every turn already resident → only the tiny index uploads.
    let reps = 5;
    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(
            arena
                .scan_weighted(&segments, &[query.as_slice()], &[])
                .unwrap(),
        );
    }
    let steady_ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;

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
        println!("  top-{k:<2}: selection overlap CPU vs paged-GPU = {ov:.3}");
    }
    println!(
        "max abs diff = {max_abs:.4}, max rel diff = {max_rel:.6}\n\
         CPU per-file scan: {cpu_ms:.1} ms   |   paged-GPU steady: {steady_ms:.2} ms  \
         (cold first scan {cold_ms:.1} ms)  ⇒ {:.1}x",
        cpu_ms / steady_ms.max(1e-6)
    );

    // The gate: same exchange SELECTION, scores within fast-math tolerance.
    assert!(
        worst_k_overlap >= 0.999,
        "paged belief scan must select the SAME exchanges as the CPU per-file scan \
         (worst top-K overlap {worst_k_overlap:.3})"
    );
    assert!(
        max_rel <= 1e-3,
        "paged-GPU vs CPU per-exchange scores exceed tolerance (max rel {max_rel:.6})"
    );
    println!(
        "\nPARITY OK — paged belief scan matches CPU per-file, {:.1}x faster (steady-state).",
        cpu_ms / steady_ms.max(1e-6)
    );
}
