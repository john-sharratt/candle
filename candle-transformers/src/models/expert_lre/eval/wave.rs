//! A discrete-event model of a batch of decode sessions streaming through the
//! layers over PCIe, estimating the token rate from stalls.
//!
//! The wave needs the union of experts its tokens route to at each layer, and
//! only part of the experts fit in VRAM. The model prefetches ahead from the
//! previous token's demand and from an optional champion predictor for cold
//! arrivals, and it evicts from behind the wave. A hard miss is an emergency load
//! on the critical path and a late prefetch is a partial stall. Compute is assumed
//! flat-out, so only stalls are modelled, and warmup is excluded so the timed
//! window measures steady state. `wave_optimize` searches prefetch depth and
//! amount, and compares the result with the simple prefetch-everything policy.

use super::*;

// ── The wave: batched decode streaming through layers over PCIe ──────────────
//
// A batch of decode sequences steps coherently through all 48 layers; at each
// layer the wave needs the UNION of experts its tokens route to.  It sweeps
// L=0→47, wraps to 0 for the next token.  Only ~60% of experts fit in VRAM, so
// the wave runs ahead of itself prefetching, and evicts from behind (L-1,
// wrapping — the longest reuse distance).  Stalls come from (a) hard misses
// (needed expert not resident → emergency load) and (b) PCIe saturation
// (prefetch can't keep up).  We model only stalls (compute is flat-out); the
// output is an estimated token rate.

pub(super) const WAVE_LAYERS: usize = 48;

pub(super) const WAVE_E: usize = 128;

/// Q4_K_M expert ≈ 3·768·2048 params · 4.5 bit.
pub(super) const EXPERT_MB: f64 = 2.6;

/// PCIe link bandwidth (given).
pub(super) const PCIE_GBPS: f64 = 8.0;

/// Expert transfer time (ms): 2.6 MB ÷ 8 GB/s.
pub(super) const TX_MS: f64 = EXPERT_MB / (PCIE_GBPS * 1000.0) * 1000.0;

/// Per-layer compute window (ms) at the 30 t/s target: (1000/30)/48.
pub(super) const TC_MS: f64 = (1000.0 / 30.0) / WAVE_LAYERS as f64;

#[derive(Clone, Copy, Default)]
pub(super) struct WaveResult {
    pub(super) rate: f64,           // estimated tokens/sec
    pub(super) hit: f64,            // % resident & ready, never prefetched (always-warm)
    pub(super) soft: f64,           // % prefetched & ready in time (latency hidden)
    pub(super) late: f64,           // % prefetched but not finished → partial stall
    pub(super) hard: f64,           // % not resident → emergency load stall
    pub(super) pcie_util: f64,      // fraction of wall-time the PCIe link was busy
    pub(super) stream_per_tok: f64, // experts transferred per token
}

/// Build the wave's per-(token, layer) demand for a batch of `b` sessions.
/// Each batch slot cycles one captured config at a distinct token offset, so the
/// per-layer union grows toward full demand (≈128/layer) as B increases —
/// modelling how a large, diverse production batch activates most experts.
pub(super) fn build_wave_demand(
    trace: &Trace,
    b: usize,
    max_tokens: usize,
) -> Vec<Vec<Vec<usize>>> {
    let cfgs = trace.configs();
    // Per config: [token][layer] = experts (decode passes only).
    let mut seqs: Vec<Vec<[Vec<usize>; WAVE_LAYERS]>> = Vec::new();
    for &c in &cfgs {
        let sub = trace.for_config(c);
        let mut seq: Vec<[Vec<usize>; WAVE_LAYERS]> = Vec::new();
        for pass in &sub.passes {
            if !pass.layers.iter().all(|(_, x)| x.len() <= 16) {
                continue;
            }
            let mut tok: [Vec<usize>; WAVE_LAYERS] = std::array::from_fn(|_| Vec::new());
            for (l, ex) in &pass.layers {
                if *l < WAVE_LAYERS {
                    tok[*l] = ex.clone();
                }
            }
            seq.push(tok);
        }
        if !seq.is_empty() {
            seqs.push(seq);
        }
    }
    let nc = seqs.len().max(1);
    // Each slot: (config index, token offset).  Distinct offsets per repeat give
    // diverse routings so the union grows with B beyond the config count.
    let slots: Vec<(usize, usize)> = (0..b.max(1)).map(|i| (i % nc, (i / nc) * 13)).collect();
    let ntok = max_tokens;
    let mut demand: Vec<Vec<Vec<usize>>> = Vec::with_capacity(ntok);
    for t in 0..ntok {
        let mut layers: Vec<Vec<usize>> = vec![Vec::new(); WAVE_LAYERS];
        for l in 0..WAVE_LAYERS {
            let mut set: HashSet<usize> = HashSet::new();
            for &(ci, off) in &slots {
                let seq = &seqs[ci];
                let tok = &seq[(t + off) % seq.len()];
                for &e in &tok[l] {
                    set.insert(e);
                }
            }
            let mut v: Vec<usize> = set.into_iter().collect();
            v.sort_unstable();
            layers[l] = v;
        }
        demand.push(layers);
    }
    demand
}

/// Discrete-event simulation of the wave.  Prefetch uses the previous token's
/// per-layer demand (recurrence) `prefetch_depth` layers ahead, capped at
/// `prefetch_cap` experts per (layer, lookahead).  Eviction drains from behind
/// the wave (L-1 first, wrapping).  Returns timing + outcome rates.
pub(super) fn wave_sim(
    demand: &[Vec<Vec<usize>>],
    budget: usize,
    prefetch_depth: usize,
    prefetch_cap: usize,
    k_champ: usize,
) -> WaveResult {
    // Optional online champion predictor for arrival prefetch of the next layer
    // (recurrence covers the sticky experts; the champion covers cold arrivals).
    let mut champ = MatrixPredictor::new(WAVE_LAYERS, WAVE_E, Scoring::Pmi { alpha: 0.5 });
    // resident[layer] = { expert : (ready_at_ms, prefetched_flag) }
    let mut resident: Vec<HashMap<usize, (f64, bool)>> = vec![HashMap::new(); WAVE_LAYERS];
    let mut count = 0usize;
    let mut clock = 0.0f64;
    let mut pcie_free = 0.0f64;
    let mut pcie_busy = 0.0f64;
    let (mut hit, mut soft, mut late, mut hard) = (0u64, 0u64, 0u64, 0u64);
    let mut transfers = 0u64;

    // Evict from behind the wave: layer (cur-1, cur-2, … wrapping).
    let evict = |resident: &mut Vec<HashMap<usize, (f64, bool)>>, count: &mut usize, cur: usize| {
        for back in 1..WAVE_LAYERS {
            let el = (cur + WAVE_LAYERS - back) % WAVE_LAYERS;
            if let Some(&k) = resident[el].keys().next() {
                resident[el].remove(&k);
                *count -= 1;
                return;
            }
        }
    };

    // Exclude warmup (the one-time cost of filling the cache) — measure
    // steady-state.  Counters and the timed window reset at `warmup`.
    let warmup = (demand.len() / 4).clamp(1, 20);
    let mut clock_warm = 0.0f64;
    let mut busy_warm = 0.0f64;

    let ntok = demand.len();
    for t in 0..ntok {
        if t == warmup {
            clock_warm = clock;
            busy_warm = pcie_busy;
            hit = 0;
            soft = 0;
            late = 0;
            hard = 0;
            transfers = 0;
        }
        if k_champ > 0 {
            champ.reset_pass();
        }
        for l in 0..WAVE_LAYERS {
            // 1. Access the experts this layer needs.
            for &e in &demand[t][l] {
                match resident[l].get(&e).copied() {
                    Some((ready, pf)) if ready <= clock => {
                        if pf {
                            soft += 1;
                            resident[l].insert(e, (ready, false)); // consumed
                        } else {
                            hit += 1;
                        }
                    }
                    Some((ready, _)) => {
                        // Prefetched but not finished: wave waits (partial stall).
                        clock = ready;
                        late += 1;
                        resident[l].insert(e, (ready, false));
                    }
                    None => {
                        // Hard miss: emergency load on the critical path.
                        if count >= budget {
                            evict(&mut resident, &mut count, l);
                        }
                        let start = clock.max(pcie_free);
                        clock = start + TX_MS;
                        pcie_free = clock;
                        pcie_busy += TX_MS;
                        transfers += 1;
                        resident[l].insert(e, (clock, false));
                        count += 1;
                        hard += 1;
                    }
                }
            }
            // 2. Compute this layer (flat-out) — the overlap window.
            clock += TC_MS;
            // 3a. Champion arrival prefetch of the immediate next layer (uses
            //     spare PCIe to fetch cold arrivals recurrence can't predict).
            if k_champ > 0 {
                let tl = (l + 1) % WAVE_LAYERS;
                for pe in champ.score_topk(l, &demand[t][l], k_champ, Scoring::Pmi { alpha: 0.5 }) {
                    if !resident[tl].contains_key(&pe) {
                        // Run ahead: evict from behind (L-1, wrapping) to make room.
                        if count >= budget {
                            evict(&mut resident, &mut count, l);
                        }
                        let start = clock.max(pcie_free);
                        pcie_free = start + TX_MS;
                        pcie_busy += TX_MS;
                        transfers += 1;
                        resident[tl].insert(pe, (pcie_free, true));
                        count += 1;
                    }
                }
                champ.observe(l, &demand[t][l]);
            }
            // 3b. Recurrence prefetch ahead using the previous token's demand.
            if t > 0 {
                for d in 1..=prefetch_depth {
                    let tl = (l + d) % WAVE_LAYERS;
                    let pred = &demand[t - 1][tl];
                    let mut c = 0usize;
                    for &pe in pred {
                        if c >= prefetch_cap {
                            break;
                        }
                        if !resident[tl].contains_key(&pe) {
                            // Run ahead: evict from behind (L-1, wrapping).
                            if count >= budget {
                                evict(&mut resident, &mut count, l);
                            }
                            let start = clock.max(pcie_free);
                            pcie_free = start + TX_MS;
                            pcie_busy += TX_MS;
                            transfers += 1;
                            resident[tl].insert(pe, (pcie_free, true));
                            count += 1;
                            c += 1;
                        }
                    }
                }
            }
        }
    }

    let steady_tok = (ntok.saturating_sub(warmup)).max(1);
    let elapsed = (clock - clock_warm).max(1e-9);
    let token_time = elapsed / steady_tok as f64;
    let acc = (hit + soft + late + hard).max(1) as f64;
    WaveResult {
        rate: 1000.0 / token_time,
        hit: 100.0 * hit as f64 / acc,
        soft: 100.0 * soft as f64 / acc,
        late: 100.0 * late as f64 / acc,
        hard: 100.0 * hard as f64 / acc,
        pcie_util: (pcie_busy - busy_warm).max(0.0) / elapsed,
        stream_per_tok: transfers as f64 / steady_tok as f64,
    }
}

/// §12 — wave simulation: estimate token rate vs prefetch depth × prefetch
/// amount at 60% VRAM, 8 GB/s PCIe, 30 t/s compute budget.
#[test]
#[ignore = "offline predictor-tuning sweep, not a correctness test: minutes per run. `cargo test -- --ignored expert_lre::eval` to run them."]
pub(super) fn wave_optimize() {
    let Some(records) = load_fixture() else {
        println!("\n[wave_optimize] no fixture.");
        return;
    };
    let trace = Trace::from_records(&records);
    let demand = build_wave_demand(&trace, trace.configs().len(), 80);

    // Diagnostics: per-layer demand and working set vs cache.
    let (mut dsum, mut dn) = (0usize, 0usize);
    let mut ws: HashSet<usize> = HashSet::new();
    for tok in &demand {
        for (l, ex) in tok.iter().enumerate() {
            dsum += ex.len();
            dn += 1;
            for &e in ex {
                ws.insert(l * WAVE_E + e);
            }
        }
    }
    let budget = (0.6 * (WAVE_LAYERS * WAVE_E) as f64).round() as usize;
    println!("\n=== §12 Wave simulation ===");
    println!(
        "tokens={} avg|demand/layer|={:.1} working-set={} VRAM budget(60%)={} slots",
        demand.len(),
        dsum as f64 / dn.max(1) as f64,
        ws.len(),
        budget
    );
    println!(
        "expert={:.1}MB  PCIe={:.0}GB/s  transfer={:.3}ms/expert  layer-compute={:.3}ms  budget/token≈{:.0} experts",
        EXPERT_MB,
        PCIE_GBPS,
        TX_MS,
        TC_MS,
        (TC_MS * WAVE_LAYERS as f64) / TX_MS
    );

    let _ = (demand, budget);
    let depths = [1usize, 2, 4, 8];
    let caps = [16usize, 64];
    let champs = [0usize, 8];
    let total = (WAVE_LAYERS * WAVE_E) as f64;
    // Optimise (depth × cap × champion-k) for a (demand, budget); return best.
    let optimise = |dem: &[Vec<Vec<usize>>], bud: usize| {
        let mut best = (0.0f64, 0usize, 0usize, 0usize, WaveResult::default());
        for &d in &depths {
            for &cap in &caps {
                for &kc in &champs {
                    let r = wave_sim(dem, bud, d, cap, kc);
                    if r.rate > best.0 {
                        best = (r.rate, d, cap, kc, r);
                    }
                }
            }
        }
        best
    };

    // ── Aggregate throughput grows with B (amortised expert loads) ──
    let bud60 = (0.60 * total).round() as usize;
    println!("\n-- per-session vs aggregate token rate at 60% VRAM (optimised) --");
    println!(
        "{:>5} {:>8} {:>8} {:>9} {:>11} {:>6} {:>6} {:>6} {:>6}",
        "B", "dem/lyr", "stream", "t/s/sess", "aggregate", "hit%", "soft%", "late%", "hard%"
    );
    for &bsz in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let dem = build_wave_demand(&trace, bsz, 60);
        let dpl = dem.iter().flatten().map(|e| e.len()).sum::<usize>() as f64
            / (dem.len() * WAVE_LAYERS) as f64;
        let (rate, _, _, _, r) = optimise(&dem, bud60);
        // hit (resident) / soft (prefetched, hidden) / late (prefetched, partial
        // stall) / hard (demand stall) — the full four-way access classification.
        println!(
            "{:>5} {:>8.1} {:>8.0} {:>9.1} {:>11.0} {:>6.0} {:>6.0} {:>6.0} {:>6.0}",
            bsz,
            dpl,
            r.stream_per_tok,
            rate,
            rate * bsz as f64,
            r.hit,
            r.soft,
            r.late,
            r.hard
        );
    }

    // ── 2D optimisation: best wave params vs (batch, VRAM residency) ──
    println!("\n-- 2D optimum params : VRAM residency × batch size --");
    println!(
        "{:>5} {:>5} {:>8} {:>9} {:>11} {:>6} {:>6} {:>12} {:>6}",
        "res%", "B", "stream", "t/s/sess", "aggregate", "hit%", "hard%", "depth/cap/kc", "pcie"
    );
    for &res in &[0.40f64, 0.50, 0.60, 0.70, 0.80, 1.00] {
        let bud = (res * total).round() as usize;
        for &bsz in &[1usize, 4, 16, 64, 256] {
            let dem = build_wave_demand(&trace, bsz, 60);
            let (rate, d, cap, kc, r) = optimise(&dem, bud);
            println!(
                "{:>5.0} {:>5} {:>8.0} {:>9.1} {:>11.0} {:>6.0} {:>6.0} {:>4}/{:<3}/{:<2} {:>5.0}%",
                res * 100.0,
                bsz,
                r.stream_per_tok,
                rate,
                rate * bsz as f64,
                r.hit,
                r.hard,
                d,
                cap,
                kc,
                r.pcie_util * 100.0
            );
        }
    }
    println!(
        "\nCompute floor = 30 t/s/session (weight-read amortised). 100% residency = control\n\
         (working set fits → no eviction → 30 t/s). depth=prefetch lookahead, cap=experts/\n\
         layer, kc=champion arrival-prefetch; eviction is always L-1 (behind, wrapping).\n"
    );

    // ── Saturation regime: drop the predictor, prefetch the full missing set ──
    // Compare the fully tuned optimum against the simple deterministic policy
    // (L+1 ahead, prefetch all missing — cap=E, no champion).  At/above the
    // streaming-saturation batch they should coincide (bandwidth-bound).
    println!(
        "-- saturation regime: tuned optimum vs prefetch-all-missing (depth=1, cap=E, kc=0) --"
    );
    println!(
        "{:>5} {:>5} {:>10} {:>14} {:>8}",
        "res%", "B", "tuned t/s", "prefetch-all", "stream"
    );
    for &res in &[0.60f64, 0.80] {
        let bud = (res * total).round() as usize;
        for &bsz in &[64usize, 256] {
            let dem = build_wave_demand(&trace, bsz, 60);
            let (tuned, ..) = optimise(&dem, bud);
            let r = wave_sim(&dem, bud, 1, WAVE_E, 0); // prefetch-all-missing
            println!(
                "{:>5.0} {:>5} {:>10.2} {:>14.2} {:>8.0}",
                res * 100.0,
                bsz,
                tuned,
                r.rate,
                r.stream_per_tok
            );
        }
    }
    println!(
        "\nAt saturation the two coincide (bandwidth-bound): the predictor adds nothing, so the\n\
         engine flips to deterministic prefetch-all-missing-L+1 + evict-L-1 — ride the wave.\n"
    );
}
