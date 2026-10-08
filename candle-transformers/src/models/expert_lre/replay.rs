//! Offline replay of residency policies over a written routing trace.
//!
//! The gate writes each config's routing (`routing_trace`, the prefill as
//! warm-up and the decode as the interval scored); this replays it under a
//! scored eviction policy with the zone's own admission rules, so a policy can
//! be judged in seconds against the same trace's LRU and Belady references
//! instead of a ten-minute gate run per idea:
//!
//! - every routed expert of an invocation is read by it, so none of them is
//!   evicted to make room for another of the same invocation;
//! - a decode miss is always installed, evicting the lowest-scored resident;
//! - a prompt-only miss takes a free slot only — a prompt fills the zone's
//!   holes and never evicts (`docs/moe_live_dispatch_design.md` §0.7.1).
//!
//! Run it after a gate has written its traces with
//! `cargo test --release -p candle-transformers --lib expert_lre::replay::tests::replay_report -- --ignored --nocapture`.

use super::belady::{belady, lru, Invocation};
use std::path::Path;

/// One trace file, decoded (`routing_trace`'s format).
pub(crate) struct TraceFile {
    pub(crate) invocations: Vec<Invocation>,
    /// Per invocation, per expert: `(tokens, decode)`.
    pub(crate) weights: Vec<Vec<(u16, bool)>>,
    pub(crate) mark: usize,
    pub(crate) pinned_requests: usize,
    pub(crate) capacity: usize,
}

impl TraceFile {
    pub(crate) fn decode(bytes: &[u8]) -> Option<Self> {
        let mut at = 0usize;
        let u32_at = |at: &mut usize| -> Option<u32> {
            let v = u32::from_le_bytes(bytes.get(*at..*at + 4)?.try_into().ok()?);
            *at += 4;
            Some(v)
        };
        let n = u32_at(&mut at)? as usize;
        let mark = u32_at(&mut at)? as usize;
        let pinned_requests = u32_at(&mut at)? as usize;
        let capacity = u32_at(&mut at)? as usize;
        let u16_at = |at: &mut usize| -> Option<u16> {
            let v = u16::from_le_bytes(bytes.get(*at..*at + 2)?.try_into().ok()?);
            *at += 2;
            Some(v)
        };
        let mut invocations = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for _ in 0..n {
            let row = u16_at(&mut at)?;
            let len = u16_at(&mut at)? as usize;
            let mut experts = Vec::with_capacity(len);
            let mut w = Vec::with_capacity(len);
            for _ in 0..len {
                experts.push(u16_at(&mut at)?);
                let x = u16_at(&mut at)?;
                w.push((x & 0x7fff, x & 0x8000 != 0));
            }
            invocations.push((row, experts));
            weights.push(w);
        }
        (at == bytes.len()).then_some(Self {
            invocations,
            weights,
            mark,
            pinned_requests,
            capacity,
        })
    }

    pub(crate) fn read(path: &Path) -> Option<Self> {
        Self::decode(&std::fs::read(path).ok()?)
    }

    /// A hit rate in percent on the measured rate's terms: the permanently
    /// resident layers' requests counted as hits.
    pub(crate) fn rate(&self, (hits, misses): (usize, usize)) -> f64 {
        let total = hits + misses + self.pinned_requests;
        100.0 * (hits + self.pinned_requests) as f64 / total.max(1) as f64
    }
}

/// How a scored policy credits and ages an expert.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScoredPolicy {
    /// Credit for an invocation that routed the expert from a decode row:
    /// a constant, or the tokens that routed it.
    pub(crate) decode_credit: Credit,
    /// Credit for a prompt-only routing.
    pub(crate) prompt_credit: f32,
    /// The recency term's decay per pass.
    pub(crate) decay: f32,
    /// The long-memory term: credit only for decode hits, its own decay, and
    /// its weight in the score.
    pub(crate) reuse_decay: f32,
    pub(crate) reuse_weight: f32,
    /// Admission: a decode miss that needs a victim takes its slot only when
    /// it scores above it (TinyLFU's rule); otherwise it stays out and the
    /// victim stays.
    pub(crate) admit_above_victim: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Credit {
    Constant(f32),
    PerToken(f32),
}

impl ScoredPolicy {
    /// The pipeline thread's policy (`cache.rs`): +1 a decode routing, +0.1 a
    /// prompt hit, the recency term ×0.85 a pass, decode reuse ×0.99 at a
    /// weight of 0.1.
    pub(crate) const ZONE: Self = Self {
        decode_credit: Credit::Constant(1.0),
        prompt_credit: 0.1,
        decay: 0.85,
        reuse_decay: 0.99,
        reuse_weight: 0.1,
        admit_above_victim: false,
    };
}

/// Hits and misses of `policy` over `t`, counted from its mark, at its
/// capacity.
pub(crate) fn scored(t: &TraceFile, policy: ScoredPolicy) -> (usize, usize) {
    let (mut hits, mut misses) = (0usize, 0usize);
    scored_observed(t, policy, |i, _, hit| {
        if i >= t.mark {
            if hit {
                hits += 1;
            } else {
                misses += 1;
            }
        }
    });
    (hits, misses)
}

/// [`scored`]'s replay, reporting every request to `observe` as
/// `(invocation, key, hit)` — the key `row << 16 | expert`.
fn scored_observed(
    t: &TraceFile,
    policy: ScoredPolicy,
    mut observe: impl FnMut(usize, usize, bool),
) {
    let key = |row: u16, e: u16| (row as usize) << 16 | e as usize;
    let mut recency: std::collections::HashMap<usize, f32> = Default::default();
    let mut reuse: std::collections::HashMap<usize, f32> = Default::default();
    let mut resident: std::collections::HashSet<usize> = Default::default();
    let mut last_row: Option<u16> = None;
    for (i, ((row, experts), weights)) in t.invocations.iter().zip(&t.weights).enumerate() {
        if last_row.is_some_and(|l| *row <= l) {
            recency.values_mut().for_each(|s| *s *= policy.decay);
            reuse.values_mut().for_each(|s| *s *= policy.reuse_decay);
        }
        last_row = Some(*row);
        let mut missed = Vec::new();
        for (&e, &(tokens, decode)) in experts.iter().zip(weights) {
            let id = key(*row, e);
            let hit = resident.contains(&id);
            observe(i, id, hit);
            if !hit {
                missed.push((id, decode));
            }
            let credit = if decode {
                match policy.decode_credit {
                    Credit::Constant(c) => c,
                    Credit::PerToken(c) => c * tokens as f32,
                }
            } else {
                policy.prompt_credit
            };
            *recency.entry(id).or_default() += credit;
            if decode && hit {
                *reuse.entry(id).or_default() += 1.0;
            }
        }
        if missed.is_empty() {
            continue;
        }
        let score = |id: &usize| {
            recency.get(id).copied().unwrap_or(0.0)
                + policy.reuse_weight * reuse.get(id).copied().unwrap_or(0.0)
        };
        let free = t.capacity.saturating_sub(resident.len());
        // Decode misses, best first; a prompt miss never takes a victim.
        let mut decode_missed: Vec<(f32, usize)> = missed
            .iter()
            .filter(|m| m.1)
            .map(|m| (score(&m.0), m.0))
            .collect();
        decode_missed.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let evict = decode_missed.len().saturating_sub(free);
        let mut refused: std::collections::HashSet<usize> = Default::default();
        if evict > 0 {
            let this: std::collections::HashSet<usize> =
                experts.iter().map(|&e| key(*row, e)).collect();
            let mut cands: Vec<(f32, usize)> = resident
                .iter()
                .filter(|id| !this.contains(id))
                .map(|id| (score(id), *id))
                .collect();
            let n = evict.min(cands.len());
            if n < cands.len() {
                cands.select_nth_unstable_by(n, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            }
            let mut victims = cands[..n].to_vec();
            victims.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            // The misses that need a victim are the weakest ones, past the
            // free slots; the strongest of them meets the weakest victim, and
            // so on up — once a pair is refused, every later one would be.
            let needing: Vec<(f32, usize)> =
                decode_missed[free.min(decode_missed.len())..].to_vec();
            let mut v = victims.into_iter();
            for (s, id) in needing {
                match v.next() {
                    Some((vs, vid)) if !policy.admit_above_victim || s > vs => {
                        resident.remove(&vid);
                    }
                    _ => {
                        refused.insert(id);
                    }
                }
            }
        }
        missed.retain(|m| !refused.contains(&m.0));
        // Decode misses first — the evictions made room for them — then
        // prompt misses into whatever is still free.
        missed.sort_by_key(|m| !m.1);
        for (id, _) in missed {
            if resident.len() < t.capacity {
                resident.insert(id);
            }
        }
    }
}

/// Hits and misses of evicting the resident used least often over the
/// scored interval itself — an oracle of each expert's popularity, without
/// its timing. Near [`belady`], the gap to the optimum is popularity a long
/// memory could learn; far below it, the gap needs the timing of reuse.
pub(crate) fn oracle_lfu(t: &TraceFile) -> (usize, usize) {
    let key = |row: u16, e: u16| (row as usize) << 16 | e as usize;
    let mut uses: std::collections::HashMap<usize, f32> = Default::default();
    for (row, experts) in &t.invocations[t.mark..] {
        for &e in experts {
            *uses.entry(key(*row, e)).or_default() += 1.0;
        }
    }
    let mut table = std::collections::HashMap::new();
    for (&id, &n) in &uses {
        table.insert(id, n);
    }
    scored_by(t, |id| table.get(&id).copied().unwrap_or(0.0))
}

/// [`scored`] with a fixed score per expert, never updated.
fn scored_by(t: &TraceFile, score: impl Fn(usize) -> f32) -> (usize, usize) {
    let key = |row: u16, e: u16| (row as usize) << 16 | e as usize;
    let mut resident: std::collections::HashSet<usize> = Default::default();
    let (mut hits, mut misses) = (0usize, 0usize);
    for (i, ((row, experts), weights)) in t.invocations.iter().zip(&t.weights).enumerate() {
        let counted = i >= t.mark;
        let mut missed = Vec::new();
        for (&e, &(_, decode)) in experts.iter().zip(weights) {
            let id = key(*row, e);
            if resident.contains(&id) {
                hits += usize::from(counted);
            } else {
                misses += usize::from(counted);
                missed.push((id, decode));
            }
        }
        let free = t.capacity.saturating_sub(resident.len());
        let evict = missed.iter().filter(|m| m.1).count().saturating_sub(free);
        if evict > 0 {
            let this: std::collections::HashSet<usize> =
                experts.iter().map(|&e| key(*row, e)).collect();
            let mut cands: Vec<(f32, usize)> = resident
                .iter()
                .filter(|id| !this.contains(id))
                .map(|&id| (score(id), id))
                .collect();
            let n = evict.min(cands.len());
            if n < cands.len() {
                cands.select_nth_unstable_by(n, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            }
            for &(_, id) in &cands[..n] {
                resident.remove(&id);
            }
        }
        missed.sort_by_key(|m| !m.1);
        for (id, _) in missed {
            if resident.len() < t.capacity {
                resident.insert(id);
            }
        }
    }
    (hits, misses)
}

/// The misses of `policy` over `t`'s scored interval, in order, as `(row,
/// expert)` keys — the stream the host tiers below VRAM serve.
pub(crate) fn miss_stream(t: &TraceFile, policy: ScoredPolicy) -> Vec<usize> {
    let mut out = Vec::new();
    scored_observed(t, policy, |i, id, hit| {
        if !hit && i >= t.mark {
            out.push(id);
        }
    });
    out
}

/// Hits of an LRU cache of `slots` over `stream`, counting only keys `cold`
/// admits — the pad caches the experts no warm slot holds.
pub(crate) fn pad_hits(
    stream: &[usize],
    slots: usize,
    cold: impl Fn(usize) -> bool,
) -> (usize, usize) {
    let inv: Vec<Invocation> = stream
        .iter()
        .filter(|&&id| cold(id))
        .map(|&id| ((id >> 16) as u16, vec![(id & 0xffff) as u16]))
        .collect();
    lru(&inv, slots, 0)
}

/// The two references over a trace file, on its own terms.
pub(crate) fn references(t: &TraceFile) -> (f64, f64) {
    (
        t.rate(lru(&t.invocations, t.capacity, t.mark)),
        t.rate(belady(&t.invocations, t.capacity, t.mark)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(invocations: Vec<Invocation>, decode: bool, mark: usize, capacity: usize) -> TraceFile {
        let weights = invocations
            .iter()
            .map(|(_, e)| e.iter().map(|_| (1u16, decode)).collect())
            .collect();
        TraceFile {
            invocations,
            weights,
            mark,
            pinned_requests: 0,
            capacity,
        }
    }

    /// The encoder's bytes decode back to the same trace.
    #[test]
    fn a_written_trace_decodes() {
        let bytes = [
            2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0x04, 0x10, 0, 0, //
            3, 0, 2, 0, 7, 0, 2, 0x80, 9, 0, 1, 0, //
            4, 0, 1, 0, 2, 1, 0xff, 0x7f,
        ];
        let t = TraceFile::decode(&bytes).unwrap();
        assert_eq!(t.invocations, vec![(3, vec![7, 9]), (4, vec![258])]);
        assert_eq!(
            t.weights,
            vec![vec![(2, true), (1, false)], vec![(32767, false)]]
        );
        assert_eq!((t.mark, t.pinned_requests, t.capacity), (1, 1, 4100));
        assert!(
            TraceFile::decode(&bytes[..bytes.len() - 1]).is_none(),
            "truncated"
        );
    }

    /// Two slots, decode routing 1 2 | 1 3 | 1 2 (one pass each, row 0): the
    /// scored policy keeps the twice-routed 1, evicting 2 for 3 and 3 for 2.
    #[test]
    fn a_scored_policy_keeps_the_frequent_expert() {
        let t = file(
            vec![(0, vec![1, 2]), (0, vec![1, 3]), (0, vec![1, 2])],
            true,
            0,
            2,
        );
        // 1 2 miss; 1 hit, 3 miss (evicts 2); 1 hit, 2 miss (evicts 3).
        assert_eq!(scored(&t, ScoredPolicy::ZONE), (2, 4));
    }

    /// A prompt-only miss takes a free slot but never evicts.
    #[test]
    fn a_prompt_miss_never_evicts() {
        let t = file(vec![(0, vec![1]), (0, vec![2]), (0, vec![1])], false, 0, 1);
        // 1 miss and installed (free); 2 miss, no free slot, not installed;
        // 1 hit.
        assert_eq!(scored(&t, ScoredPolicy::ZONE), (1, 2));
    }

    /// Admission keeps a resident that scores at least the miss: under plain
    /// frequency, two slots holding 1 (routed once) and 2 (twice), a
    /// once-routed 3 ties 1 and is refused, so the last 1 hits; without
    /// admission 3 evicts 1 and 1 misses again.
    #[test]
    fn admission_keeps_a_resident_that_scores_at_least_the_miss() {
        let t = file(
            vec![(0, vec![1, 2]), (0, vec![2]), (0, vec![3]), (0, vec![1])],
            true,
            0,
            2,
        );
        let lfu = ScoredPolicy {
            decay: 1.0,
            ..ScoredPolicy::ZONE
        };
        let adm = ScoredPolicy {
            admit_above_victim: true,
            ..lfu
        };
        assert_eq!(scored(&t, adm), (2, 3));
        assert_eq!(scored(&t, lfu), (1, 4));
    }

    /// The replay report over the traces the Flash-Next gate writes.
    #[test]
    #[ignore = "reads the traces the Flash-Next gate writes into target/routing_traces"]
    fn replay_report() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../target/routing_traces/qwen38_flash_next");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            println!("no traces in {} — run the gate first", dir.display());
            return;
        };
        let mut paths: Vec<_> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        paths.sort();
        let policies: Vec<(&str, ScoredPolicy)> = vec![
            ("zone", ScoredPolicy::ZONE),
            (
                "per-token",
                ScoredPolicy {
                    decode_credit: Credit::PerToken(1.0),
                    ..ScoredPolicy::ZONE
                },
            ),
            (
                "decay .95",
                ScoredPolicy {
                    decay: 0.95,
                    ..ScoredPolicy::ZONE
                },
            ),
            (
                "decay .99",
                ScoredPolicy {
                    decay: 0.99,
                    ..ScoredPolicy::ZONE
                },
            ),
            (
                "LFU",
                ScoredPolicy {
                    decay: 1.0,
                    ..ScoredPolicy::ZONE
                },
            ),
            (
                "zone+adm",
                ScoredPolicy {
                    admit_above_victim: true,
                    ..ScoredPolicy::ZONE
                },
            ),
            (
                "LFU+adm",
                ScoredPolicy {
                    decay: 1.0,
                    admit_above_victim: true,
                    ..ScoredPolicy::ZONE
                },
            ),
            (
                ".99+adm",
                ScoredPolicy {
                    decay: 0.99,
                    admit_above_victim: true,
                    ..ScoredPolicy::ZONE
                },
            ),
            (
                "tok .95",
                ScoredPolicy {
                    decode_credit: Credit::PerToken(1.0),
                    decay: 0.95,
                    ..ScoredPolicy::ZONE
                },
            ),
            (
                "reuse 1.0",
                ScoredPolicy {
                    reuse_weight: 1.0,
                    ..ScoredPolicy::ZONE
                },
            ),
        ];
        // The pad's question first: of the zone's misses on experts no warm
        // slot holds (a stratified 31% are warm here), how many an LRU pad of
        // each size would hold from an earlier miss.
        println!(
            "{:<22} {:>9} {:>8} {:>8} {:>8} {:>8}",
            "trace", "cold miss", "pad 512", "1024", "2048", "4096"
        );
        for p in &paths {
            let Some(t) = TraceFile::read(p) else {
                continue;
            };
            let stream = miss_stream(&t, ScoredPolicy::ZONE);
            let cold = |id: usize| (id.wrapping_mul(0x9e37_79b9) >> 7) % 100 >= 31;
            let n_cold = stream.iter().filter(|&&id| cold(id)).count();
            print!(
                "{:<22} {n_cold:>9}",
                p.file_name().unwrap().to_string_lossy()
            );
            for slots in [512, 1024, 2048, 4096] {
                let (h, _) = pad_hits(&stream, slots, cold);
                print!(" {:>7.1}%", 100.0 * h as f64 / n_cold.max(1) as f64);
            }
            println!();
        }
        print!(
            "{:<22} {:>6} {:>6} {:>8}",
            "trace", "LRU", "MIN", "oracleLFU"
        );
        for (name, _) in &policies {
            print!(" {name:>10}");
        }
        println!();
        for p in paths {
            let Some(t) = TraceFile::read(&p) else {
                continue;
            };
            let (l, m) = references(&t);
            print!(
                "{:<22} {l:>6.1} {m:>6.1} {:>8.1}",
                p.file_name().unwrap().to_string_lossy(),
                t.rate(oracle_lfu(&t))
            );
            for (_, policy) in &policies {
                print!(" {:>10.1}", t.rate(scored(&t, *policy)));
            }
            println!();
        }
    }
}
