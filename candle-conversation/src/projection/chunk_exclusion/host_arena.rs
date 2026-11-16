//! A [`BeliefArena`] whose statistics launches are computed on the host from
//! their definitions — what the learner's tests fold against, and count the
//! launches of. Every token is kept (`Keep::All`); any other launch refuses.

use std::cell::Cell;

use candle::Result;
use candle_kernels::provenance::BdpStats;

use crate::provenance::gallery_arena::{BeliefArena, PagedSegment, TokenStats, TokenVotes};
use crate::provenance::{WideQSig, FOLD_GROUPS};

/// Words a fold group holds.
const GW: usize = 8;

/// See the module documentation.
#[derive(Default)]
pub(crate) struct HostArena {
    /// Statistics launches answered.
    pub launches: Cell<usize>,
}

/// One probe token's statistics in fold group `g` against `segment`, its
/// cases reported from `case_base`.
fn stats_of(q: &WideQSig, g: usize, segment: &PagedSegment, case_base: i32) -> (BdpStats, u64) {
    let words = g * GW..(g + 1) * GW;
    let mut best = vec![0u32; segment.n_cases];
    let (mut sum, mut sumsq, mut n) = (0u64, 0u64, 0u64);
    for w in &segment.windows {
        if w.case >= segment.n_cases {
            continue;
        }
        let toks = w.turn.with_decoded(<[WideQSig]>::to_vec);
        for t in toks.get(w.start..w.end.min(toks.len())).unwrap_or(&[]) {
            if t.words.len() < FOLD_GROUPS * GW {
                continue;
            }
            let a: u32 = q.words[words.clone()]
                .iter()
                .zip(&t.words[words.clone()])
                .map(|(x, y)| (!(x ^ y)).count_ones())
                .sum();
            best[w.case] = best[w.case].max(a);
            sum += u64::from(a);
            sumsq += u64::from(a) * u64::from(a);
            n += 1;
        }
    }
    let (mut a1, mut a2, mut c1, mut c2) = (0u32, 0u32, -1i32, -1i32);
    for (c, &m) in best.iter().enumerate() {
        if m > a1 {
            (a2, c2) = (a1, c1);
            (a1, c1) = (m, c as i32);
        } else if m > a2 {
            (a2, c2) = (m, c as i32);
        }
    }
    let stats = BdpStats {
        a1,
        c1: if c1 < 0 { -1 } else { case_base + c1 },
        a2,
        c2: if c2 < 0 { -1 } else { case_base + c2 },
        sum,
        sumsq,
    };
    (stats, n)
}

impl BeliefArena for HostArena {
    fn n_groups(&self) -> usize {
        FOLD_GROUPS
    }

    fn scan_weighted(
        &self,
        _: &[PagedSegment],
        _: &[&[WideQSig]],
        _: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        Err(candle::Error::Msg(
            "the host arena answers statistics only".into(),
        ))
    }

    fn scan_segment_weighted(
        &self,
        _: &[PagedSegment],
        _: &[&[WideQSig]],
        _: &[&[f32]],
    ) -> Result<Vec<Vec<f32>>> {
        Err(candle::Error::Msg(
            "the host arena answers statistics only".into(),
        ))
    }

    fn scan_votes(&self, _: &[PagedSegment], _: &[&[WideQSig]]) -> Result<TokenVotes> {
        Err(candle::Error::Msg(
            "the host arena answers statistics only".into(),
        ))
    }

    fn scan_stats(&self, segments: &[PagedSegment], probes: &[&[WideQSig]]) -> Result<TokenStats> {
        self.launches.set(self.launches.get() + 1);
        let groups: Vec<usize> = segments
            .iter()
            .fold(segments[0].groups, |g, s| g.union(s.groups))
            .iter()
            .collect();
        let mut rows = Vec::new();
        let mut seg_tokens = vec![0u64; segments.len()];
        let mut per_probe = Vec::new();
        for probe in probes {
            let full: Vec<&WideQSig> = probe
                .iter()
                .filter(|t| t.words.len() >= FOLD_GROUPS * GW)
                .collect();
            per_probe.push(full.len());
            for q in full {
                for &g in &groups {
                    let mut base = 0i32;
                    for (s, seg) in segments.iter().enumerate() {
                        let (row, n) = stats_of(q, g, seg, base);
                        rows.push(row);
                        seg_tokens[s] = n;
                        base += seg.n_cases as i32;
                    }
                }
            }
        }
        Ok(TokenStats {
            rows,
            per_probe,
            groups,
            seg_tokens,
        })
    }
}
