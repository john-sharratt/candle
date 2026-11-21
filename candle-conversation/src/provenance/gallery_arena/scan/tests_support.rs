//! The host reference for the statistics that the scan's kernels produce.
//!
//! `host_stats` computes, for one probe token and fold group, the best agreement
//! of each case, the leading and runner-up cases, and the sum and sum of squares
//! of the agreements, from the definitions alone. A window whose case is out of
//! range is not scanned. The kernels are therefore tested against the definition,
//! not against one another. The signature helpers build deterministic inputs.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    pub(in super::super) fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: (0..24)
                .map(|w| fill.wrapping_mul(0x9E37).wrapping_add(w))
                .collect(),
        }
    }

    pub(in super::super) fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    pub(in super::super) fn rand_sig(state: &mut u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: (0..24).map(|_| xorshift(state)).collect(),
        }
    }

    /// The statistics a probe token `q` has in fold group `g` against
    /// `segment`'s windows, from the definitions: each case's best agreement,
    /// the leading and runner-up cases (ascending, strictly greater
    /// displacing), and the agreement sum and sum of squares over every token
    /// scanned. A window whose case is out of range is not scanned.
    pub(in super::super) fn host_stats(
        q: &WideQSig,
        g: usize,
        segment: &PagedSegment,
        case_base: i32,
    ) -> BdpStats {
        let gw = 8;
        let words = g * gw..(g + 1) * gw;
        let mut best = vec![0u32; segment.n_cases];
        let (mut sum, mut sumsq) = (0u64, 0u64);
        for w in &segment.windows {
            if w.case >= segment.n_cases {
                continue;
            }
            let toks = w.turn.with_decoded(<[WideQSig]>::to_vec);
            for t in &toks[w.start..w.end.min(toks.len())] {
                let a: u32 = q.words[words.clone()]
                    .iter()
                    .zip(&t.words[words.clone()])
                    .map(|(x, y)| (!(x ^ y)).count_ones())
                    .sum();
                best[w.case] = best[w.case].max(a);
                sum += u64::from(a);
                sumsq += u64::from(a) * u64::from(a);
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
        BdpStats {
            a1,
            c1: if c1 < 0 { -1 } else { case_base + c1 },
            a2,
            c2: if c2 < 0 { -1 } else { case_base + c2 },
            sum,
            sumsq,
        }
    }
}
