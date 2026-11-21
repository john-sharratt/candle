//! Fits a decode step's per-layer time against the experts it had to fetch and the
//! creep that rode it.
//!
//! The time is a fixed part, a cost per missed expert, and terms for the creep's
//! rows and attention. Co-batched steps and decode-only steps both teach it, so the
//! decode's cost is separated from the creep's by the fit itself rather than by
//! subtracting an estimate learned elsewhere. The answer is non-negative least
//! squares over the subsets of columns the steps varied in, with the fixed term and
//! the misses always kept, because a fit that cannot say what a miss costs is no
//! answer. It is solved once as each step folds in, and every later price reads
//! that answer.

/// The hit rate at resident `fraction` of the model under exponent `gamma`:
/// none with nothing resident, all with everything.
pub(super) fn hit_at(fraction: f64, gamma: f64) -> f64 {
    let f = fraction.clamp(0.0, 1.0);
    if f <= 0.0 {
        0.0
    } else {
        f.powf(gamma).clamp(0.0, 1.0)
    }
}

/// The running fit of a decode step's per-layer time against the experts it
/// had to fetch and the creep that rode it: `t = a + b·m + c·r + d·p`, `m`
/// the distinct routed experts a layer did not hold, `r` the creep's rows
/// through it and `p` the query×key pairs those rows attend — the creep's
/// cost at depth, which at 120–190K context its rows alone do not carry.
///
/// **What a missed expert costs is this card's to say.** The model it replaces
/// took a layer as its compute or its copy, whichever was longer, and put
/// every step longer than the copy into one constant compute time — so a cost
/// that grows with the experts routed but is not the bus's, the CPU pool
/// computing the misses, was flattened into a figure the same at every width.
/// On WhiteWorld (b1b3dae14, 2026-10-20) the step with sixteen decodes more
/// priced at 69.3 ms, the same as one more, where twenty-three decodes had
/// stepped 451–580 ms. The slope here is whatever a miss costs — bus, CPU or
/// both — learned from the widths the card actually stepped at.
///
/// **Learned from co-batched steps too, the creep their third term.** Taught
/// on decode-only steps alone, the fit went blind wherever every step carries
/// a creep: on BlackWorld (136060fab, 2026-10-20) it never solved through a
/// heavy-read, and the run before it froze on the phase's last decode-only
/// steps and priced 0.7–3.3 s against co-batched steps of about 0.3 s. Each
/// co-batched step is one more equation with its creep rows in it, so the
/// decode's cost is separated from the creep's by the fit itself rather than
/// by subtracting an estimate learned from the decode-only steps it lacks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct DecodeFit {
    /// The normal equations over `[1, m, q, r, p]` — `XᵀX`, `Xᵀt` and `tᵀt` —
    /// with `q` and `p` in millions so the sums stay within a few orders of
    /// the rest.
    pub(super) xx: [[f64; 5]; 5],
    pub(super) xt: [f64; 5],
    pub(super) tt: f64,
    pub(super) n: u64,
    /// The answer as of the last step folded in: solved there, once, and read
    /// by every price after it — the trade prices the decode side many times
    /// a pass, and the fit changes only when it learns.
    pub(super) cached: Option<Result<[f64; 5], FitWhy>>,
    /// The fewest and most misses its steps have had, each drawn toward their
    /// mean by the weight the sums lose — so an extreme fades with the steps
    /// that set it. `None` before the first.
    pub(super) miss_range: Option<(f64, f64)>,
}

/// Why the decode fit gave no answer, for the line that reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitWhy {
    /// Fewer steps than it believes from.
    Samples,
    /// No subset of its columns both conditioned and came out a cost.
    Unseparated,
}

impl DecodeFit {
    /// Steps folded in before the fit is believed.
    pub(super) const MIN_SAMPLES: u64 = 8;
    /// The fraction of its own sum of squares each column's pivot must keep:
    /// the steps must have varied in it apart from the others for its
    /// coefficient to be told from theirs.
    pub(super) const MIN_CONDITION: f64 = 1e-3;
    /// The positions and pairs columns' unit.
    pub(super) const MILLION: f64 = 1e6;
    /// Columns: the fixed term, the misses, the decodes' attended positions,
    /// the creep's rows, the creep's pairs.
    pub(super) const K: usize = 5;

    /// Fold one step in — `q` the positions its decode rows attended, `r` and
    /// `p` its creep's rows and pairs through the layer — the sums so far kept
    /// `1 − alpha` of their weight.
    pub(super) fn observe(&mut self, m: f64, q: f64, r: f64, p: f64, t: f64, alpha: f64) {
        let keep = 1.0 - alpha.clamp(0.0, 1.0);
        let x = [1.0, m, q / Self::MILLION, r, p / Self::MILLION];
        for i in 0..Self::K {
            for j in 0..Self::K {
                self.xx[i][j] = self.xx[i][j] * keep + x[i] * x[j];
            }
            self.xt[i] = self.xt[i] * keep + x[i] * t;
        }
        self.tt = self.tt * keep + t * t;
        self.n += 1;
        let mean = self.xx[0][1] / self.xx[0][0].max(f64::MIN_POSITIVE);
        self.miss_range = Some(match self.miss_range {
            Some((lo, hi)) => (
                m.min(mean - (mean - lo) * keep),
                m.max(mean + (hi - mean) * keep),
            ),
            None => (m, m),
        });
        self.cached = Some(self.compute());
    }

    /// `(a, b)` — per-layer seconds at no miss, and per missed expert — or
    /// `None` until the steps have varied enough to separate them.
    pub(super) fn solve(&self) -> Option<(f64, f64)> {
        self.answer().ok().map(|x| (x[0], x[1]))
    }

    /// `(a, b, c, d)`: `c` per creep row through a layer, `d` per pair it
    /// attends there.
    pub(super) fn solve_all(&self) -> Option<(f64, f64, f64, f64)> {
        self.answer()
            .ok()
            .map(|x| (x[0], x[1], x[3], x[4] / Self::MILLION))
    }

    /// Per-layer seconds per position a decode row attends, `0` when the
    /// steps did not vary in depth apart from the rest.
    pub(super) fn depth(&self) -> Option<f64> {
        self.answer().ok().map(|x| x[2] / Self::MILLION)
    }

    /// The fit's coefficients over `[1, m, q, r, p]` in its own units, or why
    /// there are none.
    ///
    /// **Non-negative least squares over its columns' subsets.** A cost cannot
    /// be negative, and the columns the steps did not vary apart from the
    /// others say nothing; so of the subsets that keep the fixed term and the
    /// misses, condition, and come out all costs, the one that leaves the
    /// least residual is the answer, each left-out column's coefficient zero.
    /// The fixed term and the misses always: a fit that cannot say what a
    /// miss costs is no answer, and the physical model prices it instead — one
    /// width's steps fitted as a flat layer time priced every dislodge free.
    /// With a creep in the steps a subset must keep a creep column: without one
    /// the decode's coefficients would be holding the creep's cost.
    pub(super) fn answer(&self) -> Result<[f64; 5], FitWhy> {
        self.cached.unwrap_or(Err(FitWhy::Samples))
    }

    /// [`Self::answer`], solved from the sums.
    pub(super) fn compute(&self) -> Result<[f64; 5], FitWhy> {
        if self.n < Self::MIN_SAMPLES {
            return Err(FitWhy::Samples);
        }
        let present = |i: usize| self.xx[i][i] > f64::EPSILON * self.xx[0][0].max(1.0);
        let creep = present(3) || present(4);
        let mut best: Option<(f64, [f64; 5])> = None;
        for mask in 0u8..8 {
            let cols: Vec<usize> = [0, 1]
                .into_iter()
                .chain((2..Self::K).filter(|&c| mask & (1 << (c - 2)) != 0))
                .collect();
            if cols.iter().any(|&c| c >= 2 && !present(c)) {
                continue;
            }
            if creep && !cols.iter().any(|&c| c >= 3) {
                continue;
            }
            let Some(coef) = self.solve_cols(&cols) else {
                continue;
            };
            let mut x = [0.0; 5];
            for (&c, v) in cols.iter().zip(&coef) {
                x[c] = *v;
            }
            if !x.iter().all(|v| v.is_finite() && *v >= 0.0) {
                continue;
            }
            let rss = self.residual(&x);
            if best.is_none_or(|(b, _)| rss < b) {
                best = Some((rss, x));
            }
        }
        best.map(|(_, x)| x).ok_or(FitWhy::Unseparated)
    }

    /// The residual sum of squares `x` leaves: `tᵀt − 2xᵀXᵀt + xᵀXᵀXx`.
    pub(super) fn residual(&self, x: &[f64; 5]) -> f64 {
        let mut xxx = 0.0;
        for i in 0..Self::K {
            for j in 0..Self::K {
                xxx += x[i] * self.xx[i][j] * x[j];
            }
        }
        let xxt: f64 = (0..Self::K).map(|i| x[i] * self.xt[i]).sum();
        self.tt - 2.0 * xxt + xxx
    }

    /// What the fit has seen, for the line that reports it: its steps and the
    /// spread of each column — the standard deviation of `m`, `q`, `r` and
    /// `p` (`q` and `p` in millions) over the weight it holds.
    /// The misses the fit has seen steps at, fewest to most (`miss_range`); a
    /// fit restored without them spans two standard deviations of its mean.
    pub(super) fn miss_support(&self) -> (f64, f64) {
        if let Some(range) = self.miss_range {
            return range;
        }
        let w = self.xx[0][0].max(f64::MIN_POSITIVE);
        let mean = self.xx[0][1] / w;
        let sd = (self.xx[1][1] / w - mean * mean).max(0.0).sqrt();
        ((mean - 2.0 * sd).max(0.0), mean + 2.0 * sd)
    }

    pub(super) fn spread(&self) -> (u64, [f64; 4]) {
        let w = self.xx[0][0].max(f64::MIN_POSITIVE);
        let sd = |i: usize| {
            let mean = self.xx[0][i] / w;
            (self.xx[i][i] / w - mean * mean).max(0.0).sqrt()
        };
        (self.n, [sd(1), sd(2), sd(3), sd(4)])
    }

    /// Least squares over `cols` by elimination on the normal equations,
    /// refusing a pivot that keeps less than [`Self::MIN_CONDITION`] of its
    /// column's sum of squares.
    pub(super) fn solve_cols(&self, cols: &[usize]) -> Option<Vec<f64>> {
        let k = cols.len();
        let mut a: Vec<Vec<f64>> = cols
            .iter()
            .map(|&i| cols.iter().map(|&j| self.xx[i][j]).collect())
            .collect();
        let mut b: Vec<f64> = cols.iter().map(|&i| self.xt[i]).collect();
        let diag: Vec<f64> = (0..k).map(|i| a[i][i]).collect();
        for p in 0..k {
            if !a[p][p].is_finite() || a[p][p] <= diag[p] * Self::MIN_CONDITION {
                return None;
            }
            let pivot = a[p].clone();
            for r in p + 1..k {
                let f = a[r][p] / pivot[p];
                for (v, &u) in a[r].iter_mut().zip(&pivot).skip(p) {
                    *v -= f * u;
                }
                b[r] -= f * b[p];
            }
        }
        let mut x = vec![0.0; k];
        for p in (0..k).rev() {
            let s: f64 = (p + 1..k).map(|c| a[p][c] * x[c]).sum();
            x[p] = (b[p] - s) / a[p][p];
        }
        Some(x)
    }
}
