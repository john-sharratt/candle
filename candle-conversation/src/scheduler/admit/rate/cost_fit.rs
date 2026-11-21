//! Fits a prefill forward's costs by least squares over the forwards the engine
//! has run.
//!
//! A forward is `T = f + X/bw + W·c + P·p`, and each observation is one equation in
//! those costs, so forwards at varied widths, residencies and depths determine them
//! together. Each observation decays the running sums by the dampening before it is
//! added, so the fit tracks a machine that moves, such as a warming expert cache or
//! thermal throttling. A cost whose feature has not varied is dropped rather than
//! guessed, and every answer is checked for physical sense, since a clamped nonsense
//! answer looks exactly like a real one downstream.

/// The running fit of a forward's costs: the copy rate, the per-token compute,
/// attention per attended pair, and the fixed cost per forward.
///
/// # Why a fit rather than two constants
///
/// A prefill forward is `T = X/bw + W·c` — non-resident bytes over the bus,
/// plus compute for the rows. Both terms are properties of *this* machine and
/// *this* checkpoint: `bw` is the card's link and how much of it the expert
/// path gets, `c` is the model's compute per row at the activation dtype. A
/// number measured on a 4090 with a 35B is wrong on a Blackwell with anything,
/// and wrong in a direction that matters — under-stating `c` makes the marginal
/// row look cheaper than it is, so the planner over-values widening and spends
/// residency for gains that are not there.
///
/// One forward is one equation in two unknowns. A *set* of forwards at
/// different widths and residencies determines both, and that is all this is:
/// the normal equations of a least squares over the features, decayed,
/// solved whenever the observations have enough spread to be conditioned.
///
/// # Forgetting
///
/// Each observation decays the accumulators by `1 − alpha` before adding
/// itself, so the fit tracks a machine whose behaviour moves — an expert cache
/// warming, a dtype change, thermal throttling — instead of averaging over the
/// whole run. With `alpha` at its default the fit has a memory of a few dozen
/// forwards.
///
/// # Four costs
///
/// `T = f + X/bw + W·c + P·p`: a fixed cost per forward, the copy, compute per
/// row, and attention per attended pair `P` — each row times the context it
/// attends. Each is a column of the normal equations, and a column that has
/// not varied is dropped rather than guessed: a fully resident card copies
/// nothing (no `X`), and forwards all at one depth make `P` a multiple of `W`
/// (no separate `p`), and either way the rest still solve.
///
/// **The intercept is a cost, not a fudge.** Leaving `f` out did not make it
/// zero — it made it invisible: with a streaming cache `X` absorbs it, but on
/// a fully resident card `X` is zero, so the projected rate collapsed to
/// `W/(W·c) = 1/c` — flat in width, and the fill refused the second row of
/// every wave as saturated. Measured on the 30B-A3B at 72 GiB: nine turns
/// queued, one admitted, 32 t/s against a batched ceiling of 518.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct CostFit {
    /// The Gram matrix of the features `[X, W, P, 1]`, decayed.
    pub(super) gram: [[f64; 4]; 4],
    /// Σ feature × T, the right-hand side.
    pub(super) rhs: [f64; 4],
    /// Observations folded in, for the conditioning guard.
    pub(super) n: u64,
}

impl CostFit {
    /// Observations required before the fit is believed at all. Two would be
    /// enough to solve the system and far too few to trust: a pair of forwards
    /// at nearly the same width gives a near-singular matrix and an answer that
    /// swings wildly on noise. The determinant guard below is the real check;
    /// this stops it being consulted while the sums are still tiny.
    pub(super) const MIN_SAMPLES: u64 = 8;

    /// The fraction of the product of the active diagonal the determinant must
    /// reach for the system to be considered conditioned.
    ///
    /// The determinant is zero exactly when the observations lie on a plane —
    /// forwards all at one width, or one residency, or one depth — from which
    /// the costs cannot be separated however many there are. Requiring a real
    /// fraction of the diagonal's product (Hadamard's bound, so the test is
    /// scale-free) means the features have genuinely varied.
    pub(super) const MIN_CONDITION: f64 = 1e-3;

    /// Feature indices.
    pub(super) const X: usize = 0;
    pub(super) const W: usize = 1;
    pub(super) const P: usize = 2;
    pub(super) const ONE: usize = 3;

    /// Fold one forward in, decaying what came before.
    pub(super) fn observe(&mut self, x: f64, w: f64, p: f64, t: f64, alpha: f64) {
        let keep = 1.0 - alpha.clamp(0.0, 1.0);
        let f = [x, w, p, 1.0];
        for i in 0..4 {
            for j in 0..4 {
                self.gram[i][j] = self.gram[i][j] * keep + f[i] * f[j];
            }
            self.rhs[i] = self.rhs[i] * keep + f[i] * t;
        }
        self.n += 1;
    }

    /// Solve for the costs, or `None` while the observations cannot separate
    /// even width from the fixed cost.
    ///
    /// The widest set of features that is conditioned wins: all four, then
    /// without attention, then without the copy, then width alone. Every
    /// answer is checked for physical sense before being returned — a copy rate
    /// positive and no faster than the link, a positive per-row compute, and
    /// non-negative fixed and attention costs — because a clamped nonsense
    /// answer is indistinguishable from a real one downstream. Each may come
    /// back at exactly zero: a card whose per-forward overhead, or whose
    /// attention at the depths seen, really is negligible should be able to say
    /// so. A set whose answer fails the check gives way to the next.
    pub(super) fn solve(&self, link_bytes_per_s: f64) -> Option<FitSolution> {
        if self.n < Self::MIN_SAMPLES {
            return None;
        }
        // Each pair is a set with attention and the same set without it. An
        // unconditioned set gives way to the next; an unphysical one with
        // attention gives way to its partner — attention at noise can come out
        // negative where the rest is sound — but an unphysical one without it
        // is the forwards describing noise or a stall, and nothing is learned.
        const PAIRS: [(&[usize], &[usize]); 2] = [
            (
                &[CostFit::X, CostFit::W, CostFit::P, CostFit::ONE],
                &[CostFit::X, CostFit::W, CostFit::ONE],
            ),
            (
                &[CostFit::W, CostFit::P, CostFit::ONE],
                &[CostFit::W, CostFit::ONE],
            ),
        ];
        for (with, without) in PAIRS {
            if let Ok(sol) = self.solve_over(with, link_bytes_per_s) {
                return Some(sol);
            }
            match self.solve_over(without, link_bytes_per_s) {
                Ok(sol) => return Some(sol),
                Err(Unsolved::Unphysical) => return None,
                Err(Unsolved::Unconditioned) => {}
            }
        }
        None
    }

    /// The least squares over the features in `set`, if conditioned and
    /// physical.
    pub(super) fn solve_over(
        &self,
        set: &[usize],
        link_bytes_per_s: f64,
    ) -> Result<FitSolution, Unsolved> {
        use Unsolved::{Unconditioned, Unphysical};
        let k = set.len();
        let mut m = [[0.0f64; 5]; 4];
        let mut diag = 1.0f64;
        for (r, &i) in set.iter().enumerate() {
            for (c, &j) in set.iter().enumerate() {
                m[r][c] = self.gram[i][j];
            }
            m[r][k] = self.rhs[i];
            diag *= self.gram[i][i];
        }
        if !(diag.is_finite() && diag > 0.0) {
            return Err(Unconditioned);
        }
        // Gaussian elimination with partial pivoting; the determinant is the
        // product of the pivots, for the conditioning guard.
        let mut det = 1.0f64;
        for col in 0..k {
            let pivot = (col..k)
                .max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))
                .ok_or(Unconditioned)?;
            if m[pivot][col] == 0.0 {
                return Err(Unconditioned);
            }
            if pivot != col {
                m.swap(pivot, col);
                det = -det;
            }
            det *= m[col][col];
            for row in col + 1..k {
                let factor = m[row][col] / m[col][col];
                let pivot_row = m[col];
                for (dst, src) in m[row][col..=k].iter_mut().zip(&pivot_row[col..=k]) {
                    *dst -= factor * src;
                }
            }
        }
        if !(det.is_finite() && det.abs() > diag * Self::MIN_CONDITION) {
            return Err(Unconditioned);
        }
        let mut sol = [0.0f64; 4];
        for row in (0..k).rev() {
            let mut s = m[row][k];
            for c in row + 1..k {
                s -= m[row][c] * sol[c];
            }
            sol[row] = s / m[row][row];
        }
        let at = |feature: usize| set.iter().position(|&i| i == feature).map(|r| sol[r]);
        let c = at(Self::W).ok_or(Unconditioned)?;
        let fixed = at(Self::ONE).ok_or(Unconditioned)?;
        let attend = at(Self::P).unwrap_or(0.0);
        let copy = at(Self::X);
        if ![c, fixed, attend].iter().all(|v| v.is_finite()) {
            return Err(Unphysical);
        }
        if c <= 0.0 || fixed < 0.0 || attend < 0.0 {
            return Err(Unphysical);
        }
        let bw = match copy {
            Some(a) => {
                if !(a.is_finite() && a > 0.0) {
                    return Err(Unphysical);
                }
                let bw = 1.0 / a;
                if !(bw.is_finite() && bw > 0.0) {
                    return Err(Unphysical);
                }
                Some(bw.min(link_bytes_per_s))
            }
            // Silent about the link: the forwards copied nothing that varied.
            None => None,
        };
        Ok(FitSolution {
            bw,
            compute_secs_per_token: c,
            fixed_secs: fixed,
            attend_secs_per_pair: at(Self::P),
        })
    }
}

/// What a solved fit determined. `bw` is `None` when the observations copied nothing and
/// so say nothing about the link, and `attend_secs_per_pair` when they were all at one
/// depth and so say nothing apart from `c`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FitSolution {
    pub(super) bw: Option<f64>,
    pub(super) compute_secs_per_token: f64,
    pub(super) fixed_secs: f64,
    pub(super) attend_secs_per_pair: Option<f64>,
}

/// Why a set of the fit's features gave no answer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Unsolved {
    /// The observations did not vary in these features enough to separate them.
    Unconditioned,
    /// They did, and the answer was not a cost: noise, or a stalled forward.
    Unphysical,
}

/// The largest index in `[lo, hi]` that `pays`, the indices that pay being a
/// prefix of the range; `None` when none does. With a `hint` the search
/// gallops out from it and bisects only the bracket it finds, so an answer at
/// or beside the hint costs a probe or two.
pub(super) fn largest_paying(
    lo: usize,
    hi: usize,
    hint: Option<usize>,
    mut pays: impl FnMut(usize) -> bool,
) -> Option<usize> {
    if lo > hi {
        return None;
    }
    // A bracket: `ok` pays (or is none), every index above `bad` does not.
    let (mut ok, mut bad): (Option<usize>, usize) = (None, hi + 1);
    if let Some(h) = hint.map(|h| h.clamp(lo, hi)) {
        if pays(h) {
            ok = Some(h);
            let mut step = 1;
            loop {
                let next = h.saturating_add(step);
                if next > hi {
                    break;
                }
                if pays(next) {
                    ok = Some(next);
                    step *= 2;
                } else {
                    bad = next;
                    break;
                }
            }
        } else {
            bad = h;
            let mut step = 1;
            loop {
                if h < lo + step {
                    break;
                }
                let next = h - step;
                if pays(next) {
                    ok = Some(next);
                    break;
                }
                bad = next;
                step *= 2;
            }
        }
    }
    // Bisect strictly between the bracket's ends.
    let mut a = ok.map_or(lo, |o| o + 1);
    let mut b = bad;
    while a < b {
        let mid = a + (b - a) / 2;
        if pays(mid) {
            ok = Some(mid);
            a = mid + 1;
        } else {
            b = mid;
        }
    }
    ok
}
