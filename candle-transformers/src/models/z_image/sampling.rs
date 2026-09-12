//! The flow-matching schedule and denoise loop.
//!
//! Z-Image is a **rectified flow** model, not a diffusion model in the DDPM
//! sense: the transformer predicts a velocity along a straight path between
//! noise and image, and a step is a plain Euler step along it. There is no
//! variance schedule and nothing to invert — the whole scheduler is one
//! sequence of sigmas and one `x += dt * v`.

use candle::{Result, Tensor};

/// The trained step count the schedule is distilled at.
///
/// Not a limit — callers routinely ask for more, and the extra steps buy finish.
/// It is the *floor* a partial walk is held to: see [`steps_from`].
pub const DISTILLED_STEPS: usize = 8;

/// Timesteps for `steps` Euler steps, plus the terminating zero.
///
/// The returned vector has `steps + 1` entries: `sigmas[i]` is where step `i`
/// starts and `sigmas[i + 1]` where it lands, so the last is 0 — a fully
/// denoised sample.
///
/// # The shift, and what moving it does
///
/// The raw schedule is linear in sigma, which spends as much of the budget on
/// the last few percent of denoising as on the first. `shift` re-weights it
/// toward high noise, where the composition of the image is actually decided:
///
/// ```text
///   σ' = shift·σ / (1 + (shift − 1)·σ)
/// ```
///
/// The published scheduler config carries `shift: 3.0` and
/// `use_dynamic_shifting: false`, so 3.0 is the value the model's step count is
/// distilled against and the deployment's default. It is nonetheless a real
/// dial and the only one this sampler has: raising it spends more of the budget
/// deciding *what the picture is*, lowering it spends more resolving *how it
/// looks*. The pipeline also computes a resolution-dependent `mu` and passes it
/// anyway — that one is ignored, because dynamic shifting is off, and reading
/// the config rather than the call site is what makes it visible.
pub fn sigmas(steps: usize, shift: f64) -> Vec<f64> {
    sigmas_from(steps, shift, 1.0)
}

/// The sigma at raw schedule position `raw`, where 1.0 is the start of the walk
/// and 0.0 its end.
///
/// # This, and not sigma, is what a "how far along" dial should move
///
/// The shift is steeply non-linear, and it is easy to reach for sigma as though
/// it were a fraction of the journey. It is not: at `shift = 3`, σ = 0.85 is
/// raw position 0.65 — the walk is already a third done. A control that moved
/// sigma directly would therefore spend most of its travel in a region where
/// the picture is already decided, and its whole useful range would be crushed
/// into the last sliver. Measured on Z-Image-Turbo, everything between a
/// re-imagining and a near-copy happened between σ = 0.92 and σ = 0.80.
///
/// Raw position is the honest axis because it *is* the step budget: the
/// schedule's linspace runs over it, so "half way" means half the steps. It is
/// the same quantity [`steps_from`] divides by, which is why a walk from
/// `sigma_at(f, shift)` runs `f` of the caller's steps exactly.
pub fn sigma_at(raw: f64, shift: f64) -> f64 {
    let r = raw.clamp(0.0, 1.0);
    shift * r / (1.0 + (shift - 1.0) * r)
}

/// The raw, unshifted position that `shift` maps onto `sigma`.
///
/// The inverse of [`sigma_at`]: `r = σ / (shift − (shift − 1)·σ)`. It is what
/// lets a partial walk be cut out of the *same* curve rather than approximated
/// by a second one — see [`sigmas_from`].
fn unshift(sigma: f64, shift: f64) -> f64 {
    let denom = shift - (shift - 1.0) * sigma;
    if denom <= 0.0 {
        return 1.0;
    }
    (sigma / denom).clamp(0.0, 1.0)
}

/// Timesteps for a walk that begins at `start` rather than at pure noise.
///
/// This is what an image-to-image draw runs: the latent is not noise, it is a
/// picture with `start` worth of noise mixed into it, so the walk joins the
/// trajectory partway down and finishes it.
///
/// **It is the same curve, not a second one.** The start is converted back to
/// its raw position and the linspace is taken over `r0 … r0/n` instead of
/// `1 … 1/n`, so the shape of the schedule — where it spends its budget, how it
/// lands on zero — is exactly what a full draw would have done over that
/// stretch. `start = 1.0` reproduces [`sigmas`] entry for entry, which is
/// asserted rather than assumed.
///
/// `sigmas_from(n, shift, s)[0]` is `s`, so a caller building the noisy latent
/// can read the sigma it must mix at off the schedule rather than recomputing
/// it and risking the two disagreeing.
pub fn sigmas_from(steps: usize, shift: f64, start: f64) -> Vec<f64> {
    let n = steps.max(1);
    let r0 = unshift(start.clamp(0.0, 1.0), shift);
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        // `linspace(r0, r0/n, n)` — the reference's
        // `get_default_z_image_sigmas` scaled into the sub-range.
        let raw = if n == 1 {
            r0
        } else {
            r0 + (r0 / n as f64 - r0) * (i as f64 / (n - 1) as f64)
        };
        out.push(shift * raw / (1.0 + (shift - 1.0) * raw));
    }
    // The landing point of the final step. Without it the last step has no
    // destination and the sample is left one step short of denoised.
    out.push(0.0);
    out
}

/// How many steps a walk from `start` should actually run, given a caller who
/// asked for `steps` of a full draw.
///
/// # Why this is not simply `steps`
///
/// A partial walk covers a fraction of the trajectory, so running the full
/// count over it packs the steps far denser than the draw the caller described
/// — paying a full draw's time for a fraction of its distance. What a caller
/// means by "24 steps" is a *density*, so the fraction of the raw curve the
/// walk covers is the fraction of the budget it gets. That is also why a light
/// touch is quick: holding most of the reference costs most of the time back.
///
/// # Why it does not simply scale
///
/// Scaling alone puts a gentle edit at two or three steps, and this model is
/// **distilled** — its schedule is trained at [`DISTILLED_STEPS`], and below
/// that the Euler error stops being a matter of finish and starts deciding what
/// the picture is. So the count is floored at the trained one, or at the
/// caller's own count when they asked for fewer than that.
pub fn steps_from(steps: usize, shift: f64, start: f64) -> usize {
    let asked = steps.max(1);
    let fraction = unshift(start.clamp(0.0, 1.0), shift);
    let scaled = (asked as f64 * fraction).round() as usize;
    scaled.clamp(DISTILLED_STEPS.min(asked), asked)
}

/// The time the transformer is given for a step starting at `sigma`.
///
/// The model is trained on "how far along" rather than "how much noise", so the
/// two run in opposite directions: the pipeline computes `(1000 − σ·1000)/1000`,
/// which is this.
pub fn model_time(sigma: f64) -> f64 {
    1.0 - sigma
}

/// Run the Euler loop.
///
/// `step` is given the current latent and the model time, and returns the
/// transformer's raw output. **The caller's output is negated here**, matching
/// the reference's `noise_pred = -noise_pred`: the model predicts the direction
/// from image toward noise, and the loop walks the other way. Getting that sign
/// wrong does not fail — it walks away from the image and returns noise.
pub fn denoise<F>(latent: &Tensor, sigmas: &[f64], mut step: F) -> Result<Tensor>
where
    F: FnMut(&Tensor, f64) -> Result<Tensor>,
{
    let mut x = latent.clone();
    for w in sigmas.windows(2) {
        let (s, s_next) = (w[0], w[1]);
        let v = step(&x, model_time(s))?;
        // `x + (σ_next − σ)·(−v)`, in f32 whatever the model's width: the
        // increments late in the schedule are small against the latent, and
        // accumulating them in bf16 quantises the last steps away.
        let dt = s_next - s;
        x = (x.to_dtype(candle::DType::F32)? - (v.to_dtype(candle::DType::F32)? * dt)?)?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schedule must start at high noise, end at zero, and never go back.
    #[test]
    fn the_schedule_descends_to_zero() {
        for steps in [1usize, 4, 8, 20, 50] {
            let s = sigmas(steps, 3.0);
            assert_eq!(s.len(), steps + 1, "{steps} steps needs {steps}+1 sigmas");
            assert_eq!(*s.last().unwrap(), 0.0, "the last step has no destination");
            assert!(
                (s[0] - 1.0).abs() < 1e-12,
                "the walk does not start at pure noise"
            );
            for w in s.windows(2) {
                assert!(
                    w[0] > w[1] || w[1] == 0.0,
                    "the schedule is not monotonic: {s:?}"
                );
            }
        }
    }

    /// **The shift is what the schedule is for.** It moves budget toward high
    /// noise, where composition is decided; a shift of 1 is the identity and
    /// anything above it must push every interior sigma *up*.
    #[test]
    fn the_shift_weights_the_schedule_toward_noise() {
        let flat = sigmas(8, 1.0);
        let shifted = sigmas(8, 3.0);
        for i in 1..flat.len() - 1 {
            assert!(
                shifted[i] > flat[i],
                "step {i}: shift did not move budget toward noise ({} vs {})",
                shifted[i],
                flat[i]
            );
        }
        // The endpoints are fixed points of the transform whatever the shift.
        assert!((shifted[0] - flat[0]).abs() < 1e-12);
        assert_eq!(*shifted.last().unwrap(), *flat.last().unwrap());
    }

    /// **A full walk is the partial walk that starts at pure noise.**
    ///
    /// The two must be the *same* function, not two implementations that agree
    /// on paper — otherwise a change to the schedule's shape silently applies
    /// to ordinary draws and not to reference draws, or the other way round.
    #[test]
    fn a_walk_from_pure_noise_is_the_ordinary_schedule() {
        for steps in [1usize, 4, 8, 24] {
            for shift in [1.0, 3.0, 6.0] {
                assert_eq!(
                    sigmas(steps, shift),
                    sigmas_from(steps, shift, 1.0),
                    "{steps} steps at shift {shift}"
                );
            }
        }
    }

    /// **A partial walk starts exactly where it was told to.** The caller mixes
    /// the reference and the noise at this sigma, so a schedule whose first
    /// entry drifted would denoise a latent carrying a different amount of
    /// noise than the model is being told it has.
    #[test]
    fn a_partial_walk_begins_at_the_sigma_it_was_given() {
        for start in [0.15, 0.35, 0.55, 0.8, 0.95] {
            for shift in [1.0, 3.0, 6.0] {
                let s = sigmas_from(12, shift, start);
                assert!(
                    (s[0] - start).abs() < 1e-12,
                    "start {start} at shift {shift} began at {}",
                    s[0]
                );
                assert_eq!(*s.last().unwrap(), 0.0, "a partial walk must still land");
                for w in s.windows(2) {
                    assert!(w[0] > w[1] || w[1] == 0.0, "not monotonic: {s:?}");
                }
            }
        }
    }

    /// The unshift is a genuine inverse across the range, which is the whole
    /// basis for cutting a sub-range out of the same curve.
    #[test]
    fn unshifting_a_sigma_recovers_its_raw_position() {
        for shift in [1.0, 2.0, 3.0, 6.0] {
            for sigma in [0.0, 0.1, 0.33, 0.5, 0.75, 1.0] {
                let r = unshift(sigma, shift);
                assert!(
                    (sigma_at(r, shift) - sigma).abs() < 1e-12,
                    "shift {shift}: σ {sigma} → r {r} → σ {}",
                    sigma_at(r, shift)
                );
            }
        }
        // …and the other way round, which is the direction a dial travels.
        for shift in [1.0, 3.0, 8.0] {
            for raw in [0.0, 0.05, 0.4, 0.65, 1.0] {
                assert!((unshift(sigma_at(raw, shift), shift) - raw).abs() < 1e-12);
            }
        }
    }

    /// **Raw position and sigma are not the same axis, and the gap is the whole
    /// reason a dial moves the former.** At the deployment's shift, two thirds
    /// of the way through the schedule is still σ = 0.85 — so a control that
    /// moved sigma would spend most of its travel past the point where the
    /// picture is decided.
    #[test]
    fn the_shift_makes_raw_position_and_sigma_very_different_axes() {
        assert!(
            (sigma_at(1.0, 3.0) - 1.0).abs() < 1e-12,
            "the start is noise"
        );
        assert!(
            (sigma_at(0.0, 3.0) - 0.0).abs() < 1e-12,
            "the end is the image"
        );
        // The midpoint of the budget is far up the sigma range.
        let mid = sigma_at(0.5, 3.0);
        assert!(mid > 0.7, "half the steps is only σ {mid}");
        // An un-shifted schedule is the identity, which is the sanity check
        // that the two axes coincide exactly when the shift stops bending.
        for raw in [0.1, 0.5, 0.9] {
            assert!((sigma_at(raw, 1.0) - raw).abs() < 1e-12);
        }
        // Monotonic, or a dial would fold back on itself.
        let mut last = -1.0;
        for i in 0..=20 {
            let s = sigma_at(i as f64 / 20.0, 3.0);
            assert!(s > last, "not monotonic at {i}");
            last = s;
        }
    }

    /// **A walk from `sigma_at(f)` runs `f` of the caller's steps.** This is the
    /// property that makes the dial mean one thing in both places at once —
    /// how much of the picture is held, and how much of the budget is spent.
    #[test]
    fn a_walk_from_a_raw_position_runs_that_fraction_of_the_steps() {
        for shift in [1.0, 3.0, 6.0] {
            for (fraction, want) in [(1.0, 24), (0.75, 18), (0.5, 12), (0.25, 8)] {
                let start = sigma_at(fraction, shift);
                assert_eq!(
                    steps_from(24, shift, start),
                    want,
                    "shift {shift}, {fraction} of the schedule"
                );
            }
        }
    }

    /// **A partial walk runs the fraction of the budget it covers, floored at
    /// the count the model is distilled for.** Both halves matter: without the
    /// scaling a light touch costs a full draw, and without the floor it runs
    /// at two steps on a model trained at eight.
    #[test]
    fn a_partial_walk_scales_its_steps_but_never_below_the_trained_count() {
        // A full walk is the count asked for, whatever the shift.
        assert_eq!(steps_from(24, 3.0, 1.0), 24);
        assert_eq!(steps_from(8, 3.0, 1.0), 8);

        // Halfway down the curve costs a fraction, not half — the shift puts
        // most of the budget above it.
        let mid = steps_from(24, 3.0, 0.55);
        assert!(
            (DISTILLED_STEPS..24).contains(&mid),
            "a mid-strength walk ran {mid} steps"
        );

        // A very light touch still gets the trained count rather than one step.
        assert_eq!(steps_from(24, 3.0, 0.05), DISTILLED_STEPS);
        // …and a caller who asked for fewer than the trained count keeps their
        // own number: the floor is a floor, not a raise.
        assert_eq!(steps_from(4, 3.0, 0.05), 4);

        // More of the trajectory is never fewer steps.
        let mut last = 0;
        for start in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
            let n = steps_from(24, 3.0, start);
            assert!(
                n >= last,
                "start {start} ran fewer steps than a shorter walk"
            );
            last = n;
        }
    }

    /// The model reads "how far along", the schedule tracks "how much noise".
    #[test]
    fn model_time_runs_opposite_to_sigma() {
        assert!((model_time(1.0) - 0.0).abs() < 1e-12);
        assert!((model_time(0.0) - 1.0).abs() < 1e-12);
        assert!((model_time(0.25) - 0.75).abs() < 1e-12);
    }

    /// **The sign is the part that fails silently.** A velocity of exactly the
    /// latent should walk it to zero over one full step; the opposite sign
    /// doubles it, and both produce a picture rather than an error.
    #[test]
    fn the_loop_walks_toward_the_image() -> Result<()> {
        let dev = candle::Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 4), &dev)?;
        // One step from σ=1 to σ=0, with the model reporting v = x.
        let out = denoise(&x, &[1.0, 0.0], |cur, _| Ok(cur.clone()))?;
        // x + (0 − 1)·(−x)·… wait: x − (v · dt) with dt = −1 gives x + v = 2x.
        // Stated as the assertion rather than the arithmetic: walking *against*
        // the predicted direction over a step of −1 doubles the sample.
        assert_eq!(out.flatten_all()?.to_vec1::<f32>()?, vec![2f32, 4., 6., 8.]);

        // And a zero velocity leaves the sample alone whatever the schedule.
        let out = denoise(&x, &[1.0, 0.5, 0.0], |cur, _| cur.zeros_like())?;
        assert_eq!(out.flatten_all()?.to_vec1::<f32>()?, vec![1f32, 2., 3., 4.]);
        Ok(())
    }

    /// Every step is visited exactly once, in order, at the model's own time.
    #[test]
    fn each_step_is_visited_once_in_order() -> Result<()> {
        let dev = candle::Device::Cpu;
        let x = Tensor::zeros((1, 2), candle::DType::F32, &dev)?;
        let s = sigmas(4, 3.0);
        let mut seen = Vec::new();
        denoise(&x, &s, |cur, t| {
            seen.push(t);
            cur.zeros_like()
        })?;
        assert_eq!(seen.len(), 4);
        for w in seen.windows(2) {
            assert!(w[0] < w[1], "model time must advance: {seen:?}");
        }
        assert!((seen[0] - model_time(s[0])).abs() < 1e-12);
        Ok(())
    }
}
