//! The subject's own colour under a soft alpha: multi-level foreground
//! estimation (Germer, Uelwer, Conrad, Harmeling, "Fast Multi-Level Foreground
//! Estimation", ICPR 2020 — pymatting's `estimate_foreground_ml`).
//!
//! # Why a cutout needs it
//!
//! A pixel the matte calls half opaque is a mix, `I = αF + (1 − α)B`, of the
//! subject's colour `F` and the background's `B`. Keeping the picture's own
//! colour `I` with that alpha puts the old background back into every soft
//! edge: a dark halo round a dog on dark grass, a green-yellow fringe between
//! a portrait's curls (AlienWorld's run on WhiteWorld, 2026-10-27). Solving for
//! `F` and keeping that instead is what makes the edge read as the subject's.
//!
//! # The method
//!
//! Each pixel's `F` and `B` minimise the compositing error plus a smoothness
//! term that pulls them toward their four neighbours' — weighted by how little
//! the alpha changes there, so colour flows along the subject and not across
//! its edge. That is a 2 × 2 solve per pixel per colour, swept in place
//! (Gauss–Seidel). Swept at full size alone it would need thousands of passes
//! to carry colour across a wide transparent region, so it starts at 1 × 1
//! and doubles: each level begins from the one below, nearest-neighbour
//! enlarged, and needs only a couple of sweeps. The constants are the paper's.

/// The smoothness weight's floor, so a flat alpha still couples neighbours.
const REGULARIZATION: f64 = 1e-5;
/// Sweeps per level at or below [`SMALL_SIZE`], where they are cheap.
const SMALL_ITERATIONS: usize = 10;
/// Sweeps per level above it.
const BIG_ITERATIONS: usize = 2;
const SMALL_SIZE: usize = 32;
/// How strongly an alpha step cuts the coupling across it.
const GRADIENT_WEIGHT: f64 = 1.0;

/// The foreground colour of a `w × h` picture under `alpha`.
///
/// `rgb` is interleaved, three values a pixel in `0..=1`; `alpha` one a
/// pixel in `0..=1`. The answer is interleaved the same way.
pub fn estimate(rgb: &[f32], alpha: &[f32], w: usize, h: usize) -> Vec<f32> {
    assert_eq!(rgb.len(), w * h * 3, "the picture is not w*h*3 values");
    assert_eq!(alpha.len(), w * h, "the alpha is not w*h values");
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let levels = (w.max(h) as f64).log2().ceil().max(0.0) as u32;
    let (mut fg, mut bg) = (vec![0f32; 3], vec![0f32; 3]);
    let (mut pw, mut ph) = (1usize, 1usize);
    for level in 0..=levels {
        let t = if levels == 0 {
            1.0
        } else {
            level as f64 / levels as f64
        };
        let lw = ((w as f64).powf(t).round() as usize).max(1);
        let lh = ((h as f64).powf(t).round() as usize).max(1);
        let image = nearest(rgb, w, h, 3, lw, lh);
        let a = nearest(alpha, w, h, 1, lw, lh);
        fg = nearest(&fg, pw, ph, 3, lw, lh);
        bg = nearest(&bg, pw, ph, 3, lw, lh);
        let sweeps = if lw <= SMALL_SIZE && lh <= SMALL_SIZE {
            SMALL_ITERATIONS
        } else {
            BIG_ITERATIONS
        };
        for _ in 0..sweeps {
            sweep(&image, &a, &mut fg, &mut bg, lw, lh);
        }
        (pw, ph) = (lw, lh);
    }
    fg
}

/// One Gauss–Seidel pass: each pixel's `F` and `B` from its own composite and
/// its neighbours' current values.
fn sweep(image: &[f32], alpha: &[f32], fg: &mut [f32], bg: &mut [f32], w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let a0 = alpha[i] as f64;
            let a1 = 1.0 - a0;
            let (mut a00, a01, mut a11) = (a0 * a0, a0 * a1, a1 * a1);
            let mut b0 = [0f64; 3];
            let mut b1 = [0f64; 3];
            for c in 0..3 {
                let v = image[i * 3 + c] as f64;
                b0[c] = a0 * v;
                b1[c] = a1 * v;
            }
            let neighbours = [
                (x.saturating_sub(1), y),
                ((x + 1).min(w - 1), y),
                (x, y.saturating_sub(1)),
                (x, (y + 1).min(h - 1)),
            ];
            for (nx, ny) in neighbours {
                let j = ny * w + nx;
                let da = REGULARIZATION + GRADIENT_WEIGHT * (a0 - alpha[j] as f64).abs();
                a00 += da;
                a11 += da;
                for c in 0..3 {
                    b0[c] += da * fg[j * 3 + c] as f64;
                    b1[c] += da * bg[j * 3 + c] as f64;
                }
            }
            let inv = 1.0 / (a00 * a11 - a01 * a01);
            for c in 0..3 {
                fg[i * 3 + c] = (inv * (a11 * b0[c] - a01 * b1[c])).clamp(0.0, 1.0) as f32;
                bg[i * 3 + c] = (inv * (a00 * b1[c] - a01 * b0[c])).clamp(0.0, 1.0) as f32;
            }
        }
    }
}

/// `src`, `sw × sh` with `ch` values a pixel, nearest-neighbour resized.
fn nearest(src: &[f32], sw: usize, sh: usize, ch: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; dw * dh * ch];
    for y in 0..dh {
        let sy = (y * sh / dh).min(sh - 1);
        for x in 0..dw {
            let sx = (x * sw / dw).min(sw - 1);
            let (s, d) = ((sy * sw + sx) * ch, (y * dw + x) * ch);
            out[d..d + ch].copy_from_slice(&src[s..s + ch]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A soft edge's colour is the subject's, not the mix.** A red subject
    /// on a green background, composited through a ramp of alpha: the picture
    /// at the half-opaque pixels is brown, and the estimate gives back red.
    #[test]
    fn a_soft_edge_recovers_the_subjects_colour() {
        let (w, h) = (64, 16);
        let (red, green) = ([1.0f32, 0.0, 0.0], [0.0f32, 1.0, 0.0]);
        let mut rgb = vec![0f32; w * h * 3];
        let mut alpha = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                // Opaque on the left third, a ramp across the middle, clear on the right.
                let a = ((40.0 - x as f32) / 16.0).clamp(0.0, 1.0);
                let i = y * w + x;
                alpha[i] = a;
                for c in 0..3 {
                    rgb[i * 3 + c] = a * red[c] + (1.0 - a) * green[c];
                }
            }
        }
        let fg = estimate(&rgb, &alpha, w, h);
        for x in 25..39 {
            let i = (h / 2) * w + x;
            let a = alpha[i];
            assert!(a > 0.05 && a < 0.95, "x={x} is not in the ramp");
            let (r, g) = (fg[i * 3], fg[i * 3 + 1]);
            assert!(
                r > 0.9 && g < 0.1,
                "x={x} at alpha {a:.2}: foreground ({r:.2}, {g:.2}) still carries the green"
            );
            // The picture itself does: the fringe this exists to remove.
            assert!(rgb[i * 3 + 1] > 0.05);
        }
    }

    /// An opaque picture's foreground is the picture.
    #[test]
    fn an_opaque_picture_is_its_own_foreground() {
        let (w, h) = (20, 12);
        let rgb: Vec<f32> = (0..w * h * 3).map(|i| (i % 7) as f32 / 7.0).collect();
        let fg = estimate(&rgb, &vec![1.0; w * h], w, h);
        for (a, b) in fg.iter().zip(&rgb) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn degenerate_sizes_are_answered() {
        assert!(estimate(&[], &[], 0, 0).is_empty());
        let fg = estimate(&[0.2, 0.4, 0.6], &[0.5], 1, 1);
        assert_eq!(fg.len(), 3);
    }
}
