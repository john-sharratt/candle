//! A member's [`ForwardCost`] fitted to its wave samples: least squares with
//! every coefficient held non-negative, each window weighted by the forwards it
//! averages.
//!
//! Five coefficients, so the normal equations are solved directly: exact
//! coordinate descent on `AᵀWA x = AᵀWy`, each coordinate clamped at zero,
//! which converges to the non-negative optimum for a positive semidefinite
//! system.

use crate::cost::{ForwardCost, Work};
use crate::experts::ExpertGeometry;
use crate::sample::{Channel, Sample};

/// How well a fit explains its samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fitted {
    pub cost: ForwardCost,
    /// Samples it was fitted to.
    pub samples: usize,
    /// Forwards those samples average.
    pub fwds: u64,
    /// Forward-weighted R².
    pub r2: f64,
    /// Forward-weighted RMS error, in milliseconds.
    pub rms_ms: f64,
}

/// The cost of `channel` fitted to `samples` over `geometry`. A sample with no
/// hit rate is left out: its expert terms are unknown. `None` with fewer
/// samples than coefficients.
pub fn fit(samples: &[Sample], channel: Channel, geometry: &ExpertGeometry) -> Option<Fitted> {
    let rows: Vec<([f64; 5], f64, f64)> = samples
        .iter()
        .filter(|s| s.channel == channel && s.fwds > 0)
        .filter_map(|s| {
            let work = Work {
                tokens: s.tokens,
                kv: s.kv,
                hit: s.hit?,
            };
            Some((work.features(geometry), s.ms, f64::from(s.fwds)))
        })
        .collect();
    if rows.len() < 5 {
        return None;
    }
    let mut g = [[0.0f64; 5]; 5];
    let mut b = [0.0f64; 5];
    for (x, y, w) in &rows {
        for i in 0..5 {
            b[i] += w * x[i] * y;
            for j in 0..5 {
                g[i][j] += w * x[i] * x[j];
            }
        }
    }
    let c = nonnegative(&g, &b);
    let cost = ForwardCost::from_coefficients(c);
    let total_w: f64 = rows.iter().map(|r| r.2).sum();
    let mean = rows.iter().map(|(_, y, w)| w * y).sum::<f64>() / total_w;
    let (mut ss_res, mut ss_tot) = (0.0, 0.0);
    for (x, y, w) in &rows {
        let p: f64 = c.iter().zip(x).map(|(c, x)| c * x).sum();
        ss_res += w * (y - p).powi(2);
        ss_tot += w * (y - mean).powi(2);
    }
    Some(Fitted {
        cost,
        samples: rows.len(),
        fwds: total_w as u64,
        r2: if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            1.0
        },
        rms_ms: (ss_res / total_w).sqrt(),
    })
}

/// The non-negative `x` minimising `½xᵀGx − bᵀx`.
fn nonnegative(g: &[[f64; 5]; 5], b: &[f64; 5]) -> [f64; 5] {
    let mut x = [0.0f64; 5];
    for _ in 0..20_000 {
        let mut moved = 0.0f64;
        for j in 0..5 {
            if g[j][j] <= 0.0 {
                continue;
            }
            let others: f64 = (0..5).filter(|&k| k != j).map(|k| g[j][k] * x[k]).sum();
            let next = ((b[j] - others) / g[j][j]).max(0.0);
            moved = moved.max((next - x[j]).abs() / (1.0 + x[j].abs()));
            x[j] = next;
        }
        if moved < 1e-13 {
            break;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: ExpertGeometry = ExpertGeometry::FLASH_NEXT;

    fn sample(channel: Channel, tokens: f64, kv: f64, hit: f64, cost: &ForwardCost) -> Sample {
        let ms = cost.ms(&Work { tokens, kv, hit }, &G);
        Sample {
            channel,
            fwds: 10,
            rows: tokens,
            tokens,
            kv,
            ms,
            hit: Some(hit),
            alone: true,
        }
    }

    /// Samples drawn from a known cost give it back, with a perfect fit.
    #[test]
    fn a_known_cost_is_recovered() {
        let truth = ForwardCost {
            base_ms: 6.0,
            token_ms: 0.08,
            mkv_ms: 1.5,
            resident_gb_ms: 1.2,
            missed_gb_ms: 45.0,
        };
        let mut samples = Vec::new();
        for (t, kv, hit) in [
            (1.0, 50_000.0, 0.99),
            (8.0, 400_000.0, 0.97),
            (24.0, 1_200_000.0, 0.95),
            (170.0, 440_000.0, 0.90),
            (436.0, 400_000.0, 0.66),
            (128.0, 3_965_000.0, 0.38),
            (16.0, 900_000.0, 0.80),
        ] {
            samples.push(sample(Channel::Decode, t, kv, hit, &truth));
        }
        let f = fit(&samples, Channel::Decode, &G).unwrap();
        for (got, want) in f.cost.coefficients().iter().zip(truth.coefficients()) {
            assert!((got - want).abs() < 1e-6 * (1.0 + want), "{got} vs {want}");
        }
        assert!(f.r2 > 0.999_999_999);
        assert_eq!((f.samples, f.fwds), (7, 70));
    }

    /// A term the data says makes forwards faster is held at zero, not fitted
    /// negative.
    #[test]
    fn a_coefficient_is_never_negative() {
        let truth = ForwardCost {
            base_ms: 10.0,
            token_ms: 0.1,
            ..ForwardCost::default()
        };
        let mut samples: Vec<Sample> = [1.0, 4.0, 16.0, 64.0, 256.0, 512.0]
            .iter()
            .map(|t| sample(Channel::Prefill, *t, 1e5, 0.9, &truth))
            .collect();
        // More attended KV, a shade faster: noise the fit must not believe.
        for (i, s) in samples.iter_mut().enumerate() {
            s.kv = 1e5 * (i as f64 + 1.0);
            s.ms -= 0.01 * i as f64;
        }
        let f = fit(&samples, Channel::Prefill, &G).unwrap();
        assert!(f.cost.coefficients().iter().all(|c| *c >= 0.0));
        assert_eq!(f.cost.mkv_ms, 0.0);
    }

    /// Too few samples, or another channel's, fit nothing.
    #[test]
    fn too_few_samples_fit_nothing() {
        let c = ForwardCost {
            base_ms: 1.0,
            ..ForwardCost::default()
        };
        let s: Vec<Sample> = (0..4)
            .map(|i| sample(Channel::Decode, f64::from(i) + 1.0, 1e5, 0.9, &c))
            .collect();
        assert!(fit(&s, Channel::Decode, &G).is_none());
        assert!(fit(&s, Channel::Prefill, &G).is_none());
    }
}
