//! A model's RoPE schedule: which frequencies a slot rotates with at each
//! length (`docs/progressive_yarn.md` §2).
//!
//! Four kinds — plain, static linear, static Llama3, YaRN — and a YaRN schedule
//! with more than one rung is *progressive*: a slot inside the trained window
//! runs exactly the RoPE the model was trained with, and a slot that outgrows
//! it moves up to the factor the vendor publishes for its length. Every other
//! schedule has one rung, used at every length.

use super::angle::AngleArithmetic;
use super::table::plain_inv_freq;
use super::yarn::{mscale, yarn_freqs};

/// One rung of a YaRN schedule: the factor a slot uses while its reach is at
/// most `ceiling`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rung {
    pub ceiling: usize,
    pub factor: f32,
}

/// How a schedule scales the plain frequencies.
#[derive(Debug, Clone, PartialEq)]
pub enum Scaling {
    /// `1 / θ^(2i/d)`, unscaled.
    Plain,
    /// Every frequency divided by `factor`, at every length.
    Linear { factor: f32 },
    /// Frequencies stated outright — by a file (`rope_freqs.weight`, Llama3
    /// scaling), a preset, or a loader's own reading of its file.
    Stated { inv_freq: Vec<f32> },
    /// YaRN over trained window `l0`, one rung per ceiling.
    ///
    /// `temperature` says whether a rung applies YaRN's `m` — the Qwen lineage
    /// does, as its reference does; DeepSeek-V4's reference applies none.
    Yarn {
        l0: usize,
        beta_fast: f64,
        beta_slow: f64,
        rungs: Vec<Rung>,
        temperature: bool,
    },
}

/// The frequencies of one rung, and the scale its Q rotary pairs take.
#[derive(Debug, Clone, PartialEq)]
pub struct RungFreqs {
    pub inv_freq: Vec<f32>,
    /// `m²`, applied to Q's rotary pairs only; exactly 1.0 when the rung has no
    /// temperature.
    pub q_rot_scale: f32,
}

/// A model's RoPE: rotary width, base, scaling, and the longest reach it
/// supports.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeSchedule {
    pub rope_dim: usize,
    pub theta: f32,
    pub scaling: Scaling,
    supported_max: usize,
    /// The arithmetic of the tables' `LO` rows — exact unless the model was
    /// calibrated on the reference's f32 product (`super::angle`).
    lo_angle: AngleArithmetic,
}

impl RopeSchedule {
    /// Plain RoPE, supported to `supported_max` positions.
    pub fn plain(rope_dim: usize, theta: f32, supported_max: usize) -> Self {
        Self {
            rope_dim,
            theta,
            scaling: Scaling::Plain,
            supported_max,
            lo_angle: AngleArithmetic::Exact,
        }
    }

    /// Static linear scaling by `factor`.
    pub fn linear(rope_dim: usize, theta: f32, factor: f32, supported_max: usize) -> Self {
        Self {
            rope_dim,
            theta,
            scaling: Scaling::Linear { factor },
            supported_max,
            lo_angle: AngleArithmetic::Exact,
        }
    }

    /// This schedule with its tables' `LO` rows in `lo`'s arithmetic.
    pub fn with_lo_angle(mut self, lo: AngleArithmetic) -> Self {
        self.lo_angle = lo;
        self
    }

    /// The arithmetic of the tables' `LO` rows.
    pub fn lo_angle(&self) -> AngleArithmetic {
        self.lo_angle
    }

    /// One rung of frequencies stated outright, `rope_dim / 2` of them —
    /// Llama3 scaling, or a loader's own reading of its file.
    pub fn stated(inv_freq: Vec<f32>, supported_max: usize) -> candle::Result<Self> {
        if inv_freq.is_empty() {
            candle::bail!("stated schedule: no frequencies");
        }
        Ok(Self {
            rope_dim: inv_freq.len() * 2,
            theta: 0.0,
            scaling: Scaling::Stated { inv_freq },
            supported_max,
            lo_angle: AngleArithmetic::Exact,
        })
    }

    /// A YaRN schedule. Its ceilings must ascend, and the last is the
    /// supported maximum. A progressive schedule's first rung has factor 1 and
    /// ceiling `l0`: inside the trained window a slot runs the trained RoPE.
    pub fn yarn(
        rope_dim: usize,
        theta: f32,
        l0: usize,
        rungs: Vec<Rung>,
        temperature: bool,
    ) -> candle::Result<Self> {
        let Some(last) = rungs.last() else {
            candle::bail!("yarn schedule: no rungs");
        };
        if rungs.windows(2).any(|w| w[0].ceiling >= w[1].ceiling) {
            candle::bail!("yarn schedule: ceilings must ascend, got {rungs:?}");
        }
        if rungs.iter().any(|r| r.factor < 1.0) {
            candle::bail!("yarn schedule: a factor below 1 in {rungs:?}");
        }
        if rungs.len() > 1 && (rungs[0].factor != 1.0 || rungs[0].ceiling != l0) {
            candle::bail!(
                "yarn schedule: a progressive schedule starts at factor 1 up to the trained \
                 window {l0}, got {:?}",
                rungs[0]
            );
        }
        let supported_max = last.ceiling;
        Ok(Self {
            rope_dim,
            theta,
            scaling: Scaling::Yarn {
                l0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                rungs,
                temperature,
            },
            supported_max,
            lo_angle: AngleArithmetic::Exact,
        })
    }

    /// The longest reach, in positions, a slot may have.
    pub fn supported_max(&self) -> usize {
        self.supported_max
    }

    /// Rotary pairs: half the rotary width.
    pub fn pairs(&self) -> usize {
        self.rope_dim / 2
    }

    /// The highest reach each rung covers, ascending. One entry, the supported
    /// maximum, for a schedule with a single rung.
    pub fn ceilings(&self) -> Vec<usize> {
        match &self.scaling {
            Scaling::Yarn { rungs, .. } => rungs.iter().map(|r| r.ceiling).collect(),
            _ => vec![self.supported_max],
        }
    }

    /// Every rung's frequencies, in rung order.
    ///
    /// A rung of factor 1 is the plain frequencies exactly as every table has
    /// computed them, so rung 1 of a progressive schedule rotates with the very
    /// frequencies the model ran before the schedule existed.
    pub fn rungs(&self) -> Vec<RungFreqs> {
        let plain = || plain_inv_freq(self.rope_dim, self.theta);
        let unit = |inv_freq| RungFreqs {
            inv_freq,
            q_rot_scale: 1.0,
        };
        match &self.scaling {
            Scaling::Plain => vec![unit(plain())],
            Scaling::Linear { factor } => vec![unit(plain().iter().map(|f| f / factor).collect())],
            Scaling::Stated { inv_freq } => vec![unit(inv_freq.clone())],
            Scaling::Yarn {
                l0,
                beta_fast,
                beta_slow,
                rungs,
                temperature,
            } => rungs
                .iter()
                .map(|r| {
                    if r.factor == 1.0 {
                        return unit(plain());
                    }
                    let inv_freq = yarn_freqs(
                        self.rope_dim,
                        self.theta as f64,
                        *l0,
                        r.factor as f64,
                        *beta_fast,
                        *beta_slow,
                    )
                    .into_iter()
                    .map(|f| f as f32)
                    .collect();
                    let m = if *temperature {
                        mscale(r.factor as f64)
                    } else {
                        1.0
                    };
                    RungFreqs {
                        inv_freq,
                        q_rot_scale: (m * m) as f32,
                    }
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hybrid() -> RopeSchedule {
        RopeSchedule::yarn(
            64,
            1e7,
            262_144,
            vec![
                Rung {
                    ceiling: 262_144,
                    factor: 1.0,
                },
                Rung {
                    ceiling: 524_288,
                    factor: 2.0,
                },
                Rung {
                    ceiling: 1_010_000,
                    factor: 4.0,
                },
            ],
            true,
        )
        .unwrap()
    }

    /// Rung 1 of a progressive schedule is the plain frequencies bit for bit,
    /// with no temperature — the trained model exactly.
    #[test]
    fn rung_one_is_the_trained_rope() {
        let r = hybrid().rungs();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].inv_freq, plain_inv_freq(64, 1e7));
        assert_eq!(r[0].q_rot_scale, 1.0);
    }

    /// Higher rungs carry the YaRN frequencies and `m²`.
    #[test]
    fn higher_rungs_carry_yarn_and_the_temperature() {
        let r = hybrid().rungs();
        let m2 = mscale(2.0);
        assert_eq!(r[1].q_rot_scale, (m2 * m2) as f32);
        let m4 = mscale(4.0);
        assert_eq!(r[2].q_rot_scale, (m4 * m4) as f32);
        let want: Vec<f32> = yarn_freqs(64, 1e7, 262_144, 4.0, 32.0, 1.0)
            .into_iter()
            .map(|f| f as f32)
            .collect();
        assert_eq!(r[2].inv_freq, want);
        // The slowest pair is interpolated: a quarter of its raw frequency.
        assert_eq!(
            r[2].inv_freq[31],
            (1.0 / 1e7f64.powf(62.0 / 64.0) / 4.0) as f32
        );
    }

    /// A static schedule without temperature (DeepSeek-V4) scales no Q.
    #[test]
    fn a_static_yarn_without_temperature_scales_nothing() {
        let s = RopeSchedule::yarn(
            64,
            160_000.0,
            65_536,
            vec![Rung {
                ceiling: 1_048_576,
                factor: 16.0,
            }],
            false,
        )
        .unwrap();
        let r = s.rungs();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].q_rot_scale, 1.0);
        assert_eq!(s.supported_max(), 1_048_576);
    }

    /// Malformed schedules are refused at construction.
    #[test]
    fn malformed_schedules_are_refused() {
        let r = |c, f| Rung {
            ceiling: c,
            factor: f,
        };
        assert!(RopeSchedule::yarn(64, 1e7, 100, vec![], true).is_err());
        assert!(RopeSchedule::yarn(64, 1e7, 100, vec![r(100, 1.0), r(100, 2.0)], true).is_err());
        assert!(RopeSchedule::yarn(64, 1e7, 100, vec![r(100, 2.0), r(200, 4.0)], true).is_err());
        assert!(RopeSchedule::yarn(64, 1e7, 100, vec![r(90, 1.0), r(200, 2.0)], true).is_err());
        assert!(RopeSchedule::stated(vec![], 131_072).is_err());
    }

    /// Linear divides every frequency; plain and stated pass theirs through.
    #[test]
    fn single_rung_schedules() {
        let plain = plain_inv_freq(128, 1e6);
        let lin = RopeSchedule::linear(128, 1e6, 2.0, 65_536).rungs();
        assert_eq!(lin.len(), 1);
        assert_eq!(
            lin[0].inv_freq,
            plain.iter().map(|f| f / 2.0).collect::<Vec<_>>()
        );
        assert_eq!(
            RopeSchedule::plain(128, 1e6, 32_768).rungs()[0].inv_freq,
            plain
        );
        assert_eq!(
            RopeSchedule::plain(128, 1e6, 32_768).ceilings(),
            vec![32_768]
        );
        let stated = RopeSchedule::stated(vec![1.0, 0.5], 8_192).unwrap();
        assert_eq!(stated.pairs(), 2);
        assert_eq!(stated.rungs()[0].inv_freq, vec![1.0, 0.5]);
        assert_eq!(stated.supported_max(), 8_192);
    }
}
