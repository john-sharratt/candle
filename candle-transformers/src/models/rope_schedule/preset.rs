//! Where a checkpoint's RoPE schedule comes from (`docs/progressive_yarn.md`
//! §2).
//!
//! The GGUFs these models ship as carry no YaRN keys — YaRN is opt-in
//! upstream — so the published parameters are properties of the model, like its
//! tokenizer revision, and live with it: in the preset for a checkpoint whose
//! architecture spans schedules (Qwen3-30B-A3B's original release is YaRN over
//! 32K; its 2507 release is native to 262K), in the loader for a lineage that
//! has one schedule throughout (Qwen3.5 / 3.6 / 3.8).

use super::declared::DeclaredScaling;
use super::schedule::{RopeSchedule, Rung};

/// A checkpoint's RoPE schedule, as its preset names it.
#[derive(Debug, Clone, PartialEq)]
pub enum RopePreset {
    /// The frequencies the file states — plain, a declared linear factor, or
    /// Llama3 scaling from `rope_freqs.weight` — at every length, up to the
    /// file's `context_length`.
    FileStated,
    /// Progressive YaRN at base `theta` over trained window `l0`: rung 1 is
    /// the file's own RoPE, and each later rung the factor the vendor publishes
    /// for its ceiling. Rungs carry YaRN's temperature, as the Qwen reference
    /// does. `theta` is the release's, which tells it from a sibling release of
    /// the same architecture trained differently (Qwen3-30B-A3B-2507: θ 1e7,
    /// native to 262K).
    ProgressiveYarn {
        theta: f32,
        l0: usize,
        rungs: Vec<Rung>,
    },
    /// The architecture's own schedule, carried by its loader.
    Lineage,
}

impl RopePreset {
    /// Qwen3 dense and the original Qwen3-30B-A3B: θ 1e6, trained at 32,768,
    /// with the factors Qwen publishes for 64K and 128K (the model cards'
    /// `rope_scaling` block).
    pub fn qwen3() -> Self {
        Self::ProgressiveYarn {
            theta: 1e6,
            l0: 32_768,
            rungs: vec![
                Rung {
                    ceiling: 32_768,
                    factor: 1.0,
                },
                Rung {
                    ceiling: 65_536,
                    factor: 2.0,
                },
                Rung {
                    ceiling: 131_072,
                    factor: 4.0,
                },
            ],
        }
    }

    /// The schedule for a GQA checkpoint whose file states `stated_inv` at base
    /// `theta` (`None` when it names none), declares `declared`, and declares
    /// `context_length`.
    ///
    /// File-stated runs the file's own frequencies — a declared YaRN as its
    /// static schedule, temperature included, whose one rung must be what the
    /// loader built. A progressive preset's first rung must be the file's own
    /// frequencies: a file that already declares a scaling cannot also be moved
    /// up rungs computed from the plain ones, and is refused rather than
    /// double-scaled. Only the two schedules that compute frequencies need θ.
    pub fn gqa_schedule(
        &self,
        stated_inv: Vec<f32>,
        theta: Option<f32>,
        declared: DeclaredScaling,
        context_length: usize,
    ) -> candle::Result<RopeSchedule> {
        let theta = || {
            theta.ok_or_else(|| {
                candle::Error::Msg(format!(
                    "rope preset: {self:?} over a file declaring {declared:?} computes its \
                     frequencies from θ, and the file names none"
                ))
            })
        };
        match self {
            Self::FileStated => match declared {
                DeclaredScaling::Yarn { .. } => {
                    let theta = theta()?;
                    let s = declared.schedule(stated_inv.len() * 2, theta, context_length)?;
                    if s.rungs()[0].inv_freq != stated_inv {
                        candle::bail!(
                            "rope preset: the file's declared yarn over θ = {theta} is not the \
                             frequencies its loader built"
                        );
                    }
                    Ok(s)
                }
                DeclaredScaling::None
                | DeclaredScaling::Linear { .. }
                | DeclaredScaling::Llama3 => RopeSchedule::stated(stated_inv, context_length),
            },
            Self::ProgressiveYarn { .. } if declared != DeclaredScaling::None => candle::bail!(
                "rope preset: a progressive schedule over a file that declares {declared:?} \
                 of its own"
            ),
            Self::ProgressiveYarn {
                theta: published,
                l0,
                rungs,
            } => {
                let theta = theta()?;
                if theta != *published {
                    candle::bail!(
                        "rope preset: the file's θ = {theta} is not the θ = {published} these \
                         rungs were published for — another release of the architecture"
                    );
                }
                let s = RopeSchedule::yarn(stated_inv.len() * 2, theta, *l0, rungs.clone(), true)?;
                if s.rungs()[0].inv_freq != stated_inv {
                    candle::bail!(
                        "rope preset: a progressive schedule over a file whose stated \
                         frequencies are not plain θ = {theta} — the file declares a scaling \
                         of its own"
                    );
                }
                Ok(s)
            }
            Self::Lineage => candle::bail!(
                "rope preset: `Lineage` names a loader-carried schedule, and this \
                 architecture carries none"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::rope_schedule::table::plain_inv_freq;

    /// Qwen3's preset over its plain file frequencies is the three published
    /// rungs, rung 1 exactly the file's.
    #[test]
    fn qwen3_over_plain_frequencies() {
        let s = RopePreset::qwen3()
            .gqa_schedule(
                plain_inv_freq(128, 1e6),
                Some(1e6),
                DeclaredScaling::None,
                40_960,
            )
            .unwrap();
        assert_eq!(s.ceilings(), vec![32_768, 65_536, 131_072]);
        assert_eq!(s.supported_max(), 131_072);
        assert_eq!(s.rungs()[0].inv_freq, plain_inv_freq(128, 1e6));
    }

    /// A file that states a scaling of its own cannot also be put on
    /// progressive rungs — whether its frequencies or its declaration say so.
    #[test]
    fn a_scaled_file_is_refused_a_progressive_preset() {
        let scaled: Vec<f32> = plain_inv_freq(128, 1e6).iter().map(|f| f / 2.0).collect();
        assert!(RopePreset::qwen3()
            .gqa_schedule(scaled, Some(1e6), DeclaredScaling::None, 40_960)
            .is_err());
        let declared = DeclaredScaling::Linear { factor: 2.0 };
        let inv = declared.inv_freq(128, 1e6).unwrap();
        assert!(RopePreset::qwen3()
            .gqa_schedule(inv, Some(1e6), declared, 40_960)
            .is_err());
    }

    /// File-stated is one rung to the file's declared window, needing no θ;
    /// `Lineage` is not a GQA schedule.
    #[test]
    fn file_stated_and_lineage() {
        let s = RopePreset::FileStated
            .gqa_schedule(vec![1.0, 0.5], None, DeclaredScaling::None, 32_768)
            .unwrap();
        assert_eq!(s.ceilings(), vec![32_768]);
        assert_eq!(s.rungs()[0].q_rot_scale, 1.0);
        assert!(RopePreset::Lineage
            .gqa_schedule(vec![1.0], Some(1e4), DeclaredScaling::None, 8)
            .is_err());
    }

    /// A schedule that computes frequencies refuses a file with no θ, and a
    /// progressive preset refuses a file whose θ is not its release's — the
    /// 2507 refresh of Qwen3-30B-A3B, native to 262K at θ 1e7, under the
    /// original's 32K rungs.
    #[test]
    fn a_computed_schedule_needs_the_release_theta() {
        assert!(RopePreset::qwen3()
            .gqa_schedule(
                plain_inv_freq(128, 1e6),
                None,
                DeclaredScaling::None,
                40_960
            )
            .is_err());
        assert!(RopePreset::qwen3()
            .gqa_schedule(
                plain_inv_freq(128, 1e7),
                Some(1e7),
                DeclaredScaling::None,
                262_144
            )
            .is_err());
    }

    /// A file declaring YaRN runs it statically, temperature included, and
    /// its loader's frequencies must be that schedule's.
    #[test]
    fn a_declared_yarn_file_runs_static_yarn() {
        let declared = DeclaredScaling::Yarn {
            factor: 4.0,
            original: 32_768,
        };
        let inv = declared.inv_freq(128, 1e6).unwrap();
        let s = RopePreset::FileStated
            .gqa_schedule(inv.clone(), Some(1e6), declared, 131_072)
            .unwrap();
        assert_eq!(s.ceilings(), vec![131_072]);
        assert!(s.rungs()[0].q_rot_scale > 1.0);
        assert!(RopePreset::FileStated
            .gqa_schedule(plain_inv_freq(128, 1e6), Some(1e6), declared, 131_072)
            .is_err());
    }
}
