//! The RoPE scaling a GQA checkpoint's metadata declares
//! (`docs/progressive_yarn.md` §2, "GGUF-declared schedules").
//!
//! A declared schedule is taken at its word, and only a declared one: a
//! `context_length` past the trained window is not a scaling (§3.1).
//!
//! - `yarn` → a **static** YaRN schedule at the declared factor over the
//!   declared original window, with its temperature — one rung, at every length;
//! - `linear` → every frequency divided by the factor;
//! - nothing declared → plain RoPE.
//!
//! Llama 3 scaling is not here: llama.cpp writes it as `rope_freqs.weight`,
//! which the Llama loader reads (`llama3.rs`).

use std::collections::HashMap;

use candle::quantized::gguf_file::Value;
use candle::Result;

use super::schedule::{RopeSchedule, Rung};
use super::table::plain_inv_freq;

/// What a file declares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeclaredScaling {
    /// No scaling: plain RoPE.
    None,
    /// Every frequency divided by `factor`.
    Linear { factor: f32 },
    /// YaRN at `factor` over the trained window `original`.
    Yarn { factor: f32, original: usize },
    /// Llama 3 scaling, whose frequencies the Llama loader builds from the
    /// file (`llama3.rs`); stated as the loader built them.
    Llama3,
}

impl DeclaredScaling {
    /// Read from GGUF metadata, under `{arch}.rope.scaling.*` and the
    /// unprefixed keys some converters write.
    ///
    /// A factor with no type is linear, as exporters that write no type apply
    /// it. A factor of 1 (or none) is no scaling. A YaRN declaration without
    /// its original window is refused: the transform is undefined without it.
    pub fn from_gguf(md: &HashMap<String, Value>, arch: &str) -> Result<Self> {
        let f32_at = |keys: &[String]| {
            keys.iter()
                .find_map(|k| md.get(k).and_then(|v| v.to_f32().ok()))
        };
        let u32_at = |keys: &[String]| {
            keys.iter()
                .find_map(|k| md.get(k).and_then(|v| v.to_u32().ok()))
        };
        let keys = |suffix: &str| [format!("{arch}.rope.{suffix}"), format!("rope.{suffix}")];
        let factor = f32_at(&keys("scaling.factor"))
            .or_else(|| f32_at(&keys("scale_factor")))
            .filter(|f| *f > 0.0 && *f != 1.0);
        let Some(factor) = factor else {
            return Ok(Self::None);
        };
        let kind = keys("scaling.type")
            .iter()
            .find_map(|k| md.get(k).and_then(|v| v.to_string().ok()).cloned());
        match kind.as_deref() {
            None | Some("linear") => Ok(Self::Linear { factor }),
            Some("yarn") => {
                let original =
                    u32_at(&keys("scaling.original_context_length")).ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "{arch}: a declared yarn scaling (factor {factor}) with no \
                         original_context_length"
                        ))
                    })? as usize;
                Ok(Self::Yarn { factor, original })
            }
            Some("none") => Ok(Self::None),
            Some("llama3") => Ok(Self::Llama3),
            Some(other) => candle::bail!(
                "{arch}: a declared rope scaling of type {other:?}, which this loader cannot run"
            ),
        }
    }

    /// The frequencies the file states over `rope_dim` at base `theta` — the
    /// schedule's one rung, which the loader's own rotation uses too.
    pub fn inv_freq(&self, rope_dim: usize, theta: f32) -> Result<Vec<f32>> {
        Ok(match *self {
            Self::None => plain_inv_freq(rope_dim, theta),
            // `1 / (s · θ^(2i/d))`: the form every GGUF exporter applies a
            // linear factor in.
            Self::Linear { factor } => (0..rope_dim / 2)
                .map(|i| 1f32 / (factor * theta.powf((2 * i) as f32 / rope_dim as f32)))
                .collect(),
            Self::Yarn { .. } => self.schedule(rope_dim, theta, usize::MAX)?.rungs()[0]
                .inv_freq
                .clone(),
            Self::Llama3 => {
                candle::bail!("Llama 3 frequencies are the Llama loader's, from the file's own")
            }
        })
    }

    /// The schedule a file of this declaration runs, to `context_length`.
    pub fn schedule(
        &self,
        rope_dim: usize,
        theta: f32,
        context_length: usize,
    ) -> Result<RopeSchedule> {
        match *self {
            Self::None => Ok(RopeSchedule::plain(rope_dim, theta, context_length)),
            Self::Linear { factor } => Ok(RopeSchedule::linear(
                rope_dim,
                theta,
                factor,
                context_length,
            )),
            Self::Yarn { factor, original } => RopeSchedule::yarn(
                rope_dim,
                theta,
                original,
                vec![Rung {
                    ceiling: context_length,
                    factor,
                }],
                true,
            ),
            Self::Llama3 => {
                candle::bail!("a Llama 3 schedule is stated from the Llama loader's frequencies")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::rope_schedule::yarn::{mscale, yarn_freqs};

    fn md(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    /// Nothing declared, or a factor of one, is plain RoPE — whatever the
    /// `context_length`.
    #[test]
    fn nothing_declared_is_plain() {
        let m = md(&[("qwen3.context_length", Value::U32(40_960))]);
        assert_eq!(
            DeclaredScaling::from_gguf(&m, "qwen3").unwrap(),
            DeclaredScaling::None
        );
        let one = md(&[("qwen3.rope.scaling.factor", Value::F32(1.0))]);
        assert_eq!(
            DeclaredScaling::from_gguf(&one, "qwen3").unwrap(),
            DeclaredScaling::None
        );
        assert_eq!(
            DeclaredScaling::None.inv_freq(128, 1e6).unwrap(),
            plain_inv_freq(128, 1e6)
        );
    }

    /// A typeless or `linear` factor divides every frequency, in the exporters'
    /// form.
    #[test]
    fn a_linear_factor_divides_every_frequency() {
        for m in [
            md(&[("qwen2.rope.scaling.factor", Value::F32(4.0))]),
            md(&[
                ("rope.scale_factor", Value::F32(4.0)),
                ("rope.scaling.type", Value::String("linear".into())),
            ]),
        ] {
            let d = DeclaredScaling::from_gguf(&m, "qwen2").unwrap();
            assert_eq!(d, DeclaredScaling::Linear { factor: 4.0 });
            let inv = d.inv_freq(64, 1e6).unwrap();
            for (i, f) in inv.iter().enumerate() {
                assert_eq!(*f, 1f32 / (4.0 * 1e6f32.powf((2 * i) as f32 / 64.0)));
            }
        }
    }

    /// A declared `yarn` is a static YaRN schedule — one rung to the file's
    /// window, at the declared factor over the declared original window, with
    /// the temperature on Q — and its frequencies are that rung's.
    #[test]
    fn a_declared_yarn_is_static_yarn() {
        let m = md(&[
            ("qwen3.rope.scaling.type", Value::String("yarn".into())),
            ("qwen3.rope.scaling.factor", Value::F32(4.0)),
            (
                "qwen3.rope.scaling.original_context_length",
                Value::U32(32_768),
            ),
        ]);
        let d = DeclaredScaling::from_gguf(&m, "qwen3").unwrap();
        assert_eq!(
            d,
            DeclaredScaling::Yarn {
                factor: 4.0,
                original: 32_768
            }
        );
        let s = d.schedule(128, 1e6, 131_072).unwrap();
        assert_eq!(s.ceilings(), vec![131_072]);
        let r = s.rungs();
        assert_eq!(r.len(), 1);
        let want: Vec<f32> = yarn_freqs(128, 1e6f32 as f64, 32_768, 4.0, 32.0, 1.0)
            .into_iter()
            .map(|f| f as f32)
            .collect();
        assert_eq!(r[0].inv_freq, want);
        assert_eq!(r[0].q_rot_scale, (mscale(4.0) * mscale(4.0)) as f32);
        assert_eq!(d.inv_freq(128, 1e6).unwrap(), want);
    }

    /// A Llama 3 declaration is the Llama loader's to build, and says so.
    #[test]
    fn a_llama3_declaration_is_the_loaders() {
        let m = md(&[
            ("llama.rope.scaling.type", Value::String("llama3".into())),
            ("llama.rope.scaling.factor", Value::F32(32.0)),
        ]);
        let d = DeclaredScaling::from_gguf(&m, "llama").unwrap();
        assert_eq!(d, DeclaredScaling::Llama3);
        assert!(d.inv_freq(128, 5e5).is_err());
    }

    /// A YaRN declaration missing its window, and a type this loader does not
    /// run, are refused rather than guessed at.
    #[test]
    fn incomplete_or_unknown_declarations_are_refused() {
        let no_window = md(&[
            ("qwen3.rope.scaling.type", Value::String("yarn".into())),
            ("qwen3.rope.scaling.factor", Value::F32(4.0)),
        ]);
        assert!(DeclaredScaling::from_gguf(&no_window, "qwen3").is_err());
        let unknown = md(&[
            ("qwen3.rope.scaling.type", Value::String("longrope".into())),
            ("qwen3.rope.scaling.factor", Value::F32(4.0)),
        ]);
        assert!(DeclaredScaling::from_gguf(&unknown, "qwen3").is_err());
    }
}
