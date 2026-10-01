//! The RoPE a GGUF states: its base and its declared scaling.

use candle::quantized::gguf_file::Value;
use candle_transformers::models::rope_schedule::DeclaredScaling;
use std::collections::HashMap;

/// A GGUF's RoPE base θ — `None` when the file states none — and the scaling it
/// declares, both under the file's own `general.architecture`.
///
/// θ is read under the keys the GQA loaders read it from, in their order:
/// `{arch}.rope.freq_base`, `{arch}.rope.theta`, then the unprefixed two.
pub(super) fn gguf_rope(
    metadata: &HashMap<String, Value>,
) -> candle::Result<(Option<f32>, DeclaredScaling)> {
    let arch = metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .unwrap_or_default();
    let theta = [
        format!("{arch}.rope.freq_base"),
        format!("{arch}.rope.theta"),
        "rope.freq_base".to_string(),
        "rope.theta".to_string(),
    ]
    .iter()
    .find_map(|k| metadata.get(k).and_then(|v| v.to_f32().ok()));
    Ok((theta, DeclaredScaling::from_gguf(metadata, &arch)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn md(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    /// The arch-prefixed key is read under the file's own arch string, and wins
    /// over the unprefixed one.
    #[test]
    fn the_arch_key_comes_first() {
        let m = md(&[
            ("general.architecture", Value::String("qwen3".into())),
            ("rope.freq_base", Value::F32(10_000.0)),
            ("qwen3.rope.freq_base", Value::F32(1_000_000.0)),
        ]);
        assert_eq!(
            gguf_rope(&m).unwrap(),
            (Some(1_000_000.0), DeclaredScaling::None)
        );
    }

    /// Past the arch keys the unprefixed ones answer, `freq_base` before `theta`.
    #[test]
    fn the_unprefixed_keys_follow() {
        let m = md(&[
            ("general.architecture", Value::String("llama".into())),
            ("rope.theta", Value::F32(20_000.0)),
            ("rope.freq_base", Value::F32(500_000.0)),
        ]);
        assert_eq!(gguf_rope(&m).unwrap().0, Some(500_000.0));
    }

    /// A file that states no base says so, rather than taking a default.
    #[test]
    fn no_key_is_none() {
        let m = md(&[("general.architecture", Value::String("qwen2".into()))]);
        assert_eq!(gguf_rope(&m).unwrap().0, None);
    }

    /// The declaration is read under the file's arch.
    #[test]
    fn the_declaration_is_the_files() {
        let m = md(&[
            ("general.architecture", Value::String("qwen3".into())),
            ("qwen3.rope.scaling.type", Value::String("yarn".into())),
            ("qwen3.rope.scaling.factor", Value::F32(4.0)),
            (
                "qwen3.rope.scaling.original_context_length",
                Value::U32(32_768),
            ),
        ]);
        assert_eq!(
            gguf_rope(&m).unwrap().1,
            DeclaredScaling::Yarn {
                factor: 4.0,
                original: 32_768
            }
        );
    }
}
