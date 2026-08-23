//! Qwen3.5 configuration: dimensions, the hybrid layer schedule, and the
//! GGUF metadata mapping.
//!
//! The whole Qwen3.5/3.8 generation is a hybrid stack: blocks of Gated
//! DeltaNet (linear-attention, recurrent-state) layers punctuated by full
//! gated-attention layers, one in every `full_attention_interval`. The GGUF
//! carries the schedule either explicitly (`{arch}.attention.recurrent_layers`,
//! a per-layer flag array) or implicitly via `{arch}.full_attention_interval`
//! (default 4: layers where `(i + 1) % 4 == 0` are full attention).
//!
//! The DeltaNet geometry rides the GGUF's SSM metadata namespace, with the
//! llama.cpp `qwen35` arch as the naming authority (see
//! `docs/qwen35_qwen38_models.md` §7.1):
//!
//! - `ssm.state_size`      → the QK/V head width (`head_k_dim == head_v_dim`)
//! - `ssm.group_count`     → number of QK heads
//! - `ssm.time_step_rank`  → number of V heads
//! - `ssm.inner_size`      → `head_v_dim × n_v_heads` (the value width)
//! - `ssm.conv_kernel`     → causal-conv kernel size over the fused QKV channels

use candle::quantized::gguf_file::Value;
use candle::Result;
use std::collections::HashMap;

// Re-exported: the config's own fields are typed by them, so this module is
// where the lineage's code (and its tests) naturally reach for them.
pub use crate::models::delta_net::{DeltaNetDims, LayerKind};

/// Full configuration for a Qwen3.5 / Qwen3.8 model (dense or MoE).
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Dense-FFN intermediate size (dense variants and any dense layers).
    pub intermediate_size: usize,
    /// Trunk layers (excludes MTP blocks).
    pub num_layers: usize,
    /// Per-trunk-layer kinds, `num_layers` long.
    pub layer_kinds: Vec<LayerKind>,
    /// Full-attention geometry.
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    /// Width of one attention head (`attention.key_length`).
    pub attn_head_dim: usize,
    /// Rotary width (`rope.dimension_count`, ggml's `n_rot`). This family
    /// uses a partial rotary factor of 0.25 — 64 of the 256 head dims
    /// rotate, the remaining 192 pass through untouched. A missing key means
    /// classic full-width RoPE (`rope_dim == attn_head_dim`).
    pub rope_dim: usize,
    /// MRoPE section split (`rope.dimension_sections`, 4 entries), in pair
    /// counts per axis, summing to `rope_dim / 2`. All-text models still
    /// declare it; a missing key is stored as `[rope_dim / 2, 0, 0, 0]`.
    pub rope_sections: [usize; 4],
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    /// Gated DeltaNet geometry (uniform across DeltaNet layers).
    pub delta_net: DeltaNetDims,
    /// MoE geometry; `None` on dense variants.
    pub moe: Option<MoeConfig>,
    /// Trailing MTP (NextN) blocks present in the checkpoint. They are loaded
    /// only by the drafter; the trunk forward ignores them.
    pub num_mtp_layers: usize,
    pub max_position_embeddings: usize,
}

/// MoE geometry for the `qwen35moe` variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeConfig {
    pub n_experts: usize,
    pub n_experts_used: usize,
    /// Per-expert FFN intermediate width (`expert_feed_forward_length`).
    pub expert_ffn_size: usize,
    /// Shared-expert FFN intermediate width (`expert_shared_feed_forward_length`,
    /// falling back to `expert_ffn_size`).
    pub shared_expert_ffn_size: usize,
    /// Whether top-k routing weights are renormalized to sum to 1
    /// (`expert_weights_norm`; Qwen3.5 uses it like Qwen3-MoE).
    pub norm_topk_prob: bool,
}

impl Qwen35Config {
    /// The layer schedule from an explicit per-layer flag array, or derived
    /// from the interval rule: `(i + 1) % interval != 0` ⇒ DeltaNet.
    pub fn schedule_from_interval(num_layers: usize, interval: usize) -> Vec<LayerKind> {
        (0..num_layers)
            .map(|i| {
                if (i + 1) % interval == 0 {
                    LayerKind::Attention
                } else {
                    LayerKind::DeltaNet
                }
            })
            .collect()
    }

    pub fn n_attention_layers(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::Attention)
            .count()
    }

    pub fn n_delta_net_layers(&self) -> usize {
        self.num_layers - self.n_attention_layers()
    }

    /// Read the config out of GGUF metadata under `arch` (`qwen35` or
    /// `qwen35moe`). `arch` is detected by the caller from
    /// `general.architecture` (with tensor-presence probing as fallback,
    /// matching `quantized_qwen3_moe`'s loader).
    pub fn from_gguf_metadata(arch: &str, md: &HashMap<String, Value>) -> Result<Self> {
        let get = |key: &str| -> Result<&Value> {
            md.get(&format!("{arch}.{key}"))
                .ok_or_else(|| candle::Error::Msg(format!("gguf: missing {arch}.{key}")))
        };
        let get_usize = |key: &str| -> Result<usize> {
            get(key).and_then(|v| value_to_usize(v, key))
        };
        let opt_usize = |key: &str| -> Option<usize> {
            md.get(&format!("{arch}.{key}"))
                .and_then(|v| value_to_usize(v, key).ok())
        };

        let num_layers_all = get_usize("block_count")?;
        let num_mtp_layers = opt_usize("nextn_predict_layers").unwrap_or(0);
        if num_mtp_layers >= num_layers_all {
            candle::bail!(
                "gguf: {arch}.nextn_predict_layers {num_mtp_layers} must be smaller than \
                 block_count {num_layers_all}"
            );
        }
        let num_layers = num_layers_all - num_mtp_layers;

        let hidden_size = get_usize("embedding_length")?;
        let num_attention_heads = get_usize("attention.head_count")?;
        let num_kv_heads = get_usize("attention.head_count_kv")?;
        let attn_head_dim = opt_usize("attention.key_length")
            .unwrap_or(hidden_size / num_attention_heads);

        // The layer schedule: an explicit per-layer recurrent flag array wins;
        // otherwise the interval rule (default 4).
        let layer_kinds = match md.get(&format!("{arch}.attention.recurrent_layers")) {
            Some(Value::Array(flags)) => {
                if flags.len() < num_layers {
                    candle::bail!(
                        "gguf: {arch}.attention.recurrent_layers has {} entries for {} layers",
                        flags.len(),
                        num_layers
                    );
                }
                flags
                    .iter()
                    .take(num_layers)
                    .map(|v| {
                        Ok(if value_to_usize(v, "recurrent_layers")? != 0 {
                            LayerKind::DeltaNet
                        } else {
                            LayerKind::Attention
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            _ => {
                let interval = opt_usize("full_attention_interval").unwrap_or(4);
                if interval == 0 {
                    candle::bail!("gguf: {arch}.full_attention_interval must be nonzero");
                }
                Self::schedule_from_interval(num_layers, interval)
            }
        };

        // DeltaNet geometry from the SSM namespace.
        let state_size = get_usize("ssm.state_size")?;
        let n_k_heads = get_usize("ssm.group_count")?;
        let n_v_heads = get_usize("ssm.time_step_rank")?;
        let inner_size = get_usize("ssm.inner_size")?;
        let conv_kernel = get_usize("ssm.conv_kernel")?;
        if inner_size != state_size * n_v_heads {
            candle::bail!(
                "gguf: {arch}.ssm.inner_size {inner_size} != state_size {state_size} × \
                 time_step_rank {n_v_heads} — this family has equal K and V head widths"
            );
        }
        if n_k_heads == 0 || n_v_heads % n_k_heads != 0 {
            candle::bail!(
                "gguf: {arch}.ssm.group_count {n_k_heads} must divide time_step_rank {n_v_heads}"
            );
        }
        if conv_kernel < 2 {
            candle::bail!("gguf: {arch}.ssm.conv_kernel {conv_kernel} must be at least 2");
        }
        let delta_net = DeltaNetDims {
            head_dim: state_size,
            n_k_heads,
            n_v_heads,
            conv_kernel,
        };

        // Rotary width; absent means classic RoPE over the full head.
        let rope_dim = opt_usize("rope.dimension_count").unwrap_or(attn_head_dim);
        if rope_dim == 0 || rope_dim % 2 != 0 || rope_dim > attn_head_dim {
            candle::bail!(
                "gguf: {arch}.rope.dimension_count {rope_dim} must be even, nonzero, and at \
                 most attention.key_length {attn_head_dim}"
            );
        }

        // MRoPE sections, in pair counts per axis over the rotary width.
        let rope_sections = match md.get(&format!("{arch}.rope.dimension_sections")) {
            Some(Value::Array(secs)) => {
                let mut out = [0usize; 4];
                if secs.len() > 4 {
                    candle::bail!(
                        "gguf: {arch}.rope.dimension_sections has {} entries, expected at most 4",
                        secs.len()
                    );
                }
                for (i, v) in secs.iter().enumerate() {
                    out[i] = value_to_usize(v, "rope.dimension_sections")?;
                }
                let total: usize = out.iter().sum();
                if total != rope_dim / 2 {
                    candle::bail!(
                        "gguf: {arch}.rope.dimension_sections sum to {total} pairs but \
                         rope.dimension_count {rope_dim} needs {}",
                        rope_dim / 2
                    );
                }
                out
            }
            _ => [rope_dim / 2, 0, 0, 0],
        };

        let rope_theta = md
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| value_to_f32(v).ok())
            .unwrap_or(1_000_000.0);
        let rms_norm_eps = get("attention.layer_norm_rms_epsilon")
            .and_then(value_to_f32)
            .map(|f| f as f64)?;
        let max_position_embeddings = opt_usize("context_length").unwrap_or(262_144);
        let vocab_size = get_usize("vocab_size").or_else(|_| {
            md.get("tokenizer.ggml.tokens")
                .and_then(|v| match v {
                    Value::Array(a) => Some(a.len()),
                    _ => None,
                })
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "gguf: neither {arch}.vocab_size nor tokenizer.ggml.tokens present"
                    ))
                })
        })?;

        // Dense FFN width; MoE variants may omit it when every layer is MoE.
        let n_experts = opt_usize("expert_count").unwrap_or(0);
        let moe = if n_experts > 1 {
            let expert_ffn_size = get_usize("expert_feed_forward_length")?;
            Some(MoeConfig {
                n_experts,
                n_experts_used: get_usize("expert_used_count")?,
                expert_ffn_size,
                shared_expert_ffn_size: opt_usize("expert_shared_feed_forward_length")
                    .unwrap_or(expert_ffn_size),
                norm_topk_prob: md
                    .get(&format!("{arch}.expert_weights_norm"))
                    .and_then(|v| value_to_usize(v, "expert_weights_norm").ok())
                    .map(|n| n != 0)
                    .unwrap_or(true),
            })
        } else {
            None
        };
        let intermediate_size = opt_usize("feed_forward_length").unwrap_or_else(|| {
            moe.map(|m| m.expert_ffn_size).unwrap_or(0)
        });
        if intermediate_size == 0 && moe.is_none() {
            candle::bail!("gguf: {arch}.feed_forward_length missing on a dense model");
        }

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            layer_kinds,
            num_attention_heads,
            num_kv_heads,
            attn_head_dim,
            rope_dim,
            rope_sections,
            rope_theta,
            rms_norm_eps,
            delta_net,
            moe,
            num_mtp_layers,
            max_position_embeddings,
        })
    }
}

fn value_to_usize(v: &Value, key: &str) -> Result<usize> {
    let n: i64 = match v {
        Value::U8(n) => *n as i64,
        Value::I8(n) => *n as i64,
        Value::U16(n) => *n as i64,
        Value::I16(n) => *n as i64,
        Value::U32(n) => *n as i64,
        Value::I32(n) => *n as i64,
        Value::U64(n) => *n as i64,
        Value::I64(n) => *n,
        Value::Bool(b) => *b as i64,
        other => candle::bail!("gguf: {key} has non-integer type {other:?}"),
    };
    if n < 0 {
        candle::bail!("gguf: {key} is negative ({n})");
    }
    Ok(n as usize)
}

fn value_to_f32(v: &Value) -> Result<f32> {
    match v {
        Value::F32(f) => Ok(*f),
        Value::F64(f) => Ok(*f as f32),
        other => match value_to_usize(other, "float key") {
            Ok(n) => Ok(n as f32),
            Err(_) => candle::bail!("gguf: expected float, got {other:?}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dn() -> DeltaNetDims {
        // The Qwen3.5-35B-A3B published geometry: 16 QK heads / 32 V heads
        // at width 128.
        DeltaNetDims {
            head_dim: 128,
            n_k_heads: 16,
            n_v_heads: 32,
            conv_kernel: 4,
        }
    }

    #[test]
    fn delta_net_dims_derive_the_conv_and_state_shapes() {
        let d = dn();
        assert_eq!(d.key_dim(), 2048);
        assert_eq!(d.value_dim(), 4096);
        // Q + K + V channels through the shared causal conv.
        assert_eq!(d.conv_dim(), 2 * 2048 + 4096);
        // One [128 × 128] matrix per V head.
        assert_eq!(d.state_elems(), 32 * 128 * 128);
        // Conv tail: kernel−1 columns per channel.
        assert_eq!(d.conv_state_elems(), 3 * (2 * 2048 + 4096));
    }

    #[test]
    fn interval_schedule_puts_attention_on_every_fourth_layer() {
        let kinds = Qwen35Config::schedule_from_interval(8, 4);
        assert_eq!(
            kinds,
            vec![
                LayerKind::DeltaNet,
                LayerKind::DeltaNet,
                LayerKind::DeltaNet,
                LayerKind::Attention,
                LayerKind::DeltaNet,
                LayerKind::DeltaNet,
                LayerKind::DeltaNet,
                LayerKind::Attention,
            ]
        );
    }

    fn base_metadata(arch: &str) -> HashMap<String, Value> {
        let mut md = HashMap::new();
        let mut put = |key: &str, v: Value| {
            md.insert(format!("{arch}.{key}"), v);
        };
        put("block_count", Value::U32(8));
        put("embedding_length", Value::U32(2048));
        put("attention.head_count", Value::U32(16));
        put("attention.head_count_kv", Value::U32(2));
        put("attention.key_length", Value::U32(256));
        put("attention.layer_norm_rms_epsilon", Value::F32(1e-6));
        put("ssm.state_size", Value::U32(128));
        put("ssm.group_count", Value::U32(16));
        put("ssm.time_step_rank", Value::U32(32));
        put("ssm.inner_size", Value::U32(4096));
        put("ssm.conv_kernel", Value::U32(4));
        put("feed_forward_length", Value::U32(5504));
        put("vocab_size", Value::U32(151_936));
        md
    }

    #[test]
    fn dense_config_parses_with_interval_default() {
        let md = base_metadata("qwen35");
        let cfg = Qwen35Config::from_gguf_metadata("qwen35", &md).unwrap();
        assert_eq!(cfg.num_layers, 8);
        assert_eq!(cfg.n_attention_layers(), 2);
        assert_eq!(cfg.n_delta_net_layers(), 6);
        assert_eq!(cfg.attn_head_dim, 256);
        assert_eq!(cfg.delta_net, dn());
        assert!(cfg.moe.is_none());
        assert_eq!(cfg.num_mtp_layers, 0);
        // No dimension_count / sections keys ⇒ classic RoPE over the full
        // head (256 dims, 128 pairs).
        assert_eq!(cfg.rope_dim, 256);
        assert_eq!(cfg.rope_sections, [128, 0, 0, 0]);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
    }

    /// The published family sets a 0.25 partial rotary factor: 64 of the 256
    /// head dims rotate, and the MRoPE sections tile that rotary width in
    /// pairs (11 + 11 + 10 = 32 = 64/2). These are the exact values in the
    /// Qwen3.5-0.8B GGUF.
    #[test]
    fn partial_rotary_width_and_sections_parse() {
        let mut md = base_metadata("qwen35");
        md.insert("qwen35.rope.dimension_count".into(), Value::U32(64));
        md.insert(
            "qwen35.rope.dimension_sections".into(),
            Value::Array(vec![
                Value::U32(11),
                Value::U32(11),
                Value::U32(10),
                Value::U32(0),
            ]),
        );
        md.insert("qwen35.rope.freq_base".into(), Value::F32(1e7));
        let cfg = Qwen35Config::from_gguf_metadata("qwen35", &md).unwrap();
        assert_eq!(cfg.attn_head_dim, 256);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.rope_sections, [11, 11, 10, 0]);
        assert_eq!(cfg.rope_theta, 1e7);
    }

    #[test]
    fn rotary_width_wider_than_the_head_is_refused() {
        let mut md = base_metadata("qwen35");
        md.insert("qwen35.rope.dimension_count".into(), Value::U32(512));
        let err = Qwen35Config::from_gguf_metadata("qwen35", &md).unwrap_err();
        assert!(err.to_string().contains("dimension_count"), "{err}");
    }

    /// Sections that do not tile the rotary width mean one of the two keys
    /// was misread — refuse rather than rope a wrong number of dims.
    #[test]
    fn sections_that_disagree_with_the_rotary_width_are_refused() {
        let mut md = base_metadata("qwen35");
        md.insert("qwen35.rope.dimension_count".into(), Value::U32(64));
        md.insert(
            "qwen35.rope.dimension_sections".into(),
            Value::Array(vec![Value::U32(11), Value::U32(11), Value::U32(11)]),
        );
        let err = Qwen35Config::from_gguf_metadata("qwen35", &md).unwrap_err();
        assert!(err.to_string().contains("dimension_sections"), "{err}");
    }

    #[test]
    fn explicit_recurrent_flags_override_the_interval() {
        let mut md = base_metadata("qwen35");
        md.insert(
            "qwen35.attention.recurrent_layers".into(),
            Value::Array(vec![
                Value::U32(0),
                Value::U32(1),
                Value::U32(1),
                Value::U32(1),
                Value::U32(0),
                Value::U32(1),
                Value::U32(1),
                Value::U32(1),
            ]),
        );
        let cfg = Qwen35Config::from_gguf_metadata("qwen35", &md).unwrap();
        assert_eq!(cfg.layer_kinds[0], LayerKind::Attention);
        assert_eq!(cfg.layer_kinds[1], LayerKind::DeltaNet);
        assert_eq!(cfg.layer_kinds[4], LayerKind::Attention);
    }

    #[test]
    fn mtp_layers_are_split_off_the_trunk() {
        let mut md = base_metadata("qwen35");
        md.insert("qwen35.block_count".into(), Value::U32(9));
        md.insert("qwen35.nextn_predict_layers".into(), Value::U32(1));
        let cfg = Qwen35Config::from_gguf_metadata("qwen35", &md).unwrap();
        assert_eq!(cfg.num_layers, 8);
        assert_eq!(cfg.num_mtp_layers, 1);
        assert_eq!(cfg.layer_kinds.len(), 8);
    }

    #[test]
    fn moe_config_parses_with_shared_expert_fallback() {
        let mut md = base_metadata("qwen35moe");
        // Re-key everything the base helper wrote under qwen35moe already;
        // add the MoE keys.
        md.insert("qwen35moe.expert_count".into(), Value::U32(256));
        md.insert("qwen35moe.expert_used_count".into(), Value::U32(8));
        md.insert(
            "qwen35moe.expert_feed_forward_length".into(),
            Value::U32(512),
        );
        md.remove("qwen35moe.feed_forward_length");
        let cfg = Qwen35Config::from_gguf_metadata("qwen35moe", &md).unwrap();
        let moe = cfg.moe.unwrap();
        assert_eq!(moe.n_experts, 256);
        assert_eq!(moe.n_experts_used, 8);
        assert_eq!(moe.expert_ffn_size, 512);
        assert_eq!(moe.shared_expert_ffn_size, 512);
        assert!(moe.norm_topk_prob);
    }

    #[test]
    fn inner_size_mismatch_is_refused() {
        let mut md = base_metadata("qwen35");
        md.insert("qwen35.ssm.inner_size".into(), Value::U32(4095));
        let err = Qwen35Config::from_gguf_metadata("qwen35", &md).unwrap_err();
        assert!(err.to_string().contains("inner_size"));
    }
}
