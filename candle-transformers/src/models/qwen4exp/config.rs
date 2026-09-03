//! `qwen4exp` configuration: geometry, the hybrid schedule, and the QSA / PLE /
//! hyper-connection hyperparameters, parsed from the GGUF metadata namespace
//! frozen in `docs/qwen38_flash_next.md` §12.1.
//!
//! The namespace deliberately mirrors `qwen35`'s wherever the architecture
//! does — the SSM keys, the attention geometry, the MoE keys, the rope keys —
//! and this parser leans on that: the shared coercions come from
//! [`super::super::qwen35::config`], and only the keys `qwen4exp` introduces
//! (`attention.compress_ratios`, `attention.indexer.*`, `hyper_connection.*`,
//! `ple.*`, `embedding_length_per_layer_input`) are parsed here.

use candle::quantized::gguf_file::Value;
use candle::Result;
use std::collections::HashMap;

use crate::models::batched_inference::KvLayers;
pub use crate::models::delta_net::{DeltaNetDims, LayerKind};
use crate::models::qwen35::config::{value_to_usize, MoeConfig, Qwen35Config};
use crate::models::qwen4exp::qsa_select::MAX_RATIO;

/// The hyper-connection (Gated Residual) geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HcConfig {
    /// Parallel residual streams (`hyper_connection.count`, 4).
    pub count: usize,
    /// Bottleneck rank of the read gate (`hyper_connection.low_rank`, 320).
    pub low_rank: usize,
}

impl HcConfig {
    /// The wide-residual row width: `count × hidden`.
    pub fn dim(&self, hidden: usize) -> usize {
        self.count * hidden
    }
}

/// The QSA indexer geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerConfig {
    /// Query heads (`attention.indexer.head_count`, 4). One shared key head.
    pub n_heads: usize,
    /// Indexer head width (`attention.indexer.key_length`, 128).
    pub head_dim: usize,
    /// Selection budget in positions (`attention.indexer.top_k`, 2048). The
    /// selected width is `top_k + ratio − 1` — whole blocks plus the tail.
    pub top_k: usize,
}

/// The PLE (per-layer n-gram embedding) configuration. One PLE layer per
/// model — llama.cpp asserts it, and this parser does too.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleConfig {
    /// Decoder index of the PLE layer (`ple.layers[0]`, zero-based on disk).
    pub layer: usize,
    /// N-gram order (`ple.ngram_size`, 3: bigrams and trigrams).
    pub ngram_size: usize,
    /// Hash heads per n-gram order (`ple.heads_per_ngram`, 8).
    pub heads_per_ngram: usize,
    /// Depthwise conv kernel over time (`ple.conv_kernel`, 4). The conv is
    /// dilated by `ngram_size`, so the carried history is
    /// `(conv_kernel − 1) × ngram_size` tokens.
    pub conv_kernel: usize,
    /// Hash-window reset token (`ple.eos_token_id`).
    pub eos_token_id: u32,
    /// Hash multipliers, one per n-gram position (`ple.layer_multipliers`).
    pub multipliers: Vec<u64>,
    /// Per-head row offset into the one stacked table (`ple.head_offsets`).
    pub head_offsets: Vec<u64>,
    /// Per-head vocabulary size (`ple.head_vocab_sizes`) — **not uniform**.
    pub head_vocab_sizes: Vec<u64>,
    /// Values each head contributes (`embedding_length_per_layer_input`, 160).
    pub head_dim: usize,
}

impl PleConfig {
    /// Total hash heads: `(ngram_size − 1) × heads_per_ngram` = 16.
    pub fn n_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Carried conv history in tokens: `(conv_kernel − 1) × ngram_size` = 9.
    pub fn conv_history(&self) -> usize {
        (self.conv_kernel - 1) * self.ngram_size
    }
}

/// Full configuration for Qwen3.8-Flash-Next (`qwen4exp`).
#[derive(Debug, Clone)]
pub struct Qwen4ExpConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// Trunk depth — blocks the sweep runs, **excluding** any draft head.
    pub num_layers: usize,
    // `layer_kinds` describes the trunk alone. A block index that may reach the
    // draft head is `num_layers + num_mtp_layers` wide; the KV side of the same
    // distinction is `session_kv_layers()` / `mtp_kv_layer()` below.
    /// NextN / MTP blocks the checkpoint carries past the trunk, from
    /// `{arch}.nextn_predict_layers`. `0` on a checkpoint with no draft head,
    /// which is every GGUF of this release's own lineage — the head exists
    /// upstream and the conversion dropped it (`docs/qwen38_flash_next.md` §14).
    ///
    /// The head is **block `num_layers`**, loaded through the same tensor names
    /// as any trunk block, so a checkpoint that carries one simply has one more
    /// `blk.N` and this is how the trunk knows to stop before it.
    pub num_mtp_layers: usize,
    pub layer_kinds: Vec<LayerKind>,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub attn_head_dim: usize,
    pub rope_dim: usize,
    pub rope_sections: [usize; 4],
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    pub delta_net: DeltaNetDims,
    pub moe: MoeConfig,
    pub hc: HcConfig,
    pub indexer: IndexerConfig,
    /// Per-layer QSA compression ratio (`attention.compress_ratios`); 0 on
    /// GDN layers and on any attention layer that attends densely.
    pub compress_ratios: Vec<usize>,
    /// The draft head's own QSA ratio, when the checkpoint declares one —
    /// `attention.compress_ratios` entry `num_layers`, the head's block.
    ///
    /// `None` on an array that covers only the trunk, where the head's ratio
    /// has to be inferred instead.
    pub head_compress_ratio: Option<usize>,
    pub ple: PleConfig,
    pub max_position_embeddings: usize,
}

impl Qwen4ExpConfig {
    /// KV layers the session allocates: the trunk's, plus one per draft head.
    ///
    /// The head is full-attention by declaration
    /// (`"mtp": {"layer_types": ["full_attention"]}`), so it attends over the
    /// sequence's history at the sequence's own length exactly as a trunk
    /// attention layer does, and a drafted position that later becomes a real
    /// one must find its keys where the verify wrote them. That needs a KV slot
    /// of its own.
    ///
    /// The head's slot sits **past** every trunk layer, so [`LayerKind`] — which
    /// describes the trunk alone, the shared parser having truncated it to
    /// `num_layers` — cannot name it and the trunk's layer sweep cannot reach
    /// it. See [`Self::mtp_kv_layer`] for why that separation is load-bearing
    /// rather than incidental.
    pub fn kv_layers(&self) -> KvLayers {
        // All **stream**, the head's layer included: `draft::head_wave_pass`
        // runs the head over the same rows at the same positions in the same
        // wave, so its layer always stands at the same length as its siblings.
        // That is what lets every session-wide operation — fork, view, prefix
        // injection, turn sealing, truncation — keep working without knowing
        // the head exists, and it is why the wave's own decode metadata covers
        // the head's slot headers rather than the head building its own.
        KvLayers::stream_only(self.n_attention_layers() + self.num_mtp_layers)
    }

    /// The KV layer a draft head writes, or `None` on a checkpoint without one.
    /// Always the last, which is what keeps it out of the trunk sweep's reach.
    ///
    /// **The trunk must never include this layer in a group it steps.** Layers
    /// stepped together are reconciled to a common block structure — one
    /// position map describes all of them — and a layer that no pass writes
    /// falls permanently behind the ones that do. The repair is silent by
    /// design (a windowed creep prefill produces the same skew legitimately),
    /// so a head that never runs does not error: it re-pads every layer in the
    /// group on every decode step, moving the trunk's own write slices. That is
    /// a wrong answer with no fault and no log — measured here as a smoke
    /// continuation degrading from "Paris. The capital of Germany is Berlin"
    /// into JSON fragments, while the test still passed its substring check.
    pub fn mtp_kv_layer(&self) -> Option<usize> {
        (self.num_mtp_layers > 0).then(|| self.n_attention_layers())
    }

    pub fn n_attention_layers(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::Attention)
            .count()
    }

    /// Read the config out of GGUF metadata under the `qwen4exp` prefix.
    pub fn from_gguf_metadata(md: &HashMap<String, Value>) -> Result<Self> {
        const ARCH: &str = "qwen4exp";
        // The shared half of the namespace parses through the lineage parser:
        // SSM geometry, attention geometry, rope, MoE, vocab, schedule.
        let base = Qwen35Config::from_gguf_metadata(ARCH, md)?;
        let moe = base.moe.ok_or_else(|| {
            candle::Error::Msg("qwen4exp: expert_count missing — every layer is MoE".to_string())
        })?;

        let get = |key: &str| -> Result<&Value> {
            md.get(&format!("{ARCH}.{key}"))
                .ok_or_else(|| candle::Error::Msg(format!("gguf: missing {ARCH}.{key}")))
        };
        let get_usize =
            |key: &str| -> Result<usize> { get(key).and_then(|v| value_to_usize(v, key)) };
        let get_u64_arr = |key: &str| -> Result<Vec<u64>> {
            match get(key)? {
                Value::Array(a) => a
                    .iter()
                    .map(|v| match v {
                        // Hash multipliers use the full u64 range; the
                        // usize coercion would reject anything past i64::MAX.
                        Value::U64(n) => Ok(*n),
                        Value::I64(n) if *n >= 0 => Ok(*n as u64),
                        other => value_to_usize(other, key).map(|n| n as u64),
                    })
                    .collect(),
                other => candle::bail!("gguf: {ARCH}.{key} is not an array: {other:?}"),
            }
        };

        let hc = HcConfig {
            count: get_usize("hyper_connection.count")?,
            low_rank: get_usize("hyper_connection.low_rank")?,
        };
        if hc.count == 0 || hc.low_rank == 0 {
            candle::bail!("qwen4exp: hyper_connection.count / low_rank must be nonzero");
        }

        let indexer = IndexerConfig {
            n_heads: get_usize("attention.indexer.head_count")?,
            head_dim: get_usize("attention.indexer.key_length")?,
            top_k: get_usize("attention.indexer.top_k")?,
        };
        if indexer.n_heads == 0 || indexer.head_dim == 0 || indexer.top_k == 0 {
            candle::bail!("qwen4exp: indexer geometry must be nonzero");
        }
        if base.rope_dim > indexer.head_dim {
            candle::bail!(
                "qwen4exp: rope width {} exceeds the indexer head width {} — the indexer \
                 keys rotate over the same rotary span as attention",
                base.rope_dim,
                indexer.head_dim
            );
        }

        let compress_ratios = match get("attention.compress_ratios")? {
            Value::Array(a) => {
                if a.len() < base.num_layers {
                    candle::bail!(
                        "gguf: {ARCH}.attention.compress_ratios has {} entries for {} layers",
                        a.len(),
                        base.num_layers
                    );
                }
                a.iter()
                    .take(base.num_layers)
                    .map(|v| value_to_usize(v, "compress_ratios"))
                    .collect::<Result<Vec<_>>>()?
            }
            other => candle::bail!("gguf: {ARCH}.attention.compress_ratios: {other:?}"),
        };
        // The draft head is `blk.{num_layers}`, so an array as long as the
        // block count declares the head's own ratio in its last entry. The
        // trunk's ratios are truncated to `num_layers` above and that entry
        // would go with them — leaving the engine to infer the head's ratio
        // from the trunk's last attention layer, which is a guess that reads as
        // fluent proposals and an accept rate that never justifies the head.
        let head_compress_ratio = match get("attention.compress_ratios")? {
            Value::Array(a) if a.len() > base.num_layers => {
                Some(value_to_usize(&a[base.num_layers], "compress_ratios")?)
            }
            _ => None,
        };
        if let Some(r) = head_compress_ratio {
            if r > MAX_RATIO {
                candle::bail!(
                    "qwen4exp: the draft head declares compress ratio {r}, past the \
                     {MAX_RATIO} the QSA entry packing expresses"
                );
            }
        }
        for (li, (&r, &kind)) in compress_ratios.iter().zip(&base.layer_kinds).enumerate() {
            if r > 0 && kind != LayerKind::Attention {
                candle::bail!(
                    "qwen4exp: layer {li} declares compress ratio {r} but is not an \
                     attention layer"
                );
            }
            // The magnitude belongs at the parse boundary beside the layer-kind
            // check. `qsa_select::pack_entry` spends two bits on the cell count,
            // and guards that with a `debug_assert!` — which compiles out, so in
            // release a ratio of 8 packs `(block << 2) | 7`, carrying the `7`
            // out of the cell field and into the block index. The mask then
            // unmasks a block the indexer never selected and leaves the selected
            // one at −inf: wrong attention, no error. The engine's
            // `SelectionTable::new` refuses it, but the CPU oracle has no such
            // gate, so the two halves would disagree silently.
            if r > MAX_RATIO {
                candle::bail!(
                    "qwen4exp: layer {li} declares compress ratio {r}, past the {MAX_RATIO} \
                     the QSA entry packing expresses"
                );
            }
        }

        let ple_layers = match get("ple.layers")? {
            Value::Array(a) if a.len() == 1 => value_to_usize(&a[0], "ple.layers")?,
            Value::Array(a) => candle::bail!(
                "qwen4exp: {} PLE layers declared — exactly one is supported (llama.cpp \
                 asserts the same)",
                a.len()
            ),
            other => candle::bail!("gguf: {ARCH}.ple.layers: {other:?}"),
        };
        if ple_layers >= base.num_layers {
            candle::bail!(
                "qwen4exp: PLE layer {ple_layers} is out of range for {} layers",
                base.num_layers
            );
        }

        let ngram_size = get_usize("ple.ngram_size")?;
        let heads_per_ngram = get_usize("ple.heads_per_ngram")?;
        let n_heads = (ngram_size.saturating_sub(1)) * heads_per_ngram;
        if ngram_size < 2 || n_heads == 0 {
            candle::bail!("qwen4exp: ple.ngram_size {ngram_size} × heads {heads_per_ngram}");
        }
        let multipliers = get_u64_arr("ple.layer_multipliers")?;
        if multipliers.len() < ngram_size {
            candle::bail!(
                "qwen4exp: {} hash multipliers for n-gram size {ngram_size}",
                multipliers.len()
            );
        }
        let head_offsets = get_u64_arr("ple.head_offsets")?;
        let head_vocab_sizes = get_u64_arr("ple.head_vocab_sizes")?;
        if head_offsets.len() < n_heads || head_vocab_sizes.len() < n_heads {
            candle::bail!(
                "qwen4exp: {} offsets / {} vocab sizes for {n_heads} PLE heads",
                head_offsets.len(),
                head_vocab_sizes.len()
            );
        }
        if head_vocab_sizes.iter().take(n_heads).any(|&v| v == 0) {
            candle::bail!("qwen4exp: a PLE head declares an empty vocabulary");
        }

        let ple = PleConfig {
            layer: ple_layers,
            ngram_size,
            heads_per_ngram,
            conv_kernel: get_usize("ple.conv_kernel")?,
            eos_token_id: get_usize("ple.eos_token_id")? as u32,
            multipliers,
            head_offsets: head_offsets[..n_heads].to_vec(),
            head_vocab_sizes: head_vocab_sizes[..n_heads].to_vec(),
            head_dim: get_usize("embedding_length_per_layer_input")?,
        };
        if ple.conv_kernel < 2 {
            candle::bail!(
                "qwen4exp: ple.conv_kernel {} must be at least 2",
                ple.conv_kernel
            );
        }
        if ple.n_heads() * ple.head_dim != base.hidden_size {
            candle::bail!(
                "qwen4exp: {} PLE heads × {} does not tile hidden {}",
                ple.n_heads(),
                ple.head_dim,
                base.hidden_size
            );
        }

        Ok(Self {
            vocab_size: base.vocab_size,
            hidden_size: base.hidden_size,
            num_layers: base.num_layers,
            // Already subtracted from `base.num_layers` by the shared parser —
            // taken here so the engine can find the head at `blk.{num_layers}`.
            num_mtp_layers: base.num_mtp_layers,
            layer_kinds: base.layer_kinds,
            num_attention_heads: base.num_attention_heads,
            num_kv_heads: base.num_kv_heads,
            attn_head_dim: base.attn_head_dim,
            rope_dim: base.rope_dim,
            rope_sections: base.rope_sections,
            rope_theta: base.rope_theta,
            rms_norm_eps: base.rms_norm_eps,
            delta_net: base.delta_net,
            moe,
            hc,
            indexer,
            compress_ratios,
            head_compress_ratio,
            ple,
            max_position_embeddings: base.max_position_embeddings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published checkpoint's exact metadata, in miniature where an array
    /// would be long. Values from `docs/qwen38_flash_next.md` §12.1.
    fn real_metadata() -> HashMap<String, Value> {
        let mut md = HashMap::new();
        let mut put = |key: &str, v: Value| {
            md.insert(format!("qwen4exp.{key}"), v);
        };
        put("block_count", Value::U32(48));
        put("context_length", Value::U32(262_144));
        put("embedding_length", Value::U32(2560));
        put("full_attention_interval", Value::U32(4));
        put("attention.head_count", Value::U32(24));
        put("attention.head_count_kv", Value::U32(2));
        put("attention.key_length", Value::U32(256));
        put("attention.value_length", Value::U32(256));
        put("attention.layer_norm_rms_epsilon", Value::F32(1e-6));
        put(
            "attention.compress_ratios",
            Value::Array(
                (0..48)
                    .map(|i| Value::I32(if (i + 1) % 4 == 0 { 4 } else { 0 }))
                    .collect(),
            ),
        );
        put("attention.indexer.head_count", Value::U32(4));
        put("attention.indexer.key_length", Value::U32(128));
        put("attention.indexer.top_k", Value::U32(2048));
        put("rope.dimension_count", Value::U32(64));
        put(
            "rope.dimension_sections",
            Value::Array(vec![
                Value::I32(11),
                Value::I32(11),
                Value::I32(10),
                Value::I32(0),
            ]),
        );
        put("rope.freq_base", Value::F32(1e7));
        put("ssm.state_size", Value::U32(128));
        put("ssm.group_count", Value::U32(16));
        put("ssm.time_step_rank", Value::U32(48));
        put("ssm.inner_size", Value::U32(6144));
        put("ssm.conv_kernel", Value::U32(4));
        put("expert_count", Value::U32(512));
        put("expert_used_count", Value::U32(10));
        put("expert_feed_forward_length", Value::U32(640));
        put("expert_shared_feed_forward_length", Value::U32(640));
        put("hyper_connection.count", Value::U32(4));
        put("hyper_connection.low_rank", Value::U32(320));
        put("ple.layers", Value::Array(vec![Value::I32(1)]));
        put("ple.ngram_size", Value::U32(3));
        put("ple.heads_per_ngram", Value::U32(8));
        put("ple.conv_kernel", Value::U32(4));
        put("ple.eos_token_id", Value::U32(248_044));
        put(
            "ple.layer_multipliers",
            Value::Array(vec![
                Value::U64(0x9E37_79B9_7F4A_7C15),
                Value::U64(0xC2B2_AE3D_27D4_EB4F),
                Value::U64(0x1656_67B1_9E37_79F9),
            ]),
        );
        put(
            "ple.head_offsets",
            Value::Array((0..16u64).map(|h| Value::U64(h * 20_000_096)).collect()),
        );
        put(
            "ple.head_vocab_sizes",
            Value::Array((0..16).map(|_| Value::U64(20_000_096)).collect()),
        );
        put("embedding_length_per_layer_input", Value::U32(160));
        put("vocab_size", Value::U32(248_320));
        md
    }

    #[test]
    fn the_published_geometry_parses() {
        let cfg = Qwen4ExpConfig::from_gguf_metadata(&real_metadata()).unwrap();
        assert_eq!(cfg.num_layers, 48);
        assert_eq!(cfg.n_attention_layers(), 12);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.attn_head_dim, 256);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.rope_sections, [11, 11, 10, 0]);
        assert_eq!(cfg.delta_net.n_v_heads, 48);
        assert_eq!(cfg.delta_net.n_k_heads, 16);
        assert_eq!(cfg.delta_net.head_dim, 128);
        assert_eq!(cfg.moe.n_experts, 512);
        assert_eq!(cfg.moe.n_experts_used, 10);
        assert_eq!(
            cfg.hc,
            HcConfig {
                count: 4,
                low_rank: 320
            }
        );
        assert_eq!(cfg.hc.dim(cfg.hidden_size), 10240);
        assert_eq!(cfg.indexer.top_k, 2048);
        assert_eq!(cfg.ple.layer, 1);
        assert_eq!(cfg.ple.n_heads(), 16);
        assert_eq!(cfg.ple.conv_history(), 9);
        assert_eq!(cfg.ple.n_heads() * cfg.ple.head_dim, cfg.hidden_size);
        // The schedule: attention on every fourth layer, QSA ratio 4 there.
        assert_eq!(cfg.layer_kinds[3], LayerKind::Attention);
        assert_eq!(cfg.compress_ratios[3], 4);
        assert_eq!(cfg.compress_ratios[0], 0);
    }

    #[test]
    fn a_ratio_on_a_gdn_layer_is_refused() {
        let mut md = real_metadata();
        md.insert(
            "qwen4exp.attention.compress_ratios".into(),
            Value::Array((0..48).map(|_| Value::I32(4)).collect()),
        );
        let err = Qwen4ExpConfig::from_gguf_metadata(&md).unwrap_err();
        assert!(err.to_string().contains("not an attention layer"), "{err}");
    }

    #[test]
    fn two_ple_layers_are_refused() {
        let mut md = real_metadata();
        md.insert(
            "qwen4exp.ple.layers".into(),
            Value::Array(vec![Value::I32(1), Value::I32(5)]),
        );
        let err = Qwen4ExpConfig::from_gguf_metadata(&md).unwrap_err();
        assert!(err.to_string().contains("exactly one"), "{err}");
    }

    #[test]
    fn ple_heads_must_tile_hidden() {
        let mut md = real_metadata();
        md.insert(
            "qwen4exp.embedding_length_per_layer_input".into(),
            Value::U32(128),
        );
        let err = Qwen4ExpConfig::from_gguf_metadata(&md).unwrap_err();
        assert!(err.to_string().contains("does not tile hidden"), "{err}");
    }
}
