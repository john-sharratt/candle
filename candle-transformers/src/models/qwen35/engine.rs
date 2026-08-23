//! How a hybrid stack describes itself to the scheduler.
//!
//! Three of the numbers the engine asks a model for mean something different
//! on a hybrid than they do on a uniform transformer, and each one is wrong
//! in a way that costs memory or correctness if it is simply inherited:
//!
//! * the **session layer count** is KV layers, not transformer depth
//!   (8 vs 32 on the 9B) — see [`super::kv_layout`];
//! * the **priced intermediate width** must cover the DeltaNet projections,
//!   which are not an FFN and can be wider than one;
//! * the **provenance capture depths** must land on layers that have a Q to
//!   capture, and three quarters of this stack does not.
//!
//! These are computed from the config so they can be tested without a
//! loaded checkpoint, and the model methods delegate to them.

use candle::DType;
use candle_nn::kv_cache::ModelGeometry;

use super::config::Qwen35Config;
use super::quantized_weights::QuantModel;
use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ProvenanceLayerIndices,
};
use crate::models::delta_net::KvLayerMap;

/// The widest per-row intermediate activation a layer of this model produces.
///
/// The wave plan sizes its FFN span from one "intermediate" width, but a
/// hybrid has two kinds of layer with unrelated widths: a dense FFN carries
/// `intermediate` per row, while a DeltaNet layer carries its fused
/// `[Q|K|V]` projection, `2·key_dim + value_dim`. The span has to hold
/// whichever is larger, or the wave is admitted wider than its transients
/// fit. On the 9B the FFN wins (12288 against 8192); on a stack with a
/// narrow FFN and wide heads it would not, and pricing on the FFN alone
/// would silently under-reserve.
pub fn priced_intermediate(cfg: &Qwen35Config) -> usize {
    let ffn = match &cfg.moe {
        // MoE prices the *per-expert* intermediate; `expert_rows` applies the
        // fan-out separately.
        Some(moe) => moe.expert_ffn_size.max(moe.shared_expert_ffn_size),
        None => cfg.intermediate_size,
    };
    ffn.max(cfg.delta_net.conv_dim())
}

/// The geometry admission prices a wave from.
pub fn wave_geometry(cfg: &Qwen35Config, act_dtype: DType) -> ModelGeometry {
    let (experts_per_tok, n_experts) = match &cfg.moe {
        Some(moe) => (moe.n_experts_used.max(1), moe.n_experts.max(1)),
        None => (1, 1),
    };
    ModelGeometry {
        hidden: cfg.hidden_size,
        intermediate: priced_intermediate(cfg),
        n_head: cfg.num_attention_heads,
        n_kv_head: cfg.num_kv_heads,
        head_dim: cfg.attn_head_dim,
        experts_per_tok,
        n_experts,
        act_dtype,
        // The int8 kernels accumulate in F32 before the cast back to
        // `act_dtype`; both buffers are live at once, so both are planned.
        accum_dtype: DType::F32,
        // This stack's Q/K/V projections run the dequantized GEMM in F32: the
        // norm's `act_dtype` output is upcast, the matmul emits F32, and the
        // result is cast back. Read off `KV_WAVE_CENSUS=labels` on the 0.8B,
        // where the six round-trip buffers are 43 MB of a 96 MB attention
        // generation — the whole gap between the priced span and the carved
        // one, which the region pad was absorbing until the attention phase
        // gained one more buffer.
        projection_accum_roundtrip: true,
        // [q|gate] interleaved projection + 64-of-256 partial rotary — the
        // gate's downstream buffers and the two permute gathers are real
        // carves on every attention layer of this lineage.
        gated_qkv: true,
        partial_rotary: true,
    }
}

/// Provenance capture depths, moved onto layers that actually attend.
///
/// The scheduler's default picks three depth fractions of the stack and a
/// four-layer window below each. On a 3:1 hybrid those land on a DeltaNet
/// layer three times out of four, where there is no Q to capture. Each band
/// is snapped down to an attention layer, and its lower endpoint becomes the
/// *previous* attention layer — a real two-point window over the layers that
/// have signatures, rather than a nominal four-layer gap that mostly spans
/// layers which contribute nothing.
///
/// Returns `None` for a stack with no attention layers at all, which has no
/// provenance to capture and must be refused by the caller rather than
/// handed indices that do not attend.
pub fn provenance_layer_indices(
    cfg: &Qwen35Config,
    map: &KvLayerMap,
) -> Option<ProvenanceLayerIndices> {
    let n = cfg.num_layers;
    if n == 0 || map.num_kv_layers() == 0 {
        return None;
    }
    // The same depth fractions the uniform default uses.
    let syn = (n * 15 / 100).max(1);
    let sem = n / 2;
    let prag = (n * 85 / 100).min(n - 1);

    let band = |depth: usize| -> (usize, usize) {
        let hi = map.snap_to_attention(depth).expect("kv layers exist");
        let kv = map.kv_index(hi).expect("snap lands on an attention layer");
        let lo = map
            .layer_of_kv(kv.saturating_sub(1))
            .expect("kv index is in range");
        (lo, hi)
    };
    let (syn_l0, syn_l4) = band(syn);
    let (sem_l0, sem_l4) = band(sem);
    let (prag_l0, prag_l4) = band(prag);
    Some(ProvenanceLayerIndices {
        syn_l0,
        syn_l4,
        sem_l0,
        sem_l4,
        prag_l0,
        prag_l4,
    })
}

/// Create a batched session whose KV is allocated per *attention* layer.
///
/// `BatchedInferenceSession` takes the count of per-layer KV chunk sets,
/// which on a hybrid is the attention-layer count — passing transformer
/// depth would allocate four times the backings, and would make admission
/// price every wave's KV at four times its real cost.
pub fn create_session(
    cfg: &Qwen35Config,
    device: &candle::Device,
    config: BatchedConfig,
) -> candle::Result<BatchedInferenceSession> {
    let kv_layers = KvLayerMap::new(&cfg.layer_kinds).num_kv_layers();
    if kv_layers == 0 {
        candle::bail!(
            "a stack with no attention layers has no KV to page — the recurrent \
             state store carries all of its history"
        );
    }
    BatchedInferenceSession::new(
        kv_layers,
        cfg.num_kv_heads,
        cfg.attn_head_dim,
        device,
        config,
    )
}

impl QuantModel {
    /// The transformer-layer ↔ KV-layer map for this stack.
    pub fn kv_map(&self) -> KvLayerMap {
        KvLayerMap::new(&self.cfg.layer_kinds)
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::{DeltaNetDims, LayerKind, MoeConfig};
    use super::*;

    /// The 9B's real geometry.
    fn nine_b() -> Qwen35Config {
        Qwen35Config {
            vocab_size: 248_320,
            hidden_size: 4096,
            intermediate_size: 12_288,
            num_layers: 32,
            layer_kinds: Qwen35Config::schedule_from_interval(32, 4),
            num_attention_heads: 16,
            num_kv_heads: 4,
            attn_head_dim: 256,
            rope_dim: 64,
            rope_sections: [11, 11, 10, 0],
            rope_theta: 1e7,
            rms_norm_eps: 1e-6,
            delta_net: DeltaNetDims {
                head_dim: 128,
                n_k_heads: 16,
                n_v_heads: 32,
                conv_kernel: 4,
            },
            moe: None,
            num_mtp_layers: 0,
            max_position_embeddings: 262_144,
        }
    }

    #[test]
    fn geometry_reports_the_hybrid_shapes() {
        let cfg = nine_b();
        let g = wave_geometry(&cfg, DType::BF16);
        assert_eq!(g.hidden, 4096);
        assert_eq!((g.n_head, g.n_kv_head, g.head_dim), (16, 4, 256));
        // Dense: the MoE terms collapse rather than needing a second branch.
        assert_eq!((g.experts_per_tok, g.n_experts), (1, 1));
        assert_eq!(g.accum_dtype, DType::F32);
        // conv_dim = 2·(16·128) + 32·128 = 8192, so the FFN's 12288 wins.
        assert_eq!(cfg.delta_net.conv_dim(), 8192);
        assert_eq!(g.intermediate, 12_288);
    }

    #[test]
    fn a_narrow_ffn_is_priced_by_the_deltanet_projections_instead() {
        // The case the FFN-only pricing would get wrong: heads wide enough
        // that the fused [Q|K|V] projection exceeds the FFN.
        let mut cfg = nine_b();
        cfg.intermediate_size = 4096;
        assert_eq!(
            priced_intermediate(&cfg),
            8192,
            "the span must hold the widest per-row buffer of EITHER layer kind"
        );
    }

    #[test]
    fn moe_prices_the_per_expert_intermediate() {
        let mut cfg = nine_b();
        cfg.moe = Some(MoeConfig {
            n_experts: 256,
            n_experts_used: 8,
            expert_ffn_size: 512,
            shared_expert_ffn_size: 512,
            norm_topk_prob: true,
        });
        let g = wave_geometry(&cfg, DType::BF16);
        assert_eq!((g.experts_per_tok, g.n_experts), (8, 256));
        // 512 per expert is narrower than the DeltaNet projection, which is
        // still carried per row — so the projection sets the floor.
        assert_eq!(g.intermediate, 8192);
    }

    #[test]
    fn provenance_bands_land_only_on_attention_layers() {
        let cfg = nine_b();
        let map = KvLayerMap::new(&cfg.layer_kinds);
        let p = provenance_layer_indices(&cfg, &map).unwrap();
        let attention = map.attention_layers();
        for (name, idx) in [
            ("syn_l0", p.syn_l0),
            ("syn_l4", p.syn_l4),
            ("sem_l0", p.sem_l0),
            ("sem_l4", p.sem_l4),
            ("prag_l0", p.prag_l0),
            ("prag_l4", p.prag_l4),
        ] {
            assert!(
                attention.contains(&idx),
                "{name} = {idx} is a DeltaNet layer and has no Q to capture"
            );
        }
        // Bands are ordered and each spans two distinct attention layers.
        assert!(p.syn_l0 <= p.syn_l4 && p.sem_l0 < p.sem_l4 && p.prag_l0 < p.prag_l4);
        assert!(p.syn_l4 <= p.sem_l4 && p.sem_l4 <= p.prag_l4);
        assert!(p.prag_l4 < cfg.num_layers);
    }

    /// The session must be sized by attention layers, not transformer depth.
    /// Getting this wrong is invisible — the model still runs — but costs 4×
    /// the KV backings and makes admission refuse 4× more prefill than the
    /// cache can actually hold.
    #[test]
    fn session_allocates_kv_per_attention_layer_not_per_layer() -> candle::Result<()> {
        let cfg = nine_b();
        let device = candle::Device::new_cuda(0)?;
        let session = create_session(&cfg, &device, BatchedConfig::default())?;
        assert_eq!(
            session.num_layers(),
            8,
            "32-layer hybrid must page 8 KV layers, not {}",
            cfg.num_layers
        );
        Ok(())
    }

    #[test]
    fn a_stack_with_no_attention_is_refused_a_session() {
        let mut cfg = nine_b();
        cfg.layer_kinds = vec![LayerKind::DeltaNet; cfg.num_layers];
        let device = candle::Device::Cpu;
        let err = match create_session(&cfg, &device, BatchedConfig::default()) {
            Ok(_) => panic!("a stack with no attention layers must not get a KV session"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("no attention layers"), "{err}");
    }

    #[test]
    fn a_stack_with_no_attention_has_no_provenance() {
        let mut cfg = nine_b();
        cfg.layer_kinds = vec![LayerKind::DeltaNet; cfg.num_layers];
        let map = KvLayerMap::new(&cfg.layer_kinds);
        assert!(
            provenance_layer_indices(&cfg, &map).is_none(),
            "must refuse rather than return indices that cannot capture"
        );
    }
}
