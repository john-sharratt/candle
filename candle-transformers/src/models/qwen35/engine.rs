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

use candle::quantized::Int8Mode;
use candle::{DType, Device};
use candle_nn::kv_cache::{ffn_work_dtype, DeltaNetWidths, ModelGeometry, SharedExpertWidths};

use super::config::{LayerKind, Qwen35Config};
use super::quantized_weights::{QuantModel, SHARED_GATE_TILE};
use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, KvLayers, ProvenanceLayerIndices,
};
use crate::models::delta_net::KvLayerMap;
use crate::models::prefill_utils::paged_decode_q8_head_dim;

/// The per-row intermediate activation the **FFN phase** carries.
///
/// Both kinds of layer in this hybrid run the same FFN, so this is one width
/// and not a maximum over the stack. It used to take
/// `.max(cfg.delta_net.conv_dim())` on the reasoning that the plan sizes its
/// FFN span from one intermediate and a DeltaNet layer's fused `[Q|K|V]`
/// projection can be wider than one. The projection is real, but it is not in
/// this phase: a DeltaNet mixer carves from the *attention* generation, which
/// is what a layer opens before its FFN whichever mixer it has. On the 0.8B
/// that `max` raised the FFN span from 3584 to 6144 columns — 49.1 MiB of a
/// 139.3 MiB span — to cover a buffer that never appears in it. The mixer is
/// priced where it allocates, through [`ModelGeometry::delta_net`].
pub fn priced_intermediate(cfg: &Qwen35Config) -> usize {
    match &cfg.moe {
        // MoE prices the *per-expert* intermediate; `expert_rows` applies the
        // fan-out separately. The shared expert is not folded in here: its
        // buffers are a chain of their own (`ModelGeometry::shared_expert`),
        // and a `max` would widen every routed row to cover it.
        Some(moe) => moe.expert_ffn_size,
        None => cfg.intermediate_size,
    }
}

/// The geometry admission prices a wave from.
///
/// `int8mode` is the session's, not the config's: the weights' KO twins are
/// chosen at load, and whether they were decides the encoding each norm's fused
/// epilogue emits — a real difference in what the span holds, not a rounding.
pub fn wave_geometry(cfg: &Qwen35Config, act_dtype: DType, int8mode: Int8Mode) -> ModelGeometry {
    let (experts_per_tok, n_experts) = match &cfg.moe {
        Some(moe) => (moe.n_experts_used.max(1), moe.n_experts.max(1)),
        None => (1, 1),
    };
    ModelGeometry {
        hidden: cfg.hidden_size,
        vocab: cfg.vocab_size,
        intermediate: priced_intermediate(cfg),
        n_head: cfg.num_attention_heads,
        n_kv_head: cfg.num_kv_heads,
        head_dim: cfg.attn_head_dim,
        experts_per_tok,
        n_experts,
        // **Asked of the stack, not assumed of the lineage.** Every checkpoint
        // in this family so far is a 3:1 hybrid, but a stack with no DeltaNet
        // layer has no mixer chain to compare the attention one against, and
        // pricing one would put a chain in the `max` that no layer can run.
        delta_net: cfg
            .layer_kinds
            .iter()
            .any(|k| matches!(k, LayerKind::DeltaNet))
            .then_some(DeltaNetWidths {
                conv_dim: cfg.delta_net.conv_dim(),
                value_dim: cfg.delta_net.value_dim(),
                n_v_heads: cfg.delta_net.n_v_heads,
            }),
        // Every MoE layer in this lineage adds an always-active shared expert
        // to the routed block, and its gate projection is stored padded to one
        // KO tile.
        shared_expert: cfg.moe.as_ref().map(|moe| SharedExpertWidths {
            intermediate: moe.shared_expert_ffn_size,
            gate_cols: SHARED_GATE_TILE,
        }),
        act_dtype,
        packed_norm: int8mode.is_int8(),
        // This lineage's callers read `int8mode` off the head itself
        // (`HybridBatched::int8mode` is `lm_head.int8mode()`), so the two are
        // the same fact here — stated separately because on the generic path
        // they are not.
        packed_head: int8mode.is_int8(),
        // Three separate projections in **both** modes, unlike the generic
        // path: this lineage's Q weight is the interleaved `[q | gate]` and does
        // not pack with K and V, so `project_qkv` never reaches
        // `qkv_segmented`. K and V arrive contiguous and neither is copied out
        // of anything.
        fused_qkv: false,
        // No Q/K/V biases in this lineage.
        qkv_bias: false,
        // The fused q8 decode combine serves this lineage's head dim on an
        // int8 session, through the same predicate the dispatch asks.
        decode_q8_context: int8mode.is_int8() && paged_decode_q8_head_dim(cfg.attn_head_dim),
        // Per-head RMSNorm on Q and K, as every Qwen3-and-later stack has.
        head_qk_norm: true,
        // The wave is flattened to `[rows, hidden]` before the layer sweep, so
        // the head-wise norms reshape for free.
        head_norm_reshapes: false,
        // The expert GEMMs' output width. The int8 grouped matmul is typed
        // `f32`; the FP path it is guarded against emits the work dtype
        // instead, so this follows the session like `packed_norm` does. Moot on
        // a dense checkpoint, where `Chain::Ffn` prices zero either way.
        accum_dtype: if int8mode.is_int8() {
            DType::F32
        } else {
            ffn_work_dtype(act_dtype)
        },
        // The `[q | gate]` interleaved projection — a lineage fact, and the one
        // flag here that is a weight shape rather than a config field.
        gated_qkv: true,
        // **Derived.** The paged kernels only know full-width RoPE, so a stack
        // whose rotary is narrower than its head dim pays two gather copies to
        // re-order Q and K; one where they are equal pays nothing. That is what
        // `rope_dim` says (`rope.dimension_count`, ggml's `n_rot`), so ask it
        // rather than assert this lineage's 64-of-256.
        partial_rotary: cfg.rope_dim < cfg.attn_head_dim,
        hyper: None,
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

/// KV layers a session of this stack allocates: one per *attention* layer,
/// plus one for the MTP draft head when the checkpoint carries it.
///
/// `BatchedInferenceSession` takes the count of per-layer KV chunk sets,
/// which on a hybrid is the attention-layer count — passing transformer
/// depth would allocate four times the backings, and would make admission
/// price every wave's KV at four times its real cost.
///
/// The head's layer sits **past** every trunk layer, so [`KvLayerMap`] — which
/// only ever yields `0..num_kv_layers()` — cannot name it and the layer sweep
/// cannot reach it. It is written by the head's own pass at the end of each
/// wave and read only when drafting. See [`super::mtp`] for why the head is a
/// layer of the model rather than a sidecar with a private cache.
pub fn session_kv_layers(cfg: &Qwen35Config) -> candle::Result<usize> {
    Ok(kv_layers(cfg)?.total())
}

/// Every KV layer this session allocates, the head's included, all of it
/// **stream**.
///
/// The head's layer is not a [`KvLayers::draft`] layer even though a head
/// writes it, and that is the deliberate design [`super::draft`] argues for:
/// `head_wave_pass` runs over the same rows at the same positions in the same
/// wave, so the layer always stands at the same length as its siblings. It is
/// covered by the wave's own metadata — [`super::draft::HeadWave::kv_layer`]
/// indexes straight into those slot headers — and every session-wide operation
/// that assumes "a sequence's layers describe one stream at one length" (fork,
/// view, prefix injection, turn sealing, truncation) keeps working without
/// being told the head exists.
///
/// `draft` is for a head that has been loaded but does not yet step its layer,
/// which is a state this model is not in.
pub fn kv_layers(cfg: &Qwen35Config) -> candle::Result<KvLayers> {
    let trunk = KvLayerMap::new(&cfg.layer_kinds).num_kv_layers();
    if trunk == 0 {
        candle::bail!(
            "a stack with no attention layers has no KV to page — the recurrent \
             state store carries all of its history"
        );
    }
    // The loader refuses anything but 0 or 1, so this is a count of heads, not
    // a schedule over them.
    Ok(KvLayers::stream_only(trunk + cfg.num_mtp_layers))
}

/// The KV layer the MTP draft head writes, or `None` on a checkpoint without
/// one. Always the last, which is what keeps it out of the sweep's reach.
pub fn mtp_kv_layer(cfg: &Qwen35Config) -> Option<usize> {
    (cfg.num_mtp_layers > 0).then(|| KvLayerMap::new(&cfg.layer_kinds).num_kv_layers())
}

/// The KV layers a wave over `[layer_start, layer_end)` **touches** — the
/// layer map's range, plus the draft head's when the window reaches the last
/// trunk layer, because that is when the head's pass runs.
///
/// One answer, three callers, deliberately: admission claims this range, the
/// failure rollback restores it, and the sweep writes it. They must be the same
/// set or the engine breaks in two different ways — a claim short of the sweep
/// is a chunk allocated from inside the forward that owns the partition, and a
/// rollback short of the sweep leaves the head's layer one token ahead of the
/// trunk's after a failed wave, which the next wave "heals" by truncating a
/// token the caller was never given.
pub fn wave_kv_range(cfg: &Qwen35Config, layer_start: usize, layer_end: usize) -> (usize, usize) {
    let map = KvLayerMap::new(&cfg.layer_kinds);
    let (start, mut end) = map.kv_range(layer_start, layer_end);
    if mtp_kv_layer(cfg).is_some() && layer_end == cfg.num_layers {
        end += 1;
    }
    (start, end)
}

/// Create a batched session whose KV is allocated per [`session_kv_layers`].
pub fn create_session(
    cfg: &Qwen35Config,
    device: &Device,
    config: BatchedConfig,
) -> candle::Result<BatchedInferenceSession> {
    BatchedInferenceSession::new(
        kv_layers(cfg)?,
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
        let g = wave_geometry(&cfg, DType::BF16, Int8Mode::Performance);
        assert_eq!(g.hidden, 4096);
        assert_eq!((g.n_head, g.n_kv_head, g.head_dim), (16, 4, 256));
        // Dense: the MoE terms collapse rather than needing a second branch.
        assert_eq!((g.experts_per_tok, g.n_experts), (1, 1));
        assert_eq!(g.accum_dtype, DType::F32);
        // **The FFN width is the FFN's, full stop.** The DeltaNet projection is
        // wider per row on some stacks (here 2·(16·128) + 32·128 = 8192 against
        // 12288, so it would not have shown), but it is not in this phase: the
        // mixer carves from the attention generation, whichever mixer a layer
        // has. Folding it in here priced a buffer where it never appears.
        assert_eq!(cfg.delta_net.conv_dim(), 8192);
        assert_eq!(g.intermediate, 12_288);
        // The mixer chain is priced, at the one width of it that is on the span.
        assert_eq!(g.delta_net.map(|d| d.n_v_heads), Some(32));
        // An int8 session's norms emit q8a128; a float session's do not.
        assert!(g.packed_norm);
        assert!(!wave_geometry(&cfg, DType::BF16, Int8Mode::Off).packed_norm);
    }

    /// **The FFN width is the FFN's, even when the mixer is wider.**
    ///
    /// This used to assert the opposite — that a stack whose DeltaNet
    /// projection exceeds its FFN is priced by the projection — and the
    /// reasoning was that the span "must hold the widest per-row buffer of
    /// either layer kind". True of the *attention* span, which is the one both
    /// mixers carve from; false of the FFN span, which every layer runs
    /// identically whichever mixer it has. Pricing it here put a buffer in a
    /// phase it never appears in: 49.1 MiB of a 139.3 MiB span on the 0.8B.
    #[test]
    fn the_ffn_width_is_the_ffns_even_when_the_mixer_is_wider() {
        let mut cfg = nine_b();
        cfg.intermediate_size = 4096;
        assert!(
            cfg.delta_net.conv_dim() > cfg.intermediate_size,
            "the case worth pinning is the one where the mixer is the wider"
        );
        assert_eq!(priced_intermediate(&cfg), 4096);
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
        let g = wave_geometry(&cfg, DType::BF16, Int8Mode::Performance);
        assert_eq!((g.experts_per_tok, g.n_experts), (8, 256));
        // The per-expert width, and only that: the FFN phase prices the FFN,
        // and `expert_rows` applies the fan-out separately.
        assert_eq!(g.intermediate, 512);
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
        let device = Device::new_cuda(0)?;
        let session = create_session(&cfg, &device, BatchedConfig::default())?;
        assert_eq!(
            session.num_layers(),
            8,
            "32-layer hybrid must page 8 KV layers, not {}",
            cfg.num_layers
        );
        Ok(())
    }

    /// **A checkpoint with an MTP head pages one KV layer more, and it is the
    /// last one.**
    ///
    /// Both halves matter and only one of them is arithmetic. The extra layer
    /// is what gives the draft head a paged history at all; putting it LAST is
    /// what keeps it out of everything else's way, because
    /// [`KvLayerMap::kv_index`] only ever yields `0..num_kv_layers()` — so the
    /// sweep cannot name it, and no range the map produces can reach it. Move
    /// it anywhere else and a trunk layer's KV silently becomes the head's.
    #[test]
    fn an_mtp_checkpoint_pages_one_more_kv_layer_and_it_is_last() -> candle::Result<()> {
        let mut cfg = nine_b();
        assert_eq!(session_kv_layers(&cfg)?, 8, "no head, no extra layer");
        assert_eq!(mtp_kv_layer(&cfg), None);

        cfg.num_mtp_layers = 1;
        assert_eq!(session_kv_layers(&cfg)?, 9);
        let head = mtp_kv_layer(&cfg).expect("a head has a layer");
        assert_eq!(head, 8, "the head's layer sits past every trunk KV layer");
        assert_eq!(
            head,
            session_kv_layers(&cfg)? - 1,
            "the head must be the LAST layer, or the map's range would cover it"
        );

        let map = KvLayerMap::new(&cfg.layer_kinds);
        assert!(
            (0..cfg.num_layers).all(|l| map.kv_index(l).is_none_or(|kv| kv < head)),
            "a trunk layer resolved to the head's KV index — the sweep would \
             write the draft head's history"
        );
        assert_eq!(
            map.kv_range(0, cfg.num_layers).1,
            head,
            "the layer MAP must stop short of the head's layer — nothing that \
             translates a trunk layer may reach it"
        );
        Ok(())
    }

    /// **A wave that reaches the last trunk layer touches the head's KV too,
    /// and a partial window does not.**
    ///
    /// The head's pass runs at the end of a complete sweep, so that is the only
    /// window whose range covers it. Admission claims this range, the failure
    /// rollback restores it, and the sweep writes it — three callers that must
    /// agree, which is why there is one function rather than three `+ 1`s.
    #[test]
    fn a_full_sweep_claims_the_head_s_kv_layer_and_a_partial_window_does_not() {
        let mut cfg = nine_b();
        let n = cfg.num_layers;

        // No head: the range is the map's, whatever the window.
        assert_eq!(wave_kv_range(&cfg, 0, n), (0, 8));
        assert_eq!(wave_kv_range(&cfg, 0, n / 2), (0, 4));

        cfg.num_mtp_layers = 1;
        assert_eq!(
            wave_kv_range(&cfg, 0, n),
            (0, 9),
            "a full sweep runs the head's pass, so its layer must be claimed"
        );
        assert_eq!(
            wave_kv_range(&cfg, 0, n / 2),
            (0, 4),
            "a half window never reaches the head's pass and must not claim its \
             layer — the claim would be storage no wave in that window writes"
        );
        assert_eq!(
            wave_kv_range(&cfg, n / 2, n),
            (4, 9),
            "the tail window is where the head's pass runs, even though the \
             window does not start at layer 0"
        );
        // Contiguous with the trunk's last KV layer: `admit_wave_kv` and
        // `rollback_wave_kv` both walk the range as a range, so a gap would
        // index a layer neither of them means.
        assert_eq!(
            wave_kv_range(&cfg, 0, n).1 - 1,
            mtp_kv_layer(&cfg).unwrap(),
            "the head's layer must be the range's last, with no gap before it"
        );
    }

    #[test]
    fn a_stack_with_no_attention_is_refused_a_session() {
        let mut cfg = nine_b();
        cfg.layer_kinds = vec![LayerKind::DeltaNet; cfg.num_layers];
        let device = Device::Cpu;
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
