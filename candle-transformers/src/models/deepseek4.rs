//! DeepSeek-V4-Flash — the model.
//!
//! Everything that makes this model *this model*: the hyperparameter values of
//! the released 0731 checkpoint, the GGUF metadata keys they are stored under,
//! and the names its weights go by on disk. The layers, attention kernels, KV
//! arena, and wave engine it runs on are shared family machinery and live in
//! [`latent_moe`](crate::models::latent_moe).
//!
//! ```no_run
//! use candle_transformers::models::deepseek4::DEEPSEEK_V4;
//! use candle_transformers::models::latent_moe::{BatchedEngine, Engine};
//! # fn main() -> candle::Result<()> {
//! # let (path, device) = (std::path::Path::new("model.gguf"), candle::Device::Cpu);
//! let engine = Engine::load(
//!     path,
//!     &DEEPSEEK_V4,
//!     &device,
//!     candle::quantized::Int8Mode::Performance,
//! )?;
//! let model = BatchedEngine::new(engine)?;
//! # let _ = model;
//! # Ok(())
//! # }
//! ```
//!
//! # The architecture
//!
//! 284B total / 13B active, 43 layers, 256 routed experts (6 active) + 1 shared,
//! MXFP4 expert weights. See `docs/deepseek_v4_flash.md` for the full design.
//!
//! * **Latent single-KV attention** — 64 query heads read one shared 512-dim KV
//!   vector per token (K ≡ V), with learned per-head sinks and output de-rotation
//!   of the 64 RoPE dims.
//! * **CSA / HCA / SWA per layer** — decided by `compress_ratios`: layers 0-1 are
//!   sliding-window only, and 2-42 alternate 4:1 Compressed Sparse Attention
//!   (compressor + indexer top-k) on even layers with 128:1 Heavily Compressed
//!   Attention on odd ones.
//! * **Manifold-Constrained Hyper-Connections** — the residual stream carries 4
//!   copies mixed by a Sinkhorn-normalized matrix.
//! * **MoE** — `sqrtsoftplus` / `noaux_tc` routing, the first 3 layers hash-routed
//!   by token id, clamped SwiGLU experts.
//!
//! # Adding a successor
//!
//! A model in the same family needs only a sibling of this file: a unit struct
//! implementing [`Arch`], with its own [`defaults`](Arch::defaults),
//! [`meta_key`](Arch::meta_key), and [`leaf`](Arch::leaf). Nothing in
//! `latent_moe` names DeepSeek-V4, and the engine's own tests run against a
//! synthetic architecture precisely so that stays true. If the successor changes
//! the *latent geometry*, see [`latent_moe::geometry::SUPPORTED`] for the extra
//! kernel-side steps.

use crate::models::latent_moe::arch::{
    Arch, Compressor, CompressorPart, Ffn, Global, Hyper, HyperPart, Meta, Weight,
};
use crate::models::latent_moe::geometry::{LatentGeometry, D512_R64_B16};
use crate::models::latent_moe::Config;

/// DeepSeek-V4-Flash-0731.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4;

/// The architecture handle. `Config::arch` points here for every V4 model.
pub static DEEPSEEK_V4: DeepSeekV4 = DeepSeekV4;

/// Number of transformer blocks in the released checkpoint.
const N_LAYERS: usize = 43;

/// Per-layer compression schedule: layers 0-1 sliding-window, then 2-42
/// alternating CSA (4:1) on even layers and HCA (128:1) on odd ones.
///
/// Used when the GGUF omits `deepseek4.attention.compress_ratios`. It decides
/// each layer's attention kind, so a wrong schedule is a wrong model rather than
/// a load error — hence it is stated here with the rest of the model's identity,
/// and `the_real_checkpoint_matches_the_declared_defaults` asserts it element-wise
/// against the shipped file.
///
/// The file's own array is 46 entries — three trailing zeros past the 43rd
/// block. Those indices are never read (`Config::compress_ratio` indexes by
/// layer), and reading them as a sliding-window tail is what an earlier version
/// of this schedule did: it cut layers 40-42 over to sliding-window, when the
/// checkpoint has them as CSA/HCA/CSA.
fn compress_ratios() -> Vec<usize> {
    (0..N_LAYERS)
        .map(|i| {
            if i < 2 {
                0
            } else if i.is_multiple_of(2) {
                4
            } else {
                128
            }
        })
        .collect()
}

impl Arch for DeepSeekV4 {
    fn id(&self) -> &'static str {
        "deepseek4"
    }

    fn geometry(&self) -> LatentGeometry {
        D512_R64_B16
    }

    fn defaults(&self) -> Config {
        Config {
            arch: &DEEPSEEK_V4,
            vocab_size: 129280,
            dim: 4096,
            moe_inter_dim: 2048,
            n_layers: N_LAYERS,
            n_hash_layers: 3,
            n_heads: 64,
            n_routed_experts: 256,
            n_shared_experts: 1,
            n_activated_experts: 6,
            score_func: "sqrtsoftplus".to_string(),
            route_scale: 1.5,
            swiglu_limit: 10.0,
            q_lora_rank: 1024,
            head_dim: 512,
            rope_head_dim: 64,
            norm_eps: 1e-6,
            o_groups: 8,
            o_lora_rank: 1024,
            window_size: 128,
            compress_ratios: compress_ratios(),
            compress_rope_theta: 160000.0,
            // YaRN parameters are NOT stored by llama.cpp's deepseek4 arch — it
            // bakes them in — so these four are only ever the values here.
            original_seq_len: 65536,
            rope_theta: 10000.0,
            rope_factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            index_n_heads: 64,
            index_head_dim: 128,
            index_topk: 512,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
        }
    }

    /// Metadata key suffixes, verified against the bartowski GGUF conversion.
    fn meta_key(&self, m: Meta) -> &'static str {
        match m {
            Meta::VocabSize => "vocab_size",
            Meta::EmbeddingLength => "embedding_length",
            Meta::ExpertFeedForwardLength => "expert_feed_forward_length",
            Meta::BlockCount => "block_count",
            Meta::HashLayerCount => "hash_layer_count",
            Meta::AttentionHeadCount => "attention.head_count",
            Meta::ExpertCount => "expert_count",
            Meta::ExpertSharedCount => "expert_shared_count",
            Meta::ExpertUsedCount => "expert_used_count",
            Meta::ExpertWeightsScale => "expert_weights_scale",
            Meta::SwigluClampExp => "swiglu_clamp_exp",
            Meta::QLoraRank => "attention.q_lora_rank",
            Meta::KeyLength => "attention.key_length",
            Meta::RopeDimensionCount => "rope.dimension_count",
            Meta::LayerNormRmsEpsilon => "attention.layer_norm_rms_epsilon",
            Meta::OutputGroupCount => "attention.output_group_count",
            Meta::OutputLoraRank => "attention.output_lora_rank",
            Meta::SlidingWindow => "attention.sliding_window",
            Meta::CompressRopeFreqBase => "attention.compress_rope_freq_base",
            Meta::RopeFreqBase => "rope.freq_base",
            Meta::IndexerHeadCount => "attention.indexer.head_count",
            Meta::IndexerKeyLength => "attention.indexer.key_length",
            Meta::IndexerTopK => "attention.indexer.top_k",
            Meta::HyperConnectionCount => "hyper_connection.count",
            Meta::HyperConnectionSinkhornIterations => "hyper_connection.sinkhorn_iterations",
            Meta::HyperConnectionEpsilon => "hyper_connection.epsilon",
            Meta::CompressRatios => "attention.compress_ratios",
        }
    }

    fn block_prefix(&self, layer: usize) -> String {
        format!("blk.{layer}.")
    }

    fn leaf(&self, w: Weight) -> &'static str {
        match w {
            Weight::AttnQA => "attn_q_a.weight",
            Weight::AttnQANorm => "attn_q_a_norm.weight",
            Weight::AttnQB => "attn_q_b.weight",
            Weight::AttnKv => "attn_kv.weight",
            Weight::AttnKvANorm => "attn_kv_a_norm.weight",
            Weight::AttnOutputA => "attn_output_a.weight",
            Weight::AttnOutputB => "attn_output_b.weight",
            Weight::AttnSinks => "attn_sinks.weight",
            Weight::AttnNorm => "attn_norm.weight",
            Weight::FfnNorm => "ffn_norm.weight",
            Weight::IndexerQB => "indexer.attn_q_b.weight",
            Weight::IndexerProj => "indexer.proj.weight",
            Weight::FfnGateInp => "ffn_gate_inp.weight",
            Weight::FfnGateTid2Eid => "ffn_gate_tid2eid.weight",
            Weight::ExpProbsBias => "exp_probs_b.bias",
            Weight::RoutedExperts(Ffn::Gate) => "ffn_gate_exps.weight",
            Weight::RoutedExperts(Ffn::Up) => "ffn_up_exps.weight",
            Weight::RoutedExperts(Ffn::Down) => "ffn_down_exps.weight",
            Weight::SharedExpert(Ffn::Gate) => "ffn_gate_shexp.weight",
            Weight::SharedExpert(Ffn::Up) => "ffn_up_shexp.weight",
            Weight::SharedExpert(Ffn::Down) => "ffn_down_shexp.weight",
            Weight::Compressor(Compressor::Attn, CompressorPart::Kv) => "attn_compressor_kv.weight",
            Weight::Compressor(Compressor::Attn, CompressorPart::Gate) => {
                "attn_compressor_gate.weight"
            }
            Weight::Compressor(Compressor::Attn, CompressorPart::Ape) => {
                "attn_compressor_ape.weight"
            }
            Weight::Compressor(Compressor::Attn, CompressorPart::Norm) => {
                "attn_compressor_norm.weight"
            }
            Weight::Compressor(Compressor::Indexer, CompressorPart::Kv) => {
                "indexer_compressor_kv.weight"
            }
            Weight::Compressor(Compressor::Indexer, CompressorPart::Gate) => {
                "indexer_compressor_gate.weight"
            }
            Weight::Compressor(Compressor::Indexer, CompressorPart::Ape) => {
                "indexer_compressor_ape.weight"
            }
            Weight::Compressor(Compressor::Indexer, CompressorPart::Norm) => {
                "indexer_compressor_norm.weight"
            }
            Weight::Hyper(Hyper::Attn, HyperPart::Fn) => "hc_attn_fn.weight",
            Weight::Hyper(Hyper::Attn, HyperPart::Base) => "hc_attn_base.weight",
            Weight::Hyper(Hyper::Attn, HyperPart::Scale) => "hc_attn_scale.weight",
            Weight::Hyper(Hyper::Ffn, HyperPart::Fn) => "hc_ffn_fn.weight",
            Weight::Hyper(Hyper::Ffn, HyperPart::Base) => "hc_ffn_base.weight",
            Weight::Hyper(Hyper::Ffn, HyperPart::Scale) => "hc_ffn_scale.weight",
        }
    }

    fn global(&self, g: Global) -> &'static str {
        match g {
            Global::Embedding => "token_embd.weight",
            Global::OutputHead => "output.weight",
            Global::OutputNorm => "output_norm.weight",
            Global::OutputHyper(HyperPart::Fn) => "output_hc_fn.weight",
            Global::OutputHyper(HyperPart::Base) => "output_hc_base.weight",
            Global::OutputHyper(HyperPart::Scale) => "output_hc_scale.weight",
        }
    }

    fn drafter(&self) -> Option<&'static dyn Arch> {
        Some(&DFLASH)
    }
}

/// The `dflash` DSpark drafter that speculates for V4.
///
/// A separate GGUF and a separate `general.architecture`, so a separate [`Arch`]
/// — but structurally a V4 model with three blocks, sharing its tensor naming
/// and latent geometry. Only the metadata namespace really differs: its
/// hyperparameters are stored under `dflash.*`, and it carries the output-side
/// hyper-connection and confidence head that the backbone has no use for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash;

/// The drafter architecture handle.
pub static DFLASH: DFlash = DFlash;

impl Arch for DFlash {
    fn id(&self) -> &'static str {
        "dflash"
    }

    fn geometry(&self) -> LatentGeometry {
        DEEPSEEK_V4.geometry()
    }

    /// The drafter's GGUF carries every hyperparameter it needs, so these are
    /// only ever a floor. They are V4's values re-pointed at this arch, with the
    /// one shape that is known a priori: three backbone blocks.
    fn defaults(&self) -> Config {
        Config {
            arch: &DFLASH,
            n_layers: 3,
            ..DEEPSEEK_V4.defaults()
        }
    }

    /// Same key suffixes as V4 — only [`Arch::id`], and so the namespace they
    /// sit in, differs.
    fn meta_key(&self, m: Meta) -> &'static str {
        DEEPSEEK_V4.meta_key(m)
    }

    fn block_prefix(&self, layer: usize) -> String {
        DEEPSEEK_V4.block_prefix(layer)
    }

    fn leaf(&self, w: Weight) -> &'static str {
        DEEPSEEK_V4.leaf(w)
    }

    fn global(&self, g: Global) -> &'static str {
        DEEPSEEK_V4.global(g)
    }
}

#[cfg(test)]
mod tests {
    use super::{compress_ratios, Arch, DEEPSEEK_V4, DFLASH, N_LAYERS};
    use crate::models::latent_moe::arch::{
        Compressor, CompressorPart, Ffn, Global, Hyper, HyperPart, Meta, Weight,
    };
    use crate::models::latent_moe::LayerKind;

    /// The released checkpoint's shape. These are the numbers a GGUF without
    /// metadata falls back to, so a typo here loads a subtly wrong model.
    #[test]
    fn defaults_match_the_released_checkpoint() {
        let cfg = DEEPSEEK_V4.defaults();
        assert_eq!(cfg.n_layers, 43);
        assert_eq!(cfg.dim, 4096);
        assert_eq!(cfg.vocab_size, 129280);
        assert_eq!(cfg.n_heads, 64);
        assert_eq!(cfg.head_dim, 512);
        assert_eq!(cfg.rope_head_dim, 64);
        assert_eq!(cfg.q_lora_rank, 1024);
        assert_eq!(cfg.o_groups, 8);
        assert_eq!(cfg.o_lora_rank, 1024);
        assert_eq!(cfg.n_routed_experts, 256);
        assert_eq!(cfg.n_activated_experts, 6);
        assert_eq!(cfg.n_shared_experts, 1);
        assert_eq!(cfg.moe_inter_dim, 2048);
        assert_eq!(cfg.n_hash_layers, 3);
        assert_eq!(cfg.window_size, 128);
        assert_eq!(cfg.index_n_heads, 64);
        assert_eq!(cfg.index_head_dim, 128);
        assert_eq!(cfg.index_topk, 512);
        assert_eq!(cfg.hc_mult, 4);
        assert_eq!(cfg.hc_sinkhorn_iters, 20);
        assert_eq!(cfg.route_scale, 1.5);
        assert_eq!(cfg.swiglu_limit, 10.0);
        assert_eq!(cfg.nope_head_dim(), 448);
    }

    /// Both architectures in this file must satisfy the rules every arch does —
    /// including the geometry/config agreement, which is otherwise easy to break
    /// by editing one of the two declarations.
    #[test]
    fn both_arches_satisfy_the_arch_invariants() {
        crate::models::latent_moe::arch::assert_arch_invariants(&DEEPSEEK_V4);
        crate::models::latent_moe::arch::assert_arch_invariants(&DFLASH);
    }

    /// The declared geometry is also the one the kernels are built for — a model
    /// whose latent no kernel implements would fail at `Engine::load`.
    #[test]
    fn declared_geometry_is_one_the_kernels_implement() {
        let g = DEEPSEEK_V4.geometry();
        assert_eq!(g.nope_dim(), DEEPSEEK_V4.defaults().nope_head_dim());
        assert!(
            crate::models::latent_moe::geometry::is_supported(g),
            "V4's latent geometry {g:?} is not in geometry::SUPPORTED"
        );
    }

    #[test]
    fn compression_schedule_assigns_the_documented_layer_kinds() {
        let r = compress_ratios();
        assert_eq!(r.len(), N_LAYERS);
        let cfg = DEEPSEEK_V4.defaults();
        // Layers 0-1: sliding window only.
        assert_eq!(cfg.layer_kind(0), LayerKind::SlidingWindow);
        assert_eq!(cfg.layer_kind(1), LayerKind::SlidingWindow);
        // 2-42 alternate CSA / HCA, starting with CSA on the even layer, and run
        // all the way to the last block — there is NO sliding-window tail.
        assert_eq!(cfg.layer_kind(2), LayerKind::Csa);
        assert_eq!(cfg.layer_kind(3), LayerKind::Hca);
        assert_eq!(cfg.layer_kind(40), LayerKind::Csa);
        assert_eq!(cfg.layer_kind(41), LayerKind::Hca);
        assert_eq!(cfg.layer_kind(42), LayerKind::Csa);
        // Only CSA layers carry an indexer.
        assert!(cfg.layer_kind(2).has_indexer());
        assert!(!cfg.layer_kind(3).has_indexer());
        assert!(cfg.layer_kind(3).compresses());
    }

    /// Raw expected tensor names. These strings are the load contract against
    /// the GGUF on disk — a rename is a load failure on a 156 GB file, so they
    /// are asserted literally rather than through the enum that produced them.
    #[test]
    fn tensor_names_match_the_gguf_on_disk() {
        let n = |w| DEEPSEEK_V4.weight(7, w);
        assert_eq!(n(Weight::AttnQA), "blk.7.attn_q_a.weight");
        assert_eq!(n(Weight::AttnQANorm), "blk.7.attn_q_a_norm.weight");
        assert_eq!(n(Weight::AttnQB), "blk.7.attn_q_b.weight");
        assert_eq!(n(Weight::AttnKv), "blk.7.attn_kv.weight");
        assert_eq!(n(Weight::AttnKvANorm), "blk.7.attn_kv_a_norm.weight");
        assert_eq!(n(Weight::AttnOutputA), "blk.7.attn_output_a.weight");
        assert_eq!(n(Weight::AttnOutputB), "blk.7.attn_output_b.weight");
        assert_eq!(n(Weight::AttnSinks), "blk.7.attn_sinks.weight");
        assert_eq!(n(Weight::AttnNorm), "blk.7.attn_norm.weight");
        assert_eq!(n(Weight::FfnNorm), "blk.7.ffn_norm.weight");
        assert_eq!(n(Weight::IndexerQB), "blk.7.indexer.attn_q_b.weight");
        assert_eq!(n(Weight::IndexerProj), "blk.7.indexer.proj.weight");
        assert_eq!(n(Weight::FfnGateInp), "blk.7.ffn_gate_inp.weight");
        assert_eq!(n(Weight::FfnGateTid2Eid), "blk.7.ffn_gate_tid2eid.weight");
        assert_eq!(n(Weight::ExpProbsBias), "blk.7.exp_probs_b.bias");
        assert_eq!(
            n(Weight::RoutedExperts(Ffn::Gate)),
            "blk.7.ffn_gate_exps.weight"
        );
        assert_eq!(
            n(Weight::RoutedExperts(Ffn::Up)),
            "blk.7.ffn_up_exps.weight"
        );
        assert_eq!(
            n(Weight::RoutedExperts(Ffn::Down)),
            "blk.7.ffn_down_exps.weight"
        );
        assert_eq!(
            n(Weight::SharedExpert(Ffn::Gate)),
            "blk.7.ffn_gate_shexp.weight"
        );
        assert_eq!(
            n(Weight::SharedExpert(Ffn::Up)),
            "blk.7.ffn_up_shexp.weight"
        );
        assert_eq!(
            n(Weight::SharedExpert(Ffn::Down)),
            "blk.7.ffn_down_shexp.weight"
        );
        assert_eq!(
            n(Weight::Compressor(Compressor::Attn, CompressorPart::Kv)),
            "blk.7.attn_compressor_kv.weight"
        );
        assert_eq!(
            n(Weight::Compressor(Compressor::Attn, CompressorPart::Ape)),
            "blk.7.attn_compressor_ape.weight"
        );
        assert_eq!(
            n(Weight::Compressor(
                Compressor::Indexer,
                CompressorPart::Norm
            )),
            "blk.7.indexer_compressor_norm.weight"
        );
        assert_eq!(
            n(Weight::Hyper(Hyper::Attn, HyperPart::Fn)),
            "blk.7.hc_attn_fn.weight"
        );
        assert_eq!(
            n(Weight::Hyper(Hyper::Ffn, HyperPart::Scale)),
            "blk.7.hc_ffn_scale.weight"
        );
        assert_eq!(DEEPSEEK_V4.global(Global::Embedding), "token_embd.weight");
        assert_eq!(DEEPSEEK_V4.global(Global::OutputHead), "output.weight");
        assert_eq!(DEEPSEEK_V4.global(Global::OutputNorm), "output_norm.weight");
        assert_eq!(
            DEEPSEEK_V4.global(Global::OutputHyper(HyperPart::Fn)),
            "output_hc_fn.weight"
        );
    }

    /// The real checkpoint on disk agrees with the defaults declared above.
    ///
    /// This is the test that would catch a metadata key renamed upstream: every
    /// key that silently stopped resolving would fall back to
    /// [`Arch::defaults`] and the two would still agree — *except* that the
    /// values here are read from the file, so a key that resolves to a
    /// different number fails loudly. Skips when the 156 GB model is absent.
    #[test]
    fn the_real_checkpoint_matches_the_declared_defaults() -> candle::Result<()> {
        use crate::models::latent_moe::{config_from_gguf, GgufModel};

        let dir = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4");
        let first = dir.join("DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf");
        if !first.exists() {
            eprintln!("[skip] real GGUF not present at {}", first.display());
            return Ok(());
        }
        let splits: Vec<std::path::PathBuf> = GgufModel::discover_splits(&first)?
            .into_iter()
            .filter(|p| p.exists())
            .collect();
        let m = GgufModel::open(&splits)?;
        let cfg = config_from_gguf(&m, &DEEPSEEK_V4)?;

        assert_eq!(cfg.n_layers, 43);
        assert_eq!(cfg.dim, 4096);
        assert_eq!(cfg.vocab_size, 129280);
        assert_eq!(cfg.n_heads, 64);
        assert_eq!(cfg.head_dim, 512);
        assert_eq!(cfg.rope_head_dim, 64);
        assert_eq!(cfg.q_lora_rank, 1024);
        assert_eq!(cfg.o_groups, 8);
        assert_eq!(cfg.o_lora_rank, 1024);
        assert_eq!(cfg.n_routed_experts, 256);
        assert_eq!(cfg.n_activated_experts, 6);
        assert_eq!(cfg.n_shared_experts, 1);
        assert_eq!(cfg.moe_inter_dim, 2048);
        assert_eq!(cfg.n_hash_layers, 3);
        assert_eq!(cfg.window_size, 128);
        assert_eq!(cfg.index_n_heads, 64);
        assert_eq!(cfg.index_head_dim, 128);
        assert_eq!(cfg.index_topk, 512);
        assert_eq!(cfg.hc_mult, 4);
        assert_eq!(cfg.hc_sinkhorn_iters, 20);
        assert_eq!((cfg.route_scale * 10.0).round(), 15.0);
        assert_eq!(cfg.swiglu_limit, 10.0);
        // The declared fallback schedule must equal the file's, layer for layer,
        // over the blocks that actually exist. (The file's array carries three
        // trailing padding entries past block 42; `compress_ratio` never reads
        // them, and treating them as a sliding-window tail is the bug this
        // assertion exists to prevent.)
        let declared = compress_ratios();
        for (layer, &want) in declared.iter().enumerate().take(cfg.n_layers) {
            assert_eq!(
                cfg.compress_ratio(layer),
                want,
                "layer {layer}: the file's compress_ratio disagrees with the declared schedule"
            );
        }
        assert_eq!(cfg.layer_kind(2), LayerKind::Csa);
        assert_eq!(cfg.layer_kind(42), LayerKind::Csa);

        // Every tensor name this arch declares must actually be in the file.
        for layer in [0usize, 2, 3, 42] {
            let kind = cfg.layer_kind(layer);
            let mut expect = vec![
                Weight::AttnQA,
                Weight::AttnQANorm,
                Weight::AttnQB,
                Weight::AttnKv,
                Weight::AttnKvANorm,
                Weight::AttnOutputA,
                Weight::AttnOutputB,
                Weight::AttnSinks,
                Weight::AttnNorm,
                Weight::FfnNorm,
                Weight::FfnGateInp,
                Weight::RoutedExperts(Ffn::Gate),
                Weight::RoutedExperts(Ffn::Up),
                Weight::RoutedExperts(Ffn::Down),
                Weight::SharedExpert(Ffn::Gate),
                Weight::SharedExpert(Ffn::Up),
                Weight::SharedExpert(Ffn::Down),
                Weight::Hyper(Hyper::Attn, HyperPart::Fn),
                Weight::Hyper(Hyper::Attn, HyperPart::Base),
                Weight::Hyper(Hyper::Attn, HyperPart::Scale),
                Weight::Hyper(Hyper::Ffn, HyperPart::Fn),
                Weight::Hyper(Hyper::Ffn, HyperPart::Base),
                Weight::Hyper(Hyper::Ffn, HyperPart::Scale),
            ];
            if cfg.is_hash_layer(layer) {
                expect.push(Weight::FfnGateTid2Eid);
            } else {
                expect.push(Weight::ExpProbsBias);
            }
            if kind.compresses() {
                for p in [
                    CompressorPart::Kv,
                    CompressorPart::Gate,
                    CompressorPart::Ape,
                    CompressorPart::Norm,
                ] {
                    expect.push(Weight::Compressor(Compressor::Attn, p));
                }
            }
            if kind.has_indexer() {
                expect.push(Weight::IndexerQB);
                expect.push(Weight::IndexerProj);
                for p in [
                    CompressorPart::Kv,
                    CompressorPart::Gate,
                    CompressorPart::Ape,
                    CompressorPart::Norm,
                ] {
                    expect.push(Weight::Compressor(Compressor::Indexer, p));
                }
            }
            for w in expect {
                let name = DEEPSEEK_V4.weight(layer, w);
                assert!(
                    m.info(&name).is_some(),
                    "layer {layer} ({kind:?}) declares {w:?} as {name:?}, absent from the GGUF"
                );
            }
        }
        for g in [Global::Embedding, Global::OutputHead, Global::OutputNorm] {
            let name = DEEPSEEK_V4.global(g);
            assert!(
                m.info(name).is_some(),
                "{g:?} declared as {name:?}, absent from the GGUF"
            );
        }
        eprintln!("[ok] real config: {cfg:?}");
        Ok(())
    }

    /// Multi-session concurrent batched forwarding — the throughput + coherence
    /// gate the wave architecture exists to serve, on the SAME shared
    /// `TestParams::run` / StoryRewrite harness the Qwen/Llama batched models
    /// use (`quantized_qwen3_moe::test_parallel_batched_forwarding`). N sessions
    /// are given the same story with a per-session name to substitute; they are
    /// ragged-batch-prefilled and decoded **batched, one wave per step**, and
    /// each session must reproduce ITS OWN name-substituted story (the harness's
    /// common-prefix reproduction check + adjacent-session distinctness — the
    /// strongest cross-session-bleed / GID-collision / decode-desync detector).
    ///
    /// This lives with the MODEL, not with the wave engine: everything it pins
    /// down is a property of DeepSeek-V4-Flash — the checkpoint paths, its
    /// dialect and EOS, the `<think>` behaviour, the batch widths its 284B/13B
    /// shape makes meaningful, and the throughput the released weights reach.
    /// The engine-side gates it used to sit beside are in `latent_moe::wave`.
    ///
    /// DeepSeek ALWAYS thinks, so every reply opens with a `<think>…</think>`
    /// block; `with_suppress_thinking(true)` makes the harness strip it
    /// (`strip_thinking_blocks`) before the reproduction match — the model's
    /// dialect can't suppress the block itself, but the validator ignores it.
    /// The per-phase `forward_wave` profile (`wave.rs`'s `pipeline_record`
    /// marks) rides the harness's profile snapshot under `--features profile`.
    ///
    /// Run ONLY this one (the bare name matches every model's copy → cargo
    /// would run all six concurrently and OOM the pinned pools):
    ///   cargo test --release --features cuda,profile --lib \
    ///     deepseek4::tests::test_parallel_batched_forwarding \
    ///     -- --ignored --nocapture --test-threads 1
    #[test]
    #[ignore]
    #[cfg(feature = "cuda")]
    fn test_parallel_batched_forwarding() -> candle::Result<()> {
        use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::gpu_test_lock::gpu_serial;
        use crate::models::latent_moe::{BatchedEngine, Engine};
        use candle::quantized::Int8Mode;
        use candle::Device;

        let _gpu = gpu_serial();
        let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !path.exists() {
            eprintln!("[skip] merged file absent");
            return Ok(());
        }
        let device = Device::new_cuda(0)?;

        let tok_path = crate::models::batch_test::test_helpers::hf_get(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )?;
        let tokenizer_json = std::fs::read_to_string(&tok_path)
            .map_err(|e| candle::Error::msg(format!("read tokenizer.json: {e}")))?;
        let eos = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle::Error::msg(format!("tokenizer: {e}")))?
            .token_to_id("<｜end▁of▁sentence｜>")
            .expect("deepseek eos id");

        // The story is long, DeepSeek thinks before reproducing it, and it is a
        // 284B model over per-token prefill — so a couple of modest batch widths
        // keep the run inside the harness timeout while still exercising genuine
        // concurrency. The `InferenceMode` is cosmetic: `create_batched_session`
        // forces the single-latent FP8 arena regardless.
        //
        // Drafter + speculative decode are ON. Under the elastic partition the
        // ~6 GB drafter is a DENSE-tier resident loaded before the span
        // reservation (`Engine::load_with_drafter`), so the span simply
        // opens smaller and the KV↔expert boundary balances what remains —
        // the old world's 20× prefill collapse (drafter + wide prefill
        // spilling transients to host at ~0 free VRAM) is gone by construction:
        // the pool cushion for activations is carved out before the span, not
        // fought over after it.
        let dspark = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
            .join("dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf");
        let params = TestParams::new(64, &tokenizer_json, Dialect::deepseek())
            .map_err(|e| candle::Error::msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true) // strip <think>…</think> before validation
            .with_stop_on_eos(vec![eos])
            .with_print_outputs(true)
            // The comparison table's `int8` column must reflect the mode the model
            // is actually loaded with (`load_model` uses `Int8Mode::Performance` —
            // int8-KO expert/attention matmuls), not the harness default (`Off`).
            .with_int8mode(Int8Mode::Performance)
            .with_speculative(5)
            .with_timeout_secs(1800);

        // `16` is the wide-wave amortization config: at 8 contexts the fixed
        // expert sweep still dominates the wall, so 16 is where the marginal
        // per-token cost — and anything quietly serial in the wave — shows up
        // as a bulk-t/s plateau instead of the ~2× the extra width should buy.
        //
        // Trailing second `1`: by the end of the sweep the streaming expert
        // cache's Markov transition matrix is warm (learned from the 1+4+8+16
        // runs), so this final single-session pass reads the STEADY-STATE
        // single-token decode rate — the leading `1` reads it cold (predictor
        // untrained), and the gap between the two is the prefetch payoff.
        let configs = [1usize, 4, 8, 16, 1]
            .into_iter()
            .map(|n| TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: n,
                num_repeats: 1,
                generate_max_len: 64,
                test_mode: Some(TestMode::StoryRewrite),
            })
            .collect::<Vec<_>>();

        let load_model = || {
            let (engine, drafter) = Engine::load_with_drafter(
                &path,
                dspark.exists().then_some(dspark.as_path()),
                &DEEPSEEK_V4,
                &device,
                Int8Mode::Performance,
            )?;
            let model = BatchedEngine::new(engine)?;
            match drafter {
                Some(d) => model.with_drafter(d),
                None => Ok(model),
            }
        };
        params.run(configs, load_model)
    }

    /// The drafter is a separate architecture reachable from the target, and it
    /// must differ where it matters (metadata namespace) while agreeing where the
    /// engine depends on it (tensor names, latent geometry).
    #[test]
    fn the_drafter_shares_naming_but_not_its_namespace() {
        let d = DEEPSEEK_V4.drafter().expect("V4 declares a DSpark drafter");
        assert_eq!(d.id(), "dflash");
        assert_eq!(d.meta(Meta::BlockCount), "dflash.block_count");
        assert_eq!(d.meta(Meta::VocabSize), "dflash.vocab_size");
        assert_eq!(d.geometry(), DEEPSEEK_V4.geometry());
        assert_eq!(
            d.weight(1, Weight::AttnQA),
            DEEPSEEK_V4.weight(1, Weight::AttnQA)
        );
        assert_eq!(d.defaults().n_layers, 3);
        // The drafter has no drafter of its own.
        assert!(d.drafter().is_none());
    }

    /// Raw expected metadata keys, namespaced under `general.architecture`.
    #[test]
    fn metadata_keys_match_the_gguf_on_disk() {
        assert_eq!(DEEPSEEK_V4.id(), "deepseek4");
        assert_eq!(DEEPSEEK_V4.meta(Meta::VocabSize), "deepseek4.vocab_size");
        assert_eq!(DEEPSEEK_V4.meta(Meta::BlockCount), "deepseek4.block_count");
        assert_eq!(
            DEEPSEEK_V4.meta(Meta::AttentionHeadCount),
            "deepseek4.attention.head_count"
        );
        assert_eq!(
            DEEPSEEK_V4.meta(Meta::KeyLength),
            "deepseek4.attention.key_length"
        );
        assert_eq!(
            DEEPSEEK_V4.meta(Meta::IndexerTopK),
            "deepseek4.attention.indexer.top_k"
        );
        assert_eq!(
            DEEPSEEK_V4.meta(Meta::CompressRatios),
            "deepseek4.attention.compress_ratios"
        );
        assert_eq!(
            DEEPSEEK_V4.meta(Meta::HyperConnectionEpsilon),
            "deepseek4.hyper_connection.epsilon"
        );
        assert_eq!(
            DEEPSEEK_V4.meta(Meta::RopeFreqBase),
            "deepseek4.rope.freq_base"
        );
    }
}
