//! What a concrete model contributes to the engine.
//!
//! Everything in this module is the part of a sparse-latent MoE model that is
//! *not* shared: the numbers its config defaults to, the GGUF metadata keys
//! those numbers are stored under, and the names its weights go by on disk. The
//! layers, kernels, and wave engine around it are model-independent and read all
//! of it through [`Arch`].
//!
//! An [`Arch`] is a zero-sized `&'static dyn` handle carried by
//! [`Config::arch`](super::config::Config::arch), so any code holding a config
//! can resolve a tensor name without being handed a second parameter — and a
//! config can never be paired with the wrong architecture's names.
//!
//! See [`models::deepseek4`](crate::models::deepseek4) for a complete
//! implementation.

use std::fmt::Debug;

use super::config::Config;
use super::geometry::LatentGeometry;

/// The three projections of a SwiGLU feed-forward expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ffn {
    Gate,
    Up,
    Down,
}

/// Which of a block's two compressors. Both compress the token stream into
/// pooled entries; the attention one feeds attention, the indexer one feeds the
/// top-k selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compressor {
    Attn,
    Indexer,
}

/// The four tensors of a compressor: two matmul projections, an additive
/// positional bias, and its norm scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressorPart {
    Kv,
    Gate,
    Ape,
    Norm,
}

/// Which of a block's two hyper-connection sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hyper {
    Attn,
    Ffn,
}

/// The three tensors of one hyper-connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperPart {
    Fn,
    Base,
    Scale,
}

/// Every weight a transformer block can hold.
///
/// This is the complete set of per-layer tensors the engine asks for, so an
/// [`Arch::leaf`] implementation is an exhaustive match — a model that renames
/// one tensor cannot silently miss another, and a new tensor added to the engine
/// breaks every arch at compile time rather than at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weight {
    /// Query down-projection to `q_lora_rank`, its norm, and its up-projection.
    AttnQA,
    AttnQANorm,
    AttnQB,
    /// Latent KV projection (K ≡ V) and its norm.
    AttnKv,
    AttnKvANorm,
    /// Output projection: the grouped `[o_groups·o_lora_rank, per_group]` down
    /// stage and the up stage back to `dim`.
    AttnOutputA,
    AttnOutputB,
    /// Per-head learned attention sink logits.
    AttnSinks,
    /// Pre-attention and pre-FFN norm scales.
    AttnNorm,
    FfnNorm,
    /// Indexer query up-projection and scoring projection.
    IndexerQB,
    IndexerProj,
    /// Router logits weight.
    FfnGateInp,
    /// Hash-layer routing table: token id → expert id, replacing the router.
    FfnGateTid2Eid,
    /// `noaux_tc` per-expert routing bias.
    ExpProbsBias,
    /// Stacked routed-expert weights `[n_routed_experts, out, in]`.
    RoutedExperts(Ffn),
    /// The always-on shared expert.
    SharedExpert(Ffn),
    Compressor(Compressor, CompressorPart),
    Hyper(Hyper, HyperPart),
}

/// Tensors that belong to the model rather than to any one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Global {
    /// Token embedding table and the output head.
    Embedding,
    OutputHead,
    /// Final norm before the head.
    OutputNorm,
    /// Output-side hyper-connection, folding the residual copies before the
    /// head. Present on drafter models, absent on the main backbone.
    OutputHyper(HyperPart),
}

/// A model hyperparameter that may be stored in GGUF metadata.
///
/// Keys are read as `{architecture}.{suffix}`, where the suffix is what
/// [`Arch::meta_key`] returns. Anything absent from the file keeps the value
/// [`Arch::defaults`] gave it, so a metadata key that a model bakes into its
/// weights instead of writing down needs no special case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Meta {
    VocabSize,
    EmbeddingLength,
    ExpertFeedForwardLength,
    BlockCount,
    HashLayerCount,
    AttentionHeadCount,
    ExpertCount,
    ExpertSharedCount,
    ExpertUsedCount,
    ExpertWeightsScale,
    SwigluClampExp,
    QLoraRank,
    KeyLength,
    RopeDimensionCount,
    LayerNormRmsEpsilon,
    OutputGroupCount,
    OutputLoraRank,
    SlidingWindow,
    CompressRopeFreqBase,
    RopeFreqBase,
    IndexerHeadCount,
    IndexerKeyLength,
    IndexerTopK,
    HyperConnectionCount,
    HyperConnectionSinkhornIterations,
    HyperConnectionEpsilon,
    CompressRatios,
}

/// The model-specific half of a sparse-latent MoE model.
///
/// Implementations are unit structs with a `&'static` instance; the engine only
/// ever holds `&'static dyn Arch`.
pub trait Arch: Debug + Send + Sync + 'static {
    /// The `general.architecture` string this model writes into its GGUF, and
    /// the prefix its metadata keys are namespaced under.
    fn id(&self) -> &'static str;

    /// The latent shape the attention kernels must be compiled for. Checked
    /// against the compiled kernels by
    /// [`paged::assert_kernel_geometry`](super::paged::assert_kernel_geometry).
    fn geometry(&self) -> LatentGeometry;

    /// Every hyperparameter, at the value the released model uses. GGUF metadata
    /// overrides individual fields; anything the file omits keeps these.
    fn defaults(&self) -> Config;

    /// The metadata key suffix `m` is stored under, after `{id()}.`.
    fn meta_key(&self, m: Meta) -> &'static str;

    /// The tensor-name prefix for `layer`, including its trailing separator.
    fn block_prefix(&self, layer: usize) -> String;

    /// The tensor name of `w` within a block, without the block prefix.
    fn leaf(&self, w: Weight) -> &'static str;

    /// The full name of a model-level tensor.
    fn global(&self, g: Global) -> &'static str;

    /// The speculative-decode drafter trained for this model, if it has one.
    ///
    /// A drafter ships as its own GGUF under its own `general.architecture`, so
    /// it is a separate [`Arch`]; naming it here is what lets the engine load one
    /// without the machinery knowing which models exist.
    fn drafter(&self) -> Option<&'static dyn Arch> {
        None
    }

    /// Full GGUF tensor name of `w` in `layer`.
    fn weight(&self, layer: usize, w: Weight) -> String {
        let mut s = self.block_prefix(layer);
        s.push_str(self.leaf(w));
        s
    }

    /// Full metadata key for `m`, namespaced under this architecture.
    fn meta(&self, m: Meta) -> String {
        format!("{}.{}", self.id(), self.meta_key(m))
    }
}

/// Every rule an [`Arch`] must satisfy regardless of which model it is.
///
/// Called by each arch's own test, so a new model gets the whole invariant set
/// by writing one test rather than by remembering what the others assert. The
/// geometry/config agreement in particular is not obvious: [`Arch::geometry`]
/// and [`Arch::defaults`] describe the same tensors from two directions, and
/// nothing but this catches them drifting apart.
#[cfg(test)]
pub(crate) fn assert_arch_invariants(arch: &'static dyn Arch) {
    let cfg = arch.defaults();
    let g = arch.geometry();

    assert!(!arch.id().is_empty(), "an arch needs a metadata namespace");
    assert_eq!(
        cfg.arch.id(),
        arch.id(),
        "{}: defaults() must carry this arch, not another",
        arch.id()
    );
    assert_eq!(
        (g.head_dim, g.rope_dim),
        (cfg.head_dim, cfg.rope_head_dim),
        "{}: geometry() and defaults() disagree about the latent",
        arch.id()
    );
    assert!(
        cfg.rope_head_dim < cfg.index_head_dim,
        "{}: the indexer rotates rope_head_dim, so its keys need a non-empty nope region",
        arch.id()
    );
    assert!(
        cfg.compress_ratios.len() >= cfg.n_layers,
        "{}: the compression schedule must cover every block",
        arch.id()
    );

    // Names must be total and injective: every weight resolves, and no two
    // weights collide onto one tensor.
    let all = every_weight();
    let names: std::collections::HashSet<String> = all.iter().map(|w| arch.weight(0, *w)).collect();
    assert_eq!(
        names.len(),
        all.len(),
        "{}: two weights resolved to the same tensor name",
        arch.id()
    );
    let prefix = arch.block_prefix(0);
    for n in &names {
        assert!(
            n.starts_with(&prefix),
            "{}: {n} is missing the block prefix {prefix:?}",
            arch.id()
        );
    }
}

/// Every [`Weight`] variant, for exhaustiveness checks in tests.
#[cfg(test)]
pub(crate) fn every_weight() -> Vec<Weight> {
    let mut v = vec![
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
        Weight::IndexerQB,
        Weight::IndexerProj,
        Weight::FfnGateInp,
        Weight::FfnGateTid2Eid,
        Weight::ExpProbsBias,
    ];
    for f in [Ffn::Gate, Ffn::Up, Ffn::Down] {
        v.push(Weight::RoutedExperts(f));
        v.push(Weight::SharedExpert(f));
    }
    for c in [Compressor::Attn, Compressor::Indexer] {
        for p in [
            CompressorPart::Kv,
            CompressorPart::Gate,
            CompressorPart::Ape,
            CompressorPart::Norm,
        ] {
            v.push(Weight::Compressor(c, p));
        }
    }
    for h in [Hyper::Attn, Hyper::Ffn] {
        for p in [HyperPart::Fn, HyperPart::Base, HyperPart::Scale] {
            v.push(Weight::Hyper(h, p));
        }
    }
    v
}

#[cfg(test)]
pub(crate) mod test_arch {
    //! A synthetic architecture for the engine's own tests.
    //!
    //! The machinery must not depend on any particular model, so its tests run
    //! against this rather than against DeepSeek-V4. It uses the shipped latent
    //! geometry (the only one the kernels are built for) with small everything
    //! else.
    //!
    //! Its tensor names are deliberately **unlike** any real model's — a
    //! `layer{n}/` prefix and short leaves instead of `blk.{n}.` and
    //! `attn_q_a.weight`. That is the point: anything in the engine that assumes
    //! DeepSeek's naming instead of going through [`Arch`] fails here rather
    //! than silently working until a second model arrives.

    use super::{Arch, Compressor, CompressorPart, Ffn, Global, Hyper, HyperPart, Meta, Weight};
    use crate::models::latent_moe::config::Config;
    use crate::models::latent_moe::geometry::LatentGeometry;

    #[derive(Debug)]
    pub struct TestArch;

    /// The instance the engine's tests run against.
    pub static TEST_ARCH: TestArch = TestArch;

    impl Arch for TestArch {
        fn id(&self) -> &'static str {
            "latent_moe_test"
        }

        /// The smallest LEGAL latent, matching this arch's `defaults()`.
        ///
        /// Deliberately NOT in [`geometry::SUPPORTED`]: no kernel is built for
        /// it, and none needs to be. This fixture exercises the eager reference
        /// path, the loader's naming layer, and the config plumbing — never a
        /// paged-kernel launch, which is what `SUPPORTED` gates. Keeping it
        /// unsupported also keeps the two ideas distinct: a geometry can be
        /// well-formed without being implemented.
        fn geometry(&self) -> LatentGeometry {
            LatentGeometry::new(64, 32, 2)
        }

        fn defaults(&self) -> Config {
            Config {
                arch: &TEST_ARCH,
                vocab_size: 128,
                dim: 64,
                moe_inter_dim: 48,
                n_layers: 6,
                n_hash_layers: 1,
                n_heads: 4,
                n_routed_experts: 8,
                n_shared_experts: 1,
                n_activated_experts: 2,
                score_func: "sqrtsoftplus".to_string(),
                route_scale: 1.5,
                swiglu_limit: 10.0,
                q_lora_rank: 32,
                // head_dim/rope_head_dim are NOT free to shrink like the rest —
                // they ARE the latent geometry, and `geometry()` below has to
                // describe them. The band rules put a floor under both: a band
                // is a whole 32-dim MMA k-step, and the rope tail must land on a
                // band boundary, so (64, 32) over 2 bands is the smallest legal
                // latent there is. Anything smaller (the 32/8 this used to
                // claim) is not a geometry at all.
                head_dim: 64,
                rope_head_dim: 32,
                norm_eps: 1e-6,
                o_groups: 2,
                o_lora_rank: 24,
                window_size: 8,
                // SWA, SWA, CSA, HCA, CSA, HCA — every layer kind is exercised.
                compress_ratios: vec![0, 0, 4, 128, 4, 128],
                compress_rope_theta: 160000.0,
                original_seq_len: 64,
                rope_theta: 10000.0,
                rope_factor: 16.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                index_n_heads: 4,
                // The indexer's compressor rotates the same `rope_head_dim`, so
                // its key width has to leave room for a non-empty nope region.
                index_head_dim: 64,
                index_topk: 8,
                hc_mult: 4,
                hc_sinkhorn_iters: 20,
                hc_eps: 1e-6,
            }
        }

        fn meta_key(&self, m: Meta) -> &'static str {
            match m {
                Meta::VocabSize => "vocab",
                Meta::EmbeddingLength => "dim",
                Meta::ExpertFeedForwardLength => "expert_dim",
                Meta::BlockCount => "layers",
                Meta::HashLayerCount => "hash_layers",
                Meta::AttentionHeadCount => "heads",
                Meta::ExpertCount => "experts",
                Meta::ExpertSharedCount => "shared_experts",
                Meta::ExpertUsedCount => "active_experts",
                Meta::ExpertWeightsScale => "route_scale",
                Meta::SwigluClampExp => "swiglu_limit",
                Meta::QLoraRank => "q_lora",
                Meta::KeyLength => "head_dim",
                Meta::RopeDimensionCount => "rope_dim",
                Meta::LayerNormRmsEpsilon => "eps",
                Meta::OutputGroupCount => "o_groups",
                Meta::OutputLoraRank => "o_lora",
                Meta::SlidingWindow => "window",
                Meta::CompressRopeFreqBase => "compress_theta",
                Meta::RopeFreqBase => "theta",
                Meta::IndexerHeadCount => "index_heads",
                Meta::IndexerKeyLength => "index_dim",
                Meta::IndexerTopK => "index_topk",
                Meta::HyperConnectionCount => "hc",
                Meta::HyperConnectionSinkhornIterations => "hc_iters",
                Meta::HyperConnectionEpsilon => "hc_eps",
                Meta::CompressRatios => "ratios",
            }
        }

        fn block_prefix(&self, layer: usize) -> String {
            format!("layer{layer}/")
        }

        fn leaf(&self, w: Weight) -> &'static str {
            match w {
                Weight::AttnQA => "qa",
                Weight::AttnQANorm => "qa_norm",
                Weight::AttnQB => "qb",
                Weight::AttnKv => "kv",
                Weight::AttnKvANorm => "kv_norm",
                Weight::AttnOutputA => "oa",
                Weight::AttnOutputB => "ob",
                Weight::AttnSinks => "sinks",
                Weight::AttnNorm => "norm_attn",
                Weight::FfnNorm => "norm_ffn",
                Weight::IndexerQB => "idx_qb",
                Weight::IndexerProj => "idx_proj",
                Weight::FfnGateInp => "router",
                Weight::FfnGateTid2Eid => "hash_table",
                Weight::ExpProbsBias => "route_bias",
                Weight::RoutedExperts(Ffn::Gate) => "e_gate",
                Weight::RoutedExperts(Ffn::Up) => "e_up",
                Weight::RoutedExperts(Ffn::Down) => "e_down",
                Weight::SharedExpert(Ffn::Gate) => "s_gate",
                Weight::SharedExpert(Ffn::Up) => "s_up",
                Weight::SharedExpert(Ffn::Down) => "s_down",
                Weight::Compressor(Compressor::Attn, CompressorPart::Kv) => "c_kv",
                Weight::Compressor(Compressor::Attn, CompressorPart::Gate) => "c_gate",
                Weight::Compressor(Compressor::Attn, CompressorPart::Ape) => "c_ape",
                Weight::Compressor(Compressor::Attn, CompressorPart::Norm) => "c_norm",
                Weight::Compressor(Compressor::Indexer, CompressorPart::Kv) => "ic_kv",
                Weight::Compressor(Compressor::Indexer, CompressorPart::Gate) => "ic_gate",
                Weight::Compressor(Compressor::Indexer, CompressorPart::Ape) => "ic_ape",
                Weight::Compressor(Compressor::Indexer, CompressorPart::Norm) => "ic_norm",
                Weight::Hyper(Hyper::Attn, HyperPart::Fn) => "hca_fn",
                Weight::Hyper(Hyper::Attn, HyperPart::Base) => "hca_base",
                Weight::Hyper(Hyper::Attn, HyperPart::Scale) => "hca_scale",
                Weight::Hyper(Hyper::Ffn, HyperPart::Fn) => "hcf_fn",
                Weight::Hyper(Hyper::Ffn, HyperPart::Base) => "hcf_base",
                Weight::Hyper(Hyper::Ffn, HyperPart::Scale) => "hcf_scale",
            }
        }

        fn global(&self, g: Global) -> &'static str {
            match g {
                Global::Embedding => "embed",
                Global::OutputHead => "head",
                Global::OutputNorm => "final_norm",
                Global::OutputHyper(HyperPart::Fn) => "out_hc_fn",
                Global::OutputHyper(HyperPart::Base) => "out_hc_base",
                Global::OutputHyper(HyperPart::Scale) => "out_hc_scale",
            }
        }
    }

    /// The engine's own fixture must satisfy every rule a real model does —
    /// otherwise it is not a stand-in for one, and tests that pass against it
    /// prove nothing about the machinery they exercise.
    #[test]
    fn the_test_arch_satisfies_the_arch_invariants() {
        super::assert_arch_invariants(&TEST_ARCH);
    }

    /// The naming scheme is deliberately unlike DeepSeek's, so anything in the
    /// engine that hardcodes `blk.{n}.` or a real tensor name fails here.
    #[test]
    fn the_test_arch_names_are_deliberately_not_deepseeks() {
        assert_eq!(TEST_ARCH.weight(3, Weight::AttnQA), "layer3/qa");
        assert_eq!(TEST_ARCH.global(Global::Embedding), "embed");
        assert_eq!(TEST_ARCH.meta(Meta::BlockCount), "latent_moe_test.layers");
        let v4_shaped = super::every_weight()
            .iter()
            .any(|w| TEST_ARCH.weight(3, *w).starts_with("blk."));
        assert!(
            !v4_shaped,
            "the test arch must not borrow DeepSeek's naming"
        );
    }
}
