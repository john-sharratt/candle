//! Groups a layer's buffers into the chains a phase can run, which is how the
//! plan combines them.
//!
//! A phase costs the largest of the chains that can open it, because a layer
//! is one kind of mixer or one kind of FFN and never both; each chain costs the
//! sum of its own buffers. Preludes are the exception: the Gated Residual
//! pre-mix runs ahead of whichever mixer or FFN the layer has, so a phase costs
//! its preludes walked first and then the largest other chain. Each alternative
//! prices only the geometry that runs it. A dense FFN prices nothing on a MoE
//! model, and the DeltaNet chain prices nothing on an all-attention stack.

use super::*;

/// A run of buffers that one layer can allocate inside one generation.
///
/// The distinction a phase alone cannot make. Within a phase the plan charges
/// the **union** of the chains that can run — a mixed wave opens one attention
/// generation and runs its decode group and its prefill group inside it, so
/// both chains' buffers are live at once. That reasoning does not extend to a
/// hybrid's two *layer kinds*: a layer is an attention layer or a DeltaNet
/// layer, never both, and a generation is one layer's phase. Summing them would
/// price a wave for a layer that does not exist.
///
/// So a phase costs the **largest** of the chains that can open it, and each
/// chain costs the **sum** of its own buffers.
///
/// This was wrong in the other direction before it was named: the DeltaNet
/// chain was undeclared and `priced_intermediate` folded its conv width into
/// the *FFN* phase — which a DeltaNet layer runs identically to an attention
/// layer, and where its mixer carves nothing. That is 49.1 MiB of a 139.3 MiB
/// FFN span on the 0.8B, charged in a phase the buffer never appears in.
///
/// **A prelude is neither.** The Gated Residual's pre-mix runs at the head of
/// the attention phase whichever mixer the layer has, and at the head of the
/// FFN phase — so it is not an alternative to the mixers but the ground each of
/// them starts from. A phase therefore costs its preludes, walked first, plus
/// the largest of its other chains walked on from there ([`Chain::is_prelude`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum Chain {
    /// The attention phase's Gated Residual pre-mix — a prelude.
    HyperAttn,
    /// The FFN phase's Gated Residual pre-mix — a prelude.
    HyperFfn,
    /// The forward's setup tables — the token ids and the ragged prefill
    /// tables — carved before the first layer and read by every one, in the
    /// forward phase. A prelude: the forward carves them whether or not it
    /// reaches the head.
    ForwardSetup,
    /// The head's Gated Residual output mix, in the forward phase — a prelude.
    HyperHead,
    /// An attention layer's mixer: norm through `o_proj`.
    Attention,
    /// A DeltaNet layer's mixer: the four projections through `w_out`. Shares
    /// the attention generation, because it is the same scope — the thing a
    /// layer opens before its FFN.
    DeltaNet,
    /// The **MoE** FFN: the router, the expert gather, the grouped GEMMs and
    /// the combine.
    Ffn,
    /// The draft head's pass: its input assembly, then — walked on from there,
    /// see [`WavePlan::phase_bytes`] — the attention prelude and the attention
    /// chain, which the head's block runs as a trunk attention layer does. An
    /// alternative to the trunk's mixers, not an addition: the pass opens the
    /// attention generation after the trunk's last layer has closed it.
    HeadPass,
    /// The PLE block: its own attention generation, opened ahead of its layer's
    /// mixer and closed before it, with no prelude.
    Ple,
    /// The **dense** FFN: norm, one fused `[gate|up]` GEMM, SiLU, multiply.
    ///
    /// The same `max` relationship the two mixers have, one phase down. A layer
    /// dispatches to `QuantFfn::Dense` or `QuantFfn::Moe`, never both, and the
    /// two chains have almost nothing in common — the dense one has no router,
    /// no gather, no expert replication, and carries its intermediates in the
    /// compute dtype rather than the accumulate one. Pricing a dense model
    /// against the MoE list is not a margin, it is a different chain: on the
    /// 0.8B it over-charged `[gate|up]` by 30.1 MiB and *under*-charged the
    /// SwiGLU output by 21.6 MiB, and only passed because the two errors
    /// partly cancelled.
    DenseFfn,
    /// Per-forward setup, read from every layer.
    Forward,
}

impl Chain {
    /// The generation this chain allocates from.
    pub fn phase(&self) -> LayerPhase {
        match self {
            Self::HyperAttn | Self::Attention | Self::DeltaNet | Self::HeadPass | Self::Ple => {
                LayerPhase::Attention
            }
            Self::HyperFfn | Self::Ffn | Self::DenseFfn => LayerPhase::Ffn,
            Self::ForwardSetup | Self::HyperHead | Self::Forward => LayerPhase::Forward,
        }
    }

    /// Whether this chain runs ahead of every other chain in its phase rather
    /// than instead of them — see the type's docs.
    pub fn is_prelude(&self) -> bool {
        matches!(
            self,
            Self::HyperAttn | Self::HyperFfn | Self::ForwardSetup | Self::HyperHead
        )
    }
}
