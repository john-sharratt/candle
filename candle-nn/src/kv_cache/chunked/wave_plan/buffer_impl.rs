//! Answers, for each buffer, which phase, chain and mode it belongs to.
//!
//! The plan partitions buffers by phase and by chain, so every variant must
//! say where it lives: whether it is a single-stream draft head's buffer,
//! which run of buffers it belongs to, and when it is live within its phase.
//! These answers are what keep the phase totals, the chain totals and the
//! census-pinned tests describing the same set of allocations.

use super::*;

impl WaveBuffer {
    /// Whether this buffer is a single-stream draft head's — priced only where
    /// [`ModelGeometry::mtp_head`] is set.
    pub fn is_mtp(&self) -> bool {
        matches!(
            self,
            Self::MtpRowEmbeds
                | Self::MtpTrunkNorm
                | Self::MtpShift
                | Self::MtpEmbedNorm
                | Self::MtpHiddenNorm
                | Self::MtpInputCat
                | Self::MtpInputOperand
                | Self::MtpInput
        )
    }

    /// Which run of buffers this one belongs to.
    pub fn chain(&self) -> Chain {
        match self {
            Self::HyperAttnNorm
            | Self::HyperAttnNormOperand
            | Self::HyperAttnLowRank
            | Self::HyperAttnGateAct
            | Self::HyperAttnGateOperand
            | Self::HyperAttnGateRaw
            | Self::HyperAttnMixed => Chain::HyperAttn,
            Self::HyperFfnNorm
            | Self::HyperFfnNormOperand
            | Self::HyperFfnLowRank
            | Self::HyperFfnGateAct
            | Self::HyperFfnGateOperand
            | Self::HyperFfnGateRaw
            | Self::HyperFfnMixed => Chain::HyperFfn,
            Self::HyperHeadNorm
            | Self::HyperHeadNormOperand
            | Self::HyperHeadLowRank
            | Self::HyperHeadGateAct
            | Self::HyperHeadGateOperand
            | Self::HyperHeadGateRaw
            | Self::HyperHeadMixed => Chain::HyperHead,
            Self::DeltaNetNorm
            | Self::DeltaNetQkvOperand
            | Self::DeltaNetQkvProj
            | Self::DeltaNetZOperand
            | Self::DeltaNetZProj
            | Self::DeltaNetBetaOperand
            | Self::DeltaNetBetaProj
            | Self::DeltaNetAlphaOperand
            | Self::DeltaNetAlphaProj
            | Self::DeltaNetLiveOperand
            | Self::DeltaNetLiveProjection
            | Self::DeltaNetLiveSplit
            | Self::DeltaNetLiveSpanPtrs
            | Self::DeltaNetLiveSpanExtents
            | Self::DeltaNetLiveConv
            | Self::DeltaNetLiveScanOut
            | Self::DeltaNetLiveScanU
            | Self::DeltaNetLiveScanW
            | Self::DeltaNetLiveScanKq
            | Self::DeltaNetLiveDecay
            | Self::DeltaNetLiveNormGate
            | Self::DeltaNetLiveOutOperand
            | Self::DeltaNetLiveOut => Chain::DeltaNet,
            Self::DenseFfnNorm | Self::DenseGateUp | Self::DenseSilu | Self::DenseSwiglu => {
                Chain::DenseFfn
            }
            Self::HeadShift
            | Self::HeadHiddenNorm
            | Self::HeadHiddenOperand
            | Self::HeadHiddenProj
            | Self::HeadInput
            | Self::MtpTrunkNorm
            | Self::MtpShift
            | Self::MtpEmbedNorm
            | Self::MtpHiddenNorm
            | Self::MtpInputCat
            | Self::MtpInputOperand
            | Self::MtpInput => Chain::HeadPass,
            Self::MtpRowEmbeds => Chain::Forward,
            Self::HeadEmbedNorm | Self::HeadEmbedOperand | Self::HeadEmbedProj => Chain::HeadPass,
            Self::PleRows | Self::PleKeyValue | Self::PleNormalized | Self::PleSpanTable => {
                Chain::Ple
            }
            Self::HeadNorm
            | Self::HeadNormF32
            | Self::HeadLogits
            | Self::HeadLogitsF32
            | Self::AcceptRows
            | Self::StencilBitmaps
            | Self::HyperResidual
            | Self::HeadRowEmbeds
            | Self::HeadScoredResidual
            | Self::HeadScoredRows => Chain::Forward,
            Self::PrefillCuSeqlens
            | Self::PrefillQLens
            | Self::PrefillKvLens
            | Self::GlueWriteTables
            | Self::ForwardResidual
            | Self::DeltaNetDecodeTable
            | Self::RowTokenIds
            | Self::ImageRows
            | Self::ImageTurns => Chain::ForwardSetup,
            other => match other.phase() {
                LayerPhase::Attention => Chain::Attention,
                LayerPhase::Ffn => Chain::Ffn,
                LayerPhase::Forward => Chain::Forward,
            },
        }
    }

    /// When this buffer is live within its phase.
    pub fn phase(&self) -> LayerPhase {
        match self {
            Self::DeltaNetNorm
            | Self::DeltaNetQkvOperand
            | Self::DeltaNetQkvProj
            | Self::DeltaNetZOperand
            | Self::DeltaNetZProj
            | Self::DeltaNetBetaOperand
            | Self::DeltaNetBetaProj
            | Self::DeltaNetAlphaOperand
            | Self::DeltaNetAlphaProj
            | Self::DeltaNetLiveOperand
            | Self::DeltaNetLiveProjection
            | Self::DeltaNetLiveSplit
            | Self::DeltaNetLiveSpanPtrs
            | Self::DeltaNetLiveSpanExtents
            | Self::DeltaNetLiveConv
            | Self::DeltaNetLiveScanOut
            | Self::DeltaNetLiveScanU
            | Self::DeltaNetLiveScanW
            | Self::DeltaNetLiveScanKq
            | Self::DeltaNetLiveDecay
            | Self::DeltaNetLiveNormGate
            | Self::DeltaNetLiveOutOperand
            | Self::DeltaNetLiveOut
            | Self::HyperAttnNorm
            | Self::HyperAttnNormOperand
            | Self::HyperAttnLowRank
            | Self::HyperAttnGateAct
            | Self::HyperAttnGateOperand
            | Self::HyperAttnGateRaw
            | Self::HyperAttnMixed
            | Self::HeadShift
            | Self::HeadHiddenNorm
            | Self::HeadHiddenOperand
            | Self::HeadHiddenProj
            | Self::HeadInput
            | Self::HeadEmbedNorm
            | Self::HeadEmbedOperand
            | Self::HeadEmbedProj
            | Self::MtpTrunkNorm
            | Self::MtpShift
            | Self::MtpEmbedNorm
            | Self::MtpHiddenNorm
            | Self::MtpInputCat
            | Self::MtpInputOperand
            | Self::MtpInput
            | Self::PleRows
            | Self::PleKeyValue
            | Self::PleNormalized
            | Self::PleSpanTable => LayerPhase::Attention,
            Self::AttnNorm
            | Self::QkvProjection
            | Self::QSplit
            | Self::QNormIn
            | Self::QNormOut
            | Self::QHeadsPacked
            | Self::GateSplit
            | Self::GateSigmoid
            | Self::GatedContext
            | Self::QRotaryPermute
            | Self::KRotaryPermute
            | Self::KSplit
            | Self::KNormIn
            | Self::KNormOut
            | Self::KHeadsPacked
            | Self::VContiguous
            | Self::QBias
            | Self::KBias
            | Self::VBias
            | Self::PrefillSlotHeaders
            | Self::AttnOutput
            | Self::PrefillKvStage
            | Self::DecodeContext
            | Self::OProjOperand
            | Self::OProjOutput
            | Self::QsaSelection => LayerPhase::Attention,
            Self::HyperFfnNorm
            | Self::HyperFfnNormOperand
            | Self::HyperFfnLowRank
            | Self::HyperFfnGateAct
            | Self::HyperFfnGateOperand
            | Self::HyperFfnGateRaw
            | Self::HyperFfnMixed
            | Self::FfnNorm
            | Self::SharedGateUp
            | Self::SharedAct
            | Self::SharedGated
            | Self::SharedDownOperand
            | Self::SharedDown
            | Self::SharedGateLogits
            | Self::SharedGateSigmoid
            | Self::SharedGatedOut
            | Self::RouterLogits
            | Self::LookAheadLogits
            | Self::RouteWeights
            | Self::RouteIndices
            | Self::RoutingTokenIds
            | Self::RoutingWeightIds
            | Self::RoutingPermutation
            | Self::RoutingTokenStarts
            | Self::MoeGather
            | Self::GateGemm
            | Self::UpGemm
            | Self::SwigluAct
            | Self::DownGemm
            | Self::MoeCombine
            | Self::MoeSharedSum
            | Self::MoeResultCast
            | Self::FfnNormOperand
            | Self::DenseFfnNorm
            | Self::DenseGateUp
            | Self::DenseSilu
            | Self::DenseSwiglu => LayerPhase::Ffn,
            Self::HyperHeadNorm
            | Self::HyperHeadNormOperand
            | Self::HyperHeadLowRank
            | Self::HyperHeadGateAct
            | Self::HyperHeadGateOperand
            | Self::HyperHeadGateRaw
            | Self::HyperHeadMixed
            | Self::HeadNorm
            | Self::HeadNormF32
            | Self::HeadLogits
            | Self::HeadLogitsF32
            | Self::AcceptRows
            | Self::StencilBitmaps
            | Self::HyperResidual
            | Self::HeadRowEmbeds
            | Self::HeadScoredResidual
            | Self::HeadScoredRows
            | Self::MtpRowEmbeds
            | Self::PrefillCuSeqlens
            | Self::PrefillQLens
            | Self::PrefillKvLens
            | Self::GlueWriteTables
            | Self::ForwardResidual
            | Self::DeltaNetDecodeTable
            | Self::RowTokenIds
            | Self::ImageRows
            | Self::ImageTurns => LayerPhase::Forward,
        }
    }
}
