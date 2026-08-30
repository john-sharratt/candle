//! Assembling a whole `QuantLayer` from a slot's views plus the layer's
//! resident residue.
//!
//! # Why the cache hands back a whole layer
//!
//! The alternative — splice slot views into the projections the forward reads
//! from `model.layers[li]` — cannot be done cheaply. `QCudaStorage`'s `Clone`
//! is documented as always-owned because *"`CudaSlice::clone` is a
//! device-to-device copy"*, so rebuilding a `QuantDeltaNetWeights` per layer
//! per wave would copy ~115M parameters per layer per forward on the 27B.
//!
//! Assembling once **per tenancy** — when a layer lands in a slot, not when it
//! is read — costs nothing on the hot path and changes no downstream signature
//! at all: the forward keeps taking `&QuantLayer` and neither the DeltaNet path
//! nor the attention path knows a slot exists.
//!
//! # What is cheap to clone here, and why that is the whole trick
//!
//! | Part | Clone cost |
//! |---|---|
//! | `QMatMul` over a slot | **not cloned** — built by `build_layer_view` from an address |
//! | `RmsNorm` | an `Arc<QTensor>` refcount bump |
//! | `Tensor` (`dt_bias`, `a`, `conv`, `norm`) | a handle, not a buffer |
//! | `w_beta`, `w_alpha` | sub-tile, so resident and dense — cloned once at assembly |
//!
//! The residue is ~0.1% of a layer, and only the *handles* to it are copied, so
//! an assembly is a few dozen refcount bumps and no device traffic.

use candle::Result;

use super::descriptor::{FfnForm, LayerTensor, MixKind};
use super::view::StreamedLayer;
use crate::models::quantized_mlp::QuantizedMlp;
use crate::models::qwen35::quantized_weights::{
    QuantAttentionWeights, QuantLayer, QuantLayerMix, ResidentResidue,
};

/// Build the layer the forward will read, over `streamed`'s slot views.
///
/// `residue` is the part of the layer that never leaves VRAM — norms, the
/// DeltaNet F32 constants, and the sub-tile gates — held once per layer for the
/// life of the process and shared into every assembly by handle.
pub fn assemble_layer(
    mut streamed: StreamedLayer,
    residue: &ResidentResidue,
    kind: MixKind,
    ffn: FfnForm,
) -> Result<QuantLayer> {
    // Consumed, not borrowed: the assembled layer owns its projections and a
    // `QMatMul` cannot be cloned into place without copying the weight.
    macro_rules! take {
        ($role:expr) => {
            streamed.take($role)?
        };
    }

    let mix = match kind {
        MixKind::DeltaNet => {
            let dn = residue.delta_net()?;
            QuantLayerMix::DeltaNet(crate::models::delta_net::QuantDeltaNetWeights {
                wqkv: take!(LayerTensor::Wqkv),
                wz: take!(LayerTensor::Wz),
                w_out: take!(LayerTensor::WOut),
                w_beta: dn.w_beta.clone(),
                w_alpha: dn.w_alpha.clone(),
                dt_bias: dn.dt_bias.clone(),
                a: dn.a.clone(),
                conv: dn.conv.clone(),
                norm: dn.norm.clone(),
            })
        }
        MixKind::Attention => {
            let at = residue.attention()?;
            QuantLayerMix::Attention(QuantAttentionWeights {
                wq: take!(LayerTensor::Wq),
                wk: take!(LayerTensor::Wk),
                wv: take!(LayerTensor::Wv),
                wo: take!(LayerTensor::Wo),
                q_norm: at.q_norm.clone(),
                k_norm: at.k_norm.clone(),
            })
        }
    };

    let down = take!(LayerTensor::FfnDown);
    let mlp = match ffn {
        FfnForm::Fused => {
            QuantizedMlp::from_repacked(Some(take!(LayerTensor::FfnGateUp)), None, None, down)?
        }
        FfnForm::Split => QuantizedMlp::from_repacked(
            None,
            Some(take!(LayerTensor::FfnGate)),
            Some(take!(LayerTensor::FfnUp)),
            down,
        )?,
    };

    Ok(QuantLayer::from_streamed(
        residue.attn_norm.clone(),
        residue.post_attn_norm.clone(),
        mix,
        mlp,
    ))
}
