//! The FFN half of one DeltaNet layer, over the whole combined buffer.
//!
//! The mixing half — quantized projections around the shared mixer core —
//! is the generic [`crate::models::delta_net::quantized`] driver; what stays
//! with the model family is this: the ten lines that drive the layer's FFN,
//! because they dispatch on the family's own [`QuantFfn`] (dense MLP vs the
//! shared-expert MoE block).
//!
//! A DeltaNet layer implements **no** engine trait: `BatchedAttentionLayer`'s
//! contract is "project Q/K/V and I attend them against a KV cache", and this
//! layer has neither, so implementing it would mean stubbing most of it out.
//! Its FFN, though, is an ordinary SwiGLU over exactly the buffer every other
//! layer's FFN sees — so rather than reshape the shared trait around a hybrid,
//! the ten lines that drive it live here.

use candle::Result;
#[cfg(feature = "cuda")]
use candle::{quantized::Int8Mode, DType, Device};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{begin_wave, LayerPhase};

#[cfg(feature = "cuda")]
use super::quantized_weights::{QuantFfn, QuantLayer};
#[cfg(feature = "cuda")]
use crate::models::batched_layer::add_ffn_residual;
#[cfg(feature = "cuda")]
use crate::models::lora::LayerLora;
#[cfg(feature = "cuda")]
use crate::models::profile::gpu_span;
#[cfg(feature = "cuda")]
use crate::models::tensor_cat::TensorCat;
#[cfg(feature = "cuda")]
use crate::models::wave_buffers::wave_root;

/// `orig_dtype` is the dtype the residual stream must come back in, captured
/// by the caller before the mixing half ran.
///
/// `lora` carries this layer's adapter pairs. Only the three FFN roles are read
/// here — a DeltaNet layer has no q/k/v/o to adapt, and the adapter this engine
/// loads reflects that: its attention pairs exist on the eight attention layers
/// and nowhere else, while its FFN pairs cover all thirty-two.
#[cfg(feature = "cuda")]
pub fn quantized_delta_net_ffn(
    layer: &QuantLayer,
    x: &mut TensorCat,
    act_dtype: DType,
    orig_dtype: DType,
    lora: LayerLora<'_>,
    decode_tokens: usize,
) -> Result<()> {
    // MLP intermediates can exceed F16's range, so accumulate in BF16 there.
    let mlp_dtype = if act_dtype == DType::F16 {
        DType::BF16
    } else {
        act_dtype
    };
    // The FFN's own transient scope: it spans the FFN through the residual add
    // that consumes its result, after which nothing it produced is live.
    let ffn_wave = match x.as_cat_tensor().device() {
        Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Ffn)?),
        _ => None,
    };
    let g_ffn = gpu_span("dn:ffn", x.as_cat_tensor().device());
    // An adapted FFN norms to float, for the reason `Qwen35AttentionLayer`'s
    // `int8mode` states in full: the fused RMSNorm→quantize kernel emits q8a128
    // and the adapter's `A` matmul needs the float that went into it.
    let mode = if lora.is_empty() {
        layer.ffn_int8mode()
    } else {
        Int8Mode::Off
    };
    // The FFN's input, before the norm quantizes it. Everything downstream in
    // this block is bounded by whether this was already bad.
    x.as_cat_tensor().assert("ffn.in");
    let acts = layer.post_attn_norm.forward_dynamic(
        x.as_cat_tensor(),
        mode,
        wave_root(ffn_wave.as_ref()),
    )?;
    match &layer.ffn {
        // The dense MLP's down projection stores `orig_dtype`, so its result
        // lands in the residual with no narrowing pass between.
        QuantFfn::Dense(m) => {
            let h = m.forward_dynamic_adapted(&acts, mlp_dtype, orig_dtype, lora)?;
            add_ffn_residual(x.as_cat_tensor_mut(), &h)?;
        }
        // The shared+routed combine, its narrowing and the residual add are one
        // launch. The FFN computes in a promoted dtype because "MLP
        // intermediates can exceed F16's range", and whether narrowing back is
        // lossless is a property of the DATA — so the residual is checked after
        // it, where an out-of-range value that became `inf` lands.
        QuantFfn::Moe(m) => {
            m.forward_residual(
                x.as_cat_tensor_mut(),
                acts,
                mlp_dtype,
                decode_tokens,
                ffn_wave.as_ref(),
            )?;
            x.as_cat_tensor().assert("ffn.moe_residual");
        }
    }
    g_ffn.end();
    Ok(())
}
