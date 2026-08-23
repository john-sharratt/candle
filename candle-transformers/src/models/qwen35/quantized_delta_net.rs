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
use candle::{DType, Device};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{begin_wave, LayerPhase};

#[cfg(feature = "cuda")]
use super::quantized_weights::{QuantFfn, QuantLayer};
#[cfg(feature = "cuda")]
use crate::models::profile::gpu_span;
#[cfg(feature = "cuda")]
use crate::models::tensor_cat::TensorCat;
#[cfg(feature = "cuda")]
use crate::models::wave_buffers::wave_root;

/// `orig_dtype` is the dtype the residual stream must come back in, captured
/// by the caller before the mixing half ran.
#[cfg(feature = "cuda")]
pub fn quantized_delta_net_ffn(
    layer: &QuantLayer,
    x: &mut TensorCat,
    act_dtype: DType,
    orig_dtype: DType,
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
    let mut h = {
        let mode = layer.ffn_int8mode();
        let acts = layer.post_attn_norm.forward_dynamic(
            x.as_cat_tensor(),
            mode,
            wave_root(ffn_wave.as_ref()),
        )?;
        match &layer.ffn {
            QuantFfn::Dense(m) => m.forward_dynamic(&acts, mlp_dtype)?,
            QuantFfn::Moe(m) => m.forward_dynamic(acts, mlp_dtype, ffn_wave.as_ref())?,
        }
    };
    h.to_dtype_mut(orig_dtype)?;
    x.to_dtype_mut(orig_dtype)?;
    x.add_mut(&h)?;
    g_ffn.end();
    // `h` borrows `ffn_wave`, so the compiler already refuses any drop order
    // but this one; both die at the end of the function.
    Ok(())
}
