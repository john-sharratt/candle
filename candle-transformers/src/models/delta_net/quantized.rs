//! The production DeltaNet layer: quantized projections around the shared
//! mixer core.
//!
//! Only the four input projections and the output projection differ from the
//! reference — everything between them is [`delta_net_mix_spans`], the same
//! function the F32 reference calls. That is deliberate: the mixer's algebra
//! carries epsilon terms that are part of the arithmetic rather than guards
//! (see the read scale in [`super::mix`]), and a second transcription of it
//! is a second chance to get those wrong.
//!
//! The layer's FFN half is *not* here: a DeltaNet layer implements no engine
//! trait, and its FFN driver is ten model-family lines living beside the
//! family's weight containers (the `qwen35` lineage's
//! `quantized_delta_net.rs`).

use candle::{LiveTensor, Result, Tensor};

// Not CUDA-gated: `gpu_span` is defined in both configurations and is a
// zero-sized no-op without `profile` + `cuda`, so the call sites need no `cfg`
// of their own.
use crate::models::profile::gpu_span;
use crate::models::quantized_matmul::QMatMul;

use super::mix::{
    delta_net_mix_spans, DeltaNetConstants, DeltaNetLayerTable, DeltaNetProjections, DeltaNetSeq,
    DeltaNetState,
};
use super::types::DeltaNetDims;

/// A DeltaNet layer's production weights.
///
/// The projections are quantized; the elementwise constants — the conv
/// kernel, `ssm_a`, the `dt` bias, and the per-head norm gain — are F32,
/// because the recurrence accumulates and must not drift (the checkpoints
/// themselves declare `mamba_ssm_dtype: float32`).
pub struct QuantDeltaNetWeights {
    /// `[conv_dim, hidden]` fused `[Q|K|V]`.
    pub wqkv: QMatMul,
    /// `[value_dim, hidden]` output gate.
    pub wz: QMatMul,
    /// `[n_v_heads, hidden]`.
    pub w_beta: QMatMul,
    /// `[n_v_heads, hidden]`.
    pub w_alpha: QMatMul,
    /// `[hidden, value_dim]`.
    pub w_out: QMatMul,
    /// F32 constants — see the struct note on why these are not quantized.
    pub dt_bias: Tensor,
    pub a: Tensor,
    pub conv: Tensor,
    pub norm: Tensor,
}

/// One production DeltaNet layer over a `[T, hidden]` activation block, from
/// a carried state. Returns `[T, hidden]`.
///
/// [`quantized_delta_net_layer_forward_spans`] with a single span — the shape
/// the reference parity tests use.
pub fn quantized_delta_net_layer_forward<'w>(
    x: &LiveTensor<'w>,
    w: &QuantDeltaNetWeights,
    dims: &DeltaNetDims,
    state: &mut DeltaNetState,
    rms_eps: f64,
) -> Result<LiveTensor<'w>> {
    let t = x.dim(0)?;
    let mut one = [DeltaNetSeq {
        start: 0,
        len: t,
        state,
    }];
    quantized_delta_net_layer_forward_spans(x, w, dims, &mut one, rms_eps, None)
}

/// One production DeltaNet layer over a `[T, hidden]` activation block holding
/// **several** sequences, named by `seqs`. Returns `[T, hidden]`; each
/// sequence's state is advanced in place.
///
/// `T` is a flat token count, not a sequence length: the caller packs however
/// many rows it has and says where each sequence begins. The five projections
/// here are row-wise, so they run **once over the whole block** — which is the
/// point, because each one re-reads its entire weight and a decode step is
/// weight-bandwidth-bound. Only the two carried steps inside
/// [`delta_net_mix_spans`] are per sequence.
///
/// # Why the mixer runs in F32 whatever the activations are
///
/// `S` is a running sum carried across every token of a sequence — the one
/// value in the stack with no bound on how many additions it accumulates. In
/// half precision it drifts, and the drift is unbounded in context length,
/// which is the opposite of what the O(1)-error design is for. The projections
/// stay in the wave's activation dtype (their kernels want it); the mixer's
/// inputs are widened at this boundary and the result narrowed on the way back
/// out to the output projection. That is two conversions per DeltaNet layer,
/// and they are the arithmetic rather than an oversight.
pub fn quantized_delta_net_layer_forward_spans<'w>(
    x: &LiveTensor<'w>,
    w: &QuantDeltaNetWeights,
    dims: &DeltaNetDims,
    seqs: &mut [DeltaNetSeq<'_>],
    rms_eps: f64,
    table: Option<&DeltaNetLayerTable>,
) -> Result<LiveTensor<'w>> {
    let act = x.dtype();
    let wide = |t: LiveTensor<'w>| -> Result<LiveTensor<'w>> {
        if t.dtype() == candle::DType::F32 {
            Ok(t)
        } else {
            t.to_dtype(candle::DType::F32)
        }
    };
    // `forward_live`, not `Module::forward`: the input is the layer's own
    // wave-scoped activation, and the projections' outputs belong in the same
    // arena. `Module` takes `&Tensor` on purpose — a module may retain what it
    // is given — so it cannot be the one to see this.
    let g_proj = gpu_span("dn:proj", x.device());
    let p = DeltaNetProjections {
        qkv: wide(w.wqkv.forward_live(x)?)?,
        z: wide(w.wz.forward_live(x)?)?,
        beta_lin: wide(w.w_beta.forward_live(x)?)?,
        alpha_lin: wide(w.w_alpha.forward_live(x)?)?,
    };
    g_proj.end();
    let c = DeltaNetConstants {
        dt_bias: &w.dt_bias,
        a: &w.a,
        conv: &w.conv,
        norm: &w.norm,
    };
    let g_mix = gpu_span("dn:mix", x.device());
    let gated = delta_net_mix_spans(&p, &c, dims, seqs, rms_eps, table)?;
    let gated = if gated.dtype() == act {
        gated
    } else {
        gated.to_dtype(act)?
    };
    g_mix.end();

    let g_out = gpu_span("dn:out_proj", x.device());
    let out = w.w_out.forward_live(&gated)?;
    g_out.end();
    Ok(out)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::models::batch_test::test_helpers::hf_get;
    use crate::models::delta_net::mix::{delta_net_layer_forward, DeltaNetWeights};
    use crate::models::qwen35::quantized_weights::{load_quantized_model, QuantLayerMix};
    use candle::quantized::{gguf_file::Content, Int8Mode};
    use candle::{DType, Device, Tensor};
    use hf_hub::RepoType;
    use std::io::{BufReader, Seek, SeekFrom};

    /// The production DeltaNet layer against the F32 reference, on the real
    /// Qwen3.5-9B Q6_K checkpoint and at its real geometry — which is the
    /// point of using the 9B rather than the 0.8B here: it has 16 K heads
    /// against 32 V heads, so the GQA broadcast inside the mixer is live
    /// (the 0.8B is 16/16 and cannot exercise it).
    ///
    /// The reference weights are produced by dequantizing the production
    /// ones, so both sides see the *same* numbers and the only difference
    /// under test is the quantized projection kernel against a plain F32
    /// matmul. That keeps the tolerance meaningful: it is Q6_K projection
    /// error carried through the recurrence, not a mismatch of intent.
    #[test]
    #[ignore = "reads the pinned Qwen3.5-9B GGUF from the HF cache (7.5 GB) and needs a GPU"]
    fn quantized_delta_net_layer_matches_the_f32_reference_on_real_weights() -> Result<()> {
        let path = hf_get(
            "unsloth/Qwen3.5-9B-GGUF",
            RepoType::Model,
            "3885219b6810b007914f3a7950a8d1b469d598a5",
            "Qwen3.5-9B-Q6_K.gguf",
        )?;
        let device = Device::new_cuda(0)?;
        let mut reader = BufReader::new(std::fs::File::open(&path)?);
        let content = Content::read(&mut reader)?;
        reader.seek(SeekFrom::Start(0))?;
        // The 9B is dense, so it needs no expert cache.
        let model = load_quantized_model(&content, &mut reader, &device, Int8Mode::Off, |_, _| {
            Ok(None)
        })?;

        let dims = model.cfg.delta_net;
        println!(
            "{} layers ({} attn / {} deltanet), hidden {}, deltanet {}k/{}v heads x {}",
            model.cfg.num_layers,
            model.cfg.n_attention_layers(),
            model.cfg.n_delta_net_layers(),
            model.cfg.hidden_size,
            dims.n_k_heads,
            dims.n_v_heads,
            dims.head_dim,
        );
        assert!(
            dims.n_v_heads > dims.n_k_heads,
            "fixture no longer exercises the GQA broadcast ({}k/{}v)",
            dims.n_k_heads,
            dims.n_v_heads
        );

        let qw = model
            .layers
            .iter()
            .find_map(|l| match &l.mix {
                QuantLayerMix::DeltaNet(w) => Some(w),
                _ => None,
            })
            .expect("the stack has DeltaNet layers");

        // Same numbers on both sides; only the projection kernel differs.
        let reference = DeltaNetWeights {
            wqkv: qw.wqkv.dequantize()?,
            wz: qw.wz.dequantize()?,
            w_beta: qw.w_beta.dequantize()?,
            w_alpha: qw.w_alpha.dequantize()?,
            w_out: qw.w_out.dequantize()?,
            dt_bias: qw.dt_bias.clone(),
            a: qw.a.clone(),
            conv: qw.conv.clone(),
            norm: qw.norm.clone(),
        };

        // A multi-token block, so the chunked scan and the conv tail both run.
        let t = 40usize;
        let x =
            Tensor::randn(0f32, 1.0, (t, model.cfg.hidden_size), &device)?.to_dtype(DType::F32)?;
        let eps = model.cfg.rms_norm_eps;

        let mut s_prod = DeltaNetState::zeros(&dims, &device)?;
        let got = quantized_delta_net_layer_forward(&x, qw, &dims, &mut s_prod, eps)?;
        let mut s_ref = DeltaNetState::zeros(&dims, &device)?;
        let want = delta_net_layer_forward(&x, &reference, &dims, &mut s_ref, eps)?;

        let rel = |a: &Tensor, b: &Tensor| -> Result<f32> {
            let diff = a.sub(b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let scale = b.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            Ok(diff / scale.max(1e-6))
        };
        let out_rel = rel(&got, &want)?;
        let state_rel = rel(&s_prod.s, &s_ref.s)?;
        println!("output rel {out_rel:.5}, carried state rel {state_rel:.5}");

        // Q6_K projection error through a 40-token scan. Loose enough to be
        // about quantization, tight enough that a wrong split, a dropped
        // scale or a transposed weight cannot pass.
        assert!(
            out_rel < 0.05,
            "production layer diverged from the reference: rel {out_rel}"
        );
        assert!(
            state_rel < 0.05,
            "carried state diverged from the reference: rel {state_rel}"
        );
        // The conv tail is elementwise on both paths and must agree closely.
        assert!(rel(&s_prod.conv_tail, &s_ref.conv_tail)? < 0.02);
        Ok(())
    }
}
