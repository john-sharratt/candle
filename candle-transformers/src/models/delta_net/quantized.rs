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
use crate::models::stacked_proj::project_grouped;

use super::mix::{
    delta_net_mix_spans, DeltaNetConstants, DeltaNetLayerTable, DeltaNetProjections, DeltaNetSeq,
    DeltaNetState,
};
use super::types::{DeltaNetDims, ZGate};

/// A DeltaNet layer's production weights.
///
/// The projections are quantized; the elementwise constants — the conv
/// kernel, `ssm_a`, the `dt` bias, and the per-head norm gain — are F32,
/// because the recurrence accumulates and must not drift (the checkpoints
/// themselves declare `mamba_ssm_dtype: float32`).
pub struct QuantDeltaNetWeights {
    /// The input projections — `[Q|K|V]`, `z`, `β`, `α` in that order —
    /// covering `conv_dim + value_dim + 2·n_v_heads` rows between them.
    ///
    /// All four contract the same activation over `hidden`, so a loader that can
    /// row-concatenate their weights hands over **one** and the layer issues one
    /// GEMM where it issued four.
    ///
    /// A list rather than one weight because stacking is a byte append over the
    /// GGUF block layout and so must precede the KO repack, which a loader
    /// applying a *per-tensor* narrowing schedule cannot do for tensors it
    /// narrows differently. qwen4exp stacks all four; qwen35's streaming loader
    /// keeps them apart because its schedule narrows `attn_qkv` alone and a
    /// stacked weight takes one target. `stacked_proj::project_grouped` walks
    /// the same code for either and the unstacked case pays nothing.
    pub proj: Vec<QMatMul>,
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
    zgate: ZGate,
) -> Result<LiveTensor<'w>> {
    let t = x.dim(0)?;
    // One carried state, as in the float single-span path: `s` is read and
    // written in place, the conv tail lands in a scratch buffer that is folded
    // back below. There is no wave to roll back here.
    let out = state.solo_out()?;
    let mut one = [DeltaNetSeq {
        start: 0,
        len: t,
        state,
        out,
        stash: None,
    }];
    let mixed =
        quantized_delta_net_layer_forward_spans(x, w, dims, &mut one, rms_eps, None, zgate)?;
    let [seq] = one;
    seq.state.absorb_solo(&seq.out)?;
    Ok(mixed)
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
/// which is the opposite of what the O(1)-error design is for.
///
/// Holding that boundary costs no tensor passes, because both matmuls that
/// straddle the mixer name their own width. The four projections ask the KO
/// kernel to store F32 out of the F32 accumulator it already has, and the
/// output projection reads the mixer's F32 directly — its activation quantizer
/// takes F32 natively — while storing the dtype the residual stream wants.
/// Every conversion on this path is a kernel's own store; none is a pass over a
/// tensor (hot-path invariant 1).
pub fn quantized_delta_net_layer_forward_spans<'w>(
    x: &LiveTensor<'w>,
    w: &QuantDeltaNetWeights,
    dims: &DeltaNetDims,
    seqs: &mut [DeltaNetSeq<'_>],
    rms_eps: f64,
    table: Option<&DeltaNetLayerTable>,
    zgate: ZGate,
) -> Result<LiveTensor<'w>> {
    let act = x.dtype();
    // `forward_live`, not `Module::forward`: the input is the layer's own
    // wave-scoped activation, and the projections' outputs belong in the same
    // arena. `Module` takes `&Tensor` on purpose — a module may retain what it
    // is given — so it cannot be the one to see this.
    let g_proj = gpu_span("dn:proj", x.device());
    // F32 out of the matmul itself, not a cast after it. The recurrence carries
    // `S` in F32, so these four have always been consumed wide — but asking the
    // KO kernel to store narrow and widening afterwards spent a full-tensor pass
    // per projection (four launches per DeltaNet layer, the single largest
    // source of `cast_f16_f32` in a prefill sweep) to recover a number the F32
    // accumulator had already computed and thrown away on the store.
    //
    // One GEMM launch per stacked group, then ONE ragged scatter for the split —
    // a single arena bump for every part, no per-part copy, no memset.
    let hv = dims.n_v_heads;
    let widths = [dims.conv_dim(), dims.value_dim(), hv, hv];
    let mut take = project_grouped(
        x,
        &w.proj,
        &widths,
        candle::DType::F32,
        "delta-net input projections",
    )?
    .into_iter();
    let p = DeltaNetProjections {
        qkv: take.next().expect("four parts requested"),
        z: take.next().expect("four parts requested"),
        beta_lin: take.next().expect("four parts requested"),
        alpha_lin: take.next().expect("four parts requested"),
    };
    g_proj.end();
    // A span that will have to rewind keeps this layer's operands, copied out
    // of the wave arena that is reclaimed at the end of the forward. The
    // destination is already allocated — see `DeltaNetSeq::stash`.
    for s in seqs.iter() {
        if let Some(slot) = s.stash.as_ref() {
            slot.ops.capture(&p, s.start, slot.row, s.len)?;
        }
    }
    let c = DeltaNetConstants {
        dt_bias: &w.dt_bias,
        a: &w.a,
        conv: &w.conv,
        norm: &w.norm,
    };
    let g_mix = gpu_span("dn:mix", x.device());
    let gated = delta_net_mix_spans(&p, &c, dims, seqs, rms_eps, table, zgate)?;
    g_mix.end();

    let g_out = gpu_span("dn:out_proj", x.device());
    // The mixer's F32 goes straight in. Narrowing it here first would be a
    // full-tensor pass per DeltaNet layer per wave that rounds away precision
    // the mixer has just computed — and the FP fallback would widen it right
    // back on the next line. `out_dtype` names what the residual stream wants,
    // so the store does the conversion the cast used to.
    let out = w.w_out.forward_live_as(&gated, act)?;
    g_out.end();
    Ok(out)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::models::batch_test::test_helpers::hf_get;
    use crate::models::delta_net::mix::{delta_net_layer_forward, DeltaNetWeights};
    use crate::models::delta_net::LayerKind;
    use crate::models::qwen35::quantized_weights::{
        load_quantized_model, LoadInputs, QuantLayerMix,
    };
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
        // Resident: this test dequantizes the projections to build the
        // reference, which wants the weights in hand rather than a slot's
        // view of them.
        let model = load_quantized_model(
            &content,
            &mut reader,
            &device,
            Int8Mode::Off,
            LoadInputs::resident(),
        )?;

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

        let dn = model
            .cfg
            .layer_kinds
            .iter()
            .position(|k| matches!(k, LayerKind::DeltaNet))
            .expect("the stack has DeltaNet layers");
        let layer = model.layers.ensure(dn)?;
        let QuantLayerMix::DeltaNet(qw) = &layer.mix else {
            unreachable!("layer {dn} is DeltaNet by its kind")
        };

        // Same numbers on both sides; only the projection kernel differs.
        let reference = DeltaNetWeights {
            // The group in canonical order — this lineage's loader does not
            // stack, so the four entries are the four projections.
            wqkv: qw.proj[0].dequantize()?,
            wz: qw.proj[1].dequantize()?,
            w_beta: qw.proj[2].dequantize()?,
            w_alpha: qw.proj[3].dequantize()?,
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
        let got = quantized_delta_net_layer_forward(&x, qw, &dims, &mut s_prod, eps, ZGate::Silu)?;
        let mut s_ref = DeltaNetState::zeros(&dims, &device)?;
        let want = delta_net_layer_forward(&x, &reference, &dims, &mut s_ref, eps, ZGate::Silu)?;

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
