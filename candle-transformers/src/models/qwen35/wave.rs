//! The hybrid layer sweep: one wave, two kinds of layer.
//!
//! The uniform engine runs `forward_layer_batched_mixed` for every layer
//! index. A hybrid cannot, because three quarters of its layers mix tokens by
//! a recurrence rather than by attending a KV cache. This module is the sweep
//! that dispatches on layer kind, and it is deliberately the *only* thing
//! that differs: attention layers are handed straight to the generic
//! per-layer body, and DeltaNet layers run their mixer plus the ten-line FFN
//! driver in `quantized_delta_net`.
//!
//! Two indexing rules hold throughout, and both are the hybrid's doing:
//!
//! * **Caches are indexed by KV layer, never by layer.** A DeltaNet layer
//!   owns no KV, so `caches[layer_idx]` would read another layer's chunks.
//!   [`KvLayerMap`] is the only thing allowed to make that translation.
//! * **Recurrent state is per sequence, not per wave row.** The mixer carries
//!   `S` and a conv tail across a sequence's tokens, so a wave holding
//!   several sequences runs the mixer once per sequence over that sequence's
//!   own row range, and only the FFN sees the whole packed buffer.

use candle::{DType, Device, Result};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{begin_wave, LayerPhase};

use super::quantized_delta_net::quantized_delta_net_ffn;
use super::quantized_weights::{QuantLayerMix, QuantModel};
use crate::models::delta_net::{
    quantized_delta_net_layer_forward_spans, DeltaNetLayerTable, DeltaNetSeq, KvLayerMap,
    RecurrentStateStore, SeqSpan, StashSlot,
};
use crate::models::rotary_layout::RotaryLayout;
use crate::models::tensor_cat::TensorCat;

/// Run one DeltaNet layer's mixing half over a packed wave buffer.
///
/// `stores` is one [`RecurrentStateStore`] per span, in span order — the
/// store is per *sequence*, holding every recurrent layer's `S` and conv
/// tail for that one sequence, so a wave carrying N sequences carries N
/// stores. Each sequence's rows are mixed against its own state, the results
/// are concatenated back into buffer order, and the whole thing is added to
/// the residual stream in one operation, so the per-sequence loop costs
/// launches rather than a scattered write.
///
/// The stores are advanced in place: this is the point at which a sequence's
/// state moves forward, which is why a failed wave must `rollback_wave`
/// rather than simply not commit.
// `stores` and `stash` are two per-span arrays the caller owns separately —
// the stores are lifted out of the model's map for the sweep, the stash slots
// are the sweep's own. Zipping them into one slice would make every caller
// build a temporary pairing to satisfy a lint.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_mix_wave(
    model: &QuantModel,
    layer_idx: usize,
    spans: &[SeqSpan],
    x: &mut TensorCat,
    stores: &mut [&mut RecurrentStateStore],
    rms_eps: f64,
    table: Option<&DeltaNetLayerTable>,
    stash: &[Option<StashSlot<'_>>],
) -> Result<()> {
    let layer = &model.layers[layer_idx];
    let QuantLayerMix::DeltaNet(w) = &layer.mix else {
        candle::bail!("delta_net_mix_wave called on a non-DeltaNet layer {layer_idx}");
    };
    if stores.len() != spans.len() || stash.len() != spans.len() {
        candle::bail!(
            "delta_net_mix_wave: {} spans against {} state stores and {} stash slots — \
             every sequence in the wave carries its own recurrent state",
            spans.len(),
            stores.len(),
            stash.len()
        );
    }
    let dims = &model.cfg.delta_net;

    let xt = x.as_cat_tensor();
    let hidden = xt.dim(xt.rank() - 1)?;
    let flat = xt.reshape((xt.elem_count() / hidden, hidden))?;

    // The mixing half's transient scope, spanning ln1 through the residual add
    // that consumes the result — the same layer scoping an attention layer's
    // half uses, and the same one [`quantized_delta_net_ffn`] opens for the
    // other half of this layer.
    #[cfg(feature = "cuda")]
    let mix_wave = match xt.device() {
        Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Attention)?),
        _ => None,
    };
    #[cfg(not(feature = "cuda"))]
    let mix_wave: Option<()> = None;

    // ln1 over the whole buffer: elementwise per row, so it does not need the
    // per-sequence split the recurrence does.
    //
    // **This is where the layer's chain is seeded.** Its input is the residual
    // stream, which crosses layers and so lives on the pool with no arena to
    // inherit; every one of the forty-odd ops downstream of here inherits from
    // its operand instead, so naming the span once — here — puts the whole
    // mixer on it without another mention of the wave.
    #[cfg(feature = "cuda")]
    let normed = layer.attn_norm.forward_rooted(&flat, mix_wave.as_ref())?;
    #[cfg(not(feature = "cuda"))]
    let normed = layer.attn_norm.forward(&flat)?;

    // **One call for the whole wave, not one per sequence.** The layer is
    // row-wise apart from the conv tail and the recurrence, so handing it every
    // sequence at once lets the five projections and the thirty-odd elementwise
    // ops run once; `DeltaNetSeq` is how the two carried steps still find their
    // own rows and their own state.
    //
    // The store hands out the layer's PAIR: the buffer the wave reads and the
    // one it writes. Nothing is read out and written back, and nothing is
    // replaced — what changes at `commit_wave` is which of the two is live,
    // which is what makes a failed wave free to undo. Taking the pair here is
    // also what records that this layer advanced, so a sweep that covers part
    // of the stack commits only the layers it actually ran.
    //
    // Both carried buffers have two halves — `s` and the conv tail alike — so
    // the layer advances without a copy anywhere.
    let mut seqs: Vec<DeltaNetSeq<'_>> = Vec::with_capacity(spans.len());
    for ((span, store), slot) in spans.iter().zip(stores.iter_mut()).zip(stash.iter()) {
        let (state, out) = store.layer_state_pair_mut(layer_idx)?;
        seqs.push(DeltaNetSeq {
            start: span.start,
            len: span.len,
            state,
            out,
            // `Some` only for a span a speculative verify will have to rewind
            // — see `super::spec`. Every other span stashes nothing, so an
            // ordinary wave copies nothing.
            stash: *slot,
        });
    }
    let mixed =
        quantized_delta_net_layer_forward_spans(&normed, w, dims, &mut seqs, rms_eps, table)?;
    drop(seqs);
    let mixed = mixed.reshape(xt.shape())?;
    x.add_mut(&mixed)?;
    // `mixed` borrows `mix_wave`, so the compiler already refuses any drop order
    // but this one; both die at the end of the function.
    drop(mixed);
    drop(mix_wave);
    Ok(())
}

/// The whole hybrid sweep over `[layer_start, layer_end)`.
///
/// `attention_layer` builds the generic per-layer wrapper for an attention
/// layer; it is a callback because the wrapper borrows the caches, which the
/// caller owns for the duration of the wave.
pub struct HybridSweep<'a> {
    pub model: &'a QuantModel,
    pub kv_map: &'a KvLayerMap,
    pub rotary: &'a RotaryLayout,
}

impl HybridSweep<'_> {
    /// Layer kinds in `[start, end)`, paired with the KV index an attention
    /// layer writes to. The wave loop drives from this rather than
    /// re-deriving the mapping per layer.
    pub fn plan(&self, start: usize, end: usize) -> Result<Vec<(usize, Option<usize>)>> {
        if start > end || end > self.model.cfg.num_layers {
            candle::bail!(
                "hybrid sweep: bad layer range [{start}, {end}) over {} layers",
                self.model.cfg.num_layers
            );
        }
        Ok((start..end)
            .map(|li| (li, self.kv_map.kv_index(li)))
            .collect())
    }

    /// The rotary layout every attention layer in this stack shares.
    pub fn rotary(&self) -> &RotaryLayout {
        self.rotary
    }
}

/// Run the FFN half of a DeltaNet layer — see [`quantized_delta_net_ffn`].
///
/// Re-exported through this module so the wave loop reads as one sweep
/// rather than reaching across modules mid-layer.
pub fn delta_net_ffn_wave(
    model: &QuantModel,
    layer_idx: usize,
    x: &mut TensorCat,
    act_dtype: DType,
    orig_dtype: DType,
) -> Result<()> {
    quantized_delta_net_ffn(&model.layers[layer_idx], x, act_dtype, orig_dtype)
}

/// The device every wave buffer of this model lives on.
pub fn model_device(model: &QuantModel) -> &Device {
    &model.device
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::delta_net::seq_spans;

    #[test]
    fn spans_tile_the_packed_buffer() {
        // The engine packs [decode | prefill | glue]: one row per decode
        // sequence, q_len rows per prefill/glue sequence.
        let spans = seq_spans(&[7, 9, 4], &[1, 5, 3]).unwrap();
        assert_eq!(
            spans,
            vec![
                SeqSpan {
                    seq: 7,
                    start: 0,
                    len: 1
                },
                SeqSpan {
                    seq: 9,
                    start: 1,
                    len: 5
                },
                SeqSpan {
                    seq: 4,
                    start: 6,
                    len: 3
                },
            ]
        );
        // Contiguous, no gaps, covering every row exactly once.
        let mut cursor = 0;
        for s in &spans {
            assert_eq!(s.start, cursor);
            cursor += s.len;
        }
        assert_eq!(cursor, 9);
    }

    #[test]
    fn spans_refuse_a_length_mismatch() {
        let err = seq_spans(&[1, 2], &[1]).unwrap_err();
        assert!(err.to_string().contains("against"), "{err}");
    }

    #[test]
    fn an_empty_wave_has_no_spans() {
        assert!(seq_spans(&[], &[]).unwrap().is_empty());
    }

    /// The property a batched recurrent layer lives or dies by: mixing two
    /// sequences packed into one buffer must equal mixing each alone.
    ///
    /// A recurrent mixer has per-sequence state, so batching it is not the
    /// trivially-safe operation batching attention is — a span computed off
    /// by one row, or a shared store, silently leaks one conversation into
    /// another. Checked on the real 9B checkpoint at its real geometry, with
    /// two sequences of different lengths so an off-by-one cannot alias.
    #[test]
    #[ignore = "reads the pinned Qwen3.5-9B GGUF from the HF cache (7.5 GB) and needs a GPU"]
    fn batched_mixing_equals_mixing_each_sequence_alone() -> Result<()> {
        use super::super::quantized_weights::load_quantized_model;
        use crate::models::batch_test::test_helpers::hf_get;
        use crate::models::delta_net::quantized_delta_net_layer_forward;
        use crate::models::delta_net::RecurrentStateStore;
        use candle::quantized::{gguf_file::Content, Int8Mode};
        use candle::{Module, Tensor};
        use std::io::{BufReader, Seek, SeekFrom};

        // The lineage's own pin, not a second copy of it: a test that names its
        // own revision keeps passing against a checkpoint the gates no longer
        // run, which is how this one was still fetching the pre-MTP 9B after
        // the pin moved.
        let spec = crate::models::quantized_qwen35::QWEN35_9B;
        let path = hf_get(spec.0, hf_hub::RepoType::Model, spec.1, spec.2)?;
        let device = Device::new_cuda(0)?;
        let mut reader = BufReader::new(std::fs::File::open(&path)?);
        let content = Content::read(&mut reader)?;
        reader.seek(SeekFrom::Start(0))?;
        // The 9B is dense, so it needs no expert cache.
        let model = load_quantized_model(
            &content,
            &mut reader,
            &device,
            Int8Mode::Off,
            None,
            |_, _| Ok(None),
        )?;

        let li = 0usize; // DeltaNet under the 3:1 schedule
        let hidden = model.cfg.hidden_size;
        let (len_a, len_b) = (5usize, 3usize);
        let total = len_a + len_b;
        let eps = model.cfg.rms_norm_eps;

        let packed = Tensor::randn(0f32, 1.0, (1, total, hidden), &device)?;
        let spans = seq_spans(&[11, 22], &[len_a, len_b])?;

        // The reference runs FIRST, deliberately. `delta_net_mix_wave` ends in
        // an in-place residual add, and `Tensor::clone` shares storage — so a
        // reference computed afterwards from the "same" tensor would be
        // reading the wave's own output. (That is a hazard of the test, not
        // of the engine, whose buffer is freshly embedded per wave.)
        let QuantLayerMix::DeltaNet(w) = &model.layers[li].mix else {
            panic!("layer {li} is not DeltaNet");
        };
        let flat = packed.reshape((total, hidden))?;
        let mut wants = Vec::new();
        for span in &spans {
            let rows = flat.narrow(0, span.start, span.len)?.contiguous()?;
            let normed = model.layers[li].attn_norm.forward(&rows)?;
            let mut solo =
                RecurrentStateStore::new(&model.cfg.layer_kinds, &model.cfg.delta_net, &device)?;
            let y = quantized_delta_net_layer_forward(
                &normed,
                w,
                &model.cfg.delta_net,
                solo.layer_state_mut(li)?,
                eps,
            )?;
            wants.push(rows.add(&y)?);
        }
        let want = Tensor::cat(&wants, 0)?;

        // Batched: one buffer, one store per sequence.
        let mut store_a =
            RecurrentStateStore::new(&model.cfg.layer_kinds, &model.cfg.delta_net, &device)?;
        let mut store_b =
            RecurrentStateStore::new(&model.cfg.layer_kinds, &model.cfg.delta_net, &device)?;
        let mut x = TensorCat::from_cat_tensor(packed, 0)?;
        {
            // **Open and commit the wave, as the sweep does.** The mixer writes
            // the half a store is NOT reading and `commit_wave` is what makes
            // that half live, so a caller that skips the bracket reads the
            // entering state back and sees two sequences that never advanced —
            // which is what this test's own isolation check then reports as a
            // span bug. `delta_net_mix_wave` is one level below `sweep`, so the
            // bracket is the test's to supply.
            store_a.begin_wave()?;
            store_b.begin_wave()?;
            let mut stores = vec![&mut store_a, &mut store_b];
            let stash = vec![None, None];
            delta_net_mix_wave(&model, li, &spans, &mut x, &mut stores, eps, None, &stash)?;
            store_a.commit_wave();
            store_b.commit_wave();
        }
        let got = x.as_cat_tensor().reshape((total, hidden))?;

        let rel = |a: &Tensor, b: &Tensor| -> Result<f32> {
            let d = a.sub(b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let s = b.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            Ok(d / s.max(1e-6))
        };
        // Per-span, so a slicing error names the span it corrupted.
        for (i, span) in spans.iter().enumerate() {
            println!(
                "  span@{} rel {:.3e}",
                span.start,
                rel(&got.narrow(0, span.start, span.len)?, &wants[i])?
            );
        }

        let diff = got
            .sub(&want)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let scale = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        println!("batched-vs-solo rel {:.3e}", diff / scale.max(1e-6));
        assert!(
            diff / scale.max(1e-6) < 1e-5,
            "packing sequences together changed the result: rel {}",
            diff / scale.max(1e-6)
        );

        // And the states advanced independently: each store must match what
        // its sequence alone would have produced, which is only true if the
        // spans sliced the buffer correctly.
        assert!(
            store_a.layer_state(li)?.s.dims() == store_b.layer_state(li)?.s.dims(),
            "both stores hold the same shape"
        );
        let sa = store_a.layer_state(li)?.s.clone();
        let sb = store_b.layer_state(li)?.s.clone();
        let cross = sa
            .sub(&sb)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            cross > 1e-6,
            "the two sequences ended in identical state — they were fed the \
             same rows, so the spans are wrong"
        );
        Ok(())
    }
}
