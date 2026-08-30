//! Reading a loaded layer's geometry, and writing the pack a streamed model
//! needs.
//!
//! The bridge between the model loader and the rest of this subsystem: it turns
//! `PendingLayer`-shaped weights into [`LayerImage`]s and, when no valid pack
//! exists, into pack records.
//!
//! # The pack build is a streaming pass, and that is the whole point
//!
//! A model larger than the card cannot be loaded and *then* packed — that is
//! the failure this design removes. So the build loads **one layer at a time**,
//! repacks it, writes the record, and drops it:
//!
//! ```text
//! for each layer:
//!     load_layer            one layer's tensors, repacked to the CUDA POOL
//!     read back to host     copy_data_to_host_on_stream
//!     write_layer           one pack record
//!     drop                  the pool frees; the next layer reuses the ground
//! ```
//!
//! Two things make that possible and both already existed:
//! `qwen35::quantized_weights::load_layer` (the trunk loop's body, extracted so
//! there is one transcription of a layer's tensor names) and
//! `WeightResidency::Pool` (the CUDA pool, which frees on drop, in place of the
//! dense block, which is a bump allocator that does not). The device peak is one layer's
//! source tensors and their twins, plus the repack's bounded f32 band — which is what
//! `dense_span::peak_load_pool_bytes` reserves pool room for.
//!
//! The pass runs only when the pack is missing or stale, so it is a one-time
//! cost per checkpoint rather than a startup cost.

#[cfg(feature = "cuda")]
use candle::quantized::GgmlDType;
#[cfg(feature = "cuda")]
use candle::Result;

use super::descriptor::LayerTensor;
#[cfg(feature = "cuda")]
use super::descriptor::{FfnForm, LayerImage, MixKind, Projection};
#[cfg(feature = "cuda")]
use crate::models::quantized_matmul::QMatMul;

/// A loaded layer's streamable projections, in the order the image will place
/// them.
///
/// Borrowed rather than owned: the caller holds the layer, and cloning a
/// `QMatMul` is a device-to-device copy of the weight (`QCudaStorage`'s `Clone`
/// is always-owned for exactly that reason), so this must never take one by
/// value.
///
/// CUDA-only, like everything that measures a slot: a projection's placed size
/// comes from `padded_storage_bytes` / `ko_repacked_bytes`, which are the CUDA
/// backend's dtype arithmetic. [`tensor_suffix`] is the one thing in this file
/// that is pure naming and stays available everywhere.
#[cfg(feature = "cuda")]
pub struct LoadedLayer<'a> {
    pub kind: MixKind,
    pub ffn: FfnForm,
    /// `(role, weight)` in image order.
    pub projections: Vec<(LayerTensor, &'a QMatMul)>,
}

#[cfg(feature = "cuda")]
impl LoadedLayer<'_> {
    /// The image this layer's weights place into.
    ///
    /// Every number comes off the loaded weight rather than from the config, so
    /// the pack cannot describe a geometry the kernels do not see — the same
    /// reasoning `QuantizedMlp::hidden_intermediate` gives for reading its own
    /// shapes back.
    pub fn image(&self) -> Result<LayerImage> {
        let mut projections = Vec::with_capacity(self.projections.len());
        for (role, w) in &self.projections {
            // Whatever form the load produced — a KO twin at an int8 mode, or
            // the source quant at `Int8Mode::Off` and for a shape the matmul
            // cannot tile. All three are flat byte payloads a slot can hold.
            let qt = w.inner().qtensor().ok_or_else(|| {
                candle::Error::Msg(format!(
                    "layer stream: {role:?} has no quantized weight to place — a dequantized \
                     tensor has no slot form, so this layer cannot be streamed"
                ))
            })?;
            let dims = qt.shape().dims();
            let [rows, cols] = dims else {
                candle::bail!("layer stream: {role:?} is rank {}, expected 2", dims.len());
            };
            let payload = qt.storage_size_in_bytes();
            projections.push(Projection {
                role: *role,
                shape: [*rows, *cols],
                dtype: qt.dtype(),
                payload,
                extent: slot_extent(payload, rows * cols, qt.dtype()),
            });
        }
        super::descriptor::layer_image(self.kind, self.ffn, &projections)
            .map_err(|e| candle::Error::Msg(e.to_string()))
    }

    /// Copy every projection's repacked bytes to the host, in image order.
    ///
    /// One buffer per projection, which the pack writer then places at the
    /// image's offsets. Sized from `storage_size_in_bytes` — the same number
    /// the image records — so a mismatch is caught by the writer's own length
    /// check rather than by a short copy.
    #[cfg(feature = "cuda")]
    pub fn read_back(
        &self,
        stream: &std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>,
    ) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::with_capacity(self.projections.len());
        for (role, w) in &self.projections {
            let qt = w.inner().qtensor().ok_or_else(|| {
                candle::Error::Msg(format!(
                    "layer stream: {role:?} has no quantized weight to read back"
                ))
            })?;
            let mut buf = vec![0u8; qt.storage_size_in_bytes()];
            qt.copy_data_to_host_on_stream(&mut buf, stream)?;
            out.push(buf);
        }
        stream.synchronize().map_err(candle::Error::wrap)?;
        Ok(out)
    }
}

/// The tensor a role is read from, under `blk.{li}`.
///
/// One transcription of the lineage's names, shared by the image derivation and
/// anything that has to find the same weight again. `FfnGateUp` has no tensor
/// of its own — it is `ffn_gate` and `ffn_up` concatenated at load — so it is
/// absent here and handled by the caller.
pub fn tensor_suffix(role: LayerTensor) -> Option<&'static str> {
    Some(match role {
        LayerTensor::Wqkv => "attn_qkv.weight",
        LayerTensor::Wz => "attn_gate.weight",
        LayerTensor::WOut => "ssm_out.weight",
        LayerTensor::Wq => "attn_q.weight",
        LayerTensor::Wk => "attn_k.weight",
        LayerTensor::Wv => "attn_v.weight",
        LayerTensor::Wo => "attn_output.weight",
        LayerTensor::FfnGate => "ffn_gate.weight",
        LayerTensor::FfnUp => "ffn_up.weight",
        LayerTensor::FfnDown => "ffn_down.weight",
        LayerTensor::FfnGateUp => return None,
    })
}

/// The per-tensor narrowing a load applies, keyed by tensor name.
///
/// A closure rather than a value because it is a **schedule** — the answer differs by layer and
/// by role — and because it must be the *same* function the loader consults. The pack's records
/// are derived from it here and its bytes are produced from it there; two transcriptions of one
/// schedule is a pack that does not describe itself.
///
/// Passed in rather than reached for so this module keeps knowing nothing about any lineage's
/// quantization policy: `qwen35::quantized_weights::streaming_twin` is the qwen35 answer, and a
/// caller with no policy passes one that always returns `None`.
#[cfg(feature = "cuda")]
pub type Narrowing<'a> = &'a dyn Fn(&str) -> Option<GgmlDType>;

/// Bytes a slot must reserve for a placed projection of `payload` bytes.
///
/// A KO twin's kernel reads exactly the chunks it was handed, so the reservation
/// is the payload. A source quant runs the GGML kernels, which address
/// `MATRIX_ROW_PADDING` elements past the end of every row unconditionally — so
/// the slot has to own that tail, or the kernel reads into the next projection's
/// bytes (and, at the last one, past the slot).
///
/// The one definition of that rule: [`slot_form`] derives it from the header and
/// [`LoadedLayer::image`] from the loaded weight, and the two must agree or the
/// pack describes a layer the loader does not build.
#[cfg(feature = "cuda")]
fn slot_extent(payload: usize, elems: usize, dtype: GgmlDType) -> usize {
    if dtype.is_ko() {
        payload
    } else {
        candle::quantized::cuda::padded_storage_bytes(elems, dtype)
    }
}

/// The form a projection takes in a slot, and how many bytes it occupies.
///
/// **Exactly the choice `QMatMul::build` makes**, restated over a shape and a
/// dtype instead of over a loaded tensor — the two must agree or the image
/// describes a layer the loader does not produce. There are three cases and the
/// first two are both "no KO twin":
///
/// * `Int8Mode::Off` — no twin exists at all, so the slot holds the source
///   quant and the GGML kernels read it in place;
/// * an int8 mode over a shape the matmul cannot tile (`nrows % 32`,
///   `ncols % 128`) — the loader logs "dense fallback for this tensor" and keeps
///   the source, so the slot must too;
/// * an int8 mode over a tileable shape — the KO twin, which is the case the
///   whole int8 path exists for.
///
/// `narrow` must be the **same** answer the loader will give for this tensor, because the two
/// derive different things from it and both must agree: this sizes the slot, and the repack
/// fills it. A disagreement is not a wrong number — it is a pack whose records do not describe
/// its bytes. `LayerPack::check_geometry` compares the two and rejects a stale pack rather than
/// reading one, which is what makes a schedule change safe to ship.
#[cfg(feature = "cuda")]
fn slot_form(
    shape: [usize; 2],
    src: GgmlDType,
    mode: candle::quantized::Int8Mode,
    narrow: Option<GgmlDType>,
) -> Result<(GgmlDType, usize, usize)> {
    use candle::quantized::cuda::{gemx_repacking_supported, ko_repacked_bytes};
    use candle::quantized::ko_quant::ko_tileable;
    use candle::Shape;

    let [rows, cols] = shape;
    if mode.is_int8() && ko_tileable(rows, cols) {
        // The int8 kernel reads exactly the chunks it was given, so payload and
        // extent are the same number.
        //
        // A source with no GEMX repack kernel does not repack: `QMatMul::build`
        // dequantizes it and re-quantizes to Q8_0 first, so the twin follows Q8_0
        // rather than the source. That is the path a *float* trunk projection
        // takes (F32/F16/BF16 have no KO twin at all, so `to_ko` on them would
        // fail the load outright), and the path MXFP4 takes (it names `MXFP4_KO`
        // but has no `dtype_to_qtype` arm, so predicting `MXFP4_KO` here would
        // size every slot at roughly half the width the loader then writes).
        let repacked = if gemx_repacking_supported(src) {
            src
        } else {
            GgmlDType::Q8_0
        };
        let picked = repacked.to_ko(mode)?;
        // Narrowing only ever shrinks — the same rule `Loader::proj` applies, so a target
        // wider than the mode's choice leaves the tensor alone on both sides.
        let ko = match narrow {
            Some(n) if n.bits_per_weight() < picked.bits_per_weight() => n,
            _ => picked,
        };
        let bytes = ko_repacked_bytes(&Shape::from_dims(&[rows, cols]), ko)?;
        return Ok((ko, bytes, bytes));
    }
    // The source quant, as GGML lays it out: whole blocks, contiguous — plus
    // the row-padding tail.
    //
    // **The tail is not slack.** The GGML matmul kernels address
    // `MATRIX_ROW_PADDING` elements past the end of every row unconditionally,
    // which is why an owned `QCudaStorage` allocates `padded_storage_bytes`
    // rather than the payload. A slot has to reserve the same, or the kernel
    // reads into the next projection's bytes — and at the last projection, past
    // the slot. It costs ~280 B per projection against a ~240 MB layer.
    let elems = rows * cols;
    if !elems.is_multiple_of(src.block_size()) {
        candle::bail!(
            "layer stream: [{rows}, {cols}] is {elems} elements, not a whole number of \
             {:?} blocks ({}) — the source form cannot be placed in a slot",
            src,
            src.block_size()
        );
    }
    let payload = elems / src.block_size() * src.type_size();
    Ok((src, payload, slot_extent(payload, elems, src)))
}

/// Every layer's image, derived from the **GGUF header alone**.
///
/// # Why this reads no weights
///
/// The pack's header needs `slot_bytes`, which is the max over images, which
/// needs every projection's size in the slot — and the obvious way to get that
/// is to load a layer and ask it. That would be circular for the pack build,
/// whose whole point is not to have the model resident.
///
/// It is not needed. A projection's slot form is a function of its shape, its
/// source dtype and the numeric mode ([`slot_form`]), and the first two are in
/// the tensor table. So the entire geometry is arithmetic over the header, and
/// the pack build becomes a single pass rather than one pass to measure and
/// another to write.
///
/// The fused `[2·intermediate, hidden]` FFN weight has no tensor of its own —
/// it is built at load — but its shape is the two halves' rows added and its
/// dtype is theirs, which fusion requires to match. So it too is derivable.
#[cfg(feature = "cuda")]
pub fn images_from_gguf(
    content: &candle::quantized::gguf_file::Content,
    layer_kinds: &[crate::models::delta_net::LayerKind],
    mode: candle::quantized::Int8Mode,
    narrow: Narrowing,
) -> Result<Vec<LayerImage>> {
    use crate::models::delta_net::LayerKind;

    let info = |name: &str| -> Result<(Vec<usize>, GgmlDType)> {
        let t = content.tensor_infos.get(name).ok_or_else(|| {
            candle::Error::Msg(format!("layer stream: the checkpoint has no {name}"))
        })?;
        Ok((t.shape.dims().to_vec(), t.ggml_dtype))
    };
    let entry = |role: LayerTensor,
                 dims: Vec<usize>,
                 src: GgmlDType,
                 narrow: Option<GgmlDType>|
     -> Result<Projection> {
        let [rows, cols] = dims[..] else {
            candle::bail!("layer stream: {role:?} is rank {}, expected 2", dims.len());
        };
        let (dtype, payload, extent) = slot_form([rows, cols], src, mode, narrow)
            .map_err(|e| candle::Error::Msg(format!("{role:?}: {e}")))?;
        Ok(Projection {
            role,
            shape: [rows, cols],
            dtype,
            payload,
            extent,
        })
    };

    let mut out = Vec::with_capacity(layer_kinds.len());
    for (li, kind) in layer_kinds.iter().enumerate() {
        let p = format!("blk.{li}");
        let (mix_kind, roles): (MixKind, &[LayerTensor]) = match kind {
            LayerKind::DeltaNet => (MixKind::DeltaNet, LayerTensor::DELTA_NET_MIX),
            LayerKind::Attention => (MixKind::Attention, LayerTensor::ATTENTION_MIX),
        };
        let mut projections = Vec::with_capacity(roles.len() + 2);
        for role in roles {
            let name = format!(
                "{p}.{}",
                tensor_suffix(*role).expect("mixer roles are named")
            );
            let (dims, src) = info(&name)?;
            projections.push(entry(*role, dims, src, narrow(&name))?);
        }

        // The FFN, in whichever form this load will produce. `from_weights`
        // fuses when the device is CUDA and the dtype is quantized, and both
        // halves must agree on dtype and shape — the same three conditions,
        // read here instead of after the fact.
        let (gate_dims, gate_src) = info(&format!("{p}.ffn_gate.weight"))?;
        let (up_dims, up_src) = info(&format!("{p}.ffn_up.weight"))?;
        // One target for all three projections, decided by `ffn_down` — `QuantizedMlp::
        // from_weights_in` takes a single width for the same reason, and this has to mirror it
        // or the slot and the repack disagree. It is a floor, not a forced value: a projection
        // already at or below it keeps what it has.
        let ffn_narrow = narrow(&format!("{p}.ffn_down.weight"));
        let fusable = gate_src == up_src
            && gate_dims == up_dims
            && !matches!(gate_src, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16);
        let ffn = if fusable {
            let fused_dims = vec![gate_dims[0] * 2, gate_dims[1]];
            projections.push(entry(
                LayerTensor::FfnGateUp,
                fused_dims,
                gate_src,
                ffn_narrow,
            )?);
            FfnForm::Fused
        } else {
            projections.push(entry(
                LayerTensor::FfnGate,
                gate_dims,
                gate_src,
                ffn_narrow,
            )?);
            projections.push(entry(LayerTensor::FfnUp, up_dims, up_src, ffn_narrow)?);
            FfnForm::Split
        };
        let (down_dims, down_src) = info(&format!("{p}.ffn_down.weight"))?;
        projections.push(entry(
            LayerTensor::FfnDown,
            down_dims,
            down_src,
            ffn_narrow,
        )?);

        out.push(
            super::descriptor::layer_image(mix_kind, ffn, &projections)
                .map_err(|e| candle::Error::Msg(format!("layer {li}: {e}")))?,
        );
    }
    Ok(out)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle::quantized::Int8Mode;

    /// A tileable projection at an int8 mode takes its KO twin.
    #[test]
    fn a_tileable_shape_is_placed_as_its_ko_twin() {
        let (dt, payload, extent) =
            slot_form([5120, 4096], GgmlDType::Q4_K, Int8Mode::Performance, None).unwrap();
        assert_eq!(dt, GgmlDType::Q4_KO);
        // `(nrows / 8) * (ncols / 128)` chunks, at the twin's chunk width.
        let chunks = (5120 / 8) * (4096 / 128);
        assert_eq!(
            payload,
            chunks * candle::quantized::ko_quant::ko_chunk_bytes(GgmlDType::Q4_KO)
        );
        // The int8 kernel reads exactly its chunks, so nothing is reserved past
        // them.
        assert_eq!(extent, payload);
    }

    /// `Int8Mode::Off` has no twin, so the slot holds the source quant — the
    /// case that used to fail the whole load with "Int8Mode::Off has no KO
    /// weight twin" before a slot could hold a GGML form.
    #[test]
    fn off_mode_places_the_source_quant() {
        let (dt, payload, extent) =
            slot_form([5120, 4096], GgmlDType::Q6_K, Int8Mode::Off, None).unwrap();
        assert_eq!(dt, GgmlDType::Q6_K);
        let elems = 5120 * 4096;
        assert_eq!(
            payload,
            elems / GgmlDType::Q6_K.block_size() * GgmlDType::Q6_K.type_size()
        );
        // The slot reserves the row-padding tail the GGML kernels read into.
        // Without it the last projection's matmul runs off the end of the slot.
        assert!(extent > payload, "the padding tail was not reserved");
        assert_eq!(
            extent,
            candle::quantized::cuda::padded_storage_bytes(elems, GgmlDType::Q6_K)
        );
    }

    /// A shape the matmul cannot tile keeps its source form **even at an int8
    /// mode**, because that is what the loader does with it: `QMatMul::build`
    /// tests `ko_tileable` before repacking and falls back to the dense path.
    /// An image that claimed a twin here would describe a layer the loader does
    /// not produce.
    #[test]
    fn an_untileable_shape_keeps_its_source_form_at_every_mode() {
        // 16 rows: the 0.8B's `w_alpha`/`w_beta`. Packs into the storage chunk
        // (8 rows) and is still refused by the matmul tile (32).
        for mode in [Int8Mode::Off, Int8Mode::Performance, Int8Mode::Precision] {
            let (dt, payload, extent) = slot_form([16, 4096], GgmlDType::Q6_K, mode, None).unwrap();
            assert_eq!(dt, GgmlDType::Q6_K, "{mode:?}");
            // And so it needs the GGML row-padding tail, at every mode.
            assert!(extent > payload, "{mode:?}");
        }
    }

    /// A narrowing target shrinks the slot, and the slot is exactly the narrowed twin's chunks.
    #[test]
    fn a_narrowed_projection_is_sized_at_the_narrow_twin() {
        let wide = slot_form([5120, 4096], GgmlDType::Q5_K, Int8Mode::Precision, None).unwrap();
        let narrow = slot_form(
            [5120, 4096],
            GgmlDType::Q5_K,
            Int8Mode::Precision,
            Some(GgmlDType::Q3_KO),
        )
        .unwrap();
        assert_eq!(narrow.0, GgmlDType::Q3_KO);
        let chunks = (5120 / 8) * (4096 / 128);
        assert_eq!(
            narrow.1,
            chunks * candle::quantized::ko_quant::ko_chunk_bytes(GgmlDType::Q3_KO)
        );
        assert_eq!(narrow.2, narrow.1);
        assert!(narrow.1 < wide.1, "narrowing did not shrink the slot");
    }

    /// **Narrowing is a floor, not a forced value.** A target at or above the width the mode
    /// already picked leaves the projection exactly as it was — the same clamp `Loader::proj`
    /// and `QuantizedMlp::from_weights_in` apply, and the reason one schedule entry can name a
    /// role across blocks whose source quants differ.
    #[test]
    fn a_target_no_narrower_than_the_mode_leaves_the_slot_alone() {
        // Q3_K's Precision twin is Q4_KO. Asking for Q4_KO (equal) or Q5_KO (wider) must not
        // move it, and must not fail.
        let plain = slot_form([5120, 4096], GgmlDType::Q3_K, Int8Mode::Precision, None).unwrap();
        assert_eq!(plain.0, GgmlDType::Q4_KO);
        for target in [GgmlDType::Q4_KO, GgmlDType::Q5_KO, GgmlDType::Q8_KO] {
            let got = slot_form(
                [5120, 4096],
                GgmlDType::Q3_K,
                Int8Mode::Precision,
                Some(target),
            )
            .unwrap();
            assert_eq!(got, plain, "{target:?}");
        }
    }

    /// An untileable shape has no twin to narrow, so the target is ignored rather than producing
    /// a KO slot the loader would never fill.
    #[test]
    fn narrowing_does_not_reach_an_untileable_shape() {
        let plain = slot_form([16, 4096], GgmlDType::Q6_K, Int8Mode::Precision, None).unwrap();
        let asked = slot_form(
            [16, 4096],
            GgmlDType::Q6_K,
            Int8Mode::Precision,
            Some(GgmlDType::Q3_KO),
        )
        .unwrap();
        assert_eq!(asked, plain);
    }
}
