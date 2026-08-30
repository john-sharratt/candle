//! The byte layout of one dense layer inside a weight-zone slot.
//!
//! The layer analogue of the expert cache's `LayerGeometry` + `slot_offsets`,
//! and the **single source of truth** for where each projection sits inside a
//! slot image. The pack writer, the warm fill and the slot view all read their
//! offsets from here, for the reason `repack_ko_into` already gives about its
//! own sizing: two copies of that arithmetic in different files is how one gets
//! corrected and the other does not, and the failure is a silent write past the
//! end of one weight into the next.
//!
//! # What a slot image holds, and what it deliberately does not
//!
//! Only the **big projections** — the weights that dominate a layer's bytes.
//! Whether one sits in a slot as its KO int8 twin or as the source quant is a
//! property of the load rather than of the image (`build::slot_form` decides,
//! and the placement's dtype records the decision); either way it is a flat
//! payload `view_repacked` can wrap. Everything else in a layer stays resident
//! for the life of the process:
//!
//! | Stays resident | Why |
//! |---|---|
//! | `attn_norm`, `post_attn_norm` | `RmsNorm`, a fused producer rather than a raw buffer |
//! | DeltaNet `dt_bias`, `a`, `conv`, `norm` | F32 by design — the recurrence accumulates and must not drift |
//! | `w_beta`, `w_alpha` | `[n_v_heads, hidden]`, sub-tile: 48 rows does not clear `nrows % 32` |
//! | `q_norm`, `k_norm` | per-head `RmsNorm`, folded into the projection |
//!
//! That residue is ~0.1% of a layer — on Qwen3.8-27B roughly 250 KB against
//! 240 MB, or ~16 MB across all 64 layers. Streaming it would buy nothing and
//! would drag the `RmsNorm` and F32-constant rebuild into a path whose whole
//! value is that a slot view is a pointer and a length. So the image is
//! uniformly "things `view_repacked` can wrap", and nothing else.
//!
//! # The FFN's form is a property of the load, not the architecture
//!
//! `QuantizedMlp::from_weights` row-concatenates gate and up into a single
//! `[2·intermediate, hidden]` weight whenever the device is CUDA and the dtype
//! is quantized — which is every checkpoint this engine runs. So a production
//! layer's FFN is **two** projections, not three, and an image built for
//! `{gate, up, down}` would describe a layer that does not exist.
//!
//! It is carried per layer as [`FfnForm`] rather than assumed, because the same
//! checkpoint fuses on CUDA and does not on CPU, and the pack has to describe
//! the layer that was actually loaded.
//!
//! # Slots are uniform; layer images are not
//!
//! The zone rests on equal-sized slots — *"relocating a slot is a memcpy
//! between two addresses of identical length rather than a compaction"* — but a
//! DeltaNet layer and an attention layer carry different tensor sets. The slot
//! is therefore sized to the **maximum** image over the model
//! ([`slot_bytes_for_layers`]) and the shorter kind leaves a tail unused. On
//! the 27B that is ~2%, against a size-classed zone that would cost the
//! property every relocation and retraction depends on.

use candle::quantized::GgmlDType;

/// Byte alignment of each projection inside a slot image.
///
/// The same 256 B the expert slots use. It is what lets every projection base
/// satisfy the matmul kernels' alignment expectations regardless of how the
/// projection before it happened to size.
pub const PROJECTION_ALIGN: usize = 256;

/// Which weight of a layer a placement names.
///
/// Exhaustive rather than a string, so a new streamed projection breaks every
/// match at compile time instead of silently landing nowhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerTensor {
    /// DeltaNet `[conv_dim, hidden]` fused `[Q|K|V]`.
    Wqkv,
    /// DeltaNet `[value_dim, hidden]` output gate.
    Wz,
    /// DeltaNet `[hidden, value_dim]`.
    WOut,
    /// Attention `[2·head_dim·n_head, hidden]` interleaved `[q|gate]`.
    Wq,
    /// Attention `[head_dim·n_kv, hidden]`.
    Wk,
    /// Attention `[head_dim·n_kv, hidden]`.
    Wv,
    /// Attention `[hidden, head_dim·n_head]`.
    Wo,
    /// Dense FFN, row-concatenated `[2·intermediate, hidden]`.
    ///
    /// The form the production path actually loads: `QuantizedMlp::from_weights`
    /// fuses gate and up whenever the device is CUDA and the dtype is
    /// quantized, which is every checkpoint this engine runs. The split form
    /// below exists for the cases fusion declines — a float weight, or a
    /// gate/up shape mismatch.
    FfnGateUp,
    /// Dense FFN `[intermediate, hidden]`, unfused.
    FfnGate,
    /// Dense FFN `[intermediate, hidden]`, unfused.
    FfnUp,
    /// Dense FFN `[hidden, intermediate]`. Never fused — its shape does not
    /// match the other two.
    FfnDown,
}

/// Whether a layer's FFN arrived fused.
///
/// Not a property of the architecture but of the load: the same checkpoint
/// yields `Fused` on CUDA and `Split` on CPU, so it is carried per layer rather
/// than assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnForm {
    /// `[FfnGateUp, FfnDown]` — the production path.
    Fused,
    /// `[FfnGate, FfnUp, FfnDown]`.
    Split,
}

impl FfnForm {
    pub fn tensors(self) -> &'static [LayerTensor] {
        match self {
            Self::Fused => &[LayerTensor::FfnGateUp, LayerTensor::FfnDown],
            Self::Split => &[
                LayerTensor::FfnGate,
                LayerTensor::FfnUp,
                LayerTensor::FfnDown,
            ],
        }
    }

    /// The on-disk discriminant — see [`LayerTensor::to_u32`].
    pub fn to_u32(self) -> u32 {
        match self {
            Self::Fused => 0,
            Self::Split => 1,
        }
    }

    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Fused,
            1 => Self::Split,
            _ => return None,
        })
    }
}

impl LayerTensor {
    /// The mixer projections of each layer kind, in image order.
    pub const DELTA_NET_MIX: &'static [Self] = &[Self::Wqkv, Self::Wz, Self::WOut];
    /// See [`Self::DELTA_NET_MIX`].
    pub const ATTENTION_MIX: &'static [Self] = &[Self::Wq, Self::Wk, Self::Wv, Self::Wo];
    /// The on-disk discriminant.
    ///
    /// Written into the pack header, so these numbers are a **file format**: a
    /// changed value silently reinterprets an existing pack's geometry as a
    /// different projection. Once a pack has been written anywhere they may
    /// only be appended to, never renumbered.
    pub fn to_u32(self) -> u32 {
        match self {
            Self::Wqkv => 0,
            Self::Wz => 1,
            Self::WOut => 2,
            Self::Wq => 3,
            Self::Wk => 4,
            Self::Wv => 5,
            Self::Wo => 6,
            Self::FfnGateUp => 7,
            Self::FfnGate => 8,
            Self::FfnUp => 9,
            Self::FfnDown => 10,
        }
    }

    /// Inverse of [`Self::to_u32`]. `None` for a discriminant this build does
    /// not know, which is a pack to rewrite rather than a value to guess at.
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Wqkv,
            1 => Self::Wz,
            2 => Self::WOut,
            3 => Self::Wq,
            4 => Self::Wk,
            5 => Self::Wv,
            6 => Self::Wo,
            7 => Self::FfnGateUp,
            8 => Self::FfnGate,
            9 => Self::FfnUp,
            10 => Self::FfnDown,
            _ => return None,
        })
    }
}

/// Which token mixer a layer uses, and so which projections its image holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixKind {
    DeltaNet,
    Attention,
}

impl MixKind {
    /// This kind's projections, in image order: mixer first, then the FFN in
    /// whichever form it loaded as.
    pub fn tensors(self, ffn: FfnForm) -> impl Iterator<Item = LayerTensor> {
        let mix = match self {
            Self::DeltaNet => LayerTensor::DELTA_NET_MIX,
            Self::Attention => LayerTensor::ATTENTION_MIX,
        };
        mix.iter().chain(ffn.tensors().iter()).copied()
    }

    /// The on-disk discriminant — see [`LayerTensor::to_u32`] on why these are
    /// a file format rather than an implementation detail.
    pub fn to_u32(self) -> u32 {
        match self {
            Self::DeltaNet => 0,
            Self::Attention => 1,
        }
    }

    /// Inverse of [`Self::to_u32`].
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::DeltaNet,
            1 => Self::Attention,
            _ => return None,
        })
    }
}

/// One projection as the checkpoint declares it, before placement.
///
/// The two byte counts are supplied rather than computed: they come from
/// `ko_repacked_bytes` and `padded_storage_bytes`, which need the CUDA-side
/// dtype tables, and keeping them inputs is what lets every placement rule in
/// this file be exercised with no GPU. They must be the **same** numbers the
/// load will produce — the two agreeing is the whole safety argument, exactly as
/// it is for an expert slot. `build::slot_form` is the one place that derives
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Projection {
    pub role: LayerTensor,
    /// `[out_features, in_features]`, as `QMatMul::weight_dims` reports it.
    pub shape: [usize; 2],
    /// The dtype the slot holds — a KO twin, or the source quant where no twin
    /// was taken. What `view_repacked` will be handed.
    pub dtype: GgmlDType,
    /// Bytes of actual weight: what the pack record holds and the H2D copies.
    pub payload: usize,
    /// Bytes the slot must **reserve** for it.
    ///
    /// Equal to `payload` for a KO twin, whose kernel reads exactly its own
    /// bytes. Larger for a source quant, because the GGML matmul kernels address
    /// `MATRIX_ROW_PADDING` elements past the end of every row — so the slot has
    /// to own that tail or the kernel reads into the next projection.
    pub extent: usize,
}

/// A projection's placement inside the slot image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    pub role: LayerTensor,
    /// Byte offset from the slot base. Always a multiple of
    /// [`PROJECTION_ALIGN`].
    pub offset: usize,
    /// The projection's own weight bytes — **not** the aligned stride to the
    /// next one. What the pack record holds and the H2D copies.
    pub bytes: usize,
    /// Bytes reserved from `offset`, which the kernel may address in full. See
    /// [`Projection::extent`].
    pub extent: usize,
    pub dtype: GgmlDType,
    pub shape: [usize; 2],
}

/// The byte layout of one layer's streamed weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerImage {
    pub kind: MixKind,
    /// Which FFN form this layer loaded as — carried because it decides the
    /// projection set as much as `kind` does.
    pub ffn: FfnForm,
    /// Placements in image order, mixer first then FFN.
    pub placements: Vec<Placement>,
    /// Bytes the image occupies, with every projection aligned. This is what a
    /// slot must be able to hold.
    pub total: usize,
}

/// Errors a layer's declared tensor set can have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    /// A projection this kind requires was not supplied.
    Missing(LayerTensor),
    /// A projection was supplied that this kind does not have.
    Unexpected(LayerTensor),
    /// The same projection was supplied twice.
    Duplicate(LayerTensor),
    /// A projection arrived as a float tensor rather than a quantized one.
    ///
    /// A slot is a flat byte payload, so it holds whichever quantized form the
    /// load produced — a KO twin at an int8 mode, the source quant at
    /// `Int8Mode::Off` or for a shape the matmul cannot tile. What it cannot
    /// hold is a weight that was never quantized: the trunk's projections all
    /// come off the checkpoint quantized, so a float here is a checkpoint or a
    /// wiring problem and is named at load rather than surfacing as a
    /// CUDA-only failure at the first view built over a slot.
    NotQuantized { role: LayerTensor, dtype: GgmlDType },
    /// A projection reserves less than it holds.
    ///
    /// The extent is what the next projection is placed past, so an extent
    /// shorter than the payload overlaps two weights.
    ShortExtent {
        role: LayerTensor,
        payload: usize,
        extent: usize,
    },
    /// A projection would occupy no bytes.
    ///
    /// A zero-length placement reserves nothing, so every later projection in
    /// the image lands where a previous one already sits. Cheap to check and
    /// impossible to diagnose from the wrong-looking activations it produces.
    Empty(LayerTensor),
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(r) => write!(f, "layer image: {r:?} was not supplied"),
            Self::Unexpected(r) => write!(f, "layer image: {r:?} does not belong to this kind"),
            Self::Duplicate(r) => write!(f, "layer image: {r:?} was supplied twice"),
            Self::NotQuantized { role, dtype } => write!(
                f,
                "layer image: {role:?} is {dtype:?} — a slot holds a quantized payload, and \
                 this projection was never quantized"
            ),
            Self::ShortExtent {
                role,
                payload,
                extent,
            } => write!(
                f,
                "layer image: {role:?} reserves {extent} B for a {payload} B payload, so the \
                 projection after it would overlap this one"
            ),
            Self::Empty(r) => write!(f, "layer image: {r:?} would occupy no bytes"),
        }
    }
}

impl std::error::Error for ImageError {}

/// Place a layer's projections into a slot image.
///
/// The order is the kind's own ([`MixKind::tensors`]) rather than the caller's,
/// so two layers of the same kind always produce byte-identical layouts however
/// their tensors were enumerated — which is what lets the pack write a record
/// for one layer and the slot view read it back for another.
pub fn layer_image(
    kind: MixKind,
    ffn: FfnForm,
    projections: &[Projection],
) -> Result<LayerImage, ImageError> {
    let want: Vec<LayerTensor> = kind.tensors(ffn).collect();
    for p in projections {
        if !want.contains(&p.role) {
            return Err(ImageError::Unexpected(p.role));
        }
        if projections.iter().filter(|q| q.role == p.role).count() > 1 {
            return Err(ImageError::Duplicate(p.role));
        }
    }

    let mut placements = Vec::with_capacity(want.len());
    let mut offset = 0usize;
    for role in want {
        let p = projections
            .iter()
            .find(|p| p.role == role)
            .ok_or(ImageError::Missing(role))?;
        if matches!(p.dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16) {
            return Err(ImageError::NotQuantized {
                role,
                dtype: p.dtype,
            });
        }
        if p.payload == 0 {
            return Err(ImageError::Empty(role));
        }
        if p.extent < p.payload {
            return Err(ImageError::ShortExtent {
                role,
                payload: p.payload,
                extent: p.extent,
            });
        }
        placements.push(Placement {
            role,
            offset,
            bytes: p.payload,
            extent: p.extent,
            dtype: p.dtype,
            shape: p.shape,
        });
        // The **extent**, not the payload: the next projection starts past
        // everything the kernel may address in this one.
        offset += align_up(p.extent);
    }
    Ok(LayerImage {
        kind,
        ffn,
        placements,
        total: offset,
    })
}

/// Bytes one zone slot must be to hold **any** layer of this model.
///
/// The max over images, for the reason §"Slots are uniform" gives: the zone
/// hands out equal-sized slots and layer images are not equal.
pub fn slot_bytes_for_layers(images: &[LayerImage]) -> usize {
    images.iter().map(|i| i.total).max().unwrap_or(0)
}

/// Round up to the projection alignment.
fn align_up(n: usize) -> usize {
    n.div_ceil(PROJECTION_ALIGN) * PROJECTION_ALIGN
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A KO-twin projection, whose extent is its payload.
    fn proj(role: LayerTensor, rows: usize, cols: usize, bytes: usize) -> Projection {
        Projection {
            role,
            shape: [rows, cols],
            dtype: GgmlDType::Q4_KO,
            payload: bytes,
            extent: bytes,
        }
    }

    /// Qwen3.8-27B's DeltaNet geometry: hidden 5120, conv_dim 10240,
    /// value_dim 6144, FFN 17408.
    fn dn_projections() -> Vec<Projection> {
        vec![
            proj(LayerTensor::Wqkv, 10240, 5120, 1000),
            proj(LayerTensor::Wz, 6144, 5120, 600),
            proj(LayerTensor::WOut, 5120, 6144, 600),
            // Fused `[2·intermediate, hidden]` — the production form.
            proj(LayerTensor::FfnGateUp, 34816, 5120, 3400),
            proj(LayerTensor::FfnDown, 5120, 17408, 1700),
        ]
    }

    /// The same layer with the FFN unfused, as a CPU load produces.
    fn dn_projections_split() -> Vec<Projection> {
        vec![
            proj(LayerTensor::Wqkv, 10240, 5120, 1000),
            proj(LayerTensor::Wz, 6144, 5120, 600),
            proj(LayerTensor::WOut, 5120, 6144, 600),
            proj(LayerTensor::FfnGate, 17408, 5120, 1700),
            proj(LayerTensor::FfnUp, 17408, 5120, 1700),
            proj(LayerTensor::FfnDown, 5120, 17408, 1700),
        ]
    }

    /// The same model's attention geometry: 24 Q / 4 KV @ head_dim 256.
    fn attn_projections() -> Vec<Projection> {
        vec![
            proj(LayerTensor::Wq, 12288, 5120, 1200),
            proj(LayerTensor::Wk, 1024, 5120, 100),
            proj(LayerTensor::Wv, 1024, 5120, 100),
            proj(LayerTensor::Wo, 5120, 6144, 600),
            proj(LayerTensor::FfnGateUp, 34816, 5120, 3400),
            proj(LayerTensor::FfnDown, 5120, 17408, 1700),
        ]
    }

    #[test]
    fn offsets_are_exact_and_aligned() {
        // Raw expected bytes, not a tolerance: every projection starts on a
        // 256 B boundary and each stride is its predecessor's payload rounded
        // up. 1000→1024, 600→768, 600→768, 3400→3584.
        let img = layer_image(MixKind::DeltaNet, FfnForm::Fused, &dn_projections()).unwrap();
        let offsets: Vec<usize> = img.placements.iter().map(|p| p.offset).collect();
        assert_eq!(offsets, vec![0, 1024, 1792, 2560, 6144]);
        assert_eq!(img.total, 6144 + 1792);

        // The payload lengths are the *unaligned* ones — a view is handed the
        // exact weight, never the padded stride.
        let bytes: Vec<usize> = img.placements.iter().map(|p| p.bytes).collect();
        assert_eq!(bytes, vec![1000, 600, 600, 3400, 1700]);
    }

    #[test]
    fn a_split_ffn_places_three_projections_where_a_fused_one_places_two() {
        // The same layer, loaded on a device that does not fuse. The pack has
        // to describe the layer that was actually loaded, so the form is part
        // of the image rather than assumed.
        let fused = layer_image(MixKind::DeltaNet, FfnForm::Fused, &dn_projections()).unwrap();
        let split =
            layer_image(MixKind::DeltaNet, FfnForm::Split, &dn_projections_split()).unwrap();
        assert_eq!(fused.placements.len(), 5);
        assert_eq!(split.placements.len(), 6);
        assert_eq!(fused.ffn, FfnForm::Fused);
        assert_eq!(split.ffn, FfnForm::Split);

        // A fused set offered as split is refused by name, not silently placed.
        assert_eq!(
            layer_image(MixKind::DeltaNet, FfnForm::Split, &dn_projections()),
            Err(ImageError::Unexpected(LayerTensor::FfnGateUp))
        );
        assert_eq!(
            layer_image(MixKind::DeltaNet, FfnForm::Fused, &dn_projections_split()),
            Err(ImageError::Unexpected(LayerTensor::FfnGate))
        );
    }

    #[test]
    fn image_order_is_the_kinds_order_not_the_callers() {
        // Two enumerations of the same layer must place identically, or a pack
        // record written from one cannot be read back through the other.
        let mut shuffled = dn_projections();
        shuffled.reverse();
        let a = layer_image(MixKind::DeltaNet, FfnForm::Fused, &dn_projections()).unwrap();
        let b = layer_image(MixKind::DeltaNet, FfnForm::Fused, &shuffled).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn both_kinds_carry_the_ffn() {
        let dn = layer_image(MixKind::DeltaNet, FfnForm::Fused, &dn_projections()).unwrap();
        let at = layer_image(MixKind::Attention, FfnForm::Fused, &attn_projections()).unwrap();
        for img in [&dn, &at] {
            let roles: Vec<LayerTensor> = img.placements.iter().map(|p| p.role).collect();
            assert!(roles.ends_with(FfnForm::Fused.tensors()));
        }
        assert_eq!(dn.placements.len(), 5);
        assert_eq!(at.placements.len(), 6);
    }

    #[test]
    fn the_slot_takes_the_larger_kind() {
        let dn = layer_image(MixKind::DeltaNet, FfnForm::Fused, &dn_projections()).unwrap();
        let at = layer_image(MixKind::Attention, FfnForm::Fused, &attn_projections()).unwrap();
        let slot = slot_bytes_for_layers(&[dn.clone(), at.clone()]);
        assert_eq!(slot, dn.total.max(at.total));
        assert!(slot >= dn.total && slot >= at.total);
    }

    #[test]
    fn an_empty_model_has_a_zero_slot() {
        assert_eq!(slot_bytes_for_layers(&[]), 0);
    }

    #[test]
    fn a_missing_projection_is_named() {
        let mut p = dn_projections();
        p.retain(|p| p.role != LayerTensor::Wz);
        assert_eq!(
            layer_image(MixKind::DeltaNet, FfnForm::Fused, &p),
            Err(ImageError::Missing(LayerTensor::Wz))
        );
    }

    #[test]
    fn a_projection_from_the_other_kind_is_refused() {
        let mut p = dn_projections();
        p.push(proj(LayerTensor::Wq, 12288, 5120, 1200));
        assert_eq!(
            layer_image(MixKind::DeltaNet, FfnForm::Fused, &p),
            Err(ImageError::Unexpected(LayerTensor::Wq))
        );
    }

    #[test]
    fn a_duplicate_is_refused() {
        let mut p = dn_projections();
        p.push(proj(LayerTensor::Wz, 6144, 5120, 600));
        assert_eq!(
            layer_image(MixKind::DeltaNet, FfnForm::Fused, &p),
            Err(ImageError::Duplicate(LayerTensor::Wz))
        );
    }

    #[test]
    fn a_zero_length_projection_cannot_be_placed() {
        // Every projection after it would be placed at an offset a previous one
        // already occupies, which shows up as wrong numbers rather than as a
        // fault. Named here instead.
        let mut p = dn_projections();
        p[1] = proj(LayerTensor::Wz, 48, 5120, 0);
        assert_eq!(
            layer_image(MixKind::DeltaNet, FfnForm::Fused, &p),
            Err(ImageError::Empty(LayerTensor::Wz))
        );
    }

    #[test]
    fn a_source_quant_is_accepted_because_off_mode_places_one() {
        // The KO twin is not the only slot form. At `Int8Mode::Off` there is no
        // twin at all, and even at an int8 mode a projection the matmul cannot
        // tile keeps its source quant — `layer_stream::build::slot_form` makes
        // that choice, and the image records whatever it chose.
        let mut p = dn_projections();
        p[0].dtype = GgmlDType::Q4_K;
        let img = layer_image(MixKind::DeltaNet, FfnForm::Fused, &p).unwrap();
        assert_eq!(img.placements[0].dtype, GgmlDType::Q4_K);
    }

    #[test]
    fn an_unquantized_projection_is_refused_before_it_reaches_a_slot() {
        // A slot holds a quantized payload. A float weight here means the
        // projection never went through the quantizer, which is a checkpoint or
        // wiring fault and is worth a name rather than a CUDA-only failure at
        // the first view.
        let mut p = dn_projections();
        p[0].dtype = GgmlDType::F16;
        assert_eq!(
            layer_image(MixKind::DeltaNet, FfnForm::Fused, &p),
            Err(ImageError::NotQuantized {
                role: LayerTensor::Wqkv,
                dtype: GgmlDType::F16,
            })
        );
        let err = layer_image(MixKind::DeltaNet, FfnForm::Fused, &p)
            .unwrap_err()
            .to_string();
        assert!(err.contains("never quantized"), "{err}");
    }

    #[test]
    fn every_ko_twin_is_accepted() {
        // The image must not care *which* KO form a layer repacked to: the mode
        // and the source width pick that per tensor, and a model mixing them is
        // ordinary (Q4_K_M puts Q6_K on some tensors).
        for dt in [
            GgmlDType::Q2_KO,
            GgmlDType::Q3_KO,
            GgmlDType::Q4_KO,
            GgmlDType::Q5_KO,
            GgmlDType::Q6_KO,
            GgmlDType::Q8_KO,
            GgmlDType::MXFP4_KO,
        ] {
            let mut p = dn_projections();
            p[0].dtype = dt;
            assert!(
                layer_image(MixKind::DeltaNet, FfnForm::Fused, &p).is_ok(),
                "{dt:?} refused"
            );
        }
    }

    #[test]
    fn a_payload_already_on_the_boundary_adds_no_padding() {
        let p = vec![
            proj(LayerTensor::Wqkv, 10240, 5120, PROJECTION_ALIGN),
            proj(LayerTensor::Wz, 6144, 5120, PROJECTION_ALIGN * 2),
            proj(LayerTensor::WOut, 5120, 6144, PROJECTION_ALIGN),
            proj(LayerTensor::FfnGateUp, 34816, 5120, PROJECTION_ALIGN),
            proj(LayerTensor::FfnDown, 5120, 17408, PROJECTION_ALIGN),
        ];
        let img = layer_image(MixKind::DeltaNet, FfnForm::Fused, &p).unwrap();
        assert_eq!(img.total, PROJECTION_ALIGN * 6);
    }
}
