//! Wrapping a slot's bytes as the layer's matmuls, without moving them.
//!
//! The layer analogue of the expert pipeline's `build_slot_view`. A slot is
//! device memory the weight zone owns; a view is a set of [`QMatMul`]s whose
//! storages point *into* it. Dropping a view releases the views and not the
//! memory (`Backing::Lease(LeaseOrigin::Foreign)`), which is what makes an
//! eviction a bookkeeping change and a relocation a memcpy rather than a
//! reload.

use ahash::{HashMap, HashMapExt};
use candle::quantized::{view_repacked, Int8Mode, QTensor};
use candle::{CudaDevice, Result};

use crate::models::layer_stream::descriptor::{LayerImage, LayerTensor};
use crate::models::quantized_matmul::QMatMul;

/// One layer's streamed projections, as matmuls over a slot.
///
/// Keyed by role rather than held in named fields because the set differs by
/// mixer kind, and a struct per kind would put the kind's shape in two places —
/// here and in [`LayerImage`] — for the layer loop to disagree about.
#[derive(Debug)]
pub struct StreamedLayer {
    projections: HashMap<LayerTensor, QMatMul>,
}

impl StreamedLayer {
    /// The matmul for `role`, or an error naming what this layer actually has.
    ///
    /// An error rather than an `Option`: a caller asking a DeltaNet layer for
    /// `Wq` has confused the layer kinds, and that is a bug to surface at the
    /// point it happens rather than a miss to handle.
    pub fn get(&self, role: LayerTensor) -> Result<&QMatMul> {
        self.projections.get(&role).ok_or_else(|| {
            let mut have: Vec<String> = self
                .projections
                .keys()
                .map(|r| format!("{r:?}"))
                .collect::<Vec<_>>();
            have.sort();
            candle::Error::Msg(format!(
                "streamed layer has no {role:?}; it carries [{}]",
                have.join(", ")
            ))
        })
    }

    /// Take `role`'s matmul **out** of the view.
    ///
    /// By value, because the layer assembled from these owns its projections
    /// and a `QMatMul` cannot be cloned into place: `QCudaStorage`'s `Clone` is
    /// always-owned precisely because `CudaSlice::clone` is a device-to-device
    /// copy, so cloning would duplicate the weight instead of aliasing the
    /// slot. Consuming the view is what keeps an assembly free of device
    /// traffic.
    pub fn take(&mut self, role: LayerTensor) -> Result<QMatMul> {
        self.projections.remove(&role).ok_or_else(|| {
            let mut have: Vec<String> = self.projections.keys().map(|r| format!("{r:?}")).collect();
            have.sort();
            candle::Error::Msg(format!(
                "streamed layer has no {role:?}; it carries [{}]",
                have.join(", ")
            ))
        })
    }

    /// How many projections this view wraps.
    pub fn len(&self) -> usize {
        self.projections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.projections.is_empty()
    }
}

/// Wrap the already-populated slot at `slot_base` as `image`'s matmuls.
///
/// Moves no bytes: the copy that filled the slot has already happened, and this
/// is pure geometry plus an address.
///
/// # Safety
///
/// `slot_base` must name a slot the zone has handed out and not reclaimed,
/// holding this layer's projections at `image`'s offsets, and it must outlive
/// the returned view. The caller's residency bookkeeping is what establishes
/// that — a view built over a slot whose transfer has not been joined reads a
/// torn weight, with nothing here able to detect it.
pub unsafe fn build_layer_view(
    image: &LayerImage,
    cuda_dev: &CudaDevice,
    slot_base: u64,
    mode: Int8Mode,
) -> Result<StreamedLayer> {
    let mut projections = HashMap::with_capacity(image.placements.len());
    for p in &image.placements {
        // `bytes` is the weight; `extent` is what the slot reserved for it, and
        // is larger for a source quant whose GGML kernel reads past every row.
        // The storage's payload stays `bytes`, so nothing downstream sees the
        // tail as weight.
        let storage = view_repacked(
            cuda_dev,
            slot_base + p.offset as u64,
            p.bytes,
            p.extent,
            p.dtype,
        )?;
        let qt = QTensor::new(storage, p.shape.to_vec())?;
        // Per projection, not per model. At an int8 mode a projection whose
        // shape the matmul cannot tile keeps its source form and runs the GGML
        // path — `layer_stream::build::slot_form` chose that when the image was
        // derived, and the dtype in the placement is the record of it.
        let placed = if p.dtype.is_ko() { mode } else { Int8Mode::Off };
        projections.insert(p.role, QMatMul::from_qtensor_view(qt, placed)?);
    }
    Ok(StreamedLayer { projections })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::layer_stream::descriptor::{layer_image, FfnForm, MixKind, Projection};
    use candle::quantized::GgmlDType;

    fn image() -> LayerImage {
        let p = |role, rows, cols| Projection {
            role,
            shape: [rows, cols],
            dtype: GgmlDType::Q4_KO,
            payload: 4096,
            extent: 4096,
        };
        layer_image(
            MixKind::DeltaNet,
            FfnForm::Fused,
            &[
                p(LayerTensor::Wqkv, 10240, 5120),
                p(LayerTensor::Wz, 6144, 5120),
                p(LayerTensor::WOut, 5120, 6144),
                p(LayerTensor::FfnGateUp, 34816, 5120),
                p(LayerTensor::FfnDown, 5120, 17408),
            ],
        )
        .unwrap()
    }

    #[test]
    fn a_missing_role_names_what_the_layer_has() {
        // Built by hand rather than over a device: the lookup's contract is
        // independent of whether a slot exists.
        let view = StreamedLayer {
            projections: HashMap::new(),
        };
        let err = view.get(LayerTensor::Wq).unwrap_err().to_string();
        assert!(err.contains("no Wq"), "{err}");
        assert!(view.is_empty());
    }

    #[test]
    fn the_image_decides_which_roles_a_view_will_carry() {
        // The view's key set is the image's, so a DeltaNet layer can never be
        // asked for an attention projection by accident.
        let img = image();
        let roles: Vec<LayerTensor> = img.placements.iter().map(|p| p.role).collect();
        assert!(roles.contains(&LayerTensor::Wqkv));
        assert!(!roles.contains(&LayerTensor::Wq));
    }
}
