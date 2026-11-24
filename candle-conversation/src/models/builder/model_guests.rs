//! The guests a model brings with it, beside the ones a deployment configures.
//!
//! A model whose release carries a vision tower names it in its spec
//! (`ModelSpec::vision`, `docs/vision_models.md` §2.2): every engine on the
//! model can read images, so the guest that runs the tower is registered with
//! the engine rather than named in a deployment's `guests.yaml`.

use super::ModelSpec;
use crate::guest::{GuestModel, GuestRegistry, VisionGuest};

/// Register the guests `spec`'s model brings on `guests`.
pub(super) fn register_model_guests(spec: &ModelSpec, guests: &mut GuestRegistry) {
    if let Some(source) = spec.vision {
        guests.register_vision(source, move || {
            Box::new(VisionGuest::new(source)) as Box<dyn GuestModel>
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guest::Guest;
    use crate::models::Model;
    use candle_transformers::models::quantized_qwen36_moe::QWEN36_VISION;
    use candle_transformers::models::quantized_qwen38_moe::FLASH_NEXT_VISION;

    /// Each model with a tower brings its own, and a model without one brings
    /// nothing.
    #[test]
    fn a_model_brings_the_tower_its_spec_names() {
        for (model, tower) in [
            (Model::Qwen38_FlashNext, &FLASH_NEXT_VISION),
            (Model::Qwen36_35B_A3B_Q4, &QWEN36_VISION),
        ] {
            let mut guests = GuestRegistry::new();
            register_model_guests(&model.preset_spec(), &mut guests);
            assert_eq!(guests.configured(), vec![Guest::Vision]);
            assert_eq!(guests.vision(), Some(tower));
        }
        let mut none = GuestRegistry::new();
        register_model_guests(&Model::Qwen3_8B_Q4.preset_spec(), &mut none);
        assert!(none.is_empty());
        assert_eq!(none.vision(), None);
    }
}
