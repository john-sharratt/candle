//! The guests a model brings with it, beside the ones a deployment configures.
//!
//! Flash-Next's vision tower ships in its own `mmproj` GGUF, pinned with the
//! model: every engine on the model can read images, so the guest that runs
//! the tower is registered with the engine rather than named in a
//! deployment's `guests.yaml`.

use super::ModelArch;
use crate::guest::{Guest, GuestModel, GuestRegistry, VisionGuest};
use candle_transformers::models::quantized_qwen38_moe::cached_vision_tower_file;

/// Register the guests `arch` brings on `guests`.
pub(super) fn register_model_guests(arch: &ModelArch, guests: &mut GuestRegistry) {
    if matches!(arch, ModelArch::Qwen4Exp) {
        guests.register(Guest::Vision, || {
            Box::new(VisionGuest::new(cached_vision_tower_file)) as Box<dyn GuestModel>
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flash-Next brings its vision tower; a model without one brings nothing.
    #[test]
    fn flash_next_brings_its_vision_tower() {
        let mut guests = GuestRegistry::new();
        register_model_guests(&ModelArch::Qwen4Exp, &mut guests);
        assert_eq!(guests.configured(), vec![Guest::Vision]);
        let mut none = GuestRegistry::new();
        register_model_guests(&ModelArch::Qwen3, &mut none);
        assert!(none.is_empty());
    }
}
