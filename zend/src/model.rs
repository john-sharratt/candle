//! The model zend runs.
//!
//! zend names its own model (`docs/parasite_model.md` §4). Its substrate is
//! bound to it (`docs/shared_model_host.md` §4), so changing it starts zend on a
//! new substrate.

use candle_conversation::models::Model;

/// The model zend runs: Qwen3.8-Flash-Next, at the card's expert rung.
pub const MODEL: Model = Model::Qwen38_FlashNext;

#[cfg(test)]
mod tests {
    use super::*;

    use candle_conversation::models::{DialectType, ModelArch};

    /// zend runs Flash-Next, speaks the Qwen3.5 dialect its prompts are written
    /// in, and names it by its preset.
    #[test]
    fn zend_runs_flash_next() {
        let spec = MODEL.spec();
        assert!(matches!(spec.arch, ModelArch::Qwen4Exp));
        assert!(matches!(spec.chat_format, DialectType::Qwen35));
        assert_eq!(MODEL.override_key().as_deref(), Some("Qwen38_FlashNext"));
    }
}
