//! The sampling zend decodes on: Qwen's "thinking, precise coding" row at 0.7,
//! with a gentle repeat penalty and every other repetition penalty off.
//!
//! The per-model preset zend receives is Qwen's *general-task* row — for the
//! Qwen3.5/3.6 MoE checkpoints temperature 1.0, presence penalty 1.5, plus DRY.
//! The same model cards publish a separate row for precise coding (temperature
//! 0.6, top_p 0.95, top_k 20, presence 0), and a coding assistant's output is
//! exactly what those penalties damage: paths, identifiers and boilerplate that
//! are only correct if they repeat. Measured on a Cline turn under the general
//! row: three commands naming `c:\Users\johna\prog\candle`, and every repeat of
//! the path came out with a space spliced in. DRY is off by default in
//! llama.cpp and was built for chat and creative writing; its own author warns
//! it corrupts names that must repeat verbatim.
//!
//! Temperature is 0.7, a little above the card's 0.6, with a repeat penalty of
//! 1.05 — Qwen3-Coder's own card value. Both are there for the reasoning block:
//! at 0.5 with no penalty at all, a GUI turn that had settled its one file read
//! spent its whole think budget rehearsing the call — "Output matches. ✅
//! Proceeds." 45 times over — until the budget closed the block, and it then
//! emitted the untagged draft it had rehearsed rather than a call. The repeat
//! penalty is multiplicative over the recent window, far gentler than presence
//! or DRY, which stay off for the damage they do to repeated paths and names.
//! A think-suppressed turn keeps the same temperature at the card's instruct
//! `top_p`, since Qwen publishes no instruct coding row.
//!
//! The pairs go into [`ModeSampling`] as well as the live fields, because a
//! turn's think mode re-adopts its row from there: setting only `temperature`
//! would be undone by the first turn that declares a mode.

use candle_conversation::{ModeSampling, SamplingConfig};

/// The temperature every zend turn samples at.
pub const TEMPERATURE: f32 = 0.7;

/// The multiplicative repeat penalty over the recent window — gentle enough
/// that a path or identifier can repeat, firm enough to break a rehearsal loop.
const REPEAT_PENALTY: f32 = 1.05;

/// Qwen's precise-coding `top_p`, for a turn that reasons.
const THINKING_TOP_P: f32 = 0.95;

/// Qwen's instruct `top_p`, for a think-suppressed turn.
const INSTRUCT_TOP_P: f32 = 0.8;

/// Qwen's `top_k`, the same in every row.
const TOP_K: i32 = 20;

/// Put `sampling` on the coding row, leaving its thinking steering (the close
/// budget, the segment ids, the EOS failsafes) as the preset set it.
pub fn apply(sampling: &mut SamplingConfig) {
    let modes = ModeSampling {
        thinking: (TEMPERATURE, THINKING_TOP_P),
        instruct: (TEMPERATURE, INSTRUCT_TOP_P),
    };
    // Adopts the row for the config's current mode into temperature/top_p.
    *sampling = std::mem::take(sampling).with_mode_sampling(modes);
    sampling.top_k = TOP_K;
    sampling.presence_penalty = 0.0;
    sampling.frequency_penalty = 0.0;
    sampling.repeat_penalty = REPEAT_PENALTY;
    sampling.cross_turn_penalty = 0.0;
    sampling.dry = None;
}

#[cfg(test)]
mod tests {
    use candle_conversation::stencil::ThinkMode;

    use super::*;

    /// The preset zend actually receives for the Qwen3.5/3.6 MoE checkpoints.
    fn general_row() -> SamplingConfig {
        SamplingConfig::for_gguf_architecture("qwen35moe")
    }

    #[test]
    fn the_general_row_becomes_the_coding_row() {
        let before = general_row();
        assert_eq!(before.presence_penalty, 1.5, "the preset this replaces");
        assert!(before.dry.is_some(), "the preset this replaces");

        let mut s = before;
        apply(&mut s);
        assert_eq!((s.temperature, s.top_p, s.top_k), (0.7, 0.95, 20));
        assert_eq!(s.presence_penalty, 0.0);
        assert_eq!(s.frequency_penalty, 0.0);
        assert_eq!(s.repeat_penalty, 1.05);
        assert_eq!(s.cross_turn_penalty, 0.0);
        assert!(s.dry.is_none());
    }

    /// A turn's think mode re-adopts its row from `ModeSampling`; the coding
    /// row must be what it finds, in both directions.
    #[test]
    fn a_think_mode_switch_keeps_the_coding_temperature() {
        let mut s = general_row();
        apply(&mut s);
        let off = s.clone().with_think_mode(ThinkMode::Off, 4096);
        assert_eq!((off.temperature, off.top_p), (0.7, 0.8));
        let back = off.with_think_mode(ThinkMode::Balanced, 4096);
        assert_eq!((back.temperature, back.top_p), (0.7, 0.95));
    }

    /// Only the sampled distribution changes; the think-block close budget and
    /// the EOS failsafes are the preset's.
    #[test]
    fn the_thinking_steering_is_left_as_the_preset_set_it() {
        let before = general_row();
        let mut s = before.clone();
        apply(&mut s);
        assert_eq!(
            s.force_segment_close_after,
            before.force_segment_close_after
        );
        assert_eq!(
            s.graceful_segment_close_after,
            before.graceful_segment_close_after
        );
        assert_eq!(s.forced_eos_after, before.forced_eos_after);
        assert_eq!(s.segment_temp_boost, before.segment_temp_boost);
    }
}
