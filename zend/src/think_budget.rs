//! A turn's thinking and answer budget, set from its think mode.
//!
//! One function for every zend turn that reasons — a dialogue turn on its
//! composer dials, a passthrough turn on the balanced dial — so the two cannot
//! drift apart. A passthrough turn used to decode on the model preset's
//! fallback, which closes the block at 300 thinking tokens and force-ends the
//! whole reply at 1000: a Cline turn's plan was cut mid-sentence at 301, and a
//! file written through a tool call could not outlast the failsafe.

use candle_conversation::stencil::ThinkMode;
use candle_conversation::SamplingConfig;

/// Program `sampling` for a turn on `mode` whose answer may run to
/// `response_tokens` past the think block. `closer` is the phrase played when
/// the hard cap cuts the block mid-sentence.
pub fn steer(sampling: &mut SamplingConfig, mode: ThinkMode, response_tokens: i32, closer: &[u32]) {
    // The in-block reflection-marker suppression is the dial's ceiling:
    // Quick/Balanced discourage the "Wait"/"Hmm" family, Deep/Exhaustive leave
    // reconsideration free.
    sampling.segment_suppress_penalty = mode.suppress_penalty();
    // The EOT close ramp's graceful/force thresholds. The steering tree gives
    // every dial ONE span, so `segment_len` runs the length of the think block
    // and this budget IS the dial — it is the only thing that separates deep
    // from balanced. The close boost ramps `</think>`+EOS over the same window,
    // building pressure into the point the force override hard-closes.
    let (graceful_eot, force_eot) = mode.eot_budget();
    sampling.graceful_segment_close_after = graceful_eot;
    sampling.force_segment_close_after = force_eot;
    sampling.segment_close_ramp_start = graceful_eot;
    sampling.segment_close_ramp_len = force_eot;
    // The EOS budget is the whole-turn backstop on total length: the think
    // budget fixes where the answer starts, so the ramp begins as the block
    // ends and is dormant during reasoning, and `response_tokens` is the room
    // above it. It can never truncate the thinking budget. (The preset's
    // eos_boost magnitude/mult stay; the boost ramps to the graceful threshold.)
    let (eos_ramp_start, graceful_eos, forced_eos) = mode.eos_budget(response_tokens);
    sampling.eos_ramp_start = eos_ramp_start;
    sampling.eos_ramp_len = graceful_eos;
    sampling.graceful_eos_after = graceful_eos;
    sampling.forced_eos_after = forced_eos;
    // Hard-cap closer: when the force budget amputates the block mid-sentence,
    // the sampler plays this phrase and then closes the block itself, so the
    // reasoning ends as intentional prose with an explicit commitment. It fires
    // only at the hard cap, only mid-sentence, and only as the block ends — a
    // rescue, not a steer.
    sampling.segment_close_script = closer.to_vec();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The preset zend receives for the Qwen3.5/3.6 MoE checkpoints — the
    /// fallback a turn decodes on when nothing programs its budget.
    fn preset() -> SamplingConfig {
        SamplingConfig::for_gguf_architecture("qwen35moe")
    }

    #[test]
    fn balanced_replaces_the_presets_fallback_budget() {
        let before = preset();
        assert_eq!(
            (before.force_segment_close_after, before.forced_eos_after),
            (300, 1000),
            "the fallback this replaces"
        );
        let mut s = before;
        steer(&mut s, ThinkMode::Balanced, 3584, &[7, 8]);
        assert_eq!(
            (s.graceful_segment_close_after, s.force_segment_close_after),
            (1024, 1536)
        );
        assert_eq!(
            (s.segment_close_ramp_start, s.segment_close_ramp_len),
            (1024, 1536)
        );
        // Thinking reserves the force cutoff; the answer's room sits above it.
        assert_eq!(s.eos_ramp_start, 1536);
        assert_eq!(s.graceful_eos_after, 1536 + 3584 * 4 / 5);
        assert_eq!(s.eos_ramp_len, s.graceful_eos_after);
        assert_eq!(s.forced_eos_after, 1536 + 3584);
        assert_eq!(
            s.segment_suppress_penalty,
            ThinkMode::Balanced.suppress_penalty()
        );
        assert_eq!(s.segment_close_script, vec![7, 8]);
    }

    /// On every dial that reasons, the answer's failsafe lies beyond the point
    /// the block is forced shut — a long thought cannot end the reply.
    #[test]
    fn the_reply_outlasts_the_longest_think_block_on_every_dial() {
        for mode in [
            ThinkMode::Quick,
            ThinkMode::Balanced,
            ThinkMode::Deep,
            ThinkMode::Exhaustive,
        ] {
            let mut s = preset();
            steer(&mut s, mode, 256, &[]);
            assert!(
                s.graceful_eos_after > s.force_segment_close_after,
                "{mode:?}: the answer's failsafe fires inside the think budget"
            );
        }
    }
}
