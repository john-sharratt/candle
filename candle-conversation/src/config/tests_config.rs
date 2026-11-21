//! Tests the configuration defaults that a silent misconfiguration would
//! otherwise hide.
//!
//! Every architecture the repository can load must have a think-close backstop
//! and an EOS failsafe, because a decode with neither runs to its ceiling and the
//! caller discards the whole output as reasoning with nothing logged. The ramp
//! tests hold each ramp's end after its start, since an inverted ramp reads as
//! plausible and collapses to a cliff. The remaining tests pin each family's
//! published temperature pairs, its DRY and repeat-penalty choices, and the
//! scheduler's prefill target against the model's ceiling.

use super::*;

#[cfg(test)]
pub(super) mod scheduler_config_tests {
    use super::SchedulerConfig;

    #[test]
    fn prefill_target_is_reliably_fillable() {
        // The scheduler per-forward target sits at the point the parallel
        // scope-ingest reliably fills every forward (amortization, design §6),
        // not chunked at 512 nor stalled waiting for a 4096-token forward the
        // reduced-scope scope KV rarely reaches on a 16 GB card. Must stay ≤ the
        // model-side `MAX_PREFILL_TOKENS` ceiling (8192).
        assert_eq!(SchedulerConfig::default().large_prefill_max_tokens, 2048);
        assert!(SchedulerConfig::default().large_prefill_max_tokens <= 8192);
    }
}

#[cfg(test)]
pub(super) mod sampling_config_tests {
    use super::SamplingConfig;

    /// Every architecture string the repository can load, and whether its family
    /// reasons in a `<think>` block.
    ///
    /// One list, used by both directions of the test below, so an arch added to
    /// the table without a steering arm — or given an arm it should not have —
    /// fails here rather than in a decode months later.
    const THINKING_ARCHES: &[&str] = &[
        "qwen3",
        "qwen3moe",
        "qwen2moe",
        "qwen35",
        "qwen35moe",
        "qwen36",
        "qwen36moe",
        "qwen38",
        "qwen38moe",
    ];

    /// **A missing arch string is silent, so this test is the thing that speaks.**
    ///
    /// `for_gguf_architecture` falls through to a conservative `_` arm for anything
    /// it does not recognise, and that arm leaves `force_segment_close_after` at `0`
    /// — disabled. A model that then fails to close its own `<think>` block runs to
    /// `max_response_tokens` and the caller throws the entire decode away as
    /// all-reasoning: an empty document, reported as a success, with nothing in any
    /// log to say what happened.
    ///
    /// That is not hypothetical. `qwen35` was absent from the table while the whole
    /// Qwen3.5/3.6/3.8 lineage loaded through it, and three separate third-party
    /// conversions were blamed for the result before the arm was found missing.
    #[test]
    fn every_thinking_arch_gets_a_close_backstop_and_an_eos_failsafe() {
        for arch in THINKING_ARCHES {
            let c = SamplingConfig::for_gguf_architecture(arch);
            assert!(
                c.force_segment_close_after > 0,
                "{arch}: no </think> backstop — it fell through to the `_` arm"
            );
            assert!(
                c.graceful_segment_close_after > 0,
                "{arch}: no graceful </think> close"
            );
            assert!(c.forced_eos_after > 0, "{arch}: no EOS failsafe");
            assert!(
                c.top_k > 0,
                "{arch}: top_k unset — the `_` arm's signature symptom"
            );
            assert!(
                SamplingConfig::non_thinking_for_gguf_architecture(arch).is_some(),
                "{arch}: reports no thinking mode, but its chat template has <think>"
            );
        }
    }

    /// The guard above only means something if the fallback really is bare — if the
    /// `_` arm ever grew a backstop of its own, every assertion there would pass for
    /// an arch that was never actually added.
    #[test]
    fn the_unknown_arch_fallback_is_genuinely_bare() {
        let c = SamplingConfig::for_gguf_architecture("something-nobody-has-heard-of");
        assert_eq!(c.force_segment_close_after, 0);
        assert_eq!(c.top_k, 0);
        assert!(SamplingConfig::non_thinking_for_gguf_architecture(
            "something-nobody-has-heard-of"
        )
        .is_none());
    }

    #[test]
    fn segment_temp_boost_defaults_to_zero() {
        assert_eq!(SamplingConfig::default().segment_temp_boost, 0.0);
        assert_eq!(SamplingConfig::argmax().segment_temp_boost, 0.0);
        assert_eq!(SamplingConfig::top_p(0.9, 0.7).segment_temp_boost, 0.0);
    }

    #[test]
    fn with_segment_temp_boost_sets_value() {
        let cfg = SamplingConfig::default().with_segment_temp_boost(0.05);
        assert_eq!(cfg.segment_temp_boost, 0.05);
    }

    #[test]
    fn think_close_budget_collapses_off_only_on_tight_budget() {
        use crate::stencil::ThinkMode;
        // Off with a short summary budget (100 < the 300 backstop): collapse to a
        // forced EMPTY block so the budget goes to the answer, not runaway thinking.
        let mut off_tight = SamplingConfig::compression();
        off_tight.set_think_close_budget(ThinkMode::Off, 100);
        assert_eq!(
            off_tight.force_segment_close_after, 1,
            "Off + tight → empty"
        );
        assert_eq!(off_tight.graceful_segment_close_after, 0);
        assert_eq!(off_tight.segment_close_ramp_len, 1);

        // Off with a roomy dialogue budget keeps the normal (220, 300) backstop.
        let mut off_roomy = SamplingConfig::compression();
        off_roomy.set_think_close_budget(ThinkMode::Off, 4096);
        let (g, f) = ThinkMode::Off.eot_budget();
        assert_eq!(
            off_roomy.force_segment_close_after, f,
            "Off + roomy → backstop"
        );
        assert_eq!(off_roomy.graceful_segment_close_after, g);

        // A REASONING dial is NEVER collapsed, even with a tiny answer budget — its
        // per-span think budget is independent of the answer length.
        let mut deep_tight = SamplingConfig::compression();
        deep_tight.set_think_close_budget(ThinkMode::Deep, 50);
        let (dg, df) = ThinkMode::Deep.eot_budget();
        assert_eq!(
            deep_tight.force_segment_close_after, df,
            "Deep never collapses"
        );
        assert_eq!(deep_tight.graceful_segment_close_after, dg);
        assert!(
            df > 50,
            "sanity: Deep's force budget exceeds the tiny answer budget"
        );
    }

    /// **Both of Qwen's published rows exist, and they are the right way round.**
    ///
    /// The lineage publishes a temperature and nucleus per think mode — `1.0/0.95`
    /// reasoning, `0.7/0.8` not — and the table carried the *instruct* row under
    /// both, because the non-thinking arm derives from the thinking one and
    /// overrode only budgets. So the reasoning numbers were nowhere in the
    /// repository and a deliberating turn decoded on the instruct card, silently:
    /// both rows are valid sampling, and nothing fails when the wrong one is used.
    #[test]
    fn each_think_mode_gets_its_own_published_temperature() {
        let thinking = SamplingConfig::for_gguf_architecture("qwen35");
        let instruct = SamplingConfig::non_thinking_for_gguf_architecture("qwen35")
            .expect("this family has a non-thinking mode");

        assert_eq!((thinking.temperature, thinking.top_p), (1.0, 0.95));
        assert_eq!((instruct.temperature, instruct.top_p), (0.7, 0.8));
        // The rest of the card is common to both rows; only the pair moves.
        assert_eq!(thinking.top_k, instruct.top_k);
        assert_eq!(thinking.presence_penalty, instruct.presence_penalty);
    }

    /// The mode swap runs in both directions and is reversible.
    ///
    /// A per-turn switch that only widened would leave a cast on the reasoning
    /// row for the rest of the process the first time one character deliberated.
    #[test]
    fn a_turns_think_mode_moves_the_sampled_pair_both_ways() {
        use crate::stencil::ThinkMode;
        let base = SamplingConfig::for_gguf_architecture("qwen35");

        let off = base.clone().with_think_mode(ThinkMode::Off, 4096);
        assert_eq!((off.temperature, off.top_p), (0.7, 0.8));

        let back = off.clone().with_think_mode(ThinkMode::Balanced, 4096);
        assert_eq!((back.temperature, back.top_p), (1.0, 0.95));

        // And every reasoning dial is the same row — the pair is about whether
        // there is a think block, not how long it is.
        for mode in [ThinkMode::Quick, ThinkMode::Deep, ThinkMode::Exhaustive] {
            let m = base.clone().with_think_mode(mode, 4096);
            assert_eq!((m.temperature, m.top_p), (1.0, 0.95), "{mode:?}");
        }
    }

    /// **A family with no published pair is left alone by the swap.**
    ///
    /// `qwen3`'s numbers were matched to a measured LM Studio reference run, not
    /// read off a card, so there is no second row to move to — and a mode switch
    /// that invented one would quietly undo that matching.
    #[test]
    fn a_family_without_a_published_pair_keeps_its_measured_numbers() {
        use crate::stencil::ThinkMode;
        let base = SamplingConfig::for_gguf_architecture("qwen3");
        assert!(base.mode_sampling.is_none());

        let (t, p) = (base.temperature, base.top_p);
        for mode in [ThinkMode::Off, ThinkMode::Deep] {
            let m = base.clone().with_think_mode(mode, 4096);
            assert_eq!((m.temperature, m.top_p), (t, p), "{mode:?}");
        }
    }

    #[test]
    fn qwen3_preset_enables_thinking_steering() {
        for arch in ["qwen3", "qwen3moe", "qwen2moe"] {
            let cfg = SamplingConfig::for_gguf_architecture(arch);
            // Thinking-only temperature boost is enabled.
            assert_eq!(
                cfg.segment_temp_boost, 0.05,
                "{arch} should set segment_temp_boost to 0.05"
            );
        }
    }

    /// **Every Qwen reasoning family carries DRY at (0.8, 1.75, 4, 512)**, for
    /// its free-text values; the families without the steering carry none.
    #[test]
    fn the_qwen_reasoning_families_carry_dry_for_free_text() {
        for arch in [
            "qwen3",
            "qwen3moe",
            "qwen2moe",
            "qwen35",
            "qwen35moe",
            "qwen36moe",
        ] {
            let dry = SamplingConfig::for_gguf_architecture(arch)
                .dry
                .unwrap_or_else(|| panic!("{arch} should enable DRY"));
            assert_eq!(
                (dry.multiplier, dry.base, dry.allowed_length, dry.range),
                (0.8, 1.75, 4, 512),
                "{arch}"
            );
        }
        for arch in ["qwen2", "llama"] {
            assert!(
                SamplingConfig::for_gguf_architecture(arch).dry.is_none(),
                "{arch}"
            );
        }
    }

    /// **The repeat penalty flips with DRY.** Qwen3.5 onwards samples a
    /// free-text value on the card's 1.0 under DRY, and everything else on the
    /// 1.05 tuned for where DRY is off; Qwen3 has one value for both.
    #[test]
    fn the_repeat_penalty_flips_with_dry() {
        let qwen36 = SamplingConfig::for_gguf_architecture("qwen36moe");
        assert_eq!(qwen36.presence_penalty, 1.5);
        assert_eq!(qwen36.repeat_penalty_for(false), 1.05);
        assert_eq!(qwen36.repeat_penalty_for(true), 1.0);
        let qwen3 = SamplingConfig::for_gguf_architecture("qwen3");
        assert_eq!(qwen3.repeat_penalty_for(false), 1.1);
        assert_eq!(qwen3.repeat_penalty_for(true), 1.1);
    }

    /// **Every ramp must actually ramp.**
    ///
    /// `eos_ramp_len` and `segment_close_ramp_len` are named "len" but hold the
    /// ramp's ABSOLUTE END, which is why every call site has a comment saying so.
    /// The kernel computes `span = max(end - start, 1)`, so an end below the start
    /// collapses the span to 1: the boost jumps from nothing to its full multiplier
    /// in a single token instead of easing in over the intended window. It is a
    /// cliff wearing a ramp's name, and it reads as a plausible pair of numbers.
    ///
    /// That shipped — the qwen3 non-thinking preset carried `(start 400, end 100)`
    /// under a comment reading "starts at 400 tokens, full by 500", so the whole
    /// family got an instant full-strength EOS boost at token 401.
    #[test]
    fn every_ramp_ends_after_it_starts() {
        let named = SamplingConfig::preset_names().iter().map(|n| {
            (
                n.to_string(),
                SamplingConfig::preset(n).expect("named preset"),
            )
        });
        let arches = ["qwen3", "qwen3moe", "qwen2moe", "qwen2", "llama", "unknown"];
        let thinking = arches.iter().map(|a| {
            (
                format!("for_gguf_architecture({a})"),
                SamplingConfig::for_gguf_architecture(a),
            )
        });
        // `None` for families with no non-thinking variant — dropped rather than
        // stood in for, so the sweep only ever checks configs that really exist.
        let non_thinking = arches.iter().filter_map(|a| {
            SamplingConfig::non_thinking_for_gguf_architecture(a)
                .map(|c| (format!("non_thinking_for_gguf_architecture({a})"), c))
        });
        let by_arch = thinking.chain(non_thinking);

        for (name, cfg) in named.chain(by_arch) {
            // A ramp is only live when its end is positive (the kernel's
            // `use_ramp` gate); a zeroed ramp is "disabled", not malformed.
            if cfg.eos_ramp_len > 0 {
                assert!(
                    cfg.eos_ramp_len > cfg.eos_ramp_start,
                    "{name}: EOS ramp ends at {} but starts at {} — the span \
                     collapses to 1 token and the boost becomes a cliff",
                    cfg.eos_ramp_len,
                    cfg.eos_ramp_start,
                );
            }
            if cfg.segment_close_ramp_len > 0 {
                assert!(
                    cfg.segment_close_ramp_len > cfg.segment_close_ramp_start,
                    "{name}: segment-close ramp ends at {} but starts at {}",
                    cfg.segment_close_ramp_len,
                    cfg.segment_close_ramp_start,
                );
            }
        }
    }

    /// The qwen3 non-thinking preset's ramp matches the window its own comment
    /// documents — the pair that was wrong, pinned so a future edit to one
    /// without the other fails here.
    #[test]
    fn qwen3_non_thinking_eos_ramp_spans_four_hundred_to_five_hundred() {
        for arch in ["qwen3", "qwen3moe", "qwen2moe"] {
            let cfg = SamplingConfig::non_thinking_for_gguf_architecture(arch)
                .unwrap_or_else(|| panic!("{arch} has a non-thinking preset"));
            assert_eq!(cfg.eos_ramp_start, 400, "{arch} ramp start");
            assert_eq!(cfg.eos_ramp_len, 500, "{arch} ramp end");
            // The ramp must also finish before the graceful failsafe, or the
            // failsafe fires while the boost is still easing in and the ramp
            // never reaches full strength.
            assert!(
                cfg.graceful_eos_after >= cfg.eos_ramp_len,
                "{arch}: graceful EOS at {} precedes the ramp's end at {}",
                cfg.graceful_eos_after,
                cfg.eos_ramp_len,
            );
        }
    }
}
