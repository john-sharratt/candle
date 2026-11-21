//! Tests that a turn's think block is steered once, and the tool-call trigger
//! stays armed.
//!
//! The think trigger fires on the block that opens the turn. After that block
//! closes, `<think>` decodes as ordinary text. The tool-call trigger remains active
//! for every call the turn makes, so steering the reasoning does not disable the
//! calls that follow it.

use super::*;

#[cfg(test)]
pub(super) mod think_steering_tests {
    use super::*;

    use crate::stencil::{TestVocab, Vocab};

    pub(in super::super) const THINK: TokenId = 151667;

    pub(in super::super) const THINK_CLOSE: TokenId = 151668;

    pub(in super::super) const TOOL_CALL: TokenId = 151657;

    pub(in super::super) fn steering() -> ThinkSteering {
        let v = TestVocab::new()
            .with_special("<think>", THINK)
            .with_special("</think>", THINK_CLOSE);
        let env = ThinkSteerEnvelope {
            think_open: THINK,
            think_close: THINK_CLOSE,
            eos: v.eos(),
            after_close: "",
        };
        let tree =
            |mode| Arc::new(compile(&compile_think_tree(mode, &env), &v).expect("think tree"));
        ThinkSteering {
            think_open: THINK,
            off: tree(ThinkMode::Off),
            quick: tree(ThinkMode::Quick),
            balanced: tree(ThinkMode::Balanced),
            deep: tree(ThinkMode::Deep),
            exhaustive: tree(ThinkMode::Exhaustive),
        }
    }

    /// The turn's think block is steered once; after it, `<think>` is text and
    /// the tool-call trigger is still armed.
    #[test]
    fn the_think_block_fires_once_and_the_call_trigger_stays() {
        let s = steering();
        let tools = TriggerRegistry::new().with_trigger(TOOL_CALL, Arc::clone(&s.off));
        for mode in [ThinkMode::Off, ThinkMode::Balanced, ThinkMode::Exhaustive] {
            let turn = s.registry_for(&tools, mode);
            assert!(
                turn.driver_for(THINK).is_some(),
                "{mode:?}: the opening block"
            );
            let rest = turn
                .after_firing(THINK)
                .unwrap_or_else(|| panic!("{mode:?}: the think trigger was not spent"));
            assert!(
                rest.driver_for(THINK).is_none(),
                "{mode:?}: a later <think> is text"
            );
            assert!(
                rest.driver_for(TOOL_CALL).is_some(),
                "{mode:?}: calls stay armed"
            );
        }
    }
}
