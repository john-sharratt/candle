//! Steering a tool call's name to the tool its reasoning named.
//!
//! The stencil's name branch is a masked choice among every catalog name; it
//! does not know what the model just decided. Here, at the moment the call
//! opens, the reasoning since the last call is read back and, when it decides
//! on exactly one of the branch's arms ([`named_arm`]), the driver is held to
//! that arm. Nothing else about the call changes: the arguments are the model's.
//!
//! Only a call opened mid-turn by its trigger token is steered. A turn that
//! *begins* inside the grammar (its scaffold prefilled before the first
//! sampled token) has no reasoning of its own to read.

use tokenizers::Tokenizer;

use crate::stencil::{arm_name, named_arm, StencilDriver, TokenId};

/// Hold `driver`'s name branch to the tool named in `generated` — the turn's
/// tokens so far — and return that name, or `None` when the reasoning does not
/// settle on one tool (or the tree opens on no branch).
pub(super) fn steer_to_named_tool(
    tokenizer: &Tokenizer,
    generated: &[u32],
    driver: &mut StencilDriver,
) -> Option<String> {
    steer_with(
        |tokens| tokenizer.decode(tokens, false).ok(),
        generated,
        driver,
    )
}

/// [`steer_to_named_tool`] over any decoder.
fn steer_with(
    decode: impl Fn(&[TokenId]) -> Option<String>,
    generated: &[TokenId],
    driver: &mut StencilDriver,
) -> Option<String> {
    let arms = driver.tree().first_branch()?.arms();
    let reasoning = decode(generated)?;
    let decoded: Vec<String> = arms
        .iter()
        .map(|arm| decode(arm).unwrap_or_default())
        .collect();
    let names: Vec<&str> = decoded
        .iter()
        .map(|text| arm_name(text.as_bytes()))
        .collect();
    let pick = named_arm(&reasoning, &names)?;
    let name = names[pick].to_string();
    driver.steer_first_branch(arms[pick].clone());
    Some(name)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::stencil::{
        compile, compile_tool_call_tree, parse_tools, StepMask, TestVocab, ToolCallEnvelope, Vocab,
    };

    const CATALOG: &str = r#"[
        {"name":"file_list","params":[{"name":"path","type":"string","required":true}]},
        {"name":"file_read","params":[{"name":"path","type":"string","required":true}]},
        {"name":"write","params":[{"name":"path","type":"string","required":true}]}]"#;

    fn driver(v: &TestVocab) -> StencilDriver {
        let tools = parse_tools(CATALOG).unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        StencilDriver::new(Arc::new(compile(&spec, v).unwrap()))
    }

    /// The call text up to the first value, taking the lowest token at every
    /// masked step — so an unsteered walk writes `file_list`.
    fn call_text(driver: &mut StencilDriver, v: &TestVocab) -> String {
        let mut text = Vec::new();
        loop {
            match driver.step() {
                StepMask::Prefill(run) => text.extend(v.decode(&run)),
                StepMask::Branch(set) => {
                    let t = set.tokens()[0];
                    text.extend(v.token_bytes(t));
                    driver.accept(t, &v.token_bytes(t));
                }
                StepMask::Free { .. } | StepMask::Done => break,
            }
        }
        String::from_utf8(text).unwrap()
    }

    fn decode(v: &TestVocab) -> impl Fn(&[TokenId]) -> Option<String> + '_ {
        |tokens| String::from_utf8(v.decode(tokens)).ok()
    }

    /// **The turn's reasoning picks the call's name.**
    #[test]
    fn the_reasoning_s_tool_is_the_call_s_name() {
        let v = TestVocab::new();
        let reasoning =
            v.encode("<think>\nThe file is new, so use the `write` tool.\n</think>\n\n");
        let mut d = driver(&v);
        assert_eq!(
            steer_with(decode(&v), &reasoning, &mut d).as_deref(),
            Some("write")
        );
        let text = call_text(&mut d, &v);
        assert!(text.contains("\"name\": \"write\""), "{text:?}");
    }

    /// Reasoning that reports on one tool and decides nothing leaves the name
    /// the model's own choice.
    #[test]
    fn reasoning_without_a_decision_leaves_the_name_free() {
        let v = TestVocab::new();
        let reasoning = v.encode("The file_read returned not_found. I'll create it.");
        let mut d = driver(&v);
        assert_eq!(steer_with(decode(&v), &reasoning, &mut d), None);
        let text = call_text(&mut d, &v);
        assert!(text.contains("\"name\": \"file_list\""), "{text:?}");
    }
}
