//! **A choice among an object's optional fields is one the object must make.**
//! A field marked [`Pick::AtLeastOne`] — the schema's `anyOf` of `required` —
//! holds the object open until a member of the choice is written, nested
//! objects included: an `invoke` body is one, and a post whose every field is
//! optional must still name a job or a brief.
//!
//! Driven byte by byte against a target call: prefills must match it, each
//! masked decode must allow its byte, and a free span writes it.

use std::sync::Arc;

use crate::stencil::compile::compile;
use crate::stencil::session::{StencilAction, StencilSession};
use crate::stencil::tool_call::{
    compile_tool_call_tree, Param, ParamType, Pick, ToolCallEnvelope, ToolSpec,
};
use crate::stencil::tree::StencilTree;
use crate::stencil::vocab::{TestVocab, TokenId, Vocab};

/// A string field, owing `pick` to the object's choice.
fn text(name: &str, pick: Pick) -> Param {
    Param {
        name: name.to_string(),
        ty: ParamType::String,
        required: false,
        enum_values: None,
        items: None,
        min_items: 0,
        max_items: None,
        properties: None,
        nullable: false,
        minimum: None,
        maximum: None,
        max_tokens: None,
        requires: Vec::new(),
        shapes: Vec::new(),
        pick,
        pattern: None,
    }
}

/// `post`, whose one argument is a required `body` object of three optional
/// fields, the first two a choice the body must make.
fn post() -> ToolSpec {
    ToolSpec {
        name: "post".to_string(),
        params: vec![Param {
            ty: ParamType::Object,
            required: true,
            properties: Some(vec![
                text("job", Pick::AtLeastOne),
                text("brief", Pick::AtLeastOne),
                text("steps", Pick::None),
            ]),
            ..text("body", Pick::None)
        }],
    }
}

fn tree(v: &TestVocab) -> Arc<StencilTree> {
    let spec = compile_tool_call_tree(&[post()], &ToolCallEnvelope::qwen3()).unwrap();
    Arc::new(compile(&spec, v).unwrap())
}

/// Write `target` through the grammar: `Ok` when every byte of it is allowed,
/// `Err` with the offset of the first byte that is not.
fn write(target: &str) -> Result<(), usize> {
    let v = TestVocab::new();
    let mut session = StencilSession::new(tree(&v));
    let bytes = target.as_bytes();
    let mut at = 0usize;
    loop {
        match session.next_action() {
            StencilAction::Prefill(toks) => {
                let wrote = v.decode(&toks);
                if !bytes[at..].starts_with(&wrote) {
                    return Err(at);
                }
                at += wrote.len();
            }
            StencilAction::MaskedDecode(set) => {
                let b = *bytes.get(at).expect("the target ended inside the call");
                if !set.contains(b as TokenId) {
                    return Err(at);
                }
                session.observe(b as TokenId, &[b]).unwrap();
                at += 1;
            }
            StencilAction::FreeDecode { .. } => {
                let b = bytes[at];
                session.observe(b as TokenId, &[b]).unwrap();
                at += 1;
            }
            StencilAction::Exit => return Ok(()),
        }
    }
}

fn call(body: &str) -> String {
    format!("<tool_call>\n{{\"name\": \"post\", \"arguments\": {{\"body\": {body}}}}}\n</tool_call>")
}

#[test]
fn a_nested_choice_holds_its_object_open_until_a_member_is_written() {
    let empty = call("{}");
    let at = write(&empty).expect_err("an empty body is no post");
    assert_eq!(&empty[at..at + 2], "}}", "refused at the body's close");

    assert_eq!(write(&call("{\"job\": \"ward-shift\"}")), Ok(()));
    assert_eq!(
        write(&call("{\"brief\": \"Walk the ward.\", \"steps\": \"go to the ward room\"}")),
        Ok(())
    );
    let at = write(&call("{\"steps\": \"go\"}")).expect_err("steps alone make no choice");
    assert!(at > 0);
}
