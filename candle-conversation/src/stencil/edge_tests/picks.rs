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

/// Where a target stopped being writable.
#[derive(Debug, PartialEq)]
enum Refused {
    /// The grammar masked out the byte at `at`.
    Masked { at: usize },
    /// The grammar wrote `wrote`, which leaves the target at `at`: the first
    /// byte it wrote differently.
    Prefilled { at: usize, wrote: String },
}

impl Refused {
    fn at(&self) -> usize {
        match self {
            Refused::Masked { at } | Refused::Prefilled { at, .. } => *at,
        }
    }
}

/// Write `target` through the grammar: `Ok` when every byte of it is allowed,
/// or where and how it was refused.
fn write(target: &str) -> Result<(), Refused> {
    let v = TestVocab::new();
    let mut session = StencilSession::new(tree(&v));
    let bytes = target.as_bytes();
    let mut at = 0usize;
    loop {
        match session.next_action() {
            StencilAction::Prefill(toks) => {
                let wrote = v.decode(&toks);
                if !bytes[at..].starts_with(&wrote) {
                    let same = bytes[at..]
                        .iter()
                        .zip(&wrote)
                        .take_while(|(a, b)| a == b)
                        .count();
                    return Err(Refused::Prefilled {
                        at: at + same,
                        wrote: String::from_utf8_lossy(&wrote).into_owned(),
                    });
                }
                at += wrote.len();
            }
            StencilAction::MaskedDecode(set) => {
                let b = *bytes.get(at).expect("the target ended inside the call");
                if !set.contains(b as TokenId) {
                    return Err(Refused::Masked { at });
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
    format!(
        "<tool_call>\n{{\"name\": \"post\", \"arguments\": {{\"body\": {body}}}}}\n</tool_call>"
    )
}

#[test]
fn a_nested_choice_holds_its_object_open_until_a_member_is_written() {
    let empty = call("{}");
    let refused = write(&empty).expect_err("an empty body is no post");
    let at = refused.at();
    assert_eq!(
        &empty[at..at + 2],
        "}}",
        "refused at the body's close: {refused:?}"
    );

    assert_eq!(write(&call("{\"job\": \"ward-shift\"}")), Ok(()));
    assert_eq!(
        write(&call(
            "{\"brief\": \"Walk the ward.\", \"steps\": \"go to the ward room\"}"
        )),
        Ok(())
    );
    let refused = write(&call("{\"steps\": \"go\"}")).expect_err("steps alone make no choice");
    assert!(refused.at() > 0, "{refused:?}");
}
