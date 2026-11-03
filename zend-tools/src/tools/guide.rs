//! `guide_list` and `guide_read`: the methods for doing work with these tools
//! — building, testing, deploying, releasing, debugging — read from the guide
//! library embedded in this binary, the one the MCP connector serves Claude, so
//! a worker reads the procedure Claude reads.

use std::sync::LazyLock;

use guides::Library;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, Replay, Tool, ToolContext, ToolError};

/// The guides, read once.
static LIBRARY: LazyLock<Library> =
    LazyLock::new(|| guides::embedded().expect("the embedded guides read"));

#[derive(Debug, Error)]
pub enum GuideError {
    #[error("there is no guide `{0}`; the guides are: {1}")]
    Unknown(String, String),
}

impl ToolError for GuideError {
    fn code(&self) -> &'static str {
        match self {
            GuideError::Unknown(..) => "unknown_guide",
        }
    }
}

/// What a guide tool answers.
#[derive(Serialize)]
pub struct GuideText {
    pub text: String,
}

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {}

pub struct GuideList;

impl Tool for GuideList {
    const NAME: &'static str = "guide_list";
    const DESCRIPTION: &'static str =
        "List the guides — the methods for building, testing, deploying, releasing and more — each \
         with when to use it.";

    type Request = ListRequest;
    type Response = GuideText;
    type Error = GuideError;

    fn run(_ctx: &ToolContext, _req: ListRequest) -> Result<GuideText, GuideError> {
        Ok(GuideText {
            text: LIBRARY.catalogue(),
        })
    }

    fn replay(_req: &ListRequest) -> Replay {
        Replay::Safe
    }
}

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReadRequest {
    /// The guide's name, as guide_list gives it. Required.
    #[validate(length(min = 1, max = 64))]
    pub name: String,
}

pub struct GuideRead;

impl Tool for GuideRead {
    const NAME: &'static str = "guide_read";
    const DESCRIPTION: &'static str =
        "Read one guide: the method for a kind of work, step by step, with what to check and what \
         to report.";

    type Request = ReadRequest;
    type Response = GuideText;
    type Error = GuideError;

    fn run(_ctx: &ToolContext, req: ReadRequest) -> Result<GuideText, GuideError> {
        let wanted = req.name.trim();
        LIBRARY
            .get(wanted)
            .map(|g| GuideText {
                text: g.body.clone(),
            })
            .ok_or_else(|| GuideError::Unknown(wanted.to_string(), LIBRARY.names().join(", ")))
    }

    fn replay(_req: &ReadRequest) -> Replay {
        Replay::Safe
    }
}

/// Every guide's name, in order: what `guide_read`'s definition offers.
pub fn names() -> Vec<String> {
    LIBRARY.names().into_iter().map(str::to_string).collect()
}

pub const GUIDE_LIST: RegisteredTool = RegisteredTool::new::<GuideList>();
pub const GUIDE_READ: RegisteredTool = RegisteredTool::new::<GuideRead>();

#[cfg(test)]
mod tests {
    use super::*;

    /// **A guide is read by name, and an unknown one names those there are**;
    /// the list names the deploy and test guides with when to use them.
    #[test]
    fn a_guide_is_read_by_name() {
        let ctx = ToolContext::new();
        let deploy = GuideRead::run(
            &ctx,
            ReadRequest {
                name: " deploy ".into(),
            },
        )
        .unwrap();
        assert!(deploy.text.contains("service_deploy"), "{}", deploy.text);
        let unknown = GuideRead::run(
            &ctx,
            ReadRequest {
                name: "nope".into(),
            },
        )
        .unwrap_err();
        assert!(unknown.to_string().contains("the guides are: "), "{unknown}");
        let list = GuideList::run(&ctx, ListRequest {}).unwrap();
        assert!(list.text.contains("\n- deploy — "), "{}", list.text);
        assert!(list.text.contains("\n- test — "), "{}", list.text);
        assert!(names().contains(&"release".to_string()));
    }
}
