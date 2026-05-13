//! file_present tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::FileError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct PresentRequest {
    #[validate(length(min = 1, max = 10))]
    pub paths: Vec<String>,
    pub title: Option<String>,
    pub mode: Option<String>,
}

#[derive(Serialize)]
pub struct PresentResponse {
    pub presented: Vec<String>,
    pub missing: Vec<String>,
}

pub struct FilePresent;

impl Tool for FilePresent {
    const NAME: &'static str = "file_present";
    const DESCRIPTION: &'static str =
        "Present one or more VFS files to the user. Returns which files were found \
         and presented vs missing.";

    type Request = PresentRequest;
    type Response = PresentResponse;
    type Error = FileError;

    fn run(ctx: &ToolContext, req: PresentRequest) -> Result<PresentResponse, FileError> {
        let mut presented = Vec::new();
        let mut missing = Vec::new();
        for path in &req.paths {
            if ctx.vfs.read(path).is_some() {
                presented.push(path.clone());
            } else {
                missing.push(path.clone());
            }
        }
        if presented.is_empty() {
            return Err(FileError::NoFilesFound);
        }
        Ok(PresentResponse { presented, missing })
    }
}

pub const FILE_PRESENT: RegisteredTool = RegisteredTool::new::<FilePresent>();
