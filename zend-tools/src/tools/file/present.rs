//! file_present tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::FileError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct PresentRequest {
    /// VFS paths to surface to the user (1–10 entries, session-relative). Required.
    #[validate(length(min = 1, max = 10))]
    pub paths: Vec<String>,
    /// Optional short heading shown above the presented files. Defaults to none.
    pub title: Option<String>,
    /// One of: auto, inline, preview. auto renders inline if small else as a preview card;
    /// inline always renders full content; preview always shows an openable card. Defaults to auto.
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
        "Surface one or more virtual-filesystem files to the user in the chat \
         interface, reporting which paths were found and shown versus \
         missing.";

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
