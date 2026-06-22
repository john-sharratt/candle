//! file_edit tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::FileError;
use crate::state::vfs::VfsError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct EditRequest {
    #[validate(length(min = 1))]
    pub path: String,
    pub old_str: String,
    pub new_str: String,
}

#[derive(Serialize)]
pub struct EditResponse {
    pub path: String,
    pub bytes: usize,
}

pub struct FileEdit;

impl Tool for FileEdit {
    const NAME: &'static str = "file_edit";
    const DESCRIPTION: &'static str =
        "Make a targeted edit to an existing VFS file by replacing a unique substring. Use \
         for: changing a value in a config, updating a function body, fixing a typo, modifying \
         one line without rewriting the whole file, applying small surgical changes. The \
         old_str must appear exactly once — if it appears multiple times the call returns \
         ambiguous and asks for more surrounding context. Triggered by \"change X to Y in the \
         file\", \"update this line\", \"modify the part where it says\", \"fix the value of\". \
         Returns path and new byte count. For full rewrites use write.";

    type Request = EditRequest;
    type Response = EditResponse;
    type Error = FileError;

    fn run(ctx: &ToolContext, req: EditRequest) -> Result<EditResponse, FileError> {
        let content = ctx
            .vfs
            .read(&req.path)
            .ok_or_else(|| FileError::NotFound(req.path.clone()))?;

        let count = content.matches(&req.old_str).count();
        if count == 0 {
            return Err(FileError::NotFound(format!(
                "old_str not found in {}",
                req.path
            )));
        }
        if count > 1 {
            return Err(FileError::Ambiguous);
        }

        let new_content = content.replacen(&req.old_str, &req.new_str, 1);
        let bytes = new_content.len();
        ctx.vfs.write(&req.path, new_content).map_err(|e| match e {
            VfsError::Full => FileError::VfsFull,
        })?;
        Ok(EditResponse {
            path: req.path,
            bytes,
        })
    }
}

pub const FILE_EDIT: RegisteredTool = RegisteredTool::new::<FileEdit>();
