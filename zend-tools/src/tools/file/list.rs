//! file_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::FileError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {
    pub prefix: Option<String>,
}

#[derive(Serialize)]
pub struct FileEntry {
    pub path: String,
    pub bytes: usize,
    pub lines: usize,
}

#[derive(Serialize)]
pub struct ListResponse {
    pub files: Vec<FileEntry>,
    pub total_bytes: usize,
}

pub struct FileList;

impl Tool for FileList {
    const NAME: &'static str = "file_list";
    const DESCRIPTION: &'static str =
        "Enumerate the files currently in the session virtual filesystem, \
         optionally narrowed to a path prefix. Returns names and sizes, not \
         file contents.";

    type Request = ListRequest;
    type Response = ListResponse;
    type Error = FileError;

    fn run(ctx: &ToolContext, req: ListRequest) -> Result<ListResponse, FileError> {
        let prefix = req.prefix.as_deref().unwrap_or("");
        let entries = ctx.vfs.list(prefix);
        let total_bytes = ctx.vfs.total_bytes();
        let files = entries.into_iter()
            .map(|(path, bytes, lines)| FileEntry { path, bytes, lines })
            .collect();
        Ok(ListResponse { files, total_bytes })
    }
}

pub const FILE_LIST: RegisteredTool = RegisteredTool::new::<FileList>();
