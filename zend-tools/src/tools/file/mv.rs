//! file_move tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::VfsStore;

use super::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct MoveRequest {
    /// The repository the file belongs to. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Path of the file to move, relative to the repository. Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// Where it goes, relative to the repository, its file name included. A file already there is refused. Required.
    #[validate(length(min = 1))]
    pub to: String,
}

#[derive(Serialize)]
pub struct MoveResponse {
    pub repo: String,
    pub path: String,
    pub to: String,
    pub bytes: usize,
}

pub struct FileMove;

impl Tool for FileMove {
    const NAME: &'static str = "file_move";
    const DESCRIPTION: &'static str =
        "Move or rename a file inside one repository, its bytes exactly as they are, with no \
         need to read it first. Use for: moving a file to another folder, renaming a file, \
         reorganising a directory. Triggered by \"move\", \"rename\", \"mv\". Refused when \
         something already stands at `to`. Returns the repo, both paths and the file's size. \
         Held in this session until committed, as `write` is. To change the file as well, edit \
         it after moving it.";

    type Request = MoveRequest;
    type Response = MoveResponse;
    type Error = FileError;

    /// Sent again after it landed, the file is no longer at `path`, so the
    /// call is refused and nothing changes.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: MoveRequest) -> Result<MoveResponse, FileError> {
        let store = ctx.files.repo(&req.repo)?;
        let bytes = move_file(&store, &req.path, &req.to)?;
        Ok(MoveResponse {
            repo: req.repo,
            path: req.path,
            to: req.to,
            bytes,
        })
    }
}

/// Move `from` to `to` in `store`, byte for byte; the file's size.
pub(crate) fn move_file(store: &VfsStore, from: &str, to: &str) -> Result<usize, FileError> {
    let bytes = store
        .read_bytes(from)?
        .ok_or_else(|| FileError::NotFound(from.to_string()))?;
    if store.read_bytes(to)?.is_some() {
        return Err(FileError::InvalidArguments(format!(
            "{to} already exists: delete it first, or name another destination"
        )));
    }
    let size = bytes.len();
    store.write_bytes(to, bytes)?;
    store.delete(from);
    Ok(size)
}

pub const FILE_MOVE: RegisteredTool = RegisteredTool::new::<FileMove>();
