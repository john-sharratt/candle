//! notes_write tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{NotesError, MAX_NOTE_BYTES};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct WriteRequest {
    /// Key identifying the note to create or update (1–256 chars). Required.
    #[validate(length(min = 1, max = 256))]
    pub key: String,
    /// Full note body to store, replacing any existing content for this key. Required.
    pub content: String,
    /// Tags to associate with the note. Defaults to none (empty tag list).
    pub tags: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct WriteResponse {
    pub key: String,
    pub bytes: usize,
    pub tags: Vec<String>,
    pub created: bool,
    pub updated_at: String,
}

pub struct NotesWrite;

impl Tool for NotesWrite {
    const NAME: &'static str = "notes_write";
    const DESCRIPTION: &'static str =
        "Write or update a persistent note by key. Notes survive across conversations. \
         Use for: saving important information, tracking progress, storing references. \
         Use notes_search to retrieve by content; notes_list to browse.";

    type Request = WriteRequest;
    type Response = WriteResponse;
    type Error = NotesError;

    fn run(ctx: &ToolContext, req: WriteRequest) -> Result<WriteResponse, NotesError> {
        if req.key.len() > 256 {
            return Err(NotesError::KeyTooLong);
        }
        if req.content.len() > MAX_NOTE_BYTES {
            return Err(NotesError::NoteTooLarge);
        }
        let tags = req.tags.unwrap_or_default();
        let (created, bytes) = ctx.notes.write(&req.key, req.content, tags.clone());
        let updated_at = ctx
            .notes
            .read(&req.key)
            .map(|n| n.updated_at)
            .unwrap_or_default();
        Ok(WriteResponse {
            key: req.key,
            bytes,
            tags,
            created,
            updated_at,
        })
    }
}

pub const NOTES_WRITE: RegisteredTool = RegisteredTool::new::<NotesWrite>();
