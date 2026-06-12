//! notes_read tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::NotesError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReadRequest {
    #[validate(length(min = 1))]
    pub key: String,
}

#[derive(Serialize)]
pub struct ReadResponse {
    pub key: String,
    pub content: String,
    pub tags: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
    pub bytes: usize,
}

pub struct NotesRead;

impl Tool for NotesRead {
    const NAME: &'static str = "notes_read";
    const DESCRIPTION: &'static str =
        "Retrieve the full body content of one stored note by its exact key. \
         Notes persist across conversations.";

    type Request = ReadRequest;
    type Response = ReadResponse;
    type Error = NotesError;

    fn run(ctx: &ToolContext, req: ReadRequest) -> Result<ReadResponse, NotesError> {
        let note = ctx
            .notes
            .read(&req.key)
            .ok_or_else(|| NotesError::NotFound(req.key.clone()))?;
        Ok(ReadResponse {
            key: note.key,
            content: note.content,
            tags: note.tags,
            created_at: note.created_at,
            updated_at: note.updated_at,
            bytes: note.bytes,
        })
    }
}

pub const NOTES_READ: RegisteredTool = RegisteredTool::new::<NotesRead>();
