//! notes_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::NotesError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {
    /// Key prefix to filter notes (e.g. `infra/`). Defaults to "" (lists all notes).
    pub prefix: Option<String>,
    /// Tags that returned notes must include. Defaults to none (no tag filter).
    pub tags: Option<Vec<String>>,
    /// Maximum notes to return (1–200). Defaults to 50.
    #[validate(range(min = 1, max = 200))]
    pub max_results: Option<u32>,
}

#[derive(Serialize)]
pub struct ListEntry {
    pub key: String,
    pub bytes: usize,
    pub tags: Vec<String>,
    pub updated_at: String,
}

#[derive(Serialize)]
pub struct ListResponse {
    pub notes: Vec<ListEntry>,
    pub total_matches: usize,
}

pub struct NotesList;

impl Tool for NotesList {
    const NAME: &'static str = "notes_list";
    const DESCRIPTION: &'static str =
        "Browse stored notes by key prefix or tag, returning each note's key, \
         tags, and timestamps — metadata only, not the body text. Use to \
         discover which notes exist before reading one.";

    type Request = ListRequest;
    type Response = ListResponse;
    type Error = NotesError;

    fn run(ctx: &ToolContext, req: ListRequest) -> Result<ListResponse, NotesError> {
        let prefix = req.prefix.as_deref().unwrap_or("");
        let tags = req.tags.as_deref().unwrap_or(&[]);
        let max = req.max_results.unwrap_or(50) as usize;
        let (entries, total) = ctx.notes.list(prefix, tags, max);
        let notes = entries
            .into_iter()
            .map(|e| ListEntry {
                key: e.key,
                bytes: e.bytes,
                tags: e.tags,
                updated_at: e.updated_at,
            })
            .collect();
        Ok(ListResponse {
            notes,
            total_matches: total,
        })
    }
}

pub const NOTES_LIST: RegisteredTool = RegisteredTool::new::<NotesList>();
