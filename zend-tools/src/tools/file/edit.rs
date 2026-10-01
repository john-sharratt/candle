//! file_edit tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::replace::apply;

use super::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct EditRequest {
    /// The repository the file belongs to. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Path of the file to edit, relative to the repository — a project file, or one this session created (e.g. `src/main.rs`). Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// The text to replace, copied from the file exactly as it stands — whole lines, with enough of the lines around it that it occurs only once. Without the line numbers `file_read` shows. Required.
    #[validate(length(min = 1))]
    pub old_text: String,
    /// The text that takes its place. Empty to delete `old_text`. Required.
    pub new_text: String,
    /// Replace every occurrence of `old_text` instead of requiring exactly one. Optional; defaults to false.
    #[serde(default)]
    pub replace_all: bool,
}

#[derive(Serialize)]
pub struct EditResponse {
    pub repo: String,
    pub path: String,
    /// Occurrences replaced; zero when the edit was already in the file.
    pub replacements: usize,
    /// The edit's result was already in the file, so nothing was written.
    pub already_applied: bool,
    /// How `old_text` was found: `exact`, or `indentation` — as whole lines
    /// with their indentation ignored, the new text re-indented to the file's.
    pub matched: &'static str,
    /// Size of the file after the edit.
    pub bytes: usize,
}

pub struct FileEdit;

impl Tool for FileEdit {
    const NAME: &'static str = "file_edit";
    const DESCRIPTION: &'static str =
        "Replace one piece of text in an existing file with another. `old_text` is the text to \
         change, copied from the file exactly as it stands — whole lines, with enough of the lines \
         around it to occur only once; `new_text` is what takes its place, empty to delete it. Use \
         for: changing a value in a config, updating a function body, adding a method after an \
         existing one, fixing a typo. Text quoted at the wrong indentation is still found, and \
         the new text is re-indented to the file's. Text that occurs more than once is refused \
         unless `replace_all` is set; an edit whose result is already in the file is reported as \
         already applied, so re-sending it is safe. Triggered by \"change X to Y in the file\", \
         \"update these lines\", \"fix the value of\", \"add this after\". Returns repo, path, \
         how many occurrences were replaced, whether it was already applied, and the new byte \
         count. For full rewrites use write.";

    type Request = EditRequest;
    type Response = EditResponse;
    type Error = FileError;

    /// An edit whose result is already in the file is reported rather than
    /// applied twice, so re-sending the same edit leaves the same file.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: EditRequest) -> Result<EditResponse, FileError> {
        // Read through the overlay, so a file that lives only in the workspace is
        // editable. Nothing is recorded unless the edit changed the file.
        let store = ctx.files.repo(&req.repo)?;
        let content = store
            .read(&req.path)?
            .ok_or_else(|| FileError::NothingToEdit(req.path.clone()))?;

        let replaced = apply(&content, &req.old_text, &req.new_text, req.replace_all)?;
        let bytes = replaced.content.len();
        // Recorded as an edit — the changed lines only, not the file they
        // landed in.
        if !replaced.already_applied {
            store.edit(&req.path, replaced.content)?;
        }
        Ok(EditResponse {
            repo: req.repo,
            path: req.path,
            replacements: replaced.replacements,
            already_applied: replaced.already_applied,
            matched: replaced.matched.as_str(),
            bytes,
        })
    }
}

pub const FILE_EDIT: RegisteredTool = RegisteredTool::new::<FileEdit>();
