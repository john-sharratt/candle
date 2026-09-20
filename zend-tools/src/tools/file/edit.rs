//! file_edit tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::patch::apply;
use super::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct EditRequest {
    /// Path of the file to edit — a project file from the working directory, or one this session created (e.g. `src/main.rs`). Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// Unified-diff body: one or more `@@ -old,count +new,count @@` hunks whose lines are prefixed with a space (context), `-` (removed) or `+` (added). Hunks are located by their context rather than by the line numbers, and a hunk whose change is already in the file is reported as already applied instead of applied twice. Give every hunk at least one context or removed line. Required.
    #[validate(length(min = 1))]
    pub patch: String,
}

#[derive(Serialize)]
pub struct EditResponse {
    pub path: String,
    /// Hunks that changed the file.
    pub hunks_applied: usize,
    /// Hunks whose change was already in the file, so nothing was written for
    /// them.
    pub hunks_already_applied: usize,
    /// Size of the file after the patch.
    pub bytes: usize,
}

pub struct FileEdit;

impl Tool for FileEdit {
    const NAME: &'static str = "file_edit";
    const DESCRIPTION: &'static str =
        "Apply a unified diff to an existing VFS file. The patch is one or more \
         `@@ -old,count +new,count @@` hunks, each line prefixed with a space for context, `-` \
         for a removed line, `+` for an added one. Use for: changing a value in a config, \
         editing several places in one file at once, updating a function body, fixing a typo. \
         Hunks are located by their context, not by their line numbers, so the numbers need \
         only be close; a hunk matching in several places is rejected as ambiguous, and a hunk \
         whose change is already in the file is reported as already applied rather than applied \
         twice, so re-sending the same patch is safe. Either every hunk lands or none does. \
         Triggered by \"change X to Y in the file\", \"apply this diff\", \"update these \
         lines\", \"fix the value of\". Returns path, how many hunks applied, how many were \
         already applied, and the new byte count. For full rewrites use write.";

    type Request = EditRequest;
    type Response = EditResponse;
    type Error = FileError;

    /// Hunks are located by their context and one whose change is already in
    /// the file is reported rather than applied twice, so re-sending the same
    /// patch leaves the same file. This is why `file_edit` is a patch.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: EditRequest) -> Result<EditResponse, FileError> {
        // Read through the overlay, so a file that lives only in the workspace is
        // editable. The write below is what copies it up — doing it here instead
        // would leave a rejected patch having dirtied the file for no reason.
        let content = ctx
            .vfs
            .read(&req.path)?
            .ok_or_else(|| FileError::NothingToEdit(req.path.clone()))?;

        let patched = apply(&content, &req.patch)?;
        let bytes = patched.content.len();
        // A patch every hunk of which was already applied changes nothing, and
        // must therefore write nothing: a copy-up here would shadow a workspace
        // file on account of an edit that did not happen.
        if patched.applied > 0 {
            ctx.vfs.write(&req.path, patched.content)?;
        }
        Ok(EditResponse {
            path: req.path,
            hunks_applied: patched.applied,
            hunks_already_applied: patched.already_applied,
            bytes,
        })
    }
}

pub const FILE_EDIT: RegisteredTool = RegisteredTool::new::<FileEdit>();
